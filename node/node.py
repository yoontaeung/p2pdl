import socket
import threading
import pickle
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils
import time

from p2pdl.training.train import train
from p2pdl.aggregator.aggregation import aggregate_models, broadcast_global_model_update
from p2pdl.evaluation.evaluation import evaluate
from p2pdl.utils.log import save_results
from p2pdl.utils.broadcast import send_echo, send_ready, send_sup
from p2pdl.utils.crypto import KeyServer, generate_key_pair, verify_signature, verify_signature_2

logging.basicConfig(level=logging.INFO)

class Node:
    def __init__(self, model, data, key_server, addr="127.0.0.1", port=12345):
        self.model = model
        self.previous_model_state = None
        self.data_loader = data
        self.addr = addr
        self.port = port
        self.neighbors = []  # Stores tuples of (addr, port)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01) 
        self.loss_fn = nn.CrossEntropyLoss()
        self.running = False
        self.received_models = []
        self._stop_event = threading.Event()  # Event to signal when to stop listening
        self.trainers_list = []
        self.testers_list = []
        self.signature_list = []
        self.sender_list = []
        self.serialized_state = None
        self.local_update = None

        self.__private_key, self.public_key = generate_key_pair()
        self.key_server = key_server
        self.key_server.register_key(self.addr, self.port, self.public_key)

        self.brb_delivered_event = threading.Event()
        self.received_echo_cnt = 0
        self.received_ready_cnt = 0
        self.received_sup_cnt = 0
        self.sent_ready = False
        self.sent_sup = False
        self.delivered = False
    

    def reset_delivered_flag(self):
        self.local_update = None
        self.sent_ready = False
        self.sent_sup = False
        self.delivered = False
        self.received_echo_cnt = 0
        self.received_ready_cnt = 0
        self.received_sup_cnt = 0
        self.signature_list = []
        self.sender_list = []
        self.received_models = []
        self.brb_delivered_event.clear()

    def set_delivered_flag(self):
        self.brb_delivered_event.set()

    def wait_for_delivered(self):
        logging.debug(f"[{self.addr}:{self.port}] Waiting for all echo and ready messages delivered ...")
        self.brb_delivered_event.wait()
        logging.debug(f"[{self.addr}:{self.port}] All messages delivered, proceeding...")

    def start(self):
        self._stop_event.clear()
        threading.Thread(target=self.listen_for_connections).start()
        logging.info(f"[{self.addr}:{self.port}] Node started running ...")

    def listen_for_connections(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.addr, self.port))
            s.listen()
            while not self._stop_event.is_set():  # Stop if the event is set
                try:
                    s.settimeout(1)  # Timeout to check for the stop event regularly
                    conn, addr = s.accept()
                    threading.Thread(target=self.handle_connection, args=(conn,)).start()
                except socket.timeout:
                    continue
    
    def stop(self):
        self._stop_event.set()  # Signal to stop listening
        logging.info(f"[{self.addr}:{self.port}] Node stop running ...")

    def handle_connection(self, conn):
        try:
            msg_len_data = conn.recv(4)
            if not msg_len_data:
                return
            msg_len = int.from_bytes(msg_len_data, byteorder='big')
            
            data = b''
            while len(data) < msg_len:
                packet = conn.recv(min(msg_len - len(data), 4096))
                if not packet:
                    break
                data += packet
            
            if len(data) == msg_len:
                command = pickle.loads(data)
                if command['type'] == 'connect':
                    neighbor_info = (command['addr'], command['port'])
                    if neighbor_info not in self.neighbors:
                        self.neighbors.append(neighbor_info)
                        logging.debug(f"Connected to {neighbor_info[0]}:{neighbor_info[1]}")

                # elif command['type'] == 'model_update':
                #     """
                #     Tester receives 'model_update' message. 
                #     """
                #     received_model_state = command['model']
                #     trainer_sender = (command['addr'], command['port'])

                #     self.received_models.append({'model': received_model_state, 'sender': trainer_sender})
                #     logging.debug(f"[{self.addr}:{self.port}] Received model update from {trainer_sender[0]}:{trainer_sender[1]} ...")
                    
                #     send_echo(self.key_server, self.__private_key, received_model_state, trainer_sender[0], trainer_sender[1], self.addr, self.port)
                elif command['type'] == 'model_update':
                    """
                    Tester receives 'model_update' message from the trainer.
                    """
                    serialized_model_update = command['model']  # Serialized model update
                    trainer_sender = (command['addr'], command['port'])

                    # Deserialize the received model update
                    received_model_update = pickle.loads(serialized_model_update)

                    # Store the received model update in the tester's state
                    self.received_models.append({'model': received_model_update, 'sender': trainer_sender})
                    logging.debug(f"[{self.addr}:{self.port}] Received model update from {trainer_sender[0]}:{trainer_sender[1]} ...")
                    
                    # Send the echo message back to the trainer
                    send_echo(self.key_server, self.__private_key, serialized_model_update, trainer_sender[0], trainer_sender[1], self.addr, self.port)
                elif command['type'] == 'echo':
                    signature = command['signature']
                    sender_addr = command['addr']
                    sender_port = command['port']
                    serialized_state_from_echo = command['serialized_state']  # The serialized update from echo
                    if self.local_update is None:
                        print("NONE local update")
                        return
                    # Use the exact serialized local update (previously sent) for verification
                    if verify_signature(self.key_server, sender_addr, sender_port, self.local_update, signature):
                        logging.info(f"[{self.addr}:{self.port}] Signature verified for echo from {sender_addr}:{sender_port}")
                        self.received_echo_cnt += 1
                        self.signature_list.append(signature)
                        self.sender_list.append({'addr': sender_addr, 'port': sender_port})  
                    else:
                        logging.warning(f"[{self.addr}:{self.port}] Signature verification failed for echo from {sender_addr}:{sender_port}")
                    
                    # Check if enough echo messages have been received to proceed
                    # TODO: replace the number to quorum
                    if self.received_echo_cnt >= 4 and self.sent_ready is False:
                        send_ready(self.signature_list, self.testers_list, self.addr, self.port, self.sender_list, self.local_update)
                        self.sent_ready = True
                
                elif command['type'] == 'ready':
                    # Received a 'ready' message from another node (trainer)
                    signature_list = command['signature_list']
                    sender_addr = command['addr']   # Trainer's addr
                    sender_port = command['port']   # Trainer's port
                    sender_list = command['sender_list']  # List of senders who sent echo messages
                    local_update = command['local_update']
                    logging.debug(f"[{self.addr}:{self.port}] Received Ready Message from {sender_addr}:{sender_port} ...")
                    
                    # Print the content of sender_list and self.testers_list for debugging purposes
                    logging.debug(f"[{self.addr}:{self.port}] Sender_list content: {sender_list}")
                    logging.debug(f"[{self.addr}:{self.port}] Testers_list content: {[(tester.addr, tester.port) for tester in self.testers_list]}")

                    # Verifying each signature in the signature_list
                    if len(signature_list) != len(sender_list):
                        logging.error(f"[{self.addr}:{self.port}] Mismatch between signature_list and sender_list lengths!")
                        return

                    for i, sender in enumerate(sender_list):
                        # Ensure we are verifying the correct signature for each sender
                        sender_addr = sender['addr']
                        sender_port = sender['port']
                        signature = signature_list[i]

                        # Verify the signature using the appropriate public key and the serialized update
                        if local_update is None:
                            logging.error(f"[{self.addr}:{self.port}] Local update is None, cannot verify signatures.")
                            return

                        # Use the serialized local update for verification (replace with the correct data if needed)
                        # serialized_local_update = pickle.dumps(self.local_update)

                        # Verify the signature
                        if verify_signature(self.key_server, sender_addr, sender_port, local_update, signature):
                            logging.debug(f"[{self.addr}:{self.port}] Signature verified for ready from {sender_addr}:{sender_port}")
                            self.received_ready_cnt += 1
                        else:
                            logging.warning(f"[{self.addr}:{self.port}] Signature verification failed for ready from {sender_addr}:{sender_port}")

                    # If enough 'ready' messages are received, proceed to send the sup message
                    if self.received_ready_cnt >= 4 and self.sent_sup is False:
                        for tester in self.testers_list:
                            tester_identity = {'addr': tester.addr, 'port': tester.port}

                            # Compare the tester's identity with entries in sender_list
                            if tester_identity not in sender_list:
                                # Tester is not in sender_list, send 'sup' with self.model_state
                                logging.debug(f"[{self.addr}:{self.port}] Sending sup with self.model_state to {tester.addr}:{tester.port}")
                                send_sup(signature_list, local_update, self.addr, self.port, tester.addr, tester.port)
                            else:
                                # Tester is in sender_list, send 'sup' with None
                                logging.debug(f"[{self.addr}:{self.port}] Sending sup with None to {tester.addr}:{tester.port}")
                                send_sup(signature_list, None, self.addr, self.port, tester.addr, tester.port)
                        self.sent_sup = True
             
                elif command['type'] == 'sup':
                    signature_list = command['signature_list']
                    sender_addr = command['addr']   # Tester's addr
                    sender_port = command['port']   # Tester's port
                    model_update = command['model_update']  # The model update being delivered
                    logging.debug(f"[{self.addr}:{self.port}] Received sup message from {sender_addr}:{sender_port} ...")

                    self.received_sup_cnt += 1
                    f = (len(self.testers_list) - 1) // 3  # Calculate f based on the system size (3f+1 model)

                    # self.received_models.append({'model': model_update, 'sender': (trainer_sender)})

                    ## TODO: If the number of received sup msgs is no more than 2f+1, resend sup msg. 
                    if self.received_sup_cnt > 2*f + 1:
                        logging.debug(f"[{self.addr}:{self.port}] Delivered!!")
                        self.delivered = True
                        self.set_delivered_flag()
                   
                elif command['type'] == 'global_model_update':
                    global_model_state = command['model']
                    self.model.load_state_dict(global_model_state)
                    logging.debug(f"[{self.addr}:{self.port}] Received global model update from {command['addr']}:{command['port']} ...")
        except Exception as e:
            logging.error(f"Error handling connection: {e}")
        finally:
            conn.close()

    def connect(self, node):
        addr, port = node.addr, node.port
        if node in self.testers_list or node in self.trainers_list:
            logging.debug(f"[{self.addr}:{self.port}] Already connected to {addr}:{port}, skipping connection.")
            return
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((addr, port))
                s.sendall(pickle.dumps({'type': 'connect', 'addr': self.addr, 'port': self.port}))
                self.neighbors.append(node)
                logging.debug(f"[{self.addr}:{self.port}] Connected to {addr}:{port}")
        except Exception as e:
            logging.error(f"[{self.addr}:{self.port}] Error connecting to {addr}:{port}: {e}")

    def send_model_to_testers(self):
        # Get the current model state
        current_model_state = self.model.state_dict()

        # Initialize local update dictionary
        local_update = {}

        # If this is the first round and previous_model_state is None, we initialize it
        if self.previous_model_state is None:
            logging.debug(f"[{self.addr}:{self.port}] First round, sending full model state as local update.")
            local_update = {key: current_model_state[key] for key in current_model_state}
        else:
            # Compute the local update as the difference between the current and previous model states
            for key in current_model_state:
                local_update[key] = current_model_state[key] - self.previous_model_state[key]

        # Store the current model state as the previous one for the next round
        self.previous_model_state = {key: value.clone() for key, value in current_model_state.items()}

        # Serialize the local update only once and store it
        serialized_update = pickle.dumps(local_update)
        self.local_update = serialized_update  # Store the serialized update for later use

        # Send the serialized local update to testers
        data = pickle.dumps({'type': 'model_update', 'model': serialized_update, 'addr': self.addr, 'port': self.port})
        msg_len = len(data)

        for tester in self.testers_list:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((tester.addr, tester.port))
                s.sendall(msg_len.to_bytes(4, byteorder='big'))
                s.sendall(data)
                logging.debug(f"Sent local update to {tester.addr}:{tester.port}")
            # Update the previous model state to the current state after sending
        # self.previous_model_state = current_model_state.copy()
    # def send_model_to_testers(self):
    #     model_state = self.model.state_dict()
    #     # Serialize the model state immediately after update
    #     serialized_state = pickle.dumps(model_state)
    #     self.serialized_state = serialized_state  # Store the serialized state
    #     self.received_models.append({'model': model_state, 'sender': (self.addr, self.port)})       # Tester stores its own trained model update for future verification
    #     data = pickle.dumps({'type': 'model_update', 'model': model_state, 'addr': self.addr, 'port': self.port})
    #     msg_len = len(data)
    #     for tester in self.testers_list:
    #         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #             s.connect((tester.addr, tester.port))
    #             s.sendall(msg_len.to_bytes(4, byteorder='big'))
    #             s.sendall(data)
    #             logging.debug(f"Sent model update to {tester.addr}:{tester.port}")

    def testing(self):
        aggregate_models(self)
        return evaluate(self)

    # def broadcast_global_model_update(self):
    #     broadcast_global_model_update(self)

    def set_start_learning(self, rounds=1, epochs=1, threshold=1e-3):
        self.running = True
        round_avg_loss = train(self, epochs)

        self.send_model_to_testers()

