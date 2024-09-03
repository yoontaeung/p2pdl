import socket
import threading
import pickle
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils

from p2pdl.training.train import train
from p2pdl.aggregator.aggregation import aggregate_models, broadcast_global_model_update
from p2pdl.evaluation.evaluation import evaluate
from p2pdl.utils.log import save_results
from p2pdl.utils.broadcast import send_echo, send_ready
from p2pdl.utils.crypto import KeyServer, generate_key_pair, verify_signature

logging.basicConfig(level=logging.INFO)

class Node:
    def __init__(self, model, data, key_server, addr="127.0.0.1", port=12345):
        self.model = model
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
        self.serialized_state = None

        self.__private_key, self.public_key = generate_key_pair()
        self.key_server = key_server
        self.key_server.register_key(self.addr, self.port, self.public_key)

        self.brb_delivered_event = threading.Event()
        self.received_echo_cnt = 0
        self.sent_ready = False
        self.received_ready_cnt = 0

    def reset_delivered_flag(self):
        self.brb_delivered_event.clear()

    def set_delivered_flag(self):
        self.brb_delivered_event.set()

    def wait_for_delivered(self):
        logging.info(f"[{self.addr}:{self.port}] Waiting for all echo and ready messages delivered ...")
        self.brb_delivered_event.wait()
        logging.info(f"[{self.addr}:{self.port}] All messages delivered, proceeding...")

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

                elif command['type'] == 'model_update':
                    """
                    Tester receives 'model_update' message. 
                    """
                    received_model_state = command['model']
                    trainer_sender = (command['addr'], command['port'])

                    self.received_models.append({'model': received_model_state, 'sender': trainer_sender})
                    logging.debug(f"[{self.addr}:{self.port}] Received model update from {trainer_sender[0]}:{trainer_sender[1]} ...")
                    
                    send_echo(self.key_server, self.__private_key, received_model_state, trainer_sender[0], trainer_sender[1], self.addr, self.port)

                elif command['type'] == 'echo':
                    signature = command['signature']
                    sender_addr = command['addr']
                    sender_port = command['port']
                    serialized_state = command['serialized_state']
                    # model_state = None
                    # for entry in self.received_models:
                    #     if entry['sender'] == (self.addr, self.port):
                    #         model_state = entry['model']
                    #         break
                    model_state = self.model.state_dict()

                    ## TODO: Use a local model update for verification, not from echo message
                    if model_state is None:
                        print("Model state is None")
                    if verify_signature(self.key_server, sender_addr, sender_port, serialized_state, signature):
                        logging.info(f"[{self.addr}:{self.port}] Signature verified for echo from {sender_addr}:{sender_port}")
                        self.received_echo_cnt += 1
                    else:
                        logging.warning(f"[{self.addr}:{self.port}] Signature verification failed for echo from {sender_addr}:{sender_port}")
                    ## TODO: Complete the logic
                    if self.received_echo_cnt >= 3 and self.sent_ready is False:
                        send_ready()
                # elif command['type'] == 'echo':
                #     signature = command['signature']
                #     tester_addr = command['addr']
                #     tester_port = command['port']

                #     # Find the model state corresponding to the current node's identity in self.received_models
                #     model_state = None
                #     for entry in self.received_models:
                #         if entry['sender'] == (self.addr, self.port):
                #             model_state = entry['model']
                #             break

                #     # logging.info(f"[{self.addr}:{self.port}] received models {self.received_models}")
                #     logging.info(f"[{self.addr}:{self.port}] Who sends ECHO message? -> {tester_addr}:{tester_port}")

                #     if model_state is None:
                #         print("Model is None")
                #     elif verify_signature(self.key_server, tester_addr, tester_port, model_state, signature):
                #         logging.info(f"[{self.addr}:{self.port}] Signature verified for echo from {tester_addr}:{tester_port}")
                #     else:
                #         logging.warning(f"[{self.addr}:{self.port}] Signature verification failed for echo from {tester_addr}:{tester_port}")
                ## elif command['type'] == 'sup':
                    ## if the sup messages are delivered more than 2f+1, then
                    ## set_delivered_flag()
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
        model_state = self.model.state_dict()
        # Serialize the model state immediately after update
        serialized_state = pickle.dumps(model_state)
        self.serialized_state = serialized_state  # Store the serialized state
        self.received_models.append({'model': model_state, 'sender': (self.addr, self.port)})       # Tester stores its own trained model update for future verification
        data = pickle.dumps({'type': 'model_update', 'model': model_state, 'addr': self.addr, 'port': self.port})
        msg_len = len(data)
        for tester in self.testers_list:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((tester.addr, tester.port))
                s.sendall(msg_len.to_bytes(4, byteorder='big'))
                s.sendall(data)
                logging.debug(f"Sent model update to {tester.addr}:{tester.port}")

    def testing(self):
        aggregate_models(self)
        evaluate(self)

    # def broadcast_global_model_update(self):
    #     broadcast_global_model_update(self)

    def set_start_learning(self, rounds=1, epochs=1, threshold=1e-3):
        self.running = True
        previous_model_state = None

        logging.debug(f"=== Round at {self.addr}:{self.port} begin... ===")
        round_avg_loss = train(self, epochs)

        self.send_model_to_testers()

        logging.debug(f"=== Round  at {self.addr}:{self.port} complete... ===")
