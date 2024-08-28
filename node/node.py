import socket
import threading
import pickle
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils

# Import the necessary functions from other modules
from p2pdl.training.train import train
from p2pdl.aggregator.aggregation import aggregate_models
from p2pdl.evaluation.evaluation import evaluate
from p2pdl.utils.log import save_results

# Set up logging
logging.basicConfig(level=logging.INFO)

class Node:
    def __init__(self, model, data, test_data, addr="127.0.0.1", port=12345):
        self.model = model
        self.data_loader = data
        self.test_data_loader = test_data
        self.addr = addr
        self.port = port
        self.neighbors = []  # Stores tuples of (addr, port)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)  # Make sure this is initialized
        self.loss_fn = nn.CrossEntropyLoss()
        self.running = False
        self.received_models = []
        self._stop_event = threading.Event()  # Event to signal when to stop listening

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
                        logging.info(f"Connected to {neighbor_info[0]}:{neighbor_info[1]}")
                elif command['type'] == 'model_update':
                    received_model_state = command['model']
                    self.received_models.append(received_model_state)
                    logging.debug(f"[{self.addr}:{self.port}] Received model update from {command['addr']}:{command['port']} ...")
        except Exception as e:
            logging.error(f"Error handling connection: {e}")
        finally:
            conn.close()

    def connect(self, addr, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((addr, port))
            s.sendall(pickle.dumps({'type': 'connect', 'addr': self.addr, 'port': self.port}))
            self.neighbors.append((addr, port))
            logging.info(f"[{self.addr}:{self.port}] Connected to {addr}:{port}")

    def send_model_to_neighbors(self):
        model_state = self.model.state_dict()
        data = pickle.dumps({'type': 'model_update', 'model': model_state, 'addr': self.addr, 'port': self.port})
        msg_len = len(data)
        for neighbor in self.neighbors:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(neighbor)
                s.sendall(msg_len.to_bytes(4, byteorder='big'))
                s.sendall(data)
                logging.debug(f"Sent model update to {neighbor[0]}:{neighbor[1]}")

    def set_start_learning(self, rounds=1, epochs=1, threshold=1e-3):
        self.running = True
        previous_model_state = None

        for round_num in range(1, rounds + 1):
            logging.info(f"====================== Round {round_num} at {self.addr}:{self.port} begin... ======================")
            round_avg_loss = train(self, epochs)

            logging.debug(f"[{self.addr}:{self.port}] Waiting for model updates from neighbors...")

            while len(self.received_models) < len(self.neighbors):
                threading.Event().wait(1)  # Small wait to avoid busy-waiting

            logging.debug(f"[{self.addr}:{self.port}] Received all model updates ...")
            
            aggregate_models(self)

            self.received_models.clear()

            evaluate(self)

            # Calculate gradient divergence and compress the model state
            current_model_state = self.model.state_dict()
            compressed_model_state = {}
            divergence_info = {}

            if previous_model_state is not None:
                for key in current_model_state:
                    difference = torch.abs(current_model_state[key] - previous_model_state[key])

                    divergence = torch.norm(difference).item()
                    divergence_info[key] = divergence

                    if divergence > threshold:
                        compressed_model_state[key] = current_model_state[key].tolist()
                        previous_model_state[key] = current_model_state[key].clone()
            else:
                compressed_model_state = {key: value.tolist() for key, value in current_model_state.items()}
                previous_model_state = current_model_state.copy()

            result_data = {
                "node": self.addr,
                "port": self.port,
                "round": round_num,
                "average_loss": round_avg_loss,
                "model_state": compressed_model_state,
                "divergence": divergence_info  
            }

            result_file = f"results_{self.port}.json"
            
            # save_results(result_data, result_file)
            
            logging.info(f"====================== Round {round_num} at {self.addr}:{self.port} complete... ======================")

        self.stop()