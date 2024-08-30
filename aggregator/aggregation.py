import logging
import pickle
import socket
from p2pdl.utils.waiting import wait_for_models

def aggregate_models(self):
    # Testers aggregate the local model updates from trainers.
    # Wait until all expected models are received or timeout occurs
    if not wait_for_models(self.received_models, len(self.trainers_list)):
        logging.warning(f"[{self.addr}:{self.port}] Proceeding with aggregation despite incomplete models.")

    logging.debug(f"[{self.addr}:{self.port}] Aggregating local model updates ...")
    num_models = len(self.received_models)  
    for key in self.model.state_dict():
        avg_param = self.model.state_dict()[key].clone()
        for received_model in self.received_models:
            avg_param += received_model[key]
        avg_param /= num_models
        self.model.state_dict()[key].copy_(avg_param)
    logging.debug(f"[{self.addr}:{self.port}] Model aggregation completed ...")

    self.broadcast_global_model_update()

def broadcast_global_model_update(self):
    # Broadcast the aggregated model to all peers. 

    model_state = self.model.state_dict()
    data = pickle.dumps({'type': 'global_model_update', 'model': model_state, 'addr': self.addr, 'port': self.port})
    msg_len = len(data)
    for neighbor in self.neighbors:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((neighbor.addr, neighbor.port))
            s.sendall(msg_len.to_bytes(4, byteorder='big'))
            s.sendall(data)
            logging.debug(f"Broadcasted global model update to {neighbor.addr}:{neighbor.port}")