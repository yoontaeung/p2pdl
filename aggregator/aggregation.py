import logging
import pickle
import socket
import torch
from p2pdl.utils.waiting import wait_for_models

def aggregate_models(self):
    # Ensure that we wait until all expected models (local updates) are received or timeout occurs.
    if not wait_for_models(self.received_models, len(self.trainers_list)):
        logging.warning(f"[{self.addr}:{self.port}] Proceeding with aggregation despite incomplete models.")

    logging.debug(f"[{self.addr}:{self.port}] Aggregating local model updates ...")
    
    # Initialize a dictionary to hold the accumulated updates (initialize with zeros)
    accumulated_updates = {key: torch.zeros_like(param) for key, param in self.model.state_dict().items()}
    
    # Count the number of received models (local updates)
    num_updates = len(self.received_models)
    
    if num_updates == 0:
        logging.error(f"[{self.addr}:{self.port}] No updates received to aggregate!")
        return
    
    # Accumulate the local updates
    for received_model in self.received_models:
        local_update = received_model['model']
        for key in accumulated_updates:
            accumulated_updates[key] += local_update[key]  # Add the local updates

    # Average the accumulated updates by dividing by the number of updates
    for key in accumulated_updates:
        accumulated_updates[key] /= num_updates

    # Apply the averaged updates to the current model parameters
    # Instead of directly adding the accumulated update, use a learning rate to scale the update
    learning_rate = 0.1  # Adjust this as necessary
    for key in self.model.state_dict():
        self.model.state_dict()[key] += learning_rate * accumulated_updates[key]

    logging.info(f"[{self.addr}:{self.port}] Model aggregation completed, applied local updates.")

    # Clear received models for the next round
    self.received_models.clear()

    # Broadcast the newly aggregated global model
    broadcast_global_model_update(self)

# def aggregate_models(self):
#     # Testers aggregate the local model updates from trainers.
#     # Wait until all expected models are received or timeout occurs
#     if not wait_for_models(self.received_models, len(self.trainers_list)):
#         logging.warning(f"[{self.addr}:{self.port}] Proceeding with aggregation despite incomplete models.")

#     logging.debug(f"[{self.addr}:{self.port}] Aggregating local model updates ...")
#     num_models = len(self.received_models)  
#     for key in self.model.state_dict():
#         avg_param = self.model.state_dict()[key].clone()
#         for received_model in self.received_models:
#             avg_param += received_model['model'][key]
#         avg_param /= num_models
#         self.model.state_dict()[key].copy_(avg_param)
#     logging.debug(f"[{self.addr}:{self.port}] Model aggregation completed ...")

#     broadcast_global_model_update(self)

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