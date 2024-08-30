import logging
import threading

def aggregate_models(self):
    while len(self.received_models) < len(self.trainers_list):
        threading.Event().wait(1)  # Small wait to avoid busy-waiting

    logging.info(f"[{self.addr}:{self.port}] Aggregating local model updates ...")
    num_models = len(self.received_models)  
    for key in self.model.state_dict():
        avg_param = self.model.state_dict()[key].clone()
        for received_model in self.received_models:
            avg_param += received_model[key]
        avg_param /= num_models
        self.model.state_dict()[key].copy_(avg_param)
    logging.debug(f"[{self.addr}:{self.port}] Model aggregation completed ...")