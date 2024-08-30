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
from p2pdl.node.node import Node

# Set up logging
logging.basicConfig(level=logging.INFO)

class Trainer(Node):
    def set_start_learning(self, rounds=1, epochs=1):
        self.running = True
        for _ in range(rounds):
            logging.info(f"====================== Round {_ + 1} at {self.addr}:{self.port} begin... ======================")
            round_avg_loss = train(self, epochs)  

            logging.debug(f"[{self.addr}:{self.port}] Waiting for model updates from neighbors...")
            
            while len(self.received_models) < len(self.neighbors):
                threading.Event().wait(1)  # Small wait to avoid busy-waiting

            logging.debug(f"[{self.addr}:{self.port}] Received all model updates ...")
            
            aggregate_models(self)  

            self.received_models.clear()

            evaluate(self)  

            model_state_dict = {key: value.tolist() for key, value in self.model.state_dict().items()}

            # Prepare the result data to be saved
            result_data = {
                "node": self.addr,
                "port": self.port,
                "round": _ + 1,
                "average_loss": round_avg_loss,
                "model_state": model_state_dict,
                "trainers": [trainer.addr for trainer in self.trainers_list],
                "testers": [tester.addr for tester in self.testers_list],
            }
            
            # Create a unique result file name based on the port number
            result_file = f"results_{self.port}.json"
            
            # Save the result to a JSON file
            save_results(result_data, result_file)
            
            logging.info(f"====================== Round {_ + 1} at {self.addr}:{self.port} complete... ======================")

        self.stop()