import threading
import random
from p2pdl.node.node import Node
# from p2pdl.node.tester import Tester
# from p2pdl.node.trainer import Trainer
from p2pdl.models.model import SimpleCNN  
from p2pdl.datasets.dataset import load_data 
# 
NUM_CLIENTS = 5
TRAINING_ROUNDS = 10
TRAINING_EPOCHS = 5

def start_learning_concurrently(node, rounds, epochs):
    node.set_start_learning(rounds=rounds, epochs=epochs)

def main():
    data_loaders = load_data(NUM_CLIENTS, dataset_name='CIFAR10')

    # Initialize nodes with the model and data loaders, with different ports
    node1 = Node(model=SimpleCNN(),data=data_loaders[0], addr="127.0.0.1",port=5001)
    node2 = Node(model=SimpleCNN(),data=data_loaders[1], addr="127.0.0.1",port=5002)
    node3 = Node(model=SimpleCNN(),data=data_loaders[2], addr="127.0.0.1",port=5003)
    node4 = Node(model=SimpleCNN(),data=data_loaders[3], addr="127.0.0.1",port=5004)
    node5 = Node(model=SimpleCNN(),data=data_loaders[4], addr="127.0.0.1",port=5005)
    
    for node in [node1, node2, node3, node4, node5]:
        node.start()

    nodes = [node1, node2, node3, node4, node5]

    for node in nodes:
        for other_node in nodes:
            if node != other_node:
                node.connect(other_node)

    # Training and evaluation for multiple rounds
    for training_round in range(TRAINING_ROUNDS):
        # Random client selection (trainers, testers)
        selected_trainers = random.sample(nodes, 3)
        remaining_nodes = [node for node in nodes if node not in selected_trainers]
        selected_testers = remaining_nodes
        print(f"Starting round {training_round + 1}") 
        print(f"trainers {selected_trainers[0].port, selected_trainers[1].port, selected_trainers[2].port}")
        print(f"testers {selected_testers[0].port, selected_testers[1].port}")
        # Assign roles and lists
        for trainer in selected_trainers:
            trainer.trainers_list = selected_trainers
            trainer.testers_list = selected_testers
        
        for tester in selected_testers:
            tester.trainers_list = selected_trainers
            tester.testers_list = selected_testers

        # Start training concurrently for trainers
        threads = []
        for trainer in selected_trainers:
            t = threading.Thread(target=start_learning_concurrently, args=(trainer, 1, TRAINING_EPOCHS))
            t.start()
            threads.append(t)

        # Wait for all trainers to complete the round
        for t in threads:
            t.join()

        # Testers evaluate the aggregated global model and broadcast results to all nodes
        for tester in selected_testers:
            tester.testing()

        print(f"Round {training_round + 1} completed\n")

    for node in [node1, node2, node3, node4, node5]:
        node.stop()
    
if __name__ == "__main__":
    main()