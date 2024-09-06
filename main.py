import threading
import random
from p2pdl.node.node import Node
# from p2pdl.node.tester import Tester
# from p2pdl.node.trainer import Trainer
from p2pdl.models.model import SimpleCNN, MLP
from p2pdl.datasets.dataset import load_data 
from p2pdl.utils.crypto import KeyServer
# 
NUM_CLIENTS = 7
TRAINING_ROUNDS = 10
TRAINING_EPOCHS = 5

def start_learning_concurrently(node, rounds, epochs):
    node.set_start_learning(rounds=rounds, epochs=epochs)

def main():
    key_server = KeyServer()
    data_loaders = load_data(NUM_CLIENTS, dataset_name='MNIST')

    # Initialize nodes with the model and data loaders, with different ports
    # Number of testers n >= 3f + 1
    node1 = Node(model=MLP(),data=data_loaders[0], key_server=key_server, addr="127.0.0.1",port=6001)
    node2 = Node(model=MLP(),data=data_loaders[1], key_server=key_server, addr="127.0.0.1",port=6002)
    node3 = Node(model=MLP(),data=data_loaders[2], key_server=key_server, addr="127.0.0.1",port=6003)
    node4 = Node(model=MLP(),data=data_loaders[3], key_server=key_server, addr="127.0.0.1",port=6004)
    node5 = Node(model=MLP(),data=data_loaders[4], key_server=key_server, addr="127.0.0.1",port=6005)
    node6 = Node(model=MLP(),data=data_loaders[5], key_server=key_server, addr="127.0.0.1",port=6006)
    node7 = Node(model=MLP(),data=data_loaders[6], key_server=key_server, addr="127.0.0.1",port=6007)
    
    for node in [node1, node2, node3, node4, node5, node6, node7]:
        node.start()

    nodes = [node1, node2, node3, node4, node5, node6, node7]

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
        print(f"testers {selected_testers[0].port, selected_testers[1].port, selected_testers[2].port, selected_testers[3].port}")
        # Assign roles and lists
        for trainer in selected_trainers:
            trainer.trainers_list = selected_trainers
            trainer.testers_list = selected_testers
        
        for tester in selected_testers:
            tester.trainers_list = selected_trainers
            tester.testers_list = selected_testers

        for node in nodes:
            node.reset_delivered_flag()

        # Start training concurrently for trainers
        threads = []
        for trainer in selected_trainers:
            t = threading.Thread(target=start_learning_concurrently, args=(trainer, 1, TRAINING_EPOCHS))
            t.start()
            threads.append(t)

        # Wait for all trainers to complete the round
        for t in threads:
            t.join()

        # Wait for brb delivered
        for tester in selected_testers:
            tester.wait_for_delivered()

        # Testers evaluate the aggregated global model and broadcast results to all nodes
        for tester in selected_testers:
            tester.testing()

        print(f"Round {training_round + 1} completed\n")

    for node in [node1, node2, node3, node4, node5, node6, node7]:
        node.stop()
    
if __name__ == "__main__":
    main()