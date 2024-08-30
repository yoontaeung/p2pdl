import threading
import random
from p2pdl.node.node import Node
from p2pdl.node.tester import Tester
from p2pdl.node.trainer import Trainer
from p2pdl.models.model import SimpleCNN  
from p2pdl.datasets.dataset import load_data 
from p2pdl.evaluation.evaluation import evaluate

NUM_CLIENTS = 5
TRAINING_ROUNDS = 3
TRAINING_EPOCHS = 3

def start_learning_concurrently(node, rounds, epochs):
    node.set_start_learning(rounds=rounds, epochs=epochs)

def main():
    data_loaders = load_data(NUM_CLIENTS, dataset_name='CIFAR10')

    # Initialize nodes with the model and data loaders, with different ports
    node1 = Node(model=SimpleCNN(),data=data_loaders[0],role=None,addr="127.0.0.1",port=5001)
    node2 = Node(model=SimpleCNN(),data=data_loaders[1],role=None,addr="127.0.0.1",port=5002)
    node3 = Node(model=SimpleCNN(),data=data_loaders[2],role=None,addr="127.0.0.1",port=5003)
    node4 = Node(model=SimpleCNN(),data=data_loaders[3],role=None,addr="127.0.0.1",port=5004)
    node5 = Node(model=SimpleCNN(),data=data_loaders[4],role=None,addr="127.0.0.1",port=5005)
    
    for node in [node1, node2, node3, node4, node5]:
        node.start()

    nodes = [node1, node2, node3, node4, node5]

    for training_round in range(TRAINING_ROUNDS):
        print(f"Starting round {training_round + 1}")
        
        selected_trainers = random.sample(nodes, 3)
        remaining_nodes = [node for node in nodes if node not in selected_trainers]
        selected_testers = random.sample(remaining_nodes, 2)

        for trainer in selected_trainers:
            trainer.role = 'trainer'
            trainer.trainers_list = selected_trainers
            trainer.testers_list = selected_testers
        
        for tester in selected_testers:
            tester.role = 'tester'
            tester.trainers_list = selected_trainers
            tester.testers_list = selected_testers

        ### Each trainer makes a connection with testers. Trainers will not be connected with each other. 
        for trainer in selected_trainers:
            for tester in selected_testers:
                trainer.connect(tester)

        threads = []
        for trainer in selected_trainers:
            t = threading.Thread(target=start_learning_concurrently, args=(trainer, 1, TRAINING_EPOCHS))
            t.start()
            threads.append(t)

        # Wait for the round to complete
        for t in threads:
            t.join()

        for tester in selected_testers:
            tester.evaluate()

        print(f"Round {training_round + 1} completed\n")

        # Disconnect all nodes after the round
        for node in selected_trainers:
            node.disconnect_all()
        # for node in selected_testers:
        #     node.disconnect_all()

if __name__ == "__main__":
    main()