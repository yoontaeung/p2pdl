import threading
from p2pdl.node.node import Node
from p2pdl.models.model import SimpleCNN  
from p2pdl.datasets.dataset import load_data 

NUM_CLIENTS = 3
TRAINING_ROUNDS = 3
TRAINING_EPOCHS = 3

def start_learning_concurrently(node, rounds, epochs):
    node.set_start_learning(rounds=rounds, epochs=epochs)

def main():
    test_loader, train_loader1, train_loader2, train_loader3 = load_data(NUM_CLIENTS)

    # Initialize nodes with the model and data loaders, with different ports
    node1 = Node(
        model=SimpleCNN(),
        data=train_loader1,
        test_data=test_loader,
        addr="127.0.0.1",
        port=5001
    )
    
    node2 = Node(
        model=SimpleCNN(),
        data=train_loader2,
        test_data=test_loader,
        addr="127.0.0.1",
        port=5002
    )

    node3 = Node(
        model=SimpleCNN(),
        data=train_loader3,
        test_data=test_loader,
        addr="127.0.0.1",
        port=5003
    )
    
    node1.start()
    node2.start()
    node3.start()

    node1.connect("127.0.0.1", 5002)
    node1.connect("127.0.0.1", 5003)

    node2.connect("127.0.0.1", 5001)
    node2.connect("127.0.0.1", 5003)

    node3.connect("127.0.0.1", 5001)
    node3.connect("127.0.0.1", 5002)

    # Training for multiple rounds (evaluation included)
    threads = []
    for node in [node1, node2, node3]:
        t = threading.Thread(target=start_learning_concurrently, args=(node, TRAINING_ROUNDS, TRAINING_EPOCHS))
        t.start()
        threads.append(t)

    # Wait for all threads to complete
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()