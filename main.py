import threading
from node import Node
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch

NUM_CLIENTS = 3
TRAINING_ROUNDS = 3
TRAINING_EPOCHS = 5

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

def exclude_digits(dataset, excluded_digits):
    including_indices = [
        idx for idx in range(len(dataset)) if dataset[idx][1] not in excluded_digits
    ]
    return torch.utils.data.Subset(dataset, including_indices)

# Data manager for MNIST
def load_data(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_set = datasets.MNIST(root='MNIST_data', train=True, download=False, transform=transform)

    total_length = len(data_set)
    split_size = total_length // NUM_CLIENTS
    torch.manual_seed(42)
    part1, part2, part3 = random_split(data_set, [split_size] * NUM_CLIENTS)

    part1 = exclude_digits(part1, excluded_digits=[0, 1, 2, 3])
    part2 = exclude_digits(part2, excluded_digits=[4, 5, 6])
    part3 = exclude_digits(part3, excluded_digits=[7, 8, 9])

    train_loader_part1 = DataLoader(part1, batch_size=batch_size, shuffle=True)
    train_loader_part2 = DataLoader(part2, batch_size=batch_size, shuffle=True)
    train_loader_part3 = DataLoader(part3, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)

    return [test_loader, train_loader_part1, train_loader_part2, train_loader_part3]

def start_learning_concurrently(node, rounds, epochs):
    node.set_start_learning(rounds=rounds, epochs=epochs)

def main():
    test_loader, train_loader1, train_loader2, train_loader3 = load_data()

    # Initialize nodes with the MLP model and MNIST data loaders, with different ports
    node1 = Node(
        model=MLP(),
        data=train_loader1,
        test_data=test_loader,
        addr="127.0.0.1",
        port=5001
    )
    
    node2 = Node(
        model=MLP(),
        data=train_loader2,
        test_data=test_loader,
        addr="127.0.0.1",
        port=5002
    )

    node3 = Node(
        model=MLP(),
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

    for t in threads:
        t.join()

    # Evaluate the final aggregated model
    # node1.evaluate()
    # node2.evaluate()
    # node3.evaluate()

if __name__ == "__main__":
    main()