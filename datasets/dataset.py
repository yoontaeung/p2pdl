import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_mnist_data(num_clients, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_set = datasets.MNIST(root='MNIST_data', train=True, download=True, transform=transform)

    total_length = len(data_set)
    split_size = total_length // num_clients
    split_sizes = [split_size] * num_clients
    split_sizes[-1] += total_length - sum(split_sizes)  # Adjust the last split size to cover any remainder

    torch.manual_seed(42)
    parts = random_split(data_set, split_sizes)

    data_loaders = [DataLoader(part, batch_size=batch_size, shuffle=True) for part in parts]

    return data_loaders

def load_cifar10_data(num_clients, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_set = datasets.CIFAR10(root='CIFAR10_data', train=True, download=True, transform=transform)

    total_length = len(data_set)
    split_size = total_length // num_clients
    split_sizes = [split_size] * num_clients
    split_sizes[-1] += total_length - sum(split_sizes)  # Adjust the last split size to cover any remainder

    torch.manual_seed(42)
    parts = random_split(data_set, split_sizes)

    data_loaders = [DataLoader(part, batch_size=batch_size, shuffle=True) for part in parts]

    return data_loaders

def load_data(num_clients, dataset_name='CIFAR10', batch_size=32):
    if dataset_name == 'MNIST':
        return load_mnist_data(num_clients, batch_size)
    elif dataset_name == 'CIFAR10':
        return load_cifar10_data(num_clients, batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")