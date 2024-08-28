import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def exclude_digits(dataset, excluded_digits):
    including_indices = [
        idx for idx in range(len(dataset)) if dataset[idx][1] not in excluded_digits
    ]
    return torch.utils.data.Subset(dataset, including_indices)

def load_mnist_data(num_clients, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_set = datasets.MNIST(root='MNIST_data', train=True, download=False, transform=transform)

    total_length = len(data_set)
    split_size = total_length // num_clients
    torch.manual_seed(42)
    parts = random_split(data_set, [split_size] * num_clients)

    train_loaders = [DataLoader(exclude_digits(part, excluded_digits=[i for i in range(3 * idx, 3 * (idx + 1))]), batch_size=batch_size, shuffle=True) for idx, part in enumerate(parts)]
    test_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)

    return [test_loader] + train_loaders

def load_cifar10_data(num_clients, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_set = datasets.CIFAR10(root='CIFAR10_data', train=True, download=True, transform=transform)

    total_length = len(data_set)
    split_size = total_length // num_clients
    split_sizes = [split_size] * num_clients
    split_sizes[0] += total_length - sum(split_sizes)  # Adjust the first split size to cover any remainder

    torch.manual_seed(42)
    parts = random_split(data_set, split_sizes)

    train_loaders = [DataLoader(part, batch_size=batch_size, shuffle=True) for part in parts]
    test_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)

    return [test_loader] + train_loaders

def load_data(num_clients, dataset_name='CIFAR10', batch_size=32):
    if dataset_name == 'MNIST':
        return load_mnist_data(num_clients, batch_size)
    elif dataset_name == 'CIFAR10':
        return load_cifar10_data(num_clients, batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")