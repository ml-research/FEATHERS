from typing import Tuple, Union, List
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torch

class FashionMNISTLoader:

    _instance = None

    def __init__(self, n_clients) -> None:
        if FashionMNISTLoader._instance is not None:
            raise RuntimeError("FashionMNISTLoader is a singleton, use instance()")
        self.n_clients = n_clients
        self._load_fmnist()
        
    @classmethod
    def instance(cls, n_clients=2):
        if FashionMNISTLoader._instance is None:
            FashionMNISTLoader._instance = FashionMNISTLoader(n_clients)
        return FashionMNISTLoader._instance

    def _load_fmnist(self):
        """
        Loads the Fashion-MNIST dataset
        """
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,), (1,))])
        train_data = torchvision.datasets.FashionMNIST('../../../datasets/femnist/', download=True, train=True, transform=transform)
        val_data = torchvision.datasets.FashionMNIST('../../../datasets/femnist/', download=True, train=False, transform=transform)
        self.train_partitions, self.val_partitions, self.test_set = partition_data(train_data, val_data, self.n_clients)
        
    def get_client_data(self):
        for train, val in zip(self.train_partitions, self.val_partitions):
            yield train, val

    def get_test(self):
        return self.test_set

class CIFAR10Loader:

    _instance = None

    def __init__(self, n_clients) -> None:
        if CIFAR10Loader._instance is not None:
            raise RuntimeError("CIFAR10Loader is a singleton, use instance()")
        self.n_clients = n_clients
        self._load_fmnist()
        
    @classmethod
    def instance(cls, n_clients=2):
        if CIFAR10Loader._instance is None:
            CIFAR10Loader._instance = CIFAR10Loader(n_clients)
        return CIFAR10Loader._instance

    def _load_fmnist(self):
        """
        Loads the Fashion-MNIST dataset
        """
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0, 1)])
        train_data = torchvision.datasets.CIFAR10('../../../datasets/cifar10/', download=True, train=True, transform=transform)
        val_data = torchvision.datasets.CIFAR10('../../../datasets/cifar10/', download=True, train=False, transform=transform)
        self.train_partitions, self.val_partitions, self.test_set = partition_data(train_data, val_data, self.n_clients)
        
    def get_client_data(self):
        for train, val in zip(self.train_partitions, self.val_partitions):
            yield train, val

    def get_test(self):
        return self.test_set

def get_dataset_loder(dataset, num_clients):
    if dataset == 'fmnist':
        return FashionMNISTLoader.instance(num_clients)
    elif dataset == 'cifar10':
        return CIFAR10Loader.instance(num_clients)
    else:
        raise ValueError('{} is not supported'.format(dataset))

def partition_data(train_set, val_set, n_clients):
    train_len = len(train_set)
    val_len = len(val_set) // 2
    
    # split validation set into validation and test set
    val_inds = np.arange(val_len)
    test_inds = np.array(range(val_len, 2*val_len))
    validation = Subset(val_set, val_inds)
    test = Subset(val_set, test_inds)

    # split sets in n_clients random and non-overlapping samples
    train_lengths = np.repeat(train_len // n_clients, n_clients)
    val_lengths = np.repeat(val_len // n_clients, n_clients)
    train_partitions = random_split(train_set, train_lengths, generator=torch.Generator().manual_seed(42))
    val_partitions = random_split(validation, val_lengths, generator=torch.Generator().manual_seed(42))

    return train_partitions, val_partitions, test

def discounted_mean(series, gamma=1.0):
    weight = gamma ** np.flip(np.arange(len(series)), axis=0)
    return np.inner(series, weight) / weight.sum()