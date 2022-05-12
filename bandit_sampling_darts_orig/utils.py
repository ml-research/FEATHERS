import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import random_split, Subset
import torchvision
import math

class FashionMNISTLoader:

    _instance = None

    def __init__(self, n_clients, skew=0) -> None:
        if FashionMNISTLoader._instance is not None:
            raise RuntimeError("FashionMNISTLoader is a singleton, use instance()")
        self.n_clients = n_clients
        self.skew = skew
        self._load_data()
        
    @classmethod
    def instance(cls, n_clients=2, skew=0):
        if FashionMNISTLoader._instance is None:
            FashionMNISTLoader._instance = FashionMNISTLoader(n_clients, skew=skew)
        return FashionMNISTLoader._instance

    def _load_data(self):
        """
        Loads the Fashion-MNIST dataset
        """
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,), (1,))])
        train_data = torchvision.datasets.FashionMNIST('../../../datasets/femnist/', download=True, train=True, transform=transform)
        val_data = torchvision.datasets.FashionMNIST('../../../datasets/femnist/', download=True, train=False, transform=transform)
        self.train_partitions, self.val_partitions, self.test_set = partition_skewed(train_data, val_data, self.n_clients, skew=self.skew)
        
    def get_client_data(self):
        for train, val in zip(self.train_partitions, self.val_partitions):
            yield train, val

    def get_test(self):
        return self.test_set

class CIFAR10Loader:

    _instance = None

    def __init__(self, n_clients, skew=0) -> None:
        if CIFAR10Loader._instance is not None:
            raise RuntimeError("CIFAR10Loader is a singleton, use instance()")
        self.n_clients = n_clients
        self.skew = skew
        self._load_data()
        
    @classmethod
    def instance(cls, n_clients=2, skew=0):
        if CIFAR10Loader._instance is None:
            CIFAR10Loader._instance = CIFAR10Loader(n_clients, skew=skew)
        return CIFAR10Loader._instance

    def _load_data(self):
        """
        Loads the Fashion-MNIST dataset
        """
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,), (1,))])
        train_data = torchvision.datasets.CIFAR10('../../../datasets/cifar10/', download=True, train=True, transform=transform)
        val_data = torchvision.datasets.CIFAR10('../../../datasets/cifar10/', download=True, train=False, transform=transform)
        self.train_partitions, self.val_partitions, self.test_set = partition_skewed(train_data, val_data, self.n_clients, skew=self.skew)
        
    def get_client_data(self):
        for train, val in zip(self.train_partitions, self.val_partitions):
            yield train, val

    def get_test(self):
        return self.test_set

def get_dataset_loder(dataset, num_clients, skew=0):
    if dataset == 'fmnist':
        return FashionMNISTLoader.instance(num_clients, skew=skew)
    elif dataset == 'cifar10':
        return CIFAR10Loader.instance(num_clients, skew=skew)
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

def label_distribution_skew(x, y, partitions, skew=1):
    def runner_split(N_labels, N_runners):
        """number of labels to assign to n clients"""
        runner_labels = round(max(1, N_labels / N_runners))
        runner_split = round(max(1, N_runners / N_labels))
        return runner_labels, runner_split

    runn_inds = []
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    N_labels = torch.unique(y).shape[0]
    n_labels, n_runners = runner_split(N_labels, partitions)
    np.random.seed(42)
    
    selected_inds = []
    for label_idx in range(0, N_labels, n_labels):
        # get a range of labels (e.g. in case of MNIST labels between 0 and 3)
        mask = torch.isin(y, torch.Tensor(np.arange(label_idx, label_idx+n_labels))).cpu().detach().numpy()
        # get index of these labels
        subset_idx = np.argwhere(mask)[:, 0]
        n_samples = subset_idx.shape[0]
        # randomly sample indices of size sample_size
        sample_size = math.floor(skew*n_samples)
        subset_idx = np.random.choice(subset_idx, sample_size, replace=False)
        selected_inds += list(subset_idx)
    
        for partition in np.array_split(subset_idx, n_runners):
            # add a data-partition for a runner
            runn_inds.append(partition)
    
    selected = np.array(selected_inds)
    mask = np.zeros(len(x))
    mask[selected] = 1
    not_selected = np.argwhere(mask == 0)
    return runn_inds, not_selected

def uniform_distribution(inds, partitions, randomise=True):
    runner_data = []
    if randomise:
        # shuffle indices
        np.random.shuffle(inds)
    # split randomly chosen data-points and labels into partitions
    for partition in np.array_split(inds, partitions):
        runner_data.append(partition)
    return runner_data

def partition_skewed(train_set, val_set, partitions, randomise=True, skew=1):
    # randomly select half of the data of the validation set to be the test-set
    ind_in_val = np.random.choice([0, 1], p=[0.5, 0.5], size=len(val_set))
    val_inds = np.argwhere(ind_in_val == 1).reshape(1, -1)[0]
    test_inds = np.argwhere(ind_in_val == 0).reshape(1, -1)[0]
    train_partitions, val_partitions, test_set = [], [], Subset(val_set, test_inds)
    train_inds = np.arange(len(train_set))
    if skew == 0:
        train_unfiorm = uniform_distribution(train_inds, partitions, randomise)
        val_uniform = uniform_distribution(val_inds, partitions, randomise)
        for t, v in zip(train_unfiorm, val_uniform):
            train_subset = Subset(train_set, t)
            val_subset = Subset(val_set, v)
            train_partitions.append(train_subset)
            val_partitions.append(val_subset)
    else:
        # build skewed data-sets
        train_selected, train_inds_remain = label_distribution_skew(train_set.data, train_set.targets, partitions, skew)
        val_selected, val_inds_remain = label_distribution_skew(val_set.data, val_set.targets, partitions, skew)
        # if skew < 1 this will contain a list of all data-points that are not already assigned to some runner
        train_uniform = uniform_distribution(train_inds_remain, partitions, randomise)
        val_uniform = uniform_distribution(val_inds_remain, partitions, randomise)
        # concatenate train and val set-indices obtained above
        for s, p in zip(train_selected, train_uniform):
            s = s.reshape(1, -1)[0]
            p = p.reshape(1, -1)[0]
            indices = np.concatenate((s, p))
            subset = Subset(train_set, indices)
            train_partitions.append(subset)
        for s, p in zip(val_selected, val_uniform):
            s = s.reshape(1, -1)[0]
            p = p.reshape(1, -1)[0]
            indices = np.concatenate((s, p))
            subset = Subset(val_set, indices)
            val_partitions.append(subset)
    return train_partitions, val_partitions, test_set

def discounted_mean(series, gamma=1.0):
    weight = gamma ** np.flip(np.arange(len(series)), axis=0)
    return np.inner(series, weight) / weight.sum()

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob, device):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(device)
    x = torch.div(x, keep_prob)
    x = torch.mul(x, mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

