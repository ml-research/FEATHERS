import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import random_split, Subset
import torchvision

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
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,), (1,))])
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


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


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

