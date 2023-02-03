from collections import OrderedDict
import os
import warnings

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_dataset_loder
from fedex_model import FMNISTCNN, CIFARCNN, NetworkCIFAR, NetworkImageNet
from rtpt import RTPT
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime as dt
import config
import argparse
from hyperparameters import Hyperparameters
from genotype import GENOTYPE

warnings.filterwarnings("ignore", category=UserWarning)
EPOCHS = 1

def train(net, trainloader, writer, epoch, optimizer, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = net(images)
        writer.add_histogram('logits', logits, i*epoch)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.)
        running_loss += loss.item()
        optimizer.step()
    writer.add_scalar('Training_Loss', running_loss, epoch)

def _test(net, testloader, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for feats, labels in testloader:
            #feats = feats.type(torch.FloatTensor)
            #labels = labels.type(torch.LongTensor)
            feats, labels = feats.to(device), labels.to(device)
            preds, _ = net(feats)
            loss += criterion(preds, labels).item()
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

def main(device, client_id):
    """Create model, load data, define Flower client, start Flower client."""

    # Load model
    #if config.DATASET == 'cifar10':
    #    net = CIFARCNN(config.IN_CHANNELS, config.OUT_CHANNELS, config.CLASSES)
    #elif config.DATASET == 'fmnist':
    #    net = FMNISTCNN()
    if config.DATASET == 'cifar10' or config.DATASET == 'fmnist':
        net = NetworkCIFAR(config.OUT_CHANNELS, config.CLASSES, config.CELLS, False, GENOTYPE, device, config.IN_CHANNELS)
    else:
        net = NetworkImageNet(config.OUT_CHANNELS, config.CLASSES, config.CELLS, False, GENOTYPE, device=device)
    net = net.to(device)

    # Load data
    dataset_loader = get_dataset_loder(config.DATASET, config.CLIENT_NR, config.DATASET_INDS_FILE, config.DATA_SKEW)
    train_data, test_data = dataset_loader.load_client_data(client_id)
    train_data, test_data = DataLoader(train_data, config.BATCH_SIZE, False), DataLoader(test_data, config.BATCH_SIZE, False)
    rtpt = RTPT('JS', 'FedEx_Client', config.ROUNDS)
    rtpt.start()

    # Flower client
    class MyClient(fl.client.NumPyClient):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.date = dt.strftime(dt.now(), '%Y:%m:%d:%H:%M:%S')
            #os.mkdir('./fedex_models/Client_{}'.format(self.date))
            self.writer = SummaryWriter("./runs/Client_{}".format(self.date))
            self.hyperparameters = Hyperparameters(config.HYPERPARAM_CONFIG_NR)
            self.hyperparameters.read_from_csv(config.HYPERPARAM_FILE)
            self.optim = torch.optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
            self.epoch = 1

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters_train(self, parameters, config):
            # obtain hyperparams and distribution
            self.distribution = parameters[-1]
            self.hyperparam_config, self.hidx = self._sample_hyperparams()
            
            # remove hyperparameter distribution from parameter list
            parameters = parameters[:-1]

            for g in self.optim.param_groups:
                g['lr'] = self.hyperparam_config['learning_rate']
                g['momentum'] = self.hyperparam_config['momentum']
                g['weight_decay'] = self.hyperparam_config['weight_decay']

            net.dropout = self.hyperparam_config['dropout']
            
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def set_parameters_evaluate(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters_train(parameters, config)
            before_loss, _ = _test(net, test_data, device)
            #net.drop_path_prob = self.hyperparam_config['dropout']
            train(net, train_data, self.writer, self.epoch, self.optim, device)
            after_loss, _ = _test(net, test_data, device)
            model_params = self.get_parameters()
            rtpt.step()
            self.epoch += 1
            return model_params, len(train_data), {'hidx': self.hidx, 'before': before_loss, 'after': after_loss}

        def evaluate(self, parameters, config):
            self.set_parameters_evaluate(parameters)
            #net.drop_path_prob = self.hyperparam_config['dropout']
            loss, accuracy = _test(net, test_data, device)
            return float(loss), len(test_data), {"accuracy": float(accuracy)}

        def _sample_hyperparams(self):
            # obtain new learning rate for this batch
            distribution = torch.distributions.Categorical(torch.FloatTensor(self.distribution))
            hyp_idx = distribution.sample().item()
            print(hyp_idx)
            hyp_config = self.hyperparameters[hyp_idx]
            return hyp_config, hyp_idx

    # Start client
    fl.client.start_numpy_client("[::]:{}".format(config.PORT), client=MyClient())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--id', type=int)

    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.gpu))
    main(device, args.id)