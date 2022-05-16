from collections import OrderedDict
import os
import warnings

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_dataset_loder
from fedex_model import NetworkCIFAR
from rtpt import RTPT
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime as dt
import config
from genotypes import GENOTYPE
import argparse

warnings.filterwarnings("ignore", category=UserWarning)
EPOCHS = 1

def train(net, trainloader, lr, writer, epoch, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    running_loss = 0
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = net(images)
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
            preds = net(feats)
            loss += criterion(preds, labels).item()
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

def main(device):
    """Create model, load data, define Flower client, start Flower client."""

    # Load model
    net = NetworkCIFAR(config.OUT_CHANNELS, config.CLASSES, config.CELL_NR, False, GENOTYPE, device, config.IN_CHANNELS)
    net.to(device)

    # Load data
    dataset_loader = get_dataset_loder(config.DATASET, config.CLIENTS, config.DATA_SKEW)
    train_data, test_data = next(dataset_loader.get_client_data())
    train_data, test_data = DataLoader(train_data, config.BATCH_SIZE, False), DataLoader(test_data, config.BATCH_SIZE, False)
    rtpt = RTPT('JS', 'HANF_Client', config.ROUNDS)
    rtpt.start()

    # Flower client
    class MyClient(fl.client.NumPyClient):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.date = dt.strftime(dt.now(), '%Y:%m:%d:%H:%M:%S')
            os.mkdir('./fedex_models/Client_{}'.format(self.date))
            self.writer = SummaryWriter("./runs/Client_{}".format(self.date))
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
            train(net, train_data, self.hyperparam_config, self.writer, self.epoch, device)
            after_loss, _ = _test(net, test_data, device)
            model_params = self.get_parameters()
            rtpt.step()
            torch.save(net, "./fedex_models/Client_{}/net_round_{}".format(self.date, self.epoch))
            self.epoch += 1
            return model_params, len(train_data), {'lr': self.hyperparam_config, 'hidx': self.hidx, 'before': before_loss, 'after': after_loss}

        def evaluate(self, parameters, config):
            self.set_parameters_evaluate(parameters)
            loss, accuracy = _test(net, test_data, device)
            return float(loss), len(test_data), {"accuracy": float(accuracy)}

        def _sample_hyperparams(self):
            # obtain new learning rate for this batch
            distribution = torch.distributions.Categorical(torch.FloatTensor(self.distribution))
            hyp_idx = distribution.sample().item()
            hyp_config = self.hyperparams[hyp_idx]
            return hyp_config, hyp_idx

    # Start client
    fl.client.start_numpy_client("[::]:{}".format(config.PORT), client=MyClient())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)

    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.gpu))
    main(device)