from collections import OrderedDict
import os
import warnings

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import FashionMNISTLoader
from fedex_model import Net
from rtpt import RTPT
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime as dt

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
EPOCHS = 1

def train(net, trainloader, lr, writer, epoch):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    running_loss = 0
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = net(images)
        writer.add_histogram('logits', logits, i*epoch)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.)
        running_loss += loss.item()
        optimizer.step()
    writer.add_scalar('Training_Loss', running_loss, epoch)

def _test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for feats, labels in testloader:
            #feats = feats.type(torch.FloatTensor)
            #labels = labels.type(torch.LongTensor)
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
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

def main():
    """Create model, load data, define Flower client, start Flower client."""

    # Load model
    net = Net()
    net.to(DEVICE)

    # Load data
    fashion_mnist_iterator = FashionMNISTLoader.instance(2)
    train_data, test_data = next(fashion_mnist_iterator.get_client_data())
    train_data, test_data = DataLoader(train_data, 64, False), DataLoader(test_data, 64, False)
    rtpt = RTPT('JS', 'HANF_Client', EPOCHS)
    rtpt.start()

    # Flower client
    class MyClient(fl.client.NumPyClient):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.date = dt.strftime(dt.now(), '%Y:%m:%d:%H:%M:%S')
            os.mkdir('./fedex_models/Client_{}'.format(self.date))
            self.writer = SummaryWriter("./runs/Client_{}".format(self.date))
            self.epoch = 1

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters_train(self, parameters, config):
            # obtain hyperparams and distribution
            self.hyperparam_config = float(parameters[-2][0])
            self.hidx = int(parameters[-1][0])
            
            # remove hyperparameter distribution from parameter list
            parameters = parameters[:-2]
            
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def set_parameters_evaluate(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters_train(parameters, config)
            before_loss, _ = _test(net, test_data)
            train(net, train_data, self.hyperparam_config, self.writer, self.epoch)
            after_loss, _ = _test(net, test_data)
            model_params = self.get_parameters()
            rtpt.step()
            torch.save(net, "./fedex_models/Client_{}/net_round_{}".format(self.date, self.epoch))
            self.epoch += 1
            return model_params, len(train_data), {'lr': self.hyperparam_config, 'hidx': self.hidx, 'before': before_loss, 'after': after_loss}

        def evaluate(self, parameters, config):
            self.set_parameters_evaluate(parameters)
            loss, accuracy = _test(net, test_data)
            return float(loss), len(test_data), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client("[::]:8081", client=MyClient())


if __name__ == "__main__":
    main()