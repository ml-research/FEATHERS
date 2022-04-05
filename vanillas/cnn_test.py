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

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:10" if torch.cuda.is_available() else "cpu")
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
        running_loss += loss.item()
        optimizer.step()
    writer.add_scalar('Training_Loss', running_loss, epoch)
    return net

def test(net, testloader):
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

# Load model
net = Net()
net.to(DEVICE)

# Load data
fashion_mnist_iterator = FashionMNISTLoader.instance(1)
train_data, test_data = next(fashion_mnist_iterator.get_client_data())
train_data, test_data = DataLoader(train_data, 64, True), DataLoader(test_data, 64, False)
rtpt = RTPT('JS', 'HANF_Client', EPOCHS)
rtpt.start()

writer = SummaryWriter('./runs/cnn/')

for e in range(0, 10):
    net = train(net, train_data, 0.01, writer, e)
    loss, acc = test(net, test_data)
    writer.add_scalar('Validation Loss', loss, e)
    writer.add_scalar('Validation Accuracy', acc, e)