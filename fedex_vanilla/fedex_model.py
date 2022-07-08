import torch.nn as nn
import torch.nn.functional as F

# Build the neural network, expand on top of nn.Module
class CIFARCNN(nn.Module):
    def __init__(self, in_channels, out_channels, classes, dropout=0.0):
        super(CIFARCNN, self).__init__()
        self.conv1 = nn.Sequential(
                                   nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   )
        self.conv2 = nn.Sequential(
                                   nn.Conv2d(out_channels, 64, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   )
        self.conv3 = nn.Sequential(
                                   nn.Conv2d(64, 64, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   )
        self.dropout = dropout
        self.fc = nn.Sequential(
                                nn.Linear(1024, 64),
                                nn.ReLU(),
                                )
        self.clf = nn.Linear(64, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(F.dropout(x.flatten(1), self.dropout))
        return self.clf(F.dropout(x, self.dropout))

class FMNISTCNN(nn.Module):
  def __init__(self, dropout=0):
    super().__init__()

    # define layers
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

    self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=10)
    self.dropout = dropout

  # define forward function
  def forward(self, t):
    # conv 1
    t = self.conv1(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # conv 2
    t = self.conv2(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # fc1
    t = t.reshape(-1, 12*4*4)
    t = self.fc1(t)
    t = F.relu(t)

    # fc2
    t = F.dropout(t, self.dropout)
    t = self.fc2(t)
    t = F.relu(t)

    # output
    t = F.dropout(t, self.dropout)
    t = self.out(t)
    # don't need softmax here since we'll use cross-entropy as activation.

    return t