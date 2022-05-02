from lib2to3.pgen2.token import RPAR
from trainer import DartsTrainer
from model import Classifier
import torch
import torchvision
from helpers import compute_accuracy
from rtpt import RTPT

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,), (1,))])
train_data = torchvision.datasets.CIFAR10('../../../datasets/cifar10/', download=True, train=True, transform=transform)
val_data = torchvision.datasets.CIFAR10('../../../datasets/cifar10/', download=True, train=False, transform=transform)

mlr = 0.025
alr = 3e-4
grad_clip = 5
wd = 3e-4
momentum = 0.9
epochs = 50
layers = 8
node_nr = 5
init_channels = 16

gpu = torch.device('cuda:1')

criterion = torch.nn.CrossEntropyLoss()
model = Classifier(10, criterion, layers, 3, init_channels, node_nr)
model.to(gpu)

rt = RTPT('JS', 'DARTS', epochs)
rt.start()

trainer = DartsTrainer(model, criterion, train_data, val_data, device=gpu, arc_learning_rate=alr, 
                        second_order_optim=True, weight_decay=wd, mlr=mlr, batch_size=64)

for e in range(epochs):
    print("Epoch {}/{}".format(e+1, epochs))
    trainer.train_one_epoch(e)
    rt.step()

    # compute val. acc.
    accs = []
    batches = 0
    for x, y in trainer.valid_loader:
        x, y = x.to(gpu), y.to(gpu)
        logits = trainer.model(x)
        accuracy = compute_accuracy(logits, y)
        accs.append(accuracy)
        batches += 1
    
    overall_acc = sum(accs) / batches
    print("Avg. accuracy: {}".format(overall_acc))