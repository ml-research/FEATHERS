from collections import OrderedDict
from copy import deepcopy
from email.policy import strict
import warnings

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.autograd import Variable
import numpy as np
from utils import get_dataset_loder, get_params
from rtpt import RTPT
import config
from hyperparameters import Hyperparameters
from tensorboardX import SummaryWriter
from datetime import datetime as dt
import argparse
from model_search import Network, TabularNetwork
from architect import Architect
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from dp_arch_optimizer import DPArchOptimizer

warnings.filterwarnings("ignore", category=UserWarning)
EPOCHS = 1


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

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, device, num_model_param_groups):

    with BatchMemoryManager(data_loader=train_queue, 
               max_physical_batch_size=config.BATCH_SIZE, optimizer=optimizer) as train_bmm:
       
       with BatchMemoryManager(data_loader=valid_queue,
               max_physical_batch_size=config.BATCH_SIZE, optimizer=architect.optimizer) as valid_bmm:
            for step, (input, target) in enumerate(train_bmm):
                if step % 10 == 0:
                    print(f'Step {step:03d}')
                    pos = len(target[target == 1])
                    neg = len(target[target == 0])
                    print(f'Frac: {pos / (pos + neg)}')
                model.train()
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                # get a random minibatch from the search queue with replacement
                input_search, target_search = next(iter(valid_bmm))
                input_search = input_search.to(device, non_blocking=True)
                target_search = target_search.to(device, non_blocking=True)
                # set parameter-lr zero during arch. step
                optimizer, plr = set_optimizer_lr(optimizer, 'params')
                architect.step(input, target, input_search, target_search, lr, optimizer, False, num_model_param_groups)
                # reset parameter-lr and set arch. lr zero
                optimizer, _ = set_optimizer_lr(optimizer, 'params', plr)
                optimizer, alr = set_optimizer_lr(optimizer, 'arch_params')
                optimizer.zero_grad()
                logits = model(input)
                loss = criterion(logits, target)
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), 5.)
                optimizer.step()
                optimizer, _ = set_optimizer_lr(optimizer, 'arch_params', alr)
                model.zero_grad()
  
    return model, optimizer

def set_optimizer_lr(optimizer, param_group, lr=0):
    if param_group == 'params':
        g = optimizer.param_groups[0]
        plr = g['lr']
        g['lr'] = lr
        return optimizer, plr
    if param_group == 'arch_params':
        g = optimizer.param_groups[1]
        alr = g['lr']
        g['lr'] = lr
        return optimizer, alr
    


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

def main(dataset, num_clients, device, client_id, classes=10, cell_nr=4, input_channels=1, out_channels=16, node_nr=7):
    """Create model, load data, define Flower client, start Flower client."""

    # Load data
    data_loader = get_dataset_loder(dataset, num_clients, config.DATASET_INDS_FILE, config.DATA_SKEW)
    train_data, test_data = data_loader.load_client_data(client_id)
    date = dt.strftime(dt.now(), '%Y:%m:%d:%H:%M:%S')
    rtpt = RTPT('JS', 'HANF_Client', EPOCHS)
    rtpt.start()

    # Flower client
    class HANFClient(fl.client.NumPyClient):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.epoch = 0
            self.hyperparameters = Hyperparameters(config.HYPERPARAM_CONFIG_NR)
            self.hyperparameters.read_from_csv(config.HYPERPARAM_FILE)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.criterion = self.criterion.to(device)
            self.dp_sigma_reward = config.DP_SIGMA_REWARD
            if config.DATASET == 'fraud':
                model = TabularNetwork(config.NODE_NR, config.FRAUD_DETECTION_IN_DIM, config.CLASSES, config.CELL_NR, self.criterion, device)
            else:
                model = Network(out_channels, classes, cell_nr, self.criterion, device, in_channels=input_channels, steps=config.NODE_NR)
            model = model.to(device)
            model_params = get_params(model, 'model')
            arch_params = get_params(model, 'arch')
            self.num_model_param_groups = len(model_params)
            optim = torch.optim.SGD([
                {'params': model_params, 'lr': 0.01, 'weight_decay': 3e-4, 'momentum': 0.9},
                {'params': arch_params, 'lr': 3e-4, 'weight_decay': 1e-3, 'momentum': 0.9}], lr=0.01, momentum=0.9, weight_decay=3e-4)
            sampler = self._get_sampler(train_data) if config.USE_WEIGHTED_SAMPLER else None
            self.train_loader = DataLoader(train_data, config.BATCH_SIZE, pin_memory=True, sampler=sampler, num_workers=2)
            self.val_loader = DataLoader(test_data, config.BATCH_SIZE, pin_memory=True, num_workers=2)
            #self.model = ModuleValidator.fix(self.model) # required to replace modules not supported by opacus (e.g. BatchNorm)
            #ModuleValidator.validate(self.model, strict=False)
            pe = PrivacyEngine()
            _, _, self.val_loader = pe.make_private(module=deepcopy(model), optimizer=optim, batch_first=True,
                                                    data_loader=self.val_loader, noise_multiplier=0.5, max_grad_norm=config.MAX_GRAD_NORM)
            self.model, self.optimizer, _ = pe.make_private(module=model, optimizer=optim, batch_first=True,
                                                   data_loader=self.train_loader, noise_multiplier=0.5, max_grad_norm=config.MAX_GRAD_NORM)

            #arch_model = arch_model.to(device)
            self.architect = Architect(self.model, self.optimizer, 0.9, 1e-3, self.criterion, device)

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def set_parameters_train(self, parameters, config):
            # obtain hyperparams and distribution
            hidx = int(parameters[-1][0])
            hyperparams = self.hyperparameters[hidx]
            self.set_current_hyperparameter_config(hyperparams, hidx)
            
            # remove hyperparameter distribution from parameter list
            parameters = parameters[:-1]
            
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            self.architect.model.load_state_dict(state_dict, strict=True)

        def set_parameters_evaluate(self, parameters):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, flwr_config):
            self.set_parameters_train(parameters, flwr_config)
            before_loss, _ = _test(self.model, self.val_loader, device)
            for e in range(EPOCHS):
                rtpt.step()
                self.epoch += 1
                self.model, self.optimizer = train(self.train_loader, self.val_loader, self.model,
                                                 self.architect, self.criterion, self.optimizer, 
                                                 self.hyperparam_config['learning_rate'], device, self.num_model_param_groups)
            after_loss, _ = _test(self.model, self.val_loader, device)
            model_params = self.get_parameters()
            before_loss += self.dp_sigma_reward * np.random.normal(0, 1)
            after_loss += self.dp_sigma_reward * np.random.normal(0, 1)
            return model_params, len(train_data), {'hidx': int(self.hidx), 'before': float(before_loss), 'after': float(after_loss)}

        def evaluate(self, parameters, config):
            self.set_parameters_evaluate(parameters)
            loss, accuracy = _test(self.model, self.val_loader, device)
            return float(loss), len(test_data), {"accuracy": float(accuracy)}

        def set_current_hyperparameter_config(self, hyperparam, idx):
            self.hyperparam_config = hyperparam
            self.hidx = idx
            g = self.optimizer.param_groups[0]
            g['lr'] = self.hyperparam_config['learning_rate']
            g['momentum'] = self.hyperparam_config['momentum']
            g['weight_decay'] = self.hyperparam_config['weight_decay']

            g = self.optimizer.param_groups[1]
            g['lr'] = self.hyperparam_config['arch_learning_rate']
            g['weight_decay'] = self.hyperparam_config['arch_weight_decay']

            # update architect's hyperparameters
            self.architect.update_hyperparameters(hyperparam)

        def _get_sampler(self, training_data):
            targets = training_data.dataset.y[training_data.indices]
            class_count = torch.tensor([len(targets[targets == t]) for t in torch.unique(targets)])
            weight = 1 / class_count
            samples_weight = torch.tensor([weight[t] for t in targets]).double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            return sampler

            
    # Start client
    fl.client.start_numpy_client("127.0.0.1:{}".format(config.PORT), client=HANFClient())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--id', type=int)

    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.gpu))
    main(config.DATASET, config.CLIENT_NR, device, args.id, config.CLASSES, config.CELL_NR, 
        config.IN_CHANNELS, config.OUT_CHANNELS, config.NODE_NR)