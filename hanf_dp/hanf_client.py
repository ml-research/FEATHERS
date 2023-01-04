from collections import OrderedDict
from copy import deepcopy
from email.policy import strict
import warnings

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from dp_arch_optimizer import DPOptimizer

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

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, device):

    with BatchMemoryManager(data_loader=train_queue, 
                max_physical_batch_size=config.BATCH_SIZE, optimizer=optimizer) as train_bmm:
        
        with BatchMemoryManager(data_loader=valid_queue,
                max_physical_batch_size=config.BATCH_SIZE, optimizer=architect.optimizer) as valid_bmm:
            for step, (input, target) in enumerate(train_bmm):
                if step % 50 == 0:
                    print(f'Step {step:03d}')
                model.train()
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                # get a random minibatch from the search queue with replacement
                input_search, target_search = next(iter(valid_bmm))
                input_search = input_search.to(device, non_blocking=True)
                target_search = target_search.to(device, non_blocking=True)
                architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=False)

                # sync model (arch -> model)
                model = sync_models(architect.model, model)
                
                optimizer.zero_grad()
                logits = model(input)
                loss = criterion(logits, target)
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), 5.)
                optimizer.step()
                model.zero_grad()

                # sync model (model -> arch)
                architect.model = sync_models(model, architect.model)
  
    return model

def sync_models(updated_model, model_tbu):
    state_dict = updated_model.state_dict()
    model_tbu.load_state_dict(state_dict, strict=True)
    return model_tbu


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
            if config.DATASET == 'fraud':
                self.model = TabularNetwork(config.NODE_NR, config.FRAUD_DETECTION_IN_DIM, config.CLASSES, config.CELL_NR, self.criterion, device)
            else:
                self.model = Network(out_channels, classes, cell_nr, self.criterion, device, in_channels=input_channels, steps=config.NODE_NR)
            arch_model = deepcopy(self.model) # since opcaus cannot register multiple hooks to the same model, we have to instantiate two models and sync them after each step
            model_optim = torch.optim.SGD(get_params(self.model, 'model'), 0.01, 0.9, 3e-4)
            arch_optim = torch.optim.Adam(get_params(arch_model, 'arch'),
                            lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
            self.train_loader = DataLoader(train_data, config.BATCH_SIZE, pin_memory=True, num_workers=2)
            self.val_loader = DataLoader(test_data, config.BATCH_SIZE, pin_memory=True, num_workers=2)
            #self.model = ModuleValidator.fix(self.model) # required to replace modules not supported by opacus (e.g. BatchNorm)
            #ModuleValidator.validate(self.model, strict=False)
            pe = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = pe.make_private(module=self.model, optimizer=model_optim, 
                                                    data_loader=self.train_loader, noise_multiplier=1., max_grad_norm=config.MAX_GRAD_NORM)
            arch_model, _, self.val_loader = pe.make_private(module=arch_model, optimizer=arch_optim, 
                                                    data_loader=self.val_loader, noise_multiplier=1., max_grad_norm=config.MAX_GRAD_NORM)
            dp_arch_optim = DPOptimizer(arch_optim, noise_multiplier=2., max_grad_norm=config.MAX_GRAD_NORM, expected_batch_size=config.BATCH_SIZE)

            self.model = self.model.to(device)
            arch_model = arch_model.to(device)
            self.architect = Architect(arch_model, dp_arch_optim, 0.9, 1e-3, self.criterion, device)

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
                self.model = train(self.train_loader, self.val_loader, self.model,
                                                 self.architect, self.criterion, self.optimizer, 
                                                 self.hyperparam_config['learning_rate'], device)
            after_loss, _ = _test(self.model, self.val_loader, device)
            model_params = self.get_parameters()
            return model_params, len(train_data), {'hidx': int(self.hidx), 'before': float(before_loss), 'after': float(after_loss)}

        def evaluate(self, parameters, config):
            self.set_parameters_evaluate(parameters)
            loss, accuracy = _test(self.model, self.val_loader, device)
            return float(loss), len(test_data), {"accuracy": float(accuracy)}

        def set_current_hyperparameter_config(self, hyperparam, idx):
            self.hyperparam_config = hyperparam
            self.hidx = idx
            if self.optimizer is None:
                self.optimizer = torch.optim.SGD(self.model.parameters(), self.hyperparam_config['learning_rate'], 
                                                momentum=self.hyperparam_config['momentum'], weight_decay=self.hyperparam_config['weight_decay'])
            else:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.hyperparam_config['learning_rate']
                    g['momentum'] = self.hyperparam_config['momentum']
                    g['weight_decay'] = self.hyperparam_config['weight_decay']

            # update architect's hyperparameters
            self.architect.update_hyperparameters(hyperparam)

            
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