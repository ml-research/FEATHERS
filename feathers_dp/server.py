import flwr as fl
from hanf_strategy import HANFStrategy
import torch.nn as nn
import torch
from model_search import Network, TabularNetwork
from model import NetworkCIFAR, NetworkImageNet, NetworkTabular
import config
from helpers import prepare_log_dirs
import argparse
from genotypes import GENOTYPE
from opacus import PrivacyEngine

def start_server_search(rounds):
    device = torch.device('cuda:{}'.format(str(config.SERVER_GPU))) 
    criterion = nn.CrossEntropyLoss()
    if config.DATASET == 'fraud':
        net = TabularNetwork(config.NODE_NR, config.FRAUD_DETECTION_IN_DIM, config.CLASSES, config.CELL_NR, criterion, device=device)
    else:        
        net = Network(config.OUT_CHANNELS, config.CLASSES, config.CELL_NR, criterion, device, in_channels=config.IN_CHANNELS, steps=config.NODE_NR)

    model = PrivacyEngine.get_compatible_module(net)


    # prepare log-directories
    prepare_log_dirs()

    # Define strategy
    strategy = HANFStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
        initial_net=model,
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        min_fit_clients=config.MIN_TRAIN_CLIENTS,
        min_eval_clients=config.MIN_VAL_CLIENTS,
        min_available_clients=config.CLIENT_NR,
        stage='search',
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:{}".format(config.PORT),
        config={"num_rounds": rounds},
        strategy=strategy,
    )

def start_server_valid(rounds):
    device = torch.device('cuda:{}'.format(str(config.SERVER_GPU))) 
    if config.DATASET == 'cifar10' or config.DATASET == 'fmnist':
        net = NetworkCIFAR(config.OUT_CHANNELS, config.CLASSES, config.CELL_NR, False, GENOTYPE, device=device, in_channels=config.IN_CHANNELS)
        model = PrivacyEngine.get_compatible_module(net)
    elif config.DATASET == 'imagenet':
        net = NetworkImageNet(config.OUT_CHANNELS, config.CLASSES, config.CELL_NR, False, GENOTYPE, device=device)
        model = PrivacyEngine.get_compatible_module(net)
    elif config.DATASET == 'fraud':
        net = NetworkTabular(config.FRAUD_DETECTION_IN_DIM, config.CLASSES, config.CELL_NR, GENOTYPE, device=device)
        model = PrivacyEngine.get_compatible_module(net)

    # prepare log-directories
    prepare_log_dirs()

    # Define strategy
    strategy = HANFStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
        initial_net=model,
        alpha=config.ALPHA,
        min_fit_clients=config.MIN_TRAIN_CLIENTS,
        min_eval_clients=config.MIN_VAL_CLIENTS,
        min_available_clients=config.CLIENT_NR,
        stage='valid',
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:{}".format(config.PORT),
        config={"num_rounds": rounds},
        strategy=strategy,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='search', type=str)

    args = parser.parse_args()
    if args.stage == 'search':
        start_server_search(config.ROUNDS)
    elif args.stage == 'valid':
        start_server_valid(config.ROUNDS)
    else:
        raise ValueError('Unknown stage: {}'.format(args.stage))
