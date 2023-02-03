import flwr as fl
from strategy import FedexStrategy
import torch.nn as nn
from fedex_model import CIFARCNN, FMNISTCNN, NetworkCIFAR, NetworkImageNet
import argparse
import config
from helpers import prepare_log_dirs
from genotype import GENOTYPE
import torch

def start_server(log_dir, rounds, dataset):
    #if config.DATASET == 'cifar10':
    #    net = CIFARCNN(config.IN_CHANNELS, config.OUT_CHANNELS, config.CLASSES)
    #elif config.DATASET == 'fmnist':
    #    net = FMNISTCNN()
    device = torch.device(f'cuda:{config.SERVER_GPU}') if torch.cuda.is_available() else torch.device('cpu')
    if config.DATASET == 'cifar10' or config.DATASET == 'fmnist':
        net = NetworkCIFAR(config.OUT_CHANNELS, config.CLASSES, config.CELLS, False, GENOTYPE, device, config.IN_CHANNELS)
    else:
        net = NetworkImageNet(config.OUT_CHANNELS, config.CLASSES, config.CELLS, False, GENOTYPE, device=device)
    

    prepare_log_dirs()
    
    # Define strategy
    strategy = FedexStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
        initial_net=net,
        log_dir=log_dir,
        min_fit_clients=config.MIN_TRAIN_CLIENTS,
        min_eval_clients=config.MIN_VAL_CLIENTS,
        min_available_clients=config.CLIENT_NR,
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:{}".format(config.PORT),
        config={"num_rounds": rounds},
        strategy=strategy,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=60)
    parser.add_argument('--log-dir')

    args = parser.parse_args()

    start_server(args.log_dir, config.ROUNDS, config.DATASET)
