import flwr as fl
from hanf_strategy import HANFStrategy
import torch.nn as nn
import torch
from model import NetworkCIFAR, NetworkImageNet
import config
from helpers import prepare_log_dirs
from genotypes import GENOTYPE

def start_server(rounds):
    device = torch.device('cuda:{}'.format(str(config.SERVER_GPU))) 
    if config.DATASET == 'cifar10':
        net = NetworkCIFAR(config.OUT_CHANNELS, config.CLASSES, config.CELL_NR, False, GENOTYPE, device=device)
    elif config.DATASET == 'imagenet':
        net = NetworkImageNet(config.OUT_CHANNELS, config.CLASSES, config.CELL_NR, False, GENOTYPE, device=device)

    # prepare log-directories
    prepare_log_dirs()

    # Define strategy
    strategy = HANFStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
        initial_net=net,
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
    start_server(config.ROUNDS)
