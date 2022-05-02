import flwr as fl
from hanf_strategy import HANFStrategy
import torch.nn as nn
import torch
from model_search import Network
import config

def start_server(rounds):
    device = torch.device('cuda:{}'.format(str(config.SERVER_GPU))) 
    criterion = nn.CrossEntropyLoss()
    net = Network(config.OUT_CHANNELS, config.CLASSES, config.CELL_NR, criterion, device)

    # Define strategy
    strategy = HANFStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
        initial_net=net,
        alpha=config.ALPHA,
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:{}".format(config.PORT),
        config={"num_rounds": rounds},
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server(config.ROUNDS)
