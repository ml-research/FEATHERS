import flwr as fl
from hanf_strategy import HANFStrategy
import torch.nn as nn
from model import Classifier
import config

def start_server(rounds):
    criterion = nn.CrossEntropyLoss()
    net = Classifier(config.CLASSES, criterion, config.CELL_NR, 
                    config.IN_CHANNELS, config.OUT_CHANNELS, config.NODE_NR)

    # Define strategy
    strategy = HANFStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
        initial_net=net,
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:8070",
        config={"num_rounds": rounds},
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server(config.ROUNDS)
