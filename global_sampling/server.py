import flwr as fl
from hanf_strategy import HANFStrategy
import torch.nn as nn
from model import Classifier



if __name__ == "__main__":

    criterion = nn.CrossEntropyLoss()
    net = Classifier(10, criterion, 4, 1, 16, 7)

    # Define strategy
    strategy = HANFStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
        initial_net=net,
        beta=0.5,
        epsilon=0.7,
        log_dir='./runs/hanf_grad_clip=7_rounds=200_no_backup'
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 200},
        strategy=strategy,
    )
