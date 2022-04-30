import flwr as fl
from bandit_strategy import HANFStrategy
import torch.nn as nn
from fedex_model import Net
import argparse

def start_server(beta, epsilon, log_dir, rounds):
    criterion = nn.CrossEntropyLoss()
    net = Net()

    # Define strategy
    strategy = HANFStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
        initial_net=net,
        beta=beta,
        epsilon=epsilon,
        log_dir=log_dir
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:8071",
        config={"num_rounds": rounds},
        strategy=strategy,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=60)
    parser.add_argument('--log-dir')
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.7)

    args = parser.parse_args()

    start_server(args.beta, args.epsilon, args.log_dir, args.rounds)
