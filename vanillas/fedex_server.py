import flwr as fl
from local_sampling_strategy import HANFStrategy
import torch.nn as nn
from fedex_model import Net



if __name__ == "__main__":
    net = Net()

    # Define strategy
    strategy = HANFStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
        initial_net=net,
        beta=0.7,
        log_dir='./runs/fedex_clip_grad_1'
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 20},
        strategy=strategy,
    )
