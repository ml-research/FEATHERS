from typing import Dict, List, Optional, Tuple
import flwr as fl
import numpy as np
import pandas as pd
from numproto import proto_to_ndarray, ndarray_to_proto
from helpers import ProtobufNumpyArray, log_model_weights, log_hyper_config, log_hyper_params
from utils import get_dataset_loder, discounted_mean
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from rtpt import RTPT
from scipy.special import logsumexp
from numpy.linalg import norm
import config
from hyperparameters import Hyperparameters
import logging
import sys
import os
from datetime import datetime as dt

DEVICE = torch.device("cuda:{}".format(config.SERVER_GPU))

def _test(net, testloader, writer, round):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for feats, labels in testloader:
            #feats = feats.type(torch.FloatTensor)
            #labels = labels.type(torch.LongTensor)
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            preds = net(feats)
            writer.add_histogram('logits', preds, round)
            loss += criterion(preds, labels).item()
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def model_improved(results, weights):
    before_losses = np.array([res.metrics['before'] for _, res in results])
    after_losses = np.array([res.metrics['after'] for _, res in results])
    avg_before = np.sum(weights * before_losses)
    avg_after = np.sum(weights * after_losses)
    return (avg_after - avg_before) < 0


class FedexStrategy(fl.server.strategy.FedAvg):

    def __init__(self, fraction_fit, fraction_eval, initial_net, 
                log_dir='./runs/', discount_factor=0.9, use_gain_avg=False, **args) -> None:
        """
        Intitialize the Fedex strategy used by flwr to aggregation of model parameters.

        Args:
            fraction_fit (_type_): Fraction of clients used for fit
            fraction_eval (_type_): Fraction of clients used for evaluation
            initial_net (_type_): Initial network to be tuned by Fedex
            log_dir (str, optional): Directory where tensorboard-logs are stored. Defaults to './runs/'.
            epsilon (float, optional): Maximum value the distribution over hyperparameters can take before being placed in a simplex. Defaults to 0.8.
            beta (int, optional): Strength of how much values are emphasized which are around those values in the distribution whose probability > epsilon.
                                When the distrbution is adjusted s.t. we avoid it collapsing into a point-mass, we allow for more emphasizement of the configurations
                                around those for which p(configuration) > epsilon holds. Smaller beta leads to a more wide-spread distribution. Defaults to 1.
        """
        super().__init__(fraction_fit=fraction_fit, fraction_eval=fraction_eval, **args)
        self.hyperparams = Hyperparameters(config.HYPERPARAM_CONFIG_NR)
        self.hyperparams.save(config.HYPERPARAM_FILE)
        log_hyper_params({'learning_rates': self.hyperparams})
        self.log_distribution = np.full(len(self.hyperparams), -np.log(len(self.hyperparams)))
        self.distribution = np.exp(self.log_distribution)
        self.eta = np.sqrt(2*np.log(len(self.hyperparams)))
        self.discount_factor = discount_factor,
        self.use_gain_avg = use_gain_avg
        self.net = initial_net
        self.net.to(DEVICE)
        initial_params = [param.cpu().detach().numpy() for _, param in self.net.state_dict().items()]
        self.initial_parameters = self.last_weights = fl.common.weights_to_parameters(initial_params)
        data_loader = get_dataset_loder(config.DATASET, config.CLIENT_NR, config.DATASET_INDS_FILE, config.DATA_SKEW)
        data_loader.partition()
        self.test_data = data_loader.load_server_data()
        self.test_loader = DataLoader(self.test_data, batch_size=config.BATCH_SIZE, pin_memory=True, num_workers=2)
        self.current_round = 1
        self.writer = SummaryWriter(log_dir)
        self.rtpt = RTPT('JS', 'FedEx_Server', config.ROUNDS)
        self.rtpt.start()
        self.distribution_history = []
        self.gain_history = [] # initialize with [0] to avoid nan-values in discounted mean
        self.log_gain_hist = []

        # logging (also logs genotypes)
        self.date = dt.strftime(dt.now(), '%Y:%m:%d:%H:%M:%S')
        self.log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=self.log_format, datefmt='%m/%d %I:%M:%S %p')
        log_prefix = 'run_{}'
        if not os.path.exists('./models/' + log_prefix.format(self.date)):
            os.mkdir('./models/' + log_prefix.format(self.date))
        fh = logging.FileHandler(os.path.join('./models/' + log_prefix.format(self.date), 'log.txt'))
        fh.setFormatter(logging.Formatter(self.log_format))
        logging.getLogger().addHandler(fh)

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        """
        Aggregate model weights and update distribution over hyperparameters during fitting phase of model.

        Args:
            rnd (int): Communication round
            results (List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]): Results sent by the clients
            failures (List[BaseException]): Failures

        Returns:
            Optional[fl.common.Weights]: Aggregated weights, the updated distribution and the possible hyperparameter-configurations.
        """

        # obtain client weights
        samples = np.array([fit_res[1].num_examples for fit_res in results])
        weights = samples / np.sum(samples)
        aggregated_weights, _ = super().aggregate_fit(rnd, results, failures)

        # log current distribution
        self.distribution_history.append(self.distribution)
        dh = np.array(self.distribution_history)
        df = pd.DataFrame(dh)
        df.to_csv('distribution_history.csv')
        rh = np.array(self.log_gain_hist)
        df = pd.DataFrame(rh)
        df.to_csv('gain_history.csv')

        gains = self.compute_gains(weights, results)
        self.update_distribution(gains, weights)
        
        # sample hyperparameters and append them to the parameters
        serialized_dist = ndarray_to_proto(self.distribution)
        aggregated_weights.tensors.append(serialized_dist.ndarray)

        # log last hyperparam-configuration
        for _, res in results:
            hidx = res.metrics['hidx']
            config = self.hyperparams[hidx]
            log_hyper_config(config, rnd, self.writer)
        self.writer.add_histogram('gains', gains, rnd)
        return aggregated_weights, {}

    def _sample_hyperparams(self):
        # obtain new learning rate for this batch
        distribution = torch.distributions.Categorical(torch.FloatTensor(self.distribution))
        hyp_idx = distribution.sample().item()
        hyp_config = self.hyperparams[hyp_idx]
        return hyp_config, hyp_idx

    def aggregate_evaluate(self, rnd: int, results, failures):
        """
        Aggregate metrics computed by clients during evaluation phase.

        Args:
            rnd (int): Communication round
            results (_type_): Results sent by clients
            failures (_type_): Failures.

        Returns:
            _type_: Aggregated loss and metrics
        """
        loss, _ = super().aggregate_evaluate(rnd, results, failures)
        accuracies = np.array([res.metrics['accuracy'] for _, res in results])
        # assign weight to each client
        summed_weights = np.array([res.num_examples for _, res in results]).sum()
        weights = np.array([res.num_examples for _, res in results]) / summed_weights
        # compute mean accuracy and log it
        mean_accuracy = np.sum(accuracies * weights)
        self.writer.add_scalar('Validation_Loss', loss, self.current_round)
        self.writer.add_scalar('Validation_Accuracy', mean_accuracy, self.current_round)
        return loss, {'accuracy': mean_accuracy}

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):
        """
        Initialize the model before training starts. Initialization sneds weights of initial_net
        passed in constructor

        Args:
            client_manager (fl.server.client_manager.ClientManager): Client Manager

        Returns:
            _type_: Initial model weights, distribution and hyperparameter configurations.
        """
        serialized_dist = ndarray_to_proto(self.distribution)
        self.initial_parameters.tensors.append(serialized_dist.ndarray)
        return self.initial_parameters

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def compute_gains(self, weights, results):
        """
        Computes the average gains/progress the model made during the last fit-call.
        Each client computes its validation loss before and after a backpop-step.
        The difference before - after is averaged and we compute (avg_before - avg_after) - gain_history.
        The gain_history is a discounted mean telling us how much gain we have obtained in the last
        rounds. If we obtain a better gain than in history, we will emphasize the corresponding
        hyperparameter-configurations in the distribution, if not these configurations get less
        likely to be sampled in the next round.

        Args:
            weights (_type_): Client weights
            results (_type_): Client results

        Returns:
            _type_: Gains
        """
        after_losses = [res.metrics['after'] for _, res in results]
        before_losses = [res.metrics['before'] for _, res in results]
        hidxs = [res.metrics['hidx'] for _, res in results]
        # compute (avg_before - avg_after)
        avg_gains = np.array([w * (a - b) for w, a, b in zip(weights, after_losses, before_losses)]).sum()
        self.gain_history.append(avg_gains)
        gains = []
        # use gain-history to obtain how much we improved on "average" in history
        baseline = discounted_mean(np.array(self.gain_history), self.discount_factor) if len(self.gain_history) > 0 else 0.0
        for hidx, al, bl, w in zip(hidxs, after_losses, before_losses, weights):
            gain = w * ((al - bl) - baseline) if self.use_gain_avg else w * (al - bl)
            client_gains = np.zeros(len(self.hyperparams))
            client_gains[hidx] = gain
            gains.append(client_gains)
        gains = np.array(gains)
        gains = gains.sum(axis=0)
        self.log_gain_hist.append(gains)
        return gains
    
    def update_distribution(self, gains, weights):
        """
        Update the distribution over the hyperparameter-search space.
        First, an exponantiated "gradient" update is made based on the gains we obtained.
        As a following step, we bound the maximum probability to be epsilon.
        Those configurations which have probability > epsilon after the exponantiated gradient step,
        are re-weighted such that near configurations are emphasized as well.
        NOTE: This re-weighting constraints our hyperparameter-search space to parameters on which an order can be defined.

        Args:
            gains (_type_): Gains obtained in last round
            weights (_type_): Weights of clients
        """
        denom = 1.0 if np.all(gains == 0.0) else norm(gains, float('inf'))
        self.log_distribution -= self.eta / denom * gains
        self.log_distribution -= logsumexp(self.log_distribution)
        self.distribution = np.exp(self.log_distribution)

    def evaluate(self, parameters: fl.common.typing.Parameters):
        params = []
        for param in parameters.tensors:
            pnpa = ProtobufNumpyArray(param)
            weight = proto_to_ndarray(pnpa)
            params.append(weight)
        self.set_parameters(params)
        loss, accuracy = _test(self.net, self.test_loader, self.writer, self.current_round)

        # log metrics to tensorboard
        self.writer.add_scalar('Test_Loss', loss, self.current_round)
        self.writer.add_scalar('Test_Accuracy', accuracy, self.current_round)

        self.current_round += 1

        # since evaluate is the last method being called in one round, step rtpt here
        self.rtpt.step()
        return float(loss), {"accuracy": float(accuracy)}