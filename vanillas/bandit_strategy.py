from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import flwr as fl
import numpy as np
import pandas as pd
from numproto import proto_to_ndarray, ndarray_to_proto
from helpers import ProtobufNumpyArray, log_model_weights, log_hyper_config, log_hyper_params
from utils import FashionMNISTLoader, discounted_mean
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from rtpt import RTPT
from scipy.special import logsumexp
from numpy.linalg import norm

DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

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


class HANFStrategy(fl.server.strategy.FedAvg):

    def __init__(self, fraction_fit, fraction_eval, initial_net, 
                log_dir='./runs/', epsilon=0.8, beta=1, nabla=0.1, 
                discount_factor=0.9, use_gain_avg=False, **args) -> None:
        """
        Intitialize the HANF strategy used by flwr to aggregation of model parameters.

        Args:
            fraction_fit (_type_): Fraction of clients used for fit
            fraction_eval (_type_): Fraction of clients used for evaluation
            initial_net (_type_): Initial network to be tuned by HANF
            log_dir (str, optional): Directory where tensorboard-logs are stored. Defaults to './runs/'.
            epsilon (float, optional): Maximum value the distribution over hyperparameters can take before being placed in a simplex. Defaults to 0.8.
            beta (int, optional): Strength of how much values are emphasized which are around those values in the distribution whose probability > epsilon.
                                When the distrbution is adjusted s.t. we avoid it collapsing into a point-mass, we allow for more emphasizement of the configurations
                                around those for which p(configuration) > epsilon holds. Smaller beta leads to a more wide-spread distribution. Defaults to 1.
        """
        super().__init__(fraction_fit=fraction_fit, fraction_eval=fraction_eval, **args)
        self.hyperparams = np.sort(10 ** (np.random.uniform(-4, 0, 40))) # sorting is important for distribution update
        log_hyper_params({'learning_rates': self.hyperparams})
        self.log_distribution = np.full(len(self.hyperparams), -np.log(self.hyperparams))
        self.distribution = np.exp(self.log_distribution)
        self.epsilon = epsilon
        self.beta = beta
        self.eta = np.sqrt(2*np.log(len(self.hyperparams)))
        self.discount_factor = discount_factor,
        self.use_gain_avg = use_gain_avg
        self.net = initial_net
        self.net.to(DEVICE)
        initial_params = [param.cpu().detach().numpy() for _, param in self.net.state_dict().items()]
        self.initial_parameters = self.last_weights = fl.common.weights_to_parameters(initial_params)
        fashion_mnist_iterator = FashionMNISTLoader.instance(2)
        self.test_data = fashion_mnist_iterator.get_test()
        self.test_loader = DataLoader(self.test_data, batch_size=64, pin_memory=True, num_workers=2)
        self.current_round = 0
        self.writer = SummaryWriter(log_dir)
        self.rtpt = RTPT('JS', 'HANF_Server', 20)
        self.rtpt.start()
        self.distribution_history = []
        self.gain_history = []
        self.current_config_idx = None
        self.old_weights = self.initial_parameters
        self.log_round = 0
        self.current_exploration = None

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

        self.log_round += 1

        if self.current_round % 10 == 0:
            print("======================= EXPLORING PHASE ======================")
            aggregated_weights = deepcopy(self.old_weights)
            if self.current_exploration is None:
                self._sample_hyperparams()
            if len(self.current_exploration) > 0:
                if len(self.current_exploration) < 10:
                    self.compute_gains(weights, results)
                self.current_config_idx = int(self.current_exploration[-1])
                self.current_exploration = self.current_exploration[:-1]
            else:
                self.compute_gains(weights, results)
                self.current_exploration = None
                self.current_round += 1
                last_gains = np.array(self.gain_history)
                max_gain_idx = np.argmin(last_gains[:,1])
                self.current_config_idx = int(last_gains[max_gain_idx, 0])
                self.gain_history = []
        else:
            self.current_round += 1
            aggregated_weights, _ = super().aggregate_fit(rnd, results, failures)
            self.old_weights = aggregated_weights
        
        # sample hyperparameters and append them to the parameters
        serialized_idx = ndarray_to_proto(np.array([self.current_config_idx]))
        serialized_hyperparams = ndarray_to_proto(self.hyperparams)
        aggregated_weights.tensors.append(serialized_idx.ndarray)
        aggregated_weights.tensors.append(serialized_hyperparams.ndarray)

        self.writer.add_scalar('Learning Rate', self.hyperparams[self.current_config_idx], rnd)

        return aggregated_weights, {}

    def _sample_hyperparams(self):
        # obtain new learning rate for this batch
        self.current_exploration = np.random.randint(0, len(self.hyperparams), 10)

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
        self.writer.add_scalar('Validation_Loss', loss, self.log_round)
        self.writer.add_scalar('Validation_Accuracy', mean_accuracy, self.log_round)
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
        serialized_idx = ndarray_to_proto(np.array([0]))
        serialized_hyperparams = ndarray_to_proto(self.hyperparams)
        self.initial_parameters.tensors.append(serialized_idx.ndarray)
        self.initial_parameters.tensors.append(serialized_hyperparams.ndarray)
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
        self.gain_history.append([hidxs[0], avg_gains])
    
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
        loss, accuracy = _test(self.net, self.test_loader, self.writer, self.log_round)

        # log metrics to tensorboard
        self.writer.add_scalar('Test_Loss', loss, self.log_round)
        self.writer.add_scalar('Test_Accuracy', accuracy, self.log_round)
        log_model_weights(self.net, self.log_round, self.writer)

        # persist model
        torch.save(self.net, './fedex_models/net_round_{}'.format(self.log_round))

        # since evaluate is the last method being called in one round, step rtpt here
        self.rtpt.step()
        return float(loss), {"accuracy": float(accuracy)}