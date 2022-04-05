from typing import Dict, List, Optional, Tuple
import flwr as fl
import numpy as np
import pandas as pd
from numproto import proto_to_ndarray, ndarray_to_proto
from helpers import ProtobufNumpyArray, log_model_weights, log_hyper_configs, log_hyper_params
from utils import FashionMNISTLoader, discounted_mean
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from rtpt import RTPT

DEVICE = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")

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

    def __init__(self, fraction_fit, fraction_eval, initial_net, log_dir='./runs/', epsilon=0.8, beta=1, **args) -> None:
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
        self.distribution, self.fixed_uniform = np.ones(len(self.hyperparams)) / len(self.hyperparams), np.ones(len(self.hyperparams)) / len(self.hyperparams)
        self.epsilon = epsilon
        self.beta = beta
        self.net = initial_net
        self.net.to(DEVICE)
        initial_params = [param.cpu().detach().numpy() for _, param in self.net.state_dict().items()]
        self.initial_parameters = self.last_weights = fl.common.weights_to_parameters(initial_params)
        fashion_mnist_iterator = FashionMNISTLoader.instance(2)
        self.test_data = fashion_mnist_iterator.get_test()
        self.test_loader = DataLoader(self.test_data, batch_size=64, pin_memory=True, num_workers=2)
        self.current_round = 1
        self.writer = SummaryWriter(log_dir)
        self.rtpt = RTPT('JS', 'HANF_Server', 20)
        self.rtpt.start()
        self.distribution_history = []
        self.gain_history = [] # initialize with [0] to avoid nan-values in discounted mean

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
        #if model_improved(results, weights):
        #    aggregated_weights, _ = super().aggregate_fit(rnd, results, failures)
        #    self.last_weights = aggregated_weights
        #else:
        #    print("Model did not improve: Use last parameters") # NOTE: HANF!
        #    aggregated_weights = self.last_weights
        aggregated_weights, _ = super().aggregate_fit(rnd, results, failures)

        # log current distribution
        self.distribution_history.append(self.distribution)
        dh = np.array(self.distribution_history)
        df = pd.DataFrame(dh)
        df.to_csv('distribution_history.csv')

        # update distribution NOTE: Activate again for full HANF!
        gains = self.compute_gains(weights, results)
        self.update_distribution(gains, weights)

        #self.writer.add_histogram('gains', gains, self.current_round)
        hyper_configs = [res.metrics for _, res in results]
        log_hyper_configs(hyper_configs, self.current_round, self.writer)
        
        # sample hyperparameters and append them to the parameters
        hyp_param, hidx = self._sample_hyperparams()
        serialized_hparam = ndarray_to_proto(np.array([hyp_param]))
        serialized_hidx = ndarray_to_proto(np.array([hidx]))
        aggregated_weights.tensors.append(serialized_hparam.ndarray)
        aggregated_weights.tensors.append(serialized_hidx.ndarray)
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
        hyp_param, hidx = self._sample_hyperparams()
        serialized_hparam = ndarray_to_proto(np.array([hyp_param]))
        serialized_hidx = ndarray_to_proto(np.array([hidx]))
        self.initial_parameters.tensors.append(serialized_hparam.ndarray)
        self.initial_parameters.tensors.append(serialized_hidx.ndarray)
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
        baseline = discounted_mean(np.array(self.gain_history), 0.9) if len(self.gain_history) > 0 else 0.0
        for hidx, al, bl, w in zip(hidxs, after_losses, before_losses, weights):
            gain = w * ((al - bl) - baseline)
            client_gains = np.zeros(len(self.hyperparams))
            client_gains[hidx] = gain
            gains.append(client_gains)
        gains = np.array(gains)
        gains = gains.sum(axis=0)
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
        gains = gains / self.distribution * np.sum(weights)
        # bring gains into range [-100, 100] to avoid overflows in exp()
        gains[gains > 100] = 100
        gains[gains < -100] = -100
        self.distribution = self.distribution * np.exp(-(0.1 * gains))
        self.distribution = self.distribution / self.distribution.sum()
        # bound distribution's max-probability
        diffs = self.distribution - self.epsilon
        js = np.argwhere(diffs > 0)
        for j in js:
            for idx in range(len(self.distribution) - 1):
                i = idx - j
                indicator = 1 if i != 0 else -1
                self.distribution[idx] = self.distribution[idx] + indicator * np.exp(-self.beta * abs(i)) * diffs[j]
        self.distribution = self.distribution / self.distribution.sum()

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
        log_model_weights(self.net, self.current_round, self.writer)

        # persist model
        torch.save(self.net, './models/net_round_{}'.format(self.current_round))
        self.current_round += 1

        # log graph in the end
        if self.current_round >= 20:
            X, y = next(iter(self.test_loader))
            X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            self.writer.add_graph(self.net, X)

        # since evaluate is the last method being called in one round, step rtpt here
        self.rtpt.step()
        return float(loss), {"accuracy": float(accuracy)}