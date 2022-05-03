from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score
import json
import os

def log_model_weights(model, step, writer):
    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, step)
    writer.flush()

def log_hyper_config(config, step, writer):
    for key, hyperparam in config.items():
        writer.add_scalar(key, hyperparam, step)

def log_hyper_params(hyper_param_dict, file_name):
    to_be_persisted = {k: list(v) for k, v in hyper_param_dict.items()}
    with open(file_name, 'w') as f:
        json.dump(to_be_persisted, f)

def compute_accuracy(logits, y):
    logits = torch.cat(logits)
    y = torch.cat(y)

    y_pred = F.softmax(logits, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)

    y_pred = y_pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    return accuracy_score(y, y_pred)      

def get_hyperparameter_id(name, client_id):
    # hyperparameter-names must have format arbitrary_name_[round_number]
    # thus we cut off "_[round_number]" and add "client_[id]_" to obtain unique
    # log-id for each client such that each hyper-parameter configuration is 
    # logged in one time-diagram per client
    split_name = name.split('_')
    split_name = split_name[:-1]
    log_name = '_'.join(split_name)
    log_name = 'client_{}_'.format(client_id) + log_name
    return log_name

def prepare_log_dirs():
    if not os.path.exists('./hyperparam-logs/'):
        os.mkdir('./hyperparam-logs')
    if not os.path.exists('./models/'):
        os.mkdir('./models')

class ProtobufNumpyArray:
    """
        Class needed to deserialize numpy-arrays coming from flower
    """
    def __init__(self, bytes) -> None:
        self.ndarray = bytes