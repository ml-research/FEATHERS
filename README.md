# HANF (Hyperparameter and Neural Architecture Search in Federated Learning)

## Experiments for Presentation
| Dataset       | Number of Clients | Reult   |
|---------------|-------------------|---------|
| FashionMNIST  | 2                 | 5/5     |
| FashionMNIST  | 5                 | 4/5     |
| FashionMNIST  | 10                | -       |
| CIFAR10       | 2                 | 5/5     |
| CIFAR10       | 5                 | 3/5     |
| CIFAR10       | 10                | -       |

## Experiments for Paper
### iid
| Dataset       | Number of Clients | Reult   |
|---------------|-------------------|---------|
| FashionMNIST  | 2                 | 5/5     |
| FashionMNIST  | 5                 | 5/5     |
| FashionMNIST  | 10                | 0/5     |
| CIFAR10       | 2                 | 0/5     |
| CIFAR10       | 5                 | 3/5     |
| CIFAR10       | 10                | 0/5     |

### non-iid (label skew)
| Dataset       | Number of Clients | Reult   |
|---------------|-------------------|---------|
| FashionMNIST  | 2                 | 0/5     |
| FashionMNIST  | 5                 | 0/5     |
| FashionMNIST  | 10                | 0/5     |
| CIFAR10       | 2                 | 0/5     |
| CIFAR10       | 5                 | 0/5     |
| CIFAR10       | 10                | 0/5     |

## Setting up Experiments
The following shows how to run the two phases of our experiments, i.e. the Search Phase where we try to identify a good neural architecture and the Evaluation Phase where we use the cells found by the Search Phase to evaluate their performance within a bigger network.
Here are two templates for the `config.py` and the `hyperparameters.py` respectively:

### Configuring the Search Phase
The `config.py` should look like:
```python
# hyperparameter configuration parameters
ROUNDS = 120 # nr. of communication rounds
ALPHA = 0.3
HYPERPARAM_CONFIG_NR = 120 # size of hyperparameter search space
BATCH_SIZE = 64

# logging
LOG_DIR = './runs/'

# server parameters
DATASET = 'DATASET' # dataset to use. Alternatives: cifar10
CLIENT_NR = {2, 5}
MIN_TRAIN_CLIENTS = {2, 5} # min. number of clients used during fit
MIN_VAL_CLIENTS = {2, 5} # min. number of clients used during evaluation
REINIT = False # reinitailize model if no improvement was made

# model initilization parameters
CLASSES = 10 # number of output-classes
CELL_NR = 8 # number of cells the search space consists of (if search phase). Else number of cells of the network
IN_CHANNELS = {1, 3} # mumber of input-channels (1 for FMNIST, 3 for CIFAR)
OUT_CHANNELS = 16 # number of output-channels
NODE_NR = 7 # number of nodes per cell

PORT = 'PORT'
GPUS = [7] # GPUs to use
SERVER_GPU = 5

DATA_SKEW = {0, 0.5} # skew of labels. 0 = iid case, 0.5 = our skewed case

# validation stage
DROP_PATH_PROB = 0.2 # probability of dropping a path in cell, similar to dropout
```

The hyperparameter-space definition in `hyperparameters.py` should look like:
```python
    self.sample_hyperparams = lambda: {
        'learning_rate': 10 ** (np.random.uniform(-4, -1)),
        'weight_decay': 10.0 ** np.random.uniform(low=-5.0, high=-1.0),
        'momentum': np.random.uniform(low=0.0, high=1.0),
        'arch_learning_rate': 10 ** (np.random.uniform(-5, -2)), 
        'arch_weight_decay': 10.0 ** np.random.uniform(low=-5.0, high=-1.0),
    }
```


### Configuring the Evaluation Phase
The `config.py` should look like:
```python
# hyperparameter configuration parameters
ROUNDS = 1500 # nr. of communication rounds
ALPHA = 0.3
HYPERPARAM_CONFIG_NR = 120 # size of hyperparameter search space
BATCH_SIZE = 96

# logging
LOG_DIR = './runs/'

# server parameters
DATASET = 'DATASET' # dataset to use. Choices: cifar10, fmnist
CLIENT_NR = {2, 5}
MIN_TRAIN_CLIENTS = {2, 5} # min. number of clients used during fit
MIN_VAL_CLIENTS = {2, 5} # min. number of clients used during evaluation
REINIT = False # reinitailize model if no improvement was made

# model initilization parameters
CLASSES = 10 # number of output-classes
CELL_NR = 20 # number of cells the search space consists of (if search phase). Else number of cells of the network
IN_CHANNELS = {1, 3} # mumber of input-channels (1 for FMNIST, 3 for CIFAR)
OUT_CHANNELS = 36 # number of output-channels
NODE_NR = 7 # number of nodes per cell

PORT = 'PORT'
GPUS = [7] # GPUs to use
SERVER_GPU = 5

DATA_SKEW = {0, 0.5} # skew of labels. 0 = iid case, 0.5 = our skewed case

# validation stage
DROP_PATH_PROB = 0.2 # probability of dropping a path in cell, similar to dropout
```

The hyperparameter-space definition in `hyperparameters.py` should look like:
```python
    self.sample_hyperparams = lambda: {
        'learning_rate': 10 ** (np.random.uniform(-4, -1)),
        'weight_decay': 10.0 ** np.random.uniform(low=-5.0, high=-1.0),
        'momentum': np.random.uniform(low=0.0, high=1.0),
        'dropout': np.random.uniform(low=0, high=0.2),
    }
```

## Starting the Experiments
Starting the experiments is straight forward:

### Search Phase
First run `python server.py --stage search` in a terminal. In a different terminal you can then run `python clients.py --stage search` and the experiments will start.

### Evaluation Phase
First run `python server.py --stage valid` in a terminal. In a different terminal you can then run `python clients.py --stage valid` and the experiments will start.