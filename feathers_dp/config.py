# hyperparameter configuration parameters
ROUNDS = 1500 # nr. of communication rounds
ALPHA = 0.3
GAMMA = 6
HYPERPARAM_CONFIG_NR = 120 # size of hyperparameter search space
BATCH_SIZE = 96

# logging
LOG_DIR = './runs/'

# server parameters
DATASET = 'fraud' # dataset to use. Alternatives: cifar10, fmnist, imagenet, fraud
CLIENT_NR = 10
MIN_TRAIN_CLIENTS = 10 # min. number of clients used during fit
MIN_VAL_CLIENTS = 10 # min. number of clients used during evaluation
REINIT = False # reinitailize model if no improvement was made

# model initilization parameters
CLASSES = 2 # number of output-classes
CELL_NR = 14 # number of cells the search space consists of (if search phase). Else number of cells of the network
IN_CHANNELS = 1 # mumber of input-channels (e.g. 3 for rgb-images)
OUT_CHANNELS = 16 # number of output-channels
NODE_NR = 7 # number of nodes per cell
FRAUD_DETECTION_IN_DIM = 7
NET_IN_DIMS = [7, 5, 3]
NET_OUT_DIMS = [5, 3, 2]

PORT = '8098'
GPUS = [6] # GPUs to use
SERVER_GPU = 6

DATA_SKEW = 0 # skew of labels. 0 = no skew, 1 only some clients hold some labels
USE_WEIGHTED_SAMPLER = True

# Differential Privacy
MAX_GRAD_NORM = 1.
DP_SIGMA_REWARD = 1e-4
NOISE_MP = 1.5

# validation stage
DROP_PATH_PROB = 0.2 # probability of dropping a path in cell, similar to dropout

HYPERPARAM_FILE = f'./hyperparam-logs/search_DARTS_{DATASET}_{CLIENT_NR}_{DATA_SKEW}.csv'
DATASET_INDS_FILE = './hyperparam-logs/indices.json'