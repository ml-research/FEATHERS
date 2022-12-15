# hyperparameter configuration parameters
ROUNDS = 500 # nr. of communication rounds
ALPHA = 0.5
GAMMA = 6
HYPERPARAM_CONFIG_NR = 120 # size of hyperparameter search space
BATCH_SIZE = 96

# logging
LOG_DIR = './runs/'

# server parameters
DATASET = 'cifar10' # dataset to use. Alternatives: cifar10
CLIENT_NR = 5
MIN_TRAIN_CLIENTS = 5 # min. number of clients used during fit
MIN_VAL_CLIENTS = 5 # min. number of clients used during evaluation
REINIT = False # reinitailize model if no improvement was made

# model initilization parameters
CLASSES = 10 # number of output-classes
CELL_NR = 8 # number of cells the search space consists of (if search phase). Else number of cells of the network
IN_CHANNELS = 3 # mumber of input-channels (e.g. 3 for rgb-images)
OUT_CHANNELS = 16 # number of output-channels
NODE_NR = 7 # number of nodes per cell
FRAUD_DETECTION_IN_DIM = 7

PORT = '8042'
GPUS = [1, 4, 5, 6, 7] # GPUs to use
SERVER_GPU = 1

DATA_SKEW = 0 # skew of labels. 0 = no skew, 1 only some clients hold some labels
USE_WEIGHTED_SAMPLER = False # use a weighted random sampler to account for class imbalances 

# validation stage
DROP_PATH_PROB = 0.2 # probability of dropping a path in cell, similar to dropout

HYPERPARAM_FILE = f'./hyperparam-logs/search_DARTS_{DATASET}_{CLIENT_NR}_{DATA_SKEW}.csv'
DATASET_INDS_FILE = './hyperparam-logs/indices.json'