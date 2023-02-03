# hyperparameter configuration parameters
ROUNDS = 1500 # nr. of communication rounds
ALPHA = 0.3
HYPERPARAM_CONFIG_NR = 120 # size of hyperparameter search space
BATCH_SIZE = 96

# logging
LOG_DIR = './runs/'

# server parameters
DATASET = 'fmnist' # dataset to use. Alternatives: cifar10
CLIENT_NR = 100
MIN_TRAIN_CLIENTS = 100 # min. number of clients used during fit
MIN_VAL_CLIENTS = 100 # min. number of clients used during evaluation

# model initilization parameters
CLASSES = 10 # number of output-classes
IN_CHANNELS = 1 # mumber of input-channels (e.g. 3 for rgb-images)
OUT_CHANNELS = 36 # number of output-channels
CELLS = 20

PORT = '8005'
GPUS = [0, 1, 3, 4, 6, 7] # GPUs to use
SERVER_GPU = 7

DATA_SKEW = 0.0 # skew of labels. 0 = no skew, 1 only some clients hold some labels

# validation stage
DROP_PATH_PROB = 0.3 # probability of dropping a path in cell, similar to dropout

DATASET_INDS_FILE = f'./hyperparam-logs/search_DARTS_{DATASET}_{CLIENT_NR}_{DATA_SKEW}.csv'
HYPERPARAM_FILE = './hyperparam-logs/indices.json'