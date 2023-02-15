# hyperparameter configuration parameters
ROUNDS = 500 # nr. of communication rounds
ALPHA = 0.3
HYPERPARAM_CONFIG_NR = 240 # size of hyperparameter search space
BATCH_SIZE = 128

# logging
LOG_DIR = './runs/'

# server parameters
DATASET = 'imagenet' # dataset to use. Alternatives: cifar10
CLIENT_NR = 5
MIN_TRAIN_CLIENTS = 5 # min. number of clients used during fit
MIN_VAL_CLIENTS = 5 # min. number of clients used during evaluation

# model initilization parameters
CLASSES = 200 # number of output-classes
IN_CHANNELS = 3 # mumber of input-channels (e.g. 3 for rgb-images)
OUT_CHANNELS = 48 # number of output-channels
CELLS = 14

PORT = '8005'
GPUS = [0, 5, 6] # GPUs to use
SERVER_GPU = 6

DATA_SKEW = 0.0 # skew of labels. 0 = no skew, 1 only some clients hold some labels

# validation stage
DROP_PATH_PROB = 0.3 # probability of dropping a path in cell, similar to dropout

DATASET_INDS_FILE = f'./hyperparam-logs/search_DARTS_{DATASET}_{CLIENT_NR}_{DATA_SKEW}.csv'
HYPERPARAM_FILE = './hyperparam-logs/indices.json'