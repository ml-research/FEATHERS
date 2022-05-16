# hyperparameter configuration parameters
ROUNDS = 120 # nr. of communication rounds
ALPHA = 0.3
HYPERPARAM_CONFIG_NR = 120 # size of hyperparameter search space
BATCH_SIZE = 64

# logging
LOG_DIR = './runs/'

# server parameters
DATASET = 'cifar10' # dataset to use. Alternatives: cifar10
CLIENT_NR = 2
MIN_TRAIN_CLIENTS = 2 # min. number of clients used during fit
MIN_VAL_CLIENTS = 2 # min. number of clients used during evaluation
REINIT = False # reinitailize model if no improvement was made

# model initilization parameters
CLASSES = 10 # number of output-classes
CELL_NR = 8 # number of cells the search space consists of (if search phase). Else number of cells of the network
IN_CHANNELS = 3 # mumber of input-channels (e.g. 3 for rgb-images)
OUT_CHANNELS = 16 # number of output-channels
NODE_NR = 7 # number of nodes per cell

PORT = '8068'
GPUS = [1, 2] # GPUs to use
SERVER_GPU = 2

DATA_SKEW = 0.5 # skew of labels. 0 = no skew, 1 only some clients hold some labels

# validation stage
DROP_PATH_PROB = 0.2 # probability of dropping a path in cell, similar to dropout