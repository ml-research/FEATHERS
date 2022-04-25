# hyperparameter configuration parameters
BETA = 0.5 # determines how widespread distribution adjustments are done
ROUNDS = 50 # nr. of communication rounds
EPSILON = 0.9 # determines max. probability mass one configuration can have
ETA = 0.01 # Learning rate to update distribution

# logging
LOG_DIR = './runs/'

# server parameters
DATASET = 'fmnist' # dataset to use. Alternatives: cifar10
CLIENT_NR = 2
MIN_TRAIN_CLIENTS = 2 # min. number of clients used during fit
MIN_VAL_CLIENTS = 2 # min. number of clients used during evaluation
REINIT = False # reinitailize model if no improvement was made

# model initilization parameters
CLASSES = 10 # number of output-classes
CELL_NR = 4 # number of cells the search space consists of
IN_CHANNELS = 1 # mumber of input-channels (e.g. 3 for rgb-images)
OUT_CHANNELS = 16 # number of output-channels
NODE_NR = 7 # number of nodes per cell