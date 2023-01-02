import os, sys
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cgpus', nargs='+', type=int)
parser.add_argument('--sgpu', type=int)

args = parser.parse_args()

for s in range(50):
    alpha = np.round(np.random.uniform(0, 1), 3)
    gamma = np.round(np.random.uniform(1, 15), 3)
    rounds = np.random.choice([100, 150, 200, 250, 300])

    os.system(f'python server.py --stage search --alpha {alpha} --gamma {gamma} --rounds {rounds} --gpu {args.sgpu} &')
    os.system('sleep 10') # wait 10 seconds until server is up
    cgpus = [str(gpu) for gpu in args.cgpus]
    cgpus_arg = ' '.join(cgpus)
    os.system(f'python clients.py --stage search --gpus {cgpus_arg}')