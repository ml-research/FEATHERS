from asyncio import subprocess
from subprocess import Popen
import config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stage', default='search', type=str)
parser.add_argument('--gpus', nargs='+', type=int)

args = parser.parse_args()

processes = []
for c in range(config.CLIENT_NR):
    gpu_idx = c % len(args.gpus)
    if args.stage == 'search':
        process = Popen(['python', 'hanf_client.py', '--gpu', str(args.gpus[gpu_idx]), '--id', str(c)])
    else:
        process = Popen(['python', 'hanf_client_valid.py', '--gpu', str(args.gpus[gpu_idx]), '--id', str(c)])
    processes.append(process)

for p in processes:
    p.wait()