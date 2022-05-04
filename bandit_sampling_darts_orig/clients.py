from asyncio import subprocess
from subprocess import Popen
import config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stage', default='search', type=str)

args = parser.parse_args()

processes = []
for c in range(config.CLIENT_NR):
    gpu_idx = c % len(config.GPUS)
    if args.stage == 'search':
        process = Popen(['python', 'hanf_client.py', '--gpu', str(config.GPUS[gpu_idx])])
    else:
        process = Popen(['python', 'hanf_client_valid.py', '--gpu', str(config.GPUS[gpu_idx])])
    processes.append(process)

for p in processes:
    p.wait()