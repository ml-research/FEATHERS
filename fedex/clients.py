from asyncio import subprocess
from subprocess import Popen
import config
import argparse


processes = []
for c in range(config.CLIENT_NR):
    gpu_idx = c % len(config.GPUS)
    process = Popen(['python', 'fedex_client.py', '--gpu', str(config.GPUS[gpu_idx]), '--id', str(c)])
    processes.append(process)

for p in processes:
    p.wait()