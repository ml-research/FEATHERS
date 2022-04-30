from asyncio import subprocess
from subprocess import Popen
import config

processes = []
for _ in range(config.CLIENT_NR):
    process = Popen(['python', 'hanf_client.py'])
    processes.append(process)

for p in processes:
    p.wait()