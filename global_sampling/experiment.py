import subprocess
import time
from datetime import datetime as dt

def run_hanf(runs=20):
    for r in range(runs):
        log_dir = './runs/hanf_{}'.format(r)
        rounds = 30
        beta = 0.5
        epsilon = 0.7
        cmd_server = ['python', 'server.py', '--log-dir', log_dir, '--rounds', str(rounds), '--beta', str(beta), '--epsilon', str(epsilon)]
        cmd_client = ['python',  'hanf_client.py']
        server_process = subprocess.Popen(cmd_server, stdout=subprocess.PIPE)
        # wait until server is up
        time.sleep(7)
        client_1_process = subprocess.Popen(cmd_client, stdout=subprocess.PIPE)
        client_2_process = subprocess.Popen(cmd_client, stdout=subprocess.PIPE)
        
        # wait for server to terminate (implies that clients have terminated)
        server_process.wait()

run_hanf()
