import subprocess
import time
from datetime import datetime as dt

def run_fedex(runs=20):
    for r in range(runs):
        log_dir = './runs/fedex_{}'.format(r)
        rounds = 30
        beta = 0.5
        epsilon = 0.7
        cmd_server = ['python', 'fedex_server.py', '--log-dir', log_dir, '--rounds', str(rounds), '--beta', str(beta), '--epsilon', str(epsilon)]
        cmd_client = ['python',  'fedex_client.py']
        server_process = subprocess.Popen(cmd_server, stdout=subprocess.PIPE)
        # wait until server is up
        time.sleep(15)
        client_1_process = subprocess.Popen(cmd_client, stdout=subprocess.PIPE)
        client_2_process = subprocess.Popen(cmd_client, stdout=subprocess.PIPE)
        
        # wait for server to terminate (implies that clients have terminated)
        server_process.wait()

run_fedex()
