import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sleep", dest='sleep_time', type=int, default=120, help="update interval in secs")

args = parser.parse_args()
job_id = 0000
state = []
bashCommand = "squeue -u s1791387 -p Standard"
while(1):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if output is None:
        put_job(state)

    state = output.decode().split()
    print(state)
    job_id = state[8]
    print(job_id)

    print(output.decode())
    time.sleep(args.sleep_time)

def put_job(state):
    job_id = state[8]
    print(job_id)
