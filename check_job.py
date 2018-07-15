import subprocess
import time
import argparse
import re

weights_pattern = re.compile("^([A-Z][0-9]+)+$")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sleep", dest='sleep_time', type=int, default=120, help="update interval in secs")

args = parser.parse_args()
job_id = 0000
state = []
bashCommand = "squeue -u s1791387 -p Short"
while(1):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    op = output.decode.split()
    print(op)
    if len(op) < 8:
        put_job(state)
    else:
        state = op
        print(state)
        print(output.decode())
    time.sleep(args.sleep_time)

def put_job(state):
    job_id = state[8]
    print(job_id)
    filename = "./logs/shrt_{}_triplet.log".format(job_id)
    with open(filename, "r") as f:
        for line in f:
