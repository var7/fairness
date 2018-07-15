import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sleep", dest='sleep_time', type=int, default=120, help="update interval in secs")

args = parser.parse_args()

bashCommand = "squeue -u s1791387 -p Short"
while(1):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    state = output.decode().split()
    print(state)
    if output is None:
        put_job()
    print(output.decode())
    time.sleep(args.sleep_time)

def put_job():
    return 0
