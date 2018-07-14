import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sleep", dest='sleep_time', type=int, default=120, help="update interval in secs")

args = parser.parse_args()

bashCommand = "squeue -u s1791387"
while(1):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output.decode())
    time.sleep(args.sleep_time)
