import subprocess
bashCommand = "squeue -u s1791387"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
print(output)
