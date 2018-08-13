#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o jupyter_%j.log  # send stdout to sample_experiment_outfile
#SBATCH -e jupyter_%j.err  # send stderr to sample_experiment_errfile
#SBATCH -t 2:00:00  # time requested in hour:minute:secon
#SBATCH -p Interactive
export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-6.0_8.0

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/
# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate fairness

export XDG_RUNTIME_DIR=${TMPDIR}
 
port=$(shuf -i 20000-30000 -n 1)
 
echo -e "\nStarting Jupyter Notebook on port ${port} on the $(hostname) server."
echo -e "\nSSH tunnel command: ssh -NL ${port}:$(hostname):${port} ${USER}@ssh.inf.ed.ac.uk"
echo -e "\nLocal URI: http://localhost:${port}"
jupyter lab --no-browser --port=${port} --ip='*'
 
date
