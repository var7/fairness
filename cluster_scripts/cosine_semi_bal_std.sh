#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o ./bal_logs/cos_semi_std_%j.log  # send stdout to sample_experiment_outfile
#SBATCH -e ./bal_logs/cos_semi_std_%j.err  # send stderr to sample_experiment_errfile
#SBATCH -t 8:00:00  # time requested in hour:minute:secon
#SBATCH -p Standard
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

python balanced_facenet_train.py -j cosine7_randsemi_std -e 24 --num-classes 6 --num-samples 5 --cosine -r -rw /home/s1791387/facescrub-data/new_data_max/balanced_model_weigths/job_cosine5_semi_std_Jul_20_1900hrs/weights_65.pth
