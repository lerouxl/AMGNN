#!/bin/bash --login
#SBATCH -p gpu_v100
#job stdout file
#SBATCH -o AMGNN.%J.log
#job stderr file
#SBATCH -e AMGNN.%J.err
#job name
#SBATCH -J AMGNN
#number of parallel processes (tasks) you are requesting - maps to MPI processes

#maximum job time in D-HH:MM
#SBATCH --time=1-23:50
#memory per process in MB
#SBATCH --mem-per-cpu=8000
#Number of GPU
#SBATCH --gres=gpu:1
#slurm configuration
# REMPLACE GPU_launch by the name of the task ( -o and -e make file with the input name, -J give a job name)

set +eu #in case of error stop the script

module purge
module load anaconda # Load the anaconda library to load the Python environment
module load proxy # Load the proxy system to be able to use internet for the WandB logging and sweep

# Activate AMGNN environement
eval "$(/apps/languages/anaconda/2021.11/bin/conda shell.bash hook)"
source activate AMGNN

# For a Wandb sweep, request the next set of parameters and run once.
wandb agent --count 1 niut/AMGNN/k0fzqk71
#python3 main.py automaticaly launched by wandb
