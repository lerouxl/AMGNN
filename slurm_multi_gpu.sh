#!/bin/bash --login
#SBATCH -p gpu_v100
#job stdout file
#SBATCH -o AMGNN.%J.log
#job stderr file
#SBATCH -e AMGNN.%J.err
#job name
#SBATCH -J AMGNN
#number of parallel processes (tasks) you are requesting - maps to MPI processes
#SBATCH --ntasks-per-node=2
# Number of nodes to use (one GPU node had 2 GPUs)
#SBATCH --nodes 1

#maximum job time in D-HH:MM
#SBATCH --time=1-12:10
#memory per process in MB
#SBATCH --mem-per-cpu=8000
#Number of GPU
#SBATCH --gres=gpu:2
#slurm configuration
# REMPLACE GPU_launch by the name of the task ( -o and -e make file with the input name, -J give a job name)

set +eu #in case of error stop the script

module purge
module load anaconda # Load the anaconda library to load the Python environment
module load proxy # Load the proxy system to be able to use internet for the WandB logging and sweep
module load CUDA/10.2

# Activate AMGNN environement
eval "$(/apps/languages/anaconda/2021.11/bin/conda shell.bash hook)"
source activate AMGNN

# From https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster_advanced.html#troubleshooting
# There are two parametres in the SLURM submission script that determine how many processes will run your training,
# the #SBATCH --nodes=X setting and #SBATCH --ntasks-per-node=Y settings.
# The numbers there need to match what is configured in your Trainer in the code:
# Trainer(num_nodes=X, devices=Y). If you change the numbers, update them in BOTH places.
srun python main.py --num_nodes 1 --devices 2