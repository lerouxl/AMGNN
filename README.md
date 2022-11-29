# AMGNN : A Graph neural network for additive manufacturing simulation
___

## Use
This graph neural network is intended to predict the state, step by step of an additive manufacturing process.

## Installation
AMGNN environement was developed on Python 3.9.7, you can install it and its library with Anaconda:
``` conda env create -f environment.yml```

If you want to precise where the environement folder will be created, use the prefix arguement:
``` conda env create --prefix ENV_PATH  -f environment.yml```

Install Simufact arc reader with : 
```pip install git+https://github.com/lerouxl/Simufact_ARC_reader.git@main```

## Configuration
### config.yml
General parameters
### sweep.yml
Sweep parameters of wandb
## Documentation generation:
```pdoc -o ./docs ./utils/config.py```

## Inputs description

This neural network is loading graph as inputs.
The nodes features are:

| ID                	| Description                                                  	|
|-------------------	|--------------------------------------------------------------	|
| 0                 	| Laser speed (Normalised by dividing it with `scaling_speed`) 	|
| 1                 	| Laser power (Normalised by dividing it with `scaling_power`) 	|
| 2                 	| Layer thickness                                              	|
| 3                 	| Time step lenght                                             	|
| 4                 	| Time step                                                    	|
| 5,6,7             	| Type of node                                                 	|
| 8                 	| Past node temperature (Normalised by `scaling_temperature`)  	|
| 9                 	| Past X displacement (Normalised by `scaling_deformation`)    	|
| 10                	| Past Y displacement (Normalised by `scaling_deformation`)    	|
| 11                	| Past Z displacement (Normalised by `scaling_deformation`)    	|
| 12                	| Process features                                             	|
| 13<14,15,16,17,18 	| Process category                                             	|
| 19                	| X coordinate (Normalised by `scaling_size`)                  	|
| 20                	| Y coordinate (Normalised by `scaling_size`)                  	|
| 21                	| Z coordinate (Normalised by `scaling_size`)                  	|


## Multi GPUs and nodes supports
If you want to use multiple GPUs and multiple nodes with a SLURM scheduler, you can use the `slurm_multi_gpu.sh` code 
for run with GPUs.

From https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster_advanced.html#troubleshooting
There are two parametres in the SLURM submission script that determine how many processes will run your training,
the #SBATCH --nodes=X setting and #SBATCH --ntasks-per-node=Y settings.
The numbers there need to match what is configured in your Trainer in the code:
Trainer(num_nodes=X, devices=Y). If you change the numbers, update them in BOTH places.
