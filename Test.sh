#!/bin/bash
# Load necessary modules
module load python/anaconda3
source $condaDotFile
source activate RLProject

# Display GPU status
nvidia-smi

# Display the Python interpreter location
which python

# Define paths
_path="C:\Users\smith\PycharmProjects\Diss"

# Change directory to the script's path
cd $_path || exit

# Execute the Python script with parameters
python $_path/PPOExperiment.py \
    -R $SLURM_JOB_ID \
    -S $_path/Configurations/PPO/BreakoutDefault.yml \
