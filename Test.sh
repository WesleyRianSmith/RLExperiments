#!/bin/bash
# Load necessary modules
module load python/anaconda3
source $condaDotFile
source activate RLDissertationProject

# Display GPU status
nvidia-smi

# Display the Python interpreter location
which python

# Define paths
_path="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/csc210168637/RLExperiments"

# Change directory to the script's path
cd $_path || exit

# Execute the Python script with parameters
python $_path/PPOExperiment.py \
    -R $SLURM_JOB_ID \
    -S $_path/Configurations/PPO/BreakoutDefault.yml \
