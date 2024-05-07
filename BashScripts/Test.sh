#!/bin/bash
# Load necessary modules
_path="/home/wesleyubuntu/RLExperiments"

# Change directory to the script's path
cd $_path || exit

# Execute the Python script with parameters
python3 $_path/PPOExperiment.py -C $_path/Configurations/PPO/BreakoutDefault.yml

