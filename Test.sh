#!/bin/bash
# Load necessary modules
module load python/anaconda3
source $condaDotFile
source activate gymnasium

# Display GPU status
nvidia-smi

# Display the Python interpreter location
which python

# to tmp ray
mkdir -p /jmain02/home/J2AD006/jxb06/cxz46-jxb06//ray/$SLURM_JOB_ID
ln -sf /jmain02/home/J2AD006/jxb06/cxz46-jxb06//ray/$SLURM_JOB_ID /tmp/ray_$SLURM_JOB_ID

# Define paths
_path="/jmain02/home/Disseration210168637"


echo "Job_ID: $SLURM_JOB_ID, Task: atari_dpber_ram_saver.py, Env: Atlantis" >> \
/jmain02/home/J2AD006/jxb06/cxz46-jxb06//run_log_atari/jobs_mapping.log

# Change directory to the script's path
cd $_path || exit

# Execute the Python script with parameters
python $_path/PPOExperiment.py \
    -R $SLURM_JOB_ID \
    -S $_path/Configurations/PPO/BreakoutDefault.yml \
