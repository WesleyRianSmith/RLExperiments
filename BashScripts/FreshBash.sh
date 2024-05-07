#!/bin/bash
module load python/anaconda3
source $condaDotFile
source activate csc1016863

nvidia-smi

which python

_path="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/csc210168637/RLExperiments"
cd $_path || exit
python $_path/PPOExperiment.py -C $_path/Configurations/PPO/BreakoutDedsdsdwadddault.yml

