path = "/jmain02/home/J2AD006/jxb06/cxz46-jxb06/csc210168637/RLExperiments"
def write_bash_script(bash_name,experiment_script, configure_file):
    bash_content = f"""#!/bin/bash
# Load necessary modules
module load python/anaconda3
source $condaDotFile
source activate csc1016863

# Display GPU status
nvidia-smi

# Display the Python interpreter location
which python

# Define paths
_path="{path}"

# Change directory to the script's path
cd $_path || exit

# Execute the Python script with parameters
python $_path/{experiment_script}.py -C $_path/Configurations/PPO/{configure_file}.yml
"""

    # Write the bash script to a file
    with open(f'BashScripts/{bash_name}.sh', 'w') as file:
        file.write(bash_content)

    print("Bash script 'run_experiment.sh' has been created.")

if __name__ == '__main__':
    write_bash_script("BreakoutDefault","PPOExperiment","BreakoutDefault")
