path = "/jmain02/home/J2AD006/jxb06/cxz46-jxb06/csc210168637/RLExperiments"
def write_bash_script(bash_name,experiment_script, configure_file):
    
	
    bash_content = f"""#!/bin/bash
module load python/anaconda3
source $condaDotFile
source activate csc1016863
    
nvidia-smi
    
which python
    
_path="{path}"
cd $_path || exit
python $_path/{experiment_script}.py -C $_path/Configurations/PPO/{configure_file}.yml
"""
    # Write the bash script to a file
    with open(f'BashScripts/{bash_name}.sh', 'w') as file:
        file.write(bash_content)

    print("Bash script 'run_experiment.sh' has been created.")

if __name__ == '__main__':
    write_bash_script("BreakoutDefault","PPOExperiment","BreakoutDefault")
