path = "/jmain02/home/J2AD006/jxb06/cxz46-jxb06/csc210168637/RLExperiments"
def write_bash_script(bash_name,experiment_script,architecture, configure_file):
    
	
    bash_content = f"""#!/bin/bash
module load python/anaconda3
source $condaDotFile
source activate csc1016863

nvidia-smi

which python

_path="{path}"
cd $_path || exit
python $_path/{experiment_script}.py -C $_path/Configurations/{architecture}/{configure_file}.yml
"""
    # Write the bash script to a file
    with open(f'BashScripts/{bash_name}.sh', 'w') as file:
        file.write(bash_content)

    print(f"Bash script '{experiment_script}.sh' has been created.")

if __name__ == '__main__':
    write_bash_script("PongPPO_Default","PPOExperiment","PPO","PongPPO_Default")
    write_bash_script("PongPPO_Tuned", "PPOExperiment","PPO", "PongPPO_Tuned")
    write_bash_script("PongDQN_Default", "DQNExperiment","DQN", "PongDQN_Default")
    write_bash_script("PongDQN_Tuned", "DQNExperiment","DQN", "PongDQN_Tuned")
    write_bash_script("PongA2C_Default", "A2CExperiment","A2C", "PongA2C_Default")
    write_bash_script("PongA2C_Tuned", "A2CExperiment","A2C", "PongA2C_Tuned")

    write_bash_script("BreakoutPPO_Default", "PPOExperiment","PPO", "BreakoutPPO_Default")
    write_bash_script("BreakoutPPO_Tuned", "PPOExperiment","PPO", "BreakoutPPO_Tuned")
    write_bash_script("BreakoutDQN_Default", "DQNExperiment","DQN", "BreakoutDQN_Default")
    write_bash_script("BreakoutDQN_Tuned", "DQNExperiment","DQN", "BreakoutDQN_Tuned")
    write_bash_script("BreakoutA2C_Default", "A2CExperiment","A2C", "BreakoutA2C_Default")
    write_bash_script("BreakoutA2C_Tuned", "A2CExperiment","A2C", "BreakoutA2C_Tuned")

    write_bash_script("SpaceInvadersPPO_Default", "PPOExperiment","PPO", "SpaceInvadersPPO_Default")
    write_bash_script("SpaceInvadersPPO_Tuned", "PPOExperiment","PPO", "SpaceInvadersPPO_Tuned")
    write_bash_script("SpaceInvadersDQN_Default", "DQNExperiment","DQN", "SpaceInvadersDQN_Default")
    write_bash_script("SpaceInvadersDQN_Tuned", "DQNExperiment","DQN", "SpaceInvadersDQN_Tuned")
    write_bash_script("SpaceInvadersA2C_Default", "A2CExperiment","A2C", "SpaceInvadersA2C_Default")
    write_bash_script("SpaceInvadersA2C_Tuned", "A2CExperiment","A2C", "SpaceInvadersA2C_Tuned")

    write_bash_script("PacManPPO_Default", "PPOExperiment","PPO", "PacManPPO_Default")
    write_bash_script("PacManPPO_Tuned", "PPOExperiment","PPO", "PacManPPO_Tuned")
    write_bash_script("PacManDQN_Default", "DQNExperiment","DQN", "PacManDQN_Default")
    write_bash_script("PacManDQN_Tuned", "DQNExperiment","DQN", "PacManDQN_Tuned")
    write_bash_script("PacManA2C_Default", "A2CExperiment","A2C", "PacManA2C_Default")
    write_bash_script("PacManA2C_Tuned", "A2CExperiment","A2C", "PacManA2C_Tuned")