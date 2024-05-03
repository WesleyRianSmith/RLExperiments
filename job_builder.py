from os import path

# Base paths and exclusion list
exclude = "dgk610,dgk613,dgk813,dgk822"
base_path = "/job_scripts"
block_exp_replay_path = path.join(base_path, "Block_ExperienceReplay")
logging_path = path.join(base_path, "logging/Atari/{env}/")
checkpoint_path = path.join(base_path, "checkpoints/Atari/{env}/")

# Script template for bash scripts
script_template = f"""#!/bin/bash
# Load necessary modules
module load python/anaconda3
source $condaDotFile
source activate gymnasium

# Display GPU status
nvidia-smi

# Display the Python interpreter location
which python

# to tmp ray
mkdir -p {base_path}/ray/$SLURM_JOB_ID
ln -sf {base_path}/ray/$SLURM_JOB_ID /tmp/ray_$SLURM_JOB_ID

# Define paths
_path="{block_exp_replay_path}"
_log="{logging_path}"
_checkpoint="{checkpoint_path}"

echo "Job_ID: $SLURM_JOB_ID, Task: {{target_file}}.py, Env: {{env}}" >> \\
{base_path}/run_log_atari/jobs_mapping.log

# Change directory to the script's path
cd $_path || exit

# Execute the Python script with parameters
python $_path/{{target_file}}.py \\
    -R $SLURM_JOB_ID \\
    -S $_path/settings/{{config}}.yml \\
    -L $_log \\
    -C $_checkpoint \\
    -E {{env}} \\
    {{extra_args}}
"""

# sbatch template for submitting jobs
sbatch_template = f"""sbatch --mem=100G \\
       --gres=gpu:1 \\
       --cpus-per-task={{num_cpu}} \\
       --time=3-1:00:00 \\
       --partition=small \\
       --exclude={exclude} \\
       {{task_path}}/tasks/{{script_name}}
"""


def create_tasks_and_commands(envs, config, task_type, num_cpu, extra_args):
    commands = []
    for env_name in envs:
        script_name = f"{env_name}_{task_type}.sh"
        # Fill in the logging and checkpoint paths based on environment
        filled_logging_path = logging_path.format(env=env_name)
        filled_checkpoint_path = checkpoint_path.format(env=env_name)
        with open(f"./tasks/{script_name}", "w") as tf:
            tf.write(script_template.format(env=env_name,
                                            target_file=f"atari_{task_type}",
                                            config=config,
                                            extra_args=extra_args,
                                            block_exp_replay_path=block_exp_replay_path,
                                            logging_path=filled_logging_path,
                                            checkpoint_path=filled_checkpoint_path))
        commands.append(
            sbatch_template.format(task_path=block_exp_replay_path, script_name=script_name, num_cpu=num_cpu))
        commands.append("\nsleep 600\n\n")
    return commands

    # "Alien", "Amidar", "Assault", "Asterix", "Asteroids",
    # "Atlantis", "BankHeist", "BeamRider", "Berzerk",
    # "Bowling", "Boxing", "Breakout", "Carnival", "Centipede",
    # "ChopperCommand", "CrazyClimber", "Defender", "FishingDerby", "Freeway",
    # "Frostbite", "Gopher", "Pong", "Qbert", "SpaceInvaders"


# Environment names
all_envs = [
    "Breakout", "BeamRider", "Qbert", "SpaceInvaders",
    "ChopperCommand", "CrazyClimber", "Defender", "FishingDerby", "Freeway",
    "Frostbite", "Gopher", "Pong"
]

# Creating scripts and commands for different types of tasks
commands_dpber = create_tasks_and_commands(all_envs,
                                           "apex_atari",
                                           "dpber",
                                           20,
                                           "-SBZ 32"
                                           )
commands_dper = create_tasks_and_commands(all_envs,
                                          "apex_atari",
                                          "dper",
                                          20,
                                          "-SBZ 1"
                                          )

# Combine all commands
all_commands = commands_dpber + commands_dper

# Write all commands to file
with open("commands_dper.sh", "w") as f:
    f.writelines(all_commands)