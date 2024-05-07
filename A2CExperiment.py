import yaml
import experiment
import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def main(config):
    directory = f"ExperimentModels"
    if not os.path.exists(directory):
        os.mkdir(directory)
    experiment_name = config["experiment_name"]
    env_id = config["env_id"]
    n_envs = config["n_envs"]
    n_stack = config["n_stack"]
    eval_frequency = config["eval_frequency"]

    env = make_atari_env(env_id, n_envs=n_envs, seed=0)
    env = VecFrameStack(env, n_stack=n_stack)
    hyperparameters = config["hyperparameters"]

    model = A2C(env=env, **hyperparameters)
    experiment.InitialiseExperiment(
        experiment_name=experiment_name, model=model, model_architecture="A2C",
        env_id=env_id, n_envs=n_envs, n_stack=n_stack, eval_frequency=eval_frequency,
        hyper_parameters=hyperparameters
    )


    experiment.TrainExperiment(experiment_name, config["steps_to_train"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run PPO experiments with configurations from a YAML file.')
    parser.add_argument('-C', '--config', type=str, required=True, help='Path to configuration YAML file')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
