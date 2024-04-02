import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
import json
import torch
import pandas as pd
import optuna
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available.")


def TrainAndEvaluate(env_id, model, total_steps):

    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    model.learn(total_steps)
    return model


def InitialiseExperiment(experiment_name, model, model_architecture, env_id, hyper_parameters):

    # Create a directory for the experiment
    directory = experiment_name
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        print(f"Directory '{directory}' already exists")
        return

    hyper_parameters_path = os.path.join(directory, "hyperparameters.txt")

    with open(hyper_parameters_path, 'w') as f:
        f.write("Initial hyperparameters, changes will not affect the model\n")
        f.write("---------------------------------------------------------\n")
        for key, value in hyper_parameters.items():
            f.write(f"{key}: {value}\n")

    # Save metadata in the experiment directory
    metadata_path = os.path.join(directory, "metadata.json")
    meta_data = {
        "env_id": env_id,
        "model_architecture": model_architecture,
        "steps_trained": 0,
    }
    with open(metadata_path, 'w') as file:
        json.dump(meta_data, file)

    # Save the initial model in the experiment directory
    new_directory = os.path.join(directory, f"{env_id}-{model_architecture}-0")
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)
    model_save_path = f"{new_directory}/model"
    model.save(model_save_path)

def TrainExperiment(experiment_name,steps):
    metadata_path = f"{experiment_name}/metadata.json"
    meta_data = {}
    with open(metadata_path, 'r') as file:
        meta_data = json.load(file)
    new_steps_trained = meta_data.get("steps_trained") + steps
    model_name = meta_data.get("env_id") + "-" + meta_data.get("model_architecture") + "-" + str(new_steps_trained)
    new_directory = f"{experiment_name}/" + model_name
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)

    env_id = meta_data.get("env_id")
    model_architecture = meta_data.get("model_architecture")
    old_model_name = env_id + "-" + model_architecture + "-" + str(meta_data.get("steps_trained"))
    model_path = f"{experiment_name}/{old_model_name}/model"
    tmp_path = f"{new_directory}/metric_logs"
    new_logger = configure(tmp_path, ["stdout", "csv"])
    if model_architecture == "PPO":
        model = PPO.load(model_path)
    elif model_architecture == "DQN":
        model = DQN.load(model_path)
    elif model_architecture == "A2C":
        model = A2C.load(model_path)
    else:
        print("No valid architecture")
        return
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    model.set_env(env)
    model.set_logger(new_logger)
    model = TrainAndEvaluate(env_id,model,steps)
    meta_data["steps_trained"] = new_steps_trained
    metadata_path = f"{experiment_name}/metadata.json"
    with open(metadata_path, 'w') as file:
        json.dump(meta_data, file)
    model_save_path = f"{new_directory}/model"
    model.save(model_save_path)












def TuneHyperparameters(model_architecture, env_id,steps ,set_params, hyper_parameter_ranges):
    def objective(trial):
        env = make_atari_env(env_id, n_envs=1, seed=0)
        env = VecFrameStack(env, n_stack=4)

        current_params = {}
        for k, v in hyper_parameter_ranges.items():
            if v[2] == "float":
                param_value = trial.suggest_float(k,v[0],v[1],log=v[3])
            elif v[2] == "int":
                param_value = trial.suggest_int(k, v[0], v[1], log=v[3])
            else:
                continue
            current_params[k] = param_value
        print(current_params)
        if model_architecture == "PPO":
            model = PPO(env=env, **set_params,**current_params)
        elif model_architecture == "DQN":
            model = DQN(env=env, **set_params, **current_params)
        elif model_architecture == "A2C":
            model = A2C(env=env, **set_params, **current_params)
        else:
            print("No valid architecture")
            return

        # Train the model
        model.learn(total_timesteps=steps)

        # Evaluate the model, you can define your own evaluation function
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

        return mean_reward
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    return trial.params