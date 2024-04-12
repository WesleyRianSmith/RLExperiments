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


def TrainAndEvaluate(env_id, n_envs, n_stack, model, total_steps):

    env = make_atari_env(env_id, n_envs=n_envs, seed=0)
    env = VecFrameStack(env, n_stack=n_stack)
    model.learn(total_steps)
    return model


def InitialiseExperiment(experiment_name, model, model_architecture, env_id, n_envs, n_stack, hyper_parameters):

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
        "n_envs": n_envs,
        "n_stack": n_stack,
        "model_architecture": model_architecture,
        "steps_trained": 0,
    }
    with open(metadata_path, 'w') as file:
        json.dump(meta_data, file)

    # Save the initial model in the experiment directory
    coded_env_id = env_id.replace("/","-")
    new_directory = os.path.join(directory, f"{coded_env_id}-{model_architecture}-0")
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
    coded_env_id = meta_data.get("env_id").replace("/","-")
    model_name = coded_env_id + "-" + meta_data.get("model_architecture") + "-" + str(new_steps_trained)
    new_directory = f"{experiment_name}/" + model_name
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)

    env_id = meta_data.get("env_id")
    model_architecture = meta_data.get("model_architecture")
    old_model_name = coded_env_id + "-" + model_architecture + "-" + str(meta_data.get("steps_trained"))
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
    env = make_atari_env(env_id, n_envs=meta_data.get("n_envs"), seed=0)
    env = VecFrameStack(env, n_stack=meta_data.get("n_stack"))
    model.set_env(env)
    model.set_logger(new_logger)
    model = TrainAndEvaluate(env_id,meta_data.get("n_envs"),meta_data.get("n_stack"),model,steps)
    meta_data["steps_trained"] = new_steps_trained
    metadata_path = f"{experiment_name}/metadata.json"
    with open(metadata_path, 'w') as file:
        json.dump(meta_data, file)
    model_save_path = f"{new_directory}/model"
    model.save(model_save_path)












def TuneHyperparameters(trial_name,model_architecture, env_id, n_envs, n_stack,steps ,set_params, hyper_parameter_ranges):
    def objective(trial):
        env = make_atari_env(env_id, n_envs=n_envs, seed=0)
        env = VecFrameStack(env, n_stack=n_stack)

        current_params = {}
        # v index mapping:
        # v[0] is lower value bound
        # v[1] is upper value bound
        # v[2] is value type
        # v[3] is whether suggestion is logarithmic
        # v[4] is what value is rounded to
        for k, v in hyper_parameter_ranges.items():
            if v[2] == "float":
                param_value = trial.suggest_float(k,v[0],v[1],log=v[3])
            elif v[2] == "int":
                param_value = trial.suggest_int(k, v[0], v[1], log=v[3])
            else:
                continue
            param_value = round(param_value, v[4])
            current_params[k] = param_value
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
        trial_info = {
            "hyperparameters": current_params,
            "mean_reward": mean_reward
        }
        trial_file = os.path.join(new_directory, f"trial_{trial.number}.json")
        with open(trial_file, 'w') as f:
            json.dump(trial_info, f)
        return mean_reward

    directory = "HyperParameterTuning"
    if not os.path.exists(directory):
        os.mkdir(directory)
    new_directory = os.path.join(directory, trial_name)
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)
    print("Best trial:")
    best_trial = study.best_trial

    print(f"  Mean Reward: {best_trial.value}")
    print("  Params: ")
    rounded_trial_params = {}
    for key, value in best_trial.params.items():
        rounded_value = round(value,hyper_parameter_ranges[key][4])
        print(f"    {key}: {rounded_value}")
        rounded_trial_params[key] = rounded_value
    trial_info = {
        "hyperparameters": rounded_trial_params,
        "mean_reward": best_trial.value
    }
    trial_file = os.path.join(new_directory, f"best_trial.json")
    with open(trial_file, 'w') as f:
        json.dump(trial_info, f)
    return rounded_trial_params