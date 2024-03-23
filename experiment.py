import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import json
import torch
import pandas as pd
import optuna
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available.")


def TrainAndEvaluate(env_id, model, df, total_steps, evaluation_count, current_steps):

    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    steps_per_eval = total_steps // evaluation_count
    for i in range(evaluation_count):
        model.learn(steps_per_eval)
        eval_env = VecFrameStack(make_atari_env(env_id, n_envs=1, seed=0), n_stack=4)
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
        df.loc[len(df)] = [(1 + i) * steps_per_eval + current_steps, mean_reward, std_reward]
    return model, df


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

    # Initialize and save an empty evaluations DataFrame in the experiment directory
    evaluations_path = os.path.join(directory, 'evaluations.csv')
    df = pd.DataFrame(columns=['training_steps', 'mean_reward', 'std_reward'])
    df.to_csv(evaluations_path, index=False)

    # Save the initial model in the experiment directory
    model_save_path = os.path.join(directory, f"{env_id}-{model_architecture}-0")
    model.save(model_save_path)

def SaveNewData(experiment_name, model, meta_data, steps_trained, df):
    new_steps_trained = meta_data.get("steps_trained") + steps_trained
    meta_data["steps_trained"] = new_steps_trained

    metadata_path = f"{experiment_name}/metadata.json"
    with open(metadata_path, 'w') as file:
        json.dump(meta_data, file)

    model_name = meta_data.get("env_id") + "-" + meta_data.get("model_architecture") + "-" + str(steps_trained)
    model_save_path = f"{experiment_name}/"+model_name
    model.save(model_save_path)
    evaluations_path = f"{experiment_name}/evaluations.csv"
    df.to_csv(evaluations_path, index=False)

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