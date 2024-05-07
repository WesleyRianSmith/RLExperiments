import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, sync_envs_normalization
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
import json
import torch
import numpy as np
import pandas as pd

import optuna
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available.")

class CustomEvalCallback(EvalCallback):
    """
    Custom callback for evaluation which includes logging of the standard deviation of rewards.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

                # Reset success rate buffer
            self._is_success_buffer = []
            print("Evaluating")
            print(self.n_eval_episodes)
            print(self.render)
            print(self.deterministic)
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,

                callback=self._log_success_callback,
            )
            print("Done Evaluating")
            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # Newly added code
            self.logger.record("eval/std_reward", float(std_reward))
            self.logger.record("eval/std_ep_length", float(std_ep_length))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training



def InitialiseExperiment(experiment_name, model, model_architecture, env_id, n_envs, n_stack, eval_frequency, hyper_parameters):
    directory = f"ExperimentModels"
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Create a directory for the experiment
    directory = f"ExperimentModels/{experiment_name}"
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
        "eval_frequency": eval_frequency,
        "model_architecture": model_architecture,
        "steps_trained": [0],
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
    metadata_path = f"ExperimentModels/{experiment_name}/metadata.json"
    meta_data = {}
    with open(metadata_path, 'r') as file:
        meta_data = json.load(file)
    steps_trained_array = meta_data.get("steps_trained")
    new_steps_trained = steps_trained_array[len(steps_trained_array)-1] + steps
    coded_env_id = meta_data.get("env_id").replace("/","-")
    model_name = coded_env_id + "-" + meta_data.get("model_architecture") + "-" + str(new_steps_trained)
    new_directory = f"ExperimentModels/{experiment_name}/" + model_name
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)

    env_id = meta_data.get("env_id")
    model_architecture = meta_data.get("model_architecture")
    old_model_name = coded_env_id + "-" + model_architecture + "-" + str(steps_trained_array[len(steps_trained_array)-1] )
    model_path = f"ExperimentModels/{experiment_name}/{old_model_name}/model"
    tmp_path = f"{new_directory}/metric_logs"
    new_logger = configure(tmp_path, ["stdout", "csv"])
    print(model_path)
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
    eval_env = make_atari_env(env_id, n_envs=meta_data.get("n_envs"), seed=0)
    eval_env = VecFrameStack(eval_env, n_stack=meta_data.get("n_stack"))


    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=None,
                                 log_path=None, eval_freq=meta_data.get("eval_frequency"),
                                 verbose=1,
                                 n_eval_episodes=10,
                                deterministic=True, render=False)
    model.learn(steps, callback=eval_callback, reset_num_timesteps=False)
    meta_data["steps_trained"].append(new_steps_trained)
    metadata_path = f"ExperimentModels/{experiment_name}/metadata.json"
    with open(metadata_path, 'w') as file:
        json.dump(meta_data, file)
    model_save_path = f"{new_directory}/model"
    model.save(model_save_path)












def TuneHyperparameters(trial_name,n_trials,model_architecture, env_id, n_envs, n_stack,steps ,set_params, hyper_parameter_ranges):
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
        # v[5] is the step value (cannot be used in tandem with v[3])
        for k, v in hyper_parameter_ranges.items():
            step = None
            if len(v) >= 6:
                step = v[5]
            if v[2] == "float":
                param_value = trial.suggest_float(k,v[0],v[1],log=v[3],step=step)
            elif v[2] == "int":
                param_value = trial.suggest_int(k, v[0], v[1], log=v[3] ,step=step)
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

        model.learn(total_timesteps=steps)
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
    study.optimize(objective, n_trials=n_trials)
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