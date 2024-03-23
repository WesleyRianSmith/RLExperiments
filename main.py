import experiment
import make_video
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import json
def InitialiseExperiment1():
    env_id = "Breakout-v4"
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    hyperparameters = {
        'policy': 'CnnPolicy',
        'verbose': 1,
        'batch_size': 512,
        'ent_coef': 0.01,
        'learning_rate': 0.00025,
        'n_epochs': 4,
        'n_steps': 128,
        'vf_coef': 0.5,
        'clip_range': 0.1
    }
    model = PPO(env=env,**hyperparameters)
    experiment.InitialiseExperiment(experiment_name="BreakoutPPO",model=model,model_architecture="PPO",env_id=env_id,hyper_parameters=hyperparameters)
def TrainExperiment(experiment_name,steps,evaluation_count):
    metadata_path = f"{experiment_name}/metadata.json"
    meta_data = {}
    with open(metadata_path, 'r') as file:
        meta_data = json.load(file)
    evaluations_path = f"{experiment_name}/evaluations.csv"
    df = pd.read_csv(evaluations_path)
    env_id = meta_data.get("env_id")
    model_architecture = meta_data.get("model_architecture")
    model_name = env_id + "-" + model_architecture + "-" + str(meta_data.get("steps_trained"))
    model_path = f"{experiment_name}/{model_name}"
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
    model, df = experiment.TrainAndEvaluate(env_id,model,df,steps,evaluation_count,meta_data.get("steps_trained"))
    print(df)
    total_steps_trained = meta_data.get("steps_trained") + steps
    experiment.SaveNewData(experiment_name,model,meta_data,total_steps_trained,df)

def InitialiseExperiment2():
    env_id = "Breakout-v4"
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    hyperparameters = {
        'policy': 'CnnPolicy',
        'verbose': 1,
        'buffer_size': 100000
    }
    model = DQN(env=env, **hyperparameters)
    experiment.InitialiseExperiment(experiment_name="BreakoutDQN",model=model,model_architecture="DQN",
                                    env_id=env_id,hyper_parameters=hyperparameters)
def InitialiseExperiment3():
    env_id = "Breakout-v4"
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    hyperparameters = {
        'policy': 'CnnPolicy',
        'verbose': 1,
    }
    model = A2C(env=env, **hyperparameters)
    experiment.InitialiseExperiment(experiment_name="BreakoutA2C",model=model,model_architecture="A2C",
                                    env_id=env_id,hyper_parameters=hyperparameters)
def InitialiseExperiment4():
    env_id = "ALE/MsPacman-v5"
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    hyperparameters = {
        'policy': 'CnnPolicy',
        'verbose': 1,
        'batch_size': 512,
        'ent_coef': 0.01,
        'learning_rate': 0.00025,
        'n_epochs': 4,
        'n_steps': 128,
        'vf_coef': 0.5,
        'clip_range': 0.1
    }
    model = PPO(env=env, **hyperparameters)
    experiment.InitialiseExperiment(experiment_name="MsPacmanPPO",model=model,model_architecture="PPO",env_id=env_id,hyper_parameters=hyperparameters)
InitialiseExperiment4()
#TrainExperiment("BreakoutPPO",1000000,5)
#model = DQN.load("BreakoutDQN/Breakout-v4-DQN-2602000")
make_video.create_video("BreakoutDQN","latest",500)
hyper_parameter_ranges = {
    "learning_rate": [0.00001, 0.001,"float",True],
    "gamma": [0.9,0.9999,"float",True],
    "exploration_final_eps": [0.01, 0.02,"float", False],
    "exploration_initial_eps": [0.9, 1.1,"float", False]
}
hyperparameters = {
    "policy": "CnnPolicy",
    'verbose': 1,
    'buffer_size': 100000
    }
#experiment.TuneHyperparameters("DQN","Breakout-v4",50000,hyperparameters,hyper_parameter_ranges)