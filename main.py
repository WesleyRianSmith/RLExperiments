import experiment
import make_video
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

#import optuna
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


def InitialiseExperiment2():
    env_id = "Breakout-v4"
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    hyperparameters = {
        'policy': 'CnnPolicy',
        'verbose': 1,
        'buffer_size': 25000
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

InitialiseExperiment2()
for i in range(10):
    experiment.TrainExperiment("BreakoutDQN",1000)
InitialiseExperiment3()
for i in range(10):
    experiment.TrainExperiment("BreakoutA2C",1000)
#model = DQN.load("BreakoutPPO/Breakout-v4-DQN-2602000")
make_video.create_video("BreakoutPPO","latest",500,True)
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