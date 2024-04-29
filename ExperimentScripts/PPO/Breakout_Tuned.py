import experiment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

experiment_name = "BreakoutPPO_Tuned"
env_id = "BreakoutNoFrameskip-v4"
n_envs = 4
n_stack = 4
eval_frequency = 100000 / n_envs
def CreateFolder():
    env = make_atari_env(env_id, n_envs=n_envs, seed=0)
    env = VecFrameStack(env, n_stack=n_stack)
    hyperparameters = {
        'policy': 'CnnPolicy',
        'verbose': 1,
        "learning_rate": 0.000298, "gamma": 0.99845, "ent_coef": 0.0399, "batch_size": 384, "n_steps": 384

    }
    model = PPO(env=env,**hyperparameters)
    experiment.InitialiseExperiment(experiment_name=experiment_name, model=model, model_architecture="PPO",
                                    env_id=env_id
                                    , n_envs=n_envs, n_stack=n_stack, eval_frequency=eval_frequency,
                                    hyper_parameters=hyperparameters)

def Train(steps,iterations):
    for i in range(iterations):
        experiment.TrainExperiment(experiment_name, steps)


