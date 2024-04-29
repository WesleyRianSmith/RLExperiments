import experiment
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

experiment_name = "PacManDQN_Default"
env_id = "ALE/Pacman-v5"
n_envs = 4
n_stack = 4
eval_frequency = 10000/ n_envs
def CreateFolder():
    env = make_atari_env(env_id, n_envs=n_envs, seed=0)
    env = VecFrameStack(env, n_stack=n_stack)
    hyperparameters = {
        'policy': 'CnnPolicy',
        'verbose': 1,
        'buffer_size': 100_000
    }
    model = DQN(env=env,**hyperparameters)
    experiment.InitialiseExperiment(experiment_name=experiment_name, model=model, model_architecture="DQN",
                                    env_id=env_id
                                    , n_envs=n_envs, n_stack=n_stack, eval_frequency=eval_frequency,
                                    hyper_parameters=hyperparameters)

def Train(steps,iterations):
    for i in range(iterations):
        experiment.TrainExperiment(experiment_name, steps)

