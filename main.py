import experiment
import ExperimentScripts.PPO.Breakout_Default as PPOBreakout
import ExperimentScripts.PPO.Pacman_Default as PPOPacMan
import ExperimentScripts.PPO.Pong_Default as PPOPong
import ExperimentScripts.PPO.SpaceInvaders_Default as PPOSpaceInvaders

import ExperimentScripts.DQN.Breakout_Default as DQNBreakout
import ExperimentScripts.DQN.Pacman_Default as DQNPacMan
import ExperimentScripts.DQN.Pong_Default as DQNPong
import ExperimentScripts.DQN.SpaceInvaders_Default as DQNSpaceInvaders

import ExperimentScripts.A2C.Breakout_Default as A2CBreakout
import ExperimentScripts.A2C.Pacman_Default as A2CPacMan
import ExperimentScripts.A2C.Pong_Default as A2CPong
import ExperimentScripts.A2C.SpaceInvaders_Default as A2CSpaceInvaders

import ExperimentScripts.PPO.Breakout_Tuned as PPOBreakout_Tuned
import ExperimentScripts.PPO.Pacman_Tuned as PPOPacMan_Tuned
import ExperimentScripts.PPO.Pong_Tuned as PPOPong_Tuned
import ExperimentScripts.PPO.SpaceInvaders_Tuned as PPOSpaceInvaders_Tuned

import ExperimentScripts.DQN.Breakout_Tuned as DQNBreakout_Tuned
import ExperimentScripts.DQN.Pacman_Tuned as DQNPacMan_Tuned
import ExperimentScripts.DQN.Pong_Tuned as DQNPong_Tuned
import ExperimentScripts.DQN.SpaceInvaders_Tuned as DQNSpaceInvaders_Tuned

import ExperimentScripts.A2C.Breakout_Tuned as A2CBreakout_Tuned
import ExperimentScripts.A2C.Pacman_Tuned as A2CPacMan_Tuned
import ExperimentScripts.A2C.Pong_Tuned as A2CPong_Tuned
import ExperimentScripts.A2C.SpaceInvaders_Tuned as A2CSpaceInvaders_Tuned
def CreateAll():
    PPOBreakout.CreateFolder()
    PPOPacMan.CreateFolder()
    PPOPong.CreateFolder()
    PPOSpaceInvaders.CreateFolder()

    DQNBreakout.CreateFolder()
    DQNPacMan.CreateFolder()
    DQNPong.CreateFolder()
    DQNSpaceInvaders.CreateFolder()

    A2CBreakout.CreateFolder()
    A2CPacMan.CreateFolder()
    A2CPong.CreateFolder()
    A2CSpaceInvaders.CreateFolder()

    PPOBreakout_Tuned.CreateFolder()
    PPOPacMan_Tuned.CreateFolder()
    PPOPong_Tuned.CreateFolder()
    PPOSpaceInvaders_Tuned.CreateFolder()

    DQNBreakout_Tuned.CreateFolder()
    DQNPacMan_Tuned.CreateFolder()
    DQNPong_Tuned.CreateFolder()
    DQNSpaceInvaders_Tuned.CreateFolder()

    A2CBreakout_Tuned.CreateFolder()
    A2CPacMan_Tuned.CreateFolder()
    A2CPong_Tuned.CreateFolder()
    A2CSpaceInvaders_Tuned.CreateFolder()

def TrainAll(steps):
    PPOBreakout.Train(steps,1)
    PPOPacMan.Train(steps,1)
    PPOPong.Train(steps,1)
    PPOSpaceInvaders.Train(steps,1)

    DQNBreakout.Train(steps,1)
    DQNPacMan.Train(steps,1)
    DQNPong.Train(steps,1)
    DQNSpaceInvaders.Train(steps,1)

    A2CBreakout.Train(steps,1)
    A2CPacMan.Train(steps,1)
    A2CPong.Train(steps,1)
    A2CSpaceInvaders.Train(steps,1)

    PPOBreakout_Tuned.Train(steps,1)
    PPOPacMan_Tuned.Train(steps,1)
    PPOPong_Tuned.Train(steps,1)
    PPOSpaceInvaders_Tuned.Train(steps,1)

    DQNBreakout_Tuned.Train(steps,1)
    DQNPacMan_Tuned.Train(steps,1)
    DQNPong_Tuned.Train(steps,1)
    DQNSpaceInvaders_Tuned.Train(steps,1)

    A2CBreakout_Tuned.Train(steps,1)
    A2CPacMan_Tuned.Train(steps,1)
    A2CPong_Tuned.Train(steps,1)
    A2CSpaceInvaders_Tuned.Train(steps,1)

ppo_hyper_parameter_ranges = {
    "learning_rate": [0.00001, 0.0003, "float", True,6],
    "gamma": [0.9,0.9999,"float", True, 5],
    "ent_coef": [0.001, 0.05, "float", True, 4],
    "batch_size": [128, 512, "int", False, 1, 128],
    "n_steps": [128, 512, "int", False, 1, 128],
}
ppo_hyperparameters = {
    "policy": "CnnPolicy",
    'verbose': 1,
    }

dqn_hyper_parameter_ranges = {
    "learning_rate": [0.00001, 0.0003, "float", True,6],
    "exploration_initial_eps": [0.01, 1, "float", True, 6],
    "exploration_final_eps": [0.001,0.1,"float", True, 5],
    "exploration_fraction": [0.01,0.05,"float", True, 5],
    "train_freq": [2,16,"int", False, 1, 2]
}
dqn_hyperparameters = {
    "policy": "CnnPolicy",
    'verbose': 1,
    'buffer_size': 100_000,
    'gradient_steps': 1
    }

a2c_hyper_parameter_ranges = {
    "learning_rate": [0.00001, 0.0003, "float", True,6],
    "ent_coef": [0.001, 0.05, "float", True, 4],
    "gamma": [0.9,0.9999,"float", True, 5],
    "max_grad_norm": [0.1,1,"float", True, 3],
    "n_steps": [5, 50, "int", False, 1, 5],
}
a2c_hyperparameters = {
    "policy": "CnnPolicy",
    'verbose': 1,
    }

#PPO_Breakout_Default.Train(500_000, 1)
#experiment.TuneHyperparameters("PacmanDQN",50,"A2C",
                              # "SpaceInvadersNoFrameskip-v4",4,4,100000,
                               #a2c_hyperparameters,a2c_hyper_parameter_ranges)
