import imageio
from IPython.display import Video
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import json
import os
from stable_baselines3.common.evaluation import evaluate_policy
import time
def create_video(experiment_name,steps_trained,steps_to_play,render_recording):
    metadata_path = f"ExperimentModels/{experiment_name}/metadata.json"
    meta_data = {}
    with open(metadata_path, 'r') as file:
        meta_data = json.load(file)
    env_id = meta_data.get("env_id")
    model_architecture = meta_data.get("model_architecture")
    if steps_trained == "latest":
        steps_trained = meta_data.get("steps_trained")
    coded_env_id = meta_data.get("env_id").replace("/", "-")
    model_path = f"ExperimentModels/{experiment_name}/{coded_env_id}-{model_architecture}-{str(steps_trained)}/model"
    print(model_path)
    if not os.path.isfile(model_path + ".zip"):
        print("No valid model file path")
        return False
    if model_architecture == "PPO":
        model = PPO.load(model_path)
    elif model_architecture == "DQN":
        model = DQN.load(model_path)
    elif model_architecture == "A2C":
        model = A2C.load(model_path)
    else:
        print("Invalid model architecture")
        return False
    eval_env = VecFrameStack(make_atari_env(env_id, n_envs=meta_data.get("n_envs"), seed=0), n_stack=meta_data.get("n_stack"))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(mean_reward, std_reward)
    obs = eval_env.reset()
    if render_recording:
        eval_env.render(mode='human')
    frames = []
    for i in range(steps_to_play):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if render_recording:
            eval_env.render(mode='human')
        frames.append(eval_env.render(mode='rgb_array'))

    video_path = f"ExperimentModels/{experiment_name}/{coded_env_id}-{model_architecture}-{str(steps_trained)}/video.mp4"
    imageio.mimwrite(video_path, frames, fps=12)  # fps is frames per second
    Video(video_path)
