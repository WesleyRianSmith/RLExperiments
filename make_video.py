import imageio
from IPython.display import Video
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import json
import os
from stable_baselines3.common.evaluation import evaluate_policy
import time
def create_video(experiment_name,steps_trained,steps_to_play):
    metadata_path = f"{experiment_name}/metadata.json"
    meta_data = {}
    with open(metadata_path, 'r') as file:
        meta_data = json.load(file)
    env_id = meta_data.get("env_id")
    model_architecture = meta_data.get("model_architecture")
    if steps_trained == "latest":
        steps_trained = meta_data.get("steps_trained")
    model_path = f"{experiment_name}/{env_id}-{model_architecture}-{str(steps_trained)}"
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
    eval_env = VecFrameStack(make_atari_env(env_id, n_envs=1, seed=0), n_stack=4)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(mean_reward, std_reward)
    obs = eval_env.reset()
    eval_env.render(mode='human')
    frames = []
    for i in range(steps_to_play):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render(mode='human')
        frames.append(eval_env.render(mode='rgb_array'))




    # Your existing code to generate frames...

    # Create a video from the frames
    video_path = 'myvideo.mp4'
    imageio.mimwrite(video_path, frames, fps=12)  # fps is frames per second

    # To display the video in a Jupyter notebook or Google Colab, use the following:

    Video(video_path)