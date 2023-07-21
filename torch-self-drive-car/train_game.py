import os.path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnvWrapper
import random
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from torch import nn

SHOW = True

GameEnvName = "SelfDriveCarEnv-v0"
ModelName = "./models/custom_game_model"

gym.register(
    id=GameEnvName,
    entry_point="self_drive_car_env:SelfDriveCarEnv",
)

num_envs = 1  # 指定環境數量

if __name__ == '__main__':
    env = gym.make(GameEnvName)
    if SHOW:
        env = gym.wrappers.HumanRendering(env)
    env = Monitor(env, "./video", override_existing=True)
    vec_env = DummyVecEnv([lambda: env])

    # 建立 PPO 模型
    policy_kwargs = dict(net_arch=[7, 4], activation_fn=nn.ReLU)
    model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)
    if os.path.exists(ModelName):
        model.load(ModelName)
    # 開始訓練模型
    model.learn(total_timesteps=20000)
    # 儲存訓練好的模型
    model.save(ModelName)
