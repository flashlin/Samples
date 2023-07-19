import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnvWrapper
import random
from self_drive_car_env import SelfDriveCarEnv
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor


GameEnvName = "SelfDriveCarEnv-v0"


gym.register(
    id=GameEnvName,
    entry_point="self_drive_car_env:SelfDriveCarEnv",
)

num_envs = 1  # 指定環境數量

if __name__ == '__main__':
    # 建立多環境
    env = gym.make(GameEnvName)
    env = Monitor(env, "./video", override_existing=True)
    # env = DummyVecEnv([lambda: env])

    # 建立 PPO 模型
    model = PPO("MlpPolicy", env, verbose=1)
    # 開始訓練模型
    model.learn(total_timesteps=10)
    # 儲存訓練好的模型
    model.save("./models/custom_game_model")
