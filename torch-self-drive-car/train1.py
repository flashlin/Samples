import gym
from gym.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnvWrapper
import random
from self_drive_car_env import SelfDriveCarEnv
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor


GameEnvName = "SelfDriveCarEnv-v0"
gym.register(
    id=GameEnvName,
    entry_point=SelfDriveCarEnv,
)


# 建立單一環境創建函式
def make_env(seed):
    def _init():
        env = SelfDriveCarEnv()
        env = ActionMasker(env, SelfDriveCarEnv.get_action_mask)
        env = Monitor(env)
        # env.seed(seed)
        # env.action_space = gym.spaces.Discrete(5)
        return env
    return _init


class EnvWrapper(object):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs

    def render(self):
        self.env.render()


num_envs = 1  # 指定環境數量

if __name__ == '__main__':
    # 建立多環境
    vector_env = gym.vector.make(GameEnvName, num_envs=num_envs)
    # env = Monitor(env, "./save", override_existing=True)
    # seed_set = set()
    # while len(seed_set) < num_envs:
    #     seed_set.add(random.randint(0, 1e9))
    # env = SubprocVecEnv([make_env(seed=s) for s in seed_set])
    # env = DummyVecEnv([lambda: vector_env.envs[0]])

    # 建立 PPO 模型
    model = PPO("MlpPolicy", vector_env, verbose=1)
    # 開始訓練模型
    model.learn(total_timesteps=10000)
    # 儲存訓練好的模型
    model.save("./models/custom_game_model")
