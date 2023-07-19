import math
import time  # For debugging.
import numpy as np
from game import SelfDriveCarGame
import gymnasium as gym


class SelfDriveCarEnv(gym.Env):
    def __init__(self, silent_mode=True):
        super().__init__()
        self.game = SelfDriveCarGame(silent_mode=silent_mode)
        self.game.reset()
        self.action_space = gym.spaces.Discrete(5)

        space = self.game.get_observation_space()
        space_width = len(space)
        self.observation_space = gym.spaces.Box(
            low=-2000, high=2000,
            shape=(space_width,),
            dtype=np.float32
        )
        self.done = False
        limit_step = False
        if limit_step:
            self.step_limit = 1000
        else:
            self.step_limit = 1e9  # Basically no limit.
        self.reward_step_counter = 0
        self.truncated = False

    def reset(self, seed=None, options=None):
        self.game.reset()
        self.done = False
        self.reward_step_counter = 0
        obs = self._generate_observation()
        info = self.game.get_observation_info()
        return obs, info.to_dict()

    def seed(self, seed=None):
        pass

    def step(self, action):
        self.done = self.game.step(action)
        obs = self._generate_observation()
        info = self.game.get_observation_info()

        reward = 0.0
        self.reward_step_counter += 1

        if self.reward_step_counter > self.step_limit:  # Step limit reached, game over.
            self.reward_step_counter = 0
            self.truncated = True

        for radar_line in info.radar_lines:
            reward += radar_line

        return obs, reward * 0.1, self.done, self.truncated, info.to_dict()

    def render(self):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])

    def _check_action_validity(self, action) -> bool:
        done = self.game.step(action)
        self.game.rollback_state()
        return done is False

    def _generate_observation(self):
        space = self.game.get_observation_space()
        obs = np.array(space, dtype=np.float32)
        # space_width = len(space)
        # obs = np.zeros((space_width, 1), dtype=np.float32)
        # for idx, num in enumerate(space):
        #     obs[idx] = num
        # # obs[tuple(np.transpose(self.game.snake))] = np.linspace(0.8, 0.2, len(self.game.snake), dtype=np.float32)
        # # obs[tuple(self.game.snake[0])] = 1.0
        return obs
