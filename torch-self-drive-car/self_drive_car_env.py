import math
import time  # For debugging.
import numpy as np

from car import Action
from constants import RadarLineLength
from game import SelfDriveCarGame
import gymnasium as gym


class SelfDriveCarEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.window = None
        self.render_mode = 'rgb_array'
        self.metadata = {
            'render_fps': 30,
            'render_modes': ['human', 'rgb_array']
        }
        self.game = SelfDriveCarGame()
        self.game.reset()
        self.action_space = gym.spaces.Discrete(4)  # NONE, UP, LEFT, RIGHT

        space = self.game.get_observation_space()
        space_width = len(space)
        self.observation_space = gym.spaces.Box(
            low=0, high=300,
            shape=(space_width,),
            dtype=np.float32
        )
        self.done = False
        limit_step = False
        if limit_step:
            self.step_limit = 5000
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

    def step(self, action: int):
        self.done = self.game.step(Action(action))
        obs = self._generate_observation()
        info = self.game.get_observation_info()

        if self.done:
            return obs, -100, self.done, self.truncated, info.to_dict()

        reward = 0.0
        self.reward_step_counter += 1

        if self.reward_step_counter > self.step_limit:
            self.reward_step_counter = 0
            self.truncated = True

        for radar_line in info.radar_lines:
            reward += radar_line

        reward += info.speed * 10
        # print(f"{self.done=} {reward=}")
        return obs, reward * 0.1, self.done, self.truncated, info.to_dict()

    def render(self, mode='human'):
        if mode == 'human':
            return self._render_frame()
        self.game.render()

    def _render_frame(self):
        if self.window is None:
            self.game.set_silent_mode(False)
            self.window = True
        self.game.render()
        return self.game.get_frame_image()

    def _generate_observation(self):
        space = self.game.get_observation_space()
        obs = np.array(space, dtype=np.float32)
        return obs
