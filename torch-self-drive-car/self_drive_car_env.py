import math
import time  # For debugging.
import gym
import numpy as np

from game import SelfDriveCarGame


class SelfDriveCarEnv(gym.Env):
    def __init__(self, silent_mode=True):
        super().__init__()
        self.game = SelfDriveCarGame(silent_mode=silent_mode)
        self.game.reset()

        self.action_space = gym.spaces.Discrete(5)  # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN 4: None

        space = self.game.get_observation_space()
        space_width = len(space)
        self.observation_space = gym.spaces.Box(
            low=-2000, high=2000,
            shape=(space_width, 1),
            dtype=np.float32
        )
        self.done = False
        limit_step = False
        if limit_step:
            self.step_limit = 1000
        else:
            self.step_limit = 1e9  # Basically no limit.
        self.reward_step_counter = 0

    def reset(self):
        self.game.reset()
        self.done = False
        self.reward_step_counter = 0
        obs = self._generate_observation()
        return obs

    def step(self, action):
        self.done = self.game.step(action)
        obs = self._generate_observation()
        info = self.game.get_observation_info()

        reward = 0.0
        self.reward_step_counter += 1

        if self.reward_step_counter > self.step_limit:  # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True

        if self.done:
            return obs, reward * 0.1, self.done, info

        for radar_line in info.radar_lines:
            reward += radar_line

        reward = reward * 0.1
        return obs, reward, self.done, info

    def render(self):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])

    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):

        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0:  # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1:  # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2:  # RIGHT
            if current_direction == "LEFT":
                return False
            else:
                col += 1

        elif action == 3:  # DOWN
            if current_direction == "UP":
                return False
            else:
                row += 1

        if (row, col) == self.game.food:
            game_over = (
                    (row, col) in snake_list  # The snake won't pop the last cell if it ate food.
                    or row < 0
                    or row >= self.board_size
                    or col < 0
                    or col >= self.board_size
            )
        else:
            game_over = (
                    (row, col) in snake_list[:-1]  # The snake will pop the last cell if it did not eat food.
                    or row < 0
                    or row >= self.board_size
                    or col < 0
                    or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    def _generate_observation(self):
        space = self.game.get_observation_space()
        space_width = len(space)
        obs = np.zeros((space_width, 1), dtype=np.float32)
        for idx, num in enumerate(space):
            obs[idx] = num
        # obs[tuple(np.transpose(self.game.snake))] = np.linspace(0.8, 0.2, len(self.game.snake), dtype=np.float32)
        # obs[tuple(self.game.snake[0])] = 1.0
        return obs
