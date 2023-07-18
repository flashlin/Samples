import math
import time  # For debugging.
import gym
import numpy as np

from main import SelfDriveCarGame


class SelfDriveCarEnv(gym.Env):
    def __init__(self, silent_mode=True):
        super().__init__()
        self.game = SelfDriveCarGame(silent_mode=silent_mode)
        self.game.reset()

        self.action_space = gym.spaces.Discrete(5)  # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN 4: None

        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(self.game.board_size, self.game.board_size),
            dtype=np.float32
        )  # 0: empty, 0.5: snake body, 1: snake head, -1: food

        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size
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
        self.done, info = self.game.step(action)
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        if self.reward_step_counter > self.step_limit:  # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True

        if self.done:  # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            # reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)
            # return obs, reward * 0.1, self.done, info

            # Linear penalty decay.
            reward = info["snake_size"] - self.grid_size  # (-max_growth, 0)
            return obs, reward * 0.1, self.done, info

        elif info["food_obtained"]:  # food eaten
            # Reward on num_steps between getting food.
            reward = math.exp((self.grid_size - self.reward_step_counter) / self.grid_size)  # (0, e)
            self.reward_step_counter = 0  # Reset reward step counter

        else:
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(
                    info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info[
                    "snake_size"]  # No upper limit might enable the agent to master shorter scenario faster and more firmly.
            else:
                reward = - 1 / info["snake_size"]
            # print(reward*0.1)
            # time.sleep(1)

        reward = reward * 0.1  # Scale reward
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

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
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

    # EMPTY: 0; SnakeBODY: 0.5; SnakeHEAD: 1; FOOD: -1;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.float32)
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(0.8, 0.2, len(self.game.snake), dtype=np.float32)
        obs[tuple(self.game.snake[0])] = 1.0
        obs[tuple(self.game.food)] = -1.0
        return obs

# Test the environment using random actions
# NUM_EPISODES = 100
# RENDER_DELAY = 0.001
# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     env = SnakeEnv(silent_mode=False)

# # Test Init Efficiency
# print(MODEL_PATH_S)
# print(MODEL_PATH_L)
# num_success = 0
# for i in range(NUM_EPISODES):
#     num_success += env.reset()
# print(f"Success rate: {num_success/NUM_EPISODES}")

# sum_reward = 0

# # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
# action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]

# for _ in range(NUM_EPISODES):
#     obs = env.reset()
#     done = False
#     i = 0
#     while not done:
#         plt.imshow(obs, interpolation='nearest')
#         plt.show()
#         action = env.action_space.sample()
#         # action = action_list[i]
#         i = (i + 1) % len(action_list)
#         obs, reward, done, info = env.step(action)
#         sum_reward += reward
#         if np.absolute(reward) > 0.001:
#             print(reward)
#         env.render()

#         time.sleep(RENDER_DELAY)
#     # print(info["snake_length"])
#     # print(info["food_pos"])
#     # print(obs)
#     print("sum_reward: %f" % sum_reward)
#     print("episode done")
#     # time.sleep(100)

# env.close()
# print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
