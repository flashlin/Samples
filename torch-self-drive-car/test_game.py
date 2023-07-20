import time
import random
from stable_baselines3 import PPO
from self_drive_car_env import SelfDriveCarEnv

MODEL_PATH = r"./models/custom_game_model"
NUM_EPISODE = 10

RENDER = True
FRAME_DELAY = 0.05  # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

env = SelfDriveCarEnv()

# Load the trained model
model = PPO.load(MODEL_PATH)

total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0

for episode in range(NUM_EPISODE):
    obs = env.reset()
    episode_reward = 0
    done = False

    num_step = 0
    info = None

    sum_step_reward = 0

    retry_limit = 9
    print(f"=================== Episode {episode + 1} ==================")

    step_counter = 0
    while not done:
        action, _ = model.predict(obs)

        prev_mask = env.get_action_mask()
        prev_direction = env.game.direction

        num_step += 1

        obs, reward, done, info = env.step(action)

        if done:
            last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
            print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")

        sum_step_reward += reward  # Accumulate step rewards.
        episode_reward += reward

        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
# print(f"=================== Summary ==================")
# print(
#     f"Average Score: {total_score / NUM_EPISODE}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {total_reward / NUM_EPISODE}")
