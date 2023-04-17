import gymnasium as gym
from genetic_functions import test

env = gym.make("ALE/Pacman-ram-v5", render_mode="human")

env.reset()
for _ in range(10000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    print(info)


    if terminated or truncated:
        observation, info = env.reset()
env.close()
