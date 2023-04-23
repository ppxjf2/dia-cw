import gymnasium as gym
import numpy as np

from agent import Agent
from brain import Brain
from population import Population
pop = 3
env = gym.make("ALE/MsPacman-ram-v5", render_mode="human")
obs_space = env.observation_space
print("The observation space: {}".format(obs_space))

env.reset()

# test = Population(env)
# test.createPop(pop)
# test.runGeneration()

# for i in range(pop):
#     print(test.agents[i].fitnessPrint())
    

for i in range(10000):
    #print(i)
    action = env.action_space.sample()  # this is where you would insert your policy

    observation, reward, terminated, truncated, info = env.step(action)
    #print(observation)
    #print(reward)

    if terminated or truncated:
        observation, info = env.reset()
        print(observation)
        break
env.close()
