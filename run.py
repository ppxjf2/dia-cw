import gymnasium as gym
import numpy as np
import pandas as pd 
import csv
import math

from monte_carlo import MonteCarlo

env = gym.make("MsPacman-ramDeterministic-v4", render_mode="human", obs_type="ram")
env.reset()

action = np.loadtxt("best_brain.txt", dtype="uint8", delimiter=' ')

rewardSum = 0
lastObservation = []

test = MonteCarlo(env)
map = test.mapGen()

print(test.mapGen())

# movement starts at 66 frames
for i in range(20000):
    observation, reward, terminated, truncated, info = env.step(1)

    pacman_x = math.floor((observation[10]+2)/8)-1
    pacman_y = math.floor((observation[16]+10)/12)
    
    # print(math.floor((pacman_x)/8))
    # print(math.floor((pacman_y+10)/12)+1)

    # print(map[pacman_y][pacman_x])z
    
    
    # rewardSum += reward

    if terminated or truncated:
        observation, info = env.reset()
        # print(observation)
        break   

env.close()
