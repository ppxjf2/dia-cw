import gymnasium as gym
import numpy as np
import pandas as pd 
import csv
import math

from monte_carlo import MonteCarlo

env = gym.make("MsPacman-ramDeterministic-v4", render_mode="rgb_array", obs_type="ram")
env.reset()

action = np.loadtxt("best_brain.txt", dtype="uint8", delimiter=' ')

rewardSum = 0
lastObservation = []

test = MonteCarlo(env)
test.train(1000)

# # movement starts at 66 frames
# for i in range(1):
    
#     observation, reward, terminated, truncated, info = env.step(1)
#     test.setCoordinates(observation)
        
#     map = test.mapGen()
#     test.ghostMapReset()

#     print(test.pillMap)
#     route = test.bfs("pill", test.graph, (test.pacman_y, test.pacman_x))
#     print(route[-1])
       
    
#     # rewardSum += reward

#     if terminated or truncated:
#         observation, info = env.reset()
#         # print(observation)
#         break   

env.close()
