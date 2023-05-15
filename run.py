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
# map = test.mapGen()

# visited = []
# test.ghostMapReset()
# print(test.red_y)
# print(test.pacman_y)


# print(test.ghostMap)

# test.bfs(visited, test.graph, (1,1))

# print(test.mapGen())

# movement starts at 66 frames
for i in range(1):
    
    observation, reward, terminated, truncated, info = env.step(1)
    test.setCoordinates(observation)

    #pacman_x = math.floor((observation[10]+2)/8)-1
    #pacman_y = math.floor((observation[16]+10)/12)
        
    map = test.mapGen()
    test.ghostMapReset()

    print(test.ghostMap)
    print(test.bfs(test.graph, (test.pacman_y, test.pacman_x)))


    # print(map[pacman_y][pacman_x])z
    
    
    # rewardSum += reward

    if terminated or truncated:
        observation, info = env.reset()
        # print(observation)
        break   

env.close()
