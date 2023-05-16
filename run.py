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

# test.train(1000)
# test.save()

env = gym.make("MsPacman-ramDeterministic-v4", render_mode="rgb_array", obs_type="ram")
example = MonteCarlo(env)

example.load()
example.run(env)
example.runPillCollect(env)
example.runGhostEvade(env)

env.close()
