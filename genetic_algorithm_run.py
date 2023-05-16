import gymnasium as gym
import numpy as np
import pandas as pd 
import csv

from agent import Agent
from brain import Brain
from population import Population
from monte_carlo import MonteCarlo

import matplotlib.pyplot as plt

pop = 50
gen = 1000

def bestAgentRun():
    env = gym.make("MsPacman-ramDeterministic-v4", render_mode="human", obs_type="ram")
    env.reset()

    action = np.loadtxt("best_brain.txt", dtype="uint8", delimiter=' ')

    rewardSum = 0
    lastObservation = []
    graphScore = []

    # movement starts at 66 frames
    for i in range(20000):
        observation, reward, terminated, truncated, info = env.step(action[i])
        np.set_printoptions(threshold=np.inf)

        rewardSum += reward
        
        graphScore.append(rewardSum)

        if terminated or truncated:
            observation, info = env.reset()
            print("Survived: " + str(i))
            # print(observation)
            break   

    # print(rewardSum)
    env.close()

    df = pd.DataFrame(graphScore)
    df.to_csv("GAScore.csv", index=False)

# train new genetic algorithm
def train():
    env = gym.make("MsPacmanDeterministic-v4", render_mode="rgb_array")

    obs_space = env.observation_space.shape[0]
    print("The observation space: {}".format(obs_space))

    env.reset()

    test = Population(env, 20000)
    test.createNewGeneration(pop)
    test.runGeneration()
        
    best_gen = []

    file = open('best.txt', 'w') 
    file.write('Best Parents for next Generation') 
    file.close()

    for i in range(gen):
        agents = []

        
        for j in range(pop):
            agents.append(test.agents[j].brain.actions)
        
        np.savetxt("generations/generation_" + str(i+1) + ".txt", agents, fmt="%d")
        
        test.naturalSelection(pop)
        test.mutateChild()
        test.runGeneration()

        bestAgentIndex = test.bestAgent()
        best_gen = "\nGeneration " + str(i+1) + " Parent " + str(bestAgentIndex) + " best with a fitness of " + str(test.agents[bestAgentIndex].fitness)

        file = open('best.txt', 'a') # Open a file in append mode
        file.write(best_gen) # Write some text
        file.close() # Close the file

# graph the score
def graphs():
    GAScore = pd.read_csv("GAScore.csv")
    MCScore = pd.read_csv("MCScore.csv")
    MCPillScore = pd.read_csv("MCPillScore.csv")
    MCGhostScore = pd.read_csv("MCGhostScore.csv")

    plt.figure()
    plt.plot(GAScore, label="Score at each time step", alpha=1)
    plt.plot(MCScore, label="Score at each tim step", alpha=1)
    plt.plot(MCPillScore, label="Score at each tim step", alpha=1)
    plt.plot(MCGhostScore, label="Score at each tim step", alpha=1)
    plt.xlabel("Time Step")
    plt.ylabel("Score")
    plt.legend(["Genetic Algorithm", "Monte-Carlo All Behaviours", "Monte-Carlo Pill Collect", "Monte-Carlo Ghost Evade"])
    plt.show()


# uncomment to train new GA
# train()

# uncomment to run the best agent 
bestAgentRun()

# Uncomment to run graph
# graphs()