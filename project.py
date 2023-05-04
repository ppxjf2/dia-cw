import gymnasium as gym
import numpy as np

from agent import Agent
from brain import Brain
from population import Population
# from montecarlo import MonteCarlo

pop = 50
gen = 1000

# env = gym.make("MsPacmanDeterministic-v4", render_mode="rgb_array")

# obs_space = env.observation_space.shape[0]
# print("The observation space: {}".format(obs_space))

# env.reset()

# test = Population(env, 20000)
# test.createNewGeneration(pop)
# test.runGeneration()
    
# best_gen = []

# file = open('best.txt', 'w') 
# file.write('Best Parents for next Generation') 
# file.close()

# for i in range(gen):
#     agents = []

    
#     for j in range(pop):
#         agents.append(test.agents[j].brain.actions)
#         #print(test.agents[j].brain.actions)
    
#     np.savetxt("generations/generation_" + str(i+1) + ".txt", agents, fmt="%d")
    
#     test.naturalSelection(pop)
#     test.mutateChildren()
#     test.runGeneration()

#     bestAgentIndex = test.bestAgent()
#     best_gen = "\nGeneration " + str(i+1) + " Parent " + str(bestAgentIndex) + " best with a fitness of " + str(test.agents[bestAgentIndex].fitness)

#     file = open('best.txt', 'a') # Open a file in append mode
#     file.write(best_gen) # Write some text
#     file.close() # Close the file





# for j in range(pop):
#     print(test.agents[j].fitness)

env = gym.make("MsPacmanDeterministic-v4", render_mode="human", obs_type="ram")
env.reset()


action = np.loadtxt("best_brain.txt", dtype="uint8", delimiter=' ')

rewardSum = 0
lastObservation = []

for i in range(100):
    #print(i)
    # action = env.action_space.sample()  # this is where you would insert your policy

    observation, reward, terminated, truncated, info = env.step(3)
    np.set_printoptions(threshold=np.inf)
    observation = observation.reshape(observation.shape[0], -1)
    
    if(np.array_equal(observation, lastObservation)):
        print(observation)
        lastObservation = observation
        
    np.savetxt("observations/observation" + str(i) + ".txt", observation, fmt="%d")
    rewardSum += reward

    if terminated or truncated:
        observation, info = env.reset()
        print(observation)
        break
    
print(rewardSum)
env.close()
