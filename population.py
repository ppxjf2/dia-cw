from agent import Agent 
from brain import Brain
import gymnasium as gym
import random

class Population:
    
    def __init__(self, env, frames):
        self.frames = frames
        self.agents = []
        self.generation = 1
        self.env = env
        self.generation = 0
        self.fitsum = 0

            
    def createNewGeneration(self, size):
        for i in range(size):
            self.agents.append(Agent(self.frames))
    
    def createNextGeneration(self, size):
        agents = []
        for i in range(size):
            agents.append(Agent(self.frames))
            
        return agents
            
    def runGeneration(self):
        self.generation += 1
        for i in range(len(self.agents)):
            fitness=0
            self.env.reset()
            for j in range(self.frames):

                action = self.agents[i].brain.actions[j]

                observation, reward, terminated, truncated, info = self.env.step(action)

                fitness += reward

                if terminated or truncated:
                    observation, info = self.env.reset()
                    break
                #self.env.close()
                
            print("Generation " + str(self.generation) + " Agent " + str(i) + " Fitness " + str(fitness))
            self.agents[i].fitnessValue(fitness)
            
    def fitnessSum(self):
        sum = 0
        for i in range(len(self.agents)):
            sum += self.agents[i].fitness
        return sum

    def newPrent(self, fitnessSum):
        rng = random.randint(0, fitnessSum)

        currentVal = 0

        for i in range(len(self.agents)):
            currentVal += self.agents[i].fitness
            if (currentVal > rng):
                return self.agents[i]
            
        return self.agents[0]
    
    def bestAgent(self) -> int:
        best = 0
        bestAgentIndex = 0
        
        for i in range(len(self.agents)):
            if (self.agents[i].fitness > best):
                best = self.agents[i].fitness
                bestAgentIndex = i 
                
        return bestAgentIndex
    
    def naturalSelection(self,size):
        bestAgentIndex = 0
       
        children = self.createNextGeneration(size)

        fitnessSum = self.fitnessSum()
        
        bestAgentIndex = self.bestAgent()
        print("Parent " + str(bestAgentIndex) + " best with a fitness of " + str(self.agents[bestAgentIndex].fitness))

        children[0] = self.agents[bestAgentIndex].createChild()
        children[1] = self.agents[bestAgentIndex].createChild()
        
        for i in range(2, len(self.agents)):
            parent = self.newPrent(fitnessSum)
            children[i] = parent.createChild()
    
        self.agents = children
    
    def mutateChildren(self):
        for i in range(1, len(self.agents)):
            self.agents[i].brain.mutate()
    

    