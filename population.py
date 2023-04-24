from agent import Agent 
from brain import Brain
import gymnasium as gym

class Population:
    
    def __init__(self, env):
        self.agents = []
        self.generation = 1
        self.env = env
        self.generation = 0
        self.fitsum = 0

            
    def createPop(self, size):
        for i in range(size):
            self.agents.append(Agent())
                
            
    def runGeneration(self):
        self.generation += 1
        for i in range(len(self.agents)):
            fitness=0
            self.env.reset()
            print(str(self.generation) + " - " + str(i))
            for j in range(1000):

                action = self.agents[i].brain.actions[j]

                observation, reward, terminated, truncated, info = self.env.step(action)

                fitness += reward

                if terminated or truncated:
                    observation, info = self.env.reset()
                    break
                #self.env.close()
            self.agents[i].fitnessValue(fitness)
            

    
    def selectParent(self):
        
        return 
    
    def naturalSelection(self):
        parent = self.selectParent()
        
        pass
    
    def mutateChildren(self):
        
        pass
    
    def setBestAgent(self):
        
        pass
    
    