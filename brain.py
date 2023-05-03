import math
import numpy as np
import random


class Brain:
    actions = []
    step = 0

    def __init__(self, size):
        self.actions = []   
        self.size = size     
        for i in range(size):
            self.actions.append(random.randint(0, 4))

    # returns a copy of the brain, this does not hurt the brain
    def clone(self):
        clone = Brain(self.size)
        
        for i in range(len(self.actions)):
            clone.actions[i] = self.actions[i]
        
        return clone
    
    # mutates the brain by setting some of the actions to random vectors
    def mutate(self):
        mutationRate = 0.01
        
        for i in range(len(self.actions)):
            rng = random.random()
            
            if (rng < mutationRate):
                # random direction  
                self.actions[i] = random.randint(0, 4)
                
    def loadBrain(self, actions):
        self.actions = actions