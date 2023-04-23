import math
import numpy as np
import random


class Brain:
    actions = []
    step = 0

    def __init__(self, size):        
        for i in range(size):
            self.actions.append(random.randint(0, 8))

    # returns a copy of the brain, this does not hurt the brain
    def clone(self):
        clone = Brain()
        
        for i in range(len(self.actions)):
            clone.actions.append(self.actions[i])
        
        return clone
    
    # mutates the brain by setting some of the actions to random vectors
    def mutate(self):
        mutationRate = 0.01
        
        for i in range(len(self.actions)):
            rand = random.random()
            
            if (rand < mutationRate):
                # random direction 
                self.actions[i] = random.randint(0, 8)