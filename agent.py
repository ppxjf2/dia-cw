from brain import Brain

class Agent:
    
    def __init__(self, frames):
        self.frames = frames
        self.brain = Brain(frames)
        self.fitness = 0 

    def fitnessValue(self, fitness):
        self.fitness = fitness
        
    def createChild(self): 
        child = Agent(self.frames)
        child.brain = self.brain.clone()
        return child
    
    def setBrain(self, brain):
        self.brain = brain
