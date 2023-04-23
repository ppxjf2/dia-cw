from brain import Brain

class Agent:
    
    def __init__(self):
        self.brain = Brain(1000)
        self.fitness = 0 
        self.isBest = False

    def fitnessValue(self, fitness):
        self.fitness = fitness
        
        
    def fitnessPrint(self):
        return self.fitness


    def gimmeBaby(self): 
        baby = Agent()
        baby.brain = self.brain.clone()
        return baby
