import random

class Action_T1():
    def __init__(self):
        self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]
    
    def n(self):
        return len(self.actions)
    
    def sample(self):
        return random.choice(self.actions)
    
    def get_actions(self):
        return self.actions