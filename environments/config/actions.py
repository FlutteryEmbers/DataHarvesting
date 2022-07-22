import numpy as np

class Discrete():
    def __init__(self):
        self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]
        self.n = len(self.actions)

    def get_indexed_action(self, n):
        return self.actions[n]

    def get_actions(self):
        return self.actions

    def sample(self):
        return np.random.randint(0, len(self.actions)-1)

class Continuous():
    def __init__(self) -> None:
        pass