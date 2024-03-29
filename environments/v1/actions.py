from tkinter import N
import numpy as np
import math

class Discrete():
    def __init__(self, time_scale=2):
        self.time_scale = time_scale
        actions = np.array([[0, 1.0], [0, -1.0], [1.0, 0], [-1.0, 0], [0, 0]]) / self.time_scale
        self.actions = actions.tolist()
        self.n = len(self.actions)

    def get_action(self, n):
        return self.actions[n]

    def get_actions(self):
        return self.actions

    def sample(self):
        return np.random.randint(0, len(self.actions)-1)

class LinearDiscrete():
    def __init__(self) -> None:
        self.actions = [[0, 0.1], [0, 0]]
        self.n = len(self.actions)
    
    def get_action(self, n):
        return self.actions[n]

    def sample(self):
        return np.random.randint(0, len(self.actions)-1)


class Continuous():
    def __init__(self) -> None:
        self.shape = 2
        self.high = 1
        self.max_speed = 1
        self.max_angle = 360

    def get_action(self, action):
        r = action[0] * self.max_speed
        theta = action[1] * self.max_angle
        
        x = r * math.cos(math.radians(theta))
        y = r * math.sin(math.radians(theta))

        return [x, y]

    def sample(self):
        return np.random.rand(2)