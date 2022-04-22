import numpy as np
from .actions import Action_T1

# Discrete Position; Single Agent
class Environment():
    def __init__(self, reward_func):
        self.reward_func = reward_func
        
    def init(self, board, startAt, arrivalAt, data_volume):
        self.board = board
        self.startAt = startAt
        self.arrivalAt = arrivalAt
        self.current_position = startAt
        self.data_volume_initial = data_volume
        self.data_volume_remaining = data_volume

    def reset(self):
        self.current_position = self.startAt
        self.data_volume_remaining = self.data_volume_initial
        self.reward = 0
    
    def get_state(self):
        return (self.board, self.current_position, self.data_volume_remaining)

    def step(self, action):
        is_done = True
        next_position = []
        for (c1, c2) in zip(self.current_position, action):
            next_position.append(c1+c2)
        self.current_position = next_position
        
        for (c1, c2) in zip(self.current_position, self.target):
            if c1 != c2:
                is_done = False
        reward = self.reward_func()
        return next_position, reward, is_done

    def reward(self):
        return self.reward

    def action_space(self):
        action = Action_T1()
        return action

    def visualizer(self):
        pass