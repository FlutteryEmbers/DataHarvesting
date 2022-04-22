import numpy as np
from .Environment import Action_T1
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
        is_done = False
        # return next_state, reward, is_done, _

    def reward(self):
        return self.reward

    def action_space(self):
        action = Action_T1()
        return action

    def visualizer(self):
        pass