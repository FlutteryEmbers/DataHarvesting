import numpy as np
from .actions import Action_T1
from config.reward_function import RewardFunction_1
import config

# Discrete Position; Single Agent
class Environment():
    def __init__(self):
        self.reward_func = RewardFunction_1
        
    def init(self, board, startAt, arrivalAt, data_volume):
        self.board = board
        self.startAt = startAt
        self.arrivalAt = arrivalAt
        self.current_position = startAt
        self.data_volume_required = data_volume
        self.data_volume_collected = [0]*len(data_volume)
        self.num_steps = 0
        self.tower_location = self._get_tower_location()

    def reset(self):
        self.current_position = self.startAt
        self.data_volume_collected = [0]*len(self.data_volume_required)
        self.reward = 0
        self.num_steps = 0
    
    def get_state(self):
        return (self.board, self.current_position, self.data_volume_collected)

    def _get_tower_location(self):
        tower_location = []
        for i in range(10):
            for j in range(10):
                if self.board[i][j] == 1:
                    tower_location.append([i, j])
        return tower_location

    def step(self, action):
        is_done = True
        self.num_steps += 1
        next_position = []
        for (c1, c2) in zip(self.current_position, action):
            next_position.append(c1+c2)
        self.current_position = next_position
        
        # 判断是否到达终点
        for (c1, c2) in zip(self.current_position, self.arrivalAt):
            if c1 != c2:
                is_done = False
        reward = self.reward_func(self.num_steps)
        
        self.data_volume_remaining = config.Phi_dif_transmitting_speed(self.current_position, self.tower_location, )
        return next_position, reward, is_done, self.num_steps

    def reward(self):
        return self.reward

    def action_space(self):
        action = Action_T1()
        return action

    def visualizer(self):
        pass