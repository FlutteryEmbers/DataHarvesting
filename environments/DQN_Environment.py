from json import tool
from matplotlib.style import available
import numpy as np
import random
from config.reward_function import RewardFunction_1
import utils.tools as tools

# Discrete Position; Single Agent
class DQN_Environment():
    def __init__(self, board):
        self.reward_func = RewardFunction_1
        self.board = board
        self.action_space = _action_class(board)
        self.tower_location = self._get_tower_location()
        
    def init(self, startAt, arrivalAt, data_volume):
        self.startAt = startAt
        self.arrivalAt = arrivalAt
        self.current_position = startAt
        self.data_volume_required = data_volume
        self.data_volume_collected = [0]*len(data_volume)
        self.num_steps = 0
        
    def reset(self):
        self.current_position = self.startAt
        self.data_volume_collected = [0]*len(self.data_volume_required)
        self.reward = 0
        self.num_steps = 0
    
    def get_state(self):
        return (self.board, self.current_position, self.data_volume_collected)

    def _get_tower_location(self):
        tower_location = []
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 1:
                    tower_location.append([i, j])
        return tower_location

    def step(self, action):
        is_done = False
        self.num_steps += 1
        self.current_position = tools.ListAddition(self.current_position, action)
        # 判断是否到达终点
        if self.current_position == self.arrivalAt:
            is_done = True

        reward = self.reward_func(self.num_steps)
        
        # self.data_volume_remaining = config.Phi_dif_transmitting_speed(self.current_position, self.tower_location, )
        return self.current_position, reward, is_done, self.num_steps

    def reward(self):
        return self.reward

    def get_action_space(self):
        return self.action_space

    def visualizer(self):
        pass

class _action_class():
    def __init__(self, board):
        self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]
        self.board = board
        self.x_limit = len(board)
        self.y_limit = len(board[0])
        print(self.x_limit)
        print(self.y_limit)
    
    def n(self):
        return len(self.actions)
    
    def sample(self, position):
        actions = self.get_available_actions(position)
        return random.choice(actions)
    
    def get_available_actions(self, position):
        # print("position", end=':')
        # print(position)
        valid_actions = []
        for i in range(len(self.actions)):
            action = self.actions[i]
            next_position = tools.ListAddition(action, position)
            # print(next_position, end=',')
            # print(action)
            if next_position[0] >= 0 and next_position[0] < self.x_limit and next_position[1] >= 0 and next_position[1] < self.y_limit:
                valid_actions.append(action)
        # print(valid_actions)
        return valid_actions

    def get_actions(self):
        return self.actions