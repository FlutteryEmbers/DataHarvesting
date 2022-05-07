import numpy as np
import random
from collections import namedtuple
from config.transmission_model import Phi_dif_transmitting_speed
import utils.tools as tools

dqn_state = namedtuple('state', field_names=['board', 'current_position', 'data_volumn_collected'])
# Discrete Position; Single Agent
class DQN_Environment():
    def __init__(self, board):
        # self.reward_func = self.test_reward_function
        self.board = board
        self.action_space = _action_class(board)
        self.tower_location = self._get_tower_location()
        self.x_limit = len(board)
        self.y_limit = len(board[0])
        
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
        return self.get_state(), self.current_position
    
    def get_state(self):
        geo_map = self.board
        location_map = self.board
        transmission_map = self.board

        [x, y] = self.current_position
        # print(x, y)
        location_map[x][y] = 1
        
        for x, y, tower_no in self.tower_location:
            transmission_map[x][y] = self.data_volume_collected[tower_no-1]

        return [geo_map, location_map, transmission_map]

    def _get_tower_location(self):
        tower_location = []
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] > 0:
                    tower_location.append([i, j, self.board[i][j]])
        return tower_location

    def step(self, action_index):
        action = self.action_space.get_actions(action_index)
        is_done = False
        self.num_steps += 1
        next_position = tools.ListAddition(self.current_position, action)
        # self.current_position[0] = max(0, min(len(self.board), self.current_position[0]))
        # self.current_position[1] = max(0, min(len(self.board[0]), self.current_position[1]))

        # NOTE: 是否出界; 如果出界
        if next_position[0] >= 0 and next_position[0] < self.x_limit and next_position[1] >= 0 and next_position[1] < self.y_limit:
            self.current_position = next_position

        # 判断是否到达终点
        if self.current_position == self.arrivalAt:
            is_done = True
 
        reward = self.test_reward_function()
        if is_done:
            reward += 10000
        # self.data_volume_remaining = config.Phi_dif_transmitting_speed(self.current_position, self.tower_location, )
        return self.get_state(), reward, is_done, self.current_position

    def reward(self):
        return self.reward

    def test_reward_function(self):
        return -self.num_steps

    def get_action_space(self):
        return self.action_space

    def render(self):
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
    
    def sample_valid_action(self, position):
        actions_index = self.get_available_actions(position)
        return random.choice(actions_index)

    def get_indexed_action(self, n):
        return self.actions[n]

    def get_available_actions(self, position):
        # print("position", end=':')
        # print(position)
        valid_actions_index = []
        for i in range(len(self.actions)):
            action = self.actions[i]
            next_position = tools.ListAddition(action, position)
            # print(next_position, end=',')
            # print(action)
            if next_position[0] >= 0 and next_position[0] < self.x_limit and next_position[1] >= 0 and next_position[1] < self.y_limit:
                valid_actions_index.append(i)
        # print(valid_actions)
        return valid_actions_index

    def get_actions(self):
        return self.actions