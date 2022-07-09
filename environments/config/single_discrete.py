import numpy as np
import random
from collections import namedtuple
from .transmission_model import Phi_dif_transmitting_speed
from utils.buffer import Info
import copy
from numpy import linalg as LNG 
import math
random.seed(10)

dqn_state = namedtuple('state', field_names=['board', 'current_position', 'data_volumn_collected'])

# NOTE: Discrete Position; Single Agent
class DQN_Environment():
    def __init__(self, board):
        # self.reward_func = self.test_reward_function
        self.board = board
        self.action_space = action_class(board)
        self.tower_location = self._get_tower_location()
        self.x_limit = len(board)
        self.y_limit = len(board[0])
        self.reward = 0
        self.action_sequence = []

        self.running_info = Info(board_structure=self.board, num_turrent=len(self.tower_location))
        
    def init(self, startAt, arrivalAt, data_volume):
        self.startAt = startAt
        self.arrivalAt = arrivalAt
        self.current_position = startAt
        self.data_volume_required = data_volume
        self.data_volume_collected = [0]*len(data_volume)
        self.data_transmitting_rate_list = [0]*len(data_volume)
        self.num_steps = 0
        
    def reset(self):
        self.current_position = self.startAt
        self.data_volume_collected = [0]*len(self.data_volume_required)
        self.reward = 0
        self.num_steps = 0
        self.action_sequence = []

        self.running_info.reset()
        return self.get_state_linear(), self.current_position
    
    def get_state_map(self):
        geo_map = copy.deepcopy(self.board[:][:]) 
        location_map = copy.deepcopy(self.board[:][:])
        transmission_map = copy.deepcopy(self.board[:][:])

        [x, y] = self.current_position
        # print(x, y)
        location_map[x][y] = 1
        
        for x, y, tower_no in self.tower_location:
            transmission_map[x][y] = self.data_volume_collected[tower_no-1]

        return [geo_map, location_map, transmission_map]

    def get_linear_state_length(self):
        state = []
        state += self.current_position
        state += self.data_volume_collected
        # state += self.tower_location
        for x, y, _ in self.tower_location:
            state.append(x)
            state.append(y)
        return len(state)

    def get_state_linear(self):
        state = []
        state += self.current_position
        state += self.data_volume_collected
        for x, y, _ in self.tower_location:
            state.append(x)
            state.append(y)
        return state

    def _get_tower_location(self):
        tower_location = []
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] > 0:
                    tower_location.append([i, j, self.board[i][j]])
        tower_location.sort(key = lambda x:x[2])
        # print(tower_location)
        # for i in range(tower_location):  
        return tower_location

    def step(self, action_index):
        self.action_sequence.append(action_index)
        action = self.action_space.get_indexed_action(action_index)
        is_done = False
        self.num_steps += 1
        next_position = np.array(self.current_position) + np.array(action)
        # self.current_position[0] = max(0, min(len(self.board), self.current_position[0]))
        # self.current_position[1] = max(0, min(len(self.board[0]), self.current_position[1]))

        # NOTE: 是否出界; 如果未出界更新位置
        if next_position[0] >= 0 and next_position[0] < self.x_limit and next_position[1] >= 0 and next_position[1] < self.y_limit:
            self.current_position = next_position.tolist()

        data_volume_collected, data_transmitting_rate_list = Phi_dif_transmitting_speed(self.current_position, self.tower_location, self.data_volume_collected, self.data_volume_required)
        self.data_volume_collected = data_volume_collected.tolist()
        self.data_transmitting_rate_list = data_transmitting_rate_list.tolist()

        data_volume_left = np.array(self.data_volume_required) - np.array(self.data_volume_collected)

        self.running_info.store(position_t=self.current_position, action_t=action_index,
                                    data_collected_t=self.data_volume_collected, 
                                    data_left_t=data_volume_left.tolist(), data_collect_rate_t = data_transmitting_rate_list.tolist())

        # NOTE: 判断是否到达终点
        # if self.data_volume_collected == self.data_volume_required:
        #     is_done = True
        if not data_volume_left.any():
            is_done = True

        reward = self.test_reward_function()
        reward -= 1 # 每步减少reward 1


        if is_done:
            # reward += 100
            # reward -= 0.5 * np.max(data_volume_left)
            reward -= 5 * LNG.norm(np.array(self.current_position) - np.array(self.arrivalAt))

        '''
        if self.num_steps > 5000:
            reward -= 100
            # reward += 10 * math.log(np.sum(np.array(self.data_volume_collected)))
            is_done = True
        '''
        return self.get_state_linear(), reward, is_done, self.current_position

    def test_reward_function(self):
        transmission_reward = 0.5*sum(self.data_transmitting_rate_list)/len(self.data_transmitting_rate_list)
        # print(transmission_reward)
        return transmission_reward

    def get_action_space(self):
        return self.action_space

    def render(self):
        pass

    def view(self):
        print('data left = ', np.array(self.data_volume_required) - np.array(self.data_volume_collected), 'steps taken = ', self.num_steps)
        return self.running_info

class action_class():
    def __init__(self, board):
        self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]
        self.board = board
        self.x_limit = len(board)
        self.y_limit = len(board[0])
        # print(self.x_limit)
        # print(self.y_limit)
    
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
            next_position = np.array(action) + np.array(position)
            # print(next_position, end=',')
            # print(action)
            if next_position[0] >= 0 and next_position[0] < self.x_limit and next_position[1] >= 0 and next_position[1] < self.y_limit:
                valid_actions_index.append(i)
        # print(valid_actions)
        return valid_actions_index

    def get_actions(self):
        return self.actions