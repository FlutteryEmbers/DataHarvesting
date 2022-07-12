import numpy as np
import random
from .transmission_model import Phi_dif_transmitting_speed
from utils.buffer import Info
from numpy import linalg as LNG 

class Task():
    def __init__(self, x_limit, y_limit, tower_location) -> None:
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.tower_location = tower_location
        self.num_tower = len(tower_location)

    def set_mission(self, start_at, arrival_at, dv_required) -> None:
        self.dv_required = dv_required
        self.start_at = start_at
        self.arrival_at = arrival_at
        
        self.current_position = self.start_at
        self.dv_left = self.dv_required
        self.dv_collected = [0]*len(self.dv_required)
        self.dv_transmittion_rate = [0]*len(self.dv_required)

    def reset(self):
        self.current_position = self.start_at
        self.dv_left = self.dv_required
        self.dv_collected = [0]*len(self.dv_required)
        self.dv_transmittion_rate = [0]*len(self.dv_required)

    def update_dv_info(self, dv_left, dv_collected, dv_transmittion_rate):
        self.dv_left = dv_left
        self.dv_collected = dv_collected
        self.dv_transmittion_rate = dv_transmittion_rate

    def update_position(self, position):
        position = position.tolist()
        if position[0] >= 0 and position[0] < self.x_limit and position[1] >= 0 and position[1] < self.y_limit:
            self.current_position = position
    
    def get_current_status(self):
        return self.current_position, self.tower_location, self.dv_collected, self.dv_left, self.dv_transmittion_rate, self.dv_required

    def get_state(self):
        current_position = np.array(self.current_position)
        dv_collected = np.array(self.dv_collected)
        tower_location = np.array(self.tower_location)

        state = np.concatenate((current_position, dv_collected, tower_location), axis=None)
        return state.tolist()

    def get_reward(self):
        transmission_reward = 0.5*sum(self.dv_transmittion_rate)/len(self.dv_transmittion_rate)
        return transmission_reward

    def is_done(self):
        if not np.array(self.dv_left).any():
            return True

        return False

# NOTE: Discrete Position; Single Agent
class Agent():
    def __init__(self, env):
        # self.reward_func = self.test_reward_function
        self.status_tracker = env
        self.action_space = Actions()
        self.running_info = Info(board_structure=self.status_tracker, num_turrent=self.status_tracker.num_tower)
        
        self.reward = 0
        self.num_steps = 0

    def reset(self):
        self.reward = 0
        self.num_steps = 0

        self.status_tracker.reset()
        self.running_info.reset()

        return self.status_tracker.get_state(), self.status_tracker.current_position
    
    def step(self, action_index):
        action = self.action_space.get_indexed_action(action_index)
        self.num_steps += 1

        current_position, tower_location, dv_collected, dv_left, dv_transmittion_rate, dv_required = self.status_tracker.get_current_status()

        next_position = np.array(current_position) + np.array(action)
        self.status_tracker.update_position(next_position)
        

        data_volume_collected, data_transmitting_rate_list = Phi_dif_transmitting_speed(self.status_tracker.current_position, tower_location, dv_collected, dv_required)

        data_volume_left = np.array(dv_required) - np.array(data_volume_collected)

        self.status_tracker.update_dv_info(dv_collected=data_volume_collected.tolist(), 
                                    dv_transmittion_rate=data_transmitting_rate_list.tolist(), 
                                    dv_left=data_volume_left.tolist())

        self.running_info.store(position_t=self.status_tracker.current_position, action_t=action_index,
                                    data_collected_t=data_volume_collected.tolist(), 
                                    data_left_t=data_volume_left.tolist(), data_collect_rate_t = data_transmitting_rate_list.tolist())

        # NOTE: 判断是否到达终点
        reward = self.status_tracker.get_reward()
        reward -= 1 # 每步减少reward 1

        if self.status_tracker.is_done():
            reward -= 5 * LNG.norm(np.array(self.status_tracker.current_position) - np.array(self.status_tracker.arrival_at))

        return self.status_tracker.get_state(), reward, self.status_tracker.is_done(), self.status_tracker.current_position

    def view(self):
        print('data left = ', np.array(self.status_tracker.dv_required) - np.array(self.status_tracker.dv_collected), 'steps taken = ', self.num_steps)
        return self.running_info

class Actions():
    def __init__(self):
        self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]
        self.n = len(self.actions)

    def get_indexed_action(self, n):
        return self.actions[n]

    def get_actions(self):
        return self.actions

    def sample(self):
        return random.randint(0, len(self.actions)-1)