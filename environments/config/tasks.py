import numpy as np
import numpy.random as rand
from .transmission_model import Phi_dif_Model
from loguru import logger

class Single_Task():
    def __init__(self, x_limit, y_limit, tower_location) -> None:
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.tower_location = tower_location
        self.num_tower = len(tower_location)
        self.transmitting_model = Phi_dif_Model(x_limit=self.x_limit, y_limit=self.y_limit, tower_position=self.tower_location, rounding=2)
        self.name = 'Single_Task'

    def set_mission(self, start_at, arrival_at, dv_required) -> None:
        self.dv_required = dv_required
        self.start_at = start_at
        self.arrival_at = arrival_at
        
        self.current_position = self.start_at[:]
        self.dv_left = self.dv_required[:]
        self.dv_collected = [0]*len(self.dv_required)
        self.dv_transmittion_rate = [0]*len(self.dv_required)
        logger.trace('position: {} towers: {} dv_require: {}'.format(self.current_position, self.tower_location, self.dv_required))
        
    def reset(self):
        self.current_position = self.start_at[:]
        self.dv_left = self.dv_required[:]
        self.dv_collected = [0]*len(self.dv_required)
        self.dv_transmittion_rate = [0]*len(self.dv_required)
        logger.trace('position: {} towers: {} dv_require: {}'.format(self.current_position, self.tower_location, self.dv_required))

    def resume(self, position, dv_collected):
        self.current_position = position[:]
        self.dv_collected = dv_collected[:]
        self.dv_left = (np.array(self.dv_required) - np.array(self.dv_collected)).tolist()

    def get_state(self):
        current_position = np.array(self.current_position)
        tower_location = np.array(self.tower_location)
        dv_collected = np.array(self.dv_collected)
        dv_required = np.array(self.dv_required)

        dv_collected_ratio = dv_collected/dv_required
        # state = np.concatenate((current_position, dv_collected_ratio, tower_location), axis=None)
        # state = np.concatenate((current_position, dv_collected_ratio, dv_required, tower_location), axis=None)
        state = np.concatenate((current_position, dv_collected), axis=None)
        # state = np.concatenate((current_position, dv_collected), axis=None)
        return state.tolist()

    ## Important
    def get_reward(self):
        dv_transmisttion_rate = np.array(self.dv_transmittion_rate)
        dv_required = np.array(self.dv_required)
        transmission_reward = 10*np.sum(dv_transmisttion_rate/dv_required)
        return transmission_reward

    def update_dv_info(self, dv_left, dv_collected, dv_transmittion_rate):
        self.dv_left = dv_left
        self.dv_collected = dv_collected
        self.dv_transmittion_rate = dv_transmittion_rate

    def update_position(self, action):
        old_position = np.array(self.current_position[:])
        next_position = np.array(self.current_position[:]) + np.array(action[:])
        next_position = np.round(next_position , 2)
        # next_position = next_position.tolist()
        if next_position[0] >= 0 and next_position[0] < self.x_limit:
            self.current_position[0] = next_position[0]

        if next_position[1] >= 0 and next_position[1] < self.y_limit:
            self.current_position[1] = next_position[1]
        
        update_action = np.array(self.current_position[:]) - old_position
        self.update_dv_status(old_position, update_action)

        return self.current_position[:], self.dv_collected, self.dv_transmittion_rate, self.dv_left
    
    def update_dv_status(self, position, action):
        delta = 10
        N = 2
        d_action = np.array(action) / delta
        position = np.array(position[:])
        for _ in range(delta):
            position = position + d_action
            self.dv_collected,  self.dv_transmittion_rate, self.dv_left = \
                self.transmitting_model.update_dv_status(position, self.dv_collected, self.dv_required, 1.0/(delta*N*10))

    def get_current_status(self):
        return self.current_position, self.tower_location, self.dv_collected, self.dv_left, self.dv_transmittion_rate, self.dv_required

    def is_done(self):
        if not np.array(self.dv_left).any():
            return True

        return False

    def get_goal(self):
        return np.concatenate((self.arrival_at, self.dv_required), axis=None)

    def description(self):
        return ['x_limit: {}'.format(self.x_limit), 'y_limit: {}'.format(self.y_limit),\
                'start_at: {}'.format(self.start_at), 'arrival_at: {}'.format(self.arrival_at),\
                'dv_required: {}'.format(self.dv_required)]


'''
class Status_Tracker(object):
    def __init__(self, x_limit, y_limit, tower_location) -> None:
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.tower_location = tower_location
        self.num_tower = len(tower_location)
        self.transmitting_model = Phi_dif_Model(x_limit=self.x_limit, y_limit=self.y_limit, tower_position=self.tower_location)
    
    def set_mission(self, start_at, arrival_at, dv_required) -> None:
        self.dv_required = dv_required
        self.start_at = start_at
        self.arrival_at = arrival_at
        
        self.current_position = self.start_at[:]
        self.dv_left = self.dv_required[:]
        self.dv_collected = [0]*len(self.dv_required)
        self.dv_transmittion_rate = [0]*len(self.dv_required)
'''