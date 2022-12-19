import numpy as np
import math
import random
from environments.v2 import models
from utils.buffer import Info
from numpy import linalg as LNG
from utils.tools import Timer
from utils import io
from loguru import logger


timer = Timer()
# NOTE: Discrete Position; Single Agent
class Agent():
    def __init__(self, x_limit, y_limit, start_at, arrival_at, tower_location, dv_required, phi_config_file, save_file, rounding = 2, control_time_scale = 2,
        action_type = 'Discrete', max_episode_steps = 100):
        # self.reward_func = self.test_reward_function
        self._max_episode_steps = max_episode_steps
        # self.status_tracker = task
        self.board = models.Board(x_limit=x_limit, y_limit=y_limit, start_at=start_at, arrival_at=arrival_at,
            tower_location=tower_location, dv_required=dv_required, 
            phi_config_file=phi_config_file, save_file=save_file, rounding=rounding, control_time_scale=control_time_scale)

        self.action_type = action_type
        self.goal = self.board.get_goal()
        
        if self.action_type == 'Discrete':
            self.action_space = models.Actions.Discrete(time_scale = control_time_scale)       
        elif self.action_type == 'Continuous':
            self.action_space = models.Actions.Continuous()
        elif self.action_type == '1D':
            self.action_space = models.Actions.LinearDiscrete()

        self.running_info = Info(board_structure=self.board, num_turrent=self.board.num_towers)
        
        self.reward = 0
        self.num_steps = 0

    def reset(self):
        self.reward = 0
        self.num_steps = 0

        self.board.reset()
        self.running_info.reset()

        s = self.board.get_state()
        # logger.debug(s)
        # return s, self.status_tracker.current_position
        return s


    def step(self, action, verbose = 0, type_reward = 'default'):
        if verbose > 0:
            logger.debug('angular representation: {}'.format(action))
        action = self.action_space.get_action(action)

        self.num_steps += 1

        data_volume_collected, data_transmitting_rate_list, data_volume_left = self.board.update_agent_state(i=0, action=action)
        
        position = self.board.get_agent_position(0).tolist()

        self.running_info.store(position_t=position, action_t=action,
                                    data_collected_t=data_volume_collected, 
                                    data_left_t=data_volume_left, data_collect_rate_t = data_transmitting_rate_list)

        done = False 

        if type_reward == 'HER':
            reward = -1
            if self.board.is_dv_collection_done() and self.board.is_all_arrived():
                # print('HER complete')
                reward = 0
                done = True

        elif type_reward == 'Simple':
            reward = -1
            if self.board.is_done():
                reward = -np.sum(abs(np.array(self.status_tracker.current_position) - np.array(self.status_tracker.arrival_at))) * self.action_space.time_scale
                done = True

        elif type_reward == 'Default':
            reward = self.board.get_reward()
            reward -= 1 # 每步减少reward 1
            

            # NOTE: 判断是否到达终点
            if self.board.is_dv_collection_done():
                reward = -np.sum(abs(np.array(self.board.get_agent_position(0)) - np.array(self.board.get_agent_goal(0))))
                done = True

        else:
            logger.critical('Invalid Reward Type')

        s = self.board.get_state()
        return s, reward, done, position

    def get_state(self):
        return self.board.get_state()

    def view(self):
        logger.info('data left = {} steps taken = {}'.format(np.array(self.board.dv_required) - np.array(self.board.dv_collected), self.num_steps))
        return self.running_info

    def save_task_info(self, output_dir):
        logs = self.board.description()
        io.save_log(output_dir, logs)
        return logs