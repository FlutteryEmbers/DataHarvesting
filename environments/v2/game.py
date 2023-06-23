import numpy as np
import math
import random
from environments.v2 import models
from environments.v2 import controller
from utils.buffer import Info
from numpy import linalg as LNG
from utils.tools import Timer
from utils import io
from loguru import logger
import pygame

timer = Timer()
# NOTE: Discrete Position; Single Agent
class Agent():
    def __init__(self, x_limit, y_limit, start_at, arrival_at, tower_location, dv_required, phi_config_file, save_file, rounding = 2, control_time_scale = 2,
        action_type = 'Discrete', moving_target = 'stationary', max_episode_steps = 100):
        self.args = {}
        self.args['max_episode_steps'] = max_episode_steps
        self.args['x_limit'] = x_limit
        self.args['y_limit'] = y_limit
        self.args['start_at'] = start_at
        self.args['arrival_at'] = arrival_at
        self.args['tower_location'] = tower_location
        self.args['dv_required'] = dv_required
        self.args['phi_config_file'] = phi_config_file
        self.args['save_file'] = save_file
        self.args['rounding'] = rounding
        self.args['control_time_scale'] = control_time_scale
        self.args['action_type'] = action_type
        self.args['target_move_type'] = moving_target

        # self.reward_func = self.test_reward_function
        self._max_episode_steps = max_episode_steps
        # self.status_tracker = task
        self.board = models.Board(x_limit=x_limit, y_limit=y_limit, start_at=start_at, arrival_at=arrival_at,
            tower_location=tower_location, dv_required=dv_required, 
            phi_config_file=phi_config_file, save_file=save_file, rounding=rounding, control_time_scale=control_time_scale, args=self.args)

        self.action_type = action_type
        self.goal = self.board.get_goal()
        self.signal_range = [2, 3, 3.7, 4.3, 4.9]
        '''
        if self.action_type == 'Discrete':
            self.action_space = models.Actions.Discrete(time_scale = control_time_scale)       
        elif self.action_type == 'Continuous':
            self.action_space = models.Actions.Continuous()
        elif self.action_type == '1D':
            self.action_space = models.Actions.LinearDiscrete()
        '''
        self.action_space = controller.Actions[self.action_type](max_speed = 1/control_time_scale)
        self.running_info = Info(board_structure=self.board, num_turrent=self.board.num_towers)
        
        self.reward = 0
        self.num_steps = 0

        self.window = None
        self.clock = None
        self.window_y = y_limit
        self.window_x = x_limit
        self.icon_size = 30

    def reset(self):
        self.reward = 0
        self.num_steps = 0

        self.board.reset()
        self.running_info.reset()

        s = self.board.get_state()
        self._render_frame()
        # logger.debug(s)
        # return s, self.status_tracker.current_position
        return s

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def step(self, action, args, verbose = 0):
        if verbose > 0:
            logger.debug('angular representation: {}'.format(action))
        action = self.action_space.get_action(action)

        self.num_steps += 1

        data_volume_collected, data_transmitting_rate_list, data_volume_left = self.board.update_agents(joint_actions=action)
        
        position = self.board.get_all_agents_position().tolist()

        self.running_info.store(position_t=position, action_t=action,
                                    data_collected_t=data_volume_collected, 
                                    data_left_t=data_volume_left, data_collect_rate_t = data_transmitting_rate_list)

        done = False 

        if args.type_reward == 'HER':
            reward = -1
            if self.board.is_dv_collection_done() and self.board.is_all_arrived():
                # print('HER complete')
                reward = 0
                done = True

        elif args.type_reward == 'Simple':
            reward = -1
            if self.board.is_done():
                reward = -np.sum(abs(np.array(self.status_tracker.current_position) - np.array(self.status_tracker.arrival_at))) * self.action_space.time_scale
                done = True

        elif args.type_reward == 'Shaped_Reward':
            reward = -1 # 每步减少reward 1
            pos_penalty = -0.1 * np.linalg.norm(abs(np.array(self.board.get_agent_position(0)) - np.array(self.board.get_agent_goal(0))))
            dv_penalty = -0.1 * np.linalg.norm(data_volume_left)

            # if self.num_steps >= self._max_episode_steps:
                  
            # NOTE: 判断是否到达终点
            if self.board.is_dv_collection_done() or self.num_steps >= self._max_episode_steps:
                # reward = 0
                pos_penalty = -10 * np.linalg.norm(abs(np.array(self.board.get_agent_position(0)) - np.array(self.board.get_agent_goal(0))))
                dv_penalty = -10 * np.linalg.norm(data_volume_left)
                ## reward = -np.sum(abs(np.array(self.board.get_agent_position(0)) - np.array(self.board.get_agent_goal(0))))
                done = self.board.is_dv_collection_done()

            reward  += pos_penalty + dv_penalty

        elif args.type_reward == 'Negative_Shaped_Reward':
            reward = -1 # 每步减少reward 1
            pos_penalty = 0.1 * np.linalg.norm(abs(self.board.get_all_agents_position() - self.board.get_all_agents_goal()), axis=1)
            dv_penalty = 0.1 * np.linalg.norm(data_volume_left)

            # if self.num_steps >= self._max_episode_steps:
                  
            # NOTE: 判断是否到达终点
            if self.board.is_dv_collection_done() or self.num_steps >= self._max_episode_steps:
                # reward = 0
                pos_penalty = 10 * np.linalg.norm(abs(self.board.get_all_agents_position() - self.board.get_all_agents_goal()), axis=1)
                dv_penalty = 10 * np.linalg.norm(data_volume_left)
                ## reward = -np.sum(abs(np.array(self.board.get_agent_position(0)) - np.array(self.board.get_agent_goal(0))))
                done = self.board.is_dv_collection_done()

            reward  += pos_penalty + dv_penalty

        elif args.type_reward == 'Lagrangian':
            reward = -1
            if self.board.is_dv_collection_done() or self.num_steps >= self._max_episode_steps:
                # reward = 0
                pos_penalty = 10 * np.linalg.norm(abs(self.board.get_all_agents_position() - self.board.get_all_agents_goal()), axis=1).mean()
                dv_penalty = 10 * np.linalg.norm(data_volume_left)
                # reward = - (pos_penalty + dv_penalty) 
                done = self.board.is_dv_collection_done()
            # done = True

        else:
            logger.critical('Invalid Reward Type')


        if done:
            self.num_steps += np.linalg.norm(np.array(self.board.get_agent_position(0)) - np.array(self.board.get_agent_goal(0)))
            # self.num_steps += np.linalg.norm(abs(self.board.get_all_agents_position() - self.board.get_all_agents_goal()), axis=1).sum()
        
        self.reward += reward
        self.running_info.final_reward = self.reward
        self.running_info.final_steps = self.num_steps
        s = self.board.get_state()
        
        return s, reward, done, position

    def get_state(self):
        return self.board.get_state()

    def view(self):
        logger.info('data left = {} steps taken = {}'.format(np.array(self.board.targets.dv_required) - np.array(self.board.targets.dv_collected), self.num_steps))
        return self.running_info

    def save_task_info(self, output_dir):
        logs = self.board.description()
        io.save_log(output_dir, logs)
        io.save_config(output_dir=output_dir, args=self.args, name='env_config')
        return logs

    def render(self, display = False):
        if display:
            self._render_frame()

    def _render_frame(self):
        color_wheel = ["#cc99ff50", "#ff99ff50", "#ffb36650", "#ff4d9450", "#80b3ff50"]
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_x*self.icon_size, self.window_y*self.icon_size))
            # self.window = pygame.display.set_mode((1000, 1000))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_x*self.icon_size, self.window_y*self.icon_size))
        canvas = pygame.Surface((1000, 1000))
        canvas.fill((255, 255, 255))

        for start_at in self.board.agents.start_at:
            pygame.draw.rect(
                canvas,
                (255, 0, 255),
                pygame.Rect(
                    self.icon_size * np.array(start_at),
                    (self.icon_size, self.icon_size),
                ),
            )

        for arrival_at in self.board.agents.arrival_at:
            pygame.draw.rect(
                canvas,
                (255, 255, 0),
                pygame.Rect(
                    self.icon_size * np.array(arrival_at),
                    (self.icon_size, self.icon_size),
                ),
            )

        pygame.draw.line(
            canvas,
            0,
            (0, self.icon_size * self.window_y),
            (1000, self.icon_size * self.window_y),
            width=3,
        )

        pygame.draw.line(
            canvas,
            0,
            (self.icon_size * self.window_x, 0),
            (self.icon_size * self.window_x, 1000),
            width=3,
        )

        for agent_position in self.board.agents.current_position:
            # agent_position = self.board.agents.current_position[i]
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (agent_position) * self.icon_size,
                self.icon_size / 5,
            )
            
        i = 0
        for target_position in self.board.targets.tower_location:
            
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (target_position) * self.icon_size,
                self.icon_size / 5,
            )

            pygame.draw.circle(
                canvas,
                (0, 200-10*i, 50*i),
                (target_position) * self.icon_size,
                self.signal_range[i] * self.icon_size,
                width=3
            )
            i += 1

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(15)
        