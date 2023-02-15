import numpy as np
import math
from environments.v2.transmission_model import Phi_dif_Model
from environments.v2.controller import Target_Movement_Circular
from utils.buffer import Info

class Agent_List():
    def __init__(self, num_tower, start_at, arrival_at) -> None:
        self.start_at = start_at
        self.arrival_at = arrival_at
        self.num_agents = len(start_at)
        self.num_tower = num_tower

        self.current_position = np.array(start_at, dtype=np.float64)
        self.activate = np.ones(self.num_agents, dtype=bool)
        self.done = np.zeros(self.num_agents, dtype=bool)
        self.dv_collected = np.zeros((self.num_agents, num_tower))
        self.position_t = [self.current_position]

    def update_agent(self, i, position):
        if not self.done[i]:
            self.current_position[i] = position
            self.done[i] = not (self.current_position[i] - np.array(self.arrival_at[i])).any()

    def update_history(self):
        self.position_t.append(self.current_position)

    def reset(self):
        self.current_position = np.array(self.start_at, dtype=np.float64)
        self.activate = np.ones(self.num_agents, dtype=bool)
        self.done = np.zeros(self.num_agents, dtype=bool)
        self.dv_collected = np.zeros((self.num_agents, self.num_tower))
        self.position_t = [self.current_position]
    
    def is_done(self):
        return self.done.all()

    def get_state(self):
        return self.current_position.flatten()

class Targets():
    def __init__(self, tower_location, dv_required, args = None) -> None:
        self.start_at = tower_location
        self.dv_required = dv_required
        self.num_towers = len(tower_location)

        self.tower_location = np.array(self.start_at)
        self.dv_collected = np.zeros(self.num_towers, dtype=np.float64)
        self.dv_left = np.array(self.dv_required, dtype=np.float64)
        self.dv_transmittion_rate = np.zeros(self.num_towers, dtype=np.float64)
        self.is_moving = False

        if args != None:
            print('The Target Is Moving')
            self.is_moving = True 
            if args['type'] == 'circular':
                self.movement = Target_Movement_Circular(centers=tower_location, radius=args['radius'], w=args['w'], w_0=args['w_0'])

            self.tower_location = self.movement.locations
        else:
            print('Target Is Stationary')

    def reset(self):
        self.tower_location = np.array(self.start_at)
        self.dv_collected = np.zeros(self.num_towers, dtype=np.float64)
        self.dv_left = np.array(self.dv_required, dtype=np.float64)
        self.dv_transmittion_rate = np.zeros(self.num_towers, dtype=np.float64)
        if self.is_moving:
            self.movement.reset()
            self.tower_location = self.movement.locations

    def update_position(self, time_scale):
        if self.is_moving:
            self.movement.update(time_scale=time_scale)
            self.tower_location = self.movement.locations

    def update_dv_state(self, transmitting_rate_list):
        dv_collected_updated = self.dv_collected + np.array(transmitting_rate_list)
        dv_collected_updated = np.minimum(dv_collected_updated, self.dv_required)

        transmitting_rate = dv_collected_updated - self.dv_collected
        self.dv_left = np.array(self.dv_required) - dv_collected_updated
        self.dv_collected = dv_collected_updated

        return  self.dv_collected, transmitting_rate, self.dv_left

class Board():
    def __init__(self, x_limit, y_limit, start_at, arrival_at, tower_location, dv_required, phi_config_file, save_file, args = None, rounding = 2, control_time_scale = 2) -> None:
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.control_time_scale = control_time_scale

        self.num_towers = len(tower_location)
        self.targets = Targets(tower_location=tower_location, dv_required=dv_required, args=args)

        self.agents = Agent_List(num_tower=self.num_towers, start_at=start_at, arrival_at=arrival_at)
        self.transmitting_model = Phi_dif_Model(x_limit=x_limit, y_limit=y_limit, tower_position=tower_location, \
            phi_config_file= phi_config_file, save_file=save_file, rounding=rounding)
        self.running_log = Info(board_structure=None, num_turrent=self.num_towers)
        
    def reset(self):
        self.agents.reset()
        self.targets.reset()

    def update_agents(self, joint_actions):
        for i in range(self.agents.num_agents):
            if not self.agents.done[i]:
                self.update_agent_state(i, joint_actions[i])

    def update_agent_state(self, i, action):
        prev_position = np.array(self.agents.current_position[i], dtype=np.float64)
        # position = self.agents.current_position[i].tolist()
        position = np.array(self.agents.current_position[i], dtype=np.float64)
        next_position = np.array(self.agents.current_position[i]) + np.array(action[:], dtype=np.float64)
        next_position = np.round(next_position , 2)
        # next_position = next_position.tolist()
        if next_position[0] >= 0 and next_position[0] < self.x_limit:
            position[0] = next_position[0]

        if next_position[1] >= 0 and next_position[1] < self.y_limit:
            position[1] = next_position[1]

        
        self.agents.update_agent(i, position)
        update_action = np.array(position[:]) - prev_position
        dv_collected, cumulative_rate, dv_left = self.update_dv_status(prev_position, update_action)

        return dv_collected, cumulative_rate, dv_left

    def update_dv_status(self, position, action, communication_time_scale = 10):
        d_action = np.array(action) / communication_time_scale
        position = np.array(position[:])
        cumulative_rate = np.zeros(self.num_towers, dtype=np.float64)
        for _ in range(communication_time_scale):
            position = position + d_action
            if self.targets.is_moving:
                self.targets.update_position(1.0/(communication_time_scale*self.control_time_scale))

                transmitting_rate_list = self.transmitting_model.get_transmission_rate_dynamic(agent_position=position, tower_location=self.targets.tower_location, \
                    time_ratio=1.0/(communication_time_scale*self.control_time_scale))
            else:
                transmitting_rate_list = self.transmitting_model.get_transmission_rate_stationary(agent_position=position, time_ratio=1.0/(communication_time_scale*self.control_time_scale))
            
            dv_collected, dv_transmittion_rate_step, dv_left = self.targets.update_dv_state(transmitting_rate_list)
            cumulative_rate += np.array(dv_transmittion_rate_step)

        self.dv_transmittion_rate = cumulative_rate.tolist()
        return dv_collected, cumulative_rate, dv_left

    def is_dv_collection_done(self):
        return not np.array(self.targets.dv_left).any()

    def is_all_arrived(self):
        return self.agents.is_done()

    def get_state(self):
        return np.append(self.agents.current_position, self.targets.dv_collected)

    def get_agent_goal(self, i):
        return self.agents.arrival_at[i]

    def get_goal(self):
        arrival_at = self.get_agent_goal(0)
        return np.concatenate((arrival_at, self.targets.dv_required), axis=None)

    def get_agent_position(self, i):
        return self.agents.current_position[i]

    def description(self):
        return ['x_limit: {}'.format(self.x_limit), 'y_limit: {}'.format(self.y_limit),\
                #'start_at: {}'.format(self.start_at), 'arrival_at: {}'.format(self.arrival_at),\
                'dv_required: {}'.format(self.targets.dv_required)]

class Actions():
    class Discrete():
        def __init__(self, time_scale=2):
            self.time_scale = time_scale
            actions = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]) / self.time_scale
            self.actions = actions.tolist()
            self.n = len(self.actions)

        def get_action(self, n):
            return self.actions[n]

        def get_actions(self):
            return self.actions

        def sample(self):
            return np.random.randint(0, len(self.actions)-1)

    class LinearDiscrete():
        def __init__(self) -> None:
            self.actions = [[0, 0.1], [0, 0]]
            self.n = len(self.actions)
        
        def get_action(self, n):
            return self.actions[n]

        def sample(self):
            return np.random.randint(0, len(self.actions)-1)


    class Continuous():
        def __init__(self) -> None:
            self.shape = 2
            self.high = 1
            self.max_speed = 1
            self.max_angle = 360

        def get_action(self, action):
            r = action[0] * self.max_speed
            theta = action[1] * self.max_angle
            
            x = r * math.cos(math.radians(theta))
            y = r * math.sin(math.radians(theta))

            return [x, y]

        def sample(self):
            return np.random.rand(2)

class Agent():
    def __init__(self, start_at, arrival_at, num_tower) -> None:
        self.start_at = start_at[:]
        self.arrival_at = arrival_at[:]

        self.current_position = start_at[:]
        self.dv_collected = np.zeros(num_tower)
        self.activate = True
        self.done = False

        self.position_t = [self.start_at]

    def update_state(self, position):
        self.current_position = position[:]
        self.position_t.append(self.current_position)

    def reset(self):
        self.current_position = self.start_at[:]
        self.done = False

class Tower():
    def __init__(self, location, dv_required) -> None:
        self.location = location[:]
        self.dv_required = dv_required
        self.dv_left = dv_required
        self.done = False

    def reset(self):
        self.dv_left = self.dv_required
        self.done = False
