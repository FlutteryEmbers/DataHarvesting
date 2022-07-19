import numpy as np
import numpy.random as rand
from .transmission_model_v1 import Phi_dif_Model

class Random_Task():
    def __init__(self, x_limit, y_limit) -> None:
        self.x_limit = x_limit
        self.y_limit = y_limit
        
    def set_mission(self, start_at, arrival_at) -> None:
        self.start_at = start_at
        self.arrival_at = arrival_at
        self.current_position = self.start_at
        self.random_init_state()
        

    def random_init_state(self):
        # self.num_tower = rand.randint(3, 5)
        self.num_tower = 3
        x_coordinates = np.arange(1, self.x_limit, 1)
        y_coordinates = np.arange(1, self.y_limit, 1)
        rand.shuffle(x_coordinates)
        rand.shuffle(y_coordinates)

        tower_location = []
        dv_require = [30]*self.num_tower
        for i in range(self.num_tower):
            tower_location.append([x_coordinates[i], y_coordinates[i]])
            # dv_require[i] = rand.randint(30, 30)
        
        self.tower_location = tower_location
        self.transmitting_model = Phi_dif_Model(x_limit=self.x_limit, y_limit=self.y_limit, tower_position=self.tower_location)
        self.dv_required = dv_require

        print('tower_locations = ', tower_location, 'dv_require = ', dv_require)
        # print(dv_require)
        # print(self.num_tower)
        self.dv_left = self.dv_required
        self.dv_collected = [0]*len(self.dv_required)
        self.dv_transmittion_rate = [0]*len(self.dv_required)

    def reset(self):
        self.current_position = self.start_at
        self.random_init_state()

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
        dv_required = np.array(self.dv_required)

        state = np.concatenate((current_position, dv_collected, tower_location), axis=None)
        # state = np.concatenate((current_position, dv_collected, dv_required, tower_location), axis=None)
        
        return state.tolist()

    def is_done(self):
        if not np.array(self.dv_left).any():
            return True

        return False

    def get_reward(self):
        transmission_reward = 0.5*sum(self.dv_transmittion_rate)/len(self.dv_transmittion_rate)
        return transmission_reward


# rt = Random_Task(x_limit=10, y_limit=10)
# rt.random_init_state()