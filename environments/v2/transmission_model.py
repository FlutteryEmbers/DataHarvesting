from json import tool
import numpy as np
import math
import sys
from utils import tools, io
from tqdm import tqdm

# config_name = "configs/config_trans_model_2_D_2"
# save_file = "map/run1"
class Phi_dif_Model():
    def __init__(self, x_limit, y_limit, tower_position, phi_config_file, save_file, rounding = 0) -> None:
        self.phi_config_file = phi_config_file
        self.save_file = 'cache/map/{}'.format(save_file)
        Config = tools.load_config(self.phi_config_file)

        self.time_ratio = Config['TIME_RATIO']
        self.B = Config['B']
        self.height = Config['HEIGHT']
        self.K = Config['K']
        self.N = Config['N']
        self.Phi_list = np.array(Config['PHI_LIST'])
        self.signal_range = Config['signal_range']

        self.x_limit = x_limit
        self.y_limit = y_limit
        self.precision = 10 ** -rounding
        self.rounding = rounding

        self.tower_position = tower_position
        try:
            self.signal_map = self.load_map()
        except:
            self.signal_map = self.init_signal_map()
        

    def get_transmission_rate_stationary(self, agent_position, time_ratio):
        x = round(agent_position[0], self.rounding)
        y = round(agent_position[1], self.rounding)
        if (x, y) not in self.signal_map:
            sys.exit("signal_map not initialized correctly")

        transmitting_rate_list = np.array(self.signal_map[(x, y)]) * time_ratio
        '''
        dv_collected_updated = np.array(dv_collected) + np.array(transmitting_rate_list)*time_ratio
        dv_collected_updated = np.minimum(dv_collected_updated, dv_required)

        transmitting_rate = dv_collected_updated - dv_collected
        dv_left = dv_required - dv_collected_updated
        '''
        # return dv_collected_updated.tolist(),  transmitting_rate.tolist(), dv_left.tolist()
        return transmitting_rate_list

    def get_transmission_rate_dynamic(self, agent_position, tower_location, time_ratio):
        agent_position = np.array(agent_position)
        tower_location = np.array(tower_location)

        relative_distance = np.linalg.norm(agent_position - tower_location, axis=1)
        data_transmitting_rate = self.Phi_list * self.B * np.log2(1 + self.K / (self.N * (relative_distance*relative_distance + pow(self.height, 2))))
        data_transmitting_rate = time_ratio * data_transmitting_rate

        return data_transmitting_rate
    
    def init_signal_map(self):
        signal_map = {}
        x_position = np.arange(0, self.x_limit, self.precision, dtype=float)
        y_position = np.arange(0, self.y_limit, self.precision, dtype=float)
        print('Building Map')
        print(self.x_limit, self.y_limit, self.tower_position, self.Phi_list)
        # for i in tqdm(range(len(x_position))):
        #     x = x_position[i]
        for x in tqdm(x_position):
            # print(x)
            for y in y_position:
                rate_at_XY = []
                agent_position = [round(x, self.rounding), round(y, self.rounding)]
                for i in range(len(self.tower_position)):
                    tower = self.tower_position[i]
                    rate_at_XY_t = self.get_transmission_rate_signal(agent_position=agent_position, tower_location=tower, 
                                                                    phi=self.Phi_list[i], b=self.B, height=self.height,
                                                                    k=self.K, n=self.N)
                    rate_at_XY.append(rate_at_XY_t)

                signal_map[(round(x, self.rounding), round(y, self.rounding))] = rate_at_XY
    
        self.save_map(signal_map)
        # sys.exit('Done Building Map')
        return signal_map

    def get_transmission_rate_signal(self, agent_position, tower_location, phi, height, b, k, n):
        agent_position = np.array(agent_position)
        tower_location = np.array(tower_location)

        relative_distance = np.linalg.norm(agent_position - tower_location)
        data_transmitting_rate = phi * b * math.log2(1 + k / (n * (pow(relative_distance, 2) + pow(height, 2))))

        return data_transmitting_rate

    def save_map(self, map):
        io.dump_to_file(self.save_file, map)

    def load_map(self):
        print('loading {}'.format(self.save_file))
        signal_map = io.load_from_file(self.save_file)
        signal_stregth = self.get_transmission_rate_signal(agent_position=[1, 1], tower_location=self.tower_position[0], 
                                            phi=self.Phi_list[0], height=self.height,b=self.B, k=self.K, n=self.N)
        if signal_map[(1.0, 1.0)][0] != signal_stregth:
            sys.exit('Incorrect Singal Map')
        return io.load_from_file(self.save_file)