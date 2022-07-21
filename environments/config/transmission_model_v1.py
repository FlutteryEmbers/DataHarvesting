import numpy as np
import math
import sys
import yaml

class Phi_dif_Model():
    def __init__(self, x_limit, y_limit, tower_position, rounding = 0, time_ratio=1, B=0.5, height=0.5, K=8, N=3, Phi_list=np.array([5,4,3, 3, 3])) -> None:
        with open("configs/config_trans_model.yaml", 'r') as stream:
            Config = yaml.safe_load(stream)
        # print(Config)
        if Config == None:
            sys.exit('Trans_model initial not correctly')

        self.time_ratio = Config['TIME_RATIO']
        self.B = Config['B']
        self.height = Config['HEIGHT']
        self.K = Config['K']
        self.N = Config['N']
        self.Phi_list = np.array(Config['PHI_LIST'])

        self.x_limit = x_limit
        self.y_limit = y_limit
        self.precision = 10 ** -rounding
        self.rounding = rounding

        self.tower_position = tower_position
        self.signal_map = self.init_signal_map()

    def update_dv_status(self, agent_position, dv_collected, dv_required):
        x = round(agent_position[0], self.rounding)
        y = round(agent_position[1], self.rounding)
        if (x, y) not in self.signal_map:
            sys.exit("signal_map not initialized correctly")

        transmitting_rate_list = self.signal_map[(x, y)]
        dv_collected_updated = np.array(dv_collected) + np.array(transmitting_rate_list*self.time_ratio)
        dv_collected_updated = np.minimum(dv_collected_updated, dv_required)

        transmitting_rate = dv_collected_updated - dv_collected
        dv_left = dv_required - dv_collected_updated

        return dv_collected_updated.tolist(),  transmitting_rate.tolist(), dv_left.tolist()
        
    
    def init_signal_map(self):
        signal_map = {}
        x_position = np.arange(0, self.x_limit, self.precision, dtype=float)
        y_position = np.arange(0, self.y_limit, self.precision, dtype=float)

        for x in x_position:
            for y in y_position:
                rate_at_XY = []
                agent_position = [x, y]
                for i in range(len(self.tower_position)):
                    tower = self.tower_position[i]
                    rate_at_XY_t = self.get_transmission_rate_signal(agent_position=agent_position, tower_location=tower, 
                                                                    phi=self.Phi_list[i], b=self.B, height=self.height,
                                                                    k=self.K, n=self.N)
                    rate_at_XY.append(rate_at_XY_t)

                signal_map[(x, y)] = rate_at_XY

        return signal_map

    def get_transmission_rate_signal(self, agent_position, tower_location, phi, height, b, k, n):
        agent_position = np.array(agent_position)
        tower_location = np.array(tower_location)

        relative_distance = np.linalg.norm(agent_position - tower_location)
        data_transmitting_rate = phi * b * math.log2(1 + k / (n * (pow(relative_distance, 2) + pow(height, 2))))

        return data_transmitting_rate