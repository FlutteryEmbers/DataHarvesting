import math
import numpy as np
from numpy import linalg as LNG 

time_ratio=1
B=0.5
height=0.5
K=8
N=3
Phi_list=np.array([5,4,3])

def Phi_dif_transmitting_speed(agent_position, tower_location, data_volume_collected, data_volume_required):
     n_target_number = len(data_volume_collected)
  
     data_transmitting_rate_list=np.zeros(n_target_number)
     distance=np.zeros(n_target_number)

     for i in range(0, n_target_number):
          target_position = [tower_location[i][0], tower_location[i][1]]
          distance[i] = LNG.norm(np.array(agent_position) - np.array(target_position))

          if data_volume_collected[i] >= data_volume_required[i]:
               data_transmitting_rate_list[i] = 0
          else:
               data_transmitting_rate_list[i] = Phi_list[i] * B * math.log2(1 + K / (N * (pow(distance[i], 2) + pow(height, 2))))              

     
     # WARNING: DATA_TYPE
     data_volume_collected = np.array(data_volume_collected) + data_transmitting_rate_list*time_ratio
     for i in range(0, n_target_number):
          data_volume_collected[i] = min(data_volume_collected[i], data_volume_required[i])
          
     return data_volume_collected,data_transmitting_rate_list
