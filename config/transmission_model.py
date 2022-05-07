import math
import numpy as np

time_ratio=0.1
B=0.5
height=0.5
K=2
N=3
Phi_list=np.array([5,4,3])

def Phi_dif_transmitting_speed(agent_position, X_target_position, data_volume_collected, data_volume_required):
     n_target_number = len(data_volume_collected)
  
     data_transmitting_rate_list=np.zeros((1, n_target_number))
     distance=np.zeros((1, n_target_number))

     for i in range(0,n_target_number):
          distance[i] = abs(np.array(agent_position) - X_target_position[i])

          if data_volume_collected[i]>= data_volume_required[i]:
               data_transmitting_rate_list[i] = 0
          else:
               data_transmitting_rate_list[i] = Phi_list[i]*B*math.log2(1+K/(N*(distance[i]^2 + height^2)))
     
     # WARNING: DATA_TYPE
     data_volume_collected = np.array(data_volume_collected) + data_transmitting_rate_list*time_ratio
  
     return data_volume_collected,data_transmitting_rate_list
