import math
import numpy as np

def Phi_dif_transmitting_speed(agent_position,X_target_position,r_list,R_list,n_target_number,Phi_list):

  B=0.5
  height=0.5
  K=2
  N=3
  
  data_transmitting_rate_list=np.zeros((1, n_target_number))
  distance=np.zeros((1, n_target_number))

  for i in range(0,n_target_number):
      distance[i] = abs(agent_position-X_target_position[i])
      if R_list[i]>= r_list[i]:
        data_transmitting_rate_list[i] = 0
      else:
        data_transmitting_rate_list[i] = Phi_list[i]*B*math.log2(1+K/(N*(distance[i]^2 + height^2)))
  
  return data_transmitting_rate_list