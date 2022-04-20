import math
import numpy as np

def Phi_dif_transmitting_speed(agent_position,X_target,r_list,R_list,n_target_number,Phi_list)

  B=0.5
  height=0.5
  K=2
  N=3
  
  data_rate=np.zeros((1, n_target_number))
  distance=np.zeros((1, n_target_number))

  for i in range(1,n_target_number):
      distance(i) = abs(agent_position-X_target(i))
      if R(i)>= r(i):
        data_rate(i) = 0
      else:
        data_rate(i) = Phi_dif(i)*B*math.log2(1+K/(N*(distance(i)^2 + height^2)))
  
  return data_rate
