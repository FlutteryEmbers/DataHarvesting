import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../../..'))
import numpy as np
from environments.v2.controller import Actions

MA = Actions['MA_Continuous'](max_speed = 1)
SA = Actions['Continuous'](max_speed = 1)
action = np.random.rand(2)

print(action)
MA.get_action(action)
print(action)
SA.get_action(action)
