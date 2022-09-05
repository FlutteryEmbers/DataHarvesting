from utils import io  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from environments.config.transmission_model import Phi_dif_Model
model = Phi_dif_Model(x_limit=10, y_limit=10, tower_position=[[7, 2], [1, 8]], rounding=0, 
                        phi_config_file='configs/config_trans_model_2_D_3.yaml', save_file='prime')

transmission_model = model.signal_map
heat_t1 = np.zeros((10, 10))
heat_t2 = np.zeros((10, 10))
for index in transmission_model:
    i, j = index
    i = int(i)
    j = int(j)
    heat_t1[i, j] = transmission_model[index][0] + transmission_model[index][1]
    # heat_t2[i, j] = transmission_model[index][1]

# plt.imshow(heat_t1, cmap='viridis', interpolation='nearest')
sns.set()
ax = sns.heatmap(heat_t1, fmt='.1f', cmap="Blues", annot=True)
ax.invert_yaxis()
# ax = sns.heatmap(heat_t2, fmt='.3f', cmap="Blues")
plt.show()
# print(heat_t2[1, 2])