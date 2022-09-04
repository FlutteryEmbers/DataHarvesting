from utils import io  
import matplotlib.pyplot as plot
import numpy as np

transmission_model = io.load_from_file('map/borad_loarder_v3')
heat_t1 = np.zeros((10*100, 10*100))
heat_t2 = np.zeros((10*100, 10*100))
for index in transmission_model:
    i, j = index
    i = int(i*100)
    j = int(j*100)
    heat_t1[i, j] = transmission_model[index][0]
    heat_t2[i, j] = transmission_model[index][1]

print(heat_t2[1, 2])