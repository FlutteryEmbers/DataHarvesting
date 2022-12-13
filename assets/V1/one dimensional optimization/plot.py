from cProfile import label
from turtle import pos
from matplotlib import pyplot as plt
import numpy as np
x_limit = 10
y_limit = 10
tower_locations = [[7, 2], [1, 8]]
start_at = [0, 0]
end_at = [2, 5]
curved_path = False

stopping_time = {}
with open('data/position_t.txt') as f:
    paths = [0]
    for line in f: # read rest of lines
        # print(line)
        x, y, z = line.split()
        point = [float(z)]
        paths.append(float(z))
        if tuple(point) in stopping_time:
            stopping_time[tuple(point)] += 1
        else:
            stopping_time[tuple(point)] = 0
t_2 = []
optimal_path = []
position = 0.0
n_1 = 0
n_2 = 0
for i in range(7200):
    t_2.append(i*0.001)
    if round(position, 3) == 2.387 and n_1 < 768:
        n_1 += 1
    elif round(position, 3) == 4.410 and n_2 < 433:
        n_2 += 1
    else:
        position += 0.001
    optimal_path.append(position)
# print(paths)
# print(stopping_time)

t = np.arange(0, len(paths)/10, 0.1)

plt.plot(t, paths, label="DRL Solution")
plt.plot(t_2, optimal_path, label="Parameterized Optimal Solution")
plt.axhline(y=2, color='#ffcccc', linestyle='--')
plt.axhline(y=2.5, color='#ffcccc', linestyle='--')
plt.axhline(y=4.5, color='#ffcccc', linestyle='--')
plt.legend(loc='upper left')
plt.xlabel("time")
plt.ylabel("position")
plt.savefig('one-dimension.png', bbox_inches='tight', pad_inches=0.1, format='png', dpi=300)