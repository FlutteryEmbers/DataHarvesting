from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import math

K = [1, 10 , 100, 1000, 10000]
x = np.arange(-500, 500)
# print(x)
B = 0.5
fig, ax = plt.subplots()
for k in K:
    y = B*np.log2(1+k/x**2)
    ax.plot(x, y, label='K = {}'.format(k))

plt.legend(loc='upper right')
plt.savefig('{}.png'.format('assets/broadcast'))
