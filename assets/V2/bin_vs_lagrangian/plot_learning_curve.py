import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../../..'))

from random import random
from turtle import color
from utils import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_filename(name, random_index):
    return 'assets/V2/lagrangian/{}/seed_{}.pickle'.format(name, random_index)

trail_names = ['binary', 'lagrangian']
colors = ['#ff8c1a', '#0066ff']
face_colors = ['#ffb366', '#66a3ff']

seeds = [10, 15, 243]

minL = float('inf')
for seed in seeds:
    for name in trail_names:
        rewards = io.load_from_file(get_filename(name, seed))
        minL = min(minL, len(rewards))

trails_stats = {}
for name in trail_names:
    stats = np.zeros((len(seeds), minL))
    for i in range(len(seeds)):
        rewards = io.load_from_file(get_filename(name, seeds[i]))
        for j in range(minL):
            stats[i, j] = rewards[j]

    trails_stats[name] = stats

x = np.arange(minL)
clrs = sns.color_palette("husl", 5)
fig, ax = plt.subplots()

for i in range(len(trail_names)):
    name = trail_names[i]
    mean_arr = np.mean(trails_stats[name], axis=0)
    std_arr = np.mean(trails_stats[name], axis=0)
    ax.plot(x, mean_arr, label=name, color=colors[i])
    ax.fill_between(x, mean_arr-std_arr, mean_arr+std_arr, alpha=0.3, facecolor=face_colors[i])

plt.legend(loc='lower right')
plt.xlabel("iteration (each point corresponding to 1000 learning steps)")
plt.yscale('symlog')
plt.ylabel("reward")
plt.savefig('{}.png'.format('assets/V2/lagrangian/rewards'), bbox_inches='tight', pad_inches=0.1, format='png', dpi=300)