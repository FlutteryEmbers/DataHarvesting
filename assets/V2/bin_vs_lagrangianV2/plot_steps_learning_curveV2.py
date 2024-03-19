import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../../..'))

from random import random
from turtle import color
from utils import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_filename(name, random_index, config_no):
    return 'assets/V2/bin_vs_lagrangianV2/{}/config_{}/seed_{}/logs/ppo_history_steps.pickle'.format(name, config_no, random_index)

trail_names = ['config_5', 'config_5_binary', 'config_5_binary_wo_pen']
alt_names = ['with_Lagrangian', 'with_terminal_penalty', 'without_terminal_penalty']
colors = ['#ff8c1a', '#0066ff', '#e600e6']
face_colors = ['#ffb366', '#66a3ff', '#ffccff']

seeds = [10, 15, 243]
configs = [1]

minL = float('inf')
for seed in seeds:
    for name in trail_names:
        for config in configs:
            rewards = io.load_from_file(get_filename(name, seed, config))
            minL = min(minL, len(rewards))

trails_stats = {}
for name in trail_names:
    stats = np.zeros((len(seeds)*len(configs), minL))
    idx = 0
    for i in range(len(seeds)):
        for config in configs:
            rewards = io.load_from_file(get_filename(name, seeds[i], config))
            #min_reward = rewards[0]
            for j in range(minL):
                #min_reward = min(min_reward, rewards[j])
                stats[idx, j] = rewards[j]
                # stats[idx, j] = min_reward
            idx += 1

    trails_stats[name] = stats

x = np.arange(0, minL*1000, 1000)
clrs = sns.color_palette("husl", 5)
fig, ax = plt.subplots()

for i in range(len(trail_names)):
    alt_name = alt_names[i]
    name = trail_names[i]
    mean_arr = np.mean(trails_stats[name], axis=0)
    std_arr = np.std(trails_stats[name], axis=0)
    min_arr = np.min(trails_stats[name], axis=0)
    max_arr = np.max(trails_stats[name], axis=0)
    ax.plot(x, mean_arr, label=alt_name, color=colors[i])
    ax.fill_between(x, mean_arr-std_arr, mean_arr+std_arr, alpha=0.3, facecolor=face_colors[i])
    print('name: {}, mean: {}, std: {}'.format(alt_name, mean_arr[-1], std_arr[-1]))
    # ax.fill_between(x, min_arr, max_arr, alpha=0.3, facecolor=face_colors[i])
plt.legend(loc='upper right')
plt.xlabel("learning steps")
# plt.yscale('symlog')
plt.ylabel("steps")
plt.savefig('{}.png'.format('assets/V2/bin_vs_lagrangianV2/steps'), bbox_inches='tight', pad_inches=0.1, format='png', dpi=300)