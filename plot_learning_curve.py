from random import random
from utils import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

her_name = 'batch_train_ddqn_her'
pher_name = 'batch_train_ddqn_pher'
length = 45
def get_filename(name, sample, random_index):
    return 'assets/sampling/{}/{}/random_seed_{}/ddqn_random_seed_{}_history_rewards.pickle'.format(name, sample, random_index, random_index)

her_result = np.zeros((12, length))
for i in range(12):
    rewards = io.load_from_file(get_filename(her_name, 0, i))
    for j in range(length):
        her_result[i, j] = rewards[j]

her_mean = np.mean(her_result, axis=0)
her_std = np.std(her_result, axis=0)
print(np.mean(her_result, axis=0))

pher_result = np.zeros((12, length))
for i in range(12):
    rewards = io.load_from_file(get_filename(pher_name, 0, i))
    # print(len(rewards))
    for j in range(length):
        pher_result[i, j] = rewards[j]

pher_mean = np.mean(pher_result, axis=0) 
pher_std = np.std(pher_result, axis=0)
print(np.mean(pher_result, axis=0))

x = np.arange(length)
clrs = sns.color_palette("husl", 5)
fig, ax = plt.subplots()

ax.plot(x, her_mean, label="HER_DDQN with Uniformly Sampling")
ax.fill_between(x, her_mean-her_std, her_mean+her_std, alpha=0.3, facecolor=clrs[0])

ax.plot(x, pher_mean, label="HER_DDQN with Prioritized Sampling")
ax.fill_between(x, pher_mean-pher_std, pher_mean+pher_std, alpha=0.3, facecolor=clrs[1])
plt.legend(loc='lower right')
plt.xlabel("iteration (each point corresponding to 1000 learning steps)")
plt.ylabel("reward")
plt.savefig('{}.png'.format('assets/sampling'))