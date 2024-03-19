import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../../../..'))

import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
# plt.figure(figsize=(10,5))
file_2 = 'assets/V2/DDQN_vs_PPO/P1/Aug18-21_02-ppo_stationary_vanilla/best_case/dv_collected.pickle'
file_1 = 'assets/V2/DDQN_vs_PPO/P1/ddqn_ma/best_case/dv_collected.pickle'

fig, axs = plt.subplots(2, figsize=(10, 10))
#fig.suptitle('Vertically stacked subplots')
N = 30
with open(file_1, 'rb') as handle:
    data = pickle.load(handle)
    handle.close()

t = [i+1 for i in range(N)]
# print(data)
#t = [i+1 for i in range(self.timestamp)]
plt.figure(figsize=(10,5))
# ax = plt.gca() #you first need to get the axis handle
# ax.set_aspect(0.8) #sets the height to width ratio to 1.5. 
for i in range(4):
    res = [0] * N
    for j in range(N):
        res[j] = data[i][j] if j < len(data[i]) else data[i][-1]

    # self.num_plots += 1
    # plot_curve(t, data[i], self.filename(type=type, turrent=i), self.num_plots)
    axs[0].plot(t, res, label="target {}".format(i+1))

axs[0].legend(loc='lower right', prop={'size': 18})
axs[0].set_ylabel('Data Volumn Collected', labelpad=0.01, fontsize=20)
axs[0].set_xlabel("time", labelpad=0.001, loc='right')
axs[0].xaxis.set_label_coords(0, -0.025)
axs[0].set_title('Case A: DDQN', fontsize=20, pad=0.01)



with open(file_2, 'rb') as handle:
    data = pickle.load(handle)
    handle.close()

t = [i+1 for i in range(N)]
# print(data)
#t = [i+1 for i in range(self.timestamp)]
plt.figure(figsize=(10,5))
# ax = plt.gca() #you first need to get the axis handle
# ax.set_aspect(0.8) #sets the height to width ratio to 1.5. 
for i in range(4):
    res = [0] * N
    for j in range(N):
        res[j] = data[i][j] if j < len(data[i]) else data[i][-1]
    # self.num_plots += 1
    # plot_curve(t, data[i], self.filename(type=type, turrent=i), self.num_plots)
    axs[1].plot(t, res, label="target {}".format(i+1))

axs[1].legend(loc='lower right', prop={'size': 18})
axs[1].set_ylabel('Data Volumn Collected', labelpad=0.001, fontsize=20)
axs[1].set_xlabel("time", labelpad=0.001, loc='right')
axs[1].xaxis.set_label_coords(0, -0.025)
axs[1].set_title('Case B: PPO', fontsize=20, pad=0.01)


fig.savefig('{}.png'.format('ddqn_vs_ppo_12'), bbox_inches='tight', pad_inches=0.1, format='png', dpi=300)
fig.savefig('{}.eps'.format('ddqn_vs_ppo_12'), bbox_inches='tight', pad_inches=0.1, format='eps', dpi=300)