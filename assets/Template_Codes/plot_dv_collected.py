import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
# plt.figure(figsize=(10,5))

file_1 = 'board_loader_v3/batch_train_ddqn/0/random_seed_0/dv_collected.pickle'

fig, axs = plt.subplots(2, figsize=(10, 10))
#fig.suptitle('Vertically stacked subplots')

with open(file_1, 'rb') as handle:
    data = pickle.load(handle)
    handle.close()

t = [i+1 for i in range(len(data[0]))]

# ax = plt.gca() #you first need to get the axis handle
# ax.set_aspect(0.8) #sets the height to width ratio to 1.5. 
for i in range(2):
    # self.num_plots += 1
    # plot_curve(t, data[i], self.filename(type=type, turrent=i), self.num_plots)
    axs[1].plot(t, data[i], label="target {}".format(i + 1))

axs[1].legend(loc='lower right', prop={'size': 18})
axs[1].set_ylabel('Data Volumn Collected', labelpad=0.01, fontsize=20)
axs[1].set_xlabel("time", labelpad=0.001, loc='right')
axs[1].xaxis.set_label_coords(0, -0.025)
axs[1].set_title('Reward Engineering', fontsize=20, pad=0.01)

file_2 = 'board_loader_v3/batch_train_ddqn_her/0/random_seed_0/dv_collected.pickle'

with open(file_2, 'rb') as handle:
    data = pickle.load(handle)
    handle.close()

t = [i+1 for i in range(len(data[0]))]
plt.figure(figsize=(10,5))
# ax = plt.gca() #you first need to get the axis handle
# ax.set_aspect(0.8) #sets the height to width ratio to 1.5. 
for i in range(2):
    # self.num_plots += 1
    # plot_curve(t, data[i], self.filename(type=type, turrent=i), self.num_plots)
    axs[0].plot(t, data[i], label="target {}".format(i + 1))

axs[0].legend(loc='lower right', prop={'size': 18})
axs[0].set_ylabel('Data Volumn Collected', labelpad=0.001, fontsize=20)
axs[0].set_xlabel("time", labelpad=0.001, loc='right')
axs[0].xaxis.set_label_coords(0, -0.025)
axs[0].set_title('Binary Reward', fontsize=20, pad=0.01)


fig.savefig('{}.png'.format('dv_collected_engineering_vs_no_engineering'), bbox_inches='tight', pad_inches=0.1, format='png', dpi=300)
fig.savefig('{}.eps'.format('dv_collected_engineering_vs_no_engineering'), bbox_inches='tight', pad_inches=0.1, format='eps', dpi=300)

