import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
import numpy as np
from utils import graph
import matplotlib.pyplot as plt
import pickle

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions) -> None:
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.next_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class Info():
    def __init__(self, board_structure, num_turrent, output_dir = '') -> None:
        self.board = board_structure
        self.num_turrent = num_turrent
        self.timestamp = 0

        self.position_t = []
        self.action_t = []
        self.data_collected_t = []
        self.data_left_t = []
        self.data_collect_rate_t = []

        self.output_dir = output_dir

        self.num_plots = 0
        self.final_reward = 0
        self.final_steps = 0
        

    def store(self, position_t, action_t, data_collected_t, data_left_t, data_collect_rate_t):
        self.timestamp += 1
        self.position_t.append(position_t)
        self.action_t.append(action_t)

        self.data_collected_t.append(data_collected_t)
        self.data_left_t.append(data_left_t)
        self.data_collect_rate_t.append(data_collect_rate_t)

    def save(self, sub_dir = '', plot = True):
        output_dir = self.output_dir + sub_dir
        # tools.mkdir(self.output_dir)

        data_collected = np.array(self.data_collected_t).T.tolist()
        data_left = np.array(self.data_left_t).T.tolist()
        data_collect_rate = np.array(self.data_collect_rate_t).T.tolist()
        paths = np.transpose(self.position_t, (1, 0, 2))

        # print(len(data_collected))
        if plot:
            self.plot_dv_info(filename='{}/data_collected'.format(output_dir), data=data_collected)
            self.plot_dv_info(filename='{}/data_left'.format(output_dir), data=data_left)
            self.plot_dv_info(filename='{}/data_collect_rate'.format(output_dir), data=data_collect_rate)

            graph.plot_path(x_limit = self.board.x_limit, y_limit= self.board.y_limit,\
                                    start_at = self.board.start_at, end_at=self.board.arrival_at,\
                                    tower_locations=self.board.tower_location, agent_paths = paths,\
                                    signal_range = self.board.signal_range, dir=output_dir)

        with open(output_dir + '/position_t.txt', 'w') as f:
            timestamp = 0
            for line in self.position_t:
                strings = str(timestamp) 
                for i in line:
                    strings += ' {}'.format(i)
                f.write(strings + '\n')
                timestamp += 1

        with open(output_dir + '/final_reward.txt', 'w') as f2:
            f2.write('final rewards: {}, final steps: {}'.format(self.final_reward, self.final_steps))

        with open(output_dir + '/path.pickle', 'wb') as handle:
            pickle.dump(self.position_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(output_dir + '/dv_collected.pickle', 'wb') as handle:
            pickle.dump(data_collected, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        handle.close()
        plt.close('all')
        
    def plot_dv_info(self, filename, data):
        # self.mkdir(type)
        t = [i+1 for i in range(self.timestamp)]
        plt.figure(figsize=(10,5))
        # ax = plt.gca() #you first need to get the axis handle
        # ax.set_aspect(0.8) #sets the height to width ratio to 1.5. 
        for i in range(self.num_turrent):
            # self.num_plots += 1
            # plot_curve(t, data[i], self.filename(type=type, turrent=i), self.num_plots)
            plt.plot(t, data[i], label="target {}".format(i))

        plt.legend(loc='upper left')
        # plt.title(filename)
        plt.xlabel("Time")
        plt.ylabel("Data Volumn")

        plt.savefig('{}.png'.format(filename), bbox_inches='tight', pad_inches=0.1, format='png', dpi=300)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0.1, format='eps', dpi=300)
        plt.close('all')

    def reset(self):
        self.timestamp = 0
        self.position_t = []
        self.action_t = []
        self.data_collected_t = []
        self.data_left_t = []
        self.data_collect_rate_t = []

