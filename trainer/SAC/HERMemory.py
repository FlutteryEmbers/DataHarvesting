import numpy as np
import torch


class HindsightExperienceReplayMemory(object):
    """
    Hindsight Experience replay - Takes size, input dimensions and number of actions as parameters
    """
    def __init__(self, memory_size, input_dims, n_actions):
        super(HindsightExperienceReplayMemory, self).__init__()
        self.max_mem_size = memory_size
        self.counter = 0

        # initializes the state, next_state, action, reward, and terminal experience memory
        self.state_memory = np.zeros((memory_size, input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((memory_size, input_dims), dtype=np.float32)
        self.reward_memory = np.zeros((memory_size, 1), dtype=np.float32)
        self.action_memory = np.zeros((memory_size, n_actions), dtype=np.int64)
        self.terminal_memory = np.zeros((memory_size, 1), dtype=bool)
        self.goal_memory = np.zeros((memory_size, input_dims), dtype=np.float32)

        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)

    def add_experience(self, state, action, reward, next_state, done, goal):
        """
        Adds new experience to the memory
        """
        curr_index = self.counter % self.max_mem_size

        self.state_memory[curr_index] = state
        self.action_memory[curr_index] = action
        self.reward_memory[curr_index] = reward
        self.next_state_memory[curr_index] = next_state
        self.terminal_memory[curr_index] = done
        self.goal_memory[curr_index] = goal

        self.counter += 1

    def get_random_experience(self, batch_size):
        """
        Returns any random memory from the experience replay memory
        """
        rand_index = np.random.choice(min(self.counter, self.max_mem_size), batch_size, replace=False)

        rand_state = self.state_memory[rand_index]
        rand_action = self.action_memory[rand_index]
        rand_reward = self.reward_memory[rand_index]
        rand_next_state = self.next_state_memory[rand_index]
        rand_done = self.terminal_memory[rand_index]
        rand_goal = self.goal_memory[rand_index]

        rand_state = torch.tensor(rand_state, dtype=torch.float).to(self.device)
        rand_action = torch.tensor(rand_action, dtype=torch.float).to(self.device)
        rand_reward = torch.tensor(rand_reward, dtype=torch.float).to(self.device)
        rand_next_state = torch.tensor(rand_next_state, dtype=torch.float).to(self.device)
        rand_done = torch.tensor(rand_done, dtype=torch.float).to(self.device)
        rand_goal = torch.tensor(rand_goal, dtype=torch.float).to(self.device)

        return rand_state, rand_action, rand_reward, rand_next_state, rand_done, rand_goal