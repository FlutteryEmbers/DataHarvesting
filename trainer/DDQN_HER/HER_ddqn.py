from loguru import logger
import torch
import torch.nn as nn
import numpy as np
from trainer.DDQN_HER.networks import MLP
from trainer.DDQN_HER.HERBuffer import HindsightExperienceReplayMemory as ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDQN(object):
    def __init__(self, env, config, network_config, eval_name = 'her_ddqn_eval', target_name = 'her_ddqn_target') -> None:
        self.env = env
        self.batch_size = config['BATCH_SIZE']
        self.lr = config['LR']
        self.epsilon = config['EPSILON']
        self.decay_epsilon = config['EPSILON_DECAY_RATE']
        self.min_epsilon = config['MINIMUN_EPSILON']
        self.gamma = config['GAMMA']
        self.target_replace_iter = config['TARGET_REPLACE_ITER']
        self.memory_capaciy = config['MEMORY_CAPACITY']

        self.eval_name = eval_name
        self.target_name = target_name

        self.network_type  = env.state_mode
        
        logger.warning('Using FC network as Backend')
        self.inputs= len(env.board.get_state())
        self.outputs=env.action_space.n
        self.goal = env.goal

        self.eval_net = MLP(inputs=self.inputs, goals=len(self.goal), outputs=self.outputs, name=self.eval_name, \
                                fc_dim1=network_config['MLP']['FC1'], fc_dim2=network_config['MLP']['FC2']).to(device=device)

        self.target_net = MLP(inputs=self.inputs, goals=len(self.goal), outputs=self.outputs, name=self.target_name, \
                                fc_dim1=network_config['MLP']['FC1'], fc_dim2=network_config['MLP']['FC2']).to(device=device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = ReplayBuffer(memory_size=self.memory_capaciy, input_dims=self.inputs)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.env = env

    def choose_action(self, state, goal, disable_exploration=False):
        if np.random.random() > self.epsilon or disable_exploration:
            concat_state_goal = np.concatenate([state, goal])
            state = torch.tensor(np.array([concat_state_goal]), dtype=torch.float).to(self.eval_net.device)
            actions = self.eval_net.forward(state)

            action = torch.argmax(actions).item()
        else:
            action = self.env.action_space.sample()

        return action

    def store_transition(self, s, a, r, s_, done, goal):
        self.memory.add_experience(s, a, r, s_, done, goal)
        self.memory_counter += 1

    def get_sample_experience(self):
        """
        Gives a sample experience from the hindsight experience replay memory
        """
        state, action, reward, next_state, done, goal = self.memory.get_random_experience(
            self.batch_size)

        b_s = torch.tensor(state).to(self.eval_net.device)
        b_a = torch.tensor(action).to(self.eval_net.device)
        b_r = torch.tensor(reward).to(self.eval_net.device)
        b_s_ = torch.tensor(next_state).to(self.eval_net.device)
        is_done = torch.tensor(done, dtype=torch.int).to(self.eval_net.device)
        b_goal = torch.tensor(goal).to(self.eval_net.device)

        return b_s, b_a, b_r, b_s_, is_done, b_goal 

    def learn(self):
        if self.memory_counter < self.memory.max_mem_size:
            return

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        b_s, b_a, b_r, b_s_, is_done, b_goal = self.get_sample_experience()

        b_s = torch.cat((b_s, b_goal), 1)
        b_s_ = torch.cat((b_s_, b_goal), 1)
        
        b_a = b_a.unsqueeze(1)

        q_eval = self.eval_net(b_s).gather(1, b_a)

        q_eval_values = self.eval_net(b_s_).detach()
        _, a_prime = q_eval_values.max(1)

        q_target_values = self.target_net(b_s_).detach()
        q_target_s_a_prime = q_target_values.gather(1, a_prime.unsqueeze(1))
        q_target_s_a_prime = q_target_s_a_prime.squeeze()

        q_target = b_r.reshape(self.batch_size, 1) + self.gamma * q_target_s_a_prime.view(self.batch_size, 1) * (1 - is_done.reshape(self.batch_size, 1))

        loss = self.loss_func(q_eval, q_target).to(self.eval_net.device)

        '''
        batches = np.arange(self.batch_size)
        q_pred = self.eval_net.forward(b_s)[batches, b_a]
        q_next = self.target_net.forward(b_s_).max(dim=1)[0]

        q_next[is_done] = 0.0
        q_target = b_r + self.gamma * q_next

        # Computes loss and performs backpropagation
        loss = self.loss_func(q_target, q_pred).to(self.eval_net.device)
        '''
        # loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()                                     
        loss.backward()                                                 
        self.optimizer.step()
        self.decrement_epsilon()

    def save_models(self, mode):
        self.eval_net.save_checkpoint(mode=mode)
        self.target_net.save_checkpoint(mode=mode)

    def load_models(self, mode, checkpoints = None):
        self.eval_net.load_checkpoint(mode=mode)
        self.target_net.load_checkpoint(mode=mode)


    def unpack_memory(self, name, batch_samples):
        result = []
        for i in range(self.batch_size):
            result.append([i, batch_samples[i][name]])
        return result
    
    def lr_decay(self, max_train_steps):
        lr_now = self.lr * (1 - self.learn_step_counter / max_train_steps)
        
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def decrement_epsilon(self):
        """
        Decrements the epsilon after each step till it reaches minimum epsilon (0.1)
        epsilon = epsilon - decrement (default is 1e-5)
        """
        self.epsilon = self.epsilon - self.decay_epsilon if self.epsilon > self.min_epsilon \
            else self.min_epsilon