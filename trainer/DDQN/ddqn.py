from loguru import logger
import torch
import torch.nn as nn
import numpy as np
from .networks import MLP, CNN
from utils.buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDQN(object):
    def __init__(self, env, config, network_config, eval_name = 'ddqn_eval', target_name = 'ddqn_target') -> None:
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
        if self.network_type == 'CNN':
            logger.warning('Using CNN network as Backend')
            state = env.status_tracker.get_state(mode='CNN')
            self.inputs= len(state)
            self.x_limit = env.status_tracker.x_limit
            self.y_limit = env.status_tracker.y_limit
            self.info_length = len(state) - self.x_limit*self.y_limit

            self.eval_net, self.target_net = CNN(self.x_limit, self.y_limit, self.info_length, env.action_space.n, self.eval_name).to(device=device), CNN(self.x_limit, self.y_limit, self.info_length, env.action_space.n, self.target_name).to(device=device)
        else:
            logger.warning('Using FC network as Backend')
            self.inputs= len(env.status_tracker.get_state())
            self.outputs=env.action_space.n
            
            self.eval_net = MLP(inputs=self.inputs, outputs=self.outputs, name=self.eval_name, fc_dim1=network_config['MLP']['FC1'], fc_dim2=network_config['MLP']['FC2']).to(device=device)
            self.target_net = MLP(inputs=self.inputs, outputs=self.outputs, name=self.target_name, fc_dim1=network_config['MLP']['FC1'], fc_dim2=network_config['MLP']['FC2']).to(device=device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = ReplayBuffer(max_size=self.memory_capaciy, input_shape=self.inputs, n_actions=1)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.env = env

    def choose_action(self, state, disable_exploration=False):
        self.eval_net.eval()
        state = torch.FloatTensor(np.array(state)).to(device)
        state = torch.unsqueeze(state, dim=0)
        '''
        global EPSILON
        if self.learn_step_counter % 100000 == 0:
            EPSILON = EPSILON * 0.99
            print('EPSILON = ', EPSILON)
        '''
        if np.random.uniform() < self.epsilon and not disable_exploration:
           action = self.env.action_space.sample()

        elif self.network_type == 'CNN':
            q_value = self.eval_net(state, self.x_limit, self.y_limit)
            _, action_value = torch.max(q_value, dim=1)
            action = int(action_value.item())
        else:
            q_value = self.eval_net(state)
            _, action_value = torch.max(q_value, dim=1)
            action = int(action_value.item())

        self.eval_net.train()
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory.store_transition(s, a, r, s_, done)
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter < self.memory.mem_size:
            return

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        b_s = torch.tensor(state, dtype=torch.float).to(device=device)
        b_a = torch.tensor(action, dtype=torch.long).to(device=device)
        b_r = torch.tensor(reward, dtype=torch.float).to(device=device)
        b_s_ = torch.tensor(new_state, dtype=torch.float).to(device=device)
        is_done = torch.tensor(done, dtype=torch.int).to(device=device)

        if self.network_type == 'CNN':
            q_eval = self.eval_net(b_s, self.x_limit, self.y_limit).gather(1, b_a)

            q_eval_values = self.eval_net(b_s_, self.x_limit, self.y_limit).detach()
            _, a_prime = q_eval_values.max(1)

            q_target_values = self.target_net(b_s_, self.x_limit, self.y_limit).detach()
            q_target_s_a_prime = q_target_values.gather(1, a_prime.unsqueeze(1))
            q_target_s_a_prime = q_target_s_a_prime.squeeze()
        
        else:
            q_eval = self.eval_net(b_s).gather(1, b_a)

            q_eval_values = self.eval_net(b_s_).detach()
            _, a_prime = q_eval_values.max(1)

            q_target_values = self.target_net(b_s_).detach()
            q_target_s_a_prime = q_target_values.gather(1, a_prime.unsqueeze(1))
            q_target_s_a_prime = q_target_s_a_prime.squeeze()

        q_target = b_r.reshape(self.batch_size, 1) + self.gamma * q_target_s_a_prime.view(self.batch_size, 1) * (1 - is_done.reshape(self.batch_size, 1))

        loss = self.loss_func(q_eval, q_target)

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