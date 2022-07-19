from operator import mod
from turtle import done, position
from unittest import result
import torch
import torch.nn as nn
import numpy as np
# from nets.CNN import CNN
from .networks import MLP
import random
from utils.buffer import ReplayBuffer

random.seed(10)

BATCH_SIZE = 32
LR = 0.00001
EPSILON = 0.95
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDQN(object):
    def __init__(self, inputs, outputs, env, eval_name = 'ddqn_eval', target_name = 'ddqn_target') -> None:
        self.eval_name = eval_name
        self.target_name = target_name

        self.eval_net, self.target_net = MLP(inputs, outputs, self.eval_name).to(device=device), MLP(inputs, outputs, self.target_name).to(device=device)
        # self.eval_net, self.target_net = CNN(h, w, outputs).to(device=device), CNN(h, w, outputs).to(device=device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = ReplayBuffer(max_size=MEMORY_CAPACITY, input_shape=inputs, n_actions=1)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.env = env

    def choose_action(self, state, position, disable_exploration=False):
        state = torch.FloatTensor(np.array(state)).to(device)
        state = torch.unsqueeze(state, dim=0)
        '''
        global EPSILON
        if self.learn_step_counter % 100000 == 0:
            EPSILON = EPSILON * 0.99
            print('EPSILON = ', EPSILON)
        '''
        if np.random.uniform() > EPSILON and not disable_exploration:
           action = self.env.action_space.sample()
        else:
            q_value = self.eval_net(state)
            _, action_value = torch.max(q_value, dim=1)
            action = int(action_value.item())
            
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory.store_transition(s, a, r, s_, done)
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        state, action, reward, new_state, done = self.memory.sample_buffer(BATCH_SIZE)
        b_s = torch.tensor(state, dtype=torch.float).to(device=device)
        b_a = torch.tensor(action, dtype=torch.long).to(device=device)
        b_r = torch.tensor(reward, dtype=torch.float).to(device=device)
        b_s_ = torch.tensor(new_state, dtype=torch.float).to(device=device)
        is_done = torch.tensor(done, dtype=torch.int).to(device=device)

        q_eval = self.eval_net(b_s).gather(1, b_a)

        q_eval_values = self.eval_net(b_s_).detach()
        _, a_prime = q_eval_values.max(1)

        q_target_values = self.target_net(b_s_).detach()
        q_target_s_a_prime = q_target_values.gather(1, a_prime.unsqueeze(1))
        q_target_s_a_prime = q_target_s_a_prime.squeeze()
        q_target = b_r.reshape(BATCH_SIZE, 1) + GAMMA * q_target_s_a_prime.view(BATCH_SIZE, 1) * (1 - is_done.reshape(BATCH_SIZE, 1))

        loss = self.loss_func(q_eval, q_target)

        # loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()                                     
        loss.backward()                                                 
        self.optimizer.step()
    
    def save_models(self, mode):
        self.eval_net.save_checkpoint(mode=mode)
        self.target_net.save_checkpoint(mode=mode)

    def load_models(self, checkpoints, mode):
        self.eval_net.load_checkpoint(checkpoint=checkpoints, mode=mode)
        self.target_net.load_checkpoint(checkpoint=checkpoints, mode=mode)


    def unpack_memory(self, name, batch_samples):
        result = []
        for i in range(BATCH_SIZE):
            result.append([i, batch_samples[i][name]])
        return result