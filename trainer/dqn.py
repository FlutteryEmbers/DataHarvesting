from turtle import done, position
from unittest import result
import torch
import torch.nn as nn
import numpy as np
from nets.CNN import CNN
from collections import namedtuple, deque
import random

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def append(self, experience):
        """Save a transition"""
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(object):
    def __init__(self, h, w, outputs, env) -> None:
        self.eval_net, self.target_net = CNN(h, w, outputs), CNN(h, w, outputs)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = ExperienceBuffer(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters, lr=LR)
        self.loss_func = nn.MSELoss()
        self.env = env

    def choose_action(self, state, position):
        state = torch.FloatTensor(np.array(state)).to(device)
        state = torch.unsqueeze(state, dim=0)

        if np.random.uniform() < EPSILON:
            q_value = self.eval_net(state)
            _, action_value = torch.max(q_value, dim=1)
            action = int(action_value.item())
        else:
            # action = np.random.randint(0 , 5)
            self.env.get_action_space().sample_valid_action(position)
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory.append(Experience(s, a, r, s_, done))
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        batch_samples = self.memory.sample(BATCH_SIZE)
        ''''
        batch_state = []
        for i in range(BATCH_SIZE):
            batch_state.append([1, batch_samples[i].state])
        '''
        batch_state = self.unpack_memory("state", batch_samples)
        batch_state = torch.FloatTensor(batch_state)

        batch_action = self.unpack_memory("action", batch_samples)
        batch_action = torch.LongTensor(batch_action)

        batch_reward = self.unpack_memory("reward", batch_samples)
        batch_reward = torch.FloatTensor(batch_reward)

        batch_state_ = self.unpack_memory("next_state", batch_samples)
        batch_state_ = torch.FloatTensor(batch_state_)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_state_).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()                                     
        loss.backward()                                                 
        self.optimizer.step()   

    def unpack_memory(self, name, batch_samples):
        result = []
        for i in range(BATCH_SIZE):
            result.append([i, batch_samples[i][name]])
        return result