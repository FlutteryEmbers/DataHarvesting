from turtle import done, position
from unittest import result
import torch
import torch.nn as nn
import numpy as np
from nets.CNN import CNN
from collections import namedtuple, deque
import random

random.seed(10)

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.95
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
        self.eval_net, self.target_net = CNN(h, w, outputs).to(device=device), CNN(h, w, outputs).to(device=device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = ExperienceBuffer(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.env = env

    def choose_action(self, state, position):
        state = torch.FloatTensor(np.array(state)).to(device)
        state = torch.unsqueeze(state, dim=0)
        '''
        global EPSILON
        if self.learn_step_counter % 100000 == 0:
            EPSILON = EPSILON * 0.99
            print('EPSILON = ', EPSILON)
        '''
        if np.random.uniform() < EPSILON:
            q_value = self.eval_net(state)
            _, action_value = torch.max(q_value, dim=1)
            action = int(action_value.item())
        else:
            # action = np.random.randint(0 , 5)
            action = self.env.get_action_space().sample_valid_action(position)
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory.append(Experience(s, a, r, s_, done))
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        batch_samples = self.memory.sample(BATCH_SIZE)
        # WARNING: Might Break
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
        '''
        batch = Experience(*zip(*batch_samples))
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool).to(device=device)
        non_final_next_states = torch.cat([torch.FloatTensor(s_) for s_ in batch.next_state if s_ is not None]).reshape(32, 3, 5, 5).to(device=device)
        # state_batch = torch.cat([torch.FloatTensor(s) for s in batch.state if s is not None])
        state_batch = torch.cat(tuple(torch.FloatTensor(batch.state))).reshape((32,3,5,5)).to(device=device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64).to(device=device)
        reward_batch = torch.FloatTensor(batch.reward).to(device=device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.eval_net(state_batch).gather(1, action_batch.view(-1, 1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = self.loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

        # loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()                                     
        loss.backward()                                                 
        self.optimizer.step()   

    def unpack_memory(self, name, batch_samples):
        
        result = []
        for i in range(BATCH_SIZE):
            result.append([i, batch_samples[i][name]])
        return result