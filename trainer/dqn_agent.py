import random
import numpy as np

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
from estimators import DQN
import time

GAMMA = 0.9
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


Experience = namedtuple('Experience', 
                            field_names=['state', 'action', 'reward', 'done', 'next_state'])


class ExperienceBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def append(self, experience):
        """Save a transition"""
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
    
    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon= 0.0, device= 'cuda'):
        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_sapce.sample()
        else:
            state_action = np.array([self.state], copy=False)
            state_value = torch.tensor(state.a).to(device)
            q_val_vector = net(state_value)
            _, action_value = torch.max(q_val_vector, dim=1)
            action = int(action_value.item())
        
        #env method
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)

        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward
    
    def get_loss(batch, net, target_net, device='cuda'):
        states, actions, rewards, dones, next_states = batch
        states_value = torch.tensor(np.array(states.copy=False)).to(device)
        next_states_value = torch.tensor(np.array(next.states.copy = False)).to(device)
        actions_value = torch.tensor(actions).to(device)
        rewards_value = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        state_action_values = net(states_value).gather(1, actions_value.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = target_net(next_states_value).max(1)[0]
            next_state_values[done_mask] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards_value

        return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    device = torch.device("cuda")
    net = DQN(env.observation_space.shape, env.action_sapce.n).to(device)
    target_net =DQN(env.observation_space.shape, env.action_sapce.n).to(device)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    optimizaer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_reward = []

    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    
    best_m_reward = None
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx/EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_reward.append(reward)
            speed = (frame_idx - ts_frame)/(time.time()-ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_reward[-100:])
            print("%d: done %d games, reward: %0.3f", "eps %f, speed, %.2f f/s"%(frame_idx, len(total_reward),mean_reward, epsilon, speed))
            if best_mean


