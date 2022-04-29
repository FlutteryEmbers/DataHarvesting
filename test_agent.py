from turtle import position
from estimators.DQN import DQN
from environments.DQN_Environment import DQN_Environment

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initEnvironment():
    board = [[0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]]

    env = DQN_Environment(board=board)
    env.init(startAt=[0,0], arrivalAt=[5,4], data_volume=[2,1,3])
    return env

if __name__ == "__main__":
    env = initEnvironment()
    net = DQN(6, 3, 3, env.get_action_space().n()).to(device)
    state_array = env.get_state()
    # action = env.action_space().sample(state.current_position)
    # state = env.step()
    state = torch.tensor(np.array(state_array)).to(device)
    state = torch.unsqueeze(state, dim=0)
    print(state.size())
    q_value = net(state)
    print(q_value)