from turtle import position
from estimators.DQN import DQN
from environments.DQN_Environment import DQN_Environment

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initEnvironment():
    board = [[0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]]

    env = DQN_Environment(board=board)
    env.init(startAt=[0,0], arrivalAt=[5,4], data_volume=[2,1,3])
    return env

if __name__ == "__main__":
    env = initEnvironment()
    net = DQN(6, 3, 3, env.get_action_space().n())
    state = env.get_state()
    # action = env.action_space().sample(state.current_position)
    # state = env.step()
    board = torch.tensor(np.array(state.board))
    position = torch.tensor(np.array(state.current_position))
    collected = torch.tensor(np.array(state.data_volumn_collected))
    q_value = net(board, position, collected)
    print(q_value)