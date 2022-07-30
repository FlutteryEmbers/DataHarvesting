import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import tools

class MLP(nn.Module):
    def __init__(self, inputs, goals, outputs, name, fc_dim1=256, fc_dim2=256, chkpt_dir='model/her_q_networks'):
        super(MLP, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.num_checkpoints = 0
        
        self.fc1 = nn.Linear(inputs + goals, fc_dim1)
        self.fc2 = nn.Linear(fc_dim1, fc_dim2)
        self.output = nn.Linear(fc_dim2, outputs)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = T.cat((state, goal), 1).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

    def save_checkpoint(self, mode = 'Default'):
        self.num_checkpoints += 1
        tools.save_network_params(mode=mode, checkpoint_file=self.checkpoint_file, 
                                    state_dict=self.state_dict(), num_checkpoints=self.num_checkpoints)

    def load_checkpoint(self, mode = 'Default'):
        state_dict = tools.load_network_params(mode=mode, checkpoint_file=self.checkpoint_file)
        self.load_state_dict(state_dict)