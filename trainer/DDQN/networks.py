import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import tools

class MLP(nn.Module):
    def __init__(self, inputs, outputs, name, fc_dim1=256, fc_dim2=256, chkpt_dir='model/q_networks'):
        super(MLP, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.num_checkpoints = 0
        
        self.fc1 = nn.Linear(inputs, fc_dim1)
        self.fc2 = nn.Linear(fc_dim1, fc_dim2)
        self.output = nn.Linear(fc_dim2, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
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

class CNN(nn.Module):
    def __init__(self, h, w, info, outputs, name, chkpt_dir='model/q_networks'):
        super(CNN, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.num_checkpoints = 0

        self.kernel_size = 2
        self.stride = 1

        self.conv1 = nn.Conv2d(1, 16, kernel_size=self.kernel_size, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.kernel_size, stride=self.stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self.kernel_size, stride=self.stride)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = self.kernel_size, stride = self.stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        total_linear_input_size = convw * convh * 32 + info
        self.head = nn.Linear(total_linear_input_size, outputs)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state, h, w):
        x, info = T.split(state, h*w, dim=1)
        # x = T.unsqueeze(x, dim=0)
        # info = T.unsqueeze(info, dim=0)
        x = x.view(-1, h, w).unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        total_linear = T.cat((x, info), dim=1)

        return self.head(total_linear)

    def save_checkpoint(self, mode = 'Default'):
        self.num_checkpoints += 1
        tools.save_network_params(mode=mode, checkpoint_file=self.checkpoint_file, 
                                    state_dict=self.state_dict(), num_checkpoints=self.num_checkpoints)

    def load_checkpoint(self, mode = 'Default'):
        state_dict = tools.load_network_params(mode=mode, checkpoint_file=self.checkpoint_file)
        self.load_state_dict(state_dict)