import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, inputs, outputs, name, chkpt_dir='model/q_networks'):
        super(MLP, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.num_checkpoints = 0
        
        self.fc1 = nn.Linear(inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

    def save_checkpoint(self, mode = 'Default'):
        self.num_checkpoints += 1

        if mode == 'Default':
            print('... saving best model ...')
            checkpoint_file = self.checkpoint_file
        elif mode == 'DR':
            print('... saving DR model ...')
            checkpoint_file = self.checkpoint_file + '_' + mode
        else:
            print('... saving model with ckpt ...')
            checkpoint_file = self.checkpoint_file + '_' + str(self.num_checkpoints)

        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint=None, mode = 'Default'):
        if mode == 'Default':
            print('... loading Best model ...')
            checkpoint_file = self.checkpoint_file 
        elif mode == 'DR':
            print('... loading DR model ...')
            checkpoint_file = self.checkpoint_file + '_' + mode
        else:
            print('... saving model with ckpt ...')
            checkpoint_file = self.checkpoint_file + '_' + str(checkpoint)
        self.load_state_dict(T.load(checkpoint_file))


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

        if mode == 'Default':
            print('... saving best model ...')
            checkpoint_file = self.checkpoint_file
        elif mode == 'DR':
            print('... saving DR model ...')
            checkpoint_file = self.checkpoint_file + '_' + mode
        else:
            print('... saving model with ckpt ...')
            checkpoint_file = self.checkpoint_file + '_' + str(self.num_checkpoints)

        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint=None, mode = 'Default'):
        if mode == 'Default':
            print('... loading Best model ...')
            checkpoint_file = self.checkpoint_file 
        elif mode == 'DR':
            print('... loading DR model ...')
            checkpoint_file = self.checkpoint_file + '_' + mode
        else:
            print('... saving model with ckpt ...')
            checkpoint_file = self.checkpoint_file + '_' + str(checkpoint)
        self.load_state_dict(T.load(checkpoint_file))