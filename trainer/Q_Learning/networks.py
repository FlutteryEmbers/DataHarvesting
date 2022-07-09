import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self, inputs, outputs, name, chkpt_dir='model/q_networks'):
        super(MLP, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.num_checkpoints = 0
        
        self.fc1 = nn.Linear(inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.num_checkpoints += 1
        checkpoint_file = self.checkpoint_file + '_' + str(self.num_checkpoints)
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint=None):
        checkpoint_file = self.checkpoint_file + '_' + str(checkpoint)
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(checkpoint_file))

class CNN(nn.Module):
    
    def __init__(self, h, w, outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 2, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        total_linear_input_size = convw * convh * 32
        self.head = nn.Linear(total_linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        total_linear = x.view(x.size(0), -1)

        return self.head(total_linear)