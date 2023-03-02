import torch as T
import torch.nn as nn
import torch.nn.functional as F
from utils import tools
import os

class Net(nn.Module):
    def __init__(self, args, name='adversial', chkpt_dir='cache/model/ppo') -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, args.state_dim)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.name = name
        self.delta = args.delta
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.num_checkpoints = 0
        self.optimizer = T.optim.Adam(self.parameters(), lr=0.001)
        self.to(self.device)

    def forward(self, state):
        state = T.tensor(state, dtype=T.float32).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = T.clamp(self.out(x), max=self.delta, min=0).cpu()
        return out

    def save_checkpoint(self, mode = 'Default'):
        self.num_checkpoints += 1
        tools.save_network_params(mode=mode, checkpoint_file=self.checkpoint_file, 
                                    state_dict=self.state_dict(), num_checkpoints=self.num_checkpoints)

    def load_checkpoint(self, mode = 'Default', chkpt_dir=None):
        if chkpt_dir != None:
            self.checkpoint_file = chkpt_dir + self.name
        state_dict = tools.load_network_params(mode=mode, checkpoint_file=self.checkpoint_file)
        self.load_state_dict(state_dict)
