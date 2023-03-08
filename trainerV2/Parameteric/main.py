import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

import torch
import torch.nn as nn
from utils import tools

class Policy(nn.Module):
    def __init__(self, num_targets) -> None:
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(num_targets, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)

if __name__ == "__main__":
    args = tools.load_config("configs/config_test.yaml")
    print(args['ARRAY'][0] - 1)