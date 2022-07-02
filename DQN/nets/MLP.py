import torch
import torch.nn as nn
import torch.nn.functional as F
# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    
    def __init__(self, inputs, outputs):
        super(MLP, self).__init__()
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