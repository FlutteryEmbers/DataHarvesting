import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Adversal(nn.Module):
    def __init__(self, n_state, delta) -> None:
        super(Adversal, self).__init__()
        self.fc1 = nn.Linear(n_state, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_state)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.delta = delta

    def forward(self, state):
        state = T.tensor(state).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x)) * self.delta
