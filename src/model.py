import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizontalPolicy(nn.Module):
    def __init__(self, input_dim, K=25):
        super().__init__()
        self.fc = nn.Linear(input_dim, K)

    def forward(self, state):
        x = state.mean(dim=0)
        return F.softmax(self.fc(x), dim=-1)


class VerticalPolicy(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 4)

    def forward(self, state, idx):
        x = state[idx]
        return F.softmax(self.fc(x), dim=-1)