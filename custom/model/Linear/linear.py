import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim = None):
        super().__init__()
        out_dim = out_dim if out_dim else dim

        self.l1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x