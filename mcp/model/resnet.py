from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(32, 32)
