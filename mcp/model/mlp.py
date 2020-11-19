import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        size_input: int,
        size_hidden: int,
        size_output: int,
        n_hiddens: int,
        dropout: float,
    ):
        super().__init__()
        self.input = nn.Linear(size_input, size_hidden)
        self.output = nn.Linear(size_hidden, size_output)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.hiddens = nn.ModuleList(
            [nn.Linear(size_hidden, size_hidden) for _ in range(n_hiddens)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.activation(x)
        x = self.dropout(x)

        for hidden in self.hiddens:
            x = hidden(x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.output(x)

        return x
