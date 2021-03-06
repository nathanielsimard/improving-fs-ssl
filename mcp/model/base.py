import abc

import torch
from torch import nn


class Model(abc.ABC, nn.Module):
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, file_path: str) -> None:
        """Save the model weights to disk."""
        torch.save(self.state_dict(), file_path)

    def load(self, file_path: str, device: torch.device) -> None:
        """Load the model weights from disk."""
        self.load_state_dict(torch.load(file_path, map_location=device))

    def freeze_weights(self):
        freeze_weights(self)

    def defreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True


def freeze_weights(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False
