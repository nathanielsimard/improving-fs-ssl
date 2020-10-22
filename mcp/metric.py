import abc

import torch


class Metric(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> float:
        pass


class Accuracy(Metric):
    def __call__(self, x: torch.Tensor, y: torch.Tensor, logit=True) -> float:
        if logit:
            x = torch.argmax(x, dim=-1)

        print(x.device)
        print(y.device)

        return (
            100.0
            * torch.as_tensor(x == y, dtype=torch.float32, device=x.device)
            .mean()
            .item()
        )
