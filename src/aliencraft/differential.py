# differential operators on tensor fields
import torch
import torch.nn as nn
import torch.nn.functional as F


class Differential(nn.Module):
    def __init__(self, num_channels: int):
        super(Differential, self).__init__()
        grad_x = (
            torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(num_channels, 1, 1, 1)
        )

        grad_y = (
            torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(num_channels, 1, 1, 1)
        )

        self.num_channels = num_channels

        self.register_buffer("grad_x", grad_x)
        self.register_buffer("grad_y", grad_y)

    def grad(self, field):
        grad_x = F.conv2d(field, self.grad_x, padding=1, groups=self.num_channels)
        grad_y = F.conv2d(field, self.grad_y, padding=1, groups=self.num_channels)
        return 0.5 * torch.stack([grad_x, grad_y], dim=1)
