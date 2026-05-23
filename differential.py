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

        laplacian = (
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(num_channels, 1, 1, 1)
        )

        self.num_channels = num_channels

        self.register_buffer("grad_x", grad_x)
        self.register_buffer("grad_y", grad_y)
        self.register_buffer("laplacian_operator", laplacian)

    def grad(self, field):
        grad_x = F.conv2d(field, self.grad_x, padding=1, groups=self.num_channels)
        grad_y = F.conv2d(field, self.grad_y, padding=1, groups=self.num_channels)
        return 0.5 * torch.stack([grad_x, grad_y], dim=1)

    def laplacian(self, field: torch.Tensor):
        laplacian = F.conv2d(
            field, self.laplacian_operator, padding=1, groups=self.num_channels
        )
        return laplacian

    def div(
        self,
        field: torch.Tensor,  # B, h, w, 2 (vector field)
    ):
        assert field.shape[-1] == 2, "Vector field must have 2 components"
        x_component = field[..., 0].unsqueeze(1)  # B, 1, h, w
        y_component = field[..., 1].unsqueeze(1)  # B, 1, h, w
        grad_x = 0.5 * F.conv2d(x_component, self.grad_x, padding=1)  # B, 1, h, w
        grad_y = 0.5 * F.conv2d(y_component, self.grad_y, padding=1)  # B, 1, h, w
        summed = grad_x + grad_y  # B, 1, h, w
        return summed.squeeze(1)  # B, h, w

    def curl(
        self,
        field: torch.Tensor,  # B, h, w, 2 (vector field)
    ):
        assert field.shape[-1] == 2, "Vector field must have 2 components"
        x_component = field[..., 0].unsqueeze(1)  # B, 1, h, w
        y_component = field[..., 1].unsqueeze(1)  # B, 1, h, w
        grad_y = 0.5 * F.conv2d(
            y_component, self.grad_x, padding=1, groups=self.num_channels
        )  # B, 1, h, w
        grad_x = 0.5 * F.conv2d(
            x_component, self.grad_y, padding=1, groups=self.num_channels
        )  # B, 1, h, w
        summed = grad_y - grad_x  # B, 1, h, w
        return summed.squeeze(1)  # B, h, w
