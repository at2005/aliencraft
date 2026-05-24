import torch
import torch.nn.functional as F
from differential import Differential
import math


class Types:
    def __init__(self, batch_size: int, num_properties: int, height: int, width: int):
        self.properties = torch.zeros(batch_size, height, width, num_properties)

    def set_properties(self, properties: torch.Tensor):
        self.properties = properties


class TechTree:
    def __init__(self):
        self.root = None
        # relations describe how types compose with one other
        self.num_materials = 0
        self.num_total_products = 100

        # each entry is composed of two other entries
        self.tech_tree = torch.zeros(self.num_total_products, self.num_total_products)


class Universe:
    def __init__(
        self,
        batch_size: int,
        width: int,
        height: int,
        num_types: int,
        num_properties: int,
        num_fields: int,
    ):
        # each cell has a type
        self.width = width
        self.height = height
        self.num_types = num_types
        self.num_properties = num_properties
        self.num_fields = num_fields
        self.batch_size = batch_size

        self.grid = torch.zeros(batch_size, width, height)
        self.velocity_field = torch.zeros(batch_size, width, height, 2)
        # and each type has a properties tensor
        self.types = Types(batch_size, num_properties, width, height)
        # forces are mediated by scalar vfields
        self.fields = torch.randn(batch_size, num_fields, width, height)

        self.differential = Differential(num_fields)

        self.dt = 0.01
        self.field_vel = 10.0

        self.source_types = [1, 2, 3]
        self.damping = 0.4

        ii, jj = torch.meshgrid(
            torch.arange(self.width), torch.arange(self.height), indexing="ij"
        )

        self.positions = torch.stack([ii, jj], dim=-1).unsqueeze(0)  #

        self.grads = self.build_grads()

    def build_grads(self):
        return torch.tensor(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / math.sqrt(2), -1 / math.sqrt(2)],
                [-1 / math.sqrt(2), 1 / math.sqrt(2)],
                [-1 / math.sqrt(2), -1 / math.sqrt(2)],
            ]
        )  # 8, 2

    def quintic_smoothing(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(self, a, b, t):
        return a + t * (b - a)

    def perlin(
        self,
        scale: float,
        seed: torch.Tensor,  # b,
    ):

        positions = self.positions * scale  # b, h, w, 2

        cell = positions.floor()  # b, h, w, 2
        local = positions - cell

        def hash(
            x: torch.Tensor,  # b, h, w
            y: torch.Tensor,  # b, h, w
        ):
            h = (
                (seed.unsqueeze(-1).unsqueeze(-1) * 2654435761)
                ^ (x.long() * 73856093)
                ^ (y.long() * 19349663)
            )
            h = h ^ (h >> 16)
            h = h * 0x85EBCA6B
            h = h ^ (h >> 13)
            h = h * 0xC2B2AE35
            h = h ^ (h >> 16)
            return (h % self.grads.shape[0]).long()

        x = cell[..., 0]
        y = cell[..., 1]
        grad1: torch.Tensor = self.grads[hash(x, y)]
        grad2: torch.Tensor = self.grads[hash(x + 1, y)]
        grad3: torch.Tensor = self.grads[hash(x, y + 1)]
        grad4: torch.Tensor = self.grads[hash(x + 1, y + 1)]

        dot1 = grad1[..., 0] * local[..., 0] + grad1[..., 1] * local[..., 1]
        dot2 = grad2[..., 0] * (local[..., 0] - 1) + grad2[..., 1] * local[..., 1]
        dot3 = grad3[..., 0] * local[..., 0] + grad3[..., 1] * (local[..., 1] - 1)
        dot4 = grad4[..., 0] * (local[..., 0] - 1) + grad4[..., 1] * (local[..., 1] - 1)

        smoothed_positions = self.quintic_smoothing(local)
        u = smoothed_positions[..., 0]
        v = smoothed_positions[..., 1]

        a = self.lerp(dot1, dot2, u)
        b = self.lerp(dot3, dot4, u)
        c = self.lerp(a, b, v)

        return c

    def seed_universe(self):
        fields = []
        for i in range(self.num_types):
            seed = torch.randint(0, 1000000, (self.batch_size,))
            perlin = self.perlin(scale=0.05 / (0.5 * math.sqrt(i + 1)), seed=seed)
            p_min = perlin.amin(dim=(-2, -1), keepdim=True)
            p_max = perlin.amax(dim=(-2, -1), keepdim=True)
            perlin = (perlin - p_min) / (p_max - p_min + 1e-8)
            fields.append(perlin)

        fields = torch.stack(fields, dim=-1)  # b, h, w, num_types
        grid = fields.argmax(dim=-1)  # b, h, w
        self.grid = grid

    def move(self, grid: torch.Tensor, velocity: torch.Tensor):
        new_positions = self.positions + velocity * self.dt
        new_positions = (
            new_positions.round()
            .clamp(0, self.width - 1)
            .clamp(0, self.height - 1)
            .long()
        )
        new_grid = torch.zeros_like(grid)
        new_grid[:, new_positions[..., 0], new_positions[..., 1]] = grid[
            :, self.positions[..., 0], self.positions[..., 1]
        ]
        return new_grid

    def step_grid(self):
        # compute forces for each cell
        field_force = -self.differential.grad(self.fields)  # b, 2, num_fields, h, w
        field_force = field_force.permute(0, 3, 4, 1, 2)  # b, h, w, 2, num_fields
        field_force = field_force.sum(dim=-1)  # b, h, w, 2
        self.velocity_field = self.velocity_field + field_force * self.dt
        return self.move(self.grid, self.velocity_field)

    def step_fields(self):
        # wave propagation for now
        delta_fields = self.field_vel * self.differential.laplacian(self.fields)

        # sources are applied using curl or div operators
        # we need to gather the types of cells that can act as sources
        # mask = torch.isin(
        #     self.grid, torch.tensor(self.source_types, device=self.grid.device)
        # )
        # # source_term = (self.differential.curl(self.fields) + self.differential.div(self.fields)) * mask
        # source_term = self.differential.laplacian(self.fields) * mask
        return self.fields + delta_fields * self.dt - self.damping * self.fields

    def step(self, action=None):
        self.fields = self.step_fields()
        self.grid = self.step_grid()


if __name__ == "__main__":
    universe = Universe(
        batch_size=1, width=4, height=4, num_types=6, num_properties=10, num_fields=3
    )  # create a universe
    universe.seed_universe()
    print(universe.grid)
