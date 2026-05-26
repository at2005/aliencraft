import torch
import torch.nn.functional as F
from differential import Differential
import math
import itertools


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
        self.grid_velocity = torch.zeros(batch_size, width, height, 2)
        # forces are mediated by scalar vfields
        self.fields = torch.zeros(batch_size, num_fields, width, height)
        self.field_velocity = torch.zeros(batch_size, num_fields, width, height)

        self.field_matter_affinity = torch.nn.Embedding(num_types, num_fields)
        self.field_matter_affinity.weight.data.uniform_(-1, 1)

        self.num_common_types = 4

        self.differential = Differential(num_fields)

        self.dt = 0.1
        self.field_vel = 1.0

        self.damping = 0.90
        self.field_damping = 0.99

        ii, jj = torch.meshgrid(
            torch.arange(self.width), torch.arange(self.height), indexing="ij"
        )

        self.positions = torch.stack([ii, jj], dim=-1).unsqueeze(0)  #

        self.grads = self.build_grads()

        ii, jj = torch.meshgrid(torch.arange(7), torch.arange(7), indexing="ij")
        positions = torch.stack([ii, jj], dim=-1)  # 7, 7, 2
        distances = (positions[..., 0] - 3) ** 2 + (positions[..., 1] - 3) ** 2
        distances = distances.sqrt()
        inverse_distances = 1.0 / (distances + 1)
        self.distance_kernel = (
            inverse_distances.unsqueeze(0).unsqueeze(0).repeat(self.num_types, 1, 1, 1)
        )  # num_types, 1, 7, 7
        assert self.distance_kernel.shape == (
            self.num_types,
            1,
            7,
            7,
        ), "distance_kernel is not the correct shape"
        self.properties = torch.rand(
            self.batch_size, self.num_types, self.num_properties
        )

        # each type affects one field
        self.fields_affected_by_types = torch.randint(
            0, self.num_fields, (self.batch_size, self.num_types)
        )  # b, num_types

        self.mass_scale = 5.0
        self.margolus_conv = torch.nn.Conv2d(16, 1, kernel_size=2, stride=2)

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
        for i in range(self.num_common_types):
            seed = torch.randint(0, 1000000, (self.batch_size,))
            perlin = self.perlin(scale=0.005 / (0.5 * math.sqrt(i + 1)), seed=seed)
            p_min = perlin.amin(dim=(-2, -1), keepdim=True)
            p_max = perlin.amax(dim=(-2, -1), keepdim=True)
            perlin = (perlin - p_min) / (p_max - p_min + 1e-8)
            fields.append(perlin)

        fields = torch.stack(fields, dim=-1)  # b, h, w, num_types
        grid = fields.argmax(dim=-1)  # b, h, w

        sparse_mask = torch.rand(self.batch_size, self.width, self.height) < 0.01
        rare_types = torch.randint(
            self.num_common_types,
            self.num_types,
            (self.batch_size, self.width, self.height),
        )  # b, h, w
        grid[sparse_mask] = rare_types[sparse_mask]

        self.grid = grid

        I = torch.eye(4)
        perms = torch.tensor(list(itertools.permutations(range(4))))
        self.permutation_matrices = I[perms]

    def block_rule(self, blocks: torch.Tensor):
        blocks = blocks.reshape(
            blocks.shape[0], blocks.shape[1], blocks.shape[2], -1
        )  # b, h//2, w//2, 4
        mod_blocks = blocks.sum(dim=-1) % 24  # b, h//2, w//2
        perm_matrix = self.permutation_matrices[mod_blocks]
        blocks = perm_matrix.float() @ blocks.unsqueeze(-1).float()
        blocks = blocks.squeeze(-1)
        blocks = blocks.view(
            blocks.shape[0], blocks.shape[1], blocks.shape[2], 2, 2
        ).long()
        return blocks

    def move(self, grid: torch.Tensor, velocity: torch.Tensor, step: int):
        new_positions = self.positions + velocity * self.dt
        new_positions = (
            new_positions.round()
            .clamp(0, self.width - 1)
            .clamp(0, self.height - 1)
            .long()
        )
        # how do we choose to move matter?
        # we need to figure out how to handle rigidity
        # so like, cells can push against each other, exert force on each other etc.
        # simplest soln: use ca-like rules to update local grids

        if step % 2 == 0:
            grid = torch.roll(grid, shifts=(1, 1), dims=(-2, -1))

        blocks = grid.reshape(
            -1, grid.shape[-2] // 2, 2, grid.shape[-1] // 2, 2
        )  # b, h//2, 2, w//2, 2
        blocks = blocks.permute(0, 1, 3, 2, 4)  # b, h//2, w//2, 2, 2
        new_blocks: torch.Tensor = self.block_rule(blocks)  # b, h//2, w//2, 2, 2
        new_blocks = new_blocks.permute(0, 1, 3, 2, 4)  # b, h//2, 2, w//2, 2
        new_grid = new_blocks.reshape_as(grid)
        if step % 2 == 0:
            new_grid = torch.roll(new_grid, shifts=(-1, -1), dims=(-2, -1))
        return new_grid

    def step_grid(self, step: int):
        # compute forces for each cell
        # field_matter_affinity is b, num_types, num_fields
        affinity: torch.Tensor = self.field_matter_affinity(
            self.grid
        )  # b, h, w, num_fields
        affinity = affinity.unsqueeze(-2)  # b, h, w, 1, num_fields
        field_force = -self.differential.grad(self.fields)  # b, 2, num_fields, h, w
        field_force = field_force.permute(0, 3, 4, 1, 2)  # b, h, w, 2, num_fields
        field_force_weighted = (field_force * affinity).sum(dim=-1)  # b, h, w, 2
        self.grid_velocity = (
            self.damping * self.grid_velocity + field_force_weighted * self.dt
        )
        return self.move(self.grid, self.grid_velocity, step)

    def step_fields(self):
        # wave propagation for now
        delta_fields = (self.field_vel**2) * self.differential.laplacian(self.fields)
        # compute distance r from source types and populate fields
        # we compute distance from each source type to everywhere else in the grid
        # then the field value is the sum of the fields from all source types at that distance
        # we look at each cell value and see if there are any source types up to radius r from that cell
        # then the force is equal to 1/r since it's 2d
        grid_types = self.grid.unsqueeze(-1)  # b, h, w, 1
        types = (
            torch.arange(self.num_types)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # 1, 1, 1, num_types
        grid_types_masked = grid_types == types
        grid = grid_types_masked.squeeze(1).float()  # b, h, w, num_types
        grid = grid.permute(0, 3, 1, 2)  # b, num_types, h, w

        # so we have a deconstructed map of types in each cell
        # we now compute a convolution of each type with a kernel that captures the distance from each type
        grid = F.conv2d(
            grid, self.distance_kernel, padding=3, groups=self.num_types
        )  # b, num_types, h, w

        mass_equivalent = self.mass_scale * (
            self.properties[..., 0].unsqueeze(-1).unsqueeze(-1)
        )  # b, num_types, 1, 1

        # this tells us the field values if we assume inverse 1/r distance from the type weighted by the mass equivalent of the type
        mass_adjusted_field = grid * mass_equivalent  # b, num_types, h, w
        delta_fields_matter = torch.zeros_like(self.fields)  # b, num_fields, h, w

        field_idx = self.fields_affected_by_types.unsqueeze(-1).unsqueeze(-1)
        field_idx = field_idx.expand(
            -1, -1, self.fields.shape[-2], self.fields.shape[-1]
        )
        delta_fields_matter = delta_fields_matter.scatter_add(
            dim=1, index=field_idx, src=mass_adjusted_field
        )  # b, num_fields, h, w

        total_delta_fields = delta_fields + delta_fields_matter
        self.field_velocity = (
            self.damping * self.field_velocity + total_delta_fields * self.dt
        )
        return self.field_damping * self.fields + self.field_velocity * self.dt

    def step(self, step: int, action=None):
        self.fields = self.step_fields()
        self.grid = self.step_grid(step)
        # print(self.fields)


if __name__ == "__main__":
    universe = Universe(
        batch_size=1, width=10, height=10, num_types=6, num_properties=10, num_fields=3
    )  # create a universe
    universe.seed_universe()
    for step in range(10):
        universe.step(step)
