import torch
import torch.nn.functional as F
from differential import Differential
import math


class AlienCraftWorld(torch.nn.Module):
    RELOAD_BUFFER_NAMES = (
        "grid",
        "grid_velocity",
        "fields",
        "field_velocity",
        "field_vel",
        "positions",
        "grads",
        "distance_kernel",
        "properties",
        "fields_affected_by_types",
        "batch_idx",
        "agent_position",
        "agent_inventory",
        "sprite_positions",
        "prop_matrix",
        "tech_tree_progress",
        "nb_offsets",
        "sprites",
        "colour_palette",
        "actuators",
        "place_type_actuator",
    )

    def __init__(
        self,
        batch_size: int,
        width: int,
        height: int,
        num_types: int,
        num_common_types: int,
        num_sparse_types: int,
        num_properties: int,
        num_fields: int,
        sprite_resolution: int,
        visual_field_size: int,
        device: str,
        driven_fields: bool = False,
        actuator_random: bool = False,
    ):
        super().__init__()
        # each cell has a type
        self.width = width
        self.height = height
        self.device = device
        # for now
        assert height == width, "height must be equal to width"
        self.num_types = num_types
        self.num_properties = num_properties
        self.num_fields = num_fields
        self.batch_size = batch_size
        self.visual_field_size = visual_field_size
        self.driven_fields = driven_fields
        self.actuator_random = actuator_random
        self.grid = torch.zeros(batch_size, width, height, device=self.device)
        self.grid_velocity = torch.zeros(
            batch_size, width, height, 2, device=self.device
        )
        # forces are mediated by scalar vfields
        self.fields = torch.zeros(
            batch_size, num_fields, width, height, device=self.device
        )
        self.field_velocity = torch.zeros(
            batch_size, num_fields, width, height, device=self.device
        )

        self.field_matter_affinity = torch.nn.Embedding(num_types, num_fields).to(
            self.device
        )
        num_active_types = num_types - 1
        assert num_active_types > 0, "num_active_types must be greater than 0"
        active_type_ids = torch.randperm(num_types)[:num_active_types].to(self.device)
        with torch.no_grad():
            self.field_matter_affinity.weight.zero_()
            self.field_matter_affinity.weight[active_type_ids] = torch.empty_like(
                self.field_matter_affinity.weight[active_type_ids]
            ).uniform_(-1.0, 1.0)

        self.num_common_types = num_common_types
        self.num_sparse_types = num_sparse_types

        assert (
            self.num_common_types + self.num_sparse_types <= self.num_types
        ), "num_common_types + num_sparse_types must be less than num_types"

        self.differential = Differential(num_fields).to(self.device)

        self.dt = 0.1
        self.field_vel = torch.randint(
            1, 3, (self.batch_size, self.num_fields), device=self.device
        ).float()  # b, num_fields

        self.damping = 0.90
        self.field_damping = 1.0

        ii, jj = torch.meshgrid(
            torch.arange(self.width, device=self.device),
            torch.arange(self.height, device=self.device),
            indexing="ij",
        )

        self.positions = torch.stack([ii, jj], dim=-1).unsqueeze(0)  #

        self.grads = self.build_grads()

        ii, jj = torch.meshgrid(
            torch.arange(7, device=self.device),
            torch.arange(7, device=self.device),
            indexing="ij",
        )
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
        self.properties = torch.zeros(
            self.batch_size, self.num_types, self.num_properties, device=self.device
        )

        self.properties.uniform_(-1.0, 1.0)

        # each type affects one field
        self.fields_affected_by_types = torch.randint(
            0, self.num_fields, (self.batch_size, self.num_types), device=self.device
        )  # b, num_types

        self.mass_scale = 5.0
        self.batch_idx = torch.arange(self.batch_size, device=self.device)

        self.agent_position = torch.randint(
            0, self.width, (self.batch_size, 2), device=self.device
        ).long()  # b, 2
        # inventory stores object counts for each type
        self.agent_inventory = torch.zeros(
            self.batch_size, self.num_types, device=self.device
        ).long()

        self.sprite_resolution = sprite_resolution
        ii_sprite, jj_sprite = torch.meshgrid(
            torch.arange(self.sprite_resolution, device=self.device),
            torch.arange(self.sprite_resolution, device=self.device),
            indexing="ij",
        )
        self.sprite_positions = torch.stack([ii_sprite, jj_sprite], dim=-1).unsqueeze(
            0
        )  # b, sprite_resolution, sprite_resolution, 2

        self.prop_matrix = torch.eye(
            self.num_properties, device=self.device
        ) + 1e-2 * torch.randn(
            self.batch_size,
            self.num_properties,
            self.num_properties,
            device=self.device,
        )

        # measures progress down the tech tree, true if the type has been discovered
        self.tech_tree_progress = torch.zeros(
            self.batch_size, self.num_types, device=self.device
        ).bool()

        assert (
            self.visual_field_size % self.sprite_resolution == 0
        ), "visual_field_size must be divisible by sprite_resolution"
        
        nb_start = -self.visual_field_size // 2 + self.sprite_resolution // 2
        nb_end = nb_start + self.visual_field_size // self.sprite_resolution

        ii_nb, jj_nb = torch.meshgrid(
            torch.arange(nb_start, nb_end, device=self.device),
            torch.arange(nb_start, nb_end, device=self.device),
            indexing="ij",
        )
        nb_offsets = torch.stack(
            [ii_nb, jj_nb], dim=-1
        )  # (visual_field_size // sprite_res, visual_field_size // sprite_res, 2)
        self.register_buffer("nb_offsets", nb_offsets, persistent=False)

        self.init_sprites()
        self.create_colour_palette()
        self.create_actuators()
        self.seed_universe()

        self.initial_grid_copy = self.grid.detach().clone()
        self.initial_agent_position = self.agent_position.detach().clone()

    def register_all_buffers(self):
        for name in self.RELOAD_BUFFER_NAMES:
            if not hasattr(self, name):
                continue

            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                continue

            value = value.detach().clone().contiguous()
            if name in self._buffers:
                self._buffers[name] = value
                self._non_persistent_buffers_set.discard(name)
                continue

            if name in self.__dict__:
                del self.__dict__[name]
            self.register_buffer(name, value, persistent=True)

        return self

    def create_actuators(self):
        if not self.actuator_random:
            self.actuators = (
                torch.eye(2, device=self.device)
                .unsqueeze(0)
                .expand(self.batch_size, -1, -1)
            )
            self.place_type_actuator = (
                torch.eye(self.num_types, device=self.device)
                .unsqueeze(0)
                .expand(self.batch_size, -1, -1)
            )
            return

        self.actuators = torch.randn(self.batch_size, 2, 2, device=self.device)
        self.place_type_actuator = torch.randn(
            self.batch_size, self.num_types, self.num_types, device=self.device
        )
        # make positive semidefinite
        self.actuators = self.actuators @ self.actuators.transpose(
            1, 2
        ) + 1e-4 * torch.eye(2, device=self.device)
        self.place_type_actuator = (
            self.place_type_actuator @ self.place_type_actuator.transpose(1, 2)
            + 1e-4 * torch.eye(self.num_types, device=self.device)
        )

        # test for invertibility, will throw runtime error if not invertible
        self.actuators.inverse()
        self.place_type_actuator.inverse()

    def create_colour_palette(self):
        colour_palette = (
            torch.arange(self.num_types, device=self.device) * 0.61803398875
        ) % 1.0  # num_types
        colour_palette = colour_palette.unsqueeze(0).expand(
            self.batch_size, -1
        )  # b, num_types
        # permute the colour palette
        perm = torch.randperm(self.num_types)
        colour_palette = colour_palette[:, perm]  # b, num_types

        colour_palette = colour_palette.unsqueeze(-1)  # b, num_types, 1
        saturation_contrast = (
            torch.tensor([0.48, 0.68], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.batch_size, self.num_types, -1)
        )  # b, num_types, 2

        hsv = torch.cat(
            [colour_palette, saturation_contrast], dim=-1
        )  # b, num_types, 3
        self.colour_palette = self.hsv_to_rgb(hsv)  # b, num_types, 3
        self.colour_palette = (
            (self.colour_palette * 255).floor().long()
        )  # b, num_types, 3

    def hsv_to_rgb(self, hsv: torch.Tensor):
        h, s, v = hsv.unbind(-1)
        i = torch.floor(h * 6).long()
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        choices = torch.stack(
            [
                torch.stack([v, t, p], dim=-1),
                torch.stack([q, v, p], dim=-1),
                torch.stack([p, v, t], dim=-1),
                torch.stack([p, q, v], dim=-1),
                torch.stack([t, p, v], dim=-1),
                torch.stack([v, p, q], dim=-1),
            ],
            dim=-2,
        )
        idx = (i % 6).unsqueeze(-1).unsqueeze(-1).expand(*i.shape, 1, 3)
        return choices.gather(-2, idx).squeeze(-2)

    def init_sprites(self):
        perlin = self.perlin(
            scale=0.5,
            positions=self.sprite_positions,
            seed=torch.randint(
                0, 1000000, (self.batch_size * self.num_types,), device=self.device
            ),
        )

        perlin = perlin.reshape(
            self.batch_size,
            self.num_types,
            self.sprite_resolution,
            self.sprite_resolution,
        )  # b, num_types, sprite_resolution, sprite_resolution

        # normalise between 0 and 1
        amin = perlin.amin(dim=(-2, -1), keepdim=True)
        amax = perlin.amax(dim=(-2, -1), keepdim=True)
        perlin = (perlin - amin) / (amax - amin + 1e-8)

        self.sprites = perlin

    def craft(
        self,
        mat1_type: torch.Tensor,  # b,
        mat2_type: torch.Tensor,  # b,
    ):
        # you craft with two types to get a new type
        m = torch.max(mat1_type, mat2_type)
        n = torch.min(mat1_type, mat2_type)
        new_type = m * (m + 1) // 2 + n + 1 + self.num_common_types
        mat1_props = self.properties[self.batch_idx, mat1_type]
        mat2_props = self.properties[self.batch_idx, mat2_type]
        inherited_props: torch.Tensor = self.prop_matrix @ mat1_props.unsqueeze(
            -1
        ) + self.prop_matrix @ mat2_props.unsqueeze(-1)
        inherited_props = inherited_props.squeeze(-1)
        inherited_props = inherited_props / (
            inherited_props.norm(dim=-1, keepdim=True) + 1e-8
        )
        return torch.where(new_type < self.num_types, new_type, -1), inherited_props

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
        ).to(
            self.device
        )  # 8, 2

    def quintic_smoothing(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(self, a, b, t):
        return a + t * (b - a)

    def perlin(
        self,
        scale: float,
        positions: torch.Tensor,
        seed: torch.Tensor,  # b,
    ):

        positions = positions * scale  # b, h, w, 2

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
            seed = torch.randint(0, 1000000, (self.batch_size,), device=self.device)
            perlin = self.perlin(
                scale=0.05 / (0.5 * math.sqrt(i + 1)),
                positions=self.positions,
                seed=seed,
            )
            p_min = perlin.amin(dim=(-2, -1), keepdim=True)
            p_max = perlin.amax(dim=(-2, -1), keepdim=True)
            perlin = (perlin - p_min) / (p_max - p_min + 1e-8)
            fields.append(perlin)

        fields = torch.stack(fields, dim=-1)  # b, h, w, num_types
        grid = fields.argmax(dim=-1)  # b, h, w

        sparse_mask = (
            torch.rand(self.batch_size, self.width, self.height, device=self.device)
            < 0.01
        )
        rare_types = torch.randint(
            self.num_common_types,
            self.num_common_types + self.num_sparse_types,
            (self.batch_size, self.width, self.height),
            device=self.device,
        )  # b, h, w
        grid[sparse_mask] = rare_types[sparse_mask]

        self.grid = grid

    def occupany_function(self, grid: torch.Tensor, positions: torch.Tensor):
        is_occupied = (
            grid[
                self.batch_idx.unsqueeze(-1).unsqueeze(-1),
                positions[..., 0],
                positions[..., 1],
            ]
            > 0
        )  # b, h, w
        is_occupied = is_occupied.unsqueeze(-1)  # b, h, w, 1
        return is_occupied

    def move(self, grid: torch.Tensor, velocity: torch.Tensor, step: int):
        new_positions = self.positions + velocity * self.dt
        new_positions = (new_positions.floor().long()) % self.width  # b, h, w, 2
        new_grid = torch.zeros_like(grid)

        is_occupied = self.occupany_function(grid, new_positions)

        # move the cells that are not occupied to their new positions
        new_positions = new_positions.where(~is_occupied, self.positions)
        is_able_to_move = (
            grid[
                self.batch_idx.unsqueeze(-1).unsqueeze(-1),
                self.positions[..., 0],
                self.positions[..., 1],
            ]
            > 0
        )  # b, h, w
        is_able_to_move = is_able_to_move.unsqueeze(-1)  # b, h, w, 1
        new_positions = new_positions.where(is_able_to_move, self.positions)

        alive = grid > 0

        new_positions = new_positions.where(alive.unsqueeze(-1), self.positions)
        # we only want to write cells into positions such that the positions in the original
        # grid were alive
        b = self.batch_idx.unsqueeze(-1).unsqueeze(-1).expand_as(grid)
        new_grid[
            b[alive], new_positions[..., 0][alive], new_positions[..., 1][alive]
        ] = grid[alive]
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
        delta_fields = self.field_vel.pow(2).unsqueeze(-1).unsqueeze(
            -1
        ) * self.differential.laplacian(
            self.fields
        )  # b, num_fields, h, w
        # compute distance r from source types and populate fields
        # we compute distance from each source type to everywhere else in the grid
        # then the field value is the sum of the fields from all source types at that distance
        # we look at each cell value and see if there are any source types up to radius r from that cell
        # then the force is equal to 1/r since it's 2d
        grid_types = self.grid.unsqueeze(-1)  # b, h, w, 1
        types = (
            torch.arange(self.num_types, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # 1, 1, num_types
        grid_types_masked = grid_types == types
        grid = grid_types_masked.squeeze(1).float()  # b, h, w, num_types
        grid = grid.permute(0, 3, 1, 2)  # b, num_types, h, w

        # so we have a deconstructed map of types in each cell
        # we now compute a convolution of each type with a kernel that captures the distance from each type

        if self.driven_fields:
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

    def actuator_project(self, action: torch.Tensor):
        # action is b, action_dim
        # we feed the continuous action into an actuator that is altered per-universe
        # how do we handle non motion actions? we project through the actuator to
        # get the type of the action. the actions can be:
        # - move in a direction
        # - pick/place an object (which should be described by two separate vectors)
        # - craft an object
        # - noop
        direction = action[..., :2]  # b, 2
        # type of object to place
        place_type = action[..., 2 : 2 + self.num_types]
        direction = self.actuators @ direction.unsqueeze(-1)  # b, 2, 1
        direction = direction.squeeze(-1)  # b, 2
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        actuated_place_type = self.place_type_actuator @ place_type.unsqueeze(
            -1
        )  # b, num_types
        actuated_place_type = actuated_place_type.squeeze(-1)
        argmaxxed_place_type = actuated_place_type.argmax(dim=-1)

        isqrt2 = 1.0 / math.sqrt(2)
        directions_to_project = torch.tensor(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [isqrt2, isqrt2],
                [isqrt2, -isqrt2],
                [-isqrt2, isqrt2],
                [-isqrt2, -isqrt2],
            ],
            device=self.device,
        ).unsqueeze(
            0
        )  # 1, 8, 2

        proj = (direction.unsqueeze(-2) * directions_to_project).sum(dim=-1)  # b, 8
        argmax_direction = proj.argmax(dim=-1)  # b,
        motion_mask = argmax_direction < 4
        craft_mask = argmax_direction == 4
        pick_mask = argmax_direction == 5
        place_mask = argmax_direction == 6
        return (
            motion_mask,
            craft_mask,
            pick_mask,
            place_mask,
            directions_to_project[0, argmax_direction % 4].long(),
            argmaxxed_place_type,
        )

    def apply_action(self, action: torch.Tensor):
        motion_mask, craft_mask, pick_mask, place_mask, motion_direction, place_type = (
            self.actuator_project(action)
        )

        place_mask = place_mask & (
            self.agent_inventory[self.batch_idx, place_type] > 0
        )  # b,

        self.agent_position = self.agent_position.where(
            ~motion_mask.unsqueeze(-1),
            (self.agent_position + motion_direction) % self.width,
        )

        type_at_position = self.grid[
            self.batch_idx, self.agent_position[..., 0], self.agent_position[..., 1]
        ]
        pick_mask = pick_mask & (type_at_position > 0)

        self.agent_inventory[self.batch_idx, type_at_position] = torch.where(
            pick_mask,
            self.agent_inventory[self.batch_idx, type_at_position] + 1,
            self.agent_inventory[self.batch_idx, type_at_position],
        )

        self.grid[
            self.batch_idx, self.agent_position[..., 0], self.agent_position[..., 1]
        ] = torch.where(pick_mask, 0, type_at_position)

        # place the type ahead of the agent
        new_grid = self.grid.clone()

        new_grid[
            self.batch_idx,
            (self.agent_position[..., 0] + 1) % self.width,
            self.agent_position[..., 1],
        ] = place_type
        self.grid = torch.where(
            place_mask.unsqueeze(-1).unsqueeze(-1), new_grid, self.grid
        )

        # remove the type from the agent's inventory
        self.agent_inventory[self.batch_idx, place_type] = torch.where(
            place_mask,
            self.agent_inventory[self.batch_idx, place_type] - 1,
            self.agent_inventory[self.batch_idx, place_type],
        )

        # for crafting we look at types at either side of the agent's position
        left_type = self.grid[
            self.batch_idx,
            (self.agent_position[..., 0] - 1) % self.width,
            self.agent_position[..., 1],
        ]
        right_type = self.grid[
            self.batch_idx,
            (self.agent_position[..., 0] + 1) % self.width,
            self.agent_position[..., 1],
        ]
        # ok so we replace the types at left and right with empty space
        # and add the new type to inventory
        new_grid = self.grid.clone()
        crafted_type, crafted_type_properties = self.craft(left_type, right_type)

        new_grid[
            self.batch_idx,
            (self.agent_position[..., 0] - 1) % self.width,
            self.agent_position[..., 1],
        ] = torch.where(crafted_type == -1, left_type, 0)
        new_grid[
            self.batch_idx,
            (self.agent_position[..., 0] + 1) % self.width,
            self.agent_position[..., 1],
        ] = torch.where(crafted_type == -1, right_type, 0)

        already_exists = self.tech_tree_progress[self.batch_idx, crafted_type]

        # condition for marking the type as discovered in the tech tree and updating properties
        cond = (~already_exists) & (crafted_type != -1) & craft_mask
        self.properties[self.batch_idx, crafted_type] = torch.where(
            cond.unsqueeze(-1),
            crafted_type_properties,
            self.properties[self.batch_idx, crafted_type],
        )
        self.tech_tree_progress[self.batch_idx, crafted_type] = (
            cond | self.tech_tree_progress[self.batch_idx, crafted_type]
        )

        new_grid[
            self.batch_idx, self.agent_position[..., 0], self.agent_position[..., 1]
        ] = crafted_type

        new_grid = torch.where(
            craft_mask.unsqueeze(-1).unsqueeze(-1), new_grid, self.grid
        )

        self.grid = new_grid.where(new_grid != -1, self.grid)

    def render(self, agent_view=False):
        # render the universe with the sprites
        if agent_view:
            nb_offsets = self.nb_offsets.unsqueeze(0) + self.agent_position.unsqueeze(
                1
            ).unsqueeze(
                1
            )  # (b, visual_field_size, visual_field_size, 2)
            nb_offsets = nb_offsets % self.width
            rows = nb_offsets[..., 0]
            cols = nb_offsets[..., 1]
            grid = self.grid[
                self.batch_idx.unsqueeze(-1).unsqueeze(-1), rows, cols
            ]  # (b, visual_field_size, visual_field_size)
        else:
            grid = self.grid  # (b, h, w)

        sprite_grid = self.sprites[
            self.batch_idx.unsqueeze(-1).unsqueeze(-1), grid
        ]  # b, h, w, sprite_resolution, sprite_resolution

        sprite_grid = sprite_grid.permute(
            0, 1, 3, 2, 4
        )  # b, h, sprite_resolution, w, sprite_resolution

        sprite_grid = sprite_grid.reshape(
            self.batch_size,
            grid.shape[-2] * self.sprite_resolution,
            grid.shape[-1] * self.sprite_resolution,
        )  # b, h * sprite_resolution, w * sprite_resolution

        return sprite_grid, grid

    def get_obs_for_agent(self, agent_view=False, normalise=False):
        rendered_grid, grid = self.render(
            agent_view
        )  # b, h * sprite_resolution, w * sprite_resolution

        rgb_grid = self.colour_palette[
            self.batch_idx.unsqueeze(-1).unsqueeze(-1), grid
        ]  # b, h, w, 3

        rgb_grid = rgb_grid.unsqueeze(2).unsqueeze(4)  # b, h, 1, w, 1, 3
        rgb_grid = rgb_grid.expand(
            -1, -1, self.sprite_resolution, -1, self.sprite_resolution, -1
        )  # b, h, sprite_resolution, w, sprite_resolution, 3

        rgb_grid = rgb_grid.reshape(
            self.batch_size,
            grid.shape[-2] * self.sprite_resolution,
            grid.shape[-1] * self.sprite_resolution,
            3,
        )  # b, h * sprite_resolution, w * sprite_resolution, 3

        noise = (
            0.45 + 0.75 * rendered_grid
        )  # b, h * sprite_resolution, w * sprite_resolution
        noise = noise.unsqueeze(
            -1
        )  # b, h * sprite_resolution, w * sprite_resolution, 1
        final_colour = (
            (rgb_grid * noise).clamp(0, 255).floor().long()
        )  # b, h * sprite_resolution, w * sprite_resolution, 3

        if normalise:
            # normalise betwee 0 and 1
            final_colour = final_colour / 255.0

        return final_colour

    def step(self, step: int, action=None):
        if action is not None:
            self.apply_action(action)
        self.fields = self.step_fields()
        self.grid = self.step_grid(step)

    def reset_velocities(self):
        self.grid_velocity = torch.zeros(
            self.batch_size, self.width, self.height, 2, device=self.device
        )
        self.field_velocity = torch.zeros(
            self.batch_size,
            self.num_fields,
            self.width,
            self.height,
            device=self.device,
        )

    def reset(self):
        with torch.no_grad():
            self.grid = torch.zeros(
                self.batch_size, self.width, self.height, device=self.device
            )
            self.fields = torch.zeros(
                self.batch_size,
                self.num_fields,
                self.width,
                self.height,
                device=self.device,
            )
            self.reset_velocities()
            self.tech_tree_progress = torch.zeros(
                self.batch_size, self.num_types, device=self.device
            ).bool()
            self.field_matter_affinity.weight.zero_()
            self.properties = torch.zeros(
                self.batch_size, self.num_types, self.num_properties, device=self.device
            )
            self.properties.uniform_(-1.0, 1.0)
            self.prop_matrix = torch.eye(
                self.num_properties, device=self.device
            ) + 1e-2 * torch.randn(
                self.batch_size,
                self.num_properties,
                self.num_properties,
                device=self.device,
            )

            num_active_types = self.num_types // 2
            assert num_active_types > 0, "num_active_types must be greater than 0"
            active_type_ids = torch.randperm(self.num_types)[:num_active_types].to(
                self.device
            )
            self.field_matter_affinity.weight[active_type_ids] = torch.empty_like(
                self.field_matter_affinity.weight[active_type_ids]
            ).uniform_(-1.0, 1.0)
            self.fields_affected_by_types = torch.randint(
                0,
                self.num_fields,
                (self.batch_size, self.num_types),
                device=self.device,
            )  # b, num_types
            self.agent_position = torch.randint(
                0, self.width, (self.batch_size, 2), device=self.device
            ).long()
            self.agent_inventory = torch.zeros(
                self.batch_size, self.num_types, device=self.device
            ).long()

            self.init_sprites()
            self.create_colour_palette()
            self.create_actuators()
            self.seed_universe()


if __name__ == "__main__":
    torch.inference_mode()
    universe = AlienCraftWorld(
        batch_size=2,
        width=10,
        height=10,
        num_types=6,
        visual_field_size=3,
        num_properties=10,
        num_fields=3,
        num_common_types=2,
        num_sparse_types=2,
        sprite_resolution=4,
    )  # create a universe
    for step in range(1):
        action = torch.randn(1, 2 + universe.num_types)
        universe.step(step, action)
    obs = universe.get_obs_for_agent(agent_view=True)
    print(obs.shape)
