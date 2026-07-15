import torch
import torch.nn.functional as F
import math


def spatial_grad(field):
    # central differences per channel; field (b, c, h, w) -> (b, 2, c, h, w)
    c = field.shape[1]
    kx = field.new_tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    grad_x = F.conv2d(field, kx.expand(c, 1, 3, 3), padding=1, groups=c)
    grad_y = F.conv2d(field, kx.T.expand(c, 1, 3, 3), padding=1, groups=c)
    return 0.5 * torch.stack([grad_x, grad_y], dim=1)


class AlienCraftWorld(torch.nn.Module):
    RELOAD_BUFFER_NAMES = (
        "grid",
        "grid_velocity",
        "fields",
        "positions",
        "grads",
        "distance_kernel",
        "field_matter_affinity",
        "properties",
        "batch_idx",
        "agent_position",
        "agent_inventory",
        "sprite_positions",
        "prop_tensor",
        "sens_tensor",
        "tech_tree_progress",
        "nb_offsets",
        "sprites",
        "colour_palette",
        "colour_projection",
        "field_colour_directions",
        "craft_f",
        "craft_f_act",
        "craft_thresh",
        "nca_w1",
        "nca_w2",
        "nca_b1",
        "nca_b2",
        "nca_alpha",
        "nca_state",
        "field_act1",
        "field_act2",
        "chem_act",
        "sens_act",
        "emission_vec",
        "emission_act",
        "force_tensor",
        "force_act",
        "gate_tensor",
        "gate_act",
        "gate_ema_beta",
        "rms_ema",
        "actuators",
        "pointer_actuator",
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
        self.grid = torch.zeros(batch_size, width, height, device=self.device)
        self.grid_velocity = torch.zeros(
            batch_size, width, height, 2, device=self.device
        )
        # forces are mediated by scalar vfields
        self.fields = torch.zeros(
            batch_size, num_fields, width, height, device=self.device
        )

        self.field_matter_affinity = self._init_field_matter_affinity(num_types)

        self.num_common_types = num_common_types
        self.num_sparse_types = num_sparse_types

        assert (
            self.num_common_types + self.num_sparse_types <= self.num_types
        ), "num_common_types + num_sparse_types must be less than num_types"

        self.dt = 0.1
        self.damping = 0.90

        ii, jj = torch.meshgrid(
            torch.arange(self.width, device=self.device),
            torch.arange(self.height, device=self.device),
            indexing="ij",
        )

        self.positions = torch.stack([ii, jj], dim=-1).unsqueeze(0)  #

        self.grads = self.build_grads()

        self._init_distance_kernel()
        self.properties = torch.zeros(
            self.batch_size, self.num_types, self.num_properties, device=self.device
        )

        self.properties.uniform_(-1.0, 1.0)

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

        self.prop_tensor = self._init_mix_tensor(self.num_properties)
        self.sens_tensor = self._init_mix_tensor(self.num_fields)
        self._init_nca()

        # true if the type has been discovered; naturals start discovered,
        # crafted types claim free slots in order
        self.tech_tree_progress = torch.zeros(
            self.batch_size, self.num_types, device=self.device
        ).bool()
        self.tech_tree_progress[:, 1 : num_common_types + num_sparse_types] = True

        assert (
            self.visual_field_size % self.sprite_resolution == 0
        ), "visual_field_size must be divisible by sprite_resolution"
        
        # window is in cell units, centered on the agent (agent at index n//2)
        n = self.visual_field_size // self.sprite_resolution
        nb_start = -(n // 2)
        nb_end = nb_start + n

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
        self._init_craft_law()
        self.seed_universe()

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

    def _sample_series(self, k=4):
        # bounded sampled nonlinearity: sum_k a_k sin(w_k x + p_k), sum|a| = 1
        a = torch.randn(self.batch_size, k, device=self.device)
        a = a / a.abs().sum(1, keepdim=True).clamp_min(1e-6)
        w = torch.exp(
            torch.empty(self.batch_size, k, device=self.device).uniform_(
                math.log(0.5), math.log(4.0)
            )
        )
        p = torch.rand(self.batch_size, k, device=self.device) * 2 * math.pi
        return torch.stack([a, w, p], dim=1)

    def _series(self, x, act):
        a, w, p = act[:, 0], act[:, 1], act[:, 2]
        shape = [x.shape[0]] + [1] * (x.dim() - 1) + [act.shape[-1]]
        y = torch.sin(w.view(shape) * x.unsqueeze(-1) + p.view(shape))
        return (y * a.view(shape)).sum(-1)

    def _init_nca(self):
        # per-universe random propagation law over visible + hidden channels
        b, f = self.batch_size, self.num_fields
        c = f + 5
        m, cin = 16, c + f
        gain = torch.exp(
            torch.empty(b, 1, 1, 1, 1, device=self.device).uniform_(
                math.log(0.25), math.log(8.0)
            )
        )
        self.nca_w1 = torch.randn(b, m, cin, 3, 3, device=self.device) * gain / (cin * 9) ** 0.5
        self.nca_w2 = torch.randn(b, c, m, 3, 3, device=self.device) * gain / (m * 9) ** 0.5
        self.nca_b1 = torch.randn(b, m, device=self.device) * 0.5
        self.nca_b2 = torch.randn(b, c, device=self.device) * 0.5
        # per-channel EMA rates down to ~200-step integrators: timescale
        # separation is what lets sampled laws express slow modes and
        # relaxation oscillations (tied single alpha provably cannot)
        self.nca_alpha = torch.exp(
            torch.empty(b, c, 1, 1, device=self.device).uniform_(
                math.log(0.005), 0.0
            )
        )
        self.field_act1 = self._sample_series()
        self.field_act2 = self._sample_series()
        self.chem_act = self._sample_series()
        self.sens_act = self._sample_series()
        # sampled emission law: strength = bounded series over the full
        # property vector (no anointed mass coordinate); reciprocity kept —
        # emission still points along the type's affinity
        self.emission_vec = torch.randn(b, self.num_properties, device=self.device)
        self.emission_act = self._sample_series()
        # sampled locomotion law: bounded force from a bilinear coupling of
        # the local field state (gradients + values) with the cell type's
        # affinity and properties; conservation and collisions stay fixed
        # the fan-in root keeps the bilinear form near unit variance, and the
        # gain — how nonlinear the sampled law runs — is itself sampled
        field_state_dim = 3 * f
        coupling_dim = f + self.num_properties
        force_gain = torch.exp(
            torch.empty(b, 1, 1, 1, device=self.device).uniform_(
                math.log(1.25), math.log(20.0)
            )
        )
        self.force_tensor = (
            force_gain
            * torch.randn(b, 2, field_state_dim, coupling_dim, device=self.device)
            / (field_state_dim * coupling_dim) ** 0.5
        )
        self.force_act = self._sample_series()
        self._init_gate()
        self.nca_state = torch.zeros(b, c, self.width, self.height, device=self.device)
        self.rms_ema = torch.zeros(b, device=self.device)

    def _init_gate(self):
        d = 2 * self.num_fields + self.num_properties
        gain = torch.exp(
            torch.empty(self.batch_size, 1, 1, device=self.device).uniform_(
                math.log(2.0), math.log(12.0)
            )
        )
        g = torch.randn(self.batch_size, d, d, device=self.device)
        self.gate_tensor = gain * 0.5 * (g + g.transpose(-1, -2)) / d
        self.gate_act = self._sample_series()
        horizon = torch.exp(
            torch.empty(self.batch_size, device=self.device).uniform_(
                math.log(20.0), math.log(500.0)
            )
        )
        self.gate_ema_beta = 1.0 / horizon

    def _init_distance_kernel(self):
        # radially symmetric distance kernel
        ii, jj = torch.meshgrid(
            torch.arange(7, device=self.device),
            torch.arange(7, device=self.device),
            indexing="ij",
        )
        r = ((ii - 3) ** 2 + (jj - 3) ** 2).float().sqrt()
        r = r / r.max()
        powers = torch.stack([r**n for n in range(5)])
        kernel = torch.einsum(
            "bn,nxy->bxy", torch.randn(self.batch_size, 5, device=self.device), powers
        )
        self.distance_kernel = kernel / kernel.abs().sum((1, 2), keepdim=True).clamp_min(1e-6)

    def _init_mix_tensor(self, dim: int):
        # sample the bilinear map
        gain = torch.exp(
            torch.empty(self.batch_size, 1, 1, 1, device=self.device).uniform_(
                math.log(2.0), math.log(12.0)
            )
        )
        tensor = torch.randn(self.batch_size, dim, dim, dim, device=self.device)
        tensor = 0.5 * (tensor + tensor.transpose(-1, -2))
        return gain * tensor / dim

    def _init_field_matter_affinity(self, num_active_types: int):
        # per-universe random subset of field-coupled types
        assert num_active_types > 0, "num_active_types must be greater than 0"
        active = torch.rand(self.batch_size, self.num_types, device=self.device).argsort(
            dim=-1
        )[:, :num_active_types]
        mask = torch.zeros(
            self.batch_size, self.num_types, dtype=torch.bool, device=self.device
        ).scatter_(1, active, True)
        values = torch.empty(
            self.batch_size, self.num_types, self.num_fields, device=self.device
        ).uniform_(-1.0, 1.0)
        return values * mask.unsqueeze(-1)  # b, num_types, num_fields

    def create_actuators(self):
        self.actuators = torch.randn(self.batch_size, 2, 2, device=self.device)
        # make positive semidefinite
        self.actuators = self.actuators @ self.actuators.transpose(
            1, 2
        ) + 1e-4 * torch.eye(2, device=self.device)

        # test for invertibility, will throw runtime error if not invertible
        self.actuators.inverse()

        # ensure all types (mostly) reachable
        p = self.num_properties
        q = torch.linalg.qr(torch.randn(self.batch_size, p, p, device=self.device)).Q
        s = torch.exp(
            torch.empty(self.batch_size, p, device=self.device).uniform_(
                math.log(0.5), math.log(2.0)
            )
        )
        self.pointer_actuator = q @ (s.unsqueeze(-1) * q.transpose(1, 2))

    def create_colour_palette(self):
        # colour = orthonormal projection of properties; a rotation at P=3,
        # so the full property state is readable from pixels
        assert self.num_properties >= 3, "colour projection needs >= 3 properties"
        proj = torch.randn(
            self.batch_size, self.num_properties, 3, device=self.device
        )
        self.colour_projection = torch.linalg.qr(proj).Q
        self.refresh_colour_palette()

        # unit RGB shimmer direction per field
        directions = torch.randn(
            self.batch_size, self.num_fields, 3, device=self.device
        )
        self.field_colour_directions = directions / (
            directions.norm(dim=-1, keepdim=True) + 1e-8
        )

    def refresh_colour_palette(self):
        # unit tanh gain: invertible through 8-bit colour without blowup
        rgb = 0.5 + 0.35 * torch.tanh(
            self.properties @ self.colour_projection
        )  # b, num_types, 3
        self.colour_palette = (rgb * 255).floor().long()

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

    def _init_craft_law(self):
        # craftability is a sampled symmetric law over the parents' observable properties
        assert (
            self.num_common_types + self.num_sparse_types >= 2
        ), "need at least one non-empty natural type"
        p = self.num_properties
        f = torch.randn(self.batch_size, p, p, device=self.device)
        self.craft_f = 0.5 * (f + f.transpose(-1, -2))
        self.craft_f_act = self._sample_series()
        self.craft_eps = 0.2
        density = torch.empty(self.batch_size, device=self.device).uniform_(0.25, 0.40)
        pa = torch.empty(self.batch_size, 4096, p, device=self.device).uniform_(-1, 1)
        pb = torch.empty(self.batch_size, 4096, p, device=self.device).uniform_(-1, 1)
        scores = self._series(
            torch.einsum("bjk,bnj,bnk->bn", self.craft_f, pa, pb), self.craft_f_act
        ).sort(dim=1).values
        idx = ((1.0 - density) * (scores.shape[1] - 1)).long()
        self.craft_thresh = scores[self.batch_idx, idx]

    def craft(
        self,
        mat1_type: torch.Tensor,  # b,
        mat2_type: torch.Tensor,  # b,
    ):
        # the craft law says whether the pair reacts; the chemistry law says
        # what comes out: a known type if the result lands within craft_eps
        # of one, else the next free slot (-1 if inert or slots exhausted)
        mat1_props = self.properties[self.batch_idx, mat1_type]
        mat2_props = self.properties[self.batch_idx, mat2_type]
        reacts = (
            self._series(
                torch.einsum("bjk,bj,bk->b", self.craft_f, mat1_props, mat2_props),
                self.craft_f_act,
            )
            > self.craft_thresh
        ) & (mat1_type > 0) & (mat2_type > 0)
        inherited_props = self._series(
            torch.einsum("bijk,bj,bk->bi", self.prop_tensor, mat1_props, mat2_props),
            self.chem_act,
        )
        mat1_sens = self.field_matter_affinity[self.batch_idx, mat1_type]
        mat2_sens = self.field_matter_affinity[self.batch_idx, mat2_type]
        inherited_sens = self._series(
            torch.einsum("bijk,bj,bk->bi", self.sens_tensor, mat1_sens, mat2_sens),
            self.sens_act,
        )
        dist = (self.properties - inherited_props.unsqueeze(1)).norm(dim=-1)
        dist = dist.masked_fill(~self.tech_tree_progress, torch.inf)
        match_dist, match_idx = dist.min(dim=1)
        matched = match_dist < self.craft_eps
        num_natural = self.num_common_types + self.num_sparse_types
        free = ~self.tech_tree_progress[:, num_natural:]
        new_type = torch.where(matched, match_idx, num_natural + free.long().argmax(1))
        new_type = torch.where(
            reacts & (matched | free.any(1)), new_type, torch.full_like(new_type, -1)
        )
        return new_type, inherited_props, inherited_sens

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
        # layout statistics reroll with the layout: terrain correlation
        # length and sparse density are sampled per reset, crafter-style
        scale_mult = torch.exp(
            torch.empty(self.batch_size, 1, 1, 1, device=self.device).uniform_(
                math.log(0.4), math.log(2.5)
            )
        )
        fields = []
        for i in range(self.num_common_types):
            seed = torch.randint(0, 1000000, (self.batch_size,), device=self.device)
            perlin = self.perlin(
                scale=scale_mult * (0.05 / (0.5 * math.sqrt(i + 1))),
                positions=self.positions,
                seed=seed,
            )
            p_min = perlin.amin(dim=(-2, -1), keepdim=True)
            p_max = perlin.amax(dim=(-2, -1), keepdim=True)
            perlin = (perlin - p_min) / (p_max - p_min + 1e-8)
            fields.append(perlin)

        fields = torch.stack(fields, dim=-1)  # b, h, w, num_types
        grid = fields.argmax(dim=-1)  # b, h, w

        sparse_density = torch.exp(
            torch.empty(self.batch_size, 1, 1, device=self.device).uniform_(
                math.log(0.003), math.log(0.03)
            )
        )
        sparse_mask = (
            torch.rand(self.batch_size, self.width, self.height, device=self.device)
            < sparse_density
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
        # dt is already integrated into velocity; round keeps sub-threshold
        # matter in place symmetrically (floor moved anything with the
        # slightest negative velocity a full cell per step)
        new_positions = self.positions + velocity
        new_positions = (new_positions.round().long()) % self.width  # b, h, w, 2
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

        # contested targets: nobody moves, so matter is conserved
        b = self.batch_idx.unsqueeze(-1).unsqueeze(-1).expand_as(grid)
        claims = torch.zeros_like(grid)
        claims.index_put_(
            (b[alive], new_positions[..., 0][alive], new_positions[..., 1][alive]),
            torch.ones_like(grid[alive]),
            accumulate=True,
        )
        contested = claims[b, new_positions[..., 0], new_positions[..., 1]] > 1
        new_positions = new_positions.where(~contested.unsqueeze(-1), self.positions)

        # we only want to write cells into positions such that the positions in the original
        # grid were alive
        new_grid[
            b[alive], new_positions[..., 0][alive], new_positions[..., 1][alive]
        ] = grid[alive]
        return new_grid

    def step_grid(self, step: int):
        # sampled locomotion law: force = bounded series over a bilinear
        # coupling of the local field state with the cell type's affinity
        # and properties (gradient descent is one point in this family)
        affinity = self.field_matter_affinity[
            self.batch_idx.view(-1, 1, 1), self.grid
        ]  # b, h, w, num_fields
        props = self.properties[self.batch_idx.view(-1, 1, 1), self.grid]
        grad = spatial_grad(self.fields)  # b, 2, num_fields, h, w
        field_state = torch.cat(
            [
                grad.reshape(
                    self.batch_size, 2 * self.num_fields, self.width, self.height
                ),
                self.fields,
            ],
            dim=1,
        )  # b, 3F, h, w
        coupling = torch.cat([affinity, props], dim=-1)  # b, h, w, F + P
        q = torch.einsum(
            "bjcd,bchw,bhwd->bjhw", self.force_tensor, field_state, coupling
        )
        force = self._series(q, self.force_act).permute(0, 2, 3, 1)  # b, h, w, 2

        self.grid_velocity = self.damping * self.grid_velocity + force * self.dt
        return self.move(self.grid, self.grid_velocity, step)

    def _nca_conv(self, x, w, bias):
        b = x.shape[0]
        out = F.conv2d(
            x.reshape(1, -1, *x.shape[2:]),
            w.reshape(-1, w.shape[2], 3, 3),
            padding=1,
            groups=b,
        ).reshape(b, -1, *x.shape[2:])
        return out + bias.view(b, -1, 1, 1)

    def step_fields(self, step: int):
        # reciprocity: matter sources fields along the same affinity it feels
        # them through; the sampled NCA is the propagation law
        grid = (
            (self.grid.unsqueeze(-1) == torch.arange(self.num_types, device=self.device))
            .float()
            .permute(0, 3, 1, 2)
        )  # b, num_types, h, w
        if self.driven_fields:
            b, t, h, w = grid.shape
            kernel = self.distance_kernel.view(b, 1, 1, 7, 7).expand(b, t, 1, 7, 7)
            grid = F.conv2d(
                grid.reshape(1, b * t, h, w),
                kernel.reshape(b * t, 1, 7, 7),
                padding=3,
                groups=b * t,
            ).reshape(b, t, h, w)
        # emission strength is a sampled law over the full property vector;
        # any positive constant cancels in the std normalisation below
        emission = self._series(
            torch.einsum("btp,bp->bt", self.properties, self.emission_vec),
            self.emission_act,
        )  # b, num_types
        source = (
            emission.unsqueeze(-1) * self.field_matter_affinity
        )  # b, num_types, num_fields
        src = torch.einsum("btxy,btf->bfxy", grid, source)
        src = src / src.flatten(1).std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
        x = torch.cat([self.nca_state, src], dim=1)
        h = self._series(self._nca_conv(x, self.nca_w1, self.nca_b1), self.field_act1)
        new = self._series(self._nca_conv(h, self.nca_w2, self.nca_b2), self.field_act2)
        self.nca_state = (1 - self.nca_alpha) * self.nca_state + self.nca_alpha * new
        visible = self.nca_state[:, : self.num_fields]
        rms = visible.pow(2).mean((1, 2, 3)).sqrt()
        blended = (1 - self.gate_ema_beta) * self.rms_ema + self.gate_ema_beta * rms
        self.rms_ema = torch.where(self.rms_ema == 0, rms, blended)
        return visible.clone()

    def actuator_project(self, action: torch.Tensor):
        # action = 2 motion dims + num_properties pointer dims, both through
        # per-universe actuators. The pointer selects the nearest held type
        # in property space: types are addressed by what they are, not by
        # slot index, and craft_eps keeps distinct types addressable
        direction = action[..., :2]  # b, 2
        pointer = action[..., 2 : 2 + self.num_properties]  # b, num_properties
        direction = self.actuators @ direction.unsqueeze(-1)  # b, 2, 1
        direction = direction.squeeze(-1)  # b, 2
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        pointer = (self.pointer_actuator @ pointer.unsqueeze(-1)).squeeze(-1)
        dist = (self.properties - pointer.unsqueeze(1)).norm(dim=-1)  # b, num_types
        dist = dist.masked_fill(self.agent_inventory <= 0, torch.inf)
        pointed_place_type = dist.argmin(dim=-1)

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
            pointed_place_type,
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
        craft_mask = craft_mask & self.craft_glow_gate(left_type, right_type)
        # ok so we replace the types at left and right with empty space
        # and add the new type to inventory
        new_grid = self.grid.clone()
        crafted_type, crafted_type_properties, crafted_type_sens = self.craft(
            left_type, right_type
        )

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
        self.field_matter_affinity[self.batch_idx, crafted_type] = torch.where(
            cond.unsqueeze(-1),
            crafted_type_sens,
            self.field_matter_affinity[self.batch_idx, crafted_type],
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

    def gate_score_map(self, left_type, right_type):
        # sampled gate law over [local fields / ambient EMA, pair affinity,
        # pair properties], symmetric in the pair; open where positive
        rms = self.rms_ema.clamp_min(1e-6).view(-1, 1, 1, 1)
        u = self.fields / (rms * math.sqrt(self.num_fields))  # b, F, h, w
        pair = torch.cat(
            [
                self.field_matter_affinity[self.batch_idx, left_type]
                + self.field_matter_affinity[self.batch_idx, right_type],
                self.properties[self.batch_idx, left_type]
                + self.properties[self.batch_idx, right_type],
            ],
            dim=-1,
        )  # b, F + P
        z = torch.cat(
            [
                u,
                pair.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, self.width, self.height),
            ],
            dim=1,
        )  # b, 2F + P, h, w
        q = torch.einsum("bij,bixy,bjxy->bxy", self.gate_tensor, z, z)
        return self._series(q, self.gate_act)

    def craft_glow_gate(self, left_type, right_type):
        score = self.gate_score_map(left_type, right_type)
        return (
            score[
                self.batch_idx, self.agent_position[..., 0], self.agent_position[..., 1]
            ]
            > 0
        )

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
            fields_local = self.fields.permute(0, 2, 3, 1)[
                self.batch_idx.unsqueeze(-1).unsqueeze(-1), rows, cols
            ]  # (b, visual_field_size, visual_field_size, num_fields)
        else:
            grid = self.grid  # (b, h, w)
            fields_local = self.fields.permute(0, 2, 3, 1)  # b, h, w, num_fields

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

        return sprite_grid, grid, fields_local

    def get_obs_for_agent(self, agent_view=False, normalise=False):
        self.refresh_colour_palette()
        rendered_grid, grid, fields_local = self.render(
            agent_view
        )  # b, h * sprite_resolution, w * sprite_resolution

        rgb_grid = self.colour_palette[
            self.batch_idx.unsqueeze(-1).unsqueeze(-1), grid
        ].float()  # b, h, w, 3

        # shimmer: field-coupling glow mapped to per-field RGB directions
        affinity = self.field_matter_affinity[
            self.batch_idx.view(-1, 1, 1), grid
        ]  # b, h, w, num_fields
        shimmer = torch.tanh(fields_local * affinity / 4.0)
        shift = torch.einsum(
            "bhwf,bfc->bhwc", shimmer, self.field_colour_directions
        )  # b, h, w, 3
        rgb_grid = rgb_grid + 40.0 * shift

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
        self.fields = self.step_fields(step)
        self.grid = self.step_grid(step)

    def reset_velocities(self):
        self.grid_velocity = torch.zeros(
            self.batch_size, self.width, self.height, 2, device=self.device
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
            self.tech_tree_progress[
                :, 1 : self.num_common_types + self.num_sparse_types
            ] = True
            self.properties = torch.zeros(
                self.batch_size, self.num_types, self.num_properties, device=self.device
            )
            self.properties.uniform_(-1.0, 1.0)
            self.prop_tensor = self._init_mix_tensor(self.num_properties)
            self.sens_tensor = self._init_mix_tensor(self.num_fields)

            self.field_matter_affinity = self._init_field_matter_affinity(
                self.num_types
            )
            self._init_nca()
            self._init_distance_kernel()
            self.agent_position = torch.randint(
                0, self.width, (self.batch_size, 2), device=self.device
            ).long()
            self.agent_inventory = torch.zeros(
                self.batch_size, self.num_types, device=self.device
            ).long()

            self.init_sprites()
            self.create_colour_palette()
            self.create_actuators()
            self._init_craft_law()
            self.seed_universe()
