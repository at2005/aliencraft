# minimal gymnasium interface over AlienCraftWorld
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .filter import load_laws, sample_edge_world
from .world import AlienCraftWorld

# slots are storage, not the tech tree: the reachable set is whatever the
# sampled chemistry closes over, banded to [10, 300] by the filters
DEFAULT_WORLD_KWARGS = dict(
    width=64,
    height=64,
    num_types=400,
    num_common_types=4,
    num_sparse_types=1,
    num_properties=3,
    num_fields=3,
    sprite_resolution=4,
    visual_field_size=32,
    driven_fields=True,
)


class AlienCraftEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(
        self,
        device: str = "cpu",
        max_episode_steps: int = 1000,
        render_mode: str = None,
        complexity_band: tuple = (0.1, 0.65),
        pool: str = None,
        **world_kwargs,
    ):
        kwargs = {**DEFAULT_WORLD_KWARGS, **world_kwargs}
        self.world = AlienCraftWorld(batch_size=1, device=device, **kwargs)
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.complexity_band = complexity_band
        self.pool = torch.load(pool) if pool else None

        obs_size = self.world.visual_field_size
        self.observation_space = spaces.Box(
            0.0, 1.0, (obs_size, obs_size, 3), np.float32
        )
        # 2 motion dims + num_properties pointer dims (nearest held type)
        self.action_space = spaces.Box(
            -1.0, 1.0, (2 + self.world.num_properties,), np.float32
        )
        self._t = 0

    def _obs(self):
        obs = self.world.get_obs_for_agent(agent_view=True, normalise=True)
        return obs[0].detach().cpu().numpy().astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # world generation draws from torch's global RNG
            torch.manual_seed(seed)
        with torch.no_grad():
            # reject universes that are frozen, noise, source-blind, or muddy
            if self.pool is not None:
                load_laws(self.world, self.pool[self.np_random.integers(len(self.pool))])
            elif self.complexity_band is None:
                self.world.reset()
            else:
                sample_edge_world(self.world, self.complexity_band)
        self._t = 0
        return self._obs(), {"discovered_types": 0}

    def step(self, action):
        action = torch.as_tensor(
            action, dtype=torch.float32, device=self.world.device
        ).reshape(1, -1)
        before = self.world.tech_tree_progress.sum().item()
        self._t += 1
        with torch.no_grad():
            self.world.step(self._t, action)
        discovered = self.world.tech_tree_progress.sum().item()
        reward = float(discovered - before)
        truncated = self._t >= self.max_episode_steps
        info = {
            "discovered_types": discovered,
            "agent_position": self.world.agent_position[0].tolist(),
        }
        return self._obs(), reward, False, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            frame = self.world.get_obs_for_agent(agent_view=False)
            return frame[0].detach().cpu().numpy().astype(np.uint8)
        return None


gym.register(id="AlienCraft-v0", entry_point=AlienCraftEnv)
