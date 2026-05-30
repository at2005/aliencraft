import argparse
import sys
import types
from pathlib import Path

import pygame
import torch

try:
    from .world import AlienCraftWorld as Universe
except ImportError:
    from world import AlienCraftWorld as Universe

ALIEN_DIR = Path(__file__).resolve().parents[3] / "alien"
sys.path.insert(0, str(ALIEN_DIR))
from config import load_world_model_config


def _install_world_model_import_stubs():
    replay = types.ModuleType("replay")

    class _UnavailableReplayBuffer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ReplayBuffer is unavailable in render.py")

    class _RLBatch:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    replay.ReplayBuffer = _UnavailableReplayBuffer
    replay.RLBatch = _RLBatch
    sys.modules["replay"] = replay

    wandb = types.ModuleType("wandb")
    wandb.init = lambda *args, **kwargs: None
    wandb.log = lambda *args, **kwargs: None
    sys.modules.setdefault("wandb", wandb)

    tqdm_module = types.ModuleType("tqdm")

    class _Tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(()) if self.iterable is None else iter(self.iterable)

        def update(self, *args, **kwargs):
            return None

    tqdm_module.tqdm = _Tqdm
    sys.modules.setdefault("tqdm", tqdm_module)


def _import_world_model():
    try:
        from world_model import WorldModel, load_model_state
    except ModuleNotFoundError as exc:
        if exc.name not in {"posix_ipc", "wandb", "tqdm"}:
            raise
        sys.modules.pop("world_model", None)
        _install_world_model_import_stubs()
        return _load_world_model_from_source()
    except TypeError as exc:
        if "unsupported operand type(s) for |" not in str(exc):
            raise
        sys.modules.pop("world_model", None)
        _install_world_model_import_stubs()
        return _load_world_model_from_source()

    return WorldModel, load_model_state


def _load_world_model_from_source():
    module = types.ModuleType("world_model")
    module.__file__ = str(ALIEN_DIR / "world_model.py")
    module.__package__ = ""
    sys.modules["world_model"] = module
    source = (ALIEN_DIR / "world_model.py").read_text()
    code = compile(
        "from __future__ import annotations\n" + source,
        str(ALIEN_DIR / "world_model.py"),
        "exec",
    )
    exec(code, module.__dict__)
    return module.WorldModel, module.load_model_state


WorldModel, load_model_state = _import_world_model()

TRAINING_UNIVERSE_KWARGS = {
    "width": 128,
    "height": 128,
    "num_types": 20,
    "num_properties": 10,
    "num_fields": 3,
    "num_common_types": 8,
    "num_sparse_types": 4,
}


def _load_checkpoint_policy(device):
    world_model_config = load_world_model_config(
        str(ALIEN_DIR / "configs/world_model.json")
    )
    config = world_model_config.model_config
    policy = WorldModel(world_model_config).to(device)
    load_model_state(
        policy,
        str(ALIEN_DIR / "models/world_model_checkpoint.pth"),
        map_location=device,
    )
    policy.eval()
    state = {
        "prev_action": torch.zeros(1, config.action_dim, device=device),
        "latent": torch.zeros(1, config.d_model, device=device),
        "recurrent": torch.zeros(1, config.recurrent_dim, device=device),
    }
    return policy, config, state


def _make_training_universe(config, device, sprite_resolution, batch_size=1):
    return Universe(
        batch_size=batch_size,
        visual_field_size=config.visual_field_size,
        sprite_resolution=sprite_resolution,
        device=device,
        **TRAINING_UNIVERSE_KWARGS,
    )


def _policy_action(policy, state, universe):
    obs = universe.get_obs_for_agent(agent_view=True, normalise=True)
    state["recurrent"] = policy.update_recurrent_state(
        state["prev_action"], state["latent"], state["recurrent"]
    )
    obs_embedding = policy.embed_observation(obs)
    state["latent"], _ = policy.sample_latent(obs_embedding, state["recurrent"])
    action_mean, action_std = policy.get_action_params(
        state["latent"], state["recurrent"]
    )
    action = torch.distributions.Normal(action_mean, action_std).sample()
    state["prev_action"] = action
    return action


def _describe_action(universe, action):
    motion, craft, pick, place, direction, place_type = universe.actuator_project(
        action
    )
    if motion[0]:
        name = f"move {tuple(direction[0].tolist())}"
    elif craft[0]:
        name = "craft"
    elif pick[0]:
        name = "pick"
    elif place[0]:
        name = f"place type {place_type[0].item()}"
    else:
        name = "noop"
    raw = action[0, :4].detach().cpu().tolist()
    return f"{name} raw=({raw[0]:+.2f}, {raw[1]:+.2f}, {raw[2]:+.2f}, {raw[3]:+.2f})"


def _describe_agent_position(universe):
    pos = universe.agent_position[0].detach().cpu().tolist()
    return f"pos=({pos[0]}, {pos[1]})"


def render(universe, batch_index=0, cell_size=6):
    pygame.init()
    frame = _render_frame(universe, batch_index)
    screen = _make_screen(frame, cell_size)
    _draw_frame(screen, frame, cell_size)
    _draw_inventory(screen, universe, batch_index, 0)
    pygame.display.flip()
    return screen


def render_animation(
    universe,
    steps=10,
    batch_index=0,
    cell_size=6,
    fps=4,
    policy=None,
    state=None,
    random_actions=False,
):
    pygame.init()
    frame = _render_frame(universe, batch_index)
    screen = _make_screen(frame, cell_size)
    clock = pygame.time.Clock()
    registered_crafts = 0

    for step in range(1, steps + 1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if random_actions:
            action = torch.randn(1, 2 + universe.num_types, device=universe.device)
        elif policy is not None:
            action = _policy_action(policy, state, universe)
        else:
            action = None
        if action is not None:
            registered_crafts += _newly_registered_craft_count(
                universe, action, batch_index
            )
        universe.step(step, action)
        frame = _render_frame(universe, batch_index)
        _draw_frame(screen, frame, cell_size)
        _draw_inventory(screen, universe, batch_index, registered_crafts)
        caption = f"aliencraft step {step} | {_describe_agent_position(universe)}"
        if action is not None:
            caption += f" | {_describe_action(universe, action)}"
        pygame.display.set_caption(caption)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def _render_frame(universe, batch_index):
    frame = universe.get_obs_for_agent()[batch_index].byte()
    return _draw_agent_marker(universe, frame, batch_index)


def _draw_agent_marker(universe, frame, batch_index):
    x, y = universe.agent_position[batch_index].detach().cpu().tolist()
    size = universe.sprite_resolution
    x0, y0 = x * size, y * size
    frame = frame.clone()
    frame[x0 : x0 + size, y0 : y0 + size] = frame.new_tensor([255, 255, 255])
    if size > 2:
        frame[x0 + 1 : x0 + size - 1, y0 + 1 : y0 + size - 1] = frame.new_tensor(
            [255, 40, 220]
        )
    return frame


def _newly_registered_craft_count(universe, action, batch_index):
    _, craft_mask, _, _, _, _ = universe.actuator_project(action)
    if not craft_mask[batch_index]:
        return 0
    pos = universe.agent_position
    left_type = universe.grid[
        universe.batch_idx,
        (pos[..., 0] - 1) % universe.width,
        pos[..., 1],
    ]
    right_type = universe.grid[
        universe.batch_idx,
        (pos[..., 0] + 1) % universe.width,
        pos[..., 1],
    ]
    crafted_type, _ = universe.craft(left_type, right_type)
    crafted_type = crafted_type[batch_index].item()
    if crafted_type == -1:
        return 0
    return int(not universe.tech_tree_progress[batch_index, crafted_type].item())


def _draw_frame(screen, frame, cell_size):
    surface = _make_surface(frame)
    if cell_size != 1:
        width, height = surface.get_size()
        surface = pygame.transform.scale(
            surface, (width * cell_size, height * cell_size)
        )
    screen.blit(surface, (0, 0))


def _draw_inventory(screen, universe, batch_index, registered_crafts):
    font = pygame.font.SysFont(None, 20)
    inventory = universe.agent_inventory[batch_index].detach().cpu()
    entries = [
        (idx, int(count)) for idx, count in enumerate(inventory.tolist()) if count
    ]
    rows = [("inventory", None), (f"new crafts: {registered_crafts}", None)] + [
        (f"type {idx}: {count}", idx) for idx, count in entries[:10]
    ]
    if not entries:
        rows.append(("empty", None))
    if len(entries) > 10:
        rows.append((f"+{len(entries) - 10} more", None))

    width = 150
    height = 12 + 20 * len(rows)
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))
    colours = universe.colour_palette[batch_index].detach().cpu()

    for row, (text, type_idx) in enumerate(rows):
        y = 6 + row * 20
        if type_idx is not None:
            colour = tuple(int(x) for x in colours[type_idx].tolist())
            pygame.draw.rect(panel, colour, pygame.Rect(8, y + 3, 10, 10))
            text_x = 24
        else:
            text_x = 8
        panel.blit(font.render(text, True, (255, 255, 255)), (text_x, y))

    screen.blit(panel, (8, 8))


def _make_surface(frame):
    pixels = frame.detach().cpu().byte()
    return pygame.surfarray.make_surface(pixels.permute(1, 0, 2).numpy())


def _make_screen(frame, cell_size):
    height, width = frame.shape[:2]
    return pygame.display.set_mode((width * cell_size, height * cell_size))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random",
        action="store_true",
        help="take random raw actions instead of actions from the checkpoint policy",
    )
    parser.add_argument(
        "--sprite-resolution",
        type=int,
        default=4,
        help="pixels per universe cell in the generated material sprite texture",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = "cpu"
    with torch.inference_mode():
        if args.random:
            policy, state = None, None
            config = load_world_model_config(
                str(ALIEN_DIR / "configs/world_model.json")
            ).model_config
        else:
            policy, config, state = _load_checkpoint_policy(device)
        universe = _make_training_universe(
            config,
            device,
            args.sprite_resolution,
        )
        render_animation(
            universe,
            steps=1000,
            cell_size=4,
            policy=policy,
            state=state,
            random_actions=args.random,
        )
