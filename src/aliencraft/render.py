import argparse
import math
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
ENV_STATE_PATH = ALIEN_DIR / "env_state.pth"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
        from world_model import WorldModel, load_model_state, load_universe
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

    return WorldModel, load_model_state, load_universe


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
    return module.WorldModel, module.load_model_state, module.load_universe


WorldModel, load_model_state, load_universe = _import_world_model()
_MANUAL_WAIT = object()

TRAINING_UNIVERSE_KWARGS = {
    "width": 2048,
    "height": 2048,
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


def _load_torch_state_dict(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _infer_universe_kwargs_from_state(config, state_dict):
    if "grid" not in state_dict:
        raise ValueError("environment state checkpoint is missing 'grid'")
    if "properties" not in state_dict:
        raise ValueError("environment state checkpoint is missing 'properties'")
    if "fields" not in state_dict:
        raise ValueError("environment state checkpoint is missing 'fields'")

    grid = state_dict["grid"]
    properties = state_dict["properties"]
    fields = state_dict["fields"]
    if grid.ndim != 3:
        raise ValueError(f"expected 'grid' to have 3 dims, got {tuple(grid.shape)}")
    if properties.ndim != 3:
        raise ValueError(
            f"expected 'properties' to have 3 dims, got {tuple(properties.shape)}"
        )
    if fields.ndim != 4:
        raise ValueError(f"expected 'fields' to have 4 dims, got {tuple(fields.shape)}")

    batch_size, width, height = grid.shape
    _, num_types, num_properties = properties.shape
    _, num_fields, _, _ = fields.shape

    if "sprite_positions" in state_dict:
        sprite_resolution = state_dict["sprite_positions"].shape[1]
    elif "sprites" in state_dict:
        sprite_resolution = state_dict["sprites"].shape[2]
    else:
        raise ValueError(
            "environment state checkpoint is missing sprite resolution tensors"
        )

    if "nb_offsets" in state_dict:
        visual_field_size = state_dict["nb_offsets"].shape[0] * sprite_resolution
    else:
        visual_field_size = config.visual_field_size

    num_common_types = min(TRAINING_UNIVERSE_KWARGS["num_common_types"], num_types)
    num_sparse_types = min(
        TRAINING_UNIVERSE_KWARGS["num_sparse_types"],
        max(0, num_types - num_common_types),
    )

    kwargs = {
        "batch_size": batch_size,
        "width": width,
        "height": height,
        "num_types": num_types,
        "num_properties": num_properties,
        "num_fields": num_fields,
        "num_common_types": num_common_types,
        "num_sparse_types": num_sparse_types,
        "visual_field_size": visual_field_size,
        "sprite_resolution": sprite_resolution,
    }
    return kwargs


def _make_training_universe(config, device, sprite_resolution, batch_size=1):
    return Universe(
        batch_size=batch_size,
        visual_field_size=config.visual_field_size,
        sprite_resolution=sprite_resolution,
        device=device,
        **TRAINING_UNIVERSE_KWARGS,
    )


def _make_env_state_universe(config, device, checkpoint_path):
    state_dict = _load_torch_state_dict(checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"expected {checkpoint_path} to contain a state dict, got "
            f"{type(state_dict).__name__}"
        )

    universe_kwargs = _infer_universe_kwargs_from_state(config, state_dict)
    universe = Universe(device=device, **universe_kwargs)
    return load_universe(universe, str(checkpoint_path), map_location=device)


def _update_policy_state(policy, state, universe):
    obs = universe.get_obs_for_agent(agent_view=True, normalise=True)
    state["recurrent"] = policy.update_recurrent_state(
        state["prev_action"], state["latent"], state["recurrent"]
    )
    obs_embedding = policy.embed_observation(obs)
    state["latent"], _ = policy.sample_latent(obs_embedding, state["recurrent"])


def _policy_action(policy, state):
    action_mean, action_std = policy.get_action_params(
        state["latent"], state["recurrent"]
    )
    return torch.distributions.Normal(action_mean, action_std).sample()


def _decode_latent_frame(policy, latent, recurrent, batch_index):
    decoded = policy.decode_latent(latent, recurrent)
    return decoded[batch_index].detach().float().clamp(0, 1).mul(255).byte()


def _advance_dream(policy, state, universe, action, batch_index, show_posterior):
    prev_action = (
        action if action is not None else torch.zeros_like(state["prev_action"])
    )
    state["recurrent"] = policy.update_recurrent_state(
        prev_action, state["latent"], state["recurrent"]
    )
    prior_latent, _ = policy.get_next_latent_prediction(state["recurrent"])
    prior_frame = _decode_latent_frame(
        policy, prior_latent, state["recurrent"], batch_index
    )

    if show_posterior:
        obs = universe.get_obs_for_agent(agent_view=True, normalise=True)
        true_frame = obs[batch_index].detach().float().clamp(0, 1).mul(255).byte()
        obs_embedding = policy.embed_observation(obs)
        posterior_latent_for_state, posterior_distribution = policy.sample_latent(
            obs_embedding, state["recurrent"]
        )
        posterior_frame = _decode_latent_frame(
            policy, posterior_distribution.mean, state["recurrent"], batch_index
        )
        frame = torch.cat([prior_frame, posterior_frame, true_frame], dim=1)
        state["latent"] = posterior_latent_for_state
    else:
        frame = prior_frame
        state["latent"] = prior_latent
    return frame


def _initial_frame(universe, batch_index, policy=None, dream=False, show_posterior=False):
    if dream:
        frame = universe.get_obs_for_agent(agent_view=True)[batch_index].byte()
        if show_posterior:
            return torch.cat([frame, frame, frame], dim=1)
        return frame
    return _render_frame(universe, batch_index)


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
    fps=12,
    policy=None,
    state=None,
    random_actions=False,
    manual_actions=False,
    dream=False,
    show_posterior=False,
):
    pygame.init()
    if dream and (policy is None or state is None):
        raise ValueError("--dream requires a checkpoint policy")
    frame = _initial_frame(universe, batch_index, policy, dream, show_posterior)
    screen = _make_screen(frame, cell_size)
    _draw_frame(screen, frame, cell_size)
    _draw_inventory(screen, universe, batch_index, 0)
    pygame.display.flip()
    clock = pygame.time.Clock()
    registered_crafts = 0
    manual_state = {"selected_place_type": 0}
    step = 0

    if dream:
        _update_policy_state(policy, state, universe)

    while step < steps:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if policy is not None and not dream:
            _update_policy_state(policy, state, universe)

        if manual_actions:
            action = _manual_action_from_input(universe, events, manual_state)
            if action is _MANUAL_WAIT:
                caption = (
                    f"aliencraft step {step} | {_describe_agent_position(universe)}"
                )
                if dream:
                    caption += " | dream"
                    if show_posterior:
                        caption += " | prior/posterior/true"
                caption += (
                    f" | manual place={manual_state['selected_place_type']}"
                    " | waiting"
                )
                pygame.display.set_caption(caption)
                clock.tick(fps)
                continue
        elif random_actions:
            action = torch.randn(1, 2 + universe.num_types, device=universe.device)
        elif policy is not None:
            action = _policy_action(policy, state)
        else:
            action = None

        step += 1
        if action is not None:
            registered_crafts += _newly_registered_craft_count(
                universe, action, batch_index
            )
        universe.step(step, action)
        if dream:
            frame = _advance_dream(
                policy, state, universe, action, batch_index, show_posterior
            )
        else:
            frame = _render_frame(universe, batch_index)
        _draw_frame(screen, frame, cell_size)
        _draw_inventory(screen, universe, batch_index, registered_crafts)
        caption = f"aliencraft step {step} | {_describe_agent_position(universe)}"
        if dream:
            caption += " | dream"
            if show_posterior:
                caption += " | prior/posterior/true"
        if manual_actions:
            caption += f" | manual place={manual_state['selected_place_type']}"
        if action is not None:
            caption += f" | {_describe_action(universe, action)}"
        pygame.display.set_caption(caption)
        pygame.display.flip()
        if policy is not None:
            state["prev_action"] = (
                action
                if action is not None
                else torch.zeros_like(state["prev_action"])
            )
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


def _manual_action_from_input(universe, events, manual_state):
    for event in events:
        if event.type != pygame.KEYDOWN:
            continue
        if event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS):
            manual_state["selected_place_type"] = (
                manual_state["selected_place_type"] + 1
            ) % universe.num_types
            continue
        if event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
            manual_state["selected_place_type"] = (
                manual_state["selected_place_type"] - 1
            ) % universe.num_types
            continue
        if pygame.K_0 <= event.key <= pygame.K_9:
            selected = event.key - pygame.K_0
            if selected < universe.num_types:
                manual_state["selected_place_type"] = selected
            continue
        if event.key in (pygame.K_PERIOD, pygame.K_RETURN, pygame.K_n):
            return None
        if event.key in (pygame.K_d, pygame.K_RIGHT):
            return _manual_projected_action(universe, (0.0, 1.0))
        if event.key in (pygame.K_a, pygame.K_LEFT):
            return _manual_projected_action(universe, (0.0, -1.0))
        if event.key in (pygame.K_s, pygame.K_DOWN):
            return _manual_projected_action(universe, (1.0, 0.0))
        if event.key in (pygame.K_w, pygame.K_UP):
            return _manual_projected_action(universe, (-1.0, 0.0))
        if event.key == pygame.K_c:
            isqrt2 = 1.0 / math.sqrt(2.0)
            return _manual_projected_action(universe, (isqrt2, isqrt2))
        if event.key in (pygame.K_e, pygame.K_SPACE):
            isqrt2 = 1.0 / math.sqrt(2.0)
            return _manual_projected_action(universe, (isqrt2, -isqrt2))
        if event.key == pygame.K_p:
            isqrt2 = 1.0 / math.sqrt(2.0)
            return _manual_projected_action(
                universe,
                (-isqrt2, isqrt2),
                place_type=manual_state["selected_place_type"],
            )

    return _MANUAL_WAIT


def _manual_projected_action(universe, projected_direction, place_type=None):
    action = torch.zeros(
        universe.batch_size,
        2 + universe.num_types,
        device=universe.device,
        dtype=universe.actuators.dtype,
    )
    direction = torch.tensor(
        projected_direction, device=universe.device, dtype=universe.actuators.dtype
    ).expand(universe.batch_size, -1)
    action[:, :2] = torch.linalg.solve(
        universe.actuators, direction.unsqueeze(-1)
    ).squeeze(-1)

    if place_type is not None:
        action[:, 2 + place_type] = 1.0

    return action


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
    if not universe.craft_glow_gate(left_type, right_type)[batch_index].item():
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
        "--manual",
        action="store_true",
        help="take actions from pygame keyboard input instead of the checkpoint policy",
    )
    parser.add_argument(
        "--dream",
        action="store_true",
        help=(
            "render decoded prior predictions from get_next_latent_prediction() "
            "instead of real environment observations"
        ),
    )
    parser.add_argument(
        "--show-posterior",
        action="store_true",
        help="with --dream, show posterior decode beside the prior decode",
    )
    parser.add_argument(
        "--sprite-resolution",
        type=int,
        default=4,
        help="pixels per universe cell for a fresh generated material sprite texture",
    )
    parser.add_argument(
        "--env-state",
        type=Path,
        default=ENV_STATE_PATH,
        help="environment state checkpoint to load",
    )
    parser.add_argument(
        "--fresh-env",
        action="store_true",
        help="create a fresh environment instead of loading --env-state",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="number of animation steps to render",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=8,
        help="screen pixels per rendered observation pixel; does not change env size",
    )
    args = parser.parse_args()
    if args.cell_size < 1:
        parser.error("--cell-size must be at least 1")
    if args.show_posterior and not args.dream:
        parser.error("--show-posterior requires --dream")
    return args


if __name__ == "__main__":
    args = _parse_args()
    device = "cpu"
    with torch.inference_mode():
        if (args.random or args.manual) and not args.dream:
            policy, state = None, None
            config = load_world_model_config(
                str(ALIEN_DIR / "configs/world_model.json")
            ).model_config
        else:
            policy, config, state = _load_checkpoint_policy(device)
        if args.fresh_env:
            universe = _make_training_universe(
                config,
                device,
                args.sprite_resolution,
            )
        else:
            universe = _make_env_state_universe(config, device, args.env_state)
        render_animation(
            universe,
            steps=args.steps,
            cell_size=args.cell_size,
            policy=policy,
            state=state,
            random_actions=args.random,
            manual_actions=args.manual,
            dream=args.dream,
            show_posterior=args.show_posterior,
        )
