# generation filters: a sampled universe is kept only if these pass
import copy
import gzip
import os

import torch
import torch.nn.functional as F

# per-universe tensors that define a universe; everything else is state
GENOME_KEYS = (
    "grid", "agent_position", "properties", "field_matter_affinity",
    "prop_tensor", "sens_tensor", "craft_map", "nca_w1", "nca_w2", "nca_b1",
    "nca_b2", "nca_alpha", "field_act1", "field_act2", "chem_act", "sens_act",
    "gate_coeffs", "craft_energy_scale", "gate_ema_beta", "distance_kernel",
    "colour_projection", "field_colour_directions", "sprites", "actuators",
)


def snapshot_genome(world, i):
    rec = {k: getattr(world, k)[i].detach().cpu().clone() for k in GENOME_KEYS}
    rec["grid"] = rec["grid"].to(torch.uint8)
    rec["craft_map"] = rec["craft_map"].to(torch.int16)
    return rec


def load_genome(world, record, spin=100):
    with torch.no_grad():
        world.reset()
        for k in GENOME_KEYS:
            getattr(world, k)[0] = record[k].to(world.device)
        world.refresh_colour_palette()
        for t in range(spin):
            world.step(t)


def generate_pool(path, n, batch_size=48, device="cpu", band=(0.1, 0.65), **world_kwargs):
    # batched universe generation: dynamics filters once, gate redrawn
    # blockwise; accepted genomes appended to a growing .pt file
    from .env import DEFAULT_WORLD_KWARGS
    from .world import AlienCraftWorld

    records = torch.load(path) if os.path.exists(path) else []
    kwargs = {**DEFAULT_WORLD_KWARGS, **world_kwargs}
    world = AlienCraftWorld(batch_size=batch_size, device=device, **kwargs)
    with torch.no_grad():
        while len(records) < n:
            world.reset()
            init_grid = world.grid.detach().cpu().clone()
            init_pos = world.agent_position.detach().cpu().clone()
            stats = edge_stats(world)
            ok = accept(
                dict(
                    stats,
                    climbable=torch.ones_like(stats["climbable"]),
                    bite=torch.zeros_like(stats["bite"]),
                ),
                band,
            )
            fr = rungs_open(world)
            gate_ok = (fr > 0).all(1) & (fr.median(1).values <= 0.6)
            for _ in range(60):
                need = ok & ~gate_ok
                if not need.any():
                    break
                old = (
                    world.gate_coeffs.clone(),
                    world.craft_energy_scale.clone(),
                    world.gate_ema_beta.clone(),
                )
                world._init_gate()
                keep = ~need
                world.gate_coeffs[keep] = old[0][keep]
                world.craft_energy_scale[keep] = old[1][keep]
                world.gate_ema_beta[keep] = old[2][keep]
                fr = rungs_open(world)
                gate_ok = gate_ok | (
                    need & (fr > 0).all(1) & (fr.median(1).values <= 0.6)
                )
            for i in (ok & gate_ok).nonzero().flatten().tolist():
                rec = snapshot_genome(world, i)
                rec["grid"] = init_grid[i].to(torch.uint8)
                rec["agent_position"] = init_pos[i]
                records.append(rec)
            torch.save(records, path)
            print(f"pool: {len(records)}/{n}", flush=True)
    return records


def sample_edge_world(world, band=(0.1, 0.65), tries=96, gate_tries=60):
    # dynamics filters don't depend on the gate genome, so the gate is
    # resampled blockwise on worlds whose dynamics already passed
    with torch.no_grad():
        for i in range(tries):
            world.reset()
            stats = edge_stats(world)
            stats_sans_gate = dict(
                stats,
                climbable=torch.ones_like(stats["climbable"]),
                bite=torch.zeros_like(stats["bite"]),
            )
            if not bool(accept(stats_sans_gate, band)[0]):
                continue
            for _ in range(gate_tries):
                fr = rungs_open(world)
                if bool((fr > 0).all(1)[0]) and float(fr.median(1).values[0]) <= 0.6:
                    return i + 1
                world._init_gate()
    raise RuntimeError(f"no edge universe found in {tries} samples")


def accept(stats, band=(0.1, 0.65)):
    return (
        (stats["complexity"] >= band[0]) & (stats["complexity"] <= band[1])
        & (stats["persistence"] >= 0.2) & (stats["persistence"] <= 0.995)
        & (stats["linearity"] >= 0.4) & (stats["linearity"] <= 0.97)
        & (stats["sensitivity"] >= 0.05)
        & (stats["spread"] >= 30.0)
        & (stats["climbable"] >= 0.999)
        & (stats["bite"] <= 0.6)
    )


def stencil_fit(world, traj):
    # ridge fit of a 3x3 linear stencil; R^2 near 1 = linear order,
    # near 0 = chaos, the edge lives between
    T, b, f = traj.shape[0], world.batch_size, world.num_fields
    X = F.unfold(traj[:-1].reshape(-1, f, world.width, world.height), 3, padding=1)
    X = X.reshape(T - 1, b, f * 9, -1).permute(1, 0, 3, 2).reshape(b, -1, f * 9)
    Y = traj[1:].permute(1, 0, 3, 4, 2).reshape(b, -1, f)
    keep = torch.randperm(X.shape[1], device=X.device)[:4000]
    X = torch.cat([X[:, keep], torch.ones_like(X[:, keep, :1])], dim=-1)
    Y = Y[:, keep]
    A = X.transpose(1, 2) @ X + 1e-3 * torch.eye(X.shape[-1], device=X.device)
    W = torch.linalg.solve(A, X.transpose(1, 2) @ Y)
    res = (X @ W - Y).pow(2).flatten(1).sum(1)
    tot = (Y - Y.mean((1, 2), keepdim=True)).pow(2).flatten(1).sum(1)
    return 1 - res / tot.clamp_min(1e-9)


def rungs_open(world):
    # per-rung gate open-area fraction right now, with the tech tree rolled
    # out hypothetically; > 0 means open somewhere
    saved = world.properties, world.field_matter_affinity
    props = world.properties.clone()
    sens = world.field_matter_affinity.clone()
    world.properties, world.field_matter_affinity = props, sens
    num_nat = world.num_common_types + world.num_sparse_types
    open_fracs = []
    for t in range(num_nat, world.num_types):
        flat = (world.craft_map == t).flatten(1).float().argmax(1)
        a, b = flat // world.num_types, flat % world.num_types
        props[:, t] = world._series(
            torch.einsum(
                "bijk,bj,bk->bi",
                world.prop_tensor,
                props[world.batch_idx, a],
                props[world.batch_idx, b],
            ),
            world.chem_act,
        )
        sens[:, t] = world._series(
            torch.einsum(
                "bijk,bj,bk->bi",
                world.sens_tensor,
                sens[world.batch_idx, a],
                sens[world.batch_idx, b],
            ),
            world.sens_act,
        )
        score = world.gate_score_map(a, b)
        open_fracs.append((score > 0).flatten(1).float().mean(1))
    world.properties, world.field_matter_affinity = saved
    return torch.stack(open_fracs, dim=1)  # b, rungs


def colour_spread(world):
    # median pairwise colour distance across the rolled-out tech tree
    props = world.properties.clone()
    for t in range(world.num_common_types + world.num_sparse_types, world.num_types):
        flat = (world.craft_map == t).flatten(1).float().argmax(1)
        pa = props[world.batch_idx, flat // world.num_types]
        pb = props[world.batch_idx, flat % world.num_types]
        props[:, t] = world._series(
            torch.einsum("bijk,bj,bk->bi", world.prop_tensor, pa, pb), world.chem_act
        )
    rgb = 255 * (0.5 + 0.35 * torch.tanh(props @ world.colour_projection))
    d = (rgb.unsqueeze(2) - rgb.unsqueeze(1)).abs().amax(-1)
    iu = torch.triu_indices(world.num_types, world.num_types, 1, device=world.device)
    return d[:, iu[0], iu[1]].median(1).values


def edge_stats(world, spin: int = 100, steps: int = 200, every: int = 4):
    # gzip complexity, one-step persistence, linear predictability, source
    # sensitivity vs a massless ghost, colour spread
    frames, traj = [], []
    with torch.no_grad():
        for t in range(spin):
            world.step(t)
        rms = world.fields.pow(2).mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
        rung_fracs = None
        for t in range(steps):
            world.step(spin + t)
            traj.append(world.fields.clone())
            if t % 50 == 0:
                fr = rungs_open(world)
                rung_fracs = fr if rung_fracs is None else torch.maximum(rung_fracs, fr)
            if t % every == 0:
                state = torch.cat(
                    [
                        world.grid.to(torch.uint8).unsqueeze(1),
                        (world.fields / rms * 25)
                        .clamp(-127, 127)
                        .to(torch.int8)
                        .view(torch.uint8),
                    ],
                    dim=1,
                )
                frames.append(state.cpu())
        ghost = copy.deepcopy(world)
        ghost.properties[..., 0] = 0.0
        for t in range(30):
            world.step(spin + steps + t)
            ghost.step(spin + steps + t)
        diff = (world.fields - ghost.fields).pow(2).mean((1, 2, 3)).sqrt()
        sens = diff / world.fields.pow(2).mean((1, 2, 3)).sqrt().clamp_min(1e-6)
        traj = torch.stack(traj)
        res = (traj[1:] - traj[:-1]).pow(2).sum((0, 2, 3, 4))
        tot = (traj[1:] - traj[1:].mean((0, 2, 3, 4), keepdim=True)).pow(2).sum(
            (0, 2, 3, 4)
        )
        persistence = 1 - res / tot.clamp_min(1e-9)
        linearity = stencil_fit(world, traj)
        spread = colour_spread(world)
        rung_fracs = torch.maximum(rung_fracs, rungs_open(world))
        climbable = (rung_fracs > 0).float().mean(1)
        bite = rung_fracs.median(1).values
    frames = torch.stack(frames)
    complexity = torch.tensor(
        [
            len(gzip.compress(frames[:, b].numpy().tobytes())) / frames[:, b].numel()
            for b in range(world.batch_size)
        ],
        device=world.device,
    )
    return dict(
        complexity=complexity,
        persistence=persistence,
        linearity=linearity,
        sensitivity=sens,
        spread=spread,
        climbable=climbable,
        bite=bite,
    )
