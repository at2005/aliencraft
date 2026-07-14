# generation filters: a sampled universe is kept only if these pass
import copy
import gzip
import os

import torch
import torch.nn.functional as F

# per-universe tensors that define a universe's laws; layout (grid, agent
# start) is rerolled fresh on every load, crafter-style
LAW_KEYS = (
    "properties", "field_matter_affinity",
    "prop_tensor", "sens_tensor", "craft_f", "craft_f_act", "craft_thresh",
    "nca_w1", "nca_w2", "nca_b1",
    "nca_b2", "nca_alpha", "field_act1", "field_act2", "chem_act", "sens_act",
    "emission_vec", "emission_act", "force_tensor", "force_act",
    "gate_tensor", "gate_act", "gate_ema_beta", "distance_kernel",
    "colour_projection", "field_colour_directions", "sprites", "actuators",
    "pointer_actuator",
)


def snapshot_laws(world, i):
    return {k: getattr(world, k)[i].detach().cpu().clone() for k in LAW_KEYS}


def load_laws(world, record, spin=100):
    with torch.no_grad():
        world.reset()
        for k in LAW_KEYS:
            getattr(world, k)[0] = record[k].to(world.device)
        world.refresh_colour_palette()
        for t in range(spin):
            world.step(t)


def _series1(world, i, x, act):
    return world._series(x.unsqueeze(0), act[i : i + 1]).squeeze(0)


def closure(world, max_rounds=30):
    # hypothetical closure of the naturals under the chemistry + craft laws:
    # which types are reachable, at what depth, by which first recipe.
    # Dedup order mirrors the runtime allocation, so a climb that replays
    # `pairs` in order lands each child in its listed slot.
    num_nat = world.num_common_types + world.num_sparse_types
    cap = world.num_types - num_nat
    out = []
    for i in range(world.batch_size):
        props = world.properties[i, 1:num_nat].clone()
        sens = world.field_matter_affinity[i, 1:num_nat].clone()
        slots = torch.arange(1, num_nat, device=world.device)
        depth = torch.zeros(num_nat - 1, dtype=torch.long, device=world.device)
        pairs = []
        capped = False
        for _ in range(max_rounds):
            n = props.shape[0]
            ii, jj = torch.triu_indices(n, n, 0, device=world.device)
            score = _series1(
                world, i,
                torch.einsum("jk,nj,nk->n", world.craft_f[i], props[ii], props[jj]),
                world.craft_f_act,
            )
            ok = score > world.craft_thresh[i]
            if capped or not ok.any():
                break
            ii, jj = ii[ok], jj[ok]
            child = _series1(
                world, i,
                torch.einsum("ijk,nj,nk->ni", world.prop_tensor[i], props[ii], props[jj]),
                world.chem_act,
            )
            child_sens = _series1(
                world, i,
                torch.einsum("ijk,nj,nk->ni", world.sens_tensor[i], sens[ii], sens[jj]),
                world.sens_act,
            )
            cdepth = torch.maximum(depth[ii], depth[jj]) + 1
            fresh = (
                (torch.cdist(child, props).min(1).values >= world.craft_eps)
                .nonzero()
                .flatten()
            )
            added = 0
            for k in fresh.tolist():
                if capped or (child[k] - props).norm(dim=-1).min() < world.craft_eps:
                    continue
                slot = num_nat + len(pairs)
                pairs.append((int(slots[ii[k]]), int(slots[jj[k]]), slot))
                props = torch.cat([props, child[k : k + 1]])
                sens = torch.cat([sens, child_sens[k : k + 1]])
                slots = torch.cat([slots, torch.tensor([slot], device=world.device)])
                depth = torch.cat([depth, cdepth[k : k + 1]])
                added += 1
                capped = len(pairs) >= cap
            if added == 0:
                break
        rgb = 255 * (0.5 + 0.35 * torch.tanh(props @ world.colour_projection[i]))
        d = (rgb.unsqueeze(1) - rgb.unsqueeze(0)).abs().amax(-1)
        iu = torch.triu_indices(props.shape[0], props.shape[0], 1)
        out.append(
            dict(
                props=props,
                sens=sens,
                depth=depth,
                capped=capped,
                pairs=torch.tensor(pairs, dtype=torch.long).reshape(-1, 3),
                spread=float(d[iu[0], iu[1]].median()) if props.shape[0] > 1 else 0.0,
            )
        )
    return out


def gate_fracs(world, closures):
    # open-area fraction of each closure recipe's gate right now, with the
    # hypothetical types written into their slots
    num_nat = world.num_common_types + world.num_sparse_types
    saved = world.properties.clone(), world.field_matter_affinity.clone()
    counts = [cl["pairs"].shape[0] for cl in closures]
    for i, cl in enumerate(closures):
        world.properties[i, num_nat : num_nat + counts[i]] = cl["props"][num_nat - 1 :]
        world.field_matter_affinity[i, num_nat : num_nat + counts[i]] = cl["sens"][
            num_nat - 1 :
        ]
    fr = torch.zeros(world.batch_size, max(max(counts), 1), device=world.device)
    valid = torch.zeros_like(fr, dtype=torch.bool)
    a = torch.ones(world.batch_size, dtype=torch.long, device=world.device)
    b = torch.ones_like(a)
    for k in range(fr.shape[1]):
        for i, cl in enumerate(closures):
            if k < counts[i]:
                a[i], b[i] = cl["pairs"][k, 0], cl["pairs"][k, 1]
                valid[i, k] = True
        score = world.gate_score_map(a, b)
        fr[:, k] = (score > 0).flatten(1).float().mean(1)
    world.properties, world.field_matter_affinity = saved
    return fr, valid


def _gate_snapshots(world, k=5, every=25):
    # field-state snapshots for scoring gate redraws; the dynamics don't
    # depend on the gate law, so one trajectory serves every redraw
    snaps = []
    for _ in range(k):
        for t in range(every):
            world.step(t)
        snaps.append((world.fields.clone(), world.rms_ema.clone()))
    return snaps


def _gate_over_snapshots(world, closures, snaps):
    saved = world.fields, world.rms_ema
    frs = []
    for fields, rms in snaps:
        world.fields, world.rms_ema = fields, rms
        fr, valid = gate_fracs(world, closures)
        frs.append(fr)
    world.fields, world.rms_ema = saved
    return torch.stack(frs), valid  # k, b, recipes


def _gate_stats(fr, valid):
    # per-universe over k snapshots: frac_gates_ever_open = union coverage (every
    # recipe opens at some snapshot), min_frac_gates_open = worst per-snapshot open
    # fraction (the frontier never fully closes), median_gate_open_area =
    # median over recipes of each recipe's best-snapshot open area
    frac_gates_ever_open, min_frac_gates_open, median_gate_open_area = [], [], []
    for i in range(fr.shape[1]):
        v = valid[i]
        if not v.any():
            zero = fr.new_tensor(0.0)
            frac_gates_ever_open.append(zero)
            min_frac_gates_open.append(zero)
            median_gate_open_area.append(zero)
            continue
        f = fr[:, i, v]  # k, recipes
        frac_gates_ever_open.append((f > 0).any(0).float().mean())
        min_frac_gates_open.append((f > 0).float().mean(1).min())
        median_gate_open_area.append(f.max(0).values.median())
    return torch.stack(frac_gates_ever_open), torch.stack(min_frac_gates_open), torch.stack(median_gate_open_area)


def generate_pool(path, n, batch_size=48, device="cpu", band=(0.1, 0.65), **world_kwargs):
    # batched universe generation: dynamics filters once, gate redrawn
    # blockwise; accepted universes appended to a growing .pt file
    from .env import DEFAULT_WORLD_KWARGS
    from .world import AlienCraftWorld

    records = torch.load(path) if os.path.exists(path) else []
    kwargs = {**DEFAULT_WORLD_KWARGS, **world_kwargs}
    world = AlienCraftWorld(batch_size=batch_size, device=device, **kwargs)
    with torch.no_grad():
        while len(records) < n:
            world.reset()
            stats = edge_stats(world)
            ok = accept(
                dict(
                    stats,
                    frac_gates_ever_open=torch.ones_like(stats["frac_gates_ever_open"]),
                    min_frac_gates_open=torch.ones_like(stats["min_frac_gates_open"]),
                    median_gate_open_area=torch.zeros_like(stats["median_gate_open_area"]),
                ),
                band,
            )
            cl = closure(world)
            snaps = _gate_snapshots(world)
            fr, valid = _gate_over_snapshots(world, cl, snaps)
            frac_gates_ever_open, min_frac_gates_open, median_gate_open_area = _gate_stats(fr, valid)
            gate_ok = (frac_gates_ever_open >= 0.999) & (min_frac_gates_open >= 0.25) & (median_gate_open_area <= 0.6)
            for _ in range(60):
                need = ok & ~gate_ok
                if not need.any():
                    break
                old = (
                    world.gate_tensor.clone(),
                    world.gate_act.clone(),
                    world.gate_ema_beta.clone(),
                )
                world._init_gate()
                keep = ~need
                world.gate_tensor[keep] = old[0][keep]
                world.gate_act[keep] = old[1][keep]
                world.gate_ema_beta[keep] = old[2][keep]
                fr, valid = _gate_over_snapshots(world, cl, snaps)
                frac_gates_ever_open, min_frac_gates_open, median_gate_open_area = _gate_stats(fr, valid)
                gate_ok = gate_ok | (
                    need & (frac_gates_ever_open >= 0.999) & (min_frac_gates_open >= 0.25) & (median_gate_open_area <= 0.6)
                )
            for i in (ok & gate_ok).nonzero().flatten().tolist():
                records.append(snapshot_laws(world, i))
            torch.save(records, path)
            print(f"pool: {len(records)}/{n}", flush=True)
    return records


def sample_edge_world(world, band=(0.1, 0.65), tries=288, gate_tries=60):
    # dynamics + closure filters don't depend on the gate law, so the gate
    # is resampled blockwise on worlds that already passed
    with torch.no_grad():
        for i in range(tries):
            world.reset()
            stats = edge_stats(world)
            stats_sans_gate = dict(
                stats,
                frac_gates_ever_open=torch.ones_like(stats["frac_gates_ever_open"]),
                min_frac_gates_open=torch.ones_like(stats["min_frac_gates_open"]),
                median_gate_open_area=torch.zeros_like(stats["median_gate_open_area"]),
            )
            if not bool(accept(stats_sans_gate, band)[0]):
                continue
            cl = closure(world)
            snaps = _gate_snapshots(world)
            for _ in range(gate_tries):
                fr, valid = _gate_over_snapshots(world, cl, snaps)
                frac_gates_ever_open, min_frac_gates_open, median_gate_open_area = _gate_stats(fr, valid)
                if (
                    bool(frac_gates_ever_open[0] >= 0.999)
                    and float(min_frac_gates_open[0]) >= 0.25
                    and float(median_gate_open_area[0]) <= 0.6
                ):
                    return i + 1
                world._init_gate()
    raise RuntimeError(f"no edge universe found in {tries} samples")


def accept(stats, band=(0.1, 0.65)):
    return (
        (stats["complexity"] >= band[0]) & (stats["complexity"] <= band[1])
        & (stats["persistence"] >= 0.2) & (stats["persistence"] <= 0.995)
        & (stats["linearity"] >= 0.4) & (stats["linearity"] <= 0.97)
        & (stats["sensitivity"] >= 0.05)
        & (stats["mobility"] >= 0.05) & (stats["mobility"] <= 0.95)
        & (stats["spread"] >= 30.0)
        & (stats["crafted"] >= 10) & (stats["crafted"] <= 300)
        & (stats["depth"] >= 3)
        & (stats["frac_gates_ever_open"] >= 0.999)
        & (stats["min_frac_gates_open"] >= 0.25)
        & (stats["median_gate_open_area"] <= 0.6)
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


def edge_stats(world, spin: int = 100, steps: int = 200, every: int = 4):
    # gzip complexity, one-step persistence, linear predictability, source
    # sensitivity vs a silent ghost, closure reach/depth/colour spread
    frames, traj = [], []
    with torch.no_grad():
        for t in range(spin):
            world.step(t)
        rms = world.fields.pow(2).mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
        cl = closure(world)
        snaps = [(world.fields.clone(), world.rms_ema.clone())]
        for t in range(steps):
            world.step(spin + t)
            traj.append(world.fields.clone())
            if t % 50 == 0:
                snaps.append((world.fields.clone(), world.rms_ema.clone()))
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
        # silent ghost: zero the emission series' amplitudes so matter
        # sources nothing, then measure how much the fields care
        ghost.emission_act[:, 0] = 0.0
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
        snaps.append((world.fields.clone(), world.rms_ema.clone()))
        fr, valid = _gate_over_snapshots(world, cl, snaps)
        frac_gates_ever_open, min_frac_gates_open, median_gate_open_area = _gate_stats(fr, valid)
    frames = torch.stack(frames)
    complexity = torch.tensor(
        [
            len(gzip.compress(frames[:, b].numpy().tobytes())) / frames[:, b].numel()
            for b in range(world.batch_size)
        ],
        device=world.device,
    )
    # matter mobility: fraction of initially-occupied cells whose content
    # ever changes (the gzip band can't see this — grid is 1 of 4 channels
    # and the churning field bytes mask a frozen one)
    g = frames[:, :, 0]  # T, b, h, w
    alive = g[0] > 0
    moved = ((g != g[0]).any(0) & alive).flatten(1).sum(1)
    mobility = (moved / alive.flatten(1).sum(1).clamp_min(1)).to(world.device)
    dev = world.device
    return dict(
        complexity=complexity,
        persistence=persistence,
        linearity=linearity,
        sensitivity=sens,
        spread=torch.tensor([c["spread"] for c in cl], device=dev),
        mobility=mobility,
        crafted=torch.tensor([c["pairs"].shape[0] for c in cl], device=dev),
        capped=torch.tensor([c["capped"] for c in cl], device=dev),
        depth=torch.tensor([int(c["depth"].max()) for c in cl], device=dev),
        frac_gates_ever_open=frac_gates_ever_open,
        min_frac_gates_open=min_frac_gates_open,
        median_gate_open_area=median_gate_open_area,
    )
