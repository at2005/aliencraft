# generation filters: a sampled universe is kept only if these pass
import copy
import gzip

import torch
import torch.nn.functional as F


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
        for t in range(steps):
            world.step(spin + t)
            traj.append(world.fields.clone())
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
    frames = torch.stack(frames)
    complexity = torch.tensor(
        [
            len(gzip.compress(frames[:, b].numpy().tobytes())) / frames[:, b].numel()
            for b in range(world.batch_size)
        ]
    )
    return dict(
        complexity=complexity,
        persistence=persistence,
        linearity=linearity,
        sensitivity=sens,
        spread=spread,
    )
