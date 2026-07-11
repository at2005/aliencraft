# play AlienCraft: WASD move, E pick, Q place, Tab cycle inventory,
# Space craft (uses the cells left/right of you), R new universe, Esc quit
#   uv run python play.py
import math

import numpy as np
import pygame
import torch

from aliencraft import AlienCraftWorld
from aliencraft.filter import sample_edge_world

SCALE = 3
SIDEBAR = 220
ISQRT2 = 1.0 / math.sqrt(2)
OCTANT = {
    "craft": torch.tensor([ISQRT2, ISQRT2]),
    "pick": torch.tensor([ISQRT2, -ISQRT2]),
    "place": torch.tensor([-ISQRT2, ISQRT2]),
    "noop": torch.tensor([-ISQRT2, -ISQRT2]),
}

torch.manual_seed(0)
world = AlienCraftWorld(
    batch_size=1, device="cpu", width=64, height=64, num_types=100,
    num_common_types=4, num_sparse_types=1, num_properties=3, num_fields=3,
    sprite_resolution=4, visual_field_size=64, driven_fields=True,
)


def new_universe():
    print("sampling universe...", flush=True)
    print(f"accepted after {sample_edge_world(world, tries=999)} tries")


new_universe()


def action_for(direction, place_type=None):
    a = torch.zeros(1, 2 + world.num_types)
    a[0, :2] = torch.linalg.inv(world.actuators[0]) @ direction
    if place_type is not None:
        a[0, 2 + place_type] = 1.0
    return a


def neighbours():
    x, y = world.agent_position[0].tolist()
    left = int(world.grid[0, (x - 1) % world.width, y])
    right = int(world.grid[0, (x + 1) % world.width, y])
    return left, right


pygame.init()
world_px = world.width * world.sprite_resolution * SCALE
screen = pygame.display.set_mode((world_px + SIDEBAR, world_px))
pygame.display.set_caption("AlienCraft")
font = pygame.font.SysFont("monospace", 15)
clock = pygame.time.Clock()
pygame.key.set_repeat(180, 70)

selected = 0
flash, flash_t = "", 0
step = 0
SHIMMER_BOOST = 3.0  # display-only exaggeration of the field tint
running = True
with torch.no_grad():
    while running:
        pending = action_for(OCTANT["noop"])
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type != pygame.KEYDOWN:
                continue
            k = ev.key
            if k == pygame.K_ESCAPE:
                running = False
            elif k == pygame.K_d:
                pending = action_for(torch.tensor([1.0, 0.0]))
            elif k == pygame.K_a:
                pending = action_for(torch.tensor([-1.0, 0.0]))
            elif k == pygame.K_s:
                pending = action_for(torch.tensor([0.0, 1.0]))
            elif k == pygame.K_w:
                pending = action_for(torch.tensor([0.0, -1.0]))
            elif k == pygame.K_e:
                pending = action_for(OCTANT["pick"])
            elif k == pygame.K_q:
                held = (world.agent_inventory[0] > 0).nonzero().flatten().tolist()
                if held:
                    pending = action_for(
                        OCTANT["place"], place_type=held[selected % len(held)]
                    )
            elif k == pygame.K_TAB:
                selected += 1
            elif k == pygame.K_SPACE:
                before = world.tech_tree_progress[0].sum().item()
                l, r = neighbours()
                pending = action_for(OCTANT["craft"])
                world.step(step, pending)
                step += 1
                after = world.tech_tree_progress[0].sum().item()
                if after > before:
                    flash, flash_t = f"NEW TYPE from {l}+{r}!", 40
                elif l and r:
                    flash, flash_t = "craft failed", 20
                pending = None
            elif k == pygame.K_r:
                new_universe()
                flash, flash_t = "new universe", 30
        if pending is not None:
            world.step(step, pending)
            step += 1
        # time-warp: run extra noop steps per frame so slow universes
        # visibly breathe at human timescales
        for _ in range(3):
            world.step(step, action_for(OCTANT["noop"]))
            step += 1

        world.field_colour_directions *= SHIMMER_BOOST
        frame = world.get_obs_for_agent(agent_view=False)[0]
        world.field_colour_directions /= SHIMMER_BOOST
        img = frame.clamp(0, 255).byte().numpy()
        surf = pygame.transform.scale(
            pygame.surfarray.make_surface(img), (world_px, world_px)
        )
        screen.fill((12, 12, 16))
        screen.blit(surf, (0, 0))

        # agent marker, with a pulsing ring when a craft is ready here
        ax, ay = world.agent_position[0].tolist()
        cell = world.sprite_resolution * SCALE
        pygame.draw.rect(
            screen, (255, 255, 255), (ax * cell, ay * cell, cell, cell), 2
        )
        l, r = neighbours()
        ready = (
            l and r
            and int(world.craft_map[0, l, r]) != -1
            and bool(world.craft_glow_gate(torch.tensor([l]), torch.tensor([r]))[0])
        )
        if ready:
            radius = int(cell * (1.2 + 0.35 * math.sin(step * 0.25)))
            pygame.draw.circle(
                screen, (255, 210, 60),
                (ax * cell + cell // 2, ay * cell + cell // 2), radius, 3,
            )

        # sidebar: discoveries, craft readout, inventory
        ui = [f"discovered: {int(world.tech_tree_progress[0].sum())}/95"]
        l, r = neighbours()
        if l and r:
            recipe = int(world.craft_map[0, l, r])
            gate = bool(
                world.craft_glow_gate(
                    torch.tensor([l]), torch.tensor([r])
                )[0]
            )
            ui.append(f"pair {l}+{r}:")
            ui.append(f"  recipe: {'-> ' + str(recipe) if recipe != -1 else 'none'}")
            ui.append(f"  gate: {'OPEN' if gate else 'closed'}")
        else:
            ui.append("stand between two")
            ui.append("materials to craft")
        ui.append("")
        ui.append("inventory (Tab, Q):")
        held = (world.agent_inventory[0] > 0).nonzero().flatten().tolist()
        for i, t in enumerate(held[:12]):
            mark = ">" if i == selected % max(1, len(held)) else " "
            ui.append(f"{mark} type {t} x{int(world.agent_inventory[0, t])}")
        for i, line in enumerate(ui):
            screen.blit(
                font.render(line, True, (230, 230, 230)),
                (world_px + 10, 12 + 18 * i),
            )
        for i, t in enumerate(held[:12]):
            colour = world.colour_palette[0, t].tolist()
            pygame.draw.rect(
                screen, colour,
                (world_px + SIDEBAR - 28, 12 + 18 * (len(ui) - len(held[:12]) + i), 14, 14),
            )
        if flash_t > 0:
            flash_t -= 1
            screen.blit(
                font.render(flash, True, (255, 220, 80)), (10, world_px - 26)
            )

        pygame.display.flip()
        clock.tick(15)

pygame.quit()
