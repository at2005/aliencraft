import torch
import pygame
import colorsys
import argparse
from world import Universe


COLORS = [
    (0, 0, 0),  # empty
    (78, 121, 167),  # muted blue
    (89, 161, 79),  # moss
    (242, 142, 43),  # ochre
    (225, 87, 89),  # clay
    (118, 183, 178),  # mineral teal
    (176, 122, 161),  # mauve
    (156, 117, 95),  # umber
    (186, 176, 172),  # stone
    (237, 201, 72),  # amber
]


def render(universe, batch_index=0, cell_size=6):
    pygame.init()
    frame = _render_frame(universe, batch_index)
    screen = _make_screen(frame, cell_size)
    _draw_frame(screen, frame, cell_size)
    pygame.display.flip()
    return screen


def render_animation(universe, steps=10, batch_index=0, cell_size=6, fps=4):
    pygame.init()
    frame = _render_frame(universe, batch_index)
    screen = _make_screen(frame, cell_size)
    clock = pygame.time.Clock()

    for step in range(1, steps + 1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        universe.step(step)
        frame = _render_frame(universe, batch_index)
        _draw_frame(screen, frame, cell_size)
        pygame.display.set_caption(f"aliencraft step {step}")
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def _render_frame(universe, batch_index):
    noise = universe.render()[batch_index]
    material_frame = _material_frame(universe, batch_index, noise.shape)
    base_colors = _palette(int(universe.grid.max().item()) + 1)[material_frame]
    shade = _shade(noise).unsqueeze(-1)
    return (base_colors * shade).clamp(0, 255).byte()


def _draw_frame(screen, frame, cell_size):
    surface = _make_surface(frame)
    if cell_size != 1:
        width, height = surface.get_size()
        surface = pygame.transform.scale(
            surface, (width * cell_size, height * cell_size)
        )
    screen.blit(surface, (0, 0))


def _make_surface(frame):
    pixels = frame.detach().cpu().byte()
    return pygame.surfarray.make_surface(pixels.permute(1, 0, 2).numpy())


def _make_screen(frame, cell_size):
    height, width = frame.shape[:2]
    return pygame.display.set_mode((width * cell_size, height * cell_size))


def _material_frame(universe, batch_index, frame_shape):
    grid = universe.grid[batch_index].detach().cpu().long()
    sprite_resolution = universe.sprite_resolution
    materials = (
        grid.unsqueeze(-1)
        .unsqueeze(-1)
        .expand(*grid.shape, sprite_resolution, sprite_resolution)
    )
    return materials.permute(0, 2, 1, 3).reshape(frame_shape)


def _shade(noise):
    noise = noise.detach().cpu().float()
    noise_min = noise.amin()
    noise_max = noise.amax()
    noise = (noise - noise_min) / (noise_max - noise_min + 1e-8)
    return 0.45 + noise * 0.75


def _palette(num_colors):
    colors = []
    for cell_type in range(num_colors):
        if cell_type < len(COLORS):
            colors.append(COLORS[cell_type])
        else:
            hue = (cell_type * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.48, 0.68)
            colors.append((round(r * 255), round(g * 255), round(b * 255)))
    return torch.tensor(colors, dtype=torch.float32)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sprite-resolution",
        type=int,
        default=4,
        help="pixels per universe cell in the generated material sprite texture",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    torch.inference_mode()
    universe = Universe(
        batch_size=1,
        width=200,
        height=200,
        num_types=10,
        num_common_types=4,
        num_sparse_types=2,
        num_properties=10,
        num_fields=2,
        sprite_resolution=args.sprite_resolution,
    )  # create a universe
    universe.seed_universe()
    render_animation(universe, steps=1000)
