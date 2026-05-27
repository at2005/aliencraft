import pygame
import argparse
from world import Universe
import torch


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
    return universe.get_obs_for_agent()[batch_index].byte()


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
