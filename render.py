import torch
import pygame
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


def _color_for_type(cell_type):
    if cell_type < len(COLORS):
        return COLORS[cell_type]

    hue = (cell_type * 0.61803398875) % 1.0
    color = pygame.Color(0)
    color.hsva = (hue * 360, 48, 68, 100)
    return color


def render(universe, batch_index=0, cell_size=6):
    pygame.init()
    screen = _make_screen(universe.grid[batch_index], cell_size)
    _draw_grid(screen, universe.grid[batch_index], cell_size)
    pygame.display.flip()
    return screen


def render_animation(universe, steps=10, batch_index=0, cell_size=6, fps=4):
    pygame.init()
    screen = _make_screen(universe.grid[batch_index], cell_size)
    clock = pygame.time.Clock()

    for step in range(1, steps + 1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        universe.step(step)
        _draw_grid(screen, universe.grid[batch_index], cell_size)
        pygame.display.set_caption(f"aliencraft step {step}")
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def _draw_grid(screen, grid, cell_size):
    grid = grid.detach().cpu().long()
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            color = _color_for_type(int(grid[x, y]))
            rect = (x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)


def _make_screen(grid, cell_size):
    width, height = grid.shape[0], grid.shape[1]
    return pygame.display.set_mode((width * cell_size, height * cell_size))


if __name__ == "__main__":
    torch.inference_mode()
    universe = Universe(
        batch_size=1,
        width=100,
        height=100,
        num_types=5,
        num_properties=10,
        num_fields=2,
    )  # create a universe
    universe.seed_universe()
    render_animation(universe, steps=100)
