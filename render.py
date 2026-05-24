import pygame
from world import Universe


COLORS = [
    (70, 230, 120),  # bright green
    (70, 130, 230),  # bright blue
    (90, 220, 120),
    (240, 210, 80),
    (180, 90, 240),
    (240, 140, 70),
    (180, 220, 220),
    (120, 120, 240),
]


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

        universe.step()
        _draw_grid(screen, universe.grid[batch_index], cell_size)
        pygame.display.set_caption(f"aliencraft step {step}")
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def _draw_grid(screen, grid, cell_size):
    grid = grid.detach().cpu().long()
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            color = COLORS[int(grid[x, y]) % len(COLORS)]
            rect = (x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)


def _make_screen(grid, cell_size):
    width, height = grid.shape[0], grid.shape[1]
    return pygame.display.set_mode((width * cell_size, height * cell_size))


if __name__ == "__main__":
    universe = Universe(
        batch_size=1,
        width=100,
        height=100,
        num_types=5,
        num_properties=10,
        num_fields=3,
    )  # create a universe
    universe.seed_universe()
    render_animation(universe, steps=10)
