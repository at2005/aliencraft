import argparse
from pathlib import Path

from PIL import Image
import torch

from world import Universe


def sample_perlin(width: int, height: int, scale: float) -> torch.Tensor:
    universe = Universe(
        batch_size=1,
        width=width,
        height=height,
        num_types=2,
        num_properties=1,
        num_fields=1,
    )
    return universe.perlin(scale=scale)[0]


def normalize_for_display(noise: torch.Tensor, percentile: float) -> torch.Tensor:
    values = noise.detach().float()
    limit = torch.quantile(values.abs().flatten(), percentile)

    if not torch.isfinite(limit) or limit <= 1e-8:
        return torch.full_like(values, 0.5)

    return (0.5 + 0.5 * values / limit).clamp(0.0, 1.0)


def save_image(display_values: torch.Tensor, output_path: Path, zoom: int) -> None:
    image = (display_values.detach().cpu().numpy() * 255).round().astype("uint8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(image)
    if zoom > 1:
        pil_image = pil_image.resize(
            (pil_image.width * zoom, pil_image.height * zoom),
            resample=Image.Resampling.NEAREST,
        )
    pil_image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Universe.perlin() to a PNG.")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.995,
        help="Symmetric contrast percentile for display normalization.",
    )
    parser.add_argument("--zoom", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("perlin.png"))
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("width and height must be positive")
    if not 0.0 < args.percentile <= 1.0:
        raise SystemExit("percentile must be in (0, 1]")
    if args.zoom <= 0:
        raise SystemExit("zoom must be positive")

    noise = sample_perlin(width=args.width, height=args.height, scale=args.scale)
    display_values = normalize_for_display(noise, args.percentile)
    save_image(display_values, args.out, args.zoom)

    print(f"saved {args.out.resolve()}")
    print(
        "noise "
        f"min={noise.min().item():.5f} "
        f"max={noise.max().item():.5f} "
        f"mean={noise.mean().item():.5f} "
        f"std={noise.std().item():.5f}"
    )
    if noise.std() <= 1e-8:
        print(
            "flat image: this sampler hits Perlin lattice points for this scale, "
            "so every dot product is zero"
        )


if __name__ == "__main__":
    main()
