from dataclasses import dataclass
from pathlib import Path

import build123d as bd
from build123d_ease import show
from loguru import logger


@dataclass
class Part1Spec:
    """Specification for part1."""

    part1_radius: float = 20

    def __post_init__(self) -> None:
        """Post initialization checks."""
        assert self.part1_radius > 0, "part1_radius must be positive"


def make_part1(spec: Part1Spec) -> bd.Part | bd.Compound:
    """Create a CAD model of part1."""
    p = bd.Part(None)

    p += bd.Cylinder(radius=spec.part1_radius, height=20)

    return p


if __name__ == "__main__":
    parts = {
        "part1": show(make_part1(Part1Spec())),
    }

    logger.info("Showing CAD model(s)")

    (export_folder := Path(__file__).parent.with_name("build")).mkdir(
        exist_ok=True
    )
    for name, part in parts.items():
        assert isinstance(part, bd.Part | bd.Solid | bd.Compound), (
            f"{name} is not an expected type ({type(part)})"
        )
        if not part.is_manifold:
            logger.warning(f"Part '{name}' is not manifold")

        bd.export_stl(part, str(export_folder / f"{name}.stl"))
        bd.export_step(part, str(export_folder / f"{name}.step"))
