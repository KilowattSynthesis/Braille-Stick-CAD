import json
import math
from dataclasses import dataclass
from pathlib import Path

import build123d as bd
from build123d_ease import show
from loguru import logger

# Braille dot numbers:
# 1 4
# 2 5
# 3 6
# 7 8 # <- These two are optional.

# An int represents a cell by the 8-bit binary representation of the raised
# braille dots.
# Do a bitwise OR together for all raised dots, like:
# (1 << (dot_number - 1)) | (1 << (dot_number - 1)) | ...


@dataclass
class BrailleStickSpec:
    """Specification for braille_stick."""

    cells_each_face: tuple[
        list[str | int], list[str | int], list[str | int]
    ] = (
        list("ABCDEFGHI"),
        list("JKLMNOPQR"),
        list("STUVWXYZ#"),
    )

    # Stick dimensions.
    stick_face_width: float = 16.0
    stick_face_thickness: float = 2.0
    stick_corner_radius: float = 1.0

    # Braille dimensions.
    dot_pitch_x: float = 2.5
    dot_pitch_y: float = 2.5
    cell_pitch_x: float = 6.0

    def __post_init__(self) -> None:
        """Post initialization checks."""
        data = {"polygon_inscribed_radius": self.polygon_inscribed_radius}
        logger.info(f"Braille stick data: {json.dumps(data, indent=4)}")

    @property
    def cell_count(self) -> int:
        """Number of braille cells."""
        return max(
            len(cells_on_face) for cells_on_face in self.cells_each_face
        )

    @property
    def total_length(self) -> float:
        """Total length of the braille stick."""
        return self.cell_count * self.cell_pitch_x + 10

    @property
    def side_count(self) -> int:
        """Number of sides of the polygon."""
        return len(self.cells_each_face)

    @property
    def polygon_inscribed_radius(self) -> float:
        """Inscribed radius of the polygon."""
        return inscribed_radius(
            n=self.side_count, side_length=self.stick_face_width
        )


def inscribed_radius(n: int, side_length: float) -> float:
    """Calculate the inscribed radius of a regular polygon.

    Args:
        n (int): Number of sides of the polygon.
        side_length (float): Length of each side.

    Returns:
        float: The radius of the inscribed circle.

    Raises:
        ValueError: If n is less than 3.

    """
    if n < 3:  # noqa: PLR2004
        msg = "A polygon must have at least 3 sides."
        raise ValueError(msg)
    return side_length / (2 * math.tan(math.pi / n))


def make_braille_stick(spec: BrailleStickSpec) -> bd.Part | bd.Compound:
    """Create a CAD model of braille_stick.

    Rod will extend in the +X direction. Text reads left-to-right (-X to +X).
    """
    p = bd.Part(None)

    polygon_face_outside = bd.RegularPolygon(
        radius=spec.polygon_inscribed_radius,
        side_count=spec.side_count,
    )
    polygon_inside_face = bd.RegularPolygon(
        radius=spec.polygon_inscribed_radius - spec.stick_face_thickness,
        side_count=spec.side_count,
    )
    polygon_face_outside = bd.fillet(
        polygon_face_outside.vertices(), radius=spec.stick_corner_radius
    )
    polygon_inside_face = bd.fillet(
        polygon_inside_face.vertices(), radius=spec.stick_corner_radius
    )
    polygon_outline = bd.Plane.YZ * bd.Sketch(
        polygon_face_outside - polygon_inside_face
    )
    assert isinstance(polygon_outline, bd.Sketch)  # Type checking mostly.

    # Create the stick by extruding the polygon outline.
    stick = bd.extrude(
        polygon_outline,
        amount=spec.total_length,
    )

    p += stick

    return p


if __name__ == "__main__":
    parts = {
        "braille_stick": show(make_braille_stick(BrailleStickSpec())),
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
