import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

# Mapping Source: https://stackoverflow.com/questions/41922629/convert-text-to-braille-unicode-in-python
ASCII_TO_BRAILLE_UNICODE_MAPPING = {
    " ": "⠀",  # Note: Key is a space. Value is not a space.
    "!": "⠮",
    '"': "⠐",
    "#": "⠼",
    "$": "⠫",
    "%": "⠩",
    "&": "⠯",
    "": "⠄",
    "(": "⠷",
    ")": "⠾",
    "*": "⠡",
    "+": "⠬",
    ",": "⠠",
    "-": "⠤",
    ".": "⠨",
    "/": "⠌",
    "0": "⠴",
    "1": "⠂",
    "2": "⠆",
    "3": "⠒",
    "4": "⠲",
    "5": "⠢",
    "6": "⠖",
    "7": "⠶",
    "8": "⠦",
    "9": "⠔",
    ":": "⠱",
    ";": "⠰",
    "<": "⠣",
    "=": "⠿",
    ">": "⠜",
    "?": "⠹",
    "@": "⠈",
    "a": "⠁",
    "b": "⠃",
    "c": "⠉",
    "d": "⠙",
    "e": "⠑",
    "f": "⠋",
    "g": "⠛",
    "h": "⠓",
    "i": "⠊",
    "j": "⠚",
    "k": "⠅",
    "l": "⠇",
    "m": "⠍",
    "n": "⠝",
    "o": "⠕",
    "p": "⠏",
    "q": "⠟",
    "r": "⠗",
    "s": "⠎",
    "t": "⠞",
    "u": "⠥",
    "v": "⠧",
    "w": "⠺",
    "x": "⠭",
    "y": "⠽",
    "z": "⠵",
    "[": "⠪",
    "\\": "⠳",
    "]": "⠻",
    "^": "⠘",
    "_": "⠸",
}
assert len(set(ASCII_TO_BRAILLE_UNICODE_MAPPING.keys())) == len(
    ASCII_TO_BRAILLE_UNICODE_MAPPING
), "Braille unicode mapping must be unique! Keys have duplicates."
assert len(set(ASCII_TO_BRAILLE_UNICODE_MAPPING.values())) == len(
    ASCII_TO_BRAILLE_UNICODE_MAPPING
), "Braille unicode mapping must be unique! Values have duplicates."


class BrailleCell:
    """Representation of a single braille cell."""

    def __init__(self, cell_input: str | int) -> None:
        """Create a braille cell.

        Args:
            cell_input (str | int): Braille cell representation. Can be either
                a single character, or an int representing the braille cell in
                binary, or a braille unicode character.

        """
        if isinstance(cell_input, str):
            if len(cell_input) != 1:
                msg = "String input must be a single character."
                raise ValueError(msg)

            # If it's in the 0x2800 to 0x28FF range, it's a braille unicode
            # character.
            if ord(cell_input) in range(0x2800, 0x28FF + 1):
                _cell_int_repr = ord(cell_input) - 0x2800
            else:
                # Convert the character to its braille unicode representation.
                cell_unicode_char = ASCII_TO_BRAILLE_UNICODE_MAPPING[
                    # TODO(KilowattSynthesis): Implement case-sensitive logic.
                    cell_input.lower()
                ]
                _cell_int_repr = ord(cell_unicode_char) - 0x2800

        elif isinstance(cell_input, int):
            _cell_int_repr = cell_input
        else:
            msg = "Input must be a string or an integer."
            raise TypeError(msg)

        # Set attribute(s).
        self.cell_int_repr: int = _cell_int_repr

    @property
    def cell_unicode_character(self) -> str:
        """Get the unicode character representation of the braille cell."""
        return chr(self.cell_int_repr + 0x2800)

    @property
    def cell_ascii_character(self) -> str:
        """Get the ASCII character representation of the braille cell."""
        for (
            ascii_char,
            braille_char,
        ) in ASCII_TO_BRAILLE_UNICODE_MAPPING.items():
            if braille_char == self.cell_unicode_character:
                return ascii_char

        # If we get here, the cell is not in the mapping.
        msg = (
            "ASCII character not found for braille cell "
            f"|{self.cell_unicode_character}|"
        )
        raise ValueError(msg)

    @property
    def dot_count(self) -> Literal[6, 8]:
        """Get the "type" of braille cell (either 6-dot or 8-dot)."""
        if self.cell_int_repr > 0b111111:  # noqa: PLR2004
            return 8
        return 6

    def get_raised_xy_point_list(
        self, dots_per_cell_mode: Literal[6, 8]
    ) -> list[tuple[float, float]]:
        """Get the XY coordinates of the raised dots in the braille cell.

        Args:
            dots_per_cell_mode (Literal[6, 8]): The number of dots per cell
                (either 6 or 8).

        Returns:
            list[tuple[float, float]]: A list of tuples representing the XY
                coordinates of the raised dots. Assuming the dots are all
                "1 unit" apart. In 6-dot mode, the dots are arranged in a 2x3
                grid with the middle dots at Y=0. In 8-dot mode, the dots
                are arranged in a 2x4 grid with the inner dots at Y=0.5 and
                Y=-0.5. Top-left corner is (-X, +Y), like Quadrant 1.

        """
        if dots_per_cell_mode not in (6, 8):
            msg = "Invalid braille cell mode. Must be either 6 or 8."
            raise ValueError(msg)

        dot_to_coord_mapping_mode_6 = {
            1: (-0.5, 1.0),
            2: (-0.5, 0.0),
            3: (-0.5, -1.0),
            4: (0.5, 1.0),
            5: (0.5, 0.0),
            6: (0.5, -1.0),
            7: (-0.5, -2.0),
            8: (0.5, -2.0),
        }

        dot_coords = [
            dot_to_coord_mapping_mode_6[dot_num]
            for dot_num in dot_to_coord_mapping_mode_6
            if (self.cell_int_repr >> (dot_num - 1)) & 1 > 0
        ]

        # If 8-dot mode, shift all north by 0.5.
        if dots_per_cell_mode == 8:  # noqa: PLR2004
            for i, (x, y) in enumerate(dot_coords):
                dot_coords[i] = (x, y + 0.5)

        return dot_coords


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
    stick_face_width: float = 28.0
    stick_face_thickness: float = 2.0
    stick_corner_radius: float = 1.0

    # Braille dimensions.
    dot_pitch_x: float = 2.5
    dot_pitch_y: float = 2.5
    cell_pitch_x: float = 6.0
    dot_diameter: float = 1.5
    dot_height: float = 1.0

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

    @property
    def normalized_cells_each_face(self) -> list[list[BrailleCell]]:
        """Get the normalized cells for each face."""
        return [
            [BrailleCell(cell) for cell in face]
            for face in self.cells_each_face
        ]

    @property
    def dots_per_cell_mode_each_face(self) -> list[Literal[6, 8]]:
        """Get the dots per cell mode for each face."""
        return [
            max(cell.dot_count for cell in face)
            for face in self.normalized_cells_each_face
        ]


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

    # Find the faces of the stick, and draw braille on them (treating them as
    # sketch surfaces).
    # Select the side_count largest faces.
    faces = sorted(stick.faces(), key=lambda f: f.area, reverse=True)
    faces = faces[: spec.side_count]
    assert len(faces) == spec.side_count, (
        f"Expected {spec.side_count} faces, got {len(faces)}"
    )
    for _face_num, (face, normalized_cell_list, dot_mode) in enumerate(
        zip(
            faces,
            spec.normalized_cells_each_face,
            spec.dots_per_cell_mode_each_face,
            strict=True,
        )
    ):
        # Create a sketch on the face.
        sketch = bd.Sketch(None)

        # Draw the braille dots.
        for cell_num, normalized_cell in enumerate(normalized_cell_list):
            for (
                dot_offset_x,
                dot_offset_y,
            ) in normalized_cell.get_raised_xy_point_list(
                dots_per_cell_mode=dot_mode
            ):
                dot_x = (
                    dot_offset_x * spec.dot_pitch_x
                    + spec.cell_pitch_x * cell_num
                    - (spec.cell_pitch_x * (spec.cell_count - 1) / 2)
                )
                dot_y = dot_offset_y * spec.dot_pitch_y

                sketch += (
                    bd.Plane(face)  # pyright: ignore[reportOperatorIssue]
                    * bd.Pos((dot_y, -dot_x))
                    * bd.Circle(
                        radius=spec.dot_diameter / 2,
                    )
                )

        # Extrude the sketch to create the braille dots.
        # assert isinstance(sketch, bd.Compound)  # Type checking.
        dot_extrude = bd.extrude(
            sketch,  # pyright: ignore[reportArgumentType]
            amount=spec.dot_height,
            clean=True,
        )
        p += dot_extrude
        logger.debug(f"Done with face {_face_num + 1} of {spec.side_count}")

    return p


def test_inscribed_radius() -> None:
    """Test the inscribed radius function."""
    # Comparison
    assert math.isclose(
        inscribed_radius(n=6, side_length=2), math.sqrt(3), rel_tol=1e-9
    )
    assert math.isclose(
        1 / math.sqrt(3), inscribed_radius(n=3, side_length=2), rel_tol=1e-9
    )


if __name__ == "__main__":
    test_inscribed_radius()

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

        bd.export_stl(part, str(export_folder / f"{name}.stl"))
        bd.export_step(part, str(export_folder / f"{name}.step"))
