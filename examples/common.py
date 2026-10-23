"""Shared input, output, and display helpers for the examples."""

import os
import sys
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"


def image_path(filename: str) -> Path:
    """Resolve a bundled or user-configured input image."""
    configured_directory = os.environ.get("CV_LEARNING_IMAGE_DIR")
    if not configured_directory:
        raise RuntimeError(
            "Set CV_LEARNING_IMAGE_DIR to the directory containing input images."
        )
    image_directory = Path(configured_directory).expanduser()
    return image_directory / filename


def load_image(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    path = image_path(filename)
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def output_path(relative_path: str) -> Path:
    path = OUTPUT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_image(relative_path: str, image: np.ndarray) -> Path:
    path = output_path(relative_path)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save output image: {path}")
    print(f"Saved image: {path.relative_to(REPOSITORY_ROOT)}")
    return path


def make_contact_sheet(
    images: list[tuple[str, np.ndarray]],
    columns: int = 2,
    tile_size: tuple[int, int] = (480, 360),
) -> np.ndarray:
    """Arrange labeled images into a consistent BGR contact sheet."""
    if not images:
        raise ValueError("At least one image is required for a contact sheet.")

    tile_width, tile_height = tile_size
    rows = (len(images) + columns - 1) // columns
    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3),
        245,
        dtype=np.uint8,
    )

    for index, (label, image) in enumerate(images):
        if image is None:
            raise ValueError(f"Image for '{label}' is empty.")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        image_height, image_width = image.shape[:2]
        scale = min(
            tile_width / image_width,
            (tile_height - 32) / image_height,
        )
        resized = cv2.resize(
            image,
            (max(1, int(image_width * scale)), max(1, int(image_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(index, columns)
        x_offset = column * tile_width
        y_offset = row * tile_height
        y_image = y_offset + 32 + (tile_height - 32 - resized.shape[0]) // 2
        x_image = x_offset + (tile_width - resized.shape[1]) // 2
        sheet[
            y_image:y_image + resized.shape[0],
            x_image:x_image + resized.shape[1],
        ] = resized
        cv2.putText(
            sheet,
            label,
            (x_offset + 12, y_offset + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    return sheet


def show_images(images: list[tuple[str, np.ndarray]]) -> None:
    """Display saved results only when the caller supplies --show."""
    if "--show" not in sys.argv:
        return
    for title, image in images:
        cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
