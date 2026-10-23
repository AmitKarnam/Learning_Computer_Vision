"""Resize and combine two images using saturated pixel addition."""

import cv2

from examples.common import load_image, make_contact_sheet, save_image, show_images


def main() -> None:
    background = load_image("rectangle.jpg")
    overlay = load_image("checkerboard_color.png")
    height, width = background.shape[:2]
    overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)
    combined = cv2.addWeighted(background, 0.5, overlay, 0.5, 0)
    result = make_contact_sheet(
        [("Background", background), ("Overlay", overlay), ("50/50 blend", combined)],
        columns=3,
    )
    save_image("image_operations/image_addition/blended_images.png", result)
    show_images([("Image addition", result)])


if __name__ == "__main__":
    main()
