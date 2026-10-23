"""Explore resizing, flipping, cropping, and direct pixel edits."""

import cv2

from examples.common import load_image, make_contact_sheet, save_image, show_images


def main() -> None:
    image = load_image("ktm_adventure_390_x.jpeg")
    checkerboard = load_image("checkerboard_18x18.png", cv2.IMREAD_GRAYSCALE)

    edited_checkerboard = checkerboard.copy()
    edited_checkerboard[2:4, 2:4] = 127

    resized = cv2.resize(image, (300, 200), interpolation=cv2.INTER_CUBIC)
    cropped = image[100:200, 100:200]
    results = [
        ("Original", image),
        ("Resized", resized),
        ("Vertical flip", cv2.flip(image, 0)),
        ("Horizontal flip", cv2.flip(image, 1)),
        ("Both axes flipped", cv2.flip(image, -1)),
        ("100 x 100 crop", cropped),
        ("Edited checkerboard", edited_checkerboard),
    ]
    result = make_contact_sheet(results, columns=2)
    save_image("fundamentals/image_manipulation/transforms.png", result)
    show_images([("Image manipulations", result)])


if __name__ == "__main__":
    main()
