"""Demonstrate brightness, contrast, and bitwise image operations."""

import cv2

from examples.common import load_image, make_contact_sheet, save_image, show_images


def main() -> None:
    image = load_image("rocks.jpg")
    brighter = cv2.convertScaleAbs(image, alpha=1.0, beta=50)
    lower_contrast = cv2.convertScaleAbs(image, alpha=0.8, beta=0)
    higher_contrast = cv2.convertScaleAbs(image, alpha=1.2, beta=0)

    rectangle = load_image("rectangle.jpg")
    circle = load_image("circle.jpg")
    circle = cv2.resize(circle, (rectangle.shape[1], rectangle.shape[0]))
    bitwise_and = cv2.bitwise_and(rectangle, circle)
    bitwise_or = cv2.bitwise_or(rectangle, circle)
    results = [
        ("Original", image),
        ("Brighter", brighter),
        ("Lower contrast", lower_contrast),
        ("Higher contrast", higher_contrast),
        ("Bitwise AND", bitwise_and),
        ("Bitwise OR", bitwise_or),
    ]
    result = make_contact_sheet(results, columns=2)
    save_image("image_operations/image_enhancement/enhancement.png", result)
    show_images([("Image enhancement", result)])


if __name__ == "__main__":
    main()
