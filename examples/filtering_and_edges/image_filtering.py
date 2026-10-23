"""Compare common smoothing filters on a single input image."""

import cv2

from examples.common import load_image, make_contact_sheet, save_image, show_images


def main() -> None:
    image = load_image("rocks.jpg")
    results = [
        ("Original", image),
        ("Box blur", cv2.blur(image, (9, 9))),
        ("Gaussian blur", cv2.GaussianBlur(image, (9, 9), 0)),
        ("Median blur", cv2.medianBlur(image, 9)),
        ("Bilateral filter", cv2.bilateralFilter(image, 9, 75, 75)),
    ]
    result = make_contact_sheet(results, columns=2)
    save_image("filtering_and_edges/image_filtering/filters.png", result)
    show_images([("Image filters", result)])


if __name__ == "__main__":
    main()
