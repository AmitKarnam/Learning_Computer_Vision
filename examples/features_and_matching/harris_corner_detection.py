"""Detect and visualize Harris corners."""

import cv2

from examples.common import load_image, make_contact_sheet, save_image, show_images


def main() -> None:
    image = load_image("checkerboard_color.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    response = cv2.cornerHarris(gray.astype("float32"), 2, 3, 0.04)
    response = cv2.dilate(response, None)
    threshold = 0.01 * response.max()

    marked = image.copy()
    marked[response > threshold] = (0, 0, 255)
    response_view = cv2.normalize(
        response,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype("uint8")
    result = make_contact_sheet(
        [("Input image", image), ("Harris response", response_view), ("Corners", marked)]
    )
    save_image("features_and_matching/harris_corner_detection/corners.png", result)
    show_images([("Harris corners", result)])


if __name__ == "__main__":
    main()
