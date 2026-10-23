"""Read an image and visualize its individual color channels."""

import cv2
import numpy as np

from examples.common import load_image, make_contact_sheet, save_image, show_images


def main() -> None:
    image = load_image("ktm_adventure_390_x.jpeg")
    blue, green, red = cv2.split(image)
    zeros = np.zeros_like(blue)
    channels = [
        ("Original", image),
        ("Blue channel", cv2.merge((blue, zeros, zeros))),
        ("Green channel", cv2.merge((zeros, green, zeros))),
        ("Red channel", cv2.merge((zeros, zeros, red))),
    ]
    result = make_contact_sheet(channels)
    save_image("fundamentals/image_io/color_channels.png", result)
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels")
    show_images([("Color channels", result)])


if __name__ == "__main__":
    main()
