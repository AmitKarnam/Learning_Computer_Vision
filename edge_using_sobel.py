import numpy as np
import cv2
from convolution import calculate_padding_length, build_new_image_matrix_with_reflect_padding

imgPath = "C:\\Users\\amitk\\workspace\\Learning_Computer_Vision\\images\\rocks.jpg"

def SobelConvolute(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Generic Sobel convolution.
    Returns raw float32 gradient values.
    """

    k = kernel.shape[0]
    h, w, c = img.shape

    paddingLength = calculate_padding_length(k)
    paddedImg = build_new_image_matrix_with_reflect_padding(
        img,
        paddingLength
    )

    output = np.zeros((h, w, c), dtype=np.float32)

    for i in range(h):
        for j in range(w):

            region = paddedImg[i:i+k, j:j+k]

            for ch in range(c):
                output[i, j, ch] = np.sum(
                    region[:, :, ch] * kernel
                )

    return output


def ComputeSobelEdges(img: np.ndarray):
    """
    Computes:
        Gx          -> Horizontal gradient
        Gy          -> Vertical gradient
        Magnitude   -> Edge strength
        Direction   -> Edge orientation (degrees)

    Returns:
        gx,
        gy,
        magnitude,
        direction_deg
    """

    sobel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [1,  2,  1],
        [0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float32)

    # Raw gradients
    gx = SobelConvolute(img, sobel_x)
    gy = SobelConvolute(img, sobel_y)

    # Edge strength
    magnitude = np.hypot(gx, gy)

    # Edge orientation
    direction_rad = np.arctan2(gy, gx)

    direction_deg = np.rad2deg(direction_rad)

    return gx, gy, magnitude, direction_deg


if __name__ == "__main__":
    originalImg = cv2.imread(imgPath)
    img = originalImg.astype(np.float32)
    gx, gy, magnitude, direction = ComputeSobelEdges(img)

    magnitude_display = np.clip(
        magnitude,
        0,
        255
    ).astype(np.uint8)

    cv2.imshow("Original Image", originalImg)

    cv2.imshow("Sobel Magnitude", magnitude_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()