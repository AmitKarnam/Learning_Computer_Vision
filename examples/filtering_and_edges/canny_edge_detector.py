import numpy as np
import cv2
from examples.common import load_image, make_contact_sheet, save_image, show_images
from examples.filtering_and_edges.convolution import (
    build_new_image_matrix_with_reflect_padding,
    calculate_padding_length,
)


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

def NonMaximaSupression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray: 
    """
    NMS is removal non maxima edges to reduce thick, blurry edges. NMS thins these bands down to a single-pixel width.
    """
    # In the direction of gradient keep only the largest pixel value and convert the rest to 0
    # if angle is 0 , check left and right
    # if angle is 45, check top-right and bottom-left
    # if angle is 90, check top and bottom
    # if angle is 135, check top-left and bottom-right

    rows,columns,channel = magnitude.shape
    output = np.zeros((rows, columns, channel), dtype=np.float32)

    direction = (direction + 180) % 180

    for ch in range(channel):
        for i in range(1,rows-1):
            for j in range(1,columns-1):

                angle = direction[i,j,ch]
                current = magnitude[i,j,ch]

                # if angle is 0 to 22.5 or 157.5 to 180 ( 0 degree )
                if (0 <= angle < 22.5) or (157.5 < angle <= 180):
                    neighbor1 = magnitude[i,j-1,ch]
                    neighbor2 = magnitude[i,j+1,ch]

                # if angle is 22.5 to 67.5 ( 45 degree )
                elif 22.5 <= angle < 67.5:
                    neighbor1 = magnitude[i-1,j+1,ch]
                    neighbor2 = magnitude[i+1,j-1,ch]

                # if angle is 67.5 to 112.5 ( 90 degree )
                elif 67.5 <= angle < 112.5:
                    neighbor1 = magnitude[i-1,j,ch]
                    neighbor2 = magnitude[i+1,j,ch]

                # if angle 112.5 to 157.5 ( 135 degree )
                else:
                    neighbor1 = magnitude[i - 1, j - 1, ch]
                    neighbor2 = magnitude[i + 1, j + 1, ch]

                if current >= neighbor1 and current >= neighbor2:
                    output[i,j,ch] = current
                else:
                    output[i,j,ch] = 0
    
    return output

def DoubleThreshold(magnitude: np.ndarray) -> np.ndarray:
    HIGH = magnitude.max() * 0.2
    LOW = HIGH * 0.5
    rows,colums,ch = magnitude.shape

    result = np.zeros((rows,colums,ch), dtype=np.float32)

    for c in range(ch):
        for row in range(rows):
            for column in range(colums):
                value = magnitude[row,column,c]
                if value >= HIGH:
                    result[row,column,c] = HIGH
                elif value >= LOW:
                    result[row,column,c] = LOW

    return result

def HysteresisThreshold(DoubleThresholdArray: np.ndarray) -> np.ndarray:
    """
    Traversal of the array to check if the strong edge is connected to weak edge using DFS
    """
    HIGH = float(DoubleThresholdArray.max())
    if HIGH == 0:
        return np.zeros_like(DoubleThresholdArray)
    LOW = HIGH * 0.5

    rows, colums, ch = DoubleThresholdArray.shape

    visited = np.zeros((rows, colums, ch), dtype=np.float32)
    result = np.zeros((rows, colums, ch), dtype=np.float32)

    stack = []

    # 8-connectivity neighborhood 
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    strong_indices = np.where(DoubleThresholdArray >= HIGH)
    
    stack = list(zip(strong_indices[0], strong_indices[1], strong_indices[2]))

    while len(stack) > 0:
        curr_r, curr_c, curr_h = stack.pop()

        if visited[curr_r, curr_c, curr_h] == 1.0:
            continue

        visited[curr_r, curr_c, curr_h] = 1.0
        result[curr_r, curr_c, curr_h] = HIGH

        # Scan all 8 surrounding spatial neighbors within the current channel
        for dr, dc in neighbor_offsets:
            next_r = curr_r + dr
            next_c = curr_c + dc

            # Boundary enforcement
            if 0 <= next_r < rows and 0 <= next_c < colums:
                val = DoubleThresholdArray[next_r, next_c, curr_h]
                
                is_weak = (val >= LOW) and (val < HIGH)
                is_unvisited = visited[next_r, next_c, curr_h] == 0.0
                
                if is_weak and is_unvisited:
                    stack.append((next_r, next_c, curr_h))

    return result

    


if __name__ == "__main__":
    originalImg = load_image("rocks.jpg")
    img = originalImg.astype(np.float32)
    gx, gy, magnitude, direction = ComputeSobelEdges(img)

    NMSArray = NonMaximaSupression(magnitude,direction)

    DoubleThresholdResult = DoubleThreshold(NMSArray)

    HysteresisThresholdResult = HysteresisThreshold(DoubleThresholdResult)

    HIGH = magnitude.max() * 0.2
    LOW = HIGH * 0.5
    gray = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
    cannyUsingCV2 = cv2.Canny(gray, threshold1=LOW, threshold2=HIGH)
    suppressed_display = cv2.normalize(
        NMSArray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    results = [
        ("Original", originalImg),
        ("Non-maximum suppression", suppressed_display),
        ("Double threshold", DoubleThresholdResult.astype(np.uint8)),
        ("Custom hysteresis", HysteresisThresholdResult.astype(np.uint8)),
        ("OpenCV Canny", cannyUsingCV2),
    ]
    result = make_contact_sheet(results, columns=2)
    save_image("filtering_and_edges/canny_edge_detector/edge_stages.png", result)
    show_images([("Canny edge detection", result)])