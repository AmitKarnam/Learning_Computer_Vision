import cv2
import numpy as np
import math
from dataclasses import dataclass

NumberOfImagesPerOctave = 3.0
NumberOfGaussianImagesPerOctave = NumberOfImagesPerOctave + 3.0 # We need enough images to calculate the DoG
ContrastThreshold = 0.03 * 255
NumberOfLayersInImagePyramid = 3

@dataclass
class KeyPoint:
    x: int
    y: int
    octave: int
    layer: int
    response: float
    orientation: float = 0.0

def DifferenceOfGaussian(image_pyramid: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    """
    Compute the Difference-of-Gaussian (DoG) pyramid from an image pyramid.

    For each octave in `image_pyramid`, compute the difference between
    successive Gaussian-blurred images to produce DoG images for that octave.

    Args:
        image_pyramid: A list of octaves, where each octave is a list of
            Gaussian-blurred images (numpy arrays) sorted by increasing blur.

    Returns:
        A list of octaves where each octave is a list of DoG images (numpy arrays).
    """
    DoGPyramid = []
    for octave in image_pyramid:
        DoGImages = []
        for i in range(len(octave) - 1):
            DoG = octave[i + 1] - octave[i]
            DoGImages.append(DoG)
        DoGPyramid.append(DoGImages)
        
    return DoGPyramid
    
def BuildImagePyramids(img: np.ndarray, number_of_layers_in_pyramid: int) -> list[list[np.ndarray]]:
    """
    Build an image pyramid consisting of multiple octaves.

    Each octave is produced by calling `BuildOctave` with a different
    downscaling factor. The resize factor used for octave `i` is
    `(1/2) ** i`.

    Args:
        img: Input grayscale image as a numpy array.
        number_of_layers_in_pyramid: Number of octaves/layers to build.

    Returns:
        A list of octaves where each octave is a list of Gaussian-blurred
        images (numpy arrays).
    """
    images_pyramid = [
        BuildOctave(img, (1/2)**resizing_factor)
        for resizing_factor in range(number_of_layers_in_pyramid)
    ]

    return images_pyramid

def BuildOctave(img: np.ndarray, resize_factor: float) -> list[np.ndarray]:
    """
    Build a single octave: resize the input image and produce a series of
    Gaussian-blurred images for that octave.

    Args:
        img: Input grayscale image as a numpy array.
        resize_factor: Scaling factor applied to `img` before computing blurs
            (e.g. 0.5 for half-size).

    Returns:
        A list of Gaussian-blurred images (numpy arrays) for this octave.
    """
    images_in_octave = []
    resizedImg = cv2.resize(img,(0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA )

    # sigma = 1.0, 1.260, 1.587, 2.000, 2.520, 3.175

    for i in range(int(NumberOfGaussianImagesPerOctave)):
        sigma = 2**(i/NumberOfImagesPerOctave)
        _,kernel = buildKernelMatrix(sigma)

        GaussianBlurImg = GaussianConvolution(resizedImg,kernel)
        images_in_octave.append(GaussianBlurImg)

    return images_in_octave

def buildKernelMatrix(sigma: float) -> tuple[int,np.ndarray]:
    """
    Build a Gaussian kernel matrix for a given sigma.

    The kernel radius is taken as `r = ceil(3*sigma)` and the kernel size
    is `2*r + 1`. The returned kernel is normalized so that its sum equals 1.

    Args:
        sigma: Standard deviation of the Gaussian.

    Returns:
        A tuple `(kernel_size, kernel)` where `kernel_size` is an int and
        `kernel` is a 2D numpy array containing the Gaussian weights.
    """
    r = int(np.ceil(3*sigma))
    kernel_size = 2*r + 1

    x = np.arange(-r,r+1,1)
    y = np.arange(-r,r+1)

    # comment pending
    xx,yy = np.meshgrid(x,y)

    kernel = np.exp(
        -(xx**2 + yy**2) / (2*sigma**2)
    )
    
    kernel /= kernel.sum()

    return kernel_size, kernel

def GaussianConvolution(
    img: np.ndarray,
    kernel: np.ndarray
) -> np.ndarray:
    """
    Convolve an image with a 2D kernel using reflect padding.

    This function performs a straightforward spatial convolution by sliding
    the kernel over the image. Input and output are numpy arrays with the
    same spatial dimensions. The output is a float64 array (not clipped).

    Args:
        img: 2D grayscale image as a numpy array.
        kernel: 2D convolution kernel (should be normalized for Gaussian).

    Returns:
        The convolved image as a numpy array of dtype float64.
    """
    kernel_size = kernel.shape[0]
    radius = kernel_size // 2

    padded_img = np.pad(
        img,
        ((radius, radius), (radius, radius)),
        mode="reflect"
    )

    output = np.zeros_like(img, dtype=np.float64)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):

            window = padded_img[
                y:y + kernel_size,
                x:x + kernel_size
            ]

            output[y, x] = np.sum(window * kernel)

    return output

def ScaleSpaceExtremaDetection(dog_pyramid: list[list[np.ndarray]]) -> list[KeyPoint]:


    """
    Detect scale-space extrema in a Difference-of-Gaussian (DoG) pyramid.

    For each pixel in each interior DoG layer of each octave, compares the
    pixel against its 26 neighbors (8 in the same layer, 9 in the layer
    above, 9 in the layer below). A pixel that is strictly greater than, or
    strictly less than, all 26 neighbors is recorded as a keypoint.

    Args:
        dog_pyramid: A list of octaves, where each octave is a list of DoG
            images (numpy arrays) sorted by increasing blur.

    Returns:
        A list of KeyPoint instances marking detected extrema.
    """
    keypoints = []

    for index,octave in enumerate(dog_pyramid):
        for layer in range(1,len(octave)-1):
            height,width = octave[layer].shape

            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    candidate = octave[layer][y][x]

                    # Neighbours same "sigma": octave[layer][y][x-1] ; octave[layer][y][x+1]
                    # Upper Neighbours same "sigma": octave[layer][y-1][x-1] ; octave[layer][y-1][x] ; octave[layer][y-1][x+1]
                    # Lower Neighbours same "sigma": octave[layer][y+1][x-1] ; octave[layer][y+1][x] ; octave[layer][y+1][x+1]

                    # Neighbours  "sigma-1": octave[layer-1][y][x-1] ; octave[layer-1][y][x] ; octave[layer-1][y][x+1]
                    # Upper Neighbours "sigma-1": octave[layer-1][y-1][x-1] ; octave[layer-1][y-1][x] ; octave[layer-1][y-1][x+1]
                    # Lower Neighbours "sigma-1": octave[layer-1][y+1][x-1] ; octave[layer-1][y+1][x] ; octave[layer-1][y+1][x+1]

                    # Neighbours  "sigma+1": octave[layer+1][y][x-1] ; octave[layer+1][y][x] ; octave[layer+1][y][x+1]
                    # Upper Neighbours "sigma+1": octave[layer+1][y-1][x-1] ; octave[layer+1][y-1][x] ; octave[layer+1][y-1][x+1]
                    # Lower Neighbours "sigma+1": octave[layer+1][y+1][x-1] ; octave[layer+1][y+1][x] ; octave[layer+1][y+1][x+1]

                    is_max = True
                    is_min = True

                    curr = octave[layer]
                    prev = octave[layer - 1]
                    next_ = octave[layer + 1]

                    neighbors = (
                        curr[y-1][x-1], curr[y-1][x], curr[y-1][x+1],
                        curr[y][x-1],                 curr[y][x+1],
                        curr[y+1][x-1], curr[y+1][x], curr[y+1][x+1],

                        prev[y-1][x-1], prev[y-1][x], prev[y-1][x+1],
                        prev[y][x-1],   prev[y][x],   prev[y][x+1],
                        prev[y+1][x-1], prev[y+1][x], prev[y+1][x+1],

                        next_[y-1][x-1], next_[y-1][x], next_[y-1][x+1],
                        next_[y][x-1],   next_[y][x],   next_[y][x+1],
                        next_[y+1][x-1], next_[y+1][x], next_[y+1][x+1],
                    )

                    for neighbor in neighbors:
                        if candidate <= neighbor:
                            is_max = False

                        if candidate >= neighbor:
                            is_min = False

                        if not is_max and not is_min:
                            break

                    if is_max or is_min:
                        keypoints.append(
                            KeyPoint(x=x, y=y, octave=index, layer=layer, response=candidate)
                        )

    return keypoints

def _compute_3d_gradient(octave_dog: list[np.ndarray], layer: int, y: int, x: int) -> np.ndarray:
    """
    Calculates rate of change using central difference apporximation 
    """
    d_dx = 0.5 * (octave_dog[layer][y, x+1] - octave_dog[layer][y, x-1])
    d_dy = 0.5 * (octave_dog[layer][y+1, x] - octave_dog[layer][y-1, x])
    d_ds = 0.5 * (octave_dog[layer+1][y,x]  - octave_dog[layer-1][y,x])

    return np.array([d_dx,d_dy,d_ds],dtype = np.float64)

def _compute_hessian(octave_dog: list[np.ndarray], layer: int, y: int, x: int) -> np.ndarray:
    """
    Hessain calculates the " rate of change of rate of change of D"; i.e Rate of change of slope(x+1) wrt to slope(x-1)
    slope(x+1) = D(x+1) - D(x)
    slope(x-1) = D(x) - D(x-1)
    """

    center = octave_dog[layer][y, x]

    d_xx = octave_dog[layer][y, x + 1] + octave_dog[layer][y, x - 1] - 2.0 * center
    d_yy = octave_dog[layer][y + 1, x] + octave_dog[layer][y - 1, x] - 2.0 * center
    d_ss = octave_dog[layer + 1][y, x] + octave_dog[layer - 1][y, x] - 2.0 * center

    d_xy = 0.25 * (
        octave_dog[layer][y + 1, x + 1] - octave_dog[layer][y + 1, x - 1]
        - octave_dog[layer][y - 1, x + 1] + octave_dog[layer][y - 1, x - 1]
    )
    d_xs = 0.25 * (
        octave_dog[layer + 1][y, x + 1] - octave_dog[layer + 1][y, x - 1]
        - octave_dog[layer - 1][y, x + 1] + octave_dog[layer - 1][y, x - 1]
    )
    d_ys = 0.25 * (
        octave_dog[layer + 1][y + 1, x] - octave_dog[layer + 1][y - 1, x]
        - octave_dog[layer - 1][y + 1, x] + octave_dog[layer - 1][y - 1, x]
    )

    return np.array(
        [[d_xx, d_xy, d_xs],
         [d_xy, d_yy, d_ys],
         [d_xs, d_ys, d_ss]],
        dtype=np.float64,
    )

def _is_edge_like(hessian: np.ndarray,curvature_ratio: float = 10.0,) -> bool:
    # Extract the spatial 2x2 Hessian.
    H_xy = hessian[:2, :2]

    trace = np.trace(H_xy)
    determinant = np.linalg.det(H_xy)

    # A nonpositive determinant fails the spatial curvature test.
    if determinant <= 0:
        return True

    ratio = (trace * trace) / determinant
    threshold = ((curvature_ratio + 1) ** 2) / curvature_ratio

    return ratio >= threshold

def KeypointLocalization(keypoints: list[KeyPoint],dog_pyramid: list[list[np.ndarray]],max_iterations: int = 5,) -> list[KeyPoint]:

    localized_keypoints = []

    for keypoint in keypoints:
        octave_dog = dog_pyramid[keypoint.octave]

        x = keypoint.x
        y = keypoint.y
        layer = keypoint.layer

        converged = False

        for _ in range(max_iterations):
            # Ensure the 3D neighborhood exists.
            if not (1 <= layer < len(octave_dog) - 1 and 1 <= y < octave_dog[layer].shape[0] - 1 and 1 <= x < octave_dog[layer].shape[1] - 1):
                break

            gradient_of_DoG = _compute_3d_gradient(octave_dog, layer, y, x)
            hessian_of_DoG = _compute_hessian(octave_dog, layer, y, x)

            try:
                delta = np.linalg.solve(hessian_of_DoG, -gradient_of_DoG)
            except np.linalg.LinAlgError:
                break

            if np.all(np.abs(delta) < 0.5):
                converged = True
                break

            x += int(np.sign(delta[0])) if abs(delta[0]) >= 0.5 else 0
            y += int(np.sign(delta[1])) if abs(delta[1]) >= 0.5 else 0
            layer += int(np.sign(delta[2])) if abs(delta[2]) >= 0.5 else 0

        if not converged:
            continue

        # Calculating contrast and elimnating low contrast values
        # there is no universal threshold independent of image intensity scaling and implementation.Lowe's original SIFT method uses a contrast threshold of approximately (0.03) when image intensities are normalized to ([0,1]).
        # NOTE: OpenCV's SIFT applies its contrast threshold with additional scaling related to the number of scale layers.
        contrast = octave_dog[layer][y][x] + (0.5* np.dot(gradient_of_DoG,delta))
        if abs(contrast) < ContrastThreshold:
            continue

        # Edge Rejection Goal: Even if a keypoint has strong contrast, can we reliably determine where it is located?
        if _is_edge_like(hessian_of_DoG):
            continue

        localized_keypoint = KeyPoint(
            x=float(x + delta[0]),
            y=float(y + delta[1]),
            octave=keypoint.octave,
            layer=float(layer + delta[2]),
            response=float(contrast),
        )

        localized_keypoints.append(localized_keypoint)

    return localized_keypoints

def _compute_2d_gradient(image: np.ndarray,y: int,x: int,) -> tuple[float, float]:
    """
    Compute image gradients using central differences.

    Returns:
        (dx, dy)
    """
    dx = 0.5 * (
        float(image[y, x + 1]) -
        float(image[y, x - 1])
    )

    dy = 0.5 * (
        float(image[y + 1, x]) -
        float(image[y - 1, x])
    )

    return dx, dy

def _smooth_orientation_histogram(
    histogram: np.ndarray,
    iterations: int = 6,
) -> np.ndarray:
    """
    Smooth the circular orientation histogram using
    the five-element kernel [1, 4, 6, 4, 1] / 16.
    """
    histogram = histogram.copy()

    for _ in range(iterations):
        histogram = (
            np.roll(histogram, 2)
            + 4.0 * np.roll(histogram, 1)
            + 6.0 * histogram
            + 4.0 * np.roll(histogram, -1)
            + np.roll(histogram, -2)
        ) / 16.0

    return histogram

def _find_orientation_peaks(
    histogram: np.ndarray,
    peak_ratio: float = 0.8,
) -> list[float]:
    """
    Find dominant orientation peaks and interpolate their
    positions using a quadratic approximation.

    Returns:
        Peak orientations in radians in [0, 2*pi).
    """
    num_bins = len(histogram)
    max_value = float(np.max(histogram))

    if max_value <= 0:
        return []

    orientations = []

    for i in range(num_bins):
        left = histogram[(i - 1) % num_bins]
        center = histogram[i]
        right = histogram[(i + 1) % num_bins]

        # Local maximum with sufficient magnitude.
        if not (center > left and center > right):
            continue

        if center < peak_ratio * max_value:
            continue

        # Quadratic interpolation of the histogram peak.
        denominator = left - 2.0 * center + right

        if abs(denominator) > 1e-12:
            offset = 0.5 * (left - right) / denominator
        else:
            offset = 0.0

        # Keep interpolation within the adjacent-bin interval.
        offset = float(np.clip(offset, -0.5, 0.5))

        peak_bin = (i + offset) % num_bins

        angle = 2.0 * np.pi * peak_bin / num_bins
        orientations.append(float(angle))

    return orientations

def OrientationAssignment(image_pyramid: list[list[np.ndarray]],keypoint_array: list[KeyPoint],num_bins: int = 36,orientation_peak_ratio: float = 0.8,) -> list[KeyPoint]:
    """
    Assign dominant orientations to localized SIFT keypoints.

    For each keypoint:
      1. Select the corresponding Gaussian image.
      2. Estimate the keypoint scale.
      3. Calculate the Gaussian weighting sigma and radius.
      4. Compute gradients in the neighborhood.
      5. Accumulate a Gaussian-weighted orientation histogram.
      6. Smooth the histogram.
      7. Create an oriented keypoint for each dominant peak.

    Assumptions:
      - Each octave contains six Gaussian images.
      - DoG layer k = Gaussian[k+1] - Gaussian[k].
      - The Gaussian blur sigma at Gaussian layer j is
        2**(j / NumberOfImagesPerOctave).
      - Keypoint coordinates are in octave-local pixels.
      - Keypoint.layer is the refined DoG layer coordinate.

    Returns:
        A list of oriented keypoints. A single input keypoint
        may generate multiple output keypoints.
    """
    oriented_keypoints: list[KeyPoint] = []

    for keypoint in keypoint_array:
        octave = keypoint.octave

        if not (0 <= octave < len(image_pyramid)):
            continue

        gaussian_octave = image_pyramid[octave]

        # 1. Map the DoG scale to a Gaussian image.x    
        # DoG[k] = Gaussian[k+1] - Gaussian[k].
        gaussian_layer_float = keypoint.layer + 1.0

        gaussian_layer = int(round(gaussian_layer_float))
        gaussian_layer = max(
            0,
            min(gaussian_layer, len(gaussian_octave) - 1),
        )

        gaussian_img = gaussian_octave[gaussian_layer]

        # 2. Estimate the keypoint's local Gaussian scale.
        # Use the refined layer coordinate for the scale, rather than the rounded index used to select an image.
        sigma_keypoint = (
            2.0 ** (
                gaussian_layer_float / NumberOfImagesPerOctave
            )
        )

        # 3. Compute Gaussian weighting sigma and radius.
        sigma_weight = 1.5 * sigma_keypoint
        radius = int(math.ceil(3.0 * sigma_weight))

        # 4. Get the integer pixel around which to sample.
        x_center = int(round(keypoint.x))
        y_center = int(round(keypoint.y))

        height, width = gaussian_img.shape[:2]

        x_min = max(1, x_center - radius)
        x_max = min(width - 1, x_center + radius + 1)

        y_min = max(1, y_center - radius)
        y_max = min(height - 1, y_center + radius + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        # 5. Initialize the orientation histogram.
        histogram = np.zeros(num_bins, dtype=np.float64)

        bin_width = 2.0 * np.pi / num_bins
        two_pi = 2.0 * np.pi

        # 6. Compute gradients and accumulate orientations.
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):

                dx_from_keypoint = x - keypoint.x
                dy_from_keypoint = y - keypoint.y

                distance_squared = (
                    dx_from_keypoint**2
                    + dy_from_keypoint**2
                )

                # Gaussian spatial weighting.
                weight = math.exp(
                    -distance_squared
                    / (2.0 * sigma_weight**2)
                )

                # Gradient at this pixel.
                dx, dy = _compute_2d_gradient(
                    gaussian_img, y, x
                )

                magnitude = math.hypot(dx, dy)

                if magnitude == 0.0:
                    continue

                # Gradient orientation in [0, 2*pi).
                angle = math.atan2(dy, dx) % two_pi

                # Convert angle to a continuous histogram bin.
                bin_position = angle / bin_width

                lower_bin = int(math.floor(bin_position))
                fraction = bin_position - lower_bin

                lower_bin %= num_bins
                upper_bin = (lower_bin + 1) % num_bins

                # Weight gradient magnitude by spatial distance,
                # then interpolate between adjacent bins.
                contribution = weight * magnitude

                histogram[lower_bin] += (
                    contribution * (1.0 - fraction)
                )

                histogram[upper_bin] += (
                    contribution * fraction
                )

        # 7. Smooth the circular histogram.
        histogram = _smooth_orientation_histogram(
            histogram,
            iterations=6,
        )

        # 8. Find the dominant orientation peaks.
        peak_angles = _find_orientation_peaks(
            histogram,
            peak_ratio=orientation_peak_ratio,
        )

        # 9. Produce an oriented keypoint for each peak.
        for angle in peak_angles:
            oriented_keypoints.append(
                KeyPoint(
                    x=keypoint.x,
                    y=keypoint.y,
                    octave=keypoint.octave,
                    layer=keypoint.layer,
                    response=keypoint.response,
                    orientation=angle,
                )
            )

    return oriented_keypoints

def ComputeDescriptor(
    image_pyramid: list[list[np.ndarray]],
    keypoint: KeyPoint,
    num_spatial_bins: int = 4,
    num_orientation_bins: int = 8,
) -> np.ndarray | None:
    """
    Compute a 128-dimensional SIFT descriptor for one oriented keypoint.

    Uses:
      - 4 x 4 spatial cells
      - 8 orientation bins per cell
      - Trilinear interpolation over x, y, and orientation
      - Gaussian weighting
      - L2 normalization, clipping at 0.2, and renormalization

    Assumptions:
      - Keypoint coordinates are octave-local pixel coordinates.
      - DoG[k] = Gaussian[k+1] - Gaussian[k].
      - KeyPoint.layer is the refined DoG layer coordinate.
      - Gaussian scale at layer j is 2 ** (j / 3).

    Returns:
        A normalized 128-dimensional descriptor, or None if the
        keypoint is too close to the image boundary.
    """

    d = num_spatial_bins
    n = num_orientation_bins

    # --------------------------------------------------
    # 1. Select the Gaussian image for this keypoint.
    # --------------------------------------------------
    octave = keypoint.octave

    if not (0 <= octave < len(image_pyramid)):
        return None

    gaussian_octave = image_pyramid[octave]

    gaussian_layer_float = keypoint.layer + 1.0
    gaussian_layer = int(round(gaussian_layer_float))

    gaussian_layer = max(
        0,
        min(gaussian_layer, len(gaussian_octave) - 1),
    )

    image = gaussian_octave[gaussian_layer].astype(
        np.float64, copy=False
    )

    height, width = image.shape

    # --------------------------------------------------
    # 2. Estimate the keypoint scale.
    # --------------------------------------------------
    sigma_keypoint = 2.0 ** (
        gaussian_layer_float / NumberOfImagesPerOctave
    )

    # Spatial histogram spacing in octave-local pixels.
    hist_width = 3.0 * sigma_keypoint

    # Gaussian weighting standard deviation.
    sigma_descriptor = 2.0 * hist_width

    # --------------------------------------------------
    # 3. Set up the rotated descriptor window.
    # --------------------------------------------------
    cos_t = math.cos(keypoint.orientation)
    sin_t = math.sin(keypoint.orientation)

    # Bounding radius large enough to cover the rotated grid.
    radius = int(
        round(
            hist_width
            * math.sqrt(2.0)
            * (d + 1)
            * 0.5
        )
    )

    x_center = int(round(keypoint.x))
    y_center = int(round(keypoint.y))

    # Gradients require neighboring pixels.
    if (
        x_center - radius < 1
        or x_center + radius >= width - 1
        or y_center - radius < 1
        or y_center + radius >= height - 1
    ):
        return None

    # --------------------------------------------------
    # 4. Initialize the 4 x 4 x 8 histogram.
    # --------------------------------------------------
    histogram = np.zeros((d, d, n), dtype=np.float64)

    two_pi = 2.0 * math.pi

    # --------------------------------------------------
    # 5. Visit pixels in the descriptor neighborhood.
    # --------------------------------------------------
    for y in range(
        y_center - radius,
        y_center + radius + 1,
    ):
        for x in range(
            x_center - radius,
            x_center + radius + 1,
        ):

            # Position relative to the keypoint.
            dx = x - keypoint.x
            dy = y - keypoint.y

            # Rotate coordinates into the keypoint frame.
            x_rot = dx * cos_t + dy * sin_t
            y_rot = -dx * sin_t + dy * cos_t

            # Convert to continuous spatial histogram coordinates.
            x_bin = x_rot / hist_width + d / 2.0 - 0.5
            y_bin = y_rot / hist_width + d / 2.0 - 0.5

            # Skip pixels that cannot contribute to any cell.
            if not (
                -1.0 <= x_bin < d
                and -1.0 <= y_bin < d
            ):
                continue

            # ------------------------------------------
            # 6. Calculate image gradients.
            # ------------------------------------------
            dx_img = 0.5 * (
                image[y, x + 1] - image[y, x - 1]
            )

            dy_img = 0.5 * (
                image[y + 1, x] - image[y - 1, x]
            )

            magnitude = math.hypot(dx_img, dy_img)

            if magnitude == 0.0:
                continue

            angle = math.atan2(dy_img, dx_img) % two_pi

            # Express orientation relative to keypoint.
            relative_angle = (
                angle - keypoint.orientation
            ) % two_pi

            # Continuous orientation-bin coordinate.
            orientation_bin = (
                relative_angle * n / two_pi
            )

            # ------------------------------------------
            # 7. Apply Gaussian weighting.
            # ------------------------------------------
            gaussian_weight = math.exp(
                -(x_rot**2 + y_rot**2)
                / (2.0 * sigma_descriptor**2)
            )

            contribution = magnitude * gaussian_weight

            # ------------------------------------------
            # 8. Trilinear interpolation.
            # ------------------------------------------
            x0 = math.floor(x_bin)
            y0 = math.floor(y_bin)
            o0 = math.floor(orientation_bin)

            fx = x_bin - x0
            fy = y_bin - y0
            fo = orientation_bin - o0

            for xi, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
                if not (0 <= xi < d):
                    continue

                for yi, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
                    if not (0 <= yi < d):
                        continue

                    for oi, wo in (
                        (o0, 1.0 - fo),
                        (o0 + 1, fo),
                    ):
                        oi %= n

                        histogram[yi, xi, oi] += (
                            contribution * wx * wy * wo
                        )

    # --------------------------------------------------
    # 9. Flatten the histogram into 128 values.
    # --------------------------------------------------
    descriptor = histogram.ravel()

    # --------------------------------------------------
    # 10. L2 normalize.
    # --------------------------------------------------
    norm = np.linalg.norm(descriptor)

    if norm < 1e-12:
        return None

    descriptor /= norm

    # --------------------------------------------------
    # 11. Clip large values and normalize again.
    # --------------------------------------------------
    descriptor = np.clip(descriptor, 0.0, 0.2)

    norm = np.linalg.norm(descriptor)

    if norm < 1e-12:
        return None

    descriptor /= norm

    return descriptor

def ComputeDescriptors(
    image_pyramid: list[list[np.ndarray]],
    keypoints: list[KeyPoint],
) -> tuple[list[KeyPoint], np.ndarray]:

    valid_keypoints = []
    descriptors = []

    for keypoint in keypoints:
        descriptor = ComputeDescriptor(
            image_pyramid,
            keypoint,
        )

        if descriptor is None:
            continue

        valid_keypoints.append(keypoint)
        descriptors.append(descriptor)

    if not descriptors:
        return valid_keypoints, np.empty((0, 128), dtype=np.float64)

    return valid_keypoints, np.vstack(descriptors)

def VisualizeKeypoints(
    
    image: np.ndarray,
    keypoints: list[KeyPoint],
) -> np.ndarray:
    """
    Draw custom SIFT keypoints on the original image.
    """

    output = cv2.cvtColor(
        image.astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )

    cv_keypoints = []

    for kp in keypoints:
        octave_scale = 2 ** kp.octave

        x = kp.x * octave_scale
        y = kp.y * octave_scale

        # Approximate the detected keypoint's scale.
        sigma = 2.0 ** (
            (kp.layer + 1.0) / NumberOfImagesPerOctave
        )

        size = 2.0 * sigma * octave_scale

        cv_keypoints.append(
            cv2.KeyPoint(
                float(x),
                float(y),
                float(size),
                float(math.degrees(kp.orientation)),
                float(kp.response),
            )
        )

    return cv2.drawKeypoints(
        output,
        cv_keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def main():
    input_img = cv2.imread("C:\\Users\\amitk\\workspace\\Learning_Computer_Vision\\images\\Tour_Eiffel.jpg", cv2.IMREAD_GRAYSCALE,)

    if input_img is None:
        raise FileNotFoundError("Could not load the input image.")

    image_pyramid = BuildImagePyramids(
        input_img,
        number_of_layers_in_pyramid=3,
    )

    dog_pyramid = DifferenceOfGaussian(image_pyramid)

    extrema = ScaleSpaceExtremaDetection(dog_pyramid)

    localized_keypoints = KeypointLocalization(
        extrema,
        dog_pyramid,
    )

    oriented_keypoints = OrientationAssignment(
        image_pyramid,
        localized_keypoints,
    )

    custom_keypoint_image = VisualizeKeypoints(
    input_img,
    oriented_keypoints, 
    )

    cv2.imshow("My SIFT Keypoints", custom_keypoint_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # descriptor_keypoints, descriptors = ComputeDescriptors(
    #     image_pyramid,
    #     oriented_keypoints, 
    # )

   

    # print("Oriented keypoints:", len(oriented_keypoints))
    # print("Keypoints with descriptors:", len(descriptor_keypoints))
    # print("Descriptor matrix shape:", descriptors.shape)

    # if len(descriptors) > 0:
    #     print("First descriptor:")
    #     print(descriptors[0])
    #     print("Descriptor norm:", np.linalg.norm(descriptors[0]))


if __name__ == "__main__":
    main()

main()