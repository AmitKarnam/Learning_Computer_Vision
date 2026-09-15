import cv2
import numpy as np

NumberOfImagesPerOctave = 3.0
NumberOfGaussianImagesPerOctave = NumberOfImagesPerOctave + 3.0 # We need enough images to calculate the DoG

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


def main():
    """
    Example entry point that demonstrates building an image pyramid and
    computing the Difference-of-Gaussian (DoG) arrays for a test image.

    This function reads a sample image from disk, converts it to grayscale,
    constructs the pyramid and DoG, and prints a representation of the
    resulting DoG array.
    """
    input_img = cv2.imread("C:\\Users\\amitk\\workspace\\Learning_Computer_Vision\\images\\Tour_Eiffel.jpg")
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    ImagePyramid = BuildImagePyramids(input_img,3)
    DoGArray = DifferenceOfGaussian(ImagePyramid)

    print(DoGArray)


    # cv2.imshow("Original Image", input_img)

    # for i in range(len(list_of_images)):
    #     image = list_of_images[i]
    #     image_display = np.clip(image, 0, 255).astype(np.uint8)

    #     cv2.imshow(f"Image : {i}", image_display)   


    #StartSIFTPipeline()

main()