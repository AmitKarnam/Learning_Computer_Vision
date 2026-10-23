import cv2
import numpy as np

def ImageAdditionOrBrightness(imgPath: str):
    img = cv2.imread(imgPath)

    matrix = np.ones(img.shape, dtype="uint8") * 50

    brightImage = cv2.add(img,matrix)
    darkImage = cv2.subtract(img,matrix)

    cv2.imshow("Bright Image", brightImage)
    cv2.imshow("Original Image", img)
    cv2.imshow("Darker Image", darkImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ImageMultiplicationOrContrast(imgPath: str):
    img = cv2.imread(imgPath)

    lighterContrastMatrix = np.ones(img.shape) * 0.8
    darkerContrastMatrix = np.ones(img.shape) * 1.2

    lighterContrastImg = np.uint8(cv2.multiply(np.float64(img),lighterContrastMatrix))
    darkerContrastImg = np.uint8(np.clip(cv2.multiply(np.float64(img),darkerContrastMatrix),0,255))

    cv2.imshow("Lighter Contrast Image", lighterContrastImg)
    cv2.imshow("Original Image", img)
    cv2.imshow("Darker Contrast Image", darkerContrastImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ImageMasking(imgPath: str):
    coke = cv2.imread("./images/coco-cola-logo.png")
    colorCheckerboard = cv2.imread("./images/checkerboard_color.png")

def BitWiseOperation():
    """
    Bitwise operations are simple pixel-level logical operations that act on binary bits — the 0s and 1s that make up pixel values.
    
    1. AND → output bit is 1 if both bits are 1

    2. OR → output bit is 1 if either bit is 1

    3. NOT → output bit is the inverse of the input bit

    4. XOR → output bit is 1 if bits are different

    Parameters
    
    => src1: The first input image (or array).

    => src2: The second input image (or array). This parameter is not present in cv2.bitwise_not().

    => dst: An optional output array (image) where the result will be stored. If not provided, a new array is created.

    => mask: An optional 8-bit single-channel array (image) that acts as a mask. This mask determines which pixels in the input images will be processed.
        1. How the mask works: For each pixel location (x, y), if the corresponding pixel in the mask array is non-zero, the bitwise operation is performed on the src1 and src2 (if applicable) pixels at (x, y), and the result is stored in the dst image at (x, y).
        2. If the corresponding pixel in the mask array is zero, no operation is performed, and the pixel in the dst image at (x, y) will typically be set to 0 (black), or retain its original value if dst was pre-initialized and not fully overwritten. 
        3. The mask must have the same dimensions (width and height) as the input images, but it must be a single-channel (grayscale) image.
    """
    rect = cv2.imread("./images/rectangle.jpg")
    circle = cv2.imread("./images/circle.jpg")

    bitwise_and = cv2.bitwise_and(rect,circle)
    bitwise_or = cv2.bitwise_or(rect,circle)

    cv2.imshow("Bitwise And of Images",bitwise_and)
    cv2.imshow("Bitwise OR of Images",bitwise_or)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #ImageAdditionOrBrightness("./images/ktm_adventure_390_x.jpeg")
    #ImageMultiplicationOrContrast("./images/ktm_adventure_390_x.jpeg")
    BitWiseOperation()