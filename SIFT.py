import cv2
import numpy as np

NumberOfImagesPerOctave = 3.0
NumberOfGaussianImagesPerOctave = NumberOfImagesPerOctave + 3.0 # We need enough images to calculate the DoG

def image_preprocess(input_img: np.ndarray) -> np.ndarray:
    """
    Pre-Processing before starting the SIFT pipeline
    """
    input_img_grayscale = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    return input_img_grayscale

def image_resize(input_img: np.ndarray, scale_factor: np.int64):
    return (cv2.resize(input_img,))

def Gussian_Scale_Space(input_img: np.ndarray):
    pass

def BuildImagePyramids(img: np.ndarray, number_of_layers_in_pyramid: int) -> list[list[np.ndarray]]:
    images_pyramid = []

    images_pyramid.append(BuildOctave(img,(1/2)**(resizing_factor)) for resizing_factor in range(number_of_layers_in_pyramid))

    return images_pyramid

def BuildOctave(img: np.ndarray, resize_factor: float) -> list[np.ndarray]:
    images_in_octave = []
    resizedImg = cv2.resize(img,(0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA )

    # sigma = 1.0, 1.260, 1.587, 2.000, 2.520, 3.175
    
    for i in range(int(NumberOfGaussianImagesPerOctave)):
        sigma = 2**(i/NumberOfImagesPerOctave)
        _,kernel = buildKernelMatrix(sigma)

        #print(f"{i} Kernel {kernel}:")

        GaussianBlurImg = GaussianConvolution(resizedImg,kernel)
        images_in_octave.append(GaussianBlurImg)

    return images_in_octave

def buildKernelMatrix(sigma: float) -> tuple[int,np.ndarray]:
    """
    This method builds the kernel matrix used in Convolution. It takes the value of sigma and calculated the size of the kernel matrix. Size is equal to 2*(3*sigma) + 1
    Radius of a kernel matric is 3*sigma
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
    input_img = cv2.imread("C:\\Users\\amitk\\workspace\\Learning_Computer_Vision\\images\\Tour_Eiffel.jpg")
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    list_of_images = BuildOctave(input_img,1)

    cv2.imshow("Original Image", input_img)

    for i in range(len(list_of_images)):
        image = list_of_images[i]
        image_display = np.clip(image, 0, 255).astype(np.uint8)

        cv2.imshow(f"Image : {i}", image_display)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Resizemain():
    input_img = cv2.imread("C:\\Users\\amitk\\workspace\\Learning_Computer_Vision\\images\\Tour_Eiffel.jpg")



    # 1. Resize to 1/2 (50%) of the original size
    img_half = cv2.resize(input_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # 2. Resize to 1/4 (25%) of the original size
    img_quarter = cv2.resize(input_img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    img_15 = cv2.resize(input_img, (0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)

    cv2.imshow("Original Image", input_img)
    cv2.imshow("Image 50%", img_half)
    cv2.imshow("25% Image", img_quarter)
    cv2.imshow("15% Image", img_15)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
   


    #StartSIFTPipeline()

main()