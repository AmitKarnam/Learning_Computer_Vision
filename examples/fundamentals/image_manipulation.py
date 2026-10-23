import cv2

def ManipulatePixelInImage(imgPath: str):
    img = cv2.imread(imgPath,0)
    img[2][2] = 127
    img[2][3] = 127
    img[3][2] = 127
    img[3][3] = 127

    cv2.imwrite("altered_checkerboard.png",img)


    #print(img)


def ImageResizing(imgPath: str):
    """
        Resize an image using OpenCV.

        Parameters:
            src (numpy.ndarray): The source (input) image.
            dsize (tuple[int, int], optional): Desired output size as (width, height).
                If both fx and fy are specified, dsize can be set to None.
            fx (float, optional): Scale factor along the horizontal axis.
            fy (float, optional): Scale factor along the vertical axis.
            interpolation (int, optional): Interpolation method to use.
                Common options:
                    - cv2.INTER_NEAREST: Nearest-neighbor interpolation (fastest, lowest quality)
                    - cv2.INTER_LINEAR: Bilinear interpolation (default)
                    - cv2.INTER_CUBIC: Bicubic interpolation (better for upscaling)
                    - cv2.INTER_AREA: Pixel area relation (recommended for shrinking)
                    - cv2.INTER_LANCZOS4: Lanczos interpolation (high quality for up/downscaling)
        Returns:
            numpy.ndarray: The resized image.
    """
    img = cv2.imread(imgPath)
    newIimg = cv2.resize(img,(300,200),interpolation=cv2.INTER_CUBIC)

    cv2.imshow("Nee Resized Image",newIimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ImageFlip(imgPath: str):
    """
    Flip an image using OpenCV.

    Parameters:
        src (numpy.ndarray): The input image (a 2D or 3D array) to flip.
        flipCode (int): Code that specifies the flip direction.
            0  -> Flip vertically (around the x-axis)
            1  -> Flip horizontally (around the y-axis)
            -1 -> Flip both vertically and horizontally

    Returns:
        numpy.ndarray: The flipped image.
    """

    img = cv2.imread(imgPath)

    xAxis = cv2.flip(img,0)
    yAxis = cv2.flip(img,1)
    bothAxis = cv2.flip(img,-1)

    cv2.imshow("Original Image", img)
    cv2.imshow("X Axis Flip",xAxis)
    cv2.imshow("Y Axis Flip", yAxis)
    cv2.imshow("Both Axis Flip", bothAxis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
     

def ImageCropping(imgPath: str):
    img = cv2.imread(imgPath,cv2.IMREAD_COLOR)

    croppedImg = img[100:200,100:200]

    cv2.imshow("Cropped Image",croppedImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #ManipulatePixelInImage("/home/amit-karnam/Learning_Computer_Vision/images/checkerboard_18x18.png")
    #ImageResizing("/home/amit-karnam/Learning_Computer_Vision/images/checkerboard_18x18.png")
    #ImageCropping("/home/amit-karnam/Learning_Computer_Vision/images/ktm_adventure_390_x.jpeg")
    ImageFlip("/home/amit-karnam/Learning_Computer_Vision/images/ktm_adventure_390_x.jpeg")