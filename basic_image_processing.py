import cv2
import numpy as np

# Note: Open CV uses Blue;Green;Red Channel order to display color images

def CVImageRead():
    """
        1. imread is a method used to read an image and return it in a numpy array format.
        2. Argument 1: File path
        3. Argument 2: Image Format
            => 0:  Read the image in gray scale
            => 1:  Read the image uncahnged, but exclude transperency.
            => -1: Read image unchanged, this includes all the properties of the original image including transperency
    """
    img = cv2.imread("/home/amit-karnam/Learning_CV/ktm_adventure_390_x.jpeg",-1)

    if img is None:
        print("Error loading image")
    else:
        print("Image loaded correctly")

    return img

def CVImageShow(img):
    """
        1. imshow is used to show an image using the image array.
        2. Argument 1: Image window name, this is the window on which the image is displayed.
        3. Argument 2: The image array.
        4. The image window might open and closely very rapidly, we need to use waitKey() so that we can view the image
        5. destroyAllWindows() is used to close all the windows that have been opened with imshow(). 
    """
    cv2.imshow("KTM Adventure 390 X",img)
    # waitKey(0): wait for a key press before closing the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def CVImageOverWrite(img):
    """
        1. imwrite is used to write an image into a file.
        2. Argument 1: File path
        3. Argument 2: The image array
    """
    if cv2.imwrite("ktm_adventure_390_x.jpeg",img=img):
        print("Image overwritten successfully")
    else:
        print("Unable to overwrite image")

if __name__ == "__main__":
    img = CVImageRead()
    
    """Printing Dimensions of the image array"""
    #print("Image Size: ", img.shape)
    #print("Image Data type: ", img.dtype)

    """Splitting the image into it's color channels """
    b,g,r = cv2.split(img)

    # Zero Channels 
    #zeros = np.zeros_like(b)

    # Merge each channel to visualize its color contribution
    #blue_img = cv2.merge([b, zeros, zeros])
    #green_img = cv2.merge([zeros, g, zeros])
    #red_img = cv2.merge([zeros, zeros, r])


    CVImageOverWrite(img=img)

    """Printing the image using image show imshow method"""
    #CVImageShow(img=img)