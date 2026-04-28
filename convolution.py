import numpy as np
import cv2

imgPath = "C:\\Users\\amitk\\Learning_Computer_Vision\\images\\ktm_adventure_390_x.jpeg"

def calculate_padding_length(n: int) -> int:
    if n == 0 or n%2 == 0:
        return -1
    return (n-1) // 2

def build_new_image_matrix_with_reflect_padding(img: np.ndarray, pad: np.ndarray) -> np.ndarray:
    # TODO: Break this into basic matrix multiplication 
    return np.pad(img, ((pad, pad), (pad, pad), (0,0)), mode='reflect')

def convolute(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Given a n*n matrix with n*n kernel return a output matrix, which is a result of convolution operation"""
    k = kernel.shape[0]
    h,w,c = img.shape

    padding_length = calculate_padding_length(len(kernel))
    if padding_length <= 0:
        return "Error with padding length"

    paddedImg = build_new_image_matrix_with_reflect_padding(img, padding_length)

    output = np.zeros(img.shape, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = paddedImg[i:i+k,j:j+k]
            for ch in range(c):
                output[i,j,ch] = np.sum(region[:,:,ch] * kernel)
    
    return output


def main():
    img = cv2.imread(imgPath)

    img = img.astype(np.float32)

    kernel = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ], dtype=np.float32)

    imgBlur = convolute(img, kernel)

    result = np.clip(imgBlur,0,255).astype(np.uint8)

    cv2.imshow("Original Image", img.astype(np.uint8))
    cv2.imshow("Convolution Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


