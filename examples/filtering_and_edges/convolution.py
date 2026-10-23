import numpy as np
from examples.common import load_image, make_contact_sheet, save_image, show_images


def calculate_padding_length(n: int) -> int:
    if n == 0 or n%2 == 0:
        return -1
    return (n-1) // 2

def build_new_image_matrix_with_reflect_padding(img: np.ndarray, pad: int) -> np.ndarray:
    # TODO: Break this into basic matrix multiplication 
    padded_matrix = np.pad(img, ((pad, pad), (pad, pad), (0,0)), mode='reflect')
    return padded_matrix

def BoxBlurConvolute(img: np.ndarray) -> np.ndarray:
    """Given a n*n matrix return a clipped output matrix, which is a result of convolution operation"""
    kernel = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ], dtype=np.float32)

    k = kernel.shape[0]
    h,w,c = img.shape

    padding_length = calculate_padding_length(len(kernel))
    if padding_length <= 0:
        raise ValueError("Kernel size must be an odd positive integer.")

    paddedImg = build_new_image_matrix_with_reflect_padding(img, padding_length)

    output = np.zeros(img.shape, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = paddedImg[i:i+k,j:j+k]
            for ch in range(c):
                output[i,j,ch] = np.sum(region[:,:,ch] * kernel)

    output = np.clip(output,0,255).astype(np.uint8)
    
    return output

def GuassianBlurConvolute(img: np.array) -> np.ndarray:
    """ Gaussian Blur, where the emphasis is on nearby pixels than the distant ones"""
    kernel = np.array([
        [1/16,2/16,1/16],
        [2/16,4/16,2/16],
        [1/16,2/16,1/16],
    ],dtype=np.float32)

    k = kernel.shape[0]
    h,w,c = img.shape

    paddingLength = calculate_padding_length(k)
    if paddingLength < 0:
        raise ValueError("Kernel size must be an odd positive integer.")
    
    paddedImg = build_new_image_matrix_with_reflect_padding(img,paddingLength)

    output = np.zeros(img.shape, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = paddedImg[i:i+k,j:j+k]
            for ch in range(c):
                output[i,j,ch] = np.sum(region[:,:,ch] * kernel)

    output = np.clip(output,0,255).astype(np.uint8)

    return output

def SobelXConvolute(img: np.array) -> np.ndarray:
    """ It highlights vertical edges. """

    kernel = np.array([
        [1,0,-1],
        [2,0,-2],
        [1,0,-1],
    ], dtype=np.float32)

    k = kernel.shape[0]
    h,w,c = img.shape

    paddingLength = calculate_padding_length(k)
    paddedImg = build_new_image_matrix_with_reflect_padding(img,paddingLength)

    output = np.zeros(img.shape,dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = paddedImg[i:i+k,j:j+k]
            for ch in range(c):
                output[i,j,ch] = np.sum(region[:,:,ch] * kernel)

    output = np.abs(output)
    output = np.clip(output,0,255).astype(np.uint8)

    return output

def SobelYConvolute(img: np.array) -> np.ndarray:
    """ It highlights horizontal edges. """

    kernel = np.array([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1],
    ], dtype=np.float32)

    k = kernel.shape[0]
    h,w,c = img.shape

    paddingLength = calculate_padding_length(k)
    paddedImg = build_new_image_matrix_with_reflect_padding(img,paddingLength)

    output = np.zeros(img.shape,dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = paddedImg[i:i+k,j:j+k]
            for ch in range(c):
                output[i,j,ch] = np.sum(region[:,:,ch] * kernel)

    output = np.abs(output)
    output = np.clip(output,0,255).astype(np.uint8)

    return output

def LaplaceConvolute(img: np.array) -> np.ndarray:
    """ Detect regions where intensity changes rapidly """

    kernel = np.array([
        [0,-1,0],
        [-1,4,-1],
        [0,-1,0],
    ], dtype=np.float32)

    k = kernel.shape[0]
    h,w,c = img.shape

    paddingLength = calculate_padding_length(k)
    paddedImg = build_new_image_matrix_with_reflect_padding(img,paddingLength)

    output = np.zeros(img.shape,dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = paddedImg[i:i+k,j:j+k]
            for ch in range(c):
                output[i,j,ch] = np.sum(region[:,:,ch] * kernel)

    output = np.clip(output,0,255).astype(np.uint8)

    return output



def main():
    img = load_image("rocks.jpg").astype(np.float32)
    boxBlur = BoxBlurConvolute(img)
    gaussianBlur = GuassianBlurConvolute(img)
    xAxisEdge = SobelXConvolute(img)
    yAxisEdge = SobelYConvolute(img)
    laplaceConvolution = LaplaceConvolute(img)
    laplaceOverGuassian = LaplaceConvolute(gaussianBlur)

    results = [
        ("Original", img.astype(np.uint8)),
        ("Box blur", boxBlur),
        ("Gaussian blur", gaussianBlur),
        ("Sobel X", xAxisEdge),
        ("Sobel Y", yAxisEdge),
        ("Laplacian", laplaceConvolution),
        ("Laplacian after Gaussian", laplaceOverGuassian),
    ]
    result = make_contact_sheet(results, columns=2)
    save_image("filtering_and_edges/convolution/kernels.png", result)
    show_images([("Convolution kernels", result)])


if __name__ == "__main__":
    main()
