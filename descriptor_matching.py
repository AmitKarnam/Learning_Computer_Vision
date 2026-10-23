import cv2
import matplotlib.pyplot as plt
import numpy as np

def ComputeDescriptorsSIFT(img: np.ndarray):
    sift_obj = cv2.SIFT_create()
    keypoints,descriptors = sift_obj.detectAndCompute(img, None)

    return keypoints,descriptors

def ComputerDescriptorsORB(img: np.ndarray):
    orb_obj = cv2.ORB_create()
    keypoints,descriptors = orb_obj.detectAndCompute(img,None)

    return keypoints,descriptors

def BruteForceSearchWithRegularMatchesCrossCheckFalse(descriptorsA, descriptorsB):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.match(descriptorsA, descriptorsB)
    return sorted(matches, key=lambda m: m.distance)

def BruteForceSearchWithRegularMatchesCrossCheckTrue(descriptorsA, descriptorsB): 
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptorsA,descriptorsB)

    return matches

def  BruteForceSearchWithSiftKNN(descriptorsA, descriptorsB):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptorsA,descriptorsB,k=2)

    good_descriptors = []
    for m,n in matches:
        if m.distance < (0.75*n.distance):
            good_descriptors.append([m])

    return good_descriptors

def BruteForceSearchWithORB(descriptorsA, descriptorsB):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptorsA,descriptorsB,k=2)

    good_descriptors = []
    for m,n in matches:
        if m.distance < (0.75*n.distance):
            good_descriptors.append(m)

    return good_descriptors
   

def main():
    imgA = cv2.imread("C:\\Users\\amitk\\workspace\\Learning_Computer_Vision\\images\\Tour_Eiffel.jpg")
    imgB = cv2.imread("C:\\Users\\amitk\\workspace\\Learning_Computer_Vision\\images\\Eiffle_Tower.jpg")

    keypointsA, descriptorsA = ComputeDescriptorsSIFT(imgA)
    keypointsB, descriptorsB = ComputeDescriptorsSIFT(imgB)

    # # For SIFT + BF with regular matches + CrossCheck set to False
    # matches = BruteForceSearchWithRegularMatchesCrossCheckFalse(descriptorsA,descriptorsB)
    # SIFT_BF_REGULAR_MATCHES  = cv2.drawMatches(imgA, keypointsA, imgB, keypointsB, matches[:100], None,  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Resultant descriptor comparision from SIFT and Brute-Force with Regular matching and Cross Check False", SIFT_BF_REGULAR_MATCHES)

    # # For SIFT + BF with regular matches + CrossCheck set to True
    # matches = BruteForceSearchWithRegularMatchesCrossCheckTrue(descriptorsA,descriptorsB)
    # SIFT_BF_REGULAR_MATCHES  = cv2.drawMatches(imgA, keypointsA, imgB, keypointsB, matches[:100], None,  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Resultant descriptor comparision from SIFT and Brute-Force with Regular matching and Cross Check True", SIFT_BF_REGULAR_MATCHES)

    # For SIFT + BF with KNN
    good_descriptors = BruteForceSearchWithSiftKNN(descriptorsA=descriptorsA,descriptorsB=descriptorsB)
    SIFT_BF_KNN_imgResult = cv2.drawMatchesKnn(imgA,keypointsA,imgB,keypointsB,good_descriptors,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Resultant descriptor comparision from SIFT and Brute-Force KNN matching",SIFT_BF_KNN_imgResult)

    # For ORB + BF with KNN
    keypointsORB_A, descriptorsORB_A = ComputerDescriptorsORB(imgA)
    keypointsORB_B, descriptorsORB_B = ComputerDescriptorsORB(imgB)
    good_descriptors = BruteForceSearchWithORB(descriptorsORB_A,descriptorsORB_B)
    ORB_BF_KNN_imgResult = cv2.drawMatches(imgA, keypointsORB_A, imgB, keypointsORB_B, good_descriptors, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Resultant descriptor comparision from ORB and Brute-ForceKNN matching", ORB_BF_KNN_imgResult)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()