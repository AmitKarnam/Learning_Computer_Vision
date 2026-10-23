import cv2

def main():
    try:
        img1 = cv2.imread("./images/checkerboard_color.png")
        img2 = cv2.imread("./images/rectangle.jpg")

        h2, w2 = img2.shape[:2]
        img1_resized = cv2.resize(img1, (w2, h2),
                              interpolation=cv2.INTER_AREA if (img1.shape[0] > h2 or img1.shape[1] > w2) else cv2.INTER_CUBIC)
    
    except Exception as e:
        print("An error occured",e)

    finalImg = cv2.add(img2,img1_resized)

    cv2.imshow("Final Image after adding",finalImg)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()