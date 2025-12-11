import cv2

def access_camera():
    web_cam_feed = cv2.VideoCapture(0)

    if not web_cam_feed:
        print("Error: Could not open video stream.")
        exit()

    while True:
        ret,frame = web_cam_feed.read()

        if not ret:
            print("Error: Error reading video stream feed")
            break

        greyFrame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

        meanThreshFrame = cv2.adaptiveThreshold(greyFrame,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

        gaussianThreshFrame = cv2.adaptiveThreshold(greyFrame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        cv2.imshow("Video Stream Feed Mean Thresholding",meanThreshFrame)
        cv2.imshow("Video Stream Feed Gaussian Thresholding", gaussianThreshFrame)



        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            

    web_cam_feed.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    access_camera()