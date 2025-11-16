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

        cv2.imshow("Video Stream Feed",frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    web_cam_feed.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    access_camera()