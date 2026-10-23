"""Capture adaptive-threshold previews from a camera."""

from datetime import datetime

import cv2

from examples.common import make_contact_sheet, save_image


def access_camera() -> None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open camera device 0.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from camera device 0.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_threshold = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
            gaussian_threshold = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            cv2.imshow("Mean threshold", mean_threshold)
            cv2.imshow("Gaussian threshold", gaussian_threshold)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                result = make_contact_sheet(
                    [
                        ("Mean adaptive threshold", mean_threshold),
                        ("Gaussian adaptive threshold", gaussian_threshold),
                    ]
                )
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                save_image(f"camera/adaptive_threshold_{timestamp}.png", result)
            elif key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    access_camera()
