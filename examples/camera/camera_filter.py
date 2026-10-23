"""Switch between live camera filters and save the current result."""

from datetime import datetime

import cv2

from examples.common import make_contact_sheet, save_image


FILTERS = ("Original", "Gaussian blur", "Canny edges", "Adaptive threshold")


def main() -> None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open camera device 0.")

    selected_filter = 0
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from camera device 0.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if selected_filter == 0:
                result = frame
            elif selected_filter == 1:
                result = cv2.GaussianBlur(frame, (13, 13), 0)
            elif selected_filter == 2:
                result = cv2.Canny(gray, 80, 150)
            else:
                result = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    7,
                )
            cv2.imshow(
                f"{FILTERS[selected_filter]} | 1-4 change filter, s save, q quit",
                result,
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                selected_filter = key - ord("1")
            elif key == ord("s"):
                sheet = make_contact_sheet([(FILTERS[selected_filter], result)])
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                save_image(f"camera/filter_{timestamp}.png", sheet)
            elif key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
