import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE_PATH = r"C:\\Users\\amitk\\workspace\\Learning_Computer_Vision\\images\\rocks.jpg"

# Half strip size.
# Actual strip height = 2 * STRIP_HALF_HEIGHT
STRIP_HALF_HEIGHT = 10


class ImageGradientAnalyzer:

    def __init__(self, image_path):

        self.img_bgr = cv2.imread(image_path)

        if self.img_bgr is None:
            raise Exception(
                f"Unable to load image: {image_path}"
            )

        self.img = cv2.cvtColor(
            self.img_bgr,
            cv2.COLOR_BGR2RGB
        )

        self.height, self.width, _ = self.img.shape

        self.selected_row = self.height // 2

        # --------------------------------------------------
        # SobelX
        # --------------------------------------------------

        self.sobel = cv2.Sobel(
            self.img.astype(np.float32),
            cv2.CV_32F,
            dx=1,
            dy=0,
            ksize=3
        )

        # --------------------------------------------------
        # Laplacian
        # --------------------------------------------------

        self.laplacian = cv2.Laplacian(
            self.img.astype(np.float32),
            cv2.CV_32F
        )

        # --------------------------------------------------
        # Combined Sobel Magnitude
        # --------------------------------------------------

        self.sobel_combined = np.sqrt(
            self.sobel[:, :, 0] ** 2 +
            self.sobel[:, :, 1] ** 2 +
            self.sobel[:, :, 2] ** 2
        )

        # --------------------------------------------------
        # Combined Laplacian Magnitude
        # --------------------------------------------------

        self.laplacian_combined = np.sqrt(
            self.laplacian[:, :, 0] ** 2 +
            self.laplacian[:, :, 1] ** 2 +
            self.laplacian[:, :, 2] ** 2
        )

        self.create_figure()

    def get_strip_bounds(self):

        start = max(
            0,
            self.selected_row - STRIP_HALF_HEIGHT
        )

        end = min(
            self.height,
            self.selected_row + STRIP_HALF_HEIGHT
        )

        return start, end

    def get_row_data(self):

        start, end = self.get_strip_bounds()

        # --------------------------------------------------
        # Average RGB strip
        # --------------------------------------------------

        rgb_strip = np.mean(
            self.img[start:end],
            axis=0
        )

        red = rgb_strip[:, 0]
        green = rgb_strip[:, 1]
        blue = rgb_strip[:, 2]

        # --------------------------------------------------
        # Average Sobel strip
        # --------------------------------------------------

        sobel_strip = np.mean(
            self.sobel[start:end],
            axis=0
        )

        sobel_r = sobel_strip[:, 0]
        sobel_g = sobel_strip[:, 1]
        sobel_b = sobel_strip[:, 2]

        # --------------------------------------------------
        # Average Laplacian strip
        # --------------------------------------------------

        lap_strip = np.mean(
            self.laplacian[start:end],
            axis=0
        )

        lap_r = lap_strip[:, 0]
        lap_g = lap_strip[:, 1]
        lap_b = lap_strip[:, 2]

        # --------------------------------------------------
        # Combined Sobel
        # --------------------------------------------------

        combined_sobel = np.mean(
            self.sobel_combined[start:end],
            axis=0
        )

        # --------------------------------------------------
        # Combined Laplacian
        # --------------------------------------------------

        combined_lap = np.mean(
            self.laplacian_combined[start:end],
            axis=0
        )

        return (
            red,
            green,
            blue,
            sobel_r,
            sobel_g,
            sobel_b,
            combined_sobel,
            lap_r,
            lap_g,
            lap_b,
            combined_lap,
        )

    def create_figure(self):

        self.fig, self.axes = plt.subplots(
            6,
            1,
            figsize=(16, 18)
        )

        self.ax_image = self.axes[0]
        self.ax_rgb = self.axes[1]
        self.ax_sobel = self.axes[2]
        self.ax_combined_sobel = self.axes[3]
        self.ax_laplacian = self.axes[4]
        self.ax_combined_laplacian = self.axes[5]

        self.fig.canvas.mpl_connect(
            "button_press_event",
            self.on_click
        )

        self.update_display()

    def update_display(self):

        (
            red,
            green,
            blue,
            sobel_r,
            sobel_g,
            sobel_b,
            combined_sobel,
            lap_r,
            lap_g,
            lap_b,
            combined_lap,
        ) = self.get_row_data()

        start, end = self.get_strip_bounds()

        # ==================================================
        # IMAGE
        # ==================================================

        self.ax_image.clear()

        self.ax_image.imshow(self.img)

        self.ax_image.axhline(
            start,
            color="yellow",
            linewidth=2
        )

        self.ax_image.axhline(
            end,
            color="yellow",
            linewidth=2
        )

        self.ax_image.fill_between(
            [0, self.width],
            start,
            end,
            color="yellow",
            alpha=0.25
        )

        self.ax_image.set_title(
            f"Selected Strip ({start} -> {end})"
        )

        # ==================================================
        # RGB PROFILE
        # ==================================================

        self.ax_rgb.clear()

        self.ax_rgb.plot(
            red,
            color="red",
            label="Red"
        )

        self.ax_rgb.plot(
            green,
            color="green",
            label="Green"
        )

        self.ax_rgb.plot(
            blue,
            color="blue",
            label="Blue"
        )

        self.ax_rgb.set_title(
            "RGB Intensity Profile"
        )

        self.ax_rgb.set_ylabel(
            "Intensity"
        )

        self.ax_rgb.grid(True)
        self.ax_rgb.legend()

        # ==================================================
        # SOBEL RGB
        # ==================================================

        self.ax_sobel.clear()

        self.ax_sobel.plot(
            sobel_r,
            color="red",
            label="Red Sobel"
        )

        self.ax_sobel.plot(
            sobel_g,
            color="green",
            label="Green Sobel"
        )

        self.ax_sobel.plot(
            sobel_b,
            color="blue",
            label="Blue Sobel"
        )

        self.ax_sobel.axhline(
            0,
            linestyle="--"
        )

        self.ax_sobel.set_title(
            "RGB SobelX Response"
        )

        self.ax_sobel.set_ylabel(
            "Gradient"
        )

        self.ax_sobel.grid(True)
        self.ax_sobel.legend()

        # ==================================================
        # COMBINED SOBEL
        # ==================================================

        self.ax_combined_sobel.clear()

        self.ax_combined_sobel.plot(
            combined_sobel,
            color="black",
            linewidth=1.5,
            label="Combined Sobel Magnitude"
        )

        self.ax_combined_sobel.set_title(
            "Combined Sobel Magnitude"
        )

        self.ax_combined_sobel.set_ylabel(
            "Magnitude"
        )

        self.ax_combined_sobel.grid(True)
        self.ax_combined_sobel.legend()

        # ==================================================
        # RGB LAPLACIAN
        # ==================================================

        self.ax_laplacian.clear()

        self.ax_laplacian.plot(
            lap_r,
            color="red",
            label="Red Laplacian"
        )

        self.ax_laplacian.plot(
            lap_g,
            color="green",
            label="Green Laplacian"
        )

        self.ax_laplacian.plot(
            lap_b,
            color="blue",
            label="Blue Laplacian"
        )

        self.ax_laplacian.axhline(
            0,
            linestyle="--"
        )

        self.ax_laplacian.set_title(
            "RGB Laplacian Response"
        )

        self.ax_laplacian.set_ylabel(
            "2nd Derivative"
        )

        self.ax_laplacian.grid(True)
        self.ax_laplacian.legend()

        # ==================================================
        # COMBINED LAPLACIAN
        # ==================================================

        self.ax_combined_laplacian.clear()

        self.ax_combined_laplacian.plot(
            combined_lap,
            color="black",
            linewidth=1.5,
            label="Combined Laplacian Magnitude"
        )

        self.ax_combined_laplacian.set_title(
            "Combined Laplacian Magnitude"
        )

        self.ax_combined_laplacian.set_xlabel(
            "Pixel Position"
        )

        self.ax_combined_laplacian.set_ylabel(
            "Magnitude"
        )

        self.ax_combined_laplacian.grid(True)
        self.ax_combined_laplacian.legend()

        self.fig.tight_layout()

        self.fig.canvas.draw_idle()

        print(
            f"Row={self.selected_row} | "
            f"SobelMax={np.max(combined_sobel):.2f} | "
            f"LaplaceMax={np.max(combined_lap):.2f}"
        )

    def on_click(self, event):

        if event.inaxes != self.ax_image:
            return

        if event.ydata is None:
            return

        self.selected_row = int(event.ydata)

        self.selected_row = max(
            0,
            min(
                self.height - 1,
                self.selected_row
            )
        )

        self.update_display()

    def run(self):

        plt.show()


def main():

    analyzer = ImageGradientAnalyzer(
        IMAGE_PATH
    )

    analyzer.run()


if __name__ == "__main__":
    main()