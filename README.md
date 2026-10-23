# Learning Computer Vision

A hands-on collection of Python examples for learning image processing and computer vision with OpenCV and NumPy. The repository is organized by topic; each example writes its result to `outputs/` and can optionally display it in a window.

## Getting started

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:CV_LEARNING_IMAGE_DIR = (Resolve-Path .\assets\images).Path
```

For macOS/Linux, activate the virtual environment with `source .venv/bin/activate` and set the image directory with:

```sh
export CV_LEARNING_IMAGE_DIR="$(pwd)/assets/images"
```

`CV_LEARNING_IMAGE_DIR` is read from the environment, not embedded as a machine-specific path in the examples. Set it to any directory containing the sample image filenames. The checked-in inputs are in `assets/images/`.

## Run an example

Run commands from the repository root:

```sh
python -m examples.fundamentals.image_io
python -m examples.filtering_and_edges.canny_edge_detector
python -m examples.features_and_matching.descriptor_matching
```

Results are created under `outputs/<topic>/<example>/`. Pass `--show` to display the result interactively; saving output does not require a display. `outputs/` is git-ignored.

## Visual outputs

Image-processing examples save labeled result images when run:

| Topic | Example result |
| --- | --- |
| Fundamentals | `outputs/fundamentals/image_io/color_channels.png` |
| Filtering and edges | `outputs/filtering_and_edges/image_filtering/filters.png` |
| Features and matching | `outputs/features_and_matching/descriptor_matching/matching_methods.png` |

## Learning paths

| Topic | What you will explore |
| --- | --- |
| [Fundamentals](examples/fundamentals/README.md) | Image I/O, channels, resizing, cropping, and flips |
| [Image operations](examples/image_operations/README.md) | Image blending, brightness, contrast, and bitwise operations |
| [Filtering and edges](examples/filtering_and_edges/README.md) | Convolution, smoothing, gradients, and Canny edge detection |
| [Features and matching](examples/features_and_matching/README.md) | Harris corners, SIFT, ORB, descriptor matching, and RANSAC |
| [Camera](examples/camera/README.md) | Live camera thresholding and filter previews |
| [Metadata](examples/metadata/README.md) | Reading selected EXIF photography metadata |

## Dependencies

Install the packages listed in [requirements.txt](requirements.txt). Camera examples also require an available camera and a graphical display.

## Notes

- Inputs are read from `CV_LEARNING_IMAGE_DIR`; examples never overwrite them.
- Generated output images and metadata are written under `outputs/`.
- The hand-written SIFT example is an educational implementation, not a replacement for the optimized OpenCV implementation.
- Verify the provenance and redistribution rights of sample images before reusing them in other projects.
