# Fundamentals

These examples introduce the basic image representation and common geometric edits.

| Module | Concepts | Output |
| --- | --- | --- |
| `image_io` | Reading images, image dimensions, and BGR channel visualization | `outputs/fundamentals/image_io/color_channels.png` |
| `image_manipulation` | Direct pixel edits, resize interpolation, flips, and cropping | `outputs/fundamentals/image_manipulation/transforms.png` |

Run from the repository root after setting `CV_LEARNING_IMAGE_DIR`:

```sh
python -m examples.fundamentals.image_io
python -m examples.fundamentals.image_manipulation
```

Add `--show` to display the saved result in an OpenCV window.
