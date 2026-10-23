# Image operations

Explore pixel-wise combinations and intensity adjustments.

| Module | Concepts | Output |
| --- | --- | --- |
| `image_addition` | Image sizing and weighted blending | `outputs/image_operations/image_addition/blended_images.png` |
| `image_enhancement` | Brightness, contrast, bitwise AND, and bitwise OR | `outputs/image_operations/image_enhancement/enhancement.png` |

Run from the repository root:

```sh
python -m examples.image_operations.image_addition
python -m examples.image_operations.image_enhancement
```

Both examples save a labeled comparison sheet. Add `--show` to open it.
