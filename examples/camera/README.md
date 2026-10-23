# Camera

These examples use camera device `0` and display live frames. They require a working camera and graphical display.

| Module | Controls | Saved output |
| --- | --- | --- |
| `accessing_camera` | `s` saves mean and Gaussian adaptive-threshold previews; `q` quits | `outputs/camera/adaptive_threshold_<timestamp>.png` |
| `camera_filter` | `1`-`4` selects original, blur, Canny, or adaptive threshold; `s` saves; `q` quits | `outputs/camera/filter_<timestamp>.png` |

Run from the repository root:

```sh
python -m examples.camera.accessing_camera
python -m examples.camera.camera_filter
```
