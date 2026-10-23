# Filtering and edges

Learn how image kernels and gradients reveal structure at different scales.

| Module | Concepts | Output |
| --- | --- | --- |
| `convolution` | Hand-written box, Gaussian, Sobel, and Laplacian convolutions | `outputs/filtering_and_edges/convolution/kernels.png` |
| `image_filtering` | OpenCV box, Gaussian, median, and bilateral filters | `outputs/filtering_and_edges/image_filtering/filters.png` |
| `image_gradient_analysis` | RGB strip profiles, Sobel response, and Laplacian response | `outputs/filtering_and_edges/image_gradient_analysis/gradient_profile.png` |
| `canny_edge_detector` | Sobel gradients, non-maximum suppression, double threshold, hysteresis, and OpenCV comparison | `outputs/filtering_and_edges/canny_edge_detector/edge_stages.png` |

Run any module from the repository root:

```sh
python -m examples.filtering_and_edges.convolution
python -m examples.filtering_and_edges.image_filtering
python -m examples.filtering_and_edges.image_gradient_analysis
python -m examples.filtering_and_edges.canny_edge_detector
```

Add `--show` to display the OpenCV result or interact with the gradient plot. In the gradient plot, click the image to move the selected row; the initial plot is saved automatically.
