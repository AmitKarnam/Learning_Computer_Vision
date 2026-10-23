# Features and matching

Compare local feature detection and matching techniques, from hand-coded scale-space processing to OpenCV's optimized descriptors.

| Module or note | Concepts | Output |
| --- | --- | --- |
| `harris_corner_detection` | Harris corner response and thresholding | `outputs/features_and_matching/harris_corner_detection/corners.png` |
| `sift_from_scratch` | Gaussian scale space, Difference of Gaussians, keypoint localization, orientation, and descriptors | `outputs/features_and_matching/sift_from_scratch/keypoints.png` |
| `descriptor_matching` | SIFT/ORB brute-force matching, ratio test, cross-check, and RANSAC inliers | `outputs/features_and_matching/descriptor_matching/matching_methods.png` |
| [sift-pyramid.md](sift-pyramid.md) | Scale-space pyramid notes | -- |

Run from the repository root:

```sh
python -m examples.features_and_matching.harris_corner_detection
python -m examples.features_and_matching.sift_from_scratch
python -m examples.features_and_matching.descriptor_matching
```

These examples can take longer than the simpler image operations. Add `--show` to display the saved comparison.
