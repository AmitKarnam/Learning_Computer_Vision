# SIFT scale-space pyramid

This diagram summarizes the image pyramid used by the hand-written SIFT example. Each octave is blurred at multiple scales, then adjacent Gaussian images are differenced to form Difference-of-Gaussian (DoG) images.

```text
Input grayscale image
        |
        +-- resize x 1.0 --> [G0 G1 G2 G3 G4 G5]
        |                       |
        |                       +--> [D0 D1 D2 D3 D4]
        |
        +-- resize x 0.5 --> [G0 G1 G2 G3 G4 G5]
        |                       |
        |                       +--> [D0 D1 D2 D3 D4]
        |
        +-- resize x 0.25 -> [G0 G1 G2 G3 G4 G5]
                                |
                                +--> [D0 D1 D2 D3 D4]
```
