Input grayscale image
        │
        ├── resize × 1.0 ──→ [G0 G1 G2 G3 G4 G5]
        │                         │
        │                         └──→ [D0 D1 D2 D3 D4]
        │
        ├── resize × 0.5 ──→ [G0 G1 G2 G3 G4 G5]
        │                         │
        │                         └──→ [D0 D1 D2 D3 D4]
        │
        └── resize × 0.25 ─→ [G0 G1 G2 G3 G4 G5]
                                  │
                                  └──→ [D0 D1 D2 D3 D4]