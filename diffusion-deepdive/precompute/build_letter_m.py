"""Render the letter 'M' as ~300 jittered points along its strokes.

The five strokes form a classic 'M':
  s0: bottom-left  -> top-left          (left vertical)
  s1: top-left     -> middle-bottom     (left diagonal of the V)
  s2: middle-bottom -> top-right        (right diagonal of the V)
  s3: top-right    -> bottom-right      (right vertical)

We sample uniformly along arc-length proportional to each stroke's length,
total ~300 samples. Each sample is a stroke point with a small Gaussian jitter
in the direction perpendicular to that stroke (sigma = 0.02). Center+scale to
fit in [-1, 1]^2 with margin.

Seeded RNG: numpy default_rng(42).
"""
import numpy as np

SEED = 42
N_TOTAL = 300
PERP_JITTER_SIGMA = 0.02
MARGIN = 0.10  # keep points inside [-1+margin, 1-margin]^2

# Letter 'M' strokes in raw coordinates. Top y=1, bottom y=0; left x=0, right x=1.
# Middle dip goes to y=0.30 (a typical M doesn't dip all the way down).
STROKES = [
    ((0.0, 0.0), (0.0, 1.0)),    # left vertical
    ((0.0, 1.0), (0.5, 0.30)),   # left diagonal of V
    ((0.5, 0.30), (1.0, 1.0)),   # right diagonal of V
    ((1.0, 1.0), (1.0, 0.0)),    # right vertical
]


def build_letter_m():
    rng = np.random.default_rng(SEED)

    strokes = [(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64))
               for a, b in STROKES]
    lengths = np.array([np.linalg.norm(b - a) for a, b in strokes])
    total_len = lengths.sum()

    # Allocate samples per stroke proportional to length, summing to N_TOTAL exactly.
    raw = lengths / total_len * N_TOTAL
    counts = np.floor(raw).astype(int)
    remainder = N_TOTAL - counts.sum()
    # distribute remainder by largest fractional parts (deterministic)
    fracs = raw - counts
    order = np.argsort(-fracs)
    for i in range(remainder):
        counts[order[i]] += 1
    assert counts.sum() == N_TOTAL

    points = []
    expected_centers = []  # for per-point invariant 3
    for (a, b), n in zip(strokes, counts):
        if n == 0:
            continue
        d = b - a
        L = np.linalg.norm(d)
        unit = d / L
        # perpendicular (rotate by +90 deg); 2D so this is unambiguous
        perp = np.array([-unit[1], unit[0]])
        # uniform along arc-length, with both endpoints included for visual clarity
        ts = np.linspace(0.0, 1.0, n)
        for t in ts:
            center = a + t * d
            jitter = rng.normal(0.0, PERP_JITTER_SIGMA)
            p = center + jitter * perp
            points.append(p)
            expected_centers.append(center)

    pts = np.array(points)        # (N, 2)
    centers = np.array(expected_centers)

    # Center & scale.
    cmin = pts.min(axis=0)
    cmax = pts.max(axis=0)
    center_xy = 0.5 * (cmin + cmax)
    extent = (cmax - cmin).max()
    target_extent = 2.0 - 2.0 * MARGIN  # so points sit inside [-1+margin, 1-margin]
    scale = target_extent / extent
    pts_norm = (pts - center_xy) * scale
    centers_norm = (centers - center_xy) * scale

    # Sanity clip: tiny numerical fudge factor before assertions.
    assert pts_norm.min() >= -1.0 + 1e-9 - 1e-6
    assert pts_norm.max() <= 1.0 - 1e-9 + 1e-6

    # Per-point perpendicular distance from its expected stroke center —
    # used by invariant 3 in build_data.py.
    perp_dists = np.linalg.norm(pts_norm - centers_norm, axis=1)

    bbox = {
        "xMin": float(pts_norm[:, 0].min()),
        "xMax": float(pts_norm[:, 0].max()),
        "yMin": float(pts_norm[:, 1].min()),
        "yMax": float(pts_norm[:, 1].max()),
    }

    return {
        "points": pts_norm.tolist(),
        "bbox": bbox,
        "_perp_dists": perp_dists.tolist(),  # keep around for invariants in build_data
    }


if __name__ == "__main__":
    out = build_letter_m()
    pts = np.array(out["points"])
    print(f"N points     : {len(pts)}")
    print(f"x-range      : [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"y-range      : [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    pd = np.array(out["_perp_dists"])
    print(f"perp dist max: {pd.max():.4f}  (sigma was {PERP_JITTER_SIGMA})")
