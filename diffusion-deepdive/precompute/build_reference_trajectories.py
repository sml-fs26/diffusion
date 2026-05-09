"""Run the trained MNIST DDPM in reverse for 4 fixed seeds and snapshot at
t ∈ {200, 150, 100, 50, 0}. Used as a fast fallback in Scene 6 of the viz.

Note on `t = 200`: the schedule index runs 0..199, so the snapshot at "t=200"
is x_T (pure noise) before any reverse step. The remaining snapshots use the
state x_t after we've stepped from t=200 down to t (i.e. snapshot at t=150
means we've completed reverse steps t = 200 → 150).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from train_mnist_ddpm import (
    TinyMLP, HIDDEN_DIM, T, BETA_MIN, BETA_MAX, INPUT_DIM, ART_DIR,
)

SEED = 42
N_TRAJECTORIES = 4
SNAPSHOT_TS = [T, 150, 100, 50, 0]   # 200, 150, 100, 50, 0

# The MLP-without-conv backbone is intentionally underpowered — see scene 7's
# takeaways. With high-loss noise predictions, the reverse trajectory can
# diverge exponentially (small ε̂ errors get amplified by 1/√α_t at every
# step). Clipping each reverse-step state into a generous range matches what
# production DDPM implementations do and keeps the demo stable. Disable by
# setting CLIP_RANGE = None.
CLIP_RANGE = (-1.5, 1.5)


def make_schedule_np():
    t = np.arange(T, dtype=np.float64)
    betas = BETA_MIN + (BETA_MAX - BETA_MIN) * t / (T - 1)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars


def reverse_sample(model, seed: int, snapshot_ts):
    """DDPM reverse process. Returns dict {t: pixels (np float32 array length 784)}."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    betas_np, alphas_np, alpha_bars_np = make_schedule_np()
    betas = torch.from_numpy(betas_np).float()
    alphas = torch.from_numpy(alphas_np).float()
    alpha_bars = torch.from_numpy(alpha_bars_np).float()

    x = torch.randn(1, INPUT_DIM, generator=g)  # x_T ~ N(0, I)

    snaps = {}
    if T in snapshot_ts:
        snaps[T] = x.numpy().reshape(-1).astype(np.float32).copy()

    model.eval()
    with torch.no_grad():
        for t in reversed(range(T)):  # t = T-1, T-2, …, 0
            tt = torch.tensor([t], dtype=torch.long)
            eps = model(x, tt)
            ab_t = alpha_bars[t]
            a_t = alphas[t]
            b_t = betas[t]

            # mean of p(x_{t-1} | x_t)
            coef = (1 - a_t) / torch.sqrt(1 - ab_t)
            mean = (1.0 / torch.sqrt(a_t)) * (x - coef * eps)

            if t > 0:
                # use beta_t as the variance (DDPM "fixed_small" choice)
                noise = torch.randn(x.shape, generator=g)
                x = mean + torch.sqrt(b_t) * noise
            else:
                x = mean

            if CLIP_RANGE is not None:
                x = torch.clamp(x, CLIP_RANGE[0], CLIP_RANGE[1])

            # We snapshot using t-index where the convention is:
            #   "snapshot at t=k" means we've just produced x_k, so the next
            #   reverse step would consume index (k-1). After the loop body
            #   for index t, the state holds x_t (with our naming x_{t-1}
            #   internally → relabel: after consuming index t, current x is x_t).
            # To stay consistent with the spec, snapshot when t equals one of
            # snapshot_ts (excluding T which was already grabbed).
            if t in snapshot_ts and t != T:
                snaps[t] = x.numpy().reshape(-1).astype(np.float32).copy()

    return snaps


def build_reference_trajectories():
    weights_path = ART_DIR / "mnist_ddpm.pt"
    assert weights_path.exists(), f"missing {weights_path}; run train_mnist_ddpm.py first"
    model = TinyMLP(HIDDEN_DIM)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    out = []
    for i in range(N_TRAJECTORIES):
        seed = SEED + i + 1   # 43, 44, 45, 46  — fixed
        snaps_dict = reverse_sample(model, seed, SNAPSHOT_TS)
        snaps = []
        for t in SNAPSHOT_TS:
            assert t in snaps_dict, f"missing snapshot t={t} for seed {seed}"
            snaps.append({"t": int(t), "pixels": snaps_dict[t].tolist()})
        out.append({"seed": seed, "snapshots": snaps})
    return out


if __name__ == "__main__":
    trajs = build_reference_trajectories()
    for tr in trajs:
        print(f"seed {tr['seed']}: {len(tr['snapshots'])} snapshots")
        for s in tr["snapshots"]:
            arr = np.array(s["pixels"])
            print(f"  t={s['t']:3d}  mean={arr.mean():+.3f}  std={arr.std():.3f}  "
                  f"range=[{arr.min():+.2f},{arr.max():+.2f}]")
