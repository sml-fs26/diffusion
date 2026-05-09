"""Use the trained MNIST UNet to generate a pool of reverse-process trajectories.

Each trajectory is a sequence of x_t snapshots from t=T down to t=0. The
viz's scene 6 cycles through these on `re-roll seed` to give the audience
multiple distinct generated digits without running conv inference live in the
browser.

Schedule: T=200 (matches the rest of the viz). Snapshots taken at every
SNAPSHOT_STRIDE'th step (default 10), giving 21 snapshots per trajectory.

Reverse step uses σ_t = √β_t (DDPM "fixed_small"). x is clipped to [-1.5, 1.5]
each step — the UNet was trained against MSE not perceptual loss, and clipping
keeps dynamic range bounded across long reverse chains. Cosmetic at best for a
well-trained UNet; matches what the live JS reverse path does.

Output: list of {seed, snapshots: [{t, pixels (length 784)}]}.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from train_mnist_unet import TinyUNet, T, BETA_MIN, BETA_MAX, ART_DIR


N_TRAJECTORIES = 16
SNAPSHOT_STRIDE = 10        # snapshot every 10 reverse steps → 21 snapshots
CLIP_RANGE = (-1.5, 1.5)
SEED_BASE = 1000


def _make_schedule():
    t = np.arange(T, dtype=np.float64)
    betas = BETA_MIN + (BETA_MAX - BETA_MIN) * t / (T - 1)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars


def _snapshot_ts():
    """Return the t indices at which we record a snapshot, sorted descending.

    Snapshot at t means: state x_t (the value before reverse-stepping it down
    to x_{t-1}). t=T is x_T (pure noise); t=0 is the final generated digit.
    """
    ts = list(range(T, -1, -SNAPSHOT_STRIDE))
    if 0 not in ts:
        ts.append(0)
    return ts  # e.g. [200, 190, ..., 10, 0]


@torch.no_grad()
def reverse_one(model: TinyUNet, seed: int, snapshot_ts):
    g = torch.Generator(device="cpu").manual_seed(seed)
    betas_np, alphas_np, alpha_bars_np = _make_schedule()
    betas = torch.from_numpy(betas_np).float()
    alphas = torch.from_numpy(alphas_np).float()
    alpha_bars = torch.from_numpy(alpha_bars_np).float()

    x = torch.randn(1, 1, 28, 28, generator=g)  # x_T
    snaps = {}
    if T in snapshot_ts:
        snaps[T] = x.numpy().reshape(-1).astype(np.float32).copy()

    model.eval()
    for t in reversed(range(T)):  # t = T-1, T-2, ..., 0
        tt = torch.tensor([t], dtype=torch.long)
        eps = model(x, tt)
        ab_t = alpha_bars[t]
        a_t = alphas[t]
        b_t = betas[t]

        coef = (1.0 - a_t) / torch.sqrt(1.0 - ab_t)
        mean = (1.0 / torch.sqrt(a_t)) * (x - coef * eps)

        if t > 0:
            noise = torch.randn(x.shape, generator=g)
            x = mean + torch.sqrt(b_t) * noise
        else:
            x = mean

        if CLIP_RANGE is not None:
            x = torch.clamp(x, CLIP_RANGE[0], CLIP_RANGE[1])

        # The state we just produced is x_t (the loop variable's value).
        if t in snapshot_ts and t != T:
            snaps[t] = x.numpy().reshape(-1).astype(np.float32).copy()

    return snaps


def build_unet_trajectories():
    weights_path = ART_DIR / "mnist_unet.pt"
    assert weights_path.exists(), f"missing {weights_path}; run train_mnist_unet.py first"

    model = TinyUNet()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    snapshot_ts = _snapshot_ts()
    print(f"snapshots per trajectory: {len(snapshot_ts)} (t = {snapshot_ts[:3]} … {snapshot_ts[-2:]})")

    out = []
    for i in range(N_TRAJECTORIES):
        seed = SEED_BASE + i
        snaps_dict = reverse_one(model, seed, snapshot_ts)
        snaps = []
        for t in snapshot_ts:
            assert t in snaps_dict, f"missing snapshot t={t} for seed {seed}"
            snaps.append({"t": int(t), "pixels": snaps_dict[t].tolist()})
        out.append({"seed": seed, "snapshots": snaps})
        last = np.array(snaps[-1]["pixels"])
        print(f"  traj {i+1:2d}/{N_TRAJECTORIES} seed={seed}: "
              f"x_0 mean={last.mean():+.3f} std={last.std():.3f}")
    return out


if __name__ == "__main__":
    trajs = build_unet_trajectories()
    out_path = ART_DIR / "unet_trajectories.json"
    with open(out_path, "w") as f:
        json.dump(trajs, f)
    size_kb = out_path.stat().st_size / 1024
    print(f"\nwrote {out_path}  ({size_kb:.0f} KB)")
