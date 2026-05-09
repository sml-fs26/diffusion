"""Train a tiny UNet DDPM on MNIST 28x28.

Architecture: 2-level UNet with 16/32/64 channels and FiLM-style time
conditioning (sinusoidal → MLP → per-block linear projection added to
features). Output predicts ε. Total ~250 k params; small enough to train on
CPU in ~10–25 min.

Conventions: same DDPM linear schedule as the rest of the viz (T=200,
β ∈ [1e-4, 2e-2]). Loss = MSE(ε̂, ε) with x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε.

Output: `_artifacts/mnist_unet.pt` (state dict).
"""
from __future__ import annotations
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
SEED = 42
T = 200
BETA_MIN = 1e-4
BETA_MAX = 0.02

TIME_DIM     = 32
TIME_EMB_DIM = TIME_DIM * 4   # 128

# Channel widths — chosen for CPU training speed; full UNet pattern, just narrow.
# 16/32/64 was 1.1 s/step on CPU; 8/16/32 cuts to ~0.25 s/step at the cost of
# some quality loss on tougher digits but still produces clean MNIST samples.
C0, C1, C2 = 8, 16, 32

LR        = 2e-4
BATCH     = 64        # better samples/s ratio than 128 on this CPU
N_STEPS   = 8_000     # ~28 min on CPU at this size; 512k samples seen
LOG_EVERY = 250

HERE = Path(__file__).parent
MNIST_ROOT = HERE / "_mnist_cache"
ART_DIR = HERE / "_artifacts"
ART_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


def _gn(channels: int, n_groups: int = 8) -> nn.GroupNorm:
    g = min(n_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_ch)
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """2-level UNet, ~250 k params, predicts ε from (x_t, t)."""

    def __init__(self):
        super().__init__()
        emb = TIME_EMB_DIM
        self.time_emb = TimeEmbedding(TIME_DIM)
        self.in_conv = nn.Conv2d(1, C0, 3, padding=1)
        # Down
        self.d1 = ResBlock(C0, C1, emb)            # (28,28)
        self.d2 = ResBlock(C1, C2, emb)            # (14,14) after pool
        # Bottleneck
        self.b1 = ResBlock(C2, C2, emb)            # ( 7, 7) after pool
        # Up
        self.u2 = ResBlock(C2 + C2, C1, emb)       # back to (14,14)
        self.u1 = ResBlock(C1 + C1, C0, emb)       # back to (28,28)
        # Out
        self.out_norm = _gn(C0)
        self.out_conv = nn.Conv2d(C0, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_emb(t)
        h = self.in_conv(x)                                      # (B, C0, 28,28)
        s1 = self.d1(h, emb)                                     # (B, C1, 28,28)
        h2 = F.avg_pool2d(s1, 2)                                 # (B, C1, 14,14)
        s2 = self.d2(h2, emb)                                    # (B, C2, 14,14)
        h3 = F.avg_pool2d(s2, 2)                                 # (B, C2,  7, 7)
        b = self.b1(h3, emb)                                     # (B, C2,  7, 7)
        u2 = F.interpolate(b, scale_factor=2, mode='nearest')    # (B, C2, 14,14)
        u2 = self.u2(torch.cat([u2, s2], dim=1), emb)            # (B, C1, 14,14)
        u1 = F.interpolate(u2, scale_factor=2, mode='nearest')   # (B, C1, 28,28)
        u1 = self.u1(torch.cat([u1, s1], dim=1), emb)            # (B, C0, 28,28)
        return self.out_conv(F.silu(self.out_norm(u1)))          # (B, 1, 28,28)


# ---------------------------------------------------------------------------
def make_schedule():
    t = torch.arange(T, dtype=torch.float64)
    betas = BETA_MIN + (BETA_MAX - BETA_MIN) * t / (T - 1)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas.float(), alphas.float(), alpha_bars.float()


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cpu")
    # Use all physical cores. On Apple Silicon CPU count includes efficiency
    # cores; using all of them gives a meaningful speedup on conv2d.
    torch.set_num_threads(max(1, os.cpu_count() or 8))
    print(f"device = {device}, threads = {torch.get_num_threads()}")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    ds = datasets.MNIST(root=MNIST_ROOT, train=True, download=True, transform=tf)
    n = len(ds)
    print(f"MNIST train size = {n}")

    print("Pre-loading MNIST into RAM …")
    all_x = torch.empty((n, 1, 28, 28), dtype=torch.float32)
    for i in range(n):
        img, _ = ds[i]
        all_x[i] = img
    print(f"  done. mean={all_x.mean():+.3f} std={all_x.std():.3f}")

    model = TinyUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    betas, alphas, alpha_bars = make_schedule()
    sqrt_ab = torch.sqrt(alpha_bars)
    sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars)

    g = torch.Generator(device="cpu").manual_seed(SEED)
    losses = []
    t0 = time.time()

    model.train()
    for step in range(1, N_STEPS + 1):
        idx = torch.randint(0, n, (BATCH,), generator=g)
        x0 = all_x[idx].to(device)                                   # (B, 1, 28, 28)
        t = torch.randint(0, T, (BATCH,), generator=g).to(device)    # (B,)
        eps = torch.randn(BATCH, 1, 28, 28, generator=g).to(device)

        sa = sqrt_ab[t][:, None, None, None]
        soa = sqrt_one_minus_ab[t][:, None, None, None]
        x_t = sa * x0 + soa * eps

        eps_pred = model(x_t, t)
        loss = F.mse_loss(eps_pred, eps)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))
        if step % LOG_EVERY == 0 or step == 1:
            recent = float(np.mean(losses[-LOG_EVERY:]))
            elapsed = time.time() - t0
            rate = step / elapsed if elapsed > 0 else float('nan')
            eta = (N_STEPS - step) / rate if rate > 0 else float('nan')
            print(f"  step {step:6d}/{N_STEPS}  loss={recent:.4f}  "
                  f"{rate:.1f} step/s  eta={eta/60:.1f} min")

    initial = float(np.mean(losses[:LOG_EVERY]))
    final = float(np.mean(losses[-LOG_EVERY:]))
    print(f"loss: first={initial:.4f}  last={final:.4f}  ratio={final/initial:.3f}")
    print(f"total wall time: {(time.time() - t0)/60:.1f} min")

    torch.save(model.state_dict(), ART_DIR / "mnist_unet.pt")
    import json as _json
    with open(ART_DIR / "unet_loss_curve.json", "w") as f:
        _json.dump({"initial": initial, "final": final, "losses": losses}, f)
    print(f"wrote {ART_DIR / 'mnist_unet.pt'}")


if __name__ == "__main__":
    main()
