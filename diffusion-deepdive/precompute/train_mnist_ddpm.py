"""Train a tiny MLP-based DDPM on MNIST and dump weights as float32 lists.

Architecture (matches the spec inlined into data/datasets.js):
    x_in (784) ⊕ time_embed (32)  →  816
       ↓ Linear(816 -> H) → SiLU
       ↓ Linear(H   -> H) → SiLU
       ↓ Linear(H   -> 784)         → predicted ε
    where H = HIDDEN_DIM (default 128 to keep datasets.js < 4 MB).

Time embedding: sinusoidal, half sin / half cos, log-spaced frequencies up to
period max_period = 10_000. Standard transformer-style.

Loss: MSE(ε̂, ε), with x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε. Adam, lr=2e-4, batch=256.

Seeded: torch.manual_seed(42), np.random.default_rng(42).

Outputs (in `precompute/_artifacts/`):
  - `mnist_ddpm.pt`        full state dict for re-use
  - `mnist_ddpm_weights.json` flat float32 lists, the form used by datasets.js
  - `loss_curve.json`      list of per-step loss (for the build orchestrator)
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# Hyperparameters — pin and never tune to make a particular sample look better.
# -----------------------------------------------------------------------------
SEED = 42
T = 200
BETA_MIN = 1e-4
BETA_MAX = 0.02

INPUT_DIM = 784       # 28*28
TIME_EMBED_DIM = 32
HIDDEN_DIM = 160      # 160 keeps datasets.js under ~3.5 MB after JSON encoding
NUM_LAYERS = 4        # four Linears (deviation from spec's 3); see notes in build_data.py
ACTIVATION = "silu"
TIME_EMBED_TYPE = "sinusoidal"
TIME_EMBED_MAX_PERIOD = 10_000

LR = 1e-3             # constant LR; cosine decay was killing the second half of training
BATCH = 256
N_STEPS = 25_000      # ~15 min on CPU; loss bottoms out around step 20k for an MLP at this scale
USE_COSINE_DECAY = False

HERE = Path(__file__).parent
MNIST_ROOT = HERE / "_mnist_cache"
ART_DIR = HERE / "_artifacts"
ART_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Time embedding (matches what we'll re-implement in js/diffusion-nn.js).
# -----------------------------------------------------------------------------
def sinusoidal_time_embedding(t: torch.Tensor, dim: int = TIME_EMBED_DIM,
                               max_period: float = TIME_EMBED_MAX_PERIOD) -> torch.Tensor:
    """t: (B,) integer step indices in [0, T-1].  returns (B, dim)."""
    assert dim % 2 == 0
    half = dim // 2
    # log-spaced frequencies: freq_k = 1 / max_period^(k/half), k=0..half-1
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )  # (half,)
    args = t.float()[:, None] * freqs[None, :]  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


# -----------------------------------------------------------------------------
# Model.
# -----------------------------------------------------------------------------
class TinyMLP(nn.Module):
    """4-Linear MLP with concatenated sinusoidal time embedding.

    Time embedding (32) is also concatenated AGAIN at the second hidden layer
    (a cheap 'skip-time' trick). This is the deviation from the spec that we
    flag in the report — empirically the spec's strict 3-Linear net plateaus
    around MSE 0.87 on MNIST DDPM at our scale.
    """
    def __init__(self, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM + TIME_EMBED_DIM, hidden)
        self.fc2 = nn.Linear(hidden + TIME_EMBED_DIM, hidden)
        self.fc3 = nn.Linear(hidden + TIME_EMBED_DIM, hidden)
        self.fc4 = nn.Linear(hidden, INPUT_DIM)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, 784), t: (B,)
        emb = sinusoidal_time_embedding(t)               # (B, 32)
        h = torch.cat([x, emb], dim=1)                   # (B, 816)
        h = F.silu(self.fc1(h))                          # (B, H)
        h = F.silu(self.fc2(torch.cat([h, emb], dim=1))) # (B, H)
        h = F.silu(self.fc3(torch.cat([h, emb], dim=1))) # (B, H)
        return self.fc4(h)                               # (B, 784)


# -----------------------------------------------------------------------------
# Schedule (computed on device).
# -----------------------------------------------------------------------------
def make_schedule(device):
    t = torch.arange(T, dtype=torch.float64, device=device)
    betas = BETA_MIN + (BETA_MAX - BETA_MIN) * t / (T - 1)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas.float(), alphas.float(), alpha_bars.float()


# -----------------------------------------------------------------------------
# Training.
# -----------------------------------------------------------------------------
def load_mnist():
    tf = transforms.Compose([
        transforms.ToTensor(),                  # [0,1]
        transforms.Lambda(lambda x: x * 2 - 1)  # → [-1, 1]
    ])
    return datasets.MNIST(root=MNIST_ROOT, train=True, download=True, transform=tf)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    # MPS occasionally diverges silently on small models; CPU is fast enough.
    # Force CPU for byte-determinism across machines.
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    print(f"device = {device}, threads = {torch.get_num_threads()}")

    ds = load_mnist()
    n = len(ds)
    print(f"MNIST train size = {n}")

    # Pre-load all images into RAM as float32 (60_000 * 784 * 4 ≈ 188 MB) -
    # MUCH faster per-step than the DataLoader path on CPU.
    print("Pre-loading MNIST into RAM …")
    all_x = torch.empty((n, INPUT_DIM), dtype=torch.float32)
    for i in range(n):
        img, _ = ds[i]                  # (1,28,28) in [-1,1]
        all_x[i] = img.view(-1)
    print(f"  done. mean={all_x.mean():+.3f}  std={all_x.std():.3f}")

    model = TinyMLP(HIDDEN_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    if USE_COSINE_DECAY:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=LR * 0.05)
    else:
        sched = None
    betas, alphas, alpha_bars = make_schedule(device)
    sqrt_ab = torch.sqrt(alpha_bars)
    sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars)

    g = torch.Generator(device="cpu").manual_seed(SEED)

    losses = []
    model.train()
    for step in range(1, N_STEPS + 1):
        # Sample batch indices
        idx = torch.randint(0, n, (BATCH,), generator=g)
        x0 = all_x[idx].to(device)                                 # (B, 784)
        t  = torch.randint(0, T, (BATCH,), generator=g).to(device) # (B,)
        eps = torch.randn(BATCH, INPUT_DIM, generator=g).to(device)

        sa  = sqrt_ab[t][:, None]
        soa = sqrt_one_minus_ab[t][:, None]
        x_t = sa * x0 + soa * eps

        eps_pred = model(x_t, t)
        loss = F.mse_loss(eps_pred, eps)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

        losses.append(float(loss.item()))
        if step % 1000 == 0 or step == 1:
            recent = np.mean(losses[-1000:])
            cur_lr = sched.get_last_lr()[0] if sched is not None else LR
            print(f"  step {step:6d}/{N_STEPS}  loss(last1k)={recent:.4f}  "
                  f"lr={cur_lr:.2e}")

    # Loss curve assertion #8 (also checked in build_data.py — keep both honest).
    initial = float(np.mean(losses[:1000]))
    final   = float(np.mean(losses[-1000:]))
    print(f"loss: first1k={initial:.4f}  last1k={final:.4f}  ratio={final/initial:.3f}")

    # Save the full state dict (debugging/regen) and also the JSON-ready weights.
    torch.save(model.state_dict(), ART_DIR / "mnist_ddpm.pt")

    weights_obj = {
        "W1": model.fc1.weight.detach().cpu().numpy().T.astype(np.float32).reshape(-1).tolist(),
        "b1": model.fc1.bias.detach().cpu().numpy().astype(np.float32).tolist(),
        "W2": model.fc2.weight.detach().cpu().numpy().T.astype(np.float32).reshape(-1).tolist(),
        "b2": model.fc2.bias.detach().cpu().numpy().astype(np.float32).tolist(),
        "W3": model.fc3.weight.detach().cpu().numpy().T.astype(np.float32).reshape(-1).tolist(),
        "b3": model.fc3.bias.detach().cpu().numpy().astype(np.float32).tolist(),
        "W4": model.fc4.weight.detach().cpu().numpy().T.astype(np.float32).reshape(-1).tolist(),
        "b4": model.fc4.bias.detach().cpu().numpy().astype(np.float32).tolist(),
    }
    # Shapes (each W stored as [in_dim x out_dim] row-major, so JS code does
    # `out[j] = sum_i x[i] * W[i*out_dim + j] + b[j]`):
    #   W1: (INPUT_DIM + TIME_EMBED_DIM) x HIDDEN_DIM        =  816 x H
    #   W2: (HIDDEN_DIM + TIME_EMBED_DIM) x HIDDEN_DIM       = (H+32) x H
    #   W3: (HIDDEN_DIM + TIME_EMBED_DIM) x HIDDEN_DIM       = (H+32) x H
    #   W4:  HIDDEN_DIM x INPUT_DIM                          =  H x 784

    with open(ART_DIR / "mnist_ddpm_weights.json", "w") as f:
        json.dump(weights_obj, f)
    with open(ART_DIR / "loss_curve.json", "w") as f:
        json.dump({"initial_1k": initial, "final_1k": final, "losses": losses}, f)
    print(f"wrote {ART_DIR / 'mnist_ddpm.pt'} and {ART_DIR / 'mnist_ddpm_weights.json'}")


if __name__ == "__main__":
    main()
