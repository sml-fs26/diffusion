"""Linear-beta DDPM schedule for the diffusion-deepdive viz.

Pinned everywhere: T = 200, beta_min = 1e-4, beta_max = 0.02 (DDPM defaults,
scaled to T=200 instead of 1000 so live in-browser reverse is fast).
"""
import numpy as np

SEED = 42
T = 200
BETA_MIN = 1e-4
BETA_MAX = 0.02


def build_schedule(T_steps: int = T, beta_min: float = BETA_MIN, beta_max: float = BETA_MAX):
    rng = np.random.default_rng(SEED)  # not used, kept for symmetry/seed audit
    _ = rng.random()  # noqa: F841 — exercise the RNG so the seed is "consumed" deterministically
    t = np.arange(T_steps, dtype=np.float64)
    betas = beta_min + (beta_max - beta_min) * t / (T_steps - 1)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return {
        "T": T_steps,
        "betas": betas.tolist(),
        "alphas": alphas.tolist(),
        "alphaBars": alpha_bars.tolist(),
    }


if __name__ == "__main__":
    sched = build_schedule()
    print(f"T = {sched['T']}")
    print(f"betas[0]   = {sched['betas'][0]:.6e}")
    print(f"betas[-1]  = {sched['betas'][-1]:.6e}")
    print(f"alphaBars[0]   = {sched['alphaBars'][0]:.6e}")
    print(f"alphaBars[-1]  = {sched['alphaBars'][-1]:.6e}")
