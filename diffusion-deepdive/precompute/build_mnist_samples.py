"""Pick one MNIST training-set image per class 0..9 and normalise to [-1, 1].

Deterministic: we walk the training set in index order and take the first
occurrence of each label. The picked indices are pinned at the bottom of the
returned dict so a future maintainer can verify reproducibility.
"""
import os
import numpy as np
import torch
from torchvision import datasets, transforms

SEED = 42
MNIST_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mnist_cache")


def get_mnist_train():
    """Load MNIST training set as a tensor dataset (no augmentation)."""
    tf = transforms.ToTensor()  # outputs [0, 1] floats, shape (1, 28, 28)
    ds = datasets.MNIST(root=MNIST_ROOT, train=True, download=True, transform=tf)
    return ds


def build_mnist_samples():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds = get_mnist_train()
    picked = {}
    picked_idx = {}
    for i in range(len(ds)):
        img, label = ds[i]
        label = int(label)
        if label in picked:
            continue
        # img: torch tensor (1,28,28) in [0,1]. Convert to [-1,1] flat 784.
        arr = img.numpy().reshape(-1).astype(np.float64)  # 784
        arr = arr * 2.0 - 1.0
        picked[label] = arr
        picked_idx[label] = i
        if len(picked) == 10:
            break

    samples = []
    for label in range(10):
        samples.append({
            "label": label,
            "pixels": picked[label].tolist(),
            "_index": picked_idx[label],
        })
    return samples


if __name__ == "__main__":
    samples = build_mnist_samples()
    for s in samples:
        arr = np.array(s["pixels"])
        print(f"label {s['label']}  idx={s['_index']:5d}  "
              f"pixels in [{arr.min():.2f},{arr.max():.2f}]  "
              f"mean={arr.mean():+.3f} std={arr.std():.3f}")
