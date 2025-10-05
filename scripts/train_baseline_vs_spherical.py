# scripts/train_baseline_vs_spherical.py

import torch
from torch.utils.data import DataLoader
from models.equivariant_point_block import EquivariantPointBlock
from scripts.cache_builder import CACHE_DIR
import numpy as np
import json
from pathlib import Path

# Load stats
stats_path = Path(r"D:\era5_cache\stats_500hpa_2024_09.json")
with open(stats_path, "r") as f:
    stats_data = json.load(f)
VARS = stats_data["VARS"]
stats = {k: tuple(v) for k, v in stats_data["stats"].items()}

# Load baseline cache
Xb = np.load(CACHE_DIR / "baseline_ctx_src_64x128.npy")
Yb = np.load(CACHE_DIR / "baseline_tgt_64x128.npy")

# Load spherical cache
Xs = np.load(CACHE_DIR / "spherical_ctx_src_TCN.npy")
Ys = np.load(CACHE_DIR / "spherical_tgt_CN.npy")

# Convert to torch tensors
Xb = torch.from_numpy(Xb)
Yb = torch.from_numpy(Yb)
Xs = torch.from_numpy(Xs)
Ys = torch.from_numpy(Ys)

# Dummy model setup (replace with actual)
class DummyBaseline(torch.nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(C, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, C, 3, padding=1)
        )
    def forward(self, x):  # x: [B, C, H, W]
        return self.net(x)

class DummySpherical(torch.nn.Module):
    def __init__(self, C, N):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(C, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, C)
        )
    def forward(self, x):  # x: [B, C, N]
        return self.net(x.transpose(1, 2)).transpose(1, 2)

# Instantiate models
C = len(VARS)
N = Xs.shape[-1]
baseline_model = DummyBaseline(C)
spherical_model = DummySpherical(C, N)

# Example forward pass
xb_out = baseline_model(Xb[0])  # [C, H, W]
xs_out = spherical_model(Xs[0])  # [C, N]

print("Baseline output shape:", xb_out.shape)
print("Spherical output shape:", xs_out.shape)