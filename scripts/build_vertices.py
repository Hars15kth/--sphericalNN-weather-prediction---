# scripts/build_vertices.py

import numpy as np
import torch

def build_spherical_vertices(H=64, W=128):
    lats = np.linspace(-90.0, 90.0, H, endpoint=True)
    lons = np.linspace(0.0, 360.0, W, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_grid)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    latlon = np.stack([lat_grid, lon_grid], axis=-1).reshape(-1, 2).astype(np.float32)
    idx_flat = np.arange(H * W, dtype=np.int64)

    return {"xyz": xyz, "latlon": latlon, "idx_flat": idx_flat, "H": H, "W": W}

def knn_on_sphere_chunked(xyz_np: np.ndarray, K: int = 16, block: int = 2048):
    xyz = torch.from_numpy(xyz_np)  # [N, 3]
    N = xyz.shape[0]
    K = min(K, max(1, N - 1))
    knn_idx = torch.empty(N, K, dtype=torch.long)

    with torch.no_grad():
        for i0 in range(0, N, block):
            i1 = min(N, i0 + block)
            d = torch.cdist(xyz[i0:i1], xyz, p=2)  # [B, N]
            d[torch.arange(i1 - i0), torch.arange(i0, i1)] = 1e9  # mask self
            topk = torch.topk(-d, k=K, dim=1).indices
            knn_idx[i0:i1] = topk.cpu()

    return knn_idx.numpy().astype(np.int64)

# Example usage
verts = build_spherical_vertices(H=64, W=128)
verts["knn_idx"] = knn_on_sphere_chunked(verts["xyz"], K=16, block=1024)
print("Built spherical vertices:", {"H": verts["H"], "W": verts["W"], "N": verts["H"] * verts["W"]})
print("KNN shape:", verts["knn_idx"].shape)