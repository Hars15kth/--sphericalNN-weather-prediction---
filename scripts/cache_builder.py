# scripts/cache_builder.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from pathlib import Path
from build_vertices import build_spherical_vertices

DATA_PATH = Path(r"D:\era5_cache\era5_500hpa_2024_09.nc")
STATS_PATH = Path(r"D:\era5_cache\stats_500hpa_2024_09.json")
CACHE_DIR = Path(r"D:\era5_cache\runtime")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CACHE = CACHE_DIR / "baseline_ctx_src_64x128.npy"
TARGET_CACHE = CACHE_DIR / "baseline_tgt_64x128.npy"
SPHERICAL_CACHE = CACHE_DIR / "spherical_ctx_src_TCN.npy"
SPHERICAL_TGT = CACHE_DIR / "spherical_tgt_CN.npy"

USE_TORCH_RESIZE = False
CHUNK = 48
torch.set_num_threads(min(8, max(1, os.cpu_count() or 4)))

ds = xr.open_dataset(str(DATA_PATH), engine="netcdf4")
if "pressure_level" in ds.dims or "pressure_level" in ds.coords:
    ds = ds.sel(pressure_level=500.0) if 500.0 in ds["pressure_level"].values else ds.isel(pressure_level=0)

lats = ds["latitude"].values
if lats[0] > lats[-1]:
    ds = ds.isel(latitude=slice(None, None, -1))

VARS = ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"]
C = len(VARS)
T = ds.sizes["time"]

# Baseline cache
if not (BASELINE_CACHE.exists() and TARGET_CACHE.exists()):
    Xb = np.empty((T, C, 64, 128), dtype=np.float32)
    Yb = np.empty((T, C, 64, 128), dtype=np.float32)
    for t0 in range(0, T, CHUNK):
        t1 = min(t0 + CHUNK, T)
        chunk = [ds[v].isel(time=slice(t0, t1)).values[:, None, ...].astype(np.float32) for v in VARS]
        block = np.concatenate(chunk, axis=1)
        if USE_TORCH_RESIZE:
            ten = torch.from_numpy(block)
            ten_rs = F.interpolate(ten, size=(64, 128), mode="bilinear", align_corners=False).numpy()
        else:
            H0, W0 = block.shape[2], block.shape[3]
            step_h, step_w = max(1, H0 // 64), max(1, W0 // 128)
            ten_rs = block[:, :, ::step_h, ::step_w][:, :, :64, :128]
            pad_h, pad_w = 64 - ten_rs.shape[2], 128 - ten_rs.shape[3]
            if pad_h > 0 or pad_w > 0:
                ten_rs = np.pad(ten_rs, ((0,0),(0,0),(0,pad_h),(0,pad_w)), mode="constant")
        Xb[t0:t1] = ten_rs
        Yb[t0:t1] = ten_rs
    np.save(BASELINE_CACHE, Xb)
    np.save(TARGET_CACHE, Yb)
    print("Baseline caches saved.")
else:
    print("Baseline caches already exist.")

# Spherical cache
if not (SPHERICAL_CACHE.exists() and SPHERICAL_TGT.exists()):
    verts = build_spherical_vertices(H=64, W=128)
    lons = ds["longitude"].values
    lat_idx = np.searchsorted(ds["latitude"].values, verts["latlon"][:, 0], side="left")
    lon_targets = ((verts["latlon"][:, 1] - lons[0] + 360) % 360) + lons[0]
    lon_idx = np.searchsorted(lons, lon_targets, side="left")
    lat_idx = np.clip(lat_idx, 0, len(ds["latitude"]) - 1)
    lon_idx = np.clip(lon_idx, 0, len(lons) - 1)
    N = lat_idx.shape[0]

    Xs = np.empty((T, C, N), dtype=np.float32)
    Ys = np.empty((T, C, N), dtype=np.float32)
    for t0 in range(0, T, CHUNK):
        t1 = min(t0 + CHUNK, T)
        chunk = [ds[v].isel(time=slice(t0, t1)).values[:, None, ...].astype(np.float32) for v in VARS]
        block = np.concatenate(chunk, axis=1)
        gather = block[:, :, lat_idx, lon_idx]
        Xs[t0:t1] = gather
        Ys[t0:t1] = gather
    np.save(SPHERICAL_CACHE, Xs)
    np.save(SPHERICAL_TGT, Ys)
    print("Spherical caches saved.")
else:
    print("Spherical caches already exist.")