# scripts/preprocess_stats.py

import os
import json
import numpy as np
import xarray as xr
from pathlib import Path

DATA_PATH = Path(r"D:\era5_cache\era5_500hpa_2024_09.nc")
assert DATA_PATH.exists(), f"Missing ERA5 file: {DATA_PATH}"

# Try multiple engines for robustness
try:
    ds = xr.open_dataset(str(DATA_PATH), engine="netcdf4")
except Exception:
    try:
        ds = xr.open_dataset(str(DATA_PATH), engine="h5netcdf")
    except Exception:
        ds = xr.open_dataset(str(DATA_PATH), engine="scipy")

# Select 500 hPa level
if "pressure_level" in ds.dims or "pressure_level" in ds.coords:
    ds = ds.sel(pressure_level=500.0) if 500.0 in ds["pressure_level"].values else ds.isel(pressure_level=0)

# Normalize time dimension
if "valid_time" in ds.dims or "valid_time" in ds.coords:
    ds = ds.rename({"valid_time": "time"})

# Canonical variable mapping
VAR_MAP = {
    "t": "temperature", "ta": "temperature",
    "z": "geopotential", "zg": "geopotential",
    "u": "u", "ua": "u",
    "v": "v", "va": "v"
}
required = ["temperature", "geopotential", "u", "v"]
present = set(ds.data_vars)
mapped = {raw: canon for raw, canon in VAR_MAP.items() if raw in present and canon not in mapped.values()}
assert set(mapped.values()) == set(required), f"Missing variables: {required}"

# Dimensions
T, H, W = ds.sizes["time"], ds.sizes["latitude"], ds.sizes["longitude"]

# Streaming accumulators
acc_sum, acc_sum2, acc_count = {c: 0.0 for c in required}, {c: 0.0 for c in required}, {c: 0 for c in required}
chunk_t = 16

for t0 in range(0, T, chunk_t):
    t1 = min(T, t0 + chunk_t)
    for canon in required:
        raw = [k for k, v in mapped.items() if v == canon][0]
        arr = ds[raw].isel(time=slice(t0, t1)).values.astype(np.float64)
        acc_sum[canon] += arr.sum()
        acc_sum2[canon] += (arr * arr).sum()
        acc_count[canon] += arr.size
    print(f"Processed time slice [{t0}:{t1})")

# Final stats
stats = {}
for canon in required:
    n = acc_count[canon]
    mean = acc_sum[canon] / max(n, 1)
    var = max(acc_sum2[canon] / max(n, 1) - mean * mean, 0.0)
    std = float(np.sqrt(var) + 1e-6)
    stats[canon] = (float(mean), std)

# Save to JSON
out_file = Path(r"D:\era5_cache\stats_500hpa_2024_09.json")
with open(out_file, "w") as f:
    json.dump({"VARS": required, "stats": stats}, f, indent=2)
print("Saved stats to:", out_file)