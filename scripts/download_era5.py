# scripts/download_era5.py

import cdsapi
import os
import shutil
from pathlib import Path

# Step 1: Download to temp location
tmp_dir = Path(os.environ.get("TEMP", r"C:\Users\hars\AppData\Local\Temp"))
tmp_nc = tmp_dir / "era5_500hpa_2024_09.tmp"

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"
        ],
        "pressure_level": ["500"],
        "year": "2024",
        "month": "09",
        "day": [f"{d:02d}" for d in range(1, 31 + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [90, -180, -90, 180],
    },
    str(tmp_nc)
)

# Step 2: Atomic copy to D:\era5_cache
dst_dir = Path(r"D:\era5_cache")
dst_dir.mkdir(parents=True, exist_ok=True)
dst_final = dst_dir / "era5_500hpa_2024_09.nc"
shutil.copy2(tmp_nc, dst_final)

print("Downloaded and copied to:", dst_final)
print("Size (MB):", round(dst_final.stat().st_size / (1024 * 1024), 2))