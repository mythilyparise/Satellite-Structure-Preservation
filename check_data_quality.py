
from pathlib import Path

import numpy as np
import rasterio

folders = [
    "data/processed/reflectance/sentinel2",
    "data/processed/reflectance/landsat",
    "data/processed/experiment/reference_10m",
    "data/processed/experiment/synthetic_30m",
    "data/processed/experiment/reconstructed_10m",
]

for folder in folders:
    path = Path(folder)

    print("\n" + "=" * 60)
    print(f"FOLDER: {folder}")
    print("=" * 60)

    files = sorted(path.glob("*.tif"))

    if not files:
        print("No TIFF files found!")
        continue

    for file in files:
        with rasterio.open(file) as src:
            data = src.read(1, masked=True).filled(np.nan)

        valid = data[np.isfinite(data)]

        if valid.size == 0:
            print(f"\n{file.name}: NO VALID PIXELS")
            continue

        print(f"\nFile: {file.name}")
        print(f"Shape: {data.shape}")
        print(f"Valid pixels: {len(valid)}")
        print(f"Valid percentage: {len(valid) / data.size * 100:.2f}%")
        print(f"Minimum: {valid.min():.4f}")
        print(f"Maximum: {valid.max():.4f}")
