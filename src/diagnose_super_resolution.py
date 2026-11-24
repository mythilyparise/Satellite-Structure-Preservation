
import rasterio
import numpy as np
from pathlib import Path

folders = {
    "NATIVE REFERENCE": Path("data/processed/experiment/reference_10m"),
    "BICUBIC": Path("data/processed/super_resolution/bicubic_10m"),
    "SRCNN": Path("data/processed/super_resolution/srcnn_10m"),
}

bands = ["blue", "green", "red", "nir"]

for name, folder in folders.items():
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    for band in bands:
        path = folder / f"{band}.tif"

        if not path.exists():
            print(f"{band.upper()}: FILE NOT FOUND - {path}")
            continue

        with rasterio.open(path) as src:
            image = src.read(1).astype(np.float64)
            nodata = src.nodata

            valid = np.isfinite(image)

            if nodata is not None:
                valid &= image != nodata

            values = image[valid]

            if values.size == 0:
                print(f"{band.upper()}: No valid pixels")
                continue

            print(
                f"{band.upper():5} | "
                f"Mean: {values.mean():.5f} | "
                f"Std: {values.std():.5f} | "
                f"Min: {values.min():.5f} | "
                f"Max: {values.max():.5f}"
            )

print("\nDiagnostic completed.")
