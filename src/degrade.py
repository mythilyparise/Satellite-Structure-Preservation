
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "reflectance" / "sentinel2"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "experiment"

REFERENCE_DIR = OUTPUT_DIR / "reference_10m"
LOW_RES_DIR = OUTPUT_DIR / "synthetic_30m"
RECON_DIR = OUTPUT_DIR / "reconstructed_10m"

BANDS = ["blue", "green", "red", "nir"]

NODATA = -9999.0
SCALE_FACTOR = 3


# --------------------------------------------------
# CREATE OUTPUT FOLDERS
# --------------------------------------------------

for folder in [REFERENCE_DIR, LOW_RES_DIR, RECON_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# PROCESS ONE BAND
# --------------------------------------------------

def process_band(band):

    print(f"\nProcessing band: {band.upper()}")

    input_path = INPUT_DIR / f"{band}.tif"

    if not input_path.exists():
        raise FileNotFoundError(
            f"\nInput file not found: {input_path}\n"
            "Check that the reflectance folder contains "
            "blue.tif, green.tif, red.tif and nir.tif."
        )

    # Read original Sentinel-2 image
    with rasterio.open(input_path) as src:

        image = src.read(1).astype(np.float32)

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

        print("Original image shape:", image.shape)
        print("Resolution:", src.res)

    # --------------------------------------------------
    # STEP 1: SAVE ORIGINAL 10m REFERENCE
    # --------------------------------------------------

    reference_path = REFERENCE_DIR / f"{band}.tif"

    profile.update(
        dtype="float32",
        nodata=NODATA,
        compress="deflate"
    )

    with rasterio.open(reference_path, "w", **profile) as dst:
        dst.write(image, 1)

    # --------------------------------------------------
    # STEP 2: CROP TO MULTIPLES OF 3
    # --------------------------------------------------

    height, width = image.shape

    new_height = (height // SCALE_FACTOR) * SCALE_FACTOR
    new_width = (width // SCALE_FACTOR) * SCALE_FACTOR

    image = image[:new_height, :new_width]

    low_height = new_height // SCALE_FACTOR
    low_width = new_width // SCALE_FACTOR

    print("Cropped image shape:", image.shape)

    # --------------------------------------------------
    # STEP 3: DEGRADE 10m IMAGE TO 30m
    # Using 3x3 block averaging
    # --------------------------------------------------

    blocks = image.reshape(
        low_height,
        SCALE_FACTOR,
        low_width,
        SCALE_FACTOR
    )

    valid = np.isfinite(blocks) & (blocks != NODATA)

    # Only use blocks where all 9 pixels are valid
    valid_blocks = np.all(valid, axis=(1, 3))

    sums = np.where(valid, blocks, 0).sum(axis=(1, 3))

    low_image = np.full(
        (low_height, low_width),
        NODATA,
        dtype=np.float32
    )

    low_image[valid_blocks] = (
        sums[valid_blocks] / (SCALE_FACTOR ** 2)
    )

    # 30m pixel size
    low_transform = transform * Affine.scale(
        SCALE_FACTOR,
        SCALE_FACTOR
    )

    low_profile = profile.copy()

    low_profile.update(
        height=low_height,
        width=low_width,
        transform=low_transform,
        dtype="float32",
        nodata=NODATA
    )

    low_path = LOW_RES_DIR / f"{band}.tif"

    with rasterio.open(low_path, "w", **low_profile) as dst:
        dst.write(low_image, 1)

    # --------------------------------------------------
    # STEP 4: RECONSTRUCT 30m IMAGE BACK TO 10m
    # Using bilinear interpolation
    # --------------------------------------------------

    reconstructed = np.full(
        image.shape,
        NODATA,
        dtype=np.float32
    )

    reproject(
        source=low_image,
        destination=reconstructed,
        src_transform=low_transform,
        src_crs=crs,
        src_nodata=NODATA,
        dst_transform=transform,
        dst_crs=crs,
        dst_nodata=NODATA,
        resampling=Resampling.bilinear
    )

    recon_profile = profile.copy()

    recon_profile.update(
        height=new_height,
        width=new_width,
        transform=transform,
        dtype="float32",
        nodata=NODATA
    )

    recon_path = RECON_DIR / f"{band}.tif"

    with rasterio.open(recon_path, "w", **recon_profile) as dst:
        dst.write(reconstructed, 1)

    # --------------------------------------------------
    # STEP 5: PRINT RESULTS
    # --------------------------------------------------

    valid_low = low_image[low_image != NODATA]
    valid_recon = reconstructed[reconstructed != NODATA]

    print("10m reference:", image.shape)
    print("30m degraded:", low_image.shape)
    print("10m reconstructed:", reconstructed.shape)

    print("Valid 30m pixels:", len(valid_low))
    print("Valid reconstructed pixels:", len(valid_recon))

    print("Saved all three images successfully!")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("========== SYNTHETIC DEGRADATION ==========")

    print("\nReading input from:")
    print(INPUT_DIR)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"\nReflectance folder does not exist: {INPUT_DIR}"
        )

    for band in BANDS:
        process_band(band)

    print("\n========== EXPERIMENT COMPLETED ==========")
    print("Reference images:", REFERENCE_DIR)
    print("30m images:", LOW_RES_DIR)
    print("Reconstructed images:", RECON_DIR)


if __name__ == "__main__":
    main()
