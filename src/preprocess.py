

from pathlib import Path
import numpy as np
import rasterio


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = PROCESSED_DIR / "reflectance"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. SATELLITE CONFIGURATION
# ==========================================

SATELLITES = {
    "sentinel2": {
        "scale": 0.0001,
        "offset": 0.0
    },

    "landsat": {
        "scale": 0.0000275,
        "offset": -0.2
    }
}

BANDS = ["blue", "green", "red", "nir"]

NODATA_VALUE = -9999.0


# ==========================================
# 3. CONVERT DIGITAL NUMBERS TO REFLECTANCE
# ==========================================

def convert_band(satellite, band):

    input_path = RAW_DIR / satellite / f"{band}.tif"

    output_folder = OUTPUT_DIR / satellite
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{band}.tif"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_path}"
        )

    scale = SATELLITES[satellite]["scale"]
    offset = SATELLITES[satellite]["offset"]

    print(f"\nProcessing {satellite} - {band}")

    with rasterio.open(input_path) as src:

        # Read original pixel values
        original = src.read(1)

        # Identify invalid pixels
        valid_mask = original != 0

        # Convert DN into surface reflectance
        reflectance = np.full(
            original.shape,
            NODATA_VALUE,
            dtype=np.float32
        )

        reflectance[valid_mask] = (
            original[valid_mask].astype(np.float32) * scale
            + offset
        )

        # Prepare output metadata
        profile = src.profile.copy()

        profile.update({
            "dtype": "float32",
            "nodata": NODATA_VALUE,
            "compress": "deflate"
        })

        # Save reflectance image
        with rasterio.open(output_path, "w", **profile) as dst:

            dst.write(reflectance, 1)

            dst.update_tags(
                satellite=satellite,
                band=band,
                scale_factor=scale,
                offset=offset,
                units="surface_reflectance"
            )

    # Display statistics
    valid_values = reflectance[valid_mask]

    print("Saved:", output_path)
    print("Dimensions:", reflectance.shape)
    print("Minimum reflectance:", float(valid_values.min()))
    print("Maximum reflectance:", float(valid_values.max()))
    print("Valid pixels:", int(valid_mask.sum()))
    print("Invalid pixels:", int((~valid_mask).sum()))


# ==========================================
# 4. PROCESS ALL BANDS
# ==========================================

def preprocess_all():

    print("\n========== REFLECTANCE PREPROCESSING ==========")

    for satellite in SATELLITES:

        print("\n" + "=" * 45)
        print("SATELLITE:", satellite.upper())
        print("=" * 45)

        for band in BANDS:

            convert_band(satellite, band)

    print("\nALL BANDS CONVERTED SUCCESSFULLY!")


# ==========================================
# 5. VERIFY OUTPUT FILES
# ==========================================

def verify_outputs():

    print("\n========== OUTPUT VERIFICATION ==========")

    for satellite in SATELLITES:

        folder = OUTPUT_DIR / satellite

        for band in BANDS:

            file_path = folder / f"{band}.tif"

            with rasterio.open(file_path) as src:

                data = src.read(1, masked=True)

                print(
                    f"{satellite}/{band}.tif | "
                    f"Shape: {src.shape} | "
                    f"Resolution: {src.res} | "
                    f"Data type: {src.dtypes[0]} | "
                    f"CRS: {src.crs}"
                )


# ==========================================
# 6. MAIN
# ==========================================

if __name__ == "__main__":

    preprocess_all()

    verify_outputs()
