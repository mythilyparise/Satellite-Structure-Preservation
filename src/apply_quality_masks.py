
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation


ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "data" / "processed" / "reflectance"
MASK_DIR = ROOT / "data" / "raw"

OUTPUT_DIR = ROOT / "data" / "processed" / "reflectance_masked"
RESULT_DIR = ROOT / "results" / "quality_mask"

BANDS = ["blue", "green", "red", "nir"]


# ==========================================
# ALIGN MASK AND ITS COVERAGE TO IMAGE GRID
# ==========================================

def align_mask_and_coverage(mask_path, reference_path):

    with rasterio.open(mask_path) as src:

        source = src.read(1)

        source_transform = src.transform
        source_crs = src.crs

        source_coverage = np.ones(
            source.shape,
            dtype=np.uint8
        )

    with rasterio.open(reference_path) as ref:

        destination = np.zeros(
            (ref.height, ref.width),
            dtype=source.dtype
        )

        coverage = np.zeros(
            (ref.height, ref.width),
            dtype=np.uint8
        )

        reproject(
            source=source,
            destination=destination,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=Resampling.nearest
        )

        reproject(
            source=source_coverage,
            destination=coverage,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=Resampling.nearest
        )

    return destination, coverage.astype(bool)


# ==========================================
# SENTINEL-2 CLOUD MASK
# ==========================================

def create_sentinel_mask():

    print("\n========== SENTINEL-2 MASK ==========")

    reference = INPUT_DIR / "sentinel2" / "blue.tif"
    scl_path = MASK_DIR / "sentinel2" / "scl.tif"

    scl, coverage = align_mask_and_coverage(
        scl_path,
        reference
    )

    # SCL classes excluded:
    # 1 = Saturated / defective
    # 3 = Cloud shadow
    # 8 = Medium probability cloud
    # 9 = High probability cloud
    # 10 = Thin cirrus
    # 11 = Snow / ice

    excluded_classes = [1, 3, 8, 9, 10, 11]

    print("Aligned SCL shape:", scl.shape)
    print("SCL coverage pixels:", np.count_nonzero(coverage))
    print("SCL uncovered pixels:", np.count_nonzero(~coverage))

    values, counts = np.unique(scl[coverage], return_counts=True)

    print("\nAligned SCL distribution:")

    for value, count in zip(values, counts):
        print(f"Class {value}: {count}")

    cloud_mask = (
        coverage
        & np.isin(scl, excluded_classes)
    )

    original_cloud_pixels = np.count_nonzero(cloud_mask)

    # Expand actual cloud detections only.
    if original_cloud_pixels > 0:

        cloud_mask = binary_dilation(
            cloud_mask,
            iterations=3
        )

        # Never expand beyond the classified footprint.
        cloud_mask &= coverage

    print("\nCloud/defective pixels before dilation:",
          original_cloud_pixels)

    print("Cloud/defective pixels after dilation:",
          np.count_nonzero(cloud_mask))

    return cloud_mask, coverage


# ==========================================
# LANDSAT CLOUD MASK
# ==========================================

def create_landsat_mask():

    print("\n========== LANDSAT MASK ==========")

    reference = INPUT_DIR / "landsat" / "blue.tif"
    qa_path = MASK_DIR / "landsat" / "qa_pixel.tif"

    qa, coverage = align_mask_and_coverage(
        qa_path,
        reference
    )

    # QA_PIXEL:
    # Bit 0 = Fill
    # Bit 1 = Dilated cloud
    # Bit 2 = Cirrus
    # Bit 3 = Cloud
    # Bit 4 = Cloud shadow
    # Bit 5 = Snow

    excluded_bits = (
        (1 << 0)
        | (1 << 1)
        | (1 << 2)
        | (1 << 3)
        | (1 << 4)
        | (1 << 5)
    )

    cloud_mask = (
        coverage
        & ((qa.astype(np.uint16) & excluded_bits) != 0)
    )

    original_cloud_pixels = np.count_nonzero(cloud_mask)

    if original_cloud_pixels > 0:

        cloud_mask = binary_dilation(
            cloud_mask,
            iterations=1
        )

        cloud_mask &= coverage

    print("Aligned QA shape:", qa.shape)
    print("QA coverage pixels:", np.count_nonzero(coverage))
    print("Cloud/invalid pixels before dilation:",
          original_cloud_pixels)

    print("Cloud/invalid pixels after dilation:",
          np.count_nonzero(cloud_mask))

    return cloud_mask, coverage


# ==========================================
# APPLY MASK TO ALL FOUR BANDS
# ==========================================

def apply_mask(satellite, cloud_mask, coverage):

    print(f"\n========== {satellite.upper()} ==========")

    input_folder = INPUT_DIR / satellite
    output_folder = OUTPUT_DIR / satellite

    output_folder.mkdir(parents=True, exist_ok=True)

    reference_path = input_folder / "blue.tif"

    with rasterio.open(reference_path) as ref:

        reference_data = ref.read(1)

        reference_valid = np.isfinite(reference_data)

        if ref.nodata is not None:
            reference_valid &= reference_data != ref.nodata

    total_valid = np.count_nonzero(reference_valid)

    # Count only actual detected mask pixels.
    masked_pixels = np.count_nonzero(
        cloud_mask & reference_valid
    )

    percentage = (
        masked_pixels / total_valid * 100
        if total_valid else 0
    )

    coverage_gaps = np.count_nonzero(
        reference_valid & ~coverage
    )

    print("Valid reference pixels:", total_valid)
    print("Masked pixels:", masked_pixels)
    print(f"Masked percentage: {percentage:.4f}%")
    print("Valid image pixels without QA coverage:", coverage_gaps)

    for band in BANDS:

        input_path = input_folder / f"{band}.tif"
        output_path = output_folder / f"{band}.tif"

        with rasterio.open(input_path) as src:

            data = src.read(1).astype(np.float32)

            nodata = src.nodata

            if nodata is None:
                nodata = -9999.0

            invalid = ~np.isfinite(data)

            if src.nodata is not None:
                invalid |= data == src.nodata

            # Only actual detected quality issues are masked.
            final_invalid = invalid | cloud_mask

            data[final_invalid] = nodata

            profile = src.profile.copy()

            profile.update({
                "dtype": "float32",
                "nodata": nodata,
                "compress": "deflate"
            })

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data, 1)

        print("Saved:", output_path.name)

    return {
        "Satellite": satellite,
        "Valid Pixels": int(total_valid),
        "Masked Pixels": int(masked_pixels),
        "Masked Percentage": percentage,
        "QA Coverage Gaps": int(coverage_gaps)
    }


# ==========================================
# MAIN
# ==========================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("CORRECTED SATELLITE QUALITY MASKING")
    print("========================================")

    sentinel_mask, sentinel_coverage = create_sentinel_mask()

    landsat_mask, landsat_coverage = create_landsat_mask()

    results = []

    results.append(
        apply_mask(
            "sentinel2",
            sentinel_mask,
            sentinel_coverage
        )
    )

    results.append(
        apply_mask(
            "landsat",
            landsat_mask,
            landsat_coverage
        )
    )

    summary = pd.DataFrame(results)

    output_path = RESULT_DIR / "quality_mask_summary.csv"

    summary.to_csv(output_path, index=False)

    print("\n========== CORRECTED SUMMARY ==========")
    print(summary.to_string(index=False))

    print("\nSaved:", output_path)
    print("\nMASKING COMPLETE!")


if __name__ == "__main__":
    main()
