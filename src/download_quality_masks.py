
from pathlib import Path

import planetary_computer
import rasterio

from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds, Window


# ==========================================
# 1. CONFIGURATION
# ==========================================

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

BBOX = [76.755, 30.710, 76.805, 30.755]

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

SCENES = {
    "sentinel2": {
        "collection": "sentinel-2-l2a",
        "scene_id": "S2A_MSIL2A_20251106T053241_R105_T43RFP_20251106T083512",
        "asset": "SCL",
        "output": "scl.tif"
    },

    "landsat": {
        "collection": "landsat-c2-l2",
        "scene_id": "LC09_L2SP_147039_20251105_02_T1",
        "asset": "qa_pixel",
        "output": "qa_pixel.tif"
    }
}


# ==========================================
# 2. CONNECT TO CATALOG
# ==========================================

def connect_catalog():

    print("Connecting to Planetary Computer...")

    catalog = Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace
    )

    print("Connected successfully!")

    return catalog


# ==========================================
# 3. DOWNLOAD QUALITY MASK
# ==========================================

def download_quality_mask(catalog, name, details):

    print("\n========================================")
    print("PROCESSING:", name.upper())
    print("========================================")

    output_dir = RAW_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / details["output"]

    if output_path.exists():
        print("Already exists:", output_path)
        print("Skipping download.")
        return

    search = catalog.search(
        collections=[details["collection"]],
        ids=[details["scene_id"]]
    )

    items = list(search.items())

    if not items:
        raise RuntimeError(
            f"Scene not found: {details['scene_id']}"
        )

    item = items[0]

    print("Scene:", item.id)
    print("Available assets:", list(item.assets.keys()))

    asset_name = details["asset"]

    if asset_name not in item.assets:
        raise KeyError(
            f"Asset {asset_name} not found. "
            f"Available assets: {list(item.assets.keys())}"
        )

    asset = item.assets[asset_name]

    print("Quality asset:", asset_name)

    with rasterio.open(asset.href) as src:

        bounds = transform_bounds(
            "EPSG:4326",
            src.crs,
            *BBOX
        )

        window = from_bounds(
            *bounds,
            transform=src.transform
        )

        window = window.round_offsets().round_lengths()

        full_window = Window(
            0, 0, src.width, src.height
        )

        window = window.intersection(full_window)

        if window.width <= 0 or window.height <= 0:
            raise ValueError(
                f"No overlap for {name} quality mask."
            )

        data = src.read(1, window=window)

        profile = src.profile.copy()

        profile.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": src.window_transform(window),
            "compress": "deflate"
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

    print("Saved:", output_path)
    print("Shape:", data.shape)
    print("Data type:", data.dtype)


# ==========================================
# 4. VERIFY MASK FILES
# ==========================================

def verify_masks():

    print("\n========== QUALITY MASK VERIFICATION ==========")

    files = [
        RAW_DIR / "sentinel2" / "scl.tif",
        RAW_DIR / "landsat" / "qa_pixel.tif"
    ]

    for file in files:

        print("\n", file.name)

        if not file.exists():
            print("NOT FOUND")
            continue

        with rasterio.open(file) as src:

            print("Shape:", src.height, "x", src.width)
            print("CRS:", src.crs)
            print("Resolution:", src.res)
            print("Data type:", src.dtypes[0])
            print("NoData:", src.nodata)


# ==========================================
# 5. MAIN
# ==========================================

def main():

    catalog = connect_catalog()

    for name, details in SCENES.items():

        download_quality_mask(
            catalog,
            name,
            details
        )

    verify_masks()

    print("\nQUALITY MASK DOWNLOAD COMPLETE!")


if __name__ == "__main__":
    main()
