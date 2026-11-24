
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. SELECTED SCENES AND BAND MAPPING
# ==========================================

SELECTED_SCENES = {
    "Sentinel-2": {
        "collection": "sentinel-2-l2a",
        "scene_id": "S2A_MSIL2A_20251106T053241_R105_T43RFP_20251106T083512",
        "bands": {
            "blue": "B02",
            "green": "B03",
            "red": "B04",
            "nir": "B08"
        }
    },

    "Landsat": {
        "collection": "landsat-c2-l2",
        "scene_id": "LC09_L2SP_147039_20251105_02_T1",
        "bands": {
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir": "nir08"
        }
    }
}


# ==========================================
# 3. CONNECT TO PLANETARY COMPUTER
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
# 4. GET SELECTED SCENE
# ==========================================

def get_scene(catalog, satellite, details):

    print(f"\nSearching for {satellite} scene...")

    search = catalog.search(
        collections=[details["collection"]],
        ids=[details["scene_id"]]
    )

    items = list(search.items())

    if not items:
        raise RuntimeError("Scene not found!")

    item = items[0]

    print("Found:", item.id)

    return item


# ==========================================
# 5. DOWNLOAD AND CLIP BAND
# ==========================================

def download_band(asset, output_path):

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
            raise ValueError("Study area does not overlap image!")

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

    print("Downloaded:", output_path.name)


# ==========================================
# 6. DOWNLOAD SATELLITE BANDS
# ==========================================

def download_satellite(catalog, satellite, details):

    print("\n================================")
    print("DOWNLOADING:", satellite)
    print("================================")

    item = get_scene(catalog, satellite, details)

    folder_name = satellite.lower().replace("-", "")

    satellite_dir = RAW_DIR / folder_name
    satellite_dir.mkdir(parents=True, exist_ok=True)

    for band_name, asset_name in details["bands"].items():

        if asset_name not in item.assets:

            raise KeyError(
                f"{asset_name} not found. "
                f"Available: {list(item.assets.keys())}"
            )

        output_path = satellite_dir / f"{band_name}.tif"

        print(f"\nDownloading {band_name} ({asset_name})...")

        download_band(
            item.assets[asset_name],
            output_path
        )

    print(f"\n{satellite} download completed!")


# ==========================================
# 7. VERIFY DOWNLOADED IMAGES
# ==========================================

def verify_downloads():

    print("\n========== DOWNLOAD VERIFICATION ==========")

    for satellite in ["sentinel2", "landsat"]:

        folder = RAW_DIR / satellite

        print(f"\n{satellite.upper()}")

        for file in sorted(folder.glob("*.tif")):

            with rasterio.open(file) as src:

                print(
                    f"{file.name} | "
                    f"Shape: {src.height} x {src.width} | "
                    f"CRS: {src.crs} | "
                    f"Resolution: {src.res}"
                )


# ==========================================
# 8. MAIN
# ==========================================

def main():

    catalog = connect_catalog()

    for satellite, details in SELECTED_SCENES.items():

        download_satellite(
            catalog,
            satellite,
            details
        )

    verify_downloads()

    print("\nALL SATELLITE BANDS DOWNLOADED!")


if __name__ == "__main__":
    main()
