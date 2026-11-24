
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from rasterio.warp import reproject, Resampling

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

EXPERIMENT_DIR = ROOT / "data" / "processed" / "experiment"

REFERENCE_DIR = EXPERIMENT_DIR / "reference_10m"
LOW_RES_DIR = EXPERIMENT_DIR / "synthetic_30m"

SR_DATA_DIR = ROOT / "data" / "processed" / "super_resolution"

BICUBIC_DIR = SR_DATA_DIR / "bicubic_10m"

# Existing SRCNN results: read only, never overwrite
OLD_SRCNN_DIR = SR_DATA_DIR / "srcnn_10m"

# New experiment outputs
RESIDUAL_DIR = SR_DATA_DIR / "residual_srcnn_10m"

RESULT_DIR = ROOT / "results" / "super_resolution"
RESIDUAL_RESULT_DIR = RESULT_DIR / "residual_experiment"

MODEL_DIR = RESULT_DIR / "model"

for folder in [
    BICUBIC_DIR,
    RESIDUAL_DIR,
    RESIDUAL_RESULT_DIR,
    MODEL_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


BANDS = ["blue", "green", "red", "nir"]

NODATA = -9999.0

RANDOM_STATE = 42

PATCH_SIZE = 15
PATCH_STRIDE = 4

BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.0001

TRAIN_END = 300
VAL_END = 360

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds():

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    torch.set_num_threads(2)


# ============================================================
# RESIDUAL SRCNN MODEL
# ============================================================

class ResidualSRCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=9,
                padding=4
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=16,
                out_channels=4,
                kernel_size=5,
                padding=2
            )
        )

        # Start with zero residual.
        # Initially, model output equals bicubic input.
        final_layer = self.features[-1]

        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, x):

        residual = self.features(x)

        return x + residual


# ============================================================
# STEP 1: LOAD IMAGES
# ============================================================

def load_images():

    print("\n========== LOADING EXPERIMENT DATA ==========")

    reference_bands = []
    low_bands = []

    reference_profile = None
    reference_transform = None
    reference_crs = None

    low_transform = None
    low_crs = None

    for band in BANDS:

        reference_path = REFERENCE_DIR / f"{band}.tif"
        low_path = LOW_RES_DIR / f"{band}.tif"

        if not reference_path.exists():
            raise FileNotFoundError(reference_path)

        if not low_path.exists():
            raise FileNotFoundError(low_path)

        with rasterio.open(reference_path) as src:

            reference = src.read(1).astype(np.float32)

            if reference_profile is None:

                reference_profile = src.profile.copy()
                reference_transform = src.transform
                reference_crs = src.crs

        with rasterio.open(low_path) as src:

            low = src.read(1).astype(np.float32)

            if low_transform is None:

                low_transform = src.transform
                low_crs = src.crs

        reference_bands.append(reference)
        low_bands.append(low)

        print(
            f"{band.upper()}: "
            f"Reference={reference.shape}, "
            f"30m={low.shape}"
        )

    expected_height = low_bands[0].shape[0] * 3
    expected_width = low_bands[0].shape[1] * 3

    reference_cube = np.stack(
        [
            band[:expected_height, :expected_width]
            for band in reference_bands
        ],
        axis=-1
    )

    low_cube = np.stack(low_bands, axis=-1)

    if reference_cube.shape[:2] != (
        expected_height,
        expected_width
    ):
        raise ValueError(
            "Reference and 30m image dimensions are not aligned."
        )

    print("\nReference cube:", reference_cube.shape)
    print("30m cube:", low_cube.shape)

    return (
        reference_cube,
        low_cube,
        reference_profile,
        reference_transform,
        reference_crs,
        low_transform,
        low_crs
    )


# ============================================================
# STEP 2: GENERATE BICUBIC IMAGE
# ============================================================

def generate_bicubic(
    low_cube,
    reference_cube,
    reference_profile,
    reference_transform,
    reference_crs,
    low_transform,
    low_crs
):

    print("\n========== GENERATING BICUBIC IMAGES ==========")

    height, width, number_of_bands = reference_cube.shape

    bicubic_cube = np.full(
        (height, width, number_of_bands),
        NODATA,
        dtype=np.float32
    )

    for index, band in enumerate(BANDS):

        output = np.full(
            (height, width),
            NODATA,
            dtype=np.float32
        )

        reproject(
            source=low_cube[:, :, index],
            destination=output,
            src_transform=low_transform,
            src_crs=low_crs,
            src_nodata=NODATA,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            dst_nodata=NODATA,
            resampling=Resampling.cubic
        )

        bicubic_cube[:, :, index] = output

        output_profile = reference_profile.copy()

        output_profile.update({
            "height": height,
            "width": width,
            "dtype": "float32",
            "count": 1,
            "nodata": NODATA,
            "compress": "deflate"
        })

        output_path = BICUBIC_DIR / f"{band}.tif"

        with rasterio.open(
            output_path,
            "w",
            **output_profile
        ) as dst:

            dst.write(output, 1)

        print("Saved bicubic:", output_path.name)

    return bicubic_cube


# ============================================================
# STEP 3: VALID PIXEL MASK
# ============================================================

def create_valid_mask(reference_cube, bicubic_cube):

    reference_valid = (
        np.all(np.isfinite(reference_cube), axis=-1)
        & np.all(reference_cube != NODATA, axis=-1)
    )

    bicubic_valid = (
        np.all(np.isfinite(bicubic_cube), axis=-1)
        & np.all(bicubic_cube != NODATA, axis=-1)
    )

    valid_mask = reference_valid & bicubic_valid

    print("\n========== VALID PIXEL CHECK ==========")
    print("Valid pixels:", int(np.sum(valid_mask)))

    return valid_mask


# ============================================================
# STEP 4: EXTRACT PATCHES
# ============================================================

def extract_patches(
    input_cube,
    target_cube,
    valid_mask,
    col_start,
    col_end
):

    input_patches = []
    target_patches = []

    height, width, _ = input_cube.shape

    col_end = min(col_end, width)

    for row in range(
        0,
        height - PATCH_SIZE + 1,
        PATCH_STRIDE
    ):

        for col in range(
            col_start,
            col_end - PATCH_SIZE + 1,
            PATCH_STRIDE
        ):

            mask_patch = valid_mask[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            if not np.all(mask_patch):
                continue

            input_patch = input_cube[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
                :
            ]

            target_patch = target_cube[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
                :
            ]

            input_patches.append(input_patch)
            target_patches.append(target_patch)

    if not input_patches:
        raise ValueError(
            f"No valid patches found in columns "
            f"{col_start}:{col_end}."
        )

    input_patches = np.asarray(
        input_patches,
        dtype=np.float32
    )

    target_patches = np.asarray(
        target_patches,
        dtype=np.float32
    )

    input_patches = np.transpose(
        input_patches,
        (0, 3, 1, 2)
    )

    target_patches = np.transpose(
        target_patches,
        (0, 3, 1, 2)
    )

    return input_patches, target_patches


# ============================================================
# STEP 5: PREPARE TRAINING AND VALIDATION DATA
# ============================================================

def prepare_training_data(
    bicubic_cube,
    reference_cube,
    valid_mask
):

    print("\n========== EXTRACTING PATCHES ==========")

    train_x, train_y = extract_patches(
        bicubic_cube,
        reference_cube,
        valid_mask,
        0,
        TRAIN_END
    )

    val_x, val_y = extract_patches(
        bicubic_cube,
        reference_cube,
        valid_mask,
        TRAIN_END,
        VAL_END
    )

    print("Training patches:", len(train_x))
    print("Validation patches:", len(val_x))

    train_dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y)
    )

    val_dataset = TensorDataset(
        torch.from_numpy(val_x),
        torch.from_numpy(val_y)
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader


# ============================================================
# STEP 6: TRAIN RESIDUAL SRCNN
# ============================================================

def train_residual_srcnn(train_loader, val_loader):

    print("\n========== TRAINING RESIDUAL SRCNN ==========")
    print("Training device:", DEVICE)

    model = ResidualSRCNN().to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_loss = float("inf")
    best_state = None

    history = []

    for epoch in range(EPOCHS):

        model.train()

        train_loss_total = 0.0
        train_batches = 0

        for inputs, targets in train_loader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(inputs)

            loss = criterion(
                predictions,
                targets
            )

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_batches += 1

        train_loss = train_loss_total / train_batches

        model.eval()

        validation_loss_total = 0.0
        validation_batches = 0

        with torch.no_grad():

            for inputs, targets in val_loader:

                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                predictions = model(inputs)

                loss = criterion(
                    predictions,
                    targets
                )

                validation_loss_total += loss.item()
                validation_batches += 1

        validation_loss = (
            validation_loss_total / validation_batches
        )

        history.append({
            "Epoch": epoch + 1,
            "Training Loss": train_loss,
            "Validation Loss": validation_loss
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Validation Loss: {validation_loss:.8f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "Residual SRCNN training failed."
        )

    model.load_state_dict(best_state)

    model_path = MODEL_DIR / "residual_srcnn.pth"

    torch.save(
        model.state_dict(),
        model_path
    )

    print("\nBest validation loss:", best_validation_loss)
    print("Saved model:", model_path)

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        RESIDUAL_RESULT_DIR / "training_loss.csv",
        index=False
    )

    save_training_plot(history_df)

    return model, history_df


# ============================================================
# STEP 7: SAVE TRAINING PLOT
# ============================================================

def save_training_plot(history):

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        history["Epoch"],
        history["Training Loss"],
        label="Training Loss"
    )

    ax.plot(
        history["Epoch"],
        history["Validation Loss"],
        label="Validation Loss"
    )

    ax.set_title("Residual SRCNN Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path = (
        RESIDUAL_RESULT_DIR / "training_loss.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved training plot:", output_path)


# ============================================================
# STEP 8: GENERATE RESIDUAL SRCNN IMAGE
# ============================================================

def generate_residual_image(
    model,
    bicubic_cube,
    valid_mask,
    reference_profile
):

    print("\n========== GENERATING RESIDUAL SRCNN IMAGE ==========")

    model.eval()

    input_array = np.nan_to_num(
        bicubic_cube,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    input_array = np.where(
        input_array == NODATA,
        0.0,
        input_array
    )

    input_tensor = torch.from_numpy(
        np.transpose(
            input_array,
            (2, 0, 1)
        )[None, ...].astype(np.float32)
    )

    with torch.no_grad():

        prediction = model(
            input_tensor.to(DEVICE)
        )

    prediction = prediction.cpu().numpy()[0]

    srcnn_cube = np.transpose(
        prediction,
        (1, 2, 0)
    )

    srcnn_cube = np.clip(
        srcnn_cube,
        0.0,
        1.0
    )

    srcnn_cube[~valid_mask] = NODATA

    height, width, _ = srcnn_cube.shape

    output_profile = reference_profile.copy()

    output_profile.update({
        "height": height,
        "width": width,
        "dtype": "float32",
        "count": 1,
        "nodata": NODATA,
        "compress": "deflate"
    })

    for index, band in enumerate(BANDS):

        output_path = RESIDUAL_DIR / f"{band}.tif"

        with rasterio.open(
            output_path,
            "w",
            **output_profile
        ) as dst:

            dst.write(
                srcnn_cube[:, :, index],
                1
            )

        print("Saved residual SRCNN:", output_path.name)

    return srcnn_cube


# ============================================================
# STEP 9: LOAD PREVIOUS SRCNN RESULTS
# ============================================================

def load_old_srcnn(reference_cube):

    paths = [
        OLD_SRCNN_DIR / f"{band}.tif"
        for band in BANDS
    ]

    if not all(path.exists() for path in paths):
        print("\nPrevious SRCNN files not found. Skipping old SRCNN comparison.")
        return None

    old_bands = []

    for path in paths:

        with rasterio.open(path) as src:
            old_bands.append(
                src.read(1).astype(np.float32)
            )

    old_cube = np.stack(old_bands, axis=-1)

    if old_cube.shape != reference_cube.shape:
        print("\nPrevious SRCNN dimensions do not match. Skipping old SRCNN.")
        return None

    print("\nPrevious SRCNN loaded for comparison.")

    return old_cube


# ============================================================
# STEP 10: EVALUATE HELD-OUT TEST REGION
# ============================================================

def calculate_metrics(
    reference_cube,
    bicubic_cube,
    residual_cube,
    valid_mask,
    old_srcnn_cube=None
):

    print("\n========== HELD-OUT TEST EVALUATION ==========")

    test_mask = valid_mask.copy()

    test_mask[:, :VAL_END] = False

    if not np.any(test_mask):
        raise ValueError("No valid test pixels found.")

    methods = {
        "Bicubic": bicubic_cube,
        "Residual SRCNN": residual_cube
    }

    if old_srcnn_cube is not None:
        methods["Original SRCNN"] = old_srcnn_cube

    rows = []

    for index, band in enumerate(BANDS):

        reference_values = reference_cube[:, :, index][test_mask]

        for method_name, method_cube in methods.items():

            predicted_values = method_cube[:, :, index][test_mask]

            error = reference_values - predicted_values

            rmse = float(
                np.sqrt(np.mean(error ** 2))
            )

            mae = float(
                np.mean(np.abs(error))
            )

            psnr = (
                float(20 * np.log10(1.0 / rmse))
                if rmse > 0 else float("inf")
            )

            rows.append({
                "Band": band.upper(),
                "Method": method_name,
                "RMSE": rmse,
                "MAE": mae,
                "PSNR (dB)": psnr,
                "Test Pixels": int(np.sum(test_mask))
            })

            print(
                f"{band.upper():5} | {method_name:16} | "
                f"RMSE={rmse:.6f} | "
                f"MAE={mae:.6f} | "
                f"PSNR={psnr:.3f} dB"
            )

    metrics_df = pd.DataFrame(rows)

    metrics_df.to_csv(
        RESIDUAL_RESULT_DIR / "super_resolution_metrics.csv",
        index=False
    )

    return metrics_df, test_mask


# ============================================================
# STEP 11: SAVE SPECTRAL STATISTICS
# ============================================================

def save_spectral_statistics(
    reference_cube,
    bicubic_cube,
    residual_cube,
    valid_mask,
    old_srcnn_cube=None
):

    print("\n========== SPECTRAL STATISTICS ==========")

    methods = {
        "Native Reference": reference_cube,
        "Bicubic": bicubic_cube,
        "Residual SRCNN": residual_cube
    }

    if old_srcnn_cube is not None:
        methods["Original SRCNN"] = old_srcnn_cube

    rows = []

    for method_name, cube in methods.items():

        for index, band in enumerate(BANDS):

            values = cube[:, :, index][valid_mask]

            rows.append({
                "Method": method_name,
                "Band": band.upper(),
                "Mean": float(np.mean(values)),
                "Standard Deviation": float(np.std(values)),
                "Minimum": float(np.min(values)),
                "Maximum": float(np.max(values))
            })

    statistics_df = pd.DataFrame(rows)

    statistics_df.to_csv(
        RESIDUAL_RESULT_DIR / "spectral_statistics.csv",
        index=False
    )

    print(statistics_df.to_string(index=False))

    return statistics_df


# ============================================================
# STEP 12: RGB VISUALIZATION
# ============================================================

def make_rgb(cube, valid_mask, reference_cube):

    channel_indices = [2, 1, 0]

    rgb_channels = []

    for index in channel_indices:

        reference_values = reference_cube[:, :, index][valid_mask]

        low = np.percentile(reference_values, 2)
        high = np.percentile(reference_values, 98)

        channel = cube[:, :, index].copy()

        channel = np.where(
            channel == NODATA,
            low,
            channel
        )

        channel = np.clip(
            (channel - low) / (high - low + 1e-8),
            0,
            1
        )

        rgb_channels.append(channel)

    return np.stack(rgb_channels, axis=-1)


def save_rgb_comparison(
    reference_cube,
    bicubic_cube,
    residual_cube,
    valid_mask,
    old_srcnn_cube=None
):

    print("\n========== SAVING RGB COMPARISON ==========")

    images = [
        reference_cube,
        bicubic_cube
    ]

    titles = [
        "Native Sentinel-2 10m",
        "Bicubic 30m to 10m"
    ]

    if old_srcnn_cube is not None:

        images.append(old_srcnn_cube)
        titles.append("Original SRCNN")

    images.append(residual_cube)
    titles.append("Residual SRCNN")

    number_of_images = len(images)

    fig, axes = plt.subplots(
        1,
        number_of_images,
        figsize=(6 * number_of_images, 6)
    )

    if number_of_images == 1:
        axes = [axes]

    for ax, cube, title in zip(axes, images, titles):

        rgb = make_rgb(
            cube,
            valid_mask,
            reference_cube
        )

        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()

    output_path = (
        RESIDUAL_RESULT_DIR / "super_resolution_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    set_random_seeds()

    print("\n==============================================")
    print("RESIDUAL SRCNN SATELLITE EXPERIMENT")
    print("==============================================")

    print("Device:", DEVICE)

    (
        reference_cube,
        low_cube,
        reference_profile,
        reference_transform,
        reference_crs,
        low_transform,
        low_crs
    ) = load_images()

    bicubic_cube = generate_bicubic(
        low_cube,
        reference_cube,
        reference_profile,
        reference_transform,
        reference_crs,
        low_transform,
        low_crs
    )

    valid_mask = create_valid_mask(
        reference_cube,
        bicubic_cube
    )

    if reference_cube.shape[1] <= VAL_END + PATCH_SIZE:
        raise ValueError(
            "Image width is too small for the configured split."
        )

    train_loader, val_loader = prepare_training_data(
        bicubic_cube,
        reference_cube,
        valid_mask
    )

    model, history = train_residual_srcnn(
        train_loader,
        val_loader
    )

    residual_cube = generate_residual_image(
        model,
        bicubic_cube,
        valid_mask,
        reference_profile
    )

    old_srcnn_cube = load_old_srcnn(reference_cube)

    metrics_df, test_mask = calculate_metrics(
        reference_cube,
        bicubic_cube,
        residual_cube,
        valid_mask,
        old_srcnn_cube
    )

    statistics_df = save_spectral_statistics(
        reference_cube,
        bicubic_cube,
        residual_cube,
        valid_mask,
        old_srcnn_cube
    )

    save_rgb_comparison(
        reference_cube,
        bicubic_cube,
        residual_cube,
        valid_mask,
        old_srcnn_cube
    )

    print("\n==============================================")
    print("RESIDUAL SRCNN EXPERIMENT COMPLETED")
    print("==============================================")

    print("\nNew results folder:")
    print(RESIDUAL_RESULT_DIR)

    print("\nTest region: columns", VAL_END, "to", reference_cube.shape[1] - 1)

    print("\nEvaluation metrics:")
    print(metrics_df.to_string(index=False))

    print("\nPrevious SRCNN results have been preserved.")
    print("All comparisons use the same held-out test region.")


if __name__ == "__main__":
    main()
