
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from numpy.lib.stride_tricks import sliding_window_view

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors

import umap


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "experiment"

REFERENCE_DIR = DATA_DIR / "reference_10m"
RECON_DIR = DATA_DIR / "reconstructed_10m"

RESULTS_DIR = PROJECT_ROOT / "results" / "dimensionality_reduction"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["blue", "green", "red", "nir"]

PATCH_SIZE = 5
STRIDE = 3
MAX_PATCHES = 1500

K = 10
RANDOM_STATE = 42

NODATA = -9999.0


# ==================================================
# STEP 1: LOAD ORIGINAL AND RECONSTRUCTED IMAGES
# ==================================================

def load_images():

    print("\n========== LOADING IMAGES ==========")

    reference_bands = []
    reconstructed_bands = []

    for band in BANDS:

        reference_path = REFERENCE_DIR / f"{band}.tif"
        reconstructed_path = RECON_DIR / f"{band}.tif"

        if not reference_path.exists():
            raise FileNotFoundError(reference_path)

        if not reconstructed_path.exists():
            raise FileNotFoundError(reconstructed_path)

        with rasterio.open(reference_path) as src:
            reference = src.read(1).astype(np.float32)

        with rasterio.open(reconstructed_path) as src:
            reconstructed = src.read(1).astype(np.float32)

        # Ensure both images have identical dimensions.

        height = min(
            reference.shape[0],
            reconstructed.shape[0]
        )

        width = min(
            reference.shape[1],
            reconstructed.shape[1]
        )

        reference = reference[:height, :width]
        reconstructed = reconstructed[:height, :width]

        reference_bands.append(reference)
        reconstructed_bands.append(reconstructed)

        print(
            f"{band.upper()} | "
            f"Reference: {reference.shape} | "
            f"Reconstructed: {reconstructed.shape}"
        )

    reference_cube = np.stack(reference_bands, axis=-1)

    reconstructed_cube = np.stack(
        reconstructed_bands,
        axis=-1
    )

    print("\nImage loading completed.")

    return reference_cube, reconstructed_cube


# ==================================================
# STEP 2: CALCULATE IMAGE RECONSTRUCTION ERROR
# ==================================================

def calculate_image_error(reference, reconstructed):

    print("\n========== IMAGE RECONSTRUCTION ERROR ==========")

    rows = []

    for i, band in enumerate(BANDS):

        original = reference[:, :, i]
        restored = reconstructed[:, :, i]

        valid = (
            np.isfinite(original)
            & np.isfinite(restored)
            & (original != NODATA)
            & (restored != NODATA)
        )

        if not np.any(valid):
            raise ValueError(
                f"No common valid pixels in {band} band."
            )

        difference = original[valid] - restored[valid]

        rmse = np.sqrt(np.mean(difference ** 2))
        mae = np.mean(np.abs(difference))

        rows.append({
            "Band": band,
            "RMSE": float(rmse),
            "MAE": float(mae),
            "Valid Pixels": int(valid.sum())
        })

        print(f"{band.upper()} RMSE: {rmse:.6f}")
        print(f"{band.upper()} MAE: {mae:.6f}")

    return rows


# ==================================================
# STEP 3: EXTRACT IMAGE PATCHES
# ==================================================

def extract_patches(reference, reconstructed):

    print("\n========== PATCH EXTRACTION ==========")

    valid_reference = (
        np.isfinite(reference)
        & (reference != NODATA)
    )

    valid_reconstructed = (
        np.isfinite(reconstructed)
        & (reconstructed != NODATA)
    )

    valid_map = (
        np.all(valid_reference, axis=-1)
        & np.all(valid_reconstructed, axis=-1)
    )

    valid_windows = sliding_window_view(
        valid_map,
        (PATCH_SIZE, PATCH_SIZE)
    )

    valid_windows = valid_windows.all(axis=(-1, -2))

    reference_windows = sliding_window_view(
        reference,
        (PATCH_SIZE, PATCH_SIZE),
        axis=(0, 1)
    )

    reconstructed_windows = sliding_window_view(
        reconstructed,
        (PATCH_SIZE, PATCH_SIZE),
        axis=(0, 1)
    )

    reference_windows = reference_windows[
        ::STRIDE, ::STRIDE
    ]

    reconstructed_windows = reconstructed_windows[
        ::STRIDE, ::STRIDE
    ]

    valid_windows = valid_windows[
        ::STRIDE, ::STRIDE
    ]

    # Convert:
    # rows, columns, bands, height, width
    # into:
    # rows, columns, height, width, bands

    reference_windows = reference_windows.transpose(
        0, 1, 3, 4, 2
    )

    reconstructed_windows = reconstructed_windows.transpose(
        0, 1, 3, 4, 2
    )

    feature_count = PATCH_SIZE * PATCH_SIZE * len(BANDS)

    X_reference = reference_windows[
        valid_windows
    ].reshape(-1, feature_count)

    X_reconstructed = reconstructed_windows[
        valid_windows
    ].reshape(-1, feature_count)

    if len(X_reference) < 20:
        raise ValueError(
            "Too few valid patches. "
            "Check image validity and nodata values."
        )

    print("Total valid patches:", len(X_reference))
    print("Features per patch:", X_reference.shape[1])

    if len(X_reference) > MAX_PATCHES:

        rng = np.random.default_rng(RANDOM_STATE)

        selected = rng.choice(
            len(X_reference),
            size=MAX_PATCHES,
            replace=False
        )

        X_reference = X_reference[selected]
        X_reconstructed = X_reconstructed[selected]

    print(
        "Patches selected for analysis:",
        len(X_reference)
    )

    return X_reference, X_reconstructed


# ==================================================
# STEP 4: PREPARE FEATURES
# ==================================================

def prepare_features(X_reference, X_reconstructed):

    print("\n========== FEATURE SCALING ==========")

    # Fit one scaler jointly so both images use
    # exactly the same feature scaling.

    combined = np.vstack([
        X_reference,
        X_reconstructed
    ])

    scaler = StandardScaler()

    combined_scaled = scaler.fit_transform(combined)

    count = len(X_reference)

    X_reference_scaled = combined_scaled[:count]

    X_reconstructed_scaled = combined_scaled[count:]

    print("Feature scaling completed.")

    return X_reference_scaled, X_reconstructed_scaled


# ==================================================
# STEP 5: NEIGHBOURHOOD OVERLAP
# ==================================================

def calculate_neighbourhood_overlap(
    X_reference,
    X_reconstructed,
    label
):

    """
    Compare k-nearest-neighbour identities for corresponding
    patches in reference and reconstructed feature spaces.

    Both arrays must contain the same patches in the same
    row order.

    The result measures the fraction of shared neighbours.
    """

    print(f"\nCalculating neighbourhood overlap: {label}")

    count = len(X_reference)

    if count < 3:
        raise ValueError(
            "At least three patches are required."
        )

    if len(X_reconstructed) != count:
        raise ValueError(
            "Reference and reconstructed arrays "
            "must have the same number of patches."
        )

    k = min(K, count - 1)

    # IMPORTANT:
    # Request exactly k neighbours.
    # Calling kneighbors() WITHOUT passing X excludes
    # each training sample from its own neighbour list.

    reference_model = NearestNeighbors(
        n_neighbors=k
    )

    reconstructed_model = NearestNeighbors(
        n_neighbors=k
    )

    reference_model.fit(X_reference)

    reconstructed_model.fit(X_reconstructed)

    reference_neighbours = reference_model.kneighbors(
        return_distance=False
    )

    reconstructed_neighbours = (
        reconstructed_model.kneighbors(
            return_distance=False
        )
    )

    overlap_scores = []

    for i in range(count):

        original_set = set(reference_neighbours[i])

        reconstructed_set = set(
            reconstructed_neighbours[i]
        )

        common_neighbours = (
            original_set & reconstructed_set
        )

        overlap = len(common_neighbours) / k

        overlap_scores.append(overlap)

    score = float(np.mean(overlap_scores))

    print(
        f"{label} ({k}-NN overlap): {score:.4f}"
    )

    print(
        f"{label}: {score * 100:.2f}%"
    )

    return score


# ==================================================
# STEP 6: DIMENSIONALITY REDUCTION
# ==================================================

def perform_reduction(
    X_reference,
    X_reconstructed
):

    print("\n========== DIMENSIONALITY REDUCTION ==========")

    combined = np.vstack([
        X_reference,
        X_reconstructed
    ])

    count = len(X_reference)

    embeddings = {}

    metrics = {}

    # ------------------------------------------------
    # PCA
    # ------------------------------------------------

    print("\nRunning PCA...")

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE
    )

    pca_embedding = pca.fit_transform(combined)

    embeddings["PCA"] = pca_embedding

    explained = float(
        np.sum(pca.explained_variance_ratio_)
    )

    pca_reconstructed = pca.inverse_transform(
        pca_embedding
    )

    pca_rmse = float(
        np.sqrt(
            np.mean(
                (combined - pca_reconstructed) ** 2
            )
        )
    )

    metrics["PCA"] = {
        "Variance Explained (%)": explained * 100,
        "2D Feature Reconstruction RMSE": pca_rmse
    }

    print(
        f"PCA variance explained: {explained * 100:.2f}%"
    )

    print(
        f"PCA 2D reconstruction RMSE: {pca_rmse:.6f}"
    )

    # ------------------------------------------------
    # t-SNE
    # ------------------------------------------------

    print("\nRunning t-SNE...")

    perplexity = min(
        30,
        max(5, (len(combined) - 1) // 3)
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE
    )

    tsne_embedding = tsne.fit_transform(combined)

    embeddings["t-SNE"] = tsne_embedding

    print("t-SNE completed.")

    # ------------------------------------------------
    # UMAP
    # ------------------------------------------------

    print("\nRunning UMAP...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(combined) - 1),
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE
    )

    umap_embedding = reducer.fit_transform(combined)

    embeddings["UMAP"] = umap_embedding

    print("UMAP completed.")

    # ------------------------------------------------
    # TRUSTWORTHINESS
    # ------------------------------------------------

    print("\nCalculating trustworthiness...")

    for method, embedding in embeddings.items():

        # Initialize the dictionary for every method.
        # This prevents KeyError for t-SNE and UMAP.

        metrics.setdefault(method, {})

        trust_k = min(
            K,
            (count - 1) // 2
        )

        score_reference = trustworthiness(
            X_reference,
            embedding[:count],
            n_neighbors=trust_k
        )

        score_reconstructed = trustworthiness(
            X_reconstructed,
            embedding[count:],
            n_neighbors=trust_k
        )

        metrics[method][
            "Reference Trustworthiness"
        ] = float(score_reference)

        metrics[method][
            "Reconstructed Trustworthiness"
        ] = float(score_reconstructed)

        print(
            f"{method}: "
            f"Reference = {score_reference:.4f}, "
            f"Reconstructed = {score_reconstructed:.4f}"
        )

    # ------------------------------------------------
    # CROSS-IMAGE NEIGHBOURHOOD OVERLAP
    # ------------------------------------------------

    print(
        "\nCalculating cross-image overlap "
        "for each representation..."
    )

    for method, embedding in embeddings.items():

        # Ensure dictionary exists before adding metrics.

        metrics.setdefault(method, {})

        reference_embedding = embedding[:count]

        reconstructed_embedding = embedding[count:]

        score = calculate_neighbourhood_overlap(
            reference_embedding,
            reconstructed_embedding,
            f"{method} 2D projection"
        )

        metrics[method][
            "2D Projection Neighbourhood Overlap (%)"
        ] = score * 100

    return embeddings, metrics


# ==================================================
# STEP 7: SAVE PROJECTION PLOTS
# ==================================================

def save_projection_plots(embeddings, count):

    print("\n========== SAVING PROJECTION PLOTS ==========")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 16)
    )

    methods = ["PCA", "t-SNE", "UMAP"]

    for row, method in enumerate(methods):

        embedding = embeddings[method]

        # Original image projection

        axes[row, 0].scatter(
            embedding[:count, 0],
            embedding[:count, 1],
            s=8,
            alpha=0.6
        )

        axes[row, 0].set_title(
            f"{method} - Original 10m"
        )

        axes[row, 0].set_xlabel("Component 1")
        axes[row, 0].set_ylabel("Component 2")

        # Reconstructed image projection

        axes[row, 1].scatter(
            embedding[count:, 0],
            embedding[count:, 1],
            s=8,
            alpha=0.6
        )

        axes[row, 1].set_title(
            f"{method} - Reconstructed 10m"
        )

        axes[row, 1].set_xlabel("Component 1")
        axes[row, 1].set_ylabel("Component 2")

    plt.tight_layout()

    output_path = (
        RESULTS_DIR / "projection_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


# ==================================================
# STEP 8: SAVE EMBEDDINGS AND METRICS
# ==================================================

def save_results(
    embeddings,
    metrics,
    image_error,
    raw_pixel_overlap,
    count
):

    print("\n========== SAVING RESULTS ==========")

    rows = []

    for method, embedding in embeddings.items():

        for i in range(count):

            rows.append({
                "Method": method,
                "Patch": i,
                "Image": "Original 10m",
                "X": embedding[i, 0],
                "Y": embedding[i, 1]
            })

            rows.append({
                "Method": method,
                "Patch": i,
                "Image": "Reconstructed 10m",
                "X": embedding[count + i, 0],
                "Y": embedding[count + i, 1]
            })

    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "projection_coordinates.csv",
        index=False
    )

    metric_rows = []

    # Raw pixel feature-space baseline.

    metric_rows.append({
        "Method": "Raw pixel patch baseline",
        "Neighbourhood Overlap (%)": (
            raw_pixel_overlap * 100
        )
    })

    # PCA, t-SNE and UMAP metrics.

    for method, values in metrics.items():

        row = {"Method": method}

        row.update(values)

        metric_rows.append(row)

    # Image reconstruction errors.

    for error in image_error:

        metric_rows.append({
            "Method": f"Image error - {error['Band']}",
            "Image RMSE": error["RMSE"],
            "Image MAE": error["MAE"],
            "Valid Pixels": error["Valid Pixels"]
        })

    metrics_path = (
        RESULTS_DIR / "evaluation_metrics.csv"
    )

    pd.DataFrame(metric_rows).to_csv(
        metrics_path,
        index=False
    )

    print("Saved projection coordinates.")

    print(
        "Saved evaluation metrics:",
        metrics_path
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n==========================================")

    print(
        "SATELLITE IMAGE DIMENSIONALITY REDUCTION"
    )

    print("==========================================")

    # Load images

    reference, reconstructed = load_images()

    # Calculate image reconstruction error

    image_error = calculate_image_error(
        reference,
        reconstructed
    )

    # Extract image patches

    X_reference, X_reconstructed = extract_patches(
        reference,
        reconstructed
    )

    # Scale features

    (
        X_reference_scaled,
        X_reconstructed_scaled
    ) = prepare_features(
        X_reference,
        X_reconstructed
    )

    # ------------------------------------------------
    # RAW PIXEL BASELINE
    # ------------------------------------------------

    print("\n========== RAW PIXEL BASELINE ==========")

    raw_pixel_overlap = calculate_neighbourhood_overlap(
        X_reference_scaled,
        X_reconstructed_scaled,
        "Raw pixel patch baseline"
    )

    # ------------------------------------------------
    # DIMENSIONALITY REDUCTION
    # ------------------------------------------------

    embeddings, metrics = perform_reduction(
        X_reference_scaled,
        X_reconstructed_scaled
    )

    count = len(X_reference_scaled)

    # Save projection plots

    save_projection_plots(
        embeddings,
        count
    )

    # Save coordinates and metrics

    save_results(
        embeddings,
        metrics,
        image_error,
        raw_pixel_overlap,
        count
    )

    print("\n==========================================")

    print(
        "DIMENSIONALITY REDUCTION COMPLETED!"
    )

    print("==========================================")

    print("\nResults saved in:")

    print(RESULTS_DIR)


if __name__ == "__main__":

    main()
