from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed" / "experiment"

REFERENCE_DIR = DATA / "reference_10m"
RECON_DIR = DATA / "reconstructed_10m"

OUTPUT_DIR = ROOT / "results" / "landcover_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["blue", "green", "red", "nir"]

N_CLUSTERS = 6
RANDOM_STATE = 42

# Silhouette is expensive on large datasets.
# We use a representative sample for this metric only.
SILHOUETTE_SAMPLE_SIZE = 5000

NODATA = -9999.0


# ============================================================
# STEP 1: LOAD FOUR-BAND IMAGES
# ============================================================

def load_images():

    print("\n========== LOADING SATELLITE IMAGES ==========")

    reference_bands = []
    reconstructed_bands = []

    reference_profile = None

    for band in BANDS:

        reference_path = REFERENCE_DIR / f"{band}.tif"
        reconstructed_path = RECON_DIR / f"{band}.tif"

        if not reference_path.exists():
            raise FileNotFoundError(reference_path)

        if not reconstructed_path.exists():
            raise FileNotFoundError(reconstructed_path)

        # Load original reference image
        with rasterio.open(reference_path) as src:

            reference = src.read(1).astype(np.float32)

            if reference_profile is None:
                reference_profile = src.profile.copy()

        # Load reconstructed image
        with rasterio.open(reconstructed_path) as src:

            reconstructed = src.read(1).astype(np.float32)

        # Match image dimensions.
        # Crop from right/bottom edges only.
        min_height = min(
            reference.shape[0],
            reconstructed.shape[0]
        )

        min_width = min(
            reference.shape[1],
            reconstructed.shape[1]
        )

        reference = reference[:min_height, :min_width]
        reconstructed = reconstructed[:min_height, :min_width]

        reference_bands.append(reference)
        reconstructed_bands.append(reconstructed)

        print(
            f"{band.upper()} loaded: "
            f"Reference={reference.shape}, "
            f"Reconstructed={reconstructed.shape}"
        )

    # Combine four bands into image cubes
    reference_cube = np.stack(reference_bands, axis=-1)

    reconstructed_cube = np.stack(
        reconstructed_bands,
        axis=-1
    )

    # Update raster metadata for cropped reference image
    reference_profile.update({
        "height": reference_cube.shape[0],
        "width": reference_cube.shape[1]
    })

    print("\n========== IMAGE LOADING COMPLETED ==========")
    print("Reference shape:", reference_cube.shape)
    print("Reconstructed shape:", reconstructed_cube.shape)

    return reference_cube, reconstructed_cube, reference_profile


# ============================================================
# STEP 2: IDENTIFY VALID PIXELS
# ============================================================

def prepare_pixels(reference, reconstructed):

    print("\n========== PREPARING PIXELS ==========")

    valid_reference = (
        np.all(np.isfinite(reference), axis=-1)
        & np.all(reference != NODATA, axis=-1)
    )

    valid_reconstructed = (
        np.all(np.isfinite(reconstructed), axis=-1)
        & np.all(reconstructed != NODATA, axis=-1)
    )

    valid_mask = valid_reference & valid_reconstructed

    X_reference = reference[valid_mask]
    X_reconstructed = reconstructed[valid_mask]

    if len(X_reference) < N_CLUSTERS:
        raise ValueError(
            "Not enough valid pixels for clustering."
        )

    print("Valid pixels:", len(X_reference))
    print("Spectral features per pixel:", len(BANDS))

    return X_reference, X_reconstructed, valid_mask


# ============================================================
# STEP 3: STANDARDIZE FEATURES
# ============================================================

def scale_features(X_reference, X_reconstructed):

    print("\n========== STANDARDIZING FEATURES ==========")

    scaler = StandardScaler()

    X_reference_scaled = scaler.fit_transform(
        X_reference
    )

    X_reconstructed_scaled = scaler.transform(
        X_reconstructed
    )

    print("Feature standardization completed.")

    return X_reference_scaled, X_reconstructed_scaled


# ============================================================
# STEP 4: K-MEANS CLUSTERING
# ============================================================

def perform_clustering(
    X_reference_scaled,
    X_reconstructed_scaled
):

    print("\n========== K-MEANS CLUSTERING ==========")

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )

    # Fit K-Means using original reference imagery
    reference_labels = model.fit_predict(
        X_reference_scaled
    )

    # Apply same model to reconstructed imagery
    reconstructed_labels = model.predict(
        X_reconstructed_scaled
    )

    print("K-Means clustering completed.")
    print("Number of clusters:", N_CLUSTERS)

    return reference_labels, reconstructed_labels


# ============================================================
# STEP 5: CREATE CLUSTER MAPS
# ============================================================

def create_cluster_maps(
    reference_labels,
    reconstructed_labels,
    valid_mask,
    profile
):

    print("\n========== CREATING CLUSTER MAPS ==========")

    height, width = valid_mask.shape

    # Zero represents invalid pixels
    reference_map = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    reconstructed_map = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    reference_map[valid_mask] = reference_labels + 1

    reconstructed_map[valid_mask] = (
        reconstructed_labels + 1
    )

    output_profile = profile.copy()

    output_profile.update({
        "height": height,
        "width": width,
        "dtype": "uint8",
        "count": 1,
        "nodata": 0,
        "compress": "deflate"
    })

    paths = [
        OUTPUT_DIR / "original_cluster_map.tif",
        OUTPUT_DIR / "reconstructed_cluster_map.tif"
    ]

    maps = [
        reference_map,
        reconstructed_map
    ]

    for path, cluster_map in zip(paths, maps):

        with rasterio.open(
            path,
            "w",
            **output_profile
        ) as dst:

            dst.write(cluster_map, 1)

        print("Saved:", path.name)

    return reference_map, reconstructed_map


# ============================================================
# STEP 6: CLUSTER-WISE EVALUATION
# ============================================================

def evaluate_clusters(
    X_reference,
    X_reconstructed,
    reference_labels,
    reconstructed_labels
):

    print("\n========== CLUSTER-WISE EVALUATION ==========")

    rows = []

    total_pixels = len(reference_labels)

    for cluster in range(N_CLUSTERS):

        original_mask = reference_labels == cluster

        reconstructed_mask = (
            reconstructed_labels == cluster
        )

        intersection = np.sum(
            original_mask & reconstructed_mask
        )

        union = np.sum(
            original_mask | reconstructed_mask
        )

        iou = (
            intersection / union
            if union > 0 else 0.0
        )

        original_count = int(np.sum(original_mask))

        reconstructed_count = int(
            np.sum(reconstructed_mask)
        )

        dice_denominator = (
            original_count + reconstructed_count
        )

        dice = (
            2 * intersection / dice_denominator
            if dice_denominator > 0 else 0.0
        )

        # Spectral reconstruction error for pixels
        # belonging to this cluster in the reference image

        if np.any(original_mask):

            differences = (
                X_reference[original_mask]
                - X_reconstructed[original_mask]
            )

            rmse = float(
                np.sqrt(np.mean(differences ** 2))
            )

            mae = float(
                np.mean(np.abs(differences))
            )

        else:

            rmse = np.nan
            mae = np.nan

        original_percentage = (
            original_count / total_pixels * 100
        )

        reconstructed_percentage = (
            reconstructed_count / total_pixels * 100
        )

        rows.append({

            "Cluster": cluster + 1,

            "Original Pixels": original_count,

            "Reconstructed Pixels": reconstructed_count,

            "Original Percentage": original_percentage,

            "Reconstructed Percentage": reconstructed_percentage,

            "Spectral RMSE": rmse,

            "Spectral MAE": mae,

            "IoU": iou,

            "Dice Score": dice

        })

        print(
            f"Cluster {cluster + 1}: "
            f"RMSE={rmse:.5f}, "
            f"IoU={iou:.4f}, "
            f"Original pixels={original_count}"
        )

    results = pd.DataFrame(rows)

    return results


# ============================================================
# STEP 7: OVERALL CLUSTER AGREEMENT
# ============================================================

def calculate_overall_agreement(
    reference_labels,
    reconstructed_labels
):

    matrix = confusion_matrix(
        reference_labels,
        reconstructed_labels,
        labels=np.arange(N_CLUSTERS)
    )

    agreement = (
        np.trace(matrix) / np.sum(matrix)
    )

    print("\n========== OVERALL CLUSTER AGREEMENT ==========")

    print(
        f"Same-label agreement: {agreement * 100:.2f}%"
    )

    return matrix, agreement


# ============================================================
# STEP 8: CLUSTER VALIDITY METRICS
# ============================================================

def calculate_cluster_validity(
    X_reference_scaled,
    X_reconstructed_scaled,
    reference_labels,
    reconstructed_labels
):

    print("\n========== CLUSTER VALIDITY METRICS ==========")

    rows = []

    # --------------------------------------------------------
    # Silhouette score
    # --------------------------------------------------------
    # Measures cohesion and separation.
    # Range: -1 to 1.
    # Higher is generally better.

    silhouette_reference = silhouette_score(
        X_reference_scaled,
        reference_labels,
        sample_size=min(
            SILHOUETTE_SAMPLE_SIZE,
            len(X_reference_scaled)
        ),
        random_state=RANDOM_STATE
    )

    silhouette_reconstructed = silhouette_score(
        X_reconstructed_scaled,
        reconstructed_labels,
        sample_size=min(
            SILHOUETTE_SAMPLE_SIZE,
            len(X_reconstructed_scaled)
        ),
        random_state=RANDOM_STATE
    )

    print(
        f"Reference Silhouette Score: "
        f"{silhouette_reference:.5f}"
    )

    print(
        f"Reconstructed Silhouette Score: "
        f"{silhouette_reconstructed:.5f}"
    )

    rows.append({
        "Metric": "Silhouette Score",
        "Reference": silhouette_reference,
        "Reconstructed": silhouette_reconstructed,
        "Interpretation": "Higher is better"
    })

    # --------------------------------------------------------
    # Davies-Bouldin Index
    # --------------------------------------------------------
    # Measures similarity between clusters.
    # Lower values generally indicate better separation.

    db_reference = davies_bouldin_score(
        X_reference_scaled,
        reference_labels
    )

    db_reconstructed = davies_bouldin_score(
        X_reconstructed_scaled,
        reconstructed_labels
    )

    print(
        f"Reference Davies-Bouldin Index: "
        f"{db_reference:.5f}"
    )

    print(
        f"Reconstructed Davies-Bouldin Index: "
        f"{db_reconstructed:.5f}"
    )

    rows.append({
        "Metric": "Davies-Bouldin Index",
        "Reference": db_reference,
        "Reconstructed": db_reconstructed,
        "Interpretation": "Lower is better"
    })

    # --------------------------------------------------------
    # Adjusted Rand Index (ARI)
    # --------------------------------------------------------
    # Compares the reference and reconstructed assignments.
    # Adjusted for random chance.
    # 1 = identical clustering, around 0 = random agreement.

    ari = adjusted_rand_score(
        reference_labels,
        reconstructed_labels
    )

    print(
        f"Adjusted Rand Index: {ari:.5f}"
    )

    rows.append({
        "Metric": "Adjusted Rand Index (ARI)",
        "Reference": np.nan,
        "Reconstructed": np.nan,
        "Interpretation": "Reference vs reconstructed",
        "Comparison Score": ari
    })

    # --------------------------------------------------------
    # Normalized Mutual Information (NMI)
    # --------------------------------------------------------
    # Measures shared information between two clusterings.
    # 0 = no shared information.
    # 1 = identical clustering.

    nmi = normalized_mutual_info_score(
        reference_labels,
        reconstructed_labels
    )

    print(
        f"Normalized Mutual Information: {nmi:.5f}"
    )

    rows.append({
        "Metric": "Normalized Mutual Information (NMI)",
        "Reference": np.nan,
        "Reconstructed": np.nan,
        "Interpretation": "Reference vs reconstructed",
        "Comparison Score": nmi
    })

    validity_results = pd.DataFrame(rows)

    return validity_results


# ============================================================
# STEP 9: VISUALIZE CLUSTER MAPS
# ============================================================

def save_cluster_visualization(
    reference_map,
    reconstructed_map
):

    print("\n========== SAVING CLUSTER VISUALIZATION ==========")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    original_plot = axes[0].imshow(
        reference_map,
        cmap="tab10",
        vmin=0,
        vmax=N_CLUSTERS
    )

    axes[0].set_title(
        "Original 10m - Spectral Clusters"
    )

    axes[0].set_xlabel("Pixel column")
    axes[0].set_ylabel("Pixel row")

    axes[1].imshow(
        reconstructed_map,
        cmap="tab10",
        vmin=0,
        vmax=N_CLUSTERS
    )

    axes[1].set_title(
        "Reconstructed 10m - Spectral Clusters"
    )

    axes[1].set_xlabel("Pixel column")
    axes[1].set_ylabel("Pixel row")

    colorbar = fig.colorbar(
        original_plot,
        ax=axes,
        ticks=range(N_CLUSTERS + 1),
        shrink=0.8
    )

    colorbar.set_label(
        "Cluster ID (0 = invalid pixel)"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "cluster_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


# ============================================================
# STEP 10: SAVE EVALUATION RESULTS
# ============================================================

def save_results(
    results,
    matrix,
    agreement,
    validity_results
):

    # Existing cluster metrics
    results.to_csv(
        OUTPUT_DIR / "cluster_metrics.csv",
        index=False
    )

    # Existing confusion matrix
    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Original_{i + 1}"
            for i in range(N_CLUSTERS)
        ],
        columns=[
            f"Reconstructed_{i + 1}"
            for i in range(N_CLUSTERS)
        ]
    )

    matrix_df.to_csv(
        OUTPUT_DIR / "cluster_confusion_matrix.csv"
    )

    # NEW: Cluster validity metrics
    validity_results.to_csv(
        OUTPUT_DIR / "cluster_validity_metrics.csv",
        index=False
    )

    # Summary
    summary = pd.DataFrame([{

        "Number of clusters": N_CLUSTERS,

        "Overall same-label agreement (%)": agreement * 100,

        "Mean cluster IoU": results["IoU"].mean(),

        "Mean cluster spectral RMSE": (
            results["Spectral RMSE"].mean()
        ),

        "Reference Silhouette Score": (
            validity_results.loc[
                validity_results["Metric"] == "Silhouette Score",
                "Reference"
            ].iloc[0]
        ),

        "Reconstructed Silhouette Score": (
            validity_results.loc[
                validity_results["Metric"] == "Silhouette Score",
                "Reconstructed"
            ].iloc[0]
        ),

        "Reference Davies-Bouldin Index": (
            validity_results.loc[
                validity_results["Metric"] == "Davies-Bouldin Index",
                "Reference"
            ].iloc[0]
        ),

        "Reconstructed Davies-Bouldin Index": (
            validity_results.loc[
                validity_results["Metric"] == "Davies-Bouldin Index",
                "Reconstructed"
            ].iloc[0]
        ),

        "Adjusted Rand Index (ARI)": (
            validity_results.loc[
                validity_results["Metric"] == "Adjusted Rand Index (ARI)",
                "Comparison Score"
            ].iloc[0]
        ),

        "Normalized Mutual Information (NMI)": (
            validity_results.loc[
                validity_results["Metric"] == "Normalized Mutual Information (NMI)",
                "Comparison Score"
            ].iloc[0]
        )

    }])

    summary.to_csv(
        OUTPUT_DIR / "cluster_summary.csv",
        index=False
    )

    print("\nAll evaluation files saved.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("SATELLITE SPECTRAL STRUCTURE ANALYSIS")
    print("==========================================")

    # Step 1
    reference, reconstructed, profile = load_images()

    # Step 2
    X_reference, X_reconstructed, valid_mask = prepare_pixels(
        reference,
        reconstructed
    )

    # Step 3
    X_reference_scaled, X_reconstructed_scaled = scale_features(
        X_reference,
        X_reconstructed
    )

    # Step 4
    reference_labels, reconstructed_labels = perform_clustering(
        X_reference_scaled,
        X_reconstructed_scaled
    )

    # Step 5
    reference_map, reconstructed_map = create_cluster_maps(
        reference_labels,
        reconstructed_labels,
        valid_mask,
        profile
    )

    # Step 6
    results = evaluate_clusters(
        X_reference,
        X_reconstructed,
        reference_labels,
        reconstructed_labels
    )

    # Step 7
    matrix, agreement = calculate_overall_agreement(
        reference_labels,
        reconstructed_labels
    )

    # Step 8 - NEW VALIDITY METRICS
    validity_results = calculate_cluster_validity(
        X_reference_scaled,
        X_reconstructed_scaled,
        reference_labels,
        reconstructed_labels
    )

    # Step 9
    save_cluster_visualization(
        reference_map,
        reconstructed_map
    )

    # Step 10
    save_results(
        results,
        matrix,
        agreement,
        validity_results
    )

    print("\n==========================================")
    print("SPECTRAL STRUCTURE ANALYSIS COMPLETED!")
    print("==========================================")

    print("\nResults saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()