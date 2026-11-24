
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import colormaps
from html import unescape


# ============================================================
# TERRA | SATELLITE INTELLIGENCE
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed" / "experiment"
RESULTS = ROOT / "results" / "dimensionality_reduction"
CLUSTER_RESULTS = ROOT / "results" / "landcover_analysis"

SR_DATA = ROOT / "data" / "processed" / "super_resolution"
SR_RESULTS = ROOT / "results" / "super_resolution"
SR_RESIDUAL_RESULTS = SR_RESULTS / "residual_experiment"

REFERENCE = DATA / "reference_10m"
DEGRADED = DATA / "synthetic_30m"
RECONSTRUCTED = DATA / "reconstructed_10m"

COORDINATES_FILE = RESULTS / "projection_coordinates.csv"
METRICS_FILE = RESULTS / "evaluation_metrics.csv"

CLUSTER_METRICS_FILE = CLUSTER_RESULTS / "cluster_metrics.csv"
CLUSTER_SUMMARY_FILE = CLUSTER_RESULTS / "cluster_summary.csv"
CLUSTER_MATRIX_FILE = CLUSTER_RESULTS / "cluster_confusion_matrix.csv"
ORIGINAL_CLUSTER_MAP = CLUSTER_RESULTS / "original_cluster_map.tif"
RECON_CLUSTER_MAP = CLUSTER_RESULTS / "reconstructed_cluster_map.tif"
CLUSTER_FIGURE = CLUSTER_RESULTS / "cluster_comparison.png"

# Super-resolution outputs
BICUBIC_SR = SR_DATA / "bicubic_10m"
ORIGINAL_SRCNN_SR = SR_DATA / "srcnn_10m"
RESIDUAL_SRCNN_SR = SR_DATA / "residual_srcnn_10m"

SR_METRICS_FILE = SR_RESIDUAL_RESULTS / "super_resolution_metrics.csv"
SR_LOSS_FILE = SR_RESIDUAL_RESULTS / "training_loss.csv"
SR_STATS_FILE = SR_RESIDUAL_RESULTS / "spectral_statistics.csv"
SR_COMPARISON_FIGURE = SR_RESIDUAL_RESULTS / "super_resolution_comparison.png"
SR_LOSS_FIGURE = SR_RESIDUAL_RESULTS / "training_loss.png"

BANDS = ["blue", "green", "red", "nir"]
N_CLUSTERS = 6

st.set_page_config(
    page_title="TERRA | Satellite Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080e18;
    --panel: #101a29;
    --panel2: #131f30;
    --border: #243348;
    --muted: #8c9db3;
    --text: #edf4fc;
    --cyan: #43d9d0;
    --blue: #62a9ff;
    --green: #4bd6a0;
}
.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, #102a3c 0%, transparent 35%),
        #080e18;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #0c1522;
    border-right: 1px solid #243348;
}
[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    color: #edf4fc !important;
    letter-spacing: -0.5px;
}
p, label {
    color: #aebdd0;
}
.terra-brand {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: 4px;
    color: #f4f8ff;
}
.terra-sub {
    color: #43d9d0;
    font-size: 10px;
    letter-spacing: 3px;
    margin-top: 4px;
}
.eyebrow {
    color: #43d9d0;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero-title {
    font-size: 37px;
    line-height: 1.15;
    font-weight: 800;
    color: #f2f7ff;
    letter-spacing: -1.5px;
}
.hero-description {
    color: #92a5bb;
    font-size: 14px;
    max-width: 760px;
    line-height: 1.8;
}
.metric-card {
    background: linear-gradient(145deg, #142438, #101a29);
    border: 1px solid #263b50;
    border-radius: 13px;
    padding: 19px 20px;
    min-height: 120px;
}
.metric-label {
    color: #91a4ba;
    font-size: 11px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}
.metric-value {
    color: #f3f8ff;
    font-family: 'DM Mono', monospace;
    font-size: 27px;
    font-weight: 500;
    margin-top: 10px;
}
.metric-note {
    color: #43d9d0;
    font-size: 11px;
    margin-top: 5px;
}
.section-heading {
    font-size: 19px;
    font-weight: 700;
    color: #f0f6ff;
}
.section-caption {
    color: #8295ad;
    font-size: 12px;
    margin-top: -5px;
}
.panel {
    background: #101a29;
    border: 1px solid #243348;
    border-radius: 14px;
    padding: 18px;
}
.status-pill {
    display: inline-block;
    border: 1px solid #1f665f;
    color: #59e0c7;
    background: #102e30;
    border-radius: 30px;
    padding: 5px 11px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
}
.small-muted {
    color: #8498af;
    font-size: 11px;
}
hr {
    border-color: #243348 !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    color: #9db0c7;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    color: #43d9d0 !important;
}
.stButton button {
    border-radius: 8px;
    border: 1px solid #315168;
    background: #13283a;
    color: #e8f5ff;
}
[data-testid="stMetric"] {
    background: #101a29;
    border: 1px solid #243348;
    border-radius: 12px;
    padding: 14px;
}
[data-testid="stDataFrame"] {
    border: 1px solid #243348;
    border-radius: 10px;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data
def load_raster(path):
    with rasterio.open(path) as src:
        image = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        image[image == nodata] = np.nan

    image[~np.isfinite(image)] = np.nan
    return image


@st.cache_data
def load_table(path):
    frame = pd.read_csv(path)
    cleaned_columns = []

    for column in frame.columns:
        name = unescape(str(column)).replace("**", "").strip()
        cleaned_columns.append(name)

    frame.columns = cleaned_columns
    return frame


@st.cache_data
def load_matrix(path):
    return pd.read_csv(path, index_col=0)


def metric_value(frame, method, column):
    if column not in frame.columns or "Method" not in frame.columns:
        return None

    result = frame.loc[frame["Method"] == method, column]

    if result.empty or pd.isna(result.iloc[0]):
        return None

    return float(result.iloc[0])


def format_metric(value, suffix=""):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}{suffix}"


def find_column(frame, candidates):
    """Find a column using case-insensitive matching."""
    lookup = {str(col).strip().lower(): col for col in frame.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]

    return None


def image_figure(image, title, cmap="viridis", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#101a29")
    ax.set_facecolor("#101a29")

    valid = image[np.isfinite(image)]

    if len(valid) == 0:
        ax.text(
            0.5, 0.5, "No valid pixels",
            color="white", ha="center", va="center"
        )
        ax.axis("off")
        return fig

    if vmin is None or vmax is None:
        low, high = np.percentile(valid, [2, 98])
    else:
        low, high = float(vmin), float(vmax)

    if low == high:
        high = low + 1e-6

    im = ax.imshow(
        image,
        cmap=cmap,
        vmin=low,
        vmax=high,
        interpolation="nearest"
    )

    ax.set_title(title, color="#edf4fc", fontsize=12, pad=12)
    ax.tick_params(colors="#8498af", labelsize=8)
    ax.set_xlabel("Pixel column", color="#8498af")
    ax.set_ylabel("Pixel row", color="#8498af")

    for spine in ax.spines.values():
        spine.set_color("#243348")

    colorbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    colorbar.ax.tick_params(colors="#8498af", labelsize=8)
    fig.tight_layout()
    return fig


def projection_figure(frame, method):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#101a29")
    ax.set_facecolor("#101a29")

    colors = {
        "Original 10m": "#43d9d0",
        "Reconstructed 10m": "#ffad66"
    }

    for image_name in ["Original 10m", "Reconstructed 10m"]:
        subset = frame[frame["Image"] == image_name]
        ax.scatter(
            subset["X"], subset["Y"],
            s=15, alpha=0.62,
            c=colors[image_name],
            label=image_name,
            edgecolors="none"
        )

    ax.set_title(
        f"{method} | PATCH EMBEDDING",
        color="#edf4fc", fontsize=13, pad=14
    )
    ax.set_xlabel("Dimension 1", color="#91a4ba")
    ax.set_ylabel("Dimension 2", color="#91a4ba")
    ax.tick_params(colors="#8498af", labelsize=8)
    ax.grid(color="#29394c", alpha=0.5, linewidth=0.6)

    for spine in ax.spines.values():
        spine.set_color("#29394c")

    ax.legend(
        facecolor="#162538",
        edgecolor="#29394c",
        labelcolor="#edf4fc"
    )
    fig.tight_layout()
    return fig


def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def cluster_map_figure(image, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#101a29")
    ax.set_facecolor("#101a29")

    cmap = colormaps["tab10"].resampled(N_CLUSTERS + 1)
    bounds = np.arange(-0.5, N_CLUSTERS + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    shown = ax.imshow(
        image,
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    ax.set_title(title, color="#edf4fc", fontsize=12, pad=12)
    ax.set_xlabel("Pixel column", color="#8498af")
    ax.set_ylabel("Pixel row", color="#8498af")
    ax.tick_params(colors="#8498af", labelsize=8)

    colorbar = fig.colorbar(
        shown,
        ax=ax,
        ticks=range(N_CLUSTERS + 1),
        fraction=0.035,
        pad=0.03
    )
    colorbar.ax.set_yticklabels(
        ["Invalid"] + [f"Cluster {i}" for i in range(1, N_CLUSTERS + 1)]
    )
    colorbar.ax.tick_params(colors="#8498af", labelsize=8)

    fig.tight_layout()
    return fig


def cluster_error_figure(cluster_metrics):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#101a29")
    ax.set_facecolor("#101a29")

    x = cluster_metrics["Cluster"].astype(str)
    ax.bar(x, cluster_metrics["Spectral RMSE"], label="RMSE", alpha=0.9)
    ax.set_title(
        "Spectral reconstruction error by reference cluster",
        color="#edf4fc",
        pad=12
    )
    ax.set_xlabel("Reference spectral cluster", color="#8498af")
    ax.set_ylabel("RMSE", color="#8498af")
    ax.tick_params(colors="#8498af")
    ax.grid(axis="y", color="#29394c", alpha=0.5)

    for spine in ax.spines.values():
        spine.set_color("#29394c")

    fig.tight_layout()
    return fig


def confusion_matrix_figure(matrix):
    values = matrix.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#101a29")
    ax.set_facecolor("#101a29")

    im = ax.imshow(values, cmap="magma")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_xlabel("Reconstructed cluster", color="#8498af")
    ax.set_ylabel("Original cluster", color="#8498af")
    ax.set_title(
        "Pixel transition matrix",
        color="#edf4fc",
        pad=12
    )
    ax.tick_params(colors="#8498af")

    threshold = values.max() / 2 if values.size else 0

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col, row, f"{values[row, col]:,}",
                ha="center", va="center",
                color="white" if values[row, col] > threshold else "#edf4fc",
                fontsize=8
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ============================================================
# SUPER-RESOLUTION FUNCTIONS
# ============================================================

def super_resolution_status():
    required_files = [
        SR_METRICS_FILE,
        SR_LOSS_FILE,
        SR_STATS_FILE,
        SR_COMPARISON_FIGURE
    ]

    missing_files = [
        path for path in required_files if not path.exists()
    ]

    return len(missing_files) == 0, missing_files


def super_resolution_metric_chart(frame):
    """Plot held-out RMSE by band and super-resolution method."""
    band_col = find_column(frame, ["Band", "Spectral Band"])
    method_col = find_column(frame, ["Method", "Model"])
    rmse_col = find_column(frame, ["RMSE", "Image RMSE"])

    if not all([band_col, method_col, rmse_col]):
        return None

    plot_data = frame.copy()
    plot_data[rmse_col] = pd.to_numeric(
        plot_data[rmse_col], errors="coerce"
    )
    plot_data = plot_data.dropna(subset=[rmse_col])

    if plot_data.empty:
        return None

    pivot = plot_data.pivot_table(
        index=band_col,
        columns=method_col,
        values=rmse_col,
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#101a29")
    ax.set_facecolor("#101a29")

    pivot.plot(kind="bar", ax=ax, width=0.78)

    ax.set_title(
        "Held-out test RMSE by spectral band",
        color="#edf4fc",
        pad=14
    )
    ax.set_xlabel("Spectral band", color="#8498af")
    ax.set_ylabel("RMSE (lower is better)", color="#8498af")
    ax.tick_params(colors="#8498af")
    ax.grid(axis="y", color="#29394c", alpha=0.5)
    ax.legend(
        facecolor="#162538",
        edgecolor="#29394c",
        labelcolor="#edf4fc"
    )

    for spine in ax.spines.values():
        spine.set_color("#29394c")

    fig.tight_layout()
    return fig


def super_resolution_loss_chart(frame):
    """Plot training and validation loss when columns are available."""
    epoch_col = find_column(frame, ["Epoch"])
    train_col = find_column(
        frame,
        ["Training Loss", "Train Loss", "training_loss"]
    )
    val_col = find_column(
        frame,
        ["Validation Loss", "Val Loss", "validation_loss"]
    )

    if epoch_col is None:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#101a29")
    ax.set_facecolor("#101a29")

    plotted = False

    if train_col:
        ax.plot(
            frame[epoch_col],
            frame[train_col],
            label="Training loss",
            linewidth=2
        )
        plotted = True

    if val_col:
        ax.plot(
            frame[epoch_col],
            frame[val_col],
            label="Validation loss",
            linewidth=2
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(
        "Residual SRCNN training history",
        color="#edf4fc",
        pad=12
    )
    ax.set_xlabel("Epoch", color="#8498af")
    ax.set_ylabel("MSE loss", color="#8498af")
    ax.tick_params(colors="#8498af")
    ax.grid(color="#29394c", alpha=0.5)
    ax.legend(
        facecolor="#162538",
        edgecolor="#29394c",
        labelcolor="#edf4fc"
    )

    for spine in ax.spines.values():
        spine.set_color("#29394c")

    fig.tight_layout()
    return fig


def super_resolution_image_paths(band):
    return {
        "Native Reference": REFERENCE / f"{band}.tif",
        "Bicubic": BICUBIC_SR / f"{band}.tif",
        "Original SRCNN": ORIGINAL_SRCNN_SR / f"{band}.tif",
        "Residual SRCNN": RESIDUAL_SRCNN_SR / f"{band}.tif"
    }


# ============================================================
# VALIDATE CORE PROJECT OUTPUTS
# ============================================================

required = [COORDINATES_FILE, METRICS_FILE]

for band in BANDS:
    required += [
        REFERENCE / f"{band}.tif",
        DEGRADED / f"{band}.tif",
        RECONSTRUCTED / f"{band}.tif"
    ]

missing = [str(path) for path in required if not path.exists()]

if missing:
    st.error("Project output files are missing.")
    for file in missing:
        st.code(file)
    st.stop()

coordinates = load_table(str(COORDINATES_FILE))
metrics = load_table(str(METRICS_FILE))

# Cluster outputs are optional.
cluster_required = [
    CLUSTER_METRICS_FILE,
    CLUSTER_SUMMARY_FILE,
    CLUSTER_MATRIX_FILE,
    ORIGINAL_CLUSTER_MAP,
    RECON_CLUSTER_MAP
]

cluster_missing = [
    str(path) for path in cluster_required if not path.exists()
]

cluster_available = len(cluster_missing) == 0

if cluster_available:
    cluster_metrics = load_table(str(CLUSTER_METRICS_FILE))
    cluster_summary = load_table(str(CLUSTER_SUMMARY_FILE))
    cluster_matrix = load_matrix(str(CLUSTER_MATRIX_FILE))
else:
    cluster_metrics = None
    cluster_summary = None
    cluster_matrix = None

# Super-resolution outputs are optional so other pages remain usable.
sr_available, sr_missing = super_resolution_status()

if sr_available:
    sr_metrics = load_table(str(SR_METRICS_FILE))
    sr_loss = load_table(str(SR_LOSS_FILE))
    sr_stats = load_table(str(SR_STATS_FILE))
else:
    sr_metrics = None
    sr_loss = None
    sr_stats = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 25px 0;">
        <div class="terra-brand">TERRA<span style="color:#43d9d0">.</span></div>
        <div class="terra-sub">SATELLITE INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="eyebrow">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    navigation = [
        "Overview",
        "Image Laboratory",
        "Structure Analysis",
        "Spectral Clusters",
        "Super-Resolution",
        "Evaluation"
    ]

    page = st.radio(
        "Navigation",
        navigation,
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<div class="eyebrow">EXPERIMENT CONTROLS</div>',
        unsafe_allow_html=True
    )

    selected_band = st.selectbox(
        "Spectral band",
        BANDS,
        format_func=lambda value: value.upper()
    )

    selected_method = st.selectbox(
        "Projection method",
        ["PCA", "t-SNE", "UMAP"]
    )

    st.divider()

    st.markdown("""
    <div class="small-muted">
        EXPERIMENT<br><br>
        Structure Preservation<br>
        in Super-Resolved Imagery<br><br>
        DATA SOURCE<br>
        Sentinel-2 L2A<br><br>
        STUDY REGION<br>
        Chandigarh, India<br><br>
        GROUP<br>
        G2 | INT396
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="eyebrow">GEOSPATIAL RESEARCH PLATFORM / INT396</div>
<div class="hero-title">Satellite Structure<br>Intelligence.</div>
<p class="hero-description">
Investigating whether additional detail in reconstructed satellite
imagery preserves genuine spatial and spectral structure.
</p>
""", unsafe_allow_html=True)

st.markdown(
    '<span class="status-pill">● EXPERIMENT DATA LOADED</span>',
    unsafe_allow_html=True
)
st.write("")


# ============================================================
# GLOBAL METRICS
# ============================================================

neighbourhood = metric_value(
    metrics,
    "Raw pixel patch baseline",
    "Neighbourhood Overlap (%)"
)

variance = metric_value(
    metrics,
    "PCA",
    "Variance Explained (%)"
)

patch_count = int(coordinates["Patch"].nunique())

selected_rmse = metric_value(
    metrics,
    f"Image error - {selected_band}",
    "Image RMSE"
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        "NEIGHBOURHOOD OVERLAP",
        format_metric(neighbourhood, "%"),
        "Raw-pixel 10-NN baseline"
    )

with c2:
    metric_card(
        "PCA VARIANCE",
        format_metric(variance, "%"),
        "First two principal components"
    )

with c3:
    metric_card(
        "PATCH SAMPLES",
        f"{patch_count:,}",
        "Image patches analyzed"
    )

with c4:
    metric_card(
        f"{selected_band.upper()} RMSE",
        format_metric(selected_rmse),
        "Image reconstruction error"
    )

st.write("")
st.divider()


# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-heading">Research overview</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-caption">THE EXPERIMENT AT A GLANCE</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.45, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Resolution transformation**")

        a, b, c = st.columns(3)
        a.metric("REFERENCE", "10 m")
        b.metric("DEGRADED", "30 m")
        c.metric("RECONSTRUCTED", "10 m")

        st.markdown("""
        <p class="small-muted">
        A controlled experiment using Sentinel-2 imagery.
        The original 10m image is aggregated into 30m pixels
        and then reconstructed to a 10m grid.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Analysis methods**")
        st.markdown("""
        - Principal Component Analysis
        - t-distributed Stochastic Neighbour Embedding
        - Uniform Manifold Approximation and Projection
        - K-Means spectral clustering
        - Residual Super-Resolution CNN
        """)
        st.markdown(
            '<p class="small-muted">Four spectral bands · '
            '100 features per image patch · Six spectral clusters</p>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("**Current image preview**")

    preview = load_raster(str(REFERENCE / f"{selected_band}.tif"))
    st.pyplot(
        image_figure(
            preview,
            f"REFERENCE IMAGE | {selected_band.upper()}"
        ),
        use_container_width=True
    )
    plt.close("all")

    if cluster_available:
        st.success("Spectral clustering outputs are available.")
    else:
        st.info(
            "Spectral clustering dashboard section is waiting for its "
            "analysis files."
        )

    if sr_available:
        st.success("Super-resolution experiment outputs are available.")
    else:
        st.info(
            "Super-resolution outputs are not yet fully available. "
            "The Super-Resolution page will show which files are missing."
        )


# ============================================================
# PAGE: IMAGE LABORATORY
# ============================================================

elif page == "Image Laboratory":

    st.markdown("### Image laboratory")
    st.markdown(
        '<div class="section-caption">MULTI-RESOLUTION IMAGE COMPARISON</div>',
        unsafe_allow_html=True
    )

    reference = load_raster(str(REFERENCE / f"{selected_band}.tif"))
    degraded = load_raster(str(DEGRADED / f"{selected_band}.tif"))
    reconstructed = load_raster(
        str(RECONSTRUCTED / f"{selected_band}.tif")
    )

    reference_valid = reference[np.isfinite(reference)]

    if reference_valid.size == 0:
        st.error("Original image contains no valid pixels.")
        st.stop()

    shared_vmin, shared_vmax = np.percentile(
        reference_valid, [2, 98]
    )

    if shared_vmin == shared_vmax:
        shared_vmax = shared_vmin + 1e-6

    tab1, tab2 = st.tabs([
        "Three-stage comparison",
        "Reconstruction difference"
    ])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**01 / ORIGINAL**")
            st.caption("10 metre reference")
            st.pyplot(
                image_figure(
                    reference, "Original 10m",
                    vmin=shared_vmin, vmax=shared_vmax
                ),
                use_container_width=True
            )

        with col2:
            st.markdown("**02 / DEGRADED**")
            st.caption("30 metre synthetic image")
            st.pyplot(
                image_figure(
                    degraded, "Synthetic 30m",
                    vmin=shared_vmin, vmax=shared_vmax
                ),
                use_container_width=True
            )

        with col3:
            st.markdown("**03 / RECONSTRUCTED**")
            st.caption("10 metre interpolated image")
            st.pyplot(
                image_figure(
                    reconstructed, "Reconstructed 10m",
                    vmin=shared_vmin, vmax=shared_vmax
                ),
                use_container_width=True
            )

        st.caption(
            "All images use the same contrast range derived from the "
            "original image. Upsampling restores the grid size, not "
            "the spatial detail lost during degradation."
        )

        plt.close("all")

    with tab2:
        height = min(reference.shape[0], reconstructed.shape[0])
        width = min(reference.shape[1], reconstructed.shape[1])

        original = reference[:height, :width]
        restored = reconstructed[:height, :width]
        difference = np.abs(original - restored)

        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor("#101a29")
        ax.set_facecolor("#101a29")

        im = ax.imshow(difference, cmap="magma")
        ax.set_title(
            f"Absolute reconstruction difference | {selected_band.upper()}",
            color="#edf4fc"
        )
        ax.tick_params(colors="#8498af")
        fig.colorbar(im, ax=ax, label="Absolute reflectance error")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Brighter areas indicate larger pixel-wise differences "
            "between the original and reconstructed images."
        )


# ============================================================
# PAGE: STRUCTURE ANALYSIS
# ============================================================

elif page == "Structure Analysis":

    st.markdown("### Structure analysis")
    st.markdown(
        '<div class="section-caption">LOW-DIMENSIONAL PATCH REPRESENTATIONS</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"**Selected method: {selected_method}**")

    method_data = coordinates[
        coordinates["Method"] == selected_method
    ]

    if method_data.empty:
        st.warning("No projection data found for this method.")
    else:
        st.pyplot(
            projection_figure(method_data, selected_method),
            use_container_width=True
        )
        plt.close("all")

        st.markdown("""
        <div class="panel">
        <b>How to read this projection</b><br><br>
        Each point represents an image patch. Points that appear
        close together have similar projected representations.
        Compare the original and reconstructed distributions
        to investigate changes in local structure.
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        method_row = metrics[
            metrics["Method"] == selected_method
        ]

        if not method_row.empty:
            row = method_row.iloc[0]
            left, right = st.columns(2)

            if "Reference Trustworthiness" in row.index:
                left.metric(
                    "Original trustworthiness",
                    format_metric(
                        row["Reference Trustworthiness"] * 100,
                        "%"
                    )
                )

            if "Reconstructed Trustworthiness" in row.index:
                right.metric(
                    "Reconstructed trustworthiness",
                    format_metric(
                        row["Reconstructed Trustworthiness"] * 100,
                        "%"
                    )
                )


# ============================================================
# PAGE: SPECTRAL CLUSTERS
# ============================================================

elif page == "Spectral Clusters":

    st.markdown("### Spectral structure comparison")
    st.markdown(
        '<div class="section-caption">'
        'K-MEANS CLUSTERING · ORIGINAL VS RECONSTRUCTED IMAGERY'
        '</div>',
        unsafe_allow_html=True
    )

    if not cluster_available:
        st.warning(
            "Cluster analysis files are not available yet. "
            "Run the cluster analysis script first."
        )
        st.write("Missing files:")
        for file in cluster_missing:
            st.code(file)
        st.stop()

    summary_row = cluster_summary.iloc[0]

    agreement = float(
        summary_row["Overall same-label agreement (%)"]
    )
    mean_iou = float(summary_row["Mean cluster IoU"])
    mean_cluster_rmse = float(
        summary_row["Mean cluster spectral RMSE"]
    )

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "SAME-LABEL AGREEMENT",
            f"{agreement:.2f}%"
        )

    with b:
        st.metric(
            "MEAN CLUSTER IoU",
            f"{mean_iou:.4f}"
        )

    with c:
        st.metric(
            "MEAN CLUSTER RMSE",
            f"{mean_cluster_rmse:.5f}"
        )

    st.caption(
        "Agreement compares cluster IDs assigned by the same K-Means "
        "model to corresponding pixels. It is not classification accuracy."
    )

    st.divider()

    original_clusters = load_raster(str(ORIGINAL_CLUSTER_MAP))
    reconstructed_clusters = load_raster(str(RECON_CLUSTER_MAP))

    map1, map2 = st.columns(2)

    with map1:
        st.markdown("**ORIGINAL REFERENCE — 10m**")
        st.pyplot(
            cluster_map_figure(
                original_clusters,
                "Original spectral cluster map"
            ),
            use_container_width=True
        )

    with map2:
        st.markdown("**RECONSTRUCTED — 10m GRID**")
        st.pyplot(
            cluster_map_figure(
                reconstructed_clusters,
                "Reconstructed spectral cluster map"
            ),
            use_container_width=True
        )

    plt.close("all")

    st.divider()

    st.markdown("**Cluster-wise reconstruction error**")

    highest_error = cluster_metrics.loc[
        cluster_metrics["Spectral RMSE"].idxmax()
    ]

    st.info(
        f"Cluster {int(highest_error['Cluster'])} has the highest "
        f"spectral RMSE ({highest_error['Spectral RMSE']:.5f}) "
        "among the six reference spectral clusters."
    )

    st.pyplot(
        cluster_error_figure(cluster_metrics),
        use_container_width=True
    )
    plt.close("all")

    st.markdown("**Detailed cluster metrics**")
    st.dataframe(
        cluster_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("**Cluster transition matrix**")
    st.caption(
        "Rows represent original cluster IDs; columns represent "
        "reconstructed cluster IDs. Values count corresponding pixels."
    )

    st.pyplot(
        confusion_matrix_figure(cluster_matrix),
        use_container_width=True
    )
    plt.close("all")

    st.dataframe(
        cluster_matrix,
        use_container_width=True
    )

    st.warning(
        "These are unsupervised spectral clusters, not verified land-cover "
        "classes. Without ground-truth labels, do not interpret cluster IDs "
        "as confirmed vegetation, water, buildings, or soil categories."
    )


# ============================================================
# PAGE: SUPER-RESOLUTION
# ============================================================

elif page == "Super-Resolution":

    st.markdown("### Super-resolution experiment")
    st.markdown(
        '<div class="section-caption">'
        'BICUBIC BASELINE VS ORIGINAL SRCNN VS RESIDUAL SRCNN'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    This experiment compares interpolation with two learned
    super-resolution approaches. All quantitative metrics below
    are calculated on the held-out spatial test region.
    """)

    if not sr_available:
        st.warning(
            "Some super-resolution result files are missing. "
            "The other dashboard pages remain available."
        )

        for file in sr_missing:
            st.code(str(file))

        st.stop()

    # --------------------------------------------------------
    # RESULTS SUMMARY
    # --------------------------------------------------------

    band_col = find_column(sr_metrics, ["Band", "Spectral Band"])
    method_col = find_column(sr_metrics, ["Method", "Model"])
    rmse_col = find_column(sr_metrics, ["RMSE", "Image RMSE"])
    mae_col = find_column(sr_metrics, ["MAE", "Image MAE"])
    psnr_col = find_column(sr_metrics, ["PSNR", "PSNR (dB)", "PSNR dB"])

    if not all([band_col, method_col, rmse_col]):
        st.error(
            "The super-resolution metrics CSV does not contain "
            "recognizable Band, Method and RMSE columns."
        )
        st.write("Detected columns:", list(sr_metrics.columns))
        st.stop()

    st.markdown("**Held-out test performance**")
    st.caption(
        "Lower RMSE and MAE are better. Higher PSNR is better."
    )

    display_columns = [
        col for col in [
            band_col, method_col, rmse_col, mae_col, psnr_col
        ] if col is not None
    ]

    display_metrics = sr_metrics[display_columns].copy()
    st.dataframe(
        display_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    # --------------------------------------------------------
    # RMSE CHART
    # --------------------------------------------------------

    st.markdown("**RMSE comparison across spectral bands**")

    rmse_figure = super_resolution_metric_chart(sr_metrics)

    if rmse_figure is not None:
        st.pyplot(rmse_figure, use_container_width=True)
        plt.close(rmse_figure)
    else:
        st.info(
            "The RMSE chart could not be generated from the available "
            "CSV column structure. The metric table is still available."
        )

    st.divider()

    # --------------------------------------------------------
    # IMAGE COMPARISON
    # --------------------------------------------------------

    st.markdown("**Visual comparison**")

    image_paths = super_resolution_image_paths(selected_band)

    existing_images = {
        name: path for name, path in image_paths.items()
        if path.exists()
    }

    if not existing_images:
        st.warning(
            f"No super-resolution raster outputs were found for "
            f"{selected_band.upper()}."
        )
    else:
        reference_path = image_paths["Native Reference"]

        if reference_path.exists():
            native_image = load_raster(str(reference_path))
            native_valid = native_image[np.isfinite(native_image)]

            if native_valid.size:
                visual_vmin, visual_vmax = np.percentile(
                    native_valid, [2, 98]
                )
            else:
                visual_vmin, visual_vmax = None, None
        else:
            visual_vmin, visual_vmax = None, None

        image_columns = st.columns(len(existing_images))

        for column, (name, path) in zip(
            image_columns, existing_images.items()
        ):
            with column:
                st.markdown(f"**{name}**")
                image = load_raster(str(path))

                st.pyplot(
                    image_figure(
                        image,
                        f"{name} | {selected_band.upper()}",
                        vmin=visual_vmin,
                        vmax=visual_vmax
                    ),
                    use_container_width=True
                )
                plt.close("all")

        st.caption(
            "The same reference-derived display range is used for "
            "available images. The residual SRCNN is evaluated against "
            "the synthetic native-resolution reference."
        )

    if SR_COMPARISON_FIGURE.exists():
        st.markdown("**Experiment comparison figure**")
        st.image(
            str(SR_COMPARISON_FIGURE),
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    st.markdown("**Residual SRCNN training history**")

    loss_figure = super_resolution_loss_chart(sr_loss)

    if loss_figure is not None:
        st.pyplot(loss_figure, use_container_width=True)
        plt.close(loss_figure)
    elif SR_LOSS_FIGURE.exists():
        st.image(
            str(SR_LOSS_FIGURE),
            use_container_width=True
        )
    else:
        st.info(
            "Training loss data was found, but its column names "
            "could not be used to create a chart."
        )

    st.caption(
        "Training and validation loss show how the model's prediction "
        "error changed over training epochs. Validation loss is used "
        "to monitor performance on held-out training patches."
    )

    st.divider()

    # --------------------------------------------------------
    # SPECTRAL STATISTICS
    # --------------------------------------------------------

    st.markdown("**Spectral statistics**")
    st.caption(
        "Compare the distribution of reflectance values across "
        "the native reference and reconstructed outputs."
    )

    st.dataframe(
        sr_stats,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.markdown("**How to interpret this experiment**")

    st.markdown("""
    - **Bicubic:** a conventional interpolation baseline. It enlarges
      the image grid but cannot recover information removed by
      downsampling.
    - **Original SRCNN:** the first learned convolutional
      super-resolution model.
    - **Residual SRCNN:** learns corrections to the bicubic image
      rather than generating the entire image from scratch.
    """)

    st.info(
        "In the current held-out experiment, Residual SRCNN achieved "
        "lower RMSE than bicubic interpolation in all four spectral "
        "bands. This is a modest improvement in a synthetic degradation "
        "experiment, not proof that real-world missing satellite detail "
        "has been recovered."
    )

    st.warning(
        "The native 10m image is the reference used to create synthetic "
        "30m observations. Therefore, this experiment evaluates "
        "reconstruction under controlled degradation. It does not "
        "establish the same performance on independently acquired "
        "30m satellite imagery."
    )


# ============================================================
# PAGE: EVALUATION
# ============================================================

elif page == "Evaluation":

    st.markdown("### Experimental evaluation")
    st.markdown(
        '<div class="section-caption">'
        'QUANTITATIVE PERFORMANCE AND VALIDATION'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("**All dimensionality-reduction evaluation metrics**")
    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    st.write("")
    st.markdown("**Image reconstruction errors**")

    error_rows = metrics[
        metrics["Method"].astype(str).str.startswith("Image error")
    ]

    if not error_rows.empty:
        error_columns = [
            col for col in [
                "Method",
                "Image RMSE",
                "Image MAE",
                "Valid Pixels"
            ]
            if col in error_rows.columns
        ]

        st.dataframe(
            error_rows[error_columns],
            use_container_width=True,
            hide_index=True
        )

    st.write("")
    st.markdown("**Projection quality comparison**")

    comparison_methods = [
        "Raw pixel patch baseline", "PCA", "t-SNE", "UMAP"
    ]

    comparison_columns = [
        "Method",
        "Neighbourhood Overlap (%)",
        "2D Projection Neighbourhood Overlap (%)",
        "Reference Trustworthiness",
        "Reconstructed Trustworthiness",
        "Variance Explained (%)",
        "2D Feature Reconstruction RMSE"
    ]

    available_comparison_columns = [
        column for column in comparison_columns
        if column in metrics.columns
    ]

    comparison_rows = metrics[
        metrics["Method"].isin(comparison_methods)
    ].copy()

    comparison_rows = comparison_rows[
        available_comparison_columns
    ]

    st.dataframe(
        comparison_rows,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The raw-pixel baseline is reported in 'Neighbourhood Overlap'. "
        "PCA, t-SNE and UMAP are compared using '2D Projection "
        "Neighbourhood Overlap'. Trustworthiness is shown as a separate "
        "embedding-quality measure; it is not the same as overlap. "
        "PCA variance explained and 2D feature reconstruction RMSE "
        "are PCA-specific measures."
    )

    st.divider()
    st.markdown("**Spectral clustering evaluation**")

    if cluster_available:
        st.dataframe(
            cluster_summary,
            use_container_width=True,
            hide_index=True
        )

        st.dataframe(
            cluster_metrics,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "Spectral clustering evaluation will appear after its output "
            "files are generated."
        )

    st.divider()
    st.markdown("**Super-resolution evaluation**")

    if sr_available:
        st.dataframe(
            sr_metrics,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "Super-resolution evaluation will appear after its output "
            "files are generated."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px 0 5px 0;">
    <div style="font-size:13px; letter-spacing:3px;
                font-weight:700; color:#dce9f7;">
        TERRA<span style="color:#43d9d0">.</span>
    </div>
    <div style="font-size:10px; color:#71869e;
                letter-spacing:1.5px; margin-top:7px;">
        SATELLITE INTELLIGENCE · INT396 · GROUP G2
    </div>
</div>
""", unsafe_allow_html=True)
