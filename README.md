# Satellite Structure Preservation in Super-Resolved Imagery

**An Unsupervised Learning Research Project | Satellite Image Analysis | Dimensionality Reduction | Super-Resolution**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Research-Unsupervised%20Learning-purple" alt="Research"/>
</p>

## Overview

Satellite imagery provides valuable information for environmental monitoring, land-cover analysis, and geospatial intelligence. However, increasing spatial resolution through image reconstruction does not necessarily guarantee the preservation of meaningful spatial structures.

This project investigates **whether spatial relationships and structural patterns are preserved when medium-resolution satellite imagery is reconstructed at a higher resolution.**

Using Sentinel-2 and Landsat imagery, the project combines unsupervised learning, dimensionality reduction, clustering, and deep-learning-based super-resolution to evaluate structural consistency and reconstruction quality.

The system also provides an interactive Streamlit dashboard for exploring satellite imagery, dimensionality reduction projections, cluster distributions, and experimental results.

## Research Objectives

- Investigate spatial structure preservation in reconstructed satellite imagery.
- Apply PCA, t-SNE, and UMAP to high-dimensional multispectral image patches.
- Evaluate neighbourhood preservation and embedding trustworthiness.
- Analyze changes in spectral clustering structure after reconstruction.
- Compare bicubic interpolation, SRCNN, and residual SRCNN.
- Quantitatively evaluate reconstruction quality using RMSE, MAE, and PSNR.
- Develop an interactive satellite imagery analysis dashboard.

## Dataset

The project uses multispectral satellite imagery from:

| Satellite | Data source | Spectral bands |
|---|---|---|
| Sentinel-2 | Sentinel-2 Level-2A | Blue, Green, Red, NIR |
| Landsat 9 | Landsat Collection 2 Level-2 | Blue, Green, Red, NIR |

**Study region:** Chandigarh, India.

The selected scenes were acquired on November 5–6, 2025.

The imagery was processed into standardized reflectance datasets and used for subsequent experiments.

*Note: The selected Sentinel-2 and Landsat scenes have different acquisition dates and native spatial resolutions.*

## Methodology

### 1. Data Acquisition and Preprocessing

- Satellite imagery retrieved using Microsoft Planetary Computer STAC.
- Multispectral bands extracted and standardized.
- Reflectance scaling and nodata handling.
- Common spatial reference and raster alignment.
- Supplementary quality assessment using Sentinel-2 SCL and Landsat QA_PIXEL.

### 2. Synthetic Resolution Degradation

A controlled degradation experiment was performed using Sentinel-2 imagery.

- Native 10 m imagery was retained as the reference.
- Spatial information was reduced through 3 × 3 block aggregation.
- The resulting synthetic 30 m imagery was reconstructed at 10 m resolution using interpolation.

This provides a controlled reference for comparing reconstruction methods.

### 3. Dimensionality Reduction

Multispectral image patches were represented as high-dimensional feature vectors.

Each patch contains:

- 5 × 5 spatial pixels.
- 4 spectral bands.
- 100 features per patch.

The following dimensionality reduction techniques were evaluated:

| Technique | Purpose |
|---|---|
| PCA | Linear dimensionality reduction and variance analysis |
| t-SNE | Nonlinear visualization of local neighbourhood relationships |
| UMAP | Nonlinear manifold learning and structural visualization |

Evaluation included:

- Neighbourhood overlap.
- Trustworthiness.
- Explained variance.
- Feature reconstruction RMSE.
- Image reconstruction RMSE and MAE.

### 4. Unsupervised Clustering

K-Means clustering was applied to multispectral reference imagery.

The learned cluster assignments were compared against reconstructed imagery using the same reference-derived cluster centroids.

Evaluation metrics:

- Silhouette Score.
- Davies–Bouldin Index.
- Adjusted Rand Index (ARI).
- Normalized Mutual Information (NMI).
- Cluster intersection over union (IoU).

These measurements evaluate cluster structure and agreement, not ground-truth land-cover classification accuracy.

### 5. Super-Resolution

Three reconstruction approaches were compared:

| Method | Description |
|---|---|
| Bicubic | Classical interpolation baseline |
| SRCNN | Convolutional neural network for image reconstruction |
| Residual SRCNN | Residual-learning variant initialized to preserve the bicubic input |

The residual SRCNN uses a convolutional architecture with three feature-extraction stages and learns corrections relative to the bicubic input.

Models were trained using synthetic low-resolution and reference image patches.

Evaluation used a spatially held-out test region and included:

- Root Mean Squared Error (RMSE).
- Mean Absolute Error (MAE).
- Peak Signal-to-Noise Ratio (PSNR).

## Experimental Results

### Dimensionality Reduction

| Method | Neighbourhood Overlap |
|---|---:|
| Raw pixel patch baseline | 29.21% |
| PCA | 23.32% |
| t-SNE | 48.95% |
| UMAP | 39.49% |

Additional PCA results:

- Explained variance (first two components): **58.38%**
- 2D feature reconstruction RMSE: **0.6452**

### Embedding Trustworthiness

| Method | Reference | Reconstructed |
|---|---:|---:|
| PCA | 0.8258 | 0.9201 |
| t-SNE | 0.9270 | 0.9796 |
| UMAP | 0.8990 | 0.9547 |

### Clustering Evaluation

| Metric | Reference | Reconstructed |
|---|---:|---:|
| Silhouette Score | 0.3344 | 0.3288 |
| Davies–Bouldin Index | 0.8925 | 0.9023 |

Additional agreement metrics:

- ARI: **0.2478**
- NMI: **0.3298**

The reconstructed imagery shows a slight reduction in cluster separation and limited agreement with reference cluster assignments.

### Super-Resolution Comparison

Held-out test results:

| Band | Bicubic RMSE | SRCNN RMSE | Residual SRCNN RMSE |
|---|---:|---:|---:|
| Blue | 0.01784 | 0.02249 | **0.01706** |
| Green | 0.01836 | 0.02403 | **0.01757** |
| Red | 0.02114 | 0.02739 | **0.02009** |
| NIR | 0.03156 | 0.05005 | **0.03050** |

Residual SRCNN achieved lower RMSE than bicubic interpolation across all four spectral bands, with an average improvement of approximately **4.3%**.

This demonstrates improved reconstruction performance on the synthetic held-out test region. It does not establish recovery of genuine missing spatial detail in real-world satellite imagery.

## Interactive Dashboard

A Streamlit application was developed to explore the experiments and visualize results.

**Dashboard modules:**

1. Overview
2. Image Laboratory
3. Structure Analysis
4. Spectral Clusters
5. Evaluation
6. Super-Resolution

The dashboard provides image comparisons, projection visualizations, clustering results, and reconstruction metrics.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Machine Learning | Scikit-learn |
| Dimensionality Reduction | PCA, t-SNE, UMAP |
| Image Processing | Rasterio, NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Satellite Data | Sentinel-2, Landsat 9 |
| Data Access | Microsoft Planetary Computer, STAC |

## Project Structure

```text
Satellite-Structure-Preservation/
│
├── app/
│   └── hello.py
│
├── src/
│   ├── data.py
│   ├── preprocess.py
│   ├── degrade.py
│   ├── dimensionality_reduction.py
│   ├── cluster_analysis.py
│   ├── super_resolution.py
│   ├── apply_quality_masks.py
│   ├── download_quality_masks.py
│   └── diagnose_super_resolution.py
│
├── results/
│   ├── dimensionality_reduction/
│   ├── landcover_analysis/
│   ├── quality_mask/
│   └── super_resolution/
│       ├── model/
│       └── residual_experiment/
│
├── check_data_quality.py
├── requirements.txt
└── .gitignore
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/mythilyparise/Satellite-Structure-Preservation.git

cd Satellite-Structure-Preservation
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
python -m streamlit run app/hello.py
```

The application will open in your browser.

**Data availability:** Raw and processed satellite rasters are excluded from this repository to avoid uploading large datasets. The complete experiments require the corresponding satellite data and generated intermediate files. The committed results folder contains the principal evaluation outputs and visualizations.

## Limitations

- Super-resolution experiments use synthetically degraded imagery rather than independently acquired high-resolution ground truth.
- The SRCNN results are specific to the selected study region and experimental setup.
- Sentinel-2 and Landsat observations were acquired on different dates.
- Cluster assignments are unsupervised and are not verified land-cover labels.
- Neighbourhood preservation metrics were evaluated for the dimensionality reduction experiments, not separately for every super-resolution method.
- No cloud or defective pixels were flagged in the classified quality-mask coverage; some Sentinel-2 pixels had no SCL coverage.

## Future Work

- Evaluate neighbourhood preservation independently across bicubic and learned super-resolution outputs.
- Explore additional super-resolution architectures.
- Validate using independent high-resolution satellite imagery.
- Expand experiments across multiple geographical regions and acquisition dates.
- Investigate land-cover-specific structural degradation.
- Improve the dashboard with additional geospatial analysis capabilities.

## Research Context

**Project:** Structure Preservation in Super-Resolved Satellite Imagery  
**Course:** Unsupervised Learning  
**Institution:** Lovely Professional University  
**Group:** G2

## Author

**Mythily Parise**  
B.Tech Computer Science and Engineering

GitHub: [@mythilyparise](https://github.com/mythilyparise)

---

*This repository contains the implementation, experimental outputs, and visualization dashboard for an academic research project on unsupervised learning and structural preservation in satellite imagery.*
