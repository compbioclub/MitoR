# MITOR: predicting T-cell mitochondria hijacking from tumor single-cell RNA sequencing data

**MITOR** is a Python tool for single-cell RNA-seq data analysis, providing an integrated pipeline for data initialization, enMT profile prediction, cell type prediction, and result visualization.



## Overview

![Figure1](D:\MITOR_standard\Figure1.png)

## 📥 Installation

**From GitHub (recommended for the latest version)**

```
pip install git+https://github.com/<username>/<repo>.git
```



## 🚀 Quick Start

### 1️⃣ Running the MITOR Pipeline

The following is an example of how to use MITOR in a complete workflow:

```python
from mitor import init_pipeline, run_enMT_profile, run_predict, plot_result, plot_info

# File paths
adata_path = "demo/scRNAseq_bench2.h5ad"
mixed_cell_names_path = "demo/mixed_cell_names.csv"

# Initialize pipeline
predictor_RNA, ref_adata_dict, mixed_cell_list, adata_RNA = init_pipeline(adata_path, mixed_cell_names_path)

# Run enMT profile prediction
run_enMT_profile(predictor_RNA, ref_adata_dict)

# Run cell prediction
run_predict(predictor_RNA, adata_RNA, mixed_cell_list)

print("✅ MITOR pipeline completed!")

```

#### 📌Explanation

1. **`init_pipeline(adata_path, mixed_cell_names_path)`**
   Loads reference data and initializes single-cell dataset (`adata_RNA`) , reference data dict (`ref_adata_dict`) and the predictor model (`predictor_RNA`).
2. **`run_enMT_profile(predictor_RNA, ref_adata_dict)`**
   Runs the enMT-based profile prediction using the reference data.
3. **`run_predict(predictor_RNA, adata_RNA, mixed_cell_list)`**
   Predicts cell types for the mixed samples.

### 2️⃣ Obtaining Prediction Results

After running the main pipeline steps, you can extract key prediction results using **`get_result`** and **`get_fraction`**:

```python
from mitor.pipeline import get_result, get_fraction

# Extract predicted labels for mixed cells
mixture_labels = adata_RNA[adata_RNA.obs["cell_name"].isin(mixed_cell_list)].obs.MTtransfer.apply(lambda x: x.lower())

# Retrieve performance metrics
metrics = get_result(predictor_RNA, mixture_labels)

# Retrieve fraction data for each cell
fractions = get_fraction(predictor_RNA)

```

#### 📌 Explanation:

- **`get_result(predictor_RNA, mixture_labels, feature="Prob")`** *(default)*:Uses Prob-based strategies.
- **`get_result(predictor_RNA, mixture_labels, feature="Rank")`**: Uses Rank-based strategies.
- **`get_result(predictor_RNA, mixture_labels, feature="Consensus(Interaction)")`**: Uses Consensus(Interaction)-based strategies
- **`get_result(predictor_RNA, mixture_labels, feature="Consensus(Union)")`**: Uses Consensus(Union)-based strategies.
- **`get_fraction(predictor_RNA)`**: Returns T cell and cancer cell fractions of each mixed cell.
- **`output path`**: all file save at **output/**.

​	

### 3️⃣ Visualizing Results

MITOR provides built-in visualization tools:

```
from mitor.pipeline import plot_result, plot_info

# Visualization of AUROC and AUPRC of Prob-base and Rank-base
plot_result(predictor_RNA, mixture_labels)

# plot key : mixture cells, U/B values, and F matrix
plot_info(predictor_RNA, plots=["mixture_cell"])
```





