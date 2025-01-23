# MitoR: Predicting T-cell Mitochondria Hijacking from Tumor Single-cell RNA Sequencing Data

## Overview

![Figure1](./figure/Figure1.png)

The schematic overview of MitoR.

+ A. The Poisson-Gamma model for fitting profiles of T cell and cancer cell endogenous mitochondria (enMT). 
+ B. The Poisson-Gamma mixture model for predicting each cell's T cell and cancer cell fraction proportions. 
+ C. Four strategies with specific cutoffs to identify hijacker and non-hijacker cancer cells.

## 📥 Installation





## 🚀 Quick Start

### 1️⃣ Running the MITOR Pipeline

The following is an example of how to use MITOR in a complete workflow:

```python
from mitor.pipeline import init_pipeline, run_enMT_profile, run_predict, plot_result, plot_info

# File paths
adata_path = "./demo/scRNAseq_bench2.h5ad"
mixed_cell_names_path = "./demo/mixed_cell_names.csv"

# Initialize pipeline
predictor_RNA, ref_adata_dict, mixed_cell_list, adata_RNA = init_pipeline(adata_path, mixed_cell_names_path)

# Run enMT profile prediction
run_enMT_profile(predictor_RNA, ref_adata_dict)

# Run cell prediction
run_predict(predictor_RNA, adata_RNA, mixed_cell_list)

print("✅ MITOR pipeline completed!")

```



### 2️⃣ Obtaining Prediction Results

After running the pipeline, you can extract key prediction results:

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

```python
from mitor.pipeline import plot_result, plot_info

# Visualization of AUROC and AUPRC of Prob-base and Rank-base
plot_result(predictor_RNA, mixture_labels)

# plot key : mixture cells, U/B values, and F matrix
plot_info(predictor_RNA, plots=["mixture_cell"])
```





