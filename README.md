<div align="center">

# Prether

**Protein Thermal Adaptation Classifier**

*A dual-backbone fusion model for classifying proteins into mesophiles, thermophiles, and psychrophiles.*

</div>

---

<details open>
<summary><b>Contents</b></summary>

- [Overview](#-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Performance](#-performance)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
  - [Download Backbone Models](#1-download-backbone-models-required)
  - [Train](#2-train)
  - [Test](#3-test)
  - [Predict](#4-predict)
- [Released Weights](#-released-weights)
- [Notes](#-notes)

</details>

---

## Overview

---

## Dataset

The dataset used in this study consists of protein sequences annotated with thermal adaptation classes.

Each sequence is assigned to one of three classes:

| Label | Class |
|---|---|
| 0 | Mesophile |
| 1 | Thermophile |
| 2 | Psychrophile |

The dataset was constructed based on previously published protein temperature adaptation datasets. After quality filtering and sequence redundancy reduction, the data were divided into training and independent test sets.

The processed datasets used in this study are provided:
```
data/
├── train.csv
└── test.csv
```

The original dataset was obtained from:

Pei H., Li X., et al. Identification of Thermophilic Proteins Based on Sequence-Based Bidirectional Representations from Transformer-Embedding Features. *Applied Sciences*, 2023, 13(5): 2858.

**Prether** predicts a protein's thermal adaptation class from its amino acid sequence:

| Label | Class | Description |
|:-----:|:------|:------------|
| `0` | **Mesophile**　嗜温 | Optimal growth at moderate temperatures (~20–45°C) |
| `1` | **Thermophilic**　嗜热 | Optimal growth at high temperatures (~45–80°C) |
| `2` | **Psychrophilic**　嗜冷 | Optimal growth at low temperatures (~-15-20°C) |

### Architecture

Prether combines **two frozen protein language models** with a **trainable BiGRU fusion head**:

| Component | Model | Hidden Dim |
|:----------|:------|:----------:|
| Backbone 1 | [`ProtT5-XL-UniRef50`](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) | 1024 |
| Backbone 2 | [`ProPrime_650M_OGT_Prediction`](https://huggingface.co/AI4Protein/ProPrime_650M_OGT_Prediction) | 1280 |
| Fusion Head | BiGRU → Attention Pooling → MLP | 512 |

> **Why two models?** ProtT5 provides general-purpose protein representations, while ProPrime is specifically pre-trained for optimal growth temperature (OGT) prediction. Their token-level hidden states are fused via BiGRU to capture complementary signals.

---

## Installation

### Requirements

- **Python** ≥ 3.9
- **CUDA-capable GPU** recommended (CPU supported but slow)
- **Disk** ~8 GB for backbone models

```bash
git clone https://github.com/your-org/Prether.git
cd Prether

# Install dependencies
pip install -r requirements.txt

# Download backbone models (~7.5 GB)
# (If you're in a region with limited Hugging Face access, use the mirror:)
#   Windows:  set HF_ENDPOINT=https://hf-mirror.com && python scripts/download_backbones.py
#   Linux:    HF_ENDPOINT=https://hf-mirror.com python scripts/download_backbones.py
python scripts/download_backbones.py
```

---

## Quick Start

Predict thermal classes for your own sequences in 3 steps:

```bash
# 1. Prepare input (CSV with columns: id,seq)
#    Linux: head examples/example_test.csv
#    Windows: Get-Content examples/example_test.csv -Head 5

# 2. Run prediction
python predict/predict.py \
  --input_path examples/example_test.csv \
  --input_type csv \
  --ckpt_dir train/fusion_out \
  --out_dir my_predictions

# 3. View results
#    Linux: cat my_predictions/predictions_ensemble.csv
#    Windows: Get-Content my_predictions/predictions_ensemble.csv
```

> Supports **CSV** and **FASTA** input. See [Predict](#4-predict) for full options.

---

## Performance

5-fold cross-validation mean results on the independent test set:

| Metric | Score | Metric | Score |
|:---|---:|---:|---:|
| **Accuracy (ACC)** | **0.9492** | Precision<sub>macro</sub> | 0.9498 |
| **MCC** | **0.9248** | AUROC<sub>macro</sub> | 0.9917 |
| Sensitivity<sub>macro</sub> | 0.9482 | AUROC<sub>micro</sub> | 0.9931 |
| Specificity<sub>macro</sub> | 0.9750 | AUPRC<sub>macro</sub> | 0.9803 |
| F1<sub>macro</sub> | 0.9479 | AUPRC<sub>micro</sub> | 0.9873 |

| Per-class F1 | Score |
|:---|:---:|
| Mesophile (嗜温) | **0.9588** |
| Thermophilic (嗜热) | **0.9431** |
| Psychrophilic (嗜冷) | **0.9418** |

> Visual artifacts (ROC curves, confusion matrix) are available in [`results/`](results/).

---

## 📖 Usage

### 1. Download Backbone Models (Required)

```bash
python scripts/download_backbones.py
```

This creates two local directories:

| Directory | Source | Size |
|:---|:---|---:|
| `prot_t5_xl_uniref50/` | `Rostlab/prot_t5_xl_uniref50` | ~5 GB |
| `ProPrime_650M_OGT_Prediction/` | `AI4Protein/ProPrime_650M_OGT_Prediction` | ~2.5 GB |

> ⚠️ These directories are **not** included in the repository due to size limits.

### 2. Train

Train the fusion head with 5-fold cross-validation:

```bash
python train/train.py \
  --train_csv data/train.csv \
  --prott5_model_path prot_t5_xl_uniref50 \
  --proprime_model_path ProPrime_650M_OGT_Prediction \
  --mode bigru-fusion \
  --out_dir train/fusion_out
```

<details>
<summary><b>Key training options</b></summary>

| Flag | Default | Description |
|:---|:---|:---|
| `--epochs` | 20 | Max training epochs |
| `--patience` | 7 | Early stopping patience |
| `--lr` | 1e-4 | Learning rate |
| `--batch_size` | 8 | Batch size per GPU |
| `--k_folds` | 5 | Number of CV folds |
| `--fp16` | — | Enable AMP mixed precision |
| `--use_cache` | — | Cache backbone embeddings to disk |
| `--tensorboard` | — | Log to TensorBoard |

</details>

### 3. Test

Evaluate all 5 folds on the test set:

```bash
python test/test.py \
  --test_csv data/test.csv \
  --train_csv data/train.csv \
  --prott5_model_path prot_t5_xl_uniref50 \
  --proprime_model_path ProPrime_650M_OGT_Prediction \
  --mode bigru-fusion \
  --ckpt_dir train/fusion_out \
  --out_dir results/test_out
```

Output includes per-fold and ensemble metrics, ROC/PR curves, confusion matrices, and t-SNE/UMAP visualizations.

### 4. Predict

```bash
# From CSV
python predict/predict.py \
  --input_path your_sequences.csv \
  --input_type csv \
  --ckpt_dir train/fusion_out \
  --prott5_model_path prot_t5_xl_uniref50 \
  --proprime_model_path ProPrime_650M_OGT_Prediction \
  --mode bigru-fusion \
  --out_dir pred_output

# From FASTA
python predict/predict.py \
  --input_path your_sequences.fasta \
  --input_type fasta \
  --ckpt_dir train/fusion_out \
  --out_dir pred_output
```

**Output files:**

| File | Description |
|:---|:---|
| `predictions_per_fold.csv` | Per-fold probabilities and predictions |
| `predictions_ensemble.csv` | **Mean-ensemble** probabilities and final prediction |

---

## Released Weights

The repository includes 5 pre-trained fusion head checkpoints:

```
train/fusion_out/
├── best_head_fold_1.pt    (13.5 MB)
├── best_head_fold_2.pt    (13.5 MB)
├── best_head_fold_3.pt    (13.5 MB)
├── best_head_fold_4.pt    (13.5 MB)
└── best_head_fold_5.pt    (13.5 MB)
```

Each checkpoint contains both the model `state_dict` and training `args` for reproducibility.

---

## Notes

- The two backbone models (`ProtT5` / `ProPrime`) are **not included** — download them via `scripts/download_backbones.py`.
- `results/` contains curated summary artifacts; full raw evaluation outputs are generated by `test.py`.
- Input can be **CSV** (`id,seq`) or **FASTA** format.
- For large-scale prediction, enable `--use_cache` and `--fp16` to reduce memory usage and speed up repeated runs.

---

