# Semantic Segmentation with Inbuilt ResNet50 and a Custom-built Swin-Transformer + Mask2Former

This project implements a full end-to-end semantic segmentation pipeline using PyTorch. It includes:
- **Dataset Preparation:** Using the COCO 2017 dataset, generating segmentation masks, balancing classes, and saving preprocessed data in LMDB format.
- **Model Architectures:**
  - **Custom Model:** A segmentation network built with a Swin Transformer backbone and a Mask2Former-style decoder.
  - **Inbuilt Model:** An alternative baseline using torchvision's FCN-ResNet50.
- **Training Pipeline:** Fully configurable via YAML files with mixed precision (AMP), gradient clipping, class weighted loss with Hungarian matching, and WandB logging.
- **Inference:** Two approaches:
  - **Command-Line Inference** to output predicted masks.
  - **Interactive Web Dashboard (using Plotly Dash)** to upload images and visualize predictions.

---

## Table of Contents
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Architectures](#model-architectures)
  - [Custom Segmentation Model](#custom-segmentation-model)
  - [Inbuilt FCN-ResNet50 Model](#inbuilt-fcn-resnet50-model)
- [Training Pipeline](#training-pipeline)
- [Inference](#inference)
  - [Command-Line Inference](#command-line-inference)
  - [Interactive Web Dashboard](#interactive-web-dashboard)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Reproducibility](#reproducibility)
- [Additional Notes](#additional-notes)
- [Contact](#contact)

---

## Directory Structure

```
segmentation/
├── configs/
│   ├── dataset.yaml
│   ├── environment.yaml
│   ├── model.yaml
│   └── train.yaml
├── datasets/
├── inference/
├── models/
├── utils/
├── train.py
├── setup.py
└── README.md
```

---

## Requirements

- **Python 3.9**
- **Key Libraries:**
  - PyTorch and torchvision
  - WandB
  - LMDB
  - Omegaconf
  - Plotly Dash
  - numpy, pandas, matplotlib, opencv, pycocotools, tqdm, aiohttp, requests

---

## Installation

```bash
git clone <repository-url>
cd segmentation/
conda env create -f configs/environment.yaml
conda activate segmentation
python setup.py install
```

---

## Dataset Preparation

```bash
python datasets/prepare_dataset.py --config configs/dataset.yaml
```

- Downloads and extracts COCO annotations
- Filters and balances images
- Generates pixel-level masks
- Converts images and masks to LMDB

---

## Model Architectures

### Custom Segmentation Model

- Swin Transformer backbone
- Mask2Former decoder
- Hungarian + CE + BCE + Dice Loss

### Inbuilt FCN-ResNet50 Model

- torchvision baseline model
- FCN head + ResNet50 backbone
- Toggle via `cfg.model.inference_prebuilt`

---

## Training Pipeline

```bash
python train.py --config configs/train.yaml
```

- Uses LMDB datasets
- AMP, gradient clipping, class weights
- WandB logging
- Saves checkpoints and test visualizations

---

## Inference

### Interactive Web Dashboard

```bash
python inference/inference.py
```

- Upload image
- Visualize predicted mask + overlay (Plotly Dash)

---

## Evaluation and Metrics

- Visual results saved per test epoch

---

## Reproducibility

```bash
python datasets/prepare_dataset.py --config configs/dataset.yaml
python train.py --config configs/train.yaml
python inference/inference.py
```

---

## Additional Notes

- Modular, extendable codebase
- YAML config-driven
- Dual-mode inference
- LMDB-optimized I/O

---

## Sample Inference Results

Check assests folder

---

## Contact

adithm@iisc.ac.in
