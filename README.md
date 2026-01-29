# Multi-Resolution Ensemble Learning with Quantitative Interpretability for Enhanced Breast Cancer Histopathology Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18413089.svg)](https://doi.org/10.5281/zenodo.18413089)

## 📌 Context
This repository contains the source code, experimental pipeline, and deployment tools for the manuscript **"Multi-Resolution Ensemble Learning with Quantitative Interpretability for Enhanced Breast Cancer Histopathology Classification"**, submitted to *The Visual Computer* (2026).

It implements the **MRE-MRPE framework** (Multi-Resolution Ensemble with Model-Rate Performance Ensemble weighting) and the quantitative XAI analysis described in the paper.

## 💻 Hardware Optimization Note
**Platform:** Apple Silicon (M4 Chip)
**Backend:** Metal Performance Shaders (MPS)

> **Note for Reproducibility:** The latency benchmarks and speed-accuracy trade-offs (FP16 vs. FP32) detailed in the manuscript were conducted on an Apple M4 device using the `mps` accelerator. While this code is fully compatible with NVIDIA GPUs (`cuda`) and CPUs, performance metrics will vary on different hardware.

## 📂 Repository Structure
* `notebook.ipynb`: The complete research pipeline, including data preprocessing, model training (EfficientNet, MobileNet, ShuffleNet), MRPE weight calculation, and statistical analysis.
* `app.py`: The deployable **Gradio** web interface ("Glass Box" tool) for clinical decision support.
* `requirements.txt`: List of dependencies.
* `ground_truth_regions.csv`: Proxy ground truth coordinates for XAI IoU validation.
* `pathologist_scores.csv`: Simulated pathologist scoring data for correlation analysis.

## 🚀 Usage

### 1. Installation
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt