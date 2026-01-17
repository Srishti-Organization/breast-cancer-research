# MRE-MRPE: Multi-Resolution Ensemble Framework for Breast Cancer Histopathology

This repository contains the implementation of the **MRE-MRPE framework**, a deep learning approach for classifying breast cancer histopathology images into four classes: **Benign, InSitu, Invasive, and Normal**. 

This work accompanies the paper *"MRE-MRPE: A Multi-Resolution Ensemble Framework with Speed-Accuracy Weighting and Quantitative Interpretability for Breast Cancer Histopathology"* by **Srishti Rao Punaroor, Tanmoy Hazra, and Rahul Dixit**.

## 📌 Project Overview

Accurately classifying breast cancer from Whole Slide Images (WSIs) is challenging due to the heterogeneous nature of tissue features at different magnifications. This framework addresses this by:

1.  **Multi-Resolution Training**: Training lightweight CNNs (MobileNetV3, EfficientNet-B0, ShuffleNetV2) on patches extracted at three different resolutions ($256^2, 512^2, 1024^2$).
2.  **Ensemble Learning**: Implementing a Model-Rate Performance Ensemble (MRPE) to balance accuracy and inference latency.
3.  **Explainable AI (XAI)**: Utilizing Grad-CAM to generate heatmaps and quantitatively assessing them against proxy ground truth regions and pathologist scores.
4.  **Hardware Optimization**: Benchmarking FP16 (Half-Precision) vs. FP32 performance specifically on Apple Silicon (MPS).

## 📊 Dataset

The model is trained and evaluated on the **ICIAR 2018 BACH (BreAst Cancer Histology) Dataset**.
* **Classes**: Normal, Benign, InSitu, Invasive.
* **Source**: [ICIAR 2018 Challenge](https://iciar2018-challenge.grand-challenge.org/)

*Note: You must download the dataset separately and place the photos in `data/raw/bach/ICIAR2018_BACH_Challenge/Photos`.*

## 🛠️ Requirements

The project requires Python 3.8+ and the following libraries:

```txt
torch
torchvision
timm
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
gradio
pytorch-grad-cam
statsmodels