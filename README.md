# MRE-MRPE: Multi-Resolution Ensemble Framework for Breast Cancer Histopathology

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-726E88?style=for-the-badge&logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

This repository contains the official implementation of the **MRE-MRPE framework**, a deep learning approach for classifying breast cancer histopathology images into four classes: **Benign, InSitu, Invasive, and Normal**. 

This work accompanies the paper:
> **"MRE-MRPE: A Multi-Resolution Ensemble Framework with Speed-Accuracy Weighting and Quantitative Interpretability for Breast Cancer Histopathology"** > *Srishti Rao Punaroor, Tanmoy Hazra, and Rahul Dixit* > *Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology, India*

---

## 📌 Project Overview

Accurately classifying breast cancer from Whole Slide Images (WSIs) is challenging due to the heterogeneous nature of tissue features at different magnifications. This framework addresses this challenge through:

1.  **Multi-Resolution Training**: Training lightweight CNNs (**MobileNetV3**, **EfficientNet-B0**, **ShuffleNetV2**) on patches extracted at three distinctive resolutions ($256^2, 512^2, 1024^2$).
2.  **Ensemble Learning**: Implementing a **Model-Rate Performance Ensemble (MRPE)** to balance classification accuracy against inference latency.
3.  **Explainable AI (XAI)**: Utilizing **Grad-CAM** to generate heatmaps and quantitatively assessing them against proxy ground truth regions.
4.  **Hardware Optimization**: Benchmarking **FP16 (Half-Precision)** vs. **FP32** performance specifically on Apple Silicon (MPS).

---

## 📊 Dataset

The model is trained and evaluated on the **ICIAR 2018 BACH (BreAst Cancer Histology) Dataset**.

* **Classes**: Normal, Benign, InSitu, Invasive.
* **Source**: [ICIAR 2018 Challenge](https://iciar2018-challenge.grand-challenge.org/)

> **⚠️ Important**: Due to licensing, the dataset is not included in this repository. 
> 1. Download the dataset from the official challenge website.
> 2. Place the images in: `data/raw/bach/ICIAR2018_BACH_Challenge/Photos`

---

## 🛠️ Installation & Requirements

The project is designed for **Python 3.8+**. We recommend using a virtual environment.

```bash
# Clone the repository
git clone [https://github.com/srishti-rao/breast-cancer-research.git](https://github.com/srishti-rao/breast-cancer-research.git)
cd breast-cancer-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision timm numpy pandas matplotlib seaborn scikit-learn Pillow gradio pytorch-grad-cam statsmodels