# 🧠 Alzheimer Model — MRI Detection with MedSAM + ResNet18

A hybrid deep learning model for classifying Alzheimer's disease stages from MRI scans using **MedSAM brain structure segmentation** combined with a **fine-tuned ResNet18** image classifier. Achieves **97.27% test accuracy** on the Alzheimer's MRI Dataset.

---

## 📌 Project Overview

This project presents a novel two-branch architecture that fuses:
1. **Structural features** extracted by MedSAM (Medical Segment Anything Model) — capturing volumetric and morphological properties of key brain regions.
2. **Visual features** from a pre-trained ResNet18 CNN — learning image-level patterns associated with neurodegeneration.

The model classifies MRI scans into four clinically relevant stages:

| Stage | Label |
|---|---|
| Non Demented | 0 |
| Very Mild Demented | 1 |
| Mild Demented | 2 |
| Moderate Demented | 3 |

---

## 🏗️ Architecture

```
Input MRI Image
      │
      ├──────────────────────────┐
      │                          │
 ResNet18 Backbone          MedSAM Segmenter
 (frozen early layers)      (ViT-B checkpoint)
      │                          │
 512-dim CNN features       12-dim structural features
      │                          │  (volume ratio, compactness,
      └────────────┬─────────────┘   intensity per brain region)
                   │
            Fusion Layer (768-dim)
                   │
        FC → BN → ReLU → Dropout
        FC → BN → ReLU → Dropout
                   │
           4-class Softmax
```

**Brain regions segmented by MedSAM:**
- Hippocampus
- Temporal Lobe (left + right)
- Ventricles
- Entorhinal Cortex (left + right)

For each region, 3 features are extracted: **Volume Ratio**, **Compactness**, **Mean Intensity** → 12 features total.

---

## 📊 Results

| Metric | Value |
|---|---|
| Best Validation Accuracy | **98.44%** |
| Final Test Accuracy | **97.27%** |
| Training Epochs | 30 |

**Per-class Performance (Test Set):**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Non Demented | 0.9841 | 0.9688 | 0.9764 | 640 |
| Very Mild Demented | 0.9563 | 0.9777 | 0.9669 | 448 |
| Mild Demented | 0.9778 | 0.9832 | 0.9805 | 179 |
| Moderate Demented | 0.9167 | 0.8462 | 0.8800 | 13 |
| **Weighted Avg** | **0.9728** | **0.9727** | **0.9727** | **1280** |

---

## 🗂️ Dataset

- **Source:** [Alzheimer MRI Dataset on Kaggle](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset) (`sachinkumar413/alzheimer-mri-dataset`)
- **Total Images:** 6,400
- **Split:** 72% Train / 8% Validation / 20% Test (stratified)

---

## ⚙️ Setup & Usage

### 1. Prerequisites

This notebook is designed to run on **Google Colab** with a T4 GPU.

```bash
pip install kaggle
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. Dataset Download

Add your Kaggle API credentials in **Cell 1** and run — the dataset downloads and extracts automatically.

### 3. Download MedSAM Weights

The notebook downloads `medsam_vit_b.pth` automatically from Zenodo.

### 4. Run Cells in Order

| Cell | Description |
|---|---|
| Cell 1 | Setup: installs packages, downloads dataset & MedSAM weights |
| Cell 2 | MedSAM segmentation + feature pre-computation (runs once, ~53 min) |
| Cell 3 | Model training (30 epochs, ~9 min) |
| Cell 4 | Evaluation, classification report & confusion matrix |
| Cell 5 | Single-image inference — upload your own MRI scan |
| Cell 6 | Inspect raw MedSAM structural measurements |
| Cell 7 | Save trained model & features to Google Drive |

---

## 🔬 How It Works

### MedSAM Feature Extraction

MedSAM uses bounding box prompts to segment anatomically meaningful brain regions. For each region, we extract:
- **Volume Ratio** — proportion of image area occupied by the structure (proxy for atrophy)
- **Compactness** — shape regularity (4π·Area / Perimeter²)
- **Mean Intensity** — average pixel brightness within the region (tissue density proxy)

These 12 features encode structural changes that are hallmarks of Alzheimer's progression (e.g., hippocampal atrophy, ventricular enlargement).

### Fusion Classifier

The structural features are passed through a dedicated 3-layer MLP branch and concatenated with ResNet18's 512-dim global average pooled features. A fusion head (2 FC layers with BatchNorm and Dropout) maps the combined 768-dim vector to 4 class logits.

### Training Details

- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Cross-Entropy
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Batch Size:** 32
- **Augmentation:** Horizontal Flip, ±10° Rotation, Color Jitter (brightness & contrast)
- **ResNet18 partial freezing:** first 20 parameters frozen for transfer learning stability

---

## 📁 Output Files

| File | Description |
|---|---|
| `best_hybrid_model.pth` | Saved model weights (best validation accuracy) |
| `precomputed_medsam_features.pt` | Pre-computed MedSAM features for all 6,400 images |
| `final_confusion_matrix.png` | Raw count + normalized confusion matrix |

---

## 🧪 Single Image Inference

After training, Cell 5 allows you to upload any MRI image and receive:
- MedSAM segmentation overlay visualization
- Predicted dementia stage with confidence score
- Full probability distribution across all 4 classes

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Deep learning framework |
| `segment-anything` (Facebook) | MedSAM backbone |
| `opencv-python` | Image processing, contour detection |
| `Pillow` | Image loading |
| `scikit-learn` | Train/val/test split, metrics |
| `matplotlib`, `seaborn` | Visualization |
| `tqdm` | Progress bars |
| `kaggle` | Dataset download |

---

## 📜 License

This project is for research and educational purposes. The MedSAM weights are subject to the [Segment Anything Model license](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE).

---

## 🙏 Acknowledgements

- **MedSAM** — Ma et al., *Segment Anything in Medical Images* ([Zenodo](https://zenodo.org/records/10689643))
- **Alzheimer MRI Dataset** — Sachin Kumar (Kaggle)
- **Segment Anything Model** — Meta AI Research
