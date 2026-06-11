# Automated Intracranial Brain Hemorrhage Detection using CT Scans

This repository contains a complete, end-to-end deep learning pipeline for the detection and classification of **Intracranial Hemorrhage (ICH)** from head Computed Tomography (CT) scans. 

The pipeline uses the **RSNA Intracranial Hemorrhage Detection** dataset, implementing custom medical image preprocessing (Hounsfield Units conversion and brain windowing), multi-label deep learning classification using a **ResNet50** architecture, progressive fine-tuning, and threshold-based inference.

---

## 📌 Project Overview

Intracranial hemorrhage (bleeding inside the skull) is a life-threatening medical emergency. Rapid and accurate detection is critical for patient survival. This project implements an automated solution to:
1. Detect the presence of any intracranial hemorrhage.
2. Identify the specific subtype(s) of hemorrhage present in the CT scan.

The model classifies five distinct subtypes of brain hemorrhage:
*   **Epidural (EDH)**
*   **Intraparenchymal (IPH)**
*   **Intraventricular (IVH)**
*   **Subarachnoid (SAH)**
*   **Subdural (SDH)**

---

## 🛠️ Repository Structure

*   **`brain_hemorrhage_detection.ipynb`**: The main Jupyter Notebook containing the entire pipeline divided into structured steps:
    1.  **Dataset Preprocessing & Splits**: Loading metadata, filtering/pivoting labels, train-val-test splitting, copying raw DICOM images, and balancing classes.
    2.  **Medical Windowing & PNG Conversion**: Applying Hounsfield Units (HU) conversion and brain-specific windowing to DICOMs, followed by resizing and saving to PNG.
    3.  **Model Architecture & Pre-Training**: Initializing a ResNet50 model with custom dense, batch normalization, and dropout layers with frozen base weights.
    4.  **Progressive Fine-Tuning**: Unfreezing top layers and training with progressive learning rates (from $10^{-4}$ down to $10^{-6}$) using Early Stopping.
    5.  **Inference Pipeline**: Functions to load the trained model, preprocess new CT images, and display bleed predictions with confidence levels.

---

## 🧬 Medical Image Preprocessing Pipeline

Raw CT images are distributed in the **DICOM** (`.dcm`) format, which contains raw 12-bit or 16-bit attenuation values. Standard image models cannot handle these values directly. The preprocessing pipeline performs the following critical clinical transformations:

### 1. Hounsfield Unit (HU) Rescaling
CT scanner pixel values (attenuation coefficients) are converted to Hounsfield Units (HU) using rescale slope and intercept metadata attributes:
$$\text{HU} = (\text{Pixel Value} \times \text{Rescale Slope}) + \text{Rescale Intercept}$$

### 2. Brain Windowing
A raw CT scan has a very wide dynamic range (often -1000 HU to +3000 HU). However, the human brain and acute blood exist in a very narrow range. To highlight hemorrhages, we apply a **Brain Window** centered at **Level (L) = 40 HU** with a **Width (W) = 80 HU**:
*   **Window Range**: $[L - \frac{W}{2}, L + \frac{W}{2}] = [0\text{ HU}, 80\text{ HU}]$
*   Any values $< 0\text{ HU}$ are clipped to 0 (appearing black).
*   Any values $> 80\text{ HU}$ are clipped to 80 (appearing white).
*   This range maximizes contrast between normal brain tissue and fresh blood, which typically measures between $+50\text{ HU}$ and $+70\text{ HU}$.

### 3. Normalization and Resizing
*   The clipped values are normalized to a $[0, 1]$ float range.
*   Images are resized to $224 \times 224$ pixels to match the input specifications of modern deep learning backbones.
*   Images are exported to PNG format for efficient loading during training.

---

## 🏗️ Neural Network Architecture

The model uses a transfer learning approach built on top of **ResNet50** (pretrained on ImageNet) to perform multi-label classification:

```
                  [ Input Image (224x224x3) ]
                              │
                    [ ResNet50 Backbone ] (Pretrained, initially frozen)
                              │
                 [ Global Average Pooling 2D ]
                              │
                   [ Batch Normalization ]
                              │
                 [ Dense Layer (512, ReLU) ]
                              │
                    [ Dropout (Rate = 0.5) ]
                              │
                 [ Dense Layer (128, ReLU) ]
                              │
                    [ Dropout (Rate = 0.3) ]
                              │
               [ Dense Output Layer (5, Sigmoid) ]
          (Epidural, Intraparenchymal, Intraventricular, Subarachnoid, Subdural)
```

### Key Training Details
*   **Loss Function**: Binary Cross-Entropy (calculated independently for each of the 5 labels, making it ideal for multi-label classification where multiple bleed types can coexist).
*   **Optimizer**: Adam with progressive learning rate schedules:
    *   *Phase 1*: Warm-up head training ($LR = 10^{-4}$) for 10 epochs.
    *   *Phase 2*: Fine-tuning last 50 layers ($LR = 10^{-5}$) with Early Stopping (patience = 2).
    *   *Phase 3*: Final fine-tuning ($LR = 10^{-6}$) for 3 epochs.

---

## 🚀 Getting Started

### 📋 Prerequisites & Setup
You can run this notebook directly on **Kaggle** or **Google Colab** (with Google Drive mounted), or locally. Install the required libraries via `pip`:

```bash
pip install tensorflow pydicom opencv-python pandas numpy scikit-learn matplotlib seaborn
```

### 💽 Dataset Download
The code expects the dataset from the [RSNA Intracranial Hemorrhage Detection Challenge](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection). 
*   Place the `stage_2_train` folder (containing DICOMs) and `stage_2_train.csv` inside your designated base directory (e.g. `/content/drive/MyDrive/hemorrhage` or `/kaggle/input/...`).

### 🏃 Running the Pipeline
1. Open `brain_hemorrhage_detection.ipynb` in your Jupyter environment.
2. Update the `base_path` variable in the Preprocessing and Training sections to point to your dataset directory.
3. Run the cells sequentially:
    *   Cells 1-10 perform dataset exploration, pivots, class balancing (8,000 normal and 8,000 hemorrhage scans), and split generation.
    *   Cells 11-14 perform DICOM windowing, PNG conversion, and zipping.
    *   Cells 15-20 load the PNG dataset, configure `ImageDataGenerator` data augmentations, and compile the ResNet50 network.
    *   Cells 21-25 train and progressively fine-tune the model, saving weights to `resnet_multilabel.h5`.
    *   Cells 26-28 implement inference and testing on validation/test images.

---

## 🔮 Inference & Classification Logic

During inference, a test image is preprocessed (using matching ResNet50 preprocessing) and passed to the model. The output contains 5 probabilities (one for each subtype). The decision logic is as follows:

1.  **Hemorrhage Detection**:
    *   If the maximum probability across all 5 classes is below **0.3** ($\max(P) < 0.3$), the scan is classified as **No Hemorrhage Detected ❌**.
2.  **Hemorrhage Subtyping**:
    *   If a hemorrhage is detected, any subtype with a probability above **0.2** ($P > 0.2$) is marked as **Detected ✅** and listed in order of confidence.
    *   If no individual subtype exceeds the threshold of 0.2, the model outputs **Uncertain Type**.

### Sample Output format:
```text
Testing image: /path/to/test_image.png
Hemorrhage Detected ✅

Type(s) of Bleed:
- subdural (85.40%)
- subarachnoid (42.15%)

Confidence Scores:
epidural: 2.10%
intraparenchymal: 10.45%
intraventricular: 1.12%
subarachnoid: 42.15%
subdural: 85.40%
```

---

## 🛡️ License
This project is for educational and research purposes. The model should not be used as a primary diagnostic tool in clinical settings without rigorous clinical validation.
