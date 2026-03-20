# 🌾 Rice Leaf Disease Detection
### PRCP-1001 | Deep Learning Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prcp-1001-rice-leaf-disease-detection.streamlit.app/)

> **Live Demo:** [https://prcp-1001-rice-leaf-disease-detection.streamlit.app/](https://prcp-1001-rice-leaf-disease-detection.streamlit.app/)

---

## 📌 Project Overview

Rice is a staple food crop that sustains over half the world's population. Diseases affecting rice leaves pose a serious threat to agricultural productivity, often leading to significant yield losses if not identified and treated at an early stage.

This project develops a **deep learning-based image classification system** capable of automatically identifying rice leaf diseases from photographs. The model classifies each input image into one of three disease categories using **InceptionV3 Two-Phase Transfer Learning** — achieving a final accuracy of **95.65%**.

### Disease Classes

| Disease | Description |
|---|---|
| 🟡 Bacterial Leaf Blight | Water-soaked lesions along leaf margins that turn yellow and white |
| 🟤 Brown Spot | Dark brown oval spots with yellow halos scattered across the leaf blade |
| ⚫ Leaf Smut | Small angular black spots appearing on both sides of the leaf surface |

---

## 🗂️ Project Structure

```
Rice-Leaf-Disease-Detection/
│
├── app.py                          ← Streamlit web application
├── requirements.txt                ← Python dependencies
├── Riceleafff.ipynb                ← Full project notebook
│
├── Dataa/                          ← Dataset directory
│   ├── Bacterial Leaf Blight/      ← 40 images
│   ├── Brown Spot/                 ← 40 images
│   └── Leaf Smut/                  ← 39 images
│
├── rice_disease_best_model.keras   ← Best deployed model (InceptionV3)
├── best_inc_best_model.keras       ← Best checkpoint saved during training
├── model_results.json              ← Saved model performance metrics
│
└── assets/
    ├── image_dimensions.png
    ├── rgb_analysis.png
    ├── brightness_contrast.png
    └── correlation_heatmap.png
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Total Images | 119 JPG images |
| Total Classes | 3 disease categories |
| Bacterial Leaf Blight | 40 images |
| Brown Spot | 40 images |
| Leaf Smut | 39 images |
| Image Resolution | Widths range from 250px to 3081px (12× variation) |
| Color Space | RGB |
| Missing / Corrupt Images | None |

> **Note:** The dataset contains only diseased leaf images. No healthy rice leaf images are included — the model is trained purely for disease classification and cannot distinguish a healthy leaf from a diseased one.

---

## 🔍 Exploratory Data Analysis

A comprehensive visual and statistical exploration was performed on the dataset:

- **Sample Image Visualization** — 5 samples per disease class displayed in a 3×5 grid
- **Class Distribution Analysis** — confirmed near-perfectly balanced classes (40 / 40 / 39)
- **Image Integrity Check** — OpenCV readability test on all images; zero corrupt files found
- **Outlier Detection** — IQR and Z-score methods applied to width, height, and aspect ratio
- **Image Dimension Analysis** — width range 250–3081px confirmed; standardised resizing mandatory
- **Enhanced Dimension Scatter Plot** — per-class resolution patterns with mean reference lines
- **RGB Channel Distribution** — dominant Green channel confirmed across all classes as expected for plant leaves
- **Brightness & Contrast Analysis** — large differences in brightness across classes mitigated by augmentation
- **Aspect Ratio Analysis** — histogram, boxplot, and violin plots per class
- **Correlation Heatmap** — 9 image metrics correlated; strong brightness-channel relationships confirmed

> **Key Finding:** Brown Spot and Leaf Smut show significant visual similarity — both present as small dark spots on the leaf surface. This is the primary source of misclassification difficulty for all models.

---

## ⚙️ Preprocessing & Augmentation

### Input Standardisation
Due to highly varying image resolutions, all images were resized to fixed dimensions per model:

| Model | Input Size |
|---|---|
| Custom CNN, VGG16, MobileNetV2, ResNet50 | 128 × 128 |
| InceptionV3 Baseline | 150 × 150 |
| InceptionV3 Two-Phase (Final) | 200 × 200 |

### Augmentation Pipeline (Training Only)

```python
ImageDataGenerator(
    rescale            = 1./255,
    rotation_range     = 40,
    zoom_range         = 0.3,
    horizontal_flip    = True,
    vertical_flip      = True,
    width_shift_range  = 0.3,
    height_shift_range = 0.3,
    shear_range        = 0.3,
    brightness_range   = [0.6, 1.4],
    channel_shift_range= 30.0,
    fill_mode          = 'nearest'
)
```

The validation generator applied **rescaling only** to ensure unbiased evaluation. The 80/20 train/validation split produced 96 training images and 23 validation images.

---

## 🤖 Models Built & Evaluated

Five architectures were trained and compared:

| Rank | Model | Validation Accuracy |
|---|---|---|
| 1st | **InceptionV3 (Two-Phase Fine-Tuning)** | **95.65%** |
| 2nd | MobileNetV2 | 82.61% |
| 3rd | VGG16 | 69.57% |
| 4th | Custom CNN | 60.87% |
| 5th | ResNet50 | 34.78% |

---

## 🏆 Best Model — InceptionV3 Two-Phase Transfer Learning

The final model uses a structured **two-phase training strategy** on InceptionV3 pretrained on ImageNet:

### Phase 1 — Custom Head Warmup
- All 311 InceptionV3 base layers **frozen**
- Only the custom head trained: `GlobalAveragePooling2D → Dense(256) → BatchNormalization → Dropout(0.2) → Dense(128) → Dropout(0.2) → Dense(3, Softmax)`
- Learning rate: **0.001**
- Up to 30 epochs with EarlyStopping (patience=10)

### Phase 2 — Fine-Tuning
- Last **100 InceptionV3 layers unfrozen**
- Learning rate reduced to **0.000005** (200× smaller) to protect pretrained weights
- Up to 120 epochs with EarlyStopping (patience=25) monitoring `val_accuracy`
- Best weights saved to `best_inc_best_model.keras`

> **Why two phases?** Unfreezing base layers without a warmup phase caused noisy gradients to corrupt pretrained weights early in training. Stabilising the custom head first before fine-tuning resolved this instability entirely.

---

## 📉 Classification Report — Best Model

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Bacterial Leaf Blight | 1.00 | 1.00 | 1.00 |
| Brown Spot | 1.00 | 0.875 | 0.933 |
| Leaf Smut | 0.875 | 1.00 | 0.933 |
| **Macro Average** | **0.958** | **0.958** | **0.978** |

- **Single misclassification:** One Brown Spot image predicted as Leaf Smut — an expected failure at the most visually ambiguous class boundary
- **Bacterial Leaf Blight:** Perfect scores across all metrics — distinctive margin lesions make this class uniquely identifiable

---

## 🔍 GradCAM — Model Explainability

GradCAM (Gradient-weighted Class Activation Mapping) was implemented to verify the model focuses on actual disease symptoms rather than background or irrelevant patterns.

- Gradients extracted from InceptionV3's final convolutional layer `mixed10`
- Heatmap overlaid on original image — **red regions = highest model attention**
- A **70% minimum confidence threshold** rejects non-rice-leaf inputs automatically
- GradCAM confirmed the model consistently focuses on lesion and spot regions

---

## ⚠️ Challenges Faced

1. **Small Dataset Size** — Only 119 images. Addressed through 9-technique augmentation and ImageNet transfer learning.
2. **High Resolution Variation** — 12× width variation across images. Resolved through fixed-size resizing per architecture.
3. **Visual Similarity Between Classes** — Brown Spot and Leaf Smut both appear as small dark spots. Caused the single misclassification in the final model.
4. **Training Instability** — Partial base layer unfreezing caused gradient noise. Resolved through the two-phase strategy.
5. **Small Validation Set Variance** — 23 validation images meant each prediction = ~4.35% accuracy. EarlyStopping monitored `val_loss` for stability.
6. **Background Inconsistency** — Some images had backgrounds removed creating visual inconsistency across classes.

---

## 🚀 Live Demo

The best performing model (InceptionV3 Two-Phase) is deployed as an interactive Streamlit web application:

**[👉 Try the Live App](https://prcp-1001-rice-leaf-disease-detection.streamlit.app/)**

Upload a rice leaf image and the model will predict the disease class with a confidence score. Images below 70% confidence are automatically rejected as non-rice-leaf inputs.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data Manipulation | NumPy, Pandas |
| Image Processing | OpenCV, Pillow |
| Visualisation | Matplotlib, Seaborn |
| Deep Learning | TensorFlow, Keras |
| Transfer Learning | InceptionV3, VGG16, MobileNetV2, ResNet50 (ImageNet weights) |
| Explainability | GradCAM (custom implementation) |
| Web Application | Streamlit |
| Statistical Analysis | SciPy, Scikit-learn |

---

## 💻 Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/shamveelmusthafa/PRCP-1001-Rice-Leaf-Disease-Detection
cd PRCP-1001-Rice-Leaf-Disease-Detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

---

## 📁 Key Files

| File | Description |
|---|---|
| `app.py` | Streamlit web application for live disease prediction |
| `Riceleafff.ipynb` | Complete project notebook with all analysis and models |
| `requirements.txt` | All Python dependencies for deployment |
| `rice_disease_best_model.keras` | Final deployed model (InceptionV3 Two-Phase) |
| `model_results.json` | Saved accuracy results for all 5 models |

---

## 👤 Author

**Shamveel Musthafa**
- GitHub: [@shamveelmusthafa](https://github.com/shamveelmusthafa)

---

*Project completed as part of the Datamites Data Science program — PRCP-1001*
