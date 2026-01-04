
# Thyroid Disorder Detection — ResNet50 + Grad-CAM (Explainable AI) 🏥

A compact research project that combines deep learning (ResNet50 + Grad-CAM) on ultrasound images with a classical machine learning model (Random Forest) on blood-test features to produce a fused, explainable thyroid disorder diagnostic aid.

---

## Table of Contents 📚

- [Project Overview](#project-overview-)
- [Features](#features-)
- [Repository Structure](#repository-structure-)
- [Quick Start](#quick-start-)
- [Usage](#usage-)
- [Training / Reproducing Models](#training--reproducing-models-)
- [Dataset](#dataset-)
- [Model Files](#model-files-)
- [Notes & Limitations ⚠️](#notes--limitations-️)
- [Contributing & Contact](#contributing--contact-)
- [License](#license-)

---

## Project Overview 💡

This project demonstrates a hybrid system to help detect thyroid conditions using:

- A ResNet50-based deep learning model trained on ultrasound images (with Grad-CAM visualizations for explainability).
- A Random Forest model trained on blood-test features (TSH, T3, T4, etc.).
- A fusion layer (in the Streamlit app) that combines DL and ML outputs to provide a final, interpretable diagnosis (hyperthyroid / hypothyroid / normal).

The Streamlit app (`integrated.py`) provides an interactive UI for image upload, blood-test inputs, visual Grad-CAM overlays, and combined diagnosis.

---

## Features ✅

- Image classification (Benign / Malignant) using ResNet50.
- Blood test classification (hyperthyroid / hypothyroid / normal) using Random Forest.
- Grad-CAM heatmaps for model explainability.
- Combined diagnosis with weighted fusion of DL and ML confidences.
- Example training scripts and a Jupyter notebook in `backend/`.

---

## Repository Structure 🔧

- `integrated.py` — Main Streamlit app (combined image + blood-test analysis, Grad-CAM visualization).
- `app.py` — Simple Streamlit demo that trains a RandomForest on `thyroid_dataset.csv` and predicts from inputs.
- `ml.py` — Example ML training / evaluation script (Random Forest + SHAP explainability snippet).
- `new_thyroid_resnet50_best.h5` — Pretrained ResNet model (used by `integrated.py`).
- `ml_model.pkl`, `ml_scaler.pkl` — Pickled RandomForest model and scaler used by the app.
- `thyroid_dataset.csv` — CSV with blood-test features and target labels.
- `backend/`
  - `dl.py` — Deep learning training script (ResNet50, image augmentations, save model).
  - `Thyroid Disorder Detection Using ResNet with Grad-CAM.ipynb` — Training/analysis notebook with Grad-CAM examples.
  - `dataset thyroid/` — Expected dataset layout (train/test folders with class subfolders).
- `.gitignore` — Typical excludes.

---

## Quick Start 🚀

Requirements:

- Python 3.8+ (Windows tested here)
- Recommended to use a virtual environment

Install common dependencies (example):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install streamlit pandas numpy scikit-learn tensorflow==2.10 pillow opencv-python matplotlib shap joblib
```

(You can adjust TensorFlow version to match your CUDA / GPU setup or use CPU-only builds.)

Run the full integrated app:

```bash
streamlit run integrated.py
```

Run the simple ML demo:

```bash
streamlit run app.py
```

Run ML training/evaluation:

```bash
python ml.py
```

Train DL model (example):

```bash
python backend/dl.py
```

Or open the notebook at `backend/Thyroid Disorder Detection Using ResNet with Grad-CAM.ipynb` to follow an interactive workflow.

---

## Usage — Notes on the Streamlit App 🖥️

- `integrated.py` provides three pages via the sidebar:
  - **Deep Learning (Image)**: Upload an ultrasound image and obtain a Benign / Malignant prediction with a Grad-CAM overlay.
  - **Machine Learning (Blood Test)**: Enter blood-test values (TSH, T3, T4, T4U, FTI, age, sex) to get a prediction and class probabilities.
  - **Combined Analysis**: Run both analyses and fuse results into a final diagnosis.

- Pretrained model files (`.h5` and `.pkl`) are loaded at startup. If any model files are missing, the app will show an error.

---

## Training / Reproducing Models 🧪

Deep Learning (images):

1. Place your images under `backend/dataset thyroid/train/<class>/` and `backend/dataset thyroid/test/<class>/` (e.g., `Benign/`, `Malignant/`, `normal thyroid/`).
2. Tweak `backend/dl.py` (image size, augmentations, epochs) as needed and run it.
3. Saved model filename: `thyroid_resnet50_benign_malignant_normal.h5` (as per `dl.py`), or you may save alternate checkpoints.

Machine Learning (blood tests):

1. `ml.py` demonstrates training a RandomForest and printing evaluation metrics. It does not persist models by default.
2. To save the trained model and scaler, add a snippet after training (example):

```python
import joblib
joblib.dump(model, 'ml_model.pkl')
joblib.dump(scaler, 'ml_scaler.pkl')
```

This will create the files used by `integrated.py`.

---

## Dataset 📁

- `thyroid_dataset.csv` — The CSV used by the ML demo. Columns expected by the app: `['age','sex','TSH','T3','T4','T4U','FTI','class']`.
- Image dataset expected layout:

```
backend/dataset thyroid/
  ├─ train/
  │   ├─ Benign/
  │   ├─ Malignant/
  │   └─ normal thyroid/
  └─ test/
      ├─ Benign/
      ├─ Malignant/
      └─ normal thyroid/
```

Make sure class subfolders contain only images for that class.

---

## Model Files 📦

- `new_thyroid_resnet50_best.h5` — ResNet50-based model used in the `integrated.py` app. (Contains weights and architecture; ensure TensorFlow/Keras compatibility when loading.)
- `thyroid_resnet50_benign_malignant_normal.h5` — Example DL model saved by `backend/dl.py`.
- `ml_model.pkl`, `ml_scaler.pkl` — Pickled RandomForest classifier and StandardScaler for the ML side.

---

## Notes & Limitations ⚠️

- This project is for research/educational purposes only — **not** a medical device. Always consult healthcare professionals for diagnosis.
- Model performance depends on dataset size, class balance, and image quality. Evaluate carefully before any real-world use.
- Grad-CAM gives a heuristic visualization of model attention; interpretation requires domain expertise.

---

## Contributing & Contact ✉️

- Contributions are welcome. Please open issues or PRs for bug fixes, improvements, or dataset additions.
- For quick questions, add an issue describing the request and include environment details (OS, Python, TensorFlow version).

---
