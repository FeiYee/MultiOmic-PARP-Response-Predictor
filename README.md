---

# Multi-Omic PARP Inhibitor Response Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-Pending-blue.svg)]()

> A robust and reproducible computational oncology framework for predicting response to PARP inhibitors (e.g., Olaparib) using multi-omic data integration. Designed for translational cancer research with strict control of information leakage.

---

## 🧭 Workflow Overview

![Workflow](https://github.com/FeiYee/MultiOmic-PARP-Response-Predictor/blob/main/results/figures/Workflow.jpg)

---

## ✨ Key Features

* **Rigorous study design**
  Three-tier validation framework:
  *Pharmacogenomic modeling → Cross-platform validation → Clinical correlation analysis*
  Ensures zero information leakage throughout the pipeline.

* **Multi-modal feature integration**
  Integrates:

  * Gene expression profiles
  * Somatic mutation data
  * Pathway activity signatures
    Capturing biological signals across multiple regulatory layers.

* **Strong predictive performance**

  * 5-fold cross-validation: **AUROC = 0.855**
  * Independent validation: **AUROC = 0.79**

* **Biological interpretability**
  SHAP-based analysis identifies key molecular determinants of PARP inhibitor sensitivity.

* **Reproducibility-focused**
  Preprocessed datasets and modular scripts enable one-click full reproduction.

---

## 📊 Performance Metrics

| Validation Tier           | Dataset         | AUROC                       | AUPRC                                                             |
| ------------------------- | --------------- | --------------------------- | ----------------------------------------------------------------- |
| Internal cross-validation | GDSC2 (n=407)   | 0.855 (95% CI: 0.797–0.914) | 0.852 (95% CI: 0.795–0.908)                                       |
| Independent validation    | GDSC1 (n=352)   | 0.79 (95% CI: 0.73–0.85)    | —                                                                 |
| Clinical correlation      | TCGA-OV (n=599) | —                           | Significant association with BRCA mutation (OR = 2.34, p = 0.002) |

> ⚠️ **Disclaimer**: This tool is intended for academic research only and does not provide clinical decision support.

---

## 🛠️ Installation

### Requirements

* Python ≥ 3.10

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/PARP-Predictor.git
cd PARP-Predictor

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Download Datasets

Preprocessed datasets (~3.41 GB):

👉 [OneDrive Download](https://1drv.ms/u/c/0c0bd60421357fab/IQDALBEfYrajQZZCuNzSX2mTAft-X3nehqePYdtekJdqZAc?e=0WgG1w)

Place all `.csv` files into:

```
data/
```

---

### 2. Reproduce Full Analysis

Run scripts sequentially:

```bash
# Step 1: Model training & evaluation
python scripts/01_train_model.py

# Step 2: Generate figures
python scripts/02_generate_figures.py

# Step 3: SHAP interpretability
python scripts/03_shap_analysis.py

# Step 4: Supplementary analyses
python scripts/04_supplementary_analysis.py

# Step 5: External validation
python scripts/05_external_validation.py
```

---

### 3. Custom Prediction

```python
import pandas as pd
import joblib

# Load trained model
model = joblib.load("results/models/rf_final_model.pkl")

# Load user data
sample = pd.read_csv("your_sample.csv", index_col=0)

# Predict sensitivity score
score = model.predict_proba(sample)[:, 1][0]

print(f"PARPi Sensitivity Score: {score:.3f}")
```

> Higher scores indicate increased predicted sensitivity to PARP inhibitors.

---

## 📁 Project Structure

```
PARP-Predictor/
├── scripts/
│   ├── 01_train_model.py
│   ├── 02_generate_figures.py
│   ├── 03_shap_analysis.py
│   ├── 04_supplementary_analysis.py
│   └── 05_external_validation.py
├── results/
│   ├── figures/
│   └── models/
├── dataset/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📝 Citation

If you use this work, please cite:

```
[To be updated]
Author Name, et al.
Multi-omic prediction of PARP inhibitor response in high-grade serous ovarian cancer.
[Journal Name], 2026.
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for details.

---
