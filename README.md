# Multi-Omics PARP Inhibitor Response Predictor

> A robust computational oncology tool for predicting PARP inhibitor (Olaparib) response from multi-omic data. Fully reproducible, no information leakage, and built for translational cancer research.

\---

## ✨ Key Features

* **Rigorous study design**: Three-tier validation framework (pharmacogenomic modeling → cross-platform validation → clinical correlation analysis) with zero information leakage
* **Multi-modal feature integration**: Combines gene expression, somatic mutation, and pathway activity signatures to capture multi-level biological signals
* **Strong predictive performance**: 5-fold cross-validation AUROC=0.855, independent cross-platform validation AUROC=0.79
* **Biological interpretability**: Built-in SHAP analysis to identify biologically meaningful markers of PARPi response
* **Production-ready**: Preprocessed datasets available, one-click reproducibility of all results

\---

## 📊 Performance Metrics

|Validation Tier|Dataset|AUROC|AUPRC|
|-|-|-|-|
|Internal cross-validation|GDSC2 (n=407)|0.855 (95%CI: 0.797-0.914)|0.852 (95%CI: 0.795-0.908)|
|Independent cross-platform validation|GDSC1 (n=352)|0.79 (95%CI: 0.73-0.85)|-|
|Clinical correlation|TCGA-OV (n=599)|-|Significant association with BRCA mutation status (OR=2.34, p=0.002)|

> ⚠️ \\\*\\\*Disclaimer\\\*\\\*: This tool is for academic research purposes only and does not constitute clinical advice.

\---

## 🛠️ Installation

### Requirements

Python 3.10+

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/\\\[your-username]/PARP-Predictor.git
cd PARP-Predictor

# Install required packages
pip install -r requirements.txt
```

\---

## 🚀 Usage

### 1\. Download Datasets

Preprocessed aligned datasets (3.41GB) are available for download from:
[**OneDrive Download**](https://1drv.ms/u/c/0c0bd60421357fab/IQDALBEfYrajQZZCuNzSX2mTAft-X3nehqePYdtekJdqZAc?e=0WgG1w)  
Replace the link above with your actual OneDrive share link.

After downloading, place all `.csv` files into the `data/` directory.

### 2\. Reproduce Full Analysis

Run the scripts in numbered order to reproduce all results:

```bash
# 1. Train model and evaluate performance
python scripts/01\\\_train\\\_model.py

# 2. Generate result visualizations
python scripts/02\\\_generate\\\_figures.py

# 3. SHAP interpretability analysis
python scripts/03\\\_shap\\\_analysis.py

# 4. Supplementary analyses (permutation test, correlation matrix, etc.)
python scripts/04\\\_supplementary\\\_analysis.py

# 5. External cohort validation
python scripts/05\\\_external\\\_validation.py
```

### 3\. Custom Prediction

```python
import pandas as pd
import joblib

# Load trained model
model = joblib.load("results/models/rf\\\_final\\\_model.pkl")

# Load your data (must include the 2000 high-variance gene expression + mutation features)
your\\\_sample = pd.read\\\_csv("your\\\_sample.csv", index\\\_col=0)

# Predict sensitivity score (higher = more sensitive to PARP inhibitors)
sensitivity\\\_score = model.predict\\\_proba(your\\\_sample)\\\[:, 1]\\\[0]
print(f"PARPi Sensitivity Score: {sensitivity\\\_score:.3f}")
```

\---

## 📁 Project Structure

```
PARP-Predictor/
├── scripts/                 # Analysis scripts (run in numbered order)
│   ├── 01\\\_train\\\_model.py               # Model training and performance evaluation
│   ├── 02\\\_generate\\\_figures.py          # Result visualization generation
│   ├── 03\\\_shap\\\_analysis.py             # SHAP interpretability analysis
│   ├── 04\\\_supplementary\\\_analysis.py    # Supplementary analyses
│   └── 05\\\_external\\\_validation.py       # External cohort validation
├── results/                 # Output directory
│   ├── figures/             # Generated figures (300 DPI)
│   └── models/              # Trained model files
├── dataset/                    # Dataset directory (download separately)
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── LICENSE                  # License
```

\---

## 📝 Citation

If you use this tool in your research, please cite:

```
\\\[To be updated] Author Name, et al. Multi-omic prediction of PARP inhibitor response in high-grade serous ovarian cancer. \\\[Journal Name], 2026.
```

\---

## 🤝 Contributing

Issues and pull requests are welcome.

\---

## ⚖️ License

Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.

