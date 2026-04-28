# 🧠 Epileptic Seizure Prediction — Effect of Preprocessing, Regularization, and Class Imbalance

> **Data Mining — Semester Major Assignment**  
> Author: **Ishfaq Ghani** ([msds.256409240@imsciences.edu.pk](mailto:msds.256409240@imsciences.edu.pk))  
> Institute of Management Sciences (IMSciences), Peshawar

---

## 📌 Overview

This project systematically investigates how **preprocessing pipeline ordering**, **model complexity**, **regularization strategies**, and **class-imbalance handling** affect generalization performance of logistic regression classifiers for EEG-based epileptic seizure prediction.

We evaluate **3 EEG datasets** (4 variants), **2 preprocessing pipelines**, **3 regularization methods**, and **4 class-balancing strategies** — totaling **~50+ controlled experiments**.

---

## 🎯 Research Questions

1. **Q1:** Does the *ordering* of preprocessing operations measurably affect performance?
2. **Q2:** Which regularization (L1, L2, Elastic Net) generalizes best across heterogeneous EEG datasets?
3. **Q3:** Does Elastic Net consistently outperform L1 and L2 individually?
4. **Q4:** How does class-imbalance handling interact with regularization choice?

---

## 🏆 Key Findings

| Finding | Evidence |
|---------|----------|
| **Preprocessing order matters** | Pipeline B beats A by **+0.310 F1** on average (4/4 datasets) |
| **L1 generalizes best** | Wins on **4/4 datasets**, lowest cross-dataset std |
| **Elastic Net does not consistently win** | Beats L1 on 0/4, L2 on 1/4 — L1 alone is sufficient |
| **Class balancing is critical** | F1 improves **12×** on imbalanced data (0.03 → 0.35) |

### 🥇 Best Configuration
**Pipeline B + L1 + Class Weighting** → F1 up to **0.9042**, PR-AUC up to **0.9691**

---

## 📊 Datasets

| ID | Dataset | Samples | Features | Imbalance | Source |
|----|---------|---------|----------|-----------|--------|
| D1 | UCI Epileptic Seizure Recognition | 11,500 | 178 | 1:4 | [UCI ML Repo](https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition) |
| D2 | Bonn University EEG | 400 | 4,097 | 1:3 | [Bonn EEG](https://www.upf.edu/web/ntsa/downloads) |
| D3 | CHB-MIT Scalp EEG (balanced) | 3,546 | 5,888 | 1:1 | [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) |
| D3_imb | CHB-MIT (imbalanced subset) | 1,462 | 5,888 | 1:3 | [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) |

---

## 🔬 Methodology

### Preprocessing Pipelines

**Pipeline A:** `Z-score Normalize → Bandpass Filter (0.5–40 Hz) → ANOVA F-test Feature Selection (k=50)`

**Pipeline B:** `Statistical/FFT Feature Extraction → StandardScaler → PCA (n=10)`

### Model
Logistic Regression (scikit-learn) with three regularization variants:
- **L1 (Lasso):** sparsity-inducing
- **L2 (Ridge):** weight shrinkage  
- **Elastic Net:** convex combination (α = 0.5)

### Class-Imbalance Strategies
- No balancing (baseline)
- SMOTE oversampling
- Random undersampling
- Class weighting (`class_weight='balanced'`)

### Evaluation
Stratified 80/20 train-test split (seed=42). Metrics: Accuracy, F1, Precision, Recall, **PR-AUC** (primary for imbalance), ROC-AUC.

---

## 📈 Results Summary

### Per-Dataset Best Performance (Pipeline B + L1)

| Dataset | F1 | PR-AUC | Best Strategy |
|---------|------|--------|---------------|
| D1 (UCI) | **0.9042** | **0.9691** | None needed |
| D2 (Bonn) | **0.8649** | **0.9815** | None needed |
| D3 (CHB-MIT bal) | 0.6176 | 0.7573 | None needed |
| D3_imb (CHB-MIT) | **0.3543** | 0.2462 | Class Weighting |

### Imbalance Handling Impact (D3_imb)

| Strategy | F1 | Precision | Recall |
|----------|------|-----------|--------|
| No Balancing | 0.030 | 1.000 | 0.015 |
| SMOTE | 0.337 | 0.230 | 0.636 |
| Undersampling | 0.349 | 0.243 | 0.621 |
| **Class Weighting** | **0.354** | 0.239 | **0.682** |

---

## 📁 Repository Structure

```
seizure-prediction-ml/
├── README.md                              # This file
├── Seizure_Prediction_Notebook.ipynb     # Main Colab notebook (all code)
├── report/
│   └── Seizure_Prediction_Report.pdf    # IEEE-format report
├── presentation/
│   └── Seizure_Prediction_Presentation.pptx
├── results/
│   ├── baseline_results.csv
│   ├── regularization_complete.csv
│   ├── stability_analysis.csv
│   ├── sparsity_analysis.csv
│   ├── imbalance_results.csv
│   ├── interaction_analysis.csv
│   └── master_results.csv
└── figures/                               # 18 result figures
    ├── Q1_preprocessing_order.png
    ├── Q2_generalization.png
    ├── Q3_elasticnet.png
    ├── Q4_interaction.png
    ├── validation_curve.png
    ├── learning_curve.png
    └── ... (more figures)
```

---

## 🚀 How to Reproduce

### Option 1: Google Colab (recommended)

1. Open `Seizure_Prediction_Notebook.ipynb` in Google Colab
2. Mount your Google Drive
3. Download the 3 datasets:
   - **D1:** [UCI Kaggle mirror](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition)
   - **D2:** [Bonn EEG Kaggle mirror](https://www.kaggle.com/datasets/yasserhessein/epileptic-seizure-recognition-bonn-university)
   - **D3:** [CHB-MIT Kaggle mirror](https://www.kaggle.com/datasets/abhishekinnvonix/the-chbmit-scalp-eeg-database)
4. Update dataset paths in notebook to match your Drive layout
5. Run all cells sequentially (top to bottom)

### Option 2: Local Python

```bash
# Clone repo
git clone https://github.com/<your-username>/seizure-prediction-ml.git
cd seizure-prediction-ml

# Create virtual env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook Seizure_Prediction_Notebook.ipynb
```

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
scipy
python-pptx
```

Install via:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn scipy python-pptx
```

---

## 📚 References

1. R. G. Andrzejak et al., "Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity," *Phys. Rev. E*, 2001.
2. A. H. Shoeb, "Application of machine learning to epileptic seizure onset detection and treatment," PhD thesis, MIT, 2009.
3. N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, 2002.
4. R. Tibshirani, "Regression shrinkage and selection via the lasso," *J. Royal Stat. Soc. B*, 1996.
5. H. Zou and T. Hastie, "Regularization and variable selection via the elastic net," *J. Royal Stat. Soc. B*, 2005.
6. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, 2011.
7. S. Wong et al., "EEG datasets for seizure detection and prediction—A review," *Epilepsia Open*, 2023.
8. H. He and E. A. Garcia, "Learning from imbalanced data," *IEEE TKDE*, 2009.

---

## 📜 License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Course:** Data Mining
- **Institution:** Institute of Management Sciences (IMSciences), Peshawar
- **Submitted:** April 2026

---

⭐ If you find this helpful, please star the repo!
