# 🏥 Obesity Risk Classification & Predictive Analytics

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

> **Multi-class classification model predicting obesity risk from lifestyle surveys — outputs feed directly into a Power BI care coordinator risk-tiering dashboard.**

---

## Business Overview

In healthcare analytics, **preventative risk stratification** costs a fraction of post-diagnosis treatment. This project flags individuals at high obesity risk using non-invasive lifestyle surveys — enabling public health systems and insurers to proactively target intervention programs.

Most ML projects stop at model accuracy. This one goes further: output probability scores are exported as a structured dataset ready for a **Power BI care coordinator dashboard** where risk tiers (Low / Medium / High) drive real operational decisions.

---

## Technical Approach

| Step | Detail |
|------|--------|
| **Data Source** | ObesityDataSet.csv - 2,111 records, 17 features |
| **Preprocessing** | One-Hot Encoding, Label Encoding, MinMaxScaler |
| **Class Imbalance** | SMOTE - synthetic oversampling for minority weight classes |
| **Models Trained** | Random Forest, Gradient Boosting, Decision Tree, KNN |
| **Evaluation** | Accuracy, Confusion Matrix, ROC-AUC per class |
| **Explainability** | SHAP values - feature contribution per prediction |

---

## Key Findings

- Family History is the #1 Risk Driver - 3.2x higher obesity probability vs. no family history
- Transportation as a Lifestyle Proxy - Automobile users showed 42% higher obesity rates vs. walkers
- Random Forest Won - Best overall; Gradient Boosting had highest binary AUC
- SMOTE Impact - Improved minority-class recall by 28%, reducing under-diagnosis risk
- SHAP Insight - family_history_with_overweight, FCVC, and NCP were the top 3 features

---

## What Makes This Stand Out

| Feature | Standard Notebook | This Project |
|---------|-----------------|--------------|
| Output | Class label only | Probability score per class |
| Explainability | None | SHAP waterfall + summary plots |
| Pipeline | Raw script | sklearn.Pipeline - reproducible |
| BI Integration | None | CSV export -> Power BI risk dashboard |
| Imbalance Handling | Basic | SMOTE with k-neighbors tuning |

---

## Tools & Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML | Scikit-Learn, Imbalanced-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Seaborn, Matplotlib |
| BI Output | CSV -> Power BI / Tableau |

---

## Project Structure

```
Obesity-Risk-Classification/
├── data/
│   └── ObesityDataSet.csv
├── notebooks/
│   └── Obesity_Risk_Analysis.ipynb
├── outputs/
│   ├── risk_scores.csv
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/omkarpallerla/Obesity-Risk-Classification.git
cd Obesity-Risk-Classification
pip install -r requirements.txt
jupyter notebook notebooks/Obesity_Risk_Analysis.ipynb
```

---

## Model Results

| Model | Accuracy | Macro AUC | Notes |
|-------|----------|-----------|-------|
| **Random Forest** | **94.2%** | **0.97** | Best overall |
| Gradient Boosting | 92.8% | 0.96 | Best binary AUC |
| Decision Tree | 87.1% | 0.91 | Most interpretable |
| KNN | 83.4% | 0.88 | Baseline |

---

Built by Omkar Pallerla - MS Business Analytics, ASU - BI Engineer - Azure | GCP | Databricks Certified
