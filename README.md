# 📡 Telco Customer Churn Analysis

> End-to-end machine learning project to identify at-risk telecom customers and drive targeted retention strategies.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Analysis Pipeline](#analysis-pipeline)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Business ROI](#business-roi)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Visualisations](#visualisations)
- [Author](#author)

---

## Overview

Customer churn is one of the most costly problems in the telecom industry — acquiring a new customer costs 5–7× more than retaining an existing one. This project builds a full end-to-end churn prediction pipeline:

- Exploratory data analysis across demographic, service, billing, and contract features
- Statistical significance testing (T-tests + Chi-Square + Cramér's V)
- Multivariate segment analysis to identify compound high-risk profiles
- Feature engineering with domain-driven interaction variables
- Machine learning modelling with Logistic Regression, Random Forest, and XGBoost
- Threshold optimisation for a recall-focused business objective
- SHAP explainability at both global and individual customer level
- Business ROI simulation to quantify campaign value

---

## Dataset

**Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Rows | 7,043 |
| Features | 21 |
| Target | `Churn` (Yes / No) |
| Churn Rate | ~26% |

**Feature Groups:**

| Group | Features |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |

---

## Project Structure

```
telco-churn-analysis/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── telco_customer_churn.ipynb              # Main analysis notebook
├── PROJECT_REPORT.pdf                       # Detailed project report
├── README.md                               # This file
│
└── outputs/                                # Saved visualisation PNGs
    ├── univariate_categorical.png
    ├── univariate_numerical.png
    ├── churn_rate_categorical_features.png
    ├── Churn_rate_num_features.png
    ├── KDE_PLOT_Churners_vs_Non-Churners.png
    ├── cramers_v_churn.png
    ├── Correlation_Matrix.png
    ├── Scatter_tenure_vs_totalCharges.png
    ├── Pairplot_Num_feature_by_churn.png
    ├── Contract_×_InternetService_churn_heatmap.png
    ├── Churn_Rate_(%)_Contract_x_PaymentMethod.png
    ├── Churn Rate (%) - Payment Method x Internet Service.png
    ├── Churn Rate by Tenure Group & Contract Type.png
    ├── Churn Rate by Number of Add-on Services Subscribed.png
    ├── ROC Curve - Model Comparison.png
    ├── Confusion matrix.png
    ├── XGBoost - Top 15 Feature Importances.png
    ├── XGBoost — Precision, Recall & F1 vs Decision Threshold.png
    ├── SHAP Summary Plot — Feature Impact on Churn Prediction.png
    ├── Mean absolute SHAP Value — Global Feature Importance.png
    ├── SHAP Dependence features.png
    ├── Top Features Differentiating Churners from Non-Churners.png
    └── business_roi_analysis.png
```

---

## Analysis Pipeline

```
Raw Data
  └─▶ Data Understanding & Cleaning
        └─▶ Univariate Analysis
              └─▶ Bivariate Analysis (Categorical + Numerical vs Churn)
                    └─▶ Statistical Tests (T-test, Chi-Square, Cramér's V)
                          └─▶ Multivariate Analysis (Compound Segments)
                                └─▶ Feature Engineering
                                      └─▶ Modelling (LR / RF / XGBoost)
                                            └─▶ Threshold Optimisation
                                                  └─▶ Hyperparameter Tuning
                                                        └─▶ SHAP Explainability
                                                              └─▶ Business ROI Analysis
```

---

## Key Findings

### Highest-Risk Customer Profile
> Month-to-month contract + Fiber Optic internet + Electronic Check payment + tenure < 12 months

### Compound Segment Churn Rates

| Segment | Churn Rate |
|---|---|
| Month-to-month + Fiber Optic | ~54.6% |
| Month-to-month + Electronic Check | ~53.7% |
| Tenure 0–12 months + No Online Security | ~High |
| Senior Citizen + Month-to-Month | ~High |
| 1 Add-on Service only | ~45% |

### Key Churn Drivers (SHAP)

1. **Contract Type** — Month-to-month = highest risk; two-year = near zero churn
2. **Tenure** — Churn drops sharply after 12 months; first year is the danger zone
3. **Monthly Charges** — Higher bills increase price sensitivity and churn likelihood
4. **HighRisk Flag** — Fiber Optic + month-to-month compound effect is a strong signal
5. **Payment Method** — Electronic check users churn more; auto-pay creates passive loyalty
6. **Support Services** — Customers without OnlineSecurity or TechSupport churn more

---

## Model Performance

Three models were trained and evaluated:

| Model | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.511 | 0.778 | 0.617 | 0.848 |
| Random Forest | 0.555 | 0.687 | 0.614 | 0.840 |
| **XGBoost (default)** | 0.533 | 0.781 | 0.633 | **0.843** |
| **XGBoost (threshold = 0.373)** | **0.478** | **0.850** | **0.612** | **0.843** |

> XGBoost at a recall-optimised threshold of 0.373 was selected as the final model. It catches 85% of churners while maintaining a usable precision level for retention campaigns.

**Imbalance handling:**
- Logistic Regression & Random Forest → `class_weight='balanced'`
- XGBoost → `scale_pos_weight = non-churners / churners`

---

## Business ROI

Using a 30% post-outreach retention assumption and ₹500 cost per customer contact:

| Metric | Value |
|---|---|
| Churners caught by model | ~85% of actual churners |
| Total customers flagged | Churners caught + False alarms |
| Revenue protected | Saved customers × Estimated CLV |
| Campaign break-even retention rate | ~17% |
| ROI multiple at 30% retention | 0.8 return per ₹1 spent |

> The model becomes profitable at a retention rate as low as 17%, making it viable even under conservative campaign assumptions.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Statistical testing | `scipy.stats` |
| Machine learning | `scikit-learn`, `xgboost` |
| Explainability | `shap` |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Srishh-ti/Telco-Customer-Churn.git
cd telco-churn-analysis
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost shap
```

### 3. Run the notebook

```bash
jupyter notebook telco_customer_churn.ipynb
```

> Make sure the dataset CSV is in the same directory as the notebook, or update the file path in the `pd.read_csv()` cell.

---

## Visualisations

| Analysis Stage | Chart |
|---|---|
| Univariate — Categorical | Bar charts with % annotations for all 17 categorical features |
| Univariate — Numerical | Histograms with skewness annotations |
| Bivariate — Categorical | Churn rate bar charts with avg churn baseline |
| Bivariate — Numerical | Box plots + KDE plots by churn class |
| Statistical Tests | Cramér's V bar chart, T-test console output |
| Multivariate | Heatmaps (Contract × Internet, Contract × Payment, Payment × Internet), Grouped bar by tenure, Services vs churn rate |
| Modelling | ROC curves, Confusion matrices, Feature importances |
| Threshold Tuning | Precision-Recall-F1 vs threshold curve |
| SHAP | Summary dot plot, Bar plot, Dependence plots, Waterfall plots |
| Business ROI | Campaign waterfall + ROI sensitivity curve |

---

## Author

**SRISHTI**

Dataset: IBM Telco Customer Churn Dataset  
Best-performing model: XGBoost

---

*Feel free to fork, star ⭐, or raise an issue if you have suggestions!*
