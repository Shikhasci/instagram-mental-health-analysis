# Instagram Usage Patterns & Mental Health Outcomes
### A Statistical Inference and Prediction Study
**Data 210P | University of California, Irvine | Winter 2026**

---

## Overview

This project investigates statistical associations between Instagram usage 
patterns and three outcome:perceived stress score, self-reported happiness, 
and daily active Instagram minutes using 1,547,896 synthetic user profiles 
spanning 58 variables (SMUBL dataset, Kaggle 2025).

A dual inferential and predictive framework is applied across parametric and 
tree ensemble model families, with SHAP decompositions providing interpretable 
feature attribution.

---

## Key Findings

| Finding | Result |
|---|---|
| Instagram behavior → Stress variance | R² = 0.714 (OLS) |
| Lifestyle/demographic contribution | ΔR² = 0.000 |
| Passive vs Active use ratio (stress) | 2.5× (β* 0.637 vs 0.252) |
| Best stress classifier AUC | 0.929 (XGBoost) |
| Best daily minutes prediction | R² = 0.998 (LightGBM) |
| LASSO features retained (stress) | 9 of 25 |

---

## Repository Structure
```
├── Final_EDA.py          # Part 1: Data loading, EDA, feature engineering
├── Final_inference.py    # Part 2: OLS nested models, GLMs, interaction tests  
├── Final_prediction.py   # Part 3: Regression benchmark, classification, SHAP
├── data/                 # Place instagram_usage_lifestyle_1million.csv here
└── outputs/              # Generated figures and tables saved here
```

---

## Dataset

**SMUBL — Social Media User Behavior & Lifestyle**  
1,547,896 synthetic Instagram user profiles | 58 variables | CC0 License  
Source: [Kaggle — rockyt07](https://www.kaggle.com/datasets/rockyt07/social-media-user-analysis)

> The dataset is not included in this repository due to file size.  
> Download it from Kaggle and place the CSV in the `/data` folder before running.

---

## Setup & Usage

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap statsmodels pyarrow
```

### Run order
```bash
python Final_EDA.py          
python Final_inference.py    
python Final_prediction.py   
```

> **macOS / Apple Silicon note:** If you get a `libomp.dylib` error with LightGBM or XGBoost:
> ```bash
> brew install libomp
> pip install lightgbm xgboost --upgrade --force-reinstall
> ```

---

## Methods Summary

| Component | Approach |
|---|---|
| Feature engineering | Passive Use Index, Active Use Index (VIF reduction), log1p transforms |
| Inferential models | OLS (nested, HC3 robust SEs), Poisson/NB2 GLM, Ordered Logistic, interaction tests |
| Predictive models | OLS, Ridge, LASSO, Elastic Net, Random Forest, XGBoost, LightGBM |
| Classification | Logistic Regression, RF, XGBoost, LightGBM (high-stress binary, PSS ≥ 30) |
| Explainability | SHAP TreeExplainer (global importance, beeswarm, dependence plots) |
| Effect size thresholds | \|β*\| > 0.05, ΔR² > 0.01, \|ρ\| > 0.10 (at n = 1.5M, p-values are uninformative) |

---

*Shikha Patel | UC Irvine | Winter 2026*
