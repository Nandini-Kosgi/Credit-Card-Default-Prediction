# Credit Card Default Prediction

This project builds an end-to-end **Machine Learning pipeline** to predict the probability that a customer will default on their credit card within the next two years. It uses **LightGBM**, **SMOTE**, and carefully designed **feature engineering** to handle real-world, imbalanced financial data.

---

## 🚀 Project Workflow

1. **Data Ingestion**
   - Read training/validation datasets (`credit-card-default.ipynb` for experiments; `engine.py` for the production pipeline).

2. **Data Preprocessing**
   - Handles missing values (e.g., `MonthlyIncome`, `NumberOfDependents`), caps extreme past-due counts, and normalizes skewed numeric features.

3. **Feature Engineering**
   - Creates signals such as delinquency aggregates, income-to-debt ratios, and utilization features to improve class separability.

4. **Class Imbalance Handling**
   - Balances minority default cases using **SMOTE** for fairer learning.

5. **Scaling**
   - Standardizes numeric features with **StandardScaler** to stabilize model training.

6. **Model Training**
   - **LightGBM Classifier** configured via `ML_pipeline/model_params.py` (called inside `engine.py`).
   - End-to-end flow defined in `engine.py` with modular steps from `ML_pipeline/*`.

7. **Evaluation & Outputs**
   - Metrics: **ROC-AUC, F1, Precision, Recall, Accuracy**.
   - Validation predictions saved to `output/test.csv`.

---

## 📂 Repository Structure

```
├── credit-card-default.ipynb      
├── engine.py                      
├── ML_pipeline/                   
│   └── test.csv                  
└── README.md
```

> Note: `ML_pipeline` package is expected to contain:
> - `dataset.py` (read_data), `data_splitting.py`, `data_preprocessing.py`,
> - `feature_engineering.py`, `upsampling_minorityClass.py`,
> - `scaling_features.py`, `model_params.py`, `train_model.py`, `predict_model.py`.

---

## ⚙️ Tech Stack

- **Python 3.9+**
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `imbalanced-learn`

---

## ▶️ Quickstart

```bash
# 1) Create & activate a virtual environment (example for Windows PowerShell)
py -3.11 -m venv .venv
. .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the pipeline
python engine.py
```

Outputs will be written to `output/test.csv`.

---

## 📊 Results (example description)

- Using **SMOTE + LightGBM** improved **Recall** for default class and increased **ROC-AUC** over a baseline logistic model.
- Modular pipeline yields consistent, reproducible experiments and is easy to extend (e.g., with Optuna tuning or SHAP).
- Practical impact: earlier and more accurate default detection to inform credit risk decisions.

---

## 🧭 Roadmap

- Hyperparameter tuning (Optuna/Hyperopt)
- Explainability (SHAP)
- Model registry & tracking (MLflow)
- REST API + Docker for deployment (FastAPI)
- CI/CD and batch/stream scoring

---

## 👤 Author

Nandini Kosgi 
