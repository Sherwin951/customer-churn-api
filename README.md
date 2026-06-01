# 📊 Customer Churn Predictor + REST API

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-black?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Level](https://img.shields.io/badge/Level-Intermediate-yellow)

An intermediate ML project that predicts customer churn for a telecom company using **Random Forest**, **Gradient Boosting**, and **Logistic Regression** — then serves the best model through a **Flask REST API**.

---

## Overview

Two-part project:
1. **`train.py`** — Trains and evaluates three models on a synthetic telecom dataset, saves the best
2. **`app.py`** — Flask REST API that serves real-time churn predictions with risk level scoring

---

## Features

- Generates a realistic 2,000-record synthetic telecom churn dataset with probability-weighted labels
- Trains three models via sklearn `Pipeline` (Imputer → Scaler → Classifier):
  - Random Forest (n_estimators=200, class_weight='balanced')
  - Gradient Boosting (n_estimators=100, learning_rate=0.1)
  - Logistic Regression (C=0.5, class_weight='balanced')
- Evaluates with accuracy, ROC-AUC, StratifiedKFold CV, and classification report
- Generates and saves:
  - `eda.png` — 3-panel EDA: churn distribution, tenure histogram, monthly charges boxplot
  - `feature_importance.png` — Random Forest feature importances
- Flask REST API with:
  - `POST /predict` — returns prediction, churn probability, and risk level
  - `GET /` — API documentation and example curl command

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| scikit-learn | Random Forest, Gradient Boosting, Logistic Regression, Pipeline |
| Flask | REST API |
| Pandas | Data manipulation |
| Matplotlib | EDA visualizations |
| NumPy | Numerical operations |
| joblib | Model serialization |

---

## Project Structure

```
04_customer_churn_api/
├── train.py            # Model training script
├── app.py              # Flask REST API
├── requirements.txt    # Dependencies
├── model/              # Saved artifacts (generated after training)
│   ├── churn_model.pkl
│   └── features.pkl
└── *.png               # Generated EDA visualizations
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Sherwin951/customer-churn-api.git
cd customer-churn-api
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```

### 4. Start the API
```bash
python app.py
```

The API will be running at `http://localhost:5000`

---

## API Usage

### GET /
Returns API documentation and example usage.

### POST /predict
**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 3,
    "monthly_charges": 89.5,
    "total_charges": 268.5,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "senior_citizen": 0,
    "dependents": 0,
    "num_services": 3,
    "paperless_billing": 1
  }'
```

**Response:**
```json
{
  "prediction": "Churn",
  "churn_probability": 0.7821,
  "risk_level": "High"
}
```

### Risk Levels
| Probability | Risk Level |
|-------------|------------|
| ≥ 0.65 | 🔴 High |
| 0.35 – 0.65 | 🟡 Medium |
| < 0.35 | 🟢 Low |

---

## Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `tenure` | int | Months as a customer (1–72) |
| `monthly_charges` | float | Monthly bill amount ($) |
| `total_charges` | float | Total amount charged ($) |
| `contract` | string | `Month-to-month` / `One year` / `Two year` |
| `internet_service` | string | `DSL` / `Fiber optic` / `No` |
| `tech_support` | string | `Yes` / `No` / `No internet service` |
| `senior_citizen` | int | `0` or `1` |
| `dependents` | int | `0` or `1` |
| `num_services` | int | Number of subscribed services (1–7) |
| `paperless_billing` | int | `0` or `1` |

---

## Sample Training Output

```
Generated 2000 records | Churn rate: 34.2%

─────────────────────────────────────────────
  Random Forest
  Accuracy: 0.8175  |  AUC-ROC: 0.8903
  CV AUC  : 0.8847 ± 0.0201

              precision  recall  f1-score
    Retained     0.87    0.88      0.87
     Churned     0.72    0.70      0.71
```

---

## Author

**Sherwin Alle**  
M.S. Computer Science — California State University, Fresno  
[GitHub](https://github.com/Sherwin951) · [Email](mailto:alle.sherwin9999@gmail.com)
