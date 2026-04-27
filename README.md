# 🤖 AntiGravity — ML Analysis & Explainability System

> **Train. Compare. Explain.**
> *An end-to-end machine learning platform that doesn't just predict — it understands.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green?style=flat-square)
![License](https://img.shields.io/badge/License-Educational-lightgrey?style=flat-square)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Metrics & Evaluation](#-metrics--evaluation)
- [Explainability with SHAP](#-explainability-with-shap)
- [Local Setup](#-local-setup)
- [Project Structure](#-project-structure)
- [Engineering Highlights](#-engineering-highlights)
- [Design Tradeoffs](#-design-tradeoffs)
- [Evaluation Integrity](#-evaluation-integrity)
- [Known Limitations](#-known-limitations)
- [Roadmap](#-roadmap)
- [Example Use Cases](#-example-use-cases)
- [Author](#-author)

---

## 🎯 Overview

**AntiGravity** is a full-stack machine learning system designed for both **prediction** and **interpretability**. Upload any CSV dataset, and AntiGravity automatically preprocesses your data, trains multiple competing models, selects the best performer, and explains model predictions using SHAP and feature-based interpretation — all through a clean, interactive Streamlit interface.

Built as a second-year CS project at VIT Pune, AntiGravity demonstrates that knowing *why* a model makes a decision is just as valuable as the decision itself.

**What this project showcases:**

✅ **Automated ML Pipeline** — Preprocessing → Training → Evaluation → Selection, with minimal manual intervention  
✅ **Multi-Model Competition** — Multiple algorithms trained in parallel and compared objectively  
✅ **SHAP Explainability** — Feature importance visualized for every prediction  
✅ **Dual Problem Support** — Handles Classification and Regression automatically  
✅ **Heuristic-based task detection** — Determines classification vs. regression from target column  
✅ **Clean Architecture** — Separated concerns across `app/`, `src/`, and `models/`  

---

## ✨ Key Features

### 🔹 Mode 1 — Prediction System

- Supports **Classification** and **Regression** tasks
- Trains and compares multiple models simultaneously:
  - **Classification:** Logistic Regression, SVM, Random Forest
  - **Regression:** Linear, Ridge, Lasso, Random Forest Regressor
- Displays side-by-side model performance comparison
- Provides SHAP-based explanations:
  - Mode 1: Per-prediction (local) explanation
  - Mode 2: Global feature importance analysis

### 🔹 Mode 2 — Dataset Analysis & Explainability

- Upload **any CSV dataset** — no prior setup required
- Largely automated pipeline:
  - Missing value handling (rows with missing values removed)
  - Categorical feature encoding
  - Feature scaling & normalization
  - Model training & evaluation using train-test split
  - Best model selection
- **Auto-detects task type:** binary classification, multiclass classification, or regression

### 📊 Advanced Visualizations

- Confusion Matrix (raw + normalized)
- Model comparison leaderboard table
- Target distribution analysis
- SHAP feature importance plots (global & per-prediction)

---

## 🏗 Architecture

```
┌──────────────────────────────────┐
│         User (Browser)           │
└────────────────┬─────────────────┘
                 │ Streamlit UI
                 ▼
┌──────────────────────────────────┐
│         app/app.py               │  ← Entry point & routing
│         Mode 1 | Mode 2          │
└───────┬──────────────┬───────────┘
        │              │
        ▼              ▼
┌───────────┐   ┌──────────────┐
│  mode1.py │   │   mode2.py   │  ← UI layer
└─────┬─────┘   └──────┬───────┘
      │                │
      ▼                ▼
┌──────────────────────────────────┐
│            src/                  │
│  preprocess_clf / preprocess_reg │  ← Data cleaning & encoding
│  train_clf / train_reg           │  ← Model training & comparison
│  predict.py                      │  ← Inference and model prediction handling
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│          models/                 │  ← Serialized best models (.pkl)
│          data/                   │  ← Input datasets
└──────────────────────────────────┘
```

**Design Principles:**

- **Separation of Concerns** — UI, logic, and data layers are fully decoupled
- **Pluggable Models** — New algorithms can be added to `train_clf.py` / `train_reg.py` without touching the UI
- **Single Preprocessing Interface** — Both modes share the same underlying preprocessing pipeline
- **Stateless Inference** — Models are saved to disk; predictions can run independently of training

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Interactive UI & file upload |
| ML Models | Scikit-learn | Training, evaluation, selection |
| Data Processing | Pandas, NumPy | Cleaning, encoding, scaling |
| Visualization | Matplotlib, Seaborn | Charts, confusion matrices |
| Explainability | SHAP | Feature importance & prediction explanation |
| Serialization | Joblib / Pickle | Saving & loading trained models |
| Version Control | Git + GitHub | Source control & collaboration |

---

## 🔄 How It Works

```
Step 1: Upload CSV Dataset
         ↓
Step 2: Select Target Column
         ↓
Step 3: Auto Preprocessing
        ├── Remove rows with missing values
        ├── Encode categorical features
        └── Scale numerical features
         ↓
Step 4: Train Multiple Models in Parallel
         ↓
Step 5: Evaluate & Compare Performance
         ↓
Step 6: Auto-Select Best Model
         ↓
Step 7: Generate SHAP Explanation
         ↓
Step 8: Display Interactive Visualizations
```

---

## 📉 Metrics & Evaluation

### Classification Tasks

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct prediction rate |
| **Precision** | Of all positive predictions, how many were correct |
| **Recall** | Of all actual positives, how many were found |
| **F1 Score** | Harmonic mean of precision and recall |

### Regression Tasks

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error — average prediction distance |
| **R² Score** | Proportion of variance explained by the model |

### Model Comparison Table

All trained models are ranked side-by-side in a leaderboard table, making it transparent why the best model was selected.

---

## 🔍 Explainability with SHAP

SHAP (SHapley Additive exPlanations) makes the black box transparent:

- **Which features** drove each individual prediction
- **Direction of influence** — does a feature push the prediction up or down?
- **Magnitude** — how much does each feature contribute to the final output?
- **Global importance** — which features matter most across the entire dataset?

```
Prediction = base_value + SHAP(feature_1) + SHAP(feature_2) + ... + SHAP(feature_n)
```

> ⚠️ SHAP support is currently optimized for **tree-based models** (e.g., Random Forest). Linear model SHAP uses the linear explainer.

---

## 💻 Local Setup

### Prerequisites

- Python 3.9+
- pip
- (Recommended) virtualenv

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/antigravity-ml.git
cd antigravity-ml
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app/app.py
```

✅ App now running at `http://localhost:8501`

### 5. Quick Test

1. Open the app in your browser
2. Select **Mode 2 — Dataset Analysis**
3. Upload any CSV from the `data/` folder
4. Select your target column and click **Analyze**

---

## 📂 Project Structure

```
MLCP/
│
├── app/
│   ├── app.py              # Main entry point & mode selector
│   ├── mode1.py            # Prediction system UI
│   └── mode2.py            # Dataset analysis UI
│
├── src/
│   ├── train.py            # Generic training orchestration
│   ├── train_clf.py        # Classification model training
│   ├── train_reg.py        # Regression model training
│   ├── preprocess_clf.py   # Classification preprocessing pipeline
│   ├── preprocess_reg.py   # Regression preprocessing pipeline
│   └── predict.py          # Inference & SHAP explanation logic
│
├── models/                 # Saved model files (.pkl)
├── data/                   # Sample datasets for testing
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🎓 Engineering Highlights

### 1. Automated Task Detection

AntiGravity inspects the target column to determine problem type automatically:
- **≤ 10 unique values** → Classification (binary or multiclass)
- **Continuous values** → Regression

No user configuration required.

### 2. Unified Preprocessing Interface

Both modes follow consistent preprocessing principles for cleaning, encoding, and scaling.
The pipeline handles:
- Missing value handling (rows with missing values are removed)
- Label encoding & one-hot encoding
- Standard scaling before model training

### 3. SHAP Integration

```python
# Tree-based explainer (Random Forest, etc.)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Linear explainer (Logistic Regression, Ridge, etc.)
explainer = shap.TreeExplainer(best_model, X_train)
shap_values = explainer.shap_values(X_test)
```

### 4. Model Serialization

Best models are persisted to `models/` using Joblib, allowing predictions to be made without retraining on every app restart.

### 5. Streamlit State Management

`st.session_state` is used to preserve generated input values and improve user interaction flow.

---

## ⚖️ Design Tradeoffs

### Why Streamlit Instead of Flask + React?

| Aspect | Streamlit | Flask + React |
|--------|-----------|---------------|
| Development Speed | ✅ Extremely fast | ❌ Requires separate frontend |
| ML Integration | ✅ Native Python objects | ⚠️ Requires serialization layer |
| Customizability | ❌ Limited styling | ✅ Full control |
| Deployment | ✅ Single service | ❌ Two services to manage |
| Audience | ✅ Data scientists | ✅ End users / production apps |

**Decision:** Streamlit is ideal for ML-focused applications where rapid prototyping and Python-native integrations matter more than custom UI.

---

### Why Multiple Models Instead of One?

Training a single model assumes you know which algorithm suits the data. By training several in parallel and comparing objectively, AntiGravity lets the data decide — which is more rigorous and often yields better results.

---

## 🧪 Evaluation Integrity

A deliberate effort was made to ensure evaluation results are honest and reproducible:

- **Unseen test data** — All metrics are computed on a held-out test set, never on training data
- **Consistent evaluation** — The same train-test split is applied uniformly across all competing models
- **Objective selection** — The best model is chosen purely on performance metrics, with no manual override
- **Aligned outputs** — The confusion matrix and SHAP explanations correspond specifically to the selected best model, not a generic baseline

> 👉 This reflects a recent bug fix that corrected metric leakage in model comparison — evaluation integrity is not an afterthought.

---

## ⚠️ Known Limitations

- SHAP is currently best supported for tree-based models; linear model explanations are less detailed
- Large datasets (100k+ rows) may cause UI lag in Streamlit
- Class imbalance is not yet handled automatically (e.g., SMOTE, class weights)
- No support for time-series or NLP datasets

---

## 🚧 Roadmap

### Phase 1: Core System ✅
- [x] Mode 1 — Prediction system with model comparison
- [x] Mode 2 — Dataset analysis & auto pipeline
- [x] SHAP explainability integration
- [x] Confusion matrix & evaluation metrics

### Phase 2: Visualization Enhancements *(Next)*
- [ ] ROC curve & AUC visualization
- [ ] Precision-Recall curve
- [ ] Residual plots for regression
- [ ] Interactive SHAP force plots (per-prediction drill-down)

### Phase 3: User Experience
- [ ] Downloadable PDF reports (metrics + SHAP plots)
- [ ] Dataset statistics summary page
- [ ] Dark mode toggle
- [ ] Model export / download (.pkl)

### Phase 4: Advanced ML
- [ ] LIME integration (model-agnostic explanations)
- [ ] Automated hyperparameter tuning (GridSearchCV / Optuna)
- [ ] Class imbalance handling (SMOTE, class weights)
- [ ] Cross-validation folds configuration

### Phase 5: Infrastructure
- [ ] Cloud deployment (Streamlit Cloud / Hugging Face Spaces)
- [ ] API layer (Flask REST endpoints for model inference)
- [ ] Database-backed experiment tracking
- [ ] Unit tests for preprocessing & training pipelines

---

## 🧪 Example Use Cases

| Domain | Task | Target Column Example |
|--------|------|-----------------------|
| 🏥 Healthcare | Medical diagnosis prediction | `diagnosis` (benign/malignant) |
| 🏠 Real Estate | House price estimation | `sale_price` (continuous) |
| 👤 Marketing | Customer churn prediction | `churned` (yes/no) |
| 💳 Finance | Fraud detection | `is_fraud` (0/1) |
| 🎓 Education | Student performance analysis | `final_grade` (continuous) |

---

## 👨‍💻 Author

**Aditya Rana**  
Second Year Computer Science (AI) Student — VIT Pune  
*Built to explore the intersection of machine learning and interpretability.*

[![GitHub](https://img.shields.io/badge/GitHub-your--username-black?style=flat-square&logo=github)](https://github.com/your-username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)

---

## 🤝 Contributing

Contributions are welcome! Areas that need attention:

- [ ] Add pytest unit tests for preprocessing pipelines
- [ ] Improve SHAP support for SVM models
- [ ] Add example datasets to `data/` folder
- [ ] Write OpenAPI-style docstrings for `src/` modules
- [ ] Build a Dockerized deployment setup

---

## 📜 License

This project is for **educational purposes** only.

---

⭐ *If this project helped you learn ML explainability or pipeline design, consider starring the repo!*

> *Built with curiosity — because understanding a model matters more than just using it.*