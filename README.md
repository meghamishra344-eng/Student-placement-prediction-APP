# 🎓 Placement Readiness — Student Placement Prediction System

A machine-learning web app that predicts whether a student will be placed, explains **why** using SHAP, and generates a personalised **action plan** showing exactly which improvements would raise their chances — plus batch scoring for an entire cohort at once.

**🔗 Live demo:** *paste your Streamlit app link here*

---

## The problem

College placement cells usually discover which students are struggling only after interviews start — when it's too late to help. This tool flags at-risk students **early**, explains the specific factors holding each one back, and quantifies how much realistic improvements (clearing backlogs, raising a coding score, taking an internship) would move their placement probability.

## Features

| | Feature | What it does |
|---|---|---|
| 🎯 | **Single-student prediction** | Enter a student's academics, skill scores, and experience → placement probability with a three-band readiness meter (At risk / Borderline / Placement-ready) |
| 🔍 | **SHAP explanations** | A per-student chart showing which factors pushed the prediction toward *Placed* or *Not Placed* |
| 🧭 | **Action plan (what-if engine)** | The model is re-run with realistic improvements applied — e.g. *"Raise Coding Score from 58 → 68: probability rises to 29.8% (+12.4 pts)"* |
| 📂 | **Batch scoring** | Upload a CSV of a whole cohort → every student scored, ranked most-at-risk first, with a downloadable results file |
| 📊 | **Model performance tab** | Hold-out metrics, confusion matrix, and feature importances — full transparency about how good the model is |
| 🎚️ | **Adjustable decision threshold** | Sidebar slider: raise the bar to flag more students for early intervention |

## The model

- **Algorithm:** Random Forest (300 trees, max depth 20, balanced class weights) inside a scikit-learn `Pipeline` with median/mode imputation, scaling, and one-hot encoding
- **Data:** 10,000 students (62% placed / 38% not placed)
- **Feature engineering:** `Academic_Avg`, `Skill_Avg`, `Total_Experience`, and a `CGPA × Skill` interaction on top of 16 raw inputs
- **Fairness:** demographic fields (gender, state, city, college type) are deliberately **excluded** — predictions use academics, skills, and experience only

### Honest evaluation

Metrics are computed on a **20% hold-out test set** (2,000 students the model never saw), then the final model is refit on all data:

| Metric | Score |
|---|---|
| Accuracy | **86.8%** |
| ROC-AUC | **0.936** |
| Precision | 88.8% |
| Recall | 90.6% |

Random Forest was benchmarked against Gradient Boosting, Extra Trees, and soft-voting ensembles and came out on top. Adding demographic features gave **no** accuracy gain — confirming they can be excluded at zero cost.

### No pickle, no version headaches

The model trains directly from the CSV at app startup (cached with `@st.cache_resource`), so there is no serialized model file and no scikit-learn / Python version-mismatch problems on deployment.

## Repository structure

```
├── app.py                                  # the Streamlit app (model + UI)
├── requirements.txt                        # dependencies
├── Student_Placement_Dataset_10000_v2.csv  # training data
├── MODELTASK.ipynb                         # EDA & model development notebook
└── .streamlit/
    └── config.toml                         # visual theme
```

## Run it locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

The first load takes ~30 seconds while the model trains and validates; after that it's cached.

## Tech stack

Python · scikit-learn · SHAP · Streamlit · pandas · matplotlib

---

*Built as an end-to-end ML project: EDA → feature engineering → model selection & honest evaluation → explainability → deployed decision-support tool.*
