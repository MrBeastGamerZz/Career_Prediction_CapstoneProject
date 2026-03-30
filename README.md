# Career Prediction System — B.Tech Final Year Capstone Project

## Project Overview
An end-to-end AI-powered Career Prediction system that predicts the best-fit IT career role for a student 
based on their academic performance, technical skills, interests, tools experience, and project portfolio.

## Model Architecture

```
Input (86 raw features)
    ↓ Feature Engineering  →  142 features
    ↓ Correlation Drop     →  126 features
    ↓ Standard Scaling + Imputation
    ↓
 ┌──────────────────────────────────────────────────┐
 │              Stacking Ensemble                   │
 │   LightGBM + XGBoost + CatBoost + Logistic Reg  │
 │                     ↓                            │
 │         Logistic Regression (Meta-Learner)       │
 └──────────────────────────────────────────────────┘
    ↓
 Top-3 Career Role Predictions + Confidence %
```

## Dataset
- 30,000 student records
- 86 input features (academic, skills, interests, tools, experience)
- 7 career domains, 30 final job roles (merged from 40 original)

## Model Performance
- Test Accuracy: ~85%
- Test Macro F1: ~83%
- Test Top-3 Accuracy: ~99%

## How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the model (if artifacts_layer/ is missing)
```bash
python train_model.py
```

### Step 3: Launch the web app
```bash
python app.py
```

Then open: http://localhost:5000

## Project Structure
```
career_app/
├── app.py                    # Flask web application
├── train_model.py            # Full training pipeline
├── career_dataset_final.csv  # Dataset (30,000 records)
├── requirements.txt
├── templates/
│   └── index.html            # Beautiful UI
└── artifacts_layer/          # Saved model artifacts
    ├── stacking_ensemble.joblib
    ├── preprocess.joblib
    ├── label_encoder.joblib
    ├── feature_order.json
    ├── role_to_domain.json
    ├── role_merge_map.json
    ├── dropped_correlated_features.json
    └── model_meta.json
```

## Career Domains Covered
1. Software Engineering (Backend, Frontend, Full-Stack, Mobile, Game Dev)
2. Data & Artificial Intelligence (Data Science, ML, Data Engineering, BI)
3. Cybersecurity (SOC, Penetration Testing, Cloud Security, GRC)
4. Cloud, DevOps & Platform Engineering (DevOps, Cloud Architect, SRE)
5. UI/UX & Product (UX Designer, Product Manager)
6. Quality Assurance & Testing (QA, Automation, Performance Testing)
7. Systems & Infrastructure (Embedded, Database, Network)
