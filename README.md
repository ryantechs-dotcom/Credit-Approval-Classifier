# Credit Card Approval Classifier

Predicts whether a credit card application is approved from applicant demographics and finances. The trained model is served through a small Flask web form.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)

## Data & preprocessing

The dataset has 1,548 applicants, each joined by ID to a binary approval label. After cleaning, 1,518 rows remain.

- **Dropped** contact fields (phone, work phone, email), which carry no signal, plus birthday and occupation, which has heavy missingness
- **Removed** 30 rows missing gender or income
- **Ordinal-encoded** education, treating "Higher education" and "Academic degree" as the same top level
- **One-hot encoded** gender and marital status, and binarized car and property ownership

This leaves 14 features: car/property ownership, children, family size, annual income, education level, days employed, gender, and marital status.

## Modeling

Unstratified 70/30 train/test split (1,062 / 456 applicants), `random_state=42`.

| Model | Test accuracy |
|---|---|
| Linear regression (sanity baseline) | n/a (R² ≈ 0) |
| **Random forest** (10 trees) | **90.4%** |

**Context.** Only about 11% of the labels are positive, so accuracy by itself overstates performance. The next revision reports precision, recall, and ROC-AUC against a majority-class baseline.

## Serving

`app.py` loads the pickled pipeline and exposes:

- `GET /` renders the application form
- `POST /predict` maps the form fields to the 14 model features and renders the prediction

```bash
pip install -r requirements.txt
python app.py        # http://127.0.0.1:5000
```

## Responsible-ML notes & next steps

Gender and marital status are used as features. In real lending they are protected characteristics under the U.S. Equal Credit Opportunity Act, so a production version would:

- Drop protected attributes and test for proxy leakage (e.g. family size)
- Audit error rates across groups and add SHAP explanations for individual decisions
- Replace plain accuracy with class-aware metrics and a calibrated threshold
- Swap pickle for a versioned model artifact and validate inputs
