<div align="center">

# 🏙️ Real Estate Investment Advisor

**Predict whether an Indian property is a good investment, and what it could be worth in 5 years.**

![Python](https://img.shields.io/badge/Python-3.12-1F6F68?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-E0A030?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-models-5B9BC4)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0E3B3F?logo=mlflow&logoColor=white)

### [▶ Open the live app](https://real-estate-investment-advisor-mn6qws3xyo8wytcttrg8ci.streamlit.app/)

</div>
## Overview

Buying property is a big decision, and most buyers have little way to judge long-term returns. This project turns a dataset of **250,000 Indian property listings** into two models:

| Question | Model type | Output |
|---|---|---|
| Is this a good investment? | Classification (XGBoost) | Yes or No, with a confidence score |
| What will it be worth in 5 years? | Regression (XGBoost) | Estimated price with a likely range |

A Streamlit app wraps both models: enter a property and the verdict and forecast update live. MLflow tracks every training run.

> **Important:** the dataset contains no price history, and its features are statistically independent of price (it appears synthetic). The 5-year price and the "good investment" label are therefore **engineered from documented domain rules**, not observed market outcomes. See [Limitations](#limitations).

## Features

- **Live predictions:** the Yes/No verdict, confidence bar and 5-year forecast update as you change any input.
- **Price path and range:** expected price year by year, with a shaded likely range.
- **Explainability:** infrastructure score breakdown, growth drivers and feature importance for both models.
- **Market explorer:** filter by state, city, BHK, type, price and size, with heatmaps and charts.
- **Model comparison:** three classifiers and three regressors compared side by side.
- **Experiment tracking:** all runs logged in MLflow.

<table>
<tr>
<td width="50%"><img src="docs/screenshots/02_why_this_result.png" alt="Why this result"></td>
<td width="50%"><img src="docs/screenshots/03_explore_market.png" alt="Explore the market"></td>
</tr>
<tr>
<td align="center"><sub>Price path, infrastructure breakdown and growth drivers</sub></td>
<td align="center"><sub>Market explorer with filters, donut and grouped charts</sub></td>
</tr>
</table>

## Results

Evaluated on a held-out test set of 50,000 properties (20% split).

**Classification: Good Investment**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.858 | 0.762 | 0.717 | 0.739 | 0.931 |
| Random Forest | 0.877 | 0.786 | 0.768 | 0.777 | 0.948 |
| **XGBoost** | **0.880** | **0.789** | **0.780** | **0.784** | **0.950** |

**Regression: price after 5 years (lakhs)**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 23.59 | 18.11 | 0.9830 |
| Random Forest | 19.65 | 14.01 | 0.9882 |
| **XGBoost** | **18.48** | **13.00** | **0.9896** |

These scores measure how well the models recover the rule-based targets plus noise. They are not evidence of real-world market accuracy.

![Model scores](docs/screenshots/04_model_scores.png)

## How it works

1. **Preprocessing:** duplicates and missing values handled, impossible floor numbers capped, price per sq ft recomputed in rupees, outliers flagged and winsorised.
2. **Feature engineering:** an infrastructure score (schools, hospitals, transport, amenities, security), school density, amenity count, price relative to the city median, age buckets, tier-1 city flag and more.
3. **Targets:** yearly growth is built from rules: a 5% base, plus bonuses for tier-1 cities, infrastructure, under-construction projects and prices below the city median, minus an age penalty, plus random noise. The 5-year price compounds that growth. A property is a **good investment** if growth is at least 7.5% a year and its price per sq ft is within 110% of the city median (28% of listings qualify).
4. **Modelling:** 80/20 stratified split, targets excluded from the features to prevent leakage, best models selected by F1 (classification) and RMSE (regression).
5. **App:** Streamlit interface that rebuilds the training features from the form inputs and queries the saved models.

All rule constants are at the top of `src/preprocess.py`.

## Run it locally

Requires **Python 3.11 or newer** (3.12 recommended). On Windows, keep the folder path short (for example `C:\rea`) to avoid path-length errors during installation.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO

python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app/app.py
```

The app opens at `http://localhost:8501`. Pre-trained models and a 60,000-row dataset sample are included, so no training is needed.

### Re-run the full pipeline (optional)

Place the original `india_housing_prices.csv` in `data/`, then:

```bash
pip install -r requirements-dev.txt
python src/preprocess.py     # clean data, engineer features and targets
python src/eda.py            # 20 EDA questions -> eda_outputs/
python src/train.py          # train models, log to MLflow, save the best
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Try an example

On **Check a property**, set Maharashtra, Pune, the first locality, Apartment, 3 BHK, 1500 sq ft, then compare:

| | Strong property | Weak property |
|---|---|---|
| Asking price | 115 lakhs | 200 lakhs |
| Year built | 2024 | 1995 |
| Floors (building / floor) | 15 / 6 | 5 / 2 |
| Status | Under_Construction | Ready_to_Move |
| Schools / hospitals nearby | 8 / 8 | 2 / 2 |
| Public transport | High | Low |
| Security / amenities | On / all five | Off / none |
| **Result** | **Yes**, about ₹1.92 Cr (10.8% a year) | **No**, about ₹2.44 Cr (4.1% a year) |

## Project structure

```
├── app/
│   ├── app.py                 # Streamlit app
│   └── app_classic.py         # earlier tab-based layout (reference)
├── src/
│   ├── preprocess.py          # cleaning, features, targets
│   ├── eda.py                 # 20 EDA questions
│   ├── train.py               # training, evaluation, MLflow
│   └── modeling.py            # portable model save/load
├── models/                    # saved models, metrics, feature importance
├── data/app_sample.csv        # 60k-row sample used by the app
├── eda_outputs/               # EDA charts and findings
├── docs/screenshots/          # images used in this README
├── .streamlit/config.toml     # app theme
├── requirements.txt           # app dependencies
└── requirements-dev.txt       # extra packages for EDA and training
```

The full cleaned dataset (88 MB) and the raw file are not stored in the repository. Run `src/preprocess.py` to regenerate the cleaned data.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub, including `.streamlit/` and `models/`.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) and choose **Create app**.
3. Select the repository and branch, and set the main file path to `app/app.py`.
4. In **Advanced settings**, pick **Python 3.12**, then deploy.

`requirements.txt` pins `scikit-learn==1.8.0` and `xgboost==3.4.1`, the versions the models were trained with. Each push to the repository redeploys the app.

## Tech stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Plotly · Streamlit · MLflow · Matplotlib · Seaborn

## Limitations

1. **The dataset appears synthetic.** Distributions are near-uniform and features such as size, location and amenities show almost no relationship with price, so the data cannot teach real market behaviour.
2. **The targets are rule-generated.** The 88% classifier accuracy shows how well the models recover the rules plus noise. The regression R² is high mainly because the future price is the price you enter times a growth factor.
3. The crime-rate analysis from the project brief is not possible (no such column), the "Sold" availability status never occurs, and "price trends" use `Year_Built` because there are no transaction dates.
4. The likely range in the app is the 10th to 90th percentile of test errors, not a market-risk estimate.
5. This project is a demonstration and is **not financial advice**. For production use, replace the engineered targets with observed historical prices and retrain.

## Use cases

- Investors screening listings for long-term return potential.
- Buyers comparing areas by infrastructure and relative valuation.
- Real estate platforms automating first-pass investment scoring.
