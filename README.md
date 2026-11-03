# Real Estate Investment Advisor

Predicts whether an Indian property is a **Good Investment** (classification) and its **estimated price after 5 years** (regression), with a Streamlit app and MLflow experiment tracking.

## Quick start
```bash
pip install -r requirements-dev.txt   # training + EDA (use requirements.txt for the app only)
python src/preprocess.py      # cleans data, engineers features + targets -> data/cleaned_housing.csv
python src/eda.py             # 20 EDA questions -> eda_outputs/ (charts + eda_findings.md)
python src/train.py           # trains 3+3 models, logs to MLflow, saves best to models/
streamlit run app/app.py      # launch the app
mlflow ui --backend-store-uri sqlite:///mlflow.db    # compare experiments
```
(`./run_all.sh` does everything.) Pre-trained models and the cleaned dataset are already included.

## Deploying the app
The repo is deployment-ready: `requirements.txt` holds only app dependencies, `data/app_sample.csv` is a 60,000-row sample used by the app, and the full CSVs are git-ignored. Push to GitHub and deploy on Streamlit Community Cloud with main file `app/app.py` and Python 3.12. Re-running `src/train.py` requires the full data files.

## Project structure
| Path | Purpose |
|---|---|
| `src/preprocess.py` | Cleaning, feature engineering, target creation |
| `src/eda.py` | Answers the 20 EDA questions |
| `src/train.py` | Training, evaluation, MLflow logging |
| `app/app.py` | Streamlit app (Predict / Market explorer / Model performance / About) |
| `data/cleaned_housing.csv` | Cleaned + engineered dataset (deliverable) |
| `models/` | Saved models, metrics, feature importance |
| `eda_outputs/` | EDA charts and written findings |

## Methodology
**1. Preprocessing.** 250,000 rows, no missing values or duplicates (pipeline still imputes/de-duplicates for robustness). Fixes: `Floor_No` capped at `Total_Floors` (46 % of rows violated this); `Price_per_SqFt` recomputed in INR (the original is lakhs/sqft rounded to 2 dp); outliers flagged with the IQR rule and price/sqft winsorised at 1st/99th percentile. Numeric features are standardised; categoricals one-hot encoded.

**2. Feature engineering.** Amenity count and flags, School Density Score, Hospital Score, Transport Score, composite **Infra Score** (0-1), Floor Ratio, Size per BHK, Age bucket, city/locality median price per sq ft, price relative to city median, tier-1 city flag.

**3. Targets (domain rules - read this).** The dataset has no price history, so both targets are engineered:
- Yearly growth = 5 % base + 1.5 % tier-1 city + 3 % x Infra Score + 1 % under-construction - 2 % x age/35 + 5 % x undervaluation vs city median + N(0, 1 %) noise, clipped to 0-20 %.
- `Future_Price_5Y` = price x (1 + growth)^5.
- `Good_Investment` = growth >= 7.5 % **and** price/sqft <= 110 % of city median (28 % of properties).
All constants are at the top of `src/preprocess.py`.

**4. Models.** 80/20 stratified split. Targets and anything derived from them are excluded from the features (no leakage). Best model chosen by F1 (classification) and RMSE (regression).

## Results (50,000-row test set)
**Classification - Good Investment**
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LogisticRegression | 0.858 | 0.762 | 0.717 | 0.739 | 0.931 |
| RandomForest | 0.877 | 0.785 | 0.768 | 0.777 | 0.948 |
| XGBoost | 0.880 | 0.789 | 0.780 | 0.784 | 0.950 |

**Regression - 5-year price (lakhs)**
| Model | RMSE | MAE | R2 |
|---|---|---|---|
| LinearRegression | 23.59 | 18.11 | 0.9830 |
| RandomForest | 19.65 | 14.01 | 0.9882 |
| XGBoost | 18.48 | 13.00 | 0.9896 |

Best: **XGBoost** (classifier), **XGBoost** (regressor).

## Findings (full list in `eda_outputs/eda_findings.md`)
- Prices (10-500 lakhs) and sizes (500-5000 sq ft) are close to uniformly distributed.
- Size, schools, hospitals, parking, furnishing, facing, amenities and city show **no meaningful relationship with price**; the largest raw-feature correlation is about 0.004. City averages are within ~3 % of each other.
- BHK, owner type and availability are evenly split.
- Investment potential differs only through the engineered rules: higher Infra Score, better transport, younger property and lower price vs city median raise appreciation.

## Important limitations
1. **Data appears synthetic** (uniform distributions, independent features), so it cannot teach real market behaviour.
2. **Targets are rule-generated.** Accuracy (88.0%) measures how well models recover the rules plus noise, not real-world predictive power. Regression R2 is high mainly because future price = the price you enter x a growth factor.
3. The brief's crime-rate analysis is impossible (no such column); `Sold` availability status does not occur; "price trends" use `Year_Built` since there are no transaction dates.
4. The app's 80 % range is the empirical 10th-90th percentile of test residuals, not a market-risk estimate. Not financial advice.

To make this production-grade: replace the engineered targets with observed historical prices and re-run `train.py`.

## Use cases
Investor screening of listings; buyers comparing areas by infrastructure and valuation; real estate platforms automating first-pass investment scoring.
