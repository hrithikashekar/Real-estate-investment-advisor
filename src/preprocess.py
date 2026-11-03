"""
Step 1 - Data preprocessing, feature engineering and target creation.

Run:  python src/preprocess.py
Out:  data/cleaned_housing.csv, models/city_stats.json (lookup used by the app)

IMPORTANT - about the targets
-----------------------------
The raw dataset has NO "future price" or "appreciation" column. The project brief
asks for (a) a Good Investment label "based on domain rules" and (b) a 5-year
price forecast, so both targets are *engineered* here from transparent rules
(see `annual_growth_rate`). They are a modelling assumption, not observed market
history. Change the constants below to test your own assumptions.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "india_housing_prices.csv"
CLEAN = ROOT / "data" / "cleaned_housing.csv"
CITY_STATS = ROOT / "models" / "city_stats.json"

CURRENT_YEAR = 2025
HORIZON_YEARS = 5
SEED = 42

# ---- domain-rule assumptions (documented in README) -----------------------
BASE_GROWTH = 0.05                  # baseline yearly appreciation
TIER1_CITIES = {"Mumbai", "New Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune",
                "Kolkata", "Ahmedabad", "Gurgaon", "Noida"}
TIER1_BONUS = 0.015
INFRA_WEIGHT = 0.030                # x infrastructure score (0-1)
UNDER_CONSTRUCTION_BONUS = 0.010    # capital appreciation as project completes
AGE_PENALTY = 0.020                 # x (age / 35)
VALUE_WEIGHT = 0.050                # x undervaluation vs. city median price/sqft
NOISE_STD = 0.010                   # market uncertainty on growth rate
GOOD_INVESTMENT_GROWTH = 0.075      # label threshold: yearly growth >= 7.5 %
# ---------------------------------------------------------------------------

TRANSPORT_MAP = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
AMENITY_LIST = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
    # missing values: none in this file, but keep the pipeline robust
    num = df.select_dtypes("number").columns
    cat = df.select_dtypes(exclude="number").columns
    df[num] = df[num].fillna(df[num].median())
    for c in cat:
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    for c in cat:
        df[c] = df[c].astype(str).str.strip()

    # data-quality fix: ~46 % of rows have Floor_No > Total_Floors (impossible)
    df["Floor_Fixed"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)
    df["Floor_No"] = np.minimum(df["Floor_No"], df["Total_Floors"])

    # recompute price/sqft in INR (original column is lakhs/sqft rounded to 2 dp -> lossy)
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]

    # outlier handling: flag with IQR rule, then winsorise price/sqft at 1st/99th pct
    for col in ["Price_per_SqFt", "Size_in_SqFt", "Price_in_Lakhs"]:
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        df[f"{col}_outlier"] = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).astype(int)
    lo, hi = df["Price_per_SqFt"].quantile([.01, .99])
    df["Price_per_SqFt"] = df["Price_per_SqFt"].clip(lo, hi)
    print(f"cleaning: {n0:,} -> {len(df):,} rows")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amenity_Count"] = df["Amenities"].str.split(", ").apply(len)
    for a in AMENITY_LIST:
        df[f"Has_{a}"] = df["Amenities"].str.contains(a).astype(int)
    df["Parking_Flag"] = (df["Parking_Space"] == "Yes").astype(int)
    df["Security_Flag"] = (df["Security"] == "Yes").astype(int)
    df["Under_Construction"] = (df["Availability_Status"] == "Under_Construction").astype(int)
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map(TRANSPORT_MAP)

    # School Density Score (0-1) and Healthcare score (0-1)
    df["School_Density_Score"] = df["Nearby_Schools"] / 10
    df["Hospital_Score"] = df["Nearby_Hospitals"] / 10
    # Infrastructure score: schools, hospitals, transport, amenities, security (0-1)
    df["Infra_Score"] = (0.25 * df["School_Density_Score"] + 0.25 * df["Hospital_Score"]
                         + 0.25 * df["Transport_Score"] + 0.15 * df["Amenity_Count"] / 5
                         + 0.10 * df["Security_Flag"])

    df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"].clip(lower=1)
    df["Size_per_BHK"] = df["Size_in_SqFt"] / df["BHK"]
    df["Age_Bucket"] = pd.cut(df["Age_of_Property"], [-1, 5, 15, 25, 100],
                              labels=["New(0-5)", "Recent(6-15)", "Mature(16-25)", "Old(26+)"]).astype(str)

    # relative-valuation features
    df["City_Median_PPS"] = df.groupby("City")["Price_per_SqFt"].transform("median")
    df["Locality_Median_PPS"] = df.groupby(["City", "Locality"])["Price_per_SqFt"].transform("median")
    df["PPS_vs_City"] = df["Price_per_SqFt"] / df["City_Median_PPS"]
    df["Is_Tier1"] = df["City"].isin(TIER1_CITIES).astype(int)
    return df


def annual_growth_rate(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Rule-based expected yearly appreciation (documented assumption)."""
    undervalued = (1 - df["PPS_vs_City"]).clip(-0.3, 0.3)      # + if cheaper than city median
    g = (BASE_GROWTH
         + TIER1_BONUS * df["Is_Tier1"]
         + INFRA_WEIGHT * df["Infra_Score"]
         + UNDER_CONSTRUCTION_BONUS * df["Under_Construction"]
         - AGE_PENALTY * df["Age_of_Property"] / 35
         + VALUE_WEIGHT * undervalued
         + rng.normal(0, NOISE_STD, len(df)))
    return g.clip(0.0, 0.20)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(SEED)
    df["Annual_Growth_Rate"] = annual_growth_rate(df, rng)
    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1 + df["Annual_Growth_Rate"]) ** HORIZON_YEARS
    df["Good_Investment"] = ((df["Annual_Growth_Rate"] >= GOOD_INVESTMENT_GROWTH)
                             & (df["PPS_vs_City"] <= 1.10)).astype(int)
    return df


def city_stats(df: pd.DataFrame) -> dict:
    return {
        "city_median_pps": df.groupby("City")["Price_per_SqFt"].median().to_dict(),
        "locality_median_pps": {f"{c}|{l}": v for (c, l), v in
                                df.groupby(["City", "Locality"])["Price_per_SqFt"].median().items()},
        "city_state": df.drop_duplicates("City").set_index("City")["State"].to_dict(),
        "tier1": sorted(TIER1_CITIES),
        "constants": dict(BASE_GROWTH=BASE_GROWTH, TIER1_BONUS=TIER1_BONUS, INFRA_WEIGHT=INFRA_WEIGHT,
                          UNDER_CONSTRUCTION_BONUS=UNDER_CONSTRUCTION_BONUS, AGE_PENALTY=AGE_PENALTY,
                          VALUE_WEIGHT=VALUE_WEIGHT, GOOD_INVESTMENT_GROWTH=GOOD_INVESTMENT_GROWTH,
                          HORIZON_YEARS=HORIZON_YEARS, CURRENT_YEAR=CURRENT_YEAR),
    }


def main():
    df = add_targets(engineer(clean(load_raw())))
    df.to_csv(CLEAN, index=False)
    CITY_STATS.write_text(json.dumps(city_stats(df)))
    print(f"saved {CLEAN}  shape={df.shape}")
    print("Good_Investment share:", round(df["Good_Investment"].mean(), 3))
    print(df[["Annual_Growth_Rate", "Future_Price_5Y"]].describe().round(3))


if __name__ == "__main__":
    main()
