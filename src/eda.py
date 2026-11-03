"""
Step 2 - Exploratory Data Analysis (answers the 20 questions in the brief).

Run:  python src/eda.py
Out:  eda_outputs/*.png  and  eda_outputs/eda_findings.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda_outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "cleaned_housing.csv")
sns.set_theme(style="whitegrid", context="notebook")
findings: list[str] = []


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=110)
    plt.close()


def note(q, text):
    findings.append(f"**{q}** {text}\n")


# ---------------- 1-5 price & size ----------------
sns.histplot(df["Price_in_Lakhs"], bins=50)
plt.title("Q1 Distribution of property prices (Lakhs)")
save("q01_price_dist")
note("Q1.", f"Prices run {df.Price_in_Lakhs.min():.0f}-{df.Price_in_Lakhs.max():.0f} lakhs, mean "
     f"{df.Price_in_Lakhs.mean():.0f}, median {df.Price_in_Lakhs.median():.0f}. The shape is close to uniform "
     "(not right-skewed as real markets are), which indicates a synthetic dataset.")

sns.histplot(df["Size_in_SqFt"], bins=50, color="tab:green")
plt.title("Q2 Distribution of property sizes (sq ft)")
save("q02_size_dist")
note("Q2.", f"Sizes range {df.Size_in_SqFt.min()}-{df.Size_in_SqFt.max()} sq ft (median {df.Size_in_SqFt.median():.0f}), "
     "again roughly uniform.")

sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt")
plt.title("Q3 Price per sq ft (INR) by property type")
save("q03_pps_by_type")
t = df.groupby("Property_Type")["Price_per_SqFt"].median().round(0)
note("Q3.", "Median price/sqft (INR): " + ", ".join(f"{k} {v:,.0f}" for k, v in t.items()) +
     ". Differences between types are negligible.")

s = df.sample(20000, random_state=1)
sns.scatterplot(data=s, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=.25, s=8)
plt.title("Q4 Size vs price")
save("q04_size_vs_price")
r = df[["Size_in_SqFt", "Price_in_Lakhs"]].corr().iloc[0, 1]
note("Q4.", f"Correlation between size and price is {r:.3f} - effectively zero. In real markets this would be strongly "
     "positive; here price is independent of size, so price/sqft is driven almost entirely by size (inverse).")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(y=df["Price_per_SqFt"], ax=ax[0]); ax[0].set_title("Q5 Price/sqft (winsorised)")
sns.boxplot(y=df["Size_in_SqFt"], ax=ax[1], color="tab:green"); ax[1].set_title("Q5 Size")
save("q05_outliers")
note("Q5.", f"IQR rule flags {df.Price_per_SqFt_outlier.sum():,} price/sqft outliers "
     f"({df.Price_per_SqFt_outlier.mean():.1%}), {df.Size_in_SqFt_outlier.sum()} size outliers and "
     f"{df.Price_in_Lakhs_outlier.sum()} price outliers. Price/sqft is right-skewed because small flats with large prices "
     "produce extreme ratios; it was winsorised at the 1st/99th percentile.")

# ---------------- 6-10 location ----------------
st = df.groupby("State")["Price_per_SqFt"].mean().sort_values()
st.plot.barh(figsize=(8, 6)); plt.title("Q6 Avg price per sq ft (INR) by state"); save("q06_pps_state")
note("Q6.", f"Highest avg price/sqft: {st.index[-1]} ({st.iloc[-1]:,.0f}); lowest: {st.index[0]} ({st.iloc[0]:,.0f}). "
     f"Spread between states is only {(st.iloc[-1]/st.iloc[0]-1):.1%}.")

ct = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
ct.plot.barh(figsize=(8, 10)); plt.xlim(240, ct.max() + 3); plt.title("Q7 Avg property price (lakhs) by city")
save("q07_price_city")
note("Q7.", f"Highest avg price: {ct.index[-1]} ({ct.iloc[-1]:.1f} L); lowest: {ct.index[0]} ({ct.iloc[0]:.1f} L). "
     "All 42 cities are within ~3 % of each other.")

lm = df.groupby("Locality")["Age_of_Property"].median()
lm.plot.hist(bins=20, figsize=(7, 4)); plt.title("Q8 Median property age by locality (500 localities)")
save("q08_age_locality")
note("Q8.", f"Median age across 500 localities ranges {lm.min():.0f}-{lm.max():.0f} years (overall median "
     f"{df.Age_of_Property.median():.0f}).")

ctab = pd.crosstab(df["City"], df["BHK"], normalize="index")
plt.figure(figsize=(8, 10)); sns.heatmap(ctab, annot=False, cmap="Blues"); plt.title("Q9 BHK share by city")
save("q09_bhk_city")
note("Q9.", "BHK (1-5) is evenly distributed in every city (~20 % each).")

top5 = df.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5).index
tr = df[df.Locality.isin(top5)].groupby(["Year_Built", "Locality"])["Price_per_SqFt"].mean().unstack()
tr.plot(figsize=(9, 4), marker="o", ms=3)
plt.title("Q10 Price/sqft by year built - top-5 priciest localities\n(no sale-date column exists; Year_Built used as time axis)")
save("q10_top5_trend")
note("Q10.", "Top-5 localities by average price: " + ", ".join(top5) + ". The dataset has no transaction dates, so "
     "'trend' is shown by construction year. Lines are noisy, so no strong trend should be read into them.")

# ---------------- 11-15 relationships ----------------
numc = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt", "BHK", "Age_of_Property", "Nearby_Schools",
        "Nearby_Hospitals", "Floor_No", "Total_Floors", "Infra_Score", "Annual_Growth_Rate", "Future_Price_5Y"]
plt.figure(figsize=(10, 8)); sns.heatmap(df[numc].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Q11 Correlation matrix"); save("q11_corr")
c = df[numc].corr()
raw = c.loc[["Price_in_Lakhs", "Size_in_SqFt", "BHK", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals"],
            ["Price_in_Lakhs", "Size_in_SqFt", "BHK", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals"]]
mx = raw.where(~np.eye(len(raw), dtype=bool)).abs().max().max()
note("Q11.", f"Among the raw numeric columns the largest |correlation| is {mx:.3f}: raw features are mutually independent. "
     f"Engineered targets correlate by construction (e.g. Infra_Score vs growth {c.loc['Infra_Score','Annual_Growth_Rate']:.2f}).")

for q, col, nm in [("12", "Nearby_Schools", "schools"), ("13", "Nearby_Hospitals", "hospitals")]:
    g = df.groupby(col)["Price_per_SqFt"].mean()
    g.plot(marker="o", figsize=(7, 4)); plt.title(f"Q{q} Avg price/sqft by # nearby {nm}")
    save(f"q{q}_{nm}")
    note(f"Q{q}.", f"Avg price/sqft by {nm} count varies only {g.min():,.0f}-{g.max():,.0f} INR; correlation "
         f"{df[col].corr(df.Price_per_SqFt):.3f} - no meaningful relationship in this data.")

sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs"); plt.title("Q14 Price by furnished status")
save("q14_furnished")
g = df.groupby("Furnished_Status")["Price_in_Lakhs"].mean().round(1)
note("Q14.", "Mean price (lakhs): " + ", ".join(f"{k} {v}" for k, v in g.items()) + " - no furnishing premium.")

sns.barplot(data=df, x="Facing", y="Price_per_SqFt", errorbar=None); plt.title("Q15 Price/sqft by facing")
save("q15_facing")
g = df.groupby("Facing")["Price_per_SqFt"].mean().round(0)
note("Q15.", "Mean price/sqft: " + ", ".join(f"{k} {v:,.0f}" for k, v in g.items()) + " - essentially identical.")

# ---------------- 16-20 investment / amenities / ownership ----------------
df["Owner_Type"].value_counts().plot.bar(figsize=(6, 4)); plt.title("Q16 Properties by owner type"); save("q16_owner")
note("Q16.", "Counts: " + ", ".join(f"{k} {v:,}" for k, v in df.Owner_Type.value_counts().items()) + ".")

df["Availability_Status"].value_counts().plot.bar(figsize=(6, 4), color="tab:orange")
plt.title("Q17 Properties by availability status"); save("q17_availability")
note("Q17.", "Counts: " + ", ".join(f"{k} {v:,}" for k, v in df.Availability_Status.value_counts().items()) +
     ". (The brief mentions 'Sold' but that value does not occur in this file.)")

sns.barplot(data=df, x="Parking_Space", y="Price_in_Lakhs", errorbar=None); plt.title("Q18 Price by parking space")
save("q18_parking")
g = df.groupby("Parking_Space")["Price_in_Lakhs"].mean().round(1)
note("Q18.", f"Mean price with parking {g['Yes']} L vs without {g['No']} L - parking does not materially affect price.")

am = {a: df.loc[df[f"Has_{a}"] == 1, "Price_per_SqFt"].mean() - df.loc[df[f"Has_{a}"] == 0, "Price_per_SqFt"].mean()
      for a in ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]}
pd.Series(am).plot.bar(figsize=(7, 4)); plt.title("Q19 Price/sqft difference (has - hasn't amenity)")
save("q19_amenities")
d = df.groupby("Amenity_Count")["Price_per_SqFt"].mean()
note("Q19.", f"Per-amenity price/sqft difference is within ±{max(abs(v) for v in am.values()):.0f} INR; avg price/sqft by "
     f"amenity count ranges {d.min():,.0f}-{d.max():,.0f}. Amenities do not move price here, but they raise our "
     "Infra_Score and therefore expected appreciation.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Price_per_SqFt", order=["Low", "Medium", "High"], errorbar=None, ax=ax[0])
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Good_Investment", order=["Low", "Medium", "High"], errorbar=None, ax=ax[1])
ax[0].set_title("Q20 Price/sqft"); ax[1].set_title("Q20 Share of Good Investments")
save("q20_transport")
g = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean()
note("Q20.", "Share of Good Investments by transport access: " + ", ".join(f"{k} {v:.1%}" for k, v in g.items()) +
     ". Price/sqft is unaffected, but better transport raises investment potential through the Infra_Score rule.")

# ---------------- extra: project-brief EDA items ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df, x="Age_Bucket", y="Good_Investment", errorbar=None, ax=ax[0],
            order=["New(0-5)", "Recent(6-15)", "Mature(16-25)", "Old(26+)"])
ax[0].set_title("Good-investment share by property age")
sns.boxplot(data=df, x="Good_Investment", y="Infra_Score", ax=ax[1]); ax[1].set_title("Infra score vs label")
save("extra_good_investment_drivers")
note("Extra - infrastructure vs value.", f"Infra_Score correlates {df.Infra_Score.corr(df.Price_in_Lakhs):.3f} with current price "
     f"but {df.Infra_Score.corr(df.Future_Price_5Y / df.Price_in_Lakhs):.2f} with the 5-year growth multiple.")
note("Extra - crime rate.", "The brief lists crime rate, but the dataset has no crime column, so that analysis cannot be done.")

city_gi = df.groupby("City")["Good_Investment"].mean().sort_values()
city_gi.plot.barh(figsize=(8, 10)); plt.title("Share of Good Investments by city"); save("extra_city_good_share")

body = "\n".join(findings)
(OUT / "eda_findings.md").write_text(
    "# EDA findings\n\n> Key caveat: raw features in this dataset are statistically independent of price "
    "(synthetic data). Relationships shown for `Good_Investment`, `Annual_Growth_Rate` and `Future_Price_5Y` come from the "
    "domain rules in `src/preprocess.py`, not from observed market history.\n\n" + body)
print(body)
