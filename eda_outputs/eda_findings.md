# EDA findings

> Key caveat: raw features in this dataset are statistically independent of price (synthetic data). Relationships shown for `Good_Investment`, `Annual_Growth_Rate` and `Future_Price_5Y` come from the domain rules in `src/preprocess.py`, not from observed market history.

**Q1.** Prices run 10-500 lakhs, mean 255, median 254. The shape is close to uniform (not right-skewed as real markets are), which indicates a synthetic dataset.

**Q2.** Sizes range 500-5000 sq ft (median 2747), again roughly uniform.

**Q3.** Median price/sqft (INR): Apartment 9,245, Independent House 9,305, Villa 9,190. Differences between types are negligible.

**Q4.** Correlation between size and price is -0.003 - effectively zero. In real markets this would be strongly positive; here price is independent of size, so price/sqft is driven almost entirely by size (inverse).

**Q5.** IQR rule flags 19,723 price/sqft outliers (7.9%), 0 size outliers and 0 price outliers. Price/sqft is right-skewed because small flats with large prices produce extreme ratios; it was winsorised at the 1st/99th percentile.

**Q6.** Highest avg price/sqft: Karnataka (13,150); lowest: Rajasthan (12,829). Spread between states is only 2.5%.

**Q7.** Highest avg price: Bangalore (258.5 L); lowest: Cuttack (250.8 L). All 42 cities are within ~3 % of each other.

**Q8.** Median age across 500 localities ranges 16-21 years (overall median 18).

**Q9.** BHK (1-5) is evenly distributed in every city (~20 % each).

**Q10.** Top-5 localities by average price: Locality_395, Locality_366, Locality_497, Locality_360, Locality_203. The dataset has no transaction dates, so 'trend' is shown by construction year. Lines are noisy, so no strong trend should be read into them.

**Q11.** Among the raw numeric columns the largest |correlation| is 0.004: raw features are mutually independent. Engineered targets correlate by construction (e.g. Infra_Score vs growth 0.24).

**Q12.** Avg price/sqft by schools count varies only 12,796-13,174 INR; correlation -0.000 - no meaningful relationship in this data.

**Q13.** Avg price/sqft by hospitals count varies only 12,817-13,152 INR; correlation -0.000 - no meaningful relationship in this data.

**Q14.** Mean price (lakhs): Furnished 254.4, Semi-furnished 254.3, Unfurnished 255.0 - no furnishing premium.

**Q15.** Mean price/sqft: East 12,926, North 12,927, South 12,942, West 13,040 - essentially identical.

**Q16.** Counts: Broker 83,479, Owner 83,268, Builder 83,253.

**Q17.** Counts: Under_Construction 125,035, Ready_to_Move 124,965. (The brief mentions 'Sold' but that value does not occur in this file.)

**Q18.** Mean price with parking 254.7 L vs without 254.4 L - parking does not materially affect price.

**Q19.** Per-amenity price/sqft difference is within ±90 INR; avg price/sqft by amenity count ranges 12,928-12,998. Amenities do not move price here, but they raise our Infra_Score and therefore expected appreciation.

**Q20.** Share of Good Investments by transport access: High 33.3%, Low 22.6%, Medium 27.9%. Price/sqft is unaffected, but better transport raises investment potential through the Infra_Score rule.

**Extra - infrastructure vs value.** Infra_Score correlates 0.000 with current price but 0.24 with the 5-year growth multiple.

**Extra - crime rate.** The brief lists crime rate, but the dataset has no crime column, so that analysis cannot be done.
