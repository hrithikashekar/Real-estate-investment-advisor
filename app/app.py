"""
Step 5 - Streamlit app: Real Estate Investment Advisor (redesigned UI).

Run from the project root:   streamlit run app/app.py
The previous tab-based layout is kept as app/app_classic.py.
"""
from pathlib import Path
import json
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import CURRENT_YEAR, HORIZON_YEARS, TRANSPORT_MAP  # noqa: E402
from modeling import load_portable, models_exist  # noqa: E402

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏙️", layout="wide")

INK, TEAL, OCHRE = "#0E3B3F", "#1F6F68", "#D9992B"
YES, NO, MUTED, LINE = "#1E8E5A", "#B8472D", "#5B6B6A", "#D5DEDB"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Public+Sans:wght@400;500;600&display=swap');
html, body, .stApp { font-family: 'Public Sans', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', 'Public Sans', sans-serif !important; color: #0E3B3F; letter-spacing: -0.01em; }
h4 { font-size: 1.12rem !important; margin-bottom: .3rem; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; max-width: 1300px; }

/* sidebar */
[data-testid="stSidebar"] { background: #0E3B3F; }
[data-testid="stSidebar"] * { color: #DCE8E5; }
[data-testid="stSidebar"] hr { border-color: #2B5A5D; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .15rem; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: .55rem .85rem; border-radius: 8px; width: 100%; cursor: pointer; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: #1D5559; box-shadow: inset 3px 0 0 #D9992B; }
.brand { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.35rem; font-weight: 700; color: #fff; line-height: 1.15; margin: .2rem 0 1.2rem; }
.side-note { font-size: .8rem; line-height: 1.45; color: #A9C2BE !important; }

/* banner */
.banner { position: relative; overflow: hidden; background: #0E3B3F; border-radius: 16px; padding: 1.9rem 2rem; margin-bottom: 1.3rem; min-height: 150px; }
.banner h1 { color: #fff !important; font-size: 2.1rem; margin: 0 0 .5rem; max-width: 640px; line-height: 1.12; }
.banner p { color: #BCD3CF; max-width: 560px; margin: 0; font-size: 1rem; line-height: 1.5; }
.banner svg { position: absolute; right: 0; bottom: 0; height: 100%; width: 46%; }

/* verdict */
.verdict { display: flex; border-radius: 12px; overflow: hidden; border: 1px solid #D5DEDB; background: #fff; margin-bottom: .9rem; }
.verdict .band { width: 14px; background: var(--c); }
.verdict .body { padding: 1rem 1.25rem 1.1rem; flex: 1; }
.verdict .q { color: #5B6B6A; font-size: .92rem; }
.verdict .a { font-family: 'Bricolage Grotesque', sans-serif; font-size: 3.1rem; font-weight: 700; line-height: 1.05; color: var(--c); }
.meter { height: 8px; background: #E3EAE8; border-radius: 99px; margin: .65rem 0 .4rem; position: relative; }
.meter span { display: block; height: 100%; border-radius: 99px; background: var(--c); }
.verdict .note { font-size: .86rem; color: #5B6B6A; }
.yes { --c: #1E8E5A; } .no { --c: #B8472D; }

/* forecast */
.forecast { border: 1px solid #D5DEDB; border-radius: 12px; background: #fff; padding: 1rem 1.25rem 1.2rem; margin-bottom: .9rem; }
.forecast .q { color: #5B6B6A; font-size: .92rem; }
.forecast .big { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.5rem; font-weight: 700; color: #0E3B3F; line-height: 1.1; }
.forecast .big small { font-size: .95rem; font-weight: 500; color: #5B6B6A; margin-left: .4rem; }
.chips { display: flex; gap: .45rem; flex-wrap: wrap; margin-top: .5rem; }
.chip { font-size: .82rem; padding: .18rem .65rem; border-radius: 99px; background: #E3EAE8; color: #0E3B3F; }
.chip.up { background: #DCEFE5; color: #146B41; }
.track { position: relative; height: 10px; background: #E3EAE8; border-radius: 99px; margin: 2.3rem 0 3.1rem; }
.track .band { position: absolute; top: 0; height: 100%; background: #F1D49A; border-radius: 99px; }
.track .mark { position: absolute; top: -4px; width: 4px; height: 18px; border-radius: 2px; transform: translateX(-2px); }
.track .mark span { position: absolute; white-space: nowrap; font-size: .78rem; color: #16282A; transform: translateX(-50%); }
.track .today { background: #0E3B3F; } .track .today span { bottom: 24px; }
.track .fut { background: #D9992B; } .track .fut span { top: 24px; }
.track-cap { font-size: .8rem; color: #5B6B6A; }

/* drivers, kpis */
.driver { display: flex; align-items: center; gap: .6rem; padding: .42rem 0; border-bottom: 1px solid #E3EAE8; font-size: .93rem; }
.driver:last-child { border-bottom: none; }
.driver i { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.driver b { margin-left: auto; font-weight: 600; }
.kpi { background: #fff; border: 1px solid #D5DEDB; border-top: 3px solid #1F6F68; border-radius: 8px; padding: .75rem 1rem; }
.kpi .l { color: #5B6B6A; font-size: .85rem; }
.kpi .v { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.65rem; font-weight: 700; color: #0E3B3F; }
[data-testid="stColumn"]:has(#sticky-results) { position: sticky; top: 1rem; align-self: flex-start; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; background: #FBFCFC; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ loaders
@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    cols = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
            "Age_of_Property", "Furnished_Status", "Public_Transport_Accessibility", "Availability_Status",
            "Infra_Score", "Good_Investment", "Future_Price_5Y", "Annual_Growth_Rate", "Owner_Type", "Year_Built"]
    sample = ROOT / "data" / "app_sample.csv"
    return pd.read_csv(sample if sample.exists() else ROOT / "data" / "cleaned_housing.csv", usecols=cols)


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    clf = load_portable(ROOT / "models", "classifier")
    reg = load_portable(ROOT / "models", "regressor")
    stats = json.loads((ROOT / "models" / "city_stats.json").read_text())
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    fi_c = pd.read_csv(ROOT / "models" / "feature_importance_classifier.csv", index_col=0)["importance"]
    fi_r = pd.read_csv(ROOT / "models" / "feature_importance_regressor.csv", index_col=0)["importance"]
    return clf, reg, stats, metrics, fi_c, fi_r


if not models_exist(ROOT / "models") or not (ROOT / "models" / "metrics.json").exists():
    st.error("Models not found. Run `python src/preprocess.py` then `python src/train.py` first.")
    st.stop()

df = load_data()
clf, reg, stats, metrics, fi_c, fi_r = load_models()


# ------------------------------------------------------------------ helpers
def html(s: str) -> str:
    """Strip indentation so Markdown never treats the HTML as a code block."""
    return "".join(line.strip() for line in s.splitlines())


def inr(lakhs: float) -> str:
    return f"₹{lakhs / 100:.2f} Cr" if lakhs >= 100 else f"₹{lakhs:.1f} L"


def style(fig, h=320, title=None):
    fig.update_layout(
        height=h, margin=dict(t=44 if title else 16, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Public Sans, sans-serif", color="#16282A", size=13),
        colorway=[TEAL, OCHRE, "#6C8EAD", NO, "#8A9A5B"],
        title=dict(text=title, font=dict(family="Bricolage Grotesque, sans-serif", size=17), x=0) if title else None,
        legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#DDE5E3", zerolinecolor="#DDE5E3")
    fig.update_yaxes(gridcolor="#DDE5E3", zerolinecolor="#DDE5E3")
    return fig


def show(fig, h=320, title=None):
    st.plotly_chart(style(fig, h, title), width="stretch")


def kpi(label, value):
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div></div>'


def banner():
    heights = [38, 52, 46, 70, 64, 88, 80, 104, 96, 122, 112, 138, 128, 154]
    rects = ""
    for i, h in enumerate(heights):
        fill = OCHRE if i >= len(heights) - 3 else ("#1D5A5E" if i % 2 == 0 else "#2A7075")
        rects += f'<rect x="{10 + i * 34}" y="{170 - h}" width="28" height="{h}" rx="3" fill="{fill}"/>'
    svg = f'<svg viewBox="0 0 500 170" preserveAspectRatio="xMaxYMax meet" xmlns="http://www.w3.org/2000/svg">{rects}</svg>'
    st.markdown(html(f"""
        <div class="banner">
          <h1>Real Estate Investment Advisor</h1>
          <p>Describe a property to see whether it looks like a good buy and what it could sell for in five years.</p>
          {svg}
        </div>"""), unsafe_allow_html=True)


def build_row(p: dict) -> pd.DataFrame:
    """Turn raw form inputs into the exact feature frame used in training."""
    pps = p["price"] * 1e5 / p["size"]
    city_med = stats["city_median_pps"][p["city"]]
    loc_med = stats["locality_median_pps"].get(f"{p['city']}|{p['locality']}", city_med)
    transport = TRANSPORT_MAP[p["transport"]]
    amen = len(p["amenities"])
    infra = (0.25 * p["schools"] / 10 + 0.25 * p["hospitals"] / 10 + 0.25 * transport
             + 0.15 * amen / 5 + 0.10 * int(p["security"]))
    age = CURRENT_YEAR - p["year_built"]
    floor = min(p["floor"], p["total_floors"])
    bucket = "New(0-5)" if age <= 5 else "Recent(6-15)" if age <= 15 else "Mature(16-25)" if age <= 25 else "Old(26+)"
    return pd.DataFrame([{
        "BHK": p["bhk"], "Size_in_SqFt": p["size"], "Price_in_Lakhs": p["price"], "Price_per_SqFt": pps,
        "Age_of_Property": age, "Floor_No": floor, "Total_Floors": p["total_floors"],
        "Nearby_Schools": p["schools"], "Nearby_Hospitals": p["hospitals"], "Amenity_Count": amen,
        "Parking_Flag": int(p["parking"]), "Security_Flag": int(p["security"]),
        "Under_Construction": int(p["status"] == "Under_Construction"), "Transport_Score": transport,
        "School_Density_Score": p["schools"] / 10, "Hospital_Score": p["hospitals"] / 10, "Infra_Score": infra,
        "Floor_Ratio": floor / max(p["total_floors"], 1), "Size_per_BHK": p["size"] / p["bhk"],
        "City_Median_PPS": city_med, "Locality_Median_PPS": loc_med, "PPS_vs_City": pps / city_med,
        "Is_Tier1": int(p["city"] in stats["tier1"]),
        "City": p["city"], "Property_Type": p["ptype"], "Furnished_Status": p["furnished"],
        "Public_Transport_Accessibility": p["transport"], "Facing": p["facing"], "Owner_Type": p["owner"],
        "Age_Bucket": bucket}])


# ------------------------------------------------------------------ sidebar
PAGES = ["Check a property", "Explore the market", "Model scores", "How it works"]
with st.sidebar:
    st.markdown('<div class="brand">Investment<br>Advisor</div>', unsafe_allow_html=True)
    page = st.radio("Page", PAGES, label_visibility="collapsed")
    st.divider()
    b = metrics["best"]
    st.markdown(html(f"""<div class="side-note">
        Classifier: {b['classifier']}, {metrics['classification'][b['classifier']]['accuracy']:.0%} accurate.<br>
        Forecaster: {b['regressor']}, average error {metrics['regression'][b['regressor']]['mae']:.0f} lakhs.<br><br>
        Growth rates come from rule-based assumptions on a synthetic dataset. This is a demonstration, not financial advice.
        </div>"""), unsafe_allow_html=True)

banner()


# ================================================================== CHECK A PROPERTY
def results_panel(p: dict):
    X = build_row(p)
    proba = float(clf.predict_proba(X)[0, 1])
    future = float(reg.predict(X)[0])
    lo, hi = future + metrics["best"]["residual_p10"], future + metrics["best"]["residual_p90"]
    cagr = (future / p["price"]) ** (1 / HORIZON_YEARS) - 1
    good = proba >= 0.5
    conf = proba if good else 1 - proba
    price = p["price"]

    st.markdown('<div id="sticky-results"></div>', unsafe_allow_html=True)
    st.markdown(html(f"""
        <div class="verdict {'yes' if good else 'no'}"><div class="band"></div><div class="body">
          <div class="q">Is this a good investment?</div>
          <div class="a">{'Yes' if good else 'No'}</div>
          <div class="meter"><span style="width:{conf * 100:.0f}%"></span></div>
          <div class="note">The model is {conf:.0%} confident. Chance this is a good investment: {proba:.0%}.</div>
        </div></div>"""), unsafe_allow_html=True)

    mn, mx = min(price, lo), max(price, hi)
    pad = (mx - mn) * 0.12
    mn, mx = mn - pad, mx + pad
    pos = lambda x: (x - mn) / (mx - mn) * 100  # noqa: E731
    delta_cls = "up" if future >= price else ""
    st.markdown(html(f"""
        <div class="forecast">
          <div class="q">Estimated price in {HORIZON_YEARS} years</div>
          <div class="big">{inr(future)}<small>{future:,.1f} lakhs</small></div>
          <div class="chips">
            <span class="chip {delta_cls}">{future - price:+,.1f} lakhs vs today</span>
            <span class="chip">{cagr:.1%} growth a year</span>
          </div>
          <div class="track">
            <div class="band" style="left:{pos(lo):.1f}%;width:{pos(hi) - pos(lo):.1f}%"></div>
            <div class="mark today" style="left:{pos(price):.1f}%"><span>Today {inr(price)}</span></div>
            <div class="mark fut" style="left:{pos(future):.1f}%"><span>Forecast {inr(future)}</span></div>
          </div>
          <div class="track-cap">The shaded band is the likely range: {inr(lo)} to {inr(hi)} (80% of past test errors fall inside it).</div>
        </div>"""), unsafe_allow_html=True)

    row = X.iloc[0]
    facts = [
        ("Price per sq ft vs city median", f"{row['PPS_vs_City']:.0%}", row["PPS_vs_City"] <= 1.0),
        ("Infrastructure score", f"{row['Infra_Score']:.2f} of 1", row["Infra_Score"] >= 0.55),
        ("Age of property", f"{int(row['Age_of_Property'])} years", row["Age_of_Property"] <= 15),
        ("Tier-1 city", "Yes" if p["city"] in stats["tier1"] else "No", p["city"] in stats["tier1"]),
        ("Still under construction", "Yes" if p["status"] == "Under_Construction" else "No", p["status"] == "Under_Construction"),
    ]
    rows = "".join(f'<div class="driver"><i style="background:{YES if ok else "#C9D3D0"}"></i>{k}<b>{v}</b></div>'
                   for k, v, ok in facts)
    with st.container(border=True):
        st.markdown("#### What is driving this result")
        st.markdown(html(rows), unsafe_allow_html=True)
        st.caption(f"Price per sq ft: ₹{price * 1e5 / p['size']:,.0f}. City median: ₹{stats['city_median_pps'][p['city']]:,.0f}. "
                   "Green dots favour growth.")


def page_check():
    left, right = st.columns([1.12, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Where is it?")
            c1, c2, c3 = st.columns(3)
            states = sorted(df["State"].unique())
            state = c1.selectbox("State", states, index=states.index("Maharashtra") if "Maharashtra" in states else 0)
            city = c2.selectbox("City", sorted(df.loc[df.State == state, "City"].unique()))
            locs = sorted(df.loc[df.City == city, "Locality"].unique(), key=lambda s: int(s.split("_")[-1]))
            locality = c3.selectbox("Locality", locs)
        with st.container(border=True):
            st.markdown("#### The property")
            c1, c2, c3 = st.columns(3)
            ptype = c1.selectbox("Type", sorted(df.Property_Type.unique()))
            bhk = c2.slider("BHK", 1, 5, 3)
            furnished = c3.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"])
            c1, c2 = st.columns(2)
            size = c1.number_input("Size (sq ft)", 500, 5000, 1500, step=50)
            price = c2.number_input("Asking price (lakhs)", 10.0, 500.0, 150.0, step=5.0)
            c1, c2, c3 = st.columns(3)
            year_built = c1.slider("Year built", 1990, 2025, 2012)
            total_floors = c2.slider("Floors in building", 1, 30, 12)
            floor = c3.slider("Floor number", 0, 30, 5)
            c1, c2, c3 = st.columns(3)
            status = c1.selectbox("Status", ["Ready_to_Move", "Under_Construction"])
            facing = c2.selectbox("Facing", ["North", "South", "East", "West"])
            owner = c3.selectbox("Seller", ["Owner", "Builder", "Broker"])
        with st.container(border=True):
            st.markdown("#### Surroundings and features")
            c1, c2, c3 = st.columns(3)
            schools = c1.slider("Schools nearby", 1, 10, 5)
            hospitals = c2.slider("Hospitals nearby", 1, 10, 5)
            transport = c3.select_slider("Public transport", ["Low", "Medium", "High"], value="Medium")
            c1, c2 = st.columns(2)
            parking = c1.toggle("Parking space", True)
            security = c2.toggle("Security", True)
            amenities = st.multiselect("Amenities", ["Clubhouse", "Garden", "Gym", "Playground", "Pool"], ["Gym", "Garden"])
    p = dict(state=state, city=city, locality=locality, ptype=ptype, bhk=bhk, size=size, price=price,
             year_built=year_built, total_floors=total_floors, floor=floor, furnished=furnished, schools=schools,
             hospitals=hospitals, transport=transport, status=status, facing=facing, owner=owner,
             parking=parking, security=security, amenities=amenities)
    with right:
        results_panel(p)

    st.markdown("#### What the models pay attention to")
    a, b = st.columns(2)
    for col, s, title, color in [(a, fi_c, "Good investment classifier", TEAL), (b, fi_r, "Price forecaster", OCHRE)]:
        top = s.head(10)[::-1]
        top.index = [i.replace("_", " ") for i in top.index]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=color))
        with col:
            show(fig, 340, title)

    sim = df[(df.City == p["city"]) & (df.BHK == p["bhk"]) & (df.Property_Type == p["ptype"])]
    st.markdown(f"#### Similar listings in {p['city']}")
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi("Comparable listings", f"{len(sim):,}"), unsafe_allow_html=True)
    k2.markdown(kpi("Flagged as good investments", f"{sim.Good_Investment.mean():.0%}" if len(sim) else "n/a"), unsafe_allow_html=True)
    k3.markdown(kpi("Median asking price", inr(sim.Price_in_Lakhs.median()) if len(sim) else "n/a"), unsafe_allow_html=True)


# ================================================================== EXPLORE
def page_explore():
    with st.container(border=True):
        st.markdown("#### Filter listings")
        f1, f2, f3, f4 = st.columns(4)
        fs = f1.multiselect("State", sorted(df.State.unique()))
        fc = f2.multiselect("City", sorted(df[df.State.isin(fs)].City.unique() if fs else df.City.unique()))
        fb = f3.multiselect("BHK", [1, 2, 3, 4, 5])
        ft = f4.multiselect("Property type", sorted(df.Property_Type.unique()))
        g1, g2, g3 = st.columns(3)
        prange = g1.slider("Price (lakhs)", 10, 500, (10, 500))
        srange = g2.slider("Size (sq ft)", 500, 5000, (500, 5000))
        only_good = g3.toggle("Show only good investments")

    v = df
    if fs: v = v[v.State.isin(fs)]
    if fc: v = v[v.City.isin(fc)]
    if fb: v = v[v.BHK.isin(fb)]
    if ft: v = v[v.Property_Type.isin(ft)]
    v = v[v.Price_in_Lakhs.between(*prange) & v.Size_in_SqFt.between(*srange)]
    if only_good: v = v[v.Good_Investment == 1]

    if not len(v):
        st.info("No properties match these filters. Widen the price or size range, or clear a filter.")
        return
    k = st.columns(4)
    k[0].markdown(kpi("Matching properties", f"{len(v):,}"), unsafe_allow_html=True)
    k[1].markdown(kpi("Average price", inr(v.Price_in_Lakhs.mean())), unsafe_allow_html=True)
    k[2].markdown(kpi("Average price per sq ft", f"₹{v.Price_per_SqFt.mean():,.0f}"), unsafe_allow_html=True)
    k[3].markdown(kpi("Good investments", f"{v.Good_Investment.mean():.0%}"), unsafe_allow_html=True)

    st.markdown("#### Heatmap by city and BHK")
    metric = st.radio("Colour by", ["Price per sq ft (₹)", "Good-investment share", "Forecast price (lakhs)"], horizontal=True)
    col = {"Price per sq ft (₹)": "Price_per_SqFt", "Good-investment share": "Good_Investment",
           "Forecast price (lakhs)": "Future_Price_5Y"}[metric]
    piv = v.pivot_table(index="City", columns="BHK", values=col, aggfunc="mean")
    fig = px.imshow(piv, aspect="auto", color_continuous_scale=["#EAF1EF", "#9CCBBE", "#2A8079", "#0B3B3F"])
    fig.update_xaxes(type="category", title="BHK")
    show(fig, max(420, 17 * len(piv)))

    a, b = st.columns(2)
    s = v.groupby("State")["Good_Investment"].mean().sort_values().reset_index()
    with a:
        show(px.bar(s, x="Good_Investment", y="State", orientation="h", labels={"Good_Investment": "Share of good investments"}),
             460, "Good investments by state")
    t = v.groupby("Year_Built")["Price_per_SqFt"].mean().reset_index()
    with b:
        show(px.line(t, x="Year_Built", y="Price_per_SqFt", labels={"Price_per_SqFt": "Avg ₹ per sq ft", "Year_Built": "Year built"}),
             460, "Average price per sq ft by year built")
    c, d = st.columns(2)
    with c:
        show(px.box(v.sample(min(len(v), 20000), random_state=1), x="Public_Transport_Accessibility", y="Annual_Growth_Rate",
                    category_orders={"Public_Transport_Accessibility": ["Low", "Medium", "High"]},
                    labels={"Public_Transport_Accessibility": "Public transport", "Annual_Growth_Rate": "Expected yearly growth"}),
             340, "Growth by transport access")
    with d:
        show(px.histogram(v, x="Infra_Score", color="Good_Investment", nbins=40, barmode="overlay",
                          labels={"Infra_Score": "Infrastructure score"}), 340, "Infrastructure score by outcome")

    st.markdown("#### Sample listings")
    st.dataframe(v.sample(min(len(v), 500), random_state=1)[[
        "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
        "Availability_Status", "Good_Investment", "Future_Price_5Y"]].round(1), width="stretch", hide_index=True)


# ================================================================== MODEL SCORES
def page_models():
    b = metrics["best"]
    ct = pd.DataFrame(metrics["classification"]).T.drop(columns=["confusion_matrix"])
    rt = pd.DataFrame(metrics["regression"]).T
    k = st.columns(4)
    k[0].markdown(kpi("Classifier accuracy", f"{ct.loc[b['classifier'], 'accuracy']:.1%}"), unsafe_allow_html=True)
    k[1].markdown(kpi("Classifier F1", f"{ct.loc[b['classifier'], 'f1']:.3f}"), unsafe_allow_html=True)
    k[2].markdown(kpi("Forecast error (RMSE)", f"{rt.loc[b['regressor'], 'rmse']:.1f} lakhs"), unsafe_allow_html=True)
    k[3].markdown(kpi("Forecast R²", f"{rt.loc[b['regressor'], 'r2']:.3f}"), unsafe_allow_html=True)
    st.write("")
    a, c = st.columns(2)
    with a:
        d = ct.reset_index().melt(id_vars="index", value_vars=["accuracy", "f1", "roc_auc"], var_name="Metric", value_name="Score")
        show(px.bar(d, x="index", y="Score", color="Metric", barmode="group", labels={"index": "Model"}), 340,
             "Good investment classifiers (higher is better)")
        st.dataframe(ct.drop(columns=["train_seconds"]).round(4), width="stretch")
    with c:
        show(px.bar(rt.reset_index(), x="index", y="rmse", labels={"index": "Model", "rmse": "RMSE (lakhs)"},
                    color_discrete_sequence=[OCHRE]), 340, "Price forecasters (lower error is better)")
        st.dataframe(rt.drop(columns=["train_seconds"]).round(4), width="stretch")
    st.caption("All scores come from a 20% test set the models never saw. Every run is logged in MLflow: "
               "`mlflow ui --backend-store-uri sqlite:///mlflow.db`.")


# ================================================================== HOW IT WORKS
def page_about():
    c = stats["constants"]
    with st.container(border=True):
        st.markdown("#### What the app tells you")
        st.markdown("Two things about any property: whether it looks like a **good investment**, and its **estimated price "
                    f"after {c['HORIZON_YEARS']} years**. Both update as you change the form on the first page.")
    with st.container(border=True):
        st.markdown("#### Where the answers come from")
        st.markdown(f"""
The dataset holds one price per property and no price history, so the two targets are built from stated rules:

- **Yearly growth** starts at {c['BASE_GROWTH']:.0%}, then adds a bonus for tier-1 cities and for infrastructure (schools, hospitals, transport, amenities, security), a bonus for projects still under construction, and a bonus when the price per sq ft is below the city median. It subtracts a penalty for older buildings, and includes random market noise.
- **Price after {c['HORIZON_YEARS']} years** is today's price multiplied by (1 + growth) for each year.
- **Good investment** means growth of at least {c['GOOD_INVESTMENT_GROWTH']:.1%} a year and a price per sq ft within 110% of the city median.
""")
    with st.container(border=True):
        st.markdown("#### How to read the scores")
        st.markdown("""
The models learn those rules, so their accuracy shows how well they recover the rules, not how well they predict the real market.
The forecast is accurate mostly because it starts from the price you enter.

The raw data has quirks: size, location and amenities show no relationship with price, 46% of rows listed a floor above the building's top floor (capped during cleaning), and there is no crime-rate column.

For real decisions, retrain on observed historical prices and speak to a licensed advisor. This is not financial advice.
""")


{"Check a property": page_check, "Explore the market": page_explore,
 "Model scores": page_models, "How it works": page_about}[page]()
