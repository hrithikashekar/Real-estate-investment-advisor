"""
Step 5 - Streamlit app: Real Estate Investment Advisor (v3 design).

Run from the project root:   streamlit run app/app.py
Earlier layouts are kept as app/app_classic.py (tabs) for reference.
"""
from pathlib import Path
import json
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import CURRENT_YEAR, HORIZON_YEARS, TRANSPORT_MAP  # noqa: E402
from modeling import load_portable, models_exist  # noqa: E402

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏙️", layout="wide")

# palette: cool teal-to-blue family for data, saffron and coral as contrast
INK, TEAL, SKY, MINT = "#0E3B3F", "#1F6F68", "#5B9BC4", "#8DC9B8"
OCHRE, CORAL, PLUM = "#E0A030", "#D0624A", "#7B6BA8"
YES, NO, MUTED = "#1E8E5A", "#C2492F", "#5B6B6A"
COOL = [[0, "#CBE7E3"], [0.25, "#8FCBC8"], [0.5, "#4FA3B0"], [0.75, "#2A769A"], [1, "#143F63"]]
WARM = [[0, "#F8DDA0"], [0.5, "#F0B84A"], [1, "#D0624A"]]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Public+Sans:wght@400;500;600&display=swap');
html, body, .stApp { font-family: 'Public Sans', system-ui, sans-serif; }
.stApp { background: linear-gradient(180deg, #E6F0EE 0%, #F3F7F6 320px, #F5F8F7 100%); }
h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', 'Public Sans', sans-serif !important; color: #0E3B3F; letter-spacing: -0.01em; }
h4 { font-size: 1.1rem !important; margin: 0 0 .4rem; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.3rem; max-width: 1320px; }

/* sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0A2E32 0%, #0E3B3F 55%, #145157 100%); }
[data-testid="stSidebar"] * { color: #DCE8E5; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .2rem; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] { padding: .62rem .9rem; border-radius: 10px; width: 100%; cursor: pointer; transition: background .15s; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover { background: rgba(255,255,255,.06); }
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] { background: rgba(255,255,255,.11); box-shadow: inset 3px 0 0 #E0A030; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]) { display: none; }
.brand { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.4rem; font-weight: 700; color: #fff; line-height: 1.12; margin: .2rem 0 .25rem; }
.brand-sub { font-size: .82rem; color: #8FB3AE !important; margin-bottom: 1.3rem; }
.side-note { font-size: .8rem; line-height: 1.5; color: #9DBBB6 !important; }

/* banner */
.banner { position: relative; overflow: hidden; border-radius: 18px; padding: 1.8rem 2rem; margin-bottom: 1.4rem; min-height: 150px;
  background: radial-gradient(700px 260px at 88% 125%, rgba(224,160,48,.30), transparent 62%), linear-gradient(120deg, #0A2E32 0%, #0E3B3F 45%, #17605F 100%); }
.banner h1 { color: #fff !important; font-size: 2.05rem; margin: 0 0 .5rem; max-width: 52%; line-height: 1.12; }
.banner p { color: #BCD8D3; max-width: min(46%, 540px); margin: 0; font-size: 1rem; line-height: 1.5; }
.banner svg { position: absolute; right: 0; bottom: 0; height: 100%; width: 48%; }

/* section headings */
.sec { margin: 1.5rem 0 .7rem; }
.sec h3 { font-size: 1.35rem; margin: 0; }
.sec p { margin: .1rem 0 0; color: #5B6B6A; font-size: .92rem; }

/* cards */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; background: #fff; border-color: #DCE5E2; box-shadow: 0 12px 26px -18px rgba(14,59,63,.45); }
[data-baseweb="select"] > div { background: #F1F6F5; border-radius: 10px; border: 1px solid transparent; }
[data-baseweb="select"] > div:hover { border-color: #8DC9B8; }
[data-baseweb="input"], [data-baseweb="base-input"] { background: #F1F6F5 !important; border-radius: 10px; }

/* verdict */
.verdict { display: flex; border-radius: 14px; overflow: hidden; border: 1px solid #D5DEDB; margin-bottom: .9rem; background: linear-gradient(135deg, var(--t), #fff 62%); box-shadow: 0 12px 26px -18px rgba(14,59,63,.45); }
.verdict .band { width: 14px; background: var(--c); }
.verdict .body { padding: 1rem 1.3rem 1.1rem; flex: 1; }
.verdict .q { color: #5B6B6A; font-size: .92rem; }
.verdict .a { font-family: 'Bricolage Grotesque', sans-serif; font-size: 3.2rem; font-weight: 700; line-height: 1.05; color: var(--c); }
.meter { height: 9px; background: rgba(14,59,63,.1); border-radius: 99px; margin: .65rem 0 .45rem; overflow: hidden; }
.meter span { display: block; height: 100%; border-radius: 99px; background: var(--c); animation: fill .8s ease-out; }
@keyframes fill { from { width: 0; } }
@media (prefers-reduced-motion: reduce) { .meter span { animation: none; } }
.verdict .note { font-size: .86rem; color: #5B6B6A; }
.yes { --c: #1E8E5A; --t: #E3F4EA; } .no { --c: #C2492F; --t: #FBEAE5; }

/* forecast */
.forecast { border: 1px solid #D5DEDB; border-radius: 14px; background: #fff; padding: 1rem 1.3rem 1.2rem; margin-bottom: .9rem; box-shadow: 0 12px 26px -18px rgba(14,59,63,.45); }
.forecast .q { color: #5B6B6A; font-size: .92rem; }
.forecast .big { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.55rem; font-weight: 700; color: #0E3B3F; line-height: 1.1; }
.forecast .big small { font-size: .95rem; font-weight: 500; color: #5B6B6A; margin-left: .4rem; }
.chips { display: flex; gap: .45rem; flex-wrap: wrap; margin-top: .5rem; }
.chip { font-size: .82rem; padding: .2rem .7rem; border-radius: 99px; background: #E3EDEB; color: #0E3B3F; }
.chip.up { background: #DCEFE5; color: #146B41; } .chip.down { background: #FBE6E0; color: #A63A22; }
.track { position: relative; height: 10px; background: #E3EAE8; border-radius: 99px; margin: 2.3rem 0 3.1rem; }
.track .band { position: absolute; top: 0; height: 100%; background: linear-gradient(90deg, #F6DDA4, #F0C068); border-radius: 99px; }
.track .mark { position: absolute; top: -4px; width: 4px; height: 18px; border-radius: 2px; transform: translateX(-2px); }
.track .mark span { position: absolute; white-space: nowrap; font-size: .78rem; color: #16282A; transform: translateX(-50%); }
.track .today { background: #0E3B3F; } .track .today span { bottom: 24px; }
.track .fut { background: #E0A030; } .track .fut span { top: 24px; }
.track-cap { font-size: .8rem; color: #5B6B6A; }

/* drivers and stat cards */
.driver { display: flex; align-items: center; gap: .6rem; padding: .45rem 0; border-bottom: 1px solid #E8EEEC; font-size: .93rem; }
.driver:last-child { border-bottom: none; }
.driver i { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.driver b { margin-left: auto; font-weight: 600; }
.kpi { background: #fff; border: 1px solid #DCE5E2; border-radius: 12px; padding: .8rem 1rem .85rem 1.1rem; position: relative; overflow: hidden; box-shadow: 0 12px 26px -18px rgba(14,59,63,.45); }
.kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--k, #1F6F68); }
.kpi .l { color: #5B6B6A; font-size: .85rem; }
.kpi .v { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; font-weight: 700; color: #0E3B3F; line-height: 1.25; }
.k0 { --k: #1F6F68; } .k1 { --k: #5B9BC4; } .k2 { --k: #E0A030; } .k3 { --k: #D0624A; }
[data-testid="stColumn"]:has(#sticky-results) { position: sticky; top: 1rem; align-self: flex-start; }
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
        height=h, margin=dict(t=48 if title else 16, b=14, l=12, r=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Public Sans, sans-serif", color="#16282A", size=13),
        colorway=[TEAL, SKY, OCHRE, CORAL, PLUM, MINT],
        title=dict(text=title, font=dict(family="Bricolage Grotesque, sans-serif", size=17, color=INK), x=0.01) if title else None,
        legend=dict(orientation="h", y=-0.18, x=0, title_text=""),
        hoverlabel=dict(bgcolor=INK, font=dict(color="#fff", family="Public Sans, sans-serif", size=13), bordercolor=INK))
    fig.update_xaxes(gridcolor="#E7EEEC", zeroline=False, linecolor="#D5DEDB", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#E7EEEC", zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED))
    return fig


def chart(fig, h=320, title=None):
    """Draw a chart inside a card."""
    with st.container(border=True):
        st.plotly_chart(style(fig, h, title), width="stretch", config={"displayModeBar": False})


def kpi(label, value, tone=0):
    return f'<div class="kpi k{tone}"><div class="l">{label}</div><div class="v">{value}</div></div>'


def section(title, sub=None):
    st.markdown(f'<div class="sec"><h3>{title}</h3>{f"<p>{sub}</p>" if sub else ""}</div>', unsafe_allow_html=True)


def banner(sub):
    heights = [38, 52, 46, 70, 64, 88, 80, 104, 96, 122, 112, 138, 128, 154]
    defs = ('<defs><linearGradient id="gt" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3C9A94"/><stop offset="1" stop-color="#14575B"/></linearGradient>'
            '<linearGradient id="go" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F3C05A"/><stop offset="1" stop-color="#D98B2B"/></linearGradient></defs>')
    parts = ''
    for i, h in enumerate(heights):
        x, hot = 10 + i * 34, i >= len(heights) - 3
        parts += f'<rect x="{x}" y="{170 - h}" width="28" height="{h}" rx="3" fill="url(#{"go" if hot else "gt"})"/>'
        for r in range(int((h - 10) // 16)):
            for c in (0, 1):
                parts += f'<rect x="{x + 6 + c * 10}" y="{170 - h + 8 + r * 16}" width="6" height="8" rx="1" fill="#fff" opacity="{.26 if (r + c + i) % 3 else .08}"/>'
    svg = f'<svg viewBox="0 0 500 170" preserveAspectRatio="xMaxYMax meet" xmlns="http://www.w3.org/2000/svg">{defs}{parts}</svg>'
    st.markdown(html(f"""<div class="banner"><h1>Real Estate Investment Advisor</h1><p>{sub}</p>{svg}</div>"""),
                unsafe_allow_html=True)


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
SUBS = {
    "Check a property": "Describe a property to see whether it looks like a good buy and what it could sell for in five years.",
    "Explore the market": "Compare cities, prices and expected growth across thousands of listings.",
    "Model scores": "See how well the two models perform on properties they have never seen.",
    "How it works": "How the answers are built, and how far to trust them.",
}
with st.sidebar:
    st.markdown('<div class="brand">Investment<br>Advisor</div><div class="brand-sub">Buy smarter, plan five years ahead</div>',
                unsafe_allow_html=True)
    page = st.radio("Page", PAGES, label_visibility="collapsed")
    st.divider()
    b = metrics["best"]
    st.markdown(html(f"""<div class="side-note">
        Classifier: {b['classifier']}, {metrics['classification'][b['classifier']]['accuracy']:.0%} accurate.<br>
        Forecaster: {b['regressor']}, average error {metrics['regression'][b['regressor']]['mae']:.0f} lakhs.<br><br>
        Growth rates come from rule-based assumptions on a synthetic dataset. This is a demonstration, not financial advice.
        </div>"""), unsafe_allow_html=True)

banner(SUBS[page])


# ================================================================== CHECK A PROPERTY
def predict(p: dict) -> dict:
    X = build_row(p)
    proba = float(clf.predict_proba(X)[0, 1])
    future = float(reg.predict(X)[0])
    lo, hi = future + metrics["best"]["residual_p10"], future + metrics["best"]["residual_p90"]
    return dict(X=X, proba=proba, future=future, lo=lo, hi=hi, good=proba >= 0.5,
                conf=proba if proba >= 0.5 else 1 - proba,
                cagr=(future / p["price"]) ** (1 / HORIZON_YEARS) - 1)


def results_panel(p: dict, r: dict):
    price, future, lo, hi = p["price"], r["future"], r["lo"], r["hi"]
    st.markdown('<div id="sticky-results"></div>', unsafe_allow_html=True)
    st.markdown(html(f"""
        <div class="verdict {'yes' if r['good'] else 'no'}"><div class="band"></div><div class="body">
          <div class="q">Is this a good investment?</div>
          <div class="a">{'Yes' if r['good'] else 'No'}</div>
          <div class="meter"><span style="width:{r['conf'] * 100:.0f}%"></span></div>
          <div class="note">The model is {r['conf']:.0%} confident. Chance this is a good investment: {r['proba']:.0%}.</div>
        </div></div>"""), unsafe_allow_html=True)

    mn, mx = min(price, lo), max(price, hi)
    pad = (mx - mn) * 0.12
    mn, mx = mn - pad, mx + pad
    pos = lambda x: (x - mn) / (mx - mn) * 100  # noqa: E731
    st.markdown(html(f"""
        <div class="forecast">
          <div class="q">Estimated price in {HORIZON_YEARS} years</div>
          <div class="big">{inr(future)}<small>{future:,.1f} lakhs</small></div>
          <div class="chips">
            <span class="chip {'up' if future >= price else 'down'}">{future - price:+,.1f} lakhs vs today</span>
            <span class="chip">{r['cagr']:.1%} growth a year</span>
          </div>
          <div class="track">
            <div class="band" style="left:{pos(lo):.1f}%;width:{pos(hi) - pos(lo):.1f}%"></div>
            <div class="mark today" style="left:{pos(price):.1f}%"><span>Today {inr(price)}</span></div>
            <div class="mark fut" style="left:{pos(future):.1f}%"><span>Forecast {inr(future)}</span></div>
          </div>
          <div class="track-cap">The shaded band is the likely range: {inr(lo)} to {inr(hi)} (80% of past test errors fall inside it).</div>
        </div>"""), unsafe_allow_html=True)


def price_path_fig(p, r):
    yrs = list(range(HORIZON_YEARS + 1))
    path = lambda end: [p["price"] * (end / p["price"]) ** (t / HORIZON_YEARS) for t in yrs]  # noqa: E731
    x = [CURRENT_YEAR + t for t in yrs]
    mid, lo, hi = path(r["future"]), path(r["lo"]), path(r["hi"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(224,160,48,.25)",
                             name="Likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mid, mode="lines+markers", name="Expected price",
                             line=dict(color=TEAL, width=3.5), marker=dict(size=[8] * HORIZON_YEARS + [13],
                             color=[TEAL] * HORIZON_YEARS + [OCHRE], line=dict(color="#fff", width=2)),
                             hovertemplate="%{x}: %{y:,.1f} lakhs<extra></extra>"))
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title="Lakhs")
    return fig


def infra_fig(p, X):
    row = X.iloc[0]
    names = ["Schools", "Hospitals", "Public transport", "Amenities", "Security"]
    vals = [row["School_Density_Score"], row["Hospital_Score"], row["Transport_Score"], row["Amenity_Count"] / 5, row["Security_Flag"]]
    vals = [v * 100 for v in vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=[100] * 5, orientation="h", marker_color="#E3ECEA", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Bar(y=names, x=vals, orientation="h", marker=dict(color=vals, colorscale=COOL, cmin=0, cmax=100),
                         text=[f"{v:.0f}%" for v in vals], textposition="inside", insidetextanchor="start",
                         textfont=dict(color="#fff"), showlegend=False, hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
    fig.update_layout(barmode="overlay", bargap=0.35)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 100], showticklabels=False, showgrid=False)
    return fig


def importance_fig(s, scale):
    top = s.head(10)[::-1]
    names = [i.replace("_", " ") for i in top.index]
    fig = go.Figure(go.Bar(x=top.values, y=names, orientation="h",
                           marker=dict(color=top.values, colorscale=scale),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(bargap=0.3)
    return fig


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
    r = predict(p)
    with right:
        results_panel(p, r)

    section("Why this result", "How the price could move, how the surroundings score, and what pushes growth up or down.")
    a, b, c = st.columns([1.25, 1, 0.95])
    with a:
        chart(price_path_fig(p, r), 340, f"Price path to {CURRENT_YEAR + HORIZON_YEARS}")
    with b:
        chart(infra_fig(p, r["X"]), 340, f"Infrastructure score {r['X'].iloc[0]['Infra_Score']:.2f} of 1")
    with c:
        row = r["X"].iloc[0]
        facts = [
            ("Price per sq ft vs city median", f"{row['PPS_vs_City']:.0%}", row["PPS_vs_City"] <= 1.0),
            ("Infrastructure score", f"{row['Infra_Score']:.2f} of 1", row["Infra_Score"] >= 0.55),
            ("Age of property", f"{int(row['Age_of_Property'])} years", row["Age_of_Property"] <= 15),
            ("Tier-1 city", "Yes" if city in stats["tier1"] else "No", city in stats["tier1"]),
            ("Under construction", "Yes" if status == "Under_Construction" else "No", status == "Under_Construction"),
        ]
        rows = "".join(f'<div class="driver"><i style="background:{YES if ok else "#C9D3D0"}"></i>{k}<b>{v}</b></div>'
                       for k, v, ok in facts)
        with st.container(border=True, height=372):
            st.markdown("#### Growth drivers")
            st.markdown(html(rows), unsafe_allow_html=True)
            st.caption(f"Price per sq ft ₹{price * 1e5 / size:,.0f}; city median ₹{stats['city_median_pps'][city]:,.0f}. "
                       "Green dots favour growth.")

    section("What the models pay attention to", "The inputs each model leaned on most when making predictions.")
    a, b = st.columns(2)
    with a:
        chart(importance_fig(fi_c, COOL), 360, "Good investment classifier")
    with b:
        chart(importance_fig(fi_r, WARM), 360, "Price forecaster")

    sim = df[(df.City == city) & (df.BHK == bhk) & (df.Property_Type == ptype)]
    section(f"Similar listings in {city}", f"{bhk} BHK {ptype.lower()} listings in the dataset.")
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi("Comparable listings", f"{len(sim):,}", 0), unsafe_allow_html=True)
    k2.markdown(kpi("Flagged as good investments", f"{sim.Good_Investment.mean():.0%}" if len(sim) else "n/a", 1), unsafe_allow_html=True)
    k3.markdown(kpi("Median asking price", inr(sim.Price_in_Lakhs.median()) if len(sim) else "n/a", 2), unsafe_allow_html=True)


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
    st.write("")
    k = st.columns(4)
    k[0].markdown(kpi("Matching properties", f"{len(v):,}", 0), unsafe_allow_html=True)
    k[1].markdown(kpi("Average price", inr(v.Price_in_Lakhs.mean()), 1), unsafe_allow_html=True)
    k[2].markdown(kpi("Average price per sq ft", f"₹{v.Price_per_SqFt.mean():,.0f}", 2), unsafe_allow_html=True)
    k[3].markdown(kpi("Average yearly growth", f"{v.Annual_Growth_Rate.mean():.1%}", 3), unsafe_allow_html=True)

    section("Good investments at a glance")
    a, b = st.columns([1, 1.4])
    share = v.Good_Investment.mean()
    donut = go.Figure(go.Pie(values=[share, 1 - share], labels=["Good investment", "Other"], hole=0.66, sort=False,
                             marker=dict(colors=[TEAL, "#D9E4E2"], line=dict(color="#fff", width=3)), textinfo="none",
                             hovertemplate="%{label}: %{percent}<extra></extra>"))
    donut.add_annotation(text=f"<b>{share:.0%}</b><br><span style='font-size:12px;color:{MUTED}'>good</span>",
                         showarrow=False, font=dict(size=30, family="Bricolage Grotesque, sans-serif", color=INK))
    with a:
        chart(donut, 360, "Share of good investments")
    ages = pd.cut(v.Age_of_Property, [-1, 5, 15, 25, 100], labels=["0 to 5 years", "6 to 15 years", "16 to 25 years", "26+ years"])
    gs = v.assign(Age=ages).groupby(["Age", "Availability_Status"], observed=True)["Good_Investment"].mean().reset_index()
    grp = go.Figure()
    for status, colr, nm in [("Ready_to_Move", TEAL, "Ready to move"), ("Under_Construction", OCHRE, "Under construction")]:
        sub = gs[gs.Availability_Status == status]
        grp.add_trace(go.Bar(x=sub.Age.astype(str), y=sub.Good_Investment, name=nm, marker_color=colr,
                             text=[f"{x:.0%}" for x in sub.Good_Investment], textposition="outside"))
    grp.update_layout(barmode="group", bargap=0.3)
    grp.update_yaxes(tickformat=".0%", rangemode="tozero")
    with b:
        chart(grp, 360, "Good investments by property age and status")

    section("Heatmap by city and BHK")
    metric = st.radio("Colour by", ["Price per sq ft (₹)", "Good-investment share", "Forecast price (lakhs)"], horizontal=True)
    col = {"Price per sq ft (₹)": "Price_per_SqFt", "Good-investment share": "Good_Investment",
           "Forecast price (lakhs)": "Future_Price_5Y"}[metric]
    piv = v.pivot_table(index="City", columns="BHK", values=col, aggfunc="mean")
    heat = go.Figure(go.Heatmap(z=piv.values, x=[f"{c} BHK" for c in piv.columns], y=piv.index, colorscale=COOL,
                                xgap=2, ygap=2, hovertemplate="%{y}, %{x}: %{z:,.2f}<extra></extra>"))
    heat.update_yaxes(autorange="reversed", dtick=1)
    chart(heat, max(460, 19 * len(piv)))

    section("Trends and drivers")
    a, b = st.columns(2)
    s = v.groupby("State")["Good_Investment"].mean().sort_values()
    bars = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker=dict(color=s.values, colorscale=COOL),
                            hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    bars.update_layout(bargap=0.25)
    bars.update_xaxes(tickformat=".0%")
    with a:
        chart(bars, 500, "Good investments by state")
    t = v.groupby("Year_Built")["Price_per_SqFt"].mean().reset_index()
    line = go.Figure(go.Scatter(x=t.Year_Built, y=t.Price_per_SqFt, mode="lines+markers", line=dict(color=SKY, width=3),
                                marker=dict(size=7, color=TEAL, line=dict(color="#fff", width=1.5)),
                                hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>"))
    line.update_yaxes(title="Average ₹ per sq ft", range=[0, t.Price_per_SqFt.max() * 1.25])
    with b:
        chart(line, 500, "Price per sq ft barely changes with build year")
    c, d = st.columns(2)
    box = go.Figure()
    for lvl, colr in [("Low", CORAL), ("Medium", OCHRE), ("High", TEAL)]:
        sub = v[v.Public_Transport_Accessibility == lvl]
        box.add_trace(go.Box(y=sub.Annual_Growth_Rate * 100, name=lvl, marker_color=colr, boxmean=True, line=dict(width=2)))
    box.update_yaxes(title="Expected yearly growth (%)")
    box.update_layout(showlegend=False)
    with c:
        chart(box, 360, "Growth by public transport access")
    hist = go.Figure()
    for flag, name, colr in [(0, "Other", CORAL), (1, "Good investment", TEAL)]:
        hist.add_trace(go.Histogram(x=v[v.Good_Investment == flag].Infra_Score, name=name, marker_color=colr, opacity=.78, nbinsx=40))
    hist.update_layout(barmode="overlay")
    hist.update_xaxes(title="Infrastructure score")
    with d:
        chart(hist, 360, "Infrastructure score by outcome")

    section("Sample listings", "A random sample of up to 500 matching properties.")
    st.dataframe(v.sample(min(len(v), 500), random_state=1)[[
        "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
        "Availability_Status", "Good_Investment", "Future_Price_5Y"]].round(1), width="stretch", hide_index=True)


# ================================================================== MODEL SCORES
def page_models():
    b = metrics["best"]
    ct = pd.DataFrame(metrics["classification"]).T.drop(columns=["confusion_matrix"])
    rt = pd.DataFrame(metrics["regression"]).T
    k = st.columns(4)
    k[0].markdown(kpi("Classifier accuracy", f"{ct.loc[b['classifier'], 'accuracy']:.1%}", 0), unsafe_allow_html=True)
    k[1].markdown(kpi("Classifier F1", f"{ct.loc[b['classifier'], 'f1']:.3f}", 1), unsafe_allow_html=True)
    k[2].markdown(kpi("Forecast error (RMSE)", f"{rt.loc[b['regressor'], 'rmse']:.1f} lakhs", 2), unsafe_allow_html=True)
    k[3].markdown(kpi("Forecast R²", f"{rt.loc[b['regressor'], 'r2']:.3f}", 3), unsafe_allow_html=True)
    section("Good investment classifiers", "Higher is better. Scores are on a test set the models never trained on.")
    a, c = st.columns([1.3, 1])
    fig = go.Figure()
    for m, colr, nm in [("accuracy", TEAL, "Accuracy"), ("f1", SKY, "F1"), ("roc_auc", OCHRE, "ROC-AUC")]:
        fig.add_trace(go.Bar(x=ct.index, y=ct[m], name=nm, marker_color=colr, text=[f"{v:.3f}" for v in ct[m]], textposition="outside"))
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_yaxes(range=[0.6, 1.0])
    with a:
        chart(fig, 360, "Classifier comparison")
    with c:
        with st.container(border=True, height=392):
            st.markdown("#### Scores table")
            st.dataframe(ct.drop(columns=["train_seconds"]).round(4), width="stretch")
            st.caption(f"Selected model: {b['classifier']}, chosen by F1 score.")
    section("Price forecasters", "Lower error is better. The selected model is highlighted in saffron.")
    a, c = st.columns([1.3, 1])
    colors = [OCHRE if m == b["regressor"] else SKY for m in rt.index]
    fig = go.Figure(go.Bar(x=rt.index, y=rt["rmse"], marker_color=colors, text=[f"{v:.1f}" for v in rt["rmse"]], textposition="outside"))
    fig.update_layout(bargap=0.45)
    fig.update_yaxes(title="RMSE (lakhs)", rangemode="tozero")
    with a:
        chart(fig, 360, "Forecast error by model")
    with c:
        with st.container(border=True, height=392):
            st.markdown("#### Scores table")
            st.dataframe(rt.drop(columns=["train_seconds"]).round(4), width="stretch")
            st.caption(f"Selected model: {b['regressor']}, chosen by lowest RMSE.")
    st.caption("Every run is logged in MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.")


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
