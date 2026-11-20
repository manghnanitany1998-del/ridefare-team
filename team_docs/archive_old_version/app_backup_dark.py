"""
Uber Fare Predictor - Streamlit app
Run:  python -m streamlit run app.py

Works OFFLINE: places come from data/places.csv (34,000+ cities in 244 countries
+ popular landmarks/airports). If internet is available, it also adds live
OpenStreetMap search results and real road routes (OSRM).
"""
from pathlib import Path
from datetime import datetime, date, time, timedelta
import random
import joblib
import pandas as pd
import requests
import pydeck as pdk
import streamlit as st
from model_utils import make_features

st.set_page_config(page_title="RideFare · Fare Predictor", page_icon="🚕", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================ STYLE
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
html, body, .stApp {font-family: 'Poppins', sans-serif;}
.stApp {background: radial-gradient(circle at 10% 0%, #12301f 0%, #0b0f14 45%, #07090c 100%);}
.block-container {padding-top: 1.2rem; max-width: 1250px;}
.hero {position: relative; overflow: hidden; border-radius: 24px; padding: 34px 36px;
       background: linear-gradient(115deg, #06C167 0%, #04a457 35%, #0b3d2a 100%);
       box-shadow: 0 20px 60px rgba(6,193,103,.25); margin-bottom: 22px;}
.hero h1 {font-size: 2.6rem; font-weight: 800; margin: 0; color: #fff;}
.hero p {margin: 6px 0 0; font-size: 1.05rem; color: #eafff3;}
.hero .car {position: absolute; right: 36px; top: 8px; font-size: 6rem;}
.pill {display:inline-block; background: rgba(255,255,255,.18); padding: 4px 12px;
       border-radius: 999px; font-size: .8rem; margin: 12px 8px 0 0; color:#fff;}
/* glass panels: st.container(border=True) */
[data-testid="stVerticalBlockBorderWrapper"] {background: rgba(255,255,255,.03);
       border: 1px solid rgba(255,255,255,.09) !important; border-radius: 20px !important;}
.step {font-size: .8rem; letter-spacing: .15em; color: #06C167; font-weight: 700;}
.fare-card {background: linear-gradient(160deg, #0f1a14, #0b0f14); border: 1px solid #06C16766;
            border-radius: 22px; padding: 24px; color:#e9eef3; box-shadow: 0 10px 40px rgba(6,193,103,.15);}
.fare {font-size: 3.2rem; font-weight: 800; color: #06C167; line-height: 1;}
.inr {font-size: 1.6rem; font-weight: 600; color: #fff; margin-top: 4px;}
.muted {color: #8a97a6; font-size: .88rem;}
.chip {display:inline-block; padding: 6px 12px; border-radius: 12px; background:#131c24; color:#e9eef3;
       border:1px solid #23303c; margin: 6px 6px 0 0; font-size: .9rem;}
.route {border-left: 3px dashed #06C167; padding-left: 14px; margin: 14px 0;}
.dot-g {color:#06C167;} .dot-r {color:#ff5a5a;}
div.stButton > button[kind="primary"] {background: linear-gradient(90deg,#06C167,#03e07a); color:#04130b;
    border: 0; font-weight: 800; font-size: 1.1rem; border-radius: 14px; height: 3.3rem;
    box-shadow: 0 8px 30px rgba(6,193,103,.4);}
div.stButton > button[kind="primary"] p {color:#04130b;}
[data-testid="stMetricValue"] {color:#06C167;}
/* Force dark sidebar + readable text even if the theme file isn't loaded */
[data-testid="stSidebar"] {background: #0f151c !important;}
[data-testid="stSidebar"] * {color: #e9eef3;}
[data-testid="stSidebar"] [data-testid="stExpander"] {background:#141b23; border-radius:12px;}
[data-testid="stSidebar"] [data-testid="stAlert"] {background:#13241b !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="car">🚕</div>
  <h1>RideFare</h1>
  <p>Predict your ride fare anywhere in the world, powered by machine learning.</p>
  <span class="pill">🌍 244 countries</span><span class="pill">🤖 ML model</span>
  <span class="pill">💱 Local currency + ₹</span><span class="pill">⏰ Best time to ride</span>
</div>""", unsafe_allow_html=True)

# ============================================================ MODEL + DATA
MODEL_PATH, PLACES_PATH = Path("model_bundle.joblib"), Path("data/places.csv")
if not MODEL_PATH.exists():
    st.error("Model not found. First place the CSV at data/uber.csv and run: python train_model.py")
    st.stop()
if not PLACES_PATH.exists():
    st.error("data/places.csv not found. It holds the worldwide place list for search.")
    st.stop()

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_places():
    df = pd.read_csv(PLACES_PATH, keep_default_na=False)
    df["label"] = df["name"] + ", " + df["country"]
    return df

bundle, PLACES = load_bundle(), load_places()
MODEL, MAE = bundle["model"], bundle["metrics"]["MAE"]
HEADERS = {"User-Agent": "uber_fare_hackathon_app"}

@st.cache_data(ttl=60, show_spinner=False)
def online():
    """Quick check: can we reach the internet map services?"""
    try:
        requests.get("https://photon.komoot.io/api/?q=a&limit=1", headers=HEADERS, timeout=3)
        return True
    except Exception:
        return False

@st.cache_data(show_spinner=False, ttl=3600)
def live_search(q):
    """Extra results from OpenStreetMap (streets, malls, offices) when online."""
    try:
        r = requests.get("https://photon.komoot.io/api/", params={"q": q, "limit": 8}, headers=HEADERS, timeout=5)
        out = []
        for f in r.json().get("features", []):
            p = f["properties"]
            parts = [p.get("name"), p.get("street"), p.get("city"), p.get("state")]
            name = ", ".join(dict.fromkeys(x for x in parts if x))
            lon, lat = f["geometry"]["coordinates"]
            out.append({"name": name, "country": p.get("country", ""), "lat": lat, "lon": lon, "type": "Live"})
        return pd.DataFrame(out)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def road_route(p_lat, p_lon, d_lat, d_lon):
    url = (f"https://router.project-osrm.org/route/v1/driving/"
           f"{p_lon},{p_lat};{d_lon},{d_lat}?overview=full&geometries=geojson")
    try:
        data = requests.get(url, headers=HEADERS, timeout=8).json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        r = data["routes"][0]
        return {"km": r["distance"] / 1000, "min": r["duration"] / 60, "path": r["geometry"]["coordinates"]}
    except Exception:
        return None

# Which extra inputs does the saved model expect? (empty for today's model)
NEEDS = set(bundle.get("features", [])) - {"pickup_longitude", "pickup_latitude", "dropoff_longitude",
                                           "dropoff_latitude", "passenger_count", "distance_km", "hour",
                                           "day_of_week", "month", "year", "is_weekend", "is_rush_hour"}
# Model's own currency: today's model = USD, an INR-trained model sets bundle["currency"] = "INR"
MODEL_CUR = bundle.get("currency", "USD")
# App ride type -> dataset cab type (used only if the model learned cab_type)
CAB_OF_RIDE = {"🚗 Go": "Mini", "🚙 Premier": "Sedan", "🚐 XL": "SUV", "🛺 Auto": "Auto", "🏍️ Moto": "Bike"}
WEATHER_WORD = {"☀️ Clear": "Clear", "🌧️ Rain": "Rain", "⛈️ Storm": "Storm"}

def predict(p, d, passengers, when, road_km=None, extras=None):
    """extras = values the app fills automatically: duration_min, cab_type, weather, city."""
    row = {"pickup_longitude": p["lon"], "pickup_latitude": p["lat"],
           "dropoff_longitude": d["lon"], "dropoff_latitude": d["lat"],
           "passenger_count": passengers, "pickup_datetime": when.strftime("%Y-%m-%d %H:%M:%S")}
    from model_utils import haversine_km
    straight = float(haversine_km(p["lon"], p["lat"], d["lon"], d["lat"]))
    row["distance_km"] = road_km if road_km is not None else straight
    row.update(extras or {})
    X = make_features(pd.DataFrame([row]))
    X.loc[X.index[0], "distance_km"] = row["distance_km"]
    fare = max(float(MODEL.predict(X)[0]), 0)
    if MODEL_CUR == "INR":                     # app's pricing logic works in USD internally
        fare = fare / max(usd_to_inr_default(), 1)
    return fare, straight

def usd_to_inr_default():
    return st.session_state.get("usd_to_inr", 83.0)

IS_ONLINE = online()

# ============================================================ CURRENCY (by selected country)
@st.cache_data
def load_currencies():
    p = Path("data/currencies.csv")
    if not p.exists():
        return {}
    df = pd.read_csv(p, keep_default_na=False)
    return dict(zip(df["country"], df["code"]))

@st.cache_data(ttl=3600, show_spinner=False)
def live_rates():
    """Live exchange rates (units per 1 USD) from open.er-api.com - free, no key."""
    try:
        return requests.get("https://open.er-api.com/v6/latest/USD", headers=HEADERS, timeout=4).json()["rates"]
    except Exception:
        return None

CURRENCY = load_currencies()
LIVE = live_rates() if IS_ONLINE else None
SYMBOLS = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥", "AED": "AED ",
           "SGD": "S$", "AUD": "A$", "CAD": "C$", "CHF": "CHF ", "SAR": "SAR ", "BRL": "R$",
           "ZAR": "R ", "KRW": "₩", "THB": "฿", "MYR": "RM ", "IDR": "Rp ", "PKR": "Rs ", "LKR": "Rs ",
           "BDT": "৳", "NPR": "Rs ", "RUB": "₽", "TRY": "₺", "MXN": "MX$", "NZD": "NZ$", "HKD": "HK$"}
# Offline fallback: APPROXIMATE units per 1 USD - update before the demo if needed
FALLBACK = {"USD": 1.0, "INR": 83.0, "EUR": 0.92, "GBP": 0.79, "JPY": 150.0, "AED": 3.67, "SGD": 1.35,
            "AUD": 1.52, "CAD": 1.36, "CHF": 0.88, "SAR": 3.75, "CNY": 7.2, "ZAR": 18.5, "THB": 35.5,
            "MYR": 4.7, "NZD": 1.65, "HKD": 7.8, "BDT": 110.0, "NPR": 133.0, "LKR": 300.0, "PKR": 280.0}

# ============================================================ SIDEBAR: MODEL REPORT
with st.sidebar:
    st.markdown("## 🤖 Model report")
    st.markdown(f"<div class='fare-card' style='padding:16px'><div class='muted'>Model picked</div>"
                f"<div style='font-size:1.3rem;font-weight:800;color:#06C167'>🏆 {bundle['model_name']}</div>"
                f"<div class='muted'>Lowest test RMSE · trained on {bundle.get('rows', 0):,} rows</div></div>",
                unsafe_allow_html=True)
    allm = bundle.get("all_metrics", {bundle["model_name"]: bundle["metrics"]})
    st.markdown("#### All models tried")
    tbl = pd.DataFrame({n: {k: v for k, v in m.items() if k != "band_report"} for n, m in allm.items()}).T
    tbl = tbl.sort_values("RMSE")
    tbl.insert(0, "Pick", ["🏆" if n == bundle["model_name"] else "" for n in tbl.index])
    st.dataframe(tbl.round(3), use_container_width=True)
    st.caption("Lower MAE/RMSE = better · R² closer to 1 = better")
    with st.expander("📖 What the metrics mean"):
        st.markdown("- **MAE**: average error in $\n- **RMSE**: punishes big misses (used to pick)\n"
                    "- **R²**: share of fare variation explained\n- **MedianAE**: typical error\n"
                    "- **MAPE_%**: average % error\n- **Within_$2_%**: predictions within $2 of real fare\n"
                    "- **Train_sec**: training time")
    st.markdown("#### Classification report (fare bands)")
    if bundle.get("importance"):
        st.markdown("#### What drives the fare")
        imp = pd.Series(bundle["importance"]).sort_values(ascending=True)
        nice = {"distance_km": "Distance", "hour": "Hour", "day_of_week": "Weekday", "month": "Month",
                "year": "Year", "is_weekend": "Weekend", "is_rush_hour": "Rush hour",
                "passenger_count": "Passengers"}
        imp.index = [nice.get(i, i) for i in imp.index]
        for name, v in imp.iloc[::-1].items():
            st.markdown(f"<div style='font-size:.85rem'>{name} <span class='muted'>{v:.1f}%</span></div>"
                        f"<div style='background:#1b252f;border-radius:6px;height:8px;margin:2px 0 8px'>"
                        f"<div style='width:{max(v, 1):.0f}%;background:#06C167;height:8px;border-radius:6px'></div></div>",
                        unsafe_allow_html=True)
        st.caption("Permutation importance: how much error rises when a feature is shuffled.")
        st.markdown("#### Classification report (fare bands)")
    st.caption("Fares grouped into Low / Medium / High / Very high, then scored like a classifier.")
    if any("band_report" in m for m in allm.values()):
        for n, m in allm.items():
            if "band_report" not in m:
                continue
            rep = m["band_report"]
            with st.expander(f"{'🏆 ' if n == bundle['model_name'] else ''}{n} · accuracy {rep['accuracy']:.1%}",
                             expanded=n == bundle["model_name"]):
                rdf = pd.DataFrame({k: v for k, v in rep.items() if isinstance(v, dict)}).T
                rdf["support"] = rdf["support"].astype(int)
                st.dataframe(rdf.round(3), use_container_width=True)
    else:
        st.info("Retrain (python train_model.py) to see the classification report.")

COUNTRIES = sorted(PLACES["country"].unique())
ICON = {"Landmark": "⭐", "City": "🏙️", "Live": "📍"}

# Ride types: multipliers are ILLUSTRATIVE assumptions, not from the dataset
RIDES = {"🚗 Go": 1.0, "🚙 Premier": 1.35, "🚐 XL": 1.6, "🛺 Auto": 0.6, "🏍️ Moto": 0.45}

# ---- Business rules on top of the ML fare (ILLUSTRATIVE, like real ride apps) ----
WEATHER = {"☀️ Clear": (0.0, 1.0), "🌧️ Rain": (0.3, 1.2), "⛈️ Storm": (0.6, 1.4)}      # (surge add, time factor)
TRAFFIC = {"🟢 Light": (0.0, 0.9), "🟡 Moderate": (0.1, 1.0), "🔴 Heavy": (0.3, 1.35)}
AIRPORT_FEE, BOOKING_FEE, GST, NIGHT_MULT, SURGE_CAP = 3.0, 1.0, 0.05, 1.25, 2.5
PROMOS = {"HACK20": 0.20, "FIRST50": 0.50}
CARS = {"🚗 Go": ["Maruti Swift Dzire", "Toyota Etios", "Hyundai Aura"],
        "🚙 Premier": ["Honda City", "Toyota Camry", "Skoda Slavia"],
        "🚐 XL": ["Toyota Innova Crysta", "Maruti Ertiga", "Kia Carens"],
        "🛺 Auto": ["Bajaj RE Auto"], "🏍️ Moto": ["Honda Activa", "Hero Splendor"]}
DRIVERS = ["Rahul", "Amit", "Priya", "Suresh", "Neha", "Vikram", "Arjun", "Sana", "John", "Maria"]

@st.cache_data(ttl=900, show_spinner=False)
def live_weather(lat, lon):
    """Current weather from Open-Meteo (free, no key). Returns WMO weather code or None if offline."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude": lat, "longitude": lon, "current": "weather_code,temperature_2m"},
                         headers=HEADERS, timeout=4).json()["current"]
        return int(r["weather_code"]), float(r["temperature_2m"])
    except Exception:
        return None

def auto_conditions(lat, lon, when):
    """Work out weather + traffic automatically - the user selects nothing."""
    # Traffic: from the time and day (rush hours on weekdays, quiet at night)
    h, wk = when.hour, when.weekday() < 5
    if wk and h in (8, 9, 10, 17, 18, 19, 20):
        traffic = "🔴 Heavy"
    elif 7 <= h <= 21:
        traffic = "🟡 Moderate"
    else:
        traffic = "🟢 Light"
    # Weather: live if online (only for rides within the next ~2 hours), else assume clear
    w, temp, src = "☀️ Clear", None, "assumed (offline)"
    if abs((when - datetime.now()).total_seconds()) < 2 * 3600:
        res = live_weather(round(lat, 2), round(lon, 2))
        if res:
            code, temp = res
            w = "⛈️ Storm" if code >= 95 else "🌧️ Rain" if code >= 51 else "☀️ Clear"
            src = "live (Open-Meteo)"
    else:
        src = "assumed (future trip)"
    return w, traffic, temp, src

def price(ml_fare, when, rush, weather, traffic, airport, promo):
    """Turn the ML fare into a full, realistic price with a breakdown."""
    night = when.hour >= 23 or when.hour < 5
    weekend_eve = when.weekday() >= 5 and 18 <= when.hour <= 23
    surge = 1.0 + (0.3 if rush else 0) + (0.2 if weekend_eve else 0) + WEATHER[weather][0] + TRAFFIC[traffic][0]
    surge = round(min(surge, SURGE_CAP), 1)
    surge_amt = ml_fare * (surge - 1)
    night_amt = (ml_fare + surge_amt) * (NIGHT_MULT - 1) if night else 0.0
    airport_amt = AIRPORT_FEE if airport else 0.0
    disc = PROMOS.get(promo.strip().upper(), 0) * (ml_fare + surge_amt + night_amt)
    sub = ml_fare + surge_amt + night_amt + airport_amt + BOOKING_FEE - disc
    tax = sub * GST
    return {"surge": surge, "night": night, "lines": [
        ("ML trip fare (model)", ml_fare), (f"Surge {surge}×", surge_amt), ("Night charge 25%", night_amt),
        ("Airport fee", airport_amt), ("Booking fee", BOOKING_FEE), ("Promo discount", -disc), ("GST 5%", tax)],
        "total": max(sub + tax, 0)}

# ============================================================ PLACE PICKER
def place_picker(title, key, country, default_name, dot):
    """ONE dropdown: click it and start typing - matching places appear instantly in the list."""
    st.markdown(f"<div class='step'>{dot} {title}</div>", unsafe_allow_html=True)
    pool = PLACES[PLACES["country"] == country].reset_index(drop=True)
    labels = [f"{ICON.get(t, '📍')}  {n}" for n, t in zip(pool["name"], pool["type"])]
    match = pool.index[pool["name"] == default_name]
    i = st.selectbox("Place (click and type to search)", range(len(pool)),
                     index=int(match[0]) if len(match) else None,
                     format_func=lambda i: labels[i], key=f"s_{key}_{country}",
                     placeholder="🔎 Type a place, e.g. Airport, Gate, Connaught…")
    if i is None:
        return None
    row = pool.loc[i]
    st.caption(f"📍 {row['name']}, {row['country']}")
    return {"label": f"{row['name']}, {row['country']}", "lat": float(row["lat"]), "lon": float(row["lon"])}

# ============================================================ INPUTS
box1 = st.container(border=True)
box1.markdown("### 📍 Where to?")
country = box1.selectbox("🌍 Country (used for pickup and drop-off)", COUNTRIES,
                         index=COUNTRIES.index("India"), key="country")
DEFAULTS = {"India": ("India Gate, New Delhi", "Indira Gandhi Airport, Delhi"),
            "United States": ("Times Square", "JFK Airport")}
a, swap_col, b = box1.columns([1, 0.08, 1])
with a:
    pickup = place_picker("PICKUP", "p", country, DEFAULTS.get(country, ("", ""))[0], "🟢")
with swap_col:
    st.markdown("<div style='font-size:1.8rem;text-align:center;margin-top:40px'>⇄</div>", unsafe_allow_html=True)
with b:
    dropoff = place_picker("DROP-OFF", "d", country, DEFAULTS.get(country, ("", ""))[1], "🔴")

box2 = st.container(border=True)
box2.markdown("### 🕒 When & how?")
now = datetime.now()
next_slot = (now + timedelta(minutes=15 - now.minute % 15)).replace(second=0, microsecond=0)
slots = [time(h, m) for h in range(24) for m in (0, 15, 30, 45)]
t1, t2, t3, t4 = box2.columns([1.1, 1, 1, 1.3])
mode = t1.radio("Pickup", ["⚡ Now", "📅 Schedule"], horizontal=True)
if mode == "📅 Schedule":
    day_opts = [date.today() + timedelta(days=k) for k in range(31)]
    pick_date = t2.selectbox("Date", day_opts,
                             format_func=lambda d: "Today" if d == date.today()
                             else "Tomorrow" if d == date.today() + timedelta(days=1)
                             else d.strftime("%a, %d %b"))
    pick_time = t3.selectbox("Time", slots, index=slots.index(next_slot.time()),
                             format_func=lambda t: t.strftime("%I:%M %p"))
    when = datetime.combine(pick_date, pick_time)
else:
    when = now
    t2.metric("Date", now.strftime("%d %b"))
    t3.metric("Time", now.strftime("%I:%M %p"))
passengers = t4.select_slider("👥 Passengers", options=[1, 2, 3, 4, 5, 6], value=1)
promo = ""
ride = box2.radio("Ride type", list(RIDES), horizontal=True)

with st.expander("⚙️ Settings"):
    s1, s2, s3 = st.columns(3)
    use_road = s1.toggle("Real road distance (needs internet)", value=IS_ONLINE, disabled=not IS_ONLINE)
    usd_to_inr = s2.number_input("USD → INR rate", value=float(round(LIVE["INR"], 2)) if LIVE and "INR" in LIVE else 83.0, min_value=1.0, step=0.5,
                                 help="Update to today's rate before the demo", key="usd_to_inr")
    show_hourly = s3.toggle("Show best time to ride", value=True)
    st.caption(("🟢 Online: live search + road routes on." if IS_ONLINE else
                "🟠 Offline: using built-in worldwide place list and straight-line distance."))

CODE = CURRENCY.get(country, "USD")
if CODE == "INR":
    RATE, RSRC = usd_to_inr, "settings"
elif LIVE and CODE in LIVE:
    RATE, RSRC = float(LIVE[CODE]), "live"
elif CODE in FALLBACK:
    RATE, RSRC = FALLBACK[CODE], "approx. offline"
else:
    CODE, RATE, RSRC = "USD", 1.0, "no rate found - showing USD"
SYM = SYMBOLS.get(CODE, CODE + " ")
DEC = 0 if CODE in ("JPY", "KRW", "IDR") else 2
CUR_COL = f"Fare ({CODE})"

def money(usd):
    """Fare in the pickup country's currency."""
    return f"{SYM}{usd * RATE:,.{DEC}f}"

def second(usd):
    """Line shown under the main fare: INR, or USD when the country already uses INR."""
    return f"${usd:,.2f} USD" if CODE == "INR" else f"₹{usd * usd_to_inr:,.0f} INR"

st.caption(f"💱 Showing fares in **{CODE}** for {country} "
           f"(1 USD = {RATE:,.2f} {CODE}, {RSRC})" + ("" if CODE == "INR" else " · INR shown below"))

go = st.button("🚀  Get fare estimate", type="primary", use_container_width=True)

# ============================================================ RESULT
if go:
    if pickup is None or dropoff is None:
        st.warning("Please choose both a pickup and a drop-off place.")
        st.stop()
    if abs(pickup["lat"] - dropoff["lat"]) < 1e-5 and abs(pickup["lon"] - dropoff["lon"]) < 1e-5:
        st.warning("Pickup and drop-off are the same place.")
        st.stop()

    route = None
    if use_road:
        with st.spinner("Finding the best road route…"):
            route = road_route(pickup["lat"], pickup["lon"], dropoff["lat"], dropoff["lon"])
    weather, traffic, temp, wsrc = auto_conditions(pickup["lat"], pickup["lon"], when)
    from model_utils import haversine_km
    _km0 = route["km"] if route else float(haversine_km(pickup["lon"], pickup["lat"], dropoff["lon"], dropoff["lat"]))
    _dur = (route["min"] if route else _km0 / 25 * 60) * WEATHER[weather][1] * TRAFFIC[traffic][1]
    EXTRAS = {"duration_min": round(_dur, 1), "cab_type": CAB_OF_RIDE[ride],
              "weather": WEATHER_WORD[weather], "city": pickup["label"].split(",")[0]}
    base, straight_km = predict(pickup, dropoff, passengers, when, route["km"] if route else None, EXTRAS)
    km = route["km"] if route else straight_km
    if NEEDS:
        st.caption("🧠 Extra model inputs filled automatically: " +
                   ", ".join(f"{k}={EXTRAS.get(k.replace('_code', '') if k != 'cab_code' else 'cab_type', '-')}"
                             for k in sorted(NEEDS)))
    mult = 1.0 if "cab_code" in NEEDS else RIDES[ride]     # don't double-count ride type
    rush = when.hour in [7, 8, 9, 16, 17, 18, 19]
    airport = "airport" in (pickup["label"] + dropoff["label"]).lower()
    P = price(base * mult, when, rush, weather, traffic, airport, promo)
    fare = P["total"]
    scale = fare / max(base * mult, 0.01)
    low, high = max(fare - MAE * mult * scale, 0), fare + MAE * mult * scale
    eta = (route["min"] if route else km / 25 * 60) * WEATHER[weather][1] * TRAFFIC[traffic][1]
    st.markdown(f"""<div class="glass" style="padding:12px 18px;margin:8px 0;border-radius:14px;
        background:rgba(6,193,103,.07);border:1px solid #06C16744">
        <b>📡 Conditions detected automatically</b> &nbsp;·&nbsp; {weather}{f' {temp:.0f}°C' if temp is not None else ''}
        <span class="muted">({wsrc})</span> &nbsp;·&nbsp; {traffic} <span class="muted">(from time &amp; day)</span>
        &nbsp;·&nbsp; {'🔥 Rush hour' if rush else '✅ Off-peak'}
        {f'&nbsp;·&nbsp; ⚡ {P["surge"]}× surge' if P["surge"] > 1 else ''}</div>""", unsafe_allow_html=True)

    st.markdown("### 🧾 Your trip")
    left, right = st.columns([1, 1.5])
    with left:
        st.markdown(f"""
        <div class="fare-card">
          <div class="muted">{ride} · {passengers} rider(s) · {when.strftime('%a %d %b, %I:%M %p')}</div>
          <div class="fare" style="margin-top:10px">{money(fare)}</div>
          <div class="inr">{second(fare)}</div>
          <div class="muted">Likely range {money(low)} – {money(high)}</div>
          <div class="route">
            <div><span class="dot-g">●</span> {pickup['label']}</div>
            <div class="muted" style="margin:6px 0">│ {km:.1f} km · {eta:.0f} min</div>
            <div><span class="dot-r">●</span> {dropoff['label']}</div>
          </div>
          <span class="chip">🛣️ {'Road' if route else 'Straight-line'} {km:.1f} km</span>
          <span class="chip">⏱️ {'~' if not route else ''}{eta:.0f} min</span>
          <span class="chip">{'🔥 Rush hour' if rush else '✅ Off-peak'}</span>
          {f'<span class="chip">⚡ {P["surge"]}× surge</span>' if P["surge"] > 1 else ''}
          {'<span class="chip">🌙 Night</span>' if P["night"] else ''}
          {'<span class="chip">✈️ Airport</span>' if airport else ''}
          <span class="chip">{weather} · {traffic}</span>
        </div>""", unsafe_allow_html=True)
        rng = random.Random(pickup["label"] + dropoff["label"] + ride)
        car = rng.choice(CARS[ride])
        plate = f"{rng.choice(['DL','MH','KA','TS','TN'])} {rng.randint(1,20):02d} {rng.choice('ABCDEFGH')}{rng.choice('JKLMNPRS')} {rng.randint(1000,9999)}"
        st.markdown(f"""<div class="fare-card" style="margin-top:12px;padding:16px">
          <div style="font-size:1.1rem;font-weight:700">🧑‍✈️ {rng.choice(DRIVERS)} · ⭐ {rng.uniform(4.5, 4.98):.2f}</div>
          <div class="muted">{car} · {plate}</div>
          <div style="margin-top:6px">🚖 Arriving in <b>{rng.randint(2, 9)} min</b></div></div>""",
                    unsafe_allow_html=True)
        with st.expander("🧾 Fare breakdown", expanded=True):
            bd = pd.DataFrame([(k, round(v * RATE, DEC), round(v * usd_to_inr)) for k, v in P["lines"] if abs(v) > 0.004],
                              columns=["Item", CODE, "INR (₹)"])
            bd.loc[len(bd)] = ["Total", round(fare * RATE, DEC), round(fare * usd_to_inr)]
            st.dataframe(bd, hide_index=True, use_container_width=True)
            st.caption("The ML model predicts the trip fare; surge, fees and tax are business rules on top.")
        if passengers > 1:
            st.info(f"👥 Split {passengers} ways: **{money(fare / passengers)}** ({second(fare / passengers)}) each")
        if km > 100:
            st.warning("Longer than any trip in the training data (>100 km) - estimate may be unreliable.")

    with right:
        pts = pd.DataFrame({"lon": [pickup["lon"], dropoff["lon"]], "lat": [pickup["lat"], dropoff["lat"]],
                            "color": [[6, 193, 103], [255, 90, 90]]})
        layers = [pdk.Layer("ScatterplotLayer", pts, get_position="[lon, lat]", get_fill_color="color",
                            get_radius=200, radius_min_pixels=9, stroked=True, get_line_color=[255, 255, 255],
                            line_width_min_pixels=2)]
        path = route["path"] if route else [[pickup["lon"], pickup["lat"]], [dropoff["lon"], dropoff["lat"]]]
        layers.insert(0, pdk.Layer("PathLayer", pd.DataFrame({"path": [path]}), get_path="path",
                                   get_color=[6, 193, 103], width_min_pixels=5))
        zoom = 12 if km < 10 else 10 if km < 40 else 8 if km < 200 else 5
        st.pydeck_chart(pdk.Deck(map_style="dark", layers=layers, initial_view_state=pdk.ViewState(
            latitude=(pickup["lat"] + dropoff["lat"]) / 2, longitude=(pickup["lon"] + dropoff["lon"]) / 2,
            zoom=zoom, pitch=35)), height=380)

    st.markdown("#### 🚘 Compare rides")
    cols = st.columns(len(RIDES))
    for col, (name, m) in zip(cols, RIDES.items()):
        if "cab_code" in NEEDS:      # model learned cab prices itself -> no multiplier
            b_ride, _ = predict(pickup, dropoff, passengers, when, route["km"] if route else None,
                                {**EXTRAS, "cab_type": CAB_OF_RIDE[name]})
            t = price(b_ride, when, rush, weather, traffic, airport, promo)["total"]
        else:
            t = price(base * m, when, rush, weather, traffic, airport, promo)["total"]
        col.metric(name, money(t), second(t), delta_color="off")
    st.caption("Ride-type multipliers are illustrative assumptions.")

    if show_hourly:
        st.markdown("#### ⏰ Best time to ride today")
        hrs = []
        for h in range(24):
            f, _ = predict(pickup, dropoff, passengers, when.replace(hour=h, minute=0),
                           route["km"] if route else None, EXTRAS)
            hrs.append({"Hour": f"{h:02d}:00", CUR_COL: round(f * mult * RATE, DEC)})
        hdf = pd.DataFrame(hrs).set_index("Hour")
        best = hdf[CUR_COL].idxmin()
        save = hdf[CUR_COL].max() - hdf[CUR_COL].min()
        st.bar_chart(hdf, color="#06C167", height=260)
        st.success(f"Cheapest at **{best}** (≈ {SYM}{hdf.loc[best, CUR_COL]:,.{DEC}f}). "
                   f"You could save up to {SYM}{save:,.{DEC}f} vs the most expensive hour.")

    st.session_state.setdefault("history", []).insert(0, {
        "When": when.strftime("%d %b %I:%M %p"), "From": pickup["label"], "To": dropoff["label"],
        "Ride": ride, "Km": round(km, 1), CUR_COL: round(fare * RATE, DEC), "Fare (₹)": round(fare * usd_to_inr)})

if st.session_state.get("history"):
    with st.expander(f"🕘 Recent estimates ({len(st.session_state['history'])})"):
        hist = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist, hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download CSV", hist.to_csv(index=False), "fare_estimates.csv", "text/csv")
        if c2.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

with st.expander("ℹ️ How it works"):
    st.write(
        f"**Model:** {bundle['model_name']} · test MAE ${MAE:.2f} · R² {bundle['metrics']['R2']:.3f} (full report in the sidebar)\n\n"
        "1. **Places:** a built-in list of 34,000+ cities in 244 countries plus landmarks and airports "
        "(works offline). When online, live OpenStreetMap results are added.\n"
        "2. **Distance:** real road route from OSRM when online; otherwise straight-line (Haversine).\n"
        "3. **Time features:** hour, weekday, month, year, weekend, rush hour.\n"
        "4. **Prediction:** the model uses distance + time + passengers, not the exact location, "
        "so it works in any city. Prices follow the training data (USD); ₹ is a conversion.\n"
        "5. **Range** = estimate ± the model's average test error. Ride types use illustrative multipliers."
    )
st.markdown("<div class='muted' style='text-align:center;margin-top:20px'>Built with Python · scikit-learn · Streamlit · OpenStreetMap</div>",
            unsafe_allow_html=True)
