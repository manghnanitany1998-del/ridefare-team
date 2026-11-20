"""
RideFare - ride fare predictor with an Uber-style interface.
Run:  python -m streamlit run app.py

Works OFFLINE: places come from data/places.csv (34,000+ cities in 244 countries
+ landmarks/airports). Online, it adds live OpenStreetMap results, road routes (OSRM),
live weather (Open-Meteo) and live exchange rates.
"""
from pathlib import Path
from datetime import datetime, date, time, timedelta
import random
import joblib
import pandas as pd
import requests
import pydeck as pdk
import streamlit as st
from model_utils import make_features, haversine_km

st.set_page_config(page_title="RideFare", page_icon="🚕", layout="wide", initial_sidebar_state="collapsed")

# ============================================================ STYLE (Uber-like: black & white, bold type)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, button, input, textarea, select {font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;}
.stApp {background: #ffffff;}
header[data-testid="stHeader"] {background: transparent; height: 0;}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 0.6rem; max-width: 1280px;}
h1, h2, h3, h4, p, label, span, div {color: #000;}

/* top navigation bar */
.nav {background:#000; border-radius: 0 0 18px 18px; padding: 16px 28px; display:flex; align-items:center;
      gap: 28px; margin-bottom: 30px;}
.nav .logo {font-size: 1.6rem; font-weight: 800; color:#fff; letter-spacing:-.5px;}
.nav .lk {color:#fff; font-weight:500; font-size:.95rem; opacity:.92;}
.nav .rt {margin-left:auto; display:flex; gap:10px;}
.nav .pill {background:#fff; color:#000; border-radius:999px; padding:7px 14px; font-size:.85rem; font-weight:600;}
.nav .ghost {color:#fff; border:1px solid #444; border-radius:999px; padding:7px 14px; font-size:.85rem; font-weight:500;}

/* headings */
.h-title {font-size: 2.6rem; font-weight: 800; letter-spacing: -1.2px; line-height:1.08; margin: 6px 0 6px;}
.h-sub {color:#5e5e5e !important; font-size: 1rem; margin-bottom: 18px;}
.sec {font-size: 1.3rem; font-weight: 700; letter-spacing:-.3px; margin: 22px 0 10px;}
.lbl {font-size:.78rem; font-weight:600; color:#5e5e5e !important; margin: 12px 0 4px; text-transform: uppercase; letter-spacing:.06em;}
.dot {display:inline-block; width:10px; height:10px; border-radius:50%; background:#000; margin-right:8px; vertical-align:middle;}
.sq {display:inline-block; width:10px; height:10px; background:#000; margin-right:8px; vertical-align:middle;}
.muted {color:#5e5e5e !important; font-size:.86rem;}

/* inputs: grey filled boxes like Uber */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stDateInput input {
    background:#f3f3f3 !important; border: 2px solid #f3f3f3 !important; border-radius: 10px !important; min-height: 50px;}
div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"] > div:focus-within {border-color:#000 !important; background:#fff !important;}
div[data-baseweb="popover"] li {font-size:.95rem;}

/* buttons */
div.stButton > button[kind="primary"], div.stDownloadButton > button {background:#000 !important; color:#fff !important;
    border:0 !important; border-radius:10px !important; height:52px; font-weight:600 !important; font-size:1.05rem !important;}
div.stButton > button[kind="primary"] p, div.stDownloadButton > button p {color:#fff !important;}
div.stButton > button[kind="primary"]:hover {background:#333 !important;}
div.stButton > button[kind="secondary"] {background:#eeeeee !important; color:#000 !important; border:0 !important; border-radius:999px !important;}

/* ride list = radio styled as Uber ride rows */
[data-testid="stRadio"] div[role="radiogroup"] {gap: 8px; width:100%;}
[data-testid="stRadio"] div[role="radiogroup"] > label {width:100%; border:2px solid #eeeeee; border-radius:14px;
    padding: 14px 16px; margin:0; background:#fff; transition: all .12s;}
[data-testid="stRadio"] div[role="radiogroup"] > label:hover {border-color:#cfcfcf;}
[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {border-color:#000; background:#fafafa;}
[data-testid="stRadio"] div[role="radiogroup"] > label p {font-size: 1rem;}

/* cards, chips */
.card {border:1px solid #eeeeee; border-radius:16px; padding:18px 20px; background:#fff; box-shadow:0 6px 24px rgba(0,0,0,.06);}
.fare {font-size: 2.6rem; font-weight: 800; letter-spacing:-1px; line-height:1;}
.fare2 {font-size: 1.1rem; font-weight: 600; color:#5e5e5e !important; margin-top: 4px;}
.chip {display:inline-block; background:#f3f3f3; border-radius:999px; padding:6px 12px; margin:6px 6px 0 0; font-size:.82rem; font-weight:500;}
.chip.dark {background:#000; color:#fff;}
.tl {position:relative; padding-left:26px; margin:14px 0 4px;}
.tl .a, .tl .b {position:relative; margin: 0 0 18px; font-weight:600;}
.tl .a:before {content:""; position:absolute; left:-24px; top:5px; width:10px; height:10px; border-radius:50%; background:#000;}
.tl .b:before {content:""; position:absolute; left:-24px; top:5px; width:10px; height:10px; background:#000;}
.tl:before {content:""; position:absolute; left:6px; top:14px; bottom:30px; width:2px; background:#000;}
.tl small {display:block; font-weight:400; color:#5e5e5e;}
.drv {display:flex; gap:14px; align-items:center;}
.avatar {width:56px; height:56px; border-radius:50%; background:#000; color:#fff !important; display:flex; align-items:center;
         justify-content:center; font-weight:700; font-size:1.3rem;}
.plate {background:#f3f3f3; border-radius:6px; padding:3px 9px; font-weight:700; letter-spacing:1px; font-size:.9rem;}
.empty {background:#f6f6f6; border-radius:16px; padding:26px; text-align:center; color:#5e5e5e !important;}

/* map + sidebar */
[data-testid="stDeckGlJsonChart"] {border-radius:18px; overflow:hidden;}
[data-testid="stSidebar"] {background:#f6f6f6;}
[data-testid="stSidebar"] .mcard {background:#000; border-radius:14px; padding:14px 16px;}
[data-testid="stSidebar"] .mcard * {color:#fff !important;}
[data-testid="stExpander"] {border:1px solid #eeeeee !important; border-radius:14px !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="nav">
  <span class="logo">RideFare</span>
  <span class="lk">Ride</span><span class="lk">Drive</span><span class="lk">Business</span><span class="lk">About</span>
  <div class="rt"><span class="ghost">🤖 ML fare estimator</span><span class="pill">🌍 244 countries</span></div>
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
                                           "day_of_week", "month", "year", "is_weekend", "is_rush_hour",
                                           "city_code", "payment_code"}
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



SEATS = {"🚗 Go": 4, "🚙 Premier": 4, "🚐 XL": 6, "🛺 Auto": 3, "🏍️ Moto": 1}
DESC = {"🚗 Go": "Affordable compact rides", "🚙 Premier": "Comfortable sedans, top-rated drivers",
        "🚐 XL": "SUVs for groups up to 6", "🛺 Auto": "Auto rickshaw rides", "🏍️ Moto": "Beat traffic on a bike"}
def nm(r):
    return r.split(" ", 1)[1]

# ============================================================ SIDEBAR: SETTINGS + MODEL REPORT
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    use_road = st.toggle("Real road distance (needs internet)", value=IS_ONLINE, disabled=not IS_ONLINE)
    usd_to_inr = st.number_input("USD → INR rate", value=float(round(LIVE["INR"], 2)) if LIVE and "INR" in LIVE else 83.0,
                                 min_value=1.0, step=0.5, help="Update to today's rate before the demo", key="usd_to_inr")
    show_hourly = st.toggle("Show best time to ride", value=True)
    st.caption("🟢 Online: live search, routes, weather, rates." if IS_ONLINE else
               "🟠 Offline: built-in places, straight-line distance, assumed clear weather.")
    st.divider()
    st.markdown("### 🤖 Model report")
    st.markdown(f"<div class='mcard'><div style='font-size:.8rem;opacity:.8'>Model picked</div>"
                f"<div style='font-size:1.15rem;font-weight:800'>🏆 {bundle['model_name']}</div>"
                f"<div style='font-size:.8rem;opacity:.8'>Lowest test RMSE · {bundle.get('rows', 0):,} rows</div></div>",
                unsafe_allow_html=True)
    allm = bundle.get("all_metrics", {bundle["model_name"]: bundle["metrics"]})
    st.markdown("#### All models tried")
    tbl = pd.DataFrame({n: {k: v for k, v in m.items() if k != "band_report"} for n, m in allm.items()}).T.sort_values("RMSE")
    tbl.insert(0, "Pick", ["🏆" if n == bundle["model_name"] else "" for n in tbl.index])
    st.dataframe(tbl.round(3), use_container_width=True)
    st.caption("Lower MAE/RMSE = better · R² closer to 1 = better")
    with st.expander("📖 What the metrics mean"):
        st.markdown("- **MAE**: average error\n- **RMSE**: punishes big misses (used to pick)\n"
                    "- **R²**: share of fare variation explained\n- **MedianAE**: typical error\n"
                    "- **MAPE_%**: average % error\n- **Within_$2_%**: predictions within $2\n- **Train_sec**: training time")
    if bundle.get("importance"):
        st.markdown("#### What drives the fare")
        nice = {"distance_km": "Distance", "hour": "Hour", "day_of_week": "Weekday", "month": "Month", "year": "Year",
                "is_weekend": "Weekend", "is_rush_hour": "Rush hour", "passenger_count": "Passengers",
                "duration_min": "Duration", "cab_code": "Cab type", "weather_code": "Weather", "city_code": "City"}
        for k, v in sorted(bundle["importance"].items(), key=lambda x: -x[1]):
            st.markdown(f"<div style='font-size:.85rem'>{nice.get(k, k)} <span class='muted'>{v:.1f}%</span></div>"
                        f"<div style='background:#e3e3e3;border-radius:6px;height:8px;margin:2px 0 8px'>"
                        f"<div style='width:{max(v, 1):.0f}%;background:#000;height:8px;border-radius:6px'></div></div>",
                        unsafe_allow_html=True)
        st.caption("Permutation importance: how much error rises when a feature is shuffled.")
    st.markdown("#### Classification report (fare bands)")
    st.caption("Fares grouped into bands, then scored like a classifier.")
    if any("band_report" in m for m in allm.values()):
        for n, m in allm.items():
            if "band_report" in m:
                rep = m["band_report"]
                with st.expander(f"{'🏆 ' if n == bundle['model_name'] else ''}{n} · accuracy {rep['accuracy']:.1%}",
                                 expanded=n == bundle["model_name"]):
                    rdf = pd.DataFrame({k: v for k, v in rep.items() if isinstance(v, dict)}).T
                    rdf["support"] = rdf["support"].astype(int)
                    st.dataframe(rdf.round(3), use_container_width=True)
    else:
        st.info("Retrain (python train_model.py) to see the classification report.")

# ============================================================ PLACE PICKER
def place_picker(label, key, country, default_name, shape):
    st.markdown(f"<div class='lbl'><span class='{shape}'></span>{label}</div>", unsafe_allow_html=True)
    pool = PLACES[PLACES["country"] == country].reset_index(drop=True)
    labels = [f"{ICON.get(t, '📍')}  {n}" for n, t in zip(pool["name"], pool["type"])]
    match = pool.index[pool["name"] == default_name]
    i = st.selectbox(label, range(len(pool)), index=int(match[0]) if len(match) else None,
                     format_func=lambda i: labels[i], key=f"s_{key}_{country}", label_visibility="collapsed",
                     placeholder="🔎 Search a place…")
    if i is None:
        return None
    row = pool.loc[i]
    return {"label": f"{row['name']}, {row['country']}", "short": row["name"], "lat": float(row["lat"]), "lon": float(row["lon"])}

DEFAULTS = {"India": ("India Gate, New Delhi", "Indira Gandhi Airport, Delhi"),
            "United States": ("Times Square", "JFK Airport")}

# ============================================================ LAYOUT
L, R = st.columns([1, 1.45], gap="large")

with L:
    st.markdown("<div class='h-title'>Go anywhere<br>with RideFare</div>"
                "<div class='h-sub'>Request a ride, hop in, and go. Fares predicted by machine learning.</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='lbl'>🌍 Country</div>", unsafe_allow_html=True)
    country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("India"), key="country", label_visibility="collapsed")
    dfl = DEFAULTS.get(country, ("", ""))
    pickup = place_picker("Pickup location", "p", country, dfl[0], "dot")
    dropoff = place_picker("Dropoff location", "d", country, dfl[1], "sq")

    st.markdown("<div class='lbl'>🕒 When</div>", unsafe_allow_html=True)
    mode = st.selectbox("When", ["Pickup now", "Schedule for later"], label_visibility="collapsed")
    now = datetime.now()
    if mode == "Schedule for later":
        next_slot = (now + timedelta(minutes=15 - now.minute % 15)).replace(second=0, microsecond=0)
        slots = [time(h, m) for h in range(24) for m in (0, 15, 30, 45)]
        c1, c2 = st.columns(2)
        day_opts = [date.today() + timedelta(days=k) for k in range(31)]
        pick_date = c1.selectbox("Date", day_opts, label_visibility="collapsed",
                                 format_func=lambda d: "📅 Today" if d == date.today() else
                                 "📅 Tomorrow" if d == date.today() + timedelta(days=1) else "📅 " + d.strftime("%a, %d %b"))
        pick_time = c2.selectbox("Time", slots, index=slots.index(next_slot.time()), label_visibility="collapsed",
                                 format_func=lambda t: "🕒 " + t.strftime("%I:%M %p"))
        when = datetime.combine(pick_date, pick_time)
        when_key = when.isoformat()
    else:
        when, when_key = now, "now"
    st.markdown("<div class='lbl'>👤 Riders</div>", unsafe_allow_html=True)
    passengers = st.selectbox("Riders", [1, 2, 3, 4, 5, 6], label_visibility="collapsed",
                              format_func=lambda n: f"👤 {n} rider" + ("s" if n > 1 else ""))
    st.markdown("<div class='lbl'>💳 Payment</div>", unsafe_allow_html=True)
    payment = st.selectbox("Payment", ["Card", "Cash", "UPI", "Wallet"], label_visibility="collapsed",
                           format_func=lambda p: {"Card": "💳 Card", "Cash": "💵 Cash", "UPI": "📱 UPI",
                                                  "Wallet": "👛 Wallet"}[p])
    sig = (country, pickup and pickup["label"], dropoff and dropoff["label"], when_key, passengers, payment)
    if st.button("See prices", type="primary", use_container_width=True):
        st.session_state["quote"] = sig
quoted = st.session_state.get("quote") == sig and pickup is not None and dropoff is not None

# ============================================================ CURRENCY (by selected country)
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
DEC = 0 if CODE in ("JPY", "KRW", "IDR", "INR") else 2
CUR_COL = f"Fare ({CODE})"

def money(usd):
    return f"{SYM}{usd * RATE:,.{DEC}f}"

def second(usd):
    return f"${usd:,.2f} USD" if CODE == "INR" else f"₹{usd * usd_to_inr:,.0f} INR"

def draw_map(p, d, path=None, height=560):
    pts, layers = [], []
    if p: pts.append({"lon": p["lon"], "lat": p["lat"], "c": [0, 0, 0]})
    if d: pts.append({"lon": d["lon"], "lat": d["lat"], "c": [0, 0, 0]})
    if p and d:
        line = path or [[p["lon"], p["lat"]], [d["lon"], d["lat"]]]
        layers.append(pdk.Layer("PathLayer", pd.DataFrame({"path": [line]}), get_path="path",
                                get_color=[0, 0, 0], width_min_pixels=5))
    if pts:
        layers.append(pdk.Layer("ScatterplotLayer", pd.DataFrame(pts), get_position="[lon, lat]", get_fill_color="c",
                                get_radius=180, radius_min_pixels=8, stroked=True, get_line_color=[255, 255, 255],
                                line_width_min_pixels=3))
    if p and d:
        km = float(haversine_km(p["lon"], p["lat"], d["lon"], d["lat"]))
        lat, lon = (p["lat"] + d["lat"]) / 2, (p["lon"] + d["lon"]) / 2
        zoom = 12 if km < 8 else 11 if km < 20 else 10 if km < 40 else 8 if km < 200 else 5
    elif pts:
        lat, lon, zoom = pts[0]["lat"], pts[0]["lon"], 11
    else:
        lat, lon, zoom = 20, 78, 3
    st.pydeck_chart(pdk.Deck(map_style="light", layers=layers,
                             initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=0)),
                    height=height)

# ============================================================ BEFORE QUOTE
if not quoted:
    with R:
        draw_map(pickup, dropoff)
        st.caption(f"💱 Fares in **{CODE}** for {country} (1 USD = {RATE:,.2f} {CODE}, {RSRC})"
                   + ("" if CODE == "INR" else " · INR shown alongside"))
    with L:
        st.markdown("<div class='empty' style='margin-top:16px'>Choose your pickup and dropoff, then tap "
                    "<b>See prices</b> to compare rides.</div>", unsafe_allow_html=True)

# ============================================================ AFTER QUOTE
else:
    route = None
    if use_road:
        with st.spinner("Finding the best route…"):
            route = road_route(pickup["lat"], pickup["lon"], dropoff["lat"], dropoff["lon"])
    weather, traffic, temp, wsrc = auto_conditions(pickup["lat"], pickup["lon"], when)
    _km0 = route["km"] if route else float(haversine_km(pickup["lon"], pickup["lat"], dropoff["lon"], dropoff["lat"]))
    eta = (route["min"] if route else _km0 / 25 * 60) * WEATHER[weather][1] * TRAFFIC[traffic][1]
    EXTRAS = {"duration_min": round(eta, 1), "cab_type": "Mini", "weather": WEATHER_WORD[weather],
              "city": pickup["label"].split(",")[0], "payment_method": payment}
    rkm = route["km"] if route else None
    base, straight_km = predict(pickup, dropoff, passengers, when, rkm, EXTRAS)
    km = route["km"] if route else straight_km
    rush = when.hour in [7, 8, 9, 16, 17, 18, 19]
    airport = "airport" in (pickup["label"] + dropoff["label"]).lower()
    promo = ""

    def ride_ml(name, at=None):
        t = at or when
        if "cab_code" in NEEDS:   # model learned cab prices itself -> no multiplier
            b, _ = predict(pickup, dropoff, passengers, t, rkm, {**EXTRAS, "cab_type": CAB_OF_RIDE[name]})
            return b
        b = base if at is None else predict(pickup, dropoff, passengers, t, rkm, EXTRAS)[0]
        return b * RIDES[name]

    quotes = {n: price(ride_ml(n), when, rush, weather, traffic, airport, promo) for n in RIDES}
    rng0 = random.Random(pickup["label"] + dropoff["label"])
    away = {n: rng0.randint(2, 9) for n in RIDES}
    drop_time = (when + timedelta(minutes=eta)).strftime("%I:%M %p")
    available = [n for n in RIDES if SEATS[n] >= passengers]

    with L:
        st.markdown("<div class='sec'>Choose a ride</div>", unsafe_allow_html=True)
        def ride_label(n):
            q = quotes[n]
            surge = f"  ⚡{q['surge']}×" if q["surge"] > 1 else ""
            return (f"{n.split(' ')[0]}  **{nm(n)}**  ·  👤 {SEATS[n]}  ·  {away[n]} min away{surge}  \n"
                    f"{DESC[n]}  ·  **{money(q['total'])}**  ·  {second(q['total'])}")
        ride = st.radio("Ride", available, format_func=ride_label, label_visibility="collapsed", key="ride")
        P = quotes[ride]
        fare = P["total"]
        mult = 1.0 if "cab_code" in NEEDS else RIDES[ride]
        scale = fare / max(ride_ml(ride), 0.01)
        low, high = max(fare - MAE * mult * scale, 0), fare + MAE * mult * scale
        if len(available) < len(RIDES):
            st.caption(f"Some rides are hidden: not enough seats for {passengers} riders.")

        if st.button(f"Request {nm(ride)}", type="primary", use_container_width=True):
            st.session_state["requested"] = (sig, ride)
            st.toast("Ride requested (demo) – finding your driver…", icon="🚕")

        if st.session_state.get("requested") == (sig, ride):
            rng = random.Random(pickup["label"] + dropoff["label"] + ride)
            driver = rng.choice(DRIVERS)
            plate = (f"{rng.choice(['DL','MH','KA','TS','TN'])} {rng.randint(1,20):02d} "
                     f"{rng.choice('ABCDEFGH')}{rng.choice('JKLMNPRS')} {rng.randint(1000,9999)}")
            st.markdown(f"""<div class="card" style="margin-top:14px">
              <div class="muted">Your driver is on the way · arriving in <b>{away[ride]} min</b></div>
              <div class="drv" style="margin-top:10px">
                <div class="avatar">{driver[0]}</div>
                <div><div style="font-weight:700;font-size:1.1rem">{driver} · ⭐ {rng.uniform(4.6, 4.98):.2f}</div>
                <div class="muted">{rng.choice(CARS[ride])}</div></div>
                <div style="margin-left:auto"><span class="plate">{plate}</span></div>
              </div></div>""", unsafe_allow_html=True)

        with st.expander("🧾 Fare breakdown"):
            bd = pd.DataFrame([(k, round(v * RATE, DEC), round(v * usd_to_inr)) for k, v in P["lines"] if abs(v) > 0.004],
                              columns=["Item", CODE, "INR (₹)"])
            bd.loc[len(bd)] = ["Total", round(fare * RATE, DEC), round(fare * usd_to_inr)]
            st.dataframe(bd, hide_index=True, use_container_width=True)
            st.caption("The ML model predicts the trip fare; ride type, surge, fees and tax are business rules on top.")
        if passengers > 1:
            st.caption(f"👥 Split {passengers} ways: **{money(fare / passengers)}** each ({second(fare / passengers)})")

    with R:
        draw_map(pickup, dropoff, route["path"] if route else None, height=420)
        st.markdown(f"""<div class="card" style="margin-top:14px">
          <div style="display:flex;align-items:flex-start;gap:20px">
            <div style="flex:1">
              <div class="tl">
                <div class="a">{pickup['short']}<small>{when.strftime('%a %d %b · %I:%M %p')}</small></div>
                <div class="b">{dropoff['short']}<small>Drop-off ≈ {drop_time}</small></div>
              </div>
            </div>
            <div style="text-align:right">
              <div class="muted">{nm(ride)} · {passengers} rider{'s' if passengers > 1 else ''}</div>
              <div class="fare">{money(fare)}</div>
              <div class="fare2">{second(fare)}</div>
              <div class="muted">Likely {money(low)} – {money(high)}</div>
            </div>
          </div>
          <span class="chip">🛣️ {km:.1f} km {'road' if route else 'straight-line'}</span>
          <span class="chip">⏱️ {'~' if not route else ''}{eta:.0f} min</span>
          <span class="chip">{weather}{f' {temp:.0f}°C' if temp is not None else ''}</span>
          <span class="chip">{traffic} traffic</span>
          {f'<span class="chip dark">⚡ {P["surge"]}× surge</span>' if P["surge"] > 1 else '<span class="chip">✅ No surge</span>'}
          {'<span class="chip">🌙 Night</span>' if P["night"] else ''}
          {'<span class="chip">✈️ Airport</span>' if airport else ''}
          <div class="muted" style="margin-top:10px">📡 Weather {wsrc} · traffic from time &amp; day ·
          💱 1 USD = {RATE:,.2f} {CODE} ({RSRC})</div>
        </div>""", unsafe_allow_html=True)
        if km > 100:
            st.warning("Longer than any trip in the training data (>100 km) - estimate may be unreliable.")
        if NEEDS:
            st.caption("🧠 Extra model inputs filled automatically: duration, cab type, weather, city.")

        if show_hourly:
            st.markdown("<div class='sec'>Best time to ride today</div>", unsafe_allow_html=True)
            hrs = []
            for h in range(24):
                f = ride_ml(ride, when.replace(hour=h, minute=0))
                hrs.append({"Hour": f"{h:02d}:00", CUR_COL: round(f * RATE, DEC)})
            hdf = pd.DataFrame(hrs).set_index("Hour")
            best = hdf[CUR_COL].idxmin()
            save = hdf[CUR_COL].max() - hdf[CUR_COL].min()
            st.bar_chart(hdf, color="#000000", height=220)
            st.caption(f"Cheapest trip fare at **{best}** (≈ {SYM}{hdf.loc[best, CUR_COL]:,.{DEC}f}) · "
                       f"save up to {SYM}{save:,.{DEC}f} vs the priciest hour (before surge and fees).")

    hist_key = (sig, ride)
    if st.session_state.get("_last_hist") != hist_key:
        st.session_state["_last_hist"] = hist_key
        st.session_state.setdefault("history", []).insert(0, {
            "When": when.strftime("%d %b %I:%M %p"), "From": pickup["label"], "To": dropoff["label"],
            "Ride": nm(ride), "Km": round(km, 1), CUR_COL: round(fare * RATE, DEC), "Fare (₹)": round(fare * usd_to_inr)})

# ============================================================ FOOTER
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
f1, f2 = st.columns(2)
with f1:
    if st.session_state.get("history"):
        with st.expander(f"🕘 Recent estimates ({len(st.session_state['history'])})"):
            hist = pd.DataFrame(st.session_state["history"])
            st.dataframe(hist, hide_index=True, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.download_button("Download CSV", hist.to_csv(index=False), "fare_estimates.csv", "text/csv")
            if c2.button("Clear history"):
                st.session_state["history"] = []
                st.rerun()
with f2:
    with st.expander("ℹ️ How it works"):
        st.write(
            f"**Model:** {bundle['model_name']} · test MAE ${MAE:.2f} · R² {bundle['metrics']['R2']:.3f} "
            "(full report in the sidebar, open it with **>** at the top left)\n\n"
            "1. **Places:** 34,000+ cities in 244 countries plus landmarks and airports (offline); live OpenStreetMap results when online.\n"
            "2. **Distance:** OSRM road route when online; otherwise straight-line (Haversine).\n"
            "3. **Conditions:** traffic from time and day; live weather when online.\n"
            "4. **Prediction:** the model uses distance + time + passengers, so it works in any city. "
            "Prices follow the training data (USD); other currencies are conversions.\n"
            "5. **Pricing:** ride type, surge, night/airport/booking fees and GST are illustrative business rules."
        )
st.markdown("<div class='muted' style='text-align:center;margin:18px 0'>RideFare · Python · scikit-learn · Streamlit · OpenStreetMap</div>",
            unsafe_allow_html=True)
