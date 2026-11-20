"""
SOLUTION model_utils.py for the practice dataset data/test_india_rides.csv
(no coordinates, given distance + duration, prices in INR, extra text columns).
"""
import numpy as np
import pandas as pd

FEATURES = ["passenger_count", "distance_km", "duration_min", "hour", "day_of_week",
            "month", "year", "is_weekend", "is_rush_hour", "cab_code", "weather_code", "city_code"]

ALIASES = {
    "fare_amount": ["fare_amount", "fare", "price", "fare_price", "price_inr"],          # CASE A
    "pickup_datetime": ["pickup_datetime", "timestamp", "datetime", "pickup_time", "booking_time"],
    "passenger_count": ["passenger_count", "passengers", "riders"],
    "distance_km": ["distance_km", "trip_distance_km", "distance"],
    "duration_min": ["duration_min", "trip_duration_min", "duration"],
    "pickup_longitude": ["pickup_longitude", "pickup_lon", "pickup_lng"],
    "pickup_latitude": ["pickup_latitude", "pickup_lat"],
    "dropoff_longitude": ["dropoff_longitude", "dropoff_lon", "dropoff_lng"],
    "dropoff_latitude": ["dropoff_latitude", "dropoff_lat"],
}
CAB = {"bike": 0, "auto": 1, "mini": 2, "sedan": 3, "suv": 4}
WEATHER = {"clear": 0, "rain": 1, "storm": 2}
CITY = {"delhi": 0, "mumbai": 1, "bengaluru": 2, "hyderabad": 3, "chennai": 4}


def standardize_columns(df):
    out = df.copy()
    lower = {c.lower().strip(): c for c in out.columns}
    rename = {}
    for standard, aliases in ALIASES.items():
        for alias in aliases:
            if alias in lower:
                rename[lower[alias]] = standard
                break
    return out.rename(columns=rename)


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _text_code(out, col, mapping):
    s = out[col] if col in out.columns else pd.Series("", index=out.index)
    return s.astype(str).str.strip().str.lower().map(mapping).fillna(-1)      # unknown/missing -> -1


def make_features(df):
    out = standardize_columns(df)
    dt = pd.to_datetime(out["pickup_datetime"], errors="coerce", dayfirst=True)   # dd-mm-yyyy
    if "distance_km" in out.columns:                                            # CASE B
        out["distance_km"] = pd.to_numeric(out["distance_km"], errors="coerce")
    else:
        out["distance_km"] = haversine_km(out["pickup_longitude"], out["pickup_latitude"],
                                          out["dropoff_longitude"], out["dropoff_latitude"])
    out["duration_min"] = pd.to_numeric(out["duration_min"], errors="coerce") if "duration_min" in out else np.nan
    out["passenger_count"] = (pd.to_numeric(out["passenger_count"], errors="coerce")
                              if "passenger_count" in out else 1)               # CASE D
    out["passenger_count"] = pd.Series(out["passenger_count"], index=out.index).fillna(1)
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["year"] = dt.dt.year
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_rush_hour"] = out["hour"].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
    out["cab_code"] = _text_code(out, "cab_type", CAB)                          # CASE E
    out["weather_code"] = _text_code(out, "weather", WEATHER)
    out["city_code"] = _text_code(out, "city", CITY)
    return out[FEATURES].replace([np.inf, -np.inf], np.nan)
