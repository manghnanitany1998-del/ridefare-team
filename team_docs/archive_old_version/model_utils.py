import numpy as np
import pandas as pd

# Main dataset: data/uber_trips_dataset_50k.csv
# Columns: trip_id, driver_id, rider_id, city, pickup_lat, pickup_lng, drop_lat, drop_lng,
#          distance_km, fare_amount, status, payment_method, pickup_time (dd-mm-yyyy HH:MM), drop_time
#
# Notes from inspection (20 Nov 2026):
# - distance_km is GIVEN and is the trip distance (0.87 correlation with fare).
#   The lat/lng columns do NOT match distance_km (correlation 0.00), so we use the given distance.
# - No passenger column -> filled with 1.
# - city, payment_method, hour and weekday barely change the fare in this data, but are kept
#   so the model can use them if they help.
# - IDs (trip/driver/rider) are not features: they are just labels.
FEATURES = [
    "distance_km", "passenger_count", "hour", "day_of_week", "month", "year",
    "is_weekend", "is_rush_hour", "city_code", "payment_code",
]
ALIASES = {
    "pickup_longitude": ["pickup_longitude", "pickup_lon", "pickup_lng"],
    "pickup_latitude": ["pickup_latitude", "pickup_lat"],
    "dropoff_longitude": ["dropoff_longitude", "dropoff_lon", "dropoff_lng", "drop_lng", "drop_lon"],
    "dropoff_latitude": ["dropoff_latitude", "dropoff_lat", "drop_lat"],
    "passenger_count": ["passenger_count", "passengers", "riders"],
    "pickup_datetime": ["pickup_datetime", "timestamp", "datetime", "pickup_time", "booking_time"],
    "fare_amount": ["fare_amount", "fare", "price", "fare_price", "price_inr"],
    "distance_km": ["distance_km", "trip_distance_km", "distance"],
}
CITY = {"boston": 0, "chicago": 1, "los angeles": 2, "new york": 3, "san francisco": 4, "seattle": 5}
PAYMENT = {"card": 0, "cash": 1, "upi": 2, "wallet": 3}


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
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _code(out, col, mapping):
    s = out[col] if col in out.columns else pd.Series("", index=out.index)
    return s.astype(str).str.strip().str.lower().map(mapping).fillna(-1)     # unknown -> -1


def make_features(df):
    out = standardize_columns(df)
    # Dataset dates are dd-mm-yyyy; the app sends yyyy-mm-dd. Try day-first, then the ISO format.
    raw = out["pickup_datetime"].astype(str)
    dt = pd.to_datetime(raw, errors="coerce", dayfirst=True, format="%d-%m-%Y %H:%M")
    dt = dt.fillna(pd.to_datetime(raw, errors="coerce"))
    # Distance: use the given column (the app fills it with the route distance), else Haversine
    if "distance_km" in out.columns and out["distance_km"].notna().any():
        out["distance_km"] = pd.to_numeric(out["distance_km"], errors="coerce")
    else:
        out["distance_km"] = haversine_km(out["pickup_longitude"], out["pickup_latitude"],
                                          out["dropoff_longitude"], out["dropoff_latitude"])
    pc = out["passenger_count"] if "passenger_count" in out.columns else pd.Series(1, index=out.index)
    out["passenger_count"] = pd.to_numeric(pc, errors="coerce").fillna(1)
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["year"] = dt.dt.year
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_rush_hour"] = out["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    out["city_code"] = _code(out, "city", CITY)
    out["payment_code"] = _code(out, "payment_method", PAYMENT)
    return out[FEATURES].replace([np.inf, -np.inf], np.nan)
