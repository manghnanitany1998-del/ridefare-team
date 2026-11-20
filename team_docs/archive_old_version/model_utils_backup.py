import numpy as np
import pandas as pd

# Raw latitude/longitude are NOT used as model inputs, so the model is
# location-independent and works for trips anywhere in the world.
FEATURES = [
    "passenger_count", "distance_km", "hour", "day_of_week", "month", "year",
    "is_weekend", "is_rush_hour"
]
ALIASES = {
    "pickup_longitude": ["pickup_longitude", "pickup_lon", "pickup_lng"],
    "pickup_latitude": ["pickup_latitude", "pickup_lat"],
    "dropoff_longitude": ["dropoff_longitude", "dropoff_lon", "dropoff_lng"],
    "dropoff_latitude": ["dropoff_latitude", "dropoff_lat"],
    "passenger_count": ["passenger_count", "passengers"],
    "pickup_datetime": ["pickup_datetime", "timestamp", "datetime", "pickup_time"],
    "fare_amount": ["fare_amount", "fare", "price", "fare_price"]
}

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

def make_features(df):
    out = standardize_columns(df)
    required = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count", "pickup_datetime"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Update ALIASES in model_utils.py if your dataset uses different names.")
    dt = pd.to_datetime(out["pickup_datetime"], errors="coerce", utc=True)
    out["distance_km"] = haversine_km(out["pickup_longitude"], out["pickup_latitude"], out["dropoff_longitude"], out["dropoff_latitude"])
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["year"] = dt.dt.year
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_rush_hour"] = out["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    return out[FEATURES].replace([np.inf, -np.inf], np.nan)
