"""
STEP 0 (PRACTICE ONLY): create a fake 'uber.csv' so you can run the whole
project TODAY, before the real dataset arrives tomorrow.

Run:  python generate_sample_data.py
Tomorrow: delete this uber.csv and put the real dataset in its place.

The fake data copies the layout of the popular Kaggle "Uber Fares" dataset:
key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude,
dropoff_longitude, dropoff_latitude, passenger_count
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)          # fixed seed = same data every run
N = 50_000                               # number of fake rides

# --- Random pickup points around New York City ---
plat = rng.normal(40.75, 0.04, N)
plon = rng.normal(-73.98, 0.04, N)

# --- Drop-off = pickup + a random trip (most trips short, a few long) ---
trip_km = rng.gamma(shape=2.0, scale=2.0, size=N)        # skewed: like real taxi trips
angle = rng.uniform(0, 2 * np.pi, N)
dlat = plat + (trip_km / 111) * np.sin(angle)            # 1 degree latitude ~ 111 km
dlon = plon + (trip_km / 85) * np.cos(angle)             # 1 degree longitude ~ 85 km in NYC

# --- Random dates 2009-2015 ---
start = pd.Timestamp("2009-01-01")
seconds = rng.integers(0, 6 * 365 * 24 * 3600, N)
times = start + pd.to_timedelta(seconds, unit="s")
hour = times.hour

passengers = rng.choice([1, 1, 1, 1, 2, 2, 3, 4, 5, 6], N)

# --- Fare formula (what the model must "discover") ---
peak = np.isin(hour, [7, 8, 9, 16, 17, 18, 19])
year_inflation = 1 + (times.year - 2009) * 0.04
fare = (2.5 + 1.6 * trip_km + peak * 1.5 + (hour < 6) * 0.5) * year_inflation
fare = fare + rng.normal(0, 1.5, N)                     # real-world noise

df = pd.DataFrame({
    "key": times.astype(str),
    "fare_amount": fare.round(2),
    "pickup_datetime": times.strftime("%Y-%m-%d %H:%M:%S UTC"),
    "pickup_longitude": plon, "pickup_latitude": plat,
    "dropoff_longitude": dlon, "dropoff_latitude": dlat,
    "passenger_count": passengers,
})

# --- Add dirty rows on purpose, like real data has ---
bad = rng.choice(N, 600, replace=False)
df.loc[bad[:150], "fare_amount"] = -5.0                  # negative fares
df.loc[bad[150:300], ["pickup_latitude", "pickup_longitude"]] = 0   # missing GPS
df.loc[bad[300:450], "passenger_count"] = 0              # zero passengers
df.loc[bad[450:], "dropoff_latitude"] = np.nan           # missing values

import os; os.makedirs("data", exist_ok=True)
df.to_csv("data/uber.csv", index=False)
print("Created uber.csv with", len(df), "rows")
print(df.head())
