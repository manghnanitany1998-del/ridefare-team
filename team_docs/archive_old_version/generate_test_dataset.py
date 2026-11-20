"""
PRACTICE: a "surprise" dataset that does NOT match today's layout.
Run:  python generate_test_dataset.py   -> creates data/test_india_rides.csv

Differences from uber.csv (on purpose):
- No latitude/longitude at all            -> CASE B (use the given distance)
- Different column names                  -> CASE A (add ALIASES)
- Fare in rupees (price_inr)              -> CASE F (new fare range + bands)
- Date as text "dd-mm-yyyy HH:MM"          -> needs dayfirst=True
- Extra columns: duration, cab_type, weather, city -> CASE E (extra features)
- Some dirty rows (negative price, 0 km, missing values)
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 30_000

cities = rng.choice(["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai"], N)
start = pd.Timestamp("2024-01-01")
times = start + pd.to_timedelta(rng.integers(0, 2 * 365 * 24 * 3600, N), unit="s")
hour = times.hour

km = np.round(rng.gamma(2.2, 3.0, N), 2)
peak = np.isin(hour, [8, 9, 10, 17, 18, 19, 20])
weather = rng.choice(["Clear", "Clear", "Clear", "Rain", "Storm"], N)
duration = np.round(km * rng.uniform(2.5, 4.5, N) * np.where(peak, 1.4, 1.0)
                    * np.select([weather == "Rain", weather == "Storm"], [1.2, 1.4], 1.0), 1)
cab = rng.choice(["Mini", "Sedan", "SUV", "Auto", "Bike"], N, p=[.35, .25, .15, .15, .10])
cab_mult = pd.Series(cab).map({"Mini": 1.0, "Sedan": 1.3, "SUV": 1.6, "Auto": 0.6, "Bike": 0.45}).values
city_mult = pd.Series(cities).map({"Delhi": 1.0, "Mumbai": 1.15, "Bengaluru": 1.1,
                                   "Hyderabad": 0.95, "Chennai": 0.9}).values

price = (40 + 14 * km + 1.5 * duration) * cab_mult * city_mult * np.where(peak, 1.25, 1.0)
price = np.round(price + rng.normal(0, 15, N), 0)

df = pd.DataFrame({
    "ride_id": [f"R{i:06d}" for i in range(N)],
    "booking_time": times.strftime("%d-%m-%Y %H:%M"),
    "city": cities,
    "trip_distance_km": km,
    "trip_duration_min": duration,
    "cab_type": cab,
    "weather": weather,
    "riders": rng.choice([1, 1, 1, 2, 2, 3, 4], N),
    "price_inr": price,
})

bad = rng.choice(N, 400, replace=False)
df.loc[bad[:100], "price_inr"] = -50
df.loc[bad[100:200], "trip_distance_km"] = 0
df.loc[bad[200:300], "weather"] = np.nan
df.loc[bad[300:], "riders"] = np.nan

os.makedirs("data", exist_ok=True)
df.to_csv("data/test_india_rides.csv", index=False)
print("Created data/test_india_rides.csv with", len(df), "rows")
print(df.head())
