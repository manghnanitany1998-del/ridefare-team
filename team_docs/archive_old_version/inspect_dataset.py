"""
RUN THIS FIRST on any new dataset:   python inspect_dataset.py
It tells you what the dataset has, guesses which column is which,
and tells you exactly what to change in model_utils.py.
"""
import sys
import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/uber_trips_dataset_50k.csv"
df = pd.read_csv(PATH, nrows=200000)
print(f"\n=== {PATH}: {df.shape[0]} rows x {df.shape[1]} columns ===\n")

info = pd.DataFrame({"type": df.dtypes.astype(str), "missing_%": (df.isna().mean() * 100).round(1),
                     "unique": df.nunique(), "example": df.iloc[0]})
print(info.to_string(), "\n")

# Keyword guesses for each role the project needs
ROLES = {
    "fare_amount (TARGET)": ["fare", "price", "amount", "cost", "total", "charge"],
    "pickup_datetime": ["date", "time", "timestamp", "datetime"],
    "pickup_latitude": ["pickup_lat", "start_lat", "src_lat", "origin_lat", "pickup_latitude"],
    "pickup_longitude": ["pickup_lon", "pickup_lng", "start_lon", "start_lng", "src_lon", "origin_lon"],
    "dropoff_latitude": ["drop", "end_lat", "dest_lat", "dropoff_lat"],
    "dropoff_longitude": ["dropoff_lon", "dropoff_lng", "end_lon", "end_lng", "dest_lon"],
    "passenger_count": ["passenger", "pax", "riders", "seats"],
    "distance (ready-made)": ["distance", "dist", "km", "miles", "trip_length"],
    "duration (ready-made)": ["duration", "trip_time", "minutes", "mins", "seconds"],
}
print("=== Best guesses (check these!) ===")
cols_l = {c: c.lower() for c in df.columns}
found = {}
for role, keys in ROLES.items():
    hits = [c for c, l in cols_l.items() if any(k in l for k in keys)]
    if role.startswith("dropoff_lat"):
        hits = [c for c in hits if "lat" in cols_l[c]]
    found[role] = hits
    print(f"{role:28s} -> {hits if hits else 'NOT FOUND'}")

extra = [c for c in df.columns if not any(c in v for v in found.values())]
num_extra = [c for c in extra if pd.api.types.is_numeric_dtype(df[c])]
cat_extra = [c for c in extra if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 50]
print(f"\nOther numeric columns (possible extra features): {num_extra}")
print(f"Other text columns with few values (category features): {cat_extra}")

print("\n=== What to do ===")
if not found["fare_amount (TARGET)"]:
    print("!! No fare/price column found - ask organisers which column is the target.")
has_coords = all(found[r] for r in ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"])
if has_coords:
    print("- Coordinates found -> project works as is. Add any new names to ALIASES in model_utils.py.")
elif found["distance (ready-made)"]:
    print("- No coordinates but a distance column -> use CASE B in ADAPT_NEW_DATASET.md.")
else:
    print("- No coordinates and no distance -> use CASE C in ADAPT_NEW_DATASET.md (place names / zones).")
if not found["pickup_datetime"]:
    print("- No datetime -> time features will be skipped (CASE D).")
if not found["passenger_count"]:
    print("- No passenger column -> it will be filled with 1 (CASE D).")
if num_extra or cat_extra:
    print("- Extra columns found -> add them as features (CASE E) for better accuracy.")
