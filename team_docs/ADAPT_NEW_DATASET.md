# Adapting the project to a NEW dataset (same layout, same app)

**Golden rule:** only `model_utils.py` should change. It turns ANY dataset into the same
standard feature table. `train_model.py`, `app.py` and the PPT keep the same layout.

```
new CSV -> standardize_columns() -> make_features() -> same FEATURES -> same training -> same app
```

## Step 1 – Inspect (5 min)
```
python inspect_dataset.py data/your_file.csv
```
It prints columns, types, missing %, and guesses which column is fare, date, coordinates, etc.

## Step 2 – Pick your case

### CASE A – Same idea, different column names (most likely)
e.g. `price`, `start_lat`, `trip_start_time`.
Add the names to `ALIASES` in `model_utils.py`:
```python
"fare_amount": ["fare_amount", "fare", "price", "total_amount"],
"pickup_latitude": ["pickup_latitude", "pickup_lat", "start_lat"],
```
Nothing else changes.

### CASE B – No coordinates, but a distance column (km or miles)
In `make_features`, use the given distance instead of Haversine:
```python
if "distance_km" in out.columns:            # alias e.g. "trip_distance", "distance"
    out["distance_km"] = pd.to_numeric(out["distance_km"], errors="coerce")
    # if it's in miles: out["distance_km"] *= 1.609
else:
    out["distance_km"] = haversine_km(...)
```
Add `"distance_km": ["distance", "trip_distance", "distance_km", "km"]` to ALIASES.
The app still works: it computes distance from the chosen places.

### CASE C – Only place names / zones (e.g. "Andheri", "Zone 12")
Option 1: look up each unique place once in `data/places.csv` (or a lookup table) to get lat/lon, then continue as normal.
Option 2: use the place as a category feature (CASE E) and skip distance. Weaker, but works.

### CASE D – Missing columns
- No passengers → `out["passenger_count"] = 1`
- No datetime → set hour/day/month/year to fixed values (e.g. 12, 2, 6, 2024) or drop those features from `FEATURES`.
  Important: whatever you drop from FEATURES, the app must not need it.

### CASE E – Extra useful columns (great for accuracy)
e.g. `duration_min`, `surge_multiplier`, `weather`, `car_type`, `traffic`, `city`.
1. Add numbers directly to `FEATURES` (e.g. `"duration_min"`).
2. Text categories → numbers:
```python
out["car_type"] = out["car_type"].astype(str).str.lower().map({"mini": 0, "sedan": 1, "suv": 2}).fillna(0)
```
   or one-hot: `pd.get_dummies(out["car_type"], prefix="car")`.
3. **Then give the app a value for it.** Either an input widget, or the auto value the app already computes
   (e.g. traffic/weather from `auto_conditions`, ride type from the ride picker).
   If the app can't know it, don't use it as a feature.

### CASE F – Target in another currency (₹) or a different range
- Change the fare filter in `train_model.py` (`between(0.5, 500)`) to a sensible range, e.g. `between(20, 5000)` for ₹.
- In `app.py`, the model's output is no longer USD: set the base currency so conversions are right
  (simplest: if the target is INR, divide predictions by `usd_to_inr` before the business rules, or tell me and I'll switch the app's base currency).
- Fare bands for the classification report (`BANDS` in `train_model.py`) → e.g. `[0, 150, 300, 600, inf]`.

### CASE G – Completely different target (e.g. predict trip duration or surge)
Same pipeline: change the target name in ALIASES (`"fare_amount": ["duration"]`), keep regression models,
relabel text in the app. If the target is a category (e.g. "cancelled yes/no"), it becomes
**classification**: swap to `RandomForestClassifier` / `HistGradientBoostingClassifier` and use accuracy / F1.

## Step 3 – Run in the same order
```
python explore_data.py      # update column names in it if EDA errors
python train_model.py       # prints scores, saves model + metrics + importance chart
python -m streamlit run app.py
```

## Step 4 – Update the PPT
Paste the training output to Quick → slides 5, 6, 8 and the feature-importance slide get updated.
Add one line to the data slide: "Dataset had X / lacked Y, so we did Z" – judges love this.

## Quick checks before demo
- `train_model.py` runs with no error and R² is clearly above Linear Regression's.
- Every name in `FEATURES` is something the app can fill.
- Try 2-3 trips in the app; fares look sensible for the currency.
