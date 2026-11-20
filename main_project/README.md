# Uber Fare Prediction

Predict the fare of an Uber trip **before it ends**, using 50,000 historical trips, and explain the drivers of price.

## Team
- Janak
- Darshan
- Neetu
- Tanya
- Mohit

## Approach (5 lines)
1. Filtered to the **42,540 Completed trips**; checked types, missing values, duplicates and suspicious values.
2. Validated coordinates: they do not match their city or `distance_km`, so we trust **`distance_km`** and drop coordinates.
3. Explored the data with 9 charts: fare is almost a straight line in distance; hour, day, city and payment barely matter.
4. Built **booking-time features only** (distance, hour, day of week, weekend, one-hot city and payment) under the golden rule.
5. Compared Linear Regression, Random Forest, Gradient Boosting (tuned with RandomizedSearchCV) and XGBoost on the **same 80/20 split, `random_state=42`**, chose the model by 5-fold CV on the training set.

## Results (test set, 8,508 trips)
| Model | MAE | RMSE | R² | Train R² | CV RMSE |
|---|---|---|---|---|---|
| Predict the average | 4.96 | 6.21 | 0.00 | – | 6.30 |
| **Linear Regression, distance only (final)** | **2.49** | **3.10** | **0.750** | **0.760** | **3.08** |
| Linear Regression, all features | 2.49 | 3.10 | 0.750 | 0.760 | 3.08 |
| Gradient Boosting (tuned, RandomizedSearchCV) | 2.49 | 3.11 | 0.750 | 0.762 | 3.09 |
| Gradient Boosting | 2.49 | 3.11 | 0.749 | 0.765 | 3.09 |
| XGBoost | 2.50 | 3.13 | 0.745 | 0.772 | 3.12 |
| Random Forest | 2.51 | 3.14 | 0.744 | 0.790 | 3.12 |

Baseline to beat (organisers): MAE 2.49 · RMSE 3.10 · R² 0.75. Full table: `reports/results.csv`.

**Final model: Linear Regression on `distance_km`**: lowest CV RMSE, best or tied-best on every test metric, no overfitting (train R² 0.76 vs test 0.75), and explained by two numbers: fare ≈ $2.97 + $1.86 per km. Feature importance: distance 100%, all other inputs ≈ 0%.

**Key finding:** the fare is essentially *base ≈ $3 + about $1.86 per km* plus random noise. No booking-time feature adds signal beyond distance, so more complex models land at the baseline level. On average the estimate is about **$2.50** off the real fare.

## Project structure
```
notebooks/Uber_Fare_Prediction.ipynb   full analysis (runs top to bottom)
data/uber_trips_dataset_50k.csv        dataset
data/places.csv, data/currencies.csv   place list and currencies for the app
models/uber_fare_model.pkl             saved final model (created by the notebook)
reports/figures/                       charts (created by the notebook)
reports/results.csv / results.json     results table (created by the notebook)
app/app.py                             bonus Streamlit app
presentation/RideFare_Final_Presentation.pptx  final presentation
requirements.txt
```

## How to run
```bash
python -m venv venv
source venv/Scripts/activate          # Windows Git Bash  (PowerShell: venv\Scripts\activate)
pip install -r requirements.txt

jupyter notebook notebooks/Uber_Fare_Prediction.ipynb   # Kernel -> Restart & Run All
python -m streamlit run app/app.py                       # bonus app (run from this folder)
```

## Bonus app
![App screenshot](reports/app_screenshot.png)

The app predicts the trip fare with the saved model, then adds illustrative ride-type, surge, fee and tax rules on top (these rules are assumptions, not learned from data).

**City vs intercity (map-based, no distance cutoff):** the app finds the nearest major city (within 60 km) to the pickup and the drop-off using `data/places.csv`.
- Same city at both ends → **city trip**, priced by the ML model whatever the length (e.g. IIT Jodhpur → AIIMS Jodhpur, 35 km ≈ ₹718).
- Different cities (centres > 50 km apart) → **intercity**, priced by a clearly labelled one-way outstation rule: km × ₹/km + driver allowance + tolls + 5% GST (e.g. Jaipur → Mumbai ≈ ₹17,740).
- Indian city fares are scaled by an "India local price level" factor (sidebar, default 0.12), because the model learned US fares.
