# 🚕 RideFare – Ride Fare Prediction

A machine-learning web app that estimates ride fares for trips anywhere in the world.
Pick a pickup and drop-off place and a time; the app detects traffic and weather,
gets the road route when online, and returns a fare with a full price breakdown.

Built with **Python, scikit-learn and Streamlit**.

## Features
- **Worldwide place search** – 34,000+ cities in 244 countries plus popular landmarks and airports (works offline)
- **Automatic conditions** – traffic from time and day, live weather (Open-Meteo) when online
- **Real road distance** – driving route and time from OSRM, with straight-line fallback
- **Local currency** – fare in the selected country's currency, with INR alongside
- **Realistic pricing** – ride types, surge, night and airport charges, booking fee and GST
- **Fare breakdown**, fare split per rider, ride comparison and best-time-to-travel chart
- **Model report** – selected model, metrics for all models tried, fare-band classification report and feature importance

## How it works
```
data → cleaning → feature engineering → model comparison → best model → Streamlit app
```
1. **Features:** Haversine distance, hour, weekday, month, year, weekend and rush-hour flags, passenger count. Raw coordinates are not used, so the model generalises across cities.
2. **Models:** Linear Regression, Random Forest and Histogram Gradient Boosting, trained on a log-transformed target with an 80/20 train/test split.
3. **Selection:** the model with the lowest test RMSE is saved and served by the app.
4. **Pricing:** the model predicts the trip fare; ride type, surge, fees and tax are applied as business rules on top.

## Results (sample data)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1.79 | 4.16 | 0.397 |
| Random Forest | 1.27 | 1.59 | 0.912 |
| **Histogram Gradient Boosting** | **1.23** | **1.55** | **0.916** |

Distance accounts for about 90% of the model's feature importance.

## Project structure
```
app.py                  Streamlit web app
train_model.py          Cleaning, training, evaluation and model selection
model_utils.py          Column mapping and feature engineering
explore_data.py         Exploratory data analysis and charts
inspect_dataset.py      Column checker for new datasets
generate_sample_data.py Synthetic sample dataset
data/places.csv         Worldwide place list (GeoNames)
data/currencies.csv     Country → currency mapping
model_bundle.joblib     Trained model
metrics.json            Model evaluation results
```

## Run locally
```bash
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash; use venv\Scripts\activate in PowerShell
pip install -r requirements.txt

python generate_sample_data.py      # optional: creates data/uber.csv
python train_model.py
python -m streamlit run app.py
```

## Dataset
Expected columns: `fare_amount`, `pickup_datetime`, `pickup_longitude`, `pickup_latitude`,
`dropoff_longitude`, `dropoff_latitude`, `passenger_count`. Other names can be mapped in `ALIASES` in `model_utils.py`.

## Notes
- Surge, fee and ride-type multipliers are illustrative business rules, not learned from data.
- Fares follow the pricing of the training data; other currencies are conversions.
- Place data: GeoNames (CC BY 4.0). Maps and routing: OpenStreetMap, Photon, OSRM. Weather: Open-Meteo.
