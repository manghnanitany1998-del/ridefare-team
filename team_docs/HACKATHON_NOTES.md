# Uber Fare Prediction Hackathon Project

## Recommended 5-hour plan
- 0:00-0:30: inspect columns, target, missing values, impossible coordinates, and fare distribution.
- 0:30-1:30: clean data and engineer distance/time features.
- 1:30-2:30: train Linear Regression, Random Forest, and Histogram Gradient Boosting; select lowest test RMSE.
- 2:30-3:30: build and test Streamlit.
- 3:30-4:15: improve UI, error handling, and demo examples.
- 4:15-5:00: prepare slides, rehearse a 3-5 minute demo, and keep screenshots/metrics ready.

## Why this is regression
The target, fare_amount, is continuous (for example $12.45), so this is a supervised regression problem, not classification.

## Why these models
- Linear Regression: quick, transparent baseline. It may underfit nonlinear pricing.
- Random Forest: learns nonlinear interactions and is robust, but can be slower/larger.
- Histogram Gradient Boosting: fast on tabular numeric data and often strong for nonlinear fare patterns.
The script trains all three and selects the lowest RMSE rather than claiming one model is always best.

## Expected columns
The code recognizes common aliases, but the ideal columns are:
fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

If your names differ, add them to ALIASES in model_utils.py.

## VS Code execution
1. Open the `uber_fare_hackathon` folder in VS Code.
2. Put the supplied dataset at `data/uber.csv`.
3. Open Terminal > New Terminal.
4. Create an environment:
   - Windows: `py -m venv .venv`
   - Activate: `.venv\\Scripts\\activate`
5. Install packages: `pip install -r requirements.txt`
6. Train: `python train_model.py`
7. Launch: `streamlit run app.py`
8. Open the local URL shown by Streamlit, usually http://localhost:8501.

## What each file does
- model_utils.py: standardizes column names and creates reusable distance/time features.
- train_model.py: reads, cleans, splits, trains, evaluates, selects, and saves the model.
- app.py: collects user inputs and makes a prediction.
- model_bundle.joblib: generated after training; model + metadata used by the app.
- metrics.json: generated comparison of all models.

## Core feature engineering
- Haversine distance approximates geographic straight-line distance from latitude/longitude.
- Hour captures commute and night patterns.
- Day of week/weekend captures demand differences.
- Month/year allow broad season/time trends.
- Rush-hour flag provides a simple interpretable traffic proxy.

## Metrics
- MAE: average absolute dollar error; easiest to explain.
- RMSE: penalizes large errors more; used to select the winner.
- R²: fraction of target variation explained; higher is better, but always pair with MAE/RMSE.

## Leakage and validation
Split before evaluation. Do not use columns only known after the ride (actual duration, final tolls, final total) unless the hackathon explicitly says they are available at prediction time. A random split is acceptable for a short hackathon; a time-based split is stronger if predicting future rides.

## Demo story
1. Explain the business problem.
2. Show the dataset and cleaning rules.
3. Explain distance/time features.
4. Show model comparison and chosen RMSE.
5. Enter a short and long trip in Streamlit.
6. Close with limitations: no live traffic/surge/weather and straight-line distance is not road distance.
