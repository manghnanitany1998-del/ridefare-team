from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_utils import make_features, standardize_columns, FEATURES
try:
    from xgboost import XGBRegressor      # optional: pip install xgboost
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed - training the other 3 models only (pip install xgboost)")

DATA_PATH = Path("data/uber_trips_dataset_50k.csv")     # main dataset
MODEL_PATH = Path("model_bundle.joblib")
METRICS_PATH = Path("metrics.json")
RANDOM_STATE = 42

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Put the CSV there, then run: python train_model.py")

df = pd.read_csv(DATA_PATH)
df = standardize_columns(df)
if "fare_amount" not in df.columns:
    raise ValueError("Target not found. Rename the fare/price column to fare_amount or add its name in ALIASES.")

print("Rows loaded:", len(df))
# Keep only completed trips: cancelled / no-show rows are not real rides.
if "status" in df.columns:
    print("Trip status:", df["status"].value_counts().to_dict())
    df = df[df["status"].astype(str).str.lower() == "completed"]

# Remove impossible target values.
df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")
df = df[df["fare_amount"].between(0.5, 500)]
if "passenger_count" in df.columns:
    df = df[pd.to_numeric(df["passenger_count"], errors="coerce").fillna(1).between(1, 8)]

X = make_features(df)
y = df.loc[X.index, "fare_amount"]
valid = y.notna() & X["distance_km"].between(0.1, 150)
X, y = X.loc[valid], y.loc[valid]
print("Rows after cleaning:", len(X))
if len(X) < 100:
    raise ValueError(f"Only {len(X)} usable rows remain. Inspect columns and cleaning rules.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
models = {
    "Linear Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LinearRegression())]),
    "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=180, max_depth=18, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE))]),
    "Histogram Gradient Boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=0.2, random_state=RANDOM_STATE))])
}
results, fitted = {}, {}
if HAS_XGB:
    # XGBoost: boosted trees with regularisation; often strongest on tabular data
    models["XGBoost"] = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("model", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                                         subsample=0.8, colsample_bytree=0.8,
                                                         reg_lambda=1.0, tree_method="hist",
                                                         n_jobs=-1, random_state=RANDOM_STATE))])
import time as _time
from sklearn.metrics import classification_report, median_absolute_error
# Fare bands turn the regression output into categories, so we can also show a
# classification-style report (precision / recall / F1) for each model.
BANDS = [0, 10, 20, 40, np.inf]
BAND_NAMES = ["Low (<$10)", "Medium ($10-20)", "High ($20-40)", "Very high (>$40)"]
def to_band(v):
    return pd.cut(v, BANDS, labels=BAND_NAMES, right=False).astype(str)
for name, model in models.items():
    # Log target reduces the effect of rare extreme fares and guarantees positive inverse predictions.
    wrapped = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    t0 = _time.time()
    wrapped.fit(X_train, y_train)
    train_sec = _time.time() - t0
    pred = np.maximum(wrapped.predict(X_test), 0)
    err = np.abs(y_test.values - pred)
    results[name] = {"MAE": float(mean_absolute_error(y_test, pred)), "RMSE": float(mean_squared_error(y_test, pred) ** 0.5), "R2": float(r2_score(y_test, pred)),
                     "Train_R2": float(r2_score(y_train, wrapped.predict(X_train))),   # overfitting check
                     "MedianAE": float(median_absolute_error(y_test, pred)),
                     "MAPE_%": float(np.mean(err / np.maximum(y_test.values, 1)) * 100),
                     "Within_$2_%": float(np.mean(err <= 2) * 100),
                     "Train_sec": round(train_sec, 1),
                     "band_report": classification_report(to_band(y_test), to_band(pred), zero_division=0, output_dict=True)}
    fitted[name] = wrapped

best_name = min(results, key=lambda n: results[n]["RMSE"])

# ---------- FEATURE IMPORTANCE (permutation) ----------
# Shuffle one feature at a time on the test set and measure how much RMSE gets worse.
# Bigger increase = the model relies more on that feature. Works for ANY model.
from sklearn.inspection import permutation_importance
sample = X_test.sample(min(3000, len(X_test)), random_state=RANDOM_STATE)
pi = permutation_importance(fitted[best_name], sample, y_test.loc[sample.index], n_repeats=5,
                            random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=-1)
importance = pd.Series(pi.importances_mean, index=sample.columns).clip(lower=0).sort_values(ascending=False)
importance_pct = (importance / importance.sum() * 100).round(1)
print("\nFeature importance (% of total, best model):")
print(importance_pct.to_string())

import matplotlib
matplotlib.use("Agg")                       # save to file without opening a window
import matplotlib.pyplot as plt
Path("charts").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
importance_pct.iloc[::-1].plot(kind="barh", ax=ax, color="#06C167")
ax.set_title(f"Feature importance – {best_name}"); ax.set_xlabel("% of importance")
fig.tight_layout(); fig.savefig("charts/4_feature_importance.png", dpi=150); plt.close(fig)
print("Saved charts/4_feature_importance.png")

bundle = {"model": fitted[best_name], "model_name": best_name, "features": FEATURES, "metrics": results[best_name], "all_metrics": results, "rows": len(X),
          "importance": importance_pct.to_dict()}
joblib.dump(bundle, MODEL_PATH)
METRICS_PATH.write_text(json.dumps({"best_model": best_name, "rows": len(X), "models": results}, indent=2))
summary = {n: {k: round(v, 3) for k, v in r.items() if k != "band_report"} for n, r in results.items()}
print(json.dumps({"best_model": best_name, "rows": len(X), "metrics": summary}, indent=2))
print(f"Saved {MODEL_PATH} and {METRICS_PATH}")
