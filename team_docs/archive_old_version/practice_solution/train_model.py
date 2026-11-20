"""
SOLUTION train_model.py for the practice dataset.
Run from the practice_solution folder:
    cd practice_solution
    python train_model.py
Saves files INSIDE practice_solution, so your main project is untouched.
"""
from pathlib import Path
import json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             median_absolute_error, classification_report)
from sklearn.inspection import permutation_importance
from model_utils import make_features, standardize_columns, FEATURES

DATA_PATH = Path("../data/test_india_rides.csv")
MODEL_PATH, METRICS_PATH = Path("model_bundle.joblib"), Path("metrics.json")
RS = 42

df = standardize_columns(pd.read_csv(DATA_PATH))
print("Rows loaded:", len(df))

# ---------- CLEAN (CASE F: prices in INR) ----------
df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")
df = df[df["fare_amount"].between(20, 5000)]

X = make_features(df)
y = df.loc[X.index, "fare_amount"]
valid = y.notna() & X["distance_km"].between(0.1, 150)
X, y = X.loc[valid], y.loc[valid]
print("Rows after cleaning:", len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RS)

imp = ("imputer", SimpleImputer(strategy="median"))        # fills the missing riders/duration
models = {
    "Linear Regression": Pipeline([imp, ("scale", StandardScaler()), ("model", LinearRegression())]),
    "Random Forest": Pipeline([imp, ("model", RandomForestRegressor(n_estimators=180, max_depth=18,
                               min_samples_leaf=2, n_jobs=-1, random_state=RS))]),
    "Histogram Gradient Boosting": Pipeline([imp, ("model", HistGradientBoostingRegressor(
                               max_iter=250, learning_rate=0.08, random_state=RS))]),
}

BANDS = [0, 150, 300, 600, np.inf]
NAMES = ["Low (<₹150)", "Medium (₹150-300)", "High (₹300-600)", "Very high (>₹600)"]
band = lambda v: pd.cut(v, BANDS, labels=NAMES, right=False).astype(str)

results, fitted = {}, {}
for name, m in models.items():
    w = TransformedTargetRegressor(regressor=m, func=np.log1p, inverse_func=np.expm1)
    t0 = time.time(); w.fit(X_train, y_train); sec = time.time() - t0
    p = np.maximum(w.predict(X_test), 0)
    err = np.abs(y_test.values - p)
    results[name] = {"MAE": float(mean_absolute_error(y_test, p)),
                     "RMSE": float(mean_squared_error(y_test, p) ** 0.5),
                     "R2": float(r2_score(y_test, p)),
                     "Train_R2": float(r2_score(y_train, w.predict(X_train))),   # overfitting check
                     "MedianAE": float(median_absolute_error(y_test, p)),
                     "MAPE_%": float(np.mean(err / np.maximum(y_test.values, 1)) * 100),
                     "Within_Rs20_%": float(np.mean(err <= 20) * 100),
                     "Train_sec": round(sec, 1),
                     "band_report": classification_report(band(y_test), band(p), zero_division=0, output_dict=True)}
    fitted[name] = w

best = min(results, key=lambda n: results[n]["RMSE"])

s = X_test.sample(min(3000, len(X_test)), random_state=RS)
pi = permutation_importance(fitted[best], s, y_test.loc[s.index], n_repeats=5, random_state=RS,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
imp_s = pd.Series(pi.importances_mean, index=s.columns).clip(lower=0)
imp_pct = (imp_s / imp_s.sum() * 100).round(1).sort_values(ascending=False)
print("\nFeature importance (% of total, best model):\n" + imp_pct.to_string())

joblib.dump({"model": fitted[best], "model_name": best, "features": FEATURES, "metrics": results[best],
             "all_metrics": results, "rows": len(X), "importance": imp_pct.to_dict(), "currency": "INR"},
            MODEL_PATH)
summary = {n: {k: round(v, 3) for k, v in r.items() if k != "band_report"} for n, r in results.items()}
METRICS_PATH.write_text(json.dumps({"best_model": best, "rows": len(X), "models": summary}, indent=2))
print(json.dumps({"best_model": best, "rows": len(X), "metrics": summary}, indent=2))
print("Saved", MODEL_PATH, "and", METRICS_PATH, "(inside practice_solution)")
