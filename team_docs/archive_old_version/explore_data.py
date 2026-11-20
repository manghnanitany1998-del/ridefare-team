"""
STEP 1: EXPLORE THE DATA (EDA)
Run:  python explore_data.py
Charts are saved into the 'charts' folder - they go on slides 5-6 of the PPT.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = "data/uber_trips_dataset_50k.csv"
os.makedirs("charts", exist_ok=True)
df = pd.read_csv(PATH)

print("=== SHAPE (rows, columns) ===", df.shape)
print("\n=== COLUMNS AND TYPES ===\n", df.dtypes)
print("\n=== FIRST 5 ROWS ===\n", df.head())
print("\n=== MISSING VALUES ===\n", df.isnull().sum())
print("\n=== SUMMARY ===\n", df.describe().round(2))
if "status" in df.columns:
    print("\n=== TRIP STATUS ===\n", df["status"].value_counts())

# Use completed trips only, like training
d = df[df["status"].astype(str).str.lower() == "completed"] if "status" in df.columns else df.copy()
d = d[(d["fare_amount"] > 0) & (d["distance_km"] > 0.1)].copy()
d["hour"] = pd.to_datetime(d["pickup_time"], dayfirst=True, errors="coerce").dt.hour

print("\nNegative/zero fares:", (df["fare_amount"] <= 0).sum())
print("Zero-km trips      :", (df["distance_km"] <= 0).sum())
print("Correlation of distance with fare:", round(d["distance_km"].corr(d["fare_amount"]), 3))
for c in ["city", "payment_method"]:
    if c in d.columns:
        print(f"\nAverage fare by {c}:\n", d.groupby(c)["fare_amount"].mean().round(2))

plt.figure(figsize=(7, 4))
plt.hist(d["fare_amount"], bins=60, color="#22d3ee")
plt.title("Fare distribution"); plt.xlabel("Fare ($)"); plt.ylabel("Trips")
plt.tight_layout(); plt.savefig("charts/1_fare_distribution.png", dpi=150); plt.close()

s = d.sample(min(5000, len(d)), random_state=1)
plt.figure(figsize=(7, 4))
plt.scatter(s["distance_km"], s["fare_amount"], s=3, alpha=0.4, color="#0e7490")
plt.title("Distance vs fare"); plt.xlabel("Distance (km)"); plt.ylabel("Fare ($)")
plt.tight_layout(); plt.savefig("charts/2_distance_vs_fare.png", dpi=150); plt.close()

plt.figure(figsize=(7, 4))
d.groupby("hour")["fare_amount"].mean().plot(kind="bar", color="#22d3ee")
plt.title("Average fare by hour of day"); plt.xlabel("Hour"); plt.ylabel("Avg fare ($)")
plt.tight_layout(); plt.savefig("charts/3_fare_by_hour.png", dpi=150); plt.close()

print("\nCharts saved in the 'charts' folder.")
