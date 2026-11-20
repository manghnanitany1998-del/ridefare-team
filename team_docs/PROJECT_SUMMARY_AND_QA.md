# RideFare – Uber Fare Prediction: End-to-End Summary + Judge Q&A

---

## 1. What we built (one line)
A machine-learning web app that predicts a ride fare for any trip in the world. The user picks places and a time; the app detects traffic and weather itself and shows a realistic, receipt-style price in $ and ₹.

---

## 2. End-to-end pipeline

| # | Step | File | What happens | Why |
|---|---|---|---|---|
| 0 | Practice data | `generate_sample_data.py` | Creates 50,000 fake NYC rides in `data/uber.csv`, with dirty rows on purpose | Lets us build and test before the real data arrives |
| 1 | Explore (EDA) | `explore_data.py` | Prints shape, types, missing values, statistics; saves 3 charts to `charts/` | Find errors and patterns before training |
| 2 | Clean | `train_model.py` | Drops missing values; keeps fare 0.5–500, passengers 1–8, valid GPS, distance 0.05–150 km | Bad rows teach the model wrong patterns |
| 3 | Features | `model_utils.py` | Haversine distance, hour, weekday, month, year, weekend, rush-hour flag, passengers | Raw coordinates mean little; distance and time drive fare |
| 4 | Split | `train_model.py` | 80% train, 20% test (`random_state=42`) | Score the model on rides it has never seen |
| 5 | Train | `train_model.py` | Linear Regression, Random Forest, Histogram Gradient Boosting, all inside Pipelines (median imputer, scaler for linear) with a log-transformed target | Compare a simple baseline with stronger non-linear models |
| 6 | Evaluate | `train_model.py` | MAE, RMSE, R², MedianAE, MAPE, % within $2, training time, fare-band classification report | Several views of accuracy |
| 7 | Select + save | `train_model.py` | Lowest RMSE wins → `model_bundle.joblib`, `metrics.json` | The app loads the saved model; no retraining |
| 8 | Serve | `app.py` | Streamlit website | Interactive demo |

---

## 3. The app (`app.py`)

**The user chooses only:** pickup and drop-off (country + one search-as-you-type dropdown), Now or a scheduled date/time, passengers and ride type.

**Detected automatically:**
- **Traffic:** from the time and day. Weekday 8–10 AM and 5–8 PM = Heavy; 7 AM–9 PM = Moderate; otherwise Light.
- **Weather:** live from Open-Meteo when online and the trip is within the next 2 hours; otherwise assumed clear. The source is shown on screen.
- **Rush hour, night, weekend evening, airport trip.**
- **Road distance and time** from OSRM when online; otherwise straight-line distance and about 25 km/h city speed.

**Pricing:**
`ML fare (model) × ride type → + surge (cap 2.5×) → + night 25% → + airport fee → + booking fee → + GST 5%`
Only the ML fare is learned from data. The other parts are **illustrative business rules**, like real ride apps use.

**Other features:** worldwide place list (34,000+ cities in 244 countries plus landmarks and airports, works offline), fare range (± MAE), fare breakdown, split per rider, driver card with arrival time, ride comparison, best time to travel chart, trip history with CSV download, and a sidebar model report.

---

## 4. Which evaluation model is best, and how to check

### Practice-data results (49,327 rows after cleaning)
| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Linear Regression | 1.79 | 4.16 | 0.40 |
| Random Forest | 1.25 | 1.56 | 0.915 |
| **Histogram Gradient Boosting** 🏆 | **1.23** | **1.55** | **0.916** |

**Winner: Histogram Gradient Boosting**, chosen because it has the lowest RMSE on the hidden test set. Random Forest is a very close second. Linear Regression underfits.

> These scores were run before raw coordinates were removed from the model's inputs. Retrain and use the new numbers.

### How to check (do this tomorrow on the real data)
1. `python train_model.py` prints every model's metrics and `best_model`.
2. Open `metrics.json` for the same results, saved to a file.
3. In the app sidebar: the 🏆 winner, a table of all models, and a classification report for each.
4. **Good-model checklist:**
   - Lowest RMSE and MAE, R² closest to 1.
   - RMSE not much bigger than MAE (few huge misses).
   - Beats the Linear Regression baseline clearly.
   - **Overfitting check:** compare training and test scores. Add this to `train_model.py` after fitting:
     `print(name, "train R2", r2_score(y_train, wrapped.predict(X_train)))`
     If the training R² is much higher than the test R² (gap > 0.1), the model is overfitting.
   - Optional: 5-fold cross-validation, to check the score is stable across different splits.
5. **Realistic expectation on real Uber data:** R² about 0.75–0.85 and MAE about $2–3 is good. Real fares include tolls, waiting and traffic that the data doesn't capture.

### What each metric means
- **MAE:** the average absolute error in $. Easiest to explain.
- **RMSE:** square root of the average squared error. Penalises big misses. **Used to pick the winner.**
- **R²:** the share of fare variation explained. 1 = perfect, 0 = no better than guessing the average.
- **MedianAE:** the typical error. Not affected by a few extreme misses.
- **MAPE %:** average percentage error.
- **Within $2 %:** the share of predictions within $2 of the real fare. Easy for non-technical judges.
- **Fare-band classification report:** fares grouped into Low (<$10), Medium ($10–20), High ($20–40) and Very high (>$40), then precision, recall, F1 and accuracy per band. This is regression, so a band report is an *extra* view, not the main metric.

---

## 5. Run order (VS Code, Git Bash, in the Hackathon project folder)
```
source venv/Scripts/activate          # or the venv outside OneDrive
python -m pip install -r requirements.txt
python generate_sample_data.py        # practice only – skip with real data
python explore_data.py
python train_model.py                 # MUST retrain after any feature change
python -m streamlit run app.py
```
**Tomorrow:** put the real file at `data/uber.csv`. If the column names differ, add them to `ALIASES` in `model_utils.py`. Then run `explore_data.py` and `train_model.py`, and paste the output to update slides 5, 6 and 8.

---

## 6. Judge Q&A

### Problem and data
**Q: Is this regression or classification? Why?**
Regression. The target, fare_amount, is a continuous number in dollars.

**Q: What did you find in EDA?**
Fares are right-skewed (most are small, with a long tail). Distance correlates strongly with fare (0.94 on practice data). Fares rise at rush hour. There were dirty rows: negative fares, zero passengers, GPS at (0,0), and missing values.

**Q: How did you handle outliers and missing values?**
Removed impossible rows (negative fares, invalid GPS, 0 passengers). A median imputer handles any remaining gaps inside the model pipeline.

**Q: Why a log-transform of the target?**
Fares are skewed. Using log(1+fare) shrinks the effect of rare expensive trips and guarantees positive predictions after converting back.

### Features
**Q: What is the Haversine formula?**
It gives the straight-line distance between two latitude/longitude points on a sphere (Earth radius 6371 km).

**Q: Which feature matters most?**
Distance, by far. Then time features (hour, rush hour, year). You can show this with the model's feature importance.

**Q: Why did you remove raw latitude/longitude from the features?**
The training data is all one city. A model that learns coordinates only knows that city. Using distance and time makes it work anywhere.

**Q: Why not use road distance in training?**
The dataset has no routes, and calling a routing service for every training row is slow. It's a next step: training on road distance would make training and prediction match.

### Models
**Q: Why these three models?**
Linear Regression is the simple baseline. Random Forest averages many trees. Histogram Gradient Boosting adds trees one after another, each fixing the previous ones' errors. Together they cover simple to strong.

**Q: Why does boosting beat linear regression?**
Fare isn't a straight line in distance and time. There are base fares, rush-hour effects and interactions. Trees capture these; a straight line can't.

**Q: Random Forest vs Gradient Boosting?**
A forest builds trees independently and averages them, which reduces variance. Boosting builds trees in sequence, each fixing errors, which reduces bias. Boosting is often slightly more accurate; Histogram binning makes it fast.

**Q: How do you know it isn't overfitting?**
It's scored on a 20% test set it never saw, and the train and test scores are close. Cross-validation would confirm it.

**Q: Did you tune hyperparameters?**
We used sensible settings (for example 250 iterations, learning rate 0.08, L2 regularisation). Next step: GridSearchCV or RandomizedSearchCV.

**Q: Why pick by RMSE and not MAE?**
RMSE penalises big errors more. A $20 miss hurts trust more than several $1 misses.

**Q: Why a classification report for a regression problem?**
Grouping fares into bands lets us show precision and recall per price range. It's an extra view, for example "how often do we get the price band right". The main metrics are still MAE, RMSE and R².

**Q: Why not deep learning?**
On tabular data of this size, tree ensembles usually match or beat neural networks, and train in seconds.

### App and realism
**Q: How does the app work without the user choosing weather or traffic?**
Traffic is estimated from the time and day. Weather comes live from Open-Meteo when online. Road routes come from OSRM. If offline, it falls back to clear weather and straight-line distance, and says so on screen.

**Q: Are surge, GST and fees learned by the model?**
No. The ML model predicts the base trip fare. Surge, fees and tax are business rules on top, which is how real ride apps work. The values are illustrative.

**Q: Why is the ₹ fare just a conversion?**
The model learned New York prices. For true Indian fares we'd retrain on Indian ride data. The pipeline stays the same.

**Q: How do place suggestions work offline?**
A built-in list of 34,000+ cities in 244 countries (GeoNames data) plus landmarks and airports. Pick a country, click the dropdown and type.

**Q: What's the fare range?**
The estimate ± the model's average test error (MAE).

**Q: How would you deploy it?**
Streamlit Community Cloud, or a Docker container on a cloud server. The model file loads at startup.

**Q: Is the driver card real?**
No, it's simulated for the demo. A real app would get it from the driver-matching system.

### Limitations and next steps
- Training on road distance; adding real traffic, weather and surge history as features.
- City-specific models or a per-city price factor; local currency pricing.
- Hyperparameter tuning, cross-validation, XGBoost or LightGBM.
- Monitoring model drift as prices change over time; retraining on a schedule.

---

## 7. 3-minute pitch
1. **Problem (20s):** "Riders want a fair upfront fare. We predict it from trip details. It's a regression problem."
2. **Data (30s):** "We explored N rides, removed X bad records, and found that distance drives fare."
3. **Features (20s):** "Haversine distance plus time features. No raw coordinates, so it works in any city."
4. **Models (40s):** "We compared three models on a hidden 20% test set. [Winner] won: RMSE $__, R² __, within $__ on average."
5. **Demo (50s):** pick a place in India → show the auto-detected conditions, fare breakdown, driver card, map, best time and sidebar report.
6. **Honesty + next steps (20s):** "The ML model predicts the trip fare; business rules add surge and fees. Next: road-distance training and local pricing."
