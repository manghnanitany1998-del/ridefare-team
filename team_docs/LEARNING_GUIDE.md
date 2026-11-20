# Uber Fare Prediction – Complete Learning Guide

Read this from top to bottom tonight. Run each step on your laptop as you read.

---

## 0. The big picture

You are building a **machine learning (ML) system**:

```
uber.csv  ->  clean  ->  features  ->  train models  ->  pick best  ->  fare_model.pkl  ->  Streamlit website
```

- **Machine learning** = the computer learns a pattern from past examples (past rides + their fares)
  and uses it to predict new ones.
- **Supervised learning** = every training row has the correct answer (`fare_amount`). That answer is the **target** (y).
  The other columns are **features** (X).
- **Regression** = the target is a number (dollars). If it were a category (e.g. "cheap/expensive")
  it would be **classification**. Fare prediction is regression.

---

## 1. Project files

| File | What it does | When you run it |
|---|---|---|
| `generate_sample_data.py` | Creates a FAKE `uber.csv` to practise | Today only |
| `explore_data.py` | Prints statistics + saves charts | First, on the real data |
| `train_model.py` | Cleans, builds features, trains 4 models, saves the best | Second |
| `app.py` | The Streamlit website | Third |
| `requirements.txt` | List of libraries | Once, at setup |
| `uber_fare_prediction.pptx` | Presentation | At the end |

---

## 2. Setup in VS Code (do this tonight)

1. Install **Python 3.10 or newer** from python.org. Tick **"Add Python to PATH"** during install.
2. Install **VS Code**, then the **Python** extension (Extensions tab, search "Python", by Microsoft).
3. In VS Code, use **File > Open Folder** to open the `Hackathon project` folder.
4. Open the terminal with **Terminal > New Terminal**.
5. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   You should see `(venv)` at the start of the line.
   *Why:* a venv is a private box of libraries for this project, so nothing clashes with other projects.
   *If activation is blocked by PowerShell:* run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.
6. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
7. Bottom-right of VS Code: click the Python version and choose the one with `venv`.

### What each library is for
- **pandas** – tables (DataFrames). Read CSV, filter rows, create columns.
- **numpy** – fast maths on whole columns at once (used for the distance formula).
- **matplotlib** – charts.
- **scikit-learn (sklearn)** – the main ML toolkit: models, train/test split, metrics.
- **xgboost** – a very strong boosted-tree model.
- **joblib** – saves the trained model to a file and loads it back.
- **streamlit** – turns a Python script into a web page. No HTML needed.

> `explore_data.py` needs matplotlib. If you see `No module named matplotlib`, run `pip install matplotlib`.

---

## 3. Practise run today (full project, end to end)

```
python generate_sample_data.py     # makes fake uber.csv (50,000 rides)
python explore_data.py             # look at the data, charts saved in /charts
python train_model.py              # trains models, prints scores, saves fare_model.pkl
streamlit run app.py               # opens the website at http://localhost:8501
```
Press **Ctrl + C** in the terminal to stop the website.

**Tomorrow:** delete the fake `uber.csv`, put the real file in the folder, rename it `uber.csv` (or change `DATA_PATH` in `train_model.py`), and run the last three commands again.

---

## 4. Step-by-step explanation of the code

### Step A – Load (`pd.read_csv`)
Reads the CSV into a DataFrame. `df.shape` shows (rows, columns).

### Step B – Explore (EDA, `explore_data.py`)
Never train blind. Check:
- `df.head()` – what the rows look like and what the columns are called.
- `df.dtypes` – numbers or text? Dates arrive as text and must be converted.
- `df.isnull().sum()` – how many values are missing.
- `df.describe()` – min/max reveal errors (negative fares, 0 passengers, latitude 0).
- Charts: the fare histogram (most fares are small = skewed), distance vs fare (almost a straight line;
  on the sample data the correlation is 0.94), and fare by hour (rush hours cost more).

**Tell the judges:** "EDA showed X bad rows and that distance is the main driver of fare."

### Step C – Clean
| Rule | Why |
|---|---|
| `dropna` | Models can't handle empty values |
| fare > 0 and < 200 | Negative fares are errors; huge ones are outliers that pull the model off |
| passengers 1–6 | 0 or 200 passengers is impossible |
| lat in -90..90, lon in -180..180 | These are the valid ranges on Earth |
| remove (0,0) GPS | (0,0) means the GPS was missing |
| distance 0.1–100 km | 0 km = cancelled ride; 5,000 km = GPS error |

### Step D – Feature engineering (the most important step)
The model can't understand raw coordinates well, so we turn them into meaningful numbers:
- **distance_km (Haversine formula)** – the distance between two points on a sphere (the Earth, radius 6371 km).
  Steps: convert degrees to radians, compute `a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)`,
  then `distance = 2·R·arcsin(√a)`.
- **hour** – rush hour vs night.
- **day_of_week, is_weekend** – weekday vs weekend demand.
- **month, year** – season, and fares rising over the years (inflation, price changes).
- **is_peak** – 1 if 7–9am or 4–7pm, else 0. This is a "flag" (binary feature).
- **passenger_count** – kept as it is.

`pd.to_datetime(..., utc=True)` converts the text date into a real date so we can take `.dt.hour` etc.

### Step E – Train/test split
`train_test_split(X, y, test_size=0.2, random_state=42)`
- 80% of rows **train** the model, and 20% are hidden to **test** it.
- *Why:* scoring a model on the data it learned from is like marking your own exam with the answer key.
  The test set shows how it does on rides it has never seen.
- `random_state=42` makes the random split the same every time (reproducible). 42 is just a convention.

### Step F – The four models
| Model | How it works (simple) | Strength | Weakness |
|---|---|---|---|
| **Linear Regression** | fare = a·distance + b·hour + … + c (a straight-line formula) | Fast, easy to explain, a baseline | Can't capture curves or interactions |
| **Random Forest** | Builds 100 decision trees on random parts of the data and averages their answers | Captures non-linear patterns, resists overfitting | Bigger, slower |
| **Gradient Boosting** | Builds trees one after another; each new tree fixes the errors of the previous ones | Very accurate | Slower to train |
| **XGBoost** | An optimised, regularised version of gradient boosting | Usually the best on tables, fast | More settings to tune |

**Decision tree** = a flowchart of yes/no questions, e.g. "distance > 5 km? → hour in rush? → predict $18".
**Ensemble** = many trees combined. The combined answer is better than any single tree.

Key settings:
- `n_estimators` = number of trees. More is usually better but slower.
- `max_depth` = how many questions each tree can ask. Too deep = memorises the data (overfitting).
- `learning_rate` (boosting) = how big a correction each tree makes. Smaller = slower but more careful.
- `subsample / colsample_bytree` = each tree sees a random 80% of the rows/columns, which reduces overfitting.
- `n_jobs=-1` = use all CPU cores.

**Overfitting** = the model memorises the training data and does badly on new data. Check: the training score is much better than the test score.
**Underfitting** = the model is too simple and bad on both (often linear regression here).

### Step G – Metrics
- **MAE (Mean Absolute Error)** – the average of |actual − predicted|. "On average we are off by $2.10." Easiest to explain.
- **RMSE (Root Mean Squared Error)** – square the errors, average them, take the square root. Big mistakes count more. Lower is better. **We pick the model with the lowest RMSE.**
- **R² (R-squared)** – the share of fare variation the model explains. 1.0 = perfect, 0 = no better than always guessing the average fare.

### Step H – Save (`joblib.dump`)
Saves the best model plus the feature list, model name and all scores in `fare_model.pkl`.
The website loads this file, so it never retrains: it opens instantly.

### Step I – The website (`app.py`)
- `st.set_page_config` – browser tab title and icon.
- `@st.cache_resource` – Streamlit re-runs the whole script on every click. The cache loads the model only once.
- `st.columns` – side-by-side layout.
- `st.number_input / date_input / time_input / slider` – input widgets.
- `st.button` – when clicked, we build ONE row with **exactly the same features, in the same order** as in training
  (`[FEATURES]` enforces the order), then call `model.predict`.
- `st.success / st.metric / st.map` – show the fare, the distance and the route on a map.
- `st.expander` – a collapsible table of all models' scores.

**Golden rule:** the features in the app must be computed exactly like in training. Otherwise the predictions are wrong.

---

## 5. Adapting to tomorrow's real dataset (most likely problems)

1. **Different column names** → run `explore_data.py`, look at the printed column names, and edit the `COLS = {...}` map at the top of `train_model.py`. Also edit the column names in `explore_data.py`.
2. **Very large file (millions of rows)** → after `pd.read_csv`, add `df = df.sample(200000, random_state=42)`.
3. **Already has a distance column** → you can use it directly instead of the Haversine distance.
4. **Extra columns** (e.g. weather, car type, surge) → add them in `build_features` AND in the `row` inside `app.py`. Text columns need encoding: `pd.get_dummies(df["car_type"])`.
5. **Date in a different format** → `pd.to_datetime(..., dayfirst=True)` or pass `format="%d-%m-%Y %H:%M"`.
6. **XGBoost won't install** → the script skips it automatically. The other 3 models still run.
7. **Prices in rupees** → change `< 200` in the cleaning step to a sensible limit and change `$` to `₹` in `app.py`.

---

## 6. Common errors and fixes

| Error | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python with "Add to PATH" ticked, or use `py` instead of `python` |
| `ModuleNotFoundError: No module named 'xyz'` | Activate the venv, then `pip install xyz` |
| `FileNotFoundError: uber.csv` | The CSV isn't in the same folder, or has a different name |
| `KeyError: 'fare_amount'` | The column name is different → edit `COLS` |
| `FileNotFoundError: fare_model.pkl` in the app | Run `train_model.py` first |
| `streamlit` not recognized | Use `python -m streamlit run app.py` |
| Training takes forever | Sample the data (see 5.2), or lower `n_estimators` |

---

## 7. 5-hour hackathon timeline

| Time | Task |
|---|---|
| 0:00–0:20 | Copy the dataset in, run `explore_data.py`, fix column names |
| 0:20–1:00 | Run `train_model.py`, write down the scores |
| 1:00–2:00 | Improve: tune `n_estimators`/`max_depth`, add any extra features the dataset offers |
| 2:00–3:00 | Polish the app (title, ₹/$, layout) |
| 3:00–4:00 | Put the real scores + EDA charts into the PPT (slide 6 table, add charts) |
| 4:00–4:40 | Rehearse the demo twice |
| 4:40–5:00 | Buffer for bugs |

---

## 8. What to say when presenting (about 3 minutes)

1. **Problem (20s):** "Riders want a fair, upfront fare estimate. We predict fare from trip details – a regression problem."
2. **Data (30s):** "We explored N rows, found and removed X bad records – negative fares, missing GPS, zero passengers."
3. **Features (30s):** "Raw coordinates mean little, so we computed the Haversine distance, plus hour, weekday and rush-hour flags. Distance correlates strongly with fare."
4. **Models (40s):** "We compared four models on a hidden 20% test set. [Best model] won with RMSE $__ and R² __ – on average we are within $__ of the real fare."
5. **Why it won (20s):** "Tree ensembles capture non-linear effects like rush-hour surcharges, which linear regression can't."
6. **Live demo (40s):** enter a trip and show the fare, distance and map.
7. **Next steps (10s):** weather/traffic data, address search, cloud deployment.

### Likely judge questions
- *Why not just linear regression?* It underfits; fare isn't purely linear (base fare, peak pricing).
- *How do you know it isn't overfitting?* We score it on unseen test data; the train and test scores are close.
- *Which feature matters most?* Distance. Show it with `model.feature_importances_` (for tree models).
- *How would you deploy it?* Streamlit Community Cloud or a Docker container; the model file loads at startup.

### Bonus: feature importance (add at the end of train_model.py)
```python
best = models[best_name]
if hasattr(best, "feature_importances_"):
    imp = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature importance:\n", imp.round(3))
```
