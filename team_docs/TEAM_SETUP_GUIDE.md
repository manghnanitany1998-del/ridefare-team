# RideFare: Team Setup Guide (run the project and the Streamlit app)

Follow this top to bottom on a Windows laptop. Takes about 15 minutes the first time.

---

## 1. Install the tools (once)
| Tool | Where | Notes |
|---|---|---|
| Python 3.11–3.13 | python.org/downloads | Tick **"Add Python to PATH"** during install |
| Git + Git Bash | git-scm.com | Use Git Bash for all commands below |
| VS Code (optional) | code.visualstudio.com | Install the **Python** and **Jupyter** extensions |

Check in Git Bash:
```bash
python --version
git --version
```

---

## 2. Get the code
```bash
cd ~/Desktop
git clone https://github.com/manghnanitany1998-del/ridefare-team.git
cd ridefare-team/main_project
```
(You must be added as a collaborator first. Ask Tanya.)

---

## 3. Create the virtual environment and install libraries
A **virtual environment (venv)** is a private box of libraries for this project only.

```bash
python -m venv venv
source venv/Scripts/activate          # Git Bash. PowerShell: venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
You should see `(venv)` at the start of the line. **Activate it every time you open a new terminal.**

Tip: keep the venv **outside OneDrive** if the first import is very slow:
```bash
python -m venv /c/Users/<you>/venvs/ridefare
source /c/Users/<you>/venvs/ridefare/Scripts/activate
```

If pip fails with SSL / certificate errors (company network):
```bash
python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
```
or switch to a phone hotspot.

---

## 4. Run the notebook (trains the model)
```bash
jupyter notebook notebooks/Uber_Fare_Prediction.ipynb
```
In the browser: **Kernel → Restart & Run All**. It creates:
- `models/uber_fare_model.pkl`: the saved final model (the app needs this)
- `reports/results.csv` and `reports/results.json`: the results table
- `reports/figures/*.png`: all charts

VS Code alternative: open the notebook, choose the `venv` kernel (top right), click **Run All**.

---

## 5. Run the Streamlit app
From the `main_project` folder (so the dark theme in `.streamlit/config.toml` is picked up):
```bash
python -m streamlit run app/app.py
```
It opens at **http://localhost:8501**. Stop it with **Ctrl + C**.

### What the app does
| Part | How it works |
|---|---|
| Place search | Pick a country, then pickup and drop-off. 34,000+ cities + 85 Indian landmarks from `data/places.csv` (works offline) |
| Map | Interactive Leaflet map with OpenStreetMap tiles (no API key), dark styled, when online; a drawn trip sketch when offline |
| City trip (same city on the map) | The trained model predicts the fare from distance, hour, weekday, weekend, city, payment, at any length |
| Intercity (different cities on the map) | A labelled **rule**, not the model: one-way km × ₹/km, driver allowance, tolls, 5% GST |
| India price level | Sidebar slider (default 0.12) scales Indian city fares, because the model learned US prices |
| Extras | Illustrative ride types, surge, night/airport/booking fees, GST, fare breakdown, split fare, driver card |
| Currency | Local currency from `data/currencies.csv`, ₹ shown alongside; live rates when online |
| Sidebar | Settings (USD→INR rate, road distance), model picked, all model metrics, feature importance |

### Quick test trips
- India → Hawa Mahal, Jaipur → Jaipur International Airport (model)
- India → Mehrangarh Fort, Jodhpur → Umaid Bhawan Palace, Jodhpur (model)
- India → Hawa Mahal, Jaipur → Gateway of India, Mumbai (intercity rule, about ₹17,700 for Go)

### Online vs offline
On company networks the map tiles, live search, road routes (OSRM), weather (Open-Meteo) and exchange rates may be blocked. The app still works: built-in places, straight-line distance × 1.25 for road, assumed clear weather, approximate rates. Use a hotspot to see everything live.

---

## 6. Common errors
| Error | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the venv, then `python -m pip install -r requirements.txt` |
| `Model not found` in the app | Run the notebook first (step 4) |
| `File does not exist: app/app.py` | You are in the wrong folder. `cd` into `main_project` |
| `streamlit: command not found` | Use `python -m streamlit run app/app.py` |
| Very slow first start / looks frozen | First scikit-learn import; wait. Don't press Ctrl + C. Move the venv out of OneDrive |
| White text on white / no dark theme | Start the app from the `main_project` folder and fully restart it |
| Map is blank | Network blocks map tiles; the offline sketch should show. Try a hotspot |

---

## 7. Making changes and sharing them
```bash
git pull                              # get teammates' latest changes first
# ...edit files...
git add .
git commit -m "Describe what you changed"
git push
```
If you retrain the model, push `models/uber_fare_model.pkl` and `reports/` too.

---

## 8. Repo layout
```
ridefare-team/
├── main_project/          the clean project (also pushed to the public "ridefare" repo)
│   ├── notebooks/       Uber_Fare_Prediction.ipynb
│   ├── app/app.py       Streamlit app
│   ├── data/            dataset, places, currencies
│   ├── models/          saved model (created by the notebook)
│   ├── reports/         results + charts (created by the notebook)
│   ├── presentation/    RideFare_Final_Presentation.pptx
│   ├── .streamlit/      dark theme
│   └── requirements.txt, README.md
└── team_docs/           study guide, final PPT, instructions PDF, learning notes, repo guide, archive of old code
```
