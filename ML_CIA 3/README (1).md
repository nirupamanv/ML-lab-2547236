# ML for Social Good — Mission Earth: Air Quality Risk Classification (CIA-3)

**Course:** MCA 521-4 Machine Learning | **Assessment:** CIA-3 — ML for Social Good Ensemble Challenge 
## 1. Project Overview

**Problem.** City pollution-control boards publish raw pollutant sensor readings (PM2.5, PM10, NO2,
SO2, CO, O3, etc.) daily, but raw concentration numbers aren't actionable for the public, schools, or
hospitals. This project trains a model that reads a city's same-day pollutant sensor readings and
classifies the day into the official **CPCB AQI risk category** — `Good`, `Satisfactory`, `Moderate`,
`Poor`, `Very Poor`, or `Severe` — as a decision-support layer for public-health advisories.

**Beneficiaries.** City/state pollution-control boards, school and hospital administrators, and
citizens with respiratory or cardiac conditions who need a same-day actionable risk category rather
than raw pollutant chemistry.

**Dataset.** *Air Quality Data in India (2015–2020)*, day-level pollutant readings for 26 Indian
cities, sourced from the **Central Pollution Control Board (CPCB), Government of India**, published on
Kaggle by Rohan Rao: `https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india`
(file: `city_day.csv`). See Section 3 for the exact citation note and a required action before you submit.

**Approach.** Leakage-safe preprocessing (missing-value audit, IQR outlier capping, one-hot encoding,
scaling — all fit only on the training split) → time-based train/validation/test split (70/15/15,
chronological within each city, so the model never sees the future) → a Logistic Regression baseline
compared against three tuned ensembles: Random Forest (bagging), XGBoost (boosting), and a
Random Forest + XGBoost → Logistic Regression stacking model → SHAP explainability (global + one
individual local prediction) → an explicit ethics and responsible-deployment discussion.

**Results summary** (test set, never touched during training or tuning):

| Model | Accuracy | Macro-F1 | ROC-AUC (OVR, macro) |
|---|---|---|---|
| Baseline: Logistic Regression | 0.740 | 0.655 | 0.945 |
| Bagging: Random Forest | 0.795 | 0.756 | 0.963 |
| Boosting: XGBoost | 0.790 | 0.771 | 0.966 |
| Stacking: RF + XGBoost → LR | 0.793 | 0.771 | 0.965 |

All three ensembles clearly beat the baseline, especially on macro-F1 — the metric that matters most
given the class imbalance (Severe/Good are rare classes). Full breakdown, confusion matrices, and the
reasoning behind every modelling choice are in the notebook itself, section by section, matching the
brief's question numbering (Q1–Q5). See `results/model_comparison.csv` for the exact numbers and
`figures/` for every chart referenced above.

**Academic integrity / acknowledgments.** All code in this project was written for this assignment. No
external code, figures, or pretrained models were copied from another source. The dataset itself is
third-party (see citation above) and is used under Kaggle's dataset terms for the original CPCB public
data. No personally identifiable or confidential data is used anywhere — the dataset is exclusively
environmental sensor readings (pollutant concentration, city, date).

---

## 2. Deliverable Mapping (what the brief asks for → what file satisfies it)

| Brief requirement | Where it is |
|---|---|
| Complete codebase / notebook | `AQI_Mission_Earth_CIA3.ipynb` — the entire pipeline, already executed, structured as Q1→Q5 with a markdown heading + explanation before every code cell |
| README with execution steps | This file, Section 4 |
| Trained pipeline / reproducible training script | `AQI_Mission_Earth_CIA3.ipynb` itself — every model is trained inside the notebook from raw data to metrics, so re-running it top-to-bottom **is** the reproducible training script (see Section 4). `build_notebook.py` is the script that generated the notebook's structure, included for transparency. |
| Three-minute pitch-and-demo video | Not a file in this folder — you record it yourself using `VIDEO_SCRIPT.md` (word-for-word script + shot list + recording instructions), matching the notebook's Q1–Q5 sections on screen |
| Dataset source + citation | Section 1 above and inside the notebook's Q1 section |
| No PII / confidential data | Confirmed — see Section 1 |
| Borrowed code/figures acknowledged | Confirmed — none used, see Section 1 |
| Ethics statement | `ETHICS_STATEMENT.md` (standalone copy of the notebook's Q4 ethics discussion) |
| Results / figures | `results/model_comparison.csv` and everything in `figures/` |

---

## 3. Folder Contents

```
├── AQI_Mission_Earth_CIA3.ipynb   <- MAIN DELIVERABLE. Already executed, all outputs saved inside.
├── build_notebook.py              <- script that generated the notebook (for transparency only)
├── data/
│   └── city_day.csv               <- dataset (see Section 1 — re-download before submitting, below)
├── figures/                       <- all chart PNGs, also embedded in the notebook
├── results/
│   └── model_comparison.csv       <- final metrics table
├── ETHICS_STATEMENT.md            <- standalone copy of the Q4 ethics discussion (required submission item)
├── VIDEO_SCRIPT.md                <- word-for-word 3-minute video script + shot list (Q5)
├── GITHUB_GUIDE.md                <- exact commands to push this project to GitHub
├── requirements.txt               <- exact Python packages needed
└── README.md                      <- this file
```

**Dataset note — action required before submitting:** the `city_day.csv` already in `data/` was
pulled from a public GitHub mirror of the Kaggle dataset above, because the environment this project
was built in has no Kaggle access. It's the exact same data, but **you should re-download
`city_day.csv` directly from the Kaggle link in Section 1** (free account, no cost) and replace the
file in `data/` before you submit, so your citation is accurate and traceable. Nothing else changes —
same filename, same columns, same code, same results.

---

## 4. Execution Steps

### Option A — Run it yourself, step by step (recommended)

**Step 1: Install Python** (skip if already installed)
- Download from `https://www.python.org/downloads/`, install (tick "Add to PATH" on Windows).
- Verify: open a terminal and run `python --version`.

**Step 2: Get the project onto your computer**
- Use this folder directly, or clone it from GitHub once pushed (see `GITHUB_GUIDE.md`).
- Open a terminal and `cd` into the folder:
  ```
  cd path/to/aqi_mission_earth_cia3
  ```

**Step 3: (Optional) Create a virtual environment**
```
python -m venv venv
```
Activate: Windows → `venv\Scripts\activate` | Mac/Linux → `source venv/bin/activate`

**Step 4: Install dependencies**
```
pip install -r requirements.txt
```

**Step 5: Get the dataset**
Download `city_day.csv` from the Kaggle link in Section 1, place it at `data/city_day.csv`.

**Step 6: Launch Jupyter and run the notebook**
```
jupyter notebook
```
Click `AQI_Mission_Earth_CIA3.ipynb` → menu bar → **Kernel → Restart & Run All** → wait ~2–3 minutes.
Every cell re-executes from scratch and regenerates every table/chart from your dataset copy — this
*is* the reproducible training run.

**Step 7 (optional):** re-run a single cell any time with `Shift+Enter` — e.g. to try a different
synthetic record in the live-prediction cell near the end.

### Option B — Google Colab (no local install needed)

1. Go to `https://colab.research.google.com` → File → Upload notebook → select `AQI_Mission_Earth_CIA3.ipynb`.
2. In Colab's file browser (left sidebar), create a `data/` folder and upload `city_day.csv` into it.
3. Runtime → Run all. (If a path error appears, run `!mkdir -p data figures results` in a new cell first.)

---

## 5. What to Actually Submit

- [ ] `AQI_Mission_Earth_CIA3.ipynb` (re-run once yourself first — Section 4 — so you can speak to it confidently in the video)
- [ ] This `README.md`, with your name/register number filled in (top of this file and top of the notebook)
- [ ] `results/model_comparison.csv` and the `figures/` folder
- [ ] `ETHICS_STATEMENT.md`
- [ ] The 3-minute pitch-and-demo video, recorded using `VIDEO_SCRIPT.md`
- [ ] Dataset citation (Section 1 — confirm you've swapped in the real Kaggle-downloaded CSV first)
- [ ] GitHub repo link, if your course requires one (`GITHUB_GUIDE.md`)

---

## 6. Note on `results/artifacts.pkl`

Re-running the notebook's final cells will save `results/artifacts.pkl` (the fitted pipeline + trained
models, for reuse without retraining). It is **not shipped** here — it's large (~170MB, because the
untuned-depth Random Forest is big) and depends on a class defined inside the notebook, so it only
unpickles cleanly from a session that has already run the notebook. It regenerates automatically the
moment you run the notebook yourself — nothing is missing from the actual deliverables.

---

## 7. Honesty Note on the ML Results

The metrics quoted in Section 1 (accuracy ≈0.79, macro-F1 ≈0.77 for the best ensembles vs. macro-F1
≈0.65 for the baseline) are **real numbers from an actual executed run on the real CPCB dataset**,
computed in this project's environment — nothing here is invented or estimated. Re-running it yourself
should produce numbers very close to these (small differences are possible from
`RandomizedSearchCV`'s randomness and from swapping in the freshly Kaggle-downloaded CSV), with the
same overall pattern holding: every ensemble clearly beats the linear baseline, and XGBoost/Stacking
are roughly tied for best.
