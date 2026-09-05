# Explainable AI: Predicting U.S. Employment Growth 2024–2034 from O*NET Occupation Attributes

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.52.0-black)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

> 🇩🇪 **Der Originalinhalt des Notebooks ist auf Deutsch.** Eine deutsche Fassung dieser README liegt in [`README.de.md`](README.de.md).

A single, self-contained Jupyter notebook that asks a simple question:

**Can the U.S. Bureau of Labor Statistics' projected employment change for an occupation (2024–2034) be predicted from what that occupation actually involves — the skills, knowledge, abilities, activities, work context and work styles that O\*NET records for it?**

The notebook joins two independent public datasets, trains a Random Forest regressor, benchmarks it against a mean baseline and a regularised linear model, and then uses **SHAP (SHapley Additive exPlanations)** to make both the global behaviour of the model and its individual predictions interpretable.

---

## Table of Contents

- [Research question](#research-question)
- [Results at a glance](#results-at-a-glance)
- [Data sources](#data-sources)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [What the notebook does, step by step](#what-the-notebook-does-step-by-step)
- [Modelling details](#modelling-details)
- [Explainability with SHAP](#explainability-with-shap)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Limitations](#limitations)
- [References](#references)

---

## Research question

> Which O\*NET occupation attributes are most strongly associated with the BLS-projected employment growth rate for 2024–2034, and how can the contribution of individual attributes to a concrete model prediction be explained with SHAP?

The framing is deliberately **associational, not causal**. The model learns patterns that link occupation profiles to BLS projections; it does not produce an independent forecast of the labour market.

---

## Results at a glance

Evaluated on a held-out 20 % test set (155 occupations); cross-validated R² is a 5-fold score on the training set only.

| Model | MAE (pp) | RMSE (pp) | R² (test) | CV-R² (train) |
|---|---:|---:|---:|---:|
| Baseline (predicts the mean) | 5.118 | 7.135 | −0.009 | −0.001 |
| Ridge Regression (`RidgeCV`, α = 221.2) | 4.353 | 5.928 | 0.303 | 0.349 |
| **Random Forest (tuned)** | **4.149** | **5.834** | **0.325** | **0.352** |

*MAE/RMSE are in percentage points of projected employment change.*

**How to read this:** roughly a third of the variance in BLS growth projections is recoverable from occupation attributes alone. That is a real, non-trivial signal — the baseline explains nothing — but it is far from a solved prediction problem, and the tree ensemble only marginally beats a well-regularised linear model. Both models systematically shrink predictions toward the mean, so fast-growing outliers are heavily under-predicted:

| Occupation | Actual | Predicted | Abs. error |
|---|---:|---:|---:|
| Data scientists | 33.5 % | 5.98 % | 27.52 |
| Operations research analysts | 21.5 % | 4.75 % | 16.75 |
| Adult basic education, adult secondary e… (instructors) | −13.7 % | 2.07 % | 15.77 |
| Computer numerically controlled tool programmers | 12.8 % | −2.29 % | 15.09 |
| Nuclear power reactor operators | −15.3 % | −0.42 % | 14.88 |

The target itself spans **−36.1 % to +49.9 %**.

---

## Data sources

Two independent public datasets are joined on the 6-digit SOC code.

### 1. O\*NET — occupation attributes (`datasets/*.xlsx`)

The O\*NET database describes each occupation through hundreds of rated descriptors. The files ship in *long format* (one row per occupation × element × rating scale) and are pivoted to *wide format* in the notebook.

| File | Scale ID used | Meaning | Features produced | Prefix |
|---|---|---|---:|---|
| `Abilities.xlsx` | `IM` | Importance | 52 | `ability_` |
| `Essential Skills.xlsx` | `IM` | Importance | 10 | `skill_` |
| `Knowledge.xlsx` | `IM` | Importance | 33 | `knowledge_` |
| `Work Activities.xlsx` | `IM` | Importance | 41 | `activity_` |
| `Work Context.xlsx` | `CX` | Context rating (the file's only numeric scale) | 55 | `context_` |
| `Work Styles.xlsx` | `WI` | Work Styles Impact | 21 | `style_` |
| `Job Zones.xlsx` | — | Already wide; preparation/training level 1–5 | 1 | `job_zone` |
| `Occupation Data.xlsx` | — | Occupation titles and descriptions (join metadata) | — | — |

**→ 213 O\*NET features across 923 O\*NET-SOC codes.**

### 2. BLS Employment Projections (`datasets/occupation.xlsx`, sheet `Table 1.2`)

The National Employment Matrix, 2024–34. Only rows with `Occupation type == "Line item"` are kept (individual occupations, not roll-up totals such as *"Total, all occupations"*).

- **Target:** `Employment change, percent, 2024–34`
- **Base-year covariates kept:** employment 2024, median annual wage 2024, percent self-employed 2024, typical entry education, required work experience, typical on-the-job training.

---

## Repository structure

```
explainable_ai/
├── index.ipynb          # The entire analysis: loading → cleaning → modelling → SHAP
├── requirements.txt     # Pinned dependencies
├── datasets/
│   ├── Abilities.xlsx           # O*NET: 52 ability ratings per occupation
│   ├── Essential Skills.xlsx    # O*NET: 10 basic skills
│   ├── Knowledge.xlsx           # O*NET: 33 knowledge domains
│   ├── Work Activities.xlsx     # O*NET: 41 generalised work activities
│   ├── Work Context.xlsx        # O*NET: 55 physical/social context ratings
│   ├── Work Styles.xlsx         # O*NET: 21 work style ratings
│   ├── Job Zones.xlsx           # O*NET: job zone 1–5
│   ├── Occupation Data.xlsx     # O*NET: titles + descriptions
│   └── occupation.xlsx          # BLS Employment Projections, Table 1.2
├── README.md            # This file (English)
└── README.de.md         # German version
```

There is no `src/` package — this is intentionally a single-notebook analysis, meant to be read top to bottom.

---

## Getting started

**Requirements:** Python 3.11+ (developed on 3.14).

```bash
git clone https://github.com/<your-user>/explainable_ai.git
cd explainable_ai

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab index.ipynb
```

Then run all cells top to bottom. All data is committed to the repo, so no downloads are needed. Every random operation is seeded with `RANDOM_STATE = 42`, so the numbers above reproduce exactly.

**Runtime note:** the `RandomizedSearchCV` in section 7 fits 20 candidate configurations × 5 folds of forests with 400–800 trees. Expect a few minutes on a modern laptop. It runs with `n_jobs=1` at the search level and `n_jobs=-1` inside each forest.

---

## What the notebook does, step by step

| Section | What happens |
|---|---|
| **1. Setup** | Imports, `RANDOM_STATE = 42`, plotting theme. |
| **2.1 Load O\*NET** | `pivot_onet()` reads each file, filters to a single `Scale ID`, and pivots long → wide (`index="O*NET-SOC Code"`, `columns="Element Name"`, `values="Data Value"`), prefixing every column by its source domain. |
| **2.2 Merge & aggregate** | An outer merge chains all seven wide frames together (923 × 215). The 8-digit O\*NET-SOC code is truncated to the 6-digit SOC code, which collapses 923 codes into **798** — sub-occupations (e.g. `11-1011.00` and `11-1011.03`) are averaged per `soc_code` **before** the BLS join. |
| **2.3 Load BLS** | `Table 1.2` is read with `skiprows=1`, filtered to line items, columns renamed to snake_case, numeric columns coerced with `errors="coerce"` (the source uses "—" for suppressed values), rows with a missing target dropped → **832** occupations. |
| **2.4 Final join** | Inner merge on `soc_code` → **773 occupations × 223 columns**, asserted to be one row per unique SOC code. |
| **2.5 / 3. Data audit** | Missing-value profile, duplicate check on full rows and on `occupation_title` (both zero), target range check. Explicitly *checks* for duplicates instead of blindly calling `drop_duplicates()`. |
| **4. Feature definition** | All 213 O\*NET features plus `job_zone` are kept, and derived structural features are added (see below) → **220 features** (216 numeric, 4 categorical). |
| **5. Split** | `train_test_split(test_size=0.2, random_state=42)` → 618 train / 155 test. Done **before** any imputer or scaler is fitted. |
| **6. Baseline + Ridge** | `DummyRegressor(strategy="mean")` and a Ridge pipeline with median imputation + standardisation (numeric) and one-hot encoding (categorical); α chosen by `RidgeCV` over `np.logspace(-2, 4, 30)`. |
| **7. Random Forest** | Median imputation + ordinal encoding, tuned by `RandomizedSearchCV` (20 draws, 5-fold `KFold`, `scoring="r2"`). |
| **8. Evaluation** | MAE / RMSE / R² on test + CV-R² on train for all three models; actual-vs-predicted scatter; top-5 largest absolute errors with occupation names. |
| **9. SHAP** | `TreeExplainer` on the fitted forest → global bar plot, beeswarm plot, and two single-case waterfall plots. |
| **10. Conclusion** | Interpretation and explicit caveats about causality and forecast validity. |

---

## Modelling details

### Feature set (220 features)

| Group | Count |
|---|---:|
| `context_` — work context | 55 |
| `ability_` — abilities | 52 |
| `activity_` — work activities | 41 |
| `knowledge_` — knowledge domains | 33 |
| `style_` — work styles | 21 |
| `skill_` — basic skills | 10 |
| Structural (BLS/SOC) | 8 |

The structural block adds context that pure attribute ratings cannot supply:

- `log_employment_2024`, `log_median_wage_2024` — log1p-transformed, since both distributions are strongly right-skewed
- `pct_self_employed` — share of self-employed workers in 2024
- `education_entry`, `work_experience`, `on_the_job_training` — typical qualification requirements (categorical)
- `major_group` — the first two digits of the SOC code, i.e. the SOC major occupational group, because projections cluster within groups
- `job_zone` — O\*NET preparation level 1–5

### Preprocessing

Both model pipelines wrap preprocessing in a `ColumnTransformer` **inside** the pipeline, so every imputation and scaling statistic is fitted on training folds only:

- **Ridge:** median imputation + `StandardScaler` (numeric), `OneHotEncoder(handle_unknown="ignore")` (categorical)
- **Random Forest:** median imputation (numeric), `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` (categorical) — sufficient and more compact than one-hot for tree models

Missingness is real and unevenly distributed: `work_experience` is missing for 660 of 773 rows, `on_the_job_training` for 303, `pct_self_employed` for 190, and each O\*NET column for ~22 occupations that O\*NET does not profile.

### Hyperparameter search

```python
param_distributions = {
    "rf__n_estimators":     [400, 600, 800],
    "rf__max_depth":        [None, 15, 25],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features":     [0.2, 0.33, 0.5, "sqrt"],
}
```

Selected: `n_estimators=400`, `max_depth=15`, `min_samples_leaf=4`, `max_features=0.2` — CV-R² **0.352**. The low `max_features` value is what you would expect with 220 partly redundant predictors: sampling few features per split decorrelates the trees.

---

## Explainability with SHAP

The Random Forest is explained with `shap.TreeExplainer`, computed on the preprocessed test matrix so that feature names survive into the plots. Three complementary views:

1. **Bar plot** (`plot_type="bar"`) — mean absolute SHAP value per feature: which attributes move predictions most, ignoring direction.
2. **Beeswarm plot** — the same ranking, but each occupation is a dot coloured by its feature value, revealing whether *high* (red) or *low* (blue) values push the prediction up or down.
3. **Waterfall plots** — two single occupations traced from the model's base value to its final prediction, one with a high and one with a low predicted growth rate. The notebook looks for *Data scientists* and *Cashiers* in the test split and falls back to the argmax/argmin prediction when they are not there (in the committed run: *Data scientists*, +6.0 %, and *Textile cutting machine setters, operators and tenders*, −8.7 %).

The notebook's own reading of these plots is stated plainly: a feature ranking high means **the model attends to it**, and that occupations with similar attribute profiles historically received similar BLS projections — not that the attribute causes an occupation to grow or shrink.

---

## Design decisions worth knowing

These are the choices that would otherwise be invisible, and each is argued for in the notebook itself.

**Leakage is avoided in three places.**
1. `Employment, 2034`, `Employment change, numeric, 2024–34` and `Occupational openings, 2024–34` are deliberately **not** used as features — they are algebraically derived from the target and would inflate R² to a meaningless degree.
2. The train/test split happens before any preprocessing statistic is estimated.
3. Sub-occupation aggregation happens **before** the BLS merge, so no occupation contributes rows to both sides of a duplicate.

**All available features are used, not a hand-picked subset.** Pre-selecting a handful of "plausible" attributes discards information, and it is unnecessary for interpretability: SHAP identifies the relevant features in the trained model anyway, and a Random Forest tolerates many correlated inputs.

**Only the Importance (IM) scale is used, not Level (LV).** O\*NET rates Abilities, Skills, Knowledge and Work Activities on both scales. They correlate very strongly, and adding LV was found to bring no predictive benefit.

**Known deviations in the bundled O\*NET files.** `Essential Skills.xlsx` contains only the 10 *Basic Skills* (Critical Thinking, Active Listening, …), not the full O\*NET skill taxonomy with cross-functional skills such as *Social Perceptiveness* or *Complex Problem Solving*. `Work Styles.xlsx` likewise uses a slightly different element list than the standard taxonomy (e.g. no *Independence*). This narrows taxonomic coverage, not the method.

---

## Limitations

- **R² ≈ 0.33 is a modest fit.** Two thirds of the variance in BLS projections is not explained by occupation attributes. Predictions regress heavily toward the mean, so the occupations you would most want to identify — the extreme growers and decliners — are exactly the ones predicted worst.
- **The target is a projection, not an outcome.** The model is trained on BLS *forecasts* for 2024–34, which rest on assumptions about economic, technological and demographic developments. It learns the structure of those forecasts, not the future.
- **Correlation, not causation.** SHAP attributions describe the model's behaviour, not labour-market mechanisms.
- **n = 773.** Each row is an occupation, not a worker; the sample is small for a 220-feature model, which is why regularisation and cross-validation carry a lot of weight here.
- **Heavy missingness in three structural columns** (`work_experience`, `on_the_job_training`, `pct_self_employed`) is handled by median imputation, which may itself carry signal that the model can exploit.
- **Single split.** Test metrics come from one 80/20 split with a fixed seed; repeated or nested CV would give a more honest uncertainty estimate.

---

## References

- **O\*NET Resource Center** — occupation attribute database: <https://www.onetcenter.org/database.html> (O\*NET data is provided by the U.S. Department of Labor under a CC BY 4.0 licence)
- **BLS Employment Projections** — National Employment Matrix, 2024–34: <https://www.bls.gov/emp/> (U.S. government work, public domain)
- **SHAP** — Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*: <https://github.com/shap/shap>
- **scikit-learn** — <https://scikit-learn.org>

---

## License

No code licence file is included yet — add one before sharing the repository publicly. The bundled data retains the terms of its original publishers (O\*NET: CC BY 4.0; BLS: public domain).
