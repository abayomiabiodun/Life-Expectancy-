# Life-Expectancy-

# Life Expectancy Analysis & Prediction

An end-to-end data science project that explores what drives life expectancy across 183 countries (2000-2015) and builds a machine learning model to predict it, using WHO and United Nations data.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Key Findings](#key-findings)
- [Model Results](#model-results)
- [Recommendations](#recommendations)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Future Work](#future-work)

## Problem Statement

Life expectancy at birth is a core indicator of a population's health and development, yet gaps of more than 30 years still separate the highest and lowest countries. This project aims to:

1. Identify which immunization, mortality, economic and social factors are most associated with life expectancy.
2. Measure how developed and developing countries differ, and whether the gap has narrowed.
3. Predict life expectancy accurately, including for countries the model has never seen.

This is a supervised **regression** problem. The target variable is `Life expectancy` (years).

## Dataset

- **Source:** Global Health Observatory (GHO) data repository of the World Health Organization (WHO), with matching economic data from the United Nations.
- **Coverage:** 193 countries, 2000-2015, 2,938 rows and 22 columns (20 predictors).
- **Usable data:** 10 countries appear in a single row with no life-expectancy value. After removing those rows, 183 countries (2,928 rows) remain.
- **Predictor categories:**

| Category | Variables |
|---|---|
| Immunization | Hepatitis B, Polio, Diphtheria |
| Mortality | Adult Mortality, infant deaths, under-five deaths, HIV/AIDS, Measles |
| Economic | GDP, percentage expenditure, Total expenditure, Income composition of resources |
| Social | Schooling, Alcohol, BMI, thinness (1-19 and 5-9 years), Population, Status, Year |

## Approach

| Stage | What was done |
|---|---|
| Data understanding | Structure, types, panel completeness, data dictionary |
| Data cleaning | Standardised column names, domain-based validation of values, dropped the unreliable `bmi` column (about 48% implausible values), within-country interpolation of missing values, outliers reviewed and kept |
| EDA | Target distribution, skewness, Spearman and Pearson correlations, Mann-Whitney test, Variance Inflation Factor (VIF) |
| Visualization | Time trends, country rankings, scatter relationships, multi-dimensional plots |
| Feature engineering | Log transforms, averaged vaccination coverage and thinness features, `year_idx`, binary status encoding, removal of collinear `under_five_deaths` |
| Modeling | Mean baseline, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost |
| Validation | **Group-aware split by country** (`GroupShuffleSplit` and `GroupKFold`) so scores reflect performance on unseen countries and avoid leakage |
| Tuning | `RandomizedSearchCV` with grouped cross-validation |
| Interpretation | Permutation importance, partial dependence, Ridge coefficients, error analysis, and a sensitivity check without mortality indicators |

## Key Findings

- Developed countries live about **12 years longer** on average (79.2 vs 67.1 years). The gap narrowed from 12.2 to 11.0 years between 2000 and 2015, because developing countries gained more (+5.1 vs +3.9 years).
- The strongest positive associations are **income composition of resources (HDI)** and **schooling**. The strongest negative associations are **HIV/AIDS** and **adult mortality**.
- Relationships are non-linear: GDP shows diminishing returns, and predicted life expectancy falls sharply once HIV/AIDS burden rises above low levels.
- The three most influential features in the final model are income composition of resources, adult mortality and HIV/AIDS burden.
- Alcohol appears positively correlated with life expectancy because wealthier countries drink more. This is confounding, not a health benefit.

## Model Results

Evaluated on a held-out test set of countries never seen during training:

| Model | RMSE (years) | MAE (years) | R² |
|---|---|---|---|
| Mean baseline | 9.95 | 7.98 | -0.06 |
| Ridge (tuned) | 3.38 | 2.60 | 0.88 |
| Gradient Boosting (default) | 2.75 | 2.01 | 0.92 |
| Random Forest (tuned) | 2.64 | 1.91 | 0.93 |
| XGBoost (tuned) | 2.65 | 2.05 | 0.93 |
| **XGBoost (default), selected** | **2.49** | **1.89** | **0.93** |

- The selected model is typically off by about 1.9 years, is within 3 years for about 81% of country-years and within 5 years for about 95%.
- Hyperparameter tuning produced only marginal changes, smaller than fold-to-fold variation, so the well-chosen XGBoost defaults were kept.
- **Sensitivity check:** without the two mortality indicators (which partly restate the outcome), the model still reaches R² of about 0.90 (RMSE about 3.05), and HIV/AIDS and income composition become the dominant drivers.

## Recommendations

**Policy and health programmes**
1. Prioritise HIV/AIDS prevention, testing and treatment where burden is high.
2. Invest in broad human development (education, income, living standards), not only health services.
3. Target adult mortality through non-communicable disease control, injury prevention and primary care.
4. Focus on lagging developing countries, where gains have been fastest but a gap remains.
5. Sustain immunization and nutrition programmes.

**Analytics**
1. Use the model for screening and benchmarking, not causal claims.
2. Improve source data quality (large gaps, unusable BMI values, isolated glitches).
3. Retrain regularly, since the data ends in 2015.

## Project Structure

```
.
├── Life_Expectancy_Analysis.ipynb   # Full analysis notebook (with outputs)
├── Life_Expectancy_Data.csv         # Raw dataset (add this file)
├── life_expectancy_cleaned.csv      # Generated by the notebook
├── life_expectancy_model.joblib     # Generated: trained final model
└── README.md
```

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/[your-username]/[your-repo-name].git
cd [your-repo-name]
```

**2. Install dependencies** (Python 3.9+ recommended)
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost joblib jupyter
```

**3. Add the data**

Place `Life_Expectancy_Data.csv` in the project folder, next to the notebook.

**4. Run the notebook**
```bash
jupyter notebook Life_Expectancy_Analysis.ipynb
```
Run all cells from top to bottom. A full run takes a few minutes, mostly the hyperparameter tuning step.

**5. Use the saved model** (optional)
```python
import joblib
model = joblib.load("life_expectancy_model.joblib")
# model.predict(X)  # X must contain the same engineered feature columns used in the notebook
```

## Limitations

- Findings are **associations, not causal effects**. Development level confounds many variables.
- Country-level averages hide inequality within countries, and the data ends in 2015.
- Adult mortality and infant deaths are partly outcome measures of the same health system.
- With only 183 countries, test scores depend on which countries are held out (cross-validation standard deviation of about 0.3-0.4 years).
- Some data glitches (such as one-year spikes in a few countries' series) cannot be caught by simple range checks.

## Future Work

- Add more recent data and external indicators (health-worker density, sanitation, conflict, air quality).
- Try time-series or mixed-effects models that use the panel structure.
- Use SHAP for per-country explanations.
- Produce prediction intervals for decision-making.

## Data Source & Acknowledgements

- World Health Organization, Global Health Observatory (GHO) data repository
- United Nations (economic data)

## License

[Choose a license, e.g. MIT] - see `LICENSE` for details.

## Author

[Your Name] - [LinkedIn / email / portfolio link]
