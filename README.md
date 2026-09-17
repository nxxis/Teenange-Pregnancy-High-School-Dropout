# Teen Birth Rate & High-School Dropout Disparities

An observational, population-level data science study examining whether
U.S. states and demographic groups with higher adolescent birth/fertility
rates also show higher high-school dropout rates, and how that
relationship varies by race/ethnicity, geography, and socioeconomic
context.

> **This is an observational study.** Results describe associations and
> predictive relationships across states and demographic groups. They do
> not establish that teen pregnancy causes students to leave school —
> aggregate patterns can reflect shared underlying conditions (poverty,
> school context, health access, policy) rather than a direct effect on
> any individual.

## Research questions

1. What are the trends in adolescent fertility/birth rates across states
   and demographic groups?
2. What are the trends and disparities in high-school status dropout
   rates across states and racial/ethnic groups?
3. Is adolescent fertility/birth rate statistically associated with
   high-school dropout rate at the state-demographic level?
4. Does the strength or direction of that association differ across
   racial/ethnic groups or geographic regions?
5. Do socioeconomic variables (poverty, health insurance) help explain
   differences in dropout rates?
6. Can dropout rates be predicted from adolescent fertility and
   contextual socioeconomic variables?
7. Which features contribute most to model predictions, and are
   prediction errors materially different across demographic groups?

## Data sources

| Source | Measure | Coverage |
|---|---|---|
| CDC WONDER Natality (2016-2024 Expanded) | Teen fertility rate, births, female population 15-44, by state and race, mothers aged 15-19 | Aggregated to 2017-2021 |
| NCES Digest of Education Statistics, Table 219.85b | High-school **status** dropout rate (ages 16-24, not enrolled and without a diploma), by state and race/ethnicity | 2017-2021 5-year average |
| U.S. Census Bureau ACS (Tables S1701, S2701) | Child poverty rate, child health-insurance coverage, by state | 2017-2021 5-year estimates |

All three sources cover the same 2017-2021 window: NCES publishes its
status-dropout table as a 2017-2021 average, and the CDC WONDER annual
panel is aggregated to match it (summing births and population across the
five years, then computing one rate).

## Scope

The analysis covers all race/ethnicity categories reported by the source
data, with **White, Black, and Hispanic** as the primary comparison groups
in the analysis and writeup. Two boundaries define what the dataset can
and cannot answer:

- **Stigma is not an operationalized variable.** No available source
  provides a usable state-level measure of pregnancy-related stigma, so it
  is discussed conceptually rather than measured directly.
- **The fertility-association analysis (RQ3 onward) covers non-Hispanic
  race categories only.** CDC WONDER's teen fertility data does not break
  out a Hispanic-specific rate, so Hispanic dropout rates are reported and
  compared (RQ2), but cannot be linked to a fertility value the same way
  as other groups. Separately, Pacific Islander is reported descriptively
  only (usable data in just 4 of 51 states) and excluded from regression
  and modeling.
- **`poverty_rate` and `insurance_rate` are state-level, not
  race-specific.** The ACS source tables (S1701, S2701) provide one
  child-poverty and one child-insurance value per state; the same value is
  merged onto every race/ethnicity row within that state. This is why they
  enter the regression as state-level context controls (RQ5) rather than
  as a per-race comparison, and why `figures/phase1_distributions_by_race.png`
  plots them once across states instead of faceting by race.

## Repository structure

```
├── data/
│   ├── raw/                     # source files, exactly as downloaded
│   ├── interim/
│   └── processed/               # cleaned + merged analytical datasets
├── src/
│   ├── harmonization.py         # FIPS, state abbreviation, and race/ethnicity crosswalks
│   ├── data_cleaning.py         # per-source cleaning + final merge
│   ├── plotting.py              # shared figure style/palette
│   ├── analysis.py              # data quality, disparity, and association analysis
│   └── modeling.py              # machine learning + fairness/explainability
├── figures/, tables/, results/  # generated outputs (run the scripts to populate)
├── docs/
│   └── data_dictionary.xlsx     # every field in the processed data, defined
├── requirements.txt
└── README.md
```

## Running the pipeline

```bash
pip install -r requirements.txt

# 1. Clean each source and build the merged analytical dataset
python3 src/data_cleaning.py
# -> data/processed/master_analytical_dataset.csv
# -> docs/data_dictionary.xlsx

# 2. Data quality, disparity, and association analysis
python3 src/analysis.py
# -> results/phase1_data_quality_report.csv
# -> results/phase3_regression_summary.txt
# -> tables/phase1_summary_statistics.csv, phase2_racial_disparity_gaps.csv,
#    phase2_state_rankings.csv, phase3_correlation.csv, phase3_vif.csv,
#    phase3_diagnostic_tests.csv
# -> figures/phase1_distributions_by_race.png,
#    phase2_disparity_gaps.png, phase2_map_dropout_rate.png,
#    phase2_map_hispanic_white_gap.png, phase2_map_black_white_gap.png,
#    phase2_map_fertility_rate_by_race.png, phase3_fertility_dropout_scatter.png,
#    phase3_regression_diagnostics.png
#    (each phase2_map_*.png has a companion .html with hover tooltips)

# 3. Machine learning + fairness/explainability
python3 src/modeling.py
# -> results/phase4_model_comparison.csv, phase5_fairness_report.csv
# -> figures/phase5_fairness_errors.png, phase5_shap_model_a_no_race.png,
#    phase5_shap_model_b_with_race.png
```

Raw data files are included in `data/raw/` so the pipeline runs
immediately after cloning, with no re-downloading required.

## Findings

From the merged analytical dataset (408 state × race-ethnicity rows; 174
with sufficient data quality on both sides for the core association test):

**Association.** Teen fertility rate and dropout rate are strongly
correlated: Pearson r = 0.74 (p < 10⁻³⁰, n = 174). In a multivariable
regression weighted by estimate reliability and controlling for poverty,
insurance coverage, and Census region, teen fertility rate remains a
significant positive predictor of dropout rate (p < 0.001), as does
poverty rate (p = 0.006); health-insurance coverage is not significant.
The fertility effect holds under standard errors clustered by state
(accounting for the fact that each state contributes multiple
non-independent race-group rows) and is not driven by a handful of
extreme observations — a Cook's-distance sensitivity check flags zero
high-leverage points in the primary sample. Multicollinearity is not a
concern: variance inflation factors for all predictors (fertility rate,
poverty, insurance, Census region) are below 2.3, well under the
conventional threshold of 5 (see `tables/phase3_vif.csv`), so these
coefficients can be interpreted without a collinearity caveat.

**Regression diagnostics** (see `figures/phase3_regression_diagnostics.png`,
`tables/phase3_diagnostic_tests.csv`) surface one real limitation: model
residuals are right-skewed rather than normal (Shapiro-Wilk p < 10⁻¹⁴), and
Breusch-Pagan detects mild residual heteroscedasticity remaining even after
inverse-variance weighting (p = 0.041). This means a handful of state-race
cells have actual dropout rates well above what the model predicts. At
n = 197 the coefficient estimates themselves are not seriously threatened by
this (the same fertility effect holds under clustered SEs and excluding
high-leverage points, above), but the exact p-values and confidence
intervals for the primary WLS model should be read as approximate rather
than precise.

**Disparities.** Racial dropout gaps are substantial: the median
Black-White gap is +2.1 percentage points (a 1.55x ratio); the median
Hispanic-White gap is +4.2 percentage points (a 2.0x ratio). Both vary
considerably by state (see `figures/phase2_map_black_white_gap.png` and
`phase2_map_hispanic_white_gap.png`).

**Fairness.** A random forest trained on fertility, poverty, insurance,
and region — with race excluded entirely from its inputs — still shows
sharply uneven errors across race groups. American Indian/Alaska Native
has a mean absolute error 3-7x larger than every other group, with a
strong systematic under-prediction bias (Kruskal-Wallis test on residual
distributions: p < 0.0001). Adding race back as a feature, `race_ethnicity`
ranks 2nd of 5 features by SHAP importance — behind fertility rate but
ahead of poverty and insurance — meaning race carries predictive weight
not explained by the measured socioeconomic context. This fairness
analysis is necessarily silent on Hispanic students, since no model can
be built for a group with no fertility input at all; that is a scope
limitation of the analysis, not a finding about the group.

## Data quality

Every cleaned dataset preserves *why* a value is missing rather than
collapsing it to zero or a blank:

- `dropout_quality_flag`: `reliable` / `unreliable_cv_30_50pct` (value
  retained, interpret with caution) / `suppressed_reporting_standards_not_met`
  (no reliable value exists — never treated as zero)
- `natality_quality_flag`: `reliable` / `suppressed_or_incomplete_5yr_window`
  (any year in the five-year window was suppressed or missing, so the
  aggregate is marked unknown rather than partially summed)
- Race/ethnicity comparability fields document that CDC WONDER's race
  categories include Hispanic-origin individuals (not excluded), unlike
  NCES's, so cross-source race comparisons are approximate

Full field-by-field documentation: `docs/data_dictionary.xlsx`.

## Modeling approach

Two random forests are trained on the same grouped 5-fold cross-validation
split (grouped by state, since race-rows within a state share identical
poverty/insurance/region values — a random row split would leak
information across train/test):

- **Model A** (fairness evaluation): fertility, poverty, insurance, and
  region only. Used to check whether a model that never sees race still
  produces racially uneven errors.
- **Model B** (residual disparity attribution): Model A's features plus
  race. Used to measure how much predictive weight race carries once the
  measured socioeconomic/fertility context is already accounted for.

Both a mean baseline and a linear model are reported alongside the random
forest for comparison. All models are weighted by inverse-variance
(`1/dropout_se²`), consistent with the Phase 3 regression. A fixed random
seed (42) is used throughout for reproducibility.
