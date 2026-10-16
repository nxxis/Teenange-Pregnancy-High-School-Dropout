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
- **`poverty_rate` and `uninsured_rate` are state-level, not
  race-specific.** The ACS source tables (S1701, S2701) provide one
  child-poverty and one child-uninsured value per state; the same value is
  merged onto every race/ethnicity row within that state. This is why they
  enter the regression as state-level context controls (RQ5) rather than
  as a per-race comparison, and why `figures/phase1_distributions_by_race.png`
  plots them once across states instead of faceting by race.
  `uninsured_rate` is the percent of children under 19 *without* health
  insurance (S2701_C05_011E).

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

Run from the repository root (paths are relative to it).

```bash
pip install -r requirements.txt

# 1. Clean each source and build the merged analytical dataset
python3 src/data_cleaning.py
# -> data/processed/master_analytical_dataset.csv
# -> docs/data_dictionary.xlsx

# 2. Data quality, disparity, and association analysis (~10 s)
python3 src/analysis.py
# -> results/phase1_data_quality_report.csv, phase3_regression_summary.txt
# -> tables/phase1_summary_statistics.csv, phase2_racial_disparity_gaps.csv,
#    phase2_state_rankings.csv, phase3_correlation.csv,
#    phase3_regression_ladder.csv, phase3_robustness.csv,
#    phase3_heterogeneity_slopes.csv, phase3_heterogeneity_tests.csv,
#    phase3_vif.csv, phase3_diagnostic_tests.csv
# -> figures/phase1_distributions_by_race.png, phase2_disparity_gaps.png,
#    phase2_map_dropout_rate.png, phase2_map_hispanic_white_gap.png,
#    phase2_map_black_white_gap.png, phase2_map_fertility_rate_by_race.png,
#    phase3_fertility_dropout_scatter.png, phase3_slopes_by_group.png,
#    phase3_regression_diagnostics.png
#    (each phase2_map_*.png has a companion .html with hover tooltips)

# 3. Machine learning + fairness/explainability (~2 min)
python3 src/modeling.py
# -> results/phase4_model_comparison.csv, phase4_h4_feature_value.csv,
#    phase5_fairness_report.csv, phase5_shap_importance.csv
# -> figures/phase5_fairness_errors.png, phase5_shap_model_a_no_race.png,
#    phase5_shap_model_b_with_race.png
```

Raw data files are included in `data/raw/` so the pipeline runs
immediately after cloning, with no re-downloading required.

## Analytic sample

The merged dataset has 408 state × race/ethnicity rows (51 states × 8
groups). Every association and modeling analysis (RQ3-RQ7) uses one
sample: **197 cells in 51 states** across White, Black, American
Indian/Alaska Native, Asian, and Two or more races, with a reliable
5-year teen fertility aggregate and a non-suppressed NCES dropout
estimate. 27 of those dropout estimates are flagged unreliable (CV
30-50%); they are kept, down-weighted by their larger standard error, and
dropped in a labeled sensitivity check.

## Findings

All regressions are weighted least squares (weight = 1/dropout_se²) with
standard errors clustered by state. Coefficients are percentage points of
status dropout per one additional birth per 1,000 females aged 15-19.

**Association (RQ3, H1).** Teen fertility and status dropout are strongly
correlated across cells (Pearson r = 0.69, Spearman ρ = 0.74, n = 197), and
the association is not only a between-group pattern: it holds *within*
every race group (r = 0.44-0.60, all p < 0.002;
`tables/phase3_correlation.csv`, `figures/phase3_fertility_dropout_scatter.png`).

**Regression ladder (RQ3, RQ5, H3)** — `tables/phase3_regression_ladder.csv`:

| Model | Controls | Fertility coef. (95% CI) |
|---|---|---|
| M1 | none | 0.164 (0.131, 0.196) |
| M2 | + child poverty, child uninsured | 0.154 (0.129, 0.179) |
| M3 | + Census region (primary) | 0.154 (0.129, 0.180) |
| M4 | + race/ethnicity fixed effects | 0.081 (0.032, 0.129) |
| M5 | race FE + state FE (within-state) | 0.061 (0.005, 0.117) |

Socioeconomic context explains little of the association (~6% attenuation
from M1 to M2/M3); poverty is itself a positive predictor once region is
controlled (p = 0.016), the uninsured rate is not. Roughly half of the
pooled association reflects differences in levels between race groups
(M4). The remaining association survives the strictest specification
(M5), which compares race groups *within the same state* and so absorbs
every state-level confounder, measured or not: about 0.6 pp higher
dropout per 10 additional births per 1,000.

**Robustness** — `tables/phase3_robustness.csv`. The M3 coefficient is
stable (0.150-0.195) under unweighted OLS, reliable-only estimates,
including Pacific Islander, dropping the 19 influential cells (Cook's D >
4/n), and random-effects meta-regression weights (1/(se² + τ²)); it also
stays positive and significant with a log-dropout outcome (p < 0.001).
The race-adjusted M4 coefficient is positive in every variant (0.064-0.137, p ≤ 0.003) except the log-outcome specification,
where it is borderline (p = 0.052).

**Heterogeneity (RQ4, H2)** — `tables/phase3_heterogeneity_*.csv`,
`figures/phase3_slopes_by_group.png`. Fertility slopes differ across race
groups (joint Wald F = 2.70, p = 0.041): steepest for American
Indian/Alaska Native (0.18, 95% CI 0.11-0.26), weakest and not
individually significant for Black (0.04, CI -0.01-0.10). Slopes do not
differ detectably across Census regions (F = 1.20, p = 0.32).

**Diagnostics.** VIFs are all < 2.3 (`tables/phase3_vif.csv`). On the
weighted scale, residuals show no remaining heteroscedasticity
(Breusch-Pagan p = 0.58) but are right-skewed (Shapiro-Wilk p < 10⁻¹²),
so inference relies on the cluster-robust SEs and the robustness checks
above (including the log-outcome model) rather than on normality.

**Disparities (RQ2).** The median within-state Black-White status dropout
gap is +2.1 pp (ratio 1.55, 42 states with both estimates); the median
Hispanic-White gap is +4.2 pp (ratio 2.0, 51 states). Both vary widely by
state (`figures/phase2_map_*_gap.png`).

**Prediction (RQ6, H4)** — `results/phase4_*.csv`, 20 × 5-fold
cross-validation grouped by state. The best model is the simplest that
includes race: an unweighted linear model on fertility, poverty,
uninsured rate, region and race (R² = 0.57 ± 0.02), ahead of the random
forests (≤ 0.45). H4 is only partly supported: adding poverty, uninsured
rate and region to fertility improves the random forest in every repeat
(+0.07 R²) but not the linear model (−0.02 to −0.04 R²), i.e. fertility
alone captures most of the linearly available signal.

**Prediction disparities (RQ7, H5)** — `results/phase5_fairness_report.csv`,
`figures/phase5_fairness_errors.png`. Every model that excludes race
systematically under-predicts American Indian/Alaska Native dropout, by
2.7-4.1 pp (95% state-cluster bootstrap CIs exclude 0), with mean absolute errors
2.5-8× larger than for other groups. The size of this bias depends on
training weights — under inverse-variance weighting AI/AN cells carry < 1% of the
total weight — but it appears under both schemes. A single race term in a
linear model removes it (mean residual 0.00 pp, unweighted), meaning the
AI/AN dropout level is a group-level shift that fertility, poverty,
insurance and region do not account for. In the race-aware random forest,
`race_ethnicity` ranks second in mean |SHAP| behind fertility. These are
predictive-disparity findings, not evidence of a causal mechanism. The
analysis is silent on Hispanic students (no fertility input exists); that
is a scope limitation, not a finding.

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
- `female_population_15_19` is WONDER's fertility-rate denominator under
  the 15-19 age filter, so `teen_fertility_rate` is an age-specific rate
  per 1,000 females aged 15-19

Full field-by-field documentation: `docs/data_dictionary.xlsx`.

## Modeling approach

Three feature sets are compared on identical folds: fertility only
(the H4 reference); **Model A** — fertility, poverty, uninsured rate and
region, with race excluded, used to check whether a model that never sees
race still produces racially uneven errors; and **Model B** — Model A plus
race, used to measure how much predictive weight race carries beyond the
measured context. Each is fit as a linear model and a random forest, with
a mean baseline for reference.

Cross-validation assigns whole states to folds (race rows within a state
share every state-level covariate, so a row split would leak), repeated
20 times with different seeded assignments; metrics are reported as mean
± SD across repeats. Every model is trained both with inverse-variance
weights (matching the regressions) and unweighted, because the weighting
determines how much the fit attends to small, noisy groups. Random seed
42 throughout.
