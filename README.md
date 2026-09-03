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
  compared (RQ2), but cannot be linked to a fertility value in the same
  way as other groups.

## Repository structure

```
├── data/
│   ├── raw/                     # source files, exactly as downloaded
│   ├── interim/
│   └── processed/               # cleaned + merged analytical datasets
├── src/
│   ├── harmonization.py         # FIPS + race/ethnicity crosswalks
│   ├── data_cleaning.py         # per-source cleaning + final merge
│   ├── analysis.py              # data quality, disparity, and association analysis
│   └── modeling.py              # machine learning + explainability
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

# 2. Data quality report, disparity gaps, correlation, regression
python3 src/analysis.py
# -> results/phase1_data_quality_report.csv
# -> results/phase3_regression_summary.txt
# -> tables/phase1_summary_statistics.csv
# -> tables/phase2_racial_disparity_gaps.csv
# -> tables/phase2_state_rankings.csv
# -> tables/phase3_correlation.csv
```

Raw data files are included in `data/raw/` so the pipeline runs
immediately after cloning, with no re-downloading required.

## Findings

From the merged analytical dataset (408 state × race-ethnicity rows; 174
with sufficient data quality on both sides for the core association test):

- **Teen fertility rate and dropout rate are strongly correlated**:
  Pearson r = 0.74 (p < 10⁻³⁰, n = 174).
- In a multivariable regression weighted by estimate reliability and
  controlling for poverty, insurance coverage, and Census region, teen
  fertility rate remains a significant positive predictor of dropout rate
  (p < 0.001), as does poverty rate (p = 0.006). Health-insurance coverage
  is not a significant predictor in this model.
- **Racial dropout disparities are substantial**: the median Black-White
  gap is +2.1 percentage points (a 1.55x ratio); the median Hispanic-White
  gap is +4.2 percentage points (a 2.0x ratio).

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
