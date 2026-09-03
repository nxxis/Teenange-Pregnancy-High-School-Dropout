"""
Source-specific cleaning for the teen birth / dropout project, plus the
final merge into the design doc's recommended analytical structure
(design_document.pdf section 8).

Study period: 2017-2021. All three sources below
cover this exact window using ACS 5-year estimates (poverty, insurance) or
NCES's own 2017-2021 5-year-average table (dropout), or an aggregate of
CDC WONDER's annual panel summed across 2017-2021 (teen fertility) --
matching design_document.pdf section 7.2 point 4: do not mix annual and
five-year-average estimates without explicitly modeling the difference.
Here, both sides of the merge are genuinely 5-year windows.
"""

import json

import pandas as pd

from harmonization import (
    US_STATES_PLUS_DC,
    WONDER_PLACEHOLDER_CATEGORIES,
    add_census_region,
    add_fips,
    check_duplicates,
    harmonize_nces_race,
    harmonize_wonder_race,
)

STUDY_YEARS = [2017, 2018, 2019, 2020, 2021]
STUDY_PERIOD = "2017-2021"

DROPOUT_XLSX = "data/raw/tabn219.85b.xlsx"
NATALITY_CSV = "data/raw/Natality, 2016-2024 expanded.csv"
ACS_POVERTY_JSON = "data/raw/ACS_S1701_2017-2021_state.json"
ACS_INSURANCE_JSON = "data/raw/ACS_S2701_2017-2021_state.json"

OUTPUT_DROPOUT = "data/processed/dropout_status_2017_2021.csv"
OUTPUT_NATALITY = "data/processed/teen_fertility_2017_2021.csv"
OUTPUT_ACS = "data/processed/acs_covariates_2017_2021.csv"
OUTPUT_MASTER = "data/processed/master_analytical_dataset.csv"


# --- 1. NCES status dropout rate (Table 219.85b) -------------------------

def _parse_dropout_cell(value, flag, se):
    """One (value, flag, se) triple from the raw table -> (rate, se, quality_flag).

    Suppression notation (verified against the file's own footnotes):
      '‡' as the value  -> reporting standards not met (too few cases / CV>=50%)
      '†' as the SE      -> not applicable
      '!' as the flag    -> unreliable but retained (CV 30-50%)
      ' ' as the flag     -> reliable
    """
    value_str = str(value).strip()
    se_str = str(se).strip()

    if value_str == "‡" or "†" in se_str:
        return float("nan"), float("nan"), "suppressed_reporting_standards_not_met"
    if str(flag).strip() == "!":
        return float(value), float(se), "unreliable_cv_30_50pct"
    return float(value), float(se), "reliable"


def load_dropout() -> pd.DataFrame:
    raw = pd.read_excel(DROPOUT_XLSX, sheet_name=0, header=None)

    group_header_row = raw.iloc[2, :]
    race_starts = {
        str(group_header_row[c]).strip(): c
        for c in range(1, len(group_header_row))
        if isinstance(group_header_row[c], str) and group_header_row[c].strip()
    }

    # Data rows: after the two header rows + column-number row, up to
    # "Other jurisdictions" (Puerto Rico) at the bottom of the state list.
    state_col = raw.iloc[:, 0].astype(str).str.strip()
    data_start = 4
    data_end = state_col[state_col == "Other jurisdictions"].index[0]
    data = raw.iloc[data_start:data_end].copy()
    data["state"] = data.iloc[:, 0].astype(str).str.strip()
    data = data[data["state"].isin(US_STATES_PLUS_DC)]

    records = []
    for _, row in data.iterrows():
        for race_label, col in race_starts.items():
            value, flag, se = row[col], row[col + 1], row[col + 2]
            rate, rate_se, quality_flag = _parse_dropout_cell(value, flag, se)
            records.append({
                "state": row["state"],
                "race_ethnicity": race_label,
                "dropout_rate": rate,
                "dropout_se": rate_se,
                "dropout_quality_flag": quality_flag,
            })

    long_df = pd.DataFrame(records)
    long_df = harmonize_nces_race(long_df)
    long_df = long_df.rename(columns={"race_comparable_to_nces": "dropout_race_comparable_to_nces"})
    long_df = add_fips(long_df)
    long_df["dropout_data_source"] = "NCES Digest Table 219.85b"
    long_df["dropout_source_period"] = STUDY_PERIOD

    check_duplicates(long_df, ["fips", "original_category"], "load_dropout")
    return long_df


# --- 2. CDC WONDER teen fertility rate, aggregated 2016-2024 -> 2017-2021 -

def load_natality() -> pd.DataFrame:
    df = pd.read_csv(NATALITY_CSV, on_bad_lines="skip")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    df = df[df["Year"].isin(STUDY_YEARS)]
    df = df[~df["Mother's Single Race 6"].isin(WONDER_PLACEHOLDER_CATEGORIES)]

    df["is_suppressed"] = df["Births"].astype(str) == "Suppressed"
    df["births_num"] = pd.to_numeric(df["Births"], errors="coerce")
    df["pop_num"] = pd.to_numeric(df["Female Population"], errors="coerce")

    records = []
    group_cols = ["State of Residence", "State of Residence Code", "Mother's Single Race 6"]
    for (state, fips_code, race), grp in df.groupby(group_cols):
        n_years_present = grp["Year"].nunique()
        any_suppressed = grp["is_suppressed"].any()
        missing_years = n_years_present < len(STUDY_YEARS)

        if any_suppressed or missing_years:
            births_sum = float("nan")
            pop_sum = float("nan")
            rate = float("nan")
            flag = "suppressed_or_incomplete_5yr_window"
        else:
            births_sum = grp["births_num"].sum()
            pop_sum = grp["pop_num"].sum()
            rate = (births_sum / pop_sum) * 1000 if pop_sum else float("nan")
            flag = "reliable"

        records.append({
            "state": state,
            "race_ethnicity": race,
            "births_teen": births_sum,
            "female_population_15_44": pop_sum,
            "teen_fertility_rate": rate,
            "natality_quality_flag": flag,
        })

    long_df = pd.DataFrame(records)
    long_df = harmonize_wonder_race(long_df)
    long_df = long_df.rename(columns={"race_comparable_to_nces": "natality_race_comparable_to_nces"})
    long_df = add_fips(long_df)
    long_df["natality_data_source"] = "CDC WONDER Natality 2016-2024 Expanded (age 15-19)"
    long_df["natality_source_period"] = STUDY_PERIOD

    check_duplicates(long_df, ["fips", "original_category"], "load_natality")
    return long_df


# --- 3. ACS 5-year 2017-2021 socioeconomic covariates (state-level) ------

def _load_acs_json(path: str, value_col: str, moe_col: str, out_name: str, out_moe_name: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    header, rows = data[0], data[1:]
    # The Census API appends its own "state" column (FIPS code) when queried
    # with for=state:*; rename it out of the way before renaming NAME->state,
    # to avoid two columns named "state".
    header = ["state_fips_api" if h == "state" else h for h in header]
    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={"NAME": "state"})
    df = df[df["state"].isin(US_STATES_PLUS_DC)].copy()
    df[out_name] = pd.to_numeric(df[value_col], errors="coerce")
    df[out_moe_name] = pd.to_numeric(df[moe_col], errors="coerce")
    return df[["state", out_name, out_moe_name]]


def load_acs() -> pd.DataFrame:
    # Column IDs verified against source:
    #   S1701_C03_002E = percent below poverty, under 18 years
    #   S2701_C05_011E = percent UNINSURED, under 19 years
    poverty = _load_acs_json(
        ACS_POVERTY_JSON, "S1701_C03_002E", "S1701_C03_002M",
        "poverty_rate", "poverty_rate_moe",
    )
    insurance = _load_acs_json(
        ACS_INSURANCE_JSON, "S2701_C05_011E", "S2701_C05_011M",
        "insurance_rate", "insurance_rate_moe",
    )
    # `insurance_rate` here is the UNINSURED percent (design doc names the
    # field generically; direction is documented in the data dictionary).

    merged = poverty.merge(insurance, on="state", how="inner")
    merged = add_fips(merged)
    merged["acs_data_source"] = "ACS 5-year 2017-2021 (S1701 age band: under 18; S2701 age band: under 19)"
    merged["acs_source_period"] = STUDY_PERIOD

    check_duplicates(merged, ["fips"], "load_acs")
    return merged


# --- 4. Final merge -------------------------------------------------------

def build_master_dataset() -> pd.DataFrame:
    dropout = load_dropout()
    natality = load_natality()
    acs = load_acs()

    master = dropout.merge(
        natality[[
            "fips", "race_ethnicity", "births_teen", "female_population_15_44",
            "teen_fertility_rate", "natality_quality_flag",
            "natality_race_comparable_to_nces",
            "natality_data_source", "natality_source_period",
        ]],
        on=["fips", "race_ethnicity"], how="left",
        suffixes=("", "_natality"),
    )
    master = master.merge(
        acs[[
            "fips", "poverty_rate", "poverty_rate_moe",
            "insurance_rate", "insurance_rate_moe",
            "acs_data_source", "acs_source_period",
        ]],
        on="fips", how="left",
    )

    master = add_census_region(master)
    master["year_period"] = STUDY_PERIOD
    check_duplicates(master, ["fips", "race_ethnicity"], "build_master_dataset")
    return master


def generate_data_dictionary(master: pd.DataFrame, out_path: str = "docs/data_dictionary.xlsx") -> None:
    """design_document.pdf section 9, point 9: create a data dictionary
    before modeling. Generated from code (not hand-edited), per section 15,
    point 8."""
    definitions = {
        "state": "State name (50 states + DC)",
        "fips": "2-digit state FIPS code, primary geographic join key (design doc sec. 9 pt. 1)",
        "race_ethnicity": "Harmonized race/ethnicity category (see original_category, race_comparable_to_nces)",
        "original_category": "Race/ethnicity label exactly as it appeared in the source table",
        "race_definition": "Precise definition of the source's race category, incl. whether Hispanic ethnicity is included/excluded",
        "dropout_race_comparable_to_nces": "Always True: dropout_rate is always on NCES's own Hispanic-exclusive-race-categories basis (kept for schema completeness/symmetry).",
        "natality_race_comparable_to_nces": "False for every row with a value: WONDER's race categories include Hispanic-origin mothers (not excluded), so teen_fertility_rate is NOT on the same Hispanic-exclusion basis as dropout_rate for the same race_ethnicity label. NaN for 'Hispanic'/'Total' rows, which have no WONDER equivalent at all -- treat cross-source race comparisons as approximate.",
        "dropout_rate": "Status dropout rate (%), ages 16-24, not enrolled + no diploma. NCES Digest Table 219.85b, 2017-2021 5-yr avg.",
        "dropout_se": "Standard error of dropout_rate, as published by NCES. NaN where suppressed.",
        "dropout_quality_flag": "reliable / unreliable_cv_30_50pct (retain value, use with caution) / suppressed_reporting_standards_not_met (value is NaN, not zero)",
        "dropout_data_source": "Source table for dropout_rate",
        "dropout_source_period": "Period the dropout figures cover",
        "births_teen": "Sum of births to mothers aged 15-19, 2017-2021, CDC WONDER. NaN if any year in the window was suppressed or missing for this state/race.",
        "female_population_15_44": "Sum of the CDC fertility-rate denominator population, 2017-2021.",
        "teen_fertility_rate": "births_teen / female_population_15_44 x 1,000, computed from the 5-year sums (not an average of annual rates).",
        "natality_quality_flag": "reliable / suppressed_or_incomplete_5yr_window (any of the 5 years suppressed or missing -> whole aggregate is NaN, not partially summed)",
        "natality_data_source": "Source query for teen_fertility_rate",
        "natality_source_period": "Period the natality figures cover",
        "poverty_rate": "ACS S1701_C03_002E: percent of population under 18 years below poverty level, 2017-2021 5-yr estimate. State-level only (not by race).",
        "poverty_rate_moe": "ACS margin of error for poverty_rate (90% confidence level, per ACS convention)",
        "insurance_rate": "ACS S2701_C05_011E: percent UNINSURED, under 19 years, 2017-2021 5-yr estimate (note: this is the uninsured rate, not the insured rate). State-level only (not by race).",
        "insurance_rate_moe": "ACS margin of error for insurance_rate",
        "acs_data_source": "Source tables for poverty_rate/insurance_rate, including the age-band mismatch between S1701 (under 18) and S2701 (under 19)",
        "acs_source_period": "Period the ACS figures cover",
        "year_period": "Study period for this entire dataset: 2017-2021",
        "census_region": "Standard Census 4-region grouping (Northeast/Midwest/South/West), used in place of state fixed effects for the geographic comparison.",
    }
    rows = []
    for col in master.columns:
        rows.append({
            "field": col,
            "dtype": str(master[col].dtype),
            "description": definitions.get(col, "(undocumented -- add to generate_data_dictionary)"),
        })
    pd.DataFrame(rows).to_excel(out_path, index=False)


if __name__ == "__main__":
    dropout = load_dropout()
    dropout.to_csv(OUTPUT_DROPOUT, index=False)
    print(f"Dropout: {len(dropout)} rows -> {OUTPUT_DROPOUT}")

    natality = load_natality()
    natality.to_csv(OUTPUT_NATALITY, index=False)
    print(f"Natality: {len(natality)} rows -> {OUTPUT_NATALITY}")

    acs = load_acs()
    acs.to_csv(OUTPUT_ACS, index=False)
    print(f"ACS: {len(acs)} rows -> {OUTPUT_ACS}")

    master = build_master_dataset()
    master.to_csv(OUTPUT_MASTER, index=False)
    print(f"Master: {len(master)} rows -> {OUTPUT_MASTER}")

    generate_data_dictionary(master)
    print("Data dictionary -> docs/data_dictionary.xlsx")
