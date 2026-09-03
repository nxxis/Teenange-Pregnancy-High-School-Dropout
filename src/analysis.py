"""
Phase 1-3 analysis (design_document.pdf section 10.1-10.3):

  - RQ2 (dropout trends/disparities by race) includes Hispanic -- NCES
    Table 219.85b reports a Hispanic dropout rate.
  - RQ3 onward (association with teen_fertility_rate) EXCLUDES Hispanic
    and Total -- CDC WONDER produces no fertility value for either.
  - Pacific Islander is reported descriptively only (4 of 51 states have
    usable fertility+dropout data for this group) -- excluded from the
    primary regression sample, included in a labeled sensitivity check.
  - No state fixed effects (infeasible at this N) -- census_region used
    instead for the geographic comparison (RQ4).
  - The dataset is a single 2017-2021 cross-section, not a repeated-period
    panel -- there is no year-over-year trend to plot. This is a real
    scope limitation, stated here rather than glossed over.
  - Multivariable regression is run both unweighted (OLS) and weighted by
    inverse-variance (1/dropout_se^2) -- the weighted version is the one
    that should be trusted for interpretation.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

MASTER_CSV = "data/processed/master_analytical_dataset.csv"

# Groups with a usable CDC WONDER fertility rate at all (excludes Hispanic
# and Total, which never have one).
FERTILITY_ELIGIBLE_RACES = [
    "White", "Black", "American Indian/Alaska Native", "Asian",
    "Pacific Islander", "Two or more races",
]
# The three primary comparison groups for the analysis and writeup.
FOCAL_RACES = ["White", "Black", "Hispanic"]
# Excluded from the primary regression sample for insufficient N;
# still reported descriptively.
SPARSE_RACES = ["Pacific Islander"]


def load_master() -> pd.DataFrame:
    return pd.read_csv(MASTER_CSV)


# --- Phase 1: Data Quality and Descriptive Analysis ----------------------

def phase1_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append(("n_rows_total", len(df), ""))
    rows.append(("n_states", df["fips"].nunique(), "expected 51 (50 states + DC)"))
    rows.append(("n_race_categories", df["race_ethnicity"].nunique(), "expected 8"))

    for flag_col, label in [
        ("dropout_quality_flag", "dropout"),
        ("natality_quality_flag", "natality"),
    ]:
        counts = df[flag_col].value_counts(dropna=False)
        for val, n in counts.items():
            rows.append((f"{label}_flag__{val}", n, f"{n/len(df):.1%} of all rows"))

    both_reliable = (
        (df["dropout_quality_flag"] == "reliable")
        & (df["natality_quality_flag"] == "reliable")
    ).sum()
    rows.append(("n_rows_both_dropout_and_fertility_reliable", both_reliable,
                 f"{both_reliable/len(df):.1%} of all rows -- the real usable N for RQ3"))

    # Unit sanity check (design doc sec. 10.1 pt. 4): dropout_rate is a
    # percent (roughly 0-40), teen_fertility_rate is per 1,000 (roughly
    # 0-400). If these ranges were ever swapped or rescaled upstream, this
    # catches it immediately rather than silently propagating.
    dr = df["dropout_rate"].dropna()
    fr = df["teen_fertility_rate"].dropna()
    rows.append(("dropout_rate_range", f"{dr.min():.2f} to {dr.max():.2f}",
                 "sanity check: should be a plausible percent, not per-1000"))
    rows.append(("teen_fertility_rate_range", f"{fr.min():.2f} to {fr.max():.2f}",
                 "sanity check: should be plausible per-1,000, not a percent"))
    assert dr.max() < 50, "dropout_rate exceeds a plausible percent -- check units"
    assert fr.max() < 500, "teen_fertility_rate exceeds a plausible per-1,000 rate -- check units"

    return pd.DataFrame(rows, columns=["metric", "value", "note"])


def phase1_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["dropout_rate", "teen_fertility_rate", "poverty_rate", "insurance_rate"]
    overall = df[numeric_cols].describe().T
    overall["race_ethnicity"] = "ALL"
    by_race = (
        df.groupby("race_ethnicity")[numeric_cols]
        .describe()
        .stack(level=0, future_stack=True)
        .rename_axis(["race_ethnicity", "variable"])
        .reset_index()
    )
    return by_race


# --- Phase 2: Trend and Disparity Analysis --------------------------------

def phase2_disparity_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """RQ4/RQ2, design doc sec. 10.2 pt. 4: absolute and relative racial
    gaps. Uses dropout_rate only (available for all groups incl. Hispanic),
    computed per state against the White rate in that same state."""
    pivot = df.pivot(index="state", columns="race_ethnicity", values="dropout_rate")
    records = []
    for race in ["Black", "Hispanic"]:
        if race not in pivot.columns:
            continue
        gap_abs = pivot[race] - pivot["White"]
        gap_rel = pivot[race] / pivot["White"]
        for state in pivot.index:
            records.append({
                "state": state,
                "comparison": f"{race} vs White",
                "dropout_rate_gap_pp": gap_abs.get(state),
                "dropout_rate_ratio": gap_rel.get(state),
            })
    return pd.DataFrame(records)


def phase2_state_rankings(df: pd.DataFrame, race: str, metric: str, n: int = 5) -> pd.DataFrame:
    """Highest/lowest n states for a given race and metric, excluding
    non-reliable rows so the ranking isn't driven by suppressed/unreliable
    estimates (design doc sec. 10.2 pt. 3)."""
    quality_col = "dropout_quality_flag" if metric == "dropout_rate" else "natality_quality_flag"
    sub = df[(df["race_ethnicity"] == race) & (df[quality_col] == "reliable")]
    sub = sub[["state", metric]].dropna().sort_values(metric)
    return pd.concat([
        sub.head(n).assign(rank="lowest"),
        sub.tail(n).assign(rank="highest"),
    ])


# --- Phase 3: Association Analysis ----------------------------------------

def phase3_correlation(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        df["race_ethnicity"].isin(FERTILITY_ELIGIBLE_RACES)
        & (df["dropout_quality_flag"] == "reliable")
        & (df["natality_quality_flag"] == "reliable")
    ][["dropout_rate", "teen_fertility_rate"]].dropna()

    pearson_r, pearson_p = stats.pearsonr(sub["dropout_rate"], sub["teen_fertility_rate"])
    spearman_r, spearman_p = stats.spearmanr(sub["dropout_rate"], sub["teen_fertility_rate"])

    return pd.DataFrame([
        {"method": "Pearson", "r": pearson_r, "p_value": pearson_p, "n": len(sub)},
        {"method": "Spearman", "r": spearman_r, "p_value": spearman_p, "n": len(sub)},
    ])


def _regression_sample(df: pd.DataFrame, include_sparse: bool) -> pd.DataFrame:
    races = FERTILITY_ELIGIBLE_RACES if include_sparse else [
        r for r in FERTILITY_ELIGIBLE_RACES if r not in SPARSE_RACES
    ]
    cols = ["dropout_rate", "dropout_se", "teen_fertility_rate", "poverty_rate",
            "insurance_rate", "census_region", "race_ethnicity"]
    sub = df[df["race_ethnicity"].isin(races)][cols].dropna()
    return sub


def phase3_regressions(df: pd.DataFrame) -> dict[str, object]:
    """Bivariate + multivariable regression, unweighted (OLS) and
    inverse-variance weighted (WLS, weight = 1/dropout_se^2) -- weighting
    downweights small/noisy state-race cells instead of treating a
    CV-50% estimate as equally trustworthy as a solid one. Run both with
    and without Pacific Islander (SPARSE_RACES) as the sensitivity check
    design doc sec. 10.3 pt. 7 asks for."""
    results = {}
    for label, include_sparse in [("primary", False), ("with_sparse_race", True)]:
        sub = _regression_sample(df, include_sparse)
        sub["weight"] = 1.0 / (sub["dropout_se"] ** 2)

        bivariate_ols = smf.ols("dropout_rate ~ teen_fertility_rate", data=sub).fit()
        multivariable_ols = smf.ols(
            "dropout_rate ~ teen_fertility_rate + poverty_rate + insurance_rate + C(census_region)",
            data=sub,
        ).fit()
        multivariable_wls = smf.wls(
            "dropout_rate ~ teen_fertility_rate + poverty_rate + insurance_rate + C(census_region)",
            data=sub, weights=sub["weight"],
        ).fit()

        results[label] = {
            "n": len(sub),
            "bivariate_ols": bivariate_ols,
            "multivariable_ols": multivariable_ols,
            "multivariable_wls": multivariable_wls,
        }
    return results


def _format_regression_summary(results: dict) -> str:
    lines = []
    for label, r in results.items():
        lines.append(f"\n{'='*70}\nSample: {label}  (n={r['n']})\n{'='*70}")
        lines.append("\n--- Bivariate OLS: dropout_rate ~ teen_fertility_rate ---")
        lines.append(str(r["bivariate_ols"].summary()))
        lines.append("\n--- Multivariable OLS (unweighted) ---")
        lines.append(str(r["multivariable_ols"].summary()))
        lines.append("\n--- Multivariable WLS (weighted by 1/dropout_se^2) -- TRUST THIS ONE ---")
        lines.append(str(r["multivariable_wls"].summary()))
    return "\n".join(lines)


if __name__ == "__main__":
    df = load_master()

    quality_report = phase1_data_quality(df)
    quality_report.to_csv("results/phase1_data_quality_report.csv", index=False)
    print(quality_report.to_string(index=False))

    summary_stats = phase1_summary_stats(df)
    summary_stats.to_csv("tables/phase1_summary_statistics.csv", index=False)

    gaps = phase2_disparity_gaps(df)
    gaps.to_csv("tables/phase2_racial_disparity_gaps.csv", index=False)

    rankings = pd.concat([
        phase2_state_rankings(df, "White", "dropout_rate"),
        phase2_state_rankings(df, "Black", "dropout_rate"),
        phase2_state_rankings(df, "Hispanic", "dropout_rate"),
    ])
    rankings.to_csv("tables/phase2_state_rankings.csv", index=False)

    correlation = phase3_correlation(df)
    correlation.to_csv("tables/phase3_correlation.csv", index=False)
    print("\n" + correlation.to_string(index=False))

    regressions = phase3_regressions(df)
    with open("results/phase3_regression_summary.txt", "w") as f:
        f.write(_format_regression_summary(regressions))
    print(f"\nRegression n (primary, Pacific Islander excluded): {regressions['primary']['n']}")
    print(f"Regression n (with sparse race included): {regressions['with_sparse_race']['n']}")
    print("\nFull output -> results/phase3_regression_summary.txt")
    print("Tables -> tables/phase1_summary_statistics.csv, tables/phase2_racial_disparity_gaps.csv,")
    print("          tables/phase2_state_rankings.csv, tables/phase3_correlation.csv")
