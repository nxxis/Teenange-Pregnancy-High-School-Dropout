"""
Phase 1-3 analysis (design_document.pdf section 10.1-10.3):

  - One analytic sample (association_sample) feeds every Phase 3 table,
    figure, and the Phase 4-5 models; variants are labeled sensitivity checks.

  - RQ2 (dropout trends/disparities by race) includes Hispanic -- NCES
    Table 219.85b reports a Hispanic dropout rate.
  - RQ3 onward (association with teen_fertility_rate) EXCLUDES Hispanic
    and Total -- CDC WONDER produces no fertility value for either.
  - Pacific Islander is reported descriptively only (4 of 51 states have
    usable fertility+dropout data for this group) -- excluded from the
    primary regression sample, included in a labeled sensitivity check.
  - census_region carries the geographic comparison (RQ4). State fixed
    effects can't sit alongside the state-level covariates (collinear), but
    are used in one ladder model (M5) to estimate the association within
    states, across race groups -- see decisions_log.md #11.
  - The dataset is a single 2017-2021 cross-section, not a repeated-period
    panel -- there is no year-over-year trend to plot. This is a real
    scope limitation, stated here rather than glossed over.
  - Regressions are WLS weighted by 1/dropout_se^2 with state-clustered
    SEs; unweighted OLS and random-effects meta-regression weights are
    reported as robustness checks.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from patsy import dmatrix
from scipy import optimize, stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

from harmonization import add_state_abbrev
from plotting import RACE_PALETTE, set_style

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
                 f"{both_reliable/len(df):.1%} of all rows (all races, incl. Pacific Islander)"))
    primary = association_sample(df)
    rows.append(("n_primary_association_sample", len(primary),
                 f"{primary['fips'].nunique()} states; races {', '.join(PRIMARY_RACES)}; "
                 f"{(primary['dropout_quality_flag'] != 'reliable').sum()} unreliable-but-retained "
                 "dropout cells (down-weighted) -- the N for every RQ3+ analysis"))

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
    """Summary statistics by race/ethnicity plus an "ALL" row per variable.

    The ALL rows respect each variable's unit of observation: dropout_rate
    and teen_fertility_rate pool the race-specific rows (Total excluded, so
    it isn't double-counted against its own components), while poverty_rate
    and uninsured_rate are state-level and are summarized once over the 51
    states rather than once per repeated race row."""
    race_cols = ["dropout_rate", "teen_fertility_rate"]
    state_cols = ["poverty_rate", "uninsured_rate"]
    by_race = (
        df.groupby("race_ethnicity")[race_cols + state_cols]
        .describe()
        .stack(level=0, future_stack=True)
        .rename_axis(["race_ethnicity", "variable"])
        .reset_index()
    )
    by_race = by_race[~by_race["variable"].isin(state_cols)]

    non_total = df[df["race_ethnicity"] != "Total"]
    overall_race = non_total[race_cols].describe().T
    overall_state = df.drop_duplicates("fips")[state_cols].describe().T
    overall = pd.concat([overall_race, overall_state]).rename_axis("variable").reset_index()
    overall.insert(0, "race_ethnicity", "ALL")
    return pd.concat([overall, by_race], ignore_index=True)


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
#
# Every Phase 3 table, figure, and the Phase 4-5 models draw on ONE analytic
# sample, built by association_sample(): fertility-eligible races minus
# SPARSE_RACES, any row with a non-suppressed dropout_rate (reliable or
# unreliable -- unreliable rows are retained and down-weighted, matching
# NCES's own guidance to report them with a caution flag) and a reliable
# 5-year fertility aggregate. The reliable-only and with-Pacific-Islander
# variants are run as labeled sensitivity checks, never as a silently
# different "main" sample.

PRIMARY_RACES = [r for r in FERTILITY_ELIGIBLE_RACES if r not in SPARSE_RACES]

COVARIATES = "poverty_rate + uninsured_rate"
MULTIVARIABLE_FORMULA = f"dropout_rate ~ teen_fertility_rate + {COVARIATES} + C(census_region)"


def association_sample(df: pd.DataFrame, include_sparse: bool = False,
                       reliable_only: bool = False) -> pd.DataFrame:
    races = FERTILITY_ELIGIBLE_RACES if include_sparse else PRIMARY_RACES
    cols = ["fips", "state", "race_ethnicity", "dropout_rate", "dropout_se",
            "dropout_quality_flag", "teen_fertility_rate", "poverty_rate",
            "uninsured_rate", "census_region"]
    sub = df[df["race_ethnicity"].isin(races)][cols].dropna()
    if reliable_only:
        sub = sub[sub["dropout_quality_flag"] == "reliable"]
    sub = sub.reset_index(drop=True)
    sub["weight"] = 1.0 / (sub["dropout_se"] ** 2)
    return sub


def phase3_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson + Spearman correlations (design doc sec. 10.3 pts. 1-2), on
    the primary sample, the reliable-only sensitivity sample, and within
    each race group separately. The within-race rows matter: a pooled
    correlation across race groups partly reflects between-group
    differences (e.g. AI/AN high on both measures, Asian low on both), so
    the paper needs to show the association also holds *within* groups."""
    rows = []

    def _corr(sub, sample, group):
        pr, pp = stats.pearsonr(sub["teen_fertility_rate"], sub["dropout_rate"])
        sr, sp = stats.spearmanr(sub["teen_fertility_rate"], sub["dropout_rate"])
        rows.append({"sample": sample, "group": group, "n": len(sub),
                     "pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp})

    primary = association_sample(df)
    _corr(primary, "primary", "pooled")
    _corr(association_sample(df, reliable_only=True), "reliable_only", "pooled")
    for race in PRIMARY_RACES:
        _corr(primary[primary["race_ethnicity"] == race], "primary", race)
    return pd.DataFrame(rows)


def _fit_wls(formula: str, sub: pd.DataFrame, weights=None):
    """WLS with standard errors clustered by state (design doc sec. 10.3 pt.
    6). Clustering is used for every reported model, not just one variant:
    each state contributes up to 5 race rows sharing the same state-level
    covariates, so row-independent SEs would overstate precision."""
    w = sub["weight"] if weights is None else weights
    return smf.wls(formula, data=sub, weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": sub["fips"]}
    )


def whitened_influence(model) -> OLSInfluence:
    """Influence statistics for a WLS fit. statsmodels' OLSInfluence applied
    directly to a WLS result uses the raw residuals and the unweighted
    design matrix but the *weighted* residual variance, so its Cook's
    distances and studentized residuals are on mismatched scales (this
    previously made every Cook's distance look tiny). Refitting the
    equivalent OLS on the whitened data (sqrt(w)*y on sqrt(w)*X) gives the
    correct WLS influence measures."""
    return OLSInfluence(sm.OLS(model.model.wendog, model.model.wexog).fit())


def random_effects_tau2(y: np.ndarray, X: np.ndarray, se: np.ndarray) -> float:
    """REML estimate of the between-cell variance tau^2 in a random-effects
    meta-regression y_i ~ N(X_i b, se_i^2 + tau^2). Each state-race dropout
    rate is an estimate with a known sampling SE; tau^2 is the residual
    heterogeneity the covariates don't explain. Weighting by
    1/(se^2 + tau^2) instead of 1/se^2 stops a few very precisely measured
    cells (large White populations) from dominating the fit."""
    def neg_reml(log_tau2):
        v = se ** 2 + np.exp(log_tau2)
        w = 1.0 / v
        xtwx = X.T @ (X * w[:, None])
        b = np.linalg.solve(xtwx, X.T @ (w * y))
        r = y - X @ b
        _, logdet = np.linalg.slogdet(xtwx)
        return 0.5 * (np.sum(np.log(v)) + logdet + np.sum(w * r ** 2))

    res = optimize.minimize_scalar(neg_reml, bounds=(-12, 6), method="bounded")
    return float(np.exp(res.x))


def _coef_row(model, label: str, spec: str, term: str = "teen_fertility_rate") -> dict:
    lo, hi = model.conf_int().loc[term]
    return {
        "model": label, "specification": spec, "n": int(model.nobs),
        "fertility_coef": model.params[term], "cluster_se": model.bse[term],
        "ci_low": lo, "ci_high": hi, "p_value": model.pvalues[term],
        "r_squared": model.rsquared,
    }


def phase3_regression_ladder(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The paper's main association table (RQ3, RQ5/H3): the fertility
    coefficient as controls are added step by step. All WLS (1/dropout_se^2),
    SEs clustered by state. Coefficient = percentage-point change in status
    dropout rate per 1 additional birth per 1,000 females aged 15-19.

      M1  fertility only                         (bivariate, RQ3)
      M2  + child poverty, child uninsured rate  (RQ5 / H3: attenuation)
      M3  + Census region                        (primary multivariable)
      M4  + race/ethnicity fixed effects         (association net of
                                                  race-group level differences)
      M5  race FE + state FE                     (within-state: compares race
                                                  groups in the SAME state, so
                                                  every state-level confounder,
                                                  measured or not, is absorbed;
                                                  state-level covariates drop
                                                  out because they're collinear
                                                  with state FE)
    """
    sub = association_sample(df)
    specs = [
        ("M1", "fertility only", "dropout_rate ~ teen_fertility_rate"),
        ("M2", "+ poverty, uninsured", f"dropout_rate ~ teen_fertility_rate + {COVARIATES}"),
        ("M3", "+ census region (primary)", MULTIVARIABLE_FORMULA),
        ("M4", "+ race/ethnicity FE", MULTIVARIABLE_FORMULA + " + C(race_ethnicity)"),
        ("M5", "race FE + state FE (within-state)",
         "dropout_rate ~ teen_fertility_rate + C(race_ethnicity) + C(fips)"),
    ]
    models, rows = {}, []
    for label, spec, formula in specs:
        m = _fit_wls(formula, sub)
        models[label] = m
        rows.append(_coef_row(m, label, spec))
    table = pd.DataFrame(rows)
    table["n_states"] = sub["fips"].nunique()
    m1 = table.loc[table["model"] == "M1", "fertility_coef"].iloc[0]
    table["pct_attenuation_vs_M1"] = 100 * (1 - table["fertility_coef"] / m1)
    return table, models


def phase3_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity of the fertility coefficient (design doc sec. 10.3 pt. 7)
    for the primary (M3) and race-adjusted (M4) specifications."""
    primary = association_sample(df)
    rows = []
    for label, formula in [("M3", MULTIVARIABLE_FORMULA),
                           ("M4", MULTIVARIABLE_FORMULA + " + C(race_ethnicity)")]:
        base = _fit_wls(formula, primary)
        rows.append(_coef_row(base, label, "primary: WLS 1/se^2, clustered SE"))

        ols = smf.ols(formula, data=primary).fit(cov_type="cluster", cov_kwds={"groups": primary["fips"]})
        rows.append(_coef_row(ols, label, "unweighted OLS"))

        rel = association_sample(df, reliable_only=True)
        rows.append(_coef_row(_fit_wls(formula, rel), label, "reliable dropout estimates only"))

        sparse = association_sample(df, include_sparse=True)
        rows.append(_coef_row(_fit_wls(formula, sparse), label, "incl. Pacific Islander"))

        cooks = whitened_influence(smf.wls(formula, data=primary, weights=primary["weight"]).fit()).cooks_distance[0]
        keep = primary[cooks <= 4 / len(primary)]
        r = _coef_row(_fit_wls(formula, keep), label,
                      f"excl. {len(primary) - len(keep)} influential cells (Cook's D > 4/n)")
        rows.append(r)

        y = primary["dropout_rate"].values
        X = smf.wls(formula, data=primary, weights=primary["weight"]).fit().model.exog
        tau2 = random_effects_tau2(y, X, primary["dropout_se"].values)
        re_w = 1.0 / (primary["dropout_se"] ** 2 + tau2)
        rows.append(_coef_row(_fit_wls(formula, primary, weights=re_w), label,
                              f"random-effects meta-regression weights 1/(se^2+tau^2), tau^2={tau2:.2f}"))

        log_m = _fit_wls(formula.replace("dropout_rate ~", "np.log(dropout_rate) ~"), primary)
        rows.append(_coef_row(log_m, label, "log(dropout_rate) outcome (coef ~ proportional change)"))
    return pd.DataFrame(rows)


def _slope_table(model, group_col: str, groups: list[str]) -> pd.DataFrame:
    rows = []
    for g in groups:
        term = f"C({group_col})[{g}]:teen_fertility_rate"
        lo, hi = model.conf_int().loc[term]
        rows.append({"group_variable": group_col, "group": g, "slope": model.params[term],
                     "cluster_se": model.bse[term], "ci_low": lo, "ci_high": hi,
                     "p_value": model.pvalues[term]})
    return pd.DataFrame(rows)


def _equal_slopes_test(model, group_col: str, groups: list[str]) -> dict:
    """Joint Wald test that every group's fertility slope equals the first
    group's (H0: no heterogeneity in the association across groups)."""
    names = list(model.params.index)
    terms = [f"C({group_col})[{g}]:teen_fertility_rate" for g in groups]
    R = np.zeros((len(terms) - 1, len(names)))
    for i, t in enumerate(terms[1:]):
        R[i, names.index(t)] = 1
        R[i, names.index(terms[0])] = -1
    w = model.wald_test(R, use_f=True, scalar=True)
    return {"group_variable": group_col, "test": "equal fertility slopes (Wald F, clustered)",
            "statistic": float(w.statistic), "df_num": int(w.df_num), "df_denom": float(w.df_denom),
            "p_value": float(w.pvalue)}


def phase3_heterogeneity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RQ4 / H2: does the fertility-dropout association differ by
    race/ethnicity or by region? Fits group-specific fertility slopes
    (each slope read directly off the model) with the other M4 controls,
    then tests equality of slopes jointly."""
    sub = association_sample(df)
    regions = sorted(sub["census_region"].unique())

    race_model = _fit_wls(
        f"dropout_rate ~ C(race_ethnicity) + C(race_ethnicity):teen_fertility_rate + {COVARIATES} + C(census_region)",
        sub,
    )
    region_model = _fit_wls(
        f"dropout_rate ~ C(census_region) + C(census_region):teen_fertility_rate + {COVARIATES} + C(race_ethnicity)",
        sub,
    )
    race_n = sub["race_ethnicity"].value_counts()
    region_n = sub["census_region"].value_counts()

    slopes = pd.concat([
        _slope_table(race_model, "race_ethnicity", PRIMARY_RACES).assign(
            n=lambda d: d["group"].map(race_n)),
        _slope_table(region_model, "census_region", regions).assign(
            n=lambda d: d["group"].map(region_n)),
    ], ignore_index=True)
    tests = pd.DataFrame([
        _equal_slopes_test(race_model, "race_ethnicity", PRIMARY_RACES),
        _equal_slopes_test(region_model, "census_region", regions),
    ])
    return slopes, tests


def phase3_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Multicollinearity check for the primary model's predictors (design
    doc sec. 7.3/11). VIF > 5 flags a predictor whose SE is inflated by
    collinearity; below that, coefficients can be read without a
    collinearity caveat."""
    sub = association_sample(df)
    design = dmatrix(
        f"teen_fertility_rate + {COVARIATES} + C(census_region)",
        data=sub, return_type="dataframe",
    )
    vifs = pd.DataFrame({
        "variable": design.columns,
        "vif": [variance_inflation_factor(design.values, i) for i in range(design.shape[1])],
    })
    return vifs[vifs["variable"] != "Intercept"].reset_index(drop=True)


def _fit_primary_wls(df: pd.DataFrame):
    """The primary (M3) WLS model without clustering, for diagnostics --
    residual diagnostics don't depend on the SE estimator."""
    sub = association_sample(df)
    model = smf.wls(MULTIVARIABLE_FORMULA, data=sub, weights=sub["weight"]).fit()
    return model, sub


def phase3_diagnostic_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Formal residual diagnostics for the primary WLS model, all on the
    *weighted* (whitened) scale -- the scale the model's assumptions are
    stated on. Testing raw residuals of a WLS fit for heteroscedasticity
    just rediscovers the unequal sampling variance the weights already
    model (raw-residual Breusch-Pagan p = 0.04 vs. weighted p ~ 0.6)."""
    model, _ = _fit_primary_wls(df)
    resid_std = whitened_influence(model).resid_studentized_internal

    shapiro_stat, shapiro_p = stats.shapiro(resid_std)
    bp_stat, bp_p, _, _ = het_breuschpagan(model.wresid, model.model.exog)

    return pd.DataFrame([
        {"test": "Shapiro-Wilk (normality of weighted studentized residuals)",
         "statistic": shapiro_stat, "p_value": shapiro_p},
        {"test": "Breusch-Pagan (homoscedasticity of weighted residuals)",
         "statistic": bp_stat, "p_value": bp_p},
    ])


# --- Figures ---------------------------------------------------------------

def figure_distributions_by_race(
    df: pd.DataFrame,
    out_path: str = "figures/phase1_distributions_by_race.png",
):
    """Distributions of the four core analytical variables (design doc sec.
    10.1: distributions, extreme values) -- phase1_summary_stats already
    reports these as numbers; this is the same information as a picture,
    which is what actually shows skew, spread, and outliers at a glance.

    dropout_rate and teen_fertility_rate are genuinely race-specific, so
    they're shown as one box per race. poverty_rate and uninsured_rate are
    NOT race-specific: they're state-level ACS estimates, merged onto every
    race row within a state (verified -- e.g. every California row has the
    identical poverty_rate regardless of race_ethnicity). Faceting those by
    race would draw 7 visually distinct boxes that are mathematically
    identical by construction, which reads as a (false) finding of no
    racial gap in poverty. They're plotted once instead, over the 51
    state-level values.

    Total is excluded from the by-race panels -- it's the all-races
    aggregate row (used standalone for the dropout-rate map), not a
    comparison group. A race with zero non-null values for a given
    variable (e.g. Hispanic and teen_fertility_rate -- CDC WONDER has no
    Hispanic fertility rate at all) is simply omitted from that panel
    rather than plotted as an empty box."""
    set_style()
    race_order = [
        "White", "Black", "Hispanic", "American Indian/Alaska Native",
        "Asian", "Two or more races", "Pacific Islander",
    ]
    by_race_vars = [
        ("dropout_rate", "Dropout rate (%)"),
        ("teen_fertility_rate", "Teen fertility rate (per 1,000)"),
    ]
    state_level_vars = [
        ("poverty_rate", "Child poverty rate (%)"),
        ("uninsured_rate", "Child uninsured rate (%, under 19)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    for ax, (col, label) in zip(axes[0], by_race_vars):
        present = [r for r in race_order if df.loc[df["race_ethnicity"] == r, col].notna().any()]
        data = [df.loc[df["race_ethnicity"] == r, col].dropna().values for r in present]
        colors = [RACE_PALETTE.get(r, "#333333") for r in present]

        bp = ax.boxplot(
            data, patch_artist=True, widths=0.6, showfliers=True,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markersize=3, alpha=0.5, markeredgewidth=0),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels(present, rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"{label}, by race/ethnicity")

    for ax, (col, label) in zip(axes[1], state_level_vars):
        values = df.drop_duplicates("state")[col].dropna().values
        bp = ax.boxplot(
            [values], patch_artist=True, widths=0.4, vert=False, showfliers=True,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markersize=3, alpha=0.5, markeredgewidth=0),
        )
        bp["boxes"][0].set_facecolor("#999999")
        bp["boxes"][0].set_alpha(0.75)
        ax.set_yticks([])
        ax.set_xlabel(label)
        ax.set_title(f"{label}, across states (n={len(values)})\n"
                     "same value for every race within a state -- not race-specific",
                     fontsize=10)

    fig.suptitle("Distribution of core variables (Total excluded from by-race panels)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def figure_regression_diagnostics(df: pd.DataFrame, out_path: str = "figures/phase3_regression_diagnostics.png"):
    """Residual diagnostics for the primary multivariable WLS model (design
    doc sec. 10.3: regression diagnostics) -- checks the assumptions the
    Findings section's regression results depend on, rather than just
    reporting p-values and trusting them. Studentized residuals throughout,
    since the model itself is weighted (1/dropout_se^2).

    Left panel (residuals vs. fitted): checks linearity and
    homoscedasticity -- a random scatter around zero with no funnel shape
    or curve is what "the model's assumptions are reasonable" looks like.
    Right panel (Normal Q-Q): checks whether residuals are approximately
    normal -- points following the diagonal support the CIs/p-values
    reported for this model; systematic curvature would not.

    See phase3_diagnostic_tests for the formal Shapiro-Wilk and
    Breusch-Pagan tests backing up what these plots show visually."""
    set_style()
    model, _ = _fit_primary_wls(df)
    influence = whitened_influence(model)
    fitted = model.fittedvalues
    resid_std = influence.resid_studentized_internal

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.scatter(fitted, resid_std, alpha=0.65, s=35, color="#4C72B0",
               edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="#999999", linewidth=0.9, linestyle=":")
    smoothed = lowess(resid_std, fitted, frac=0.6)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", linewidth=1.5)
    ax.set_xlabel("Fitted values (predicted dropout rate, %)")
    ax.set_ylabel("Studentized residuals")
    ax.set_title("Residuals vs. fitted")

    ax = axes[1]
    stats.probplot(resid_std, dist="norm", plot=ax)
    ax.get_lines()[0].set(markerfacecolor="#4C72B0", markeredgecolor="white",
                            markersize=5.5, alpha=0.85)
    ax.get_lines()[1].set_color("black")
    ax.set_title("Normal Q-Q")

    fig.suptitle("Regression diagnostics: primary multivariable WLS model",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_path)
    plt.close(fig)


def figure_correlation_scatter(df: pd.DataFrame, out_path: str = "figures/phase3_fertility_dropout_scatter.png"):
    """The paper's likely Figure 1: the core RQ3 relationship on the primary
    analytic sample (same rows as the regressions). Filled markers are
    reliable NCES estimates, hollow markers are retained-but-unreliable ones
    (CV 30-50%). Thin lines are within-race fits, the dashed line the
    pooled fit -- so the reader sees the association holds within groups,
    not only across them."""
    set_style()
    sub = association_sample(df)
    r, p = stats.pearsonr(sub["dropout_rate"], sub["teen_fertility_rate"])
    slope, intercept = np.polyfit(sub["teen_fertility_rate"], sub["dropout_rate"], 1)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for race in PRIMARY_RACES:
        grp = sub[sub["race_ethnicity"] == race]
        color = RACE_PALETTE.get(race, "#333333")
        reliable = grp["dropout_quality_flag"] == "reliable"
        ax.scatter(grp.loc[reliable, "teen_fertility_rate"], grp.loc[reliable, "dropout_rate"],
                   label=race, color=color, alpha=0.8, s=40, edgecolor="white", linewidth=0.5)
        ax.scatter(grp.loc[~reliable, "teen_fertility_rate"], grp.loc[~reliable, "dropout_rate"],
                   facecolor="none", edgecolor=color, alpha=0.8, s=40, linewidth=1.0)
        g_slope, g_int = np.polyfit(grp["teen_fertility_rate"], grp["dropout_rate"], 1)
        xs = np.linspace(grp["teen_fertility_rate"].min(), grp["teen_fertility_rate"].max(), 50)
        ax.plot(xs, g_slope * xs + g_int, color=color, linewidth=1.1, alpha=0.9)
    x_line = np.linspace(sub["teen_fertility_rate"].min(), sub["teen_fertility_rate"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5, linestyle="--", zorder=1)

    ax.set_xlabel("Teen fertility rate (births per 1,000 females aged 15-19)")
    ax.set_ylabel("High-school status dropout rate, ages 16-24 (%)")
    ax.set_title("Teen fertility and status dropout, by state and race/ethnicity")
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(
        0.03, 0.97, f"Pooled Pearson r = {r:.2f}\n{p_text}\nn = {len(sub)} cells, {sub['fips'].nunique()} states\n"
                    "hollow = unreliable (CV 30-50%)",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax.legend(loc="lower right", ncol=1, markerscale=1.2)
    fig.savefig(out_path)
    plt.close(fig)


def figure_heterogeneity_slopes(slopes: pd.DataFrame, tests: pd.DataFrame,
                                out_path: str = "figures/phase3_slopes_by_group.png"):
    """RQ4 coefficient plot: group-specific fertility slopes with 95% CIs
    (state-clustered), race panel and region panel, each annotated with
    the joint equal-slopes test."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for ax, group_col, title in [(axes[0], "race_ethnicity", "By race/ethnicity"),
                                 (axes[1], "census_region", "By Census region")]:
        d = slopes[slopes["group_variable"] == group_col].iloc[::-1]
        colors = [RACE_PALETTE.get(g, "#555555") for g in d["group"]]
        y = np.arange(len(d))
        ax.errorbar(d["slope"], y, xerr=[d["slope"] - d["ci_low"], d["ci_high"] - d["slope"]],
                    fmt="none", ecolor="#555555", elinewidth=1.2, capsize=3)
        ax.scatter(d["slope"], y, color=colors, s=45, zorder=3, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{g} (n={n})" for g, n in zip(d["group"], d["n"])])
        ax.axvline(0, color="#999999", linewidth=0.8, linestyle=":")
        t = tests[tests["group_variable"] == group_col].iloc[0]
        ax.set_title(f"{title}\nequal-slopes test: F={t['statistic']:.2f}, p={t['p_value']:.3f}", fontsize=11)
        ax.set_xlabel("Dropout pp per +1 birth per 1,000 (95% CI)")
    fig.suptitle("Heterogeneity of the fertility-dropout association (RQ4)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.savefig(out_path)
    plt.close(fig)


def figure_disparity_gaps(gaps: pd.DataFrame, out_path: str = "figures/phase2_disparity_gaps.png"):
    """State-by-state distribution of the Black-White and Hispanic-White
    dropout-rate gap, as a strip plot with the median marked -- shows both
    the typical gap and how much it varies across states."""
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    comparisons = ["Black vs White", "Hispanic vs White"]
    colors = [RACE_PALETTE["Black"], RACE_PALETTE["Hispanic"]]

    rng = np.random.default_rng(42)
    for i, (comp, color) in enumerate(zip(comparisons, colors)):
        vals = gaps[gaps["comparison"] == comp]["dropout_rate_gap_pp"].dropna()
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, alpha=0.6, s=35, edgecolor="white", linewidth=0.4)
        ax.hlines(vals.median(), i - 0.25, i + 0.25, color="black", linewidth=2.2, zorder=3)

    ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(comparisons)
    ax.set_ylabel("Dropout rate gap (percentage points)")
    ax.set_title("State-level racial dropout-rate gaps vs. White (median in black)")
    fig.savefig(out_path)
    plt.close(fig)


def figure_state_map(df: pd.DataFrame, value_col: str, race: str, title: str,
                      colorbar_title: str, out_path: str, color_scale: str = "Reds"):
    """State choropleth map, design doc sec. 10.2 pt. 5: only built because
    the geographic measure and period ARE directly comparable here -- one
    2017-2021 value per state, same source and definition throughout."""
    sub = df[df["race_ethnicity"] == race][["state", value_col]].dropna()
    sub = add_state_abbrev(sub)

    fig = px.choropleth(
        sub, locations="state_abbrev", locationmode="USA-states",
        color=value_col, scope="usa", color_continuous_scale=color_scale,
        labels={value_col: colorbar_title},
        hover_name="state",
        hover_data={"state_abbrev": False, value_col: ":.2f"},
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, family="Arial", color="black")),
        font=dict(family="Arial", size=13),
        margin=dict(l=10, r=10, t=60, b=10),
        geo=dict(landcolor="#d9d9d9", showland=True, lakecolor="white"),
        coloraxis_colorbar=dict(title=colorbar_title),
    )
    fig.write_image(out_path, width=1000, height=650, scale=2)

    html_path = os.path.splitext(out_path)[0] + ".html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path


def figure_fertility_map_small_multiples(
    df: pd.DataFrame,
    races: tuple[str, ...] = (
        "White", "Black", "American Indian/Alaska Native",
        "Asian", "Two or more races", "Pacific Islander",
    ),
    out_path: str = "figures/phase2_map_fertility_rate_by_race.png",
):
    """Side-by-side state maps of teen_fertility_rate, one panel per race, on
    a shared color scale. Covers every race in FERTILITY_ELIGIBLE_RACES --
    the same set the phase3 regression uses -- rather than an all-race rate
    that CDC WONDER never provides (Hispanic has none; see module
    docstring). Pacific Islander is listed last: it's the SPARSE_RACES
    group excluded from the primary regression, so its panel is mostly gray
    (12 of 51 states) -- shown here descriptively, consistent with how the
    design doc treats it everywhere else."""
    sub = df[df["race_ethnicity"].isin(races)][
        ["state", "race_ethnicity", "teen_fertility_rate"]
    ].dropna()
    sub = add_state_abbrev(sub)

    # Color scale is scaled to the non-sparse races only: Pacific Islander's
    # small-population estimates (e.g. Arkansas at ~89/1,000, from 230
    # births over a base of 2,570) are individually "reliable" per the
    # source's own flag but volatile enough that including them here would
    # wash out the color resolution for every other panel. Its cells still
    # plot on the shared scale -- they just clip to the darkest color
    # instead of stretching it.
    scale_sub = sub[~sub["race_ethnicity"].isin(SPARSE_RACES)]
    fig = px.choropleth(
        sub, locations="state_abbrev", locationmode="USA-states",
        color="teen_fertility_rate", scope="usa", color_continuous_scale="Purples",
        facet_col="race_ethnicity", facet_col_wrap=3,
        category_orders={"race_ethnicity": list(races)},
        range_color=(scale_sub["teen_fertility_rate"].min(), scale_sub["teen_fertility_rate"].max()),
        labels={"teen_fertility_rate": "Births per 1,000 women aged 15-19"},
        hover_name="state",
        hover_data={"state_abbrev": False, "race_ethnicity": True, "teen_fertility_rate": ":.1f"},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(
        title=dict(text="Teen fertility rate by state and race (2017-2021)", x=0.5,
                    font=dict(size=18, family="Arial", color="black")),
        font=dict(family="Arial", size=13),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    fig.update_geos(landcolor="#d9d9d9", showland=True, lakecolor="white")
    fig.write_image(out_path, width=1800, height=1000, scale=2)

    html_path = os.path.splitext(out_path)[0] + ".html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path


def figure_disparity_map(gaps: pd.DataFrame, comparison: str, title: str, out_path: str):
    """Choropleth of the racial dropout-rate gap itself (not a raw rate) --
    shows where the disparity is largest, not just where dropout is highest."""
    sub = gaps[gaps["comparison"] == comparison][["state", "dropout_rate_gap_pp"]].dropna()
    sub = add_state_abbrev(sub)

    fig = px.choropleth(
        sub, locations="state_abbrev", locationmode="USA-states",
        color="dropout_rate_gap_pp", scope="usa", color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        labels={"dropout_rate_gap_pp": "Gap (pp)"},
        hover_name="state",
        hover_data={"state_abbrev": False, "dropout_rate_gap_pp": ":.2f"},
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, family="Arial", color="black")),
        font=dict(family="Arial", size=13),
        margin=dict(l=10, r=10, t=60, b=10),
        geo=dict(landcolor="#d9d9d9", showland=True, lakecolor="white"),
    )
    fig.write_image(out_path, width=1000, height=650, scale=2)

    html_path = os.path.splitext(out_path)[0] + ".html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path


def _format_regression_summary(models: dict) -> str:
    lines = ["Full model output for the regression ladder (tables/phase3_regression_ladder.csv).",
             "All models: WLS weighted by 1/dropout_se^2, standard errors clustered by state."]
    for label, m in models.items():
        lines.append(f"\n{'='*70}\n{label}: {m.model.formula}\n{'='*70}")
        lines.append(str(m.summary()))
    return "\n".join(lines)


if __name__ == "__main__":
    df = load_master()

    quality_report = phase1_data_quality(df)
    quality_report.to_csv("results/phase1_data_quality_report.csv", index=False)
    print(quality_report.to_string(index=False))

    summary_stats = phase1_summary_stats(df)
    summary_stats.to_csv("tables/phase1_summary_statistics.csv", index=False)
    figure_distributions_by_race(df)

    gaps = phase2_disparity_gaps(df)
    gaps.to_csv("tables/phase2_racial_disparity_gaps.csv", index=False)
    figure_disparity_gaps(gaps)

    figure_state_map(df, "dropout_rate", "Total",
                      "High-school status dropout rate by state (2017-2021)",
                      "Dropout rate (%)", "figures/phase2_map_dropout_rate.png")
    # No "Total" fertility value exists -- CDC WONDER reports single-race
    # breakdowns only, never an all-race combined rate (unlike NCES dropout
    # data, which does have a Total row). Small multiples across every
    # FERTILITY_ELIGIBLE_RACES group on a shared color scale, rather than a
    # single race's map, so the figure speaks to disparity across all
    # comparison groups instead of standing in for a rate that doesn't exist.
    figure_fertility_map_small_multiples(df)
    figure_disparity_map(gaps, "Hispanic vs White",
                         "Hispanic-White dropout-rate gap by state",
                         "figures/phase2_map_hispanic_white_gap.png")
    figure_disparity_map(gaps, "Black vs White",
                         "Black-White dropout-rate gap by state",
                         "figures/phase2_map_black_white_gap.png")

    rankings = pd.concat([
        phase2_state_rankings(df, "White", "dropout_rate"),
        phase2_state_rankings(df, "Black", "dropout_rate"),
        phase2_state_rankings(df, "Hispanic", "dropout_rate"),
    ])
    rankings.to_csv("tables/phase2_state_rankings.csv", index=False)

    correlation = phase3_correlation(df)
    correlation.to_csv("tables/phase3_correlation.csv", index=False)
    print("\n" + correlation.to_string(index=False))
    figure_correlation_scatter(df)

    ladder, ladder_models = phase3_regression_ladder(df)
    ladder.to_csv("tables/phase3_regression_ladder.csv", index=False)
    with open("results/phase3_regression_summary.txt", "w") as f:
        f.write(_format_regression_summary(ladder_models))
    print("\nRegression ladder (fertility coefficient, clustered SE):")
    print(ladder.round(4).to_string(index=False))

    robustness = phase3_robustness(df)
    robustness.to_csv("tables/phase3_robustness.csv", index=False)
    print("\nRobustness of the fertility coefficient:")
    print(robustness.round(4).to_string(index=False))

    slopes, slope_tests = phase3_heterogeneity(df)
    slopes.to_csv("tables/phase3_heterogeneity_slopes.csv", index=False)
    slope_tests.to_csv("tables/phase3_heterogeneity_tests.csv", index=False)
    print("\nRQ4 group-specific fertility slopes:")
    print(slopes.round(4).to_string(index=False))
    print(slope_tests.round(4).to_string(index=False))
    figure_heterogeneity_slopes(slopes, slope_tests)

    vif = phase3_vif(df)
    vif.to_csv("tables/phase3_vif.csv", index=False)
    print("\nMulticollinearity check (VIF, primary sample):")
    print(vif.to_string(index=False))
    if (vif["vif"] > 5).any():
        print("WARNING: at least one predictor has VIF > 5 -- multivariable "
              "regression coefficients may be affected by collinearity.")

    diagnostic_tests = phase3_diagnostic_tests(df)
    diagnostic_tests.to_csv("tables/phase3_diagnostic_tests.csv", index=False)
    print("\nRegression diagnostic tests (primary WLS model):")
    print(diagnostic_tests.to_string(index=False))
    figure_regression_diagnostics(df)

    print("\nOutputs -> tables/, results/, figures/ (see README 'Running the pipeline')")
