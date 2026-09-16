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

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from patsy import dmatrix
from scipy import stats
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
    cols = ["fips", "dropout_rate", "dropout_se", "teen_fertility_rate", "poverty_rate",
            "insurance_rate", "census_region", "race_ethnicity"]
    sub = df[df["race_ethnicity"].isin(races)][cols].dropna()
    return sub


MULTIVARIABLE_FORMULA = "dropout_rate ~ teen_fertility_rate + poverty_rate + insurance_rate + C(census_region)"


def phase3_regressions(df: pd.DataFrame) -> dict[str, object]:
    """Bivariate + multivariable regression, unweighted (OLS) and
    inverse-variance weighted (WLS, weight = 1/dropout_se^2) -- weighting
    downweights small/noisy state-race cells instead of treating a
    CV-50% estimate as equally trustworthy as a solid one. Run both with
    and without Pacific Islander (SPARSE_RACES) as the sensitivity check
    design doc sec. 10.3 pt. 7 asks for.

    Also reports the WLS model with standard errors clustered by state
    (design doc sec. 10.3 pt. 6): every state contributes up to 8 race-group
    rows that share the same poverty_rate/insurance_rate/census_region
    values, so treating all rows as independent understates uncertainty.
    Clustering by fips corrects for that within-state non-independence.

    And a Cook's-distance sensitivity check (design doc sec. 10.3 pt. 7):
    refits the primary WLS model excluding points with unusually high
    influence, to check the fertility coefficient isn't an artifact of a
    handful of extreme state-race cells."""
    results = {}
    for label, include_sparse in [("primary", False), ("with_sparse_race", True)]:
        sub = _regression_sample(df, include_sparse)
        sub["weight"] = 1.0 / (sub["dropout_se"] ** 2)

        bivariate_ols = smf.ols("dropout_rate ~ teen_fertility_rate", data=sub).fit()
        multivariable_ols = smf.ols(MULTIVARIABLE_FORMULA, data=sub).fit()
        multivariable_wls = smf.wls(MULTIVARIABLE_FORMULA, data=sub, weights=sub["weight"]).fit()
        multivariable_wls_clustered = smf.wls(MULTIVARIABLE_FORMULA, data=sub, weights=sub["weight"]).fit(
            cov_type="cluster", cov_kwds={"groups": sub["fips"]}
        )

        cooks_d = OLSInfluence(multivariable_wls).cooks_distance[0]
        threshold = 4 / len(sub)
        high_leverage_mask = cooks_d > threshold
        sub_excl = sub.loc[~high_leverage_mask]
        multivariable_wls_excl_high_leverage = smf.wls(
            MULTIVARIABLE_FORMULA, data=sub_excl, weights=sub_excl["weight"]
        ).fit()

        results[label] = {
            "n": len(sub),
            "bivariate_ols": bivariate_ols,
            "multivariable_ols": multivariable_ols,
            "multivariable_wls": multivariable_wls,
            "multivariable_wls_clustered": multivariable_wls_clustered,
            "n_high_leverage_excluded": int(high_leverage_mask.sum()),
            "multivariable_wls_excl_high_leverage": multivariable_wls_excl_high_leverage,
        }
    return results


def phase3_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Multicollinearity check for the multivariable regression's predictors
    (design doc sec. 7.3/11: "avoid adding many correlated socioeconomic
    variables" / "may be highly correlated with one another"). Computed on
    the same primary regression sample as phase3_regressions (Pacific
    Islander excluded). A VIF above 5 flags a predictor whose standard
    error is inflated by collinearity with the others in the model --
    above 10 is a stronger warning; below that, the multivariable
    regression's coefficients can be interpreted without a collinearity
    caveat."""
    sub = _regression_sample(df, include_sparse=False)
    design = dmatrix(
        "teen_fertility_rate + poverty_rate + insurance_rate + C(census_region)",
        data=sub, return_type="dataframe",
    )
    vifs = pd.DataFrame({
        "variable": design.columns,
        "vif": [variance_inflation_factor(design.values, i) for i in range(design.shape[1])],
    })
    return vifs[vifs["variable"] != "Intercept"].reset_index(drop=True)


def _fit_primary_wls(df: pd.DataFrame):
    """Refits the same primary multivariable WLS model used in
    phase3_regressions (Pacific Islander excluded, weighted by
    1/dropout_se^2) -- shared by the diagnostic tests and diagnostic plots
    below so both look at exactly the model reported in Findings."""
    sub = _regression_sample(df, include_sparse=False)
    sub["weight"] = 1.0 / (sub["dropout_se"] ** 2)
    model = smf.wls(MULTIVARIABLE_FORMULA, data=sub, weights=sub["weight"]).fit()
    return model, sub


def phase3_diagnostic_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Formal tests backing up the residual diagnostic plots (design doc
    sec. 10.3: regression diagnostics) -- a plot alone doesn't say whether a
    pattern is significant or just noise at this sample size.

    Shapiro-Wilk tests whether the studentized residuals are consistent
    with a normal distribution (needed for the regression's p-values/CIs
    to be trustworthy at this n). Breusch-Pagan tests for heteroscedasticity
    remaining in the *response* residuals even after the model has already
    been weighted by 1/dropout_se^2 -- i.e. whether the inverse-variance
    weighting fully accounts for the unequal reliability across state-race
    cells, or whether some residual heteroscedasticity is left over."""
    model, _ = _fit_primary_wls(df)
    resid_std = OLSInfluence(model).resid_studentized_internal

    shapiro_stat, shapiro_p = stats.shapiro(resid_std)
    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)

    return pd.DataFrame([
        {"test": "Shapiro-Wilk (normality of residuals)", "statistic": shapiro_stat, "p_value": shapiro_p},
        {"test": "Breusch-Pagan (homoscedasticity)", "statistic": bp_stat, "p_value": bp_p},
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
    they're shown as one box per race. poverty_rate and insurance_rate are
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
        ("insurance_rate", "Child health-insurance rate (%)"),
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
    influence = OLSInfluence(model)
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
    """The paper's likely Figure 1: the core RQ3 relationship. One point per
    state x race-group cell, colored by race, restricted to rows where both
    measures are reliable (same population as phase3_correlation)."""
    set_style()
    sub = df[
        df["race_ethnicity"].isin(FERTILITY_ELIGIBLE_RACES)
        & (df["dropout_quality_flag"] == "reliable")
        & (df["natality_quality_flag"] == "reliable")
    ][["dropout_rate", "teen_fertility_rate", "race_ethnicity"]].dropna()

    r, p = stats.pearsonr(sub["dropout_rate"], sub["teen_fertility_rate"])
    slope, intercept = np.polyfit(sub["teen_fertility_rate"], sub["dropout_rate"], 1)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for race, grp in sub.groupby("race_ethnicity"):
        ax.scatter(
            grp["teen_fertility_rate"], grp["dropout_rate"],
            label=race, color=RACE_PALETTE.get(race, "#333333"),
            alpha=0.75, s=45, edgecolor="white", linewidth=0.5,
        )
    x_line = np.linspace(sub["teen_fertility_rate"].min(), sub["teen_fertility_rate"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5, linestyle="--", zorder=1)

    ax.set_xlabel("Teen fertility rate (births per 1,000 women aged 15-19)")
    ax.set_ylabel("High-school status dropout rate (%)")
    ax.set_title("Teen fertility rate and dropout rate, by state and race/ethnicity")
    ax.text(
        0.03, 0.97, f"Pearson r = {r:.2f}\np < 0.001\nn = {len(sub)}",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax.legend(loc="lower right", ncol=1, markerscale=1.2)
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
        lines.append("\n--- Multivariable WLS, standard errors clustered by state ---")
        lines.append("(accounts for up to 8 non-independent race-group rows per state)")
        lines.append(str(r["multivariable_wls_clustered"].summary()))
        lines.append(f"\n--- Sensitivity: WLS excluding {r['n_high_leverage_excluded']} high-leverage points "
                      f"(Cook's distance > 4/n) ---")
        lines.append(str(r["multivariable_wls_excl_high_leverage"].summary()))
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

    regressions = phase3_regressions(df)
    with open("results/phase3_regression_summary.txt", "w") as f:
        f.write(_format_regression_summary(regressions))
    print(f"\nRegression n (primary, Pacific Islander excluded): {regressions['primary']['n']}")
    print(f"Regression n (with sparse race included): {regressions['with_sparse_race']['n']}")
    print("\nFull output -> results/phase3_regression_summary.txt")

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

    print("\nTables -> tables/phase1_summary_statistics.csv, tables/phase2_racial_disparity_gaps.csv,")
    print("          tables/phase2_state_rankings.csv, tables/phase3_correlation.csv, tables/phase3_vif.csv,")
    print("          tables/phase3_diagnostic_tests.csv")
    print(f"n high-leverage points excluded in sensitivity check (primary): "
          f"{regressions['primary']['n_high_leverage_excluded']}")
    print("Figures -> figures/phase1_distributions_by_race.png, figures/phase2_disparity_gaps.png,")
    print("           figures/phase2_map_dropout_rate.png, figures/phase2_map_hispanic_white_gap.png,")
    print("           figures/phase2_map_black_white_gap.png, figures/phase2_map_fertility_rate_by_race.png,")
    print("           figures/phase3_fertility_dropout_scatter.png,")
    print("           figures/phase3_regression_diagnostics.png")
