"""
Machine learning and explainability, framed as a fairness/disparity
analysis rather than a prediction-accuracy showcase. The dataset (~200
usable rows after quality filtering) does not support a strong predictive
claim; it does support a rigorous look at whether an ML model's errors are
systematically uneven across demographic groups, and whether race carries
predictive weight beyond the measured socioeconomic/fertility context.

Two models are fit on the same population and cross-validation folds:

  Model A (fairness evaluation): teen_fertility_rate + poverty_rate +
    insurance_rate + census_region. Race is NOT a feature. This is the
    standard fairness setup -- a model that never sees race, evaluated for
    whether its errors are still uneven across race groups.

  Model B (residual disparity attribution): Model A's features + race.
    If race carries substantial SHAP weight here even after controlling
    for the same socioeconomic/fertility context, that is evidence of a
    disparity not explained by the measured covariates.

Evaluation uses grouped 5-fold cross-validation (grouped by state, since
every race-row within a state shares the same poverty/insurance/region
values -- a random row split would leak state-level information across
train/test and overstate performance). Final models for interpretation
(SHAP, coefficients) are fit on the full sample, since interpretation asks
what the fitted relationships in the available evidence are, not how well
they generalize -- that question is answered separately by the
cross-validated metrics.

Excluded from training entirely: Hispanic and Total (no CDC WONDER
fertility value exists for either) and Pacific Islander (n=4 states with
usable data -- too sparse for any reliable conclusion, fairness or
otherwise). This means the fairness analysis below cannot speak to
whether the model is fair with respect to Hispanic individuals, because
no model can be built for them from this data at all. That is a scope
limitation of the analysis, not a finding about Hispanic students.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from analysis import FERTILITY_ELIGIBLE_RACES, SPARSE_RACES, load_master
from plotting import RACE_PALETTE, set_style

RANDOM_SEED = 42
N_FOLDS = 5

TRAINING_RACES = [r for r in FERTILITY_ELIGIBLE_RACES if r not in SPARSE_RACES]
BASE_FEATURES = ["teen_fertility_rate", "poverty_rate", "insurance_rate", "census_region"]


def build_modeling_sample(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["fips", "state", "race_ethnicity", "dropout_rate", "dropout_se"] + BASE_FEATURES
    sub = df[df["race_ethnicity"].isin(TRAINING_RACES)][cols].dropna()
    return sub.reset_index(drop=True)


def _design_matrix(sub: pd.DataFrame, include_race: bool) -> pd.DataFrame:
    X = pd.get_dummies(sub[["census_region"]], drop_first=True)
    X.insert(0, "teen_fertility_rate", sub["teen_fertility_rate"].values)
    X.insert(1, "poverty_rate", sub["poverty_rate"].values)
    X.insert(2, "insurance_rate", sub["insurance_rate"].values)
    if include_race:
        race_dummies = pd.get_dummies(sub["race_ethnicity"], prefix="race", drop_first=True)
        X = pd.concat([X.reset_index(drop=True), race_dummies.reset_index(drop=True)], axis=1)
    return X.astype(float)


def cross_validate(sub: pd.DataFrame, include_race: bool) -> dict:
    """Grouped 5-fold CV (grouped by state/fips), returns out-of-fold
    predictions for baseline, linear regression, and random forest, all
    fit with sample_weight = 1/dropout_se^2 (consistent with the Phase 3
    weighted regression -- see src/analysis.py)."""
    X = _design_matrix(sub, include_race)
    y = sub["dropout_rate"].values
    weights = 1.0 / (sub["dropout_se"].values ** 2)
    groups = sub["fips"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = {name: np.full(len(sub), np.nan) for name in ["baseline", "linear", "random_forest"]}

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        w_train = weights[train_idx]

        baseline = DummyRegressor(strategy="mean").fit(X_train, y_train, sample_weight=w_train)
        oof["baseline"][test_idx] = baseline.predict(X_test)

        linear = LinearRegression().fit(X_train, y_train, sample_weight=w_train)
        oof["linear"][test_idx] = linear.predict(X_test)

        rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED, min_samples_leaf=3)
        rf.fit(X_train, y_train, sample_weight=w_train)
        oof["random_forest"][test_idx] = rf.predict(X_test)

    return oof


def model_comparison_table(sub: pd.DataFrame, oof: dict) -> pd.DataFrame:
    y = sub["dropout_rate"].values
    rows = []
    for name, preds in oof.items():
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y, preds),
            "RMSE": np.sqrt(mean_squared_error(y, preds)),
            "R2": r2_score(y, preds),
        })
    return pd.DataFrame(rows)


def fairness_report(sub: pd.DataFrame, oof_predictions: np.ndarray, model_name: str) -> pd.DataFrame:
    """Per-race-group out-of-fold error, for the model that never saw race
    as a feature (Model A). A group with a systematically nonzero mean
    residual is one the model is systematically over- or under-predicting
    for -- exactly the disparity design_document.pdf RQ7 asks about."""
    residuals = sub["dropout_rate"].values - oof_predictions
    report = pd.DataFrame({"race_ethnicity": sub["race_ethnicity"], "residual": residuals})

    rows = []
    for race, grp in report.groupby("race_ethnicity"):
        rows.append({
            "race_ethnicity": race,
            "n": len(grp),
            "MAE": grp["residual"].abs().mean(),
            "mean_residual": grp["residual"].mean(),
            "reliable_n": len(grp) >= 15,
        })
    result = pd.DataFrame(rows).sort_values("race_ethnicity")

    groups_for_test = [g["residual"].values for _, g in report.groupby("race_ethnicity")]
    stat, p = stats.kruskal(*groups_for_test)
    result.attrs["kruskal_wallis"] = {"statistic": stat, "p_value": p, "model": model_name}
    return result


def shap_analysis(sub: pd.DataFrame, include_race: bool, out_path: str, title: str):
    """Fit on the full sample (not a CV fold) for interpretation, per this
    module's docstring. Saves a native SHAP beeswarm plot (shows both the
    magnitude and direction of each feature's effect, per observation --
    the standard explainability figure for this kind of analysis, not an
    approximation of it) and returns the mean absolute SHAP value per
    feature, with one-hot race/region columns aggregated back to their
    parent variable for the printed ranking summary."""
    set_style()
    X = _design_matrix(sub, include_race)
    y = sub["dropout_rate"].values
    weights = 1.0 / (sub["dropout_se"].values ** 2)

    rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED, min_samples_leaf=3)
    rf.fit(X, y, sample_weight=weights)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, show=False, plot_size=(8.5, 0.45 * X.shape[1] + 2))
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("SHAP value (impact on predicted dropout_rate, pp)")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)

    def _parent(col: str) -> str:
        if col.startswith("census_region_"):
            return "census_region"
        if col.startswith("race_"):
            return "race_ethnicity"
        return col

    grouped = mean_abs_shap.groupby(_parent).sum().sort_values(ascending=False)
    return grouped


def figure_fairness_errors(fairness: pd.DataFrame, out_path: str = "figures/phase5_fairness_errors.png"):
    """Group-wise model error (MAE), the figure version of the fairness
    table -- sized and colored to make the American Indian/Alaska Native
    outlier immediately visible, with low-N groups marked explicitly
    rather than presented as equally reliable."""
    set_style()
    ordered = fairness.sort_values("MAE", ascending=True)
    colors = [RACE_PALETTE.get(r, "#333333") for r in ordered["race_ethnicity"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.barh(ordered["race_ethnicity"], ordered["MAE"], color=colors, edgecolor="white")
    for bar, n, reliable in zip(bars, ordered["n"], ordered["reliable_n"]):
        label = f"n={n}" + ("" if reliable else "  (low N)")
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, label,
                va="center", ha="left", fontsize=9, color="#555555")

    ax.set_xlabel("Mean absolute error, out-of-fold prediction (percentage points)")
    ax.set_title("Model error by race/ethnicity (race excluded from model inputs)")
    ax.set_xlim(0, ordered["MAE"].max() * 1.25)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    df = load_master()
    sub = build_modeling_sample(df)
    print(f"Modeling sample: {len(sub)} rows across {sub['race_ethnicity'].nunique()} race groups "
          f"({', '.join(TRAINING_RACES)})")
    print(f"Excluded from modeling entirely: Hispanic, Total (no fertility value), "
          f"{', '.join(SPARSE_RACES)} (insufficient N)")

    # --- Model A: fairness evaluation (race NOT a feature) ---
    oof_a = cross_validate(sub, include_race=False)
    comparison_a = model_comparison_table(sub, oof_a)
    comparison_a.to_csv("results/phase4_model_comparison.csv", index=False)
    print("\nModel comparison (cross-validated, race excluded):")
    print(comparison_a.to_string(index=False))

    fairness = fairness_report(sub, oof_a["random_forest"], "random_forest (Model A, no race feature)")
    fairness.to_csv("results/phase5_fairness_report.csv", index=False)
    kw = fairness.attrs["kruskal_wallis"]
    print(f"\nGroup-wise residual report (Model A random forest):")
    print(fairness.to_string(index=False))
    print(f"\nKruskal-Wallis test for equal residual distributions across groups: "
          f"H={kw['statistic']:.2f}, p={kw['p_value']:.4f}")

    figure_fairness_errors(fairness)

    shap_a = shap_analysis(sub, include_race=False,
                            out_path="figures/phase5_shap_model_a_no_race.png",
                            title="SHAP feature impact (Model A: race excluded)")
    print("\nSHAP feature importance (Model A):")
    print(shap_a.to_string())

    # --- Model B: residual disparity attribution (race included) ---
    shap_b = shap_analysis(sub, include_race=True,
                            out_path="figures/phase5_shap_model_b_with_race.png",
                            title="SHAP feature impact (Model B: race included)")
    print("\nSHAP feature importance (Model B, race included):")
    print(shap_b.to_string())
    race_rank = list(shap_b.sort_values(ascending=False).index).index("race_ethnicity") + 1
    print(f"\nrace_ethnicity ranks #{race_rank} of {len(shap_b)} features by SHAP importance in Model B "
          f"-- i.e., race still carries this much predictive weight after controlling for "
          f"fertility, poverty, insurance, and region.")

    print("\nOutputs -> results/phase4_model_comparison.csv, results/phase5_fairness_report.csv,")
    print("           figures/phase5_fairness_errors.png, figures/phase5_shap_model_a_no_race.png,")
    print("           figures/phase5_shap_model_b_with_race.png")
