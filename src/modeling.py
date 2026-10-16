"""
Machine learning and explainability (design_document.pdf sec. 10.4-10.5),
framed as a predictive-disparity analysis rather than a prediction-accuracy
showcase. The dataset (~200 state-race cells) does not support a strong
predictive claim; it does support a careful look at (a) whether contextual
socioeconomic variables add predictive value beyond teen fertility (H4),
(b) whether a model's errors are systematically uneven across demographic
groups (H5/RQ7), and (c) whether race carries predictive weight beyond the
measured fertility/socioeconomic context.

Feature sets, all fit on the same rows and the same folds:

  fertility_only : teen_fertility_rate                         (H4 reference)
  A (no race)    : teen_fertility_rate + poverty_rate + uninsured_rate +
                   census_region. Race is NOT a feature -- the standard
                   fairness setup: a model that never sees race, evaluated
                   for whether its errors are still uneven across race.
  B (with race)  : A + race_ethnicity. If race carries substantial SHAP
                   weight here, that is evidence of a disparity not
                   explained by the measured covariates.

Every model is trained under two weighting schemes, reported side by side:
"inverse_variance" (sample_weight = 1/dropout_se^2, matching the Phase 3
WLS) and "unweighted". The choice matters for the fairness question: under
inverse-variance weights the American Indian/Alaska Native cells (large
SEs) carry well under 1% of the total training weight, so a weighted model
is effectively fit to the White, Asian and Black cells and its errors for
AI/AN partly reflect that, not only the features. Reporting both separates
"the features can't explain this group's dropout level" from "the loss
function ignored this group".

Evaluation is repeated grouped K-fold cross-validation: states (not rows)
are randomly assigned to folds, repeated N_REPEATS times with different
seeded assignments. Grouping by state prevents leakage (race rows in a
state share every state-level covariate); repetition shows how much the
metrics depend on one particular fold assignment, which at 51 states is a
lot. Each row's out-of-fold prediction used for the fairness analysis is
its mean across repeats.

Final models for interpretation (SHAP) are fit on the full sample --
interpretation asks what relationships the fitted model uses, which the
cross-validated metrics answer separately from how well it generalizes.

The sample is analysis.association_sample() -- identical to the Phase 3
regressions. Excluded: Hispanic and Total (no CDC WONDER fertility value)
and Pacific Islander (too sparse). The fairness analysis therefore cannot
speak to Hispanic students; that is a scope limitation, not a finding.
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

from analysis import association_sample, load_master, PRIMARY_RACES, SPARSE_RACES
from plotting import RACE_PALETTE, set_style

RANDOM_SEED = 42
N_FOLDS = 5
N_REPEATS = 20

NUMERIC_FEATURES = ["teen_fertility_rate", "poverty_rate", "uninsured_rate"]

FEATURE_SETS = {
    "fertility_only": {"numeric": ["teen_fertility_rate"], "region": False, "race": False},
    "A_no_race": {"numeric": NUMERIC_FEATURES, "region": True, "race": False},
    "B_with_race": {"numeric": NUMERIC_FEATURES, "region": True, "race": True},
}

WEIGHTINGS = ["inverse_variance", "unweighted"]

# (model name, feature set, estimator factory)
MODEL_SPECS = [
    ("baseline_mean", "A_no_race", lambda: DummyRegressor(strategy="mean")),
    ("linear", "fertility_only", lambda: LinearRegression()),
    ("random_forest", "fertility_only",
     lambda: RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED, min_samples_leaf=3, n_jobs=-1)),
    ("linear", "A_no_race", lambda: LinearRegression()),
    ("random_forest", "A_no_race",
     lambda: RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED, min_samples_leaf=3, n_jobs=-1)),
    ("linear", "B_with_race", lambda: LinearRegression()),
    ("random_forest", "B_with_race",
     lambda: RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED, min_samples_leaf=3, n_jobs=-1)),
]


def _design_matrix(sub: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    spec = FEATURE_SETS[feature_set]
    X = sub[spec["numeric"]].reset_index(drop=True).copy()
    if spec["region"]:
        X = pd.concat([X, pd.get_dummies(sub["census_region"], prefix="census_region",
                                         drop_first=True).reset_index(drop=True)], axis=1)
    if spec["race"]:
        X = pd.concat([X, pd.get_dummies(sub["race_ethnicity"], prefix="race",
                                         drop_first=True).reset_index(drop=True)], axis=1)
    return X.astype(float)


def _grouped_folds(groups: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Randomly assign each unique group (state) to one of N_FOLDS folds of
    near-equal size; returns test indices per fold."""
    unique = rng.permutation(np.unique(groups))
    fold_of = {g: i % N_FOLDS for i, g in enumerate(unique)}
    assignment = np.array([fold_of[g] for g in groups])
    return [np.where(assignment == k)[0] for k in range(N_FOLDS)]


def _weights(sub: pd.DataFrame, weighting: str) -> np.ndarray:
    return sub["weight"].values if weighting == "inverse_variance" else np.ones(len(sub))


def repeated_grouped_cv(sub: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns (per-repeat metrics, mean out-of-fold predictions keyed by
    (model, feature_set, weighting)). Every model/weighting sees the same
    fold assignment within a repeat, so comparisons are paired."""
    y = sub["dropout_rate"].values
    groups = sub["fips"].values
    X_by_set = {fs: _design_matrix(sub, fs) for fs in FEATURE_SETS}
    rng = np.random.default_rng(RANDOM_SEED)
    keys = [(m, fs, wt) for m, fs, _ in MODEL_SPECS for wt in WEIGHTINGS]
    makers = {(m, fs): make for m, fs, make in MODEL_SPECS}

    metric_rows = []
    oof_sum = {k: np.zeros(len(sub)) for k in keys}
    for repeat in range(N_REPEATS):
        folds = _grouped_folds(groups, rng)
        oof = {k: np.full(len(sub), np.nan) for k in keys}
        for test_idx in folds:
            train_idx = np.setdiff1d(np.arange(len(sub)), test_idx)
            for name, fs, wt in keys:
                X = X_by_set[fs]
                w = _weights(sub, wt)
                est = makers[(name, fs)]().fit(X.iloc[train_idx], y[train_idx], sample_weight=w[train_idx])
                oof[(name, fs, wt)][test_idx] = est.predict(X.iloc[test_idx])
        for (name, fs, wt), preds in oof.items():
            oof_sum[(name, fs, wt)] += preds
            metric_rows.append({
                "repeat": repeat, "model": name, "feature_set": fs, "weighting": wt,
                "MAE": mean_absolute_error(y, preds),
                "RMSE": np.sqrt(mean_squared_error(y, preds)),
                "R2": r2_score(y, preds),
                "weighted_R2": r2_score(y, preds, sample_weight=sub["weight"].values),
            })
    oof_mean = {k: v / N_REPEATS for k, v in oof_sum.items()}
    return pd.DataFrame(metric_rows), oof_mean


def model_comparison_table(per_repeat: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD across CV repeats -- the SD is the honest error bar on
    each metric at this sample size."""
    agg = per_repeat.groupby(["weighting", "model", "feature_set"], sort=False).agg(
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        weighted_R2_mean=("weighted_R2", "mean"), weighted_R2_sd=("weighted_R2", "std"),
    )
    return agg.reset_index()


def h4_test(per_repeat: pd.DataFrame) -> pd.DataFrame:
    """H4: do fertility + socioeconomic features beat fertility alone?
    Paired comparison across CV repeats (same folds within a repeat), per
    learner and weighting. Reports the mean RMSE reduction, its SD, and the
    share of repeats in which the fuller model wins."""
    rows = []
    for wt in WEIGHTINGS:
        for name in ["linear", "random_forest"]:
            pr = per_repeat[(per_repeat["model"] == name) & (per_repeat["weighting"] == wt)]
            a = pr[pr["feature_set"] == "fertility_only"].set_index("repeat")
            b = pr[pr["feature_set"] == "A_no_race"].set_index("repeat")
            diff = a["RMSE"] - b["RMSE"]
            rows.append({
                "weighting": wt, "learner": name,
                "comparison": "A_no_race vs fertility_only",
                "mean_RMSE_reduction": diff.mean(), "sd_RMSE_reduction": diff.std(),
                "share_repeats_A_better": (diff > 0).mean(),
                "mean_R2_gain": (b["R2"] - a["R2"]).mean(),
            })
    return pd.DataFrame(rows)


def fairness_report(sub: pd.DataFrame, oof_predictions: np.ndarray, model_label: str) -> pd.DataFrame:
    """Per-race out-of-fold error. A group whose mean residual is
    systematically nonzero is one the model over- (negative) or under-
    (positive) predicts -- the disparity RQ7 asks about. The 95% CI on the
    mean residual is a state-level cluster bootstrap, since cells within a
    group are not independent draws."""
    residuals = sub["dropout_rate"].values - oof_predictions
    report = pd.DataFrame({"race_ethnicity": sub["race_ethnicity"].values,
                           "fips": sub["fips"].values, "residual": residuals})
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []
    for race, grp in report.groupby("race_ethnicity"):
        states = grp["fips"].unique()
        by_state = {s: grp.loc[grp["fips"] == s, "residual"].values for s in states}
        boot = []
        for _ in range(2000):
            draw = rng.choice(states, size=len(states), replace=True)
            boot.append(np.concatenate([by_state[s] for s in draw]).mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "model": model_label, "race_ethnicity": race, "n": len(grp),
            "MAE": grp["residual"].abs().mean(),
            "mean_residual": grp["residual"].mean(),
            "mean_residual_ci_low": lo, "mean_residual_ci_high": hi,
            "systematic_bias": not (lo <= 0 <= hi),
            "reliable_n": len(grp) >= 15,
        })
    result = pd.DataFrame(rows).sort_values("race_ethnicity")
    stat, p = stats.kruskal(*[g["residual"].values for _, g in report.groupby("race_ethnicity")])
    result["kruskal_wallis_H"] = stat
    result["kruskal_wallis_p"] = p
    return result


def shap_analysis(sub: pd.DataFrame, feature_set: str, out_path: str, title: str) -> pd.Series:
    """Fit on the full sample (inverse-variance weighted, consistent with
    the Phase 3 regressions) for interpretation. Saves a SHAP beeswarm and
    returns mean |SHAP| per feature, with one-hot race/region columns summed
    back to their parent variable."""
    set_style()
    X = _design_matrix(sub, feature_set)
    rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED, min_samples_leaf=3, n_jobs=-1)
    rf.fit(X, sub["dropout_rate"].values, sample_weight=sub["weight"].values)

    shap_values = shap.TreeExplainer(rf).shap_values(X)
    shap.summary_plot(shap_values, X, show=False, plot_size=(8.5, 0.45 * X.shape[1] + 2))
    fig, ax = plt.gcf(), plt.gca()
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("SHAP value (impact on predicted dropout rate, pp)")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)

    def _parent(col: str) -> str:
        if col.startswith("census_region_"):
            return "census_region"
        if col.startswith("race_"):
            return "race_ethnicity"
        return col

    return mean_abs.groupby(_parent).sum().sort_values(ascending=False)


FAIRNESS_FIGURE_SERIES = [
    ("random_forest / A_no_race / inverse_variance", "RF, race excluded, inverse-variance weights", "o", "#222222"),
    ("random_forest / A_no_race / unweighted", "RF, race excluded, unweighted", "o", "#999999"),
    ("linear / B_with_race / inverse_variance", "Linear, race included, inverse-variance weights", "s", "#222222"),
    ("linear / B_with_race / unweighted", "Linear, race included, unweighted", "s", "#999999"),
]


def figure_fairness_errors(fairness: pd.DataFrame, out_path: str = "figures/phase5_fairness_errors.png"):
    """Mean out-of-fold residual by race (cluster-bootstrap 95% CI).
    Positive = under-prediction. Circles: race-blind random forest; squares:
    race-aware linear model; dark vs light: training weights. Shows whether
    a race-blind model is systematically biased for a group, whether that
    depends on the training weights, and whether a race term removes it."""
    set_style()
    ref = fairness[fairness["model"] == FAIRNESS_FIGURE_SERIES[0][0]]
    races = list(ref.sort_values("mean_residual")["race_ethnicity"])
    n = ref.set_index("race_ethnicity")["n"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    offsets = np.linspace(-0.27, 0.27, len(FAIRNESS_FIGURE_SERIES))
    for (model_label, legend, marker, color), off in zip(FAIRNESS_FIGURE_SERIES, offsets):
        d = fairness[fairness["model"] == model_label].set_index("race_ethnicity").loc[races]
        y = np.arange(len(races)) + off
        ax.errorbar(d["mean_residual"], y,
                    xerr=[d["mean_residual"] - d["mean_residual_ci_low"],
                          d["mean_residual_ci_high"] - d["mean_residual"]],
                    fmt=marker, color=color, markersize=5, capsize=2.5, label=legend)
    ax.set_yticks(np.arange(len(races)))
    ax.set_yticklabels([f"{r} (n={n[r]})" for r in races])
    ax.axvline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Mean out-of-fold residual, pp (positive = under-predicted)")
    ax.set_title("Group-wise prediction bias (95% state-cluster bootstrap CI)")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    df = load_master()
    sub = association_sample(df)
    print(f"Modeling sample: {len(sub)} rows, {sub['fips'].nunique()} states, races: {', '.join(PRIMARY_RACES)}")
    print(f"Excluded: Hispanic, Total (no fertility value); {', '.join(SPARSE_RACES)} (insufficient N)")
    print(f"Repeated grouped CV: {N_REPEATS} x {N_FOLDS}-fold by state, seed {RANDOM_SEED}")

    per_repeat, oof = repeated_grouped_cv(sub)
    comparison = model_comparison_table(per_repeat)
    comparison.to_csv("results/phase4_model_comparison.csv", index=False)
    print("\nModel comparison (mean +/- SD over CV repeats):")
    print(comparison.round(3).to_string(index=False))

    h4 = h4_test(per_repeat)
    h4.to_csv("results/phase4_h4_feature_value.csv", index=False)
    print("\nH4 -- value of socioeconomic features beyond fertility:")
    print(h4.round(3).to_string(index=False))

    fairness = pd.concat([
        fairness_report(sub, oof[(m, fs, wt)], f"{m} / {fs} / {wt}")
        for m, fs in [("random_forest", "A_no_race"), ("linear", "A_no_race"),
                      ("random_forest", "B_with_race"), ("linear", "B_with_race")]
        for wt in WEIGHTINGS
    ], ignore_index=True)
    fairness.to_csv("results/phase5_fairness_report.csv", index=False)
    print("\nGroup-wise out-of-fold error:")
    print(fairness.drop(columns=["reliable_n"]).round(3).to_string(index=False))
    figure_fairness_errors(fairness)

    shap_a = shap_analysis(sub, "A_no_race", "figures/phase5_shap_model_a_no_race.png",
                           "SHAP feature impact (Model A: race excluded)")
    shap_b = shap_analysis(sub, "B_with_race", "figures/phase5_shap_model_b_with_race.png",
                           "SHAP feature impact (Model B: race included)")
    shap_table = pd.concat([shap_a.rename("model_A_mean_abs_shap"),
                            shap_b.rename("model_B_mean_abs_shap")], axis=1)
    shap_table.to_csv("results/phase5_shap_importance.csv", index_label="feature")
    print("\nMean |SHAP| by feature:")
    print(shap_table.round(3).to_string())
