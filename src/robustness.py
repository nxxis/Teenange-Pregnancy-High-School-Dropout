"""
Advanced robustness and sensitivity checks for the Phase 3 association
(design_document.pdf sec. 10.3 pt. 7, sec. 11), run on the same analytic
sample and model specifications as src/analysis.py:

  1. Oster (2019) coefficient stability -- how strong would selection on
     unobservables have to be, relative to the observed controls, to drive
     the fertility coefficient to zero? (delta; >1 is the usual benchmark)
  2. Wild cluster bootstrap-t (restricted, Webb weights) p-values -- the
     cluster-robust p-values in analysis.py rest on 51 state clusters,
     where asymptotic cluster SEs can over-reject; this is the standard
     small-cluster correction (Cameron, Gelbach & Miller 2008; Webb 2023).
  3. Spatial dependence -- Moran's I on state-level model residuals with
     k-nearest-neighbor weights, permutation p-value; and Conley spatial-HAC
     standard errors that allow residual correlation between nearby states.
  4. Leave-one-state-out and leave-one-race-group-out refits -- is any
     single state or group driving the result?
  5. Shapley (LMG) decomposition of the weighted R^2 across predictor
     groups -- how much explained variance belongs to fertility vs.
     poverty, uninsurance, region, and race (RQ5).

Run from the repository root after src/data_cleaning.py:
    python3 src/robustness.py
"""

import warnings
from itertools import combinations
from math import factorial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from analysis import (
    COVARIATES, MULTIVARIABLE_FORMULA, PRIMARY_RACES,
    _fit_wls, association_sample, load_master,
)
from harmonization import STATE_CENTROID
from plotting import set_style

# numpy 2.0 linked against macOS Accelerate emits spurious "divide by zero /
# overflow / invalid value encountered in matmul" RuntimeWarnings even for
# well-conditioned inputs (reproducible with random matrices). Results are
# checked for finiteness explicitly below instead.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

RANDOM_SEED = 42
N_BOOT = 9999
N_PERM = 9999
K_NEIGHBORS = 5
TERM = "teen_fertility_rate"

LADDER_FORMULAS = {
    "M1": "dropout_rate ~ teen_fertility_rate",
    "M3": MULTIVARIABLE_FORMULA,
    "M4": MULTIVARIABLE_FORMULA + " + C(race_ethnicity)",
    "M5": "dropout_rate ~ teen_fertility_rate + C(race_ethnicity) + C(fips)",
}


def _wls(formula: str, sub: pd.DataFrame):
    return smf.wls(formula, data=sub, weights=sub["weight"]).fit()


# --- 1. Oster coefficient stability ----------------------------------------

def oster_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Oster (2019) approximate bias-adjusted coefficient and the delta at
    which the fertility coefficient would be zero, comparing M1 (no
    controls) against each controlled model. Uses weighted R^2 since every
    model is WLS. R_max is reported at Oster's recommended 1.3 x R_controlled
    (capped at 1) and at the conservative bound R_max = 1.

    delta = 1 means unobservables would need to be as important as all the
    observed controls together; |delta| > 1 is the conventional threshold for
    a robust result. beta_star is the coefficient after adjusting for
    proportional selection with delta = 1."""
    sub = association_sample(df)
    base = _wls(LADDER_FORMULAS["M1"], sub)
    b0, r0 = base.params[TERM], base.rsquared
    rows = []
    for label in ["M3", "M4", "M5"]:
        m = _wls(LADDER_FORMULAS[label], sub)
        b1, r1 = m.params[TERM], m.rsquared
        for rmax_rule, rmax in [("1.3 x R_controlled", min(1.0, 1.3 * r1)), ("1.0", 1.0)]:
            movement = (b0 - b1) * (rmax - r1)
            delta = b1 * (r1 - r0) / movement if movement != 0 else np.inf
            beta_star = b1 - (b0 - b1) * (rmax - r1) / (r1 - r0)
            rows.append({
                "controlled_model": label, "rmax_rule": rmax_rule, "r_max": rmax,
                "beta_uncontrolled": b0, "r2_uncontrolled": r0,
                "beta_controlled": b1, "r2_controlled": r1,
                "delta_for_beta_zero": delta, "beta_star_delta_1": beta_star,
                "robust_delta_gt_1": abs(delta) > 1,
            })
    return pd.DataFrame(rows)


# --- 2. Wild cluster bootstrap ---------------------------------------------

_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _cluster_t(X, y_mat, k, cluster_idx, n_clusters, bread):
    """CR1 cluster-robust t-statistic for coefficient k, vectorized over
    the columns of y_mat (one column per bootstrap draw)."""
    n, p = X.shape
    beta = bread @ X.T @ y_mat
    resid = y_mat - X @ beta
    z = X @ bread[k]
    scores = np.zeros((n_clusters, y_mat.shape[1]))
    np.add.at(scores, cluster_idx, z[:, None] * resid)
    c = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - p))
    se = np.sqrt(c * (scores ** 2).sum(axis=0))
    if not (np.all(np.isfinite(beta[k])) and np.all(np.isfinite(se))):
        raise FloatingPointError("non-finite bootstrap statistic")
    return beta[k] / se, beta[k], se


def wild_cluster_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    """Restricted (null-imposed) wild cluster bootstrap-t for H0: fertility
    coefficient = 0, Webb six-point weights drawn per state, on the
    whitened (WLS) model. Also reports a 95% CI by inverting the bootstrap-t
    distribution of the unrestricted fit."""
    sub = association_sample(df)
    rng = np.random.default_rng(RANDOM_SEED)
    codes, cluster_idx = np.unique(sub["fips"].values, return_inverse=True)
    G = len(codes)
    rows = []
    for label in ["M3", "M4", "M5"]:
        m = _wls(LADDER_FORMULAS[label], sub)
        X = m.model.wexog
        y = m.model.wendog
        k = list(m.params.index).index(TERM)
        bread = np.linalg.pinv(X.T @ X)

        t_obs, b_obs, se_obs = _cluster_t(X, y[:, None], k, cluster_idx, G, bread)
        t_obs, b_obs, se_obs = t_obs[0], b_obs[0], se_obs[0]

        # Restricted fit: drop the fertility column, i.e. impose beta_k = 0.
        X_r = np.delete(X, k, axis=1)
        beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
        fit_r = X_r @ beta_r
        u_r = y - fit_r
        v = _WEBB[rng.integers(0, 6, size=(G, N_BOOT))][cluster_idx]
        t_boot, _, _ = _cluster_t(X, fit_r[:, None] + u_r[:, None] * v, k, cluster_idx, G, bread)
        p_wild = (np.abs(t_boot) >= abs(t_obs)).mean()

        # Unrestricted bootstrap-t for a CI.
        fit_u = X @ (bread @ X.T @ y)
        u_u = y - fit_u
        v2 = _WEBB[rng.integers(0, 6, size=(G, N_BOOT))][cluster_idx]
        t_u, b_u, se_u = _cluster_t(X, fit_u[:, None] + u_u[:, None] * v2, k, cluster_idx, G, bread)
        t_centered = (b_u - b_obs) / se_u
        q_lo, q_hi = np.percentile(t_centered, [2.5, 97.5])

        sm_fit = _fit_wls(LADDER_FORMULAS[label], sub)
        rows.append({
            "model": label, "n": len(sub), "n_clusters": G,
            "fertility_coef": b_obs, "cluster_se_CR1": se_obs, "t_stat": t_obs,
            "p_asymptotic_cluster": sm_fit.pvalues[TERM],
            "p_wild_cluster_bootstrap": p_wild,
            "ci_low_bootstrap_t": b_obs - q_hi * se_obs,
            "ci_high_bootstrap_t": b_obs - q_lo * se_obs,
            "n_boot": N_BOOT, "weights": "Webb 6-point, null imposed",
        })
    return pd.DataFrame(rows)


# --- 3. Spatial dependence ---------------------------------------------------

def _knn_weights(states: list[str], k: int) -> np.ndarray:
    lat = np.radians([STATE_CENTROID[s][0] for s in states])
    lon = np.radians([STATE_CENTROID[s][1] for s in states])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * np.arcsin(np.sqrt(a))
    np.fill_diagonal(dist, np.inf)
    W = np.zeros_like(dist)
    for i in range(len(states)):
        W[i, np.argsort(dist[i])[:k]] = 1.0
    return W / W.sum(axis=1, keepdims=True)


def _morans_i(x: np.ndarray, W: np.ndarray) -> float:
    z = x - x.mean()
    return (len(x) / W.sum()) * (z @ W @ z) / (z @ z)


def spatial_moran(df: pd.DataFrame) -> pd.DataFrame:
    """Moran's I of state-level residuals (weight-averaged across a state's
    race cells) for M3 and M4, k-nearest-neighbor weights on state
    centroids, permutation p-value (two-sided). Also run on the White-only
    residuals, the one group observed in all 51 states. A non-significant I
    means the clustered-by-state inference is not undermined by spillover
    between neighboring states."""
    sub = association_sample(df)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for label in ["M3", "M4", "M5"]:
        m = _wls(LADDER_FORMULAS[label], sub)
        r = sub.assign(resid=m.resid.values)
        variants = {"White cells only": r[r["race_ethnicity"] == "White"].set_index("state")["resid"]}
        if label != "M5":
            # Under state FE the weighted state-mean residual is zero by
            # construction, so only the within-state (White) residual is
            # informative for M5.
            variants["state mean residual (weighted)"] = r.groupby("state").apply(
                lambda g: np.average(g["resid"], weights=g["weight"]), include_groups=False)
        for variant, series in variants.items():
            states = list(series.index)
            W = _knn_weights(states, K_NEIGHBORS)
            x = series.values
            i_obs = _morans_i(x, W)
            perm = np.array([_morans_i(rng.permutation(x), W) for _ in range(N_PERM)])
            expected = -1 / (len(x) - 1)
            p = (np.abs(perm - expected) >= abs(i_obs - expected)).mean()
            rows.append({"model": label, "residual": variant, "n_states": len(x),
                         "morans_I": i_obs, "expected_I": expected, "p_permutation": p,
                         "weights": f"{K_NEIGHBORS}-nearest neighbors, row-standardized"})
    return pd.DataFrame(rows)


def _cell_distance_km(states: pd.Series) -> np.ndarray:
    lat = np.radians(states.map(lambda s: STATE_CENTROID[s][0]).values)
    lon = np.radians(states.map(lambda s: STATE_CENTROID[s][1]).values)
    a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def conley_standard_errors(df: pd.DataFrame, cutoffs_km=(500, 1000, 1500)) -> pd.DataFrame:
    """Conley (1999) spatial-HAC standard errors for the fertility
    coefficient, on the whitened WLS model, Bartlett kernel
    w_ij = max(0, 1 - d_ij / cutoff). Cells in the same state have
    d = 0 and weight 1, so this nests clustering by state and additionally
    allows residual correlation between nearby states -- the remedy for the
    spatial autocorrelation found by spatial_moran()."""
    sub = association_sample(df)
    dist = _cell_distance_km(sub["state"])
    rows = []
    for label in ["M3", "M4", "M5"]:
        m = _wls(LADDER_FORMULAS[label], sub)
        X, u = m.model.wexog, m.model.wendog - m.model.wexog @ m.params.values
        k = list(m.params.index).index(TERM)
        bread = np.linalg.pinv(X.T @ X)
        scores = X * u[:, None]
        n, p = X.shape
        clustered = _fit_wls(LADDER_FORMULAS[label], sub)
        rows.append({"model": label, "se_type": "clustered by state (CR1)", "cutoff_km": 0,
                     "fertility_coef": m.params[TERM], "se": clustered.bse[TERM],
                     "p_value": clustered.pvalues[TERM]})
        for cutoff in cutoffs_km:
            K = np.clip(1 - dist / cutoff, 0, None)
            V = bread @ (scores.T @ K @ scores) @ bread * n / (n - p)
            se = np.sqrt(V[k, k])
            z = m.params[TERM] / se
            rows.append({"model": label, "se_type": "Conley spatial HAC (Bartlett)", "cutoff_km": cutoff,
                         "fertility_coef": m.params[TERM], "se": se,
                         "p_value": 2 * stats.norm.sf(abs(z))})
    return pd.DataFrame(rows)


# --- 4. Leave-one-out --------------------------------------------------------

def leave_one_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit M3, M4, M5 dropping one state at a time, and M3, M4 dropping
    one race group at a time (M5 needs every group for within-state
    contrasts but is still refit). Returns (detail, summary)."""
    sub = association_sample(df)
    detail = []
    for label in ["M3", "M4", "M5"]:
        for state in sorted(sub["state"].unique()):
            m = _fit_wls(LADDER_FORMULAS[label], sub[sub["state"] != state].reset_index(drop=True))
            detail.append({"model": label, "dropped": "state", "dropped_unit": state,
                           "fertility_coef": m.params[TERM], "p_value": m.pvalues[TERM]})
        for race in PRIMARY_RACES:
            m = _fit_wls(LADDER_FORMULAS[label], sub[sub["race_ethnicity"] != race].reset_index(drop=True))
            detail.append({"model": label, "dropped": "race group", "dropped_unit": race,
                           "fertility_coef": m.params[TERM], "p_value": m.pvalues[TERM]})
    detail = pd.DataFrame(detail)
    full = {label: _fit_wls(LADDER_FORMULAS[label], sub).params[TERM] for label in ["M3", "M4", "M5"]}

    summary = []
    for (label, dropped), g in detail.groupby(["model", "dropped"], sort=False):
        lo, hi = g.loc[g["fertility_coef"].idxmin()], g.loc[g["fertility_coef"].idxmax()]
        summary.append({
            "model": label, "dropped": dropped, "n_refits": len(g),
            "full_sample_coef": full[label],
            "min_coef": lo["fertility_coef"], "min_coef_when_dropping": lo["dropped_unit"],
            "max_coef": hi["fertility_coef"], "max_coef_when_dropping": hi["dropped_unit"],
            "max_p_value": g["p_value"].max(),
            "max_p_when_dropping": g.loc[g["p_value"].idxmax(), "dropped_unit"],
            "n_refits_p_ge_0.05": int((g["p_value"] >= 0.05).sum()),
        })
    return detail, pd.DataFrame(summary)


def figure_leave_one_state_out(detail: pd.DataFrame, out_path: str = "figures/phase3_leave_one_state_out.png"):
    """Fertility coefficient after dropping each state, per model, against
    the full-sample estimate -- shows at a glance whether any single state
    drives the result."""
    set_style()
    sub_full = association_sample(load_master())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, label in zip(axes, ["M3", "M4", "M5"]):
        d = detail[(detail["model"] == label) & (detail["dropped"] == "state")].sort_values("fertility_coef")
        full = _fit_wls(LADDER_FORMULAS[label], sub_full).params[TERM]
        colors = np.where(d["p_value"] < 0.05, "#4C72B0", "#C44E52")
        ax.scatter(range(len(d)), d["fertility_coef"], c=colors, s=14)
        ax.axhline(full, color="black", linewidth=1, linestyle="--")
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
        for _, row in pd.concat([d.head(1), d.tail(1)]).iterrows():
            ax.annotate(row["dropped_unit"], (list(d.index).index(row.name), row["fertility_coef"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_xlabel("states, sorted by coefficient")
    axes[0].set_ylabel("Fertility coefficient (pp per +1/1,000)")
    fig.suptitle("Leave-one-state-out fertility coefficient (dashed = full sample; red = p >= 0.05)",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.savefig(out_path)
    plt.close(fig)


# --- 5. Shapley R^2 decomposition --------------------------------------------

PREDICTOR_GROUPS = {
    "teen_fertility_rate": "teen_fertility_rate",
    "poverty_rate": "poverty_rate",
    "uninsured_rate": "uninsured_rate",
    "census_region": "C(census_region)",
    "race_ethnicity": "C(race_ethnicity)",
}


def shapley_r2(df: pd.DataFrame) -> pd.DataFrame:
    """LMG / Shapley decomposition of M4's weighted R^2: each predictor
    group's average marginal contribution to R^2 over all orderings. Shares
    sum to the full-model R^2, so -- unlike attenuation percentages -- they
    answer "how much of the explained variation is fertility vs. context
    vs. race" without depending on entry order."""
    sub = association_sample(df)
    names = list(PREDICTOR_GROUPS)
    n = len(names)
    r2_cache = {(): 0.0}
    for size in range(1, n + 1):
        for subset in combinations(names, size):
            rhs = " + ".join(PREDICTOR_GROUPS[g] for g in subset)
            r2_cache[subset] = _wls(f"dropout_rate ~ {rhs}", sub).rsquared

    rows = []
    for g in names:
        others = [x for x in names if x != g]
        value = 0.0
        for size in range(len(others) + 1):
            for subset in combinations(others, size):
                with_g = tuple(x for x in names if x in subset or x == g)
                weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                value += weight * (r2_cache[with_g] - r2_cache[tuple(subset)])
        rows.append({"predictor_group": g, "shapley_r2": value})
    out = pd.DataFrame(rows)
    total = r2_cache[tuple(names)]
    out["share_of_explained_pct"] = 100 * out["shapley_r2"] / total
    out["full_model_r2"] = total
    return out.sort_values("shapley_r2", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = load_master()
    pd.set_option("display.width", 200)

    oster = oster_delta(df)
    oster.to_csv("tables/phase3_sensitivity_oster.csv", index=False)
    print("1. Oster coefficient stability:")
    print(oster.round(3).to_string(index=False))

    wild = wild_cluster_bootstrap(df)
    wild.to_csv("tables/phase3_wild_cluster_bootstrap.csv", index=False)
    print("\n2. Wild cluster bootstrap:")
    print(wild.round(4).to_string(index=False))

    moran = spatial_moran(df)
    moran.to_csv("tables/phase3_spatial_moran.csv", index=False)
    print("\n3. Spatial dependence (Moran's I):")
    print(moran.round(4).to_string(index=False))
    conley = conley_standard_errors(df)
    conley.to_csv("tables/phase3_conley_se.csv", index=False)
    print("\n3b. Conley spatial-HAC standard errors:")
    print(conley.round(4).to_string(index=False))

    loo_detail, loo_summary = leave_one_out(df)
    loo_detail.to_csv("tables/phase3_leave_one_out_detail.csv", index=False)
    loo_summary.to_csv("tables/phase3_leave_one_out_summary.csv", index=False)
    figure_leave_one_state_out(loo_detail)
    print("\n4. Leave-one-out:")
    print(loo_summary.round(4).to_string(index=False))

    shap_r2 = shapley_r2(df)
    shap_r2.to_csv("tables/phase3_r2_decomposition.csv", index=False)
    print("\n5. Shapley R^2 decomposition (M4):")
    print(shap_r2.round(4).to_string(index=False))
