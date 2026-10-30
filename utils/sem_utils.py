import logging
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from semopy import Model
from sklearn.decomposition import PCA
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

import constants as c

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

INDICES = ["StaticIndex", "VisualIndex", "DynamicIndex"]
INTERACTIONS = {"M6": "Static_x_Dynamic", "M7": "Visual_x_Dynamic", "M8": "Static_x_Visual"}
PREDICTOR_COVS = ["StaticIndex ~~ VisualIndex + DynamicIndex", "VisualIndex ~~ DynamicIndex"]

# M5: the static channel reaches affect only through the visual and dynamic channels
MEDIATION_MODEL = "\n".join([
    "VisualIndex ~ StaticIndex",
    "DynamicIndex ~ StaticIndex",
    "valence ~ VisualIndex + DynamicIndex",
    "arousal ~ VisualIndex + DynamicIndex",
    "valence ~~ arousal",
    "VisualIndex ~~ DynamicIndex",
])


# -------------------------------------------------------------------
# PREPROCESSING AND INDICES
# -------------------------------------------------------------------
def centre(df, cols):
    """Within-participant mean-centring."""
    return df[cols] - df.groupby(c.PARTICIPANT_ID)[cols].transform("mean")


def prepare(df, log_vars, responses, features):
    """log(x + 1), within-participant centring of responses, z-scores of features."""
    d = df.copy()
    d[log_vars] = np.log1p(d[log_vars].clip(lower=0))
    d[responses] = centre(d, responses)
    d[features] = d[features].astype(float).apply(stats.zscore)
    return d


def pc1_index(df, cols, ref):
    """First principal component of `cols`, signed so that it rises with `ref`."""
    pca = PCA(n_components=1, svd_solver="full").fit(df[cols])
    loadings = pd.Series(pca.components_[0], index=cols)
    sign = 1 if loadings[ref] >= 0 else -1

    score = pd.Series(sign * pca.transform(df[cols])[:, 0], index=df.index)
    return score, sign * loadings, pca.explained_variance_ratio_[0]


def channel_indices(df, channels, scenario):
    """The three channel indices of one scenario and their interactions, all centred
    within participant."""
    d = df.copy()
    loadings = []
    for name, (cols, ref) in zip(INDICES, channels):
        d[name], load, var = pc1_index(d, cols, ref)
        for f in cols:
            loadings.append({
                "scenario": scenario, "index": name, "feature": f,
                "loading": load[f], "variance_explained": var,
            })

    d[INDICES] = centre(d, INDICES)
    d["Static_x_Dynamic"] = d["StaticIndex"] * d["DynamicIndex"]
    d["Visual_x_Dynamic"] = d["VisualIndex"] * d["DynamicIndex"]
    d["Static_x_Visual"] = d["StaticIndex"] * d["VisualIndex"]
    d[list(INTERACTIONS.values())] = centre(d, list(INTERACTIONS.values()))
    return d, loadings


# -------------------------------------------------------------------
# PATH MODELS
# -------------------------------------------------------------------
def direct_model(outcomes, interaction=None):
    """M1-M4, or M6-M8 when an interaction term is given."""
    rhs = " + ".join(INDICES + ([interaction] if interaction else []))
    lines = [f"{y} ~ {rhs}" for y in outcomes]
    lines += [f"{a} ~~ {b}" for a, b in combinations(outcomes, 2)]
    return "\n".join(lines + PREDICTOR_COVS)


def fit_indices(model):
    """ML fit indices from the full log-likelihood (semopy's own LogLik and DoF are not
    the standard ones). Saturated models get CFI = TLI = 1 and RMSEA = 0."""
    n, s = model.n_samples, model.mx_cov
    sigma = model.calc_sigma()[0]
    p = s.shape[0]

    logdet_sigma, logdet_s = np.linalg.slogdet(sigma)[1], np.linalg.slogdet(s)[1]
    trace = np.trace(s @ np.linalg.inv(sigma))
    loglik = -n / 2 * (p * np.log(2 * np.pi) + logdet_sigma + trace)
    chi2 = max(n * (logdet_sigma + trace - logdet_s - p), 0)

    # Exogenous (co)variances count as parameters
    est = model.inspect()
    exo = model.vars["exogenous"] - model.vars["latent"]
    free_exo = ((est["op"] == "~~") & est["lval"].isin(exo) & est["rval"].isin(exo)).sum()
    k = len(model.param_vals) + len(exo) * (len(exo) + 1) // 2 - free_exo
    df = p * (p + 1) // 2 - k

    # Independence model, for CFI and TLI
    chi2_b, df_b = n * (np.log(np.diag(s)).sum() - logdet_s), p * (p - 1) // 2
    saturated = df == 0

    return {
        "df": df,
        "chi2": chi2,
        "chi2_p": np.nan if saturated else stats.chi2.sf(chi2, df),
        "CFI": 1 - max(chi2 - df, 0) / max(chi2_b - df_b, chi2 - df, 1e-12),
        "TLI": 1.0 if saturated else (chi2_b / df_b - chi2 / df) / (chi2_b / df_b - 1),
        "RMSEA": 0.0 if saturated else np.sqrt(max(chi2 - df, 0) / (df * (n - 1))),
        "AIC": 2 * k - 2 * loglik,
        "BIC": np.log(n) * k - 2 * loglik,
    }


def fit_path_model(desc, data):
    """Fit a path model with semopy (ML). Each equation is refitted with OLS for R2 and
    clip-clustered standard errors. Returns paths, R2, residuals and fit indices."""
    model = Model(desc)
    model.fit(data)
    est = model.inspect(std_est=True)
    paths = est.loc[est["op"] == "~", ["lval", "rval", "Estimate", "Est. Std"]].reset_index(drop=True)

    r2, resid, rows = {}, {}, []
    for y, grp in paths.groupby("lval", sort=False):
        fit = ols(f"{y} ~ {' + '.join(grp['rval'])}", data).fit(
            cov_type="cluster", cov_kwds={"groups": data[c.VIDEO_ID_COL]}, use_t=True
        )
        r2[y], resid[y] = fit.rsquared, fit.resid
        for x in grp["rval"]:
            rows.append({
                "lval": y, "rval": x,
                "b_ols": fit.params[x], "se_cluster": fit.bse[x], "p_cluster": fit.pvalues[x],
            })
    paths = paths.merge(pd.DataFrame(rows), on=["lval", "rval"])

    gap = (paths["Estimate"] - paths["b_ols"]).abs().max()
    if gap > 1e-3:
        log.warning(f"semopy and OLS estimates differ by up to {gap:.4f}")

    return paths, r2, pd.DataFrame(resid), fit_indices(model)


def joint_static_test(data, outcomes):
    """Clip-clustered F-test that StaticIndex has no path to any outcome (M5 vs M4),
    with the outcome equations stacked."""
    long = pd.concat([data.assign(_y=data[y], _out=y) for y in outcomes], ignore_index=True)
    rhs = " + ".join(f"C(_out):{x}" for x in INDICES)
    fit = ols(f"_y ~ 0 + C(_out) + {rhs}", long).fit(
        cov_type="cluster", cov_kwds={"groups": long[c.VIDEO_ID_COL]}, use_t=True
    )

    names = list(fit.params.index)
    R = np.zeros((len(outcomes), len(names)))
    for i, y in enumerate(outcomes):
        R[i, names.index(f"C(_out)[{y}]:StaticIndex")] = 1

    test = fit.f_test(R)
    return {
        "F": float(np.squeeze(test.fvalue)),
        "df_num": int(test.df_num),
        "df_denom": int(test.df_denom),
        "p": float(test.pvalue),
    }


def holm(p):
    """Holm-adjusted p-values."""
    return multipletests(p, method="holm")[1]


# -------------------------------------------------------------------
# DESCRIPTIVES
# -------------------------------------------------------------------
def rating_reliability(df, y):
    """ICC(1) of single ratings and ICC(k) of clip means (one-way ANOVA)."""
    aov = sm.stats.anova_lm(ols(f"{y} ~ C({c.VIDEO_ID_COL})", df).fit(), typ=1)
    ms_b, ms_w = aov["mean_sq"].iloc[0], aov["mean_sq"].iloc[1]
    k = stats.hmean(df.groupby(c.VIDEO_ID_COL)[y].count())
    return (ms_b - ms_w) / (ms_b + (k - 1) * ms_w), (ms_b - ms_w) / ms_b


# -------------------------------------------------------------------
# HELD-OUT VALIDATION
# -------------------------------------------------------------------
def loco_errors(X, y, clips):
    """Squared errors when each clip is predicted by OLS fitted on the other clips."""
    X = np.column_stack([np.ones(len(y)), X])
    pred = np.empty(len(y))
    for v in np.unique(clips):
        test = clips == v
        beta = np.linalg.lstsq(X[~test], y[~test], rcond=None)[0]
        pred[test] = X[test] @ beta
    return (y - pred) ** 2


def heldout_gain(d, y, base, gis, ground, n_boot=2000, seed=1):
    """Held-out R2 with the GIS version, the ground-level version or both, and the gain
    of adding the other version. CIs resample clips."""
    clips = d[c.VIDEO_ID_COL].to_numpy()
    ids = np.unique(clips)
    draws = np.random.default_rng(seed).integers(0, len(ids), (n_boot, len(ids)))
    t = d[y].to_numpy()

    def per_clip(errors):
        return pd.Series(errors).groupby(clips).sum().reindex(ids).to_numpy()

    sst = per_clip((t - t.mean()) ** 2)
    sse = {
        name: per_clip(loco_errors(d[base + cols].to_numpy(), t, clips))
        for name, cols in [("gis", gis), ("ground", ground), ("both", gis + ground)]
    }
    r2 = {f"cv_r2_{k}": 1 - v.sum() / sst.sum() for k, v in sse.items()}

    rows = []
    for test, single in [("ground beyond GIS", "gis"), ("GIS beyond ground", "ground")]:
        diff = sse[single] - sse["both"]
        boot = diff[draws].sum(axis=1) / sst[draws].sum(axis=1)
        rows.append({
            "test": test, **r2,
            "delta_cv_r2": diff.sum() / sst.sum(),
            "ci_low": np.quantile(boot, .025),
            "ci_high": np.quantile(boot, .975),
        })
    return rows


def heldout_comparison(X_a, X_b, t, clips, n_boot=2000, seed=1):
    """Held-out R2 of two models and the gain of model b over model a. CIs resample clips."""
    ids = np.unique(clips)
    draws = np.random.default_rng(seed).integers(0, len(ids), (n_boot, len(ids)))

    def per_clip(errors):
        return pd.Series(errors).groupby(clips).sum().reindex(ids).to_numpy()

    sst = per_clip((t - t.mean()) ** 2)
    sse_a, sse_b = (per_clip(loco_errors(X, t, clips)) for X in (X_a, X_b))
    diff = sse_a - sse_b
    boot = diff[draws].sum(axis=1) / sst[draws].sum(axis=1)

    return {
        "cv_r2_a": 1 - sse_a.sum() / sst.sum(),
        "cv_r2_b": 1 - sse_b.sum() / sst.sum(),
        "delta_cv_r2": diff.sum() / sst.sum(),
        "ci_low": np.quantile(boot, .025),
        "ci_high": np.quantile(boot, .975),
    }
