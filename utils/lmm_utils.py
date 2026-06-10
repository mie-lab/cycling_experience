import logging
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import chi2
import constants as c
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.regression.mixed_linear_model import MixedLMResults
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def lmm_contrast(fitted, contrast: dict, alpha: float = 0.05) -> dict:
    """
    Test a linear combination of fixed effects from a fitted statsmodels MixedLM.

    :param fitted: a fitted MixedLMResults object.
    :param contrast: {coefficient_name: weight}. Names must match fitted.fe_params index
                     exactly (e.g. "C(pair)[T.NB \u2192 B]"). Coefficients not listed get weight 0.
    :param alpha: for the (1-alpha) CI.
    :return: dict with estimate, se, z, p, ci_low, ci_high.

    Example: order effect (B\u2192NB vs NB\u2192B) when reference level is "B \u2192 B":
        lmm_contrast(model, {"C(pair)[T.B \u2192 NB]": 1, "C(pair)[T.NB \u2192 B]": -1})
    """
    names = list(fitted.fe_params.index)
    L = np.zeros(len(names))
    for term, w in contrast.items():
        if term not in names:
            raise KeyError(
                f"'{term}' not in fixed effects. Available: {names}"
            )
        L[names.index(term)] = w

    beta = fitted.fe_params.values
    # fixed-effect covariance block (statsmodels stacks fe then re params)
    cov = fitted.cov_params().loc[names, names].values

    est = float(L @ beta)
    se = float(np.sqrt(L @ cov @ L))
    z = est / se if se > 0 else np.nan
    p = 2 * norm.sf(abs(z)) if se > 0 else np.nan
    crit = norm.ppf(1 - alpha / 2)

    return {
        "estimate": est, "se": se, "z": z, "p": p,
        "ci_low": est - crit * se, "ci_high": est + crit * se,
    }


def run_lmm(
        df: pd.DataFrame,
        formula: str,
        groups_col: str,
        re_formula=None,
        vc_formula=None,
        convergence_method='lbfgs',
        maxiter=1000

) -> MixedLMResults:
    """
    Fits a Linear Mixed Model (LMM) with centered categorical predictors for Familiarity (F) and Overall Experience (OE).
    :param convergence_method:
    :param groups_col: column for random effects grouping
    :param vc_formula: variance components formula models by-video variability
    :param re_formula: specifies slopes for andom effects
    :param formula: model formula
    :param df: input DataFrame
    :return: fitted LMM results
    """
    model_df = df.copy()

    model = smf.mixedlm(
        formula,
        data=model_df,
        groups=model_df[groups_col],
        vc_formula=vc_formula,
        re_formula=re_formula
    )

    results = model.fit(reml=False, method=convergence_method, maxiter=maxiter)
    log.info(results.summary())
    calculate_r2_lmm(results)

    return results


def run_bayes_logistic_lmm(
        df: pd.DataFrame,
        formula: str,
        groups_col: str
):
    # Create a copy to ensure the original DataFrame is not modified
    model_df = df.copy()
    random_effects = {'a': f'0 + C({groups_col})'}

    try:
        model = BinomialBayesMixedGLM.from_formula(
            formula=formula,
            vc_formulas=random_effects,
            data=model_df
        )

        results = model.fit_vb()

        log.info(results.summary())
        return results

    except Exception as e:
        log.info(f"\n--- BAYESIAN MODEL FAILED TO FIT ---")
        log.info(f"Formula: {formula}")
        log.info(f"Error: {e}")
        return None


"""
def calculate_r2_lmm(lmm_results):
    
    'Nakagawa & Schielzeth (2013) style R² for Gaussian LMMs.'
    # Fixed-effects variance
    mu = lmm_results.model.exog @ lmm_results.fe_params
    var_f = np.var(mu, ddof=1)

    # Random-effects variance
    var_r = float(lmm_results.vcomp.sum())

    # Residual variance
    var_e = float(lmm_results.scale)

    total = var_f + var_r + var_e
    marginal_r2 = var_f / total
    conditional_r2 = (var_f + var_r) / total

    y_true = lmm_results.model.endog
    y_pred = lmm_results.fittedvalues
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    log.info(f"Conditional R² (fixed + random effects): {conditional_r2:.3f}")
    log.info(f"RMSE: {rmse: .3f}, MAE: {mae: .3f}")

    return marginal_r2, conditional_r2, rmse, mae
"""


def calculate_r2_lmm(lmm_results):
    """Nakagawa & Schielzeth (2013) R² for Gaussian LMMs (marginal + conditional)."""
    var_f = np.var(lmm_results.model.exog @ lmm_results.fe_params, ddof=1)

    # random-effect variance: random-intercept/slopes (cov_re) + variance components (vcomp)
    var_re = float(np.trace(np.atleast_2d(lmm_results.cov_re))) if lmm_results.cov_re.size else 0.0
    var_vc = float(np.sum(lmm_results.vcomp)) if len(lmm_results.vcomp) else 0.0
    var_r = var_re + var_vc

    var_e = float(lmm_results.scale)
    total = var_f + var_r + var_e

    marginal_r2 = var_f / total
    conditional_r2 = (var_f + var_r) / total

    log.info(f"Marginal R² (fixed): {marginal_r2:.3f} | "
             f"Conditional R² (fixed+random): {conditional_r2:.3f}")
    return marginal_r2, conditional_r2


def lr_test_mixed(
        model_simple: MixedLMResults,
        model_complex: MixedLMResults,
        df_mode="len_params",
        mixture=False,
        df_mixture=1
) -> tuple:
    """
    Likelihood Ratio Test for two nested Mixed Linear Models.
    :param model_simple: simpler model (null hypothesis)
    :param model_complex: more complex model (alternative hypothesis)
    :param df_mode: method to calculate degrees of freedom difference
    :param mixture: whether to use a mixture distribution for p-value calculation
    :param df_mixture: degrees of freedom for the chi-squared distribution in the mixture
    :return: (likelihood ratio statistic, degrees of freedom difference, p-value)
    """
    ll0 = model_simple.llf
    ll1 = model_complex.llf
    lr = max(0.0, 2.0 * (ll1 - ll0))

    if df_mode == "len_params":
        df0 = len(model_simple.params)
        df1 = len(model_complex.params)
    else:
        raise ValueError("Only df_mode='len_params' implemented here.")
    df_diff = df1 - df0

    if mixture:
        # 50:50 mixture of χ²₀ and χ²_{df_mixture}
        p = 0.5 * chi2.sf(lr, df_mixture)
    else:
        p = chi2.sf(lr, df_diff)

    log.info(f"--- Likelihood Ratio Test ---")
    log.info(f"lrt: {lr:.4f}, DoFD: {df_diff}, p-value: {p:.4f}")

    return lr, df_diff, p


def show_fe_names(fitted):
    for n in fitted.fe_params.index:
        log.info(repr(n))


def run_lmm_crossed(
        df: pd.DataFrame,
        formula: str,
        vc_factors: dict,          # e.g. {"participant": "participant", "clip": "spoiler_clip"}
        convergence_method="lbfgs",
        maxiter=2000,
) -> MixedLMResults:
    """
    Fit an LMM with crossed random intercepts via the single-group / variance-components trick.
    All factors in vc_factors become independent zero-mean random-intercept families.
    """
    model_df = df.copy()
    model_df["_single_group"] = 1

    vc = {name: f"0 + C({col})" for name, col in vc_factors.items()}

    model = smf.mixedlm(
        formula,
        data=model_df,
        groups=model_df["_single_group"],
        vc_formula=vc,
    )
    results = model.fit(reml=False, method=convergence_method, maxiter=maxiter)
    log.info(results.summary())
    calculate_r2_lmm(results)
    return results

def diagnose_clip_random_effect(df, clip_col="spoiler_clip", outcome="valence",
                                participant_col=c.PARTICIPANT_ID):
    occ = df[clip_col].value_counts().sort_index()
    print(f"\n=== Occupancy: {clip_col} ===")
    print(occ.to_string())
    excl = occ.drop(labels=["baseline"], errors="ignore")
    print(f"\nDistinct levels: {df[clip_col].nunique()}  "
          f"(real spoiler clips excl. baseline: {excl.shape[0]})")
    print(f"Obs/level: min={occ.min()}, median={int(occ.median())}, max={occ.max()}")

    d = df.copy()
    d["_single_group"] = 1
    vc = {"participant": f"0 + C({participant_col})", "clip": f"0 + C({clip_col})"}
    res = smf.mixedlm(f"{outcome} ~ 1", data=d, groups=d["_single_group"],
                      vc_formula=vc).fit(reml=False, method="lbfgs", maxiter=2000)

    print(f"\nConverged: {res.converged}")
    print(f"vcomp: {dict(zip(['participant','clip'], np.round(res.vcomp, 5)))}")
    print(f"residual scale: {res.scale:.5f}")
    clip_var = float(res.vcomp[-1])
    if clip_var < 1e-4:
        print(">>> Clip variance ~0: clip RE NOT identifiable. Drop it / use measured "
              "per-segment values / treat clip as fixed nuisance.")
    else:
        print(f">>> Clip ICC ≈ {clip_var / (sum(res.vcomp) + res.scale):.3f}")
    return res