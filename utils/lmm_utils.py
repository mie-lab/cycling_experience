import logging
import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.regression.mixed_linear_model import MixedLMResults
import statsmodels.formula.api as smf

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
        maxiter=1000,
        verbose=True

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
