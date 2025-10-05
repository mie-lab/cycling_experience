import logging
import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.regression.mixed_linear_model import MixedLMResults
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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


def calculate_r2_lmm(lmm_results):
    """
    Nakagawa & Schielzeth (2013) style R² for Gaussian LMMs.
    """
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
