"""REML crossed-intercept models with explicitly asymptotic Wald inference."""
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import cho_factor, cho_solve, qr
from scipy.stats import norm, spearmanr
import statsmodels
from statsmodels.regression.mixed_linear_model import MixedLM, VCSpec
from statsmodels.stats.multitest import multipletests
from scipy.sparse import csr_matrix


def fit_chemical(data, model_type, sd_ratio_limit=3.0, resid_cor_limit=0.3):
    """One chemical, original response units, categorical date/genus background.

    Constant groups plus two variance components implement crossed (not nested)
    animal and strain intercepts. Chemical standardization is numerical only.
    """
    d = data.loc[np.isfinite(data.response) & np.isfinite(data.chemical)].copy()
    info = d[["sample_id", "date", "genus", "chemical"]].drop_duplicates()
    out = dict(status="not_estimable", reason="", n_observations=len(d),
               n_strains=d.sample_id.nunique(), n_animals=d.animal_id.nunique(),
               n_dates=d.date.nunique(), n_genera=d.genus.nunique(),
               background_columns=np.nan, background_rank=np.nan,
               chemical_remaining_sd_fraction=np.nan, chemical_remaining_range=np.nan,
               chemical_min=np.nan, chemical_max=np.nan, chemical_iqr=np.nan,
               n_genera_with_chemical_variation=np.nan, max_strain_design_share=np.nan,
               beta=np.nan, se_wald=np.nan, z_wald=np.nan, ci_low=np.nan, ci_high=np.nan,
               p_wald=np.nan, effect_iqr=np.nan, converged=False, singular=False,
               animal_variance=np.nan, strain_variance=np.nan, residual_variance=np.nan,
               abs_residual_fitted_spearman=np.nan, residual_sd_ratio=np.nan,
               heteroskedasticity_flag=False, warnings="")
    residual_table = None
    if model_type not in ("overall", "within_genus"):
        raise ValueError("Unknown model type")
    try:
        if out["n_strains"] < 3 or out["n_animals"] < 2 or len(d) <= out["n_strains"]:
            raise ValueError("Insufficient strain/animal replication")
        strain_x = info.drop_duplicates("sample_id").chemical
        out.update(chemical_min=strain_x.min(), chemical_max=strain_x.max(),
                   chemical_iqr=strain_x.quantile(.75) - strain_x.quantile(.25),
                   n_genera_with_chemical_variation=int(info.groupby("genus").chemical.agg(
                       lambda x: x.max() - x.min()).gt(1e-10).sum()))
        scale = strain_x.std(ddof=1)
        if not np.isfinite(scale) or scale < 1e-12:
            out.update(status="not_identifiable", reason="Constant chemical profile")
            return out, None
        center = strain_x.mean()
        d["x"] = (d.chemical - center) / scale
        terms = ["date"] + (["genus"] if model_type == "within_genus" else [])
        # Redundant date/genus columns are removed by rank only, independently of y.
        bg_info = pd.get_dummies(info[terms].astype(str), drop_first=True, dtype=float)
        bg_info.insert(0, "intercept", 1.0)
        _, triangular, pivots = qr(bg_info.to_numpy(), mode="economic", pivoting=True)
        rank = int((np.abs(np.diag(triangular)) > 1e-9 * max(1.0, np.abs(triangular).max())).sum())
        kept = bg_info.columns[pivots[:rank]]
        basis_info = bg_info[kept].to_numpy()
        x_info = (info.chemical.to_numpy() - center) / scale
        remainder = x_info - basis_info @ np.linalg.lstsq(basis_info, x_info, rcond=None)[0]
        fraction = np.std(remainder, ddof=1) / np.std(x_info, ddof=1)
        shares = pd.Series(remainder**2, index=info.sample_id).groupby(level=0).sum()
        out.update(background_columns=len(bg_info.columns), background_rank=rank,
                   chemical_remaining_sd_fraction=fraction,
                   chemical_remaining_range=np.ptp(remainder) * scale,
                   max_strain_design_share=shares.max() / shares.sum() if shares.sum() else np.nan)
        if fraction < 1e-8:
            out.update(status="not_identifiable", reason="Chemical is in the date/genus background column space")
            return out, None
        background = pd.get_dummies(d[terms].astype(str), drop_first=True, dtype=float)
        background.insert(0, "intercept", 1.0)
        X = background[kept].copy()
        X["chemical_scaled"] = d.x
        names, column_names, matrices = [], [], []
        for label, key in [("animal", "animal_id"), ("strain", "sample_id")]:
            codes, levels = pd.factorize(d[key], sort=True)
            matrix = csr_matrix((np.ones(len(d)), (np.arange(len(d)), codes)), shape=(len(d), len(levels)))
            names.append(label)
            column_names.append([list(map(str, levels))])
            matrices.append([matrix])
        vc = VCSpec(names, column_names, matrices)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = MixedLM(d.response, X, groups=np.ones(len(d)),
                            exog_re=np.empty((len(d), 0)), exog_vc=vc)
            fit = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
            variances = dict(zip(model.exog_vc.names, fit.vcomp))
            out.update(converged=bool(fit.converged), animal_variance=variances["animal"],
                       strain_variance=variances["strain"], residual_variance=fit.scale,
                       singular=bool(np.any(np.sqrt(fit.vcomp / fit.scale) < 1e-4)),
                       beta=fit.fe_params["chemical_scaled"] / scale,
                       se_wald=fit.bse_fe["chemical_scaled"] / scale)
            out["effect_iqr"] = out["beta"] * out["chemical_iqr"]
            if np.isfinite(out["se_wald"]) and out["se_wald"] > 0:
                out["z_wald"] = out["beta"] / out["se_wald"]
                out["p_wald"] = 2 * norm.sf(abs(out["z_wald"]))
                out["ci_low"] = out["beta"] - norm.ppf(.975) * out["se_wald"]
                out["ci_high"] = out["beta"] + norm.ppf(.975) * out["se_wald"]
            # statsmodels 0.14 sparse crossed designs can fail in fittedvalues.
            # Gaussian BLUP residual = sigma^2 V^-1 (y-X beta), using the same fit.
            covariance = fit.scale * np.eye(len(d))
            for variance, components in zip(fit.vcomp, matrices):
                Z = components[0]
                covariance += variance * (Z @ Z.T).toarray()
            marginal_residual = d.response.to_numpy() - X.to_numpy() @ fit.fe_params.to_numpy()
            residual = fit.scale * cho_solve(cho_factor(covariance, lower=True), marginal_residual)
            fitted = d.response.to_numpy() - residual
            out["abs_residual_fitted_spearman"] = spearmanr(np.abs(residual), fitted).statistic
            bins = pd.qcut(fitted, 4, duplicates="drop")
            sds = pd.Series(residual).groupby(bins, observed=True).std().dropna()
            out["residual_sd_ratio"] = sds.max() / sds.min() if len(sds) >= 2 else np.nan
            out["heteroskedasticity_flag"] = bool(
                abs(out["abs_residual_fitted_spearman"]) >= resid_cor_limit or
                out["residual_sd_ratio"] >= sd_ratio_limit)
            residual_table = pd.DataFrame({"observation_id": d.observation_id.to_numpy(), "fitted": fitted,
                                           "residual": residual, "standardized_residual": residual / np.sqrt(fit.scale)})
        out["warnings"] = "; ".join(dict.fromkeys(str(w.message) for w in caught))
        out["status"] = ("not_converged" if not out["converged"] else
                         "wald_failed" if not np.isfinite(out["p_wald"]) else
                         "singular" if out["singular"] else
                         "residual_review" if out["heteroskedasticity_flag"] else
                         "warning_review" if out["warnings"] else "ok")
    except Exception as error:
        out.update(status="fit_error", reason=f"{type(error).__name__}: {error}")
    return out, residual_table


def screen_chemicals(folder):
    folder = Path(folder)
    data = pd.read_csv(folder / "animal_input.csv", dtype={"date": str})
    chemicals = pd.read_csv(folder / "chemical_input.csv", index_col="sample_id")
    settings = pd.read_csv(folder / "settings.csv").set_index("parameter").value
    method = settings["fdr_method"]
    if method not in ("BH", "BY"):
        raise ValueError("FDR method must be BH or BY")
    results, residuals = [], []
    total = 2 * len(chemicals.columns)
    for feature in chemicals:
        data["chemical"] = data.sample_id.map(chemicals[feature])
        for model_type in ("overall", "within_genus"):
            result, residual = fit_chemical(
                data, model_type, float(settings["residual_sd_ratio_limit"]),
                float(settings["abs_residual_fitted_cor_limit"]))
            results.append(dict(feature_id=feature, model=model_type, **result))
            if residual is not None:
                residuals.append(residual.assign(feature_id=feature, model=model_type))
            if len(results) % 20 == 0 or len(results) == total:
                print(f"Wald screening: {len(results)}/{total} fits", flush=True)
                pd.DataFrame(results).to_csv(folder / "results_partial.csv", index=False)
    result = pd.DataFrame(results)
    usable = (np.isfinite(result.p_wald) & result.converged &
              ~result.status.isin(["fit_error", "wald_failed", "not_estimable", "not_identifiable"]))
    # All declared tests remain in the family, including errors/nonidentifiable items.
    p = result.p_wald.where(usable, 1.0)
    result["q_wald"] = multipletests(p, method="fdr_by" if method == "BY" else "fdr_bh")[1]
    result.loc[~usable, "q_wald"] = np.nan
    result["fdr_family_size"], result["fdr_method"] = total, method
    result["nominal_fdr_hit"] = result.q_wald.le(float(settings["alpha"]))
    result["passes_model_review"] = result.status.eq("ok")
    result["candidate"] = result.nominal_fdr_hit & result.passes_model_review
    result.to_csv(folder / "model_results.csv", index=False)
    pd.concat(residuals, ignore_index=True).to_csv(folder / "residuals.csv", index=False) if residuals else pd.DataFrame(
        columns=["observation_id", "fitted", "residual", "standardized_residual", "feature_id", "model"]
    ).to_csv(folder / "residuals.csv", index=False)
    (folder / "python_versions.txt").write_text(
        f"Python {platform.python_version()}\nnumpy {np.__version__}\npandas {pd.__version__}\n"
        f"scipy {scipy.__version__}\nstatsmodels {statsmodels.__version__}\n", encoding="utf-8")
    return result
