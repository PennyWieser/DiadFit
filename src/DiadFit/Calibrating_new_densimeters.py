
import pickle
import numpy as np
from scipy.odr import ODR, Model, RealData
from scipy.optimize import curve_fit
import pandas as pd

def poly_odr_centered(beta, xc):
    """ Polynomial model definition, for centered form"""
    a, b, c, d, e = beta
    return a * xc + b * xc**2 + c * xc**3 + d * xc**4 + e


def fit_densimeter(x_data, y_data, x_err, y_err, maxit=10000):
    """
    Fit the polynomial densimeter model with scipy.odr (accounts for x and y
    uncertainty), then compute a residual-scatter term for prediction
    intervals. Returns a dict with everything needed to predict later.

    x_data = Corrected Splitting from calibratoin (with error x_err)
    y_data = Preferred density value from standard (with error y_err)
    """
    # --- clean & flatten ---
    x = np.asarray(x_data, dtype=np.float64).ravel()
    y = np.asarray(y_data, dtype=np.float64).ravel()
    sx = np.asarray(x_err, dtype=np.float64).ravel()
    sy = np.asarray(y_err, dtype=np.float64).ravel()

    if sx.size == 1:
        sx = np.full_like(x, sx.item())
    if sy.size == 1:
        sy = np.full_like(y, sy.item())

    # Really hates nans or zeros in error, so get rid of them.

    mask = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(sx) & np.isfinite(sy)
        & (sx > 0) & (sy > 0)
    )
    x, y, sx, sy = x[mask], y[mask], sx[mask], sy[mask]
    n_dropped = np.sum(~mask)
    if n_dropped:
        print(f"[fit_densimeter] Dropped {n_dropped} invalid rows. Check these! you can't afford to loose too much data'")

    # --- center x (stabilizes the quadratic fit) ---
    x_mean = np.mean(x)
    xc = x - x_mean

    # --- initial guess from ordinary least squares, in centered space ---
    p0, _ = curve_fit(
        lambda xc_, a, b, c, d, e: poly_odr_centered([a, b, c, d, e], xc_),
        xc, y
    )

    # --- ODR fit (accounts for both x and y errors) ---
    data = RealData(xc, y, sx=sx, sy=sy)
    model = Model(poly_odr_centered)
    odr_fit = ODR(data, model, beta0=p0, maxit=maxit)
    output = odr_fit.run()

    if output.info not in (1, 2):
        raise RuntimeError(
            f"ODR did not converge cleanly: info={output.info}, "
            f"stopreason={output.stopreason}"
        )
    if not np.any(output.cov_beta):
        raise RuntimeError("cov_beta is all zero — fit likely failed silently.")

    popt = output.beta
    # scale by res_var: inflates parameter uncertainty when the data show
    # more scatter than the nominal x_err/y_err predict (chi^2/dof > 1)
    pcov = output.cov_beta * output.res_var

    # --- residual scatter of data around the fitted curve ---
    # this captures intrinsic/experimental scatter NOT explained by the
    # nominal x_err/y_err inputs -- needed for a prediction interval on a
    # single new measurement (as opposed to a confidence interval on the
    # mean curve, which pcov alone gives you)
    y_model_at_data = poly_odr_centered(popt, xc)
    residuals = y - y_model_at_data
    resid_std = np.std(residuals, ddof=len(popt))

    result = {
        "popt": popt,
        "pcov": pcov,
        "x_mean": x_mean,
        "resid_std": resid_std,
        "res_var": output.res_var,
        "info": output.info,
        "stopreason": output.stopreason,
        "param_names": ["a", "b", "c", "d", "e"],
        "model_form": "a*xc + b*xc**2 + c*xc**3 + d*xc**4 + e, where xc = x - x_mean",
        "x_range": (float(np.min(x)), float(np.max(x))),
        "n_points": len(x),
    }

    print(f"ODR stop reason: {output.stopreason}")
    print(f"Reduced Chi-Squared (res_var): {output.res_var:.4f}")
    print(f"Residual scatter around curve (resid_std): {resid_std:.6f}")
    a, b, c, d, e = popt
    print(f"Coefficients (centered space): a={a:.6g} b={b:.6g} c={c:.6g} "
          f"d={d:.6g} e={e:.6g}")
    print(f"x_mean used for centering: {x_mean:.6g}")

    return result


# ---------------------------------------------------------------------------
# 3. Save / load
# ---------------------------------------------------------------------------
def save_model(result, path="densimeter_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved model to {path}")


def load_model(path="densimeter_model.pkl"):
    with open(path, "rb") as f:
        result = pickle.load(f)
    print(f"Loaded model from {path}")
    return result


# ---------------------------------------------------------------------------
# 4. Prediction -- delta method (fast, analytic)
# ---------------------------------------------------------------------------
def predict_delta(x_new, x_new_err, model_dict, include_scatter=True):
    """
    Prediction interval via first-order (delta-method) error propagation.
    Combines, in quadrature:
      1) parameter uncertainty (from pcov, already res_var-scaled)
      2) propagated input-x uncertainty (via dy/dx)
      3) residual/intrinsic scatter around the calibration curve
         (set include_scatter=False to get a confidence interval on the
         mean curve instead of a prediction interval for a new point)
    """
    popt = model_dict["popt"]
    pcov = model_dict["pcov"]
    x_mean = model_dict["x_mean"]
    resid_std = model_dict["resid_std"]

    a, b, c, d, e = popt
    x_new = np.atleast_1d(np.asarray(x_new, dtype=float))
    x_new_err = np.atleast_1d(np.asarray(x_new_err, dtype=float))
    xc = x_new - x_mean

    y_pred = a * xc + b * xc**2 + c * xc**3 + d * xc**4 + e

    # parameter uncertainty
    J_params = np.stack(
        [xc, xc**2, xc**3, xc**4, np.ones_like(xc)], axis=-1
    )
    var_params = np.einsum("ni,ij,nj->n", J_params, pcov, J_params)

    # input-x uncertainty (d/d(xc) == d/dx since xc = x - const)
    dydx = a + 2 * b * xc + 3 * c * xc**2 + 4 * d * xc**3
    var_x = (dydx * x_new_err) ** 2

    var_total = var_params + var_x
    if include_scatter:
        var_total = var_total + resid_std**2

    return {
        "y_pred": y_pred,
        "sigma_params": np.sqrt(var_params),
        "sigma_x": np.sqrt(var_x),
        "sigma_scatter": np.full_like(y_pred, resid_std) if include_scatter else np.zeros_like(y_pred),
        "sigma_total": np.sqrt(var_total),
    }


# ---------------------------------------------------------------------------
# 5. Prediction -- Monte Carlo (robust cross-check)
# ---------------------------------------------------------------------------
def predict_mc(x_new, x_new_err, model_dict, n_mc=20000, include_scatter=True,
                seed=None):
    """
    Monte Carlo prediction interval: samples parameters from N(popt, pcov),
    samples the new x from N(x_new, x_new_err), and (if include_scatter)
    adds a draw from N(0, resid_std) to represent intrinsic scatter.
    """
    rng = np.random.default_rng(seed)
    popt = model_dict["popt"]
    pcov = model_dict["pcov"]
    x_mean = model_dict["x_mean"]
    resid_std = model_dict["resid_std"]

    coef_samples = rng.multivariate_normal(popt, pcov, size=n_mc)
    x_samples = rng.normal(x_new, x_new_err, size=n_mc)
    xc_samples = x_samples - x_mean

    a = coef_samples[:, 0]; b = coef_samples[:, 1]; c = coef_samples[:, 2]
    d = coef_samples[:, 3]; e = coef_samples[:, 4]
    y_samples = a*xc_samples + b*xc_samples**2 + c*xc_samples**3 + d*xc_samples**4 + e

    if include_scatter:
        y_samples = y_samples + rng.normal(0, resid_std, size=n_mc)

    return {
        "y_pred": np.mean(y_samples),
        "sigma_total": np.std(y_samples),
        "samples": y_samples,
    }


## ACtually making a densimeter function for people
def calculate_density_generic(*, df_combo=None, corrected_split=None, split_err=None,
                                Ne_pickle_str=None, Ar_pickle_str=None,
                                pref_Ne=None, Ne_err=None,
                                model_dict=None, CI_split=0.67, CI_neon=0.67):

    """ This is a function for the generic densimeter you have saved as a pickle using the new standards from Wieser et al. (in prep).
    """

    if model_dict is None:
        raise ValueError("model_dict is required (from fit_densimeter/load_model).")

    # --- resolve Split and its error ---
    if corrected_split is not None:
        Split = corrected_split
        Split_err = split_err

    elif df_combo is not None:
        df_combo_c = df_combo.copy()

        # Case A: df_combo already has corrected splitting + error computed
        # (e.g. output of an earlier Ne-correction step) -- use directly,
        # no column names to specify.
        if 'Corrected_Splitting' in df_combo_c.columns:
            Split = df_combo_c['Corrected_Splitting']
            Split_err = df_combo_c['Corrected_Splitting_σ']

        # Case B: raw peak-fit data -- needs Ne drift correction first
        else:
            time = df_combo_c['sec since midnight']

            if Ne_pickle_str is not None:
                Ne_corr = calculate_Ne_corr_std_err_values(pickle_str=Ne_pickle_str,
                                                             new_x=time, CI=CI_neon)
                pref_Ne = Ne_corr['preferred_values']
                Split_err, pk_err = propagate_error_split_neon_peakfit(Ne_corr=Ne_corr, df_fits=df_combo_c)

                df_combo_c['Corrected_Splitting_σ'] = Split_err
                df_combo_c['Corrected_Splitting_σ_Ne'] = (
                    Ne_corr['upper_values']*df_combo_c['Splitting'] - Ne_corr['lower_values']*df_combo_c['Splitting']
                ) / 2
                df_combo_c['Corrected_Splitting_σ_peak_fit'] = pk_err
            else:
                Split_err, pk_err = propagate_error_split_neon_peakfit(df_fits=df_combo_c, Ne_err=Ne_err, pref_Ne=pref_Ne)
                df_combo_c['Corrected_Splitting_σ'] = Split_err
                df_combo_c['Corrected_Splitting_σ_Ne'] = (
                    (Ne_err+pref_Ne)*df_combo_c['Splitting'] - (Ne_err-pref_Ne)*df_combo_c['Splitting']
                ) / 2
                df_combo_c['Corrected_Splitting_σ_peak_fit'] = pk_err

            Split = df_combo_c['Splitting'] * pref_Ne
            df_combo_c['Corrected_Splitting'] = Split
    else:
        raise ValueError("Must supply either df_combo or corrected_split/split_err.")

    if isinstance(Split, (float, int)):
        Split = pd.Series([Split])
    Split = Split.reset_index(drop=True)
    Split_err = pd.Series(Split_err).reset_index(drop=True) if not np.isscalar(Split_err) else Split_err

    # --- apply the polynomial model with full uncertainty propagation ---
    popt = model_dict['popt']
    pcov = model_dict['pcov']
    x_mean = model_dict['x_mean']
    resid_std = model_dict['resid_std']
    a, b, c, d, e = popt

    x = Split.values.astype(float)
    x_err = np.asarray(Split_err, dtype=float)
    xc = x - x_mean

    y_pred = a*xc + b*xc**2 + c*xc**3 + d*xc**4 + e

    J = np.stack([xc, xc**2, xc**3, xc**4, np.ones_like(xc)], axis=-1)
    var_params = np.einsum("ni,ij,nj->n", J, pcov, J)

    dydx = a + 2*b*xc + 3*c*xc**2 + 4*d*xc**3
    var_x = (dydx * x_err) ** 2

    sigma_dens = np.sqrt(var_params + resid_std**2)
    sigma_split = np.sqrt(var_x)
    sigma_total = np.sqrt(var_params + resid_std**2 + var_x)

    x_min, x_max = model_dict['x_range']
    in_range = np.where((x >= x_min) & (x <= x_max), 'Y', 'N')
    notes = np.where(
        in_range == 'Y', 'In calibration range',
        np.where(x < x_min, 'Below lower calibration limit', 'Above upper calibration limit')
    )

    df = pd.DataFrame(data={
        'Preferred D': y_pred,
        'Corrected_Splitting': x,
        'Preferred D_σ': sigma_total,
        'Preferred D_σ_split': sigma_split,
        'Preferred D_σ_dens': sigma_dens,
        'in range': in_range,
        'Notes': notes,
    })

    if df_combo is not None:
        df_merge = pd.concat([df.drop(columns=['Corrected_Splitting']), df_combo_c.reset_index(drop=True)], axis=1)
    else:
        df_merge = df

    df_merge = df_merge.rename(columns={
        'Preferred D': 'Density g/cm3',
        'Preferred D_σ': 'σ Density g/cm3',
        'Preferred D_σ_split': 'σ Density g/cm3 (from Ne+peakfit)',
        'Preferred D_σ_dens': 'σ Density g/cm3 (from densimeter)',
        'filename_x': 'filename',
    })

    cols_to_move = [
        'filename', 'Density g/cm3', 'σ Density g/cm3',
        'σ Density g/cm3 (from Ne+peakfit)', 'σ Density g/cm3 (from densimeter)',
        'Corrected_Splitting', 'Corrected_Splitting_σ',
        'power (mW)', 'Spectral Center'
    ]
    cols_existing = [col for col in cols_to_move if col in df_merge.columns]
    df_merge = df_merge[cols_existing + [col for col in df_merge.columns if col not in cols_existing]]

    return df_merge