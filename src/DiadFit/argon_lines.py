import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import lmfit
from lmfit.models import GaussianModel, VoigtModel, LinearModel, ConstantModel, PseudoVoigtModel
from scipy.signal import find_peaks
import os
import re
from os import listdir
from os.path import isfile, join
from DiadFit.importing_data_files import *
from typing import Tuple, Optional
from dataclasses import dataclass
import matplotlib.patches as patches
from tqdm.notebook import tqdm
import re
from DiadFit.ne_lines import *



encode="ISO-8859-1"

def calculate_Ar_splitting(wavelength=532.05, line1_shift=1117, line2_shift=1447, cut_off_intensity=2000):
    """
    Calculates ideal splitting in air between lines closest to user-specified line shift. E.g. if user enters
    1117 and 1447 line, it looks for the nearest theoretical Ne line position based on your wavelength,
    and calculates the ideal splitting between these theoretical line positions. This is used to calculate the
    ideal Ne line splitting for doing the Ne correction routine of Lamadrid et al. (2017)

    Parameters
    -------------
    Wavelength: int
        Wavelength of your laser
    line1_shift: int, float
        Estimate of position of line 1
    line2_shift: int, float
        Estimate of position of line 2
    cut_off_intensity: int, float
        only searches through lines with a theoretical intensity from NIST grater than this value

    Returns
    ----------
    df of theoretical splitting, line positions used for this, and entered Ne line positions
    """

    df_Ar=calculate_Ar_line_positions(wavelength=wavelength, cut_off_intensity=cut_off_intensity)

    closest1=find_closest(df_Ar, line1_shift).loc['Raman_shift (cm-1)']
    closest2=find_closest(df_Ar, line2_shift).loc['Raman_shift (cm-1)']

    diff=abs(closest1-closest2)

    df=pd.DataFrame(data={'Ar_Split': diff,
    'Line_1': closest1,
    'Line_2': closest2,
    'Entered Pos Line 1': line1_shift,
    'Entered Pos Line 2': line2_shift}, index=[0])


    return df

def calculate_Ar_line_positions(wavelength=532.05, cut_off_intensity=2000):
    """
    Calculates Raman shift for a given laser wavelength of Ne lines, using the datatable from NIST of Ne line
    emissoin in air and the intensity of each line.

    Parameters
    ---------------
    Wavelength: float
        Wavelength of laser
    cut_off_intensity: float
        Only chooses lines with intensities greater than this

    Returns
    ------------
    pd.DataFrame
        df wih Raman shift, intensity, and emission line position in air.


    """

    Ar_emission_line_air=np.array([
2420.456,
2516.789,
2534.709,
2562.087,
2891.612,
2942.893,
2979.05,
3033.508,
3093.402,
3243.689,
3293.64,
3307.228,
3350.924,
3376.436,
3388.531,
3476.747,
3478.232,
3491.244,
3491.536,
3509.778,
3514.388,
3545.596,
3545.845,
3559.508,
3561.03,
3576.616,
3581.608,
3582.355,
3588.441,
3622.138,
3639.833,
3718.206,
3729.309,
3737.889,
3765.27,
3766.119,
3770.369,
3770.52,
3780.84,
3803.172,
3809.456,
3850.581,
3868.528,
3925.719,
3928.623,
3932.547,
3946.097,
3948.979,
3979.356,
3994.792,
4013.857,
4033.809,
4035.46,
4042.894,
4044.418,
4052.921,
4072.005,
4072.385,
4076.628,
4079.574,
4082.387,
4103.912,
4131.724,
4156.086,
4158.59,
4164.18,
4179.297,
4181.884,
4190.713,
4191.029,
4198.317,
4200.674,
4218.665,
4222.637,
4226.988,
4228.158,
4237.22,
4251.185,
4259.362,
4266.286,
4266.527,
4272.169,
4277.528,
4282.898,
4300.101,
4300.65,
4309.239,
4331.2,
4332.03,
4333.561,
4335.338,
4345.168,
4348.064,
4352.205,
4362.066,
4367.832,
4370.753,
4371.329,
4375.954,
4379.667,
4385.057,
4400.097,
4400.986,
4426.001,
4430.189,
4430.996,
4433.838,
4439.461,
4448.879,
4474.759,
4481.811,
4510.733,
4522.323,
4530.552,
4545.052,
4564.405,
4579.35,
4589.898,
4609.567,
4637.233,
4657.901,
4721.591,
4726.868,
4732.053,
4735.906,
4764.865,
4806.02,
4847.81,
4865.91,
4879.864,
4889.042,
4904.752,
4933.209,
4965.08,
5009.334,
5017.163,
5062.037,
5090.495,
5141.783,
5145.308,
5165.773,
5187.746,
5216.814,
5495.874,
5558.702,
5606.733,
5650.704,
5739.520,
5888.584,
5912.085,
6032.127,
6043.223,
6059.372,
6114.923,
6172.278,
6243.12,
6384.717,
6416.307,
6483.082,
6638.221,
6639.74,
6643.698,
6666.359,
6677.282,
6684.293,
6752.834,
6861.269,
6871.289,
6937.664,
6965.431,
7030.251,
7067.218,
7068.736,
7107.478,
7125.82,
7147.042,
7206.98,
7272.936,
7311.716,
7316.005,
7353.293,
7372.118,
7380.426,
7383.98,
7392.98,
7435.368,
7503.869,
7514.652,
7635.106,
7723.761,
7724.207,
7948.176,
8006.157,
8014.786,
8103.693,
8115.311,
8264.522,
8392.27,
8408.21,
8424.648,
8521.442,
8667.944,
8771.86,
8849.91,
9075.394,
9122.967,
9194.638,
9224.499,
9291.531,
9354.22,
9657.786,
9784.503,
10052.06,
10332.72,
10467.177,
10470.054,
10506.5,
10673.565,
10683.034,
10733.87,
10759.16,
10812.896,
11106.46,
11488.109,
11668.71,
12112.326,
12139.738,
12343.393,
12402.827,
12439.321,
12456.12,
12487.663,
12702.281,
12733.418,
12802.739,
12933.195,
12956.659,
13008.264,
13213.99,
13228.107,
13230.9,
13272.64,
13313.21,
13367.111,
13499.41,
13504.191,
13599.333,
13622.659,
13678.55,
13718.577,
14093.64,
15046.5,
15172.69,
15989.49,
16519.86,
16940.58,
20616.23,
20986.11,
23133.2,
23966.52,

    ])*0.1


    Intensity=np.array([
    2,
    3,
    3,
    5,
    8,
    70,
    30,
    15,
    15,
    7,
    8,
    7,
    8,
    8,
    8,
    25,
    7,
    15,
    30,
    25,
    25,
    25,
    25,
    30,
    30,
    25,
    8,
    15,
    25,
    8,
    7,
    12,
    25,
    15,
    50,
    15,
    1,
    7,
    8,
    8,
    15,
    25,
    12,
    12,
    15,
    8,
    25,
    1,
    7,
    12,
    15,
    15,
    7,
    50,
    1,
    30,
    70,
    25,
    8,
    12,
    8,
    50,
    100,
    12,
    11,
    1,
    12,
    1,
    3,
    1,
    6,
    11,
    8,
    8,
    8,
    30,
    30,
    1,
    6,
    3,
    25,
    4,
    200,
    7,
    3,
    8,
    25,
    70,
    15,
    3,
    1,
    1,
    250,
    15,
    8,
    15,
    70,
    25,
    15,
    50,
    15,
    25,
    70,
    130,
    50,
    15,
    15,
    7,
    12,
    30,
    70,
    3,
    1,
    7,
    130,
    7,
    130,
    130,
    200,
    12,
    130,
    7,
    200,
    15,
    100,
    250,
    200,
    50,
    15,
    250,
    25,
    7,
    12,
    70,
    15,
    25,
    25,
    7,
    30,
    25,
    8,
    1,
    7,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    30,
    50,
    8,
    1,
    2,
    8,
    8,
    7,
    15,
    8,
    3,
    12,
    4,
    7,
    4,
    1,
    300,
    4,
    300,
    3,
    1,
    1,
    30,
    2,
    60,
    1,
    1,
    2,
    6,
    7,
    300,
    1,
    1,
    600,
    400,
    700,
    400,
    300,
    600,
    600,
    700,
    600,
    1000,
    300,
    1,
    400,
    600,
    400,
    130,
    7,
    5,
    1,
    1000,
    15,
    400,
    11,
    50,
    700,
    130,
    5,
    1,
    30,
    50,
    5,
    6,
    2,
    1,
    1,
    2,
    1,
    11,
    6,
    6,
    1,
    1,
    6,
    6,
    3,
    6,
    4,
    1,
    6,
    1,
    14,
    6,
    6,
    6,
    3,
    14,
    30,
    30,
    1,
    30,
    1,
    11,
    6,
    30,
    6,
    3,
    1,
    1,
    1,
    14,
    1,
    1,
    1,
    1,


    ])


    Raman_shift=(10**7/wavelength)-(10**7/Ar_emission_line_air)

    df_Ar=pd.DataFrame(data={'Raman_shift (cm-1)': Raman_shift,
                            'Intensity': Intensity,
                            'Ar emission line in air': Ar_emission_line_air})

    df_Ar_r=df_Ar.loc[df_Ar['Intensity']>cut_off_intensity]

    df_Ar_r=df_Ar_r.loc[df_Ar['Raman_shift (cm-1)']>0]
    return df_Ar_r


## Function that is basically just the Ne function but with some renaming.


def loop_Ar_lines(*, files, spectra_path, filetype, config_ID_peaks, config, df_fit_params,  prefix, plot_figure=True, print_df=False, const_params=True):
    # Call the Neon line processing function
    df_ne = loop_Ne_lines(
        files=files,
        spectra_path=spectra_path,
        filetype=filetype,
        config_ID_peaks=config_ID_peaks,
        config=config,
        df_fit_params=df_fit_params,
        prefix=prefix,
        plot_figure=plot_figure,
        print_df=False,
        const_params=const_params
    )



    # Rename columns by replacing 'Ne' with 'Ar'
    df_ar = df_ne.rename(columns=lambda col: col.replace('Ne', 'Ar'))

    return df_ar


def Argon_id_config(*, height, distance, prominence, width, threshold,
                    peak1_cent, peak2_cent, n_peaks,
                    exclude_range_1=None, exclude_range_2=None):
    return Neon_id_config(
        height=height,
        distance=distance,
        prominence=prominence,
        width=width,
        threshold=threshold,
        peak1_cent=peak1_cent,
        peak2_cent=peak2_cent,
        n_peaks=n_peaks,
        exclude_range_1=exclude_range_1,
        exclude_range_2=exclude_range_2
    )

def identify_Ar_lines(*, path, filename, filetype, config, print_df=False):
    return identify_Ne_lines(
        path=path,
        filename=filename,
        filetype=filetype,
        config=config,
        print_df=print_df
    )

def Ar_peak_config(**kwargs):
    return Ne_peak_config(**kwargs)


def fit_Ar_lines(*, Ar, filename, path, prefix=False, config,
                 Ar_center_1, Ar_center_2,
                 Ar_prom_1, Ar_prom_2,
                 const_params=False):
    # Call the Neon version internally
    df_ne = fit_Ne_lines(
        Ne=Ar,
        filename=filename,
        path=path,
        prefix=prefix,
        config=config,
        Ne_center_1=Ar_center_1,
        Ne_center_2=Ar_center_2,
        Ne_prom_1=Ar_prom_1,
        Ne_prom_2=Ar_prom_2,
        const_params=const_params
    )

    # Rename output columns from 'Ne' to 'Ar'
    df_ar = df_ne.rename(columns=lambda col: col.replace('Ne', 'Ar'))

    return df_ar


import matplotlib.pyplot as plt

def plot_Ar_corrections(df=None, x_axis=None, x_label='index', marker='o', mec='k', mfc='r'):
    """
    Plot correction-related information for Ar spectra.
    Assumes column names have 'Ar_Corr', '1σ_Ar_Corr', etc.
    """
    if x_axis is not None:
        x = x_axis
    else:
        x = df.index

    fig, ((ax5, ax6),  (ax3, ax4), (ax1, ax2)) = plt.subplots(3, 2, figsize=(10, 12))

    # Peak 1
    ax5.errorbar(x, df['pk1_peak_cent'], xerr=0, yerr=df['error_pk1'].fillna(0).infer_objects(),
                 fmt='o', ecolor='k', elinewidth=0.8, mfc='b', ms=5, mec='k', capsize=3)
    ax5.set_xlabel(x_label)
    ax5.set_ylabel('Peak 1 center')

    # Peak 2
    ax6.plot(x, df['pk2_peak_cent'], marker, mec=mec, mfc=mfc)
    ax6.errorbar(x, df['pk2_peak_cent'], xerr=0, yerr=df['error_pk2'].fillna(0).infer_objects(),
                 fmt='o', ecolor='k', elinewidth=0.8, mfc=mfc, ms=5, mec=mec, capsize=3)
    ax6.set_xlabel(x_label)
    ax6.set_ylabel('Peak 2 center')

    # Correction vs. Peak 2
    ax3.errorbar(df['Ar_Corr'], df['pk2_peak_cent'],
                 xerr=df['1σ_Ar_Corr'].fillna(0).infer_objects(),
                 yerr=df['error_pk2'].fillna(0).infer_objects(),
                 fmt='o', ecolor='k', elinewidth=0.8, mfc='b', ms=5, mec='k', capsize=3)
    ax3.set_xlabel('Ar Correction factor')
    ax3.set_ylabel('Peak 2 center')

    # Correction vs. Peak 1
    ax4.errorbar(df['Ar_Corr'], df['pk1_peak_cent'],
                 xerr=df['1σ_Ar_Corr'].fillna(0).infer_objects(),
                 yerr=df['error_pk1'].fillna(0).infer_objects(),
                 fmt='o', ecolor='k', elinewidth=0.8, mfc='b', ms=5, mec='k', capsize=3)
    ax4.set_xlabel('Ar Correction factor')
    ax4.set_ylabel('Peak 1 center')

    # Ar Correction vs. x
    ax1.errorbar(x, df['Ar_Corr'], xerr=0, yerr=df['1σ_Ar_Corr'].fillna(0).infer_objects(),
                 fmt='o', ecolor='k', elinewidth=0.8, mfc='grey', ms=5, mec='k', capsize=3)
    ax1.set_ylabel('Ar Correction factor')
    ax1.set_xlabel(x_label)

    # Ar Correction vs. residuals (placeholder logic — update as needed)
    if 'residual_sum' in df.columns:
        ax2.errorbar(df['residual_sum'], df['Ar_Corr'],
                     xerr=0, yerr=df['1σ_Ar_Corr'].fillna(0).infer_objects(),
                     fmt='o', ecolor='k', elinewidth=0.8, mfc='grey', ms=5, mec='k', capsize=3)

    ax2.set_xlabel('Sum of pk1 and pk2 residual')
    ax2.set_ylabel('Ar Correction factor')

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.ticklabel_format(useOffset=False)

    plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax4.get_xticklabels(), rotation=30, ha='right')

    fig.tight_layout()
    return fig


def filter_Ar_Line_neighbours(*, df_combo=None, Corr_factor=None, number_av=6, offset=0.00005, file_name_filt=None):
    """
    Filters Ar correction factors that deviate by more than `offset` from the
    local median of neighboring values (defined by `number_av`).
    Optionally excludes specified filenames via `file_name_filt`.
    """
    if df_combo is not None:
        Corr_factor = df_combo['Ar_Corr']

    Corr_factor_Filt = np.zeros(len(Corr_factor), dtype=float)
    median_loop = np.zeros(len(Corr_factor), dtype=float)

    for i in range(len(Corr_factor)):
        if i < len(Corr_factor) / 2:
            median_loop[i] = np.nanmedian(Corr_factor[i:i+number_av])
        else:
            median_loop[i] = np.nanmedian(Corr_factor[i-number_av:i])

        if (
            Corr_factor[i] > (median_loop[i] + offset)
            or Corr_factor[i] < (median_loop[i] - offset)
        ):
            Corr_factor_Filt[i] = np.nan
        else:
            Corr_factor_Filt[i] = Corr_factor[i]

    ds = pd.Series(Corr_factor_Filt)

    if file_name_filt is not None:
        pattern = '|'.join(file_name_filt)
        mask = df_combo['filename_x'].str.contains(pattern)
        ds = ds.where(~mask, np.nan)

    return ds


def generate_Ar_corr_model(*, time, Ar_corr, N_poly=3, CI=0.67, bootstrap=False,
                           std_error=True, N_bootstrap=500,
                           save_fig=False, pkl_name='polyfit_data_Ar.pkl'):
    """Generates a polynomial correction model for Ar correction data."""

    x_all = np.array([time])

    if isinstance(Ar_corr, pd.DataFrame):
        y_all = np.array([Ar_corr['Ar_Corr']])
        y_err = Ar_corr['1σ_Ar_Corr']
    else:
        y_all = Ar_corr
        y_err = 0 * Ar_corr

    non_nan_indices = ~np.isnan(x_all) & ~np.isnan(y_all)
    x = x_all[non_nan_indices]
    y = y_all[non_nan_indices]

    coefficients = np.polyfit(x, y, N_poly)
    Pf = np.poly1d(coefficients)

    data = {'model': Pf, 'x': x, 'y': y}
    with open(pkl_name, 'wb') as f:
        pickle.dump(data, f)

    new_x_plot = np.linspace(np.min(x), np.max(x), 100)

    if bootstrap:
        Ar_corr2 = calculate_Ar_corr_bootstrap_values(
            pickle_str=pkl_name,
            new_x=pd.Series(new_x_plot),
            N_poly=N_poly,
            CI=CI,
            N_bootstrap=N_bootstrap
        )
    elif std_error:
        Ar_corr2 = calculate_Ar_corr_std_err_values(
            pickle_str=pkl_name,
            new_x=pd.Series(new_x_plot),
            CI=CI
        )

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax1.errorbar(x, y, xerr=0, yerr=y_err, fmt='o', ecolor='k',
                 elinewidth=0.8, mfc='grey', ms=5, mec='k', capsize=3)

    ax1.plot(new_x_plot, Ar_corr2['preferred_values'], '-k', label='best fit')
    ax1.plot(new_x_plot, Ar_corr2['lower_values'], ':k', label='lower bound')
    ax1.plot(new_x_plot, Ar_corr2['upper_values'], ':k', label='upper bound')
    ax1.plot(x, y, '+r', label='Ar lines')

    ax1.set_xlabel('sec after midnight')
    ax1.set_ylabel('Ar Corr factor')
    ax1.ticklabel_format(useOffset=False)
    ax1.legend()

    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(N_poly, 'th')
    ax1.set_title(f"{N_poly}$^{{{suffix}}}$ degree polynomial: {int(CI * 100)}% prediction interval")

    if save_fig:
        fig.savefig('Ar_line_correction.png')

    return Pf, fig


def calculate_Ar_corr_std_err_values(*, pickle_str, new_x, CI=0.67):
    with open(pickle_str, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    N_poly = model.order - 1
    x = data['x']
    y = data['y']

    new_x_array = np.asarray(new_x)
    residuals = y - model(x)
    residual_std = np.std(residuals)

    mean_x = np.mean(x)
    n = len(x)
    standard_errors = residual_std * np.sqrt(1 + 1/n + (new_x_array - mean_x)**2 / np.sum((x - mean_x)**2))

    df_dof = len(x) - (N_poly + 1)
    t_value = t.ppf((1 + CI) / 2, df_dof)

    preferred_values = model(new_x_array)
    lower_values = preferred_values - t_value * standard_errors
    upper_values = preferred_values + t_value * standard_errors

    return pd.DataFrame({
        'time': new_x_array,
        'preferred_values': preferred_values,
        'lower_values': lower_values,
        'upper_values': upper_values
    })


def calculate_Ar_corr_bootstrap_values(*, pickle_str, new_x, N_poly=3, CI=0.67, N_bootstrap=500):
    with open(pickle_str, 'rb') as f:
        data = pickle.load(f)

    Pf = data['model']
    x = data['x']
    y = data['y']

    x_values = new_x
    preferred_values = []
    lower_values = []
    upper_values = []

    for new_x in x_values:
        bootstrap_predictions = []
        for _ in range(N_bootstrap):
            bootstrap_indices = np.random.choice(len(x), size=len(x), replace=True)
            bootstrap_x = x[bootstrap_indices]
            bootstrap_y = y[bootstrap_indices]
            bootstrap_coefficients = np.polyfit(bootstrap_x, bootstrap_y, N_poly)
            bootstrap_Pf = np.poly1d(bootstrap_coefficients)
            bootstrap_predictions.append(bootstrap_Pf(new_x))

        bootstrap_predictions_sorted = np.sort(bootstrap_predictions)
        lower_idx = int(((1 - CI) / 2) * N_bootstrap)
        upper_idx = int((1 - (1 - CI) / 2) * N_bootstrap)

        preferred_values.append(Pf(new_x))
        lower_values.append(bootstrap_predictions_sorted[lower_idx])
        upper_values.append(bootstrap_predictions_sorted[upper_idx])

    return pd.DataFrame({
        'time': x_values,
        'preferred_values': preferred_values,
        'lower_values': lower_values,
        'upper_values': upper_values
    })




