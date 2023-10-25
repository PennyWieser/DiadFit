import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import lmfit
from lmfit.models import GaussianModel, VoigtModel, LinearModel, ConstantModel
from scipy.signal import find_peaks
import os
import re
from os import listdir
from os.path import isfile, join
import pickle
from scipy import stats


encode="ISO-8859-1"


def calculate_generic_std_err_values(*, pickle_str, new_x, CI=0.67):

    """
    This function loads a model from a pickle file and calculates standard error values based on the model residuals, considering a confidence interval.

    Parameters
    -----------------
    pickle_str (str)
        The path to the pickle file containing the model.
    new_x (np.array):
        x values for calculation.
    CI (float, optional):
        The confidence level for prediction intervals (default: 0.67).

    Returns:
    - pandas.DataFrame

    """
    # Load the model and the data from the pickle file
    with open(pickle_str, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    N_poly = model.order - 1
    Pf = data['model']
    x = data['x']
    y = data['y']

    # Calculate the residuals
    residuals = y - Pf(x)

    # Calculate the standard deviation of the residuals
    residual_std = np.std(residuals)

    # Calculate the standard errors for the new x values
    mean_x = np.mean(x)
    n = len(x)
    standard_errors = residual_std * np.sqrt(1 + 1/n + (new_x - mean_x)**2 / np.sum((x - mean_x)**2))

    # Calculate the degrees of freedom
    df = len(x) - (N_poly + 1)

    # Calculate the t value for the given confidence level
    t_value = stats.t.ppf((1 + CI) / 2, df)

    # Calculate the prediction intervals
    preferred_values = Pf(new_x)
    lower_values = preferred_values - t_value * standard_errors
    upper_values = preferred_values + t_value * standard_errors

    df=pd.DataFrame(data={
        'time': new_x,
        'preferred_values': preferred_values,
        'lower_values': lower_values,
        'upper_values': upper_values
    })

    return df

def plot_and_save_CO2cali_pickle(*, cali_data, CO2_dens_col='rho',Split_col='Split', split_error='split_err',CO2_dens_error='dens_err', density_range, N_poly=3, CI=0.67, std_error=True, save_fig=False,eq_division='ccmr',save_suffix=''):
    """
    This function calculates and saves three polynomial regression models as pickles for CO2 calibration data according to DeVitre et al. 2021 (low density, mid density and high density).
    The models are saved as pickles and the plot is saved as well if desired.

    Parameters
    -----------------
    cali_data (pandas.DataFrame):
        A DataFrame containing calibration data (should at least contain a density and a fermi splitting column)

    CO2_dens_col (str, optional):
        The column name corresponding to CO2 density in the cali_data DataFrame (default: 'rho').

    Split_col (str, optional):
        The column name corresponding to Fermi splitting values in the cali_data DataFrame  (default: 'Split').

    split_error (str, float, or array-like, optional):
        The column name corresponding to error in fermi splitting OR an array of splitting errors OR a float (default: 'split_err').

    CO2_dens_error (str, float or array-like, optional):
        The column name corresponding to error in CO2 density OR an array of density errors OR a float(default: 'dens_err').

    density_range (str):
        The density range to be fit ('Low', 'Medium', or 'High').

    N_poly (int, optional):
        The degree of the polynomial fit (default: 3).

    CI (float, optional):
        The confidence level for prediction intervals (default: 0.67).

    std_error (bool, optional):
        Whether to calculate and plot standard error (default: True).

    save_fig (bool, optional):
        Whether to save the plot as an image (default: False).

    eq_division (str, optional):
        Method for dividing the data based on density ('ccmr' or 'cmass', default: 'ccmr'). CCMR corresponds to the limits for each section as shown in DeVitre et al., 2021 (Chem. Geo), cmass is for those in DeVitre et al., 2023 (J. Volcanica)

    save_suffix (str, optional):
        Suffix to be added to the saved file names (default: '').

    Returns:
        Saves a .pkl file, doesnt return anything.

    """
# Define the x and y values
    try:
        if eq_division=='ccmr':
            lowcut=0.17
            midcut_low=0.12
            midcut_high=0.72
            highcut=0.65
        elif eq_division=='cmass':
            lowcut=0.20
            midcut_low=0.13
            midcut_high=0.70
            highcut=0.65
        if density_range == 'Low':
            cali_data = cali_data[cali_data[CO2_dens_col] < lowcut]

            prefix='Lowrho_'
        elif density_range == 'Medium':
            cali_data = cali_data[cali_data[CO2_dens_col].between(midcut_low, midcut_high)]

            prefix='Mediumrho_'
        elif density_range == 'High':
            cali_data = cali_data[cali_data[CO2_dens_col] > highcut]

            prefix='Highrho_'
        else:
            raise ValueError("Invalid density range. Please choose 'Low', 'Medium', or 'High'.")
    except ValueError as e:
        print(f"Warning: {e}")
        return

    x_all   = cali_data[Split_col].values
    y_all = cali_data[CO2_dens_col].values

    if not isinstance(split_error,str)==True:
        x_err=pd.Series(split_error, index=cali_data[CO2_dens_col].index).values
    else:
        x_err=cali_data[split_error].values

    if not isinstance(CO2_dens_error,str)==True:
        y_err=pd.Series(CO2_dens_error, index=cali_data[CO2_dens_col].index).values
    else:
        y_err=cali_data[CO2_dens_error].values

    non_nan_indices = ~np.isnan(x_all) & ~np.isnan(y_all)

    # Filter out NaN values
    x = x_all[non_nan_indices]
    y = y_all[non_nan_indices]
    # Perform polynomial regression

    coefficients = np.polyfit(x, y, N_poly)
    Pf = np.poly1d(coefficients)


    # Save the model and the data to a pickle file
    data = {'model': Pf, 'x': x, 'y': y}
    with open(prefix+'polyfit_data'+save_suffix+'.pkl', 'wb') as f:
        pickle.dump(data, f)

    if std_error is True:
        new_x_plot = np.linspace(np.min(x), np.max(x), 100)
        new_calidf = calculate_generic_std_err_values(pickle_str=prefix+'polyfit_data'+save_suffix+'.pkl',
                                                    new_x=pd.Series(new_x_plot), CI=CI)
        # Calculate R-squared and p-value
        residuals = y - Pf(x)
        ssr = np.sum(residuals**2)
        sst = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ssr / sst)
        df = len(x) - (N_poly + 1)
        p_value = 1 - stats.f.cdf(r_squared * (len(x) - 1), N_poly, df)
        legend_label = f'best fit: R$^2$ = {r_squared:.4f}, p-val = {p_value:.1e}'
    else:
        legend_label = 'best fit'


    # Now lets plot the confidence interval
    fig, (ax1) = plt.subplots(1, 1, figsize=(10,5))
    ax1.errorbar(x, y, xerr=x_err,yerr=y_err,
            fmt='o', ecolor='grey', elinewidth=0.8, mfc='grey', ms=5, mec='k',capsize=3,barsabove=True)

    ax1.plot(new_x_plot, new_calidf['preferred_values'], '-k', label=legend_label)
    # ax1.plot(new_x_plot, new_calidf['lower_values'], ':k', label='lower vals')
    # ax1.plot(new_x_plot, new_calidf['upper_values'], ':k', label='upper vals')
    ax1.fill_between(new_x_plot, new_calidf['lower_values'], new_calidf['upper_values'], color='gray', alpha=0.2,zorder=-2,label='Prediction interval')
    ax1.set_xlabel('Split')
    ax1.set_ylabel('Density')
    ax1.legend()

    ax1.ticklabel_format(useOffset=False)
    # this sets the ordinal suffix for the polynomial degree in the title
    if N_poly == 1:
        suffix = 'st'
    elif N_poly == 2:
        suffix = 'nd'
    elif N_poly == 3:
        suffix = 'rd'
    else:
        suffix='th'

    ax1.set_title(f"{N_poly}$^{{{suffix}}}$ degree polynomial: "+str(100*CI) + ' % prediction interval')


    if save_fig is True:
        fig.savefig(prefix+'cali_line'+save_suffix+'.png')