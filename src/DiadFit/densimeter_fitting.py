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

def plot_and_save_CO2cali_pickle(*, cali_data, density_range, N_poly=3, CI=0.67, std_error=True, save_fig=False):
# Define the x and y values
    try:
        if density_range == 'Low':
            cali_data = cali_data[cali_data['rho'] < 0.17]
#            print(np.min(cali_data['Split']))
#            print(np.max(cali_data['Split']))
            prefix='Lowrho_'
        elif density_range == 'Medium':
            cali_data = cali_data[cali_data['rho'].between(0.12, 0.72)]
#            print(np.min(cali_data['Split']))
#           print(np.max(cali_data['Split']))
            prefix='Mediumrho_'
        elif density_range == 'High':
            cali_data = cali_data[cali_data['rho'] > 0.65]
#            print(np.min(cali_data['Split']))
#            print(np.max(cali_data['Split']))
            prefix='Highrho_'
        else:
            raise ValueError("Invalid density range. Please choose 'Low', 'Medium', or 'High'.")
    except ValueError as e:
        print(f"Warning: {e}")
        return

    x_all   = cali_data['Split'].values
    y_all = cali_data['rho'].values
    x_err=cali_data['spliterr'].values
    non_nan_indices = ~np.isnan(x_all) & ~np.isnan(y_all)

    # Filter out NaN values
    x = x_all[non_nan_indices]
    y = y_all[non_nan_indices]
    # Perform polynomial regression

    coefficients = np.polyfit(x, y, N_poly)
    Pf = np.poly1d(coefficients)


    # Save the model and the data to a pickle file
    data = {'model': Pf, 'x': x, 'y': y}
    with open(prefix+'polyfit_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    if std_error is True:
        new_x_plot = np.linspace(np.min(x), np.max(x), 100)
        new_calidf = calculate_generic_std_err_values(pickle_str=prefix+'polyfit_data.pkl',
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
    ax1.errorbar(x, y, xerr=x_err,
            fmt='o', ecolor='k', elinewidth=0.8, mfc='grey', ms=5, mec='k',capsize=3,barsabove=True)

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
        fig.savefig(prefix+'cali_line.png')