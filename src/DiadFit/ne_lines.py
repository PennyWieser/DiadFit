import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import lmfit
from lmfit.models import GaussianModel, VoigtModel, LinearModel, ConstantModel, PseudoVoigtModel, SkewedVoigtModel
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
import scipy.stats as stats
import pickle


allowed_models = ["VoigtModel", "PseudoVoigtModel", "Pearson4Model", "SkewedVoigtModel"]



encode="ISO-8859-1"

## Plotting Ne lines, returns peak position
def find_closest(df, line1_shift):
    """ 
   This function finds the closest Raman shift value in the inputted dataframe to the inputted line position

   Parameters
   -------------
    df: pd.DataFrame
        Dataframe of Ne line positions based on laser wavelength from the function calculate_Ne_line_positions, or calculate_Ar_line_positions
    
    line1_shift: int, float
        input line position
error_pk2
   Returns
   -------------
   Closest theoretical line position

    """
    dist = (df['Raman_shift (cm-1)'] - line1_shift).abs()
    return df.loc[dist.idxmin()]

def find_max_row(df, target_shift, tol=2):
    """ This function is used to find the highest amplitude within a predefined ampl range for finding the right Ne line
    """
    
    df_filtered = df[(df['Raman_shift (cm-1)'] >= target_shift - tol) & (df['Raman_shift (cm-1)'] <= target_shift + tol)]
    
    # Find the row with the maximum intensity within this filtered DataFrame
    max_intensity_row = df_filtered.loc[df_filtered['Intensity'].idxmax()]
    
    return max_intensity_row

def calculate_Ne_splitting(wavelength=532.05, line1_shift=1117, line2_shift=1447, cut_off_intensity=2000):
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

    df_Ne=calculate_Ne_line_positions(wavelength=wavelength, cut_off_intensity=cut_off_intensity)
    # 
    # closest1=find_closest(df_Ne, line1_shift).loc['Raman_shift (cm-1)']
    # closest2=find_closest(df_Ne, line2_shift).loc['Raman_shift (cm-1)']
    # closest_1_int=find_closest(df_Ne, line1_shift).loc['Intensity']
    # closest_2_int=find_closest(df_Ne, line2_shift).loc['Intensity']
    # 
    # diff=abs(closest1-closest2)
    # 
    # df=pd.DataFrame(data={'Ne_Split': diff,
    # 'Line_1': closest1,
    # 'Line_2': closest2,
    # 'Line_1_int': closest_1_int,
    # 'Line_2_int': closest_2_int,
    # 'Entered Pos Line 1': line1_shift,
    # 'Entered Pos Line 2': line2_shift}, index=[0])
    # 
    # 
    # return df
# Use the new function to find the lines of interest
    closest1_row = find_max_row(df_Ne, line1_shift)
    closest2_row = find_max_row(df_Ne, line2_shift)
    
    # Extract the required values from the rows
    closest1 = closest1_row['Raman_shift (cm-1)']
    closest2 = closest2_row['Raman_shift (cm-1)']
    closest_1_int = closest1_row['Intensity']
    closest_2_int = closest2_row['Intensity']
    
    # Calculate the difference
    diff = abs(closest1 - closest2)
    
    # Create the DataFrame
    df = pd.DataFrame(data={
    'Ne_Split': diff,
    'Line_1': closest1,
    'Line_2': closest2,
    'Line_1_int': closest_1_int,
    'Line_2_int': closest_2_int,
    'Entered Pos Line 1': line1_shift,
    'Entered Pos Line 2': line2_shift
    }, index=[0])
    
    return df

def calculate_Ne_line_positions(wavelength=532.05, cut_off_intensity=2000):
    """
    Calculates Raman shift for a given laser wavelength of Ne lines, using the datatable from NIST of Ne line
    emissoin in air and the intensity of each line. 
    Data from https://physics.nist.gov/PhysRefData/ASD/lines_form.html

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

    Ne_emission_line_air=np.array([
    
    
    541.85584,
    542.009,
    542.0155,
    543.36513,
    544.7120,
    
    544.85091,
    549.44158,
    550.73442,
    551.1176,
    551.1485,
    
    552.063,
    553.36788,
    553.86510,
    555.90978,

    
    
    
    556.244160,
    556.276620,
    556.305310,
    557.603940,
    558.590500,
    558.934720,
    559.115000,
    565.256640,
    565.602580,
    565.665880,
    566.220000,
    566.254890,
    568.464700,
    568.981630,
    571.534090,
    571.887980,
    571.922480,
    571.953000,
    574.243700,
    574.829850,
    574.864460,
    576.058850,
    576.405250,
    576.441880,
    577.030670,
    580.409000,
    580.444960,
    581.140660,
    581.662190,
    582.015580,
    582.890630,
    585.24878,
    586.84165,
    587.2145,
    587.28275,
    588.1895,
    589.83287,
    590.20944,
    590.24623,
    590.27835,
    590.64294,
    591.3633,
    591.89068,
    591.9029,
    593.44522,
    593.93154,
    594.4834,
    596.16228,
    596.5471,
    596.6179,
    597.46273,
    597.55343,
    598.23753,
    598.79074,
    599.16477,
    600.09275,
    602.99968,
    604.2013,
    604.61348,
    606.45359,
    607.43376,
    609.6163,
    611.088,
    611.272,
    611.80187,
    612.84498,
    613.277,
    614.2508,
    614.30627,
    615.02985,
    615.6138,
    616.35937,
    616.609,
    617.28156,
    617.48829,
    617.52842,
    618.2146,
    618.31575,
    618.90649,
    619.30663,
    620.2974,
    620.57775,
    620.908,
    621.018,
    621.38758,
    621.72812,
    622.448,
    622.5735,
    623.9032,
    624.67294,
    624.9593,
    625.0925,
    625.2732,
    625.2905,
    625.869,
    625.87884,
    626.64952,
    627.0131,
    627.242,
    627.30141,
    627.60327,
    627.75,
    628.358,
    628.959,
    629.37447,
    629.784,
    630.47893,
    630.6113,
    630.766,
    631.36855,
    632.1116,
    632.81646,
    633.08894,
    633.44276,
    634.5653,
    635.18532,
    636.49963,
    638.29914,
    640.1076,
    640.2248,
    640.658,
    640.97469,
    642.17044,
    644.47118,
    650.65277,
    651.0877,
    651.1789,
    651.6789,
    651.7041,
    652.5584,
    652.603,
    653.28824,
    653.8685,
    653.9143,
    653.9492,
    653.9692,
    655.0653,
    655.0653,
    655.9024,
    655.9285,
    657.13,
    657.17,
    659.89528,
    660.29007,
    664.00095,
    664.08,
    665.20925,
    666.6892,
    667.82766,
    671.7043,
    672.11342,
    673.8032,
    675.95821,
    688.694,
    689.541,
    692.94672,
    702.405,
    703.24128,
    705.12922,
    705.91079,
    706.4762,
    706.7724,
    711.23075,
    713.336,
    713.854,
    717.3938,
    721.321,
    723.51978,
    724.51665
    
    
    
    
    
    
    ])
    
    Intensity=np.array([
    
    
    
    
    
    
    1500,
    12,
    500,
    2500,
    80,
    
    1500,
    500,
    250,
    30,
    150,
    
    30,
    750,
    500,
    350,

    
    
    
    
    
    1500.00,
    5000.00,
    750.00,
    350.00,
    50.00,
    500.00,
    80.00,
    750.00,
    750.00,
    5000.00,
    40.00,
    750.00,
    250.00,
    1500.00,
    350.00,
    1500.00,
    5000.00,
    750.00,
    80.00,
    5000.00,
    700.00,
    700.00,
    30.00,
    7000.00,
    500.00,
    750.00,
    5000.00,
    3000.00,
    500.00,
    5000.00,
    750.00,
    
    20000,
    750,
    750,
    5000,
    10000,
    200,
    30,
    500,
    50,
    500,
    2500,
    2500,
    80,
    750,
    500,
    5000,
    700,
    5000,
    350,
    5000,
    6000,
    80,
    1500,
    750,
    1000,
    10000,
    150,
    500,
    500,
    10000,
    3000,
    100,
    160,
    150,
    1000,
    100,
    1000,
    10000,
    1000,
    500,
    10000,
    80,
    150,
    700,
    500,
    1500,
    50,
    700,
    500,
    150,
    1000,
    70,
    70,
    1500,
    10000,
    160,
    500,
    120,
    1000,
    50,
    120,
    20,
    80,
    180,
    1000,
    10000,
    80,
    140,
    700,
    500,
    30,
    30,
    30,
    1000,
    80,
    1000,
    100,
    180,
    1500,
    60,
    3000,
    1500,
    10000,
    60,
    1000,
    1000,
    10000,
    1000,
    20000,
    90,
    1500,
    1000,
    1500,
    15000,
    60,
    140,
    120,
    100,
    22,
    120,
    1000,
    80,
    120,
    140,
    60,
    140,
    140,
    100,
    100,
    22,
    60,
    10000,
    1000,
    100,
    50,
    1500,
    1000,
    5000,
    700,
    20,
    700,
    150,
    70,
    100,
    100000,
    34000,
    85000,
    2200,
    10000,
    80,
    80,
    110,
    40,
    55,
    77000,
    300,
    300,
    77000
    
    
    ])



    Raman_shift=(10**7/wavelength)-(10**7/Ne_emission_line_air)

    df_Ne=pd.DataFrame(data={'Raman_shift (cm-1)': Raman_shift,
                            'Intensity': Intensity,
                            'Ne emission line in air': Ne_emission_line_air})

    df_Ne_r=df_Ne.loc[df_Ne['Intensity']>cut_off_intensity]
    # Lets also only return positive numbers
    df_Ne_p=df_Ne_r.loc[df_Ne_r['Raman_shift (cm-1)']>0]
    return df_Ne_p


@dataclass
class Neon_id_config:
    # Exclude a range, e.g. cosmic rays
    exclude_range_1: Optional [Tuple[float, float]] = None
    exclude_range_2: Optional [Tuple[float, float]] = None

    # Things for Scipy find peaks
    height: tuple  = 10
    distance: float= 1
    prominence: float = 10
    width: float = 1
    threshold: float = 0.6
    peak1_cent: float= 1117
    peak2_cent: float=1447

    # Number of peaks to look for
    n_peaks: float = 6






def identify_Ne_lines(*, config: Neon_id_config=Neon_id_config(), path=None, filename=None, filetype=None, plot_figure=True, print_df=False,
Ne_array=None):

    """
    Loads Ne line, uses scipy find peaks to identify peaks, overlays these,
    and returns approximate peak positions, prominences etc to feed into fitting algorithms

    Parameters
    -----------
    config: from Neon_id_config
        This is used to identify peaks using Scipy find peaks. Parameters that can be tweaked
        exclude_range_1: None, or Tuple[float, float]
            Range to exclude (e.g, cosmic ray, instrument noise)
        exclude_range_2: None, or Tuple[float, float]
            Range to exclude (e.g, cosmic ray, instrument noise)
        height, distance, prominence, width, threshold: float
            Scipy find peak parameters you can tweak
        peak1_cent: float
            Estimate of location of Ne line 1
        peak2_cent: float
            Estimate of location of Ne line 2
        n_peaks: float
            Looks through the largest N peaks of scipy in the entire spectra and identifies them on the plot

    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read

    filetype: str
        choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'

    plot_figure: bool
        If True, plots a figure highlighting the identified peaks

    print_df: bool
        if True, prints the positions of the N biggest peaks it found

    Ne_array: np.array
        Can also enter data as a numpy array, rather than as a filename, filepath and filetype

    Returns
    --------------
    Ne, df_fit_params
    Ne: np.array of spectral data (with ranges excluded)
    df_fit_params: DataFrame of approximate peak positions, prominences etc. 


    """

    # This bit extracts the data, unless you already fed it in as an array
    if filename is not None and path is not None and filetype is not None:
        Ne_in=get_data(path=path, filename=filename, filetype=filetype)
    if Ne_array is not None:
        Ne_in=Ne_array


    # This gets parameters from config file
    exclude_range_1=config.exclude_range_1
    exclude_range_2=config.exclude_range_2
    height=config.height
    distance=config.distance
    prominence=config.prominence
    width=config.width
    threshold=config.threshold
    peak1_cent=config.peak1_cent
    peak2_cent=config.peak2_cent
    n_peaks=config.n_peaks


    # This bit filters the spectra if you want to exclude a range or 2
    if exclude_range_1 is None and exclude_range_2 is None:
        Ne=Ne_in
    if exclude_range_1 is not None:
        Ne=Ne_in[ (Ne_in[:, 0]<exclude_range_1[0]) | (Ne_in[:, 0]>exclude_range_1[1]) ]

    if exclude_range_2 is not None:
        Ne=Ne_in[ (Ne_in[:, 0]<exclude_range_2[0]) | (Ne_in[:, 0]>exclude_range_2[1]) ]

    # Get X and Y coords.
    y=Ne[:, 1]
    x=Ne[:, 0]
    spec_res=np.abs(x[1]-x[0])


    # Apply Scipy Find peaks using the parameters in the config file.
    peaks = find_peaks(y,height = height, threshold = threshold, distance = distance, prominence=prominence, width=width)

    # This gets a list of peak heights
    height = peaks[1]['peak_heights']
    # This gets a list of peak positions
    peak_pos = x[peaks[0]]
    
   

    # Lets combine them in a dataframe
    df=pd.DataFrame(data={'pos': peak_pos,
                        'height': height})


    # Find bigest peaks, and take up to n peaks
    df_sort_Ne=df.sort_values('height', axis=0, ascending=False)
    df_sort_Ne_trim=df_sort_Ne[0:n_peaks]

    if print_df is True:
        print('Biggest N peaks:')
        print(df_sort_Ne_trim)

    # Get peak within +-5
    df_pk1=df_sort_Ne.loc[df['pos'].between(peak1_cent-10*spec_res, peak1_cent+10*spec_res)]
    df_pk2=df_sort_Ne.loc[df['pos'].between(peak2_cent-10*spec_res, peak2_cent+10*spec_res)]

    df_pk1_trim=df_pk1[0:1]
    df_pk2_trim=df_pk2[0:1]

    # Lets extract spectra 10 spectral units either side of the asked for peak positions
    Neon1_region=(x<(peak1_cent+10*spec_res)) & (x>(peak1_cent-10*spec_res))
    Neon1_trim_y=y[Neon1_region]
    Neon1_trim_x=x[Neon1_region]
    Neon2_region=(x<(peak1_cent+10*spec_res)) & (x>(peak1_cent-10*spec_res))
    Neon2_trim_y=y[Neon1_region]
    Neon2_trim_x=x[Neon1_region]

    # Take 25th quantile as representative of the background position
    Baseline_Neon1=np.quantile(Neon1_trim_y, 0.1)
    Baseline_Neon2=np.quantile(Neon2_trim_y, 0.1)

    if len(df_pk1_trim)==1:

        df_fit_params=pd.DataFrame(data={'Peak1_cent': df_pk1_trim['pos'],
                                        'Peak1_height': df_pk1_trim['height']})
    elif len(df_pk1_trim)>1:
        df_pk1_max=df_pk1_trim.loc[df_pk1_trim['height']==np.max(df_pk1_trim['height'])]
        df_fit_params=pd.DataFrame(data={'Peak1_cent': df_pk1_max['pos'],
                                        'Peak1_height': df_pk1_max['height']})
    elif len(df_pk1_trim)==0:


        Max_y_Neon1=np.max(Neon1_trim_y)
        x_Neon_1=x[y==Max_y_Neon1][0]
        df_fit_params=pd.DataFrame(data={'Peak1_cent': x_Neon_1,
                                    'Peak1_height': Max_y_Neon1}, index=[0])


    if len(df_pk2_trim)==1:
        df_fit_params['Peak2_cent']=df_pk2_trim['pos'].iloc[0]
        df_fit_params['Peak2_height']=df_pk2_trim['height'].iloc[0]
    elif len(df_pk2_trim)>1:
        df_pk2_max=df_pk2_trim.loc[df_pk2_trim['height']==np.max(df_pk2['height'])]

        df_fit_params['Peak2_cent']=df_pk2_max['pos'].iloc[0]
        df_fit_params['Peak2_height']=df_pk2_max['height'].iloc[0]

    elif len(df_pk2_trim)==0:

        Max_y_Neon2=np.max(Neon2_trim_y)
        x_Neon_2=x[y==Max_y_Neon2][0]
        df_fit_params['Peak2_cent']= x_Neon_2
        df_fit_params['Peak2_height']=Max_y_Neon2


    if plot_figure is True:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(11, 3.5))
        ax0.plot(x, y, '-r')
        miny=np.min(y)
        maxy=np.max(y)


        ax0.set_ylabel('Amplitude (counts)')
        ax0.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.plot(Ne_in[:, 0], Ne_in[:, 1], '-c')
        ax1.plot(Ne_in[:, 0], Ne_in[:, 1], '.c')
        ax1.plot(Ne[:, 0], Ne[:, 1], '.r')
        ax1.plot(x, y, '-r', label='filtered')
        ax1.plot(df_sort_Ne_trim['pos'], df_sort_Ne_trim['height'], '*c', label='all pks IDed')

        ax1.plot([peak1_cent-20, peak1_cent+20], [Baseline_Neon1, Baseline_Neon1], '-g', label='approx bck' )
        #ax1.plot(peak2_cent, df_pk2_trim['height'], '*k')

        pos_pk1=str(np.round(df_fit_params['Peak1_cent'].iloc[0], 2))

        nearest_pk1=peak1_cent


        #ax1.annotate('peak=' + pos_pk1, xy=(df_fit_params['Peak1_cent'].iloc[0]+3,
        #Baseline_Neon1*1.5+700), xycoords="data", fontsize=10, rotation=90)

        ax1.annotate('peak=' + pos_pk1, xy=(0.8, 0.3),xycoords="axes fraction", fontsize=10, rotation=90)

       

        ax1.plot(df_fit_params['Peak1_cent'], df_fit_params['Peak1_height'], '*k', mfc='yellow', ms=8, label='selected peak')

        ax1.legend(loc='upper center', ncol=2, fontsize=8)


        pos_pk2=str(np.round(df_fit_params['Peak2_cent'].iloc[0], 2))
        ax2.plot([peak2_cent-20, peak2_cent+20], [Baseline_Neon2, Baseline_Neon2], '-g')

        # ax2.annotate('peak=' + pos_pk2, xy=(df_fit_params['Peak2_cent'].iloc[0]-5,
        # Baseline_Neon1*2+700), xycoords="data", fontsize=10, rotation=90)
        ax2.annotate('peak=' + pos_pk2,xy=(0.8, 0.3),xycoords="axes fraction", fontsize=10, rotation=90)



        ax1.set_xlim([peak1_cent-15, peak1_cent+15])

        ax1.set_xlim([peak1_cent-10, peak1_cent+10])
        ax2.plot(Ne[:, 0], Ne[:, 1], '.r', label='input')
        ax2.plot(x, y, '-r')
        ax2.plot(df_sort_Ne_trim['pos'], df_sort_Ne_trim['height'], '*k', mfc='yellow', ms=8)
        #print(df_pk1)


        ax2.set_xlim([peak2_cent-15, peak2_cent+15])

        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)')

        ax1.set_ylim([0, 1.5*df_fit_params['Peak1_height'].iloc[0]+100])
        ax2.set_ylim([0, 1.5*df_fit_params['Peak2_height'].iloc[0]+100])
        fig.tight_layout()

    df_fit_params['Peak1_prom']=df_fit_params['Peak1_height']-Baseline_Neon1
    df_fit_params['Peak2_prom']=df_fit_params['Peak2_height']-Baseline_Neon2
    
    df_fit_params=df_fit_params.reset_index(drop=True)
    
    


    return Ne, df_fit_params



## Ne baselines
def remove_Ne_baseline_pk(Ne, N_poly_pk1_baseline=None, Ne_center_1=None,
lower_bck=None, upper_bck1=None, upper_bck2=None, sigma_baseline=None):
    """ This function uses a defined range of values to fit a baseline of Nth degree polynomial to the baseline
    around a specified peak

    Parameters
    -----------

    Ne: np.array
        np.array of x and y coordinates from the spectra

    N_poly_pk1_baseline: int
        Degree of polynomial used to fit the background

    Ne_center_1: float
        Center position for Ne line being fitted

    lower_bck: list (length 2). default [-50, -20]
        position used for lower background relative to peak, so =[-50, -20] takes a
        background -50 and -20 from the peak center

    upper_bck1: list (length 2). default [8, 15]
        position used for 1st upper background relative to peak, so =[8, 15] takes a
        background +8 and +15 from the peak center

    upper_bck2: list (length 2). default [30, 50]
        position used for 2nd upper background relative to peak, so =[30, 50] takes a
        background +30 and +50 from the peak center

    Returns
    -----------
    y_corr, Py_base, x,  Ne_short, Py_base, Baseline_y, Baseline_x
    
    y_corr (numpy.ndarray): The corrected y-values after subtracting the fitted polynomial baseline from the original data.
    Py_base (numpy.ndarray): The y-values of the fitted polynomial baseline.
    x (numpy.ndarray): The x-values of the trimmed data within the specified range.
    Ne_short (numpy.ndarray): The trimmed data within the specified range.
    Baseline_y (numpy.ndarray): The y-values of the baseline data points.
    Baseline_x (numpy.ndarray): The x-values of the baseline data points

    """

    lower_0baseline_pk1=Ne_center_1+lower_bck[0]
    upper_0baseline_pk1=Ne_center_1+lower_bck[1]
    lower_1baseline_pk1=Ne_center_1+upper_bck1[0]
    upper_1baseline_pk1=Ne_center_1+upper_bck1[1]
    lower_2baseline_pk1=Ne_center_1+upper_bck2[0]
    upper_2baseline_pk1=Ne_center_1+upper_bck2[1]

    # Trim for entire range
    Ne_short=Ne[ (Ne[:,0]>lower_0baseline_pk1) & (Ne[:,0]<upper_2baseline_pk1) ]

    # Get actual baseline
    Baseline_with_outl=Ne_short[
    ((Ne_short[:, 0]<=upper_0baseline_pk1) &(Ne_short[:, 0]>=lower_0baseline_pk1))
    |
    ((Ne_short[:, 0]<=upper_1baseline_pk1) &(Ne_short[:, 0]>=lower_1baseline_pk1))
    |
    ((Ne_short[:, 0]<=upper_2baseline_pk1) &(Ne_short[:, 0]>=lower_2baseline_pk1))]

    # Calculates the median for the baseline and the standard deviation
    Median_Baseline=np.median(Baseline_with_outl[:, 1])
    Std_Baseline=np.std(Baseline_with_outl[:, 1])

    # Removes any points in the baseline outside of 2 sigma (helps remove cosmic rays etc).
    Baseline=Baseline_with_outl[(Baseline_with_outl[:, 1]<Median_Baseline+sigma_baseline*Std_Baseline)
                                &
                                (Baseline_with_outl[:, 1]>Median_Baseline-sigma_baseline*Std_Baseline)
                               ]

    # Fits a polynomial to the baseline of degree
    Pf_baseline = np.poly1d(np.polyfit(Baseline[:, 0], Baseline[:, 1], N_poly_pk1_baseline))
    Py_base =Pf_baseline(Ne_short[:, 0])
    Baseline_ysub=Pf_baseline(Baseline_with_outl[:, 0])
    Baseline_y=Baseline[:, 1]
    Baseline_x= Baseline[:, 0]#Baseline[:, 0]
    y_corr= Ne_short[:, 1]-  Py_base
    x=Ne_short[:, 0]


    return y_corr, Py_base, x,  Ne_short, Py_base, Baseline_y, Baseline_x




def fit_Ne_pk(x, y_corr, x_span=[-10, 8], Ne_center=1117.1, amplitude=98.1, pk1_sigma=0.28,
LH_offset_mini=[1.5, 3], peaks_pk1=2, model_name='PseudoVoigtModel', block_print=True,
const_params=True, spec_res=0.4) :
    """ This function fits the 1117 Ne line as 1 or two voigt peaks

    Parameters
    -----------

    x: np.array
        x coordinate (wavenumber)

    y: np.array
        Background corrected intensiy

    x_span: list length 2. Default [-10, 8]
        Span either side of peak center used for fitting,
        e.g. by default, fits to 10 wavenumbers below peak, 8 above.

    Ne_center: float (default=1117.1)
        Center position for Ne line being fitted

    amplitude: integer (default = 98)
        peak amplitude

    sigma: float (default =0.28)
        sigma of the voigt peak

    peaks_pk1: integer
        number of peaks to fit, e.g. 1, single voigt, 2 to get shoulder peak

    LH_offset_mini: list length 2
        Forces second peak to be within -1.5 to -3 from the main peak position.

    print_report: bool
        if True, prints fit report.


    """
    if const_params is True:
        min_off=0.8
        max_off=1.2
    if const_params is False:
        min_off=0.3
        max_off=3

    # Flatten x and y if needed
    xdat=x.flatten()
    ydat=y_corr.flatten()
    
    df=pd.DataFrame(data={'Xdata': xdat,
    'Ydata': ydat})
    #df.to_clipboard(excel=True)

    # This defines the range you want to fit (e.g. how big the tails are)
    lower_pk1=Ne_center+x_span[0]
    upper_pk1=Ne_center+x_span[1]

    # This segments into the x and y variable, and variables to plot, which are a bit bigger.
    Ne_pk1_reg_x=x[(x>lower_pk1)&(x<upper_pk1)]
    Ne_pk1_reg_y=y_corr[(x>lower_pk1)&(x<upper_pk1)]
    Ne_pk1_reg_x_plot=x[(x>(lower_pk1-3))&(x<(upper_pk1+3))]
    Ne_pk1_reg_y_plot=y_corr[(x>(lower_pk1-3))&(x<(upper_pk1+3))]

    if peaks_pk1>1:

        # Setting up lmfit
        model0 = globals()[model_name](prefix='p0_')

        
        pars0 = model0.make_params()
        pars0['p0_center'].set(Ne_center, min=Ne_center-2*spec_res, max=Ne_center+2*spec_res)
        pars0['p0_amplitude'].set(amplitude, min=amplitude*min_off, max=amplitude*max_off)
       
        
        
        init0 = model0.eval(pars0, x=xdat)
        result0 = model0.fit(ydat, pars0, x=xdat)
        Center_p0=result0.best_values.get('p0_center')
        if block_print is False:
            print('first iteration, peak Center='+str(np.round(Center_p0, 4)))

        
        Amp_p0=result0.params.get('p0_amplitude')
        if block_print is False:
            print('first iteration, peak Amplitude='+str(np.round(Amp_p0, 4)))
        fwhm_p0=result0.params.get('p0_fwhm')

        model1 = globals()[model_name](prefix='p1_')
            
        pars1 = model1.make_params()
        pars1['p1_'+ 'amplitude'].set(Amp_p0, min=min_off*Amp_p0, max=max_off*Amp_p0)
        pars1['p1_'+ 'center'].set(Center_p0, min=Center_p0-spec_res/2, max=Center_p0+spec_res/2)
        pars1['p1_'+ 'sigma'].set(pk1_sigma, min=pk1_sigma*min_off, max=pk1_sigma*max_off)


        # Second wee peak
        peak = globals()[model_name](prefix='p2_')


        pars = peak.make_params()
        minp2=Center_p0-LH_offset_mini[1]
        maxp2=Center_p0-LH_offset_mini[0]
        if block_print is False:
            print('Trying to place second peak between '+str(np.round(minp2, 2))+'and'+ str(np.round(maxp2, 2)))
        

        #pars['p2_fwhm'].set(fwhm_p0/2, min=0.001, max=fwhm_p0*5)

        pars['p2_center'].set((maxp2+minp2)/2, min=minp2,
        max=maxp2)
        pars['p2_amplitude'].set(Amp_p0/10, min=Amp_p0/15, max=Amp_p0/4)
        pars['p2_sigma'].set(pk1_sigma/1.2, min=pk1_sigma/3, max=pk1_sigma)
       

        model_combo=model1+peak
        # updating pars1, the fit to peak 1, with pars values
        pars1.update(pars)
        
        
        # Attempt at stabilizing peak fit
        result = model_combo.fit(ydat, pars1, x=xdat)
        

        # pars1['p2_center'].vary = False
        # pars1['p2_amplitude'].vary = False
        # pars1['p2_sigma'].vary = False
        
        # 
        result.params['p2_center'].vary = False
        result.params['p2_amplitude'].vary = False
        result.params['p2_sigma'].vary = False


        
        model1_only = model1
        
        pars1.update(result.params)
        
        
        #result_pk1 = model1_only.fit(ydat, pars1, x=xdat)
        result = model_combo.fit(ydat, pars1, x=xdat)




    if peaks_pk1==1:
        model_combo = globals()[model_name](prefix='p1_')
        # create parameters with initial values
        pars1 = model_combo.make_params()
        pars1['p1_amplitude'].set(amplitude,  min=amplitude*min_off, max=amplitude*max_off)
        pars1['p1_' + 'center'].set(Ne_center, min=Ne_center-2*spec_res,max=Ne_center+2*spec_res)
        pars1['p1_'+ 'sigma'].set(pk1_sigma, min=pk1_sigma*min_off, max=pk1_sigma*max_off)

    
    result = model_combo.fit(ydat, pars1, x=xdat)
    
    init = model_combo.eval(pars1, x=xdat)
    # Need to check errors output
    Error_bars=result.errorbars

    # Get center value
    Center_p1=result.best_values.get('p1_center')
    error_pk1 = result.params['p1_center'].stderr
    error_pk1_amp = result.params['p1_amplitude'].stderr
    
  
    # Get mix of lorenz
    Peak1_Prop_Lor=result.best_values.get('p1_fraction')




    Area_pk1=result.best_values.get('p1_amplitude')
    sigma_pk1=result.best_values.get('p1_sigma')
    gamma_pk1=result.best_values.get('p1_gamma')


    # Evaluate the peak at 100 values for pretty plotting
    xx_pk1=np.linspace(lower_pk1, upper_pk1, 2000)

    result_pk1=result.eval(x=xx_pk1)
    comps=result.eval_components(x=xx_pk1)


    result_pk1_origx=result.eval(x=Ne_pk1_reg_x)

    Center_pk1=Center_p1


    return Center_pk1, Area_pk1, sigma_pk1, gamma_pk1, Ne_pk1_reg_x_plot, Ne_pk1_reg_y_plot, Ne_pk1_reg_x, Ne_pk1_reg_y, xx_pk1, result_pk1, error_pk1, result_pk1_origx, comps, Peak1_Prop_Lor, error_pk1_amp




## Setting default Ne fitting parameters
@dataclass
class Ne_peak_config:

    model_name: str = 'PseudoVoigtModel'
    # Things for the background positioning and fit
    N_poly_pk1_baseline: float = 1 #Degree of polynomial to fit to the baseline
    N_poly_pk2_baseline: float = 1 #Degree of polynomial to fit to the baseline
    sigma_baseline=3 # Discard things outside of this sigma on the baseline
    lower_bck_pk1: Tuple[float, float] = (-50, -25) # Background position Pk1
    upper_bck1_pk1: Tuple[float, float] = (8, 15)  # Background position Pk1
    upper_bck2_pk1: Tuple[float, float] = (30, 50)     # Background position Pk1
    lower_bck_pk2: Tuple[float, float] = (-44.2, -22)  # Background position Pk2
    upper_bck1_pk2: Tuple[float, float] = (15, 50) # Background position Pk2
    upper_bck2_pk2: Tuple[float, float] = (50, 51)     # Background position Pk1

    # Whether you want a secondary peak
    peaks_1: float=2
    peaks_2: float=1

    # SPlitting
    DeltaNe_ideal: float= 330.477634

    # Things for plotting the baseline
    x_range_baseline_pk1: float=20 #  How many units outside your selected background it shows on the baseline plot
    y_range_baseline_pk1: float= 200    # Where the y axis is cut off above the minimum baseline measurement
    x_range_baseline_pk2: float=20 #  How many units outside your selected background it shows on the baseline plot
    y_range_baseline_pk2: float= 200    # Where the y axis is cut off above the minimum baseline measurement


    # Sigma for peaks
    pk1_sigma: float = 0.4
    pk2_sigma: float = 0.4

    # Things for plotting the primary peak
    x_range_peak: float=15 # How many units to each side are shown when plotting the peak fits

    # Things for plotting the residual
    x_range_residual: float=7 # Shows how many x units to left and right is shown on residual plot

    # Things for fitting a secondary peak on pk1
    LH_offset_mini: Tuple[float, float] = (1.5, 3)
    # Same for Pk2
    LH_offset_mini2: Tuple[float, float] = None
    
    

    # Optional, by default, fits to the points inside the baseline. Can also specify as values to make a smaller peak fit.
    x_span_pk1: Optional [Tuple[float, float]] = None # Tuple[float, float] = (-10, 8)
    x_span_pk2: Optional [Tuple[float, float]] = None # Tuple[float, float] = (-5, 5)



def fit_Ne_lines(*,  config: Ne_peak_config=Ne_peak_config(),
Ne_center_1=1117.1, Ne_center_2=1147, Ne_prom_1=100, Ne_prom_2=200,
Ne=None, filename=None, path=None, prefix=True,
plot_figure=True, loop=True,
 save_clipboard=False,
    close_figure=False, const_params=True):






    """ This function reads in a user file, fits the Ne lines, and if required, saves an image
    into a new sub folder

    Parameters
    -----------

    Ne: np.array
        x coordinate (wavenumber) and y coordinate (intensity)

    filename and path: str
        used to save filename in datatable, and to make a new folder.
        
    prefix: bool
        Whether or not the filename has a prefix
        
    Ne_center_1, Ne_center_2 : float
        Approximate peak position of the 1st and 2nd Ne line

    Ne_prom_1, Ne_prom_2 : float
        Approximate prominance of the 1st and 2nd Ne line      
        
    plot_figure: bool
        Plots a figure, nice to inspect fits, makes it slower
        
    loop: bool
        
    save_clipboard: bool
        Saves results to clipboard if true
    
    close_figure: bool
        Closes figure if True (useful in some editors)
        
    const_params: bool 
        If true, means amplitude and peak sigma have to be closer to the guessed parameters (e.g. used after initial fit).  
        E.g. forced within +-20% of estimated peak parameters if True, if false, +-100%.    
        

    config parameters from Ne_peak_config():
    
        model_name: str
            allowed_models = "VoigtModel", "PseudoVoigtModel", "Pearson4Model", "SkewedVoigtModel"
            
        N_poly_pk1_baseline, N_poly_pk1_baseline: int (default 1)
            Degree of polynomial to fit to baseline around pk 
            
        lower_bck_pk1, upper_bck1_pk1, upper_bck2_pk1: tuple
        lower_bck_pk2, upper_bck1_pk2, upper_bck2_pk2: tuple
            Background positions relative to estimated peak center. 
            
        peaks_1, peaks_2: int
            Number of peaks to fit to peak 1. If you need 2 overlapping peaks, put this Ne line as pk1
            if not 1, you also need LH_offset_mini=(1.5, 3). Means second peak put between 1.5 and 3 units left of the 1st peak.
            
        DeltaNe_ideal: float
            Ideal distance between two lines calculated for your laser wavelength. 
            
        x_range_baseline_pk1, x_range_baseline_pk2: int, float
            test
            
        y_range_baseline_pk1, y_range_baseline_pk2: int, float
            test
        
        pk1_sigma, pk2_sigma: float (Default 0.4)
            Estimated sigma of each peak
        
        x_range_peak:  flt, int, or None
            How much to either side of the peak to show on the final peak fitting plot
        
        x_range_residual: flt
            How much either side of the peak is used for calculating residual
        
        x_span_pk1,  x_span_pk2: float, default None
            By default, function fits up to background, but can shrink that range here. 
            
    Returns
    ------------
    if loop is False:
        return df, Ne_pk1_reg_x_plot, Ne_pk1_reg_y_plot
    if loop is True:
        return df
        
    Also returns figure. 
    
    """
    # Check model is supported
    if config.model_name not in allowed_models:
        raise ValueError(f"Unsupported model: {config.model_name}. Supported models are: {', '.join(allowed_models)}")


    
    

    # check they havent messed up background
    if config.lower_bck_pk1[0]>config.lower_bck_pk1[1]:
        raise TypeError('Your left hand number needs to be a smaller number than the right, e.g. you can have [2, 5] but you cant have [5, 2]')
    if config.upper_bck1_pk1[0]>config.upper_bck1_pk1[1]:
        raise TypeError('Your left hand number needs to be a smaller number than the right, e.g. you can have [2, 5] but you cant have [5, 2]')
    if config.upper_bck2_pk1[0]>config.upper_bck2_pk1[1]:
        raise TypeError('Your left hand number needs to be a smaller number than the right, e.g. you can have [2, 5] but you cant have [5, 2]')

    # check they havent messed up background
    if config.lower_bck_pk2[0]>config.lower_bck_pk2[1]:
        raise TypeError('Your left hand number needs to be a smaller number than the right, e.g. you can have [2, 5] but you cant have [5, 2]')
    if config.upper_bck1_pk2[0]>config.upper_bck1_pk2[1]:
        raise TypeError('Your left hand number needs to be a smaller number than the right, e.g. you can have [2, 5] but you cant have [5, 2]')
    if config.upper_bck2_pk2[0]>config.upper_bck2_pk2[1]:
        raise TypeError('Your left hand number needs to be a smaller number than the right, e.g. you can have [2, 5] but you cant have [5, 2]')
   
    
    x=Ne[:, 0]
    spec_res=np.abs(x[1]-x[0])
    # Getting things from config file
    peaks_1=config.peaks_1
    peaks_2=config.peaks_2
    DeltaNe_ideal=config.DeltaNe_ideal

    # Estimate amplitude from prominence and sigma you entered
    Pk1_Amp=((config.pk1_sigma)*(Ne_prom_1))/0.3939
    Pk2_Amp=((config.pk2_sigma)*(Ne_prom_2))/0.3939


    #Remove the baselines
    y_corr_pk1, Py_base_pk1, x_pk1, Ne_short_pk1, Py_base_pk1, Baseline_ysub_pk1, Baseline_x_pk1=remove_Ne_baseline_pk(Ne,
    N_poly_pk1_baseline=config.N_poly_pk1_baseline, Ne_center_1=Ne_center_1, sigma_baseline=config.sigma_baseline,
    lower_bck=config.lower_bck_pk1, upper_bck1=config.upper_bck1_pk1, upper_bck2=config.upper_bck2_pk1)

    y_corr_pk2, Py_base_pk2, x_pk2, Ne_short_pk2, Py_base_pk2, Baseline_ysub_pk2, Baseline_x_pk2=remove_Ne_baseline_pk(Ne, Ne_center_1=Ne_center_2, N_poly_pk1_baseline=config.N_poly_pk2_baseline, sigma_baseline=config.sigma_baseline,
    lower_bck=config.lower_bck_pk2, upper_bck1=config.upper_bck1_pk2, upper_bck2=config.upper_bck2_pk2)


    # Have the option to override the xspan here from default. Else, trims
    if config.x_span_pk1 is None:

        x_span_pk1=[config.lower_bck_pk1[1], config.upper_bck1_pk1[0]]
        x_span_pk1_dist=abs(config.lower_bck_pk1[1]-config.upper_bck1_pk1[0])
    else:
        x_span_pk1=config.x_span_pk1
        x_span_pk1_dist=abs(config.x_span_pk1[1]-config.x_span_pk1[0])

    if config.x_span_pk2 is None:
        x_span_pk2=[config.lower_bck_pk2[1], config.upper_bck1_pk2[0]]
        x_span_pk2_dist=abs(config.lower_bck_pk2[1]-config.upper_bck1_pk2[0])
    else:
        x_span_pk2=config.x_span_pk2
        x_span_pk2_dist=abs(config.x_span_pk2[1]-config.x_span_pk2[0])

    # Fit Pk1
    cent_pk1, Area_pk1, sigma_pk1, gamma_pk1, Ne_pk1_reg_x_plot, Ne_pk1_reg_y_plot, Ne_pk1_reg_x, Ne_pk1_reg_y, xx_pk1, result_pk1, error_pk1, result_pk1_origx, comps, Peak1_Prop_Lor, error_pk1_amp= fit_Ne_pk(x_pk1, y_corr_pk1, x_span=x_span_pk1, Ne_center=Ne_center_1, model_name=config.model_name,  LH_offset_mini=config.LH_offset_mini, peaks_pk1=peaks_1, amplitude=Pk1_Amp, pk1_sigma=config.pk1_sigma,
    const_params=const_params, spec_res=spec_res)

    # Fit pk2 
    cent_pk2,Area_pk2, sigma_pk2, gamma_pk2, Ne_pk2_reg_x_plot, Ne_pk2_reg_y_plot, Ne_pk2_reg_x, Ne_pk2_reg_y, xx_pk2, result_pk2, error_pk2, result_pk2_origx, comps2, Peak2_Prop_Lor, error_pk2_amp = fit_Ne_pk( x_pk2, y_corr_pk2, x_span=x_span_pk2,  Ne_center=Ne_center_2, model_name=config.model_name, LH_offset_mini=config.LH_offset_mini2, peaks_pk1=peaks_2, amplitude=Pk2_Amp, pk1_sigma=config.pk2_sigma, const_params=const_params,spec_res=spec_res)


    # Calculate difference between peak centers, and Delta Ne
    DeltaNe=cent_pk2-cent_pk1


    Ne_Corr=DeltaNe_ideal/DeltaNe


    # Calculate maximum splitting (+1 sigma)
    error_pk1 = error_pk1 if error_pk1 is not None else 0
    error_pk2 = error_pk2 if error_pk2 is not None else 0

    
    DeltaNe_max=(cent_pk2+error_pk2)-(cent_pk1-error_pk1)
    DeltaNe_min=(cent_pk2-error_pk2)-(cent_pk1+error_pk1)
    Ne_Corr_max=DeltaNe_ideal/DeltaNe_min
    Ne_Corr_min=DeltaNe_ideal/DeltaNe_max

    # Calculating least square residual
    residual_pk1=np.sum(((Ne_pk1_reg_y-result_pk1_origx)**2)**0.5)/(len(Ne_pk1_reg_y))
    residual_pk2=np.sum(((Ne_pk2_reg_y-result_pk2_origx)**2)**0.5)/(len(Ne_pk2_reg_y))

    if plot_figure is True:
        # Make a summary figure of the backgrounds and fits
        fig, ((ax3, ax2), (ax5, ax4), (ax1, ax0)) = plt.subplots(3,2, figsize = (12,15)) # adjust dimensions of figure here
        fig.suptitle(filename, fontsize=16)

        # Setting y limits of axis
        ymin_ax1=min(Ne_short_pk1[:,1])-10
        ymax_ax1=min(Ne_short_pk1[:,1])+config.y_range_baseline_pk1
        ymin_ax0=min(Ne_short_pk2[:,1])-10
        ymax_ax0=min(Ne_short_pk2[:,1])+config.y_range_baseline_pk2
        ax1.set_ylim([ymin_ax1, ymax_ax1])
        ax0.set_ylim([ymin_ax0, ymax_ax0])

        # Setting x limits of axis

        ax1_xmin=min(Ne_short_pk1[:,0])-config.x_range_baseline_pk1
        ax1_xmax=max(Ne_short_pk1[:,0])+config.x_range_baseline_pk1
        ax0_xmin=min(Ne_short_pk2[:,0])-config.x_range_baseline_pk2
        ax0_xmax=max(Ne_short_pk2[:,0])+config.x_range_baseline_pk2
        ax0.set_xlim([ax0_xmin, ax0_xmax])
        ax1.set_xlim([ax1_xmin, ax1_xmax])

        # Adding background positions as colored bars on pk1
        ax1cop=ax1.twiny()
        #ax1cop.set_zorder(ax1.get_zorder())
        ax1cop_xmax=ax1_xmax-Ne_center_1
        ax1cop_xmin=ax1_xmin-Ne_center_1
        ax1cop.set_xlim([ax1cop_xmin, ax1cop_xmax])
        rect_pk1_b1=patches.Rectangle((config.lower_bck_pk1[0],ymin_ax1),config.lower_bck_pk1[1]-config.lower_bck_pk1[0],ymax_ax1-ymin_ax1,
                              linewidth=1,edgecolor='none',facecolor='cyan', label='bck_pk1', alpha=0.3, zorder=0)
        ax1cop.add_patch(rect_pk1_b1)
        rect_pk1_b2=patches.Rectangle((config.upper_bck1_pk1[0],ymin_ax1),config.upper_bck1_pk1[1]-config.upper_bck1_pk1[0],ymax_ax1-ymin_ax1,
                              linewidth=1,edgecolor='none',facecolor='cyan', label='bck_pk2', alpha=0.3, zorder=0)
        ax1cop.add_patch(rect_pk1_b2)
        rect_pk1_b3=patches.Rectangle((config.upper_bck2_pk1[0],ymin_ax1),config.upper_bck2_pk1[1]-config.upper_bck2_pk1[0],ymax_ax1-ymin_ax1,
                              linewidth=1,edgecolor='none',facecolor='cyan', label='bck_pk3', alpha=0.3, zorder=0)
        ax1cop.add_patch(rect_pk1_b3)

        # Adding background positions as colored bars on pk2
        ax0cop=ax0.twiny()
        ax0cop_xmax=ax0_xmax-Ne_center_2
        ax0cop_xmin=ax0_xmin-Ne_center_2
        ax0cop.set_xlim([ax0cop_xmin, ax0cop_xmax])
        rect_pk2_b1=patches.Rectangle((config.lower_bck_pk2[0],ymin_ax0),config.lower_bck_pk2[1]-config.lower_bck_pk2[0],ymax_ax0-ymin_ax0,
                              linewidth=1,edgecolor='none',facecolor='cyan', label='bck_pk2', alpha=0.3)
        ax0cop.add_patch(rect_pk2_b1)
        rect_pk2_b2=patches.Rectangle((config.upper_bck1_pk2[0],ymin_ax0),config.upper_bck1_pk2[1]-config.upper_bck1_pk2[0],ymax_ax0-ymin_ax0,
                              linewidth=1,edgecolor='none',facecolor='cyan', label='bck_pk2', alpha=0.3)
        ax0cop.add_patch(rect_pk2_b2)
        rect_pk2_b3=patches.Rectangle((config.upper_bck2_pk2[0],ymin_ax0),config.upper_bck2_pk2[1]-config.upper_bck2_pk2[0],ymax_ax0-ymin_ax0,
                              linewidth=1,edgecolor='none',facecolor='cyan', label='bck_pk3', alpha=0.3)
        ax0cop.add_patch(rect_pk2_b3)

        # Plotting trimmed data and background
        ax0.plot(Ne_short_pk2[:,0], Py_base_pk2, '-k', label='Fit. Bck')
        ax0.plot(Ne_short_pk2[:,0], Ne_short_pk2[:,1], '-r')

        ax1.plot(Ne_short_pk1[:,0], Py_base_pk1, '-k', label='Fit. Bck')
        ax1.plot(Ne_short_pk1[:,0], Ne_short_pk1[:,1], '-r')

        # Plotting data actually used in the background, after sigma exclusion
        ax1.plot(Baseline_x_pk1, Baseline_ysub_pk1, '.b', ms=6, label='Bck')
        ax0.plot(Baseline_x_pk2, Baseline_ysub_pk2, '.b', ms=5, label='Bck')


        mean_baseline=np.mean(Py_base_pk2)
        std_baseline=np.std(Py_base_pk2)

        std_baseline=np.std(Py_base_pk1)


        ax1.plot([Ne_center_1, Ne_center_1], [ymin_ax1, ymax_ax1], ':k', label='Peak')
        ax0.plot([Ne_center_2, Ne_center_2], [ymin_ax0, ymax_ax0], ':k', label='Peak')

        ax1.set_title('%.0f' %Ne_center_1+': background fitting')
        ax1.set_xlabel('Wavenumber')
        ax1.set_ylabel('Intensity')
        ax0.set_title('%.0f' %Ne_center_2+ ': background fitting')
        ax0.set_xlabel('Wavenumber')
        ax0.set_ylabel('Intensity')
        ax1cop.set_xlabel('Offset from Pk estimate')
        ax0cop.set_xlabel('Offset from Pk estimate')

        #Showing all data, not just stuff fit


        ax0.plot(Ne[:,0], Ne[:,1], '-', color='grey', zorder=0)
        ax1.plot(Ne[:,0], Ne[:,1], '-', color='grey', zorder=0)



        # ax1.legend()
        # ax0.legend()
        ax0.legend(loc='lower right', bbox_to_anchor= (-0.08, 0.8), ncol=1,
            borderaxespad=0, frameon=True, facecolor='white')

        # Actual peak fits.
        ax2.plot(Ne_pk2_reg_x_plot, Ne_pk2_reg_y_plot, 'xb', label='all data')
        ax2.plot(Ne_pk2_reg_x, Ne_pk2_reg_y, 'ok', label='data fitted')
        ax2.plot(xx_pk2, result_pk2, 'r-', label='interpolated fit')
        ax2.set_title('%.0f' %Ne_center_2+' peak fitting')
        ax2.set_xlabel('Wavenumber')
        ax2.set_ylabel('Intensity')
        if config.x_range_peak is None:
            ax2.set_xlim([cent_pk2-x_span_pk2_dist/2, cent_pk2+x_span_pk2_dist/2])
        else:
            ax2.set_xlim([cent_pk2-config.x_range_peak, cent_pk2+config.x_range_peak])


        ax3.plot(Ne_pk1_reg_x_plot, Ne_pk1_reg_y_plot, 'xb', label='all data')
        ax3.plot(Ne_pk1_reg_x, Ne_pk1_reg_y, 'ok', label='data fitted')

        ax3.set_title('%.0f' %Ne_center_1+' peak fitting')
        ax3.set_xlabel('Wavenumber')
        ax3.set_ylabel('Intensity')
        ax3.plot(xx_pk1, comps.get('p1_'), '-r', label='p1')
        if peaks_1>1:
            ax3.plot(xx_pk1, comps.get('p2_'), '-c', label='p2')
        ax3.plot(xx_pk1, result_pk1, 'g-', label='best fit')
        ax3.legend()
        if config.x_range_peak is None:
            ax3.set_xlim([cent_pk1-x_span_pk1_dist/2, cent_pk1+x_span_pk1_dist/2])
        else:
            ax3.set_xlim([cent_pk1-config.x_range_peak, cent_pk1+config.x_range_peak ])

        # Residuals for peak fits.
        ax4.plot(Ne_pk2_reg_x, Ne_pk2_reg_y-result_pk2_origx, '-r', label='residual')
        ax5.plot(Ne_pk1_reg_x, Ne_pk1_reg_y-result_pk1_origx, '-r',  label='residual')
        ax4.plot(Ne_pk2_reg_x, Ne_pk2_reg_y-result_pk2_origx, 'ok', mfc='r', label='residual')
        ax5.plot(Ne_pk1_reg_x, Ne_pk1_reg_y-result_pk1_origx, 'ok', mfc='r', label='residual')
        ax4.set_ylabel('Residual (Intensity units)')
        ax4.set_xlabel('Wavenumber')
        ax5.set_ylabel('Residual (Intensity units)')
        ax5.set_xlabel('Wavenumber')
        Residual_pk2=Ne_pk2_reg_y-result_pk2_origx
        Residual_pk1=Ne_pk1_reg_y-result_pk1_origx

        Local_Residual_pk2=Residual_pk2[(Ne_pk2_reg_x>cent_pk2-config.x_range_residual)&(Ne_pk2_reg_x<cent_pk2+config.x_range_residual)]
        Local_Residual_pk1=Residual_pk1[(Ne_pk1_reg_x>cent_pk1-config.x_range_residual)&(Ne_pk1_reg_x<cent_pk1+config.x_range_residual)]
        ax5.set_xlim([cent_pk1-config.x_range_residual, cent_pk1+config.x_range_residual])
        ax4.set_xlim([cent_pk2-config.x_range_residual, cent_pk2+config.x_range_residual])
        ax5.plot([cent_pk1, cent_pk1 ], [np.min(Local_Residual_pk1)-10, np.max(Local_Residual_pk1)+10], ':k')
        ax4.plot([cent_pk2, cent_pk2 ], [np.min(Local_Residual_pk2)-10, np.max(Local_Residual_pk2)+10], ':k')
        ax5.set_ylim([np.min(Local_Residual_pk1)-10, np.max(Local_Residual_pk1)+10])
        ax4.set_ylim([np.min(Local_Residual_pk2)-10, np.max(Local_Residual_pk2)+10])
        fig.tight_layout()

        # Save figure
        path3=path+'/'+'Ne_fit_images'
        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ 'Ne_fit_images', exist_ok=False)

        figure_str=path+'/'+ 'Ne_fit_images'+ '/'+ filename+str('_Ne_Line_Fit')+str('.png')

        fig.savefig(figure_str, dpi=200)
        if close_figure is True:
            plt.close(fig)

    if prefix is True:
        filename = filename.split(' ', 1)[1]
        
        
        
    df=pd.DataFrame(data={'filename': filename,
                          'pk2_peak_cent':cent_pk2,
                          'pk2_amplitude': Area_pk2,
                          'pk2_sigma': sigma_pk2,
                          'pk2_gamma': gamma_pk2,
                          'error_pk2': error_pk2,
                          'Peak2_Prop_Lor': Peak2_Prop_Lor,
                          'pk1_peak_cent':cent_pk1,
                          'pk1_amplitude': Area_pk1,
                          'pk1_sigma': sigma_pk1,
                          'pk1_gamma': gamma_pk1,
                          'error_pk1': error_pk1,
                          'Peak1_Prop_Lor': Peak1_Prop_Lor,

                         'deltaNe': DeltaNe,
                         'Ne_Corr': Ne_Corr,
                         'Ne_Corr_min':Ne_Corr_min,
                         'Ne_Corr_max': Ne_Corr_max,
                         'residual_pk2':residual_pk2,
                         'residual_pk1': residual_pk1,
                         'residual_pk1+pk2':residual_pk1+residual_pk2,
                         'error_pk1_amplitude': error_pk1_amp,
                         'error_pk2_amplitude': error_pk2_amp
                         }, index=[0])
                         
                         
    df_combo=df
    pk1_peak_cent_values = df_combo['pk1_peak_cent'].values
    pk1_peak_cent_errors = df_combo['error_pk1'].fillna(0).infer_objects().values
    pk2_peak_cent_values = df_combo['pk2_peak_cent'].values
    pk2_peak_cent_errors = df_combo['error_pk2'].fillna(0).infer_objects().values
    
    constant=df_combo['deltaNe']
    
    # Calculate the error on Ne_Corr using error propagation (quadrature)
    Ne_Corr_errors = np.sqrt((pk1_peak_cent_errors / constant) ** 2 + (pk2_peak_cent_errors / constant) ** 2)
    
    test_err=np.sqrt(pk1_peak_cent_errors ** 2 + pk2_peak_cent_errors ** 2)
    
    total_err=test_err/constant
    
    df.insert(1,'1σ_Ne_Corr', Ne_Corr_errors)
    df.insert(1,'1σ_Ne_Corr_test', total_err)
    
    if save_clipboard is True:
        df.to_clipboard(excel=True, header=False, index=False)

    if loop is False:
        return df, Ne_pk1_reg_x_plot, Ne_pk1_reg_y_plot
    if loop is True:
        return df



## Plot to help inspect which Ne lines to discard
def plot_Ne_corrections(df=None, x_axis=None, x_label='index', marker='o', mec='k',
                       mfc='r'):
    if x_axis is not None:
        x=x_axis
    else:
        x=df.index
    fig, ((ax5, ax6),  (ax3, ax4), (ax1, ax2)) = plt.subplots(3, 2, figsize=(10, 12))

    # Pk1 center vs. X

    ax5.errorbar(x, df['pk1_peak_cent'], xerr=0, yerr=df['error_pk1'].fillna(0).infer_objects(),
             fmt='o', ecolor='k', elinewidth=0.8, mfc='b', ms=5, mec='k', capsize=3)
    ax5.set_xlabel(x_label)
    ax5.set_ylabel('Peak 1 center')
             
    # Pk2 center vs. X
             
    ax6.plot(x, df['pk2_peak_cent'], marker,  mec='k', mfc='r')
    ax6.errorbar(x, df['pk2_peak_cent'], xerr=0, yerr=df['error_pk2'].fillna(0).infer_objects(),
             fmt='o', ecolor='k', elinewidth=0.8, mfc='r', ms=5, mec='k', capsize=3)
    
    ax6.set_xlabel(x_label)
    ax6.set_ylabel('Peak 2 center')
    
    # 
    ax3.errorbar(df['Ne_Corr'], df['pk2_peak_cent'], xerr=df['1σ_Ne_Corr'].fillna(0).infer_objects(), 
    yerr=df['error_pk2'].fillna(0).infer_objects(),
             fmt='o', ecolor='k', elinewidth=0.8, mfc='b', ms=5, mec='k', capsize=3)
             
             
    ax3.set_xlabel('Ne Correction factor')
    ax3.set_ylabel('Peak 2 center')


    ax4.errorbar(df['Ne_Corr'], df['pk1_peak_cent'], xerr=df['1σ_Ne_Corr'].fillna(0).infer_objects(), 
    yerr=df['error_pk1'].fillna(0).infer_objects(),
             fmt='o', ecolor='k', elinewidth=0.8, mfc='b', ms=5, mec='k', capsize=3)
    ax4.set_xlabel('Ne Correction factor')
    ax4.set_ylabel('Peak 1 center')
    
    # Ne correction factor vs. time
    ax1.errorbar(x, df['Ne_Corr'], xerr=0, yerr=df['1σ_Ne_Corr'].fillna(0).infer_objects(),
             fmt='o', ecolor='k', elinewidth=0.8, mfc='grey', ms=5, mec='k',capsize=3)
    
    ax1.set_ylabel('Ne Correction factor')
    ax1.set_xlabel(x_label)

    # Ne correction factor vs. residual


   
    ax2.set_xlabel('Sum of pk1 and pk2 residual')
    ax2.set_ylabel('Ne Correction factor')




    plt.setp(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax4.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax1.ticklabel_format(useOffset=False)
    ax5.ticklabel_format(useOffset=False)
    ax6.ticklabel_format(useOffset=False)
    ax2.ticklabel_format(useOffset=False)
    ax3.ticklabel_format(useOffset=False)
    ax4.ticklabel_format(useOffset=False)
    fig.tight_layout()
    return fig

## Looping Ne lines
def loop_Ne_lines(*, files, spectra_path, filetype,
        config, config_ID_peaks, df_fit_params=None, prefix=False, print_df=False,
                  plot_figure=True, single_acq=False, const_params=True):

    df = pd.DataFrame([])
    # This is for repeated acquisition of Ne lines
    if single_acq is True:

        for i in tqdm(range(0, np.shape(files)[1]-2)):
            filename=str(i)
            Ne=np.column_stack((files[:, 0], files[:, i+1]))
            

            Ne, df_fit_params=identify_Ne_lines(Ne_array=Ne,
            config=config_ID_peaks, print_df=False, plot_figure=False)


            data=fit_Ne_lines(Ne=Ne, path=spectra_path,
            config=config, prefix=prefix,
            Ne_center_1=df_fit_params['Peak1_cent'].iloc[0],
            Ne_center_2=df_fit_params['Peak2_cent'].iloc[0],
            Ne_prom_1=df_fit_params['Peak1_prom'].iloc[0],
            Ne_prom_2=df_fit_params['Peak2_prom'].iloc[0],
            const_params=const_params,
            plot_figure=plot_figure)
            df = pd.concat([df, data], axis=0)

    else:
        for i in tqdm(range(0, len(files))):
            filename=files[i]

            Ne, df_fit_params=identify_Ne_lines(path=spectra_path,
            filename=filename, filetype=filetype,
            config=config_ID_peaks, print_df=False, plot_figure=False)

            data=fit_Ne_lines(Ne=Ne, filename=filename,
            path=spectra_path, prefix=prefix,
            config=config,
            Ne_center_1=df_fit_params['Peak1_cent'].iloc[0],
            Ne_center_2=df_fit_params['Peak2_cent'].iloc[0],
            Ne_prom_1=df_fit_params['Peak1_prom'].iloc[0],
            Ne_prom_2=df_fit_params['Peak2_prom'].iloc[0],
            const_params=const_params,
            plot_figure=plot_figure)
            df = pd.concat([df, data], axis=0)





#print('working on ' + str(files[i]))


    df2=df.reset_index(drop=True)


    # Now lets reorder some columns

    cols_to_move = ['filename', 'Ne_Corr', '1σ_Ne_Corr', 'deltaNe', 'pk2_peak_cent', 'pk1_peak_cent', 'pk2_amplitude', 'pk1_amplitude', 'residual_pk2', 'residual_pk1']
    df2 = df2[cols_to_move + [
                col for col in df2.columns if col not in cols_to_move]]
                




    return df2

## Regressing Ne lines against time
from scipy.interpolate import interp1d
def reg_Ne_lines_time(df, fit='poly', N_poly=None, spline_fit=None):
    """
    Parameters
    -----------
    df: pd.DataFrame
        dataframe of stitched Ne fits and metadata information from WITEC,
        must have columns 'sec since midnight' and 'Ne_Corr'

    fit: float 'poly', or 'spline'
        If 'poly':
            N_poly: int, degree of polynomial to fit (1 if linear)
        if 'spline':
            spline_fit: The string has to be one of:
        ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
        ‘quadratic’, ‘cubic’, ‘previous’. Look up documentation for interpld

    Returns
    -----------
    figure of fit and data used to make it
    Pf: fit model, can be used to evaluate unknown data (only within x range of df for spline fits).




    """
    Px=np.linspace(np.min(df['sec since midnight']), np.max(df['sec since midnight']),
                         101)
    if fit=='poly':
        Pf = np.poly1d(np.polyfit(df['sec since midnight'], df['Ne_Corr'],
                              N_poly))

    if fit == 'spline':
            if spline_fit is None:
                raise TypeError('If you choose spline you also need to choose the type of spline fit. do help of this function to get the options')
            else:
                Pf = interp1d(df['sec since midnight'], df['Ne_Corr'], kind=spline_fit)

    Py=Pf(Px)

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 3))
    ax1.plot(df['sec since midnight'], df['Ne_Corr'], 'xk')
    ax1.plot(Px, Py, '-r')
    ax1.set_xlabel('Seconds since midnight')
    ax1.set_ylabel('Ne Correction Factor')

    ax1.ticklabel_format(useOffset=False)


    return Pf, fig


def filter_Ne_Line_neighbours(*, df_combo=None, Corr_factor=None, number_av=6, offset=0.00005, file_name_filt=None):
    """ This function discards Ne lines with a correction factor more than 
    offset away from the median value of the N points (number_av) either side of it.
    

    """
    if df_combo is not None:
        Corr_factor=df_combo['Ne_Corr']
    Corr_factor_Filt=np.zeros(len(Corr_factor), dtype=float)
    median_loop=np.zeros(len(Corr_factor), dtype=float)

    for i in range(0, len(Corr_factor)):
        if i<len(Corr_factor)/2: # For first half, do 5 after
            median_loop[i]=np.nanmedian(Corr_factor[i:i+number_av])
        if i>=len(Corr_factor)/2: # For first half, do 5 after
            median_loop[i]=np.nanmedian(Corr_factor[i-number_av:i])
        if Corr_factor[i]>(median_loop[i]+offset) or Corr_factor[i]<(median_loop[i]-offset) :
            Corr_factor_Filt[i]=np.nan
        else:
            Corr_factor_Filt[i]=Corr_factor[i]
    ds=pd.Series(Corr_factor_Filt)
    
    # Filter based on file names
    if file_name_filt is not None:
        pattern = '|'.join(file_name_filt)
        mask = df_combo['filename_x'].str.contains(pattern)
        filtered_ds = ds.where(~mask, np.nan)

    
        return filtered_ds
    else:
        return ds


## Lets make a plotting function for this notebook 

def generate_Ne_corr_model(*, time, Ne_corr, N_poly=3, CI=0.67, bootstrap=False, std_error=True, N_bootstrap=500,save_fig=False, pkl_name='polyfit_data.pkl'):
    """ This function takes time stamp and Ne correctoin data to make a predictive polynomial, which it then saves as a pkl file
    
    Parameters
    ---------------
    time: pd.Series
        Time through the run
        
    Ne_corr: df or pd.Series
        if dataframe, has to have the column 'Ne_corr'
        Else, just Ne correction factor as a pd.Series itself
        
    CI: float. Default 0.67
        Confidence interval to save as uncertainty on model. 
        
    Either:
        
    std_error: bool (True)
        calculates uncertainty on model using CI
        
    Or:
    boostrap: bool (False)
        Used for testing of method, keep as False
    
    save_fig:bool
        Saves figure. 
        
    pkl_name: str
        Name of model that is saved. 
        
        
    
        
    
    """
# Define the x and y values
    x_all   = np.array([time])
    if isinstance(Ne_corr, pd.DataFrame):
        y_all = np.array([Ne_corr['Ne_Corr']])
        y_err=Ne_corr['1σ_Ne_Corr']
    else:
        y_all=Ne_corr
        y_err=0*Ne_corr
    non_nan_indices = ~np.isnan(x_all) & ~np.isnan(y_all)

    # Filter out NaN values
    x = x_all[non_nan_indices]
    y = y_all[non_nan_indices]
    # Perform polynomial regression
  
    coefficients = np.polyfit(x, y, N_poly)
    Pf = np.poly1d(coefficients)


    # Save the model and the data to a pickle file
    data = {'model': Pf, 'x': x, 'y': y}
    with open(pkl_name, 'wb') as f:
        pickle.dump(data, f)
        
    if bootstrap is True:

        new_x_plot=np.linspace(np.min(x), np.max(x), 100)
        Ne_corr2=calculate_Ne_corr_bootstrap_values(pickle_str=pkl_name,
            new_x=pd.Series(new_x_plot), N_poly=N_poly, CI=CI, N_bootstrap=N_bootstrap)

    if std_error is True:
        new_x_plot=np.linspace(np.min(x), np.max(x), 100)
        Ne_corr2=calculate_Ne_corr_std_err_values(pickle_str=pkl_name,
        new_x=pd.Series(new_x_plot), CI=CI)

    

    # Now lets plot the prediction interval
    fig, (ax1) = plt.subplots(1, 1, figsize=(10,5))
    ax1.errorbar(x, y, xerr=0, yerr=y_err,
             fmt='o', ecolor='k', elinewidth=0.8, mfc='grey', ms=5, mec='k',capsize=3)
    
    ax1.plot(new_x_plot, Ne_corr2['preferred_values'], '-k', label='best fit')
    ax1.plot(new_x_plot, Ne_corr2['lower_values'], ':k', label='lower vals')
    ax1.plot(new_x_plot, Ne_corr2['upper_values'], ':k', label='upper vals')
    ax1.set_xlabel('sec after midnight')
    ax1.set_ylabel('Ne Corr factor')
    ax1.legend()
    ax1.plot(x, y, '+r', label='Ne lines')
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
        fig.savefig('Ne_line_correction.png')
        

    
from scipy.stats import t

def calculate_Ne_corr_std_err_values(*, pickle_str, new_x, CI=0.67):
    # Load the model and the data from the pickle file
    with open(pickle_str, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    N_poly = model.order - 1

    Pf = data['model']
    x = data['x']
    y = data['y']

    # Convert new_x to plain numpy array
    new_x_array = np.asarray(new_x)

    # Calculate the residuals
    residuals = y - Pf(x)

    # Calculate the standard deviation of the residuals
    residual_std = np.std(residuals)

    # Calculate the standard errors for the new x values
    mean_x = np.mean(x)
    n = len(x)
    standard_errors = residual_std * np.sqrt(1 + 1/n + (new_x_array - mean_x)**2 / np.sum((x - mean_x)**2))

    # Calculate the degrees of freedom
    df_dof = len(x) - (N_poly + 1)

    # Calculate the t value for the given confidence level
    t_value = t.ppf((1 + CI) / 2, df_dof)

    # Calculate the prediction intervals
    preferred_values = Pf(new_x_array)
    lower_values = preferred_values - t_value * standard_errors
    upper_values = preferred_values + t_value * standard_errors

    df_out = pd.DataFrame(data={
        'time': new_x_array,
        'preferred_values': preferred_values,
        'lower_values': lower_values,
        'upper_values': upper_values
    })

    return df_out






def calculate_Ne_corr_bootstrap_values(*, pickle_str, new_x, N_poly=3, CI=0.67, N_bootstrap=500):
    # Load the model and the data from the pickle file
    with open(pickle_str, 'rb') as f:
        data = pickle.load(f)

    Pf = data['model']
    x = data['x']
    y = data['y']

    # Define the function

    x_values=new_x
    N_poly = N_poly  # degree of the polynomial
    n_bootstrap = N_bootstrap  # number of bootstrap samples
    confidence = CI  # confidence level

    preferred_values = []
    lower_values = []
    upper_values = []

    for new_x in x_values:
        # Perform bootstrapping
        bootstrap_predictions = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_indices = np.random.choice(len(x), size=len(x), replace=True)
            bootstrap_x = x[bootstrap_indices]
            bootstrap_y = y[bootstrap_indices]

            # Perform polynomial regression on the bootstrap sample
            bootstrap_coefficients = np.polyfit(bootstrap_x, bootstrap_y, N_poly)
            bootstrap_Pf = np.poly1d(bootstrap_coefficients)

            # Calculate predicted value for the new x
            bootstrap_prediction = bootstrap_Pf(new_x)
            bootstrap_predictions.append(bootstrap_prediction)

        # Calculate prediction interval
        lower_quantile = (1 - confidence) / 2
        upper_quantile = 1 - lower_quantile
        lower_index = int(lower_quantile * n_bootstrap)
        upper_index = int(upper_quantile * n_bootstrap)
        bootstrap_predictions_sorted = np.sort(bootstrap_predictions)
        lower_value = bootstrap_predictions_sorted[lower_index]
        upper_value = bootstrap_predictions_sorted[upper_index]

        preferred_values.append(Pf(new_x))
        lower_values.append(lower_value)
        upper_values.append(upper_value)
        
    df=pd.DataFrame(data={'time': new_x,
        'preferred_values': preferred_values,
        'lower_values': lower_values,
        'upper_values': upper_values}
    )

    return df
    
    



