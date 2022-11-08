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
from DiadFit.importing_data_files import *
from typing import Tuple, Optional
from dataclasses import dataclass
import matplotlib.patches as patches
from tqdm import tqdm




encode="ISO-8859-1"

## Plotting Ne lines, returns peak position
def find_closest(df, line1_shift):
    dist = (df['Raman_shift (cm-1)'] - line1_shift).abs()
    return df.loc[dist.idxmin()]



def calculate_Ne_splitting(wavelength=532.05, line1_shift=1117, line2_shift=1447, cut_off_intensity=2000):
    """
    Calculates ideal splitting in air between lines closest to user-specified line shift
    """

    df_Ne=calculate_Ne_line_positions(wavelength=wavelength, cut_off_intensity=cut_off_intensity)

    closest1=find_closest(df_Ne, line1_shift).loc['Raman_shift (cm-1)']
    closest2=find_closest(df_Ne, line2_shift).loc['Raman_shift (cm-1)']

    diff=abs(closest1-closest2)

    df=pd.DataFrame(data={'Ne_Split': diff,
    'Line_1': closest1,
    'Line_2': closest2,
    'Entered Pos Line 1': line1_shift,
    'Entered Pos Line 2': line2_shift}, index=[0])


    return df

def calculate_Ne_line_positions(wavelength=532.05, cut_off_intensity=2000):
    """
    Calculates Ne line positions using the theoretical lines from NIST for the user inputted wavelenth
    """

    Ne_emission_line_air=np.array([
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
582.890630



    ])

    Intensity=np.array([1500.00,
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
    750.00])



    Raman_shift=(10**7/wavelength)-(10**7/Ne_emission_line_air)

    df_Ne=pd.DataFrame(data={'Raman_shift (cm-1)': Raman_shift,
                            'Intensity': Intensity,
                            'Ne emission line in air': Ne_emission_line_air})

    df_Ne_r=df_Ne.loc[df_Ne['Intensity']>cut_off_intensity]
    return df_Ne_r

def plot_Ne_lines(*, path=None, filename=None, filetype=None, n_peaks=6,
peak1_cent=1118, peak2_cent=1447, exclude_range_1=None, Ne_array=None,
exclude_range_2=None, height=10, threshold=0.6, distance=1, prominence=10, width=1,):

    """
    Loads Ne line, uses scipy find peaks to identify peaks, overlays these,
    and returns peak positions to feed into fitting algorithms

    Parameters
    -----------

    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read

    filetype: str
        Identifies type of file
        Witec_ASCII: Datafile from WITEC with metadata for first few lines
        headless_txt: Txt file with no headers, just data with wavenumber in 1st col, int 2nd
        HORIBA_txt: Datafile from newer HORIBA machines with metadata in first rows
        Renishaw_txt: Datafile from renishaw with column headings.

    n_peaks: int
        Number of peaks to return values for

    peak1_cent: int or float, default 1118
        Position to look for 1st peak in, finds peaks within +/- 5 of this

    peak2_cent: int or float, default 1447
        Position to look for 2nd peak in, finds peaks within +/- 5 of this


    height, threshold, distance, prominence, width: int
         parameters for scipy find peaks

    exclude_range_1: None or list
        users can enter a range (e.g [1100, 1112]) to exclude a part of their spectrum,
        perhaps to remove cosmic rays

    exclude_range_2: None or list
        users can enter a range (e.g [1100, 1112]) to exclude a part of their spectrum,
        perhaps to remove cosmic rays



    """

    if filename is not None and path is not None and filetype is not None:
        Ne_in=get_data(path=path, filename=filename, filetype=filetype)
    if Ne_array is not None:
        Ne_in=Ne_array



    # Exclude range
    if exclude_range_1 is None and exclude_range_2 is None:
        Ne=Ne_in
    if exclude_range_1 is not None:
        Ne=Ne_in[ (Ne_in[:, 0]<exclude_range_1[0]) | (Ne_in[:, 0]>exclude_range_1[1]) ]

    if exclude_range_2 is not None:
        Ne=Ne_in[ (Ne_in[:, 0]<exclude_range_2[0]) | (Ne_in[:, 0]>exclude_range_2[1]) ]

    # Find peaks
    y=Ne[:, 1]
    x=Ne[:, 0]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    ax0.plot(x, y, '-r')
    miny=np.min(y)
    maxy=np.max(y)
    ax0.plot([1117, 1117], [miny, maxy], ':k')
    ax0.plot([1220, 1220], [miny, maxy], ':k')
    ax0.plot([1310, 1310], [miny, maxy], ':k')
    ax0.plot([1398, 1398], [miny, maxy], ':k')
    ax0.plot([1447, 1447], [miny, maxy], ':k')
    ax0.plot([1566, 1566], [miny, maxy], ':k')

    ax0.set_ylabel('Amplitude (counts)')
    ax0.set_xlabel('Wavenumber (cm$^{-1}$)')


    peaks = find_peaks(y,height = height, threshold = threshold, distance = distance, prominence=prominence, width=width)
    # print('Found peaks at wavenumber=')
    # print(x[peaks[0]])

    n_peaks=6
    height = peaks[1]['peak_heights'] #list of the heights of the peaks
    peak_pos = x[peaks[0]] #list of the peaks positions
    df=pd.DataFrame(data={'pos': peak_pos,
                        'height': height})


    # Find bigest peaks,
    df_sort_Ne=df.sort_values('height', axis=0, ascending=False)
    df_sort_Ne_trim=df_sort_Ne[0:n_peaks]

    print('Biggest 6 peaks:')
    display(df_sort_Ne_trim)

    df_pk1=df_sort_Ne.loc[df['pos'].between(peak1_cent-5, peak1_cent+5)]
    df_pk2=df_sort_Ne.loc[df['pos'].between(peak2_cent-5, peak2_cent+5)]

    df_pk1_trim=df_pk1[0:1]
    df_pk2_trim=df_pk2[0:1]



    ax1.plot(Ne_in[:, 0], Ne_in[:, 1], '-c', label='input')
    ax1.plot(x, y, '-r', label='filtered')
    ax1.plot(df_sort_Ne_trim['pos'], df_sort_Ne_trim['height'], '*c', label='all peaks')

    if len(df_pk1_trim)==0:
        print('No peak found within +-5 wavenumbers of peak position 1, have returned user-entered peak')
        #ax1.plot(peak2_cent, df_pk2_trim['height'], '*k')
        pos_pk1=str(peak1_cent)
        ax1.annotate(pos_pk1, xy=(peak1_cent,
        100-10), xycoords="data", fontsize=10, rotation=90)
        nearest_pk1=peak1_cent

    else:
        ax1.plot(df_pk1_trim['pos'], df_pk1_trim['height'], '*k', mfc='yellow', ms=8, label='selected peak')
        pos_pk1=str(np.round(df_pk1_trim['pos'].iloc[0], 1))
        ax1.annotate(pos_pk1, xy=(df_pk1_trim['pos']-5,
        df_pk1_trim['height']-10), xycoords="data", fontsize=10, rotation=90)
        nearest_pk1=float(df_pk1_trim['pos'])

    if len(df_pk2_trim)==0:
        print('No peak found within +-5 wavenumbers of peak position 2, have returned user-entered peak')
        pos_pk2=str(peak2_cent)
        nearest_pk2=peak2_cent
        ax2.annotate(pos_pk2, xy=(nearest_pk2,
        200), xycoords="data", fontsize=10, rotation=90)

    else:
        ax2.plot(df_pk2_trim['pos'], df_pk2_trim['height'], '*k', mfc='yellow', ms=8,label='selected peak')
        ax2.legend(bbox_to_anchor=(0.3,1.1))

        pos_pk2=str(np.round(df_pk2_trim['pos'].iloc[0], 1))
        nearest_pk2=float(df_pk2_trim['pos'])

        ax2.annotate(pos_pk2, xy=(df_pk2_trim['pos']-5,
        df_pk2_trim['height']/2), xycoords="data", fontsize=10, rotation=90)

    ax1.set_xlim([peak1_cent-15, peak1_cent+15])

    ax1.set_xlim([peak1_cent-10, peak1_cent+10])

    ax2.plot(x, y, '-r')
    ax2.plot(df_sort_Ne_trim['pos'], df_sort_Ne_trim['height'], '*k', mfc='yellow', ms=8)
    #print(df_pk1)


    ax2.set_xlim([peak2_cent-15, peak2_cent+15])

    ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax2.set_xlabel('Wavenumber (cm$^{-1}$)')

    print('selected Peak 1 Pos')
    print(nearest_pk1)
    print('selected Peak 2 Pos')
    print(nearest_pk2)
    return Ne, df_sort_Ne_trim, nearest_pk1, nearest_pk2



## Ne baselines
def remove_Ne_baseline_pk1(Ne, N_poly_pk1_baseline=None, Ne_center_1=None,
lower_bck=None, upper_bck1=None, upper_bck2=None, sigma_baseline=None):
    """ This function uses a defined range of values to fit a baseline of Nth degree polynomial to the baseline
    around the 1117 peak

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

def remove_Ne_baseline_pk2(Ne, N_poly_pk2_baseline=None, Ne_center_2=None, sigma_baseline=None,
lower_bck=None, upper_bck1=None, upper_bck2=None):

    """ This function uses a defined range of values to fit a baseline of Nth degree polynomial to the baseline
    around the 1447 peak

    Parameters
    -----------

    Ne: np.array
        np.array of x and y coordinates from the spectra

    N_poly_pk1_baseline: int
        Degree of polynomial used to fit the background

    Ne_center_1: float
        Center position for Ne line being fitted

    lower_bck: list (length 2) Default [-44.2, -22]
        position used for lower background relative to peak, so =[-50, -20] takes a
        background -50 and -20 from the peak center

    upper_bck1: list (length 2). Default [15, 50]
        position used for 1st upper background relative to peak, so =[8, 15] takes a
        background +8 and +15 from the peak center

    upper_bck2: list (length 2) Default [50, 51]
        position used for 2nd upper background relative to peak, so =[30, 50] takes a
        background +30 and +50 from the peak center
    """



    lower_0baseline_pk2=Ne_center_2+lower_bck[0]
    upper_0baseline_pk2=Ne_center_2+lower_bck[1]
    lower_1baseline_pk2=Ne_center_2+upper_bck1[0]
    upper_1baseline_pk2=Ne_center_2+upper_bck1[1]
    lower_2baseline_pk2=Ne_center_2+upper_bck2[0]
    upper_2baseline_pk2=Ne_center_2+upper_bck2[1]

    # Trim for entire range
    Ne_short=Ne[ (Ne[:,0]>lower_0baseline_pk2) & (Ne[:,0]<upper_2baseline_pk2) ]

    # Get actual baseline
    Baseline_with_outl=Ne_short[
    ((Ne_short[:, 0]<upper_0baseline_pk2) &(Ne_short[:, 0]>lower_0baseline_pk2))
    |
    ((Ne_short[:, 0]<upper_1baseline_pk2) &(Ne_short[:, 0]>lower_1baseline_pk2))
    |
    ((Ne_short[:, 0]<upper_2baseline_pk2) &(Ne_short[:, 0]>lower_2baseline_pk2))]

    # Calculates the median for the baseline and the standard deviation
    Median_Baseline=np.median(Baseline_with_outl[:, 1])
    Std_Baseline=np.std(Baseline_with_outl[:, 1])

    # Removes any points in the baseline outside of 2 sigma (helps remove cosmic rays etc).
    Baseline=Baseline_with_outl[(Baseline_with_outl[:, 1]<Median_Baseline+sigma_baseline*Std_Baseline)
                                &
                                (Baseline_with_outl[:, 1]>Median_Baseline-sigma_baseline*Std_Baseline)
                               ]

    # Fits a polynomial to the baseline of degree
    Pf_baseline = np.poly1d(np.polyfit(Baseline[:, 0], Baseline[:, 1], N_poly_pk2_baseline))
    Py_base =Pf_baseline(Ne_short[:, 0])
    Baseline_ysub=Pf_baseline(Baseline[:, 0])
    Baseline_x=Baseline[:, 0]
    Baseline_y=Baseline[:, 1]
    y_corr= Ne_short[:, 1]-  Py_base
    x=Ne_short[:, 0]


    return y_corr, Py_base, x,  Ne_short, Py_base, Baseline_y, Baseline_x





def fit_pk1(x, y_corr, x_span=[-10, 8], Ne_center=1117.1, amplitude=98, sigma=0.28,
LH_offset_mini=[1.5, 3], peaks_pk1=2, block_print=True) :
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

    # Flatten x and y if needed
    xdat=x.flatten()
    ydat=y_corr.flatten()

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
        model0 = VoigtModel(prefix='p0_')#+ ConstantModel(prefix='c0')
        pars0 = model0.make_params(p0_center=Ne_center, p0_amplitude=amplitude)
        init0 = model0.eval(pars0, x=xdat)
        result0 = model0.fit(ydat, pars0, x=xdat)
        Center_p0=result0.best_values.get('p0_center')
        if block_print is False:
            print('first iteration, peak Center='+str(np.round(Center_p0, 4)))

        Center_p0_error=result0.params.get('p0_center')
        Amp_p0=result0.params.get('p0_amplitude')
        if block_print is False:
            print('first iteration, peak Amplitude='+str(np.round(Amp_p0, 4)))
        fwhm_p0=result0.params.get('p0_fwhm')
        Center_p0_errorval=float(str(Center_p0_error).split()[4].replace(",", ""))

        #Ne_center=Ne_center
        #rough_peak_positions=Ne_center-2

        model1 = VoigtModel(prefix='p1_')#+ ConstantModel(prefix='c1')
        pars1 = model1.make_params()
        pars1['p1_'+ 'amplitude'].set(Amp_p0, min=0.8*Amp_p0, max=1.2*Amp_p0)
        pars1['p1_'+ 'center'].set(Center_p0, min=Center_p0-0.2, max=Center_p0+0.2)
        pars1['p1_'+ 'fwhm'].set(fwhm_p0, min=fwhm_p0-1, max=fwhm_p0+1)


        # Second wee peak
        prefix='p2_'
        peak = VoigtModel(prefix='p2_')#+ ConstantModel(prefix='c2')
        pars = peak.make_params()
        minp2=Center_p0-LH_offset_mini[1]
        maxp2=Center_p0-LH_offset_mini[0]
        if block_print is False:
            print('Trying to place second peak between '+str(np.round(minp2, 2))+'and'+ str(np.round(maxp2, 2)))
        pars[prefix + 'center'].set(Center_p0, min=minp2,
        max=maxp2)

        pars['p2_'+ 'fwhm'].set(fwhm_p0/2, min=0.001, max=fwhm_p0*5)


        pars[prefix + 'amplitude'].set(Amp_p0/5, min=0, max=Amp_p0/2)
        pars[prefix + 'sigma'].set(0.2, min=0)

        model_combo=model1+peak
        pars1.update(pars)


    if peaks_pk1==1:
        if block_print is False:
            print('fitting a single peak, if you want the shoulder, do peaks_pk1=2')

        model_combo = VoigtModel(prefix='p1_')#+ ConstantModel()


        # create parameters with initial values
        pars1 = model_combo.make_params(amplitude=amplitude, sigma=sigma)
        pars1['p1_' + 'center'].set(Ne_center, min=Ne_center-2,
        max=Ne_center+2)






    init = model_combo.eval(pars1, x=xdat)
    result = model_combo.fit(ydat, pars1, x=xdat)
    # Need to check errors output
    Error_bars=result.errorbars


    # Get center value
    Center_p1=result.best_values.get('p1_center')
    Center_p1_error=result.params.get('p1_center')


    if peaks_pk1==1:
        Center_pk1=Center_p1
        if Error_bars is False:
            if block_print is False:
                print('Error bars not determined by function')
            error_pk1=np.nan
        else:
            error_pk1 = float(str(Center_p1_error).split()[4].replace(",", ""))


    if peaks_pk1>1:
        Center_p2=result.best_values.get('p2_center')
        Center_p2_error=result.params.get('p2_center')


        if Error_bars is False:
            if block_print is False:
                print('Error bars not determined by function')
            Center_p1_errorval=np.nan
            if peaks_pk1>1:
                Center_p2_errorval=np.nan
        else:
            Center_p1_errorval=float(str(Center_p1_error).split()[4].replace(",", ""))
            if peaks_pk1>1:
                Center_p2_errorval=float(str(Center_p2_error).split()[4].replace(",", ""))

        # Check if nonsense, e.g. if center 2 miles away, just use center 0
        if Center_p2 is not None:
            if Center_p2>Center_p0 or Center_p2<1112:
                Center_pk1=Center_p0
                error_pk1=Center_p0_errorval
                if block_print is False:
                    print('No  meaningful second peak found')

            elif Center_p1 is None and Center_p2 is None:
                if block_print is False:
                    print('No peaks found')
            elif Center_p1 is None and Center_p2>0:
                Center_pk1=Center_p2
                error_pk1=Center_p2_errorval
            elif Center_p2 is None and Center_p1>0:
                Center_pk1=Center_p1
                error_pk1=Center_p1_errorval
            elif Center_p1>Center_p2:
                Center_pk1=Center_p1
                error_pk1=Center_p1_errorval
            elif Center_p1<Center_p2:
                Center_pk1=Center_p2
                error_pk1=Center_p2_errorval

    Area_pk1=result.best_values.get('p1_amplitude')
    sigma_pk1=result.best_values.get('p1_sigma')
    gamma_pk1=result.best_values.get('p1_gamma')


    # Evaluate the peak at 100 values for pretty plotting
    xx_pk1=np.linspace(lower_pk1, upper_pk1, 2000)

    result_pk1=result.eval(x=xx_pk1)
    comps=result.eval_components(x=xx_pk1)


    result_pk1_origx=result.eval(x=Ne_pk1_reg_x)



    return Center_pk1, Area_pk1, sigma_pk1, gamma_pk1, Ne_pk1_reg_x_plot, Ne_pk1_reg_y_plot, Ne_pk1_reg_x, Ne_pk1_reg_y, xx_pk1, result_pk1, error_pk1, result_pk1_origx, comps


def fit_pk2(x, y_corr, x_span=[-5, 5], Ne_center=1447.5, amplitude=1000, sigma=0.28, print_report=False) :
    """ This function fits the 1447 Ne line as a single Voigt

    Parameters
    -----------

    x: np.array
        x coordinate (wavenumber)

    y: np.array
        Background corrected intensiy

    x_span: list length 2. Default [-5, 5]
        Span either side of peak center used for fitting,
        e.g. by default, fits to 5 wavenumbers below peak, 5 above.

    Ne_center: float (default=1447.5)
        Center position for Ne line being fitted

    amplitude: integer (default = 1000)
        peak amplitude

    sigma: float (default =0.28)
        sigma of the voigt peak


    print_report: bool
        if True, prints fit report.


    """

    # This defines the range you want to fit (e.g. how big the tails are)
    lower_pk2=Ne_center+x_span[0]
    upper_pk2=Ne_center+x_span[1]

    # This segments into the x and y variable, and variables to plot, which are a bit bigger.
    Ne_pk2_reg_x=x[(x>lower_pk2)&(x<upper_pk2)]
    Ne_pk2_reg_y=y_corr[(x>lower_pk2)&(x<upper_pk2)]
    Ne_pk2_reg_x_plot=x[(x>(lower_pk2-3))&(x<(upper_pk2+3))]
    Ne_pk2_reg_y_plot=y_corr[(x>(lower_pk2-3))&(x<(upper_pk2+3))]


    model = VoigtModel()#+ ConstantModel()


    # create parameters with initial values
    params = model.make_params(center=Ne_center, amplitude=amplitude, sigma=sigma)

    # Place bounds on center allowed
    params['center'].min = Ne_center+x_span[0]
    params['center'].max = Ne_center+x_span[1]



    result = model.fit(Ne_pk2_reg_y.flatten(), params, x=Ne_pk2_reg_x.flatten())

    # Get center value
    Center_pk2=result.best_values.get('center')
    Center_pk2_error=result.params.get('center')

    #print(result.best_values)

    Area_pk2=result.best_values.get('amplitude')
    sigma_pk2=result.best_values.get('sigma')
    gamma_pk2=result.best_values.get('gamma')
    # Have to strip away the rest of the string, as center + error

    Center_pk2_errorval=float(str(Center_pk2_error).split()[4].replace(",", ""))
    error_pk2=Center_pk2_errorval

    # Evaluate the peak at 100 values for pretty plotting
    xx_pk2=np.linspace(lower_pk2, upper_pk2, 2000)

    result_pk2=result.eval(x=xx_pk2)
    result_pk2_origx=result.eval(x=Ne_pk2_reg_x)

    if print_report is True:
         print(result.fit_report(min_correl=0.5))

    return Center_pk2,Area_pk2, sigma_pk2, gamma_pk2,  Ne_pk2_reg_x_plot, Ne_pk2_reg_y_plot, Ne_pk2_reg_x, Ne_pk2_reg_y, xx_pk2, result_pk2, error_pk2, result_pk2_origx

## Setting default Ne fitting parameters
@dataclass
class Ne_peak_config:
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

    # Things for plotting the baseline
    x_range_baseline: float=20 #  How many units outside your selected background it shows on the baseline plot
    y_range_baseline: float= 200    # Where the y axis is cut off above the minimum baseline measurement

    # Things for fitting the primary peak
    pk1_amplitude: float = 100
    pk2_amplitude: float = 100
    # Things for plotting the primary peak
    x_range_peak: float=15 # How many units to each side are shown when plotting the peak fits

    # Things for plotting the residual
    x_range_residual: float=7 # Shows how many x units to left and right is shown on residual plot

    # Things for fitting a secondary peak on 1117
    LH_offset_mini: Tuple[float, float] = (1.5, 3)

    # Optional, by default, fits to the points inside the baseline. Can also specify as values to make a smaller peak fit.
    x_span_pk1: Optional [Tuple[float, float]] = None # Tuple[float, float] = (-10, 8)
    x_span_pk2: Optional [Tuple[float, float]] = None # Tuple[float, float] = (-5, 5)



def fit_Ne_lines(*,  config: Ne_peak_config=Ne_peak_config(),
Ne_center_1=1117.1, Ne_center_2=1147, peaks_1=2,
    Ne=None, filename=None, path=None, prefix=True,
    plot_figure=True, loop=True,
    DeltaNe_ideal=330.477634, save_clipboard=True,
    close_figure=False):






    """ This function reads in a user file, fits the Ne lines, and if required, saves an image
    into a new sub folder

    Parameters
    -----------

    Ne: np.array
        x coordinate (wavenumber) and y coordinate (intensity)

    filename and path: str
        used to save filename in datatable, and to make a new folder.

    filetype: str
        Identifies type of file
        Witec_ASCII: Datafile from WITEC with metadata for first few lines
        headless_txt: Txt file with no headers, just data with wavenumber in 1st col, int 2nd
        HORIBA_txt: Datafile from newer HORIBA machines with metadata in first rows
        Renishaw_txt: Datafile from renishaw with column headings.

    amplitude: int or float
        first guess of peak amplitude

    plot_figure: bool
        if True, saves figure of fit in a new folder

    Loop: bool
        If True, only returns df.

    x_range_baseline: flt, int
        How much x range outside selected baseline the baseline selection plot shows.

    y_range_baseline: flt, int
        How much above the baseline position is shown on the y axis.

    x_range_peak: flt, int, or None
        How much to either side of the peak to show on the final peak fitting plot


    DeltaNe_ideal: float
        Theoretical distance between the two peaks you have selected. Default is 330.477634 for
        the 1117 and 1447 diad for the Cornell Raman. You can calculate this using the calculate_Ne_line_positions


    Things for Diad 1 (~1117):

        N_poly_pk1_baseline: int
            Degree of polynomial used to fit the background

        Ne_center_1: float
            Center position for Ne line being fitted

        lower_bck_1, upper_bck1, upper_bck1: 3 lists of length 2:
            Positions used for background relative to peak.[-50, -20] takes a
            background -50 and -20 from the peak center

        x_span_pk1: list length 2. Default [-10, 8]
            Span either side of peak center used for fitting,
            e.g. by default, fits to 10 wavenumbers below peak, 8 above.


        peaks_1: int
            How many peaks to fit to the 1117 diad, if 2, tries to put a shoulder peak

        LH_offset_mini: list
            If peaks>1, puts second peak within this range left of the main peak




    Things for Diad 2 (~1447):
        N_poly_pk2_baseline: int
            Degree of polynomial used to fit the background

        Ne_center_2: float
            Center position for Ne line being fitted

        lower_bck_2, upper_bck2, upper_bck2: 3 lists of length 2:
            Positions used for background relative to peak.[-50, -20] takes a
            background -50 and -20 from the peak center

        x_span_pk2: list length 2. Default [-10, 8]
            Span either side of peak center used for fitting,
            e.g. by default, fits to 10 wavenumbers below peak, 8 above.
    """


    #Remove the baselines
    y_corr_pk1, Py_base_pk1, x_pk1, Ne_short_pk1, Py_base_pk1, Baseline_ysub_pk1, Baseline_x_pk1=remove_Ne_baseline_pk1(Ne,
    N_poly_pk1_baseline=config.N_poly_pk1_baseline, Ne_center_1=Ne_center_1, sigma_baseline=config.sigma_baseline,
    lower_bck=config.lower_bck_pk1, upper_bck1=config.upper_bck1_pk1, upper_bck2=config.upper_bck2_pk1)

    y_corr_pk2, Py_base_pk2, x_pk2, Ne_short_pk2, Py_base_pk2, Baseline_ysub_pk2, Baseline_x_pk2=remove_Ne_baseline_pk2(Ne, Ne_center_2=Ne_center_2, N_poly_pk2_baseline=config.N_poly_pk2_baseline, sigma_baseline=config.sigma_baseline,
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

    # Fit the 1117 peak
    cent_pk1, Area_pk1, sigma_pk1, gamma_pk1, Ne_pk1_reg_x_plot, Ne_pk1_reg_y_plot, Ne_pk1_reg_x, Ne_pk1_reg_y, xx_pk1, result_pk1, error_pk1, result_pk1_origx, comps = fit_pk1(x_pk1, y_corr_pk1, x_span=x_span_pk1, Ne_center=Ne_center_1, LH_offset_mini=config.LH_offset_mini, peaks_pk1=peaks_1, amplitude=config.pk1_amplitude)


    # Fit the 1447 peak
    cent_pk2,Area_pk2, sigma_pk2, gamma_pk2, Ne_pk2_reg_x_plot, Ne_pk2_reg_y_plot, Ne_pk2_reg_x, Ne_pk2_reg_y, xx_pk2, result_pk2, error_pk2, result_pk2_origx = fit_pk2(x_pk2, y_corr_pk2, x_span=x_span_pk2,  Ne_center=Ne_center_2, amplitude=config.pk2_amplitude)


    # Calculate difference between peak centers, and Delta Ne
    DeltaNe=cent_pk2-cent_pk1

    Ne_Corr=DeltaNe_ideal/DeltaNe

    # Calculate maximum splitting (+1 sigma)
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
        ymax_ax1=min(Ne_short_pk1[:,1])+config.y_range_baseline
        ymin_ax0=min(Ne_short_pk2[:,1])-10
        ymax_ax0=min(Ne_short_pk2[:,1])+config.y_range_baseline
        ax1.set_ylim([ymin_ax1, ymax_ax1])
        ax0.set_ylim([ymin_ax1, ymax_ax1])

        # Setting x limits of axis

        ax1_xmin=min(Ne_short_pk1[:,0])-config.x_range_baseline
        ax1_xmax=max(Ne_short_pk1[:,0])+config.x_range_baseline
        ax0_xmin=min(Ne_short_pk2[:,0])-config.x_range_baseline
        ax0_xmax=max(Ne_short_pk2[:,0])+config.x_range_baseline
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
        ax0.plot([Ne_center_2, Ne_center_2], [ymin_ax1, ymax_ax1], ':k', label='Peak')

        ax1.set_title('Peak1: 1117 background fitting')
        ax1.set_xlabel('Wavenumber')
        ax1.set_ylabel('Intensity')
        ax0.set_title('Peak2: 1447 background fitting')
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

        filename=filename.split(' ')[1:][0]
    df=pd.DataFrame(data={'filename': filename,
                          'pk2_peak_cent':cent_pk2,
                          'pk2_amplitude': Area_pk2,
                          'pk2_sigma': sigma_pk2,
                          'pk2_gamma': gamma_pk2,
                          'error_pk2': error_pk2,
                          'pk1_peak_cent':cent_pk1,
                          'pk1_amplitude': Area_pk1,
                          'pk1_sigma': sigma_pk1,
                          'pk1_gamma': gamma_pk1,
                          'error_pk1': error_pk1,
                         'deltaNe': DeltaNe,
                         'Ne_Corr': Ne_Corr,
                         'Ne_Corr_min':Ne_Corr_min,
                         'Ne_Corr_max': Ne_Corr_max,
                         'residual_pk2':residual_pk2,
                         'residual_pk1': residual_pk1,
                         'residual_pk1+pk2':residual_pk1+residual_pk2,
                         }, index=[0])
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
    fig, ((ax5, ax6), (ax1, ax2), (ax3, ax4), ) = plt.subplots(3, 2, figsize=(10, 12))
    ax1.plot(x, df['Ne_Corr'], marker,  mec='k', mfc='grey')
    ax1.set_ylabel('Ne Correction factor')
    ax1.set_xlabel(x_label)

    ax5.plot(x, df['pk1_peak_cent'], marker,  mec='k', mfc='b')
    ax6.plot(x, df['pk2_peak_cent'], marker,  mec='k', mfc='r')
    ax5.set_xlabel(x_label)
    ax6.set_xlabel(x_label)
    ax5.set_ylabel('Peak 1 center')
    ax6.set_ylabel('Peak 2 center')


    ax2.plot( df['residual_pk2']+df['residual_pk1'], df['Ne_Corr'], marker,  mec='k', mfc='r')
    ax2.set_xlabel('Sum of pk1 and pk2 residual')
    ax2.set_ylabel('Ne Correction factor')

    ax3.plot(df['Ne_Corr'], df['pk2_peak_cent'],marker,  mec='k', mfc='r')
    ax3.set_xlabel('Ne Correction factor')
    ax3.set_ylabel('Peak 2 center')

    ax4.plot(df['Ne_Corr'], df['pk1_peak_cent'], marker,  mec='k', mfc='b')
    ax4.set_xlabel('Ne Correction factor')
    ax4.set_ylabel('Peak 1 center')

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
def loop_Ne_lines(*, files, path, filetype,
                  config, peaks_1,  Ne_center_1,
                  Ne_center_2, DeltaNe_ideal, prefix=None,
                  plot_figure=True, save_clipboard=True, single_acq=True):

    df = pd.DataFrame([])
    if single_acq is True:
        for i in tqdm(range(0, np.shape(files)[1]-2)):
            Ne=np.column_stack((files[:, 0], files[:, i+1]))
            filename=str(i)

            data=fit_Ne_lines(
            config=config, peaks_1=peaks_1,
            Ne=Ne, filename=filename, path=path, prefix=prefix,
            Ne_center_1=Ne_center_1, Ne_center_2=Ne_center_2,
            DeltaNe_ideal=DeltaNe_ideal, plot_figure=plot_figure,
            save_clipboard=save_clipboard)
            df = pd.concat([df, data], axis=0)


    if single_acq is False:
        for i in tqdm(range(0, len(files))):
            filename=files[i]
            Ne=get_data(path=path, filename=filename, filetype=filetype)
            data=fit_Ne_lines(
            config=config, peaks_1=peaks_1,
            Ne=Ne, filename=filename, path=path, prefix=prefix,
            Ne_center_1=Ne_center_1, Ne_center_2=Ne_center_2,
            DeltaNe_ideal=DeltaNe_ideal, plot_figure=plot_figure,
            save_clipboard=save_clipboard)
            df = pd.concat([df, data], axis=0)
        #print('working on ' + str(files[i]))







    df2=df.reset_index(drop=True)

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
            Pf = interp1d(df['sec since midnight'], df['Ne_Corr'], kind=spline_fit)

    Py=Pf(Px)

    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 4))
    ax1.plot(df['sec since midnight'], df['Ne_Corr'], 'ok')
    ax1.plot(Px, Py, '-r')
    ax1.set_xlabel('Seconds since midnight')
    ax1.set_ylabel('Ne Correction Factor')

    ax1.ticklabel_format(useOffset=False)


    return Pf


def filter_Ne_Line_neighbours(Corr_factor, number_av=6, offset=0.00005):
    Corr_factor_Filt=np.empty(len(Corr_factor), dtype=float)
    median_loop=np.empty(len(Corr_factor), dtype=float)

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



    return ds




