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

def plot_Ne_lines(*, path=None, filename, filetype='Witec_ASCII', n_peaks=6,
peak1_cent=1118, peak2_cent=1447, exclude_range_1=None,
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


    Ne_in=get_data(path=path, filename=filename, filetype=filetype)




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

    ax0.set_ylabel('Amplitude')
    ax0.set_xlabel('Wavenumber')


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

    df_1117=df_sort_Ne.loc[df['pos'].between(peak1_cent-5, peak1_cent+5)]
    df_1447=df_sort_Ne.loc[df['pos'].between(peak2_cent-5, peak2_cent+5)]

    df_1117_trim=df_1117[0:1]
    df_1447_trim=df_1447[0:1]



    ax1.plot(Ne_in[:, 0], Ne_in[:, 1], '-c', label='input')
    ax1.plot(x, y, '-r', label='filtered')
    ax1.plot(df['pos'], df['height'], '*c', label='all peaks')

    if len(df_1117_trim)==0:
        print('No peak found within +-5 wavenumbers of peak position 1, have returned user-entered peak')
        #ax1.plot(peak2_cent, df_1447_trim['height'], '*k')
        pos_1117=str(peak1_cent)
        ax1.annotate(pos_1117, xy=(peak1_cent,
        100-10), xycoords="data", fontsize=10, rotation=90)
        nearest_1117=peak1_cent

    else:
        ax1.plot(df_1117_trim['pos'], df_1117_trim['height'], '*k', label='selected peak')
        pos_1117=str(np.round(df_1117_trim['pos'].iloc[0], 1))
        ax1.annotate(pos_1117, xy=(df_1117_trim['pos']-5,
        df_1117_trim['height']-10), xycoords="data", fontsize=10, rotation=90)
        nearest_1117=float(df_1117_trim['pos'])

    if len(df_1447_trim)==0:
        print('No peak found within +-5 wavenumbers of peak position 2, have returned user-entered peak')
        pos_1447=str(peak2_cent)
        nearest_1447=peak2_cent
        ax2.annotate(pos_1447, xy=(nearest_1447,
        200), xycoords="data", fontsize=10, rotation=90)

    else:
        ax2.plot(df_1447_trim['pos'], df_1447_trim['height'], '*k', label='selected peak')
        ax2.legend()
        pos_1447=str(np.round(df_1447_trim['pos'].iloc[0], 1))
        nearest_1447=float(df_1447_trim['pos'])

        ax2.annotate(pos_1447, xy=(df_1447_trim['pos']-5,
        df_1447_trim['height']-100), xycoords="data", fontsize=10, rotation=90)

    ax1.set_xlim([peak1_cent-15, peak1_cent+15])

    ax1.set_xlim([peak1_cent-10, peak1_cent+10])

    ax2.plot(x, y, '-r')
    ax2.plot(df_sort_Ne_trim['pos'], df_sort_Ne_trim['height'], '*k')
    #print(df_1117)


    ax2.set_xlim([peak2_cent-15, peak2_cent+15])


    print('selected Peak 1 Pos')
    print(nearest_1117)
    print('selected Peak 2 Pos')
    print(nearest_1447)
    return Ne, df_sort_Ne_trim, nearest_1117, nearest_1447



## Ne baselines
def remove_Ne_baseline_1117(Ne, N_poly_1117_baseline=1, Ne_center_1=1117.1,
lower_bck=[-50, -25], upper_bck1=[8, 15], upper_bck2=[30, 50] ):
    """ This function uses a defined range of values to fit a baseline of Nth degree polynomial to the baseline
    around the 1117 peak

    Parameters
    -----------

    Ne: np.array
        np.array of x and y coordinates from the spectra

    N_poly_1117_baseline: int
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

    lower_0baseline_1117=Ne_center_1+lower_bck[0]
    upper_0baseline_1117=Ne_center_1+lower_bck[1]
    lower_1baseline_1117=Ne_center_1+upper_bck1[0]
    upper_1baseline_1117=Ne_center_1+upper_bck1[1]
    lower_2baseline_1117=Ne_center_1+upper_bck2[0]
    upper_2baseline_1117=Ne_center_1+upper_bck2[1]

    # Trim for entire range
    Ne_short=Ne[ (Ne[:,0]>lower_0baseline_1117) & (Ne[:,0]<upper_2baseline_1117) ]

    # Get actual baseline
    Baseline_with_outl=Ne_short[
    ((Ne_short[:, 0]<upper_0baseline_1117) &(Ne_short[:, 0]>lower_0baseline_1117))
    |
    ((Ne_short[:, 0]<upper_1baseline_1117) &(Ne_short[:, 0]>lower_1baseline_1117))
    |
    ((Ne_short[:, 0]<upper_2baseline_1117) &(Ne_short[:, 0]>lower_2baseline_1117))]

    # Calculates the median for the baseline and the standard deviation
    Median_Baseline=np.median(Baseline_with_outl[:, 1])
    Std_Baseline=np.std(Baseline_with_outl[:, 1])

    # Removes any points in the baseline outside of 2 sigma (helps remove cosmic rays etc).
    Baseline=Baseline_with_outl[(Baseline_with_outl[:, 1]<Median_Baseline+2*Std_Baseline)
                                &
                                (Baseline_with_outl[:, 1]>Median_Baseline-2*Std_Baseline)
                               ]

    # Fits a polynomial to the baseline of degree
    Pf_baseline = np.poly1d(np.polyfit(Baseline[:, 0], Baseline[:, 1], N_poly_1117_baseline))
    Py_base =Pf_baseline(Ne_short[:, 0])
    Baseline_ysub=Pf_baseline(Baseline[:, 0])
    Baseline_x=Baseline[:, 0]
    y_corr= Ne_short[:, 1]-  Py_base
    x=Ne_short[:, 0]


    return y_corr, Py_base, x,  Ne_short, Py_base, Baseline_ysub, Baseline_x

def remove_Ne_baseline_1447(Ne, N_poly_1447_baseline=1, Ne_center_2=1447.1,
lower_bck=[-44.2, -22], upper_bck1=[15, 50], upper_bck2=[50, 51]):

    """ This function uses a defined range of values to fit a baseline of Nth degree polynomial to the baseline
    around the 1447 peak

    Parameters
    -----------

    Ne: np.array
        np.array of x and y coordinates from the spectra

    N_poly_1117_baseline: int
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



    lower_0baseline_1447=Ne_center_2+lower_bck[0]
    upper_0baseline_1447=Ne_center_2+lower_bck[1]
    lower_1baseline_1447=Ne_center_2+upper_bck1[0]
    upper_1baseline_1447=Ne_center_2+upper_bck1[1]
    lower_2baseline_1447=Ne_center_2+upper_bck2[0]
    upper_2baseline_1447=Ne_center_2+upper_bck2[1]

    # Trim for entire range
    Ne_short=Ne[ (Ne[:,0]>lower_0baseline_1447) & (Ne[:,0]<upper_2baseline_1447) ]

    # Get actual baseline
    Baseline_with_outl=Ne_short[
    ((Ne_short[:, 0]<upper_0baseline_1447) &(Ne_short[:, 0]>lower_0baseline_1447))
    |
    ((Ne_short[:, 0]<upper_1baseline_1447) &(Ne_short[:, 0]>lower_1baseline_1447))
    |
    ((Ne_short[:, 0]<upper_2baseline_1447) &(Ne_short[:, 0]>lower_2baseline_1447))]

    # Calculates the median for the baseline and the standard deviation
    Median_Baseline=np.median(Baseline_with_outl[:, 1])
    Std_Baseline=np.std(Baseline_with_outl[:, 1])

    # Removes any points in the baseline outside of 2 sigma (helps remove cosmic rays etc).
    Baseline=Baseline_with_outl[(Baseline_with_outl[:, 1]<Median_Baseline+2*Std_Baseline)
                                &
                                (Baseline_with_outl[:, 1]>Median_Baseline-2*Std_Baseline)
                               ]

    # Fits a polynomial to the baseline of degree
    Pf_baseline = np.poly1d(np.polyfit(Baseline[:, 0], Baseline[:, 1], N_poly_1447_baseline))
    Py_base =Pf_baseline(Ne_short[:, 0])
    Baseline_ysub=Pf_baseline(Baseline[:, 0])
    Baseline_x=Baseline[:, 0]
    y_corr= Ne_short[:, 1]-  Py_base
    x=Ne_short[:, 0]


    return y_corr, Py_base, x,  Ne_short, Py_base, Baseline_ysub, Baseline_x





def fit_1117(x, y_corr, x_span=[-10, 8], Ne_center=1117.1, amplitude=98, sigma=0.28,
LH_offset_mini=[1.5, 3], peaks_1117=2, print_report=False) :
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

    peaks_1117: integer
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
    lower_1117=Ne_center+x_span[0]
    upper_1117=Ne_center+x_span[1]

    # This segments into the x and y variable, and variables to plot, which are a bit bigger.
    Ne_1117_reg_x=x[(x>lower_1117)&(x<upper_1117)]
    Ne_1117_reg_y=y_corr[(x>lower_1117)&(x<upper_1117)]
    Ne_1117_reg_x_plot=x[(x>(lower_1117-3))&(x<(upper_1117+3))]
    Ne_1117_reg_y_plot=y_corr[(x>(lower_1117-3))&(x<(upper_1117+3))]

    if peaks_1117>1:

        # Setting up lmfit
        model0 = VoigtModel(prefix='p0_')#+ ConstantModel(prefix='c0')
        pars0 = model0.make_params(p0_center=Ne_center, p0_amplitude=amplitude)
        init0 = model0.eval(pars0, x=xdat)
        result0 = model0.fit(ydat, pars0, x=xdat)
        Center_p0=result0.best_values.get('p0_center')
        print('first iteration, peak Center='+str(np.round(Center_p0, 4)))

        Center_p0_error=result0.params.get('p0_center')
        Amp_p0=result0.params.get('p0_amplitude')
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

        print('Trying to place second peak between '+str(np.round(minp2, 2))+'and'+ str(np.round(maxp2, 2)))
        pars[prefix + 'center'].set(Center_p0, min=minp2,
        max=maxp2)

        pars['p2_'+ 'fwhm'].set(fwhm_p0/2, min=0.001, max=fwhm_p0*5)


        pars[prefix + 'amplitude'].set(Amp_p0/5, min=0, max=Amp_p0/2)
        pars[prefix + 'sigma'].set(0.2, min=0)

        model_combo=model1+peak
        pars1.update(pars)


    if peaks_1117==1:
        print('fitting a single peak, if you want the shoulder, do peaks_1117=2')

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


    if peaks_1117==1:
        Center_1117=Center_p1
        if Error_bars is False:
            print('Error bars not determined by function')
            error_1117=np.nan
        else:
            error_1117 = float(str(Center_p1_error).split()[4].replace(",", ""))


    if peaks_1117>1:
        Center_p2=result.best_values.get('p2_center')
        Center_p2_error=result.params.get('p2_center')


        if Error_bars is False:
            print('Error bars not determined by function')
            Center_p1_errorval=np.nan
            if peaks_1117>1:
                Center_p2_errorval=np.nan
        else:
            Center_p1_errorval=float(str(Center_p1_error).split()[4].replace(",", ""))
            if peaks_1117>1:
                Center_p2_errorval=float(str(Center_p2_error).split()[4].replace(",", ""))

        # Check if nonsense, e.g. if center 2 miles away, just use center 0
        if Center_p2 is not None:
            if Center_p2>Center_p0 or Center_p2<1112:
                Center_1117=Center_p0
                error_1117=Center_p0_errorval
                print('No  meaningful second peak found')

            elif Center_p1 is None and Center_p2 is None:
                print('No peaks found')
            elif Center_p1 is None and Center_p2>0:
                Center_1117=Center_p2
                error_1117=Center_p2_errorval
            elif Center_p2 is None and Center_p1>0:
                Center_1117=Center_p1
                error_1117=Center_p1_errorval
            elif Center_p1>Center_p2:
                Center_1117=Center_p1
                error_1117=Center_p1_errorval
            elif Center_p1<Center_p2:
                Center_1117=Center_p2
                error_1117=Center_p2_errorval


    # Evaluate the peak at 100 values for pretty plotting
    xx_1117=np.linspace(lower_1117, upper_1117, 2000)

    result_1117=result.eval(x=xx_1117)
    comps=result.eval_components(x=xx_1117)


    result_1117_origx=result.eval(x=Ne_1117_reg_x)

    if print_report is True:
         #print(result.fit_report(min_correl=0.5))
         print('trying')


    return Center_1117, Ne_1117_reg_x_plot, Ne_1117_reg_y_plot, Ne_1117_reg_x, Ne_1117_reg_y, xx_1117, result_1117, error_1117, result_1117_origx, comps


def fit_1447(x, y_corr, x_span=[-5, 5], Ne_center=1447.5, amplitude=1000, sigma=0.28, print_report=False) :
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
    lower_1447=Ne_center+x_span[0]
    upper_1447=Ne_center+x_span[1]

    # This segments into the x and y variable, and variables to plot, which are a bit bigger.
    Ne_1447_reg_x=x[(x>lower_1447)&(x<upper_1447)]
    Ne_1447_reg_y=y_corr[(x>lower_1447)&(x<upper_1447)]
    Ne_1447_reg_x_plot=x[(x>(lower_1447-3))&(x<(upper_1447+3))]
    Ne_1447_reg_y_plot=y_corr[(x>(lower_1447-3))&(x<(upper_1447+3))]


    model = VoigtModel()#+ ConstantModel()


    # create parameters with initial values
    params = model.make_params(center=Ne_center, amplitude=amplitude, sigma=sigma)

    # Place bounds on center allowed
    params['center'].min = Ne_center+x_span[0]
    params['center'].max = Ne_center+x_span[1]



    result = model.fit(Ne_1447_reg_y.flatten(), params, x=Ne_1447_reg_x.flatten())

    # Get center value
    Center_1447=result.best_values.get('center')
    Center_1447_error=result.params.get('center')
    # Have to strip away the rest of the string, as center + error

    Center_1447_errorval=float(str(Center_1447_error).split()[4].replace(",", ""))
    error_1447=Center_1447_errorval

    # Evaluate the peak at 100 values for pretty plotting
    xx_1447=np.linspace(lower_1447, upper_1447, 2000)

    result_1447=result.eval(x=xx_1447)
    result_1447_origx=result.eval(x=Ne_1447_reg_x)

    if print_report is True:
         print(result.fit_report(min_correl=0.5))

    return Center_1447, Ne_1447_reg_x_plot, Ne_1447_reg_y_plot, Ne_1447_reg_x, Ne_1447_reg_y, xx_1447, result_1447, error_1447, result_1447_origx



def fit_Ne_lines(*, Ne=None, filename=None, path=None, prefix=True,
Ne_center_1=1117.1, N_poly_1_baseline=1, x_span_1=[-10, 8],
LH_offset_mini=[1.5, 3], peaks_1=2,
lower_bck_pk1=[-50, -25], upper_bck1_pk1=[8, 15], upper_bck2_pk1=[30, 50],
Ne_center_2=1147, N_poly_2_baseline=1, x_span_2=[-5, 5],
x_span_pk1_override=None, x_span_pk2_override=None,
lower_bck_pk2=[-44.2, -22], upper_bck1_pk2=[15, 50], upper_bck2_pk2=[50, 51],
amplitude=100, plot_figure=True, print_report=False, loop=False, x_range_baseline=100, y_range_baseline=1000, x_range_peak=None, DeltaNe_ideal=330.477634):




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

    x_span_pk2_override, x_span_pk1_override: list
        determins how many point to either side of peak you want to fit, if different from background positions

    DeltaNe_ideal: float
        Theoretical distance between the two peaks you have selected. Default is 330.477634 for
        the 1117 and 1447 diad for the Cornell Raman. You can calculate this using the calculate_Ne_line_positions


    Things for Diad 1 (~1117):

        N_poly_1_baseline: int
            Degree of polynomial used to fit the background

        Ne_center_1: float
            Center position for Ne line being fitted

        lower_bck_1, upper_bck1, upper_bck1: 3 lists of length 2:
            Positions used for background relative to peak.[-50, -20] takes a
            background -50 and -20 from the peak center

        x_span_1: list length 2. Default [-10, 8]
            Span either side of peak center used for fitting,
            e.g. by default, fits to 10 wavenumbers below peak, 8 above.


        peaks_1: int
            How many peaks to fit to the 1117 diad, if 2, tries to put a shoulder peak

        LH_offset_mini: list
            If peaks>1, puts second peak within this range left of the main peak




    Things for Diad 2 (~1447):
        N_poly_2_baseline: int
            Degree of polynomial used to fit the background

        Ne_center_2: float
            Center position for Ne line being fitted

        lower_bck_2, upper_bck2, upper_bck2: 3 lists of length 2:
            Positions used for background relative to peak.[-50, -20] takes a
            background -50 and -20 from the peak center

        x_span_2: list length 2. Default [-10, 8]
            Span either side of peak center used for fitting,
            e.g. by default, fits to 10 wavenumbers below peak, 8 above.
    """


    #Remove the baselines
    y_corr_1117, Py_base_1117, x_1117, Ne_short_1117, Py_base_1117, Baseline_ysub_1117, Baseline_x_1117=remove_Ne_baseline_1117(Ne, Ne_center_1=Ne_center_1,
    N_poly_1117_baseline=N_poly_1_baseline,
    lower_bck=lower_bck_pk1, upper_bck1=upper_bck1_pk1, upper_bck2=upper_bck2_pk1)

    y_corr_1447, Py_base_1447, x_1447, Ne_short_1447, Py_base_1447, Baseline_ysub_1447, Baseline_x_1447=remove_Ne_baseline_1447(Ne, Ne_center_2=Ne_center_2, N_poly_1447_baseline=N_poly_2_baseline,
    lower_bck=lower_bck_pk2, upper_bck1=upper_bck1_pk2, upper_bck2=upper_bck2_pk2)


    # Have the option to override the xspan here from default. Else, trims
    if x_span_pk1_override is None:

        x_span_pk1=[lower_bck_pk1[1], upper_bck1_pk1[0]]
        x_span_pk1_dist=abs(lower_bck_pk1[1]-upper_bck1_pk1[0])
    else:
        x_span_pk1=x_span_pk1_override
        x_span_pk1_dist=abs(x_span_pk1[1]-x_span_pk1[0])

    if x_span_pk2_override is None:
        x_span_pk2=[lower_bck_pk2[1], upper_bck1_pk2[0]]
        x_span_pk2_dist=abs(lower_bck_pk2[1]-upper_bck1_pk2[0])
    else:
        x_span_pk2=x_span_pk2_override
        x_span_pk2_dist=abs(x_span_pk2[1]-x_span_pk2[0])

    # Fit the 1117 peak
    cent_1117, Ne_1117_reg_x_plot, Ne_1117_reg_y_plot, Ne_1117_reg_x, Ne_1117_reg_y, xx_1117, result_1117, error_1117, result_1117_origx, comps = fit_1117(x_1117, y_corr_1117, x_span=x_span_pk1, Ne_center=Ne_center_1, LH_offset_mini=LH_offset_mini, peaks_1117=peaks_1, amplitude=amplitude, print_report=print_report)


    # Fit the 1447 peak
    cent_1447, Ne_1447_reg_x_plot, Ne_1447_reg_y_plot, Ne_1447_reg_x, Ne_1447_reg_y, xx_1447, result_1447, error_1447, result_1447_origx = fit_1447(x_1447, y_corr_1447, x_span=x_span_pk2,  Ne_center=Ne_center_2, amplitude=amplitude, print_report=print_report)


    # Calculate difference between peak centers, and Delta Ne
    DeltaNe=cent_1447-cent_1117

    Ne_Corr=DeltaNe_ideal/DeltaNe

    # Calculate maximum splitting (+1 sigma)
    DeltaNe_max=(cent_1447+error_1447)-(cent_1117-error_1117)
    DeltaNe_min=(cent_1447-error_1447)-(cent_1117+error_1117)
    Ne_Corr_max=DeltaNe_ideal/DeltaNe_min
    Ne_Corr_min=DeltaNe_ideal/DeltaNe_max

    # Calculating least square residual
    residual_1117=np.sum(((Ne_1117_reg_y-result_1117_origx)**2)**0.5)/(len(Ne_1117_reg_y))
    residual_1447=np.sum(((Ne_1447_reg_y-result_1447_origx)**2)**0.5)/(len(Ne_1447_reg_y))

    if plot_figure is True:
        # Make a summary figure of the backgrounds and fits
        fig, ((ax3, ax2), (ax5, ax4), (ax1, ax0)) = plt.subplots(3,2, figsize = (12,15)) # adjust dimensions of figure here
        fig.suptitle(filename, fontsize=16)

        ax0.plot(Ne_short_1447[:,0], Py_base_1447, '-k')
        ax0.plot(Ne_short_1447[:,0], Ne_short_1447[:,1], '-r')



        ax0.plot(Baseline_x_1447, Baseline_ysub_1447, '.b', ms=5, label='Selected background')

        ax0.set_title('Peak2: 1447 background fitting')
        ax0.set_xlabel('Wavenumber')
        ax0.set_ylabel('Intensity')
        mean_baseline=np.mean(Py_base_1447)
        std_baseline=np.std(Py_base_1447)
        ax0.set_ylim([min(Ne_short_1447[:,1])-10, min(Ne_short_1447[:,1])+y_range_baseline])
        ax0.set_xlim([min(Ne_short_1447[:,0])-20, max(Ne_short_1447[:,0])+y_range_baseline])
        #ax0.set_ylim([mean_baseline-50, mean_baseline+50])

        ax1.plot(Ne_short_1117[:,0], Py_base_1117, '-k')
        ax1.plot(Ne_short_1117[:,0], Ne_short_1117[:,1], '-r')
        ax1.plot(Baseline_x_1117, Baseline_ysub_1117, '.b', ms=6, label='Selected background')

        std_baseline=np.std(Py_base_1117)
        ax1.set_ylim([min(Ne_short_1117[:,1])-10, min(Ne_short_1117[:,1])+y_range_baseline])

        ax1.set_title('Peak1: 1117 background fitting')
        ax1.set_xlabel('Wavenumber')
        ax1.set_ylabel('Intensity')

        #Testing
        ax0.legend()
        ax1.legend()
        ax0.plot(Ne[:,0], Ne[:,1], '-', color='grey', zorder=0)
        ax1.plot(Ne[:,0], Ne[:,1], '-', color='grey', zorder=0)
        ax0.set_xlim([min(Ne_short_1447[:,0])-x_range_baseline, max(Ne_short_1447[:,0])+x_range_baseline])
        ax1.set_xlim([min(Ne_short_1117[:,0])-x_range_baseline, max(Ne_short_1117[:,0])+x_range_baseline])

        ax2.plot(Ne_1447_reg_x_plot, Ne_1447_reg_y_plot, 'xb', label='all data')
        ax2.plot(Ne_1447_reg_x, Ne_1447_reg_y, 'ok', label='data fitted')
        ax2.plot(xx_1447, result_1447, 'r-', label='interpolated fit')
        ax2.set_title('1447 peak fitting')
        ax2.set_xlabel('Wavenumber')
        ax2.set_ylabel('Intensity')
        if x_range_peak is None:
            ax2.set_xlim([cent_1447-x_span_pk2_dist/2, cent_1447+x_span_pk2_dist/2])
        else:
            ax2.set_xlim([cent_1447-x_range_peak, cent_1447+x_range_peak])


        ax3.plot(Ne_1117_reg_x_plot, Ne_1117_reg_y_plot, 'xb', label='all data')
        ax3.plot(Ne_1117_reg_x, Ne_1117_reg_y, 'ok', label='data fitted')

        ax3.set_title('1117 peak fitting')
        ax3.set_xlabel('Wavenumber')
        ax3.set_ylabel('Intensity')
        ax3.plot(xx_1117, comps.get('p1_'), '-r', label='p1')
        if peaks_1>1:
            ax3.plot(xx_1117, comps.get('p2_'), '-c', label='p2')
        ax3.plot(xx_1117, result_1117, 'g-', label='best fit')
        ax3.legend()
        if x_range_peak is None:
            ax3.set_xlim([cent_1117-x_span_pk1_dist/2, cent_1117+x_span_pk1_dist/2])
        else:
            ax3.set_xlim([cent_1117-x_range_peak, cent_1117+x_range_peak ])

        # Residuals for charlotte
        ax4.plot(Ne_1447_reg_x, Ne_1447_reg_y-result_1447_origx, '-r', label='residual')
        ax5.plot(Ne_1117_reg_x, Ne_1117_reg_y-result_1117_origx, '-r',  label='residual')
        ax4.plot(Ne_1447_reg_x, Ne_1447_reg_y-result_1447_origx, 'ok', mfc='r', label='residual')
        ax5.plot(Ne_1117_reg_x, Ne_1117_reg_y-result_1117_origx, 'ok', mfc='r', label='residual')
        ax4.set_ylabel('Residual (Intensity units)')
        ax4.set_xlabel('Wavenumber')
        ax5.set_ylabel('Residual (Intensity units)')
        ax5.set_xlabel('Wavenumber')

        fig.tight_layout()

        # Save figure
        path3=path+'/'+'Peak_fit_images'
        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ 'Peak_fit_images', exist_ok=False)

        figure_str=path+'/'+ 'Peak_fit_images'+ '/'+ filename+str('_Ne_Line_Fit')+str('.png')

        fig.savefig(figure_str, dpi=200)

    if prefix is True:

        filename=filename.split(' ')[1:][0]
    df=pd.DataFrame(data={'File_Name': filename,
                          '1447_peak_cent':cent_1447,
                          'error_1447': error_1447,
                          '1117_peak_cent':cent_1117,
                          'error_1117': error_1117,
                         'deltaNe': DeltaNe,
                         'Ne_Corr': Ne_Corr,
                         'Ne_Corr_min':Ne_Corr_min,
                         'Ne_Corr_max': Ne_Corr_max,
                         'residual_1447':residual_1447,
                         'residual_1117': residual_1117}, index=[0])

    df.to_clipboard(excel=True, header=False, index=False)

    if loop is False:

        return df, Ne_1117_reg_x_plot, Ne_1117_reg_y_plot
    if loop is True:
        return df


def each_Ne_Line(path=None,filename=None,  filetype=None,  nearest_1117=None,nearest_1447=None, amplitude=None, prefix=None, LH_offset_mini=None, plot_figure=False):
    """
    This function does all the steps for each Ne line, e.g. background fitting,
    """

    Ne=get_data(path=path, filename=filename, filetype=filetype)

    # How many degrees in polynomials
    N_poly_1447_baseline=1
    N_poly_1117_baseline=2
    #If you have weak Ne lines and no secondary peak, set to 1
    peaks_1117=2
    # If weak, set to 10
    amplitude=amplitude

    df, Ne_1117_reg_x_plot, Ne_1117_reg_y_plot=fit_Ne_lines(Ne=Ne,
    filename=filename, path=path,
    Ne_center_1=nearest_1117, Ne_center_2=nearest_1447,
    peaks_1117=peaks_1117,
    x_span_1447=20, x_span_1117_up=8, x_span_1117_low=10,
    LH_offset_mini=LH_offset_mini,  prefix=prefix, amplitude=amplitude, plot_figure=plot_figure)



    return df




