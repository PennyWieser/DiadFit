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
import warnings as w

encode="ISO-8859-1"

def plot_diad(*,path=None, filename=None, filetype='Witec_ASCII'):


    Spectra_df=get_data(path=path, filename=filename, filetype=filetype)

    Spectra=np.array(Spectra_df)


    fig, (ax1) = plt.subplots(1, 1, figsize=(6,4))

    miny=np.min(Spectra[:, 1])
    maxy=np.max(Spectra[:, 1])
    ax1.plot([1090, 1090], [miny, maxy], ':k', label='Magnesite')
    ax1.plot([1131, 1131], [miny, maxy], '-',  alpha=0.5,color='grey', label='Anhydrite/Mg-Sulfate')
    #ax1.plot([1136, 1136], [miny, maxy], '-', color='grey', label='Mg-Sulfate')
    ax1.plot([1151, 1151], [miny, maxy], ':c', label='SO2')
    ax1.plot([1286, 1286], [miny, maxy], '-g',  alpha=0.5,label='Diad1')
    ax1.plot([1389, 1389], [miny, maxy], '-m', alpha=0.5, label='Diad2')
    ax1.legend()
    ax1.plot(Spectra[:, 0], Spectra[:, 1], '-r')
    ax1.set_xlabel('Wavenumber (cm-1)')
    ax1.set_ylabel('Intensity')

@dataclass
class diad_id_config:
    # Exclude a range, e.g. cosmic rays
    exclude_range1: Optional [Tuple[float, float]] = None
    exclude_range2: Optional [Tuple[float, float]] = None
    # Approximate diad position
    approx_diad2_pos: Tuple[float, float]=(1379, 1395)
    approx_diad1_pos: Tuple[float, float]=(1275, 1295)

    # Thresholds for Scipy find peaks
    height: float = 400
    distance: float = 5
    threshold: float = 0.5
    width: float=0.5
    prominence: float=10

    # to plot or not to plot
    plot_figure: bool = True


def identify_diad_peaks(*, config: diad_id_config=diad_id_config(), path=None, filename, filetype='Witec_ASCII',
    n_peaks_diad1=None, n_peaks_diad2=None, block_print=True, plot_figure=True ):
    """
    This function loads your file, and excludes up to 2 user-defined ranges.
    It then uses scipy find peaks to get a first guess of peak positions to feed into later functions


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
        head_csv: CSV with a header, wavenumber in x, intensity in y
        HORIBA_txt: Datafile from newer HORIBA machines with metadata in first rows
        Renishaw_txt: Datafile from renishaw with column headings.

    n_peaks_diad1:
        How many peaks you want the code to try to identify around the LH diad
        1: Just the diad
        2: Diad and Hot Band

    n_peaks_diad2:
        How many peaks you want the code to try to identify around the RH diad
        1: Just the diad
        2: Diad and Hot Band
        3: Diad, Hot band and C13

    approx_diad1_pos, approx_diad2_pos:
        list, e.g., [1290, 1300], code looks for peaks in this range. needs tweaking on different instuments.

    exclude_range1: None or list length 2
        Excludes a region, e.g. a cosmic ray

    exclude_range2: None or list length 2
        Excludes a region, e.g. a cosmic ray

    height, threshold, distance, prominence, width: int or float
        parameters that can be tweaked from scipy find peaks

    plot_figure: bool
        if True, plots figure, if False, doesn't.

    Returns
    -----------



    """

    Diad_df=get_data(path=path, filename=filename, filetype=filetype)

    Diad=np.array(Diad_df)
    if config.exclude_range1 is None and config.exclude_range2 is None:
        Discard_str=False
    else:
        Discard_str=True
        if config.exclude_range1 is not None and config.exclude_range2 is None:
            Diad_old=Diad.copy()
            Diad=Diad[(Diad[:, 0]<config.exclude_range1[0])|(Diad[:, 0]>config.exclude_range1[1])]
            Discard=Diad_old[(Diad_old[:, 0]>=config.exclude_range1[0]) & (Diad_old[:, 0]<=config.exclude_range1[1])]

        # NEED TO FIX
        if config.exclude_range2 is not None and config.exclude_range1 is None:
            Diad_old=Diad.copy()
            Diad=Diad[(Diad[:, 0]<config.exclude_range2[0])|(Diad[:, 0]>config.exclude_range2[1])]

            Discard=Diad_old[(Diad_old[:, 0]>=config.exclude_range2[0]) & (Diad_old[:, 0]<=config.exclude_range2[1])]

        if config.exclude_range1 is not None and config.exclude_range2 is not None:
            Diad_old=Diad.copy()
            Diad=Diad[
            ((Diad[:, 0]<config.exclude_range1[0])|(Diad[:, 0]>config.exclude_range1[1]))
            &
            ((Diad[:, 0]<config.exclude_range2[0])|(Diad[:, 0]>config.exclude_range2[1]))
            ]

            Discard=Diad_old[
            ((Diad_old[:, 0]>=config.exclude_range1[0]) & (Diad_old[:, 0]<=config.exclude_range1[1]))
            |
            ((Diad_old[:, 0]>=config.exclude_range2[0]) & (Diad_old[:, 0]<=config.exclude_range2[1]))
            ]


    y=Diad[:, 1]
    x=Diad[:, 0]
    peaks = find_peaks(y,height = config.height, threshold = config.threshold,
    distance = config.distance, prominence=config.prominence, width=config.width)

    height = peaks[1]['peak_heights'] #list of the heights of the peaks
    peak_pos = x[peaks[0]] #list of the peaks positions
    df=pd.DataFrame(data={'pos': peak_pos,
                        'height': height})



    df_pks_diad1=df[(df['pos']>1220) & (df['pos']<1320) ]
    # Find peaks within the 2nd diad window
    df_pks_diad2=df[(df['pos']>1350) & (df['pos']<1430) ]



    df_sort_diad1=df_pks_diad1.sort_values('height', axis=0, ascending=False)
    df_sort_diad1_trim=df_sort_diad1[0:n_peaks_diad1]

    df_sort_diad2=df_pks_diad2.sort_values('height', axis=0, ascending=False)
    df_sort_diad2_trim=df_sort_diad2[0:n_peaks_diad2]

    if any(df_sort_diad2_trim['pos'].between(config.approx_diad2_pos[0], config.approx_diad2_pos[1])):
        diad_2_peaks=tuple(df_sort_diad2_trim['pos'].values)
    else:
        if n_peaks_diad2==1:
            if block_print is False:
                print('WARNING: Couldnt find diad2, ive guesed a peak position of ' + str(np.round(np.average(config.approx_diad2_pos), 2)) +  'to move forwards')
            diad_2_peaks=np.array([np.average(config.approx_diad2_pos)])
        if n_peaks_diad2==2:
            if block_print is False:
                print('WARNING: Couldnt find diad2, ive guesed a peak position of 1389.1 and 1410')
            diad_2_peaks=np.array([np.average(config.approx_diad2_pos)])
        if n_peaks_diad2==3:

            raise TypeError('WARNING: Couldnt find diad2, and you specified 3 peaks, try adjusting the Scipy peak parameters')

    if any(df_sort_diad1_trim['pos'].between(config.approx_diad1_pos[0], config.approx_diad1_pos[1])):
        diad_1_peaks=tuple(df_sort_diad1_trim['pos'].values)
        if n_peaks_diad2==2:
            if len(diad_1_peaks)==1:
                print('Warning - couldnt find hotband, guessing its positoin as 20 below the peak')
                diad_1_peaks=(diad_1_peaks[0], diad_1_peaks[0]-20)
    else:
        if block_print is False:
            print('WARNING: Couldnt find diad2, ive guesed a peak position of ' + str(np.round(np.average(config.approx_diad1_pos), 2)) +  'to move forwards')
        diad_1_peaks=np.array([np.average(config.approx_diad1_pos)])


    if block_print is False:
        print('Initial estimates: Diad1+HB=' +str(np.round(diad_1_peaks, 1)) + ', Diad2+HB=' + str(np.round(diad_2_peaks, 1)))


    if plot_figure is True:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12,4))

        ax0.plot(Diad[:, 0], Diad[:, 1], '-r')
        ax0.plot(df['pos'], df['height'], '*k')
        ax1.plot(df['pos'], df['height'], '*k', label='All Scipy Peaks')
        ax2.plot(df['pos'], df['height'], '*k')

        if Discard_str is not False:
            ax0.plot(Discard[:, 0], Discard[:, 1], '.c', label='Discarded')
            ax1.plot(Discard[:, 0], Discard[:, 1], '.c', label='Discarded')
            ax2.plot(Discard[:, 0], Discard[:, 1], '.c', label='Discarded')

        ax0.plot([np.average(config.approx_diad1_pos), np.average(config.approx_diad1_pos)],
        [min(Diad[:, 1]), max(Diad[:, 1])], ':k', label='Approx. D1 pos')
        ax0.plot([np.average(config.approx_diad2_pos), np.average(config.approx_diad2_pos)],
        [min(Diad[:, 1]), max(Diad[:, 1])], ':k', label='approx D2 pos')
        ax1.plot([np.average(config.approx_diad1_pos), np.average(config.approx_diad1_pos)],
        [min(Diad[:, 1]), max(Diad[:, 1])], ':k', label='approx D1 pos')
        ax1.plot([np.average(config.approx_diad2_pos), np.average(config.approx_diad2_pos)],
        [min(Diad[:, 1]), max(Diad[:, 1])], ':k', label='approx D2 pos')
        ax2.plot([np.average(config.approx_diad1_pos), np.average(config.approx_diad1_pos)],
        [min(Diad[:, 1]), max(Diad[:, 1])], ':k', label='approx D1 pos')
        ax2.plot([np.average(config.approx_diad2_pos), np.average(config.approx_diad2_pos)],
        [min(Diad[:, 1]), max(Diad[:, 1])], ':k', label='approx expt. D2 pos')

        ax0.legend()
        ax1.set_title('Diad1')
        ax1.plot(Diad[:, 0],Diad[:, 1], '-r')
        ax1.set_xlim([1200, 1350])
        ax2.set_title('Diad2')
        ax2.plot(Diad[:, 0],Diad[:, 1], '-r')
        ax2.set_xlim([1350, 1450])
        #ax0.set_ylim[np.min(Diad[:, 1]), np.max(Diad[:, 1]) ])
        fig.tight_layout()
        ax2.plot(df_sort_diad2_trim['pos'], df_sort_diad2_trim['height'], '*k', mfc='yellow', ms=10)
        ax1.plot(df_sort_diad1_trim['pos'], df_sort_diad1_trim['height'], '*k',  mfc='yellow', ms=10, label='Selected Pks')
        ax1.legend()
        ax0.set_xlabel('Wavenumber')
        ax0.set_ylabel('Intensity')
        ax1.set_xlabel('Wavenumber')
        ax1.set_ylabel('Intensity')
        ax2.set_xlabel('Wavenumber')
        ax2.set_ylabel('Intensity')

        for i in range(0, len(df_sort_diad1_trim)):
            ax1.annotate(str(np.round(df_sort_diad1_trim['pos'].iloc[i], 1)), xy=(df_sort_diad1_trim['pos'].iloc[i]-10,
            df_sort_diad1_trim['height'].iloc[i]-1/7*(df_sort_diad1_trim['height'].iloc[i]-700)), xycoords="data", fontsize=10, rotation=90)

        for i in range(0, len(df_sort_diad2_trim)):
            ax2.annotate(str(np.round(df_sort_diad2_trim['pos'].iloc[i], 1)), xy=(df_sort_diad2_trim['pos'].iloc[i]-10,
            df_sort_diad2_trim['height'].iloc[i]-1/7*(df_sort_diad2_trim['height'].iloc[i]-700)), xycoords="data", fontsize=10, rotation=90)



    return diad_1_peaks, diad_2_peaks



def remove_diad_baseline(*, path=None, filename=None, Diad_files=None, filetype='Witec_ASCII',
            exclude_range1=None, exclude_range2=None,N_poly=1, x_range_baseline=10,
            lower_bck=[1200, 1250], upper_bck=[1320, 1330], sigma=4,
            plot_figure=True):
    """ This function uses a defined range of values to fit a baseline of Nth degree polynomial to the baseline between user-specified limits

    Parameters
    -----------
    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read

    OR

    Diad_files: str
        Filename if in same folder as script.

    filetype: str
        Identifies type of file
        Witec_ASCII: Datafile from WITEC with metadata for first few lines
        headless_txt: Txt file with no headers, just data with wavenumber in 1st col, int 2nd
        HORIBA_txt: Datafile from newer HORIBA machines with metadata in first rows
        Renishaw_txt: Datafile from renishaw with column headings.


    exclude_range1: None or list length 2
        Excludes a region, e.g. a cosmic ray

    exclude_range2: None or list length 2
        Excludes a region, e.g. a cosmic ray

    sigma: int
        Number of sigma to filter out points from the mean of the baseline


    N_poly: int
        Degree of polynomial to fit to the backgroun

    lower_bck: list len 2
        wavenumbers of LH baseline region

    upper_bck: list len 2
        wavenumbers of RH baseline region


    plot_figure: bool
        if True, plots figure, if False, doesn't.

    Returns
    -----------
    y_corr, Py_base, x,  Diad_short, Py_base, Baseline_ysub, Baseline_x, Baseline

        y_corr: Background subtracted y values trimmed in baseline range
        x: initial x values trimmed in baseline range
        Diad_short: Initial data (x and y) trimmed in baseline range
        Py_base: Fitted baseline for trimmed x coordinates
        Baseline_ysub: Baseline fitted for x coordinates in baseline
        Baseline_x: x co-ordinates in baseline.
        Baseline: Filtered to remove points outside sigma*std dev of baseline



    """

    Diad_df=get_data(path=path, filename=filename, filetype=filetype)



    Diad=np.array(Diad_df)


    if exclude_range1 is not None and exclude_range2 is None:
        Diad_old=Diad.copy()
        Diad=Diad[(Diad[:, 0]<exclude_range1[0])|(Diad[:, 0]>exclude_range1[1])]
        Discard=Diad_old[(Diad_old[:, 0]>=exclude_range1[0]) & (Diad_old[:, 0]<=exclude_range1[1])]


    if exclude_range2 is not None and exclude_range1 is None:
        Diad_old=Diad.copy()
        Diad=Diad[(Diad[:, 0]<exclude_range2[0])|(Diad[:, 0]>exclude_range2[1])]

        Discard=Diad_old[(Diad_old[:, 0]>=exclude_range2[0]) & (Diad_old[:, 0]<=exclude_range2[1])]

    if exclude_range1 is not None and exclude_range2 is not None:
        Diad_old=Diad.copy()
        Diad=Diad[
        ((Diad[:, 0]<exclude_range1[0])|(Diad[:, 0]>exclude_range1[1]))
        &
        ((Diad[:, 0]<exclude_range2[0])|(Diad[:, 0]>exclude_range2[1]))
        ]

        Discard=Diad_old[
        ((Diad_old[:, 0]>=exclude_range1[0]) & (Diad_old[:, 0]<=exclude_range1[1]))
        |
        ((Diad_old[:, 0]>=exclude_range2[0]) & (Diad_old[:, 0]<=exclude_range2[1]))
        ]


    lower_0baseline=lower_bck[0]
    upper_0baseline=lower_bck[1]
    lower_1baseline=upper_bck[0]
    upper_1baseline=upper_bck[1]
    # Bit that is actually peak, not baseline
    span=[upper_0baseline, lower_1baseline]

    # lower_2baseline=1320
    # upper_2baseline=1330

    # Trim for entire range
    Diad_short=Diad[ (Diad[:,0]>lower_0baseline) & (Diad[:,0]<upper_1baseline) ]

    # Get actual baseline
    Baseline_with_outl=Diad_short[
    ((Diad_short[:, 0]<upper_0baseline) &(Diad_short[:, 0]>lower_0baseline))
         |
    ((Diad_short[:, 0]<upper_1baseline) &(Diad_short[:, 0]>lower_1baseline))]

    # Calculates the median for the baseline and the standard deviation
    Median_Baseline=np.mean(Baseline_with_outl[:, 1])
    Std_Baseline=np.std(Baseline_with_outl[:, 1])

    # Removes any points in the baseline outside of 2 sigma (helps remove cosmic rays etc).
    Baseline=Baseline_with_outl[(Baseline_with_outl[:, 1]<Median_Baseline+sigma*Std_Baseline)
                                &
                                (Baseline_with_outl[:, 1]>Median_Baseline-sigma*Std_Baseline)
                               ]


    #Baseline=Baseline_with_outl



    # Fits a polynomial to the baseline of degree
    Pf_baseline = np.poly1d(np.polyfit(Baseline[:, 0], Baseline[:, 1], N_poly))
    Py_base =Pf_baseline(Diad_short[:, 0])


    Baseline_ysub=Pf_baseline(Baseline[:, 0])
    Baseline_x=Baseline[:, 0]
    y_corr= Diad_short[:, 1]-  Py_base
    x=Diad_short[:, 0]

     # Plotting what its doing
    if plot_figure is True:
        if block_print is False:
            print('Plotting baselines here for easier inspection and tweaking')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        ax1.set_title('Background fit')
        ax1.plot(Diad[:, 0], Diad[:, 1], '-', color='grey')
        ax1.plot(Diad_short[:, 0], Diad_short[:, 1], '-r', label='region_2_bck_sub')

        #ax1.plot(Baseline[:, 0], Baseline[:, 1], '-b', label='Bck points')
        ax1.plot(Baseline[:, 0], Baseline[:, 1], '.b', label='Bck points')
        ax1.plot(Diad_short[:, 0], Py_base, '-k')



        ax1_ymin=np.min(Baseline[:, 1])-10*np.std(Baseline[:, 1])
        ax1_ymax=np.max(Baseline[:, 1])+10*np.std(Baseline[:, 1])
        ax1_xmin=lower_0baseline-30
        ax1_xmax=upper_1baseline+30
        # Adding patches


        rect_diad1_b1=patches.Rectangle((lower_0baseline, ax1_ymin),upper_0baseline-lower_0baseline,ax1_ymax-ax1_ymin,
                              linewidth=1,edgecolor='none',facecolor='cyan', label='bck', alpha=0.3, zorder=0)
        ax1.add_patch(rect_diad1_b1)
        rect_diad1_b2=patches.Rectangle((lower_1baseline, ax1_ymin),upper_1baseline-lower_1baseline,ax1_ymax-ax1_ymin,
                              linewidth=1,edgecolor='none',facecolor='cyan', alpha=0.3, zorder=0)
        ax1.add_patch(rect_diad1_b2)
        ax1.set_xlim([ax1_xmin, ax1_xmax])
        ax1.set_ylim([ax1_ymin, ax1_ymax])

        ax1.set_ylabel('Intensity')
        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Wavenumber')
        ax1.legend()



        ax2.set_title('Background subtracted')
        ax2.plot(x, y_corr, '-r')
        height_p=np.max(Diad_short[:, 1])-np.min(Diad_short[:, 1])
        ax2.set_ylim([np.min(y_corr), 1.2*height_p ])
        ax1.set_xlabel('Wavenumber')


    return y_corr, Py_base, x,  Diad_short, Py_base, Pf_baseline, Baseline_ysub, Baseline_x, Baseline, span



def add_peak(*, prefix=None, center=None,
min_cent=None, max_cent=None, min_sigma=None, max_sigma=None, amplitude=100, sigma=0.2):
    """
    This function iteratively adds peaks for lmfit
    """
    Model_combo=VoigtModel(prefix=prefix)#+ConstantModel(prefix=prefix) #Stops getting results
    peak =  Model_combo
    pars = peak.make_params()

    if min_cent is not None and max_cent is not None:
        pars[prefix + 'center'].set(center, min=min_cent, max=max_cent)
    else:
        pars[prefix + 'center'].set(center)


    pars[prefix + 'amplitude'].set(amplitude, min=0)

    if min_sigma is not None:
        pars[prefix+'sigma'].set(fwhm, max=max_sigma)
    else:
        pars[prefix + 'sigma'].set(sigma, min=0)
    return peak, pars






def fit_gaussian_voigt_diad1(*, path=None, filename=None,
                                xdat=None, ydat=None,
                                peak_pos_voigt=(1263, 1283),
                                peak_pos_gauss=None,
                                gauss_sigma=1,
                                gauss_amp=3000,
                                diad_sigma=0.2,

                                diad_amplitude=100,
                                HB_amplitude=20,
                                span=None,
                                plot_figure=True,  dpi=200):

    """ This function fits diad 1 at ~1283, and the hot band if present

    Parameters
    -----------
    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read

    xdat, ydat: pd.series
        x and background substracted y data to fit.

    peak_pos_voigt: list
        Estimates of peak positions for peaks

    peak_pos_gauss: None, int, or float
        If you want a gaussian as part of your fit, put an approximate center here

    amplitude: int, float
        Approximate amplitude of main peak

    plot_figure: bool
        if True, saves figure

    dpi: int
        dpi for saved figure

    Returns
    -----------
    result, df_out, y_best_fit, x_lin

        result: fitted model
        df_out: Dataframe of fit parameters for diad.



    """

    if peak_pos_gauss is None:
        # Fit just as many peaks as there are peak_pos_voigt

        # If peak find functoin has put out a float, does this for 1 peak
        if type(peak_pos_voigt) is float or type(peak_pos_voigt) is np.float64 or type(peak_pos_voigt) is int:
            model_F = VoigtModel(prefix='lz1_')# + ConstantModel(prefix='c1')
            pars1 = model_F.make_params()
            pars1['lz1_'+ 'amplitude'].set(diad_amplitude, min=0, max=diad_amplitude*10)
            pars1['lz1_'+ 'center'].set(peak_pos_voigt)
            pars1['lz1_'+ 'sigma'].set(diad_sigma, min=diad_sigma/10, max=diad_sigma*10)
            params=pars1
            # Sometimes length 1 can be with a comma
        else:
            #  If peak find function put out a tuple length 1
            if len(peak_pos_voigt)==1:
                model_F = VoigtModel(prefix='lz1_') #+ ConstantModel(prefix='c1')
                pars1 = model_F.make_params()
                pars1['lz1_'+ 'amplitude'].set(diad_amplitude, min=0, max=diad_amplitude*10)
                pars1['lz1_'+ 'center'].set(peak_pos_voigt[0])
                pars1['lz1_'+ 'sigma'].set(diad_sigma, min=diad_sigma/10, max=diad_sigma*10)
                params=pars1

            if len(peak_pos_voigt)==2:

                # Code from 1447
                model_prel = VoigtModel(prefix='lzp_') #+ ConstantModel(prefix='c1')
                pars2 = model_prel.make_params()
                pars2['lzp_'+ 'amplitude'].set(diad_amplitude, min=0)
                pars2['lzp_'+ 'center'].set(peak_pos_voigt[0])


                init = model_prel.eval(pars2, x=xdat)
                result_prel = model_prel.fit(ydat, pars2, x=xdat)
                comps_prel = result_prel.eval_components()

                Peakp_Cent=result_prel.best_values.get('lzp_center')
                Peakp_Area=result_prel.best_values.get('lzp_amplitude')


                # Then use these to inform next peak
                model1 = VoigtModel(prefix='lz1_')#+ ConstantModel(prefix='c1')
                pars1 = model1.make_params()
                pars1['lz1_'+ 'amplitude'].set(Peakp_Area, min=Peakp_Area/2, max=Peakp_Area*2)
                pars1['lz1_'+ 'center'].set(Peakp_Cent, min=Peakp_Cent-1, max=Peakp_Cent+2)

                # Second wee peak
                prefix='lz2_'
                peak = VoigtModel(prefix='lz2_')#+ ConstantModel(prefix='c2')
                pars = peak.make_params()
                pars[prefix + 'center'].set(min(peak_pos_voigt), min=min(peak_pos_voigt)-2, max=min(peak_pos_voigt)+2)
                pars[prefix + 'amplitude'].set(HB_amplitude, min=0, max=Peakp_Area/3)


                model_F=model1+peak
                pars1.update(pars)
                params=pars1





    if peak_pos_gauss is not None:

        model = GaussianModel(prefix='bkg_')
        params = model.make_params()
        params['bkg_'+'amplitude'].set(gauss_amp, min=gauss_amp/10, max=gauss_amp*10)
        params['bkg_'+'sigma'].set(gauss_sigma, min=gauss_sigma/10, max=gauss_sigma*10)
        params['bkg_'+'center'].set(peak_pos_gauss, min=peak_pos_gauss-10, max=peak_pos_gauss+10)






        rough_peak_positions = peak_pos_voigt
        # If you want a Gaussian background
        if type(peak_pos_voigt) is float or type(peak_pos_voigt) is np.float64 or type(peak_pos_voigt) is int:
                peak, pars = add_peak(prefix='lz1_', center=peak_pos_voigt, diad_amplitude=amplitude)
                model = peak+model
                params.update(pars)
        else:
            if len(peak_pos_voigt)==1:
                if type(peak_pos_voigt) is tuple:
                    print('im atuple')
                    peak_pos_voigt2=peak_pos_voigt[0]
                else:
                    peak_pos_voigt2=peak_pos_voigt

                peak, pars = add_peak(prefix='lz1_', center=peak_pos_voigt2, min_cent=peak_pos_voigt2-5, max_cent=peak_pos_voigt2+5)
                model = peak+model
                params.update(pars)





            if len(peak_pos_voigt)>1:
                for i, cen in enumerate(rough_peak_positions):

                    peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen, amplitude=diad_amplitude)
                    model = peak+model
                    params.update(pars)
                if type(peak_pos_voigt) is float or type(peak_pos_voigt) is np.float64 or type(peak_pos_voigt) is int:
                    peak, pars = add_peak(prefix='lz1_', center=cen, amplitude=diad_amplitude)
                    model = peak+model
                    params.update(pars)



        model_F=model

    # Regardless of model, now evalcuate it
    init = model_F.eval(params, x=xdat)
    result = model_F.fit(ydat, params, x=xdat)
    comps = result.eval_components()



    # Get first peak center
    Peak1_Cent=result.best_values.get('lz1_center')
    Peak1_Int=result.best_values.get('lz1_amplitude')
    Peak1_sigma=result.best_values.get('lz1_sigma')
    Peak1_gamma=result.best_values.get('lz1_gamma')

    df_out=pd.DataFrame(data={'Diad1_Voigt_Cent': Peak1_Cent,
                            'Diad1_Voigt_Area': Peak1_Int,
                            'Diad1_Voigt_Sigma': Peak1_sigma,
                            'Diad1_Voigt_Gamma': Peak1_gamma,

    }, index=[0])

    if Peak1_Int>=(diad_amplitude*10-0.1):
        w.warn('Diad fit right at the upper limit of the allowed fit parameter, change diad_amplitude in the config file')
    if Peak1_sigma>=(diad_amplitude*10-0.1):
        w.warn('Diad fit right at the upper limit of the allowed fit parameter, change diad_amplitude in the config file')
    if Peak1_sigma>=(diad_sigma*10-0.1):
        w.warn('Diad fit right at the upper limit of the allowed fit parameter, change diad_sigma in the config file')
    if Peak1_sigma<=(diad_sigma/10+0.1):
        w.warn('Diad fit right at the lower limit of the allowed fit parameter, change diad_sigma in the config file')



    if peak_pos_gauss is not None:
        Gauss_cent=result.best_values.get('bkg_center')
        Gauss_amp=result.best_values.get('bkg_amplitude')
        Gauss_sigma=result.best_values.get('bkg_sigma')
        df_out['Gauss_Cent']=Gauss_cent
        df_out['Gauss_Area']=Gauss_amp
        df_out['Gauss_Sigma']=Gauss_sigma
        if Gauss_sigma>=(gauss_sigma*10-0.1):
            w.warn('Best fit Gauss sigma right at the upper limit of the allowed fit parameter, change gauss_sigma in the config file')
        if Gauss_sigma<=(gauss_sigma/10+0.1):
            w.warn('Best fit Gauss  sigma is right at the lower limit of the allowed fit parameter, change gauss_sigma in the config file')
        if Gauss_amp>=(gauss_amp*10-0.1):
            w.warn('Best fit Gauss amplitude is right at the upper limit of the allowed fit parameter, change gauss_amp in the config file')
        if Gauss_amp<=(gauss_sigma/10+0.1):
            w.warn('Best fit Gauss amplitude is right at the lower limit of the allowed fit parameter, change gauss_amp in the configfile')
        if Gauss_cent<=(peak_pos_gauss-30+0.5):
            w.warn('Best fit Gauss Cent is right at the lower limit of the allowed fit parameter, change peak_pos_gauss in the configfile')
        if Gauss_cent>=(peak_pos_gauss+30-0.5):
            w.warn('Best fit Gauss Cent is right at the upper limit of the allowed fit parameter, change peak_pos_gauss in the configfile')




    x_lin=np.linspace(span[0], span[1], 2000)

    y_best_fit=result.eval(x=x_lin)
    components=result.eval_components(x=x_lin)


    x_cent_lin=np.linspace(Peak1_Cent-1, Peak1_Cent+1, 20000)

    y_cent_best_fit=result.eval(x=x_cent_lin)
    diad_height = np.max(y_cent_best_fit)
    df_out['Diad1_Combofit_Height']= diad_height
    df_out.insert(0, 'Diad1_Combofit_Cent', np.nanmean(x_cent_lin[y_cent_best_fit==diad_height]))



        # Uncommnet to get full report
    if print is True:
        print(result.fit_report(min_correl=0.5))

    # Checing for error bars
    Error_bars=result.errorbars
    if Error_bars is False:
        if block_print is False:
            print('Error bars not determined by function')



    if len(peak_pos_voigt)==2:
        Peak2_Cent=result.best_values.get('lz2_center')
        Peak2_Int=result.best_values.get('lz2_amplitude')
        df_out['HB1_Cent']=Peak2_Cent
        df_out['HB1_Area']=Peak2_Int

    if len(peak_pos_voigt)==3:
        Peak3_Cent=result.best_values.get('lz3_center')
        Peak3_Int=result.best_values.get('lz3_amplitude')
        df_out['Peak3_Cent']=Peak3_Cent
        df_out['Peak3_Area']=Peak3_Int





    if len(peak_pos_voigt)>1:
        lowerpeak=np.min([Peak1_Cent, Peak2_Cent])
        upperpeak=np.max([Peak1_Cent, Peak2_Cent])

        ax1_xlim=[Peak1_Cent-50, Peak1_Cent+20]
        ax2_xlim=[Peak1_Cent-50, Peak1_Cent+20]
    else:
        ax1_xlim=[Peak1_Cent-20, Peak1_Cent+20]
        ax2_xlim=[Peak1_Cent-20, Peak1_Cent+20]

    # Calculating residuals
    result_diad1_origx_all=result.eval(x=xdat)
    # Y evaluated at actual axes
    #print(result_diad2_origx_all)

    result_diad1_origx=result_diad1_origx_all[(xdat>span[0]) & (xdat<span[1])]
    ydat_inrange=ydat[(xdat>span[0]) & (xdat<span[1])]
    xdat_inrange=xdat[(xdat>span[0]) & (xdat<span[1])]
    residual_diad1_coords=ydat_inrange-result_diad1_origx


    residual_diad1=np.sum(((ydat_inrange-result_diad1_origx)**2)**0.5)/(len(ydat_inrange))
    df_out['Residual_Diad1']=residual_diad1


    if plot_figure is True:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5), sharey=True)
        # Residuals
        ax3.plot(xdat, ydat-result_diad1_origx, '-r')
        ax3.set_xlabel('Wavenumber')
        ax3.set_ylabel('Residual')
        ax1.plot(xdat, ydat,  '.k', label='data')



        ax1.plot(x_lin, y_best_fit, '-g', label='best fit')
        ax1.legend()

        ax2.plot(xdat, ydat, '.k')
        if peak_pos_gauss is not None:
            ax2.plot(x_lin, components.get('bkg_'), '-c', label='Gaussian bck', linewidth=1)
        if len(peak_pos_voigt)>1:
            ax2.plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='Peak2')
        ax2.plot(x_lin, components.get('lz1_'), '-b', linewidth=2, label='Peak1')
        #ax2.plot(xdat, result.best_fit, '-g', label='best fit')
        ax2.legend()
        fitspan=max(y_best_fit)-min(y_best_fit)
        ax2.set_ylim([min(y_best_fit)-fitspan/5, max(y_best_fit)+fitspan/5])

        ax1.set_ylabel('Intensity')
        ax1.set_xlabel('Wavenumber')
        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Wavenumber')

        path3=path+'/'+'diad_fit_images'
        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ 'diad_fit_images', exist_ok=False)

        if block_print is False:
            print(path)
        file=filename.rsplit('.txt', 1)[0]
        fig.savefig(path3+'/'+'Diad1_Fit_{}.png'.format(file), dpi=dpi)

    # Result = Model fitted
    #df_out=df of peak positions
    #y_best_fit = Best fit evaluated at x_lin (linspace of points covering range)
    # components - fit for different components of fit (e.g. multiple lines
    # xdat and ydat, data being fitted (background corrected)
    #ax1_xlim, ax2_xlim: Limits for 2 axes.

    return result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim, peak_pos_gauss, residual_diad1_coords, ydat_inrange,  xdat_inrange


def fit_gaussian_voigt_diad2(*,  path=None,filename=None, xdat=None, ydat=None, peak_pos_voigt=(1389, 1410),
                    diad_amplitude=100, diad_sigma=0.2, HB_amplitude=20, peak_pos_gauss=(1400), gauss_sigma=None, gauss_amp=100, span=None, plot_figure=True, dpi=200,
                    block_print=True):


    """ This function fits diad 2, at ~1389 and the hot band and C13 peak

    Parameters
    -----------
    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read

    xdat, ydat: pd.series
        x and background substracted y data to fit.

    peak_pos_voigt: list
        Estimates of peak positions for peaks.
        Fits as many peaks as positions in this list

    peak_pos_gauss: None, int, or flota
        If you want a gaussian as part of your fit, put an approximate center here

    amplitude: int, float
        Approximate amplitude of main peak

    span: list
        x bits that are actually peak for residuals. Basically upper and lower back coords

    plot_figure: bool
        if True, saves figure

    dpi: int
        dpi for saved figure

    Returns
    -----------
    result, df_out, y_best_fit, x_lin

        result: fitted model
        df_out: Dataframe of fit parameters for diad.


    """
    # Super useful in all cases to fit a first peak, which is the biggest peak


        # If peak find functoin has put out a float, does this for 1 peak

    if type(peak_pos_voigt) is np.ndarray:
        peak_pos_voigt=peak_pos_voigt[0]


    if type(peak_pos_voigt) is float or type(peak_pos_voigt) is int or type(peak_pos_voigt) is np.float64:
        initial_guess=peak_pos_voigt
        type_peak="int"
            # Sometimes length 1 can be with a comma
    else:


        if len(peak_pos_voigt)==1:
            initial_guess=peak_pos_voigt[0]
        if len(peak_pos_voigt)==2:
            initial_guess=np.min(peak_pos_voigt)
        if len(peak_pos_voigt)==3:
            initial_guess=np.median(peak_pos_voigt)

    model_ini = VoigtModel()#+ ConstantModel()

    # create parameters with initial values
    params_ini = model_ini.make_params(center=initial_guess)
    params_ini['amplitude'].set(diad_amplitude, min=0, max=diad_amplitude*10)
    params_ini['sigma'].set(diad_sigma, min=diad_sigma/10, max=diad_sigma*10)
    init_ini = model_ini.eval(params_ini, x=xdat)


    result_ini  = model_ini.fit(ydat, params_ini, x=xdat)
    comps_ini  = result_ini.eval_components()
    Center_ini=result_ini.best_values.get('center')
    Amplitude_ini=result_ini.params.get('amplitude')
    sigma_ini=result_ini.params.get('sigma')
    fwhm_ini=result_ini.params.get('fwhm')
    if block_print is False:
        print(Center_ini)
        print(sigma_ini)



    if peak_pos_gauss is None:
        # Fit just as many peaks as there are peak_pos_voigt

        if type(peak_pos_voigt) is float or type(peak_pos_voigt) is int or type(peak_pos_voigt) is np.float64:
            model_F = VoigtModel(prefix='lz1_')#+ ConstantModel(prefix='c1')
            pars1 = model_F.make_params()
            pars1['lz1_'+ 'amplitude'].set(diad_amplitude, min=0, max=diad_amplitude*10)
            pars1['lz1_'+ 'center'].set(peak_pos_voigt)
            pars1['lz1_'+ 'sigma'].set(diad_sigma, min=diad_sigma/10, max=diad_sigma*10)
            params=pars1

        else:
            if len(peak_pos_voigt)==1:
                model_F = VoigtModel(prefix='lz1_')#+ ConstantModel(prefix='c1')
                pars1 = model_F.make_params()
                pars1['lz1_'+ 'amplitude'].set(diad_amplitude, min=0, max=diad_amplitude*10)
                pars1['lz1_'+ 'center'].set(peak_pos_voigt[0])
                pars1['lz1_'+ 'sigma'].set(diad_sigma, min=diad_sigma/10, max=diad_sigma*10)
                params=pars1

            if len(peak_pos_voigt)==2:

                Peakp_Cent=Center_ini
                Peakp_Area=Amplitude_ini
                Peakp_HW=fwhm_ini



                # Then use these to inform next peak
                model1 = VoigtModel(prefix='lz1_')#+ ConstantModel(prefix='c1')
                pars1 = model1.make_params()
                pars1['lz1_'+ 'amplitude'].set(Peakp_Area, min=Peakp_Area/2, max=Peakp_Area*2)
                pars1['lz1_'+ 'center'].set(Peakp_Cent, min=Peakp_Cent-1, max=Peakp_Cent+2)

                # Second wee peak
                prefix='lz2_'
                peak = VoigtModel(prefix='lz2_')#+ ConstantModel(prefix='c2')
                pars = peak.make_params()
                pars[prefix + 'center'].set(max(peak_pos_voigt), min=max(peak_pos_voigt)-2, max=max(peak_pos_voigt)+2)
                pars[prefix + 'amplitude'].set(HB_amplitude, min=Peakp_Area/100, max=Peakp_Area/5)
                pars[prefix+ 'fwhm'].set(Peakp_HW, min=Peakp_HW/10, max=Peakp_HW*10)

                model_F=model1+peak
                pars1.update(pars)
                params=pars1

            if len(peak_pos_voigt)==3:
                if block_print is False:
                    print('Trying to iteratively fit 3 peaks')
                low_peak=np.min(peak_pos_voigt)
                med_peak=np.median(peak_pos_voigt)
                high_peak=np.max(peak_pos_voigt)
                peak_pos_left=np.array([low_peak, high_peak])

                model = VoigtModel(prefix='lz1_')#+ ConstantModel(prefix='c1')
                params = model.make_params()
                params['lz1_'+ 'amplitude'].set(amplitude, min=0)
                params['lz1_'+ 'center'].set(med_peak)

                for i, cen in enumerate(peak_pos_left):

                    peak, pars = add_peak(prefix='lz%d_' % (i+2), center=cen,
                    min_cent=cen-3, max_cent=cen+3, sigma=sigma_ini, max_sigma=sigma_ini*5)
                    model = peak+model
                    params.update(pars)

                model_F=model




    # Same, but also with a Gaussian Background
    if peak_pos_gauss is not None:

        model = GaussianModel(prefix='bkg_')
        params = model.make_params()
        params['bkg_'+'amplitude'].set(gauss_amp, min=gauss_amp/10, max=gauss_amp*10)
        params['bkg_'+'sigma'].set(gauss_sigma, min=gauss_sigma/10, max=gauss_sigma*10)
        params['bkg_'+'center'].set(peak_pos_gauss, min=peak_pos_gauss-10, max=peak_pos_gauss+10)




        rough_peak_positions = peak_pos_voigt
        # If you want a Gaussian background
        if type(peak_pos_voigt) is float or type(peak_pos_voigt) is np.float64 or type(peak_pos_voigt) is int:
            type_peak="int"
            peak, pars = add_peak(prefix='lz1_', center=peak_pos_voigt, amplitude=amplitude)
            model = peak+model
            params.update(pars)
        else:

            if len(peak_pos_voigt)==1:
                peak, pars = add_peak(prefix='lz1_', center=cen, amplitude=amplitude)
                model = peak+model
                params.update(pars)

            if len(peak_pos_voigt)==2:
                if block_print is False:
                    print('Fitting 2 voigt peaks iteratively ')
                for i, cen in enumerate(peak_pos_voigt):
                    if block_print is False:
                        print('working on voigt peak' + str(i))
                    #peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen)
                    peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen,
                    min_cent=cen-3, max_cent=cen+3, sigma=sigma_ini, max_sigma=sigma_ini*5)


                    model = peak+model
                    params.update(pars)

            if len(peak_pos_voigt)==3:
                if block_print is False:
                    print('Fitting 2 peaks iteratively, then adding C13')
                for i, cen in enumerate(peak_pos_voigt):
                    if block_print is False:
                        print('working on voigt peak' + str(i))
                    #peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen)
                    peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen,
                    min_cent=cen-3, max_cent=cen+3, sigma=sigma_ini, max_sigma=sigma_ini*2)


                    model = peak+model
                    params.update(pars)

        model_F=model



    # Regardless of fit, evaluate model
    init = model_F.eval(params, x=xdat)
    result = model_F.fit(ydat, params, x=xdat)
    comps = result.eval_components()

    #print(result.fit_report(min_correl=0.5))
    # Check if function gives error bars
    Error_bars=result.errorbars
    if Error_bars is False:
        if block_print is False:
            print('Error bars not determined by function')

    # Get first peak center
    Peak1_Cent=result.best_values.get('lz1_center')
    Peak1_Int=result.best_values.get('lz1_amplitude')
    Peak1_sigma=result.best_values.get('lz1_sigma')
    Peak1_gamma=result.best_values.get('lz1_gamma')

    if Peak1_Int>=(diad_amplitude*10-0.1):
        w.warn('Diad fit right at the upper limit of the allowed fit parameter, change diad_amplitude in the config file')
    if Peak1_sigma>=(diad_amplitude*10-0.1):
        w.warn('Diad fit right at the upper limit of the allowed fit parameter, change diad_amplitude in the config file')
    if Peak1_sigma>=(diad_sigma*10-0.1):
        w.warn('Diad fit right at the upper limit of the allowed fit parameter, change diad_sigma in the config file')
    if Peak1_sigma<=(diad_sigma/10+0.1):
        w.warn('Diad fit right at the lower limit of the allowed fit parameter, change diad_sigma in the config file')


    if peak_pos_gauss is not None:
        Gauss_cent=result.best_values.get('bkg_center')
        Gauss_amp=result.best_values.get('bkg_amplitude')
        Gauss_sigma=result.best_values.get('bkg_sigma')
        if Gauss_sigma>=(gauss_sigma*10-0.1):
            w.warn('Best fit Gauss sigma right at the upper limit of the allowed fit parameter, change gauss_sigma in the config file')
        if Gauss_sigma<=(gauss_sigma/10+0.1):
            w.warn('Best fit Gauss  sigma is right at the lower limit of the allowed fit parameter, change gauss_sigma in the config file')
        if Gauss_amp>=(gauss_amp*10-0.1):
            w.warn('Best fit Gauss amplitude is right at the upper limit of the allowed fit parameter, change gauss_amp in the config file')
        if Gauss_amp<=(gauss_sigma/10+0.1):
            w.warn('Best fit Gauss amplitude is right at the lower limit of the allowed fit parameter, change gauss_amp in the configfile')
        if Gauss_cent<=(peak_pos_gauss-30+0.5):
            w.warn('Best fit Gauss Cent is right at the lower limit of the allowed fit parameter, change peak_pos_gauss in the configfile')
        if Gauss_cent>=(peak_pos_gauss+30-0.5):
            w.warn('Best fit Gauss Cent is right at the upper limit of the allowed fit parameter, change peak_pos_gauss in the configfile')



    Peak1_Int=result.best_values.get('lz1_amplitude')
    # print('fwhm gauss')
    # print(result.best_values)


    x_lin=np.linspace(span[0], span[1], 2000)
    y_best_fit=result.eval(x=x_lin)
    components=result.eval_components(x=x_lin)

    #

    # Work out what peak is what


    if type(peak_pos_voigt) is float or type(peak_pos_voigt) is np.float64 or type(peak_pos_voigt) is int:

        Peak2_Cent=None
        Peak3_Cent=None
        ax1_xlim=[peak_pos_voigt-15, peak_pos_voigt+15]
        ax2_xlim=[peak_pos_voigt-15, peak_pos_voigt+15]

    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:
        if len(peak_pos_voigt)==1:
            Peak2_Cent=None
            Peak3_Cent=None
            ax1_xlim=[peak_pos_voigt[0]-15, peak_pos_voigt[0]+15]
            ax2_xlim=[peak_pos_voigt[0]-15, peak_pos_voigt[0]+15]
        if len(peak_pos_voigt)==2:
            Peak2_Cent=result.best_values.get('lz2_center')
            Peak2_Int=result.best_values.get('lz2_amplitude')
            Peak3_Cent=None
            ax1_xlim=[peak_pos_voigt[0]-15, peak_pos_voigt[0]+30]
            ax2_xlim=[peak_pos_voigt[0]-15, peak_pos_voigt[0]+30]

        if len(peak_pos_voigt)==3:
            Peak2_Cent=result.best_values.get('lz2_center')
            Peak2_Int=result.best_values.get('lz2_amplitude')
            Peak3_Cent=result.best_values.get('lz3_center')
            Peak3_Int=result.best_values.get('lz3_amplitude')

            ax1_xlim=[peak_pos_voigt[0]-30, peak_pos_voigt[0]+30]
            ax2_xlim=[peak_pos_voigt[0]-30, peak_pos_voigt[0]+30]

    if Peak2_Cent is None:
        df_out=pd.DataFrame(data={'Diad2_Voigt_Cent': Peak1_Cent,
                                'Diad2_Voigt_Area': Peak1_Int,
                            'Diad2_Voigt_Sigma': Peak1_sigma,
                            'Diad2_Voigt_Gamma': Peak1_gamma
        }, index=[0])

    if Peak2_Cent is not None:


        if Peak3_Cent is None:
            Peaks=np.array([Peak1_Cent, Peak2_Cent])
            # Diad is lower peak
            Diad2_Cent=np.min(Peaks)
            # Hot band is upper peak'
            HB2_Cent=np.max(Peaks)
            # Allocate areas
            if Diad2_Cent==Peak2_Cent:
                Diad2_Int=Peak2_Int
                HB2_Int=Peak1_Int
            if Diad2_Cent==Peak1_Cent:
                Diad2_Int=Peak1_Int
                HB2_Int=Peak2_Int




        if Peak3_Cent is not None:
            Peaks=np.array([Peak1_Cent, Peak2_Cent, Peak3_Cent])

            # C13 is lower peak
            C13_Cent=np.min(Peaks)
            # Diad is medium peak
            Diad2_Cent=np.median(Peaks)
            # Hot band is upper peak'
            HB2_Cent=np.max(Peaks)

            if Diad2_Cent==Peak2_Cent:
                Diad2_Int=Peak2_Int
            if Diad2_Cent==Peak1_Cent:
                Diad2_Int=Peak1_Int
            if Diad2_Cent==Peak3_Cent:
                Diad2_Int=Peak3_Int
            # Same for hotband
            if HB2_Cent==Peak2_Cent:
                HB2_Int=Peak2_Int
            if HB2_Cent==Peak1_Cent:
                HB2_Int=Peak1_Int
            if HB2_Cent==Peak3_Cent:
                HB2_Int=Peak3_Int

            # Same for C13
            if C13_Cent==Peak2_Cent:
                C13_Int=Peak2_Int
            if C13_Cent==Peak1_Cent:
                C13_Int=Peak1_Int
            if C13_Cent==Peak3_Cent:
                C13_Int=Peak3_Int

        df_out=pd.DataFrame(data={'Diad2_Voigt_Cent': Diad2_Cent,
                                'Diad2_Voigt_Area': Diad2_Int,
                                'Diad2_Voigt_Sigma': Peak1_sigma,
                                'Diad2_Voigt_Gamma': Peak1_gamma,
        }, index=[0])

        df_out['HB2_Cent']=HB2_Cent
        df_out['HB2_Area']=HB2_Int
        if  Peak3_Cent is not None:
            df_out['C13_Cent']=C13_Cent
            df_out['C13_Area']=C13_Int

    result_diad2_origx_all=result.eval(x=xdat)
    # Trim to be in range
    #print(result_diad2_origx_all)
    result_diad2_origx=result_diad2_origx_all[(xdat>span[0]) & (xdat<span[1])]
    ydat_inrange=ydat[(xdat>span[0]) & (xdat<span[1])]
    xdat_inrange=xdat[(xdat>span[0]) & (xdat<span[1])]
    residual_diad2_coords=ydat_inrange-result_diad2_origx


    x_cent_lin=np.linspace(df_out['Diad2_Voigt_Cent']-1, df_out['Diad2_Voigt_Cent']+1, 20000)
    y_cent_best_fit=result.eval(x=x_cent_lin)
    diad_height = np.max(y_cent_best_fit)
    df_out['Diad2_Combofit_Height']= diad_height
    df_out.insert(0, 'Diad2_Combofit_Cent', np.nanmean(x_cent_lin[y_cent_best_fit==diad_height]))


    residual_diad2=np.sum(((ydat_inrange-result_diad2_origx)**2)**0.5)/(len(ydat_inrange))
    df_out['Residual_Diad2']=residual_diad2

    if peak_pos_gauss is not None:
        df_out['Gauss_Cent']=Gauss_cent
        df_out['Gauss_Area']=Gauss_amp
        df_out['Gauss_Sigma']=Gauss_sigma



    if plot_figure is True:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        ax1.plot(x_lin, y_best_fit, '-g', linewidth=2, label='best fit')
        ax1.plot(xdat, ydat,  '.k', label='data')
        ax1.legend()
        ax1.set_xlim(ax1_xlim)
        ax2.set_xlim(ax2_xlim)






        if peak_pos_gauss is not None:
            ax2.plot(x_lin, components.get('bkg_'), '.c',linewidth=2,  label='Gaussian bck')
        ax2.plot(x_lin, components.get('lz1_'), '-b', linewidth=2, label='Peak1')
        ax2.plot(xdat, ydat, '.k')


    #ax2.plot(xdat, result.best_fit, '-g', label='best fit')

        fitspan=max(y_best_fit)-min(y_best_fit)
        ax2.set_ylim([min(y_best_fit)-fitspan/5, max(y_best_fit)+fitspan/5])


        ax1.set_ylabel('Intensity')
        ax1.set_xlabel('Wavenumber')
        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Wavenumber')


        if len(peak_pos_voigt)>1:
            ax2.plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='Peak2')

        if len(peak_pos_voigt)>2:
            ax2.plot(x_lin, components.get('lz3_'), '-m', linewidth=2, label='Peak3')
        # if len(peak_pos_voigt)>1:
        #     lowerpeak=np.min([Peak1_Cent, Peak2_Cent])
        #     upperpeak=np.max([Peak1_Cent, Peak2_Cent])
        #     ax1.set_xlim([lowerpeak-15, upperpeak+15])
        # if len(peak_pos_voigt)>1:
        #     ax2.set_xlim([lowerpeak-20, upperpeak+20])

        ax2.legend()

        path3=path+'/'+'diad_fit_images'
        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ 'diad_fit_images', exist_ok=False)


        file=filename.rsplit('.txt', 1)[0]
        fig.savefig(path3+'/'+'Diad2_Fit_{}.png'.format(file), dpi=dpi)




    best_fit=result.best_fit
    return result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim, peak_pos_gauss, residual_diad2_coords, ydat_inrange,  xdat_inrange

## Overall function for fitting diads in 1 single step
@dataclass
class diad1_fit_config:
    """
    Testing the documentation for these
    """
    # Do you need a gaussian? Set position here if so
    peak_pos_gauss: Optional [float] =None
    gauss_sigma: float=1
    gauss_amp: float = 3000

    diad_sigma: float=0.2



    # Degree of polynomial to use
    N_poly_bck_diad1: float =1

    # Background/baseline positions
    lower_bck_diad1: Tuple[float, float]=(1180, 1220)
    upper_bck_diad1: Tuple[float, float]=(1300, 1350)

    # Peak amplitude
    diad_amplitude: float = 100
    HB_amplitude: float = 20
    # How much to show on x anx y axis of figure showing background
    x_range_baseline: float=75
    y_range_baseline: float=100

    #Do you want to save the figure?
    plot_figure: bool = True
    dpi: float = 200
    x_range_residual: float=20

    # Do you want to return other parameters?
    return_other_params: bool =False
@dataclass
class diad2_fit_config:
    # Do you need a gaussian? Set position here if so
    peak_pos_gauss: Optional [float] =None
    gauss_sigma: float=1
    gauss_amp: float = 3000

    diad_sigma: float=0.2



    # Degree of polynomial to use
    N_poly_bck_diad2: float =1

    # Background/baseline positions
    lower_bck_diad2: Tuple[float, float]=(1300, 1360)
    upper_bck_diad2: Tuple[float, float]=(1440, 1470)

    # Peak amplitude
    diad_amplitude: float = 100
    HB_amplitude:  float = 20
    # How much to show on x anx y axis of figure showing background with all peaks imposed
    x_range_baseline: float=75
    y_range_baseline: float=100

    #Do you want to save the figure?
    plot_figure: bool = True
    dpi: float = 200
    x_range_residual: float=20

    # Do you want to return other parameters?
    return_other_params: bool =False




def fit_diad_2_w_bck(*, config1: diad2_fit_config=diad2_fit_config(), config2: diad_id_config=diad_id_config(),
    path=None, filename=None, peak_pos_voigt=None,filetype=None,
    close_figure=True):
    """ This function fits the background, then the diad + nearby peaks for Diad 2 @1389

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

    exclude_range1: None or list length 2
        Excludes a region, e.g. a cosmic ray

    exclude_range2: None or list length 2
        Excludes a region, e.g. a cosmic ray

    amplitude: int, float
        Approximate amplitude of main peak

    N_poly_bck_diad2: int
        Degree of polynomial to fit the background with

    lower_bck_diad2: list len 2
        wavenumbers of LH baseline region

    upper_bck_diad2: list len 2
        wavenumbers of RH baseline region


    peak_pos_voigt: list
        Estimates of peak positions for peaks

    peak_pos_gauss: None, int, or flota
        If you want a gaussian as part of your fit, put an approximate center here


    plot_figure: bool
        if True, saves figure

    dpi: int
        dpi for saved figure

    return_other_params: bool (default False)
        if False, just returns a dataframe of peak parameters
        if True, also returns:
            result: fit parameters
            y_best_fit: best fit to all curves in y
            x_lin: linspace of best fit coordinates in x.


    """
    Diad_df=get_data(path=path, filename=filename, filetype=filetype)
    Diad=np.array(Diad_df)
    # First, we feed data into the remove baseline function, which returns corrected data

    y_corr_diad2, Py_base_diad2, x_diad2,  Diad_short_diad2, Py_base_diad2, Pf_baseline_diad2,  Baseline_ysub_diad2, Baseline_x_diad2, Baseline_diad2, span_diad2=remove_diad_baseline(
   path=path, filename=filename, filetype=filetype, exclude_range1=config2.exclude_range1, exclude_range2=config2.exclude_range2, N_poly=config1.N_poly_bck_diad2,
    lower_bck=config1.lower_bck_diad2, upper_bck=config1.upper_bck_diad2, plot_figure=False)





    # Then, we feed this baseline-corrected data into the combined gaussian-voigt peak fitting function
    result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim, peak_pos_gauss,residual_diad2_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_diad2(path=path, filename=filename,
                    xdat=x_diad2, ydat=y_corr_diad2, diad_amplitude=config1.diad_amplitude,
                    diad_sigma=config1.diad_sigma,
                    HB_amplitude=config1.HB_amplitude,
                    peak_pos_voigt=peak_pos_voigt,
                    peak_pos_gauss=config1.peak_pos_gauss,
                    gauss_sigma=config1.gauss_sigma,  gauss_amp=config1.gauss_amp,
                    span=span_diad2, plot_figure=False)

    # get a best fit to the baseline using a linspace from the peak fitting
    ybase_xlin=Pf_baseline_diad2(x_lin)

    # We extract the full spectra to plot at the end, and convert to a dataframe
    Spectra_df=get_data(path=path, filename=filename, filetype=filetype)
    Spectra=np.array(Spectra_df)



    # Make nice figure

    figure_mosaic="""
    XY
    AB
    CD
    EE
    """

    fig,axes=plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(12, 16))
    fig.suptitle('Diad 2, file= '+ str(filename), fontsize=16, x=0.5, y=1.0)

    # Background plot for real

    # Plot best fit on the LHS, and individual fits on the RHS at the top

    axes['X'].set_title('a) Background fit')
    axes['X'].plot(Diad[:, 0], Diad[:, 1], '-', color='grey')
    axes['X'].plot(Diad_short_diad2[:, 0], Diad_short_diad2[:, 1], '-r', label='region_2_bck_sub')

    #axes['X'].plot(Baseline[:, 0], Baseline[:, 1], '-b', label='Bck points')
    axes['X'].plot(Baseline_diad2[:, 0], Baseline_diad2[:, 1], '.b', label='Bck points')
    axes['X'].plot(Diad_short_diad2[:, 0], Py_base_diad2, '-k')



    ax1_ymin=np.min(Baseline_diad2[:, 1])-10*np.std(Baseline_diad2[:, 1])
    ax1_ymax=np.max(Baseline_diad2[:, 1])+10*np.std(Baseline_diad2[:, 1])
    ax1_xmin=config1.lower_bck_diad2[0]-30
    ax1_xmax=config1.upper_bck_diad2[1]+30
    # Adding patches


    rect_diad2_b1=patches.Rectangle((config1.lower_bck_diad2[0], ax1_ymin),config1.lower_bck_diad2[1]-config1.lower_bck_diad2[0],ax1_ymax-ax1_ymin,
                            linewidth=1,edgecolor='none',facecolor='cyan', label='bck', alpha=0.3, zorder=0)
    axes['X'].add_patch(rect_diad2_b1)
    rect_diad2_b2=patches.Rectangle((config1.upper_bck_diad2[0], ax1_ymin),config1.upper_bck_diad2[1]-config1.upper_bck_diad2[0],ax1_ymax-ax1_ymin,
                            linewidth=1,edgecolor='none',facecolor='cyan', alpha=0.3, zorder=0)
    axes['X'].add_patch(rect_diad2_b2)
    axes['X'].set_xlim([ax1_xmin, ax1_xmax])
    axes['X'].set_ylim([ax1_ymin, ax1_ymax])

    axes['X'].set_ylabel('Intensity')
    axes['Y'].set_ylabel('Intensity')
    axes['Y'].set_xlabel('Wavenumber')
    axes['X'].legend()



    axes['Y'].set_title('b) Background subtracted spectra')
    axes['Y'].plot(x_diad2, y_corr_diad2, '-r')
    height_p=np.max(Diad_short_diad2[:, 1])-np.min(Diad_short_diad2[:, 1])
    axes['Y'].set_ylim([np.min(y_corr_diad2), 1.2*height_p ])
    axes['X'].set_xlabel('Wavenumber')



    axes['A'].plot(xdat, ydat,  '.k', label='data')
    axes['A'].plot( x_lin ,y_best_fit, '-g', linewidth=1, label='best fit')
    axes['A'].legend()
    axes['A'].set_ylabel('Intensity')
    axes['A'].set_xlabel('Wavenumber')
    axes['A'].set_xlim(ax1_xlim)
    axes['A'].set_title('c) Overall Best Fit')

   # individual fits
    axes['B'].plot(xdat, ydat, '.k')

    # This is for if there is more than 1 peak, this is when we want to plot the best fit
    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:
        if len(peak_pos_voigt)>1:
            axes['B'].plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='Peak2')

    axes['B'].plot(x_lin, components.get('lz1_'), '-b', linewidth=2, label='Peak1')
    if peak_pos_gauss is not None:
        axes['B'].plot(x_lin, components.get('bkg_'), '-m', label='Gaussian bck', linewidth=2)
    #ax2.plot(xdat, result.best_fit, '-g', label='best fit')
    axes['B'].legend()

    fitspan=max(y_best_fit)-min(y_best_fit)
    axes['B'].set_ylim([min(y_best_fit)-fitspan/5, max(y_best_fit)+fitspan/5])

    axes['B'].set_ylabel('Intensity')
    axes['B'].set_xlabel('Wavenumber')


    axes['B'].set_xlim(ax2_xlim)

    # Dashed lines so matches part D

    axes['B'].plot([df_out['Diad2_Voigt_Cent'], df_out['Diad2_Voigt_Cent']], [np.min(ydat), np.max(ydat)], ':b')

    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:
        if len(peak_pos_voigt)>=2:
            axes['B'].plot([df_out['HB2_Cent'], df_out['HB2_Cent']], [np.min(ydat), np.max(ydat)], ':r')

    axes['B'].set_title('c) Fit Components')

    # Background fit

    # First, set up x and y limits on axis

    if config1.x_range_baseline is not None:
        axc_xmin=df_out['Diad2_Voigt_Cent'][0]-config1.x_range_baseline
        axc_xmax=df_out['Diad2_Voigt_Cent'][0]+config1.x_range_baseline
    else:
        axc_xmin=config1.lower_bck_diad2[0]
        axc_xmax=config1.upper_bck_diad2[1]
    axc_ymin=np.min(Baseline_diad2[:, 1])-config1.y_range_baseline
    axc_ymax=np.max(Baseline_diad2[:, 1])+config1.y_range_baseline

    rect_diad2_b1=patches.Rectangle((config1.lower_bck_diad2[0],axc_ymin),config1.lower_bck_diad2[1]-config1.lower_bck_diad2[0],axc_ymax-axc_ymin,
                              linewidth=1,edgecolor='none',facecolor='cyan', label='bck', alpha=0.3, zorder=0)
    axes['C'].add_patch(rect_diad2_b1)
    rect_diad2_b2=patches.Rectangle((config1.upper_bck_diad2[0],axc_ymin),config1.upper_bck_diad2[1]-config1.upper_bck_diad2[0],axc_ymax-axc_ymin,
                              linewidth=1,edgecolor='none',facecolor='cyan', alpha=0.3, zorder=0)
    axes['C'].add_patch(rect_diad2_b2)


    axes['C'].set_title('d) Peaks overlain on data before subtraction')
    axes['C'].plot(Baseline_diad2[:, 0], Baseline_diad2[:, 1], '.b', label='bck')
    axes['C'].plot(Diad_short_diad2[:, 0], Py_base_diad2, '-k', label='Poly bck fit')
    axes['C'].plot(Diad_short_diad2[:, 0], Diad_short_diad2[:, 1], '.r', label='Data')

    axes['C'].set_ylabel('Intensity')
    axes['C'].set_xlabel('Wavenumber')


    if peak_pos_gauss is not None:

        axes['C'].plot(x_lin, components.get('bkg_')+ybase_xlin, '-m', label='Gaussian bck', linewidth=2)

    axes['C'].plot( x_lin ,y_best_fit+ybase_xlin, '-g', linewidth=2, label='Best Fit')
    axes['C'].plot(x_lin, components.get('lz1_')+ybase_xlin, '-b', label='Peak1', linewidth=1)
    if len(peak_pos_voigt)>1:
        axes['C'].plot(x_lin, components.get('lz2_')+ybase_xlin, '-r', label='Peak2', linewidth=1)


    axes['C'].legend()


    axes['C'].set_xlim([axc_xmin, axc_xmax])
    axes['C'].set_ylim([axc_ymin, axc_ymax])
    #axes['C'].plot(Diad_short[:, 0], Diad_short[:, 1], '"r', label='Data')


    # Residual on plot D
    axes['D'].set_title('f) Residuals')
    axes['D'].plot([df_out['Diad2_Voigt_Cent'], df_out['Diad2_Voigt_Cent']], [np.min(residual_diad2_coords), np.max(residual_diad2_coords)], ':b')
    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:
        if len(peak_pos_voigt)>=2:
            axes['D'].plot([df_out['HB2_Cent'], df_out['HB2_Cent']], [np.min(residual_diad2_coords), np.max(residual_diad2_coords)], ':r')

    axes['D'].plot(xdat_inrange, residual_diad2_coords, 'ok', mfc='c' )
    axes['D'].plot(xdat_inrange, residual_diad2_coords, '-c' )
    axes['D'].set_ylabel('Residual')
    axes['D'].set_xlabel('Wavenumber')
    # axes['D'].set_xlim(ax1_xlim)
    # axes['D'].set_xlim(ax2_xlim)
    Local_Residual_diad2=residual_diad2_coords[((xdat_inrange>(df_out['Diad2_Voigt_Cent'][0]-config1.x_range_residual))
                                            &(xdat_inrange<df_out['Diad2_Voigt_Cent'][0]+config1.x_range_residual))]
    axes['D'].set_xlim([df_out['Diad2_Voigt_Cent'][0]-config1.x_range_residual,
                df_out['Diad2_Voigt_Cent'][0]+config1.x_range_residual])
    #ax5.plot([cent_1117, cent_1117 ], [np.min(Local_Residual_1117)-10, np.max(Local_Residual_1117)+10], ':k')
    axes['D'].set_ylim([np.min(Local_Residual_diad2)-10, np.max(Local_Residual_diad2)+10])





    # Overal spectra
    axes['E'].set_title('g) Summary plot of raw spectra for file = ' + filename)
    axes['E'].plot(Spectra[:, 0], Spectra[:, 1], '-r')
    axes['E'].set_ylabel('Intensity')
    axes['E'].set_xlabel('Wavenumber')




    path3=path+'/'+'diad_fit_images'
    if os.path.exists(path3):
        out='path exists'
    else:
        os.makedirs(path+'/'+ 'diad_fit_images', exist_ok=False)

    fig.tight_layout()

    file=filename.rsplit('.txt', 1)[0]
    fig.savefig(path3+'/'+'diad2_Fit_{}.png'.format(file), dpi=config1.dpi)


    if close_figure is True:
        plt.close(fig)



    if  config1.return_other_params is False:
        return df_out
    else:
        return df_out, result, y_best_fit, x_lin




def fit_diad_1_w_bck(*, config1: diad1_fit_config=diad1_fit_config(), config2: diad_id_config=diad_id_config(),
    path=None, filename=None, peak_pos_voigt=None,filetype=None, close_figure=True):
    """ This function fits the background, then the diad + nearby peaks for Diad 2 @1389

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

    exclude_range1: None or list length 2
        Excludes a region, e.g. a cosmic ray

    exclude_range2: None or list length 2
        Excludes a region, e.g. a cosmic ray

    amplitude: int, float
        Approximate amplitude of main peak

    N_poly_bck_diad1: int
        Degree of polynomial to fit the background with

    lower_bck_diad1: list len 2
        wavenumbers of LH baseline region

    upper_bck_diad1: list len 2
        wavenumbers of RH baseline region


    peak_pos_voigt: list
        Estimates of peak positions for peaks

    peak_pos_gauss: None, int, or flota
        If you want a gaussian as part of your fit, put an approximate center here


    plot_figure: bool
        if True, saves figure

    dpi: int
        dpi for saved figure

    return_other_params: bool (default False)
        if False, just returns a dataframe of peak parameters
        if True, also returns:
            result: fit parameters
            y_best_fit: best fit to all curves in y
            x_lin: linspace of best fit coordinates in x.


    """
    Diad_df=get_data(path=path, filename=filename, filetype=filetype)
    Diad=np.array(Diad_df)
    # First, we feed data into the remove baseline function, which returns corrected data

    y_corr_diad1, Py_base_diad1, x_diad1,  Diad_short_diad1, Py_base_diad1, Pf_baseline_diad1,  Baseline_ysub_diad1, Baseline_x_diad1, Baseline_diad1, span_diad1=remove_diad_baseline(
   path=path, filename=filename, filetype=filetype, exclude_range1=config2.exclude_range1, exclude_range2=config2.exclude_range2, N_poly=config1.N_poly_bck_diad1,
    lower_bck=config1.lower_bck_diad1, upper_bck=config1.upper_bck_diad1, plot_figure=False)





    # Then, we feed this baseline-corrected data into the combined gaussian-voigt peak fitting function
    result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim, peak_pos_gauss,residual_diad1_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_diad1(path=path,                             filename=filename,
                    xdat=x_diad1, ydat=y_corr_diad1, diad_amplitude=config1.diad_amplitude,
                    diad_sigma=config1.diad_sigma,
                    HB_amplitude=config1.HB_amplitude,
                    peak_pos_voigt=peak_pos_voigt,
                    peak_pos_gauss=config1.peak_pos_gauss,
                    gauss_sigma=config1.gauss_sigma,  gauss_amp=config1.gauss_amp,
                    span=span_diad1, plot_figure=False)

    # get a best fit to the baseline using a linspace from the peak fitting
    ybase_xlin=Pf_baseline_diad1(x_lin)

    # We extract the full spectra to plot at the end, and convert to a dataframe
    Spectra_df=get_data(path=path, filename=filename, filetype=filetype)
    Spectra=np.array(Spectra_df)



    # Make nice figure

    figure_mosaic="""
    XY
    AB
    CD
    EE
    """

    fig,axes=plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(12, 16))
    fig.suptitle('Diad 1, file= '+ str(filename), fontsize=16, x=0.5, y=1.0)

    # Background plot for real

    # Plot best fit on the LHS, and individual fits on the RHS at the top

    axes['X'].set_title('a) Background fit')
    axes['X'].plot(Diad[:, 0], Diad[:, 1], '-', color='grey')
    axes['X'].plot(Diad_short_diad1[:, 0], Diad_short_diad1[:, 1], '-r', label='region_2_bck_sub')

    #axes['X'].plot(Baseline[:, 0], Baseline[:, 1], '-b', label='Bck points')
    axes['X'].plot(Baseline_diad1[:, 0], Baseline_diad1[:, 1], '.b', label='Bck points')
    axes['X'].plot(Diad_short_diad1[:, 0], Py_base_diad1, '-k')



    ax1_ymin=np.min(Baseline_diad1[:, 1])-10*np.std(Baseline_diad1[:, 1])
    ax1_ymax=np.max(Baseline_diad1[:, 1])+10*np.std(Baseline_diad1[:, 1])
    ax1_xmin=config1.lower_bck_diad1[0]-30
    ax1_xmax=config1.upper_bck_diad1[1]+30
    # Adding patches


    rect_diad1_b1=patches.Rectangle((config1.lower_bck_diad1[0], ax1_ymin),config1.lower_bck_diad1[1]-config1.lower_bck_diad1[0],ax1_ymax-ax1_ymin,
                            linewidth=1,edgecolor='none',facecolor='cyan', label='bck', alpha=0.3, zorder=0)
    axes['X'].add_patch(rect_diad1_b1)
    rect_diad1_b2=patches.Rectangle((config1.upper_bck_diad1[0], ax1_ymin),config1.upper_bck_diad1[1]-config1.upper_bck_diad1[0],ax1_ymax-ax1_ymin,
                            linewidth=1,edgecolor='none',facecolor='cyan', alpha=0.3, zorder=0)
    axes['X'].add_patch(rect_diad1_b2)
    axes['X'].set_xlim([ax1_xmin, ax1_xmax])
    axes['X'].set_ylim([ax1_ymin, ax1_ymax])

    axes['X'].set_ylabel('Intensity')
    axes['Y'].set_ylabel('Intensity')
    axes['Y'].set_xlabel('Wavenumber')
    axes['X'].legend()



    axes['Y'].set_title('b) Background subtracted spectra')
    axes['Y'].plot(x_diad1, y_corr_diad1, '-r')
    height_p=np.max(Diad_short_diad1[:, 1])-np.min(Diad_short_diad1[:, 1])
    axes['Y'].set_ylim([np.min(y_corr_diad1), 1.2*height_p ])
    axes['X'].set_xlabel('Wavenumber')



    axes['A'].plot(xdat, ydat,  '.k', label='data')
    axes['A'].plot( x_lin ,y_best_fit, '-g', linewidth=1, label='best fit')
    axes['A'].legend()
    axes['A'].set_ylabel('Intensity')
    axes['A'].set_xlabel('Wavenumber')
    axes['A'].set_xlim(ax1_xlim)
    axes['A'].set_title('c) Overall Best Fit')

   # individual fits
    axes['B'].plot(xdat, ydat, '.k')

    # This is for if there is more than 1 peak, this is when we want to plot the best fit
    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:
        if len(peak_pos_voigt)>1:
            axes['B'].plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='Peak2')

    axes['B'].plot(x_lin, components.get('lz1_'), '-b', linewidth=2, label='Peak1')
    if peak_pos_gauss is not None:
        axes['B'].plot(x_lin, components.get('bkg_'), '-m', label='Gaussian bck', linewidth=2)
    #ax2.plot(xdat, result.best_fit, '-g', label='best fit')
    axes['B'].legend()

    fitspan=max(y_best_fit)-min(y_best_fit)
    axes['B'].set_ylim([min(y_best_fit)-fitspan/5, max(y_best_fit)+fitspan/5])

    axes['B'].set_ylabel('Intensity')
    axes['B'].set_xlabel('Wavenumber')


    axes['B'].set_xlim(ax2_xlim)

    # Dashed lines so matches part D
    axes['B'].plot([df_out['Diad1_Voigt_Cent'], df_out['Diad1_Voigt_Cent']], [np.min(ydat), np.max(ydat)], ':b')
    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:
        if len(peak_pos_voigt)>=2:
            axes['B'].plot([df_out['HB1_Cent'], df_out['HB1_Cent']], [np.min(ydat), np.max(ydat)], ':r')

    axes['B'].set_title('c) Fit Components')

    # Background fit

    # First, set up x and y limits on axis

    if config1.x_range_baseline is not None:
        axc_xmin=df_out['Diad1_Voigt_Cent'][0]-config1.x_range_baseline
        axc_xmax=df_out['Diad1_Voigt_Cent'][0]+config1.x_range_baseline
    else:
        axc_xmin=config1.lower_bck_diad1[0]
        axc_xmax=config1.upper_bck_diad1[1]
    axc_ymin=np.min(Baseline_diad1[:, 1])-config1.y_range_baseline
    axc_ymax=np.max(Baseline_diad1[:, 1])+config1.y_range_baseline

    rect_diad1_b1=patches.Rectangle((config1.lower_bck_diad1[0],axc_ymin),config1.lower_bck_diad1[1]-config1.lower_bck_diad1[0],axc_ymax-axc_ymin,
                              linewidth=1,edgecolor='none',facecolor='cyan', label='bck', alpha=0.3, zorder=0)
    axes['C'].add_patch(rect_diad1_b1)
    rect_diad1_b2=patches.Rectangle((config1.upper_bck_diad1[0],axc_ymin),config1.upper_bck_diad1[1]-config1.upper_bck_diad1[0],axc_ymax-axc_ymin,
                              linewidth=1,edgecolor='none',facecolor='cyan', alpha=0.3, zorder=0)
    axes['C'].add_patch(rect_diad1_b2)


    axes['C'].set_title('d) Peaks overlain on data before subtraction')
    axes['C'].plot(Baseline_diad1[:, 0], Baseline_diad1[:, 1], '.b', label='bck')
    axes['C'].plot(Diad_short_diad1[:, 0], Py_base_diad1, '-k', label='Poly bck fit')
    axes['C'].plot(Diad_short_diad1[:, 0], Diad_short_diad1[:, 1], '.r', label='Data')

    axes['C'].set_ylabel('Intensity')
    axes['C'].set_xlabel('Wavenumber')


    if peak_pos_gauss is not None:

        axes['C'].plot(x_lin, components.get('bkg_')+ybase_xlin, '-m', label='Gaussian bck', linewidth=2)

    axes['C'].plot( x_lin ,y_best_fit+ybase_xlin, '-g', linewidth=2, label='Best Fit')
    axes['C'].plot(x_lin, components.get('lz1_')+ybase_xlin, '-b', label='Peak1', linewidth=1)
    if len(peak_pos_voigt)>1:
        axes['C'].plot(x_lin, components.get('lz2_')+ybase_xlin, '-r', label='Peak2', linewidth=1)




    axes['C'].legend()
    axes['C'].set_xlim([axc_xmin, axc_xmax])
    axes['C'].set_ylim([axc_ymin, axc_ymax])
    #axes['C'].plot(Diad_short[:, 0], Diad_short[:, 1], '"r', label='Data')


    # Residual on plot D
    axes['D'].set_title('f) Residuals')
    axes['D'].plot([df_out['Diad1_Voigt_Cent'], df_out['Diad1_Voigt_Cent']], [np.min(residual_diad1_coords), np.max(residual_diad1_coords)], ':b')
    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:
        if len(peak_pos_voigt)>=2:
            axes['D'].plot([df_out['HB1_Cent'], df_out['HB1_Cent']], [np.min(residual_diad1_coords), np.max(residual_diad1_coords)], ':r')

    axes['D'].plot(xdat_inrange, residual_diad1_coords, 'ok', mfc='c' )
    axes['D'].plot(xdat_inrange, residual_diad1_coords, '-c' )
    axes['D'].set_ylabel('Residual')
    axes['D'].set_xlabel('Wavenumber')
    # axes['D'].set_xlim(ax1_xlim)
    # axes['D'].set_xlim(ax2_xlim)
    Local_Residual_diad1=residual_diad1_coords[((xdat_inrange>(df_out['Diad1_Voigt_Cent'][0]-config1.x_range_residual))
                                            &(xdat_inrange<df_out['Diad1_Voigt_Cent'][0]+config1.x_range_residual))]
    axes['D'].set_xlim([df_out['Diad1_Voigt_Cent'][0]-config1.x_range_residual,
                df_out['Diad1_Voigt_Cent'][0]+config1.x_range_residual])
    #ax5.plot([cent_1117, cent_1117 ], [np.min(Local_Residual_1117)-10, np.max(Local_Residual_1117)+10], ':k')



    axes['D'].set_ylim([np.min(Local_Residual_diad1)-10, np.max(Local_Residual_diad1)+10])





    # Overal spectra
    axes['E'].set_title('g) Summary plot of raw spectra for file = ' + filename)
    axes['E'].plot(Spectra[:, 0], Spectra[:, 1], '-r')
    axes['E'].set_ylabel('Intensity')
    axes['E'].set_xlabel('Wavenumber')


    path3=path+'/'+'diad_fit_images'
    if os.path.exists(path3):
        out='path exists'
    else:
        os.makedirs(path+'/'+ 'diad_fit_images', exist_ok=False)

    fig.tight_layout()

    file=filename.rsplit('.txt', 1)[0]
    fig.savefig(path3+'/'+'Diad1_Fit_{}.png'.format(file), dpi=config1.dpi)

    if close_figure is True:
        plt.close(fig)

    if  config1.return_other_params is False:
        return df_out
    else:
        return df_out, result, y_best_fit, x_lin


def combine_diad_outputs(*, filename=None, prefix=True,
Diad1_fit=None, Diad2_fit=None, Carb_fit=None, to_csv=True,
to_clipboard=True, path=None):

    if prefix is True:
        filename=filename.split(' ')[1:][0]

    if Diad1_fit is not None and Diad2_fit is not None:

        combo=pd.concat([Diad1_fit, Diad2_fit], axis=1)
        # Fill any columns which dont' exist so can re-order the same every time
        if 'HB1_Cent' not in combo.columns:
            combo['HB1_Cent']=np.nan
            combo['HB1_Area']=0

        if 'HB2_Cent' not in combo.columns:
            combo['HB2_Cent']=np.nan
            combo['HB2_Area']=0

        if 'C13_Cent' not in combo.columns:
            combo['C13_Cent']=np.nan
            combo['C13_Area']=0

        combo['Splitting']=combo['Diad2_Voigt_Cent']-combo['Diad1_Voigt_Cent']
        cols_to_move = ['Splitting', 'Diad1_Combofit_Cent', 'Diad1_Combofit_Height', 'Diad1_Voigt_Cent', 'Diad1_Voigt_Area', 'Diad1_Voigt_Sigma', 'Diad1_Voigt_Gamma', 'Residual_Diad1', 'Diad2_Combofit_Cent', 'Diad2_Combofit_Height', 'Diad2_Voigt_Cent', 'Diad2_Voigt_Area', 'Diad2_Voigt_Sigma', 'Diad2_Voigt_Gamma', 'Residual_Diad2',
                    'HB1_Cent', 'HB1_Area', 'HB2_Cent', 'HB2_Area', 'C13_Cent', 'C13_Area']
        combo_f = combo[cols_to_move + [
                col for col in combo.columns if col not in cols_to_move]]
        combo_f=combo_f.iloc[:, 0:17]
        file=filename.rsplit('.txt', 1)[0]
        combo_f.insert(0, 'filename', file)

        if Carb_fit is None:
            if to_clipboard is True:
                combo_f.to_clipboard(excel=True, header=False, index=False)



        if Carb_fit is not None:
            combo_f=pd.concat([combo_f, Carb_fit], axis=1)
            # width=np.shape(combo_f)[1]
            # combo_f.insert(width, 'Carb_Cent',Carb_fit['Carb_Cent'])
            # combo_f.insert(width+1, 'Carb_Area',Carb_fit['Carb_Area'])
            # area_ratio=Carb_fit['Carb_Area']/(combo_f['Diad1_Voigt_Area']+combo_f['Diad2_Voigt_Area'])
            # combo_f.insert(width+2, 'Carb_Area/Diad_Area',  area_ratio)
            if to_clipboard is True:
                combo_f.to_clipboard(excel=True, header=False, index=False)


            return combo_f
    if Diad1_fit is None and Diad2_fit is None:
        df=pd.DataFrame(data={'filename': filename,
                            'Splitting': np.nan,
                                'Diad1_Voigt_Cent':np.nan,
                                    'Diad1_Voigt_Area':np.nan,
                                    'Residual_Diad1': np.nan,
                                        'Diad2_Voigt_Cent':np.nan,
                                        'Diad2_Voigt_Area':np.nan,
                                        'Residual_Diad2': np.nan,
                                            'HB1_Cent':np.nan,
                                                'HB1_Area':np.nan,
                                                'HB2_Cent':np.nan,
                                                    'HB2_Area':np.nan,
                                                        'C13_Cent': np.nan,
                                                        'C13_Area': np.nan,
                                                        })
        combo_f=pd.concat([df, Carb_fit], axis=1)
        if to_clipboard is True:
            df.to_clipboard(excel=True, header=False, index=False)
        combo_f=df

    if to_csv is True:
        if path is None:
            raise Exception('You need to specify path= for wherever you want this saved')
        path_fits=path+'/'+'Diad_Fits'
        if os.path.exists(path_fits):
            out='path exists'
        else:
            dir2=os.makedirs(path+'/'+ 'Diad_Fits', exist_ok=False)
        #filepath=Path(path+'/'+ 'Peak_Fits'+'/'+filename)
        combo_f.to_csv(path+'/'+'Diad_Fits'+'/'+'fits_'+filename)
    return combo_f


def plot_spectra(*,path=None, filename=None, filetype='Witec_ASCII'):
    """ Plots entire spectra, identifies some key phases


    Parameters
    -----------


    path: str
        Path to folder containing file

    filename: str
        filename with extension

    filetype: str
        One of the supported filetypes:
            headless_txt
            headless_csv
            HORIBA_txt
            Renishaw_txt


    Returns
    -----------
    figure showing spectra with major peak positions overlain


    """

    Spectra_df=get_data(path=path, filename=filename, filetype=filetype)

    Spectra=np.array(Spectra_df)


    fig, (ax1) = plt.subplots(1, 1, figsize=(5,4))

    miny=np.min(Spectra[:, 1])
    maxy=np.max(Spectra[:, 1])
    ax1.plot([1090, 1090], [miny, maxy+(maxy-miny)/10], '-.',color='salmon', label='Magnesite')
    ax1.plot([1131, 1131], [miny, maxy+(maxy-miny)/10], '-.k',  alpha=0.5, label='Anhydrite/Mg-Sulfate')
    #ax1.plot([1136, 1136], [miny, maxy], '-', color='grey', label='Mg-Sulfate')
    ax1.plot([1151, 1151], [miny, maxy+(maxy-miny)/10], '-.c', label='SO2')
    ax1.plot([1286, 1286], [miny, maxy+(maxy-miny)/10], '-k',  alpha=0.5,label='Diad1')
    ax1.plot([1389, 1389], [miny, maxy+(maxy-miny)/10], '-', color='darkgrey', alpha=0.5, label='Diad2')
    ax1.legend()
    ax1.plot(Spectra[:, 0], Spectra[:, 1], '-r', label='Spectra')
    ax1.set_xlabel('Wavenumber (cm$^{-1)}$')
    ax1.set_ylabel('Intensity')
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=2, mode="expand", borderaxespad=0.)
    ax1.set_ylim([miny, maxy+(maxy-miny)/10])
    #ax1.legend(bbox_to_anchor=(0, 1.05), loc='center', borderaxespad=0)



@dataclass
class carb_peak_config:
    # Selecting the two background positions
    lower_bck: Tuple[float, float]=(1060, 1065)
    upper_bck: Tuple[float, float]=(1120, 1130)

    # Background degree of polynomial
    N_poly_carb_bck: float =1
    # Seletcting the amplitude
    amplitude: float =1000

    # Selecting the peak center
    cent_carbonate: float =1090

    # outlier sigma to discard background at
    outlier_sigma: float =12

    # Number of peaks to look for in scipy find peaks
    N_peaks: float=3

    # Parameters for Scipy find peaks
    distance: float=10
    prominence: float=5
    width: float=6
    threshold: float=0.1
    height=100

    # Excluding a range for cosmic rays etc.
    exclude_range: Optional [Tuple[float, float]]=None

    # Plotting parameters
    dpi:float = 100
    plot_figure: bool = True

    # Return other parameteres e.g. intermediate outputs
    return_other_params: bool = False




def fit_carbonate_peak(*, config: carb_peak_config=carb_peak_config(),
path=None, filename=None, filetype=None,
fit_carbonate=None, plot_figure=True):

    """ This function fits a carbonate peak with a gaussian, and returns a plot

    fit_carbonate: bool
        If True, proceeds to use this function, if False, doesnt do anything else

    path: str
        Path to file

    filename: str
        filename with file extension

    filetype: str
        Filetype from one of the following options:
            Witec_ASCII
            headless_txt
            headless_csv
            HORIBA_txt
            Renishaw_txt

    lower_bck: list
        Lower range to fit background over (default [1030, 1050])

    upper_bck: list
        Upper range to fit background over (default [1140, 1200])

    N_poly: int
        Degree of polynomial to fit to background

    exclude_range: None or list
        Can select a range to exclude from fitting (e.g. a cosmic ray)

    cent: int or float
        Approximate peak center

    amplitude: int or float
        Approximate amplitude (helps the fitting algorithm converge)

    N_peaks: int
        number of peaks to try to find from scipy find peaks

    plot_figure: bool
        If true, plots figure and saves it in a subfolder in the same directory as the filepath specified

    dpi: int
        dpi to save figure at

    height, threshold, distance, prominence, width: int
        Values for Scipy find peaks that can be adjusted.

    return_other_params: bool
        if False (Default), returns just df of fit information
        if True, also returns:
            xx_carb: :linspace for plotting
            y_carb: Best fit to background-subtracted data
            result0: fit results from lmfit

    """

    if fit_carbonate is True:

        Spectra_in=get_data(path=path, filename=filename, filetype=filetype)



        # If exclude range, trim that here
        if config.exclude_range is not None:
             Spectra=Spectra_in[ (Spectra_in[:, 0]<config.exclude_range[0]) | (Spectra_in[:, 0]>config.exclude_range[1]) ]
        else:
            Spectra=Spectra_in



        lower_0baseline=config.lower_bck[0]
        upper_0baseline=config.lower_bck[1]
        lower_1baseline=config.upper_bck[0]
        upper_1baseline=config.upper_bck[1]

        # Filter out spectra outside these baselines
        Spectra_short=Spectra[ (Spectra[:,0]>lower_0baseline) & (Spectra[:,0]<upper_1baseline) ]

        # To make a nice plot, give 50 wavenumber units on either side as a buffer
        Spectra_plot=Spectra[ (Spectra[:,0]>lower_0baseline-50) & (Spectra[:,0]<upper_1baseline+50) ]

        # Find peaks using Scipy find peaks
        y=Spectra_plot[:, 1]
        x=Spectra_plot[:, 0]
        peaks = find_peaks(y,height = config.height, threshold = config.threshold,
        distance = config.distance, prominence=config.prominence, width=config.width)

        height = peaks[1]['peak_heights'] #list of the heights of the peaks
        peak_pos = x[peaks[0]] #list of the peaks positions
        df_sort=pd.DataFrame(data={'pos': peak_pos,
                            'height': height})

        df_peak_sort=df_sort.sort_values('height', axis=0, ascending=False)

        # Trim number of peaks based on user-defined N peaks
        df_peak_sort_short=df_peak_sort[0:config.N_peaks]


        # Get actual baseline
        Baseline_with_outl=Spectra_short[
        ((Spectra_short[:, 0]<upper_0baseline) &(Spectra_short[:, 0]>lower_0baseline))
            |
        ((Spectra_short[:, 0]<upper_1baseline) &(Spectra_short[:, 0]>lower_1baseline))]

        # Calculates the LH baseline
        LH_baseline=Spectra_short[
        ((Spectra_short[:, 0]<upper_0baseline) &(Spectra_short[:, 0]>lower_0baseline))]

        Mean_LH_baseline=np.nanmean(LH_baseline[:, 1])
        Std_LH_baseline=np.nanstd(LH_baseline[:, 1])

        # Calculates the RH baseline
        RH_baseline=Spectra_short[((Spectra_short[:, 0]<upper_1baseline)
        &(Spectra_short[:, 0]>lower_1baseline))]
        Mean_RH_baseline=np.nanmean(RH_baseline[:, 1])
        Std_RH_baseline=np.nanstd(RH_baseline[:, 1])

        # Removes points outside baseline

        LH_baseline_filt=LH_baseline[(LH_baseline[:, 1]<Mean_LH_baseline+config.outlier_sigma*Std_LH_baseline)
        &(LH_baseline[:, 1]>Mean_LH_baseline-config.outlier_sigma*Std_LH_baseline) ]

        RH_baseline_filt=RH_baseline[(RH_baseline[:, 1]<Mean_RH_baseline+config.outlier_sigma*Std_RH_baseline)
        &(RH_baseline[:, 1]>Mean_RH_baseline-config.outlier_sigma*Std_RH_baseline) ]

        Baseline=np.concatenate((LH_baseline_filt, RH_baseline_filt), axis=0)





        # Fits a polynomial to the baseline of degree
        Pf_baseline = np.poly1d(np.polyfit(Baseline[:, 0], Baseline[:, 1], config.N_poly_carb_bck))
        Py_base =Pf_baseline(Spectra_short[:, 0])

        Baseline_ysub=Pf_baseline(Baseline[:, 0])
        Baseline_x=Baseline[:, 0]
        y_corr= Spectra_short[:, 1]-  Py_base
        x=Spectra_short[:, 0]

        # NOw into the voigt fitting

        model0 = VoigtModel()#+ ConstantModel()

        # create parameters with initial values
        pars0 = model0.make_params()
        pars0['center'].set(config.cent_carbonate, min=config.cent_carbonate-30, max=config.cent_carbonate+30)
        pars0['amplitude'].set(config.amplitude, min=0)


        init0 = model0.eval(pars0, x=x)
        result0 = model0.fit(y_corr, pars0, x=x)
        Center_p0=result0.best_values.get('center')
        area_p0=result0.best_values.get('amplitude')



        # Make a nice linspace for plotting with smooth curves.
        xx_carb=np.linspace(min(x), max(x), 2000)
        y_carb=result0.eval(x=xx_carb)
        height=np.max(y_carb)

        # Plotting what its doing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
        fig.suptitle('Secondary Phase, file= '+ str(filename), fontsize=12, x=0.5, y=1.0)

        # Plot the peak positions and heights


        ax1.plot([1090, 1090], [min(Spectra_short[:, 1]), max(Spectra_short[:, 1])], ':r', label='Magnesite')

        ax1.set_title('Background fit')
        ax1.plot(Spectra_plot[:, 0], Spectra_plot[:, 1], '-r', label='Spectra')
        ax1.plot(RH_baseline_filt[:, 0], RH_baseline_filt[:, 1], '-b',
        lw=3,  label='bck points')
        ax1.plot(LH_baseline_filt[:, 0], LH_baseline_filt[:, 1], '-b',
        lw=3, label='_bck points')


        ax2.set_title('Bkg-subtracted, carbonate peak fit')

        ax2.plot(xx_carb, y_carb, '-k', label='Peak fit')

        ax2.plot(x, y_corr, 'ok', mfc='red', label='Bck-sub data')
        ax2.set_ylim([min(y_carb)-0.5*(max(y_carb)-min(y_carb)),
                    max(y_carb)+0.1*max(y_carb),
        ])

        ax1.plot(Spectra_short[:, 0], Py_base, '-k')

        ax1.plot(df_peak_sort_short['pos'], df_peak_sort_short['height'], '*k', mfc='yellow', label='SciPy Peaks')

        ax1.set_ylabel('Intensity')
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax2.set_ylabel('Bck-corrected Intensity')
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)')

        ax1.legend()
        ax2.legend()

        ax1.set_ylim([
        np.min(Spectra_plot[:, 1])-50,
        np.max(Spectra_plot[:, 1])+50
        ])



        if area_p0 is None:
            area_p0=np.nan
            if area_p0 is not None:
                if area_p0<0:
                    area_p0=np.nan



        if config.plot_figure is True:
            path3=path+'/'+'Carbonate_fit_images'
            if os.path.exists(path3):
                out='path exists'
            else:
                os.makedirs(path+'/'+ 'Carbonate_fit_images', exist_ok=False)

            file=filename.rsplit('.txt', 1)[0]
            fig.savefig(path3+'/'+'Carbonate_Fit_{}.png'.format(file), dpi=config.dpi)

        df=pd.DataFrame(data={'Carb_Cent': Center_p0,
        'Carb_Area': area_p0,
        'Carb_Height': height }, index=[0])



    if fit_carbonate is False:

        df=None
        xx_carb=None
        y_carb=None
        result0=None
    if config.return_other_params is True:
        return df, xx_carb, y_carb, result0
    else:
        return df



def keep_carbonate(Carb_fit, ):
    N=input("Type 'Y' if you want to save this carbonate peak, 'N' if you dont")

    if N == 'Y':
        print('Saving carbonate')
        return Carb_fit
    if N == 'N':
        Carb_fit=None
        print('Carbonate peak parameters set to None')
        return Carb_fit

# def keep_carbonate(Carb_fit):
#     N=input("Type 'Y' if you want to save this carbonate peak, 'N' if you dont")
#
#     if N == 'Y':
#         print('Saving carbonate')
#         return Carb_fit
#     if N == 'N':
#         Carb_fit=None
#         print('Carbonate peak parameters set to None')
#         return Carb_fit

def proceed_to_fit_diads(filename, Carb_fit, diads_present=False):

    if diads_present is True:
        print('Move on to fit diads')

    if diads_present is False:

        df=pd.DataFrame(data={'filename': filename,
                            'Splitting': np.nan,
                                'Diad1_Voigt_Cent':np.nan,
                                    'Diad1_Voigt_Area':np.nan,
                                        'Diad2_Voigt_Cent':np.nan,
                                        'Diad2_Voigt_Area':np.nan,
                                            'HB1_Cent':np.nan,
                                                'HB1_Area':np.nan,
                                                'HB2_Cent':np.nan,
                                                    'HB2_Area':np.nan,
                                                        'C13_Cent': np.nan,
                                                        'C13_Area': np.nan,
                                                        'Carb_Cent':Carb_fit['Carb_Cent'],
                                                        'Carb_Area': Carb_fit['Carb_Area'],
                                                        'Carb_Area/Diad_Area':np.nan
                                                        })


        if to_clipboard is True:
            df.to_clipboard(excel=True, header=False, index=False)
        print('Saved carbonate peak in diad format to clipboard')

## Fit generic peak

@dataclass
class generic_peak_config:

    # Name that gets stamped onto fits
    name: str= 'generic'
    # Selecting the two background positions
    lower_bck: Tuple[float, float]=(1060, 1065)
    upper_bck: Tuple[float, float]=(1120, 1130)

    # Background degree of polynomial
    N_poly_carb_bck: float =1
    # Seletcting the amplitude
    amplitude: float =1000

    # Selecting the peak center
    cent_generic: float =1090

    # outlier sigma to discard background at
    outlier_sigma: float =12

    # Number of peaks to look for in scipy find peaks
    N_peaks: float=3

    # Parameters for Scipy find peaks
    distance: float=10
    prominence: float=5
    width: float=6
    threshold: float=0.1
    height=100

    # Excluding a range for cosmic rays etc.
    exclude_range: Optional [Tuple[float, float]]=None

    # Plotting parameters
    dpi:float = 100
    plot_figure: bool = True

    # Return other parameteres e.g. intermediate outputs
    return_other_params: bool = False






def fit_generic_peak(*, config: generic_peak_config=generic_peak_config(),
path=None, filename=None, filetype=None,
 plot_figure=True):

    """ This function fits a generic peak with a gaussian, and returns a plot

    path: str
        Path to file

    filename: str
        filename with file extension

    filetype: str
        Filetype from one of the following options:
            Witec_ASCII
            headless_txt
            headless_csv
            HORIBA_txt
            Renishaw_txt

    lower_bck: list
        Lower range to fit background over (default [1030, 1050])

    upper_bck: list
        Upper range to fit background over (default [1140, 1200])

    N_poly: int
        Degree of polynomial to fit to background

    exclude_range: None or list
        Can select a range to exclude from fitting (e.g. a cosmic ray)

    cent: int or float
        Approximate peak center

    amplitude: int or float
        Approximate amplitude (helps the fitting algorithm converge)

    N_peaks: int
        number of peaks to try to find from scipy find peaks

    plot_figure: bool
        If true, plots figure and saves it in a subfolder in the same directory as the filepath specified

    dpi: int
        dpi to save figure at

    height, threshold, distance, prominence, width: int
        Values for Scipy find peaks that can be adjusted.

    return_other_params: bool
        if False (Default), returns just df of fit information
        if True, also returns:
            xx_generic: :linspace for plotting
            y_generic: Best fit to background-subtracted data
            result0: fit results from lmfit

    """


    Spectra_in=get_data(path=path, filename=filename, filetype=filetype)

    name=config.name

    # If exclude range, trim that here
    if config.exclude_range is not None:
            Spectra=Spectra_in[ (Spectra_in[:, 0]<config.exclude_range[0]) | (Spectra_in[:, 0]>config.exclude_range[1]) ]
    else:
        Spectra=Spectra_in



    lower_0baseline=config.lower_bck[0]
    upper_0baseline=config.lower_bck[1]
    lower_1baseline=config.upper_bck[0]
    upper_1baseline=config.upper_bck[1]

    # Filter out spectra outside these baselines
    Spectra_short=Spectra[ (Spectra[:,0]>lower_0baseline) & (Spectra[:,0]<upper_1baseline) ]

    # To make a nice plot, give 50 wavenumber units on either side as a buffer
    Spectra_plot=Spectra[ (Spectra[:,0]>lower_0baseline-50) & (Spectra[:,0]<upper_1baseline+50) ]

    # Find peaks using Scipy find peaks
    y=Spectra_plot[:, 1]
    x=Spectra_plot[:, 0]
    peaks = find_peaks(y,height = config.height, threshold = config.threshold,
    distance = config.distance, prominence=config.prominence, width=config.width)

    height = peaks[1]['peak_heights'] #list of the heights of the peaks
    peak_pos = x[peaks[0]] #list of the peaks positions
    df_sort=pd.DataFrame(data={'pos': peak_pos,
                        'height': height})

    df_peak_sort=df_sort.sort_values('height', axis=0, ascending=False)

    # Trim number of peaks based on user-defined N peaks
    df_peak_sort_short=df_peak_sort[0:config.N_peaks]


    # Get actual baseline
    Baseline_with_outl=Spectra_short[
    ((Spectra_short[:, 0]<upper_0baseline) &(Spectra_short[:, 0]>lower_0baseline))
        |
    ((Spectra_short[:, 0]<upper_1baseline) &(Spectra_short[:, 0]>lower_1baseline))]

    # Calculates the LH baseline
    LH_baseline=Spectra_short[
    ((Spectra_short[:, 0]<upper_0baseline) &(Spectra_short[:, 0]>lower_0baseline))]

    Mean_LH_baseline=np.nanmean(LH_baseline[:, 1])
    Std_LH_baseline=np.nanstd(LH_baseline[:, 1])

    # Calculates the RH baseline
    RH_baseline=Spectra_short[((Spectra_short[:, 0]<upper_1baseline)
    &(Spectra_short[:, 0]>lower_1baseline))]
    Mean_RH_baseline=np.nanmean(RH_baseline[:, 1])
    Std_RH_baseline=np.nanstd(RH_baseline[:, 1])

    # Removes points outside baseline

    LH_baseline_filt=LH_baseline[(LH_baseline[:, 1]<Mean_LH_baseline+config.outlier_sigma*Std_LH_baseline)
    &(LH_baseline[:, 1]>Mean_LH_baseline-config.outlier_sigma*Std_LH_baseline) ]

    RH_baseline_filt=RH_baseline[(RH_baseline[:, 1]<Mean_RH_baseline+config.outlier_sigma*Std_RH_baseline)
    &(RH_baseline[:, 1]>Mean_RH_baseline-config.outlier_sigma*Std_RH_baseline) ]

    Baseline=np.concatenate((LH_baseline_filt, RH_baseline_filt), axis=0)





    # Fits a polynomial to the baseline of degree
    Pf_baseline = np.poly1d(np.polyfit(Baseline[:, 0], Baseline[:, 1], config.N_poly_carb_bck))
    Py_base =Pf_baseline(Spectra_short[:, 0])

    Baseline_ysub=Pf_baseline(Baseline[:, 0])
    Baseline_x=Baseline[:, 0]
    y_corr= Spectra_short[:, 1]-  Py_base
    x=Spectra_short[:, 0]

    # NOw into the voigt fitting

    model0 = VoigtModel()#+ ConstantModel()

    # create parameters with initial values
    pars0 = model0.make_params()
    pars0['center'].set(config.cent_generic, min=config.cent_generic-30, max=config.cent_generic+30)
    pars0['amplitude'].set(config.amplitude, min=0)


    init0 = model0.eval(pars0, x=x)
    result0 = model0.fit(y_corr, pars0, x=x)
    Center_p0=result0.best_values.get('center')
    area_p0=result0.best_values.get('amplitude')



    # Make a nice linspace for plotting with smooth curves.
    xx_carb=np.linspace(min(x), max(x), 2000)
    y_carb=result0.eval(x=xx_carb)
    height=np.max(y_carb)

    # Plotting what its doing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle('Secondary Phase, file= '+ str(filename), fontsize=16, x=0.5, y=1.0)

    # Plot the peak positions and heights



    ax1.set_title('Background fit')
    ax1.plot(Spectra_plot[:, 0], Spectra_plot[:, 1], '-r', label='Spectra')
    ax1.plot(RH_baseline_filt[:, 0], RH_baseline_filt[:, 1], '-b',
    lw=3,  label='bck points')
    ax1.plot(LH_baseline_filt[:, 0], LH_baseline_filt[:, 1], '-b',
    lw=3, label='_bck points')


    ax2.set_title('Bkg-subtracted, ' + name + ' peak fit')

    ax2.plot(xx_carb, y_carb, '-k', label='Peak fit')

    ax2.plot(x, y_corr, 'ok', mfc='red', label='Bck-sub data')
    ax2.set_ylim([min(y_carb)-0.5*(max(y_carb)-min(y_carb)),
                max(y_carb)+0.1*max(y_carb),
    ])

    ax1.plot(Spectra_short[:, 0], Py_base, '-k')

    ax1.plot(df_peak_sort_short['pos'], df_peak_sort_short['height'], '*k', mfc='yellow', label='SciPy Peaks')

    ax1.set_ylabel('Intensity')
    ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax2.set_ylabel('Bck-corrected Intensity')
    ax2.set_xlabel('Wavenumber (cm$^{-1}$)')

    ax1.legend()
    ax2.legend()

    ax1.set_ylim([
    np.min(Spectra_plot[:, 1])-50,
    np.max(Spectra_plot[:, 1])+50
    ])



    if area_p0 is None:
        area_p0=np.nan
        if area_p0 is not None:
            if area_p0<0:
                area_p0=np.nan



    if plot_figure is True:
        path3=path+'/'+'Carbonate_fit_images'
        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ name + '_fit_images', exist_ok=False)

        file=filename.rsplit('.txt', 1)[0]
        fig.savefig(path3+'/'+ name +'_Fit_{}.png'.format(file), dpi=config.dpi)

    df=pd.DataFrame(data={'Peak_Cent_{}'.format(name): Center_p0,
    'Peak_Area_{}'.format(name): area_p0,
    'Peak_Height_{}'.format(name): height}, index=[0])


    if config.return_other_params is True:
        return df, xx_carb, y_carb, result0
    else:
        return df



