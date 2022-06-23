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

def plot_diad(*,path=None, filename=None, filetype='Witec_ASCII'):

    Spectra_df=get_data(path=path, filename=filename, filetype=filetype)

    Spectra=np.array(Spectra_df)


    fig, (ax1) = plt.subplots(1, 1, figsize=(7,4))

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

def identify_diad_peaks(*, path=None, filename, filetype='Witec_ASCII', n_peaks_diad1=2, n_peaks_diad2=3,
        exclude_range1=None, exclude_range2=None,
        height = 10, threshold = 0.6, distance = 1, prominence=10, width=2,
        plot_figure=True):
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
    if exclude_range1 is None and exclude_range2 is None:
        Discard_str=False
    else:
        Discard_str=True
        if exclude_range1 is not None and exclude_range2 is None:
            Diad_old=Diad.copy()
            Diad=Diad[(Diad[:, 0]<exclude_range1[0])|(Diad[:, 0]>exclude_range1[1])]
            Discard=Diad_old[(Diad_old[:, 0]>=exclude_range1[0]) & (Diad_old[:, 0]<=exclude_range1[1])]

        # NEED TO FIX
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


    y=Diad[:, 1]
    x=Diad[:, 0]
    peaks = find_peaks(y,height = height, threshold = threshold, distance = distance, prominence=prominence, width=width)

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

    if any(df_sort_diad2_trim['pos'].between(1385, 1391)):
        diad_2_peaks=tuple(df_sort_diad2_trim['pos'].values)
    else:
        if n_peaks_diad2==1:
            print('WARNING: Couldnt find diad2, ive guesed a peak position of 1389.1 to move forwards')
            diad_2_peaks=np.array([1389.1])
        if n_peaks_diad2==2:
            print('WARNING: Couldnt find diad2, ive guesed a peak position of 1389.1 and 1410')
            diad_2_peaks=np.array([1389.1, 1410])
        if n_peaks_diad2==3:
            raise TypeError('WARNING: Couldnt find diad2, and you specified 3 peaks, try adjusting the Scipy peak parameters')

    if any(df_sort_diad1_trim['pos'].between(1280, 1290)):
        diad_1_peaks=tuple(df_sort_diad1_trim['pos'].values)
    else:
        print('Couldnt find diad1, set peak guess to 1286.1')
        diad_1_peaks=np.array([1286.1])



    print('Initial estimates: Diad1+HB=' +str(np.round(diad_1_peaks, 1)) + ', Diad2+HB=' + str(np.round(diad_2_peaks, 1)))


    if plot_figure is True:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12,4))
        ax0.plot(Diad[:, 0], Diad[:, 1], '-r')
        if Discard_str is not False:
            ax0.plot(Discard[:, 0], Discard[:, 1], '.c', label='Discarded')
            ax1.plot(Discard[:, 0], Discard[:, 1], '.c', label='Discarded')
            ax2.plot(Discard[:, 0], Discard[:, 1], '.c', label='Discarded')

        ax0.plot([1286, 1286],
        [min(Diad[:, 1]), max(Diad[:, 1])], ':k', label='Approx. D1 pos')
        ax0.plot([1389, 1389],
        [min(Diad[:, 1]), max(Diad[:, 1])], ':k', label='approx D2 pos')
        ax1.plot([1286, 1286],
        [min(Diad[:, 1]), max(Diad[:, 1])], ':k', label='approx D1 pos')
        ax1.plot([1389, 1389],
        [min(Diad[:, 1]), max(Diad[:, 1])], ':k', label='approx D2 pos')
        ax2.plot([1286, 1286],
        [min(Diad[:, 1]), max(Diad[:, 1])], ':k', label='approx D1 pos')
        ax2.plot([1389, 1389],
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
        ax2.plot(df_sort_diad2_trim['pos'], df_sort_diad2_trim['height'], '*k')
        ax1.plot(df_sort_diad1_trim['pos'], df_sort_diad1_trim['height'], '*k')
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
            exclude_range1=None, exclude_range2=None,N_poly=1,
            lower_range=[1200, 1250], upper_range=[1320, 1330], sigma=4,
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

    lower_range: list len 2
        wavenumbers of LH baseline region

    upper_range: list len 2
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


    lower_0baseline=lower_range[0]
    upper_0baseline=lower_range[1]
    lower_1baseline=upper_range[0]
    upper_1baseline=upper_range[1]
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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        ax1.set_title('Background fit')
        ax1.plot(Diad_short[:, 0], Diad_short[:, 1], '-r')

        ax1.plot(Baseline[:, 0], Baseline[:, 1], '.b')
        ax1.plot(Diad_short[:, 0], Py_base, '-k')

        ax1.set_ylim([
        np.min(Baseline[:, 1])-10*np.std(Baseline[:, 1]),
        np.max(Baseline[:, 1])+10*np.std(Baseline[:, 1])
        ] )
        ax1.set_ylabel('Intensity')
        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Wavenumber')



        ax2.set_title('Background corrected')
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
                                amplitude=100,
                                span=None,
                                plot_figure=True, dpi=200):

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

    peak_pos_gauss: None, int, or flota
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
            pars1['lz1_'+ 'amplitude'].set(amplitude)
            pars1['lz1_'+ 'center'].set(peak_pos_voigt)
            params=pars1
            # Sometimes length 1 can be with a comma
        else:
            #  If peak find function put out a tuple length 1
            if len(peak_pos_voigt)==1:
                model_F = VoigtModel(prefix='lz1_') #+ ConstantModel(prefix='c1')
                pars1 = model_F.make_params()
                pars1['lz1_'+ 'amplitude'].set(amplitude)
                pars1['lz1_'+ 'center'].set(peak_pos_voigt[0])
                params=pars1

            if len(peak_pos_voigt)==2:

                # Code from 1447
                model_prel = VoigtModel(prefix='lzp_') #+ ConstantModel(prefix='c1')
                pars2 = model_prel.make_params()
                pars2['lzp_'+ 'amplitude'].set(amplitude, min=0)
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
                pars[prefix + 'amplitude'].set(amplitude/5, min=0, max=Peakp_Area/5)


                model_F=model1+peak
                pars1.update(pars)
                params=pars1





    if peak_pos_gauss is not None:

        model = GaussianModel(prefix='bkg_')
        params = model.make_params()
        params['bkg_'+'amplitude'].set(gauss_amp, min=gauss_amp/10, max=gauss_amp*10)
        params['bkg_'+'sigma'].set(gauss_sigma, min=gauss_sigma/10, max=gauss_sigma*10)
        params['bkg_'+'center'].set(peak_pos_gauss, min=peak_pos_gauss-30, max=peak_pos_gauss+30)






        rough_peak_positions = peak_pos_voigt
        # If you want a Gaussian background
        if type(peak_pos_voigt) is float or type(peak_pos_voigt) is np.float64 or type(peak_pos_voigt) is int:
                peak, pars = add_peak(prefix='lz1_', center=peak_pos_voigt, amplitude=amplitude)
                model = peak+model
                params.update(pars)
        else:
            if len(peak_pos_voigt)==1:

                peak, pars = add_peak(prefix='lz1_', center=peak_pos_voigt, min_cent=peak_pos_voigt-5, max_cent=peak_pos_voigt+5)
                model = peak+model
                params.update(pars)





            if len(peak_pos_voigt)>1:
                for i, cen in enumerate(rough_peak_positions):

                    peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen, amplitude=amplitude)
                    model = peak+model
                    params.update(pars)
                if type(peak_pos_voigt) is float or type(peak_pos_voigt) is np.float64 or type(peak_pos_voigt) is int:
                    peak, pars = add_peak(prefix='lz1_', center=cen, amplitude=amplitude)
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
    df_out=pd.DataFrame(data={'Diad1_Cent': Peak1_Cent,
                            'Diad1_Area': Peak1_Int
    }, index=[0])


    if peak_pos_gauss is not None:
        Gauss_cent=result.best_values.get('bkg_center')
        Gauss_amp=result.best_values.get('bkg_amplitude')
        Gauss_sigma=result.best_values.get('bkg_sigma')
        print('Gauss_cent='+str(Gauss_cent))
        print('Gauss_amp='+str(Gauss_amp))
        print('Gauss_sigma='+str(Gauss_sigma))



    x_lin=np.linspace(span[0], span[1], 2000)
    y_best_fit=result.eval(x=x_lin)
    components=result.eval_components(x=x_lin)


        # Uncommnet to get full report
    if print is True:
        print(result.fit_report(min_correl=0.5))

    # Checing for error bars
    Error_bars=result.errorbars
    if Error_bars is False:
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

        path3=path+'/'+'Peak_fit_images'
        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ 'Peak_fit_images', exist_ok=False)


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
                    amplitude=100, peak_pos_gauss=(1400), gauss_sigma=None, gauss_amp=100, span=None, plot_figure=True, dpi=200):


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

    init_ini = model_ini.eval(params_ini, x=xdat)


    result_ini  = model_ini.fit(ydat, params_ini, x=xdat)
    comps_ini  = result_ini.eval_components()
    Center_ini=result_ini.best_values.get('center')
    Amplitude_ini=result_ini.params.get('amplitude')
    sigma_ini=result_ini.params.get('sigma')
    fwhm_ini=result_ini.params.get('fwhm')
    print(Center_ini)
    print(sigma_ini)



    if peak_pos_gauss is None:
        # Fit just as many peaks as there are peak_pos_voigt

        if type(peak_pos_voigt) is float or type(peak_pos_voigt) is int or type(peak_pos_voigt) is np.float64:
            model_F = VoigtModel(prefix='lz1_')#+ ConstantModel(prefix='c1')
            pars1 = model_F.make_params()
            pars1['lz1_'+ 'amplitude'].set(amplitude, min=0)
            pars1['lz1_'+ 'center'].set(peak_pos_voigt)
            params=pars1

        else:
            if len(peak_pos_voigt)==1:
                model_F = VoigtModel(prefix='lz1_')#+ ConstantModel(prefix='c1')
                pars1 = model_F.make_params()
                pars1['lz1_'+ 'amplitude'].set(amplitude, min=0)
                pars1['lz1_'+ 'center'].set(peak_pos_voigt[0])
                params=pars1

            if len(peak_pos_voigt)==2:
                # Do a prelim fit
                # model_prel = VoigtModel(prefix='lzp_')#+ ConstantModel(prefix='c1')
                # pars2 = model_prel.make_params()
                # pars2['lzp_'+ 'amplitude'].set(amplitude, min=0, max=Amplitude_ini)
                # pars2['lzp_'+ 'center'].set(peak_pos_voigt[0])
                # pars2['lzp_'+ 'sigma'].set(sigma_ini, min=sigma_ini/5, max=sigma_ini*5)
                #
                #
                # init = model_prel.eval(pars2, x=xdat)
                # result_prel = model_prel.fit(ydat, pars2, x=xdat)
                # comps_prel = result_prel.eval_components()

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
                pars[prefix + 'amplitude'].set(amplitude/5, min=Peakp_Area/100, max=Peakp_Area/5)
                pars[prefix+ 'fwhm'].set(Peakp_HW, min=Peakp_HW/10, max=Peakp_HW*2)

                model_F=model1+peak
                pars1.update(pars)
                params=pars1

            if len(peak_pos_voigt)==3:
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
        params['bkg_'+'center'].set(peak_pos_gauss, min=peak_pos_gauss-30, max=peak_pos_gauss+30)




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
                print('Fitting 2 voigt peaks iteratively ')
                for i, cen in enumerate(peak_pos_voigt):
                    print('working on voigt peak' + str(i))
                    #peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen)
                    peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen,
                    min_cent=cen-3, max_cent=cen+3, sigma=sigma_ini, max_sigma=sigma_ini*5)


                    model = peak+model
                    params.update(pars)

            if len(peak_pos_voigt)==3:
                print('Fitting 2 peaks iteratively, then adding C13')
                for i, cen in enumerate(peak_pos_voigt):
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
        print('Error bars not determined by function')

    # Get first peak center
    Peak1_Cent=result.best_values.get('lz1_center')
    Peak1_Int=result.best_values.get('lz1_amplitude')

    if peak_pos_gauss is not None:
        Gauss_cent=result.best_values.get('bkg_center')
        Gauss_amp=result.best_values.get('bkg_amplitude')
        Gauss_sigma=result.best_values.get('bkg_sigma')
        print('Gauss_cent='+str(Gauss_cent))
        print('Gauss_amp='+str(Gauss_amp))
        print('Gauss_sigma='+str(Gauss_sigma))


    Peak1_Int=result.best_values.get('lz1_amplitude')
    # print('fwhm gauss')
    # print(result.best_values)


    x_lin=np.linspace(span[0], span[1], 2000)
    y_best_fit=result.eval(x=x_lin)
    components=result.eval_components(x=x_lin)



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
        df_out=pd.DataFrame(data={'Diad2_Cent': Peak1_Cent,
                                'Diad2_Area': Peak1_Int
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
            print(Peaks)
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


        print('made df')
        df_out=pd.DataFrame(data={'Diad2_Cent': Diad2_Cent,
                                'Diad2_Area': Diad2_Int
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


    residual_diad2=np.sum(((ydat_inrange-result_diad2_origx)**2)**0.5)/(len(ydat_inrange))
    df_out['Residual_Diad2']=residual_diad2



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

        path3=path+'/'+'Peak_fit_images'
        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ 'Peak_fit_images', exist_ok=False)


        file=filename.rsplit('.txt', 1)[0]
        fig.savefig(path3+'/'+'Diad2_Fit_{}.png'.format(file), dpi=dpi)




    best_fit=result.best_fit
    return result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim, peak_pos_gauss, residual_diad2_coords, ydat_inrange,  xdat_inrange

## Overall function for fitting diads in 1 single step
def fit_diad_2_w_bck(*, path=None, filename=None, filetype='headless_txt',
exclude_range1=None, exclude_range2=None, amplitude=100,
N_poly_bck_diad2=2, lower_baseline_diad2=[1320, 1350], upper_baseline_diad2=[1440, 1500],
peak_pos_voigt=(1369, 1387, 1408),peak_pos_gauss=(1380), gauss_sigma=1,  gauss_amp=3000, plot_figure=True, dpi=200, return_other_params=False):

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

    lower_baseline_diad2: list len 2
        wavenumbers of LH baseline region

    upper_baseline_diad2: list len 2
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


    #Fit baseline

    #Fit baseline

    y_corr_diad2, Py_base_diad2, x_diad2,  Diad_short, Py_base_diad2, Pf_baseline, Baseline_ysub_diad2, Baseline_x_diad2, Baseline, span=remove_diad_baseline(
   path=path, filename=filename, filetype=filetype, exclude_range1=exclude_range1, exclude_range2=exclude_range2, N_poly=N_poly_bck_diad2,
    lower_range=lower_baseline_diad2, upper_range=upper_baseline_diad2, plot_figure=False)


    # Fit voigt (+-) gaussian to data

    result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim, peak_pos_gauss,residual_diad2_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_diad2(path=path, filename=filename,
                    xdat=x_diad2, ydat=y_corr_diad2, amplitude=amplitude,
                    peak_pos_voigt=peak_pos_voigt,
                    peak_pos_gauss=peak_pos_gauss, gauss_sigma=gauss_sigma,
                    span=span, plot_figure=False)


    # Get diad data to plot
    Spectra_df=get_data(path=path, filename=filename, filetype=filetype)
    Spectra=np.array(Spectra_df)


    # Make nice figure

    figure_mosaic="""
    AB
    CD
    EE
    """
    fig,axes=plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(12, 16))

    # Overall best fit

    axes['A'].plot( x_lin ,y_best_fit, '-g', linewidth=1)
    axes['A'].plot(xdat, ydat,  '.k', label='data')
    axes['A'].legend()
    axes['A'].set_ylabel('Intensity')
    axes['A'].set_xlabel('Wavenumber')
    axes['A'].set_xlim(ax1_xlim)
    axes['A'].set_title('a) Overall Best Fit')


    # Split into separate components

    axes['B'].plot(xdat, ydat, '.k', label='data')
    if peak_pos_gauss is not None:
        axes['B'].plot(x_lin, components.get('bkg_'), '-m', label='Gaussian bck', linewidth=1)

    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:

        if len(peak_pos_voigt)==2:
            axes['B'].plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='Peak2')
        if len(peak_pos_voigt)>2:
            axes['B'].plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='Peak2')
            axes['B'].plot(x_lin, components.get('lz3_'), '-', color='yellow', linewidth=2, label='Peak3')
    axes['B'].plot(x_lin, components.get('lz1_'), '-b', linewidth=2, label='Peak1')


    #ax2.plot(xdat, result.best_fit, '-g', label='best fit')
    axes['B'].legend()
    fitspan=max(y_best_fit)-min(y_best_fit)
    axes['B'].set_ylim([min(y_best_fit)-fitspan/5, max(y_best_fit)+fitspan/5])
    axes['B'].set_ylabel('Intensity')
    axes['B'].set_xlabel('Wavenumber')
    axes['B'].set_title('b) Fit Components')
    axes['B'].set_xlim(ax2_xlim)

    # Fits, but not background subtracted to visualize background

    #Background fit on plot C
    axes['C'].set_title('c) Background fit')
    axes['C'].plot(Diad_short[:, 0], Diad_short[:, 1], '.c', label='Data')
    axes['C'].plot(Baseline[:, 0], Baseline[:, 1], '.b', label='bck')
    axes['C'].plot(Diad_short[:, 0], Py_base_diad2, '-k', label='Poly bck fit')

    axes['C'].set_ylim([
    np.min(Baseline[:, 1])-10*np.std(Baseline[:, 1]),
    np.max(Baseline[:, 1])+10*np.std(Baseline[:, 1])
    ] )
    axes['C'].set_ylabel('Intensity')
    axes['C'].set_xlabel('Wavenumber')
    ybase_xlin=Pf_baseline(x_lin)
    if peak_pos_gauss is not None:

        axes['C'].plot(x_lin, components.get('bkg_')+ybase_xlin, '-m', label='Gaussian bck', linewidth=2)

    axes['C'].plot( x_lin ,y_best_fit+ybase_xlin, '-g', linewidth=2, label='best fit')
    axes['C'].plot(x_lin, components.get('lz1_')+ybase_xlin, '-b', label='Peak1', linewidth=1)
    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:


        if len(peak_pos_voigt)>1:
            axes['C'].plot(x_lin, components.get('lz2_')+ybase_xlin, '-r', label='Peak2', linewidth=1)
        if len(peak_pos_voigt)>2:
            axes['C'].plot(x_lin, components.get('lz3_')+ybase_xlin, ':', color='yellow', linewidth=1, label='Peak3')

    axes['C'].set_title('c) Background fit')



    axes['C'].legend()

    # Residual on plot D
    axes['D'].set_title('d) Residual')


    axes['D'].plot([df_out['Diad2_Cent'], df_out['Diad2_Cent']], [np.min(residual_diad2_coords), np.max(residual_diad2_coords)], '-b')
    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:
        if len(peak_pos_voigt)>=2:
            axes['D'].plot([df_out['HB2_Cent'], df_out['HB2_Cent']], [np.min(residual_diad2_coords), np.max(residual_diad2_coords)], '-r')
        if len(peak_pos_voigt)==3:
            axes['D'].plot([df_out['C13_Cent'], df_out['C13_Cent']], [np.min(residual_diad2_coords), np.max(residual_diad2_coords)], '-', color='yellow')


    axes['D'].plot(xdat_inrange, residual_diad2_coords, 'ok', mfc='c' )
    axes['D'].plot(xdat_inrange, residual_diad2_coords, '-c' )
    axes['D'].set_ylabel('Residual')
    axes['D'].set_xlabel('Wavenumber')
    axes['D'].set_xlim(ax1_xlim)
    axes['D'].set_xlim(ax2_xlim)
    #axes['D'].set_ylim([np.min(residual_diad2_coords), 100*np.max(residual_diad2_coords)])

    axes['D'].plot([df_out['Diad2_Cent'], df_out['Diad2_Cent']], [np.min(residual_diad2_coords), np.max(residual_diad2_coords)], ':b')
    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:
        if len(peak_pos_voigt)>=2:
            axes['D'].plot([df_out['HB2_Cent'], df_out['HB2_Cent']], [np.min(residual_diad2_coords), np.max(residual_diad2_coords)], ':r')
    #axes['D'].set_xlim(ax1_xlim)






    # Overal spectra
    axes['E'].set_title(filename)
    axes['E'].plot(Spectra[:, 0], Spectra[:, 1], '-r')
    axes['E'].set_ylabel('Intensity')
    axes['E'].set_xlabel('Wavenumber')


    path3=path+'/'+'Peak_fit_images'
    fig.tight_layout()


    if os.path.exists(path3):
        out='path exists'
    else:
        os.makedirs(path+'/'+ 'Peak_fit_images', exist_ok=False)


    print(path)
    file=filename.rsplit('.txt', 1)[0]
    fig.savefig(path3+'/'+'Diad2_Fit_{}.png'.format(file), dpi=dpi)



    if return_other_params is True:
        return df_out, result, y_best_fit, x_lin
    else:
        return df_out


def fit_diad_1_w_bck(*, path=None, filename=None, filetype='headless_txt',
exclude_range1=None, exclude_range2=None,
N_poly_bck_diad1=2, lower_baseline_diad1=[1170, 1220],
upper_baseline_diad1=[1330, 1350], peak_pos_voigt=(1263, 1283),
peak_pos_gauss=(1270), amplitude=100, gauss_sigma=1,  gauss_amp=3000, plot_figure=True, dpi=200, return_other_params=False):
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

    lower_baseline_diad1: list len 2
        wavenumbers of LH baseline region

    upper_baseline_diad1: list len 2
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
    # First, we feed data into the remove baseline function, which returns corrected data

    y_corr_diad1, Py_base_diad1, x_diad1,  Diad_short, Py_base_diad1, Pf_baseline,  Baseline_ysub_diad1, Baseline_x_diad1, Baseline, span=remove_diad_baseline(
   path=path, filename=filename, filetype=filetype, exclude_range1=exclude_range1, exclude_range2=exclude_range2, N_poly=N_poly_bck_diad1,
    lower_range=lower_baseline_diad1, upper_range=upper_baseline_diad1, plot_figure=False)





    # Then, we feed this baseline-corrected data into the combined gaussian-voigt peak fitting function
    result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim, peak_pos_gauss,residual_diad1_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_diad1(path=path, filename=filename,
                    xdat=x_diad1, ydat=y_corr_diad1, amplitude=amplitude,
                    peak_pos_voigt=peak_pos_voigt,
                    peak_pos_gauss=peak_pos_gauss,
                    gauss_sigma=1,  gauss_amp=3000,
                    span=span, plot_figure=False)

    # get a best fit to the baseline using a linspace from the peak fitting
    ybase_xlin=Pf_baseline(x_lin)

    # We extract the full spectra to plot at the end, and convert to a dataframe
    Spectra_df=get_data(path=path, filename=filename, filetype=filetype)
    Spectra=np.array(Spectra_df)



    # Make nice figure

    figure_mosaic="""
    AB
    CD
    EE
    """

    fig,axes=plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(12, 16))

    # Plot best fit on the LHS, and individual fits on the RHS at the top
    axes['A'].plot( x_lin ,y_best_fit, '-g', linewidth=1, label='best fit')
    axes['A'].plot(xdat, ydat,  '.k', label='data')
    axes['A'].legend()
    axes['A'].set_ylabel('Intensity')
    axes['A'].set_xlabel('Wavenumber')
    axes['A'].set_xlim(ax1_xlim)
    axes['A'].set_title('a) Overall Best Fit')

   # individual fits
    axes['B'].plot(xdat, ydat, '.k')

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
    axes['B'].plot([df_out['Diad1_Cent'], df_out['Diad1_Cent']], [np.min(ydat), np.max(ydat)], ':b')
    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:
        if len(peak_pos_voigt)>=2:
            axes['B'].plot([df_out['HB1_Cent'], df_out['HB1_Cent']], [np.min(ydat), np.max(ydat)], ':r')

    axes['B'].set_title('b) Fit Components')

    # Background fit
    axes['C'].set_title('c) Background fit')
    axes['C'].plot(Diad_short[:, 0], Diad_short[:, 1], '.c', label='Data')
    axes['C'].plot(Baseline[:, 0], Baseline[:, 1], '.b', label='bck')
    axes['C'].plot(Diad_short[:, 0], Py_base_diad1, '-k', label='Poly bck fit')

    axes['C'].set_ylim([
    np.min(Baseline[:, 1])-10*np.std(Baseline[:, 1]),
    np.max(Baseline[:, 1])+10*np.std(Baseline[:, 1])
    ] )
    axes['C'].set_ylabel('Intensity')
    axes['C'].set_xlabel('Wavenumber')


    if peak_pos_gauss is not None:

        axes['C'].plot(x_lin, components.get('bkg_')+ybase_xlin, '-m', label='Gaussian bck', linewidth=2)

    axes['C'].plot( x_lin ,y_best_fit+ybase_xlin, '-g', linewidth=2, label='Best Fit')
    axes['C'].plot(x_lin, components.get('lz1_')+ybase_xlin, '-b', label='Peak1', linewidth=1)
    if len(peak_pos_voigt)>1:
        axes['C'].plot(x_lin, components.get('lz2_')+ybase_xlin, '-r', label='Peak2', linewidth=1)




    axes['C'].legend()


    # Residual on plot D
    axes['D'].set_title('d) Residual')
    axes['D'].plot([df_out['Diad1_Cent'], df_out['Diad1_Cent']], [np.min(residual_diad1_coords), np.max(residual_diad1_coords)], ':b')
    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not np.float64 and type(peak_pos_voigt) is not int:
        if len(peak_pos_voigt)>=2:
            axes['D'].plot([df_out['HB1_Cent'], df_out['HB1_Cent']], [np.min(residual_diad1_coords), np.max(residual_diad1_coords)], ':r')

    axes['D'].plot(xdat_inrange, residual_diad1_coords, 'ok', mfc='c' )
    axes['D'].plot(xdat_inrange, residual_diad1_coords, '-c' )
    axes['D'].set_ylabel('Residual')
    axes['D'].set_xlabel('Wavenumber')
    axes['D'].set_xlim(ax1_xlim)
    axes['D'].set_xlim(ax2_xlim)





    # Overal spectra
    axes['E'].set_title(filename)
    axes['E'].plot(Spectra[:, 0], Spectra[:, 1], '-r')
    axes['E'].set_ylabel('Intensity')
    axes['E'].set_xlabel('Wavenumber')


    path3=path+'/'+'Peak_fit_images'
    if os.path.exists(path3):
        out='path exists'
    else:
        os.makedirs(path+'/'+ 'Peak_fit_images', exist_ok=False)

    fig.tight_layout()

    file=filename.rsplit('.txt', 1)[0]
    fig.savefig(path3+'/'+'Diad1_Fit_{}.png'.format(file), dpi=dpi)



    if  return_other_params is False:
        return df_out
    else:
        return df_out, result, y_best_fit, x_lin


def combine_diad_outputs(*, filename=None, prefix=True, Diad1_fit=None, Diad2_fit=None, Carb_fit=None):

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

        combo['Splitting']=combo['Diad2_Cent']-combo['Diad1_Cent']
        cols_to_move = ['Splitting', 'Diad1_Cent', 'Diad1_Area', 'Residual_Diad1', 'Diad2_Cent', 'Diad2_Area', 'Residual_Diad2',
                    'HB1_Cent', 'HB1_Area', 'HB2_Cent', 'HB2_Area', 'C13_Cent', 'C13_Area']
        combo_f = combo[cols_to_move + [
                col for col in combo.columns if col not in cols_to_move]]
        combo_f=combo_f.iloc[:, 0:13]
        file=filename.rsplit('.txt', 1)[0]
        combo_f.insert(0, 'filename', file)

        if Carb_fit is None:
            combo_f.to_clipboard(excel=True, header=False, index=False)

            return combo_f

        if Carb_fit is not None:
            width=np.shape(combo_f)[1]
            combo_f.insert(width, 'Carb_Cent',Carb_fit['Carb_Cent'])
            combo_f.insert(width+1, 'Carb_Area',Carb_fit['Carb_Area'])
            area_ratio=Carb_fit['Carb_Area']/(combo_f['Diad1_Area']+combo_f['Diad2_Area'])
            combo_f.insert(width+2, 'Carb_Area/Diad_Area',  area_ratio)

            combo_f.to_clipboard(excel=True, header=False, index=False)


            return combo_f
    if Diad1_fit is None and Diad2_fit is None:
        df=pd.DataFrame(data={'filename': filename,
                            'Splitting': np.nan,
                                'Diad1_Cent':np.nan,
                                    'Diad1_Area':np.nan,
                                    'Residual_Diad1': np.nan,
                                        'Diad2_Cent':np.nan,
                                        'Diad2_Area':np.nan,
                                        'Residual_Diad2': np.nan,
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
        df.to_clipboard(excel=True, header=False, index=False)
        return df


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


def fit_carbonate_peak(*,path=None, filename=None, filetype='Witec_ASCII',
lower_range=[1030, 1050], upper_range=[1140, 1200], amplitude=1000,
exclude_range=None,
N_poly=2, outlier_sigma=12, cent=1090, plot_figure=True, dpi=100,
height = 20, threshold = 1, distance = 5, prominence=1, width=3,
N_peaks=2, fit_carbonate=True, return_other_params=False):

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

    lower_range: list
        Lower range to fit background over (default [1030, 1050])

    upper_range: list
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

        # read file
        if filetype == 'headless_txt' or filetype == 'headless_csv':
            Spectra_df=pd.read_csv(path+'/'+filename, sep="\t", header=None )

        elif filetype=='Witec_ASCII':
            Spectra_df=read_witec_to_df(path=path, filename=filename)

        elif filetype=='Renishaw_txt':
            Spectra_df_long=pd.read_csv(path+'/'+filename, sep="\t" )
            Spectra_df=Spectra_df_long.iloc[:, 0:2]

        elif filetype=='HORIBA_txt':
            Spectra_df=read_HORIBA_to_df(path=path, filename=filename)

        else:
            raise TypeError('Filetype not recognised')

        Spectra_in=np.array(Spectra_df)

        # If exclude range, trim that here
        if exclude_range is not None:
             Spectra=Spectra_in[ (Spectra_in[:, 0]<exclude_range[0]) | (Spectra_in[:, 0]>exclude_range[1]) ]
        else:
            Spectra=Spectra_in



        lower_0baseline=lower_range[0]
        upper_0baseline=lower_range[1]
        lower_1baseline=upper_range[0]
        upper_1baseline=upper_range[1]

        # Filter out spectra outside these baselines
        Spectra_short=Spectra[ (Spectra[:,0]>lower_0baseline) & (Spectra[:,0]<upper_1baseline) ]

        # To make a nice plot, give 50 wavenumber units on either side as a buffer
        Spectra_plot=Spectra[ (Spectra[:,0]>lower_0baseline-50) & (Spectra[:,0]<upper_1baseline+50) ]

        # Find peaks using Scipy find peaks
        y=Spectra_plot[:, 1]
        x=Spectra_plot[:, 0]
        peaks = find_peaks(y,height = height, threshold = threshold,
        distance = distance, prominence=prominence, width=width)

        height = peaks[1]['peak_heights'] #list of the heights of the peaks
        peak_pos = x[peaks[0]] #list of the peaks positions
        df_sort=pd.DataFrame(data={'pos': peak_pos,
                            'height': height})

        df_peak_sort=df_sort.sort_values('height', axis=0, ascending=False)

        # Trim number of peaks based on user-defined N peaks
        df_peak_sort_short=df_peak_sort[0:N_peaks]
        print('Found peaks at:')
        print(df_peak_sort_short)
        print('Only returning up to N_peaks')

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

        LH_baseline_filt=LH_baseline[(LH_baseline[:, 1]<Mean_LH_baseline+outlier_sigma*Std_LH_baseline)
        &(LH_baseline[:, 1]>Mean_LH_baseline-outlier_sigma*Std_LH_baseline) ]

        RH_baseline_filt=RH_baseline[(RH_baseline[:, 1]<Mean_RH_baseline+outlier_sigma*Std_RH_baseline)
        &(RH_baseline[:, 1]>Mean_RH_baseline-outlier_sigma*Std_RH_baseline) ]

        Baseline=np.concatenate((LH_baseline_filt, RH_baseline_filt), axis=0)





        # Fits a polynomial to the baseline of degree
        Pf_baseline = np.poly1d(np.polyfit(Baseline[:, 0], Baseline[:, 1], N_poly))
        Py_base =Pf_baseline(Spectra_short[:, 0])

        Baseline_ysub=Pf_baseline(Baseline[:, 0])
        Baseline_x=Baseline[:, 0]
        y_corr= Spectra_short[:, 1]-  Py_base
        x=Spectra_short[:, 0]

        # NOw into the voigt fitting

        model0 = VoigtModel()#+ ConstantModel()

        # create parameters with initial values
        pars0 = model0.make_params()
        pars0['center'].set(cent, min=cent-30, max=cent+30)
        pars0['amplitude'].set(amplitude, min=0)


        init0 = model0.eval(pars0, x=x)
        result0 = model0.fit(y_corr, pars0, x=x)
        Center_p0=result0.best_values.get('center')
        area_p0=result0.best_values.get('amplitude')



        # Make a nice linspace for plotting with smooth curves.
        xx_carb=np.linspace(min(x), max(x), 2000)
        y_carb=result0.eval(x=xx_carb)

        # Plotting what its doing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
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



        if plot_figure is True:
            path3=path+'/'+'Peak_fit_images'
            if os.path.exists(path3):
                out='path exists'
            else:
                os.makedirs(path+'/'+ 'Peak_fit_images', exist_ok=False)

            file=filename.rsplit('.txt', 1)[0]
            fig.savefig(path3+'/'+'Carbonate_Fit_{}.png'.format(file), dpi=dpi)

        df=pd.DataFrame(data={'Carb_Cent': Center_p0,
        'Carb_Area': area_p0}, index=[0])



    if fit_carbonate is False:

        df=None
        xx_carb=None
        y_carb=None
        result0=None
    if return_other_params is True:
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

def proceed_to_fit_diads(filename, Carb_fit):
    M=input("Type 'Y' if you want to try to fit these diads, 'N' if you just want to export the data now with just the carbonate")

    if M == 'Y':
        print('Proceeding to let you do diad fits')
    if M == 'N':
        df=pd.DataFrame(data={'filename': filename,
                            'Splitting': np.nan,
                                'Diad1_Cent':np.nan,
                                    'Diad1_Area':np.nan,
                                        'Diad2_Cent':np.nan,
                                        'Diad2_Area':np.nan,
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



        df.to_clipboard(excel=True, header=False, index=False)
        print('Saved carbonate peak in diad format to clipboard')







