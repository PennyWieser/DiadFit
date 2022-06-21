import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DiadFit as pf
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import scipy
from scipy import stats

from DiadFit.importing_data_files import *

def find_olivine_peak_trough_pos(smoothed_ol_y, x_new, height=1):

    """" This function identifies the peaks and troughs in the Olivine spectra


    Parameters
    -----------

    path: smoothed_ol_y
        Olivine spectra y values after applying a cubic spline, and trimming to the spectra region around the peaks
        (from function smooth_and_trim_around_olivine)
    x_new: X values corresponding to y values in smoothed_ol_y

    height: int
        Height used for scipy find peaks function. May need tweaking

    Returns:
    -----------
    Peak positions (x), peak heights (y), trough x position, trough y position



    """
    # Find peaks with Scipy
    peaks_Ol = find_peaks(smoothed_ol_y, height)
    peak_height_Ol_unsort=peaks_Ol[1]['peak_heights']
    peak_pos_Ol_unsort = x_new[peaks_Ol[0]]

    df_peaks=pd.DataFrame(data={'pos': peak_pos_Ol_unsort,
                            'height': peak_height_Ol_unsort})
    df_peaks_sort=df_peaks.sort_values('height', axis=0, ascending=False)
    df_peak_sort_short1=df_peaks_sort[0:2]
    df_peak_sort_short=df_peak_sort_short1.sort_values('pos', axis=0, ascending=True)
    peak_pos_Ol=df_peak_sort_short['pos'].values
    peak_height_Ol=df_peak_sort_short['height'].values


    # Find troughs - e..g find minimum point +3 from the 1st peak, -3 units from the 2nd peak
    trim_y_cub_Ol=smoothed_ol_y[(x_new>(peak_pos_Ol[0]+3)) & (x_new<(peak_pos_Ol[1]-3))]
    trim_x=x_new[(x_new>(peak_pos_Ol[0]+3)) & (x_new<(peak_pos_Ol[1]-3))]


    trough_y=np.min(trim_y_cub_Ol)
    trough_x=trim_x[trim_y_cub_Ol==trough_y]


    return peak_pos_Ol, peak_height_Ol, trough_y, trough_x

def smooth_and_trim_around_olivine(x_range=[800,900], x_max=900, Ol_spectra=None,
                                   MI_spectra=None):
    """
    Takes melt inclusion and olivine spectra, and trims into the region around the olivine peaks,
    and fits a cubic spline (used for unmixing spectra)

    Parameters
    -----------
    x_range: list
        range of x coordinates to smooth between (e.g. [800, 900] by default
    Ol_spectra: nd.array
        numpy array of olivine spectra (x is wavenumber, y is intensity)
    MI_spectra: nd.array
        numpy array of melt inclusion spectra (x is wavenumber, y is intensity)



    Returns:
    -----------
    x_new: x coordinates of smoothed curves
    y_cub_MI: smoothed y coordinates using a cubic spline for MI
    y_cub_Ol: smoothed y coordinates using a cubic spline for Ol

    peak_pos_Ol: x coordinates of 2 olivine peaks
    peak_height_Ol: y coordinates of 2 olivine peaks
    trough_x: x coordinate of minimum point between peaks
    trough_y: y coordinate of minimum point between peaks
    """
    x_min=x_range[0]
    x_max=x_range[1]
    # Trim to region of interest
    Filt_Ol=Ol_spectra[~(
        (Ol_spectra[:, 0]<x_min) |
        (Ol_spectra[:, 0]>x_max)
    )]
    Filt_MI=MI_spectra[~(
        (MI_spectra[:, 0]<x_min) |
        (MI_spectra[:, 0]>x_max)
    )]

    # Fit spline to data

    x_MI=Filt_MI[:, 0]
    x_Ol=Filt_Ol[:, 0]

    y_MI=Filt_MI[:, 1]
    y_Ol=Filt_Ol[:, 1]


    # Fit a  cubic spline
    f2_MI = interp1d(x_MI, y_MI, kind='cubic')
    f2_Ol = interp1d(x_Ol, y_Ol, kind='cubic')

    x_new=np.linspace(min(x_Ol),max(x_Ol), 100000)

    y_cub_MI=f2_MI(x_new)
    y_cub_Ol=f2_Ol(x_new)

    # Plot peaks and troughs on this to check they are right
    peak_pos_Ol, peak_height_Ol, trough_y, trough_x=find_olivine_peak_trough_pos(
        smoothed_ol_y=y_cub_Ol, x_new=x_new, height=1)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.plot(Ol_spectra[:, 0], Ol_spectra[:, 1], '-g', label='Ol Spectra')
    ax1.plot(MI_spectra[:, 0], MI_spectra[:, 1], '-',
             color='salmon', label='MI Spectra')

    ax2.plot(Filt_MI[:, 0], Filt_MI[:, 1], '+', color='salmon')
    ax2.plot(Filt_Ol[:, 0], Filt_Ol[:, 1], '+g')
    ax2.plot(x_new, y_cub_MI, '-', color='salmon')
    ax2.plot(x_new, y_cub_Ol, '-g')
    ax2.plot(peak_pos_Ol, peak_height_Ol, '*k',mfc='yellow', ms=10, label='Peaks')
    ax2.plot(trough_x, trough_y, 'dk', mfc='cyan', ms=10, label='Trough')
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax1.set_ylabel('Intensity')

    return x_new, y_cub_MI, y_cub_Ol, peak_pos_Ol, peak_height_Ol, trough_x, trough_y



## Unmix the olivine
def trough_or_peak_higher(spectra_x, spectra_y, peak_pos_x,
                          trough_pos_x, trough_pos_y, av_width=1, plot=False,
                         print_result=False):
    """
    This function assesses whether the line between the 2 peaks is above or below the trough position
    Called by a loop to select the optimum unmixing ratio for olivine and melt


    Parameters
    -----------
    spectra_x: x coordinates of spectra to test

    spectra_y: y coordinates of spectra to test

    peak_pos_x: x positions of 2 olivine peaks

    trough_pos_x: x position of trough

    trough_pos_y: y position of trough

    av_width: averages +- 1 width either side of the peak and troughs when doing assesment and regression

    plot: bool

    print_result: bool (default False)
        prints whether it found the trough above or below the peak

    plot: bool (default False)
        if True, Draws a plot showing the regression

    Returns:
    -----------
    Dist: Vertical height between the linear line (Y coordinate) between the 2 peaks at the x position of the trough,
    and the y coordinate of the trough.

    """


    peak1_xex=peak_pos_x[0]
    peak2_xex=peak_pos_x[1]


    # Takes average av_width to either side of peak
    peak1_y=spectra_y[(spectra_x>peak1_xex-av_width)
                     &(spectra_x<peak1_xex+av_width)]
    peak2_y=spectra_y[(spectra_x>peak2_xex-av_width)
                     &(spectra_x<peak2_xex+av_width)]

    peak1_x=spectra_x[(spectra_x>peak1_xex-av_width)
                     &(spectra_x<peak1_xex+av_width)]
    peak2_x=spectra_x[(spectra_x>peak2_xex-av_width)
                     &(spectra_x<peak2_xex+av_width)]


    trough_y=spectra_y[(spectra_x>trough_pos_x-av_width)
                     &(spectra_x<trough_pos_x+av_width)]

    trough_x=spectra_x[(spectra_x>trough_pos_x-av_width)
                     &(spectra_x<trough_pos_x+av_width)]



    x=np.concatenate((peak1_x, peak2_x), axis=0)

    y=np.concatenate((peak1_y, peak2_y), axis=0)


    Px = np.linspace(peak1_x, peak2_x, 101)



    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    Py=slope*Px+intercept
    Pred_trough=slope*np.nanmean(trough_x)+intercept

    if print_result is True:
        if np.nanmean(trough_y)>np.nanmean(Pred_trough):
            print('Trough higher than peaks, subtracted too much')

        if np.nanmean(trough_y)<np.nanmean(Pred_trough):
            print('Trough lower than peaks, not subtracted enough')
    Dist=np.nanmean(trough_y)-np.nanmean(Pred_trough)
    #Py=lr.predict(Px)
    if plot is True:
        plt.plot(peak1_x, peak1_y, '+m')
        plt.plot(peak2_x, peak2_y, '+m')
        plt.plot(trough_x, trough_y, '+g')

        plt.plot(Px, Py, '-r')
        plt.plot(trough_pos_x, Pred_trough, '*k', ms=10)
        plt.plot(spectra_x, spectra_y, '-k')
    return Dist



# Now lets mix up spectra
def make_evaluate_mixed_spectra(smoothed_Ol_y, smoothed_MI_y,
                                Ol_spectra, MI_spectra, x_new, peak_pos_Ol,
                      trough_x, trough_y, N_steps=20, av_width=2,
                               X_min=0, X_max=1):

    """
    Makes mixed spectra from  measured MI  - measured Ol * X, where X
    is a factor the user can set determining the mixing proportions to test

    Parameters
    -----------
    smoothed_Ol_y: np.array
        y coordinates of olivine around peak region after fitting cubic spline

    smoothed_MI_y: np.array
        y coordinates of melt inclusion around peak region after fitting cubic spline

    x_new: np.array
        x coordinates from smoothed Ol and MI curves

    Ol_Spectra: np.array
        Full olivine spectra, not trimmed or smoothed

    MI_Spectra: np.array
        Full MI spectra, not trimmed or smoothed

    peak_pos_Ol: list
        Peak positions (x) of Olivine peaks

    trough_x: float, int
        Peak position (x) of olivine trough

    x_min:  float or int
        Minimum x for unmixing

    x_max:  float or int
        Maximum x for unmixing

    N_steps: int
        Number of mixing steps to use between X_min and X_max



    Returns:
    -----------


    """

    N_steps=20
    MI_Mix=np.empty((N_steps,len(smoothed_MI_y)), 'float')
    Dist=np.empty(N_steps, 'float')
    X=np.linspace(X_min, X_max, N_steps)
    for i in range(0, N_steps):

        # print('Working on mixing proportion:')
        # print(X[i])

        # Geochemistry style mix
        #MI_Mix[i, :]=(smoothed_MI_y- smoothed_Ol_y*X[i])/(1-X[i])
        # True subtraction mix from Smith 2021
        MI_Mix[i, :]=smoothed_MI_y- smoothed_Ol_y*X[i]

        Dist[i]=trough_or_peak_higher(spectra_x=x_new,
                          spectra_y=MI_Mix[i, :],
                          peak_pos_x=peak_pos_Ol,
                          trough_pos_x=trough_x,
                          trough_pos_y=trough_y,
                          av_width=2)
        #print(MI_Mix)

    mix_spline = interp1d(X, Dist, kind='cubic')

    x_new_mix=np.linspace(min(X), max(X), 5000)

    y_cub_mix=mix_spline(x_new_mix)


    # Closest point to zero
    val=np.argmax(y_cub_mix>0)
    ideal_mix=x_new_mix[val]
    print('best fit proportion')
    print(ideal_mix)

    MI_Mix_Best_syn=(smoothed_MI_y-smoothed_Ol_y*ideal_mix)/(1-ideal_mix)
    MI_Mix_Best=(MI_spectra- Ol_spectra*ideal_mix)/(1-ideal_mix)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10))

    for i in range(0, N_steps):
        ax1.plot(x_new, MI_Mix[i, :], '-k')
        ax1.plot([peak_pos_Ol[0], peak_pos_Ol[0]], [0.7, 1.5], '-', color='yellow')
        ax1.plot([peak_pos_Ol[1], peak_pos_Ol[1]], [0.7, 1.5], '-', color='yellow')
        ax1.plot([trough_x, trough_x], [0.7, 1.5], '-', color='cyan')


    ax1.set_xlabel('Wavenumber (cm$^{-1}$')
    ax1.set_ylabel('Intensity ratioed to 1st coordinate')

    ax2.plot(X, Dist, 'or')
    ax2.plot([min(X), max(X)], [0, 0], '-k')
    ax2.set_xlabel('Mixing Proportion of Olivine')
    ax2.set_ylabel('Vert. Dist. between peaks reg & troughs')
    ax2.plot(x_new_mix, y_cub_mix, '-r')

    ax3.plot(MI_spectra[:, 0],MI_Mix_Best[:, 1], '-k')
    ax3.plot(MI_spectra[:, 0],MI_spectra[:, 1], '-', color='salmon')
    ax3.plot(Ol_spectra[:, 0],Ol_spectra[:, 1], '-', color='g')
    ax3.set_xlim([775, 900])


    ax4.plot(MI_spectra[:, 0],MI_Mix_Best[:, 1], '-k', label='Umixed glass')
    ax4.plot(MI_spectra[:, 0],MI_spectra[:, 1],  '-', color='salmon',label='Measured MI')
    ax4.plot(Ol_spectra[:, 0],Ol_spectra[:, 1], '-', color='g', label='Measured Ol')
    ax4.legend()
    ax3.set_xlabel('Wavenumber (cm$^{-1}$')
    ax4.set_xlabel('Wavenumber (cm$^{-1}$')
    ax3.set_ylabel('Intensity')



    return MI_Mix_Best, ideal_mix, Dist, MI_Mix, X


## Fitting silica and water peak areas
def check_if_spectra_negative(Spectra=None, peak_pos_Ol=None, tie_x_cord=2000, override=False, flip=False):
    """
    Checks if spectra is negative, e.g. if the spectra N units in is higher or lower
    than the peak position of the olivine. This may depend on your Raman system, so you can always adjust


    Parameters
    -----------
    Spectra: np.array
        Spectra from the unmixing function

    peak_pos_Ol: list
        Olivine peak positions

    tie_x_cord: int or float
        Coordinate to use as a tie point, e.g. is olivine peak higher or lower than this?

    override: bool
        If False, function determins if it wants to invert the spectra,
        if true you can override
    flip: bool
        If override is true, flip=False leaves spectra how it is, True flips the y axis.

    Returns
    -----------

    Spectra: np.array


    """
    Spectra=Spectra.copy()

    val=np.argmax(Spectra[:, 0]>tie_x_cord)

    tie_y_cord=Spectra[val, 1]

    mean_around_peak=np.nanmean(
        Spectra[:, 1][(Spectra[:, 0]>peak_pos_Ol[0])
        &
        (Spectra[:, 0]<peak_pos_Ol[0]+5)]
            )


    print('y coordinate of tie coordinate')
    print(tie_y_cord)
    print('mean y coordinate around peak')
    print(mean_around_peak)
    x=Spectra[:, 0]
    y_init=Spectra[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), sharey=True)
    ax1.set_title('Entered Spectra')
    ax2.set_title('Returned Spectra')
    ax1.plot(x, y_init, '-r')
    ax1.plot(tie_x_cord, tie_y_cord, '*k',  ms=10,  label='tie_cord')
    ax1.plot(peak_pos_Ol[0], mean_around_peak, '*k', mfc='yellow', ms=15,label='Av Ol coordinate')

    if override is False:
        if mean_around_peak>tie_y_cord:
            y=y_init


            print('peak positive, spectra left as is')

            ax2.plot(x, y, '-r')
            ax2.plot(tie_x_cord, tie_y_cord, '*k',  ms=10,  label='tie_cord')
            ax2.plot(peak_pos_Ol[0], mean_around_peak, '*k', mfc='yellow', ms=15, label='Av Ol coordinate')
            ax2.legend()

            return Spectra
        else:
            print('Peak negative, spectra inverted')

            y=-Spectra[:, 1]
            ax2.plot(x, y, '-r')
            ax2.plot(tie_x_cord, -tie_y_cord, '*k', ms=10, label='tie_cord')
            ax2.plot(peak_pos_Ol[0], -mean_around_peak, '*k', mfc='yellow', ms=15, label='Av Ol coordinate')

            ax2.legend()
            Spectra2=np.column_stack((x, y))
            return Spectra2

    if override is True:
        print('Youve choosen to override the default')

        if flip is False:
            return Spectra
        if flip is True:
            print('spectra inverted')
            x=Spectra[:, 0]
            y=-Spectra[:, 1]
            plt.plot(x, y, '-r')
            Spectra2=np.column_stack((x, y))

            return Spectra2




def fit_area_for_silicate_region(Spectra=None, lower_range_sil=[200, 300], upper_range_sil=[1240, 1500],
sigma_sil=5, exclude_range1_sil=None, exclude_range2_sil=None, N_poly_sil=2, plot_figure=True,
fit_sil='poly'):

    """
    Fits background polynomial or spline. Integrates under curve, returns trapezoid and

    Parameters
    -----------
    Spectra: np. array
        Spectra with olivine subtracted from it

    lower_range_sil: list
        LHS part of spectra to use as a background (default [200, 300])

    upper_range_sil: list
        RHS part of spectra to use as background (default [1240, 1500])

    exclude_range1_sil,  exclude_range2_sil: list or None
        Can enter up to 2 ranges (e.g. [200, 210]) to remove, helps to trim cosmic rays

    fit_sil: 'poly' or 'spline'
        Fits a polynomial or cubic spline to curve.

        N_poly_sil: int
            Degree of polynomial to fit to if fit_sil='poly'

    poly_figure: bool
        if True, plots figure of silicate region and shows background, and background subtracted data



    Returns:
    -----------
    dataframe of background positions, and various area calculations.


    """

    Sil=Spectra
    exclude=False


    # These bits of code trim out the excluded regions if relevant
    if exclude_range1_sil is not None and exclude_range2_sil is None:
        exclude=True
        Sil_old=Sil.copy()
        Sil=Sil[(Sil[:, 0]<exclude_range1_sil[0])|(Sil[:, 0]>exclude_range1_sil[1])]
        Discard=Sil_old[(Sil_old[:, 0]>=exclude_range1_sil[0])
                        & (Sil_old[:, 0]<=exclude_range1_sil[1])]


    if exclude_range2_sil is not None and exclude_range1_sil is None:
        exclude=True
        Sil_old=Sil.copy()
        Sil=Sil[(Sil[:, 0]<exclude_range2_sil[0])|(Sil[:, 0]>exclude_range2_sil[1])]

        Discard=Sil_old[(Sil_old[:, 0]>=exclude_range2_sil[0])
                        & (Sil_old[:, 0]<=exclude_range2_sil[1])]

    if exclude_range1_sil is not None and exclude_range2_sil is not None:
        exclude=True
        Sil_old=Sil.copy()
        Sil=Sil[
        ((Sil[:, 0]<exclude_range1_sil[0])|(Sil[:, 0]>exclude_range1_sil[1]))
        &
        ((Sil[:, 0]<exclude_range2_sil[0])|(Sil[:, 0]>exclude_range2_sil[1]))
        ]

        Discard=Sil_old[
        ((Sil_old[:, 0]>=exclude_range1_sil[0]) & (Sil_old[:, 0]<=exclude_range1_sil[1]))
        |
        ((Sil_old[:, 0]>=exclude_range2_sil[0]) & (Sil_old[:, 0]<=exclude_range2_sil[1]))
        ]



    # Now we calculate the edge of the baseline
    lower_0baseline_sil=lower_range_sil[0]
    upper_0baseline_sil=lower_range_sil[1]
    lower_1baseline_sil=upper_range_sil[0]
    upper_1baseline_sil=upper_range_sil[1]

    # Bit that is actually peak, not baseline
    span=[upper_0baseline_sil, lower_1baseline_sil]

    # lower_2baseline=1320
    # upper_2baseline=1330

    # Trim for entire range
    Sil_short=Sil[ (Sil[:,0]>lower_0baseline_sil)
                  & (Sil[:,0]<upper_1baseline_sil) ]

    Sil_plot=Sil[ (Sil[:,0]>(lower_0baseline_sil-100))
                  & (Sil[:,0]<(upper_1baseline_sil+100)) ]


    # Get actual baseline
    Baseline_with_outl_sil=Sil_short[
    ((Sil_short[:, 0]<upper_0baseline_sil) &(Sil_short[:, 0]>lower_0baseline_sil))
         |
    ((Sil_short[:, 0]<upper_1baseline_sil) &(Sil_short[:, 0]>lower_1baseline_sil))]

    # Calculates the median for the baseline and the standard deviation
    Median_Baseline_sil=np.mean(Baseline_with_outl_sil[:, 1])
    Std_Baseline_sil=np.std(Baseline_with_outl_sil[:, 1])

    # Removes any points in the baseline outside of 2 sigma (helps remove cosmic rays etc).
    Baseline_sil=Baseline_with_outl_sil[
    (Baseline_with_outl_sil[:, 1]<Median_Baseline_sil+sigma_sil*Std_Baseline_sil)
                                &
    (Baseline_with_outl_sil[:, 1]>Median_Baseline_sil-sigma_sil*Std_Baseline_sil)
                               ]


    #Baseline=Baseline_with_outl


    if fit_sil == 'poly':
        # Fits a polynomial to the baseline of degree
        Pf_baseline_sil = np.poly1d(np.polyfit(Baseline_sil[:, 0], Baseline_sil[:, 1], N_poly_sil))
        Py_base_sil =Pf_baseline_sil(Sil_short[:, 0])



        Baseline_ysub_sil=Pf_baseline_sil(Sil_short[:, 0])


    if  fit_sil == 'spline':
        from scipy.interpolate import CubicSpline
        mix_spline_sil = CubicSpline(Baseline_sil[:, 0], Baseline_sil[:, 1],
                                extrapolate=True)

        Baseline_ysub_sil=mix_spline_sil(Sil_short[:, 0])


        N_poly_sil='Spline'

    Baseline_x_sil=Sil_short[:, 0]
    y_corr_sil= Sil_short[:, 1]- Baseline_ysub_sil

    x_sil=Baseline_sil[:, 0]

     # Plotting what its doing
    if plot_figure is True:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        ax1.plot(Sil_plot[:, 0], Sil_plot[:, 1], '-c', label='Spectra')
        if  exclude is True:
            ax1.plot(Discard[:, 0], Discard[:, 1], '*k', label='Discarded points')

        ax1.set_title('Background fit')
        ax1.plot(Sil_short[:, 0], Sil_short[:, 1], '-r', label='Selected part')

        ax1.plot(Baseline_sil[:, 0], Baseline_sil[:, 1], '.b', label='selected bck pts.')
        ax1.plot(Sil_short[:, 0], Baseline_ysub_sil, '-k', label='fitted bck')
        ax1.legend()
        xdat_sil=(Sil_short[:, 0])
        ydat_sil=y_corr_sil

        ax1.set_ylabel('Intensity')
        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Wavenumber')



        ax2.set_title('Background corrected')
        ax2.plot(xdat_sil, y_corr_sil, '-r')
        height_p=np.max(Sil_short[:, 1])-np.min(Sil_short[:, 1])
        #ax2.set_ylim([np.min(y_corr), 1.2*height_p ])
        ax1.set_xlabel('Wavenumber')


    from numpy import trapz
    from scipy.integrate import simps
    xspace_sil=xdat_sil[1]-xdat_sil[0]
    area_trap = trapz(y_corr_sil, dx=xspace_sil)
    area_simps = simps(y_corr_sil, dx=xspace_sil)



    df_sil=pd.DataFrame(data={'Sil_LHS_Back1':lower_range_sil[0],
                          'Sil_LHS_Back2':lower_range_sil[1],
                          'Sil_RHS_Back1':upper_range_sil[0],
                          'Sil_RHS_Back2':upper_range_sil[1],
                          'Sil_N_Poly': N_poly_sil,
                          'Sil_Trapezoid_Area':area_trap,
                          'Sil_Simpson_Area': area_simps}, index=[0])
    return df_sil



def fit_area_for_water_region(Spectra=None, lower_range_water=[2750, 3100], upper_range_water=[3750, 4100],
sigma_water=5, exclude_range1_water=None, exclude_range2_water=None,
N_poly_water=2, plot_figure=True, fit_water='poly'):


    """
    Fits background polynomial or spline. Integrates under curve, returns areas

    Parameters
    -----------
    Spectra: np. array
        Spectra with olivine subtracted from it

    lower_range_water: list
        LHS part of spectra to use as a background (default [2750, 3100])

    upper_range_water: list
        RHS part of spectra to use as background (default [3750, 4100])

    exclude_range1_water,  exclude_range2_water: list or None
        Can enter up to 2 ranges (e.g. [3100, 3110]) to remove, helps to trim cosmic rays

    fit_sil: 'poly' or 'spline'
        Fits a polynomial or cubic spline to curve.

        N_poly_sil: int
            Degree of polynomial to fit to if fit_sil='poly'

    poly_figure: bool
        if True, plots figure of water region and shows background, and background subtracted data



    Returns:
    -----------
    dataframe of background positions, and various area calculations.


    """

    Water=Spectra
    exclude=False




    # These bits of code trim out the excluded regions if relevant
    if exclude_range1_water is not None and exclude_range2_water is None:
        exclude=True
        Water_old=Water.copy()
        Water=Water[(Water[:, 0]<exclude_range1_water[0])|(Water[:, 0]>exclude_range1_water[1])]
        Discard=Water_old[(Water_old[:, 0]>=exclude_range1_water[0])
                        & (Water_old[:, 0]<=exclude_range1_water[1])]


    if exclude_range2_water is not None and exclude_range1_water is None:
        exclude=True
        Water_old=Water.copy()
        Water=Water[(Water[:, 0]<exclude_range2_water[0])|(Water[:, 0]>exclude_range2_water[1])]

        Discard=Water_old[(Water_old[:, 0]>=exclude_range2_water[0])
                        & (Water_old[:, 0]<=exclude_range2_water[1])]

    if exclude_range1_water is not None and exclude_range2_water is not None:
        exclude=True
        Water_old=Water.copy()
        Water=Water[
        ((Water[:, 0]<exclude_range1_water[0])|(Water[:, 0]>exclude_range1_water[1]))
        &
        ((Water[:, 0]<exclude_range2_water[0])|(Water[:, 0]>exclude_range2_water[1]))
        ]

        Discard=Water_old[
        ((Water_old[:, 0]>=exclude_range1_water[0]) & (Water_old[:, 0]<=exclude_range1_water[1]))
        |
        ((Water_old[:, 0]>=exclude_range2_water[0]) & (Water_old[:, 0]<=exclude_range2_water[1]))
        ]


    # Now we calculate the edge of the baseline
    lower_0baseline_water=lower_range_water[0]
    upper_0baseline_water=lower_range_water[1]
    lower_1baseline_water=upper_range_water[0]
    upper_1baseline_water=upper_range_water[1]

    # Bit that is actually peak, not baseline
    span=[upper_0baseline_water, lower_1baseline_water]

    # lower_2baseline=1320
    # upper_2baseline=1330

    # Trim for entire range
    Water_short=Water[ (Water[:,0]>lower_0baseline_water)
                  & (Water[:,0]<upper_1baseline_water) ]

    Water_plot=Water[ (Water[:,0]>(lower_0baseline_water-300))
                  & (Water[:,0]<(upper_1baseline_water+300)) ]


    # Get actual baseline
    Baseline_with_outl_water=Water_short[
    ((Water_short[:, 0]<upper_0baseline_water) &(Water_short[:, 0]>lower_0baseline_water))
         |
    ((Water_short[:, 0]<upper_1baseline_water) &(Water_short[:, 0]>lower_1baseline_water))]

    # Calculates the median for the baseline and the standard deviation
    Median_Baseline_water=np.mean(Baseline_with_outl_water[:, 1])
    Std_Baseline_water=np.std(Baseline_with_outl_water[:, 1])

    # Removes any points in the baseline outside of 2 sigma (helps remove cosmic rays etc).
    Baseline_water=Baseline_with_outl_water[
    (Baseline_with_outl_water[:, 1]<Median_Baseline_water+sigma_water*Std_Baseline_water)
                                &
    (Baseline_with_outl_water[:, 1]>Median_Baseline_water-sigma_water*Std_Baseline_water)
                               ]


    #Baseline=Baseline_with_outl


    if fit_water == 'poly':
        # Fits a polynomial to the baseline of degree
        Pf_baseline_water = np.poly1d(np.polyfit(Baseline_water[:, 0], Baseline_water[:, 1], N_poly_water))
        Py_base_water =Pf_baseline_water(Water_short[:, 0])



        Baseline_ysub_water=Pf_baseline_water(Water_short[:, 0])


    if  fit_water == 'spline':
        from scipy.interpolate import CubicSpline
        mix_spline_water = CubicSpline(Baseline_water[:, 0], Baseline_water[:, 1],
                                extrapolate=True)

        Baseline_ysub_water=mix_spline_water(Water_short[:, 0])


        N_poly_water='Spline'

    Baseline_x_water=Water_short[:, 0]
    y_corr_water= Water_short[:, 1]- Baseline_ysub_water

    x_water=Baseline_water[:, 0]

     # Plotting what its doing
    if plot_figure is True:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        ax1.plot(Water_plot[:, 0], Water_plot[:, 1], '-c')
        ax1.set_title('Background fit')
        ax1.plot(Water_short[:, 0], Water_short[:, 1], '-r')
        if  exclude is True:
            ax1.plot(Discard[:, 0], Discard[:, 1], '*k', label='Discarded points')
            ax1.legend()

        ax1.plot(Baseline_water[:, 0], Baseline_water[:, 1], '.b')
        ax1.plot(Water_short[:, 0], Baseline_ysub_water, '-k')
        xdat_water=(Water_short[:, 0])
        ydat_water=y_corr_water

        ax1.set_ylabel('Intensity')
        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)')



        ax2.set_title('Background corrected')
        ax2.plot(xdat_water, y_corr_water, '-r')
        height_p=np.max(Water_short[:, 1])-np.min(Water_short[:, 1])

        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')


    from numpy import trapz
    from scipy.integrate import simps
    xspace_water=xdat_water[1]-xdat_water[0]
    area_trap = trapz(y_corr_water, dx=xspace_water)
    area_simps = simps(y_corr_water, dx=xspace_water)



    df_water=pd.DataFrame(data={'Water_LHS_Back1':lower_range_water[0],
                          'Water_LHS_Back2':lower_range_water[1],
                          'Water_RHS_Back1':upper_range_water[0],
                          'Water_RHS_Back2':upper_range_water[1],
                          'Water_N_Poly': N_poly_water,
                          'Water_Trapezoid_Area':area_trap,
                          'Water_Simpson_Area': area_simps}, index=[0])
    return df_water


    ## Stitching results together nicely for output.


def stitch_dataframes_together(df_sil=None, df_water=None, Ol_file=None, MI_file=None):
    """ Stitches results from silicate and water peaks together, ready for output

    Parameters
    -----------

    df_sil: pd.DataFrame
        DataFrame of peak area and backgroudn positions from fit_area... function for silica

    df_water: pd.DataFrame
        DataFrame of peak area and backgroudn positions from fit_area... function for water

    Ol_file: str
        Olivine file name

    MI_file: str
        MI file name

    Parameters
    -----------
    pd.DataFrame with columns


    """
    Combo_Area=pd.concat([df_sil, df_water], axis=1)
    Combo_Area.insert(0, 'Olivine filename', Ol_file)
    Combo_Area.insert(1, 'MI filename', MI_file)
    Combo_Area.insert(2, 'Trap_H2O_Sil',
                      Combo_Area['Water_Trapezoid_Area']/Combo_Area['Sil_Trapezoid_Area'])
    Combo_Area.insert(3, 'Simp_H2O_Sil',
                      Combo_Area['Water_Simpson_Area']/Combo_Area['Sil_Simpson_Area'])

    return Combo_Area