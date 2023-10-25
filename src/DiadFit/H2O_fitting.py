import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DiadFit as pf
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import scipy
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Optional
from DiadFit.importing_data_files import *
from numpy import trapz
from scipy.integrate import simps
##
def extract_xstal_MI_name(*, files, char_xstal, pos_xstal, char_MI, pos_MI,
                         prefix=True, str_prefix=" ", file_ext='.txt'):

    """ Extracts the names of the crystal and MI samples from a list of filenames

    Parameters
    ------------
    files (list): A list of filenames.

    char_xstal (str), char_MI (str):
        The character or string used to split the filenames into parts. E.g. '_' if the filename is of the form 'FM_7_MI'.

    pos_xstal (int): The index of the part of the split filename that corresponds to the crystal sample name.

    pos_MI (int): The index of the part of the split filename that corresponds to the MI sample name.

    prefix (bool, optional):
        If True, removes prefix. E.g. WITEC instruments where 01 is appended onto the first file.

    str_prefix (str, optional): The prefix that the filenames should have if `prefix` is True. Default is " ".

    file_type (str, optional): The file extension of the filenames. Default is ".txt".

    Returns:
        df_out (pandas DataFrame): A dataframe with columns "filename", "crystal_name", and "MI_name", containing the input filenames, the extracted crystal sample names, and the extracted MI sample names, respectively.

    """

    file_simple=pf.extracting_filenames_generic(names=files,
    prefix=prefix, str_prefix=str_prefix,
   file_ext=file_ext)



    xstal=np.empty(len(file_simple), dtype=object)
    MI=np.empty(len(file_simple), dtype=object)

    for i in range(0, len(file_simple)):
        name=file_simple[i]
        xstal[i]=name.split(char_xstal)[pos_xstal]
        MI[i]=name.split(char_MI)[pos_MI]
    df_out=pd.DataFrame(data={'filename': files,
                         'crystal_name': xstal,
                         'MI_name': MI})
    return df_out



def find_host_peak_trough_pos(smoothed_host_y, x_new, height=1):

    """" This function identifies the peaks and troughs in the host mineral spectra

    Parameters
    -----------

    path: smoothed_host_y
        Host spectra y values after applying a cubic spline, and trimming to the spectra region around the peaks
        (from function smooth_and_trim_around_host)
    x_new: X values corresponding to y values in smoothed_ol_y

    height: int
        Height used for scipy find peaks function. May need tweaking

    Returns:
    -----------
    Peak positions (x), peak heights (y), trough x position, trough y position



    """
    # Find peaks with Scipy
    peaks_Host= find_peaks(smoothed_host_y, height)
    peak_height_Host_unsort=peaks_Host[1]['peak_heights']
    peak_pos_Host_unsort = x_new[peaks_Host[0]]

    df_peaks=pd.DataFrame(data={'pos': peak_pos_Host_unsort,
                            'height': peak_height_Host_unsort})
    df_peaks_sort=df_peaks.sort_values('height', axis=0, ascending=False)
    df_peak_sort_short1=df_peaks_sort[0:2]
    df_peak_sort_short=df_peak_sort_short1.sort_values('pos', axis=0, ascending=True)
    peak_pos_Host=df_peak_sort_short['pos'].values
    peak_height_Host=df_peak_sort_short['height'].values


    # Find troughs - e..g find minimum point +3 from the 1st peak, -3 units from the 2nd peak
    trim_y_cub_Host=smoothed_host_y[(x_new>(peak_pos_Host[0]+3)) & (x_new<(peak_pos_Host[1]-3))]
    trim_x=x_new[(x_new>(peak_pos_Host[0]+3)) & (x_new<(peak_pos_Host[1]-3))]


    trough_y=np.min(trim_y_cub_Host)
    trough_x=trim_x[trim_y_cub_Host==trough_y]


    return peak_pos_Host, peak_height_Host, trough_y, trough_x

def smooth_and_trim_around_host(filename=None, x_range=[800,900], x_max=900, Host_spectra=None,
                                   MI_spectra=None, plot_figure=True):
    """
    Takes melt inclusion and host spectra, and trims into the region around the host peaks,
    and fits a cubic spline (used for unmixing spectra)

    Parameters
    -----------
    x_range: list
        range of x coordinates to smooth between (e.g. [800, 900] by default

    Host_spectra: nd.array
        numpy array of host spectra (x is wavenumber, y is intensity)

    MI_spectra: nd.array
        numpy array of melt inclusion spectra (x is wavenumber, y is intensity)

    filename:str
        name of file for saving figure



    Returns:
    -----------
    x_new: x coordinates of smoothed curves
    y_cub_MI: smoothed y coordinates using a cubic spline for MI
    y_cub_Host: smoothed y coordinates using a cubic spline for Ol

    peak_pos_Host: x coordinates of 2 host peaks
    peak_height_Host: y coordinates of 2 host peaks
    trough_x: x coordinate of minimum point between peaks
    trough_y: y coordinate of minimum point between peaks
    """
    x_min=x_range[0]
    x_max=x_range[1]
    # Trim to region of interest
    Filt_Host=Host_spectra[~(
        (Host_spectra[:, 0]<x_min) |
        (Host_spectra[:, 0]>x_max)
    )]
    Filt_MI=MI_spectra[~(
        (MI_spectra[:, 0]<x_min) |
        (MI_spectra[:, 0]>x_max)
    )]

    # Fit spline to data

    x_MI=Filt_MI[:, 0]
    x_Host=Filt_Host[:, 0]

    y_MI=Filt_MI[:, 1]
    y_Host=Filt_Host[:, 1]


    # Fit a  cubic spline
    f2_MI = interp1d(x_MI, y_MI, kind='cubic')
    f2_Host = interp1d(x_Host, y_Host, kind='cubic')

    x_new=np.linspace(min(x_Host),max(x_Host), 100000)

    y_cub_MI=f2_MI(x_new)
    y_cub_Host=f2_Host(x_new)

    # Plot peaks and troughs on this to check they are right
    peak_pos_Host, peak_height_Host, trough_y, trough_x=find_host_peak_trough_pos(
        smoothed_host_y=y_cub_Host, x_new=x_new, height=1)


    if plot_figure is True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3.5))
        if filename is not None:
            fig.suptitle('file='+filename)
        ax1.plot(Host_spectra[:, 0], Host_spectra[:, 1], '-g', label='Host Spectra')
        ax1.plot(MI_spectra[:, 0], MI_spectra[:, 1], '-',
                color='salmon', label='MI Spectra')

        ax2.plot(Filt_MI[:, 0], Filt_MI[:, 1], '+', color='salmon')
        ax2.plot(Filt_Host[:, 0], Filt_Host[:, 1], '+g')
        ax2.plot(x_new, y_cub_MI, '-', color='salmon')
        ax2.plot(x_new, y_cub_Host, '-g')
        ax2.plot(peak_pos_Host, peak_height_Host, '*k',mfc='yellow', ms=10, label='Peaks')
        ax2.plot(trough_x, trough_y, 'dk', mfc='cyan', ms=10, label='Trough')

        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Intensity')
        ax1.legend(fontsize=8)
        ax2.legend(fontsize=8)

        return x_new, y_cub_MI, y_cub_Host, peak_pos_Host, peak_height_Host, trough_x, trough_y, fig
    else:
        return x_new, y_cub_MI, y_cub_Host, peak_pos_Host, peak_height_Host, trough_x, trough_y



## Unmix the host
def trough_or_peak_higher(spectra_x, spectra_y, peak_pos_x,
                          trough_pos_x, trough_pos_y, av_width=1, plot=False,
                         print_result=False):
    """
    This function assesses whether the line between the 2 peaks is above or below the trough position
    Called by a loop to select the optimum unmixing ratio for host and melt


    Parameters
    -----------
    spectra_x: x coordinates of spectra to test

    spectra_y: y coordinates of spectra to test

    peak_pos_x: x positions of 2 host peaks (from find_host_peak_trough_pos)

    trough_pos_x: x position of trough (from find_host_peak_trough_pos)

    trough_pos_y: y position of trough (from find_host_peak_trough_pos)

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
def make_evaluate_mixed_spectra(*, path, filename, smoothed_host_y, smoothed_MI_y,
                                Host_spectra, MI_spectra, x_new, peak_pos_Host,
                      trough_x, trough_y, N_steps=20, av_width=2,
                               X_min=0, X_max=1, plot_figure=True, dpi=200):

    """
    This function unmixes glass and host spectra, and fits the best fit proportion 
    where the host peak and trough disapears. Specifically, it calculates the mixed spectra by 
    taking the measured MI spectra and subtracting X*Ol spectra, where X is the mixing proportions

    Parameters
    -----------
    smoothed_host_y: np.array
        y coordinates of host around peak region (from the function smooth_and_trim_around_host)

    smoothed_MI_y: np.array
        y coordinates of melt inclusion around peak region (from the function smooth_and_trim_around_host)

    x_new: np.array
        x coordinates from smoothed Ol and MI curves (from the function smooth_and_trim_around_host)

    Host_Spectra: np.array
        Full host spectra, not trimmed or smoothed (from the function get_data)

    MI_Spectra: np.array
        Full MI spectra, not trimmed or smoothed  (from the function get_data)

    peak_pos_Host: list
        Peak positions (x) of Olivine peaks (from the function smooth_and_trim_around_host)

    trough_x: float, int
        Peak position (x) of Olivine trough (from the function smooth_and_trim_around_host)

    x_min:  float or int
        Minimum mixing proportion allowed

    x_max:  float or int
        Maximum mixing proportion allowed

    N_steps: int
        Number of mixing steps to use between X_Max and X_Max. E.g. Precisoin of mixed value.

    av_width: int
        averages +- 1 width either side of the peak and troughs when doing assesment and regression



    Returns:
    -----------
    MI_Mix_Best: np.array
        Spectra of best-fit unmixed spectra (e.g. where host peak and trough the smallest)
    ideal_mix: float
        Best fit mixing proportion (i.e. X)
    Dist: float
        Vertical distance between the host peak and trough (in intensity units)
    MI_Mix: np.array
        Umixed spectra for each of the N_steps
    X: np.array
        X coordinates of unmixed spectra (along with MI_Mix and X allows plots of unmixing)

    if plot_figure is True, also returns a plot showing the unmixing process


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
        MI_Mix[i, :]=smoothed_MI_y- smoothed_host_y*X[i]

        Dist[i]=trough_or_peak_higher(spectra_x=x_new,
                          spectra_y=MI_Mix[i, :],
                          peak_pos_x=peak_pos_Host,
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
    #print('best fit proportion')
    #print(ideal_mix)

    MI_Mix_Best_syn=(smoothed_MI_y-smoothed_host_y*ideal_mix)/(1-ideal_mix)
    MI_Mix_Best=(MI_spectra- Host_spectra*ideal_mix)/(1-ideal_mix)

    if plot_figure is True:

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10))
        fig.suptitle('file='+filename)
        for i in range(0, N_steps):
            ax1.plot(x_new, MI_Mix[i, :], '-k')
            ax1.plot([peak_pos_Host[0], peak_pos_Host[0]], [0.7, 1.5], '-', color='yellow')
            ax1.plot([peak_pos_Host[1], peak_pos_Host[1]], [0.7, 1.5], '-', color='yellow')
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
        ax3.plot(Host_spectra[:, 0],Host_spectra[:, 1], '-', color='g')
        ax3.set_xlim([775, 900])


        ax4.plot(MI_spectra[:, 0],MI_Mix_Best[:, 1], '-k', label='Umixed glass')
        ax4.plot(MI_spectra[:, 0],MI_spectra[:, 1],  '-', color='salmon',label='Measured MI')
        ax4.plot(Host_spectra[:, 0],Host_spectra[:, 1], '-', color='g', label='Measured Host')
        ax4.legend()
        ax3.set_xlabel('Wavenumber (cm$^{-1}$')
        ax4.set_xlabel('Wavenumber (cm$^{-1}$')
        ax3.set_ylabel('Intensity')

        path3=path+'/'+'H2O_Silicate_images'

        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ 'H2O_Silicate_images', exist_ok=False)


        file=filename
        fig.savefig(path3+'/'+'Host_Glass_Umixing_{}.png'.format(filename), dpi=dpi)

    return MI_Mix_Best, ideal_mix, Dist, MI_Mix, X


## Fitting silica and water peak areas
def check_if_spectra_negative(*, path, filename, Spectra=None, peak_pos_Host=None, tie_x_cord=2000,
override=False, flip=False, plot_figure=True, dpi=200):
    """
    This function checks if the unmixed specta is negative, based on two tie points.
    The first tie point is the mean y coordinate of the peak position of host +5 wavenumbers,
    and the second tie point (tie_x_cord) is an optional input. If the specta is inverted, 
    this function inverts it.


    Parameters
    -----------
    Spectra: np.array
        Spectra from the function make_evaluate_mixed_spectra

    peak_pos_Host: list
        Host peak positions from the function find_host_peak_trough_pos

    tie_x_cord: int or float
        X cooordinate to use as a tie point to ask whether the host peak's y coordinate is higher or lower than this.

    override: bool
        if False, function flips the spectra if its upsideown,
        if True, you can use the input 'flip' to manually flip the spectra
    flip: bool
        If override is true, flip=False leaves spectra how it is, True flips the y axis.

    Returns
    -----------

    Spectra: np.array
        Flipped or unflipped spectra


    """
    Spectra=Spectra.copy()

    val=np.argmax(Spectra[:, 0]>tie_x_cord)

    tie_y_cord=Spectra[val, 1]

    mean_around_peak=np.nanmean(
        Spectra[:, 1][(Spectra[:, 0]>peak_pos_Host[0])
        &
        (Spectra[:, 0]<peak_pos_Host[0]+5)]
            )



    x=Spectra[:, 0]
    y_init=Spectra[:, 1]


    if plot_figure is True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3.5))
        fig.suptitle('file='+filename)
        ax1.set_title('Entered Spectra')
        ax2.set_title('Returned Spectra')
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Intensity')
        ax2.set_ylabel('Intensity')
        ax1.plot(x, y_init, '-r')
        ax1.plot(tie_x_cord, tie_y_cord, '*k',  ms=10,  label='tie_cord')
        ax1.plot(peak_pos_Host[0], mean_around_peak, '*k', mfc='yellow', ms=15,label='Av host coordinate')

    if override is False:
        if mean_around_peak>tie_y_cord:
            y=y_init


            #print('peak positive, spectra left as is')

            if plot_figure is True:

                ax2.plot(x, y, '-r')
                ax2.plot(tie_x_cord, tie_y_cord, '*k',  ms=10,  label='tie_cord')
                ax2.plot(peak_pos_Host[0], mean_around_peak, '*k', mfc='yellow', ms=15, label='Av Ol coordinate')
                ax2.legend()

        else:
            #print('Peak negative, spectra inverted')

            y=-Spectra[:, 1]
            Spectra=np.column_stack((x, y))
            if plot_figure is True:
                ax2.plot(x, y, '-r')
                ax2.plot(tie_x_cord, -tie_y_cord, '*k', ms=10, label='tie_cord')
                ax2.plot(peak_pos_Host[0], -mean_around_peak, '*k', mfc='yellow', ms=15, label='Av Ol coordinate')

                ax2.legend()





    if override is True:
        print('Youve choosen to override the default')

        if flip is False:
            return Spectra
        if flip is True:
            print('spectra inverted')
            x=Spectra[:, 0]
            y=-Spectra[:, 1]
            #plt.plot(x, y, '-r')
            Spectra=np.column_stack((x, y))

        if plot_figure is True:
            fig.tight_layout()
            path3=path+'/'+'H2O_Silicate_images'
            if os.path.exists(path3):
                out='path exists'
            else:
                os.makedirs(path+'/'+ 'H2O_Silicate_images', exist_ok=False)


            file=filename
            fig.tight_layout()
            fig.savefig(path3+'/'+'Check_if_negative_{}.png'.format(filename), dpi=dpi)

    return Spectra





@dataclass
class sil_bck_pos_Schiavi_rhyolite:
    """
    Configuration object for Silicate background positions from Schiavi et al. (2018) for rhyolites

    Parameters
    ----------
    lower_range_sil, mid_range1_sil, mid_range2_sil, upper_range_sil: Tuple[float, float]
        spectral range taken as background positions (from left to right as listed here)

    fit_silicate : str
        Type of fit for the baseline of the silicate region ('poly' or 'spline')

    N_poly_silicate: int
        Degree of polynomial fit for the baseline of the silicate region

    sigma_water: int
        Allow points on background within +-sigma_silicate * std dev of other background points

    """

    lower_range_sil: [Tuple[float, float]]=(190, 210)
    mid_range1_sil: [Tuple[float, float]]=(670, 710)
    mid_range2_sil: [Tuple[float, float]]=(840, 845)
    upper_range_sil: [Tuple[float, float]]=(1250, 1260)
    N_poly_sil: float=3
    sigma_sil: float = 5


@dataclass
class sil_bck_pos_Schiavi_andesite:
    """
    Configuration object for Silicate background positions from Schiavi et al. (2018) for andesites

    Parameters
    ----------
    lower_range_sil, mid_range1_sil, mid_range2_sil, upper_range_sil: Tuple[float, float]
    spectral range taken as background positions (from left to right as listed here)

    fit_silicate : str
        Type of fit for the baseline of the silicate region ('poly' or 'spline')
    N_poly_silicate: int
        Degree of polynomial fit for the baseline of the silicate region
    sigma_water: int
        Allow points on background within +-sigma_silicate * std dev of other background points

"""


    lower_range_sil: [Tuple[float, float]]=(230,250)
    mid_range1_sil: [Tuple[float, float]]=(645, 700)
    mid_range2_sil: [Tuple[float, float]]=(825, 840)
    upper_range_sil: [Tuple[float, float]]=(1230, 1250)
    N_poly_sil: float=3
    sigma_sil: float = 5



@dataclass
class sil_bck_pos_Schiavi_basalt:
    """
    Configuration object for Silicate background positions from Schiavi et al. (2018) for basalts

    Parameters
    ----------
    lower_range_sil, mid_range1_sil, mid_range2_sil, upper_range_sil: Tuple[float, float]
    spectral range taken as background positions (from left to right as listed here)

    fit_silicate : str
        Type of fit for the baseline of the silicate region ('poly' or 'spline')
    N_poly_silicate: int
        Degree of polynomial fit for the baseline of the silicate region
    sigma_water: int
        Allow points on background within +-sigma_silicate * std dev of other background points

"""

    lower_range_sil: [Tuple[float, float]]=(300, 340)
    mid_range1_sil: [Tuple[float, float]]=(630, 640)
    mid_range2_sil: [Tuple[float, float]]=(800,830)
    upper_range_sil: [Tuple[float, float]]=(1200, 1250)
    # HW and LW from Diego
    LW: [Tuple[float, float]]=(400, 600)
    HW: [Tuple[float, float]]=(800, 1200)
    N_poly_sil: float=3
    sigma_sil: float = 5


@dataclass
class sil_bck_pos_Schiavi_basanite:
    """
    Configuration object for Silicate background positions from Schiavi et al. (2018) for basanites

    Parameters
    ----------
    lower_range_sil, mid_range1_sil, mid_range2_sil, upper_range_sil: Tuple[float, float]
    spectral range taken as background positions (from left to right as listed here)

    fit_silicate : str
        Type of fit for the baseline of the silicate region ('poly' or 'spline')
    N_poly_silicate: int
        Degree of polynomial fit for the baseline of the silicate region
    sigma_water: int
        Allow points on background within +-sigma_silicate * std dev of other background points

"""

    lower_range_sil: [Tuple[float, float]]=(340, 360)
    mid_range1_sil: [Tuple[float, float]]=(630, 640)
    mid_range2_sil: [Tuple[float, float]]=(np.nan, np.nan)
    upper_range_sil: [Tuple[float, float]]=(1190, 1200)
    N_poly_sil: float=3
    sigma_sil: float = 5






def fit_area_for_silicate_region(*, path, filename, Spectra=None,
config1: sil_bck_pos_Schiavi_basalt(),
sigma_sil=5, exclude_range1_sil=None, exclude_range2_sil=None, plot_figure=True, save_fig=True,
fit_sil='poly', dpi=200):

    """
    Calculates the area of silicate peaks in a spectrum and fits a polynomial or spline curve to the baseline of the spectrum.


    Parameters
    ----------
    path : str
        File path

    filename : str
        File name

    Spectra : numpy.ndarray, optional
        2D array representing the spectrum data

    config1 : object
        Configuration object for silicate peak and background positions. Default values stored in the dataclasses
        'sil_bck_pos_Schiavi_basalt', _andesite, etc. Can tweak these as well.

    exclude_range1_silicate : list of float, optional
        List of two numbers representing the start and end of a range of
        wavenumbers to be excluded from the spectrum

    exclude_range2_silicate : list of float, optional
        List of two numbers representing the start and end of a second
        range of wavenumbers to be excluded from the spectrum

    plot_figure : bool, optional
        Indicates whether or not to plot the fit (default is True)

    dpi : int, optional
        Resolution of the plot in dots per inch (default is 200)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for 'Silicate_Trapezoid_Area', 'Silicate_Simpson_Area',
        as well as parameters for the selected background positions

    """


    Sil=Spectra
    exclude=False

    lower_range_sil=config1.lower_range_sil
    upper_range_sil=config1.upper_range_sil
    mid_range1_sil= config1.mid_range1_sil
    mid_range2_sil=config1.mid_range2_sil
    N_poly_sil=config1.N_poly_sil
    sigma_sil=config1.sigma_sil
    LW=(lower_range_sil[1], mid_range1_sil[0])
    if mid_range2_sil[0]>0:
        HW=(mid_range2_sil[1],  upper_range_sil[0])
        MW=(mid_range1_sil[1], mid_range2_sil[0])
    else:
        HW=(mid_range1_sil[1],upper_range_sil[0] )
        MW=None



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
    ((Sil_short[:, 0]<mid_range1_sil[1]) &(Sil_short[:, 0]>mid_range1_sil[0]))
         |
    ((Sil_short[:, 0]<mid_range2_sil[1]) &(Sil_short[:, 0]>mid_range2_sil[0]))
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
    xdat_sil=(Sil_short[:, 0])
    ydat_sil=y_corr_sil

    xspace_sil=xdat_sil[1]-xdat_sil[0]
    area_trap = trapz(y_corr_sil, dx=xspace_sil)
    area_simps = simps(y_corr_sil, dx=xspace_sil)
    # Just the LW area
    xsil_LW=xdat_sil[(xdat_sil>LW[0]) & (xdat_sil<LW[1])]
    y_corr_sil_LW=y_corr_sil[(xdat_sil>LW[0]) & (xdat_sil<LW[1])]
    xspace_sil_LW=xsil_LW[1]-xsil_LW[0]
    area_trap_LW=trapz(y_corr_sil_LW, dx=xspace_sil_LW)
    area_simp_LW=simps(y_corr_sil_LW, dx=xspace_sil_LW)


    # Just the HW area
    xsil_HW=xdat_sil[(xdat_sil>HW[0]) & (xdat_sil<HW[1])]
    y_corr_sil_HW=y_corr_sil[(xdat_sil>HW[0]) & (xdat_sil<HW[1])]
    xspace_sil_HW=xsil_HW[1]-xsil_HW[0]
    area_trap_HW=trapz(y_corr_sil_HW, dx=xspace_sil_HW)
    area_simp_HW=simps(y_corr_sil_HW, dx=xspace_sil_HW)

    # MW
    if MW is not None:
        xsil_MW=xdat_sil[(xdat_sil>MW[0]) & (xdat_sil<MW[1])]
        y_corr_sil_MW=y_corr_sil[(xdat_sil>MW[0]) & (xdat_sil<MW[1])]
        xspace_sil_MW=xsil_MW[1]-xsil_MW[0]
        area_trap_MW=trapz(y_corr_sil_MW, dx=xspace_sil_MW)
        area_simp_MW=simps(y_corr_sil_MW, dx=xspace_sil_MW)


     # Plotting what its doing
    if plot_figure is True:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        fig.suptitle('file='+filename)
        ax1.plot(Sil_plot[:, 0], Sil_plot[:, 1], '-c', label='Spectra')
        if  exclude is True:
            ax1.plot(Discard[:, 0], Discard[:, 1], '*k', label='Discarded points')

        ax1.set_title('Background fit')
        ax1.plot(Sil_short[:, 0], Sil_short[:, 1], '-r', label='Selected part')

        ax1.plot(Baseline_sil[:, 0], Baseline_sil[:, 1], '.b', label='selected bck pts.')
        ax1.plot(Sil_short[:, 0], Baseline_ysub_sil, '-k', label='fitted bck')
        ax1.legend()


        ax1.set_ylabel('Intensity')
        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Wavenumber')



        ax2.set_title('Background corrected')
        ax2.plot(xdat_sil, y_corr_sil, '-r')
        height_p=np.max(Sil_short[:, 1])-np.min(Sil_short[:, 1])
        #ax2.set_ylim([np.min(y_corr), 1.2*height_p ])
        ax1.set_xlabel('Wavenumber')
        ax2_max=np.max(xdat_sil)
        ax2_min=np.min(xdat_sil)
        ax2.plot([ax2_min, ax2_max], [0, 0], '-k')


        ax2.fill_between(xsil_LW, y_corr_sil_LW, color='red', label='LW', alpha=0.5)
        ax2.fill_between(xsil_HW, y_corr_sil_HW, color='cyan', label='HW', alpha=0.5)
        if MW is not None:
            ax2.fill_between(xsil_MW, y_corr_sil_MW, color='yellow', label='MW', alpha=0.5)


        ax2.legend()


        path3=path+'/'+'H2O_Silicate_images'
        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ 'H2O_Silicate_images', exist_ok=False)


        file=filename
        fig.savefig(path3+'/'+'Silicate_fit_{}.png'.format(filename), dpi=dpi)







    df_sil=pd.DataFrame(data={
    'Silicate_LHS_Back1':lower_range_sil[0],
                          'Silicate_LHS_Back2':lower_range_sil[1],
                          'Silicate_RHS_Back1':upper_range_sil[0],
                          'Silicate_RHS_Back2':upper_range_sil[1],
                          'Silicate_N_Poly': N_poly_sil,
                          'Silicate_Trapezoid_Area':area_trap,
                          'Silicate_Simpson_Area': area_simps,
                          'LW_Silicate_Trapezoid_Area':area_trap_LW,
                          'LW_Silicate_Simpson_Area':area_simp_LW,
                          'HW_Silicate_Trapezoid_Area':area_trap_LW,
                          'HW_Silicate_Simpson_Area':area_simp_LW,
                           }, index=[0])

    if MW is not None:
        df_sil['MW_Silicate_Trapezoid_Area']=area_trap_MW
        df_sil['MW_Silicate_Simpson_Area']=area_simp_MW
    return df_sil


@dataclass
class water_bck_pos:
    """ Configuration object for water peak and background positions.

Parameters
    ----------
    lower_bck_water, upper_bck_water: Tuple[float, float]
        2 coordinates for background to left of water peak, and to right.
    fit_water : str
        Type of fit for the baseline of the water region ('poly' or 'spline')
    N_poly_water : int
        Degree of polynomial fit for the baseline of the water region
    sigma_water: int
        Allow points on background within +-sigma*water * std dev of other background points


    """

    fit_water: str='poly'
    N_poly_water: float=3
    lower_bck_water: [Tuple[float, float]]=(2750, 3100)
    upper_bck_water: [Tuple[float, float]]=(3750, 4100)
    sigma_water=5





def fit_area_for_water_region(*, path, filename, Spectra=None, config1: water_bck_pos(),
 exclude_range1_water=None, exclude_range2_water=None, plot_figure=True,  dpi=200):

    """
    Calculates the area of water peaks in a spectrum and fits a polynomial or spline curve to the baseline of the spectrum.

    Parameters
    ----------
    path : str
        File path
    filename : str
        File name
    Spectra : numpy.ndarray, optional
        2D array representing the spectrum data
    config1 : object
        Configuration object for water peak and background positions. Default parameters stored in water_bck_pos, user can tweak.
        Parameters that need tweaking:

        fit_water: str 'poly',
        N_poly_water: str, degree of polynomial to fit to background
        lower_bck_water: [float, float], background position to left of water peak
        upper_bck_water: [float, float], background position to right of water peak

    exclude_range1_water : list of float, optional
        List of two numbers representing the start and end of a range of wavenumbers to be excluded from the spectrum
    exclude_range2_water : list of float, optional
        List of two numbers representing the start and end of a second range of wavenumbers to be excluded from the spectrum
    plot_figure : bool, optional
        Indicates whether or not to plot the fit (default is True)
    dpi : int, optional
        Resolution of the plot in dots per inch (default is 200)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for 'Water_Trapezoid_Area', 'Water_Simpson_Area', as well as parameters for the selected background positions

    """
    Water=Spectra
    exclude=False
    N_poly_water=config1.N_poly_water
    fit_water=config1.fit_water
    lower_range_water=config1.lower_bck_water
    upper_range_water=config1.upper_bck_water
    sigma_water=config1.sigma_water



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

    xdat_water=(Water_short[:, 0])

    Baseline_x_water=Water_short[:, 0]
    y_corr_water= Water_short[:, 1]- Baseline_ysub_water
    ydat_water=y_corr_water
    x_water=Baseline_water[:, 0]


    xspace_water=xdat_water[1]-xdat_water[0]
    area_trap = trapz(y_corr_water, dx=xspace_water)
    area_simps = simps(y_corr_water, dx=xspace_water)


     # Plotting what its doing
    if plot_figure is True:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        fig.suptitle('file='+filename)
        ax1.plot(Water_plot[:, 0], Water_plot[:, 1], '-c')
        ax1.set_title('Background fit')
        ax1.plot(Water_short[:, 0], Water_short[:, 1], '-r')
        if  exclude is True:
            ax1.plot(Discard[:, 0], Discard[:, 1], '*k', label='Discarded points')
            ax1.legend()

        ax1.plot(Baseline_water[:, 0], Baseline_water[:, 1], '.b')
        ax1.plot(Water_short[:, 0], Baseline_ysub_water, '-k')


        ax1.set_ylabel('Intensity')
        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)')



        ax2.set_title('Background corrected')
        ax2.plot(xdat_water, y_corr_water, '-r')
        height_p=np.max(Water_short[:, 1])-np.min(Water_short[:, 1])

        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax2.fill_between(xdat_water, y_corr_water, color='cornflowerblue', label='Water', alpha=0.5)
        ax2.legend()

        path3=path+'/'+'H2O_Silicate_images'
        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ 'H2O_Silicate_images', exist_ok=False)


        file=filename.rsplit('.txt', 1)[0]
        fig.savefig(path3+'/'+'Water_fit_{}.png'.format(filename), dpi=dpi)








    df_water=pd.DataFrame(data={'Water Filename': filename,
    'Water_LHS_Back1':lower_range_water[0],
                          'Water_LHS_Back2':lower_range_water[1],
                          'Water_RHS_Back1':upper_range_water[0],
                          'Water_RHS_Back2':upper_range_water[1],
                          'Water_N_Poly': N_poly_water,
                          'Water_Trapezoid_Area':area_trap,
                          'Water_Simpson_Area': area_simps}, index=[0])
    return df_water


    ## Stitching results together nicely for output.


def stitch_dataframes_together(df_sil, df_water,  MI_file, Host_file=None, save_csv=False, path=False):
    """ This function stitches together results from the fit_area function for silicate and water peaks and returns a DataFrame with the combined results. The DataFrame includes peak areas and background positions for both silicate and water peaks, and adds columns for the ratios of water to silicate areas.

    Parameters
    -----------

    df_sil: pd.DataFrame
        DataFrame of peak area and background positions from fit_area_for_silicate_region()

    df_water: pd.DataFrame
        DataFrame of peak area and background positions from fit_area_for_water_region()

    MI_file: str
        MI file name

    Host_file: str, optional
        Olivine file name


    Returns
    -----------
    pd.DataFrame
        DataFrame with columns for MI filename, HW:LW_Trapezoid, HW:LW_Simpson, Water_Trapezoid_Area,
        Water_Simpson_Area, Silicate_Trapezoid_Area, and Silicate_Simpson_Area.
        If Host_file is provided,
        the DataFrame will also include a column for Host filename.


    """
    Combo_Area=pd.concat([df_sil, df_water], axis=1)
    if Host_file is not None:
        Combo_Area.insert(0, 'Host filename', Host_file)
    Combo_Area.insert(1, 'MI filename', MI_file)
    Combo_Area.insert(2, 'HW:LW_Trapezoid',
                      Combo_Area['Water_Trapezoid_Area']/Combo_Area['HW_Silicate_Trapezoid_Area'])
    Combo_Area.insert(3, 'HW:LW_Simpson',
                      Combo_Area['Water_Simpson_Area']/Combo_Area['HW_Silicate_Simpson_Area'])

    if Host_file is not None:
        cols_to_move=['Host filename', 'MI filename', 'HW:LW_Trapezoid', 'HW:LW_Simpson',
     'Water_Trapezoid_Area', 'Water_Simpson_Area', 'Silicate_Trapezoid_Area', 'Silicate_Simpson_Area']
    else:
        cols_to_move=['MI filename', 'HW:LW_Trapezoid', 'HW:LW_Simpson',
     'Water_Trapezoid_Area', 'Water_Simpson_Area', 'Silicate_Trapezoid_Area', 'Silicate_Simpson_Area']



    Combo_Area = Combo_Area[cols_to_move + [
        col for col in Combo_Area.columns if col not in cols_to_move]]

    if save_csv is True:
        if path is False:
            raise TypeError('You need to enter a path to say where to save the CSV')
        filename_with_ext=Combo_Area['Water Filename'][0]
        filename, extension = os.path.splitext(filename_with_ext)
        filename = filename.split('.')[0]
        filename2=filename+ '_combo_fit.csv'
        full_path = os.path.join(path, filename2)
        Combo_Area.to_csv(full_path)

    return Combo_Area