import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# New things needed here
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import os
from os import listdir
from os.path import isfile, join
from DiadFit.importing_data_files import *
from DiadFit.diads import *

np.seterr (invalid='raise')

encode="ISO-8859-1"


def assess_diad1_skewness(*,  int_cut_off=0.3, path=None, filename=None, filetype=None, Diad_files=None,
exclude_range1=None, exclude_range2=None,
N_poly_bck_diad1=1, lower_baseline_diad1=[1220, 1250],
upper_baseline_diad1=[1300, 1350], save_fig=True, dpi=300, skewness='abs'):
    """ Assesses Skewness of Diad peaks. Useful for identifying mixed L + V phases
    (see DeVitre et al. in review)


    Parameters
    -----------
    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read

    OR

    Diad_files:str
        File name

    filetype: str
        Identifies type of file
        Witec_ASCII: Datafile from WITEC with metadata for first few lines
        headless_txt: Txt file with no headers, just data with wavenumber in 1st col, int 2nd
        HORIBA_txt: Datafile from newer HORIBA machines with metadata in first rows
        Renishaw_txt: Datafile from renishaw with column headings.


    lower_baseline_diad1: list
        Region of spectra to use a baseline (LHS)

    upper_baseline_diad1: list
        Region of spectra to use a baseline (RHS)

    N_poly_bck_diad1: int
        Degree of polynomial for background fitting

    skewness: 'abs' or 'dir'
        If 'abs', gives absolute skewness (e.g., largest possible number regardless of direction)
        if 'dir' does skewness as one side over the other

    int_cut_off: float
        Value of intensity at which to calculate the skewness (e.g. 0.15X the peak height).


    Returns
    -----------
    pd.DataFrame wtih filename, skewness of Diad 1,
    the x position of the LH and RH tie point at intensity=int_cut_off

    """

    if Diad_files is not None:
        y_corr_diad1, Py_base_diad1, x_diad1,  Diad_short, Py_base_diad1, Pf_baseline,  Baseline_ysub_diad1, Baseline_x_diad1, Baseline, span=remove_diad_baseline(
        Diad_files=Diad_files, exclude_range1=exclude_range1, exclude_range2=exclude_range2, N_poly=N_poly_bck_diad1,
        lower_range=lower_baseline_diad1, upper_range=upper_baseline_diad1, plot_figure=False)

    else:

    # First, do the background subtraction around the diad
        y_corr_diad1, Py_base_diad1, x_diad1,  Diad_short, Py_base_diad1, Pf_baseline,  Baseline_ysub_diad1, Baseline_x_diad1, Baseline, span=remove_diad_baseline(
        path=path, filename=filename, filetype=filetype, exclude_range1=exclude_range1, exclude_range2=exclude_range2, N_poly=N_poly_bck_diad1,
        lower_range=lower_baseline_diad1, upper_range=upper_baseline_diad1, plot_figure=False)



    x_lin_baseline=np.linspace(lower_baseline_diad1[0], upper_baseline_diad1[1], 100000)
    ybase_xlin=Pf_baseline(x_lin_baseline)



    # Get x and y ready to make a cubic spline through data
    x=x_diad1
    y=y_corr_diad1

    # Fit a  cubic spline
    f2 = interp1d(x, y, kind='cubic')
    x_new=np.linspace(np.min(x), np.max(x), 100000)

    y_cub=f2(x_new)



    # Use Scipy find peaks to get the biggest peak
    height=1
    peaks = find_peaks(y_cub, height)
    peak_height=peaks[1]['peak_heights']
    peak_pos = x_new[peaks[0]]

    # find max peak.  put into df because i'm lazy
    peak_df=pd.DataFrame(data={'pos': peak_pos,
                        'height': peak_height})

    # Find bigest peaks,
    df_peak_sort=peak_df.sort_values('height', axis=0, ascending=False)
    df_peak_sort_trim=df_peak_sort[0:1]
    Peak_Center=df_peak_sort_trim['pos']
    Peak_Height=df_peak_sort_trim['height']


    # Find intensity cut off
    y_int_cut=Peak_Height*int_cut_off

    # Split the array into a LHS and a RHS
    LHS_y=y_cub[x_new<=Peak_Center.values]
    RHS_y=y_cub[x_new>Peak_Center.values]

    LHS_x=x_new[x_new<=Peak_Center.values]
    RHS_x=x_new[x_new>Peak_Center.values]

    # Need to flip LHS to put into the find closest function
    LHS_y_flip=np.flip(LHS_y)
    LHS_x_flip=np.flip(LHS_x)

    val=np.argmax(LHS_y_flip<y_int_cut.values)

    val2=np.argmax(RHS_y<y_int_cut.values)


    # Find nearest x unit to this value
    y_nearest_LHS=LHS_y_flip[val]
    x_nearest_LHS=LHS_x_flip[val]

    y_nearest_RHS=RHS_y[val2]
    x_nearest_RHS=RHS_x[val2]


    # Return Skewness
    LHS_Center=abs(x_nearest_LHS-Peak_Center)
    RHS_Center=abs(x_nearest_RHS-Peak_Center)


   #Also get option to always take biggest value of skewness
    if skewness=='abs':
        if LHS_Center.values>RHS_Center.values:
            AS=LHS_Center.values/RHS_Center.values
        else:
            AS=RHS_Center.values/LHS_Center.values
    elif skewness=='dir':
        AS=RHS_Center.values/LHS_Center.values


    if save_fig is True:

        # Make pretty figure showing background subtractoin routine
        fig, ((ax2, ax1), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

        ax2.set_title('a) Spectra')
        ax2.plot(Diad_short[:, 0], Diad_short[:, 1], '-r', label='Data')
        ax2.set_xlim([lower_baseline_diad1[0]+40, upper_baseline_diad1[1]-40])
        ax1.set_title('b) Background fit')
        ax1.plot(Diad_short[:, 0], Diad_short[:, 1], '.-c', label='Data')

        ax1.plot(Baseline[:, 0], Baseline[:, 1], '.b', label='bck')
        ax1.plot(Diad_short[:, 0], Py_base_diad1, '-k', label='bck fit')
        ax3.set_title('c) Background subtracted')
        ax3.plot(x_diad1,y_corr_diad1, '-r', label='Bck sub')

    # Adds cubic interpoloation for inspection

        ax4.set_title('d) Cubic Spline')


        #print(x_nearest_LHS.dtypes)
        print(Peak_Center.dtypes)
        print(y_int_cut.dtypes)

        ax4.plot([x_nearest_LHS, Peak_Center.values], [y_int_cut.values, y_int_cut.values], '-g')

        ax4.annotate(str(np.round(LHS_Center.values[0], 2)),
                     xy=(x_nearest_LHS-3, y_int_cut+(Peak_Height-y_int_cut)/10), xycoords="data",
                     fontsize=12, color='green')
        ax4.plot(x_nearest_LHS, y_nearest_LHS, '*k', mfc='green',ms=15, label='RH tie')



        ax4.plot([Peak_Center, x_nearest_RHS], [y_int_cut, y_int_cut], '-', color='grey')
        ax4.annotate(str(np.round(RHS_Center.values[0], 2)),
                     xy=(x_nearest_RHS+3, y_int_cut+(Peak_Height-y_int_cut)/10), xycoords="data",
                      fontsize=12, color='grey')
        ax4.plot(x_nearest_RHS, y_nearest_RHS, '*k', mfc='grey', ms=15, label='LH tie')




        ax4.plot(x, y, '.r')
        ax4.plot(x_new, y_cub, '-k')

        ax4.plot([Peak_Center, Peak_Center], [Peak_Height, Peak_Height],
             '*k', mfc='blue', ms=15, label='Scipy Center')


        # Add to plot

        ax4.plot(lower_baseline_diad1[0], upper_baseline_diad1[1], [y_int_cut, y_int_cut], ':r')




        ax4.set_xlim([x_nearest_LHS-10, x_nearest_RHS+10])
        ax4.set_ylim([0-10, Peak_Height.values*1.2])
        ax4.plot([Peak_Center, Peak_Center], [0, Peak_Height], ':b')



        ax4.annotate('Skewness='+str(np.round(AS[0], 2)),
                     xy=(x_nearest_RHS+2, y_int_cut+y_int_cut+(Peak_Height-y_int_cut)/10), xycoords="data",
                      fontsize=12)

        ax2.legend()
        ax1.legend()
        ax3.legend()
        ax4.legend()
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax3.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax4.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Intensity')
        ax2.set_ylabel('Intensity')
        ax3.set_ylabel('Intensity')
        ax4.set_ylabel('Intensity')
        fig.tight_layout()



        path3=path+'/'+'Skewness_images'
        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ 'Skewness_images', exist_ok=False)


        file=filename.rsplit('.txt', 1)[0]
        fig.savefig(path3+'/'+'Diad1_skewness_{}.png'.format(file), dpi=dpi)

    df_out=pd.DataFrame(data={'filename':filename,
                              'Skewness_diad1': AS,
                              'LHS_tie_diad1': x_nearest_LHS,
                              'RHS_tie_diad1': x_nearest_RHS})

    return df_out




def assess_diad2_skewness(*,  int_cut_off=0.3, skewness='dir',path=None, filename=None, filetype=None,
                        exclude_range1=None, exclude_range2=None,
                        N_poly_bck_diad2=1, lower_baseline_diad2=[1300, 1360],
                        upper_baseline_diad2=[1440, 1470], save_fig=True, dpi=300):

    """ Assesses Skewness of Diad peaks. Useful for identifying mixed L + V phases
    (see DeVitre et al. in review)


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


    lower_baseline_diad1: list
        Region of spectra to use a baseline (LHS)

    upper_baseline_diad1: list
        Region of spectra to use a baseline (RHS)

    N_poly_bck_diad1: int
        Degree of polynomial for background fitting

    skewness: 'abs' or 'dir'
        If 'abs', gives absolute skewness (e.g., largest possible number regardless of direction)
        if 'dir' does skewness as one side over the other

    int_int_cut_off: float
        Value of intensity at which to calculate the skewness (e.g. 0.15X the peak height).


    Returns
    -----------
    pd.DataFrame wtih filename, skewness of Diad 1,
    the x position of the LH and RH tie point at intensity=int_int_cut_off

    """


# First, do the background subtraction
    y_corr_diad2, Py_base_diad2, x_diad2,  Diad_short, Py_base_diad2, Pf_baseline,  Baseline_ysub_diad2, Baseline_x_diad2, Baseline, span=remove_diad_baseline(
    path=path, filename=filename, filetype=filetype, exclude_range1=exclude_range1, exclude_range2=exclude_range2, N_poly=N_poly_bck_diad2,
    lower_range=lower_baseline_diad2, upper_range=upper_baseline_diad2, plot_figure=False)



    x_lin_baseline=np.linspace(lower_baseline_diad2[0], upper_baseline_diad2[1], 100000)
    ybase_xlin=Pf_baseline(x_lin_baseline)



# Get x and y for cubic spline
    x=x_diad2
    y=y_corr_diad2

# Fits a  cubic spline
    f2 = interp1d(x, y, kind='cubic')
    x_new=np.linspace(np.min(x), np.max(x), 100000)

    y_cub=f2(x_new)



# Use Scipy find peaks to get that cubic peak
    height=1
    peaks = find_peaks(y_cub, height)
    peak_height=peaks[1]['peak_heights']
    peak_pos = x_new[peaks[0]]

    # find max peak.  put into df because i'm lazy
    peak_df=pd.DataFrame(data={'pos': peak_pos,
                        'height': peak_height})

    df_peak_sort=peak_df.sort_values('height', axis=0, ascending=False)
    df_peak_sort_trim=df_peak_sort[0:1]
    Peak_Center=df_peak_sort_trim['pos']
    Peak_Height=df_peak_sort_trim['height']





# Find intensity cut off
    y_int_cut=Peak_Height*int_cut_off

    # Split the array into a LHS and a RHS
    LHS_y=y_cub[x_new<=Peak_Center.values]
    RHS_y=y_cub[x_new>Peak_Center.values]

    LHS_x=x_new[x_new<=Peak_Center.values]
    RHS_x=x_new[x_new>Peak_Center.values]

    # Need to flip LHS to put into the find closest function
    LHS_y_flip=np.flip(LHS_y)
    LHS_x_flip=np.flip(LHS_x)

    val=np.argmax(LHS_y_flip<y_int_cut.values)

    val2=np.argmax(RHS_y<y_int_cut.values)


    # Find nearest x unit to this value
    y_nearest_LHS=LHS_y_flip[val]
    x_nearest_LHS=LHS_x_flip[val]

    y_nearest_RHS=RHS_y[val2]
    x_nearest_RHS=RHS_x[val2]

    # Return Skewness
    LHS_Center=abs(x_nearest_LHS-Peak_Center)
    RHS_Center=abs(x_nearest_RHS-Peak_Center)

    #Added conditional to always have a ratio of the high/low
    if skewness=='abs':
        if LHS_Center.values>RHS_Center.values:
            AS=LHS_Center.values/RHS_Center.values
        else:
            AS=RHS_Center.values/LHS_Center.values
    elif skewness=='dir':
        AS=RHS_Center.values/LHS_Center.values


    if save_fig is True:

        # Make pretty figure showing background subtractoin routine
        fig, ((ax2, ax1), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

        ax2.set_title('a) Spectra')
        ax2.plot(Diad_short[:, 0], Diad_short[:, 1], '-r', label='Data')
        ax2.set_xlim([lower_baseline_diad2[0]+40, upper_baseline_diad2[1]-40])
        ax1.set_title('b) Background fit')
        ax1.plot(Diad_short[:, 0], Diad_short[:, 1], '.-c', label='Data')

        ax1.plot(Baseline[:, 0], Baseline[:, 1], '.b', label='bck')
        ax1.plot(Diad_short[:, 0], Py_base_diad2, '-k', label='bck fit')
        ax3.set_title('c) Background subtracted')
        ax3.plot( x_diad2,y_corr_diad2, '-r', label='Bck sub')

    # Adds cubic interpoloation for inspection

        ax4.set_title('d) Cubic Spline')
        ax4.plot([x_nearest_LHS, Peak_Center], [y_int_cut, y_int_cut], '-g')
        ax4.annotate(str(np.round(LHS_Center.values[0], 2)),
                     xy=(x_nearest_LHS-3, y_int_cut+(Peak_Height-y_int_cut)/10), xycoords="data",
                     fontsize=12, color='green')
        ax4.plot(x_nearest_LHS, y_nearest_LHS, '*k', mfc='green',ms=15, label='RH tie')



        ax4.plot([Peak_Center, x_nearest_RHS], [y_int_cut, y_int_cut], '-', color='grey')
        ax4.annotate(str(np.round(RHS_Center.values[0], 2)),
                     xy=(x_nearest_RHS+3, y_int_cut+(Peak_Height-y_int_cut)/10), xycoords="data",
                      fontsize=12, color='grey')
        ax4.plot(x_nearest_RHS, y_nearest_RHS, '*k', mfc='grey', ms=15, label='LH tie')




        ax4.plot(x, y, '.r')
        ax4.plot(x_new, y_cub, '-k')

        ax4.plot([Peak_Center, Peak_Center], [Peak_Height, Peak_Height],
             '*k', mfc='blue', ms=15, label='Scipy Center')


        # Add to plot

        ax4.plot(lower_baseline_diad2[0], upper_baseline_diad2[1], [y_int_cut, y_int_cut], ':r')




        ax4.set_xlim([x_nearest_LHS-10, x_nearest_RHS+10])
        ax4.set_ylim([0-10, Peak_Height.values*1.2])
        ax4.plot([Peak_Center, Peak_Center], [0, Peak_Height], ':b')



        ax4.annotate('Skewness='+str(np.round(AS[0], 2)),
                     xy=(x_nearest_RHS+2, y_int_cut+y_int_cut+(Peak_Height-y_int_cut)/10), xycoords="data",
                      fontsize=10)

        ax2.legend()
        ax1.legend()
        ax3.legend()
        ax4.legend()
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax3.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax4.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Intensity')
        ax2.set_ylabel('Intensity')
        ax3.set_ylabel('Intensity')
        ax4.set_ylabel('Intensity')
        fig.tight_layout()


        path3=path+'/'+'Skewness_images'
        if os.path.exists(path3):
            out='path exists'
        else:
            os.makedirs(path+'/'+ 'Skewness_images', exist_ok=False)


        file=filename.rsplit('.txt', 1)[0]
        fig.savefig(path3+'/'+'Diad2_skewness_{}.png'.format(file), dpi=dpi)


    df_out=pd.DataFrame(data={
                              'Skewness_diad2': AS,
                              'LHS_tie_diad2': x_nearest_LHS,
                              'RHS_tie_diad2': x_nearest_RHS})

    return df_out


def loop_diad_skewness(*, path=None, filetype=None, file_ext='.txt', skewness='abs', sort=False, exclude_str='Ne', exclude_type='.png', int_cut_off=0.15, N_poly_bck_diad1=1, exclude_range1_diad1=None,
exclude_range2_diad1=None, lower_baseline_diad1=[1220, 1250], upper_baseline_diad1=[1300, 1350],
N_poly_bck_diad2=1,  exclude_range1_diad2=None, exclude_range2_diad2=None, lower_baseline_diad2=[1300, 1360],
                        upper_baseline_diad2=[1440, 1470], save_fig=True, dpi=300):
    """ Loops over all supplied files to calculate skewness for multiple spectra.


    Parameters
    -----------
    path: str
        Folder user wishes to read data from
    sort: bool
        If true, sorts files alphabetically
    file_ext: str
        File format. Default txt, could also enter csv etc.
    exclude_str: str
        Excludes files with this string in their name. E.g. if exclude_str='Ne' it will exclude Ne lines
    exclude_type: str
        Excludes files of this type, e.g. exclude_type='png' gets rid of image files.


    lower_baseline_diad1, lower_baseline_diad2: list
        Region of spectra to use a baseline (LHS) for diad 1 and diad2

    upper_baseline_diad1, upper_baseline_diad2: list
        Region of spectra to use a baseline (RHS) for diad 1 and diad 2

    N_poly_bck_diad1, N_poly_bck_diad2: int
        Degree of polynomial for background fitting

    skewness: 'abs' or 'dir'
        If 'abs', gives absolute skewness (e.g., largest possible number regardless of direction)
        if 'dir' does skewness as one side over the other

    int_cut_off: float
        Value of intensity at which to calculate the skewness (e.g. 0.15X the peak height).


    Returns
    -----------
    pd.DataFrame wtih filename, skewness of Diad 1,
    the x position of the LH and RH tie point at intensity=int_int_cut_off



    """

    Diad_files=get_diad_files(path=path, sort=sort, file_ext=file_ext, exclude_str=exclude_str, exclude_type=exclude_type)

    df_diad1 = pd.DataFrame([])
    df_diad2 = pd.DataFrame([])
    for i in range(0, len(Diad_files)):

        filename=Diad_files[i]
        print('working on file #'+str(i))

        data_diad1=assess_diad1_skewness(int_cut_off=int_cut_off,
                            skewness=skewness, path=path, filename=filename,
                            filetype=filetype,
                            exclude_range1=exclude_range1_diad1,
                            exclude_range2=exclude_range2_diad1,
                            N_poly_bck_diad1=N_poly_bck_diad1,
                            lower_baseline_diad1=lower_baseline_diad1,
                            upper_baseline_diad1=upper_baseline_diad1,
                            save_fig=save_fig,
                            dpi=dpi)

        data_diad2=assess_diad2_skewness(int_cut_off=int_cut_off,
                            skewness=skewness, path=path, filename=filename,
                            filetype=filetype,
                            exclude_range1=exclude_range1_diad2,
                            exclude_range2=exclude_range2_diad2,
                            N_poly_bck_diad2=N_poly_bck_diad2,
                            lower_baseline_diad2=lower_baseline_diad2,
                            upper_baseline_diad2=upper_baseline_diad2,
                            save_fig=save_fig,
                            dpi=dpi)
        print(type(data_diad1))

        df_diad1 = pd.concat([df_diad1, data_diad1], axis=1)
        df_diad2 =  pd.concat([df_diad2, data_diad2], axis=1)
        df_combo=pd.concat([df_diad1, df_diad2], axis=1)

        return df_combo
        #df_diad2.concat(data_diad2)




