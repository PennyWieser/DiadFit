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

encode="ISO-8859-1"

# Requirements
# lmfit


## Functions for getting file names
def get_Ne_files(path, ID_str='Ne', file_fmt='txt', exclude_str=None, sort=True):
    """ This function takes a user path, and extracts all files which contain the ID_str

    Parameters
    -----------

    path: str
        Folder user wishes to read data from
    sort: bool
        If true, sorts files alphabetically
    ID_str: str
        Finds all files containing this string (e.g. Ne, NE)
    exclude_str: str
        Excludes files with this string in the name
    file_fmt: str
        Gets all files of this format only (e.g. txt)


    Returns
    -----------
    Returns file names as a list.

    """
    Allfiles = [f for f in listdir(path) if isfile(join(path, f))]
    Ne_files=[item for item in Allfiles if ID_str in item and file_fmt in item and exclude_str not in item]

    if sort is True:
        Ne_files=sorted(Ne_files)
    return Ne_files


def get_diad_files(path, sort=True, file_fmt='txt', exclude_str='Ne', exclude_type='.png'):
    """ This function takes a user path, and extracts all files which dont contain the excluded string and type

    Parameters
    -----------

    path: str
        Folder user wishes to read data from
    sort: bool
        If true, sorts files alphabetically
    file_fmt: str
        File format. Default txt, could also enter csv etc.
    exclude_str: str
        Excludes files with this string in their name. E.g. if exclude_str='Ne' it will exclude Ne lines
    exclude_type: str
        Excludes files of this type, e.g. exclude_type='png' gets rid of image files.

    Returns
    -----------
    Returns file names as a list.

    """
    exclude=exclude_type
    Allfiles = [f for f in listdir(path) if isfile(join(path, f))]
    Diad_files=[item for item in Allfiles if exclude_str not in item and file_fmt in item and exclude not in item]
    if sort is True:

        Diad_files2=sorted(Diad_files)
    else:
        Diad_files2=Diad_files
    return Diad_files2


def get_all_txt_files(path):
    """ This function takes a user path, and gets all the .txt. files in that path.

    Parameters
    -----------
    path: str
        Folder user wishes to read data from
    """

    Allfiles_all = [f for f in listdir(path) if isfile(join(path, f))]
    # Use only txt files
    type(Allfiles_all)
    All_files=[]
    for file in Allfiles_all:
        if '.txt' in file and 'pandas' not in file:
            All_files.append(format(file))
    return All_files


## Functions to just simply get data to plot up
def get_data(*, path=None, filename, filetype='Witec_ASCII'):
    """
    Extracts data as a np.array from user file of differen types
    """
    if filetype == 'headless_txt':
        df=pd.read_csv(path+'/'+filename, sep="\t", header=None )

    if filetype=='Witec_ASCII':
        df=read_witec_to_df(path=path, filename=filename)

    if filetype=='Renishaw_txt':
        df_long=pd.read_csv(path+'/'+filename, sep="\t" )
        df=Ne_df_long.iloc[:, 0:2]

    if filetype=='HORIBA_txt':
        df=read_HORIBA_to_df(path=path, filename=filename)

    if filetype=='headless_csv':
        df=pd.read_csv(path+str('/')+filename, header=None)

    df_in=np.array(df)

    return df_in




## Importing files
def read_HORIBA_to_df(*,  path=None, filename):
    """ This function takes in a HORIBA .txt. file with headers with #, and looks down to the row where Data starts (no #),
    and saves this to a new file called pandas_.... old file. It exports the data as a pandas dataframe

    Parameters
    -----------

    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read


    """
    path2=path+'/'+ 'Peak_fits_txt'
    if os.path.exists(path2):
        a='path exists'
    else:
        os.makedirs(path+'/'+ 'Peak_fits_txt', exist_ok=False)
        print('Ive made a new folder to store your intermediate txt files in')

    if path is None:
        fr = open(filename, 'r', encoding=encode)

        fw=open('pandas2_'+filename, 'w')
    else:
        fr = open(path+'/'+filename, 'r', encoding=encode)
        fw= open(path+'/'+'Peak_fits_txt'+'/'+'pandas2_'+filename, 'w')

    if fr.readline().startswith('#Acq. time'):
        out='HORIBA txt file recognised'
    else:
        raise TypeError('Not a HORIBA txt file with headers')
    while True:
        l=fr.readline()
        if not l.startswith('#'):

            break

    for line in fr:
        fw.write(line)

    fw.close()
    fr.close()
    if path is None:
        print(filename)
        df=pd.read_csv('pandas2_'+filename, sep="\t", header=None)
    else:
        print(filename)
        df=pd.read_csv(path+'/'+'Peak_fits_txt'+'/'+'pandas2_'+filename, sep="\t", header=None)

        return df

def read_witec_to_df(*,  path=None, filename):
    """ This function takes in a WITec ASCII.txt. file with metadata mixed with data, and looks down to the row where Data starts,
    and saves this to a new file called pandas_.... old file. It exports the data as a pandas dataframe

    Parameters
    -----------

    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read

    """
    path2=path+'/'+ 'Peak_fits_txt'
    if os.path.exists(path2):
        a='path exists'
    else:
        os.makedirs(path+'/'+ 'Peak_fits_txt', exist_ok=False)
        print('Ive made a new folder to store your intermediate txt files in')

    if path is None:
        fr = open(filename, 'r', encoding=encode)
        fw=open('pandas2_'+filename, 'w')
    else:
        fr = open(path+'/'+filename, 'r', encoding=encode)
        fw= open(path+'/'+'Peak_fits_txt'+'/'+'pandas2_'+filename, 'w')

    if fr.readline().startswith('//Exported ASCII'):
        out='ASCI file recognised'
    else:
        raise TypeError('file not an ASCI file')

    while True:
        l=fr.readline()

        if l.startswith('[Data]'):

            break

    for line in fr:
        fw.write(line)

    fw.close()
    fr.close()
    if path is None:
        print(filename)
        df=pd.read_csv('pandas2_'+filename, sep="\t")
    else:
        print(filename)
        df=pd.read_csv(path+'/'+'Peak_fits_txt'+'/'+'pandas2_'+filename, sep="\t")

    return df


## Functions for plotting and fitting Ne lines

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
    print('Found peaks at wavenumber=')
    print(x[peaks[0]])

    n_peaks=6
    height = peaks[1]['peak_heights'] #list of the heights of the peaks
    peak_pos = x[peaks[0]] #list of the peaks positions
    df=pd.DataFrame(data={'pos': peak_pos,
                        'height': height})

    # Find bigest peaks,
    df_sort_Ne=df.sort_values('height', axis=0, ascending=False)
    df_sort_Ne_trim=df_sort_Ne[0:n_peaks]
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
lower_bck_pk2=[-44.2, -22], upper_bck1_pk2=[15, 50], upper_bck2_pk2=[50, 51],
amplitude=100, plot_figure=True, print_report=False, loop=False, x_range=100, y_range=1000):


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

    x_range: flt, int
        How much x range outside selected baseline the baseline selection plot shows.
    y_range: flt, int
        How much above the baseline position is shown on the y axis.

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

    # Fit the 1117 peak
    cent_1117, Ne_1117_reg_x_plot, Ne_1117_reg_y_plot, Ne_1117_reg_x, Ne_1117_reg_y, xx_1117, result_1117, error_1117, result_1117_origx, comps = fit_1117(x_1117, y_corr_1117, x_span=x_span_1, Ne_center=Ne_center_1, LH_offset_mini=LH_offset_mini, peaks_1117=peaks_1, amplitude=amplitude, print_report=print_report)


    # Fit the 1447 peak
    cent_1447, Ne_1447_reg_x_plot, Ne_1447_reg_y_plot, Ne_1447_reg_x, Ne_1447_reg_y, xx_1447, result_1447, error_1447, result_1447_origx = fit_1447(x_1447, y_corr_1447, x_span=x_span_2,  Ne_center=Ne_center_2, amplitude=amplitude, print_report=print_report)


    # Calculate difference between peak centers, and Delta Ne
    DeltaNe=cent_1447-cent_1117
    DeltaNe_ideal=330.477634
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
        ax0.set_ylim([min(Ne_short_1447[:,1])-10, min(Ne_short_1447[:,1])+y_range])
        ax0.set_xlim([min(Ne_short_1447[:,0])-20, max(Ne_short_1447[:,0])+y_range])
        #ax0.set_ylim([mean_baseline-50, mean_baseline+50])

        ax1.plot(Ne_short_1117[:,0], Py_base_1117, '-k')
        ax1.plot(Ne_short_1117[:,0], Ne_short_1117[:,1], '-r')
        ax1.plot(Baseline_x_1117, Baseline_ysub_1117, '.b', ms=6, label='Selected background')

        std_baseline=np.std(Py_base_1117)
        ax1.set_ylim([min(Ne_short_1117[:,1])-10, min(Ne_short_1117[:,1])+y_range])

        ax1.set_title('Peak1: 1117 background fitting')
        ax1.set_xlabel('Wavenumber')
        ax1.set_ylabel('Intensity')

        #Testing
        ax0.legend()
        ax1.legend()
        ax0.plot(Ne[:,0], Ne[:,1], '-', color='grey', zorder=0)
        ax1.plot(Ne[:,0], Ne[:,1], '-', color='grey', zorder=0)
        ax0.set_xlim([min(Ne_short_1447[:,0])-x_range, max(Ne_short_1447[:,0])+x_range])
        ax1.set_xlim([min(Ne_short_1117[:,0])-x_range, max(Ne_short_1117[:,0])+x_range])

        ax2.plot(Ne_1447_reg_x_plot, Ne_1447_reg_y_plot, 'xb', label='data')
        ax2.plot(Ne_1447_reg_x, Ne_1447_reg_y, '+k', label='data')
        ax2.plot(xx_1447, result_1447, 'r-', label='interpolated fit')
        ax2.set_title('1447 peak fitting')
        ax2.set_xlabel('Wavenumber')
        ax2.set_ylabel('Intensity')
        ax2.set_xlim([cent_1447-5, cent_1447+5])


        ax3.plot(Ne_1117_reg_x_plot, Ne_1117_reg_y_plot, 'xb', label='data')
        ax3.plot(Ne_1117_reg_x, Ne_1117_reg_y, '+k', label='data')

        ax3.set_title('1117 peak fitting')
        ax3.set_xlabel('Wavenumber')
        ax3.set_ylabel('Intensity')
        ax3.plot(xx_1117, comps.get('p1_'), '-r', label='p1')
        if peaks_1>1:
            ax3.plot(xx_1117, comps.get('p2_'), '-c', label='p2')
        ax3.plot(xx_1117, result_1117, 'g-', label='best fit')
        ax3.legend()
        ax3.set_xlim([cent_1117-5,cent_1117+5 ])

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





## Functions for plotting and fitting Diads


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

    # if discard_diad1 is not None:
    #     df_pks_diad1=df_pks_diad1[~(df_pks_diad1['pos'].between(discard_diad1-0.1, discard_diad1+0.1))]
    #
    # if discard_diad2 is not None:
    #     df_pks_diad2=df_pks_diad2[~(df_pks_diad2['pos'].between(discard_diad2-0.1, discard_diad2+0.1))]

    df_sort_diad1=df_pks_diad1.sort_values('height', axis=0, ascending=False)
    df_sort_diad1_trim=df_sort_diad1[0:n_peaks_diad1]

    df_sort_diad2=df_pks_diad2.sort_values('height', axis=0, ascending=False)
    df_sort_diad2_trim=df_sort_diad2[0:n_peaks_diad2]

    if any(df_sort_diad2_trim['pos'].between(1385, 1391)):
        diad_2_peaks=tuple(df_sort_diad2_trim['pos'].values)
    else:
        if n_peaks_diad2==1:
            print('Couldnt find diad2, set peak guess to 1389.1')
            diad_2_peaks=np.array([1389.1])
        if n_peaks_diad2==2:
            print('Couldnt find diad2, set peak guess to 1389.1, 1410')
            diad_2_peaks=np.array([1389.1, 1410])


    if any(df_sort_diad1_trim['pos'].between(1283, 1290)):
        diad_1_peaks=tuple(df_sort_diad1_trim['pos'].values)
    else:
        print('Couldnt find diad1, set peak guess to 1286')
        diad_1_peaks=np.array([1286.1])



    print('Using initial estimates: Diad1+HB=' +str(np.round(diad_1_peaks, 1)) + ', Diad2+HB=' + str(np.round(diad_2_peaks, 1)))


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



def remove_diad_baseline(*, path=None, filename, filetype='Witec_ASCII',
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
    print(Std_Baseline)

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



def add_peak(*, prefix=None, center=None, min_cent=None, max_cent=None,  amplitude=100, sigma=0.2):
    """
    This function iteratively adds peaks for lmfit
    """
    Model_combo=VoigtModel(prefix=prefix) ##+ConstantModel(prefix=prefix) Stops getting results
    peak =  Model_combo
    pars = peak.make_params()

    if min_cent is not None and max_cent is not None:
        pars[prefix + 'center'].set(center, min=min_cent, max=max_cent)
    else:
        pars[prefix + 'center'].set(center)


    pars[prefix + 'amplitude'].set(amplitude, min=0)
    pars[prefix + 'sigma'].set(sigma, min=0)
    return peak, pars


def fit_gaussian_voigt_diad1(*, path=None, filename=None,
                                xdat=None, ydat=None,
                                peak_pos_voigt=(1263, 1283),
                                peak_pos_gauss=None,
                                amplitude=100, span=None,
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
        if type(peak_pos_voigt) is float or type(peak_pos_voigt) is int:
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
        params = model.make_params(center=peak_pos_gauss)
        params.add('amplitude', value=50, min=0)

        rough_peak_positions = peak_pos_voigt
        # If you want a Gaussian background
        if type(peak_pos_voigt) is float or type(peak_pos_voigt) is int:
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
                if type(peak_pos_voigt) is float or type(peak_pos_voigt) is int:
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
        print(lowerpeak)
        print(upperpeak)
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
                    amplitude=100, peak_pos_gauss=(1400), span=None, plot_figure=True, dpi=200):


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

    if peak_pos_gauss is None:
        # Fit just as many peaks as there are peak_pos_voigt

        if type(peak_pos_voigt) is float or type(peak_pos_voigt) is int:
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
                model_prel = VoigtModel(prefix='lzp_')#+ ConstantModel(prefix='c1')
                pars2 = model_prel.make_params()
                pars2['lzp_'+ 'amplitude'].set(amplitude, min=0)
                pars2['lzp_'+ 'center'].set(peak_pos_voigt[0])


                init = model_prel.eval(pars2, x=xdat)
                result_prel = model_prel.fit(ydat, pars2, x=xdat)
                comps_prel = result_prel.eval_components()

                Peakp_Cent=result_prel.best_values.get('lzp_center')
                Peakp_Area=result_prel.best_values.get('lzp_amplitude')
                Peakp_HW=result_prel.params.get('lzp_fwhm')



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
                pars[prefix+ 'fwhm'].set(Peakp_HW, min=Peakp_HW/5, max=Peakp_HW*2)

                model_F=model1+peak
                pars1.update(pars)
                params=pars1

            if len(peak_pos_voigt)==3:
                print('got here')
                low_peak=np.min(peak_pos_voigt)
                med_peak=np.median(peak_pos_voigt)
                high_peak=np.max(peak_pos_voigt)
                peak_pos_left=np.array([low_peak, high_peak])

                model = VoigtModel(prefix='lz1_')#+ ConstantModel(prefix='c1')
                params = model.make_params()
                params['lz1_'+ 'amplitude'].set(amplitude, min=0)
                params['lz1_'+ 'center'].set(med_peak)

                for i, cen in enumerate(peak_pos_left):

                    peak, pars = add_peak(prefix='lz%d_' % (i+2), center=cen, min_cent=cen-3, max_cent=cen+3 )
                    model = peak+model
                    params.update(pars)

                model_F=model




    # Same, but also with a Gaussian Background
    if peak_pos_gauss is not None:

        model = GaussianModel(prefix='bkg_')
        params = model.make_params(center=peak_pos_gauss)
        params.add('amplitude', value=amplitude, min=0)

        rough_peak_positions = peak_pos_voigt
        # If you want a Gaussian background
        if type(peak_pos_voigt) is float or type(peak_pos_voigt) is int:
                peak, pars = add_peak(prefix='lz1_', center=peak_pos_voigt, amplitude=amplitude)
                model = peak+model
                params.update(pars)
        else:

            if len(peak_pos_voigt)==1:
                peak, pars = add_peak(prefix='lz1_', center=cen, amplitude=amplitude)
                model = peak+model
                params.update(pars)

            if len(peak_pos_voigt)>1:
                print('got here')
                for i, cen in enumerate(peak_pos_voigt):

                    peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen)
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


    x_lin=np.linspace(span[0], span[1], 2000)
    y_best_fit=result.eval(x=x_lin)
    components=result.eval_components(x=x_lin)



    # Work out what peak is what
    if type(peak_pos_voigt) is float and type(peak_pos_voigt) is int:
        if len(peak_pos_voigt)==1:
            Peak2_Cent=None
            Peak3_Cent=None
            ax1_xlim=[peak_pos_voigt[0]-15, peak_pos_voigt[0]+15]
            ax2_xlim=[peak_pos_voigt[0]-15, peak_pos_voigt[0]+15]
    if type(peak_pos_voigt) is not float and type(peak_pos_voigt) is not int:
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

            ax1_xlim=[peak_pos_voigt[0]-15, peak_pos_voigt[0]+30]
            ax2_xlim=[peak_pos_voigt[0]-15, peak_pos_voigt[0]+30]

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
            ax2.plot(x_lin, components.get('bkg_'), '-c',linewidth=2,  label='Gaussian bck')
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
peak_pos_voigt=(1369, 1387, 1408),peak_pos_gauss=(1380), plot_figure=True, dpi=200):

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


    """


    #Fit baseline

    y_corr_diad2, Py_base_diad2, x_diad2,  Diad_short, Py_base_diad2, Pf_baseline, Baseline_ysub_diad2, Baseline_x_diad2, Baseline, span=remove_diad_baseline(
   path=path, filename=filename, filetype=filetype, exclude_range1=exclude_range1, exclude_range2=exclude_range2, N_poly=N_poly_bck_diad2,
    lower_range=lower_baseline_diad2, upper_range=upper_baseline_diad2, plot_figure=False)



    result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim, peak_pos_gauss,residual_diad2_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_diad2(path=path, filename=filename,
                    xdat=x_diad2, ydat=y_corr_diad2, amplitude=amplitude,
                    peak_pos_voigt=peak_pos_voigt,
                    peak_pos_gauss=peak_pos_gauss, span=span, plot_figure=False)


    # Get diad data to plot
    Spectra_df=get_data(path=path, filename=filename, filetype=filetype)

    Spectra=np.array(Spectra_df)


    # Make nice figure

    figure_mosaic="""
    AB
    DC
    EE
    """
    fig,axes=plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(10, 16))

    # Plot best fit on the LHS, and individual fits on the RHS at the top

    axes['A'].plot( x_lin ,y_best_fit, '-g', linewidth=1)
    axes['A'].plot(xdat, ydat,  '.k', label='data')
    axes['A'].legend()
    axes['A'].set_ylabel('Intensity')
    axes['A'].set_xlabel('Wavenumber')
    axes['A'].set_xlim(ax1_xlim)
    axes['A'].set_title('a) Overall Best Fit')

    axes['B'].plot(xdat, ydat, '.k', label='data')
    if peak_pos_gauss is not None:
        axes['B'].plot(x_lin, components.get('bkg_'), '-m', label='Gaussian bck', linewidth=1)
    if len(peak_pos_voigt)==2:
        axes['B'].plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='Peak2')
    if len(peak_pos_voigt)>2:
         axes['B'].plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='Peak2')
         axes['B'].plot(x_lin, components.get('lz3_'), '-k', linewidth=2, label='Peak3')
    axes['B'].plot(x_lin, components.get('lz1_'), '-b', linewidth=2, label='Peak1')


    #ax2.plot(xdat, result.best_fit, '-g', label='best fit')
    axes['B'].legend()
    fitspan=max(y_best_fit)-min(y_best_fit)
    axes['B'].set_ylim([min(y_best_fit)-fitspan/5, max(y_best_fit)+fitspan/5])
    axes['B'].set_ylabel('Intensity')
    axes['B'].set_xlabel('Wavenumber')
    axes['B'].set_title('b) Fit Components')
    axes['B'].set_xlim(ax2_xlim)



    # Residual on plot C
    axes['C'].set_title('d) Residual')
    axes['C'].plot(xdat_inrange, residual_diad2_coords, 'ok', mfc='c' )
    axes['C'].plot(xdat_inrange, residual_diad2_coords, '-c' )
    axes['C'].set_ylabel('Residual')
    axes['C'].set_xlabel('Wavenumber')
    axes['C'].set_xlim(ax1_xlim)
    axes['C'].set_xlim(ax2_xlim)
    #axes['C'].set_xlim(ax1_xlim)

    #Background fit on plot D
    #Background fit on plot D
    axes['D'].set_title('c) Background fit')
    axes['D'].plot(Diad_short[:, 0], Diad_short[:, 1], '-c', label='Data')

    axes['D'].set_ylim([
    np.min(Baseline[:, 1])-10*np.std(Baseline[:, 1]),
    np.max(Baseline[:, 1])+10*np.std(Baseline[:, 1])
    ] )
    axes['D'].set_ylabel('Intensity')
    axes['D'].set_xlabel('Wavenumber')
    ybase_xlin=Pf_baseline(x_lin)
    if peak_pos_gauss is not None:

        axes['D'].plot(x_lin, components.get('bkg_')+ybase_xlin, '-m', label='Gaussian bck', linewidth=2)

    axes['D'].plot( x_lin ,y_best_fit+ybase_xlin, '-g', linewidth=2, label='best fit')
    axes['D'].plot(x_lin, components.get('lz1_')+ybase_xlin, '-b', label='Peak1', linewidth=1)
    if len(peak_pos_voigt)>1:
        axes['D'].plot(x_lin, components.get('lz2_')+ybase_xlin, '-r', label='Peak2', linewidth=1)
    if len(peak_pos_voigt)>2:
         axes['D'].plot(x_lin, components.get('lz3_')+ybase_xlin, '-k', linewidth=1, label='Peak3')

    axes['D'].set_title('c) Background fit')


    axes['D'].plot(Baseline[:, 0], Baseline[:, 1], '.b', label='bck')
    axes['D'].plot(Diad_short[:, 0], Py_base_diad2, '-k', label='bck fit')
    axes['D'].legend()




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
    fig.savefig(path3+'/'+'Diad1_Fit_{}.png'.format(file), dpi=dpi)




    return df_out, result, y_best_fit, x_lin
   #  y_corr_diad2, Py_base_diad2, x_diad2,  Ne_short_diad2, Py_base_diad2, Baseline_ysub_diad2, Baseline_x_diad2=remove_diad_baseline(
   # path=path, filename=filename,filetype=filetype,
   #                     exclude_range1=exclude_range1, exclude_range2=exclude_range2,
   #                      N_poly=N_poly_bck_diad2,
   #  lower_range=lower_baseline_diad2, upper_range=upper_baseline_diad2)
   #
   #  result_diad2, df_out, best_fit_diad2, xdat_diad2=fit_gaussian_voigt_diad2(path=path, filename=filename, xdat=x_diad2, ydat=y_corr_diad2,
   #                  peak_pos_voigt=peak_pos_voigt,amplitude=amplitude,
   #                  peak_pos_gauss=peak_pos_gauss, plot_figure=plot_figure, dpi=dpi)
   #
   #  return df_out, result_diad2, best_fit_diad2, xdat_diad2


def fit_diad_1_w_bck(*, path=None, filename=None, filetype='headless_txt',
exclude_range1=None, exclude_range2=None,
N_poly_bck_diad1=2, lower_baseline_diad1=[1170, 1220],
upper_baseline_diad1=[1330, 1350], peak_pos_voigt=(1263, 1283),
peak_pos_gauss=(1270), amplitude=100, plot_figure=True, dpi=200):
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


    """

    # Fit baseline
    #y_corr, Py_base, x,  Diad_short, Py_base, Baseline_ysub, Baseline_x, Baseline





    y_corr_diad1, Py_base_diad1, x_diad1,  Diad_short, Py_base_diad1, Pf_baseline,  Baseline_ysub_diad1, Baseline_x_diad1, Baseline, span=remove_diad_baseline(
   path=path, filename=filename, filetype=filetype, exclude_range1=exclude_range1, exclude_range2=exclude_range2, N_poly=N_poly_bck_diad1,
    lower_range=lower_baseline_diad1, upper_range=upper_baseline_diad1, plot_figure=False)



    result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim, peak_pos_gauss,residual_diad1_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_diad1(path=path, filename=filename,
                    xdat=x_diad1, ydat=y_corr_diad1, amplitude=amplitude,
                    peak_pos_voigt=peak_pos_voigt,
                    peak_pos_gauss=peak_pos_gauss, span=span, plot_figure=False)


    # Get diad data to plot
    Spectra_df=get_data(path=path, filename=filename, filetype=filetype)

    Spectra=np.array(Spectra_df)


    # Make nice figure

    figure_mosaic="""
    AB
    DC
    EE
    """
    fig,axes=plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(10, 16))

    # Plot best fit on the LHS, and individual fits on the RHS at the top

    axes['A'].plot( x_lin ,y_best_fit, '-g', linewidth=1, label='best fit')
    axes['A'].plot(xdat, ydat,  '.k', label='data')
    axes['A'].legend()

    axes['B'].plot(xdat, ydat, '.k')

    if len(peak_pos_voigt)>1:
        axes['B'].plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='Peak2')
    axes['B'].plot(x_lin, components.get('lz1_'), '-b', linewidth=2, label='Peak1')
    if peak_pos_gauss is not None:
        axes['B'].plot(x_lin, components.get('bkg_'), '-m', label='Gaussian bck', linewidth=2)
    #ax2.plot(xdat, result.best_fit, '-g', label='best fit')
    axes['B'].legend()
    fitspan=max(y_best_fit)-min(y_best_fit)
    axes['B'].set_ylim([min(y_best_fit)-fitspan/5, max(y_best_fit)+fitspan/5])

    axes['A'].set_ylabel('Intensity')
    axes['A'].set_xlabel('Wavenumber')
    axes['B'].set_ylabel('Intensity')
    axes['B'].set_xlabel('Wavenumber')

    axes['A'].set_xlim(ax1_xlim)
    axes['B'].set_xlim(ax2_xlim)
    axes['A'].set_title('a) Overall Best Fit')
    axes['B'].set_title('b) Fit Components')

    # Residual on plot C
    axes['C'].set_title('d) Residual')
    axes['C'].plot(xdat_inrange, residual_diad1_coords, 'ok', mfc='c' )
    axes['C'].plot(xdat_inrange, residual_diad1_coords, '-c' )
    axes['C'].set_ylabel('Residual')
    axes['C'].set_xlabel('Wavenumber')
    axes['C'].set_xlim(ax1_xlim)
    axes['C'].set_xlim(ax2_xlim)

    #Background fit on plot D
    axes['D'].set_title('c) Background fit')
    axes['D'].plot(Diad_short[:, 0], Diad_short[:, 1], '-c', label='Data')

    axes['D'].set_ylim([
    np.min(Baseline[:, 1])-10*np.std(Baseline[:, 1]),
    np.max(Baseline[:, 1])+10*np.std(Baseline[:, 1])
    ] )
    axes['D'].set_ylabel('Intensity')
    axes['D'].set_xlabel('Wavenumber')

    ybase_xlin=Pf_baseline(x_lin)
    if peak_pos_gauss is not None:

        axes['D'].plot(x_lin, components.get('bkg_')+ybase_xlin, '-m', label='Gaussian bck', linewidth=2)

    axes['D'].plot( x_lin ,y_best_fit+ybase_xlin, '-g', linewidth=2)
    axes['D'].plot(x_lin, components.get('lz1_')+ybase_xlin, '-b', label='Peak1', linewidth=1)
    if len(peak_pos_voigt)>1:
        axes['D'].plot(x_lin, components.get('lz2_')+ybase_xlin, '-r', label='Peak2', linewidth=1)




    axes['D'].plot(Baseline[:, 0], Baseline[:, 1], '.b', label='bck')
    axes['D'].plot(Diad_short[:, 0], Py_base_diad1, '-k', label='bck fit')
    axes['D'].legend()

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
    print(path)
    file=filename.rsplit('.txt', 1)[0]
    fig.savefig(path3+'/'+'Diad1_Fit_{}.png'.format(file), dpi=dpi)




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

def fit_carbonate_peak(*,path=None, filename=None, filetype='Witec_ASCII',
lower_range=[1030, 1050], upper_range=[1140, 1200], amplitude=1000,
exclude_range=None,
N_poly=2, outlier_sigma=12, cent=1090, plot_figure=True, dpi=100,
height = 20, threshold = 1, distance = 5, prominence=1, width=3,
N_peaks=2, fit_carbonate=True):

    if fit_carbonate is True:

        # read file
        if filetype == 'headless_txt':
            Spectra_df=pd.read_csv(path+'/'+filename, sep="\t", header=None )

        if filetype=='Witec_ASCII':
            Spectra_df=read_witec_to_df(path=path, filename=filename)

        if filetype=='Renishaw_txt':
            Spectra_df_long=pd.read_csv(path+'/'+filename, sep="\t" )
            Spectra_df=Spectra_df_long.iloc[:, 0:2]

        if filetype=='HORIBA_txt':
            Spectra_df=read_HORIBA_to_df(path=path, filename=filename)

        Spectra_in=np.array(Spectra_df)
        if exclude_range is not None:
             Spectra=Spectra_in[ (Spectra_in[:, 0]<exclude_range[0]) | (Spectra_in[:, 0]>exclude_range[1]) ]
        else:
            Spectra=Spectra_in


        # Start at minimum value of spectra on the Cornell WITEC - alter for other machines

        lower_0baseline=lower_range[0]
        upper_0baseline=lower_range[1]
        lower_1baseline=upper_range[0]
        upper_1baseline=upper_range[1]

        Spectra_short=Spectra[ (Spectra[:,0]>lower_0baseline) & (Spectra[:,0]<upper_1baseline) ]
        Spectra_plot=Spectra[ (Spectra[:,0]>lower_0baseline-50) & (Spectra[:,0]<upper_1baseline+50) ]

        # Find other peaks
        y=Spectra_plot[:, 1]
        x=Spectra_plot[:, 0]
        peaks = find_peaks(y,height = height, threshold = threshold, distance = distance, prominence=prominence, width=width)

        height = peaks[1]['peak_heights'] #list of the heights of the peaks
        peak_pos = x[peaks[0]] #list of the peaks positions
        df_sort=pd.DataFrame(data={'pos': peak_pos,
                            'height': height})

        df_peak_sort=df_sort.sort_values('height', axis=0, ascending=False)
        df_peak_sort_short=df_peak_sort[0:N_peaks]
        print('Find Peaks also found major peaks at:')
        print(df_peak_sort_short)

        # Get actual baseline
        Baseline_with_outl=Spectra_short[
        ((Spectra_short[:, 0]<upper_0baseline) &(Spectra_short[:, 0]>lower_0baseline))
            |
        ((Spectra_short[:, 0]<upper_1baseline) &(Spectra_short[:, 0]>lower_1baseline))]


        # Calculates the median for the baseline and the standard deviation
        Median_Baseline=np.mean(Baseline_with_outl[:, 1])
        #print(Median_Baseline)
        Std_Baseline=np.std(Baseline_with_outl[:, 1])
        #print(Std_Baseline)

        # Removes any points in the baseline outside of 2 sigma (helps remove cosmic rays etc).
        Baseline=Baseline_with_outl[(Baseline_with_outl[:, 1]<Median_Baseline+outlier_sigma*Std_Baseline)
                                    &
                                    (Baseline_with_outl[:, 1]>Median_Baseline-outlier_sigma*Std_Baseline)
                                ]
        #print(Std_Baseline)




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

        #print(result0.fit_report(min_correl=0.5))


        xx_carb=np.linspace(min(x), max(x), 2000)

        y_carb=result0.eval(x=xx_carb)

        # Plotting what its doing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        ax1.plot(df_peak_sort_short['pos'], df_peak_sort_short['height'], '*k')
        ax1.plot([1090, 1090], [min(Spectra_short[:, 1]), max(Spectra_short[:, 1])], ':r')
        ax1.set_title('Background fit')
        ax1.plot(Spectra_plot[:, 0], Spectra_plot[:, 1], '-r')
        ax1.plot(Baseline[:, 0], Baseline[:, 1], 'ok', mfc='b')

        ax1.set_ylim([min(Spectra_short[:, 1]), max(Spectra_short[:, 1])])

        ax2.set_title('Back sub peak fit')

        ax2.plot(xx_carb, y_carb, '-k')

        ax2.plot(x, y_corr, 'ok', mfc='red')
        ax2.set_ylim([min(y_carb)-0.5*(max(y_carb)-min(y_carb)),
                    max(y_carb)+0.1*max(y_carb),
        ])

        ax1.plot(Spectra_short[:, 0], Py_base, '-k')
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
    return df, xx_carb, y_carb, result0



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


def calculate_density_cornell(temp='SupCrit', Split=None):
    """ Calculates density for Cornell densimeters"""

    #if temp is "RoomT":
    LowD_RT=-38.34631 + 0.3732578*Split
    HighD_RT=-41.64784 + 0.4058777*Split- 0.1460339*(Split-104.653)**2

    # IF temp is 37
    LowD_SC=-38.62718 + 0.3760427*Split
    MedD_SC=-47.2609 + 0.4596005*Split+ 0.0374189*(Split-103.733)**2-0.0187173*(Split-103.733)**3
    HighD_SC=-42.52782 + 0.4144277*Split- 0.1514429*(Split-104.566)**2

    df=pd.DataFrame(data={'Preferred D': 0,
     'in range': 'Y',
                            'Notes': 'not in range',
                            'LowD_RT': LowD_RT,
                            'HighD_RT': HighD_RT,
                            'LowD_SC': LowD_SC,
                            'MedD_SC': MedD_SC,
                            'HighD_SC': HighD_SC,
                            'Temperature': temp,
                            'Splitting': Split,

                            })

    roomT=df['Temperature']=="RoomT"
    SupCrit=df['Temperature']=="SupCrit"
    # Range for SC low density
    min_lowD_SC_Split=df['Splitting']>=102.72
    max_lowD_SC_Split=df['Splitting']<=103.16
    # Range for SC med density
    min_MD_SC_Split=df['Splitting']>103.16
    max_MD_SC_Split=df['Splitting']<=104.28
    # Range for SC high density
    min_HD_SC_Split=df['Splitting']>=104.28
    max_HD_SC_Split=df['Splitting']<=104.95
    # Range for Room T low density
    min_lowD_RoomT_Split=df['Splitting']>=102.734115670188
    max_lowD_RoomT_Split=df['Splitting']<=103.350311768435
    # Range for Room T high density
    min_HD_RoomT_Split=df['Splitting']>=104.407308904012
    max_HD_RoomT_Split=df['Splitting']<=105.1
    # Impossible densities, room T
    Imposs_lower_end=(df['Splitting']>103.350311768435) & (df['Splitting']<103.88)
    # Impossible densities, room T
    Imposs_upper_end=(df['Splitting']<104.407308904012) & (df['Splitting']>103.88)
    # Too low density
    Too_Low_SC=df['Splitting']<102.72
    Too_Low_RT=df['Splitting']<102.734115670188

    # If room T, low density, set as low density
    df.loc[roomT&(min_lowD_RoomT_Split&max_lowD_RoomT_Split), 'Preferred D'] = LowD_RT
    df.loc[roomT&(min_lowD_RoomT_Split&max_lowD_RoomT_Split), 'Notes']='Room T, low density'
    # If room T, high density
    df.loc[roomT&(min_HD_RoomT_Split&max_HD_RoomT_Split), 'Preferred D'] = HighD_RT
    df.loc[roomT&(min_HD_RoomT_Split&max_HD_RoomT_Split), 'Notes']='Room T, high density'

    # If SupCrit, high density
    df.loc[ SupCrit&(min_HD_SC_Split&max_HD_SC_Split), 'Preferred D'] = HighD_SC
    df.loc[ SupCrit&(min_HD_SC_Split&max_HD_SC_Split), 'Notes']='SupCrit, high density'
    # If SupCrit, Med density
    df.loc[SupCrit&(min_MD_SC_Split&max_MD_SC_Split), 'Preferred D'] = MedD_SC
    df.loc[SupCrit&(min_MD_SC_Split&max_MD_SC_Split), 'Notes']='SupCrit, Med density'

    # If SupCrit, low density
    df.loc[ SupCrit&(min_lowD_SC_Split&max_lowD_SC_Split), 'Preferred D'] = LowD_SC
    df.loc[SupCrit&(min_lowD_SC_Split&max_lowD_SC_Split), 'Notes']='SupCrit, low density'

    # If Supcritical, and too low
    df.loc[SupCrit&(Too_Low_SC), 'Preferred D']=LowD_SC
    df.loc[SupCrit&(Too_Low_SC), 'Notes']='Below lower calibration limit'
    df.loc[SupCrit&(Too_Low_SC), 'in range']='N'


    # If RoomT, and too low
    df.loc[roomT&(Too_Low_RT), 'Preferred D']=LowD_RT
    df.loc[roomT&(Too_Low_RT), 'Notes']='Below lower calibration limit'
    df.loc[roomT&(Too_Low_RT), 'in range']='N'

    #if splitting is zero
    SplitZero=df['Splitting']==0
    df.loc[SupCrit&(SplitZero), 'Preferred D']=np.nan
    df.loc[SupCrit&(SplitZero), 'Notes']='Splitting=0'
    df.loc[SupCrit&(SplitZero), 'in range']='N'

    df.loc[roomT&(SplitZero), 'Preferred D']=np.nan
    df.loc[roomT&(SplitZero), 'Notes']='Splitting=0'
    df.loc[roomT&(SplitZero), 'in range']='N'


    # If impossible density, lower end
    df.loc[roomT&Imposs_lower_end, 'Preferred D'] = LowD_RT
    df.loc[roomT&Imposs_lower_end, 'Notes']='Impossible Density, low density'
    df.loc[roomT&Imposs_lower_end, 'in range']='N'

    # If impossible density, lower end
    df.loc[roomT&Imposs_upper_end, 'Preferred D'] = HighD_RT
    df.loc[roomT&Imposs_upper_end, 'Notes']='Impossible Density, high density'
    df.loc[roomT&Imposs_upper_end, 'in range']='N'

    # If high densiy, and beyond the upper calibration limit
    Upper_Cal_RT=df['Splitting']>105.1
    Upper_Cal_SC=df['Splitting']>104.95

    df.loc[roomT&Upper_Cal_RT, 'Preferred D'] = HighD_RT
    df.loc[roomT&Upper_Cal_RT, 'Notes']='Above upper Cali Limit'
    df.loc[roomT&Upper_Cal_RT, 'in range']='N'

    df.loc[SupCrit&Upper_Cal_SC, 'Preferred D'] = HighD_SC
    df.loc[SupCrit&Upper_Cal_SC, 'Notes']='Above upper Cali Limit'
    df.loc[SupCrit&Upper_Cal_SC, 'in range']='N'

    return df

    ## Loop over Ne lines



## Instrument specific metadata things
## Functions to extract metadata from WITEC files (v instrument specific)

def extract_time_stamp_witec(*, path, filename):
    """ Extracts time stamps
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Start'):
            line=l
            break
    return line

def extract_laser_power_witec(*, path, filename):
    """ Extracts laser power
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Laser'):
            line=l
            break
    return line

def extract_accumulations(*, path, filename):
    """ Extracts accumulations
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Number'):
            line=l
            break
    return line


def extract_Integration_Time(*, path, filename):
    """ Extracts Integration time
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('Integration'):
            line=l
            break
    return line

def extract_objective(*, path, filename):
    """ Extracts objective magnification
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if "Magnification" in l:
            line=l
            break
    return line

def extract_duration(*, path, filename):
    """ Extracts analysis duration
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Duration'):
            line=l
            break
    return line

def extract_date(*, path, filename):
    """ Extracts date"""

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Start Date'):
            line=l
            break
    return line

def checks_if_video(*, path, filename):
    """ Checks if file is an image (as doesnt have all metadata)
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)
    l1=fr.readline()
    #print(l1)
    if 'Video' in l1:
        return 'Video'
    else:

        return 'not Video'

def checks_if_imagescan(*, path, filename):
    """ Checks if file is an imagescan (as doesnt have all metadata)
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)
    l1=fr.readline()
    #print(l1)
    if 'Scan' in l1:
        return 'Scan'
    else:

        return 'not Scan'

def checks_if_general(*, path, filename):
    """ Checks if file is a spectra file with all the right metadata
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)
    l1=fr.readline()
    #print(l1)
    if 'General' in l1:
        return 'General'
    else:
        return 'not General'

## Functions for extracting the metadata from WITEC files

def extract_acq_params(*, path, filename):
    """ This function checks what type of file you have, and if its a spectra file,
    uses the functions above to extract various bits of metadata
    """

    line_general=checks_if_general(path=path, filename=filename)
    line_video_check=checks_if_video(path=path, filename=filename)
    line_scan=checks_if_imagescan(path=path, filename=filename)

    # If not a
    if line_video_check == "Video":
        power=np.nan
        accums=np.nan
        integ=np.nan
        Obj=np.nan
        Dur=np.nan
        dat=np.nan
    if line_scan == "Scan":
        power=np.nan
        accums=np.nan
        integ=np.nan
        Obj=np.nan
        Dur=np.nan
        dat=np.nan
    if line_general == 'General':
        power=np.nan
        accums=np.nan
        integ=np.nan
        Obj=np.nan
        Dur=np.nan
        dat=np.nan

    # If a real spectra file
    if line_video_check == 'not Video' and line_general == 'not General' and line_scan == "not Scan":
        power_str=extract_laser_power_witec(path=path, filename=filename)
        power=float(power_str.split()[3])

        accums_str=extract_accumulations(path=path, filename=filename)
        accums=float(accums_str.split()[3])

        integ_str=extract_Integration_Time(path=path, filename=filename)
        integ=float(integ_str.split()[3])

        Obj_str=extract_objective(path=path, filename=filename)
        Obj=float(Obj_str.split()[2])

        Dur_str=extract_duration(path=path, filename=filename)
        Dur=Dur_str.split()[1:]

        dat_str=extract_date(path=path, filename=filename)
        dat=dat_str.split(':')[1].split(',',1)[1].lstrip( )

    return power, accums, integ, Obj, Dur, dat




def calculates_time(*, path, filename):
    """ calculates time for non video files for WITEC files"""

    # Need to throw out video and peak fit files "general"
    line_general=checks_if_general(path=path, filename=filename)
    line_video_check=checks_if_video(path=path, filename=filename)
    line_scan=checks_if_imagescan(path=path, filename=filename)

    # If not a
    if line_video_check == "Video":
        line3_sec_int=np.nan
        line2=np.nan
    if line_general == 'General':
        line3_sec_int=np.nan
        line2=np.nan
    if line_scan== "Scan":
        line3_sec_int=np.nan
        line2=np.nan
    # If a real spectra file
    if line_video_check == 'not Video' and line_general == 'not General' and line_scan == "not Scan":
        line=extract_time_stamp_witec(path=path, filename=filename)


        line2=line.strip('Start Time:\t')
        if 'PM' in line2:
            line3=line2.strip(' PM\n')
            line3_hr=line3.split(':')[0]
            line3_min=re.search(':(.*):', line3).group(1)
            line3_sec=re.search(':(.*)', line2).group(1)[3:5]


        if 'AM' in line2:
            line3=line2.strip(' AM\n')
            line3_hr=line3.split(':')[0]
            line3_min=re.search(':(.*):', line3).group(1)
            line3_sec=re.search(':(.*)', line2).group(1)[3:5]



        if line3_hr != '12' and 'PM' in line2:
            line3_sec_int=12*60*60+float(line3_hr)*60*60+float(line3_min)*60+float(line3_sec)
        else:
            line3_sec_int=float(line3_hr)*60*60+float(line3_min)*60+float(line3_sec)


    return line3_sec_int, line2

def stitch_in_loop(*, Allfiles=None, path=None, prefix=True):
    """ Stitches together WITEC metadata for all files in a loop
    """
    # string values
    time_str=[]
    filename_str=[]
    duration_str=[]
    date_str=[]
    # Numerical values
    Int_time=np.empty(len(Allfiles), dtype=float)
    objec=np.empty(len(Allfiles), dtype=float)
    time=np.empty(len(Allfiles), dtype=float)
    power=np.empty(len(Allfiles), dtype=float)
    accumulations=np.empty(len(Allfiles), dtype=float)

    for i in range(0, len(Allfiles)):
        filename1=Allfiles[i] #.rsplit('.',1)[0]
        if prefix is True:
            filename=filename1.split(' ')[1:][0]
        else:
            filename=filename1
        print('working on file' + str(filename1))
        time_num, t_str=calculates_time(path=path, filename=filename1)

        powr, accums, integ, Obj, Dur, dat=extract_acq_params(path=path,
                                                       filename=filename1)


        Int_time[i]=integ
        objec[i]=Obj
        power[i]=powr
        accumulations[i]=accums


        time[i]=time_num
        time_str.append(format(t_str))
        filename_str.append(format(filename))
        duration_str.append(format(Dur))
        date_str.append(format(dat))




    Time_Df=pd.DataFrame(data={'name': filename_str,
                               'date': date_str,
                               'power': power,
                               'Int_time': Int_time,
                               'accumulations': accumulations,
                               'Mag (X)': objec,
                               'duration': duration_str,
                   '24hr_time': time_str,
    'sec since midnight': time
                              })
    Time_Df_2=Time_Df[Time_Df['sec since midnight'].notna()]
    Time_Df_2['index']=Time_Df_2.index

    Time_Df_2=Time_Df_2.sort_values('sec since midnight', axis=0, ascending=True)
    Time_Df_2.to_clipboard(excel=True)
    print('Done')

    return Time_Df_2