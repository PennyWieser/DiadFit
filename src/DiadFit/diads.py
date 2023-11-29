import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import lmfit
from lmfit.models import GaussianModel, VoigtModel, LinearModel, ConstantModel, PseudoVoigtModel, Pearson4Model
from scipy.signal import find_peaks, gaussian
import os
import re
from os import listdir
from os.path import isfile, join
from DiadFit.importing_data_files import *
from typing import Tuple, Optional
from dataclasses import dataclass
import matplotlib.patches as patches
import warnings as w
from tqdm import tqdm
from numpy import trapz
from scipy.integrate import simps
from scipy.interpolate import interp1d

# For debuggin

#import warnings
#warnings.simplefilter('error')

encode="ISO-8859-1"
## Ratio of different peaks

def calculate_mole_fraction_2comp(peak_area_a, peak_area_b, cross_section_a, cross_section_b, instrument_eff_a, instrument_eff_b):
    """ This function calculates the molar ration of 2 components a and b based on peak areas,
    cross section and instrument efficiency

    Parameters
    ------------

    peak_area_a: int, float, pd.Series, np.array
        Peak area of component a

    peak_area_b: int, float, pd.Series, np.array
        Peak area of component b

    cross_section_a, cross_section_a: int, float
        Raman cross section for component a and b

    instrument_eff_a, instrument_eff_b: int, float
        Instrument effeciency of a and b.

    Returns
    ------------
    pd.DataFrame
        Molar ratio of a/b


    """

    Sum_phase_a=peak_area_a/(cross_section_a*instrument_eff_a)
    Sum_phase_b=peak_area_b/(cross_section_b*instrument_eff_b)

    df=pd.DataFrame(data={'A:B molar ratio': Sum_phase_a/Sum_phase_b})

    return df
def plot_diad(*,path=None, filename=None, filetype='Witec_ASCII', Spectra_x=None, Spectra_y=None):
    """This function makes a plot of the spectra for a specific file to allow visual inspectoin

    Parameters
    ----------------
    Either

    path: str
        Folder where spectra is stored
    filename: str
        Filename of specific spectra file of interest
    filetype: str
        choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'
    Or
        Spectra_x: np.array of x coordinates
        Spectra_y: np.array of y coordinates

    Returns
        plots figure with common peaks overlain


    """
    if 'CRR' in filename:
        filetype='headless_txt'

    if Spectra_x is None:
        Spectra_df=get_data(path=path, filename=filename, filetype=filetype)

        Spectra=np.array(Spectra_df)

        Spectra_x=Spectra[:, 0]
        Spectra_y=Spectra[:, 1]




    fig, (ax1) = plt.subplots(1, 1, figsize=(4,3))

    miny=np.min(Spectra_y)
    maxy=np.max(Spectra_y)
    ax1.plot([1090, 1090], [miny, maxy], ':k', label='Magnesite')
    ax1.plot([1131, 1131], [miny, maxy], '-',  alpha=0.5,color='grey', label='Anhydrite/Mg-Sulfate')
    #ax1.plot([1136, 1136], [miny, maxy], '-', color='grey', label='Mg-Sulfate')
    ax1.plot([1151, 1151], [miny, maxy], ':c', label='SO2')
    ax1.plot([1286, 1286], [miny, maxy], '-g',  alpha=0.5,label='Diad1')
    ax1.plot([1389, 1389], [miny, maxy], '-m', alpha=0.5, label='Diad2')
    ax1.legend()
    ax1.plot(Spectra_x, Spectra_y, '-r')
    ax1.set_xlabel('Wavenumber (cm-1)')
    ax1.set_ylabel('Intensity')





@dataclass
class diad_id_config:

    # Thresholds for Scipy find peaks
    height: float = 1
    width: float=0.5
    prominence: float=50

    # to plot or not to plot
    plot_figure: bool = True


    # Exclude a range, e.g. cosmic rays
    exclude_range1: Optional [Tuple[float, float]] = None
    exclude_range2: Optional [Tuple[float, float]] = None
    # Approximate diad position
    approx_diad2_pos: Tuple[float, float]=(1379, 1395)
    approx_diad1_pos: Tuple[float, float]=(1275, 1290)


    # Background positions for prominence
    LH_bck_diad1=[1180, 1220]
    LH_bck_diad2=[1330, 1350]
    RH_bck_diad1=[1330, 1350]
    RH_bck_diad2=[1450, 1470]

    Diad_window_width=30

    Diad2_window: Tuple[float, float]=(approx_diad2_pos[0]-Diad_window_width, approx_diad2_pos[1]+Diad_window_width)
    Diad1_window: Tuple[float, float]=(approx_diad1_pos[0]-Diad_window_width, approx_diad1_pos[1])

    approx_diad2_pos_3peaks: Tuple[float, float]=(1379, 1395, 1379-17)






def identify_diad_peaks(*, config: diad_id_config=diad_id_config(), path=None, filename,
 filetype='Witec_ASCII',  plot_figure=True):

    """ This function fits a spline to the spectral data. It then uses Scipy find peaks to identify the diad,
    HB and C13 peaks. It outputs parameters such as peak positions, peak prominences,
    and peak height ratios that can be very useful when splitting data into groups.
    It uses information stored in diad_id_config for most parameters

    Parameters
    -------------
    config: dataclass
        This is diad_id_config, which you should edit before you feed into the function to adjust peak finding
        parameters. E.g. to change prominence of peaks found by scipy, do config=pf.diad_id_config(prominence=12),
        then input into this function

        Things that can be adjusted in this config file
            height: 1 (threshold height for Scipy find peaks)
            width: 0.5 (threshold width for Scipy find peaks)
            prominence: 50 (threshold prominence for scipy find peaks)
            exclude_range1: None (default ) or list - excludes a certain region of the spectra (e.g, [1200, 1250])
            exclude_range2: None (default ) or list - excludes a certain region of the spectra (e.g, [1300, 1310])
            Diad_window_width: 30 - Window to the left of Diad1 and to the right of Diad2 to look fo rdiad + hotbands.
            approx_diad2_pos: (1379, 1395) - Region for code to look for diad2 (+-Diad_window_width).
            approx_diad1_pos: (1275, 1290) - Region for code to look for diad1 (+-Diad_window_width).
            LH_bck_diad1, LH_bck_diad2, RH_bck_diad1, RH_bck_diad2: Positions either side of peaks to calculate prominences (e.g. LH_bck_diad1=[1180, 1220])


    path: str
        Path where spectra is stored

    filename: str
        Specific name of file being plotted

    filetype: str
        choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'

    plot_figure: bool
        if True, plots a figure of peaks, and the identified peaks that Scipy found

    Returns
    -------------
    pd.DataFrame
        Dataframe of approximate peak positions, prominences, etc.

    np.array of x and y coordinates of spectra

    figure if plot_figure is True


    """
    Diad_df=get_data(path=path, filename=filename, filetype=filetype)


    Diad=np.array(Diad_df)



    # First lets use find peaks
    y_in=Diad[:, 1]
    x_in=Diad[:, 0]
    spec_res=np.abs(x_in[1]-x_in[0])

    # Now lets fit a cubic spline to smoooth it all out a bit
    # Get x and y ready to make a cubic spline through data
    # Fit a  cubic spline
    f2 = interp1d(x_in, y_in, kind='cubic')
    x_new=np.linspace(np.min(x_in), np.max(x_in), 100000)

    y_cub=f2(x_new)

    y=y_cub
    x=x_new



    # Spacing of hotband from main peak
    diad2_HB2_min_offset=19-spec_res
    diad2_HB2_max_offset=23+spec_res

    # Spacing of HB from main peak.
    diad1_HB1_min_offset=19.8-spec_res
    diad1_HB1_max_offset=20.4+spec_res

    # Spacing of c13 from main peak
    diad2_C13_min_offset=16.5-spec_res
    diad2_C13_max_offset=20+spec_res

    peaks = find_peaks(y, height = config.height, prominence=config.prominence, width=config.width)


    # This gets the list of peak positions.
    peak_pos = x[peaks[0]]
    height = peaks[1]['peak_heights']
    df=pd.DataFrame(data={'pos': peak_pos,
                        'height': height})

    # Here, we are looking for peaks in the diad window.

    df_pks_diad1=df[(df['pos']>config.Diad1_window[0]) & (df['pos']<config.Diad1_window[1]) ]

    # Find peaks within the 2nd diad window
    df_pks_diad2=df[(df['pos']>config.Diad2_window[0]) & (df['pos']<config.Diad2_window[1]) ]


    # Find N highest peaks within range you have selected.
    df_sort_diad1=df_pks_diad1.sort_values('height', axis=0, ascending=False)
    if len(df_sort_diad1)>2:
        df_sort_diad1_trim=df_sort_diad1 #[0:2]
    else:
        df_sort_diad1_trim=df_sort_diad1

    df_sort_diad2=df_pks_diad2.sort_values('height', axis=0, ascending=False)
    if len(df_sort_diad2)>3:
        df_sort_diad2_trim=df_sort_diad2  #[0:3]

    else:
        df_sort_diad2_trim=df_sort_diad2

    # Check if any peaks lie where we expect the second diad
    right_pos_diad2=df_sort_diad2_trim['pos'].between(config.approx_diad2_pos[0], config.approx_diad2_pos[1])
    if any(right_pos_diad2):

        manual_diad2=False
        df_sort_diad2_rightpos=df_sort_diad2_trim.loc[right_pos_diad2]
        diad_2_diad=df_sort_diad2_rightpos.loc[df_sort_diad2_rightpos['height']==np.max(df_sort_diad2_rightpos['height'])]
        df_out=pd.DataFrame(data={'filename': filename,
                                'Diad2_pos': diad_2_diad['pos'],
                                'Diad2_height': diad_2_diad['height']})




        # Allocate hot band 2 position

        if any(df_sort_diad2_trim['pos'].between(diad_2_diad['pos'].iloc[0]+diad2_HB2_min_offset, diad_2_diad['pos'].iloc[0]+diad2_HB2_max_offset)):

            diad_2_HB=df_sort_diad2_trim.loc[df_sort_diad2_trim['pos'].between(diad_2_diad['pos'].iloc[0]+diad2_HB2_min_offset, diad_2_diad['pos'].iloc[0]+diad2_HB2_max_offset)]

            df_out['HB2_pos']=diad_2_HB['pos'].iloc[0]
            df_out['HB2_height']=diad_2_HB['height'].iloc[0]


        else:

            df_out['HB2_pos']=np.nan
            df_out['HB2_height']=np.nan


        # Look for C13 position
        if any(df_sort_diad2_trim['pos'].between(diad_2_diad['pos'].iloc[0]-diad2_C13_max_offset, diad_2_diad['pos'].iloc[0]-diad2_C13_min_offset)):
            diad_2_C13=df_sort_diad2_trim.loc[df_sort_diad2_trim['pos'].between(diad_2_diad['pos'].iloc[0]-diad2_C13_max_offset, diad_2_diad['pos'].iloc[0]-diad2_C13_min_offset)]
            df_out['C13_pos']=diad_2_C13['pos'].iloc[0]
            df_out['C13_height']=diad_2_C13['height'].iloc[0]
        else:
            df_out['C13_pos']=np.nan
            df_out['C13_height']=np.nan
    # If it didnt find anything where we expect Diad2 to be, find max position
    else:
        manual_diad2=True
        # Lets find the highest bit within this range
        diad2_range=(Diad[:, 0]>config.approx_diad2_pos[0]) & (Diad[:, 0]<config.approx_diad2_pos[1])
        diad2_height=np.max(Diad[:, 1][diad2_range])
        diad2_pos=Diad[:, 0][(Diad[:, 1]==diad2_height)&diad2_range]

        df_out=pd.DataFrame(data={'filename': filename,
                                    'Diad2_pos': diad2_pos[0] ,
                                'Diad2_height':diad2_height}, index=[0])

        # Lets try to find the hotband and hope its here!
        if any(df_sort_diad2_trim['pos'].between(df_out['Diad2_pos'].iloc[0]+diad2_HB2_min_offset, df_out['Diad2_pos'].iloc[0]+diad2_HB2_max_offset)):

            diad_2_HB=df_sort_diad2_trim.loc[df_sort_diad2_trim['pos'].between(df_out['Diad2_pos'].iloc[0]+diad2_HB2_min_offset, df_out['Diad2_pos'].iloc[0]+diad2_HB2_max_offset)]

            df_out['HB2_pos']=diad_2_HB['pos'].iloc[0]
            df_out['HB2_height']=diad_2_HB['height'].iloc[0]

        else:
            df_out['HB2_pos']=np.nan
            df_out['HB2_height']=np.nan

        # Lets try and find the C13 peak

        if any(df_sort_diad2_trim['pos'].between(df_out['Diad2_pos'].iloc[0]-diad2_C13_max_offset, df_out['Diad2_pos'].iloc[0]-diad2_C13_min_offset)):
            diad_2_C13=df_sort_diad2_trim.loc[df_sort_diad2_trim['pos'].between(df_out['Diad2_pos'].iloc[0]-diad2_C13_max_offset, df_out['Diad2_pos'].iloc[0]-diad2_C13_min_offset)]

            df_out['C13_pos']=diad_2_C13['pos'].iloc[0]
            df_out['C13_height']=diad_2_C13['height'].iloc[0]
        else:

            df_out['C13_pos']=np.nan
            df_out['C13_height']=np.nan



    # Do the same for diad 1

    right_pos_diad1=df_sort_diad1_trim['pos'].between(config.approx_diad1_pos[0], config.approx_diad1_pos[1])
    if any(right_pos_diad1):
        manual_diad1=False
        df_sort_diad1_rightpos=df_sort_diad1_trim.loc[right_pos_diad1]
        diad_1_diad=df_sort_diad1_rightpos.loc[df_sort_diad1_rightpos['height']==np.max(df_sort_diad1_rightpos['height'])]

        df_out['Diad1_pos']=diad_1_diad['pos'].iloc[0]
        df_out['Diad1_height']=diad_1_diad['height'].iloc[0]

        if any(df_sort_diad1_trim['pos'].between(diad_1_diad['pos'].iloc[0]-diad1_HB1_max_offset, diad_1_diad['pos'].iloc[0]-diad1_HB1_min_offset)):
            diad_1_HB=df_sort_diad1_trim.loc[df_sort_diad1_trim['pos'].between(diad_1_diad['pos'].iloc[0]-diad1_HB1_max_offset, diad_1_diad['pos'].iloc[0]-diad1_HB1_min_offset)]
            df_out['HB1_pos']=diad_1_HB['pos'].iloc[0]
            df_out['HB1_height']=diad_1_HB['height'].iloc[0]

        else:
            df_out['HB1_pos']=np.nan
            df_out['HB1_height']=np.nan


    else:
        manual_diad1=True
        diad1_range=(Diad[:, 0]>config.approx_diad1_pos[0]) & (Diad[:, 0]<config.approx_diad1_pos[1])
        diad1_height=np.max(Diad[:, 1][diad1_range])
        diad1_pos=Diad[:, 0][(Diad[:, 1]==diad1_height)&diad1_range]


        df_out['Diad1_pos']= diad1_pos[0]
        df_out['Diad1_height']=diad1_height

        # Lets see if we can find a hotband in here now
        if any(df_sort_diad1_trim['pos'].between(df_out['Diad1_pos'].iloc[0]-diad1_HB1_max_offset, df_out['Diad1_pos'].iloc[0]-diad1_HB1_min_offset)):
            diad_1_HB=df_sort_diad1_trim.loc[df_sort_diad1_trim['pos'].between(df_out['Diad1_pos'].iloc[0]-diad1_HB1_max_offset, df_out['Diad1_pos'].iloc[0]-diad1_HB1_min_offset)]
            df_out['HB1_pos']=diad_1_HB['pos'].iloc[0]
            df_out['HB1_height']=diad_1_HB['height'].iloc[0]

        else:
            df_out['HB1_pos']=np.nan
            df_out['HB1_height']=np.nan

    # Lets get the prominence of the main peaks and hotbands now we have found them
    Diad_x=Diad[:, 0]
    Diad_y=Diad[:, 1]
    Med_LHS_diad1=np.nanmedian(Diad_y[(Diad[:, 0]>config.LH_bck_diad1[0])& (Diad[:, 0]<config.LH_bck_diad1[1])])
    Med_RHS_diad1=np.nanmedian(Diad_y[(Diad[:, 0]>config.RH_bck_diad1[0])& (Diad[:, 0]<config.RH_bck_diad1[1])])
    Med_LHS_diad2=np.nanmedian(Diad_y[(Diad[:, 0]>config.LH_bck_diad2[0])& (Diad[:, 0]<config.LH_bck_diad2[1])])
    Med_RHS_diad2=np.nanmedian(Diad_y[(Diad[:, 0]>config.RH_bck_diad2[0])& (Diad[:, 0]<config.RH_bck_diad2[1])])
    #Med_central_back_diad2=np.nanmedian(Diad[(Diad[:, 0]>1300)& (Diad[:, 0]<1350)]

    Diad_diad1=Diad_y[(Diad[:, 0]>1260)& (Diad[:, 0]<1300)]
    Diad_diad2=Diad_y[(Diad[:, 0]>1380)& (Diad[:, 0]<1400)]
    Med_bck_diad1=(Med_LHS_diad1+Med_RHS_diad1)/2
    Med_bck_diad2=(Med_RHS_diad2+Med_LHS_diad2)/2

    df_out['Diad1_Median_Bck']=Med_bck_diad1
    df_out['Diad2_Median_Bck']=Med_bck_diad2

    df_out['Diad1_abs_prom']=df_out['Diad1_height']-Med_bck_diad1
    df_out['Diad2_abs_prom']=df_out['Diad2_height']-Med_bck_diad2
    df_out['Diad1_rel_prom']=df_out['Diad1_height']/Med_bck_diad1
    df_out['Diad2_rel_prom']=df_out['Diad2_height']/Med_bck_diad2
    df_out['HB1_abs_prom']=df_out['HB1_height']-Med_LHS_diad1
    df_out['HB2_abs_prom']=df_out['HB2_height']-Med_RHS_diad2
    df_out['HB1_rel_prom']=df_out['HB1_height']/Med_LHS_diad1
    df_out['HB2_rel_prom']=df_out['HB2_height']/Med_RHS_diad2

    df_out['approx_split']=df_out['Diad2_pos']-df_out['Diad1_pos']


    # Lets get the C13 prominence
    if any(df_out['C13_pos']>-500):
        C13_back=np.quantile(Diad_y[
        ((Diad_x>(df_out['C13_pos'].iloc[0]-3))&(Diad_x<(df_out['C13_pos'].iloc[0]+3)))
        ], 0.25)

        df_out['C13_abs_prom']=df_out['C13_height']-C13_back
        df_out['C13_rel_prom']=df_out['C13_height']/C13_back
        df_out['C13_HB2_abs_prom_ratio']=df_out['HB2_abs_prom']/df_out['C13_abs_prom']
    else:
        df_out['C13_abs_prom']=np.nan
        df_out['C13_rel_prom']=np.nan
        df_out['C13_HB2_abs_prom_ratio']=np.nan

    # Lets get the prominence of the valley between hotbands and the diad

    # Find the minimum intensity between the diad and HB

    if not np.isnan(df_out['HB2_pos'].iloc[0]):


        Diad2_between=Diad[(Diad[:, 0]<df_out['HB2_pos'].iloc[0])& (Diad[:, 0]>df_out['Diad2_pos'].iloc[0])]
        Min_y_midpoint2=np.min(Diad2_between[:, 1])

        # Option here
        RHS_x_Diad2=(df_out['HB2_pos'].iloc[0])+20
        RHS_y_Diad2=np.median(Diad[:, 1][
            (Diad[:, 0]<RHS_x_Diad2+2*spec_res)
            &(Diad[:, 0]>RHS_x_Diad2-2*spec_res)])

        df_out['Diad2_HB2_Valley_prom']=Min_y_midpoint2/RHS_y_Diad2
    else:

        df_out['Diad2_HB2_Valley_prom']=np.nan

    if not np.isnan(df_out['HB1_pos'].iloc[0]):
        Diad1_between=Diad[(Diad[:, 0]>df_out['HB1_pos'].iloc[0])& (Diad[:, 0]<df_out['Diad1_pos'].iloc[0])]
        Min_y_midpoint1=np.min(Diad1_between[:, 1])

        # option here seems to work better
        LHS_x_Diad1=(df_out['HB1_pos'].iloc[0])-20
        LHS_y_Diad1=np.median(Diad[:, 1][
            (Diad[:, 0]<LHS_x_Diad1+2.1*spec_res)
            &(Diad[:, 0]>LHS_x_Diad1-2.1*spec_res)])


        df_out['Diad1_HB1_Valley_prom']=Min_y_midpoint1/LHS_y_Diad1
    else:
        df_out['Diad1_HB1_Valley_prom']=np.nan
    # Divide by the median background position found above






    # Other useful params

    # Mean of HB elevation.
    df_out['Mean_Diad_HB_Valley_prom']=(df_out['Diad2_HB2_Valley_prom']+df_out['Diad1_HB1_Valley_prom'])/2

    # Mean HB prominence
    df_out['Mean_abs_HB_prom']=(df_out['HB1_abs_prom']+df_out['HB2_abs_prom'])

    df_out['Diad2_HB2_abs_prom_ratio']=df_out['Diad2_abs_prom']/df_out['HB2_abs_prom']
    df_out['Diad1_HB1_abs_prom_ratio']=df_out['Diad1_abs_prom']/df_out['HB1_abs_prom']
    # Average these
    df_out['Av_Diad_HB_prom_ratio']=(df_out['Diad2_HB2_abs_prom_ratio']+df_out['Diad1_HB1_abs_prom_ratio'])/2
    # Parameter for amount of noise between diads vs. height of peaks

    between_diads_x=(x>diad_id_config.RH_bck_diad1[0])&(x<diad_id_config.LH_bck_diad2[1])
    std_bet_diad=np.std(y[between_diads_x])

    df_out['Diad1_prom/std_betweendiads']=df_out['Diad1_abs_prom']/std_bet_diad
    df_out['Diad2_prom/std_betweendiads']=df_out['Diad2_abs_prom']/std_bet_diad
    df_out['HB1_prom/std_betweendiads']=df_out['HB1_abs_prom']/std_bet_diad
    df_out['HB2_prom/std_betweendiads']=df_out['HB2_abs_prom']/std_bet_diad

    # New ones
    df_out['Av_Diad_prom/std_betweendiads']=(df_out['Diad1_prom/std_betweendiads']+df_out['Diad2_prom/std_betweendiads'])/2
    df_out['C13_prom/HB2_prom']=df_out['C13_abs_prom']/df_out['HB2_abs_prom']

    df_out['Left_vs_Right']=Med_LHS_diad1/Med_RHS_diad2



    # Lets sort based on columns we want near each other
    cols_to_move = ['filename', 'approx_split',
    'Diad1_pos', 'Diad2_pos','HB1_pos', 'HB2_pos', 'C13_pos',
    'Diad1_abs_prom', 'Diad2_abs_prom', 'HB1_abs_prom', 'HB2_abs_prom', 'C13_abs_prom',
    'Mean_abs_HB_prom', 'Diad2_HB2_abs_prom_ratio', 'Diad1_HB1_abs_prom_ratio',
    'Diad1_rel_prom', 'Diad2_rel_prom', 'HB1_rel_prom', 'HB2_rel_prom', 'C13_rel_prom',
    'Diad1_HB1_Valley_prom',
    'Mean_Diad_HB_Valley_prom',
    'Diad1_prom/std_betweendiads', 'Diad2_prom/std_betweendiads',
    'Av_Diad_prom/std_betweendiads', 'C13_prom/HB2_prom', 'Av_Diad_HB_prom_ratio', 'Left_vs_Right']

    df_out = df_out[cols_to_move + [
        col for col in df_out.columns if col not in cols_to_move]]
    # Now lets get approximate strength subtracting the backgrround

    # Now lets make a plot
    if plot_figure is True:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12,4))

        ax0.plot(Diad[:, 0], Diad[:, 1], '-r')
        ax0.plot(df['pos'], df['height'], '*k')
        ax1.plot(df['pos'], df['height'], '*k', label='All Scipy Peaks')
        ax2.plot(df['pos'], df['height'], '*k')
        if manual_diad1 is True:
            ax1.plot(diad1_pos[0], diad1_height, 'dk', mfc='yellow', ms=7, label='SciPyMissed')

        if manual_diad2 is True:
            ax1.plot(diad2_pos[0], diad2_height, 'dk', mfc='yellow', ms=7)
        #ax0.legend()
        ax1.set_title('Diad1')
        ax1.plot(Diad[:, 0],Diad[:, 1], '-r')
        ax1.set_xlim([config.Diad1_window[0], config.Diad1_window[1]])
        ax2.set_title('Diad2')
        ax2.plot(Diad[:, 0],Diad[:, 1], '-r')
        ax2.set_xlim([config.Diad2_window[0], config.Diad2_window[1]])
        #ax0.set_ylim[np.nanmin(Diad[:, 1]), np.nanmax(Diad[:, 1]) ])
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


    return df_out, Diad, fig if 'fig' in locals() else None









def loop_approx_diad_fits(*, spectra_path, config, Diad_Files, filetype, plot_figure):
    """ Uses the identify_diad_peaks function to loop over all files.
    Parameters
    ----------------
    config: dataclass
        See documentation for identify_diad_peaks

    spectra_path: str
        Path where spectra is stored

    Diad_Files: list
        List of file names to loop over

    filetype: str
        choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'

    plot_figure: bool
        if True, plots a figure of peaks, and the identified peaks that Scipy found

    Returns
    -----------
    pd.DataFrame of the approximate peak parameters for each file.

    np.array of y coordinates of every spectra.


    """

    # Do fit for first file to get length
    df_peaks, Diad, fig=identify_diad_peaks(
    config=config, path=spectra_path, filename=Diad_Files[0],
    filetype=filetype, plot_figure=plot_figure)

    # Now do for all files
    fit_params = pd.DataFrame([])
    x_cord=Diad[:, 0]
    data_y_all=np.empty([  len(x_cord), len(Diad_Files)], float)
    data_x_all=np.empty([  len(x_cord), len(Diad_Files)], float)
    i=0
    for file in tqdm(Diad_Files):


        df_peaks, Diad, fig=identify_diad_peaks(
        config=config, path=spectra_path, filename=file,
    filetype=filetype, plot_figure=plot_figure)

        data_y_all[:, i]=Diad[:, 1]
        data_x_all[:, i]=Diad[:, 0]
        #data = pd.concat([Diad, data], axis=0)
        fit_params = pd.concat([fit_params, df_peaks], axis=0)
        i=i+1

    fit_params=fit_params.reset_index(drop=True)
    # Lets check all data x are the same.
    a=data_x_all
    same = np.all(a[0, :] == a[0, 0])

    if same==True:

        return fit_params, data_y_all
    if same==False:
        print('You do not have the same x axis for all spectra. ')
        diff_val_row = np.where(np.any(a != a[0], axis=1))[0]
        print(diff_val_row)


        return fit_params, data_y_all



def plot_peak_params(fit_params,
                     x_param='Diad1_pos',  y1_param='approx_split',
                    y2_param='Mean_Valley_prom', y3_param='C13_abs_prom',
                    y4_param='HB2_abs_prom', fill_na=0):

    """ Makes a 2X2 figure of fit parameters, with the same x axis, and 4 different y axis

    Parameters
    -----------

    fit_params: Pandas DataFrame
        dataframe of approximate fit params from function loop_approx_diad_fits
    x_param: str
        parameter you want on the x axis of all plots
    y1_param: str
        parameter on y axis of 1st subplot (top left)
    y2_param: str
        parameter on y axis of 2nd subplot (top right)
    y3_param: str
        parameter on y axis of 3rd subplot (bottom left)
    y4_param: str
        parameter on y axis of 4th subplot (bottom right)
    fill_na: int
        The integer value to fill Nans with to show up on plots.

    Returns
    ----------
    4 part figure

    """
    fit_params_nona=fit_params.fillna(fill_na)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10*0.8,8*0.8))


    ax1.plot(fit_params_nona[x_param], fit_params_nona[y1_param],
        'xr')
    ax1.set_xlabel(x_param)
    ax1.set_ylabel(y1_param)

    ax2.plot(fit_params_nona[x_param], fit_params_nona[y2_param],
        'xr')
    ax2.set_xlabel(x_param)
    ax2.set_ylabel(y2_param)
    fig.tight_layout()

    ax3.plot(fit_params_nona[x_param], fit_params_nona[y3_param],
        'xr')
    ax3.set_xlabel(x_param)
    ax3.set_ylabel(y3_param)

    ax4.plot(fit_params_nona[x_param], fit_params_nona[y4_param],
        'xr')
    ax4.set_xlabel(x_param)
    ax4.set_ylabel(y4_param)
    fig.tight_layout()

    return fig


def filter_splitting_prominence(*, fit_params, data_y_all,
                                x_cord,
                                splitting_limits=[100, 107],
                                lower_diad1_prom=10, exclude_str):
    """ Filters Spectra based on approximate splitting, draws a plot showing spectra to discard and those to keep

    Parameters
    --------------
    fit_params: pd.dataframe
        dataframe of fit parameters from loop_approx_diad_fits

    data_y_all: np.array
        y coordinates of each spectra from loop_approx_diad_fits, used for plotting visualizatoins

    x_cord: np.array
        x coordinates of 1 spectra. Assumes all x coordinates the same length

    splitting_limits: list (e.g. [100, 107])
        only keeps spectra with splitting in this range

    lower_diad1_prom: int, float
        Only keeps spectra that meet the splitting parameter, and have an absolute
        diad1 prominence greater than this value (helps filter out other weird spectra)

    Returns
    --------------
    fit_params_filt: pd.DataFrame
        dataframe of fit parameters for spectra to keep
    data_y_filt: np.array
        y coordinates of spectra to keep
    fit_params_disc: pd.DataFrame
        dataframe of fit parameters for spectra to discard
    data_y_disc: np.array
        y coordinates of spectra to discard


    """

    reas_split=(fit_params['approx_split'].between(splitting_limits[0], splitting_limits[1]))
    reas_heigh=fit_params['Diad1_abs_prom']>lower_diad1_prom
    if exclude_str is not None:
        name_in_file=~fit_params['filename'].str.contains(exclude_str)
    else:
        name_in_file=reas_heigh

    fit_params_filt=fit_params.loc[(reas_split&reas_heigh&name_in_file)].reset_index(drop=True)
    fit_params_disc=fit_params.loc[~(reas_split&reas_heigh&name_in_file)].reset_index(drop=True)

    print('Keeping N='+str(len(fit_params_filt)))
    print('Discarding N='+str(len(fit_params_disc)))

    filt=reas_split&reas_heigh&name_in_file
    data_y_filt=data_y_all[:, (filt)]
    data_y_disc=data_y_all[:, ~(filt)]

    intc=800
    prom_filt=0
    prom_disc=0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    ax1.set_title('Spectra to Discard')
    ax2.set_title('Spectra to Keep')
    if sum(~filt)>0:
        for i in range(0, np.shape(data_y_disc)[1]):
            av_prom_disc=np.abs(np.nanmedian(fit_params_disc['Diad1_abs_prom'])/intc)
            Diff=np.nanmax(data_y_disc[:, i])-np.nanmin(data_y_disc[:, i])
            av_prom_Keep=fit_params_disc['Diad1_abs_prom'].iloc[i]
            prom_disc=prom_disc+av_prom_disc
            ax1.plot(x_cord+i*5, (data_y_disc[:, i]-np.nanmin(data_y_disc[:, i]))/Diff+i/3, '-r', lw=0.5)
            yplot=np.quantile((data_y_disc[:, i]-np.nanmin(data_y_disc[:, i]))/Diff+i/3, 0.65)
            file=fit_params_disc['filename'].iloc[i]
            file = file.replace("_CRR_DiadFit", "")
            ax1.annotate(str(file), xy=(1450+i*5, yplot),
                    xycoords="data", fontsize=8, bbox=dict(facecolor='white', edgecolor='none', pad=2))

        ax1.set_xlim([1250, 1500+i*5])
        ax1.set_xticks([])
        ax1.set_yticks([])
    if sum(filt)>0:
        for i in range(0, np.shape(data_y_filt)[1]):
            Diff=np.nanmax(data_y_filt[:, i])-np.nanmin(data_y_filt[:, i])
            av_prom_Keep=fit_params_filt['Diad1_abs_prom'].iloc[i]
            prom_filt=prom_filt+av_prom_Keep
            ax2.plot(x_cord+i*5, (data_y_filt[:, i]-np.nanmin(data_y_filt[:, i]))/Diff+i/3, '-b', lw=0.5)


        ax2.set_xlim([1250, 1450+i*5])
        ax2.set_xticks([])
        ax2.set_yticks([])

    return fit_params_filt, data_y_filt, fit_params_disc, data_y_disc


def identify_diad_group(*, fit_params, data_y,  x_cord, filter_bool,y_fig_scale=0.1, grp_filter='Weak'):

    """Sorts diads into two groups. Those meeting the 'filter_bool' criteria, and those not
     meeting this criteria. Ones meeting the criteria are shown on the left hand plot,
    ones not meeting the criteria are shown in the right hand plot

    Parameters
    -----------------
    fit_params: pd.DataFrame
        dataframe of approximate peak parameters from loop_approx_diad_fits

    data_y: np.array
        y coordinates of each file in fit_params

    x_cord: np.array
        x coordinate of 1 spectra. Assumes all spectra have the same xcoordinate

    filter_bool: Pandas.series of bool (e.g. True, False, True, False)
        This is the filter to use to subdivide spectra into two groups.

    y_fig_scale: float, int (0.1)
        Determins how far apart the spectra are displayed on the plot. Larger number, the longer the plot is
        (and easier spectra are to see). The total height of the figure is 2+y_fig_scale*number of spectra

    grp_filter: str
        The name of the group where the filter_bool is true. E.g. if you apply a filter to get the weak diads,
        enter 'Weak' here. If filter_bool is to identify Strong spectra, enter 'Medium-Strong' here.

    Returns
    -----------------
    pd.DataFrame of fit parameters where filter_bool is True
    pd.DataFrame of fit parameters where filter_bool is False
    np.array of y coordinates where filter_bool is True
    np.array of y coordinates where filter_bool is False




    """

    if np.shape(data_y)[1]==0:
        Group1_df=pd.DataFrame().reindex_like(fit_params)
        Groupnot1_df=pd.DataFrame().reindex_like(fit_params)
        Group1_np_y=np.empty(0, dtype='float')
        Groupnot1_np_y=np.empty(0, dtype='float')
        return Group1_df, Groupnot1_df, Group1_np_y, Groupnot1_np_y

    else:
        # Grp1 is where the bool is true.

        grp1=filter_bool

        fit_params_notgrp1=fit_params.loc[~grp1]


        # Find ones in group1, in dataframe and numpy form
        Group1_df=fit_params.loc[grp1]#.reset_index(drop=True)
        Group1_df_reset=fit_params.loc[grp1].reset_index(drop=True)
        index_Grp1=Group1_df.index
        Group1_np_y=data_y[:, index_Grp1]

        # Ones not in group1
        Groupnot1_df=fit_params.loc[~grp1]#.reset_index(drop=True)
        Groupnot1_df_reset=fit_params.loc[~grp1].reset_index(drop=True)
        index_Grpnot1=Groupnot1_df.index
        Groupnot1_np_y=data_y[:, index_Grpnot1]


        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 2+y_fig_scale*len(fit_params)))
        intc=8


        #
        if sum(grp1)>0:



            if sum(grp1)==1:
                i=0
                av_prom_disc=np.abs(np.nanmedian(Group1_df['Diad1_abs_prom'])/intc)
                Diff=np.nanmax(Group1_np_y[:, i])-np.nanmin(Group1_np_y[:, i])
                ax0.plot(x_cord-i*5, (Group1_np_y[:, i]-np.nanmin(Group1_np_y[:, i]))/Diff+i/3, '-r', lw=0.5)
                yplot=np.quantile((Group1_np_y[:, i]-np.nanmin(Group1_np_y[:, i]))/Diff+i/3, 0.65)
                file=Group1_df_reset['filename'].iloc[i]
                file = file.replace("_CRR_DiadFit", "")
                ax0.annotate(str(file), xy=(1450-i*5, yplot),
                    xycoords="data", fontsize=8, bbox=dict(facecolor='white', edgecolor='none', pad=2))


            else:
                for i in range(0, np.shape(Group1_np_y)[1]):



                    av_prom_disc=np.abs(np.nanmedian(Group1_df['Diad1_abs_prom'])/intc)
                    Diff=np.nanmax(Group1_np_y[:, i])-np.nanmin(Group1_np_y[:, i])
                    ax0.plot(x_cord-i*5, (Group1_np_y[:, i]-np.nanmin(Group1_np_y[:, i]))/Diff+i/3, '-r', lw=0.5)

                    yplot=np.quantile((Group1_np_y[:, i]-np.nanmin(Group1_np_y[:, i]))/Diff+i/3, 0.5)
                    file=Group1_df_reset['filename'].iloc[i]
                    file = file.replace("_CRR_DiadFit", "")
                    ax0.annotate(str(file), xy=(1450-i*5, yplot),
                        xycoords="data", fontsize=8, bbox=dict(facecolor='white', edgecolor='none', pad=2))
            ax0.set_xlim([1250-i*5, 1550])
            ax0.set_xticks([])
            ax0.set_yticks([])



            # av_prom_Group1=np.abs(np.nanmedian(Group1_df[x_param])/intc)
            # ax0.plot(x_cord, Group1_np_y[:, i]+av_prom_Group1*i, '-r')

        if sum(~grp1)>0:


            if sum(~grp1)==1:
                j=0
                av_prom_disc=np.abs(np.nanmedian(Groupnot1_df['Diad1_abs_prom'])/intc)
                Diff=np.nanmax(Groupnot1_np_y[:, j])-np.nanmin(Groupnot1_np_y[:, j])
                ax1.plot(x_cord-j*5,
                (Groupnot1_np_y[:, j]-np.nanmin(Groupnot1_np_y[:, j]))/Diff+j/3, '-k', lw=0.5)
                yplot=np.quantile((Groupnot1_np_y[:, j]-np.nanmin(Groupnot1_np_y[:, j]))/Diff+j/3, 0.65)
                file=Groupnot1_df_reset['filename'].iloc[j]
                file = file.replace("_CRR_DiadFit", "")
                ax1.annotate(str(file), xy=(1450-j*5, yplot),
                    xycoords="data", fontsize=8, bbox=dict(facecolor='white', edgecolor='none', pad=2))


            else:

                for j in range(0, np.shape(Groupnot1_np_y)[1]):

                    av_prom_disc=np.abs(np.nanmedian(Groupnot1_df['Diad1_abs_prom'])/intc)
                    Diff=np.nanmax(Groupnot1_np_y[:, j])-np.nanmin(Groupnot1_np_y[:, j])
                    ax1.plot(x_cord-j*5,
                    (Groupnot1_np_y[:, j]-np.nanmin(Groupnot1_np_y[:, j]))/Diff+j/3, '-k', lw=0.5)
                    yplot=np.quantile((Groupnot1_np_y[:, j]-np.nanmin(Groupnot1_np_y[:, j]))/Diff+j/3, 0.5)
                    file=Groupnot1_df_reset['filename'].iloc[j]
                    file = file.replace("_CRR_DiadFit", "")
                    ax1.annotate(str(file), xy=(1450-j*5, yplot),
                        xycoords="data", fontsize=8, bbox=dict(facecolor='white', edgecolor='none', pad=2))

            ax1.set_xlim([1250-j*5, 1550])
            ax1.set_xticks([])
            ax1.set_yticks([])



        if grp_filter=='Medium-Strong':
            ax0.set_title('Classified as Strong')
            ax1.set_title('Classified as Medium')
        if grp_filter=='Weak':
            ax0.set_title('Classified as Weak')
            ax1.set_title('Ones left (not classified yet)')


        plt.subplots_adjust(wspace=0)

        return Group1_df.reset_index(drop=True), Groupnot1_df.reset_index(drop=True),Group1_np_y, Groupnot1_np_y



def plot_diad_groups(*, x_cord,  Weak_np=None, Medium_np=None, Strong_np=None, y_fig_scale=0.5):
    """
    This makes a figure showing spectra classified into Weak, Medium and Strong groups on a 3 panel plot

    Parameters
    --------------
    x_cord: np.array
        x_coordinate for all spectra (must be the same)

    Weak_np: np.array
        y coordinates of spectra classified as weak

    Medium_np: np.array
        y coordinates of spectra classified as medium

    Strong_np: np.array
        y coordinates of spectra classified as strong

    y_fig_scale: float
        The total height of the figure is 2+y_fig_scale*total number of spectra entered.


    Returns
    -------------
    figure showing 3 diad groups.




    """


    #
    if len(Weak_np)>0:
        Num_Weak=np.shape(Weak_np)[1]
    else:
        Num_Weak=0
    if len(Medium_np)>0:
        Num_Medium=np.shape(Medium_np)[1]
    else:
        Num_Medium=0
    if len(Strong_np)>0:
        Num_Strong=np.shape(Strong_np)[1]
    else:
        Num_Strong=0


    Total=Num_Strong+Num_Medium+Num_Weak
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15,2+y_fig_scale*Total))
    if Num_Weak>0:
        for i in range(0, np.shape(Weak_np)[1]):
            Diff=np.nanmax(Weak_np[:, i])-np.nanmin(Weak_np[:, i])
            ax0.plot(x_cord-i*5, (Weak_np[:, i]-np.nanmin(Weak_np[:, i]))/Diff+i/3, '-r', lw=0.5)
        ax0.set_xlim([1250-i*5, 1450])
        ax0.set_xticks([])
        ax0.set_yticks([])

    if Num_Medium>0:
        for i in range(0, np.shape(Medium_np)[1]):
            Diff=np.nanmax(Medium_np[:, i])-np.nanmin(Medium_np[:, i])
            ax1.plot(x_cord-i*5, (Medium_np[:, i]-np.nanmin(Medium_np[:, i]))/Diff+i/3, '-b', lw=0.5)
        ax1.set_xlim([1250-i*5, 1450])
        ax1.set_xticks([])
        ax1.set_yticks([])

    if Num_Strong>0:
        for i in range(0, np.shape(Strong_np)[1]):
            Diff=np.nanmax(Strong_np[:, i])-np.nanmin(Strong_np[:, i])
            ax2.plot(x_cord-i*5, (Strong_np[:, i]-np.nanmin(Strong_np[:, i]))/Diff+i/3, '-g', lw=0.5)
        ax2.set_xlim([1250-i*5, 1450])
        ax2.set_xticks([])
        ax2.set_yticks([])

    ax0.set_title('Weak, N='+str(Num_Weak))
    ax1.set_title('Medium, N='+str(Num_Medium))
    ax2.set_title('Strong, N='+str(Num_Strong))
    plt.subplots_adjust(wspace=0)

    return fig


def remove_diad_baseline(*, path=None, filename=None, Diad_files=None, filetype='Witec_ASCII',
            exclude_range1=None, exclude_range2=None,N_poly=1, x_range_baseline=10,
            lower_bck=[1200, 1250], upper_bck=[1320, 1330], sigma=4,
            plot_figure=True):
    """ This function fits an Nth degree polynomial fitted through spectra points between two background positions. It returns baseline-subtracted data

    Parameters
    -----------
    path: str
        Folder where spectra is stored

    filename: str
        Specific file being read

    OR

    Diad_files: str
        Filename if in same folder as script.

    filetype: str
        Identifies type of file. choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'

    exclude_range1: None or list length 2
        Excludes a region, e.g. a cosmic ray

    exclude_range2: None or list length 2
        Excludes a region, e.g. a cosmic ray

    sigma: int
        Filters out points that are more than this number of sigma from the median of the baseline.

    N_poly: int
        Degree of polynomial to fit to the backgroun

    lower_bck: list len 2
        wavenumbers of baseline region to the left of the peak of interest

    upper_bck: list len 2
        wavenumbers to the right of the peak of interest


    plot_figure: bool
        if True, plots figure, if False, doesn't.

    Returns
    -----------
    y_corr, Py_base, x,  Diad_short, Py_base, Pf_baseline, Baseline_ysub, Baseline_x, Baseline, span

        y_corr: Background subtracted y values trimmed in baseline range
        x: initial x values trimmed in baseline range
        Diad_short: Initial data (x and y) trimmed in baseline range
        Py_base: Fitted baseline for trimmed x coordinates
        Pf_baseline: polynomial model fitted to baseline
        Baseline_ysub: Baseline fitted for x coordinates in baseline
        Baseline_x: x co-ordinates in baseline.
        Baseline: Filtered to remove points outside sigma*std dev of baseline
        span: lower and upper x coordinate of region that isnt the baseline



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



def add_peak(*, prefix=None, center=None, model_name='VoigtModel',
min_cent=None, max_cent=None, min_sigma=None, max_sigma=None, amplitude=100, min_amplitude=None, max_amplitude=None, sigma=0.2):
    """
    This function iteratively adds a peak using lmfit, used for iteratively fitting HB, C13 etc.

    Parameters
    --------------
    prefix: str
        Prefix added to peak name, e.g. 'diad1', so it reads diad1_center, diad1_sigma etc.

    center: float, int
        Estimate of peak center (e.g. x coordinate of peak)

    sigma: float, int (default 0.2)
        Estimate of sigma of peak fit

    amplitude: float, int
        Estimate of amplitude of peakf it


    model_name: str 'VoigtModel' or 'PseudoVoigtModel'
        Type of peak which is being added.

    Optional:

    min_cent, max_cent: float
        Minimum and maximum x coordinate to place peak center at

    min_sigma, max_sigma: float
        Minimum and maximum sigma of peak fit

    min_amplitude, max_amplitude: float
        Minimum and maximum sigma of peak fit

    Returns
    ---------------

    peak: Fitted model
    params: Peak fit parameters

    """
    if model_name == "VoigtModel":
        Model_combo=VoigtModel(prefix=prefix)#+ConstantModel(prefix=prefix) #Stops getting results

    if model_name == "PseudoVoigtModel":
        Model_combo=PseudoVoigtModel(prefix=prefix)#+ConstantModel(prefix=prefix) #Stops getting results

    if model_name== "Pearson4Model":
        Model_combo=Pearson4Model(prefix=prefix)#+ConstantModel(prefix=prefix) #Stops getting results
    peak =  Model_combo
    pars = peak.make_params()

    if min_cent is not None and max_cent is not None:
        pars[prefix + 'center'].set(center, min=min_cent, max=max_cent)
    else:
        pars[prefix + 'center'].set(center)


    pars[prefix + 'amplitude'].set(amplitude, min=min_amplitude, max=max_amplitude)

    if min_sigma is not None:
        pars[prefix+'sigma'].set(sigma, max=max_sigma)

    if min_sigma is not None and max_sigma is not None:
        pars[prefix+'sigma'].set(sigma, max=max_sigma, min=min_sigma)

    else:
        pars[prefix + 'sigma'].set(sigma, min=0)


    return peak, pars






## Overall function for fitting diads in 1 single step
@dataclass
class diad1_fit_config:

    # What model to use
    model_name: str = 'PseudoVoigtModel'
    fit_peaks:int = 2

    # Degree of polynomial to use
    N_poly_bck_diad1: float =1

    # Background/baseline positions
    lower_bck_diad1: Tuple[float, float]=(1180, 1220)
    upper_bck_diad1: Tuple[float, float]=(1300, 1350)

    # Do you need a gaussian? Set position here if so
    fit_gauss: Optional [bool] =False
    gauss_amp: Optional [float]=1000

    diad_sigma: float=0.2
    diad_sigma_min_allowance: float=0.2
    diad_sigma_max_allowance: float=5




    # Peak amplitude
    diad_prom: float = 100
    HB_prom: float = 20
    HB_sigma_min_allowance=0.05
    HB_sigma_max_allowance=3
    HB_amp_min_allowance=0.01
    HB_amp_max_allowance=1
    # How much to show on x anx y axis of figure showing background
    x_range_baseline: float=75
    y_range_baseline: float=100

    #Do you want to save the figure?


    dpi: float = 200
    x_range_residual: float=20

    # Do you want to return other parameters?
    return_other_params: bool =False



@dataclass
class diad2_fit_config:
    # Do you need a gaussian? Set position here if so

    # What model to use
    model_name: str = 'PseudoVoigtModel'
    fit_peaks:int = 3

    # Degree of polynomial to use
    N_poly_bck_diad2: float =1
    # Threshold intesnity
    threshold_intensity=50

    # Background/baseline positions
    lower_bck_diad2: Tuple[float, float]=(1300, 1360)
    upper_bck_diad2: Tuple[float, float]=(1440, 1470)


    fit_gauss: Optional [bool] =False
    gauss_amp: Optional [float]=1000

    diad_sigma: float=0.2
    diad_sigma_min_allowance: float=0.2
    diad_sigma_max_allowance: float=5


    # Peak amplitude
    diad_prom: float = 100
    HB_prom: float = 20
    HB_sigma_min_allowance=0.05
    HB_sigma_max_allowance=3
    HB_amp_min_allowance=0.01
    HB_amp_max_allowance=1
    # How much to show on x anx y axis of figure showing background with all peaks imposed
    x_range_baseline: float=75
    y_range_baseline: float=100

    #Do you want to save the figure?
    plot_figure: bool = True
    dpi: float = 200
    x_range_residual: float=20

    # Do you want to return other parameters?
    return_other_params: bool =False

    C13_prom: float=10



## Testing generic model


def fit_gaussian_voigt_generic_diad(config1, *, diad1=False, diad2=False, path=None, filename=None, xdat=None, ydat=None, span=None,  plot_figure=True, dpi=200, Diad_pos=None, HB_pos=None, C13_pos=None, fit_peaks=None, block_print=True):


    """ This function fits a voigt/Pseudovoigt curves to background subtracted data for your diads
    +- HBs+-C13 peaks. It can also fit a second Gaussian background iteratively with these peaks


    Parameters
    -----------
    config1: fit configuration file, from diad2_fit_config or diad1_fit_config data classes.
        Detailed descriptions of things in this file are found in docs for fit_diad_2_w_bck, and fit_diad_1_w_bck

    diad1: bool
        True if fitting Diad1, False if Diad2

    diad2: bool
        True if fitting Diad2, False if Diad1

    path: str
        Path to spectra folder. Used to make a folder inside this folder to store peak fit images in

    filename: str
        filename of specific diad being fitted

    xdat: np.array
        x coordinates of the spectra

    ydat: np.array
        y coordinates of the spectra

    span: list (e.g. [1100, 1300])
        Comes from remove_diad_baseline function. Sets min and max x coordinate of new linspace
        for plotting the overall peak fit over the data

    plot_figure: bool
        if True, plots figure from this function specifically. Even if False, if called through
        fit_diad_2_w_bck or fit_diad_1_w_bck, get a figure from those functions with relevant data on

    dpi: int
        dpi of saved figure

    Diad_pos, HB_pos, C13_pos: float
        Initial guess of Diad, HB, C13 pos

    fit_peaks: int
        Number of peaks to fit. 1= just diad, 2 = diad + HB, 3 = diad+ HB +C13

    block_print: bool
        If true, prints a few status updates as it goes to help debug

    Returns
    -------------------
    result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim, config1.fit_gauss, residual_diad_coords, ydat_inrange,  xdat_inrange

    result: lmfit.model.ModelResult (model form lmfit)
    df_out: dataframe of peak fit parameters
    y_best_fit: np.array of best peak fit y parameters
    x_lin: np.array of x coordinates corresponding to y_best_fit
    components: y coordinates of the different components of the peak fit.
    xdat: input xdat
    ydat: input ydat
    ax1_xlim, ax2_xlim: xlim for plot in final figure in fit_w_bck for different fit components
    residual_diad_coordinates: np.array of y coordinate of residual plot
    ydat_inrange: np.array of data within 'span'
    xdat_inrange:np.array of data within 'span'



    """




    # Calculate the amplitude from the sigma and the prominence
    calc_diad_amplitude=((config1.diad_sigma)*(config1.diad_prom))/0.3939
    calc_HB_amplitude=((config1.diad_sigma)*(config1.HB_prom))/0.3939

    # If ask for Gaussian, but fit peaks is 1, remove gaussian
    if fit_peaks==1 and config1.fit_gauss is True:
        config1.fit_gauss=False


    if diad2 is True and fit_peaks==3:
        calc_C13_amplitude=(0.5*(config1.diad_sigma)*(config1.C13_prom))/0.3939
    # Gets overridden if you have triggered any of the warnings
    refit=False
    refit_param='Flagged Warnings:'
    spec_res=np.abs(xdat[1]-xdat[0])






    initial_guess=Diad_pos
    HB_initial_guess=HB_pos
    C13_initial_guess=C13_pos




    if config1.model_name=="VoigtModel":
        model_ini = VoigtModel()#+ ConstantModel()
    elif config1.model_name=="PseudoVoigtModel":
        model_ini = PseudoVoigtModel()#+ ConstantModel()
    elif config1.model_name=='Pearson4Model':
        model_ini = Pearson4Model()#+ ConstantModel()
    else:
        TypeError('Choice of model not yet supported - select VoigtModel, PseudoVoigtModel or Pearson4Model')

    # Create initial peak params
    # Set peak position to 2 spectral res units either side of the initial guess made above
    params_ini = model_ini.make_params(center=initial_guess, max=initial_guess+spec_res*5, min=initial_guess-spec_res*5)
    # Set the amplitude to within X0.1 or 10X of the guessed amplitude
    params_ini['amplitude'].set(calc_diad_amplitude, min=calc_diad_amplitude/5,
                                 max=calc_diad_amplitude*5)
    # Set sigma to be the entered value, then
    params_ini['sigma'].set(config1.diad_sigma, min=config1.diad_sigma*config1.diad_sigma_min_allowance,
                                max=config1.diad_sigma*config1.diad_sigma_max_allowance)

    # Now lets tweak an initial model with just 1 peak, this really helps with the fit
    init_ini = model_ini.eval(params_ini, x=xdat)
    result_ini  = model_ini.fit(ydat, params_ini, x=xdat)
    comps_ini  = result_ini.eval_components()
    Center_ini=result_ini.params.get('center')
    Amplitude_ini=result_ini.params.get('amplitude')
    sigma_ini=result_ini.params.get('sigma')
    fwhm_ini=result_ini.params.get('fwhm')



    # For relatively weak peaks, you wont want a gaussian background
    if config1.fit_gauss is False:

        # If there is 1 peak, e.g. if have a Nan for hotband
        if fit_peaks==1:
            if config1.model_name=='VoigtModel':
                model_F = VoigtModel(prefix='lz1_') #+ ConstantModel(prefix='c1')
            if config1.model_name=='PseudoVoigtModel':
                model_F = PseudoVoigtModel(prefix='lz1_') + ConstantModel(prefix='c1')
            if config1.model_name=='Pearson4Model':
                model_F = Pearson4Model(prefix='lz1_') + ConstantModel(prefix='c1')

            pars1 = model_F.make_params()
            pars1['lz1_'+ 'amplitude'].set(calc_diad_amplitude, min=0, max=calc_diad_amplitude*10)
            pars1['lz1_'+ 'center'].set(Center_ini, min=Center_ini-5*spec_res, max=Center_ini+5*spec_res)
            pars1['lz1_'+ 'sigma'].set(config1.diad_sigma, min=config1.diad_sigma*config1.diad_sigma_min_allowance,
                         max=config1.diad_sigma*config1.diad_sigma_max_allowance)
            params=pars1


        # If there is more than one peak
        else:
            # Set up Lz1 the same for all situations
            if config1.model_name=='VoigtModel':
                model1 = VoigtModel(prefix='lz1_')# + ConstantModel(prefix='c1')
            if config1.model_name=='PseudoVoigtModel':
                model1 = PseudoVoigtModel(prefix='lz1_') #+ ConstantModel(prefix='c1')
            if config1.model_name=='Pearson4Model':
                model1 = Pearson4Model(prefix='lz1_') #+ ConstantModel(prefix='c1')
            pars1 = model1.make_params()
            pars1['lz1_'+ 'amplitude'].set(calc_diad_amplitude, min=0, max=calc_diad_amplitude*10)

            pars1['lz1_'+ 'center'].set(Center_ini, min=Center_ini-3*spec_res, max=Center_ini+3*spec_res)

            pars1['lz1_'+ 'sigma'].set(config1.diad_sigma, min=config1.diad_sigma*config1.diad_sigma_min_allowance,
                        max=config1.diad_sigma*config1.diad_sigma_max_allowance)




            if fit_peaks>1:
                model_F=model1
                params=pars1

            if fit_peaks==2:

                # Second wee peak
                prefix='lz2_'
                if config1.model_name=='VoigtModel':
                    peak = VoigtModel(prefix='lz2_')# + ConstantModel(prefix='c1')
                if config1.model_name=='PseudoVoigtModel':
                    peak = PseudoVoigtModel(prefix='lz2_')# + ConstantModel(prefix='c1')
                if config1.model_name=='Pearson4Model':
                    peak = Pearson4Model(prefix='lz2_')# + ConstantModel(prefix='c1')


                pars = peak.make_params()

                pars[prefix + 'center'].set(HB_initial_guess,
                    min=HB_initial_guess-2*spec_res, max=HB_initial_guess+2)

                pars[prefix + 'amplitude'].set(calc_HB_amplitude, min=Amplitude_ini*config1.HB_amp_min_allowance, max=Amplitude_ini*config1.HB_amp_max_allowance)
                pars[prefix+ 'sigma'].set(sigma_ini, min=sigma_ini*config1.HB_sigma_min_allowance, max=sigma_ini*config1.HB_sigma_max_allowance)


                model_F=model1+peak

                # updating pars 1 with values in pars
                pars1.update(pars)
                params=pars1


                # Attempt at stabilizing peak fit
                result = model_F.fit(ydat, pars1, x=xdat)
                result.params['lz2_center'].vary = False
                result.params['lz2_amplitude'].vary = False
                result.params['lz2_sigma'].vary = False
                final_result = model_F.fit(ydat, result.params, x=xdat)
                params.update(final_result.params)
                params.update(final_result.params)
                result=final_result







            if fit_peaks==3:
                if block_print is False:
                    print('Trying to iteratively fit 3 peaks')

                peak_pos_left=np.array([HB_initial_guess, C13_initial_guess])

                for i, cen in enumerate(peak_pos_left):

                    if i==0: # This is the hotband
                        peak, pars = add_peak(prefix='lz%d_' % (i+2), center=cen,
                        min_cent=cen-3*spec_res, max_cent=cen+3*spec_res, sigma=sigma_ini,
                        min_sigma=sigma_ini*config1.HB_sigma_min_allowance,
                        max_sigma=sigma_ini*config1.HB_sigma_max_allowance,
                        amplitude=calc_HB_amplitude,
                        min_amplitude=Amplitude_ini*config1.HB_amp_min_allowance,
                        max_amplitude=Amplitude_ini*config1.HB_amp_max_allowance,
                        model_name=config1.model_name)
                        model = peak+model1
                        params.update(pars)
                    if i==1: # This is c13
                        peak, pars = add_peak(prefix='lz%d_' % (i+2), center=cen,
                        min_cent=cen-3*spec_res, max_cent=cen+3*spec_res,
                        sigma=sigma_ini/5,
                        min_sigma=sigma_ini/20,
                        max_sigma=sigma_ini/2,
                        amplitude=calc_C13_amplitude,
                        min_amplitude=calc_C13_amplitude*(0.5*config1.HB_amp_min_allowance),
                        max_amplitude=calc_C13_amplitude*2*(config1.HB_amp_max_allowance),
                        model_name=config1.model_name)
                        model = peak+model
                        params.update(pars)




                model_F=model



                # Attempt at stabilizing peak fit
                result = model_F.fit(ydat, pars1, x=xdat)
                result.params['lz2_center'].vary = False
                result.params['lz2_amplitude'].vary = False
                result.params['lz2_sigma'].vary = False
                result.params['lz3_center'].vary = False
                result.params['lz3_amplitude'].vary = False
                result.params['lz3_sigma'].vary = False
                final_result = model_F.fit(ydat, result.params, x=xdat)
                params.update(final_result.params)
                params.update(final_result.params)
                result=final_result




    # Same, but also with a Gaussian Background
    if config1.fit_gauss is not False:


        # making the gaussian model
        model1 = GaussianModel(prefix='bkg_')
        params = model1.make_params()
        params['bkg_'+'amplitude'].set(config1.gauss_amp, min=config1.gauss_amp/100, max=config1.gauss_amp*10)
        params['bkg_'+'sigma'].set(sigma_ini*15, min=sigma_ini*10, max=sigma_ini*25)
        params['bkg_'+'center'].set(initial_guess, min=initial_guess-7, max=initial_guess+7)



        if fit_peaks==2:
            peak_pos_voigt=np.array([initial_guess, HB_initial_guess])

            for i, cen in enumerate(peak_pos_voigt):
                if i==0: # This is the Diad
                    peak, pars = add_peak(prefix='lz%d_' % (i+1),
                center=Center_ini, min_cent=cen-3*spec_res, max_cent=cen+3*spec_res,
                sigma=config1.diad_sigma,
                max_sigma=config1.diad_sigma*config1.diad_sigma_max_allowance,
                min_sigma=config1.diad_sigma*config1.diad_sigma_min_allowance,
                amplitude=calc_diad_amplitude, min_amplitude=0, max_amplitude=10*calc_diad_amplitude,
                model_name=config1.model_name)
                model = peak+model1
                params.update(pars)

                if i==1: # This is the hotband
                    peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen,
                        min_cent=cen-3*spec_res, max_cent=cen+3*spec_res, sigma=sigma_ini,
                        min_sigma=sigma_ini*config1.HB_sigma_min_allowance,
                        max_sigma=sigma_ini*config1.HB_sigma_max_allowance,
                        amplitude=calc_HB_amplitude,
                        min_amplitude=Amplitude_ini*config1.HB_amp_min_allowance,
                        max_amplitude=Amplitude_ini*config1.HB_amp_max_allowance,
                        model_name=config1.model_name)
                    model2 = peak+model
                    params.update(pars)


            # Attempt at stabilizing peak fit
# updating pars 1 with values in pars

            result = model2.fit(ydat, params, x=xdat)


            # Set the parameters of 'lz1' to be fixed
            result.params['lz2_center'].vary = False
            result.params['lz2_amplitude'].vary = False
            result.params['lz2_sigma'].vary = False
            result.params['bkg_center'].vary = False
            result.params['bkg_amplitude'].vary = False
            result.params['bkg_sigma'].vary = False

            # Perform fitting using the entire model (model2) with the stabilized 'lz1' parameters
            final_result = model2.fit(ydat, result.params, x=xdat)
            center_error = final_result.params['lz1_center'].stderr
            params.update(final_result.params)


            result=final_result
            model_F=model2


        if fit_peaks==3:
            peak_pos_voigt=np.array([initial_guess, HB_initial_guess, C13_initial_guess])

            for i, cen in enumerate(peak_pos_voigt):

                if i==0: # This is the Diad
                    peak, pars = add_peak(prefix='lz%d_' % (i+1),
                center=Center_ini, min_cent=cen-3*spec_res, max_cent=cen+3*spec_res,
                sigma=config1.diad_sigma,
                max_sigma=config1.diad_sigma*config1.diad_sigma_max_allowance,
                min_sigma=config1.diad_sigma*config1.diad_sigma_min_allowance,
                amplitude=calc_diad_amplitude, min_amplitude=0, max_amplitude=10*calc_diad_amplitude,
                model_name=config1.model_name)
                model = peak+model1
                params.update(pars)

                if i==1: # This is the hotband

                    peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen,
                        min_cent=cen-3*spec_res, max_cent=cen+3*spec_res, sigma=sigma_ini,
                        min_sigma=sigma_ini*config1.HB_sigma_min_allowance,
                        max_sigma=sigma_ini*config1.HB_sigma_max_allowance,
                        amplitude=calc_HB_amplitude,
                        min_amplitude=Amplitude_ini*config1.HB_amp_min_allowance,
                        max_amplitude=Amplitude_ini*config1.HB_amp_max_allowance,
                        model_name=config1.model_name)
                    model2 = peak+model
                    params.update(pars)

                if i==2: # This is c13
                    peak, pars = add_peak(prefix='lz%d_' % (i+1), center=cen,
                        min_cent=cen-3*spec_res, max_cent=cen+3*spec_res,
                        sigma=sigma_ini/5,
                        min_sigma=sigma_ini/20,
                        max_sigma=sigma_ini/2,
                        amplitude=calc_C13_amplitude,
                        min_amplitude=calc_C13_amplitude*(0.5*config1.HB_amp_min_allowance),
                        max_amplitude=calc_C13_amplitude*2*(config1.HB_amp_max_allowance),
                        model_name=config1.model_name)
                    model3 = peak+model2
                    params.update(pars)

            result = model3.fit(ydat, params, x=xdat)


            # Set the parameters of 'lz1' to be fixed
            result.params['lz2_center'].vary = False
            result.params['lz2_amplitude'].vary = False
            result.params['lz2_sigma'].vary = False
            result.params['lz3_center'].vary = False
            result.params['lz3_amplitude'].vary = False
            result.params['lz3_sigma'].vary = False
            result.params['bkg_center'].vary = False
            result.params['bkg_amplitude'].vary = False
            result.params['bkg_sigma'].vary = False

            # Perform fitting using the entire model (model2) with the stabilized 'lz1' parameters
            final_result = model3.fit(ydat, result.params, x=xdat)
            center_error = final_result.params['lz1_center'].stderr

            result=final_result
            model_F=model3

            params.update(final_result.params)









    # Regardless of fit, evaluate model
    init = model_F.eval(params, x=xdat)
    result = model_F.fit(ydat, params, x=xdat)
    comps = result.eval_components()

    #print(result.ci_report())

    #print(result.conf_interval(method='bootstrap'))

    #print(result.fit_report(min_correl=0.5))
    # Check if function gives error bars


    #print(result.best_values)
    # Get first peak center
    Peak1_Cent=result.best_values.get('lz1_center')
    Peak1_Int=result.best_values.get('lz1_amplitude')
    Peak1_Sigma=result.best_values.get('lz1_sigma')
    Peak1_gamma=result.best_values.get('lz1_gamma')
    Peak1_Prop_Lor=result.best_values.get('lz1_fraction')
    Peak1_fwhm=result.params['lz1_fwhm'].value

    Center_pk2_error=result.params.get('lz1_center')

    error_pk2 = result.params['lz1_center'].stderr



    x_lin=np.linspace(span[0], span[1], 2000)
    y_best_fit=result.eval(x=x_lin)
    components=result.eval_components(x=x_lin)




    if config1.fit_gauss is not False:
        Gauss_cent=result.best_values.get('bkg_center')
        Gauss_amp=result.best_values.get('bkg_amplitude')
        Gauss_sigma=result.best_values.get('bkg_sigma')

        max_Gauss=np.max(components.get('bkg_'))
        max_voigt=np.max(components.get('lz1_'))
        if max_Gauss>max_voigt/5:
            refit_param=str(refit_param)+ ' G_higher_than_diad'
        # if Gauss_sigma>=(config1.gauss_sigma*10-0.1):
        #     refit=True
        #     refit_param=str(refit_param)+' G_input_TooLowSigma'
        # if Gauss_sigma<=(config1.gauss_sigma/10+0.1):
        #     refit=True
        #     refit_param=str(refit_param)+ ' G_input_TooHighSigma'
        #
        # if Gauss_amp>=(config1.gauss_amp*10-0.1):
        #     refit=True
        #     refit_param=str(refit_param)+' G_input_TooLowAmp'
        # if Gauss_amp<=(config1.gauss_sigma+0.1):
        #     refit=True
        #     refit_param=str(refit_param)+' G_Amp_too_high'
        if Gauss_cent<=(config1.fit_gauss-30+0.5):
            refit=True
            refit_param=str(refit_param)+' G_InputTooLowCent'
        if Gauss_cent>=(initial_guess+30-0.5):
            refit=True
            refit_param=str(refit_param)+' G_InputTooHighCent'





    # print('fwhm gauss')
    # print(result.best_values)




    #

    # Work out what peak is what
    if diad2 is True:

        if fit_peaks==1:

            Peak2_Cent=None
            Peak3_Cent=None
            ax1_xlim=[Center_ini-15, Center_ini+15]
            ax2_xlim=[Center_ini-15, Center_ini+15]

        if fit_peaks==2:

            Peak2_Cent=result.best_values.get('lz2_center')
            Peak2_Int=result.best_values.get('lz2_amplitude')
            Peak2_Sigma=result.best_values.get('lz2_sigma')
            Peak3_Cent=None
            ax1_xlim=[Center_ini-15, Center_ini+30]
            ax2_xlim=[Center_ini-15, Center_ini+30]

        if fit_peaks==3:

            Peak2_Cent=result.best_values.get('lz2_center')
            Peak2_Int=result.best_values.get('lz2_amplitude')
            Peak2_Sigma=result.best_values.get('lz2_sigma')
            Peak3_Cent=result.best_values.get('lz3_center')
            Peak3_Int=result.best_values.get('lz3_amplitude')
            Peak3_Sigma=result.best_values.get('lz3_sigma')

            ax1_xlim=[Center_ini-30, Center_ini+30]
            ax2_xlim=[Center_ini-30, Center_ini+30]

        if Peak2_Cent is None:
            df_out=pd.DataFrame(data={'Diad2_Voigt_Cent': Peak1_Cent,
                                    'Diad2_cent_err': error_pk2,
                                    'Diad2_Voigt_Area': Peak1_Int,
                                'Diad2_Voigt_Sigma': Peak1_Sigma,
                                'Diad2_Voigt_Gamma': Peak1_gamma,


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
                    HB2_Sigma=Peak1_Sigma
                if Diad2_Cent==Peak1_Cent:
                    Diad2_Int=Peak1_Int
                    HB2_Int=Peak2_Int
                    HB2_Sigma=Peak2_Sigma




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
                    HB2_Sigma=Peak2_Sigma
                if HB2_Cent==Peak1_Cent:
                    HB2_Int=Peak1_Int
                    HB2_Sigma=Peak1_Sigma
                if HB2_Cent==Peak3_Cent:
                    HB2_Int=Peak3_Int
                    HB2_Sigma=Peak2_Sigma


                # Same for C13
                if C13_Cent==Peak2_Cent:
                    C13_Int=Peak2_Int
                    C13_Sigma=Peak2_Sigma
                if C13_Cent==Peak1_Cent:
                    C13_Int=Peak1_Int
                    C13_Sigma=Peak1_Sigma
                if C13_Cent==Peak3_Cent:
                    C13_Int=Peak3_Int
                    C13_Sigma=Peak3_Sigma

            df_out=pd.DataFrame(data={'Diad2_Voigt_Cent': Diad2_Cent,
            'Diad2_cent_err': error_pk2,
                                    'Diad2_Voigt_Area': Diad2_Int,
                                    'Diad2_Voigt_Sigma': Peak1_Sigma,
                                    'Diad2_Voigt_Gamma': Peak1_gamma,
            }, index=[0])

            df_out['HB2_Cent']=HB2_Cent
            df_out['HB2_Area']=HB2_Int
            df_out['HB2_Sigma']=HB2_Sigma
            if  Peak3_Cent is not None:
                df_out['C13_Cent']=C13_Cent
                df_out['C13_Area']=C13_Int
                df_out['C13_Sigma']=C13_Sigma

        result_diad2_origx_all=result.eval(x=xdat)
        # Trim to be in range
        #print(result_diad2_origx_all)
        result_diad2_origx=result_diad2_origx_all[(xdat>span[0]) & (xdat<span[1])]
        ydat_inrange=ydat[(xdat>span[0]) & (xdat<span[1])]
        xdat_inrange=xdat[(xdat>span[0]) & (xdat<span[1])]
        residual_diad_coords=ydat_inrange-result_diad2_origx


        x_cent_lin=np.linspace(df_out['Diad2_Voigt_Cent']-1, df_out['Diad2_Voigt_Cent']+1, 20000)
        y_cent_best_fit=result.eval(x=x_cent_lin)
        diad_height = np.max(y_cent_best_fit)
        df_out['Diad2_Combofit_Height']= diad_height
        df_out.insert(0, 'Diad2_Combofit_Cent', np.nanmean(x_cent_lin[y_cent_best_fit==diad_height]))


        residual_diad2=np.sum(((ydat_inrange-result_diad2_origx)**2)**0.5)/(len(ydat_inrange))
        df_out['Diad2_Residual']=residual_diad2
        df_out['Diad2_Prop_Lor']= Peak1_Prop_Lor
        df_out['Diad2_fwhm']=Peak1_fwhm

        if config1.fit_gauss is not False:
            df_out['Diad2_Gauss_Cent']=Gauss_cent
            df_out['Diad2_Gauss_Area']=Gauss_amp
            df_out['Diad2_Gauss_Sigma']=Gauss_sigma


        #Checking whether you need any flags.
        if error_pk2 is None:
            refit=True
            refit_param=str(refit_param)+' No Error'


        if Peak1_Int>=(calc_diad_amplitude*10-0.1):
            refit=True
            refit_param=str(refit_param)+' V_LowAmp'
        if Peak1_Sigma>=(calc_diad_amplitude*10-0.1):
            refit=True
            refit_param=str(refit_param)+' V_HighAmp'
        if Peak1_Sigma>=(config1.diad_sigma*config1.diad_sigma_max_allowance-0.001):
            refit=True
            refit_param=str(refit_param)+' V_Sigma_too_Low'
        if  diad_height>(3*config1.diad_prom):
            refit_param=str(refit_param)+' V_Sigma_too_Low'
        if  diad_height<(0.5*config1.diad_prom):
            refit_param=str(refit_param)+' V_Sigma_too_High'
        if Peak1_Sigma<=(config1.diad_sigma*config1.diad_sigma_min_allowance+0.001):
            refit=True
            refit_param=str(refit_param)+' V_Sigma_too_High'
        if diad_height<config1.threshold_intensity:
            refit_param=str(refit_param)+' Below threshold intensity'

    #if findme


        if plot_figure is True:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
            ax1.plot(x_lin, y_best_fit, '-g', linewidth=2, label='best fit')
            ax1.plot(xdat, ydat,  '.k', label='data')
            ax1.legend()
            ax1.set_xlim(ax1_xlim)
            ax2.set_xlim(ax2_xlim)






            if config1.fit_gauss is not False:
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


            if fit_peaks>1:
                ax2.plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='Peak2')

            if fit_peaks>2:
                ax2.plot(x_lin, components.get('lz3_'), '-m', linewidth=2, label='Peak3')


            ax2.legend()

            path3=path+'/'+'diad_fit_images'
            if os.path.exists(path3):
                out='path exists'
            else:
                os.makedirs(path+'/'+ 'diad_fit_images', exist_ok=False)


            file=filename.rsplit('.txt', 1)[0]
            fig.savefig(path3+'/'+'Diad2_Fit_{}.png'.format(file), dpi=dpi)




        best_fit=result.best_fit

        df_out['Diad2_refit']=refit_param

    if diad1 is True:



        if fit_peaks==1:
            Peak2_Cent=None
            Peak3_Cent=None
            ax1_xlim=[Center_ini-15, Center_ini+15]
            ax2_xlim=[Center_ini-15, Center_ini+15]
        if fit_peaks==2:
            Peak2_Cent=result.best_values.get('lz2_center')
            Peak2_Int=result.best_values.get('lz2_amplitude')
            Peak2_Sigma=result.best_values.get('lz2_sigma')
            # # Checking parameters for hot bands
            # if Peak2_Sigma>(sigma_ini*5-0.1):
            #     refit_param=str(refit_param)+' HB1_HighSigma'
            # if Peak2_Sigma<=(sigma_ini/20+0.1):
            #     refit_param=str(refit_param)+' HB1_LowSigma'
            #
            # if Peak2_Int>(Amplitude_ini/5-0.001):
            #     refit_param=str(refit_param)+' HB1_HighAmp'
            # if Peak2_Int<=(Amplitude_ini/100+0.001):
            #     refit_param=str(refit_param)+' HB1_LowAmp'


            Peak3_Cent=None
            ax1_xlim=[Center_ini-30, Center_ini+15]
            ax2_xlim=[Center_ini-30, Center_ini+15]



        if Peak2_Cent is None:
            df_out=pd.DataFrame(data={'Diad1_Voigt_Cent': Peak1_Cent,
            'Diad1_cent_err': error_pk2,
                                    'Diad1_Voigt_Area': Peak1_Int,
                                'Diad1_Voigt_Sigma': Peak1_Sigma,
                                'Diad1_Voigt_Gamma': Peak1_gamma,


            }, index=[0])
        if Peak2_Cent is not None:





            df_out=pd.DataFrame(data={'Diad1_Voigt_Cent': Peak1_Cent,
            'Diad1_cent_err': error_pk2,
                                    'Diad1_Voigt_Area': Peak1_Int,
                                    'Diad1_Voigt_Sigma': Peak1_Sigma,
                                    'Diad1_Voigt_Gamma': Peak1_gamma,
            }, index=[0])

            df_out['HB1_Cent']=Peak2_Cent
            df_out['HB1_Area']=Peak2_Int
            df_out['HB1_Sigma']=Peak2_Sigma


        result_diad1_origx_all=result.eval(x=xdat)
        # Trim to be in range
        #print(result_diad2_origx_all)
        result_diad1_origx=result_diad1_origx_all[(xdat>span[0]) & (xdat<span[1])]
        ydat_inrange=ydat[(xdat>span[0]) & (xdat<span[1])]
        xdat_inrange=xdat[(xdat>span[0]) & (xdat<span[1])]
        residual_diad_coords=ydat_inrange-result_diad1_origx


        x_cent_lin=np.linspace(df_out['Diad1_Voigt_Cent']-1, df_out['Diad1_Voigt_Cent']+1, 20000)
        y_cent_best_fit=result.eval(x=x_cent_lin)
        diad_height = np.max(y_cent_best_fit)
        df_out['Diad1_Combofit_Height']= diad_height
        df_out.insert(0, 'Diad1_Combofit_Cent', np.nanmean(x_cent_lin[y_cent_best_fit==diad_height]))


        residual_diad1=np.sum(((ydat_inrange-result_diad1_origx)**2)**0.5)/(len(ydat_inrange))
        df_out['Diad1_Residual']=residual_diad1
        df_out['Diad1_Prop_Lor']= Peak1_Prop_Lor
        df_out['Diad1_fwhm']=Peak1_fwhm

        if config1.fit_gauss is not False:
            df_out['Diad1_Gauss_Cent']=Gauss_cent
            df_out['Diad1_Gauss_Area']=Gauss_amp
            df_out['Diad1_Gauss_Sigma']=Gauss_sigma



        if plot_figure is True:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
            ax1.plot(x_lin, y_best_fit, '-g', linewidth=2, label='best fit')
            ax1.plot(xdat, ydat,  '.k', label='data')
            ax1.legend()
            ax1.set_xlim(ax1_xlim)
            ax2.set_xlim(ax2_xlim)






            if config1.fit_gauss is not False:
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


            if fit_peaks>1:
                ax2.plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='Peak2')

            if fit_peaks>2:
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
            fig.savefig(path3+'/'+'Diad1_Fit_{}.png'.format(file), dpi=dpi)




        best_fit=result.best_fit

        df_out['Diad1_refit']=refit_param

        # Final check - that Gaussian isnt anywhere near the height of the diad


    df_out=df_out.fillna(0)

    return result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim, residual_diad_coords, ydat_inrange,  xdat_inrange



def fit_diad_2_w_bck(*, config1: diad2_fit_config=diad2_fit_config(), config2: diad_id_config=diad_id_config(),
    path=None, filename=None, peak_pos_voigt=None,filetype=None,
    plot_figure=True, close_figure=False, Diad_pos=None, HB_pos=None, C13_pos=None):
    """ This function fits the background (using the function remove_diad_baseline) and then
    fits the peaks using fit_gaussian_voigt_generic_diad()
    It then checks if any parameters are right at the permitted edge (meaning fitting didnt converge),
    and tries fitting if so. Then it makes a plot
    of the various parts of the fitting procedure, and returns a dataframe of the fit parameters.


    Parameters
    -----------
    config1: by default uses diad2_fit_config(). Options to edit within this:

        N_poly_bck_diad2: int
            degree of polynomial to fit to the background.

        model_name: str
            What type of peak you want to fit, either 'PseudoVoigtModel' or 'VoigtModel'

        fit_peaks: int
            Number of peaks to fit. 1 = just diad, 2= diad and Hotband, 3= diad, hotband, C13

        lower_bck_diad2, upper_bck_diad2: [float, float]
            background position to left and right of overal diad region (inc diad+-HB+-C13)

        fit_gauss: bool
            If True, fits a gaussian peak

        gauss_amp:  float
            Estimate of amplitude of Gaussian peak (we find a good first guess is 2* the absolute prominence of the HB peak)

        diad_sigma, HB_sigma: float
            Estimate of the amplitude of the diad peak/HB peak

        diad_prom, HB_prom, C13_prom: float
            Estimate of prominence of peak. Used alongside sigma to estimate the amplitude of the diad to aid peak fitting.


        diad_sigma_min_allowance, diad_sigma_min_allowance: float
        HB_sigma_min_allowance, HB_sigma_max_allowance: float
            Factor that sigma of peak is allowed to deviate from 1st guess (e.g. max=2, 2* guess, min=0.5, 0.5 * guess).

        diad_amp_min_allowance, diad_amp_min_allowance: float
        HB_amp_min_allowance, HB_amp_max_allowance: float
            Factor that the amplitude of peak is allowed to deviate from 1st guess (e.g. max=2, 2* guess, min=0.5, 0.5 * guess).

        * Note, for C13, we find the code works better, if sigma is set as 1st fit of diad sigma/5, allowed to vary
        between diad_sigma/20 to diad_sigma/2. The min and max amplitude are got 0.5 and 2* the allowance entered for HB2

        x_range_baseline: float or None
            Sets x limit of plot showing baselines to diad center - x_range_baseline, and +x_range_baseline

        y_range_baseline: float or None
            Sets y axis of baseline plot as minimum balue of baseline region -y_range_baseline/3,
            and the upper value as max value of baseline region + y_range_baseline
            Baseline region is the defined region of baseline minus any points outside the sigma filter

        x_range_residual: float
            Distance either side of diad center to show on residual plot.

        dpi: float
            dpi to save figure at of overall fits

    config2: diad_id_config() file. Only thing it uses from this is any excluded ranges

    path: str
        Folder user wishes to read data from

    filename: str
        Specific name of file being used.

    filetype: str
        Identifies type of file. choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'

    plot_figure: bool
        if True, saves figure

    Diad_pos: float
        Estimate of diad position

    HB_pos: float
        Estimate of HB position

    C13_pos: float
        Estimate of C13 position

    close_fig: bool
        if True, displays figure in Notebook. If False, closes it (you can still find it saved)

    return_other_params: bool (default False)
        if False, just returns a dataframe of peak parameters
        if True, also returns:
            result: fit parameters
            y_best_fit: best fit to all curves in y
            x_lin: linspace of best fit coordinates in x.

    Returns
    ------------
    see above. Depends in returns_other_params




    """
    if 'CRR' in filename:
        filetype='headless_txt'

    # Check number of peaks makes sense
    fit_peaks=config1.fit_peaks
    Diad_df=get_data(path=path, filename=filename, filetype=filetype)
    Diad=np.array(Diad_df)


    if fit_peaks==2:
        if np.isnan(HB_pos)==True or config1.HB_prom<0:
            fit_peaks=1
            fit_gauss=False



    if fit_peaks==3:
        # This tests if HB is Nan and C13 is Nan
        if np.isnan(HB_pos)==True and np.isnan(C13_pos)==True:
            fit_peaks=1
            fit_gauss=False
        # If only HB is Nan, lets instead allocate a HB positoin.
        elif np.isnan(HB_pos):
            HB_region=[Diad_pos+19, Diad_pos+23]
            filter=(Diad[:, 0]>=HB_region[0])& (Diad[:,0]<=HB_region[1])
            TrimDiad=Diad[filter]
            filter2=(Diad[:, 0]>=(HB_region[0]-5))& (Diad[:,0]<=(HB_region[1]+5))
            TrimDiad2=Diad[filter2]
            maxy=np.max(TrimDiad[:, 1])
            maxx=TrimDiad[TrimDiad[:, 1]==maxy]
            HB_pos=float(maxx[:, 0][0])
            miny=np.min(TrimDiad2[:, 1])
            config1.HB_prom=maxy-miny
            config1.gauss_amp=2*(maxy-miny)




        # If no C13, reduce fit peaks to 2.
        elif np.isnan(C13_pos)==True or config1.C13_prom<10:
            fit_peaks=2



    # First, we feed data into the remove baseline function, which returns corrected data

    y_corr_diad2, Py_base_diad2, x_diad2,  Diad_short_diad2, Py_base_diad2, Pf_baseline_diad2,  Baseline_ysub_diad2, Baseline_x_diad2, Baseline_diad2, span_diad2=remove_diad_baseline(
   path=path, filename=filename, filetype=filetype, exclude_range1=config2.exclude_range1, exclude_range2=config2.exclude_range2, N_poly=config1.N_poly_bck_diad2,
    lower_bck=config1.lower_bck_diad2, upper_bck=config1.upper_bck_diad2, plot_figure=False)





    # Then, we feed this baseline-corrected data into the combined gaussian-voigt peak fitting function
    result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim,residual_diad_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_generic_diad(config1, diad2=True, path=path, filename=filename,
    xdat=x_diad2, ydat=y_corr_diad2,
    span=span_diad2, plot_figure=False, Diad_pos=Diad_pos, HB_pos=HB_pos, C13_pos=C13_pos, fit_peaks=fit_peaks)

    if any(df_out['Diad2_refit'].str.contains('Below threshold intensity')):
        print('below your threshold intensity, we have filled with nans')


    else:

        # Try Refitting once
        if str(df_out['Diad2_refit'].iloc[0])!='Flagged Warnings:':
            print('refit attempt 1')
            factor=2

            import copy
            config_tweaked=config1
            config_tweaked=copy.copy(config1)

            if any(df_out['Diad2_refit'].str.contains(' No Error')):
                if fit_peaks>1:
                    fit_peaks=fit_peaks-1




            # if any(df_out['Diad2_refit'].str.contains('V_HighAmp')):
            #     config_tweaked.diad_amplitude=calc_diad_amplitude/10
            if any(df_out['Diad2_refit'].str.contains('V_Sigma_too_Low')):
                config_tweaked.diad_sigma=config1.diad_sigma*factor
            if any(df_out['Diad2_refit'].str.contains('V_Sigma_too_High')):
                config_tweaked.diad_sigma=config1.diad_sigma/factor
            # Gaussian fit bits
            if any(df_out['Diad2_refit'].str.contains('G_input_TooHighSigma')):
                config_tweaked.gauss_sigma=config1.gauss_sigma/factor
            if any(df_out['Diad2_refit'].str.contains('G_input_TooLowSigma')):
                config_tweaked.gauss_sigma=config1.gauss_sigma*factor
            if any(df_out['Diad2_refit'].str.contains('G_input_TooLowAmp')):
                config_tweaked.gauss_amplitude=config1.gauss_amp*factor
            if any(df_out['Diad2_refit'].str.contains('G_Amp_too_high')):
                config_tweaked.gauss_amplitude=config1.gauss_amp/factor
            if any(df_out['Diad2_refit'].str.contains('G_higher_than_diad')):
                config_tweaked.gauss_amplitude=config1.gauss_amp/factor
                #config_tweaked.gauss_sigma=config1.gauss_sigma*factor
            # Hot band fit bits
            # if any(df_out['Diad2_refit'].str.contains('HB2_LowAmp')):
            #     config_tweaked.HB_amplitude=calc_HB_amplitude/2
            # if any(df_out['Diad2_refit'].str.contains('HB2_HighAmp')):
            #     config_tweaked.HB_amplitude=calc_HB_amplitude*2


            result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim,residual_diad_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_generic_diad(config_tweaked, diad2=True, path=path, filename=filename,
            xdat=x_diad2, ydat=y_corr_diad2,
        span=span_diad2, plot_figure=False, Diad_pos=Diad_pos, HB_pos=HB_pos, C13_pos=C13_pos, fit_peaks=fit_peaks)
            i=2
            while str(df_out['Diad2_refit'].iloc[0])!='Flagged Warnings:':

                print('refit attempt  ='+str(i) + ', '+str(df_out['Diad2_refit'].iloc[0]))
                print(df_out['Diad2_refit'].iloc[0])


                config_tweaked2=copy.copy(config_tweaked)
                factor2=factor*2

                #
                # if any(df_out['Diad2_refit'].str.contains('V_LowAmp')):
                #     config_tweaked2.diad_amplitude=config_tweaked.diad_amplitude*10
                # if any(df_out['Diad2_refit'].str.contains('V_HighAmp')):
                #     config_tweaked2.diad_amplitude=config_tweaked.diad_amplitude/10
                if any(df_out['Diad2_refit'].str.contains('V_Sigma_too_Low')):
                    config_tweaked2.diad_sigma=config_tweaked.diad_sigma*factor2
                if any(df_out['Diad2_refit'].str.contains('V_Sigma_too_High')):
                    config_tweaked2.diad_sigma=config_tweaked.diad_sigma/factor2
                # Gaussian fit bits
                if any(df_out['Diad2_refit'].str.contains('G_input_TooHighSigma')):
                    config_tweaked2.gauss_sigma=config_tweaked.gauss_sigma/factor2
                if any(df_out['Diad2_refit'].str.contains('G_input_TooLowSigma')):
                    config_tweaked2.gauss_sigma=config_tweaked.gauss_sigma*factor2
                if any(df_out['Diad2_refit'].str.contains('G_input_TooLowAmp')):
                    config_tweaked2.gauss_amplitude=config_tweaked.gauss_amp*factor2
                if any(df_out['Diad2_refit'].str.contains('G_Amp_too_high')):
                    config_tweaked2.gauss_amplitude=config_tweaked.gauss_amp/factor2
                if any(df_out['Diad2_refit'].str.contains('G_higher_than_diad')):
                    config_tweaked2.gauss_amplitude=config_tweaked.gauss_amp/factor2
                    #config_tweaked2.gauss_sigma=config_tweaked.gauss_sigma*factor2
            # # Hot bands bits
            #     if any(df_out['Diad2_refit'].str.contains('HB2_LowAmp')):
            #         config_tweaked2.HB_amplitude=config_tweaked.HB_amplitude/2
            #     if any(df_out['Diad2_refit'].str.contains('HB2_HighAmp')):
            #         config_tweaked2.HB_amplitude=config_tweaked.HB_amplitude*2


                result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim,residual_diad_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_generic_diad(config_tweaked2, diad2=True, path=path, filename=filename,
            xdat=x_diad2, ydat=y_corr_diad2, span=span_diad2, plot_figure=False, Diad_pos=Diad_pos, HB_pos=HB_pos, C13_pos=C13_pos, fit_peaks=fit_peaks)
                i=i+1
                if i>5:
                    print('Got to 5 iteratoins and still couldnt adjust the fit parameters')
                    break


    # get a best fit to the baseline using a linspace from the peak fitting
    ybase_xlin=Pf_baseline_diad2(x_lin)

    # We extract the full spectra to plot at the end, and convert to a dataframe
    Spectra_df=get_data(path=path, filename=filename, filetype=filetype)
    Spectra=np.array(Spectra_df)



    # Make nice figure
    if plot_figure is True:

        figure_mosaic="""
        XY
        AB
        CD
        EE
        """

        fig,axes=plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(12, 16))
        if df_out['Diad2_refit'] is not False:
            fig.suptitle('Diad 2, file= '+ str(filename) + ' \n' + str(df_out['Diad2_refit'].iloc[0]), fontsize=16, x=0.5, y=1.0)
        else:
            fig.suptitle('Diad 2, file= '+ str(filename), fontsize=16, x=0.5, y=1.0)
        # Background plot for real

        # Plot best fit on the LHS, and individual fits on the RHS at the top

        axes['X'].set_title('a) Background fit')
        axes['X'].plot(Diad[:, 0], Diad[:, 1], '-', color='grey')
        axes['X'].plot(Diad_short_diad2[:, 0], Diad_short_diad2[:, 1], '-r', label='Spectra')

        #axes['X'].plot(Baseline[:, 0], Baseline[:, 1], '-b', label='Bck points')
        axes['X'].plot(Baseline_diad2[:, 0], Baseline_diad2[:, 1], '.b', label='bel. Bck. pts')
        axes['X'].plot(Diad_short_diad2[:, 0], Py_base_diad2, '-k', label='bck. poly fit')



        ax1_ymin=np.min(Baseline_diad2[:, 1])-10*np.std(Baseline_diad2[:, 1])
        ax1_ymax=np.max(Baseline_diad2[:, 1])+10*np.std(Baseline_diad2[:, 1])
        ax1_xmin=config1.lower_bck_diad2[0]-30
        ax1_xmax=config1.upper_bck_diad2[1]+30
        # Adding patches


        rect_diad2_b1=patches.Rectangle((config1.lower_bck_diad2[0], ax1_ymin),config1.lower_bck_diad2[1]-config1.lower_bck_diad2[0],ax1_ymax-ax1_ymin,
                                linewidth=1,edgecolor='none',facecolor='cyan', label='sel. bck. region', alpha=0.3, zorder=0)
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



        axes['A'].plot(xdat, ydat,  '.k', label='bck. sub. data')
        axes['A'].plot( x_lin ,y_best_fit, '-g', linewidth=1, label='best fit')
        axes['A'].legend()
        axes['A'].set_ylabel('Intensity')
        axes['A'].set_xlabel('Wavenumber')
        axes['A'].set_xlim(ax1_xlim)
        axes['A'].set_title('c) Overall Best Fit')

    # individual fits
        axes['B'].plot(xdat, ydat, '.k')

        # This is for if there is more than 1 peak, this is when we want to plot the best fit

        if fit_peaks>1:


            axes['B'].plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='HB2')

        axes['B'].plot(x_lin, components.get('lz1_'), '-b', linewidth=2, label='Diad2')
        if config1.fit_gauss is not False:
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


        if fit_peaks>=2:
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
        axc_ymin=np.min(Baseline_diad2[:, 1])-config1.y_range_baseline/3
        axc_ymax=np.max(Baseline_diad2[:, 1])+config1.y_range_baseline

        rect_diad2_b1=patches.Rectangle((config1.lower_bck_diad2[0],axc_ymin),config1.lower_bck_diad2[1]-config1.lower_bck_diad2[0],axc_ymax-axc_ymin,
                                linewidth=1,edgecolor='none',facecolor='cyan', label='bck', alpha=0.3, zorder=0)



        axes['C'].set_title('d) Peaks overlain on data before subtraction')
        axes['C'].plot(Diad_short_diad2[:, 0], Diad_short_diad2[:, 1], '.r', label='data')
        axes['C'].plot(Baseline_diad2[:, 0], Baseline_diad2[:, 1], '.b', label='bck')
        axes['C'].plot(Diad_short_diad2[:, 0], Py_base_diad2, '-k', label='Poly bck fit')


        axes['C'].set_ylabel('Intensity')
        axes['C'].set_xlabel('Wavenumber')


        if config1.fit_gauss is not False:

            axes['C'].plot(x_lin, components.get('bkg_')+ybase_xlin, '-m', label='Gaussian bck', linewidth=2)

        axes['C'].plot( x_lin ,y_best_fit+ybase_xlin, '-g', linewidth=2, label='Best Fit')
        axes['C'].plot(x_lin, components.get('lz1_')+ybase_xlin, '-b', label='Diad2', linewidth=1)
        if fit_peaks>1:
            axes['C'].plot(x_lin, components.get('lz2_')+ybase_xlin, '-r', label='HB2', linewidth=1)
        if fit_peaks>2:
            axes['C'].plot(x_lin, components.get('lz3_')+ybase_xlin, '-c', label='C13', linewidth=1)

        axes['C'].legend(ncol=3, loc='lower center')



        axes['C'].set_xlim([axc_xmin, axc_xmax])
        axes['C'].set_ylim([axc_ymin, axc_ymax])
        #axes['C'].plot(Diad_short[:, 0], Diad_short[:, 1], '"r', label='Data')


        # Residual on plot D
        axes['D'].set_title('f) Residuals')
        axes['D'].plot([df_out['Diad2_Voigt_Cent'], df_out['Diad2_Voigt_Cent']], [np.min(residual_diad_coords), np.max(residual_diad_coords)], ':b')
        if fit_peaks>1:
                axes['D'].plot([df_out['HB2_Cent'], df_out['HB2_Cent']], [np.min(residual_diad_coords), np.max(residual_diad_coords)], ':r')

        axes['D'].plot(xdat_inrange, residual_diad_coords, 'ok', mfc='c' )
        axes['D'].plot(xdat_inrange, residual_diad_coords, '-c' )
        axes['D'].set_ylabel('Residual')
        axes['D'].set_xlabel('Wavenumber')
        # axes['D'].set_xlim(ax1_xlim)
        # axes['D'].set_xlim(ax2_xlim)
        Local_Residual_diad2=residual_diad_coords[((xdat_inrange>(df_out['Diad2_Voigt_Cent'][0]-config1.x_range_residual))
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

    # Lets calculate peak skewness here.
    Skew50=assess_diad2_skewness(config1=diad2_fit_config(), int_cut_off=0.5, path=path, filename=filename, filetype=filetype,
    skewness='abs', height=1, prominence=5, width=0.5, plot_figure=False)
    Skew80=assess_diad2_skewness(config1=diad2_fit_config(), int_cut_off=0.3, path=path, filename=filename, filetype=filetype,
    skewness='abs', height=1, prominence=5, width=0.5, plot_figure=False)
    df_out['Diad2_Asym50']=Skew50['Skewness_diad2']
    df_out['Diad2_Asym70']=Skew80['Skewness_diad2']
    df_out['Diad2_Yuan2017_sym_factor']=(df_out['Diad2_fwhm'])*(df_out['Diad2_Asym50']-1)
    df_out['Diad2_Remigi2021_BSF']=df_out['Diad2_fwhm']/df_out['Diad2_Combofit_Height']


    if  config1.return_other_params is False:
        return df_out
    else:
        return df_out, result, y_best_fit, x_lin




def fit_diad_1_w_bck(*, config1: diad1_fit_config=diad1_fit_config(), config2: diad_id_config=diad_id_config(),
    path=None, filename=None, filetype=None,  plot_figure=True, close_figure=True, Diad_pos=None, HB_pos=None):
    """ This function fits the background (using the function remove_diad_baseline) and then fits the peaks
    using fit_gaussian_voigt_generic_diad()
    It then checks if any parameters are right at the permitted edge (meaning fitting didnt converge),
    and tries fitting if so. Then it makes a plot
    of the various parts of the fitting procedure, and returns a dataframe of the fit parameters.


    Parameters
    -----------
    config1: by default uses diad1_fit_config(). Options to edit within this:

        N_poly_bck_diad1: int
            degree of polynomial to fit to the background.

        model_name: str
            What type of peak you want to fit, either 'PseudoVoigtModel' or 'VoigtModel'

        fit_peaks: int
            Number of peaks to fit. 1 = just diad, 2= diad

        lower_bck_diad1, upper_bck_diad1: [float, float]
            background position to left and right of overal diad region (inc diad+-HB+-C13)

        fit_gauss: bool
            If True, fits a gaussian peak

        gauss_amp:  float
            Estimate of amplitude of Gaussian peak (we find a good first guess is 2* the absolute prominence of the HB peak)

        diad_sigma, HB_sigma: float
            Estimate of the amplitude of the diad peak/HB peak

        diad_prom, HB_prom, C13_prom: float
            Estimate of prominence of peak. Used alongside sigma to estimate the amplitude of the diad to aid peak fitting.


        diad_sigma_min_allowance, diad_sigma_min_allowance: float
        HB_sigma_min_allowance, HB_sigma_max_allowance: float
            Factor that sigma of peak is allowed to deviate from 1st guess (e.g. max=2, 2* guess, min=0.5, 0.5 * guess).

        diad_amp_min_allowance, diad_amp_min_allowance: float
        HB_amp_min_allowance, HB_amp_max_allowance: float
            Factor that the amplitude of peak is allowed to deviate from 1st guess (e.g. max=2, 2* guess, min=0.5, 0.5 * guess).


        x_range_baseline: float or None
            Sets x limit of plot showing baselines to diad center - x_range_baseline, and +x_range_baseline

        y_range_baseline: float or None
            Sets y axis of baseline plot as minimum balue of baseline region -y_range_baseline/3,
            and the upper value as max value of baseline region + y_range_baseline
            Baseline region is the defined region of baseline minus any points outside the sigma filter

        x_range_residual: float
            Distance either side of diad center to show on residual plot.

        dpi: float
            dpi to save figure at of overall fits

    config2: diad_id_config() file. Only thing it uses from this is any excluded ranges

    path: str
        Folder user wishes to read data from

    filename: str
        Specific name of file being used.

    filetype: str
        Identifies type of file. choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'

    plot_figure: bool
        if True, saves figure

    Diad_pos: float
        Estimate of diad position

    HB_pos: float
        Estimate of HB position


    close_fig: bool
        if True, displays figure in Notebook. If False, closes it (you can still find it saved)

    return_other_params: bool (default False)
        if False, just returns a dataframe of peak parameters
        if True, also returns:
            result: fit parameters
            y_best_fit: best fit to all curves in y
            x_lin: linspace of best fit coordinates in x.

    Returns
    ------------
    see above. Depends in returns_other_params


"""
    if 'CRR' in filename:
        filetype='headless_txt'

    fit_peaks=config1.fit_peaks

    if fit_peaks==2:
        if np.isnan(HB_pos)==True or config1.HB_prom<-50:
            fit_peaks=1
            config1.fit_gauss=False
            #print('Either no hb position, or prominence<-50, using 1 fit')

    Diad_df=get_data(path=path, filename=filename, filetype=filetype)
    Diad=np.array(Diad_df)
    # First, we feed data into the remove baseline function, which returns corrected data

    y_corr_diad1, Py_base_diad1, x_diad1,  Diad_short_diad1, Py_base_diad1, Pf_baseline_diad1,  Baseline_ysub_diad1, Baseline_x_diad1, Baseline_diad1, span_diad1=remove_diad_baseline(
   path=path, filename=filename, filetype=filetype, exclude_range1=config2.exclude_range1, exclude_range2=config2.exclude_range2, N_poly=config1.N_poly_bck_diad1,
    lower_bck=config1.lower_bck_diad1, upper_bck=config1.upper_bck_diad1, plot_figure=False)





    # Then, we feed this baseline-corrected data into the combined gaussian-voigt peak fitting function

    result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim,residual_diad_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_generic_diad(config1,diad1=True, path=path, filename=filename,
                    xdat=x_diad1, ydat=y_corr_diad1,
                    span=span_diad1, plot_figure=False,
                    Diad_pos=Diad_pos, HB_pos=HB_pos,fit_peaks=fit_peaks)



    # if df_out['Diad1_cent_err'].iloc[0]==0:
    #     if config1.fit_peaks>1:
    #         config_tweaked.fit_peaks=config1.fit_peaks-1
    #         print('reducing peaks by 1 to try to get an error')


    # Try Refitting once
    if str(df_out['Diad1_refit'].iloc[0])!='Flagged Warnings:':
        print('refit attempt 1')
        factor=2
        import copy
        config_tweaked=config1
        config_tweaked=copy.copy(config1)

        if any(df_out['Diad1_refit'].str.contains(' No Error')):
            if config1.fit_peaks>2:
                config_tweaked.fit_peaks=config1.fit_peaks-1
                fit_peaks=config_tweaked.fit_peaks


        if any(df_out['Diad1_refit'].str.contains('V_Sigma_too_Low')):
            config_tweaked.diad_sigma=config1.diad_sigma*factor
        if any(df_out['Diad1_refit'].str.contains('V_Sigma_too_High')):
            config_tweaked.diad_sigma=config1.diad_sigma/factor
        # Gaussian fit bits
        if any(df_out['Diad1_refit'].str.contains('G_input_TooHighSigma')):
            config_tweaked.gauss_sigma=config1.gauss_sigma/factor
        if any(df_out['Diad1_refit'].str.contains('G_input_TooLowSigma')):
            config_tweaked.gauss_sigma=config1.gauss_sigma*factor
        if any(df_out['Diad1_refit'].str.contains('G_input_TooLowAmp')):
            config_tweaked.gauss_amplitude=config1.gauss_amp*factor
        if any(df_out['Diad1_refit'].str.contains('G_Amp_too_high')):
            config_tweaked.gauss_amplitude=config1.gauss_amp/factor
        if any(df_out['Diad1_refit'].str.contains('G_higher_than_diad')):
            config_tweaked.gauss_amplitude=config1.gauss_amp/factor
            config_tweaked.gauss_sigma=config1.gauss_sigma*factor
        # # Hot band fit bits
        # if any(df_out['Diad1_refit'].str.contains('HB1_LowAmp')):
        #     config_tweaked.HB_amplitude=calc_HB_amplitude/2
        # if any(df_out['Diad1_refit'].str.contains('HB1_HighAmp')):
        #     config_tweaked.HB_amplitude=calc_HB_amplitude*2

        result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim,residual_diad_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_generic_diad(config_tweaked, diad1=True, path=path, filename=filename,
                    xdat=x_diad1, ydat=y_corr_diad1,
                    span=span_diad1, plot_figure=False, Diad_pos=Diad_pos, HB_pos=HB_pos, fit_peaks=config_tweaked.fit_peaks)
        i=2
        while str(df_out['Diad1_refit'].iloc[0])!='Flagged Warnings:':

            print('refit attempt  ='+str(i) + ', '+str(df_out['Diad1_refit'].iloc[0]))
            print(df_out['Diad1_refit'].iloc[0])

            import copy
            config_tweaked2=copy.copy(config_tweaked)
            factor2=factor*2


            # if any(df_out['Diad1_refit'].str.contains('V_LowAmp')):
            #     config_tweaked2.diad_amplitude=config_tweaked.diad_amplitude*10
            # if any(df_out['Diad1_refit'].str.contains('V_HighAmp')):
            #     config_tweaked2.diad_amplitude=config_tweaked.diad_amplitude/10
            if any(df_out['Diad1_refit'].str.contains('V_Sigma_too_Low')):
                config_tweaked2.diad_sigma=config_tweaked.diad_sigma*factor2
            if any(df_out['Diad1_refit'].str.contains('V_Sigma_too_High')):
                config_tweaked2.diad_sigma=config_tweaked.diad_sigma/factor2
            # Gaussian fit bits
            if any(df_out['Diad1_refit'].str.contains('G_input_TooHighSigma')):
                config_tweaked2.gauss_sigma=config_tweaked.gauss_sigma/factor2
            if any(df_out['Diad1_refit'].str.contains('G_input_TooLowSigma')):
                config_tweaked2.gauss_sigma=config_tweaked.gauss_sigma*factor2
            if any(df_out['Diad1_refit'].str.contains('G_input_TooLowAmp')):
                config_tweaked2.gauss_amplitude=config_tweaked.gauss_amp*factor2
            if any(df_out['Diad1_refit'].str.contains('G_Amp_too_high')):
                config_tweaked2.gauss_amplitude=config_tweaked.gauss_amp/factor2
            if any(df_out['Diad1_refit'].str.contains('G_higher_than_diad')):
                config_tweaked2.gauss_amplitude=config_tweaked.gauss_amp/factor2
                config_tweaked2.gauss_sigma=config_tweaked.gauss_sigma*factor2
            # Hot bands bits
            # if any(df_out['Diad1_refit'].str.contains('HB1_LowAmp')):
            #     config_tweaked2.HB_amplitude=config_tweaked.HB_amplitude/2
            # if any(df_out['Diad1_refit'].str.contains('HB1_HighAmp')):
            #     config_tweaked2.HB_amplitude=config_tweaked.HB_amplitude*2


            result, df_out, y_best_fit, x_lin, components, xdat, ydat, ax1_xlim, ax2_xlim,residual_diad_coords, ydat_inrange,  xdat_inrange=fit_gaussian_voigt_generic_diad(config_tweaked2,diad1=True, path=path, filename=filename,
                    xdat=x_diad1, ydat=y_corr_diad1,
                    span=span_diad1, plot_figure=False, Diad_pos=Diad_pos, HB_pos=HB_pos, fit_peaks=config_tweaked.fit_peaks)
            i=i+1
            if i>5:
                print('Got to 5 iteratoins and still couldnt adjust the fit parameters')
                break






    # get a best fit to the baseline using a linspace from the peak fitting
    ybase_xlin=Pf_baseline_diad1(x_lin)

    # We extract the full spectra to plot at the end, and convert to a dataframe
    Spectra_df=get_data(path=path, filename=filename, filetype=filetype)
    Spectra=np.array(Spectra_df)



    # Make nice figure
    if plot_figure is True:

        figure_mosaic="""
        XY
        AB
        CD
        EE
        """

        fig,axes=plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(12, 16))
        fig.suptitle('Diad 1, file= '+ str(filename) + ' \n' + str(df_out['Diad1_refit'].iloc[0]), fontsize=16, x=0.5, y=1.0)

        # Background plot for rea

        # Plot best fit on the LHS, and individual fits on the RHS at the top

        axes['X'].set_title('a) Background fit')
        axes['X'].plot(Diad[:, 0], Diad[:, 1], '-', color='grey')
        axes['X'].plot(Diad_short_diad1[:, 0], Diad_short_diad1[:, 1], '-r', label='Spectra')

        #axes['X'].plot(Baseline[:, 0], Baseline[:, 1], '-b', label='Bck points')
        axes['X'].plot(Baseline_diad1[:, 0], Baseline_diad1[:, 1], '.b', label='Sel. bck. pts')
        axes['X'].plot(Diad_short_diad1[:, 0], Py_base_diad1, '-k', label='bck. poly fit')



        ax1_ymin=np.min(Baseline_diad1[:, 1])-10*np.std(Baseline_diad1[:, 1])
        ax1_ymax=np.max(Baseline_diad1[:, 1])+10*np.std(Baseline_diad1[:, 1])
        ax1_xmin=config1.lower_bck_diad1[0]-30
        ax1_xmax=config1.upper_bck_diad1[1]+30
        # Adding patches


        rect_diad1_b1=patches.Rectangle((config1.lower_bck_diad1[0], ax1_ymin),config1.lower_bck_diad1[1]-config1.lower_bck_diad1[0],ax1_ymax-ax1_ymin,
                                linewidth=1,edgecolor='none',facecolor='cyan', label='sel. bck. region', alpha=0.3, zorder=0)
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



        axes['A'].plot(xdat, ydat,  '.k', label='bck. sub. data')
        axes['A'].plot( x_lin ,y_best_fit, '-g', linewidth=1, label='best fit')
        axes['A'].legend()
        axes['A'].set_ylabel('Intensity')
        axes['A'].set_xlabel('Wavenumber')
        axes['A'].set_xlim(ax1_xlim)
        axes['A'].set_title('c) Overall Best Fit')

    # individual fits
        axes['B'].plot(xdat, ydat, '.k')

        # This is for if there is more than 1 peak, this is when we want to plot the best fit
        if fit_peaks>1:

            axes['B'].plot(x_lin, components.get('lz2_'), '-r', linewidth=2, label='HB1')

        axes['B'].plot(x_lin, components.get('lz1_'), '-b', linewidth=2, label='Diad1')
        if config1.fit_gauss is not False:
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

        if fit_peaks>2:
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




        axes['C'].set_title('d) Peaks overlain on data before subtraction')
        axes['C'].plot(Diad_short_diad1[:, 0], Diad_short_diad1[:, 1], '.r', label='Data')
        axes['C'].plot(Baseline_diad1[:, 0], Baseline_diad1[:, 1], '.b', label='bck')
        axes['C'].plot(Diad_short_diad1[:, 0], Py_base_diad1, '-k', label='Poly bck fit')


        axes['C'].set_ylabel('Intensity')
        axes['C'].set_xlabel('Wavenumber')


        if config1.fit_gauss is not False:

            axes['C'].plot(x_lin, components.get('bkg_')+ybase_xlin, '-m', label='Gaussian bck', linewidth=2)

        axes['C'].plot( x_lin ,y_best_fit+ybase_xlin, '-g', linewidth=2, label='Best Fit')
        axes['C'].plot(x_lin, components.get('lz1_')+ybase_xlin, '-b', label='Diad1', linewidth=1)

        if fit_peaks>1:
            axes['C'].plot(x_lin, components.get('lz2_')+ybase_xlin, '-r', label='HB1', linewidth=1)




        axes['C'].legend(ncol=3, loc='lower center')
        axes['C'].set_xlim([axc_xmin, axc_xmax])
        axes['C'].set_ylim([axc_ymin, axc_ymax])
        #axes['C'].plot(Diad_short[:, 0], Diad_short[:, 1], '"r', label='Data')


        # Residual on plot D
        axes['D'].set_title('f) Residuals')
        axes['D'].plot([df_out['Diad1_Voigt_Cent'], df_out['Diad1_Voigt_Cent']], [np.min(residual_diad_coords), np.max(residual_diad_coords)], ':b')
        if fit_peaks>2:
            axes['D'].plot([df_out['HB1_Cent'], df_out['HB1_Cent']], [np.min(residual_diad_coords), np.max(residual_diad_coords)], ':r')

        axes['D'].plot(xdat_inrange, residual_diad_coords, 'ok', mfc='c' )
        axes['D'].plot(xdat_inrange, residual_diad_coords, '-c' )
        axes['D'].set_ylabel('Residual')
        axes['D'].set_xlabel('Wavenumber')
        # axes['D'].set_xlim(ax1_xlim)
        # axes['D'].set_xlim(ax2_xlim)
        Local_Residual_diad1=residual_diad_coords[((xdat_inrange>(df_out['Diad1_Voigt_Cent'][0]-config1.x_range_residual))
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

    # Lets calculate peak skewness here.
    Skew50=assess_diad1_skewness(config1=diad1_fit_config(), int_cut_off=0.5, path=path, filename=filename, filetype=filetype,
    skewness='abs', height=1, prominence=5, width=0.5, plot_figure=False)
    Skew80=assess_diad1_skewness(config1=diad1_fit_config(), int_cut_off=0.3, path=path, filename=filename, filetype=filetype,
    skewness='abs', height=1, prominence=5, width=0.5, plot_figure=False)

    df_out['Diad1_Asym50']=Skew50['Skewness_diad1']
    df_out['Diad1_Asym70']=Skew80['Skewness_diad1']
    df_out['Diad1_Yuan2017_sym_factor']=(df_out['Diad1_fwhm'])*(df_out['Diad1_Asym50']-1)
    df_out['Diad1_Remigi2021_BSF']=df_out['Diad1_fwhm']/df_out['Diad1_Combofit_Height']




    if  config1.return_other_params is False:
        return df_out
    else:
        return df_out, result, y_best_fit, x_lin


def combine_diad_outputs(*, filename=None, prefix=True,
Diad1_fit=None, Diad2_fit=None, to_csv=False,
to_clipboard=False, path=None):
    """ This function combines the dataframes from fit_diad_2_w_bck and fit_diad_1_w_bck into a
    single dataframe

    Parameters
    ---------------

    filename: str
        filename data is for

    prefix: bool
        if True, splits filename on ' ', if False, leaves as is

    Diad1_fit: pd.DataFrame
        pd.DataFrame of fit from fit_diad_1_w_bck function

    Diad2_fit: pd.DataFrame
        pd.DataFrame of fit from fit_diad_2_w_bck function

    to_clipboard: bool
        If True, copies merged dataframe to clipboard

    to_csv: bool
        if True, saves fit to Csv, by making a subfolder 'Diad_Fits' inside the folder 'path'




    """

    if prefix is True:
        filename = filename.split(' ', 1)[1]

    if Diad1_fit is not None and Diad2_fit is not None:

        combo=pd.concat([Diad1_fit, Diad2_fit], axis=1)
        # Fill any columns which dont' exist so can re-order the same every time
        if 'HB1_Cent' not in combo.columns:
            combo['HB1_Cent']=np.nan
            combo['HB1_Area']=np.nan
            combo['HB1_Sigma']=np.nan
        if 'HB2_Cent' not in combo.columns:
            combo['HB2_Cent']=np.nan
            combo['HB2_Area']=np.nan
            combo['HB2_Sigma']=np.nan
        if 'C13_Cent' not in combo.columns:
            combo['C13_Cent']=np.nan
            combo['C13_Area']=np.nan
            combo['C13_Sigma']=np.nan
        if 'Diad1_Gauss_Cent' not in combo.columns:
            combo['Diad1_Gauss_Cent']=np.nan
            combo['Diad1_Gauss_Area']=np.nan
            combo['Diad1_Gauss_Sigma']=np.nan
        if 'Diad2_Gauss_Cent' not in combo.columns:
            combo['Diad2_Gauss_Cent']=np.nan
            combo['Diad2_Gauss_Area']=np.nan
            combo['Diad2_Gauss_Sigma']=np.nan


        combo['Splitting']=combo['Diad2_Voigt_Cent']-combo['Diad1_Voigt_Cent']
        #combo['Split_err_abs']=combo['Diad1_cent_err']+combo['Diad2_cent_err']
        combo['Split_σ']=(combo['Diad1_cent_err']**2+combo['Diad2_cent_err']**2)**0.5
        cols_to_move = ['Splitting', 'Split_σ',  'Diad1_Combofit_Cent', 'Diad1_cent_err', 'Diad1_Combofit_Height', 'Diad1_Voigt_Cent', 'Diad1_Voigt_Area', 'Diad1_Voigt_Sigma',  'Diad1_Residual', 'Diad1_Prop_Lor', 'Diad1_fwhm', 'Diad1_refit','Diad2_Combofit_Cent', 'Diad2_cent_err',  'Diad2_Combofit_Height', 'Diad2_Voigt_Cent', 'Diad2_Voigt_Area', 'Diad2_Voigt_Sigma', 'Diad2_Voigt_Gamma', 'Diad2_Residual', 'Diad2_Prop_Lor', 'Diad2_fwhm', 'Diad2_refit',
                    'HB1_Cent', 'HB1_Area', 'HB1_Sigma', 'HB2_Cent', 'HB2_Area', 'HB2_Sigma', 'C13_Cent', 'C13_Area', 'C13_Sigma',
                    'Diad2_Gauss_Cent', 'Diad2_Gauss_Area','Diad2_Gauss_Sigma', 'Diad1_Gauss_Cent', 'Diad1_Gauss_Area','Diad1_Gauss_Sigma', 'Diad1_Asym50', 'Diad1_Asym70', 'Diad1_Yuan2017_sym_factor',
'Diad1_Remigi2021_BSF','Diad2_Asym50', 'Diad2_Asym70', 'Diad2_Yuan2017_sym_factor',
'Diad2_Remigi2021_BSF']




        combo_f = combo[cols_to_move + [
                col for col in combo.columns if col not in cols_to_move]]
        combo_f=combo_f.iloc[:, 0:len(cols_to_move)]
        file=filename.rsplit('.txt', 1)[0]
        combo_f.insert(0, 'filename', file)

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




def calculate_HB_Diad_area_ratio(df):
    """" Calculates two area ratios of hotbands and diads"""
    df_c=df.copy()
    df_c['HB_Diad_Ratio']=(df_c['HB1_Area']+df_c['HB2_Area'])/(df_c['Diad1_Voigt_Area']+df_c['Diad2_Voigt_Area'])
    df_c['HB_Total_Ratio']=(df_c['HB1_Area']+df_c['HB2_Area'])/(df_c['Diad1_Voigt_Area']+df_c['Diad2_Voigt_Area']+df_c['HB1_Area']+df_c['HB2_Area'])
    return df_c






## Fit generic peak

@dataclass
class generic_peak_config:

    # Name that gets stamped onto fits
    name: str= 'generic'
    # Selecting the two background positions
    lower_bck: Tuple[float, float]=(1060, 1065)
    upper_bck: Tuple[float, float]=(1120, 1130)

    #
    model_name: str='PseudoVoigtModel'
    x_range_bck: float=10

    # Background degree of polynomial
    N_poly_carb_bck: float =1
    # Seletcting the amplitude
    amplitude: float =1000

    # Selecting the peak center
    cent: float =1090

    # outlier sigma to discard background at
    outlier_sigma: float =12



    # Parameters for Scipy find peaks
    distance: float=10
    prominence: float=5
    width: float=6
    threshold: float=0.1
    height:float =100

    # Excluding a range for cosmic rays etc.
    exclude_range: Optional [Tuple[float, float]]=None


    # Return other parameteres e.g. intermediate outputs
    return_other_params: bool = False

    N_peaks: int= 1

    int_cut_off: float=0.05





def fit_generic_peak(*, config: generic_peak_config=generic_peak_config(),
path=None, filename=None, filetype=None,
 plot_figure=True, dpi=200):

    """ This function fits a generic peak with various options

    config: from dataclass generic_peak_config

        model_name: str, 'GaussianModel', 'VoigtModel', 'PseudoVoigtModel', 'Spline', 'Poly',
            Fits Gaussian, Voigt, or PseudoVoigt, or fits a spline or polynomail and gets area underneath

        name: str
            Name of feature you are fitting. Used to make column headings in output (e.g., if SO2, outputs are Peak_Cent_SO2, Peak_Area_SO2)

        int_cut_off: float
            fits peak out to int_cut_off* peak height.

        lower_bkc: Tuple[float, float]
            Upper and lower x coordinate for background to left of peak

        upper_bkc: Tuple[float, float]
            Upper and lower x coordinate for background to right of peak

        x_range_bck: float
            sets xlim of figure, as the peak center + x_range_bck, and peak cent - x_range_bck

        N_poly_carb_bck: int
            Degree of polynomial to fit to the background positions

        amplitude: float
            Approx amplitude of peak.

        cent: float
            Approx cent of peak

        outlier_sigma: float
            Discards points on the background outside outlier_sigma of the median value

        distance, prominence, threshold, height: floats
            parameters for scipy find_peaks

        exclude_range: Optional [Tuple[float, float]]
            Excludes region of spectra between these values

        dpi: float
            dpi to save figure at

        return_other_params: bool
            if True, returns other peak fit parameters and fits (df, xx_carb, y_carb, result0),
            else just returns df of peak positions, heights etc


    path: str
        Path to file

    filename: str
        filename with file extension

    filetype: str
        choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'


    plot_figure: bool
        If true, plots figure and saves it in a subfolder in the same directory as the filepath specified

    dpi: int
        dpi to save figure at

    Returns
    -----------
    df of peak fits, unless return_other_params is True, then also returns x and y coordinate of fit, and fit params

    """


    Spectra_in=get_data(path=path, filename=filename, filetype=filetype)

    int_cut_off=config.int_cut_off
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
    Spectra_fit=Spectra[ (Spectra[:,0]>upper_0baseline) & (Spectra[:,0]<lower_1baseline) ]

    # To make a nice plot, give 50 wavenumber units on either side as a buffer
    Spectra_plot=Spectra[ (Spectra[:,0]>lower_0baseline-50) & (Spectra[:,0]<upper_1baseline+50) ]

    # Find peaks using Scipy find peaks
    y=Spectra_fit[:, 1]
    x=Spectra_fit[:, 0]
    spec_res=np.abs(x[0]-x[1])


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
    Py_base_fit =Pf_baseline(Spectra_fit[:, 0])



    Baseline_ysub=Pf_baseline(Baseline[:, 0])
    Baseline_x=Baseline[:, 0]
    y_corr= Spectra_short[:, 1]-Py_base
    y_corr_fit=Spectra_fit[:, 1]-Py_base_fit
    x_fit=Spectra_fit[:, 0]
    x_corr=Spectra_short[:, 0]



    if config.model_name not in ['GaussianModel', 'VoigtModel', 'PseudoVoigtModel', 'Spline']:
        raise TypeError('model_name not a permitted input, Please select either GaussianModel, VoigtModel, PseudoVoigtModel, or Spline')


    if config.model_name in ['GaussianModel', 'VoigtModel', 'PseudoVoigtModel']:


        # NOw into the voigt fitting
        if config.model_name=='VoigtModel':
            model0 = VoigtModel()#+ ConstantModel()
        if config.model_name == 'PseudoVoigtModel':
            model0 = PseudoVoigtModel()
        if config.model_name=='GaussianModel':
            model0=GaussianModel()

        # create parameters with initial values
        pars0 = model0.make_params()
        pars0['center'].set(config.cent, min=config.cent-30, max=config.cent+30)
        pars0['amplitude'].set(config.amplitude, min=0)


        init0 = model0.eval(pars0, x=x)
        result0 = model0.fit(y_corr_fit, pars0, x=x)
        Center_p0=result0.best_values.get('center')
        area_p0=result0.best_values.get('amplitude')



        # Make a nice linspace for plotting with smooth curves.
        xx_carb=np.linspace(min(x), max(x), 2000)
        y_carb=result0.eval(x=xx_carb)
        height=np.max(y_carb)

        # Moved this form down below, think it belongs here
        if area_p0 is None:
            area_p0=np.nan
            if area_p0 is not None:
                if area_p0<0:
                    area_p0=np.nan


        df=pd.DataFrame(data={'filename': filename,
    'Peak_Cent_{}'.format(name): Center_p0,
    'Peak_Area_{}'.format(name): area_p0,
    'Peak_Height_{}'.format(name): height,
    'Model_name': config.model_name}, index=[0])

    else:

        if  config.model_name == 'Spline':
            from scipy.interpolate import CubicSpline
            # fit a spline first to find intensity cut off
            mix_spline_sil = interp1d(x_fit, y_corr_fit,
                                    kind='cubic')


            x_new=np.linspace(np.min(x_fit), np.max(x_fit), 1000)
            Baseline_ysub_sil=mix_spline_sil(x_new)

            x_max = x_new[np.argmax(Baseline_ysub_sil)]
            cent=x_max
            Peak_Center=x_max

            max_y=np.max(y_corr_fit)

            # Lets trim x and y based on intensity values
        # Find intensity cut off

            y_int_cut=max_y*int_cut_off

            # Check peak isnt at edge of window, else wont work

            if Peak_Center==np.max(x_fit) or Peak_Center==np.min(x_fit):
                print('peak at edge of window, setting params to nans')



                df=pd.DataFrame(data={'filename': filename,
            'Peak_Cent_{}'.format(name): np.nan,
            'Peak_Area_{}'.format(name): np.nan,
            'Peak_Height_{}'.format(name): np.nan,
            'Model_name': config.model_name}, index=[0])

            else:



            # Split the array into a LHS and a RHS

                LHS_y=Baseline_ysub_sil[x_new<Peak_Center] # Was originally <=
                RHS_y=Baseline_ysub_sil[x_new>Peak_Center]

                LHS_x=x_new[x_new<Peak_Center] # Was originally <=
                RHS_x=x_new[x_new>Peak_Center]

                # Need to flip LHS to put into the find closest function
                LHS_y_flip=np.flip(LHS_y)
                LHS_x_flip=np.flip(LHS_x)




                val=np.argmax(LHS_y_flip<y_int_cut)
                if val == 0 and not (LHS_y_flip[0] < y_int_cut):
                    val = np.argmin(LHS_y_flip)

                val2=np.argmax(RHS_y<y_int_cut)
                if val2 == 0 and not (RHS_y[0] < y_int_cut):
                    val2 = np.argmin(RHS_y)



                # Find nearest x unit to this value
                y_nearest_LHS=LHS_y_flip[val]
                x_nearest_LHS=LHS_x_flip[val]

                y_nearest_RHS=RHS_y[val2]
                x_nearest_RHS=RHS_x[val2]




                diff_LHS=Peak_Center-x_nearest_LHS
                diff_RHS=x_nearest_RHS-Peak_Center



                n_res=0


                x_new_skewness=np.linspace(x_nearest_LHS-spec_res*n_res, x_nearest_RHS+spec_res*n_res)


                x_new=x_new_skewness


                Baseline_ysub_sil=mix_spline_sil(x_new_skewness)






                N_poly_sil='Spline'


                xspace_sil=x_new[1]-x_new[0]
                area_trap = trapz(Baseline_ysub_sil, dx=xspace_sil)
                area_simps = simps(Baseline_ysub_sil, dx=xspace_sil)



                area_p0=area_trap




                df=pd.DataFrame(data={'filename': filename,
            'Peak_Cent_{}'.format(name): x_max,
            'Peak_Area_{}'.format(name): area_p0,
            'Peak_Height_{}'.format(name): max_y,
            'Model_name': config.model_name}, index=[0])






    # Plotting what its doing

    if plot_figure is True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        fig.suptitle('Secondary Phase, file= '+ str(filename), fontsize=12, x=0.5, y=1.0)

        # Plot the peak positions and heights



        ax1.set_title('Background fit')

        ax1.plot(Spectra_plot[:, 0], Spectra_plot[:, 1], '-r', label='Spectra')
        ax1.plot(RH_baseline_filt[:, 0], RH_baseline_filt[:, 1], '.b',
        label='bck points')
        ax1.plot(LH_baseline_filt[:, 0], LH_baseline_filt[:, 1], '-b',
        lw=3, label='_bck points')

        #ax2.plot(x, y_corr_fit, '-r', label='Bck-sub data', lw=0.5)
        ax2.plot(x, y_corr_fit, '.r', label='Bck-sub data')
        ax1.plot(Spectra_short[:, 0], Py_base, '-k', label='Bck Poly')


        if config.model_name != 'Spline' and config.model_name != 'Poly':
            ax2.plot(xx_carb, y_carb, '-k', label='Peak fit')
            cent=df['Peak_Cent_{}'.format(name)].iloc[0]
            ax2.set_xlim([cent-config.x_range_bck, cent+config.x_range_bck])
        else:
            ax2.plot(x_new, Baseline_ysub_sil, '-k')
            ax2.fill_between(x_new, Baseline_ysub_sil, color='yellow', label='fit area', alpha=0.5)


        ax2.set_title('Bkg-subtracted, ' + name + ' peak fit')








    # ax2.set_ylim([min(y_carb)-0.5*(max(y_carb)-min(y_carb)),
    #             max(y_carb)+0.1*max(y_carb),
    # ])



        if find_peaks is True:

            ax1.plot(df_peak_sort_short['pos'], df_peak_sort_short['height'], '*k', mfc='yellow', label='SciPy Peaks')
            ax1.plot(Spectra_short[:, 0], Py_base, '-k')

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






        file=filename.rsplit('.txt', 1)[0]
        if plot_figure is True:
            path3=path+'/'+'Secondary_fit_Images'

            if os.path.exists(path3):
                out='path exists'
            else:
                os.makedirs(path3, exist_ok=False)

            file=filename.rsplit('.txt', 1)[0]
            fig.savefig(path3+'/'+ name +'_Fit_{}.png'.format(file), dpi=dpi)



    if config.return_other_params is True:
        return df, xx_carb, y_carb, result0
    else:
        return df

def peak_prominence(x, y, peak_position):
    """ This function calculates the approximate prominence of a peak,
    by finding the local minimum either side of the peak

    Parameters
    ----------
    x: np.array
        x coordinates (wavenumber)
    y: np.array
        y coordinates (intensity)

    peak:position: int, float
        Peak position to calculate prominence for

    Returns
    --------
    float: prominence of peak in y coordinates.

    """
    # Find the index of the peak position
    peak_index = np.where(x == peak_position)[0][0]

    # Find the local minimums on both sides of the peak
    left_valley_index = peak_index
    right_valley_index = peak_index

    # Find the left valley
    while left_valley_index > 0 and y[left_valley_index] >= y[left_valley_index - 1]:
        left_valley_index -= 1

    # Find the right valley
    while right_valley_index < len(x) - 1 and y[right_valley_index] >= y[right_valley_index + 1]:
        right_valley_index += 1

    # Calculate prominence as the difference between peak height and higher valley
    peak_height = y[peak_index]
    left_valley_height = y[left_valley_index]
    right_valley_height = y[right_valley_index]

    prominence = peak_height - max(left_valley_height, right_valley_height)

    return prominence



# Plotting up secondary peaks

from scipy.signal import find_peaks
def plot_secondary_peaks(*, Diad_Files, path, filetype,
        xlim_plot=[1040, 1200], xlim_peaks=[1060, 1100],
        height=100, threshold=0.1, distance=10, prominence=5, width=6,
         prominence_filter=False, sigma=3, sigma_window=30, find_peaks_filter=False,
         just_plot=False, yscale=0.2, gaussian_smooth=False, smoothing_window=5, smooth_std=3):

    """ This function plots spectra stacked vertically ontop of each other, in a specific region.
    It also has two options to identify peak positions.

    Parameters
    ---------------
    Diad_Files: list
        List of filenames to plot

    path: str
        Path where spectra are stored

    filetype: str
        Choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'

    xlim_plot: Tuple[float, float]
        Range of x coordinates to make plot over

    xlim_peaks: Tuple[float, float]
        Range of x coordinates to search for peaks in

    Optional parameters for Smoothing spectra prior to finding peaks

        gaussian_smooth:bool - If True, applies a gaussian smoothing filter using np.convolve defined by a
         smoothing_window and a smooth_std.


    If you want to actually ID peaks rather than just plot them, choose from:

    prominence_filter: bool
        if True, finds max y coordinate in the window xlim_peaks. It then calculates the prominence by
        comparing this peak height to the average of the median of the 3 datapoints at the left and right of the window.
        If the peak is more than 'prominence' units above the average of these two end point medians, it
        is classified as a peak.
        Also need to specify prominence: float

    find_peaks_filter: bool

        if True, uses scipy find peaks to find peaks. Can tweak distance, prominence, width,
        threshold, and height.

    just_plot: bool
        if True, just plots spectra, doesnt try to find peaks

    y_scale: float
        Figure size is 2+y_scale * number of spectra

    Returns
    -------------
    df_peaks, x_data, y_data, fig

    df_peaks: pd.DataFrame for each file of peak position, height, prominence and file name. Filled with Nans if didnt find a peak
    x_data: np.array of x coordinates (Assumes the same for all files)
    y_data: np.array of y coordinates of each spectra
    fig: figure it has plotted


    """
    if prominence_filter is True and (find_peaks_filter is True or just_plot is True):
        raise TypeError('Only one of sigma_filter, ind_peaks_filter and just_plot can be True')
    fig, (ax1) = plt.subplots(1, 1, figsize=(10,2+yscale*len(Diad_Files)))

    i=0
    Y=0
    peak_pos_saved=np.empty(len(Diad_Files), dtype=float)
    peak_prom_saved=np.empty(len(Diad_Files), dtype=float)
    peak_height_saved=np.empty(len(Diad_Files), dtype=float)
    peak_bck=np.empty(len(Diad_Files), dtype=float)
    y_star=np.empty(len(Diad_Files), dtype=float)
    yplot=np.empty(len(Diad_Files), dtype=float)
    Diad_df=get_data(path=path, filename=Diad_Files[0], filetype=filetype)
    x_data=Diad_df[:, 0]
    y_data=np.empty([  len(x_data), len(Diad_Files)], float)


    for file in Diad_Files:




        Diad_df=get_data(path=path, filename=file, filetype=filetype)
        Diad=np.array(Diad_df)


        # First lets use find peaks
        y2=Diad[:, 1]
        x=Diad[:, 0]

        if gaussian_smooth is True:
            y = np.convolve(y2, gaussian(smoothing_window, std=smooth_std), mode='same')
        else:
            y=y2



        # Region of interest
        Region_peaks = ((x>xlim_peaks[0])& (x<xlim_peaks[1]))
        Region_plot=((x>xlim_plot[0])& (x<xlim_plot[1]))
        x_trim=x[Region_peaks]
        y_trim=y[Region_peaks]
        x_plot=x[Region_plot]
        y_plot=y[Region_plot]
        y_data[:, i]=y



        # IF using scipy find peaks
        if find_peaks_filter is True:
            # Find peaks for trimmed spectra (e.g. in region peaks have been asked for)
            peaks = find_peaks(y_trim,height = height, threshold = threshold,
            distance = distance, prominence=prominence, width=width)


            height_PEAK = peaks[1]['peak_heights'] #list of the heights of the peaks
            peak_pos = x_trim[peaks[0]] #list of the peaks positions

            df_sort=pd.DataFrame(data={'pos': peak_pos,
                                'height': height_PEAK})

            df_peak_sort=df_sort.sort_values('height', axis=0, ascending=False)

            # Trim number of peaks based on user-defined N peaks



            #print(df_peak_sort_short)
            if len(df_peak_sort>=1):

                df_peak_sort_short=df_peak_sort[0:1]
                peak_pos_saved[i]=df_peak_sort_short['pos'].values
                peak_height_saved[i]=df_peak_sort_short['height'].values
            else:

                peak_pos_saved[i]=np.nan
                peak_height_saved[i]=np.nan

            peak_bck[i]=np.quantile(y_plot, 0.2)
            av_y=np.quantile(y_plot, 0.5)
            Diff=np.max(y_plot)-np.min(y_plot)
            Y_sum=np.max(y_plot)/Diff

            ax1.plot(x_plot, ((y_plot-np.min(y_plot))/Diff)+i, '-r')
            ax1.plot(x_plot, ((y_plot-np.min(y_plot))/Diff)+i, '.k', ms=1)
            if len(height_PEAK)>0:


                ax1.plot(df_peak_sort_short['pos'], (df_peak_sort_short['height']-np.min(y_plot))/Diff+i, '*k', mfc='yellow', ms=10)

            yplot[i]=np.min(((y_plot-np.min(y_plot))/Diff)+i)

            ax1.annotate(str(file), xy=(xlim_plot[1]+0.5, yplot[i]),
                         xycoords="data", fontsize=8)
            #ax1.set_xlim(xlim)
            Y=Y+Y_sum


            # calculate prominence
            y_edge1=np.median(y_trim[0:3])
            y_edge2=np.median(y_trim[-3:])
            peak_prom_saved[i]=peak_height_saved[i]-(y_edge1+y_edge2)/2

            i=i+1


        elif prominence_filter is True:
                # Find max value in region

            maxy=np.max(y_trim)
            xpos=x_trim[y_trim==maxy][0]
            #print(xpos)
            # if (xpos+sigma_window)>np.max(x_trim):
            #     Raise TypeError('Your peak finding window is smaller than sigma window')
            # y_around_max=y_trim[(x_trim>(xpos+sigma_window))&(x_trim<(xpos-sigma_window))]
            # stdy=np.nanstd(y_around_max)
            # mediany=np.quantile(y_around_max, 0.5)
            # if maxy>mediany+stdy*sigma+prominence:
            #     pos_x=x_trim[y_trim==maxy][0]
            # else:
            #     pos_x=np.nan

            #prominence_calc = peak_prominence(x_trim, y_trim, xpos)

            y_edge1=np.median(y_trim[0:3])
            y_edge2=np.median(y_trim[-3:])
            #print(y_edge2)
            prominence_calc=maxy-(y_edge1+y_edge2)/2
            peak_prom_saved[i]=prominence_calc



            if prominence_calc>prominence:
                pos_x=x_trim[y_trim==maxy][0]
            else:
                pos_x=np.nan




            if pos_x>0:
                peak_pos_saved[i]=pos_x
                peak_height_saved[i]=maxy

            else:
                peak_pos_saved[i]=np.nan
                peak_height_saved[i]=np.nan

            peak_bck[i]=np.quantile(y_plot, 0.2)
            av_y=np.quantile(y_plot, 0.5)
            Diff=np.max(y_plot)-np.min(y_plot)
            Y_sum=np.max(y_plot)/Diff
            # IF there is something that passes the filter
            if pos_x>0:
                y_star[i]=(peak_height_saved[i]-np.min(y_plot))/Diff+i
                ax1.plot(pos_x, y_star[i], '*k', mfc='yellow', ms=10)
            else:
                y_star[i]=np.nan

            ax1.plot(x_plot, ((y_plot-np.min(y_plot))/Diff)+i, '-r')
            ax1.plot(x_plot, ((y_plot-np.min(y_plot))/Diff)+i, '.k', ms=1)
            yplot[i]=np.min(((y_plot-np.min(y_plot))/Diff)+i)
            ax1.annotate(str(file), xy=(xlim_plot[1]+0.5, yplot[i]),
                         xycoords="data", fontsize=8)
            Y=Y+Y_sum
            i=i+1

        elif just_plot is True:
            Diff=np.max(y_plot)-np.min(y_plot)
            Y_sum=np.max(y_plot)/Diff
            ax1.plot(x_plot, ((y_plot-np.min(y_plot))/Diff)+i, '-r')
            ax1.plot(x_plot, ((y_plot-np.min(y_plot))/Diff)+i, '.k', ms=1)
            yplot[i]=np.min(((y_plot-np.min(y_plot))/Diff)+i)
            ax1.annotate(str(file), xy=(xlim_plot[1]+0.5, yplot[i]),
                         xycoords="data", fontsize=8)
            Y=Y+Y_sum
            i=i+1



    df_peaks=pd.DataFrame(data={'pos': peak_pos_saved,
                                        'height': peak_height_saved,
                                            'prom': peak_prom_saved})

    #if sigma_filter is True:
        #ax1.plot(df_peaks['pos'][y_star>0], y_star[y_star>0], '*k', mfc='yellow', ms=10)










    ax1.plot([1078, 1078], [0, yplot[-1]+2], '-g', lw=1, label='Na$_2$CO$_3$')
    ax1.plot([1085, 1085], [0, yplot[-1]+2], '-', color='cornflowerblue', lw=1, label='CaCO$_3$')

    ax1.plot([1094, 1094], [0, yplot[-1]+2], '-c', lw=1, label='MgCO$_3$')
    ax1.plot([1097, 1097], [0, yplot[-1]+2], '-b', lw=1, label='Dolomite')
    ax1.plot([1131, 1131], [0, yplot[-1]+2], '-', color='grey', lw=1, label='CaSO$_4$')
    ax1.plot([1136, 1136], [0, yplot[-1]+2], '--', color='rosybrown', lw=1, label='MgSO$_4$')
    ax1.plot([1151, 1151], [0, yplot[-1]+2], '-k', lw=1, label='SO$_2$')


    ax1.fill_between(xlim_peaks, -0.5, yplot[-1]+3, color='blue', alpha=0.1)


    ax1.legend(ncol=7,loc='upper center', fontsize=10,  bbox_to_anchor=[0.5, 1.05])
    ax1.set_ylim([-0.5, yplot[-1]+3])
    ax1.set_xlim([xlim_plot[0], xlim_plot[1]+0.5])

    # df_peaks=pd.DataFrame(data={'filename': Diad_Files,
    #                         'pos': peak_pos_saved,
    #                             'prom': peak_height_saved-peak_bck})



    ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax1.set_yticks([])

    x_data=x_trim

    df_peaks['file_names']=Diad_Files

    return df_peaks, x_data, y_data, fig


## Diad Skewness from DeVitre et al.

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






def assess_diad1_skewness(*,  config1: diad1_fit_config=diad1_fit_config(), int_cut_off=0.3, path=None, filename=None, filetype=None,
skewness='abs', height=1, prominence=5, width=0.5, plot_figure=True, peak_fit_routine=False, peak_pos=None, peak_height=None, dpi=200):
    """ Assesses Skewness of Diad peaks. Useful for identifying mixed L + V phases
    (see DeVitre et al. in review)


    Parameters
    -----------
    config1: diad1_fit_config() dataclass. Import parameters for this function are.

        lower_bck_diad1, upper_bck_diad1: [float, float]
            background position to left and right of overal diad region (inc diad+-HB+-C13)

        N_poly_bck_diad1: int
            degree of polynomial to fit to the background.

        exclude_range1, exclude_range2: Tuple[float, float]
            Range to exclude

    int_cut_off: float
        Value of intensity at which to calculate the skewness (e.g. 0.15X the peak height).

    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read


    filetype: str
        Identifies type of file
        choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'

    skewness: 'abs' or 'dir'
        If 'abs', gives absolute skewness (e.g., largest possible number regardless of direction)
        if 'dir' does skewness as one side over the other


    height, prominence, width: float
        Scipy find peak parameters

    plot_figure: bool
        if True, plots and saves a figure.

    dpi: int
        dpi to save image at

    peak_fit_routine: bool (default False) -
        if True, doesnt find max of spline, but uses the peak position you have found from the diad fitting functions.
        Also needs peak_pos (peak center in cm-1) and peak_height


    Returns
    -----------
    pd.DataFrame with information about file and skewness parameters.

    """



    y_corr_diad1, Py_base_diad1, x_diad1,  Diad_short, Py_base_diad1, Pf_baseline,  Baseline_ysub_diad1, Baseline_x_diad1, Baseline, span=remove_diad_baseline(
    path=path, filename=filename, filetype=filetype, N_poly=config1.N_poly_bck_diad1,
    lower_bck=config1.lower_bck_diad1, upper_bck=config1.upper_bck_diad1, plot_figure=False)



    x_lin_baseline=np.linspace(config1.lower_bck_diad1[0], config1.upper_bck_diad1[1], 100000)
    ybase_xlin=Pf_baseline(x_lin_baseline)



    # Get x and y ready to make a cubic spline through data
    x=x_diad1
    y=y_corr_diad1

    # Fit a  cubic spline
    f2 = interp1d(x, y, kind='cubic')
    x_new=np.linspace(np.min(x), np.max(x), 100000)

    y_cub=f2(x_new)



    # Use Scipy find peaks to get the biggest peak
    if peak_fit_routine is False:
        height=height
        peaks = find_peaks(y_cub, height=height, prominence=prominence, width=width)
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

    # now we are using peak parameters we got from Voigt/Pseudovoigt etc.
    if peak_fit_routine is True:
        Peak_Height=peak_height
        Peak_Center=peak_pos



    # Find intensity cut off
    y_int_cut=Peak_Height.iloc[0]*int_cut_off

    # Split the array into a LHS and a RHS

    LHS_y=y_cub[x_new<=Peak_Center.values]
    RHS_y=y_cub[x_new>Peak_Center.values]

    LHS_x=x_new[x_new<=Peak_Center.values]
    RHS_x=x_new[x_new>Peak_Center.values]

    # Need to flip LHS to put into the find closest function
    LHS_y_flip=np.flip(LHS_y)
    LHS_x_flip=np.flip(LHS_x)

    val=np.argmax(LHS_y_flip<y_int_cut)

    val2=np.argmax(RHS_y<y_int_cut)


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


    if plot_figure is True:

        # Make pretty figure showing background subtractoin routine
        fig, ((ax2, ax1), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
        fig.suptitle('Diad 1 Skewness file= '+ str(filename), fontsize=16, x=0.5, y=1.0)

        ax2.set_title('a) Spectra')
        ax2.plot(Diad_short[:, 0], Diad_short[:, 1], '-r', label='Data')
        ax2.set_xlim([config1.lower_bck_diad1[0]+40, config1.upper_bck_diad1[1]-40])
        ax1.set_title('b) Background fit')
        ax1.plot(Diad_short[:, 0], Diad_short[:, 1], '.-c', label='Data')

        ax1.plot(Baseline[:, 0], Baseline[:, 1], '.b', label='bck')
        ax1.plot(Diad_short[:, 0], Py_base_diad1, '-k', label='bck fit')
        ax3.set_title('c) Background subtracted')
        ax3.plot(x_diad1,y_corr_diad1, '-r', label='Bck sub')

    # Adds cubic interpoloation for inspection

        ax4.set_title('d) Cubic Spline')




        ax4.plot([x_nearest_LHS, Peak_Center.iloc[0]], [y_int_cut, y_int_cut], '-g')

        ax4.annotate(str(np.round(LHS_Center.values[0], 2)),
                     xy=(x_nearest_LHS-3, y_int_cut+(Peak_Height-y_int_cut)/10), xycoords="data",
                     fontsize=12, color='green')
        ax4.plot(x_nearest_LHS, y_nearest_LHS, '*k', mfc='green',ms=15, label='RH tie')



        ax4.plot([Peak_Center.iloc[0], x_nearest_RHS], [y_int_cut, y_int_cut], '-', color='grey')
        ax4.annotate(str(np.round(RHS_Center.values[0], 2)),
                     xy=(x_nearest_RHS+3, y_int_cut+(Peak_Height.iloc[0]-y_int_cut)/10), xycoords="data",
                      fontsize=12, color='grey')
        ax4.plot(x_nearest_RHS, y_nearest_RHS, '*k', mfc='grey', ms=15, label='LH tie')




        ax4.plot(x, y, '.r')
        ax4.plot(x_new, y_cub, '-k')

        ax4.plot([Peak_Center.iloc[0], Peak_Center.iloc[0]], [Peak_Height.iloc[0], Peak_Height.iloc[0]],
             '*k', mfc='blue', ms=15, label='Scipy Center')


        # Add to plot

        ax4.plot(config1.lower_bck_diad1[0], config1.upper_bck_diad1[1], [y_int_cut, y_int_cut], ':r')




        ax4.set_xlim([x_nearest_LHS-10, x_nearest_RHS+10])
        ax4.set_ylim([0-10, Peak_Height.iloc[0]*1.2])
        ax4.plot([Peak_Center.iloc[0], Peak_Center.iloc[0]], [0, Peak_Height.iloc[0]], ':b')



        ax4.annotate('Skewness='+str(np.round(AS[0], 2)),
                     xy=(x_nearest_RHS+2, y_int_cut+y_int_cut+(Peak_Height.iloc[0]-y_int_cut)/10), xycoords="data",
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



def assess_diad2_skewness(*,  config1: diad2_fit_config=diad1_fit_config(), int_cut_off=0.3, path=None, filename=None, filetype=None,
skewness='abs', height=1, prominence=5, width=0.5, plot_figure=True, dpi=200, peak_fit_routine=False, peak_pos=None, peak_height=None):

    """ Assesses Skewness of Diad peaks. Useful for identifying mixed L + V phases
    (see DeVitre et al. in review)


    Parameters
    -----------
    config1: diad2_fit_config() dataclass. Import parameters for this function are.

        lower_bck_diad2, upper_bck_diad2: [float, float]
            background position to left and right of overal diad region (inc diad+-HB+-C13)

        N_poly_bck_diad1: int
            degree of polynomial to fit to the background.

        exclude_range1, exclude_range2: Tuple[float, float]
            Range to exclude

    int_cut_off: float
        Value of intensity at which to calculate the skewness (e.g. 0.15X the peak height).

    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read


    filetype: str
        Identifies type of file
        choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'

    skewness: 'abs' or 'dir'
        If 'abs', gives absolute skewness (e.g., largest possible number regardless of direction)
        if 'dir' does skewness as one side over the other


    height, prominence, width: float
        Scipy find peak parameters

    plot_figure: bool
        if True, plots and saves a figure.

    dpi: int
        dpi to save image at

    peak_fit_routine: bool (default False) -
        if True, doesnt find max of spline, but uses the peak position you have found from the diad fitting functions.
        Also needs peak_pos (peak center in cm-1) and peak_height



    Returns
    -----------
    pd.DataFrame with information about file and skewness parameters.

    """


# First, do the background subtraction
    y_corr_diad2, Py_base_diad2, x_diad2,  Diad_short, Py_base_diad2, Pf_baseline,  Baseline_ysub_diad2, Baseline_x_diad2, Baseline, span=remove_diad_baseline(
    path=path, filename=filename, filetype=filetype, N_poly=config1.N_poly_bck_diad2,
    lower_bck=config1.lower_bck_diad2, upper_bck=config1.upper_bck_diad2, plot_figure=False)



    x_lin_baseline=np.linspace(config1.lower_bck_diad2[0], config1.upper_bck_diad2[1], 100000)
    ybase_xlin=Pf_baseline(x_lin_baseline)



# Get x and y for cubic spline
    x=x_diad2
    y=y_corr_diad2

# Fits a  cubic spline
    f2 = interp1d(x, y, kind='cubic')
    x_new=np.linspace(np.min(x), np.max(x), 100000)

    y_cub=f2(x_new)


    if peak_fit_routine is False:
# Use Scipy find peaks to get that cubic peak
        peaks = find_peaks(y_cub, height=height, prominence=prominence, width=width)
        peak_height=peaks[1]['peak_heights']
        peak_pos = x_new[peaks[0]]

        # find max peak.  put into df because i'm lazy
        peak_df=pd.DataFrame(data={'pos': peak_pos,
                            'height': peak_height})

        df_peak_sort=peak_df.sort_values('height', axis=0, ascending=False)
        df_peak_sort_trim=df_peak_sort[0:1]
        Peak_Center=df_peak_sort_trim['pos'].iloc[0]
        Peak_Height=df_peak_sort_trim['height'].iloc[0]

    if peak_fit_routine is True:
        Peak_Height=peak_height[0]
        Peak_Center=peak_pos[0]




# Find intensity cut off
    y_int_cut=Peak_Height*int_cut_off

    # Split the array into a LHS and a RHS
    LHS_y=y_cub[x_new<=Peak_Center]
    RHS_y=y_cub[x_new>Peak_Center]

    LHS_x=x_new[x_new<=Peak_Center]
    RHS_x=x_new[x_new>Peak_Center]

    # Need to flip LHS to put into the find closest function
    LHS_y_flip=np.flip(LHS_y)
    LHS_x_flip=np.flip(LHS_x)

    val=np.argmax(LHS_y_flip<y_int_cut)

    val2=np.argmax(RHS_y<y_int_cut)


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
        if LHS_Center>RHS_Center:
            AS=LHS_Center/RHS_Center
        else:
            AS=RHS_Center/LHS_Center
    elif skewness=='dir':
        AS=RHS_Center/LHS_Center


    if plot_figure is True:

        # Make pretty figure showing background subtractoin routine
        fig, ((ax2, ax1), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
        fig.suptitle('Diad 2 Skewness file= '+ str(filename), fontsize=16, x=0.5, y=1.0)

        ax2.set_title('a) Spectra')
        ax2.plot(Diad_short[:, 0], Diad_short[:, 1], '-r', label='Data')
        ax2.set_xlim([config1.lower_bck_diad2[0]+40, config1.upper_bck_diad2[1]-40])
        ax1.set_title('b) Background fit')
        ax1.plot(Diad_short[:, 0], Diad_short[:, 1], '.-r', label='Data')

        ax1.plot(Baseline[:, 0], Baseline[:, 1], '.b', label='bck')
        ax1.plot(Diad_short[:, 0], Py_base_diad2, '-k', label='bck fit')
        ax3.set_title('c) Background subtracted')
        ax3.plot( x_diad2,y_corr_diad2, '-r', label='Bck sub')

    # Adds cubic interpoloation for inspection

        ax4.set_title('d) Cubic Spline')
        ax4.plot([x_nearest_LHS, Peak_Center], [y_int_cut, y_int_cut], '-g')
        ax4.annotate(str(np.round(LHS_Center, 2)),
                     xy=(x_nearest_LHS-3, y_int_cut+(Peak_Height-y_int_cut)/10), xycoords="data",
                     fontsize=12, color='green')
        ax4.plot(x_nearest_LHS, y_nearest_LHS, '*k', mfc='green',ms=15, label='RH tie')



        ax4.plot([Peak_Center, x_nearest_RHS], [y_int_cut, y_int_cut], '-', color='grey')
        ax4.annotate(str(np.round(RHS_Center, 2)),
                     xy=(x_nearest_RHS+3, y_int_cut+(Peak_Height-y_int_cut)/10), xycoords="data",
                      fontsize=12, color='grey')
        ax4.plot(x_nearest_RHS, y_nearest_RHS, '*k', mfc='grey', ms=15, label='LH tie')




        ax4.plot(x, y, '.r')
        ax4.plot(x_new, y_cub, '-k')

        ax4.plot([Peak_Center, Peak_Center], [Peak_Height, Peak_Height],
             '*k', mfc='blue', ms=15, label='Scipy Center')


        # Add to plot

        ax4.plot(config1.lower_bck_diad2[0], config1.upper_bck_diad2[1], [y_int_cut, y_int_cut], ':r')




        ax4.set_xlim([x_nearest_LHS-10, x_nearest_RHS+10])
        ax4.set_ylim([0-10, Peak_Height*1.2])
        ax4.plot([Peak_Center, Peak_Center], [0, Peak_Height], ':b')



        ax4.annotate('Skewness='+str(np.round(AS, 2)),
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
                              'RHS_tie_diad2': x_nearest_RHS}, index=[0])

    return df_out


def loop_diad_skewness(*, Diad_files, path=None, filetype=None,  skewness='abs', sort=False, int_cut_off=0.15,
config_diad1: diad1_fit_config=diad1_fit_config(), config_diad2: diad2_fit_config=diad2_fit_config(), prominence=10, width=1, height=1):
    """ Loops over all supplied files to calculate skewness for multiple spectra using the functions
    assess_diad1_skewness() and assess_diad2_skewness()


    Parameters
    -----------
    Diad_Files: list
        List of filenames to loop through

    path: str
        Folder user wishes to read data from

    filetype: str
        Identifies type of file
        choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'

    skewness: 'abs' or 'dir'
        If 'abs', gives absolute skewness (e.g., largest possible number regardless of direction)
        if 'dir' does skewness as one side over the other



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



    df_diad1 = pd.DataFrame([])
    df_diad2 = pd.DataFrame([])
    df_merged=pd.DataFrame([])
    for i in range(0, len(Diad_files)):

        filename=Diad_files[i]
        print('working on file #'+str(i))

        data_diad1=assess_diad1_skewness(config1=config_diad1,
        int_cut_off=int_cut_off, prominence=prominence, height=height, width=width,
        skewness=skewness, path=path, filename=filename,
        filetype=filetype)

        data_diad2=assess_diad2_skewness(config1=config_diad2,
        int_cut_off=int_cut_off, prominence=prominence, height=height, width=width,
        skewness=skewness, path=path, filename=filename,
        filetype=filetype)


        df_diad1 = pd.concat([df_diad1, data_diad1], axis=0)
        df_diad2 =  pd.concat([df_diad2, data_diad2], axis=0)



    df_combo=pd.concat([df_diad1, df_diad2], axis=1)

    cols_to_move=['filename', 'Skewness_diad1', 'Skewness_diad2']

    df_combo = df_combo[cols_to_move + [
        col for col in df_combo.columns if col not in cols_to_move]]




    return df_combo






