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
import pickle
from pathlib import Path
from DiadFit.ne_lines import *
from scipy.stats import t
from DiadFit.importing_data_files import *
encode="ISO-8859-1"

## Cornell densimeters
def calculate_density_cornell(*, temp='SupCrit', Split, split_err=None):
    """ This function converts Diad Splitting into CO$_2$ density using the densimeters of DeVitre et al. (2021)
    This should only be used for the Cornell Raman, not other Ramans at present

    Parameters
    -------------
    temp: str
        'SupCrit' if measurements done at 37C
        'RoomT' if measurements done at 24C

    Split: int, float, pd.Series, np.array

    Returns
    --------------
    pd.DataFrame
        Prefered Density (based on different equations being merged), and intermediate calculations




    """

    #if temp is "RoomT":
    LowD_RT=-38.34631 + 0.3732578*Split
    HighD_RT=-41.64784 + 0.4058777*Split- 0.1460339*(Split-104.653)**2

    # IF temp is 37
    LowD_SC=-38.62718 + 0.3760427*Split
    MedD_SC=-47.2609 + 0.4596005*Split+ 0.0374189*(Split-103.733)**2-0.0187173*(Split-103.733)**3
    HighD_SC=-42.52782 + 0.4144277*Split- 0.1514429*(Split-104.566)**2

    if isinstance(Split, pd.Series) or isinstance(Split, np.ndarray):

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

    else:
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

                                }, index=[0])


    roomT=df['Temperature']=="RoomT"
    SupCrit=df['Temperature']=="SupCrit"
    # If splitting is 0
    zero=df['Corrected_Splitting']==0

    # Range for SC low density
    min_lowD_SC_Split=df['Corrected_Splitting']>=102.72
    max_lowD_SC_Split=df['Corrected_Splitting']<=103.16
    # Range for SC med density
    min_MD_SC_Split=df['Corrected_Splitting']>103.16
    max_MD_SC_Split=df['Corrected_Splitting']<=104.28
    # Range for SC high density
    min_HD_SC_Split=df['Corrected_Splitting']>=104.28
    max_HD_SC_Split=df['Corrected_Splitting']<=104.95
    # Range for Room T low density
    min_lowD_RoomT_Split=df['Corrected_Splitting']>=102.734115670188
    max_lowD_RoomT_Split=df['Corrected_Splitting']<=103.350311768435
    # Range for Room T high density
    min_HD_RoomT_Split=df['Corrected_Splitting']>=104.407308904012
    max_HD_RoomT_Split=df['Corrected_Splitting']<=105.1
    # Impossible densities, room T
    Imposs_lower_end=(df['Corrected_Splitting']>103.350311768435) & (df['Splitting']<103.88)
    # Impossible densities, room T
    Imposs_upper_end=(df['Corrected_Splitting']<104.407308904012) & (df['Splitting']>103.88)
    # Too low density
    Too_Low_SC=df['Corrected_Splitting']<102.72
    Too_Low_RT=df['Corrected_Splitting']<102.734115670188

    df.loc[zero, 'Preferred D']=0
    df.loc[zero, 'Notes']=0


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
    SplitZero=df['Corrected_Splitting']==0
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


    #df.loc[zero, 'in range']='Y'
    # If high densiy, and beyond the upper calibration limit
    Upper_Cal_RT=df['Corrected_Splitting']>105.1
    Upper_Cal_SC=df['Corrected_Splitting']>104.95

    df.loc[roomT&Upper_Cal_RT, 'Preferred D'] = HighD_RT
    df.loc[roomT&Upper_Cal_RT, 'Notes']='Above upper Cali Limit'
    df.loc[roomT&Upper_Cal_RT, 'in range']='N'

    df.loc[SupCrit&Upper_Cal_SC, 'Preferred D'] = HighD_SC
    df.loc[SupCrit&Upper_Cal_SC, 'Notes']='Above upper Cali Limit'
    df.loc[SupCrit&Upper_Cal_SC, 'in range']='N'

    if split_err is not None:
        df2=calculate_dens_error(temp, Split, split_err)

        df.insert(1, 'dens+1σ', df2['max_dens'])
        df.insert(1, 'dens-1σ', df2['min_dens'])
        df.insert(3, '1σ', (df2['max_dens']-df2['min_dens'])/2 )

    return df

def calculate_dens_error(temp, Split, split_err):

    max_dens=calculate_density_cornell(temp=temp, Split=Split+split_err)
    min_dens=calculate_density_cornell(temp=temp, Split=Split-split_err)
    df=pd.DataFrame(data={
                        'max_dens': max_dens['Preferred D'],
                        'min_dens': min_dens['Preferred D']})

    return df




def propagate_error_split_neon_peakfit(Ne_corr, df_fits):
    """ This function propagates errors in your Ne correction model and peak fits by quadrature

    """
    # Get the error on Neon things
    Ne_err=(Ne_corr['upper_values']-Ne_corr['lower_values'])/2
    # Get the peak fit errors
    Diad1_err=df_fits['Diad1_cent_err'].fillna(0)
    Diad2_err=df_fits['Diad2_cent_err'].fillna(0)
    split_err=(Diad1_err**2 + Diad2_err**2)**0.5
    Combo_err= (((df_fits['Splitting']* (Ne_err))**2) +  (Ne_corr['preferred_values'] *split_err  )**2 )**0.5




    return Combo_err, split_err

## Error for densimeters

def calculate_Densimeter_std_err_values(*, pickle_str, corrected_split, corrected_split_err, CI_neon=0.67, CI_split=0.67, str_d='LowD') :
    """
    This function propagates uncertainty from the densimeter polynomial and the error on the splitting (which comes from both the Ne line correction and the peak fitting error)
    """
    # Corrected splitting
    new_x=corrected_split
    new_x_uncertainty=corrected_split_err

    # This gets the uncertainty from the densimeter
    DiadFit_dir=Path(__file__).parent
    with open(DiadFit_dir/pickle_str, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    N_poly = model.order - 1

    Pf = data['model']
    x = data['x']
    y = data['y']

    # Calculate the residuals
    residuals = y - Pf(x)

    # Calculate the standard deviation of the residuals
    residual_std = np.std(residuals)

    # Calculate the standard errors for the new x values
    mean_x = np.mean(x)
    n = len(x)
    standard_errors = residual_std * np.sqrt(1 + 1 / n + (new_x - mean_x) ** 2 / np.sum((x - mean_x) ** 2))

    # Calculate the degrees of freedom
    df = len(x) - (N_poly + 1)

    # Calculate the t value for the given confidence level
    t_value_split = t.ppf((1 + CI_split) / 2, df)
    t_value_dens = t.ppf((1 + CI_neon) / 2, df)

    # Calculate the prediction intervals
    preferred_values = Pf(new_x)
    lower_values = preferred_values - t_value_dens * standard_errors
    upper_values = preferred_values + t_value_dens * standard_errors
    uncertainty_from_dens=(upper_values-lower_values)/2



    # Calculate the propagated uncertainty in new_x by evaluating at the top and bottom of the value
    max_split=new_x + new_x_uncertainty
    min_split=new_x - new_x_uncertainty
    max_density=  Pf(max_split)
    min_density=  Pf(min_split)
    uncertainty_split=(max_density-min_density)/2


    # Calculate the total uncertainty in the density estimation
    total_uncertainty = np.sqrt(uncertainty_split ** 2 + uncertainty_from_dens ** 2)


    df=pd.DataFrame(data={
        str_d+'_Corrected_Splitting': new_x,
        str_d+'_Density': preferred_values,
        str_d + '_Density_σ': total_uncertainty,
        str_d+'_Density+1σ': preferred_values-total_uncertainty,
        str_d+'_Density-1σ': preferred_values+total_uncertainty,
        str_d+'_Density_σ_dens': (uncertainty_from_dens),
        str_d+'_Density_σ_split': (uncertainty_split),

    })

    return df


## UCBerkeley densimeters
def calculate_density_ucb(*, df_combo, Ne_pickle_str='polyfit_data.pkl',  temp='SupCrit', split_err=0, CI_split=0.67, CI_neon=0.67):
    """ This function converts Diad Splitting into CO$_2$ density using densimeters of UCB

    Parameters
    -------------
    df_combo: pandas DataFrame
        data frame of peak fitting information

    Ne_corr: pandas DataFrame
        dataframe of Ne correction factors

    temp: str
        'SupCrit' if measurements done at 37C
        'RoomT' if measurements done at 24C - Not supported at Berkeley.

    Split: int, float, pd.Series, np.array

    Returns
    --------------
    pd.DataFrame
        Prefered Density (based on different equatoins being merged), and intermediate calculations




    """
    time=df_combo['sec since midnight']
    Ne_corr=calculate_Ne_corr_std_err_values(pickle_str=Ne_pickle_str,
    new_x=time, CI=CI_neon)

    # Lets calculate  corrected splitting and the error on this.
    Split=df_combo['Splitting']*Ne_corr['preferred_values']

    Split_err, pk_err=propagate_error_split_neon_peakfit(Ne_corr=Ne_corr, df_fits=df_combo)
    df_combo['Corrected_Splitting_σ']=Split_err
    df_combo['Corrected_Splitting_σ_Ne']=(Ne_corr['upper_values']*df_combo['Splitting']-Ne_corr['lower_values']*df_combo['Splitting'])/2
    df_combo['Corrected_Splitting_σ_peak_fit']=pk_err

    if temp=='RoomT':
        raise TypeError('Sorry, no UC Berkeley calibration at 24C, please enter temp=SupCrit')
    if isinstance(Split, float) or isinstance(Split, int):
        Split=pd.Series(Split)
    # #if temp is "RoomT":
    DiadFit_dir=Path(__file__).parent

    LowD_RT=-38.34631 + 0.3732578*Split
    HighD_RT=-41.64784 + 0.4058777*Split- 0.1460339*(Split-104.653)**2

    # IF temp is 37
    pickle_str_lowr='Lowrho_polyfit_data.pkl'
    with open(DiadFit_dir/pickle_str_lowr, 'rb') as f:
        lowrho_pickle_data = pickle.load(f)
    pickle_str_medr='Mediumrho_polyfit_data.pkl'
    with open(DiadFit_dir/pickle_str_medr, 'rb') as f:
        medrho_pickle_data = pickle.load(f)
    pickle_str_highr='Highrho_polyfit_data.pkl'
    with open(DiadFit_dir/pickle_str_highr, 'rb') as f:
        highrho_pickle_data = pickle.load(f)

    lowrho_model = lowrho_pickle_data['model']
    medrho_model = medrho_pickle_data['model']
    highrho_model = highrho_pickle_data['model']

    #

    LowD_SC = pd.Series(lowrho_model(Split), index=Split.index)
    lowD_error=calculate_Densimeter_std_err_values(corrected_split=Split, corrected_split_err=Split_err,
    pickle_str=pickle_str_lowr,  CI_neon=CI_neon, CI_split=CI_split, str_d='LowD')
    medD_error=calculate_Densimeter_std_err_values(corrected_split=Split, corrected_split_err=Split_err,
    pickle_str=pickle_str_medr,  CI_neon=CI_neon, CI_split=CI_split, str_d='MedD')
    highD_error=calculate_Densimeter_std_err_values(corrected_split=Split, corrected_split_err=Split_err,
    pickle_str=pickle_str_highr,   CI_neon=CI_neon, CI_split=CI_split,  str_d='HighD')
    MedD_SC = pd.Series(medrho_model(Split), index=Split.index)
    HighD_SC = pd.Series(highrho_model(Split), index=Split.index)




    df=pd.DataFrame(data={'Preferred D': 0,
    'Corrected_Splitting': Split,
    'Preferred D_σ': 0,
    'Preferred D_σ_split': 0,
    'Preferred D_σ_Ne': 0,
    'Preferred D_σ_pkfit': 0,
     'Preferred D_σ_dens': 0,
        'in range': 'Y',
                                'Notes': 'not in range',
                                'LowD_RT':np.nan,
                                'HighD_RT': np.nan,
                                'LowD_SC': LowD_SC,
                                'LowD_SC_σ': lowD_error['LowD_Density_σ'],
                                'MedD_SC': MedD_SC,
                                'MedD_SC_σ': medD_error['MedD_Density_σ'],
                                'HighD_SC': HighD_SC,
                                'HighD_SC_σ': highD_error['HighD_Density_σ'],
                                'Temperature': temp,


                                })




    roomT=df['Temperature']=="RoomT"
    SupCrit=df['Temperature']=="SupCrit"
    # If splitting is 0
    zero=df['Corrected_Splitting']==0

    # Range for SC low density
    min_lowD_SC_Split=df['Corrected_Splitting']>=102.7623598753032
    max_lowD_SC_Split=df['Corrected_Splitting']<=103.1741034592534
    # Range for SC med density
    min_MD_SC_Split=df['Corrected_Splitting']>103.0608505403591
    max_MD_SC_Split=df['Corrected_Splitting']<=104.3836704771313
    # Range for SC high density
    min_HD_SC_Split=df['Corrected_Splitting']>=104.2538992302499
    max_HD_SC_Split=df['Corrected_Splitting']<=105.3438707618937
    # Range for Room T low density
    min_lowD_RoomT_Split=df['Corrected_Splitting']>=102.734115670188
    max_lowD_RoomT_Split=df['Corrected_Splitting']<=103.350311768435
    # Range for Room T high density
    min_HD_RoomT_Split=df['Corrected_Splitting']>=104.407308904012
    max_HD_RoomT_Split=df['Corrected_Splitting']<=105.1
    # Impossible densities, room T
    Imposs_lower_end=(df['Corrected_Splitting']>103.350311768435) & (df['Corrected_Splitting']<103.88)
    # Impossible densities, room T
    Imposs_upper_end=(df['Corrected_Splitting']<104.407308904012) & (df['Corrected_Splitting']>103.88)
    # Too low density
    Too_Low_SC=df['Corrected_Splitting']<102.7623598753032
    Too_Low_RT=df['Corrected_Splitting']<102.734115670188

    df.loc[zero, 'Preferred D']=0
    df.loc[zero, 'Notes']=0





    # If SupCrit, high density
    df.loc[ SupCrit&(min_HD_SC_Split&max_HD_SC_Split), 'Preferred D'] = HighD_SC
    df.loc[ SupCrit&(min_HD_SC_Split&max_HD_SC_Split), 'Preferred D_σ'] = highD_error['HighD_Density_σ']
    df.loc[ SupCrit&(min_HD_SC_Split&max_HD_SC_Split), 'Preferred D_σ_split'] = highD_error['HighD_Density_σ_split']
    df.loc[ SupCrit&(min_HD_SC_Split&max_HD_SC_Split), 'Preferred D_σ_dens'] = highD_error['HighD_Density_σ_dens']
    df.loc[ SupCrit&(min_HD_SC_Split&max_HD_SC_Split), 'Notes']='SupCrit, high density'
    # If SupCrit, Med density
    df.loc[SupCrit&(min_MD_SC_Split&max_MD_SC_Split), 'Preferred D'] = MedD_SC
    df.loc[SupCrit&(min_MD_SC_Split&max_MD_SC_Split), 'Preferred D_σ'] = medD_error['MedD_Density_σ']
    df.loc[SupCrit&(min_MD_SC_Split&max_MD_SC_Split), 'Preferred D_σ_split'] = medD_error['MedD_Density_σ_split']
    df.loc[SupCrit&(min_MD_SC_Split&max_MD_SC_Split), 'Preferred D_σ_dens'] = medD_error['MedD_Density_σ_dens']
    df.loc[SupCrit&(min_MD_SC_Split&max_MD_SC_Split), 'Notes']='SupCrit, Med density'

    # If SupCrit, low density
    df.loc[ SupCrit&(min_lowD_SC_Split&max_lowD_SC_Split), 'Preferred D'] = LowD_SC
    df.loc[ SupCrit&(min_lowD_SC_Split&max_lowD_SC_Split), 'Preferred D_σ'] = lowD_error['LowD_Density_σ']
    df.loc[ SupCrit&(min_lowD_SC_Split&max_lowD_SC_Split), 'Preferred D_σ_split'] = lowD_error['LowD_Density_σ_split']
    df.loc[ SupCrit&(min_lowD_SC_Split&max_lowD_SC_Split), 'Preferred D_σ_dens'] = lowD_error['LowD_Density_σ_dens']
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
    SplitZero=df['Corrected_Splitting']==0
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


    #df.loc[zero, 'in range']='Y'
    # If high densiy, and beyond the upper calibration limit
    Upper_Cal_RT=df['Corrected_Splitting']>105.1
    Upper_Cal_SC=df['Corrected_Splitting']>105.3438707618937

    df.loc[roomT&Upper_Cal_RT, 'Preferred D'] = HighD_RT
    df.loc[roomT&Upper_Cal_RT, 'Notes']='Above upper Cali Limit'
    df.loc[roomT&Upper_Cal_RT, 'in range']='N'

    df.loc[SupCrit&Upper_Cal_SC, 'Preferred D'] = HighD_SC
    df.loc[SupCrit&Upper_Cal_SC, 'Notes']='Above upper Cali Limit'
    df.loc[SupCrit&Upper_Cal_SC, 'in range']='N'


    df_merge1=pd.concat([df_combo, Ne_corr], axis=1).reset_index(drop=True)
    df_merge=pd.concat([df, df_merge1], axis=1).reset_index(drop=True)

    df_merge = df_merge.rename(columns={'Preferred D': 'Density g/cm3'})
    df_merge = df_merge.rename(columns={'Preferred D_σ': 'σ Density g/cm3'})
    df_merge = df_merge.rename(columns={'Preferred D_σ_split': 'σ Density g/cm3 (from Ne+peakfit)'})
    df_merge = df_merge.rename(columns={'Preferred D_σ_dens': 'σ Density g/cm3 (from densimeter)'})
    df_merge = df_merge.rename(columns={'filename_x': 'filename'})


    #
    #

    cols_to_move = ['filename', 'Density g/cm3', 'σ Density g/cm3','σ Density g/cm3 (from Ne+peakfit)', 'σ Density g/cm3 (from densimeter)',
     'Corrected_Splitting', 'Corrected_Splitting_σ',
    'Corrected_Splitting_σ_Ne', 'Corrected_Splitting_σ_peak_fit', 'power (mW)', 'Spectral Center']
    df_merge = df_merge[cols_to_move + [
        col for col in df_merge.columns if col not in cols_to_move]]






    return df_merge

## functions to neatly merge secondary phases in
import os.path
from os import path
def merge_in_carb_SO2(df_combo, file1_name='Carb_Peak_fits.xlsx', file2_name='SO2_Peak_fits.xlsx', prefix=False, str_prefix=" "):
    """
    This function checks for files with secondary phases in the path with names file1_name and file2_name, if they are there
    it will merge them together into a dataframe. It then merges this with the dataframe 'df_combo' of your other fits
    """

    if path.exists(file1_name):
        Carb=pd.read_excel(file1_name)
    else:
        Carb=None
    if path.exists(file2_name):
        SO2=pd.read_excel(file2_name)
    else:
        SO2=None
    if SO2 is not None and Carb is not None:
        Sec_Phases=pd.merge(SO2, Carb, on='filename', how='outer').reset_index(drop=True)
    elif SO2 is not None and Carb is None:
        Sec_Phases=SO2
    elif SO2 is None and Carb is not None:
        Sec_Phases=Carb
    else:
        Sec_Phases=None
    if Sec_Phases is not None:
        print('Made a df!')



    # Remove these to get the pure file name
    if Sec_Phases is not None:
        file_sec_phase=extracting_filenames_generic(
            prefix=prefix, str_prefix=str_prefix,
            names=Sec_Phases['filename'].reset_index(drop=True),
        file_type='.txt')


    else:
        df_combo_sec_phase=df_combo

    df_combo['name_for_matching']=df_combo['Name_for_Secondary_Phases']


    if Sec_Phases is not None:
        Sec_Phases['name_for_matching']=file_sec_phase
        df_combo_sec_phase=df_combo.merge(Sec_Phases,
        on='name_for_matching', how='outer').reset_index(drop=True)

        print(Sec_Phases['name_for_matching'])

    else:
        df_combo_sec_phase=df_combo

    if 'Peak_Area_Carb' in df_combo_sec_phase.columns:
        df_combo_sec_phase['Carb_Diad_Ratio']=(df_combo_sec_phase['Peak_Area_Carb']/(df_combo_sec_phase['Diad1_Voigt_Area']
                        +df_combo_sec_phase['Diad2_Voigt_Area']))
    if 'Peak_Area_SO2' in df_combo_sec_phase.columns:
        df_combo_sec_phase['SO2_Diad_Ratio']=(df_combo_sec_phase['Peak_Area_SO2']/(df_combo_sec_phase['Diad1_Voigt_Area']
                        +df_combo_sec_phase['Diad2_Voigt_Area']))



    return df_combo_sec_phase

## Merge peak fits together
def merge_fit_files():
    """
    This function merges the files Discarded_df.xlsx, Weak_Diads.xlsx, Medium_Diads.xlsx, Strong_Diads.xlsx
    if they exist into one combined dataframe
    """
    import os.path
    from os import path
    if path.exists('Discarded_df.xlsx'):
        discard=pd.read_excel('Discarded_df.xlsx')
    else:
        discard=None
    if path.exists('Weak_Diads.xlsx'):
        grp1=pd.read_excel('Weak_Diads.xlsx')
    else:
        grp1=None
    if path.exists('Medium_Diads.xlsx'):
        grp2=pd.read_excel('Medium_Diads.xlsx')
    else:
        grp2=None
    if path.exists('Strong_Diads.xlsx'):
        grp3=pd.read_excel('Strong_Diads.xlsx')
    else:
        grp3=None
    df2=pd.concat([grp1, grp2, grp3], axis=0).reset_index(drop=True)
    if discard is not None:
        discard_cols=discard[discard.columns.intersection(df2.columns)]
        df2=pd.concat([df2, discard_cols]).reset_index(drop=True)
    return df2





