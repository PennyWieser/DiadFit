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
def calculate_density_cornell_old(*, temp='SupCrit', Split, split_err=None):
    """ This function converts Diad Splitting into CO$_2$ density using the densimeters of DeVitre et al. (2021)
    This should only be used for the Cornell Raman, not other Ramans at present. This is an older version of the function
    calculate_density_cornell - it does not have the same error propagation capabilities, but we keep it here for backwards
    compatability.

    Parameters
    -------------
    temp: str
        'SupCrit' if measurements done at 37C
        'RoomT' if measurements done at 24C

    Split: int, float, pd.Series, np.array
        Corrected splitting in cm-1

    Split_err: int, float, pd.Series (Optional)
        Error on corrected splitting in cm-1


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
        'Corrected_Splitting': Split,
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
        'Corrected_Splitting': Split,
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
    """
    This function is used by the function calculate_density_cornell_old to calculate a max and min density
    corresponding to splitting values + - 1 sigma

    Parameters
    -------------
    temp: str
        'SupCrit' if measurements done at 37C
        'RoomT' if measurements done at 24C

    Split: int, float, pd.Series, np.array
        Corrected splitting in cm-1

    Split_err: int, float, pd.Series (Optional)
        Error on corrected splitting in cm-1

    Returns
    ---------
    pd.DataFrame
        Dataframe with column for max calculated density and min calculated density.
    """

    max_dens=calculate_density_cornell(temp=temp, Split=Split+split_err)
    min_dens=calculate_density_cornell(temp=temp, Split=Split-split_err)
    df=pd.DataFrame(data={
                        'max_dens': max_dens['Preferred D'],
                        'min_dens': min_dens['Preferred D']})

    return df




def propagate_error_split_neon_peakfit(*, df_fits, Ne_corr=None, Ne_err=None, pref_Ne=None):
    """ This function propagates errors in your Ne correction model and peak fits by quadrature.

    Parameters
    -----------------

    df_fits: pd.DataFrame
        Dataframe of peak fitting parameters. Must contain columns for 'Diad1_cent_err', 'Diad2_cent_err', 'Splitting'

    Choose either:

    Ne_corr: pd.DataFrame (Optional)
        Dataframe with columns for 'upper_values' and 'lower values', e.g. the upper and lower bounds of the error on the Ne correction model

    Or

    pref_Ne and Ne_err: float, int, pd.Series, np.array
        A preferred value of the Ne correction factor and the error (e.g. pref_Ne=0.998, Ne_err=0.001). Used for
        rapid peak fitting before developing Ne lines.


    Returns
    -----------------
    two pd.Series, the error on the splitting, and the combined error from the splitting and the Ne correction model.

    """
    # Get the error on Neon things
    if isinstance(Ne_corr, pd.DataFrame):
        Ne_err=(Ne_corr['upper_values']-Ne_corr['lower_values'])/2
        print(np.mean(Ne_err))
        pref_Ne=Ne_corr['preferred_values']

    elif pref_Ne is not None and Ne_err is not None:
        print('using fixed values for Ne error and Ne factor')
    else:
        raise TypeError('you either ne Ne_corr as a dataframe, or to give a value for pref_Ne and Ne_err')




    # Get the peak fit errors
    Diad1_err=df_fits['Diad1_cent_err'].fillna(0)
    Diad2_err=df_fits['Diad2_cent_err'].fillna(0)
    split_err=(Diad1_err**2 + Diad2_err**2)**0.5
    Combo_err= (((df_fits['Splitting']* (Ne_err))**2) +  (pref_Ne *split_err  )**2 )**0.5




    return Combo_err, split_err

## Error for densimeters

def calculate_Densimeter_std_err_values(*, pickle_str, corrected_split, corrected_split_err, CI_dens=0.67, CI_split=0.67, str_d='LowD') :

    """
    This function propagates uncertainty from the densimeter polynomial fit and the overall error on the corrected splitting (from both the peak fitting and the Ne line correction).

    Parameters
    -----------------
    pickle_str: str
        Name of Pickle with regression model for a specific part of the densimeter. Need to be in the same path as the notebook you are calling this in.

    corrected_split: pd.Series
        panda series of corrected splitting (cm-1)

    corrected_split_err: pd. Series
        panda series of error on corrected splitting (contributions from both peak fitting and the Ne correction model if relevant)

    str_d: str
            string of what density equation it came from, appended onto column headings.

    CI_split: float
        confidence interval for splitting propagation. Should be set the same as CI_

    CI_split: float

    Returns
    -----------------
    pd.DataFrame
        Dataframe of preferred density, and error from each source of uncertainty (Density_σ_dens=error from densimeter, Density_σ_split = error from spliting).

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
    mean_x = np.nanmean(x)
    n = len(x)


    standard_errors = residual_std * np.sqrt(1 + 1 / n + (new_x - mean_x) ** 2 / np.sum((x - mean_x) ** 2))

    # Calculate the degrees of freedom
    df = len(x) - (N_poly + 1)

    # Calculate the t value for the given confidence level
    t_value_split = t.ppf((1 + CI_split) / 2, df)
    t_value_dens = t.ppf((1 + CI_dens) / 2, df)

    # Calculate the prediction intervals from the densimeter
    preferred_values = Pf(new_x)
    lower_values = preferred_values - t_value_dens * standard_errors
    upper_values = preferred_values + t_value_dens * standard_errors
    uncertainty_from_dens=(upper_values-lower_values)/2



    # Calculate the propagated uncertainty in splitting
    max_split=new_x + new_x_uncertainty
    min_split=new_x - new_x_uncertainty
    max_density=  Pf(max_split)
    min_density=  Pf(min_split)
    uncertainty_split=(max_density-min_density)/2


    # Calculate the total uncertainty in the density estimation
    total_uncertainty = np.sqrt(uncertainty_split ** 2 + uncertainty_from_dens ** 2)


    df=pd.DataFrame(data={

        str_d+'_Density': preferred_values,
        str_d + '_Density_σ': total_uncertainty,
        str_d+'_Density+1σ': preferred_values-total_uncertainty,
        str_d+'_Density-1σ': preferred_values+total_uncertainty,
        str_d+'_Density_σ_dens': (uncertainty_from_dens),
        str_d+'_Density_σ_split': (uncertainty_split),

    })

    return df
## Function for if we dont have a densimeter yet

def calculate_errors_no_densimeter(*, df_combo, Ne_pickle_str='polyfit_data.pkl',  temp='SupCrit', split_err=0, CI_split=0.67, CI_neon=0.67):
    """ This function calculates the error from just the Ne line correction method. It is largely redundant, 
    still in use for some older projects using an old workflow

    """
    time=df_combo['sec since midnight']
    Ne_corr=calculate_Ne_corr_std_err_values(pickle_str=Ne_pickle_str,
    new_x=time, CI=CI_neon)

    # Lets calculate  corrected splitting and the error on this.
    Split=df_combo['Splitting']*Ne_corr['preferred_values']
    df_combo['Corrected_Splitting']=Split
    Split_err, pk_err=propagate_error_split_neon_peakfit(Ne_corr=Ne_corr, df_fits=df_combo)
    df_combo['Corrected_Splitting_σ']=Split_err
    df_combo['Corrected_Splitting_σ_Ne']=(Ne_corr['upper_values']*df_combo['Splitting']-Ne_corr['lower_values']*df_combo['Splitting'])/2
    df_combo['Corrected_Splitting_σ_peak_fit']=pk_err

    cols_to_move = ['filename',
     'Corrected_Splitting', 'Corrected_Splitting_σ',
    'Corrected_Splitting_σ_Ne', 'Corrected_Splitting_σ_peak_fit']
    df_merge = df_combo[cols_to_move + [
        col for col in df_combo.columns if col not in cols_to_move]]

    return df_combo



def calculate_density_cornell(*,  lab='CMASS', df_combo=None, temp='SupCrit', 
CI_split=0.67, CI_neon=0.67,  Ne_pickle_str=None, pref_Ne=None, Ne_err=None, corrected_split=None, split_err=None):
    """ This function converts Diad Splitting into CO$_2$ density using the Cornell densimeters. Use lab='CCMR' for CCMR and lab='CMASS' for Esteban Gazels lab. 

    Parameters
    -------------

    lab: str. 'CMASS' or 'CCMR'
        Name of the lab where the analy
    Either:

    df_combo: pandas DataFrame
        data frame of peak fitting information
        
    Or:
    corrected_split: pd.Series
        Corrected splitting  (cm-1)  
        
    Split_err: float, int
        Error on corrected splitting

    temp: str
        'SupCrit' if measurements done at 37C
        'RoomT' if measurements done at 24C - Not supported yet but could be added if needed. 

    CI_neon: float
        Default 0.67. Confidence interval to use, e.g. 0.67 returns 1 sigma uncertainties. If you use another number,
        note the column headings will still say sigma.

    CI_split: float
        Default 0.67. Confidence interval to use, e.g. 0.67 returns 1 sigma uncertainties. If you use another number,
        note the column headings will still say sigma.
        
        


    Either

    Ne_pickle_str: str
        Name of Ne correction model

    OR

    pref_Ne, Ne_err: float, int
        For quick and dirty fitting can pass a preferred value for your instrument before you have a chance to 
        regress the Ne lines (useful when first analysing new samples. )




    Returns
    --------------
    pd.DataFrame
        Prefered Density (based on different equatoins being merged), and intermediate calculations
    """
    if corrected_split is not None:
        Split=corrected_split
    if df_combo is not None:
        df_combo_c=df_combo.copy()
        time=df_combo_c['sec since midnight']

        if Ne_pickle_str is not None:

            # Calculating the upper and lower values for Ne to get that error
            Ne_corr=calculate_Ne_corr_std_err_values(pickle_str=Ne_pickle_str,
            new_x=time, CI=CI_neon)
            # Extracting preferred correction values
            pref_Ne=Ne_corr['preferred_values']
            Split_err, pk_err=propagate_error_split_neon_peakfit(Ne_corr=Ne_corr, df_fits=df_combo_c)

            df_combo_c['Corrected_Splitting_σ']=Split_err
            df_combo_c['Corrected_Splitting_σ_Ne']=(Ne_corr['upper_values']*df_combo_c['Splitting']-Ne_corr['lower_values']*df_combo_c['Splitting'])/2
            df_combo_c['Corrected_Splitting_σ_peak_fit']=pk_err

        # If using a single value for quick dirty fitting
        else:
            Split_err, pk_err=propagate_error_split_neon_peakfit(df_fits=df_combo_c, Ne_err=Ne_err, pref_Ne=pref_Ne)



            df_combo_c['Corrected_Splitting_σ']=Split_err

            df_combo_c['Corrected_Splitting_σ_Ne']=((Ne_err+pref_Ne)*df_combo_c['Splitting']-(Ne_err-pref_Ne)*df_combo_c['Splitting'])/2
            df_combo_c['Corrected_Splitting_σ_peak_fit']=pk_err

        Split=df_combo_c['Splitting']*pref_Ne

    else:
       Split_err=(split_err*Split).astype(float)


    if temp=='RoomT':
        raise TypeError('Sorry, this function doesnt yet support the calibration at 24C, please enter temp=SupCrit')
    if isinstance(Split, float) or isinstance(Split, int):
        Split=pd.Series(Split)
    # #if temp is "RoomT":
    DiadFit_dir=Path(__file__).parent

    LowD_RT=-38.34631 + 0.3732578*Split
    HighD_RT=-41.64784 + 0.4058777*Split- 0.1460339*(Split-104.653)**2

    # IF temp is 37
    if lab=='CMASS':
        # This gets the densimeter at low density
        pickle_str_lowr='Lowrho_polyfit_data_CMASS.pkl'
        with open(DiadFit_dir/pickle_str_lowr, 'rb') as f:
            lowrho_pickle_data = pickle.load(f)
    
        # This gets the densimeter at medium density
        pickle_str_medr='Mediumrho_polyfit_data_CMASS.pkl'
        with open(DiadFit_dir/pickle_str_medr, 'rb') as f:
            medrho_pickle_data = pickle.load(f)
        # This gets the densimeter at high density.
        pickle_str_highr='Highrho_polyfit_data_CMASS.pkl'
        with open(DiadFit_dir/pickle_str_highr, 'rb') as f:
            highrho_pickle_data = pickle.load(f)
    elif lab=='CCMR':
        pickle_str_lowr='Lowrho_polyfit_data_CCMR.pkl'
        with open(DiadFit_dir/pickle_str_lowr, 'rb') as f:
            lowrho_pickle_data = pickle.load(f)
    
        # This gets the densimeter at medium density
        pickle_str_medr='Mediumrho_polyfit_data_CCMR.pkl'
        with open(DiadFit_dir/pickle_str_medr, 'rb') as f:
            medrho_pickle_data = pickle.load(f)
        # This gets the densimeter at high density.
        pickle_str_highr='Highrho_polyfit_data_CCMR.pkl'
        with open(DiadFit_dir/pickle_str_highr, 'rb') as f:
            highrho_pickle_data = pickle.load(f)        
        
    
    else:
        raise TypeError('Lab name not recognised. enter CCMR or CMASS')

    # this allocates the model
    lowrho_model = lowrho_pickle_data['model']
    medrho_model = medrho_pickle_data['model']
    highrho_model = highrho_pickle_data['model']

    # Each of these lines get the density, and then the error on that density.

    LowD_SC = pd.Series(lowrho_model(Split), index=Split.index)
    lowD_error=calculate_Densimeter_std_err_values(corrected_split=Split, corrected_split_err=Split_err,
    pickle_str=pickle_str_lowr,  CI_dens=CI_neon, CI_split=CI_split, str_d='LowD')

    MedD_SC = pd.Series(medrho_model(Split), index=Split.index)
    medD_error=calculate_Densimeter_std_err_values(corrected_split=Split, corrected_split_err=Split_err,
    pickle_str=pickle_str_medr,  CI_dens=CI_neon, CI_split=CI_split, str_d='MedD')

    HighD_SC = pd.Series(highrho_model(Split), index=Split.index)
    highD_error=calculate_Densimeter_std_err_values(corrected_split=Split, corrected_split_err=Split_err,
    pickle_str=pickle_str_highr,   CI_dens=CI_neon, CI_split=CI_split,  str_d='HighD')





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


        
    if Ne_pickle_str is not None:
        df_merge1=pd.concat([df_combo_c, Ne_corr], axis=1).reset_index(drop=True)
    else:
        df_merge1=df

    df_merge=pd.concat([df, df_merge1], axis=1).reset_index(drop=True)

        
    

    df_merge=pd.concat([df, df_merge1], axis=1).reset_index(drop=True)

    df_merge = df_merge.rename(columns={'Preferred D': 'Density g/cm3'})
    df_merge = df_merge.rename(columns={'Preferred D_σ': 'σ Density g/cm3'})
    df_merge = df_merge.rename(columns={'Preferred D_σ_split': 'σ Density g/cm3 (from Ne+peakfit)'})
    df_merge = df_merge.rename(columns={'Preferred D_σ_dens': 'σ Density g/cm3 (from densimeter)'})
    df_merge = df_merge.rename(columns={'filename_x': 'filename'})


    #
    #

    if Ne_pickle_str is not None:
        cols_to_move = ['filename', 'Density g/cm3', 'σ Density g/cm3','σ Density g/cm3 (from Ne+peakfit)', 'σ Density g/cm3 (from densimeter)',
        'Corrected_Splitting', 'Corrected_Splitting_σ',
        'Corrected_Splitting_σ_Ne', 'Corrected_Splitting_σ_peak_fit', 'power (mW)', 'Spectral Center']
        df_merge = df_merge[cols_to_move + [
            col for col in df_merge.columns if col not in cols_to_move]]
    elif pref_Ne is not None:
        cols_to_move = ['filename', 'Density g/cm3', 'σ Density g/cm3','σ Density g/cm3 (from Ne+peakfit)', 'σ Density g/cm3 (from densimeter)',
        'Corrected_Splitting', 'Corrected_Splitting_σ',
        'Corrected_Splitting_σ_Ne', 'Corrected_Splitting_σ_peak_fit']
        df_merge = df_merge[cols_to_move + [
            col for col in df_merge.columns if col not in cols_to_move]]


    return df_merge


## functions to neatly merge secondary phases in
import os.path
from os import path
def merge_in_carb_SO2(df_combo, file1_name='Carb_Peak_fits.xlsx', file2_name='SO2_Peak_fits.xlsx', 
prefix=False, str_prefix=" ", file_ext='.txt'):
    """
    This function checks for .xlsx files with secondary phases in the path with names 
    file1_name and file2_name, if they are there it will merge them together into a dataframe.
     It then merges this with the dataframe 'df_combo' of your other fits

    Parameters
    -----------------
    df_combo: pd.DataFrame
        Dataframe of peak fitting parameters for diads
    
    file1_name: str
        Name of first excel spreadsheet of secondary phases to merge in

    file2_name: str
        Name of second excel spreadsheet of secondary phases to merge in

    prefix: bool
        If True, removes prefix followed by str_prefix from file name, 
        e.g. if your spectra are called 01 SO2_acquisition, the file name would become SO2_acquisition
        for easier merging. 
    
    file_ext: str
        Removes from file name for merging dataframes (as above).

    Returns
    ----------------
    pd.DataFrame
        Dataframe of combined peak parameters from secondary phases and diads. 

    





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
            prefix=prefix, str_prefix=str_prefix, file_ext=file_ext,
            names=Sec_Phases['filename'].reset_index(drop=True))



    else:
        df_combo_sec_phase=df_combo




    if Sec_Phases is not None:
        Sec_Phases['filename']=file_sec_phase
        df_combo_sec_phase=df_combo.merge(Sec_Phases,
        on='filename', how='outer').reset_index(drop=True)



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
def merge_fit_files(path):
    """
    This function merges the files Discarded_df.xlsx, Weak_Diads.xlsx, Medium_Diads.xlsx, Strong_Diads.xlsx
    if they exist into one combined dataframe. 

    Parameters
    -----------------
    path: str
        Path on computer where these files are stored

    Returns
    ------------
    pd.DataFrame
        pandas dataframe where 3 sets of peak fits are merged. 

    """
    import os
    import pandas as pd

    if os.path.exists(os.path.join(path, 'Discarded_df.xlsx')):
        discard = pd.read_excel(os.path.join(path, 'Discarded_df.xlsx'))
    else:
        discard = None

    if os.path.exists(os.path.join(path, 'Weak_Diads.xlsx')):
        grp1 = pd.read_excel(os.path.join(path, 'Weak_Diads.xlsx'))
    else:
        grp1 = None

    if os.path.exists(os.path.join(path, 'Medium_Diads.xlsx')):
        grp2 = pd.read_excel(os.path.join(path, 'Medium_Diads.xlsx'))
    else:
        grp2 = None

    if os.path.exists(os.path.join(path, 'Strong_Diads.xlsx')):
        grp3 = pd.read_excel(os.path.join(path, 'Strong_Diads.xlsx'))
    else:
        grp3 = None

    df2 = pd.concat([grp1, grp2, grp3], axis=0).reset_index(drop=True)

    if discard is not None:
        discard_cols=discard[discard.columns.intersection(df2.columns)]
        df2=pd.concat([df2, discard_cols]).reset_index(drop=True)
    return df2

## New UC Berkeley using 1220

def calculate_density_ucb(*, Ne_line_combo='1117_1447', df_combo=None, temp='SupCrit', 
CI_split=0.67, CI_neon=0.67,  Ne_pickle_str=None, pref_Ne=None, Ne_err=None, corrected_split=None, split_err=None):
    """ This function converts Diad Splitting into CO$_2$ density using the UC Berkeley calibration line
    developed by DeVitre and Wieser in 2023. 

    Parameters
    -------------
    Ne_line_combo: str, '1117_1447', '1220_1447', '1220_1400'
        Combination of Ne lines used for drift correction
        
    Either:

    df_combo: pandas DataFrame
        data frame of peak fitting information
        
    Or:
    corrected_split: pd.Series
        Corrected splitting  (cm-1)  
        
    Split_err: float, int
        Error on corrected splitting

    temp: str
        'SupCrit' if measurements done at 37C
        'RoomT' if measurements done at 24C - Not supported yet but could be added if needed. 

    CI_neon: float
        Default 0.67. Confidence interval to use, e.g. 0.67 returns 1 sigma uncertainties. If you use another number,
        note the column headings will still say sigma.

    CI_split: float
        Default 0.67. Confidence interval to use, e.g. 0.67 returns 1 sigma uncertainties. If you use another number,
        note the column headings will still say sigma.
        
        


    Either

    Ne_pickle_str: str
        Name of Ne correction model

    OR

    pref_Ne, Ne_err: float, int
        For quick and dirty fitting can pass a preferred value for your instrument before you have a chance to 
        regress the Ne lines (useful when first analysing new samples. )




    Returns
    --------------
    pd.DataFrame
        Prefered Density (based on different equatoins being merged), and intermediate calculations

    """
    if corrected_split is not None:
        Split=corrected_split
    if df_combo is not None:
        df_combo_c=df_combo.copy()
        time=df_combo_c['sec since midnight']

        if Ne_pickle_str is not None:

            # Calculating the upper and lower values for Ne to get that error
            Ne_corr=calculate_Ne_corr_std_err_values(pickle_str=Ne_pickle_str,
            new_x=time, CI=CI_neon)
            # Extracting preferred correction values
            pref_Ne=Ne_corr['preferred_values']
            Split_err, pk_err=propagate_error_split_neon_peakfit(Ne_corr=Ne_corr, df_fits=df_combo_c)

            df_combo_c['Corrected_Splitting_σ']=Split_err
            df_combo_c['Corrected_Splitting_σ_Ne']=(Ne_corr['upper_values']*df_combo_c['Splitting']-Ne_corr['lower_values']*df_combo_c['Splitting'])/2
            df_combo_c['Corrected_Splitting_σ_peak_fit']=pk_err

        # If using a single value for quick dirty fitting
        else:
            Split_err, pk_err=propagate_error_split_neon_peakfit(df_fits=df_combo_c, Ne_err=Ne_err, pref_Ne=pref_Ne)



            df_combo_c['Corrected_Splitting_σ']=Split_err

            df_combo_c['Corrected_Splitting_σ_Ne']=((Ne_err+pref_Ne)*df_combo_c['Splitting']-(Ne_err-pref_Ne)*df_combo_c['Splitting'])/2
            df_combo_c['Corrected_Splitting_σ_peak_fit']=pk_err

        Split=df_combo_c['Splitting']*pref_Ne

    else:
       Split_err=(split_err*Split).astype(float)



    # This is for if you just have splitting



    # This propgates the uncertainty in the splitting from peak fitting, and the Ne correction model




    if temp=='RoomT':
        raise TypeError('Sorry, no UC Berkeley calibration at 24C, please enter temp=SupCrit')
    if isinstance(Split, float) or isinstance(Split, int):
        Split=pd.Series(Split)
    # #if temp is "RoomT":
    DiadFit_dir=Path(__file__).parent

    LowD_RT=-38.34631 + 0.3732578*Split
    HighD_RT=-41.64784 + 0.4058777*Split- 0.1460339*(Split-104.653)**2

    # IF temp is 37
    if Ne_line_combo=='1220_1447':
    # This gets the densimeter at low density
        pickle_str_lowr='Lowrho_polyfit_dataUCB_1220_1447.pkl'
        with open(DiadFit_dir/pickle_str_lowr, 'rb') as f:
            lowrho_pickle_data = pickle.load(f)

        # This gets the densimeter at medium density
        pickle_str_medr='Mediumrho_polyfit_dataUCB_1220_1447.pkl'
        with open(DiadFit_dir/pickle_str_medr, 'rb') as f:
            medrho_pickle_data = pickle.load(f)
        # This gets the densimeter at high density.
        pickle_str_highr='Highrho_polyfit_dataUCB_1220_1447.pkl'
        with open(DiadFit_dir/pickle_str_highr, 'rb') as f:
            highrho_pickle_data = pickle.load(f)

    if Ne_line_combo=='1220_1400':
        pickle_str_lowr='Lowrho_polyfit_dataUCB_1220_1400.pkl'
        with open(DiadFit_dir/pickle_str_lowr, 'rb') as f:
            lowrho_pickle_data = pickle.load(f)

        # This gets the densimeter at medium density
        pickle_str_medr='Mediumrho_polyfit_dataUCB_1220_1400.pkl'
        with open(DiadFit_dir/pickle_str_medr, 'rb') as f:
            medrho_pickle_data = pickle.load(f)
        # This gets the densimeter at high density.
        pickle_str_highr='Highrho_polyfit_dataUCB_1220_1400.pkl'
        with open(DiadFit_dir/pickle_str_highr, 'rb') as f:
            highrho_pickle_data = pickle.load(f)
            
    if Ne_line_combo=='1117_1400':
        pickle_str_lowr='Lowrho_polyfit_dataUCB_1117_1400.pkl'
        with open(DiadFit_dir/pickle_str_lowr, 'rb') as f:
            lowrho_pickle_data = pickle.load(f)

        # This gets the densimeter at medium density
        pickle_str_medr='Mediumrho_polyfit_dataUCB_1117_1400.pkl'
        with open(DiadFit_dir/pickle_str_medr, 'rb') as f:
            medrho_pickle_data = pickle.load(f)
        # This gets the densimeter at high density.
        pickle_str_highr='Highrho_polyfit_dataUCB_1117_1400.pkl'
        with open(DiadFit_dir/pickle_str_highr, 'rb') as f:
            highrho_pickle_data = pickle.load(f)


    if Ne_line_combo=='1117_1447':
    # This gets the densimeter at low density
        pickle_str_lowr='Lowrho_polyfit_data.pkl'
        with open(DiadFit_dir/pickle_str_lowr, 'rb') as f:
            lowrho_pickle_data = pickle.load(f)

        # This gets the densimeter at medium density
        pickle_str_medr='Mediumrho_polyfit_data.pkl'
        with open(DiadFit_dir/pickle_str_medr, 'rb') as f:
            medrho_pickle_data = pickle.load(f)
        # This gets the densimeter at high density.
        pickle_str_highr='Highrho_polyfit_data.pkl'
        with open(DiadFit_dir/pickle_str_highr, 'rb') as f:
            highrho_pickle_data = pickle.load(f)

    # this allocates the model
    lowrho_model = lowrho_pickle_data['model']
    medrho_model = medrho_pickle_data['model']
    highrho_model = highrho_pickle_data['model']

    # Each of these lines get the density, and then the error on that density.

    LowD_SC = pd.Series(lowrho_model(Split), index=Split.index)
    lowD_error=calculate_Densimeter_std_err_values(corrected_split=Split, corrected_split_err=Split_err,
    pickle_str=pickle_str_lowr,  CI_dens=CI_neon, CI_split=CI_split, str_d='LowD')

    MedD_SC = pd.Series(medrho_model(Split), index=Split.index)
    medD_error=calculate_Densimeter_std_err_values(corrected_split=Split, corrected_split_err=Split_err,
    pickle_str=pickle_str_medr,  CI_dens=CI_neon, CI_split=CI_split, str_d='MedD')

    HighD_SC = pd.Series(highrho_model(Split), index=Split.index)
    highD_error=calculate_Densimeter_std_err_values(corrected_split=Split, corrected_split_err=Split_err,
    pickle_str=pickle_str_highr,   CI_dens=CI_neon, CI_split=CI_split,  str_d='HighD')





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

    if Ne_pickle_str is not None:
        df_merge1=pd.concat([df_combo_c, Ne_corr], axis=1).reset_index(drop=True)
        df_merge=pd.concat([df, df_merge1], axis=1).reset_index(drop=True)
    elif Ne_pickle_str is None and df_combo is not None:
        df_merge=pd.concat([df, df_combo_c], axis=1).reset_index(drop=True)
    else:
        df_merge=df

    

    df_merge = df_merge.rename(columns={'Preferred D': 'Density g/cm3'})
    df_merge = df_merge.rename(columns={'Preferred D_σ': 'σ Density g/cm3'})
    df_merge = df_merge.rename(columns={'Preferred D_σ_split': 'σ Density g/cm3 (from Ne+peakfit)'})
    df_merge = df_merge.rename(columns={'Preferred D_σ_dens': 'σ Density g/cm3 (from densimeter)'})
    df_merge = df_merge.rename(columns={'filename_x': 'filename'})


    #
    #

    if Ne_pickle_str is not None: # If its not none, have all the columns for Ne
        cols_to_move = ['filename', 'Density g/cm3', 'σ Density g/cm3','σ Density g/cm3 (from Ne+peakfit)', 'σ Density g/cm3 (from densimeter)',
        'Corrected_Splitting', 'Corrected_Splitting_σ',
        'Corrected_Splitting_σ_Ne', 'Corrected_Splitting_σ_peak_fit', 'power (mW)', 'Spectral Center']
        df_merge = df_merge[cols_to_move + [
            col for col in df_merge.columns if col not in cols_to_move]]
    elif pref_Ne is not None and df_combo is not None: #If Pref Ne, 
        cols_to_move = ['filename', 'Density g/cm3', 'σ Density g/cm3','σ Density g/cm3 (from Ne+peakfit)', 'σ Density g/cm3 (from densimeter)',
        'Corrected_Splitting', 'Corrected_Splitting_σ',
        'Corrected_Splitting_σ_Ne', 'Corrected_Splitting_σ_peak_fit']
        df_merge = df_merge[cols_to_move + [
            col for col in df_merge.columns if col not in cols_to_move]]
            
    elif df_combo is None:
        
        cols_to_move = ['Density g/cm3', 'σ Density g/cm3','σ Density g/cm3 (from Ne+peakfit)', 'σ Density g/cm3 (from densimeter)',
        'Corrected_Splitting']
        df_merge = df_merge[cols_to_move + [
            col for col in df_merge.columns if col not in cols_to_move]]







    return df_merge


