import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inspect
from scipy.interpolate import CubicSpline
from pathlib import Path
from pickle import load
import pickle


DiadFit_dir=Path(__file__).parent


def ind_density_homog_T_CO2(T_C, print=False):
    """ Calculates CO2 density for a specified homogenization temperature in Celcius
    using the Span and Wanger (1996) equation of state.
    Only works for one sample, use loop_density_homog_T
    if you have more than one temperature

    Parameters
    --------------
    T_C: int or float
        Temperature in celcius
    Returns
    -------------
    pd.DataFrame:
        colums for Liq_gcm3 and gas_gcm3

    """

    with open(DiadFit_dir/'Phase_Boundary.pck', 'rb') as f:
            my_loaded_model=load(f)

    P_MPa=my_loaded_model(T_C)
    # Need to check if its greater than PCrit
    PCrit=7.3773
    TCrit=30.9782

    P_Pa=P_MPa*10**6
    T_K=T_C+273.15

    try:
        import CoolProp.CoolProp as cp
    except ImportError:
        raise RuntimeError('You havent installed CoolProp, which is required to convert FI densities to pressures. If you have python through conda, run conda install -c conda-forge coolprop in your command line')


    Phase=cp.PhaseSI('P', P_Pa, 'T', T_K,'CO2')
    if print is True:

        print('T='+str(T_C))
        print('P='+str(P_MPa))
        print('Phase coolprop says='+str(Phase))

    Density_kgm3_liq=np.nan
    Density_kgm3_gas=np.nan
    Density_kgm3_supcrit_liq=np.nan
    Density_kgm3_supcrit_gas=np.nan
    Density_kgm3_supcrit=np.nan
    if P_MPa<PCrit and T_C<TCrit:
        Density_kgm3_liq=cp.PropsSI('D', 'P|liquid', P_Pa, 'T', T_K, 'CO2')
        Density_kgm3_gas=cp.PropsSI('D', 'P|gas', P_Pa, 'T', T_K, 'CO2')


    # if P_MPa>PCrit and T_C<TCrit:
    #     Density_kgm3_supcrit_liq=cp.PropsSI('D', 'P|supercritical_liquid', P_Pa, 'T', T_K, 'CO2')
    # if P_MPa<PCrit and T_C>=TCrit:
    #     Density_kgm3_supcrit_gas=cp.PropsSI('D', 'P|supercritical_gas', P_Pa, 'T', T_K, 'CO2')
    # if P_MPa>PCrit and T_C>TCrit:
    #     Density_kgm3_supercrit=cp.PropsSI('D', 'P|supercritical', P_Pa, 'T', T_K, 'CO2')

    df=pd.DataFrame(data={'Liq_gcm3': Density_kgm3_liq/1000,
    'Gas_gcm3': Density_kgm3_gas/1000,
    'T_C': T_C
}, index=[0]
    )


    return df


def calculate_CO2_density_homog_T(T_C, Sample_ID=None):
    """ Calculates CO2 density for a specified homogenization temperature in Celcius
    using the Span and Wanger (1996) equation of state.

    Parameters
    --------------
    T_C: int, float, pd.series
        Temperature in celcius
    Sample_ID: int, pd.series (optional)
        SampleID
    Returns
    -------------
    pd.DataFrame:
        colums for Liq_gcm3 and gas_gcm3

    """
    if isinstance(T_C, float) or isinstance(T_C, int):
        Density2=ind_density_homog_T_CO2(T_C)
    else:
        Density=pd.DataFrame([])
        if isinstance(T_C, pd.Series):
            T_C_np=np.array(T_C)
        for i in range(0, len(T_C)):
            data=ind_density_homog_T_CO2(T_C[i])
            Density = pd.concat([Density, data], axis=0)

        Density2=Density.reset_index(drop=True)
    if Sample_ID is not None:
        if isinstance(Sample_ID, str):
            Density2['Sample_ID']=Sample_ID

        elif len(Sample_ID)==len(T_C):
            Density2['Sample_ID']=Sample_ID
        else:
            w.warn('SampleID not same length as temp, havent added a column as a result')
    return Density2




def propagate_microthermometry_uncertainty_1sam(*, T_C, sample_i=0, error_T_C=0.3, N_dup=1000,
        error_dist_T_C='uniform', error_type_T_C='Abs', len_loop=1):

    if len_loop==1:
        df_c=pd.DataFrame(data={'T_C': T_C}, index=[0])
    else:
        df_c=pd.DataFrame(data={'T_C': T_C})


    # Temperature error distribution
    if error_type_T_C=='Abs':
        error_T_C=error_T_C
    if error_type_T_C =='Perc':
        error_T_C=df_c['T_C'].iloc[sample_i]*error_T_C/100
    if error_dist_T_C=='normal':
        Noise_to_add_T_C = np.random.normal(0, error_T_C, N_dup)
    if error_dist_T_C=='uniform':
        Noise_to_add_T_C = np.random.uniform(- error_T_C, +
                                                      error_T_C, N_dup)

    T_C_with_noise=Noise_to_add_T_C+df_c['T_C'].iloc[sample_i]

    return T_C_with_noise

def propagate_microthermometry_uncertainty(T_C, Sample_ID=None, sample_i=0, error_T_C=0.3, N_dup=1000,
        error_dist_T_C='uniform', error_type_T_C='Abs', len_loop=1):

    # Set up empty things to fill up.

    if type(T_C) is pd.Series:
        len_loop=len(T_C)
    else:
        len_loop=1
    print(len_loop)

    All_outputs=pd.DataFrame([])
    Std_density_gas=np.empty(len_loop)
    Std_density_liq=np.empty(len_loop)
    Mean_density_gas=np.empty(len_loop)
    Mean_density_liq=np.empty(len_loop)
    Sample=np.empty(len_loop,  dtype=np.dtype('U100') )

    for i in range(0, len_loop):

        # If user has entered a pandas series for error, takes right one for each loop
        if type(error_T_C) is pd.Series:
            error_T_C=error_T_C.iloc[i]
        else:
            error_T_C=error_T_C

        if type(T_C) is pd.Series:
            T_C_i=T_C.iloc[i]
        else:
            T_C_i=T_C

        # Check of
        if Sample_ID is None:
            Sample[i]=i

        elif isinstance(Sample_ID, str):
            Sample[i]=Sample_ID
        else:
            Sample[i]=Sample_ID.iloc[i]

        Temp_MC=propagate_microthermometry_uncertainty_1sam(T_C=T_C_i,
        sample_i=0, error_T_C=error_T_C, N_dup=N_dup,
        error_dist_T_C=error_dist_T_C, error_type_T_C=error_type_T_C, len_loop=1)

        Sample2=Sample[i]
        MC_T=calculate_CO2_density_homog_T(T_C=Temp_MC, Sample_ID=Sample2)




        # MC for each FI
        All_outputs=pd.concat([All_outputs, MC_T], axis=0)

        # get av and mean
        Std_density_gas[i]=np.nanstd(MC_T['Gas_gcm3'])
        Std_density_liq[i]=np.nanstd(MC_T['Liq_gcm3'])
        Mean_density_gas[i]=np.nanmean(MC_T['Gas_gcm3'])
        Mean_density_liq[i]=np.nanmean(MC_T['Liq_gcm3'])




    Av_outputs=pd.DataFrame(data={'Sample_ID': Sample,
                                      'Mean_density_Gas_gcm3': Mean_density_gas,
                                      'Std_density_Gas_gcm3': Std_density_gas,
                                       'Mean_density_Liq_gcm3': Mean_density_liq,
                                      'Std_density_Liq_gcm3': Std_density_liq,
                                      'error_T_C': error_T_C})

    return Av_outputs, All_outputs


