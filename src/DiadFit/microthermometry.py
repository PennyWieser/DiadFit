import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inspect
from scipy.interpolate import CubicSpline
from pathlib import Path
from pickle import load
import pickle


DiadFit_dir=Path(__file__).parent


def calculate_density_homog_T_CO2(T_C, print=False):
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
    Density_gcm3: int or float
        Density of CO2 liquid in g/cm3.

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

    if P_MPa>PCrit and T_C<TCrit:
        Density_kgm3_supcrit_liq=cp.PropsSI('D', 'P|supercritical_liquid', P_Pa, 'T', T_K, 'CO2')
    if P_MPa<PCrit and T_C>=TCrit:
        Density_kgm3_supcrit_gas=cp.PropsSI('D', 'P|supercritical_gas', P_Pa, 'T', T_K, 'CO2')
    if P_MPa>PCrit and T_C>TCrit:
        Density_kgm3_supercrit=cp.PropsSI('D', 'P|supercritical', P_Pa, 'T', T_K, 'CO2')

    df=pd.DataFrame(data={'Liq_gcm3': Density_kgm3_liq/1000,
    'Gas_gcm3': Density_kgm3_gas/1000,
    'Supercrit_Liq_gcm3': Density_kgm3_supcrit_liq/1000,
    'Supercrit_Gas_gcm3': Density_kgm3_supcrit_gas/1000,
    'Supercrit_Fluid_gcm3': Density_kgm3_supcrit/1000}, index=[0]
    )


    return df


def loop_density_homog_T_CO2(T_C):
    """ Calculates CO2 density for a specified homogenization temperature in Celcius
    using the Span and Wanger (1996) equation of state.

    """
    Density=pd.DataFrame([])
    if isinstance(T_C, pd.Series):
        T_C_np=np.array(T_C)
    for i in range(0, len(T_C)):
        data=calculate_density_homog_T_CO2(T_C[i])
        Density = pd.concat([Density, data], axis=0)

    Density.reset_index(drop=True)
    return Density



