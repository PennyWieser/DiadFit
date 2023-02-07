import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize


def calc_pressure_SP1994(T_K, density_gcm3):
    T=T_K-273.15
    T0=-273.15
    MolConc=density_gcm3/44
    a1=0*(T-T0)**-4+0*(T-T0)**-2+1826134/(T-T0)+79.224365+0*(T-T0)+0*(T-T0)**2
    a2=0*(T-T0)**-4+0*(T-T0)**-2+0/(T-T0)+0.00006656066+0.0000057152798*(T-T0)+0.00000000030222363*(T-T0)**2
    a3=0*(T-T0)**-4+0*(T-T0)**-2+0/(T-T0)+0.0059957845+0.000071669631*(T-T0)+0.0000000062416103*(T-T0)**2
    a4=0*(T-T0)**-4+0*(T-T0)**-2-1.3270279/(T-T0)-0.15210731+0.00053654244*(T-T0)-0.000000071115142*(T-T0)**2
    a5=0*(T-T0)**-4+0*(T-T0)**-2+0.12456776/(T-T0)+4.9045367+0.009822056*(T-T0)+0.0000055962121*(T-T0)**2
    a6=0*(T-T0)**-4+0*(T-T0)**-2+0/(T-T0)+0.75522299+0*(T-T0)+0*(T-T0)**2
    a7=-393446440000*(T-T0)**-4+90918237*(T-T0)**-2+427767.16/(T-T0)-22.347856+0*(T-T0)+0*(T-T0)**2
    a8=0*(T-T0)**-4+0*(T-T0)**-2+402.82608/(T-T0)+119.71627+0*(T-T0)+0*(T-T0)**2
    a9=0*(T-T0)**-4+22995650*(T-T0)**-2-78971.817/(T-T0)-63.376456+0*(T-T0)+0*(T-T0)**2
    a10=0*(T-T0)**-4+0*(T-T0)**-2+95029.765/(T-T0)+18.038071+0*(T-T0)+0*(T-T0)**2



    P_MPa=(0.1*83.14472*(T-T0)*(MolConc+a1*MolConc**2-MolConc**2*((a3
    +2*a4*MolConc+3*a5*MolConc**2+4*a6*MolConc**3)/(a2+
    a3*MolConc+a4*MolConc**2+a5*MolConc**3+a6*MolConc**4)**2)
    +a7*MolConc**2*np.exp(-a8*MolConc)
    +a9*MolConc**2*np.exp(-a10*MolConc)))
    return P_MPa



def calculate_Pressure_Sterner_Pitzer_1994_test(*, T_K=1400, calc_type='T_h',
T_h=None, phase=None, density_gcm3=None, return_array=False):
    """ This function calculates Pressure using Sterner and Pitzer (1994) using either a homogenization temp,
    or a CO2 density. You must also input a temperature of your system (e.g. 1400 K for a basalt)

    Parameters
    -------

    T_K: int, float, pd.Series
        Temperature in Kelvin to find P at (e.g. temp fluid was trapped at)

    calc_type: str
        'T_h' if you want to use homogenization temp
        'density_g_cm3' if you want to use an entered density

    Either:

        T_h: int, float, pd.Series (optional)
            homogenization temp during microthermometry
        phase: str
            'Gas' or 'Liquid', the phase your inclusion homogenizes too
    Or:

        density_gcm3: int, float, pd.Series
            Density of your inclusion in g/cm3, e.g. from Raman spectroscopy

    return_array: bool
        if True, returns a pd.array not a df.



    Returns
    -------
    Pandas.DataFrame
        Has columns for T_K, T_h, Liq_density, Gas_density, P_MPa. Non relevant variables filled with NaN

    """

    T=T_K-273.15
    T0=-273.15
    if calc_type!='density_gcm3' and calc_type!='T_h':
        raise TypeError('you must specify the calc_type as T_h or density_gcm3')
    if calc_type=='density_gcm3':
        density_to_use=density_gcm3/44
        Liq_density=np.nan
        gas_density=np.nan



    if calc_type=='T_h':


        T_K_hom=T_h+273.15
        TempTerm=1-T_K_hom/304.1282
        Liq_density=(np.exp(1.9245108*TempTerm**0.34-0.62385555*TempTerm**0.5-0.32731127*TempTerm**1.6666667+0.39245142*TempTerm**1.8333333)*0.4676)
        gas_density=(np.exp(-1.7074879*TempTerm**0.34-0.8227467*TempTerm**0.5-4.6008549*TempTerm**1-10.111178*TempTerm**2.333333-29.742252*TempTerm**4.6666667)*0.4676)
    # Pressure stuff


        if phase=='Liq':
            density_to_use=Liq_density
        if  phase=='Gas':
            density_to_use=gas_density


    P_MPa=calc_pressure_SP1994(T_K=T_K, density_gcm3=density_to_use)

    if return_array is True:
        return P_MPa


    else:

        if isinstance(P_MPa, float) or isinstance(P_MPa, int):
            df=pd.DataFrame(data={'T_h': T_h,
                                'T_K': T_K,
                                'Liq_density_gcm3': Liq_density,
                                'Gas_density_gcm3': gas_density,
                                'P_MPa': P_MPa}, index=[0])
        else:


            df=pd.DataFrame(data={'T_h': T_h,
                                'T_K': T_K,
                                'Liq_density_gcm3': Liq_density,
                                'Gas_density_gcm3': gas_density,
                                'P_MPa': P_MPa})



        return df


## What about inverting for pressure?
import scipy
from scipy.optimize import minimize
# What we are trying to do, is run at various densities, until pressure matches input pressure

def objective_function(density_gcm3, T_K, target_pressure):
    # The objective function that you want to minimize
    calculated_pressure = calc_pressure_SP1994(density_gcm3=density_gcm3, T_K=T_K)
    objective = np.abs(calculated_pressure - target_pressure)
    return objective

def calculate_Density_Sterner_Pitzer_1994(T_K, target_pressure):
    # Solve for density using Scipy's minimize function
    initial_guess = 1 # Provide an initial guess for the density
    result = minimize(objective_function, initial_guess, bounds=((0, 2), ), args=(T_K, target_pressure))
    return result.x

def calculate_Density_Sterner_Pitzer_1994_loop(T_K, target_pressure):
    Density=np.empty(len(target_pressure))
    for i in range(0, len(target_pressure)):
        Density[i]=calculate_Density_Sterner_Pitzer_1994(T_K=T_K, target_pressure=target_pressure[i])
    return Density
