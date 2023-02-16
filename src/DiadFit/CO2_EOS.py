import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inspect
from scipy.interpolate import CubicSpline
import scipy
from scipy.optimize import minimize

from pathlib import Path
from pickle import load
import pickle


DiadFit_dir=Path(__file__).parent



## Calculating density for a given homogenization temp - Only available with Span and Wanger, but have equations

def calculate_CO2_density_homog_T(T_h_C, Sample_ID=None, homog_to=None, EOS='SW96'):
    """ Calculates CO2 density for a specified homogenization temperature in Celcius
    using the Span and Wanger (1996) equation of state.

    Parameters
    --------------
    T_h_C: int, float, pd.series
        Temperature in celcius
    Sample_ID: int, pd.series (optional)
        SampleID
    homog_to: str ('L', 'V'), pd.series with strings. Optional
        If specified, returns an additional column 'Bulk Density' to choose between the liquid and gas.
    EOS: str, 'SW96'
        Here for consistency with other functions, only supported for SW96
    Returns
    -------------
    pd.DataFrame:
        colums for Liq_gcm3 and gas_gcm3, bulk density if homog_to specified

    """
    if EOS!='SW96':
        raise TypeError('At the moment, only Span and Wanger (SW96) EOS can be used to convert T_h_C into density')

    T_K_hom=T_h_C+273.15
    TempTerm=1-T_K_hom/304.1282
    # This is equation 3.14 from Span and Wanger (1996)
    Liq_density=(np.exp(1.9245108*TempTerm**0.34-0.62385555*TempTerm**0.5-0.32731127*TempTerm**1.6666667+0.39245142*TempTerm**1.8333333)*0.4676)
    # This is equation 3.15 from Span and Wanger (1996)
    gas_density=(np.exp(-1.7074879*TempTerm**0.34-0.8227467*TempTerm**0.5-4.6008549*TempTerm**1-10.111178*TempTerm**2.333333-29.742252*TempTerm**4.6666667)*0.4676)

    if isinstance(Liq_density, float):
        df=pd.DataFrame(data={'Bulk_gcm3': np.nan,
    'Liq_gcm3': Liq_density,
    'Gas_gcm3': gas_density,
    'T_h_C': T_h_C,
    'homog_to': 'no input'}, index=[0])

    else:


        df=pd.DataFrame(data={'Bulk_gcm3': np.nan,
    'Liq_gcm3': Liq_density,
    'Gas_gcm3': gas_density,
    'T_h_C': T_h_C,
    'homog_to': 'no input'})

    return df


    #     raise TypeError('Make sure homog_to is L or V, no other options are supported')





## There is another way of doing this, we have parameterized the phase boundary from the NIST webbok. This gives very slghtly different answers, one above is favoured
def ind_density_homog_T_h_CO2_SW96_loaded_phase_boundary_1sam(T_h_C, print=False, homog_to=None):
    """ Calculates CO2 density for a specified homogenization temperature in Celcius
    using the Span and Wanger (1996) equation of state.
    Only works for one sample, use loop_density_homog_T
    if you have more than one temperature

    Parameters
    --------------
    T_h_C: int or float
        Temperature in celcius

    homog_to: str ('L', 'V'), pd.series with strings. Optional
        If specified, returns an additional column 'Bulk Density' to choose between the liquid and gas.
    Returns
    -------------
    pd.DataFrame:
        colums for Liq_gcm3 and gas_gcm3

    """

    with open(DiadFit_dir/'Phase_Boundary.pck', 'rb') as f:
            my_loaded_model=load(f)

    P_MPa=my_loaded_model(T_h_C)
    # Need to check if its greater than PCrit
    PCrit=7.3773
    TCrit=30.9782

    P_Pa=P_MPa*10**6
    T_K=T_h_C+273.15

    try:
        import CoolProp.CoolProp as cp
    except ImportError:
        raise RuntimeError('You havent installed CoolProp, which is required to convert FI densities to pressures. If you have python through conda, run conda install -c conda-forge coolprop in your command line')


    Phase=cp.PhaseSI('P', P_Pa, 'T', T_K,'CO2')
    if print is True:

        print('T='+str(T_h_C))
        print('P='+str(P_MPa))
        print('Phase coolprop says='+str(Phase))

    Density_kgm3_liq=np.nan
    Density_kgm3_gas=np.nan
    Density_kgm3_supcrit_liq=np.nan
    Density_kgm3_supcrit_gas=np.nan
    Density_kgm3_supcrit=np.nan
    if P_MPa<PCrit and T_h_C<TCrit:
        Density_kgm3_liq=cp.PropsSI('D', 'P|liquid', P_Pa, 'T', T_K, 'CO2')
        Density_kgm3_gas=cp.PropsSI('D', 'P|gas', P_Pa, 'T', T_K, 'CO2')


    # if P_MPa>PCrit and T_h_C<TCrit:
    #     Density_kgm3_supcrit_liq=cp.PropsSI('D', 'P|supercritical_liquid', P_Pa, 'T', T_K, 'CO2')
    # if P_MPa<PCrit and T_h_C>=TCrit:
    #     Density_kgm3_supcrit_gas=cp.PropsSI('D', 'P|supercritical_gas', P_Pa, 'T', T_K, 'CO2')
    # if P_MPa>PCrit and T_h_C>TCrit:
    #     Density_kgm3_supercrit=cp.PropsSI('D', 'P|supercritical', P_Pa, 'T', T_K, 'CO2')

    df=pd.DataFrame(data={'Bulk_gcm3': np.nan,
    'Liq_gcm3': Density_kgm3_liq/1000,
    'Gas_gcm3': Density_kgm3_gas/1000,
    'T_h_C': T_h_C,
    'homog_to': 'no input'
}, index=[0]
    )
    if homog_to=='L':
        df['Bulk_gcm3']=Density_kgm3_liq/1000
        df['homog_to']='L'
    elif homog_to=='V':
        df['Bulk_gcm3']=Density_kgm3_gas/1000
        df['homog_to']='V'
    elif homog_to is None:
        a=1
    else:
        raise TypeError('Make sure homog_to is L or V, no other options are supported')


    return df


# This calls the function above and stitches it for all samples
def calculate_CO2_density_homog_T_SW96_NIST(T_h_C, Sample_ID=None, homog_to=None):
    """ Calculates CO2 density for a specified homogenization temperature in Celcius
    using the Span and Wanger (1996) equation of state.
    Parameterizes based on NIST webook. Very similar to above using the root equations (rounding error issues in webbook?)

    Parameters
    --------------
    T_h_C: int, float, pd.series
        Homogenization temperature in celcius
    Sample_ID: int, pd.series (optional)
        SampleID
    homog_to: str ('L', 'V'), pd.series with strings. Optional
        If specified, returns an additional column 'Bulk Density' to choose between the liquid and gas.
    Returns
    -------------
    pd.DataFrame:
        colums for Liq_gcm3 and gas_gcm3, bulk density if homog_to specified

    """
    if isinstance(T_h_C, float) or isinstance(T_h_C, int):
        if homog_to is None:
            Density2=ind_density_homog_T_h_CO2_SW96_loaded_phase_boundary_1sam(T_h_C)

        else:
            Density2=ind_density_homog_T_h_CO2_SW96_loaded_phase_boundary_1sam(T_h_C, homog_to=homog_to)

    else:
        Density=pd.DataFrame([])
        if isinstance(T_h_C, pd.Series):
            T_h_C_np=np.array(T_h_C)
        for i in range(0, len(T_h_C)):
            if homog_to is None:
                data=ind_density_homog_T_h_CO2_SW96_loaded_phase_boundary_1sam(T_h_C[i])
            else:
                data=ind_density_homog_T_h_CO2_SW96_loaded_phase_boundary_1sam(T_h_C[i], homog_to=homog_to[i])
            Density = pd.concat([Density, data], axis=0)

        Density2=Density.reset_index(drop=True)
    if Sample_ID is not None:
        if isinstance(Sample_ID, str):
            Density2['Sample_ID']=Sample_ID

        elif len(Sample_ID)==len(T_h_C):
            Density2['Sample_ID']=Sample_ID
        else:
            w.warn('SampleID not same length as temp, havent added a column as a result')


    return Density2

## Calculating density for a given pressure and temperature

def calculate_rho_gcm3_for_P_T(P_kbar, T_K, EOS='SW96'):

    if EOS=='SW96':

        if isinstance(P_kbar, pd.Series):
            P_kbar=np.array(P_kbar)
        if isinstance(T_K, pd.Series):
            T_K=np.array(T_K)

        try:
            import CoolProp.CoolProp as cp
        except ImportError:
            raise RuntimeError('You havent installed CoolProp, which is required to convert FI densities to pressures. If you have python through conda, run conda install -c conda-forge coolprop in your command line')

        P_Pa=P_kbar*10**8
        Density_gcm3=cp.PropsSI('D', 'P', P_Pa, 'T', T_K, 'CO2')/1000



    if EOS=='SP94':

        Density_gcm3=calculate_SP19942(T_K=T_K, target_pressure_MPa=P_kbar*100)

    return pd.Series(Density_gcm3)








# using CoolProp/SW96
def calculate_rho_gcm3_for_P_T_SW96(P_kbar, T_K):
    if isinstance(P_kbar, pd.Series):
        P_kbar=np.array(P_kbar)
    if isinstance(T_K, pd.Series):
        T_K=np.array(T_K)

    P_Pa=P_kbar*10**8
    Density_gcm3=cp.PropsSI('D', 'P', P_Pa, 'T', T_K, 'CO2')/1000

    return pd.Series(Density_gcm3)

# Generic fun

## Calculate density for a given pressure and temperature using Sterner and Pitzer
#def calculate_rho_gcm3_for_P_T_SP94(P_kbar, T_K):


## Generic function for converting rho and T into Pressure

def calculate_P_for_rho_T(density_gcm3, T_K, EOS='SW96'):

    if EOS=='SW96':
        df=calculate_P_for_rho_T_SW96(density_gcm3=density_gcm3, T_K=T_K)
    elif EOS=='SP94':
        df=calculate_P_for_rho_T_SP94(density_gcm3=density_gcm3, T_K=T_K)
    else:
        raise TypeError('Please choose either SP94 or SW96 as an EOS')
    return df

# Calculating P for a given density and Temperature using Coolprop
def calculate_P_for_rho_T_SW96(density_gcm3, T_K):
    if isinstance(density_gcm3, pd.Series):
        density_gcm3=np.array(density_gcm3,)
    if isinstance(T_K, pd.Series):
        T_K=np.array(T_K)
    Density_kgm3=density_gcm3*1000

    try:
        import CoolProp.CoolProp as cp
    except ImportError:
        raise RuntimeError('You havent installed CoolProp, which is required to convert FI densities to pressures. If you have python through conda, run conda install -c conda-forge coolprop in your command line')

    try:
        import CoolProp.CoolProp as cp
    except ImportError:
        raise RuntimeError('You havent installed CoolProp, which is required to convert FI densities to pressures. If you have python through conda, run conda install -c conda-forge coolprop in your command line')

    P_kbar=cp.PropsSI('P', 'D', Density_kgm3, 'T', T_K, 'CO2')/10**8
    df=pd.DataFrame(data={'P_kbar': P_kbar,
                            'P_MPa': P_kbar*100,
                            'T_K': T_K,
                            'density_gcm3': density_gcm3
                            })
    return df

# Calculating P for a given density and Temp, Sterner and Pitzer
def calculate_P_for_rho_T_SP94(T_K, density_gcm3, scalar_return=False):
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
    if scalar_return is True:
        return P_MPa
    else:

        df=pd.DataFrame(data={'P_kbar': P_MPa/100,
                            'P_MPa': P_MPa,
                            'T_K': T_K,
                            'density_gcm3': density_gcm3

                            })
        return df



## Overall function convertiong density or T_h_C into a pressure


def calculate_P_SP1994(*, T_K=1400,
T_h_C=None, phase=None, density_gcm3=None, return_array=False):
    """ This function calculates Pressure using Sterner and Pitzer (1994) using either a homogenization temp,
    or a CO2 density. You must also input a temperature of your system (e.g. 1400 K for a basalt)

    Parameters
    -------

    T_K: int, float, pd.Series
        Temperature in Kelvin to find P at (e.g. temp fluid was trapped at)



    Either:

        T_h_C: int, float, pd.Series (optional)
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
        Has columns for T_K, T_h_C, Liq_density, Gas_density, P_MPa. Non relevant variables filled with NaN

    """

    T=T_K-273.15
    T0=-273.15

    if density_gcm3 is not None and T_h_C is not None:
        raise TypeError('Enter either density_gcm3 or T_h_C, not both')
    if density_gcm3 is not None:
        density_to_use=density_gcm3/44
        Liq_density=np.nan
        gas_density=np.nan



    if T_h_C is not None:


        T_K_hom=T_h_C+273.15
        TempTerm=1-T_K_hom/304.1282
        Liq_density=(np.exp(1.9245108*TempTerm**0.34-0.62385555*TempTerm**0.5-0.32731127*TempTerm**1.6666667+0.39245142*TempTerm**1.8333333)*0.4676)
        gas_density=(np.exp(-1.7074879*TempTerm**0.34-0.8227467*TempTerm**0.5-4.6008549*TempTerm**1-10.111178*TempTerm**2.333333-29.742252*TempTerm**4.6666667)*0.4676)
    # Pressure stuff


        if phase=='Liq':
            density_to_use=Liq_density
        if  phase=='Gas':
            density_to_use=gas_density


    P_MPa=calculate_P_for_rho_T_SP94(T_K=T_K, density_gcm3=density_to_use)

    if return_array is True:
        return P_MPa


    else:

        if isinstance(P_MPa, float) or isinstance(P_MPa, int):
            df=pd.DataFrame(data={'T_h_C': T_h_C,
                                'T_K': T_K,
                                'Liq_density_gcm3': Liq_density,
                                'Gas_density_gcm3': gas_density,
                                'P_MPa': P_MPa}, index=[0])
        else:


            df=pd.DataFrame(data={'T_h_C': T_h_C,
                                'T_K': T_K,
                                'Liq_density_gcm3': Liq_density,
                                'Gas_density_gcm3': gas_density,
                                'P_MPa': P_MPa})



        return df


## Sterner and Pitzer inverting for Density
import scipy
from scipy.optimize import minimize
# What we are trying to do, is run at various densities, until pressure matches input pressure

def objective_function(density_gcm3, T_K, target_pressure_MPa):
    # The objective function that you want to minimize
    calculated_pressure = calculate_P_for_rho_T_SP94(density_gcm3=density_gcm3, T_K=T_K, scalar_return=True)
    objective = np.abs(calculated_pressure - target_pressure_MPa)
    return objective

def calculate_Density_Sterner_Pitzer_1994(T_K, target_pressure_MPa):
    # Solve for density using Scipy's minimize function
    initial_guess = 1 # Provide an initial guess for the density
    result = minimize(objective_function, initial_guess, bounds=((0, 2), ), args=(T_K, target_pressure_MPa))
    return result.x

def calculate_SP19942(T_K, target_pressure_MPa):

    if isinstance(target_pressure_MPa, float) or isinstance(target_pressure_MPa, int):
        print('yes')
        target_p=np.array(target_pressure_MPa)
        Density=calculate_Density_Sterner_Pitzer_1994(T_K=T_K, target_pressure_MPa=target_p)
    else:
        Density=np.empty(len(target_pressure_MPa))
        for i in range(0, len(target_pressure_MPa)):
            Density[i]=calculate_Density_Sterner_Pitzer_1994(T_K=T_K, target_pressure_MPa=target_pressure_MPa[i])
    return Density


