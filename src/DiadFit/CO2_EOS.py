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
import math
DiadFit_dir=Path(__file__).parent

## Visualizing coexisting phases




## Calculating density for a given homogenization temp - Only available with Span and Wanger, but have equations

def calculate_CO2_density_homog_T(T_h_C, EOS='SW96', Sample_ID=None, homog_to=None, set_to_critical=False):
    """ Calculates CO2 density for a specified homogenization temperature in Celcius
    using eq 3.14 and 3.15 from the Span and Wanger (1996) equation of state.

    Parameters
    --------------
    T_h_C: int, float, pd.series
        Temperature in celcius

    EOS: str, 'SW96'
        Here for consistency with other functions, only supported for SW96

    Optional

    Sample_ID: int, pd.series
        Sample ID, will append to the final dataframe

    homog_to: str ('L', 'V'), pd.series with strings. Optional
        If specified, returns an additional column 'Bulk Density' to choose between the liquid and gas.

    set_to_critical: bool
        Default False. If true, if you enter T_h_C which exceeds 30.9782 (the critical point of CO2) it replaces your entered Temp with that temp.

    Returns
    -------------
    pd.DataFrame:
        colums for Liq_gcm3 and gas_gcm3, bulk density if homog_to specified

    """
    if isinstance(homog_to, str):
        if homog_to=='L' or homog_to=='V':
            a=1
        else:
            raise TypeError('unsupported input for homog_to, has to be L or V')


    # IF its a float or integer, just tell people outright
    if isinstance(T_h_C, float) or isinstance(T_h_C, int):
        if T_h_C>=30.9782: # 29.878:
        #print('Sorry, algorithm cant converge for Ts above 29.878')
            raise TypeError('Sorry, algorithm cant converge for T_h_C above 30.9782')

    # If its a panda series, set critical is false, raise a type error
    if isinstance(T_h_C, pd.Series) or isinstance(T_h_C, np.ndarray):

        if any(T_h_C>=30.9782) and set_to_critical is False:
            raise TypeError('Sorry, algorithm cant converge for Ts above 30.9782. You can put set_to_critical=True and this T_ will be replacd with 30.9782')
        elif any(T_h_C>=30.9782) and set_to_critical is True:
            print('found some with too high temps, are setting to 30.9782C - the max homog T ')
            if isinstance(T_h_C, pd.Series):
                T_h_C = np.where(T_h_C > 30.9782, 30.9782, T_h_C)
            elif isinstance(T_h_C, np.ndarray):

                T_h_C[T_h_C > 30.9782] = 30.9782





    if EOS!='SW96':
        raise TypeError('At the moment, only Span and Wanger (SW96) EOS can be used to convert T_h_C into density')

    T_K_hom=T_h_C+273.15
    TempTerm=1-T_K_hom/304.1282
    # This is equation 3.14 from Span and Wanger (1996)


    Liq_density=(np.exp(1.9245108*TempTerm**0.34-0.62385555*TempTerm**0.5-0.32731127*TempTerm**1.6666667+0.39245142*TempTerm**1.8333333)*0.4676)

    # This is equation 3.15 from Span and Wanger (1996)
    gas_density=(np.exp(-1.7074879*TempTerm**0.34-0.8227467*TempTerm**0.5-4.6008549*TempTerm**1-10.111178*TempTerm**2.333333-29.742252*TempTerm**4.6666667)*0.4676)



    if isinstance(Liq_density, float):
        if homog_to is None:
            df=pd.DataFrame(data={'Bulk_gcm3': np.nan,
        'Liq_gcm3': Liq_density,
        'Gas_gcm3': gas_density,
        'T_h_C': T_h_C,
        'homog_to': homog_to}, index=[0])
        else:
            if homog_to=='L':
                df=pd.DataFrame(data={'Bulk_gcm3': Liq_density,
            'Liq_gcm3': Liq_density,
            'Gas_gcm3': gas_density,
            'T_h_C': T_h_C,
            'homog_to': homog_to}, index=[0])

            elif homog_to=='V':
                df=pd.DataFrame(data={'Bulk_gcm3': gas_density,
            'Liq_gcm3': Liq_density,
            'Gas_gcm3': gas_density,
            'T_h_C': T_h_C,
            'homog_to': homog_to}, index=[0])

            else:
                df=pd.DataFrame(data={'Bulk_gcm3': np.nan,
            'Liq_gcm3': Liq_density,
            'Gas_gcm3': gas_density,
            'T_h_C': T_h_C,
            'homog_to': homog_to}, index=[0])


    else:
        # If they havent specified
        if homog_to is None:
            df=pd.DataFrame(data={'Bulk_gcm3': np.nan,
    'Liq_gcm3': Liq_density,
    'Gas_gcm3': gas_density,
    'T_h_C': T_h_C,
    'homog_to': homog_to})

        # If its a string, e.g. same for all samples
        elif isinstance(homog_to, str):
            if homog_to=='L':
                df=pd.DataFrame(data={'Bulk_gcm3': Liq_density,
        'Liq_gcm3': Liq_density,
        'Gas_gcm3': gas_density,
        'T_h_C': T_h_C,
        'homog_to': homog_to})
            if homog_to=='V':
                df=pd.DataFrame(data={'Bulk_gcm3': gas_density,
        'Liq_gcm3': Liq_density,
        'Gas_gcm3': gas_density,
        'T_h_C': T_h_C,
        'homog_to': homog_to})



        # If its a panda series
        else:

            df=pd.DataFrame(data={'Bulk_gcm3': np.nan,
        'Liq_gcm3': Liq_density,
        'Gas_gcm3': gas_density,
        'T_h_C': T_h_C,
        'homog_to': homog_to})


            homog_L=homog_to=='L'
            df.loc[homog_L, 'Bulk_gcm3']=Liq_density
            homog_L=homog_to=='V'
            df.loc[homog_L, 'Bulk_gcm3']=gas_density

    if Sample_ID is not None:
        df['Sample_ID']=Sample_ID







    return df





## There is another way of doing this, we have parameterized the phase boundary from the NIST webbok. This gives very slghtly different answers, we favour the method above, this is here incase!

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

    Print: bool
        Prints the phase

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

## Calculating density for a given pressure and temperature - have a generic function, that calls the individual EOS depending on which one you select

def calculate_rho_for_P_T(P_kbar, T_K, EOS='SW96'):
    """ This function calculates CO2 density in g/cm3 for a known Pressure (in kbar), a known T (in K), and a specified EOS

    Parameters
    ---------------------
    P_kbar: int, float, pd.Series, np.array
        Pressure in kbar

    T_K: int, float, pd.Series, np.array
        Temperature in Kelvin

    EOS: str
        'SW96' for Span and Wanger (1996), 'SP94' for Sterner and Pitzer (1994)


    Returns
    --------------------
    pd.Series
        CO2 density in g/cm3

    """

    if EOS=='SW96':
        CO2_dens_gcm3=calculate_rho_for_P_T_SW96(P_kbar, T_K)


    if EOS=='SP94':

        CO2_dens_gcm3=calculate_rho_for_P_T_SP94(T_K=T_K, P_kbar=P_kbar)




    return pd.Series(CO2_dens_gcm3)






# Function for Span and Wanger (1996)

def calculate_rho_for_P_T_SW96(P_kbar, T_K):
    """ This function calculates CO2 density in g/cm3 for a known Pressure (in kbar), a known T (in K) for the Span and Wagner (1996) EOS

    Parameters
    ---------------------
    P_kbar: int, float, pd.Series, np.array
        Pressure in kbar

    T_K: int, float, pd.Series, np.array
        Temperature in Kelvin

    Returns
    --------------------
    pd.Series
        CO2 density in g/cm3

    """
    if isinstance(P_kbar, pd.Series):
        P_kbar=np.array(P_kbar)
    if isinstance(T_K, pd.Series):
        T_K=np.array(T_K)

    P_Pa=P_kbar*10**8

    try:
        import CoolProp.CoolProp as cp
    except ImportError:
        raise RuntimeError('You havent installed CoolProp, which is required to convert FI densities to pressures. If you have python through conda, run conda install -c conda-forge coolprop in your command line')

    CO2_dens_gcm3=cp.PropsSI('D', 'P', P_Pa, 'T', T_K, 'CO2')/1000

    return pd.Series(CO2_dens_gcm3)

# Function for Sterner and Pitzer, references functions down below
def calculate_rho_for_P_T_SP94(P_kbar, T_K):
    """ This function calculates CO2 density in g/cm3 for a known Pressure (in kbar), a known T (in K) for the Sterner and Pitzer EOS
    it references the objective functions to solve for density.

    Parameters
    ---------------------
    P_kbar: int, float, pd.Series, np.array
        Pressure in kbar

    T_K: int, float, pd.Series, np.array
        Temperature in Kelvin

    Returns
    --------------------
    pd.Series
        CO2 density in g/cm3

    """
    target_pressure_MPa=P_kbar*100
    Density=calculate_SP19942(T_K=T_K, target_pressure_MPa=target_pressure_MPa)
    return  pd.Series(Density)



## Generic function for converting rho and T into Pressure

def calculate_P_for_rho_T(CO2_dens_gcm3, T_K, EOS='SW96', Sample_ID=None):
    """ This function calculates P in kbar for a specified CO2 density in g/cm3, a known T (in K), and a specified EOS

    Parameters
    ---------------------
    CO2_dens_gcm3: int, float, pd.Series, np.array
        CO2 density in g/cm3

    T_K: int, float, pd.Series, np.array
        Temperature in Kelvin

    EOS: str
        'SW96' for Span and Wanger (1996),  'SP94' for Sterner and Pitzer (1994),
        or  'DZ06' for Pure CO2 from Duan and Zhang (2006)

    Sample_ID: str, pd.Series
        Sample ID to be appended onto output dataframe

    Returns
    --------------------
    pd.DataFrame
        Pressure in kbar, MPa and input parameters

    """


    if EOS=='SW96':
        df=calculate_P_for_rho_T_SW96(CO2_dens_gcm3=CO2_dens_gcm3, T_K=T_K)
    elif EOS=='SP94':
        df=calculate_P_for_rho_T_SP94(CO2_dens_gcm3=CO2_dens_gcm3, T_K=T_K)
    elif EOS=='DZ06':
        df=calculate_P_for_rho_T_DZ06(CO2_dens_gcm3=CO2_dens_gcm3, T_K=T_K)


    else:
        raise TypeError('Please choose either SP94, SW96, or DZ06 as an EOS')

    if Sample_ID is not None:
        df['Sample_ID']=Sample_ID

    # Replace infinities with nan
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

# Calculating P for a given density and Temperature using Coolprop

def calculate_P_for_rho_T_DZ06(CO2_dens_gcm3, T_K):
    """ This function calculates P in kbar for a specified CO2 density in g/cm3 and a known T (in K) for the pure CO2 Duan and Zhang (2006) EOS (e.g. XH2O=0)

    Parameters
    ---------------------
    CO2_dens_gcm3: int, float, pd.Series, np.array
        CO2 density in g/cm3

    T_K: int, float, pd.Series, np.array
        Temperature in Kelvin


    Returns
    --------------------
    pd.DataFrame
        Pressure in kbar. MPa and input parameters

    """
    P_MPa=calculate_Pressure_DZ2006(density=CO2_dens_gcm3, T_K=T_K, XH2O=0)/10



    df=pd.DataFrame(data={'P_kbar': P_MPa/100,
                            'P_MPa': P_MPa,
                            'T_K': T_K,
                            'CO2_dens_gcm3': CO2_dens_gcm3

                            }, index=[0])

    return df




def calculate_P_for_rho_T_SW96(CO2_dens_gcm3, T_K):
    """ This function calculates P in kbar for a specified CO2 density in g/cm3 and a known T (in K) for the Span and Wanger (1996) EOS

    Parameters
    ---------------------
    CO2_dens_gcm3: int, float, pd.Series, np.array
        CO2 density in g/cm3

    T_K: int, float, pd.Series, np.array
        Temperature in Kelvin


    Returns
    --------------------
    pd.DataFrame
        Pressure in kbar. MPa and input parameters

    """
    if isinstance(CO2_dens_gcm3, pd.Series):
        CO2_dens_gcm3=np.array(CO2_dens_gcm3,)
    if isinstance(T_K, pd.Series):
        T_K=np.array(T_K)
    Density_kgm3=CO2_dens_gcm3*1000

    try:
        import CoolProp.CoolProp as cp
    except ImportError:
        raise RuntimeError('You havent installed CoolProp, which is required to convert FI densities to pressures. If you have python through conda, run conda install -c conda-forge coolprop in your command line')


    P_kbar=cp.PropsSI('P', 'D', Density_kgm3, 'T', T_K, 'CO2')/10**8
    if isinstance(P_kbar, float):
        df=pd.DataFrame(data={'P_kbar': P_kbar,
                                'P_MPa': P_kbar*100,
                                'T_K': T_K,
                                'CO2_dens_gcm3': CO2_dens_gcm3
                                }, index=[0])
    else:
        df=pd.DataFrame(data={'P_kbar': P_kbar,
                                'P_MPa': P_kbar*100,
                                'T_K': T_K,
                                'CO2_dens_gcm3': CO2_dens_gcm3
                                })

    return df
# OVeral
# Calculating P for a given density and Temp, Sterner and Pitzer

def calculate_P_for_rho_T_SP94(T_K, CO2_dens_gcm3, scalar_return=False):

    """ This function calculates P in kbar for a specified CO2 density in g/cm3 and a known T (in K) for the Sterner and Pitzer (1994) EOS

    Parameters
    ---------------------
    CO2_dens_gcm3: int, float, pd.Series, np.array
        CO2 density in g/cm3

    T_K: int, float, pd.Series, np.array
        Temperature in Kelvin


    Returns
    --------------------
    pd.DataFrame
        Pressure in kbar. MPa and input parameters

    """

    T=T_K-273.15
    T0=-273.15
    MolConc=CO2_dens_gcm3/44
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

    elif isinstance(P_MPa, pd.Series) or isinstance(P_MPa, np.ndarray):


        df=pd.DataFrame(data={'P_kbar': P_MPa/100,
                            'P_MPa': P_MPa,
                            'T_K': T_K,
                            'CO2_dens_gcm3': CO2_dens_gcm3

                            })


    else:
        df=pd.DataFrame(data={'P_kbar': P_MPa/100,
                            'P_MPa': P_MPa,
                            'T_K': T_K,
                            'CO2_dens_gcm3': CO2_dens_gcm3

                            }, index=[0])

    return df



## Inverting for temp if you know density and pressure
def calculate_T_for_rho_P(CO2_dens_gcm3, P_kbar, EOS='SW96', Sample_ID=None):
    """ This function calculates P in kbar for a specified CO2 density in g/cm3, a known T (in K), and a specified EOS

    Parameters
    ---------------------
    CO2_dens_gcm3: int, float, pd.Series, np.array
        CO2 density in g/cm3

    P_kbar: int, float, pd.Series, np.array
        Pressure in kbar

    EOS: str
        'SW96' for Span and Wanger (1996), or 'SP94' for Sterner and Pitzer (1994)

    Sample_ID: str, pd.Series
        Sample ID to be appended onto output dataframe

    Returns
    --------------------
    pd.DataFrame
        Temp in Kelvin and input parameters

    """


    if EOS=='SW96':
        df=calculate_T_for_rho_P_SW96(CO2_dens_gcm3=CO2_dens_gcm3, P_kbar=P_kbar)
    elif EOS=='SP94':
        df=calculate_T_for_rho_P_SP94(CO2_dens_gcm3=CO2_dens_gcm3, P_kbar=P_kbar)
    else:
        raise TypeError('Please choose either SP94 or SW96 as an EOS')

    if Sample_ID is not None:
        df['Sample_ID']=Sample_ID

    return df

def calculate_T_for_rho_P_SW96(CO2_dens_gcm3, P_kbar):
    """ This function calculates Temperature for a known CO2 density in g/cm3  and a known Pressure (in kbar)
    for the Span and Wagner (1996) EOS

    Parameters
    ---------------------
    CO2_dens_gcm3: int, float, pd.Series, np.array
        CO2 density in g/cm3

    P_kbar: int, float, pd.Series, np.array
        Pressure in kbar


    Returns
    --------------------
    pd.Series
        Pressure in kbar

    """
    if isinstance(P_kbar, pd.Series):
        P_kbar=np.array(P_kbar)
    if isinstance(CO2_dens_gcm3, pd.Series):
        CO2_dens_gcm3=np.array(CO2_dens_gcm3)

    P_Pa=P_kbar*10**8

    try:
        import CoolProp.CoolProp as cp
    except ImportError:
        raise RuntimeError('You havent installed CoolProp, which is required to convert FI densities to pressures. If you have python through conda, run conda install -c conda-forge coolprop in your command line')

    Temp=cp.PropsSI('T', 'D', CO2_dens_gcm3*1000, 'P', P_Pa, 'CO2')


    return pd.Series(Temp)

def calculate_T_for_rho_P_SP94(P_kbar, CO2_dens_gcm3):
    """ This function calculates Temp for a known Pressure (in kbar) and a known CO2 density in g/cm3
    using the Sterner and Pitzer EOS.
    it references the objective functions to solve for density.

    Parameters
    ---------------------
    P_kbar: int, float, pd.Series, np.array
        Pressure in kbar

   CO2_dens_gcm3: int, float, pd.Series, np.array
        CO2 density in g/cm3

    Returns
    --------------------
    pd.Series
        Temp in K

    """
    target_pressure_MPa=P_kbar*100
    Temp=calculate_SP1994_Temp(CO2_dens_gcm3=CO2_dens_gcm3, target_pressure_MPa=target_pressure_MPa)
    return  pd.Series(Temp)





## Overall function convertiong density or T_h_C into a pressure


def calculate_P_SP1994(*, T_K=1400,
T_h_C=None, phase=None, CO2_dens_gcm3=None, return_array=False):
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

        CO2_dens_gcm3: int, float, pd.Series
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

    if CO2_dens_gcm3 is not None and T_h_C is not None:
        raise TypeError('Enter either CO2_dens_gcm3 or T_h_C, not both')
    if CO2_dens_gcm3 is not None:
        density_to_use=CO2_dens_gcm3/44
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


    P_MPa=calculate_P_for_rho_T_SP94(T_K=T_K, CO2_dens_gcm3=density_to_use)

    if return_array is True:
        return P_MPa


    else:

        if isinstance(P_MPa, float) or isinstance(P_MPa, int):
            df=pd.DataFrame(data={'T_h_C': T_h_C,
                                'T_K': T_K,
                                'Liq_CO2_dens_gcm3': Liq_density,
                                'Gas_CO2_dens_gcm3': gas_density,
                                'P_MPa': P_MPa}, index=[0])
        else:


            df=pd.DataFrame(data={'T_h_C': T_h_C,
                                'T_K': T_K,
                                'Liq_CO2_dens_gcm3': Liq_density,
                                'Gas_CO2_dens_gcm3': gas_density,
                                'P_MPa': P_MPa})



        return df


## Sterner and Pitzer inverting for Density
import scipy
from scipy.optimize import minimize
# What we are trying to do, is run at various CO2 densities, until pressure matches input pressure

def objective_function_CO2_dens(CO2_dens_gcm3, T_K, target_pressure_MPa):
    """ This function minimises the offset between the calculated and target pressure.
    It finds the temp and CO2 density that correspond to the target pressure

    Parameters
    -------

    T_K: int, float
        Temperature in Kelvin to find P at (e.g. temp fluid was trapped at)

    CO2_dens_gcm3: int, float
        CO2 density in g/cm3

    target_pressure_MPa: int, float
        Pressure of CO2 fluid in MPa




    """
    # The objective function that you want to minimize
    calculated_pressure = calculate_P_for_rho_T_SP94(CO2_dens_gcm3=CO2_dens_gcm3, T_K=T_K, scalar_return=True)
    objective = np.abs(calculated_pressure - target_pressure_MPa)
    return objective

def calculate_Density_Sterner_Pitzer_1994(T_K, target_pressure_MPa):
    """ This function uses the objective function 'objective_function_CO2_dens' above to solve for CO2 density for a given pressure and Temp

    Parameters
    -------

    T_K: int, float
        Temperature in Kelvin to find P at (e.g. temp fluid was trapped at)


    target_pressure_MPa: int, float
        Pressure of CO2 fluid in MPa

    Returns
    -------
    CO2 density


    """
    initial_guess = 1 # Provide an initial guess for the density
    result = minimize(objective_function_CO2_dens, initial_guess, bounds=((0, 2), ), args=(T_K, target_pressure_MPa))
    return result.x


def calculate_SP19942(T_K, target_pressure_MPa):
    """ This function Solves for CO2 density for a given temp and pressure using the objective and minimise functions above.
    """
    if isinstance(target_pressure_MPa, float) or isinstance(target_pressure_MPa, int):
        target_p=np.array(target_pressure_MPa)
        Density=calculate_Density_Sterner_Pitzer_1994(T_K=T_K, target_pressure_MPa=target_p)
    else:
        Density=np.zeros(len(target_pressure_MPa))
        for i in range(0, len(target_pressure_MPa)):
            Density[i]=calculate_Density_Sterner_Pitzer_1994(T_K=T_K, target_pressure_MPa=target_pressure_MPa[i])
    return Density

# Lets do the same to solve for tempreature here


def objective_function_Temp(T_K, CO2_dens_gcm3, target_pressure_MPa):
    """ This function minimises the offset between the calculated and target pressure
    """
    # The objective function that you want to minimize
    calculated_pressure = calculate_P_for_rho_T_SP94(T_K=T_K, CO2_dens_gcm3=CO2_dens_gcm3, scalar_return=True)
    objective = np.abs(calculated_pressure - target_pressure_MPa)
    return objective


def calculate_Temp_Sterner_Pitzer_1994(CO2_dens_gcm3, target_pressure_MPa):
    """ This function uses the objective function above to solve for Temp for a given pressure and CO2 density
    """
    initial_guess = 1200 # Provide an initial guess for the density
    result = minimize(objective_function_Temp, initial_guess, bounds=((0, 2000), ), args=(CO2_dens_gcm3, target_pressure_MPa))
    return result.x

def calculate_SP1994_Temp(CO2_dens_gcm3, target_pressure_MPa):
    """ This function Solves for Temp for a given CO2 density and pressure using the objective and minimise functions above.
    """
    if isinstance(target_pressure_MPa, float) or isinstance(target_pressure_MPa, int):
        target_p=np.array(target_pressure_MPa)
        Density=calculate_Temp_Sterner_Pitzer_1994(CO2_dens_gcm3=CO2_dens_gcm3, target_pressure_MPa=target_p)
    else:
        Density=np.zeros(len(target_pressure_MPa))
        for i in range(0, len(target_pressure_MPa)):
            Density[i]=calculate_Temp_Sterner_Pitzer_1994(CO2_dens_gcm3=CO2_dens_gcm3, target_pressure_MPa=target_pressure_MPa[i])
    return Density



## Combined CO2 and H2O-CO2 EOS files to avoid circular imports for pure DZ EOS.


# Set up constants.
Tc1 = 647.25
Pc1 = 221.19
Tc2 = 301.1282
Pc2 = 73.773

# Set up low pressure and high pressure parameters for CO2.

aL1 = [0] * 16  # Assuming the array is 1-indexed like in C.
#So we dont need to adjust everything

aL1[1] = 4.38269941 / 10**2
aL1[2] = -1.68244362 / 10**1
aL1[3] = -2.36923373 / 10**1
aL1[4] = 1.13027462 / 10**2
aL1[5] = -7.67764181 / 10**2
aL1[6] = 9.71820593 / 10**2
aL1[7] = 6.62674916 / 10**5
aL1[8] = 1.06637349 / 10**3
aL1[9] = -1.23265258 / 10**3
aL1[10] = -8.93953948 / 10**6
aL1[11] = -3.88124606 / 10**5
aL1[12] = 5.61510206 / 10**5
aL1[13] = 7.51274488 / 10**3  # alpha for H2O
aL1[14] = 2.51598931  # beta for H2O
aL1[15] = 3.94 / 10**2  # gamma for H2O

# Higher pressure parameters - 0.2- 1 GPa
aH1 = [0] * 16  # Assuming the array is 1-indexed like in C

aH1[1] = 4.68071541 / 10**2
aH1[2] = -2.81275941 / 10**1
aH1[3] = -2.43926365 / 10**1
aH1[4] = 1.10016958 / 10**2
aH1[5] = -3.86603525 / 10**2
aH1[6] = 9.30095461 / 10**2
aH1[7] = -1.15747171 / 10**5
aH1[8] = 4.19873848 / 10**4
aH1[9] = -5.82739501 / 10**4
aH1[10] = 1.00936000 / 10**6
aH1[11] = -1.01713593 / 10**5
aH1[12] = 1.63934213 / 10**5
aH1[13] = -4.49505919 / 10**2  # alpha for H2O
aH1[14] = -3.15028174 / 10**1  # beta for H2O
aH1[15] = 1.25 / 10**2  # gamma for H2O


# Low presure CO2 parameters.

aL2 = [0] * 16  # Assuming the array is 1-indexed like in C

aL2[1] = 1.14400435 / 10**1
aL2[2] = -9.38526684 / 10**1
aL2[3] = 7.21857006 / 10**1
aL2[4] = 8.81072902 / 10**3
aL2[5] = 6.36473911 / 10**2
aL2[6] = -7.70822213 / 10**2
aL2[7] = 9.01506064 / 10**4
aL2[8] = -6.81834166 / 10**3
aL2[9] = 7.32364258 / 10**3
aL2[10] = -1.10288237 / 10**4
aL2[11] = 1.26524193 / 10**3
aL2[12] = -1.49730823 / 10**3
aL2[13] = 7.81940730 / 10**3  # alpha for CO2
aL2[14] = -4.22918013  # beta for CO2
aL2[15] = 1.585 / 10**1  # gamma for CO2



# High pressure CO2 parameters.
aH2 = [0] * 16  # Assuming the array is 1-indexed like in C
aH2[1] = 5.72573440 / 10**3
aH2[2] = 7.94836769
aH2[3] = -3.84236281 * 10.0
aH2[4] = 3.71600369 / 10**2
aH2[5] = -1.92888994
aH2[6] = 6.64254770
aH2[7] = -7.02203950 / 10**6
aH2[8] = 1.77093234 / 10**2
aH2[9] = -4.81892026 / 10**2
aH2[10] = 3.88344869 / 10**6
aH2[11] = -5.54833167 / 10**4
aH2[12] = 1.70489748 / 10**3
aH2[13] = -4.13039220 / 10**1  # alpha for CO2
aH2[14] = -8.47988634  # beta for CO2
aH2[15] = 2.800 / 10**2  # gamma for CO2



import pandas as pd
import numpy as np

def ensure_series(a, b, c):
    # Determine the target length
    lengths = [len(a) if isinstance(a, (pd.Series, np.ndarray)) else None,
               len(b) if isinstance(b, (pd.Series, np.ndarray)) else None,
               len(c) if isinstance(c, (pd.Series, np.ndarray)) else None]
    lengths = [l for l in lengths if l is not None]
    target_length = max(lengths) if lengths else 1

    # Convert each input to a Series of the target length
    if not isinstance(a, (pd.Series, np.ndarray)):
        a = pd.Series([a] * target_length)
    else:
        a = pd.Series(a)

    if not isinstance(b, (pd.Series, np.ndarray)):
        b = pd.Series([b] * target_length)
    else:
        b = pd.Series(b)

    if not isinstance(c, (pd.Series, np.ndarray)):
        c = pd.Series([c] * target_length)
    else:
        c = pd.Series(c)

    return a.reset_index(drop=True), b.reset_index(drop=True), c.reset_index(drop=True)


def ensure_series_4(a, b, c, d):
    # Determine the target length
    lengths = [len(a) if isinstance(a, (pd.Series, np.ndarray)) else None,
               len(b) if isinstance(b, (pd.Series, np.ndarray)) else None,
               len(c) if isinstance(c, (pd.Series, np.ndarray)) else None,
               len(d) if isinstance(d, (pd.Series, np.ndarray)) else None]
    lengths = [l for l in lengths if l is not None]
    target_length = max(lengths) if lengths else 1

    # Convert each input to a Series of the target length
    if not isinstance(a, (pd.Series, np.ndarray)):
        a = pd.Series([a] * target_length)
    else:
        a = pd.Series(a)

    if not isinstance(b, (pd.Series, np.ndarray)):
        b = pd.Series([b] * target_length)
    else:
        b = pd.Series(b)

    if not isinstance(c, (pd.Series, np.ndarray)):
        c = pd.Series([c] * target_length)
    else:
        c = pd.Series(c)

    if not isinstance(d, (pd.Series, np.ndarray)):
        d = pd.Series([d] * target_length)
    else:
        d = pd.Series(d)

    return a.reset_index(drop=True), b.reset_index(drop=True), c.reset_index(drop=True), d.reset_index(drop=True)



## Pure EOS functions
# First, we need the pure EOS
def pureEOS(i, V, P, B, C, D, E, F, Vc, TK, b, g):
    """
    This function calculates the compressability factor for a pure EOS using the modified Lee-Kesler
    equation.

    i=0 for H2O, i=1 for CO2.

    You input a volume, and it returns the difference between the compresability factor, and that calculated at the input P, V and T_K.
    E.g. gives the residual so that you can iterate.
    """
    CF = (1.0 + (B[i] * Vc[i] / V) + (C[i] * Vc[i] *
    Vc[i] / (V * V)) + (D[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] /
     (V * V * V * V)))

    CF += ((E[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] /
    (V * V * V * V * V)))

    CF += ((F[i] * Vc[i] * Vc[i] / (V * V)) *
    (b[i] + g[i] * Vc[i] * Vc[i] / (V * V)) *
    math.exp(-g[i] * Vc[i] * Vc[i] / (V * V)))

    return CF - (P * V) / (83.14467 * TK)

def pureEOS_CF(i, V, P, B, C, D, E, F, Vc, TK, b, g):
    """
    This function calculates the compressability factor for a pure EOS using the modified Lee-Kesler
    equation.

    i=0 for H2O, i=1 for CO2.

    You input a volume, and it returns the difference between the compresability factor, and that calculated at the input P, V and T_K.
    E.g. gives the residual so that you can iterate.
    """
    CF = (1.0 + (B[i] * Vc[i] / V) + (C[i] * Vc[i] *
    Vc[i] / (V * V)) + (D[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] /
     (V * V * V * V)))

    CF += ((E[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] /
    (V * V * V * V * V)))

    CF += ((F[i] * Vc[i] * Vc[i] / (V * V)) *
    (b[i] + g[i] * Vc[i] * Vc[i] / (V * V)) *
    math.exp(-g[i] * Vc[i] * Vc[i] / (V * V)))

    return CF


# Volume iterative function using Netwon-Raphson method.
def purevolume(i, V, P, B, C, D, E, F, Vc, TK, b, g):
    """ Using the pure EOS, this function solves for the best molar volume (in cm3/mol) using the pureEOS residual calculated above

    It returns the volume.

    """
    for iter in range(1, 51):
        # Calculate the derivative of the pureEOS function at (V, P)
        diff = (pureEOS(i, V + 0.0001, P,B, C, D, E, F, Vc, TK, b, g) - pureEOS(i, V, P, B, C, D, E, F, Vc, TK, b, g)) / 0.0001

        # Update the volume using the Newton-Raphson method
        Vnew = V - pureEOS(i, V, P, B, C, D, E, F, Vc, TK, b, g) / diff

        # Check if the update is within the tolerance (0.000001)
        if abs(Vnew - V) <= 0.000001:
            break

        # Update V for the next iteration
        V = Vnew

    # Return the final estimated volume
    return V

def purepressure(i, V, P, TK):
    """ Using the pure EOS, this function solves for the best pressure (in bars) using the pureEOS residual calculated above

    It returns the pressure in bars

    """
    for iter in range(1, 51):
        # Calculate the derivative of the pureEOS function at (V, P)
        k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess=get_EOS_params(P, TK)

        diff = (pureEOS(i, V, P + 0.0001, B, C, D, E, F, Vc, TK, b, g) - pureEOS(i, V, P, B, C, D, E, F, Vc, TK, b, g)) / 0.0001

        # Update the pressure using the Newton-Raphson method
        Pnew = P - pureEOS(i, V, P, B, C, D, E, F, Vc, TK, b, g) / diff

        # Dont allow negative solutions
        if Pnew < 0:
            Pnew = 30000

        # Check if the update is within the tolerance (0.000001)
        if abs(Pnew - P) <= 0.000001:
            break

        # Update P for the next iteration
        P = Pnew

    # Return the final estimated pressure
    return P




def mol_vol_to_density(mol_vol, XH2O):
    """ Converts molar volume (cm3/mol) to density (g/cm3) for a given XH2O"""
    density=((1-XH2O)*44 + (XH2O)*18)/mol_vol
    return density

def pure_lnphi(i, Z, B, Vc, V, C, D, E, F, g, b):
    """
    This function calculates the fugacity coefficient (kbar) from the equation of state for a pure fluid

    """
    lnph = Z[i] - 1.0 - math.log(Z[i]) + (B[i] * Vc[i] / V[i]) + (C[i] * Vc[i] * Vc[i] / (2.0 * V[i] * V[i]))
    lnph += (D[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] / (4.0 * V[i] * V[i] * V[i] * V[i]))
    lnph += (E[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] / (5.0 * V[i] * V[i] * V[i] * V[i] * V[i]))
    lnph += (F[i] / (2.0 * g[i])) * (b[i] + 1.0 - (b[i] + 1.0 + g[i] * Vc[i] * Vc[i] / (V[i] * V[i])) * math.exp(-g[i] * Vc[i] * Vc[i] / (V[i] * V[i])))

    return lnph



## Mixing between species
def cbrt_calc(x):
    """
    This function calculates the cubic root that can deal with negative numbers.
    """
    if x >= 0:
        return math.pow(x, 1/3)
    else:
        return -math.pow(-x, 1/3)


def mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK):
    """ This function is like the one for the pureEOS. It calculates the compressability factor, and
    then calculates the compressability factor based on the P-V-T you entered. It returns the residual between those two values.
    """
    CF = 1.0 + (BVc / V) + (CVc2 / (V * V)) + (DVc4 / (V * V * V * V)) + (EVc5 / (V * V * V * V * V))
    CF += (FVc2 / (V * V)) * (bmix + gVc2 / (V * V)) * np.exp(-gVc2 / (V * V))

    return CF - (P * V) / (83.14467 * TK)


def mixEOS_CF(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK):
    """ This function is like the one for the pureEOS. It calculates the compressability factor, and
    then calculates that based on the P-V-T you entered. It does not return the residual
    """
    CF = 1.0 + (BVc / V) + (CVc2 / (V * V)) + (DVc4 / (V * V * V * V)) + (EVc5 / (V * V * V * V * V))
    CF += (FVc2 / (V * V)) * (bmix + gVc2 / (V * V)) * np.exp(-gVc2 / (V * V))

    return CF


def mixvolume(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK):
    """ This function iterates in volume space to get the best match volume (cm3/mol) to the entered pressure using the mixEOS function above.

    """
    for iter in range(1, 51):
        diff = ((mixEOS(V + 0.0001, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK)
    - mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK)) / 0.0001)
        Vnew = V - mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK) / diff
        if abs(Vnew - V) <= 0.000001:
            break
        V = Vnew

    return V

import warnings as w

## We are going to have to use a look up table to help the netwon raphson converge.

# Load the lookup table from the CSV file
DiadFit_dir=Path(__file__).parent
file_str='lookup_table_noneg.csv'
dz06_lookuptable=pd.read_csv(DiadFit_dir/file_str)
#df = pd.read_csv('lookup_table_noneg.csv')





def get_initial_guess(V_target, T_K_target, XH2O_target):
    # Calculate the Euclidean distance from the target point to all points in the table
    # We normalize each dimension by its range to give equal weight to all parameters
    df=dz06_lookuptable

    # code to find best value

    P_range = df['P_kbar'].max() - df['P_kbar'].min()
    T_K_range = df['T_K'].max() - df['T_K'].min()
    XH2O_range = df['XH2O'].max() - df['XH2O'].min()
    V_range = df['V'].max() - df['V'].min()

    # Calculate normalized distances
    distances = np.sqrt(
        ((df['P_kbar'] - df['P_kbar'].mean()) / P_range) ** 2 +
        ((df['T_K'] - T_K_target) / T_K_range) ** 2 +
        ((df['XH2O'] - XH2O_target) / XH2O_range) ** 2 +
        ((df['V'] - V_target) / V_range) ** 2
    )

    # Drop NaN values from distances
    non_nan_distances = distances.dropna()

    # Check if all distances are NaN
    if non_nan_distances.empty:
        return 10

    # Find the index of the closest row in the DataFrame
    closest_index = non_nan_distances.idxmin()

    # Retrieve the P_kbar value from the closest row
    initial_guess_P = df.iloc[closest_index]['P_kbar']

    return initial_guess_P




def mixpressure(P, V, TK, Y):
    """ This function iterates in pressure space to get the best match in bars to the entered volume in cm3/mol using the mixEOS function above.

    """


    for iter in range(1, 51):
        k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess=get_EOS_params(P, TK)
        Bij, Vcij, BVc_prm, BVc, Cijk, Vcijk, CVc2_prm, CVc2, Dijklm, Vcijklm, DVc4_prm, DVc4, Eijklmn, Vcijklmn, EVc5_prm,  EVc5, Fij, FVc2_prm, FVc2, bmix, b_prm, gijk, gVc2_prm, gVc2=mixing_rules(B, C,D, E, F, Vc, Y, b,    g, k1_temperature, k2_temperature, k3_temperature)

        diff = ((mixEOS(V, P + 0.0001, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK)
        - mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK)) / 0.0001)
        Pnew = P - mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK) / diff




        if abs(Pnew - P) <= 0.000001:
            break

        P = Pnew

    return P

        # # Dont allow negative solutions
        # if Pnew<0:
        #     Pnew = 3000

        # if Pnew < 10000 and Pnew>0 and V<50:
        #     w.warn('Sometimes the adapted Newton Raphson method will find a second root at lower (or negative pressure). This initially found a root at P=' + str(np.round(Pnew, 2)) + ', V=' + str(np.round(V)) + '. The algorithm has started its search again at P=3000 bars. Double check your results make sense')
        #
        #     Pnew = 10000  # Replace 0.0001 with a small positive value that makes sense for your system
        #


# def mixpressure(P_init, V, TK, Y, max_restarts=3):
#     """This function iterates in pressure space to get the best match to the entered volume using the mixEOS function above."""
#
#     restarts = 0
#     while restarts <= max_restarts:
#         P = P_init
#         for iter in range(1, 51):
#             # Your EOS parameters and mixing rules calculations here
#
#             diff = ((mixEOS(V, P + 0.0001, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK) - mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK)) / 0.0001)
#             Pnew = P - mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK) / diff
#
#             # Don't allow unrealistic solutions but provide a chance to reset
#             if Pnew < 5000 and V > 10:
#                 warnings.warn('Root forced above 5000 bars due to conditions, attempting restart...')
#                 Pnew = 5000  # Force the pressure up due to your condition
#                 break  # Break out of the current iteration loop to allow for a restart
#
#             if abs(Pnew - P) <= 0.000001:  # Convergence criterion
#                 return Pnew  # Return the converged value
#             P = Pnew
#
#         restarts += 1  # Increment the number of restarts attempted
#         P_init = 5000  # Set a new starting point that might be closer to the suspected real root
#
#     warnings.warn('Max restarts reached, solution may not be optimal.')
#     return P  # Return the last computed pressure if all restart attempts fail






def mix_lnphi(i, Zmix, BVc_prm, CVc2_prm, DVc4_prm, EVc5_prm, FVc2_prm, FVc2, bmix, b_prm, gVc2, gVc2_prm, Vmix):
    """ This function calculates lnphi values"""
    lnph=0

    lnph = -math.log(Zmix)
    lnph += (BVc_prm[i] / Vmix)
    lnph += (CVc2_prm[i] / (2.0 * Vmix ** 2))
    lnph += (DVc4_prm[i] / (4.0 * Vmix ** 4))
    lnph += (EVc5_prm[i] / (5.0 * Vmix ** 5))
    lnph += ((FVc2_prm[i] * bmix + b_prm[i] * FVc2) / (2 * gVc2)) * (1.0 - math.exp(-gVc2 / (Vmix ** 2)))
    lnph += ((FVc2_prm[i] * gVc2 + gVc2_prm[i] * FVc2 - FVc2 * bmix * (gVc2_prm[i] - gVc2)) / (2.0 * gVc2 ** 2)) * (1.0 - (gVc2 / (Vmix ** 2) + 1.0) * math.exp(-gVc2 / (Vmix ** 2)))
    lnph += -(((gVc2_prm[i] - gVc2) * FVc2) / (2 * gVc2 ** 2)) * (2.0 - (((gVc2 ** 2) / (Vmix ** 4)) + (2.0 * gVc2 / (Vmix ** 2)) + 2.0) * math.exp(-gVc2 / (Vmix ** 2)))


    return lnph



def mix_fugacity_ind(*, P_kbar, T_K, XH2O, Vmix):
    """ This function calculates fugacity for a single sample.
    It returns the activity of each component (fugacity/fugacity in pure component)


    """

    P=P_kbar*1000
    TK=T_K
    XCO2=1-XH2O
    Y = [0] * 2
    Y[0]=XH2O
    Y[1]=XCO2

    # Calculate the constants you neeed
    k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess=get_EOS_params(P, TK)


    lnphi2kbL = [0.0, 0.0]
    lnphi2kbH = [0.0, 0.0]

    lnphi = [0.0, 0.0]
    phi_mix = [0.0, 0.0]
    activity = [0.0, 0.0]
    f = [0.0, 0.0]

    # Calculate at pressure of interest
    Z_pure = [0.0, 0.0]
    V_pure = [0.0, 0.0]


    # Initial guess for volumne

    if P<=2000:
        Vguess=1000
    elif P>20000:
        Vguess=10
    else:
        Vguess=100

    V_pure[1]=purevolume(1, Vguess, P, B, C, D, E, F, Vc, TK, b, g)
    V_pure[0]=purevolume(0, Vguess, P, B, C, D, E, F, Vc, TK, b, g)
    Z_pure[0]=P*V_pure[0]/(83.14467*TK)
    Z_pure[1]=P*V_pure[1]/(83.14467*TK)


    #H2O pure
    lnphi0=pure_lnphi(0, Z_pure, B, Vc, V_pure, C, D, E, F, g, b)
    #CO2 pure
    lnphi1=pure_lnphi(1, Z_pure, B, Vc, V_pure, C, D, E, F, g, b)


    # Funny maths you have to do incase P>2000 bars


    # First, calculate parameters with low pressure coefficients
    k1_temperature_LP, k2_temperature_LP, k3_temperature_LP, a1_LP, a2_LP, g_LP, b_LP, Vc_LP, B_LP, C_LP, D_LP, E_LP, F_LP, Vguess=get_EOS_params(500, TK)
    Z_pure_LP_2000 = [0.0, 0.0]
    V_pure_LP_2000 = [0.0, 0.0]
    V_pure_LP_2000[0]=purevolume(0, 100, 2000, B_LP, C_LP, D_LP, E_LP, F_LP, Vc_LP, TK, b_LP, g_LP)
    V_pure_LP_2000[1]=purevolume(1, 100, 2000, B_LP, C_LP, D_LP, E_LP, F_LP, Vc_LP, TK, b_LP, g_LP)

    Z_pure_LP_2000[0]=2000.0*V_pure_LP_2000[0]/(83.14467*TK)
    Z_pure_LP_2000[1]=2000.0*V_pure_LP_2000[1]/(83.14467*TK)

    # Low pressure
    lnphi0_LP=pure_lnphi(0, Z_pure_LP_2000, B_LP, Vc_LP, V_pure_LP_2000, C_LP, D_LP, E_LP, F_LP, g_LP, b_LP)
    lnphi1_LP=pure_lnphi(1, Z_pure_LP_2000, B_LP, Vc_LP, V_pure_LP_2000, C_LP, D_LP, E_LP, F_LP, g_LP, b_LP)



    # Same with high P
    k1_temperature_HP, k2_temperature_HP, k3_temperature_HP, a1_HP, a2_HP, g_HP, b_HP, Vc_HP, B_HP, C_HP, D_HP, E_HP, F_HP, Vguess=get_EOS_params(3000, TK)
    Z_pure_HP_2000 = [0.0, 0.0]
    V_pure_HP_2000 = [0.0, 0.0]
    V_pure_HP_2000[0]=purevolume(0, 100, 2000, B_HP, C_HP, D_HP, E_HP, F_HP, Vc_HP, TK, b_HP, g_HP)
    V_pure_HP_2000[1]=purevolume(1, 100, 2000, B_HP, C_HP, D_HP, E_HP, F_HP, Vc_HP, TK, b_HP, g_HP)
    Z_pure_HP_2000[0]=2000.0*V_pure_HP_2000[0]/(83.14467*TK)
    Z_pure_HP_2000[1]=2000.0*V_pure_HP_2000[1]/(83.14467*TK)


    #pure_HP
    lnphi0_HP=pure_lnphi(0, Z_pure_HP_2000, B_HP, Vc_HP, V_pure_HP_2000, C_HP, D_HP, E_HP, F_HP, g_HP, b_HP)
    lnphi1_HP=pure_lnphi(1, Z_pure_HP_2000, B_HP, Vc_HP, V_pure_HP_2000, C_HP, D_HP, E_HP, F_HP, g_HP, b_HP)

    if P>2000:
        # This is a weird thing described on Page 6 of Yoshimura -
        lnphi0=lnphi0-lnphi0_HP+lnphi0_LP
        lnphi1=lnphi1-lnphi1_HP+lnphi1_LP

    phi0_pure=math.exp(lnphi0)
    phi1_pure=math.exp(lnphi1)



    # Now we need to do the mixed fugacity part of this
    #--------------------------------------------------------------------------
    k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess=get_EOS_params(P, TK)
    Bij, Vcij, BVc_prm, BVc, Cijk, Vcijk, CVc2_prm, CVc2, Dijklm, Vcijklm, DVc4_prm, DVc4, Eijklmn, Vcijklmn, EVc5_prm,  EVc5, Fij, FVc2_prm, FVc2, bmix, b_prm, gijk, gVc2_prm, gVc2=mixing_rules(B, C, D, E, F, Vc, Y, b, g, k1_temperature, k2_temperature, k3_temperature)
    Zmix=(P*Vmix)/(83.14467*TK)
    lnphi_mix = [0.0, 0.0]
    phi_mix = [0.0, 0.0]
    lnphi_mix[0]=mix_lnphi(0,  Zmix, BVc_prm, CVc2_prm, DVc4_prm, EVc5_prm, FVc2_prm,FVc2, bmix, b_prm, gVc2, gVc2_prm, Vmix)
    lnphi_mix[1]=mix_lnphi(1,  Zmix, BVc_prm, CVc2_prm, DVc4_prm, EVc5_prm, FVc2_prm,FVc2, bmix, b_prm, gVc2, gVc2_prm, Vmix)



    # But what if P>2000, well we need to do these calcs at low and high P
    # High P - using Parameters from up above
    Bij_HP, Vcij_HP, BVc_prm_HP, BVc_HP, Cijk_HP, Vcijk_HP, CVc2_prm_HP, CVc2_HP, Dijklm_HP, Vcijklm_HP, DVc4_prm_HP, DVc4_HP, Eijklmn_HP, Vcijklmn_HP, EVc5_prm_HP,  EVc5_HP, Fij_HP, FVc2_prm_HP, FVc2_HP, bmix_HP, b_prm_HP, gijk_HP, gVc2_prm_HP, gVc2_HP=mixing_rules(B_HP, C_HP, D_HP, E_HP, F_HP, Vc_HP, Y, b_HP, g_HP, k1_temperature_HP, k2_temperature_HP, k3_temperature_HP)
    Vmix_HP=mixvolume(100, 2000, BVc_HP, CVc2_HP, DVc4_HP, EVc5_HP, FVc2_HP, bmix_HP, gVc2_HP, TK)
    Zmix_HP=(2000*Vmix_HP)/(83.14467*TK)
    lnphi_mix_HP = [0.0, 0.0]
    lnphi_mix_HP[0]=mix_lnphi(0,  Zmix_HP, BVc_prm_HP, CVc2_prm_HP, DVc4_prm_HP, EVc5_prm_HP, FVc2_prm_HP,FVc2_HP, bmix_HP, b_prm_HP, gVc2_HP, gVc2_prm_HP, Vmix_HP)
    lnphi_mix_HP[1]=mix_lnphi(1,  Zmix_HP, BVc_prm_HP, CVc2_prm_HP, DVc4_prm_HP, EVc5_prm_HP, FVc2_prm_HP, FVc2_HP, bmix_HP, b_prm_HP, gVc2_HP, gVc2_prm_HP, Vmix_HP)


    # Same for LP
    Bij_LP, Vcij_LP, BVc_prm_LP, BVc_LP, Cijk_LP, Vcijk_LP, CVc2_prm_LP, CVc2_LP, Dijklm_LP, Vcijklm_LP, DVc4_prm_LP, DVc4_LP, Eijklmn_LP, Vcijklmn_LP, EVc5_prm_LP,  EVc5_LP, Fij_LP, FVc2_prm_LP, FVc2_LP, bmix_LP, b_prm_LP, gijk_LP, gVc2_prm_LP, gVc2_LP=mixing_rules(B_LP, C_LP, D_LP, E_LP, F_LP,Vc_LP, Y, b_LP, g_LP, k1_temperature_LP, k2_temperature_LP, k3_temperature_LP)
    Vmix_LP=mixvolume(100, 2000, BVc_LP, CVc2_LP, DVc4_LP, EVc5_LP, FVc2_LP, bmix_LP, gVc2_LP, TK)
    Zmix_LP=(2000*Vmix_LP)/(83.14467*TK)
    lnphi_mix_LP = [0.0, 0.0]
    lnphi_mix_LP[0]=mix_lnphi(0,  Zmix_LP, BVc_prm_LP, CVc2_prm_LP, DVc4_prm_LP, EVc5_prm_LP, FVc2_prm_LP, FVc2_LP, bmix_LP, b_prm_LP, gVc2_LP, gVc2_prm_LP, Vmix_LP)
    lnphi_mix_LP[1]=mix_lnphi(1,  Zmix_LP, BVc_prm_LP, CVc2_prm_LP, DVc4_prm_LP, EVc5_prm_LP, FVc2_prm_LP,FVc2_LP, bmix_LP, b_prm_LP, gVc2_LP, gVc2_prm_LP, Vmix_LP)

    if P>2000:
        lnphi_mix[0]=lnphi_mix[0]-lnphi_mix_HP[0]+lnphi_mix_LP[0]
        lnphi_mix[1]=lnphi_mix[1]-lnphi_mix_HP[1]+lnphi_mix_LP[1]


    phi_mix[0]=math.exp(lnphi_mix[0])
    phi_mix[1]=math.exp(lnphi_mix[1])







    activity[0] = phi_mix[0] * Y[0] / phi0_pure
    activity[1] = phi_mix[1] * Y[1] / phi1_pure

    f[0] = Y[0] * P * phi_mix[0] / 1000.0  # fugacity in kbar
    f[1] = Y[1] * P * phi_mix[1] / 1000.0  # fugacity in kbar




    return f[0], f[1], activity[0], activity[1], Zmix





def mixing_rules(B, C, D, E, F, Vc, Y, b, g, k1_temperature, k2_temperature, k3_temperature):
    """ This function applies the DZ06 mixing rules"""

    Bij = np.zeros((2, 2))
    Vcij = np.zeros((2, 2))
    BVc_prm = np.zeros(2)
    b_prm=np.zeros(2)
    BVc = 0.0
    Cijk = np.zeros((2, 2, 2))
    Vcijk = np.zeros((2, 2, 2))
    CVc2_prm = np.zeros(2)
    CVc2 = 0.0
    Dijklm = np.zeros((2, 2, 2, 2, 2))
    Vcijklm = np.zeros((2, 2, 2, 2, 2))
    Eijklmn=np.zeros((2, 2, 2, 2, 2, 2))
    Vcijklmn=np.zeros((2, 2, 2, 2, 2, 2))
    DVc4_prm=np.zeros(2)
    DVc4=0

    EVc5_prm = np.zeros(2)
    EVc5 = 0.0
    Fij = np.zeros((2, 2))
    FVc2_prm = np.zeros(2)
    FVc2 = 0.0
    gijk = np.zeros((2, 2, 2))
    gVc2_prm = np.zeros(2)
    gVc2 = 0.0

    for i in range(2):
        for j in range(2):
            k1 = 1.0 if i == j else k1_temperature
            Bij[i, j] = pow((cbrt_calc(B[i]) + cbrt_calc(B[j]))/2, 3.0) * k1

    for i in range(2):
        for j in range(2):
            Vcij[i, j] = pow((cbrt_calc(Vc[i]) + cbrt_calc(Vc[j]))/2, 3.0)


    for i in range(2):
        for j in range(2):
            BVc_prm[i] += 2 * Y[j] * Bij[i, j] * Vcij[i, j]



    for i in range(2):
        for j in range(2):

            BVc += Y[i] * Y[j] * Bij[i, j] * Vcij[i, j]

    for i in range(2):
        for j in range(2):
            for k in range(2):
                k2 = 1.0 if i == j and j == k else k2_temperature
                Cijk[i, j, k] = pow((cbrt_calc(C[i]) + cbrt_calc(C[j]) + cbrt_calc(C[k]))/3, 3.0) * k2

    for i in range(2):
        for j in range(2):
            for k in range(2):
                Vcijk[i, j, k] = pow((cbrt_calc(Vc[i]) + cbrt_calc(Vc[j]) + cbrt_calc(Vc[k]))/3, 3.0)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                CVc2_prm[i] += 3 * Y[j] * Y[k] * Cijk[i, j, k] * Vcijk[i, j, k] * Vcijk[i, j, k]

    CVc2=0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                CVc2 += Y[i] * Y[j] * Y[k] * Cijk[i, j, k] * Vcijk[i, j, k] * Vcijk[i, j, k]


    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        Dijklm[i, j, k, l, m] = pow((cbrt_calc(D[i]) + cbrt_calc(D[j]) + cbrt_calc(D[k]) + cbrt_calc(D[l]) + cbrt_calc(D[m]))/5, 3.0)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        Vcijklm[i, j, k, l, m] = pow((cbrt_calc(Vc[i]) + cbrt_calc(Vc[j]) + cbrt_calc(Vc[k]) + cbrt_calc(Vc[l]) + cbrt_calc(Vc[m]))/5, 3.0)


    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        DVc4_prm[i] += 5.0 * Y[j] * Y[k] * Y[l] * Y[m] * Dijklm[i, j, k, l, m] * pow(Vcijklm[i, j, k, l, m], 4.0)


    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        DVc4 += Y[i] * Y[j] * Y[k] * Y[l] * Y[m] * Dijklm[i, j, k, l, m] * pow(Vcijklm[i, j, k, l, m], 4)

# Missing Eijklmn,
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            Eijklmn[i, j, k, l, m, n] = pow((cbrt_calc(E[i]) + cbrt_calc(E[j]) + cbrt_calc(E[k]) + cbrt_calc(E[l]) + cbrt_calc(E[m]) + cbrt_calc(E[n]))/6, 3.0)


    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            Vcijklmn[i, j, k, l, m, n] = pow((cbrt_calc(Vc[i]) + cbrt_calc(Vc[j]) + cbrt_calc(Vc[k]) + cbrt_calc(Vc[l]) + cbrt_calc(Vc[m]) + cbrt_calc(Vc[n]))/6, 3.0)



    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            EVc5_prm[i] += 6.0 * Y[j] * Y[k] * Y[l] * Y[m] * Y[n] * Eijklmn[i, j, k, l, m, n] * pow(Vcijklmn[i, j, k, l, m, n], 5.0)


    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            EVc5 += Y[i] * Y[j] * Y[k] * Y[l] * Y[m] * Y[n] * Eijklmn[i, j, k, l, m, n] * pow(Vcijklmn[i, j, k, l, m, n], 5.0)

    for i in range(2):
        for j in range(2):
            Fij[i, j] = pow((cbrt_calc(F[i]) + cbrt_calc(F[j]))/2, 3.0)

    for i in range(2):
        for j in range(2):
            FVc2_prm[i] += 2.0 * Y[j] * Fij[i, j] * Vcij[i, j] * Vcij[i, j]

    for i in range(2):
        for j in range(2):
            FVc2 += Y[i] * Y[j] * Fij[i, j] * Vcij[i, j] * Vcij[i, j]

    bmix = Y[0] * b[0] + Y[1] * b[1]

    b_prm[0] = b[0]
    b_prm[1] = b[1]

    for i in range(2):
        for j in range(2):
            for k in range(2):
                k3 = 1.0 if i == j and j == k else k3_temperature
                gijk[i, j, k] = pow((cbrt_calc(g[i]) + cbrt_calc(g[j]) + cbrt_calc(g[k]))/3, 3.0) * k3

    for i in range(2):
        for j in range(2):
            for k in range(2):
                gVc2_prm[i] += 3.0 * Y[j] * Y[k] * gijk[i, j, k] * Vcijk[i, j, k] * Vcijk[i, j, k]


    for i in range(2):
        for j in range(2):
            for k in range(2):
                gVc2 += Y[i] * Y[j] * Y[k] * gijk[i, j, k] * Vcijk[i, j, k] * Vcijk[i, j, k]


    return Bij, Vcij, BVc_prm, BVc, Cijk, Vcijk, CVc2_prm, CVc2, Dijklm, Vcijklm, DVc4_prm, DVc4, Eijklmn, Vcijklmn, EVc5_prm,  EVc5, Fij, FVc2_prm, FVc2, bmix, b_prm, gijk, gVc2_prm, gVc2


## Getting EOS contsants themselves

def get_EOS_params(P, TK):
    """ This function returns the EOS 'constants' if you know the pressure (in bars) and temperature (in Kelvin)

    """

    a1 = np.zeros(16)
    a2 = np.zeros(16)
    b = np.zeros(2)
    g = np.zeros(2)
    Vc = np.zeros(1)
    B = np.zeros(2)
    C = np.zeros(2)
    D = np.zeros(2)
    E = np.zeros(2)
    F = np.zeros(2)
    V = np.zeros(2)
    Vc = np.zeros(2)








    # Initial guess for volumne

    if P<=2000:
        Vguess=1000
    elif P>20000:
        Vguess=10
    else:
        Vguess=100


    if P <= 2000.0:
        for i in range(16):
            a1[i] = aL1[i]
            a2[i] = aL2[i]
        # These are the binary interaction parameters
        k1_temperature = 3.131 - (5.0624 / 10**3.0) * TK + (1.8641 / 10**6) * TK**2 - 31.409 / TK
        k2_temperature = -46.646 + (4.2877 / 10**2.0) * TK - (1.0892 / 10**5) * TK**2 + 1.5782 * 10**4 / TK
        k3_temperature = 0.9
    else:
        for i in range(16):
            a1[i] = aH1[i]
            a2[i] = aH2[i]
        # Same, but for higher pressures
        k1_temperature = 9.034 - (7.9212 / 10**3) * TK + (2.3285 / 10**6) * TK**2 - 2.4221 * 10**3 / TK
        k2_temperature = -1.068 + (1.8756 / 10**3) * TK - (4.9371 / 10**7) * TK**2 + 6.6180 * 10**2 / TK
        k3_temperature = 1.0

    b[0] = a1[14]  # beta for H2O
    b[1] = a2[14]  # beta for CO2
    g[0] = a1[15]  # gamma for H2O
    g[1] = a2[15]  # gamma for CO2

    Vc[0] = 83.14467 * Tc1 / Pc1
    B[0] = a1[1] + a1[2] / ((TK / Tc1) * (TK / Tc1)) + a1[3] / ((TK / Tc1) * (TK / Tc1) * (TK / Tc1))
    C[0] = a1[4] + a1[5] / ((TK / Tc1) * (TK / Tc1)) + a1[6] / ((TK / Tc1) * (TK / Tc1) * (TK / Tc1))
    D[0] = a1[7] + a1[8] / ((TK / Tc1) * (TK / Tc1)) + a1[9] / ((TK / Tc1) * (TK / Tc1) * (TK / Tc1))
    E[0] = a1[10] + a1[11] / ((TK / Tc1) * (TK / Tc1)) + a1[12] / ((TK / Tc1) * (TK / Tc1) * (TK / Tc1))
    F[0] = a1[13] / ((TK / Tc1) * (TK / Tc1) * (TK / Tc1))

    Vc[1] = 83.14467 * Tc2 / Pc2
    B[1] = a2[1] + a2[2] / ((TK / Tc2) * (TK / Tc2)) + a2[3] / ((TK / Tc2) * (TK / Tc2) * (TK / Tc2))
    C[1] = a2[4] + a2[5] / ((TK / Tc2) * (TK / Tc2)) + a2[6] / ((TK / Tc2) * (TK / Tc2) * (TK / Tc2))
    D[1] = a2[7] + a2[8] / ((TK / Tc2) * (TK / Tc2)) + a2[9] / ((TK / Tc2) * (TK / Tc2) * (TK / Tc2))
    E[1] = a2[10] + a2[11] / ((TK / Tc2) * (TK / Tc2)) + a2[12] / ((TK / Tc2) * (TK / Tc2) * (TK / Tc2))
    F[1] = a2[13] / ((TK / Tc2) * (TK / Tc2) * (TK / Tc2))
    return k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess

## Lets wrap all these functions up.

def calculate_molar_volume_ind_DZ2006(*, P_kbar, T_K, XH2O):
    """ This function calculates molar volume (cm3/mol) for a known pressure (kbar), T in K and XH2O (mol frac) for a single value
    """

    P=P_kbar*1000
    TK=T_K

    # Calculate the constants you neeed
    k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess=get_EOS_params(P, TK)

    if XH2O==0:
        mol_vol=purevolume(1, Vguess, P, B, C, D, E, F, Vc, TK, b, g)

    if XH2O==1:
        mol_vol=purevolume(0, Vguess, P, B, C, D, E, F, Vc, TK, b, g)

    else:
        XCO2=1-XH2O
        Y = [0] * 2
        Y[0]=XH2O
        Y[1]=XCO2
        Bij, Vcij, BVc_prm, BVc, Cijk, Vcijk, CVc2_prm, CVc2, Dijklm, Vcijklm, DVc4_prm, DVc4, Eijklmn, Vcijklmn, EVc5_prm,  EVc5, Fij, FVc2_prm, FVc2, bmix, b_prm, gijk, gVc2_prm, gVc2=mixing_rules(B, C,D, E, F, Vc, Y, b,    g, k1_temperature, k2_temperature, k3_temperature)


        mol_vol=mixvolume(Vguess, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, T_K)

    if mol_vol<0:
        mol_vol=np.nan

    return mol_vol


def calculate_molar_volume_DZ2006(*, P_kbar, T_K, XH2O):
    """ Used to calculate molar volume (cm3/mol) in a loop for multiple inputs


    """

    P_kbar, T_K, XH2O=ensure_series(P_kbar, T_K, XH2O)

    # Check all the same length
    lengths = [len(P_kbar), len(T_K), len(XH2O)]
    if len(set(lengths)) != 1:
        raise ValueError("All input Pandas Series must have the same length.")

    # Set up loop
    mol_vol=np.zeros(len(P_kbar), float)

    for i in range(0, len(P_kbar)):
        mol_vol[i]=calculate_molar_volume_ind_DZ2006(P_kbar=P_kbar.iloc[i].astype(float), T_K=T_K.iloc[i].astype(float), XH2O=XH2O.iloc[i].astype(float))





    return mol_vol

def calculate_Pressure_ind_DZ2006(*, mol_vol, T_K, XH2O, Pguess=None):
    """ This function calculates pressure  (in bars) for a known molar volume, T in K and XH2O (mol frac) for a single value. It uses a look up table to get pressure, then a newton and raphson method (implemented in the function mixpressure) to find the best fit pressure. There are some densities, T_K and XH2O values where the volume is negative.
    """
    V=mol_vol
    # if Pguess is None:
    #     if V>1000:
    #         Pguess=1000
    #     elif V<10:
    #         Pguess=20000
    #     else:
    #         Pguess=200

    # Lets get P guess from a look up table
    # uses a look up table
    Pguess=get_initial_guess(V_target=V, T_K_target=T_K, XH2O_target=XH2O)*1000

    if Pguess <= 0:
            return np.nan


    TK=T_K

    # lets do for low pressure initially


    # if XH2O==0:
    #     P=purepressure(1,  V, Pguess, TK)
    #
    # elif XH2O==1:
    #     P=purepressure(0, V, Pguess, TK)
    #
    # else:
    XCO2=1-XH2O
    Y = [0] * 2
    Y[0]=XH2O
    Y[1]=XCO2

    # Now iteratively solve for pressure starting from this initial guess.

    P=mixpressure(Pguess, V, T_K, Y)

    return P

def calculate_Pressure_DZ2006(*, mol_vol=None, density=None, T_K, XH2O):
    """ Used to calculate pressure in a loop for multiple inputs.
    Den - bulk density.

    Parameters
    ----------------
    mol_vol: molar volume in g/cm3
    density: density in g/cm3
    T_K: entrapment temperature in kelvin
    XH2O: molar fraction of H2O in the fluid

    Returns
     ----------------
     Pressure in bars


    """
    # Make all a panda series



    if mol_vol is None and density is not None:
        mol_vol=density_to_mol_vol(density=density, XH2O=XH2O)

    mol_vol, T_K, XH2O=ensure_series(mol_vol, T_K, XH2O)

    # Check all the same length
    lengths = [len(mol_vol), len(T_K), len(XH2O)]
    if len(set(lengths)) != 1:
        raise ValueError("All input Pandas Series must have the same length.")

    # Set up loop
    P=np.zeros(len(mol_vol), float)

    for i in range(0, len(mol_vol)):
        P[i]=calculate_Pressure_ind_DZ2006(mol_vol=mol_vol.iloc[i].astype(float), T_K=T_K.iloc[i].astype(float), XH2O=XH2O.iloc[i].astype(float))



    return P


def mix_fugacity(*, P_kbar, T_K, XH2O, Vmix):

    """ Used to calculate fugacity, compressability and activities for a panda series

    """
    # Make everything a pandas series

    P_kbar, T_K, XH2O, Vmix=ensure_series_4(P_kbar, T_K, XH2O, Vmix)



    #Check all the same length
    lengths = [len(P_kbar), len(T_K), len(XH2O), len(Vmix)]
    if len(set(lengths)) != 1:
        raise ValueError("All input Pandas Series must have the same length.")

    f=np.zeros(len(P_kbar), float)
    a_CO2=np.zeros(len(P_kbar), float)
    a_H2O=np.zeros(len(P_kbar), float)
    f_CO2=np.zeros(len(P_kbar), float)
    f_H2O=np.zeros(len(P_kbar), float)
    Zmix=np.zeros(len(P_kbar), float)
    for i in range(0, len(P_kbar)):

        f_H2O[i], f_CO2[i], a_H2O[i], a_CO2[i], Zmix[i]=mix_fugacity_ind(P_kbar=P_kbar.iloc[i].astype(float), T_K=T_K.iloc[i].astype(float), XH2O=XH2O.iloc[i].astype(float), Vmix=Vmix.iloc[i].astype(float))

    return f_H2O, f_CO2, a_H2O,a_CO2,  Zmix


def mol_vol_to_density(*, mol_vol, XH2O):
    """ Converts molar mass g/mol to densit g/cm3 for a given XH2O"""
    density=((1-XH2O)*44 + (XH2O)*18)/mol_vol
    return density

def density_to_mol_vol(*, density, XH2O):
    """ Converts density in g/cm3 to molar volume (mol/cm3) for a given XH2O"""
    mol_vol=((1-XH2O)*44 + (XH2O)*18)/density
    return mol_vol



def calc_prop_knownP_EOS_DZ2006(*, P_kbar=1, T_K=1200, XH2O=1):
    """ This function calculates molar volume, density, compressability factor, fugacity, and activity for mixed H2O-CO2 fluids
    using the EOS of Span and Wanger. It assumes you know P, T, and XH2O.

    Parameters
    -------------------
    P_kbar: float, np.array, pd.Series
        Pressure in kbar
    T_K: float, np.array, pd.Series
        Temperature in Kelvin
    XH2O: float, np.array, pd.Series
        Molar fraction of H2O in the fluid phase.

    Returns
    -------------------
    pd.DataFrame

    """



    # First, check all pd Series


    mol_vol=calculate_molar_volume_DZ2006(P_kbar=P_kbar, T_K=T_K, XH2O=XH2O)


    f_H2O, f_CO2, a_H2O, a_CO2, Zmix=mix_fugacity(P_kbar=P_kbar, T_K=T_K, XH2O=XH2O,
                                                      Vmix=mol_vol)
    density=mol_vol_to_density(mol_vol=mol_vol, XH2O=XH2O)
    # 'T_K': T_K,
    # 'P_kbar': P_kbar,
    # 'XH2O': XH2O,
    #


    df=pd.DataFrame(data={'P_kbar': P_kbar,
                          'T_K': T_K,
                          'XH2O': XH2O,
                          'XCO2': 1-XH2O,
                          'Molar Volume (cm3/mol)': mol_vol,
                          'Density (g/cm3)': density,
                          'Compressability_factor': Zmix,
                          'fugacity_H2O (kbar)': f_H2O,
                          'fugacity_CO2 (kbar)': f_CO2,
                          'activity_H2O': a_H2O,
                          'activity_CO2': a_CO2})

    return df



def calculate_entrapment_P_XH2O(*, XH2O, CO2_dens_gcm3, T_K, T_K_ambient=37+273.15, fast_calcs=False, Hloss=True):
    """" This function calculates pressure for a measured CO$_2$ density, temperature and estimate of initial XH2O.
    It first corrects the density to obtain a bulk density for a CO2-H2O mix, assuming that H2O was lost from the inclusion.
    correcting for XH2O. It assumes that H2O has been lost from the inclusion (see Hansteen and Klugel, 2008 for method). It also calculates using other
    pure CO2 equation of states for comparison

    Parameters
    ----------------------
    XH2O: float, pd.Series.
        The molar fraction of H2O in the fluid. Should be between 0 and 1. Can get an estimate from say VESical.

    CO2_dens_gcm3: float, pd.Series
        Measured CO2 density in g/cm3

    T_K: float, pd.Series
        Temperature in Kelvin fluid was trapped at

    T_K_ambient: pd.Series
        Temperature in Kelvin Raman measurement was made at.

    fast_calcs: bool (default False)
        If True, only performs one EOS calc for DZ06, not 4 (with water, without water, SP94 and SW96).
        also specify H2Oloss=True or False



    Returns
    -----------------------------
    if fast_calcs is False:
    pd.DataFrame:
        Columns showing:
        P_kbar_pureCO2_SW96: Pressure calculated for the measured CO$_2$ density using the pure CO2 EOS from Span and Wanger (1996)
        P_kbar_pureCO2_SP94: Pressure calculated for the measured CO$_2$ density using the pure CO2 EOS from Sterner and Pitzer (1994)
        P_kbar_pureCO2_DZ06: Pressure calculated from the measured CO$_2$ density using the pure CO2 EOs from Duan and Zhang (2006)
        P_kbar_mixCO2_DZ06_Hloss: Pressure calculated from the reconstructed mixed fluid density using the mixed EOS from Duan and Zhang (2006) assuming H loss
        P_kbar_mixCO2_DZ06_noHloss: Pressure calculated from the reconstructed mixed fluid density using the mixed EOS from Duan and Zhang (2006) assuming H loss
        P Mix_Hloss/P Pure DZ06: Correction factor - e.g. how much deeper the pressure is from the mixed EOS with H loss
        P Mix_noHloss/P Pure DZ06: Correction factor - e.g. how much deeper the pressure is from the mixed EOS (assuming no H loss)
        rho_mix_calc_noHloss: Bulk density calculated (C+H)
        rho_mix_calc_Hloss: Bulk density calculated (C+H) after h loss
        CO2_dens_gcm3: Input CO2 density
        T_K: input temperature
        XH2O: input molar fraction of H2O

    if fast_calcs is True:
        P_kbar_mixCO2_DZ06: Pressure calculated from the reconstructed mixed fluid density using the mixed EOS from Duan and Zhang (2006)



    """

    XH2O, rho_meas, T_K=ensure_series(a=XH2O, b=CO2_dens_gcm3, c=T_K)
    alpha=XH2O/(1-XH2O)

    # All inputs 194 up to here


    # IF water is lost
    rho_orig_H_loss=rho_meas*(1+alpha*(18/44))
    # IF water isnt lost

    # Calculate mass ratio from molar ratio
    XH2O_mass=(XH2O*18)/((1-XH2O)*44 +(XH2O*18) )
    # Calculate pressure in CO2 fluid - returns answer in kbar
    P_CO2=calculate_P_for_rho_T_SW96(CO2_dens_gcm3, T_K_ambient)['P_kbar']
    # Now calculate density of H2O fluid

    # See https://www.youtube.com/watch?v=6wE4Tk6OjGY
    Ptotal=P_CO2/(1-XH2O) # Calculate the total pressure from the pressure we know for CO2.
    P_H2O=Ptotal*XH2O # Now calculate the pressure of H2O. You could also do this as PTot*XH2O

    # calculate density of H2O using EOS
    H2O_dens=calculate_rho_for_P_T_H2O(P_kbar=P_H2O,T_K=T_K_ambient)
    H2O_dens=H2O_dens.reset_index(drop=True)

    # Calculate the bulk density by re-arranging the two volume equations
    nan_mask = H2O_dens==0

    # Debugging




    rho_orig_no_H_loss=(rho_meas*H2O_dens)/((1-XH2O_mass)*H2O_dens+XH2O_mass*rho_meas)




    rho_orig_no_H_loss = np.where(nan_mask, rho_meas, rho_orig_no_H_loss)




    if fast_calcs is True:
        if Hloss is True:
            P=calculate_Pressure_DZ2006(density=rho_orig_H_loss, T_K=T_K, XH2O=XH2O)
        if Hloss is False:
            P=calculate_Pressure_DZ2006(density=rho_orig_no_H_loss, T_K=T_K, XH2O=XH2O)
        return P/1000

    else:

        # Lets calculate the pressure using SW96
        P_SW=calculate_P_for_rho_T(T_K=T_K, CO2_dens_gcm3=rho_meas, EOS='SW96')
        P_SP=calculate_P_for_rho_T(T_K=T_K, CO2_dens_gcm3=rho_meas, EOS='SP94')
        # Run Duan and Zhang with no XHO@ to start with
        P_DZ=calculate_Pressure_DZ2006(density=rho_meas, T_K=T_K, XH2O=XH2O*0)
        # Now doing it with XH2O for two different densities

        P_DZ_mix_H_loss=calculate_Pressure_DZ2006(density=rho_orig_H_loss, T_K=T_K, XH2O=XH2O)
        P_DZ_mix_noH_loss=calculate_Pressure_DZ2006(density=rho_orig_no_H_loss, T_K=T_K, XH2O=XH2O)

        df=pd.DataFrame(data={
            'P_kbar_pureCO2_SW96': P_SW['P_kbar'],
            'P_kbar_pureCO2_SP94': P_SP['P_kbar'],
            'P_kbar_pureCO2_DZ06': P_DZ/1000,
            'P_kbar_mixCO2_DZ06_Hloss': P_DZ_mix_H_loss/1000,
            'P_kbar_mixCO2_DZ06_no_Hloss': P_DZ_mix_noH_loss/1000,
            'P Mix_Hloss/P Pure DZ06': P_DZ_mix_H_loss/P_DZ,
            'P Mix_no_Hloss/P Pure DZ06': P_DZ_mix_noH_loss/P_DZ,
            'rho_mix_calc_Hloss': rho_orig_H_loss,
            'rho_mix_calc_noHloss': rho_orig_no_H_loss,
            'CO2_dens_gcm3': rho_meas,
            'T_K': T_K,
            'XH2O': XH2O})

        return df


def calculate_rho_for_P_T_H2O(P_kbar, T_K):
    """ This function calculates H2O density in g/cm3 for a known Pressure (in kbar), a known T (in K) using the Wanger and Pru (2002) EOS from CoolProp
    doi:10.1063/1.1461829.

    Parameters
    ---------------------
    P_kbar: int, float, pd.Series, np.array
        Pressure in kbar

    T_K: int, float, pd.Series, np.array
        Temperature in Kelvin

    Returns
    --------------------
    pd.Series
        H2O density in g/cm3

    """
# Convert inputs to numpy arrays if they are not already
    if isinstance(P_kbar, (int, float)):
        P_kbar = np.array([P_kbar])
    elif isinstance(P_kbar, (pd.Series, list)):
        P_kbar = np.array(P_kbar)

    if isinstance(T_K, (int, float)):
        T_K = np.array([T_K])
    elif isinstance(T_K, (pd.Series, list)):
        T_K = np.array(T_K)

    # Ensure both arrays are the same shape
    P_kbar, T_K = np.broadcast_arrays(P_kbar, T_K)

    P_Pa = P_kbar * 10**8

    try:
        import CoolProp.CoolProp as cp
    except ImportError:
        raise RuntimeError('You havent installed CoolProp, which is required to convert FI densities to pressures. If you have python through conda, run conda install -c conda-forge coolprop in your command line')

    H2O_dens_gcm3 = np.zeros_like(P_kbar, dtype=float)

    non_zero_indices = P_kbar != 0
    if np.any(non_zero_indices):
        H2O_dens_gcm3[non_zero_indices] = cp.PropsSI('D', 'P', P_Pa[non_zero_indices], 'T', T_K[non_zero_indices], 'H2O') / 1000

    return pd.Series(H2O_dens_gcm3)















