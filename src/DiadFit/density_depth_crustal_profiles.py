import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inspect



def rasmussen(P_kbar):
    """ Calculates Depth for a given pressure using a 4th degree fit to the supporting information of Rasmussen et al. 2022,
    overall best fit density vs. depth. Above 5.24 kbar, returns Nan

    Parameters
    -------------
    P_kbar: int, float, pd.series
        Pressure in kbar

    Returns
    -------------
    Depth in km (same datatype as input)

    """
    P=P_kbar
    if P<5.2474630296099205:
        b=-0.0025915704129682504
        c=0.037299399171806996
        d=-0.21276925206828018
        e=4.273285349609203
        f=0.00430844920014402
        D= b*(P**4) + c*(P**3) + d*(P**2) + e*P + f
    else:
        D=np.nan
    return D

def hill_zucca(P_kbar):

    """Calculates Depth for a given pressure using the parameterization of Hill and Zucca (1987),
    as given in Putirka (2017) Down the Crater Elements supplement for Hawaii

    Parameters
    -------------
    P_kbar: int, float, pd.series
        Pressure in kbar

    Returns
    -------------
    Depth in km (same datatype as input)


    """
    P=P_kbar

    D=-2.77*10**(-5) * (P**4) - 2.0*10**(-3)*(P**3) - 4.88*10**(-2)*P**2 + 3.6*P - 6.34*(10**(-2))

    return D

def ryan_lerner(P_kbar):
    """ Calculates depth for a given pressure using the Parameterization of Ryan 1987, actual equation from Lerner et al. 2021
    After 16.88 km (455 MPa), returns NaN

    Parameters
    -------------
    P_kbar: int, float, pd.series
        Pressure in kbar

    Returns
    -------------
    Depth in km (same datatype as input)


    """
    P=P_kbar*100
    if P<455.09090909:
        D=(4.578*10**(-8) *P**3) - (4.151*10**(-5) *P**2) + (4.652*10**(-2) *P)
    else:
        D=np.nan

    return D

def mavko_debari(P_kbar):
    """ Calculates depth for a given pressure using the parameterization of Mavko and Thompson (1983) and DeBari and Greene (2011)
    as given in Putirka (2017) Down the Crater Elements supplement, used for Cascades

    Parameters
    -------------
    P_kbar: int, float, pd.series
        Pressure in kbar

    Returns
    -------------
    Depth in km (same datatype as input)


    """
    P=P_kbar
    D=0.4853881 + 3.6006116*P - 0.0117368*(P-1.3822)**2


    return D


def prezzi(P_kbar):
    """Calculates depth for a given pressure using the parameterization of Prezzi et al. (2009),
    as given in Putirka (2017) Down the Crater Elements supplement.
    Used for Andes.

    Parameters
    -------------
    P_kbar: int, float, pd.series
        Pressure in kbar

    Returns
    -------------
    Depth in km (same datatype as input)


    """
    P=P_kbar
    D=4.88 + 3.30*P - 0.0137*(P - 18.01)**2

    return D


def prezzi(P_kbar):
    """Calculates depth for a given pressure using the parameterization of Prezzi et al. (2009),
    as given in Putirka (2017) Down the Crater Elements supplement.
    Used for Andes.

    Parameters
    -------------
    P_kbar: int, float, pd.series
        Pressure in kbar

    Returns
    -------------
    Depth in km (same datatype as input)



    """
    P=P_kbar
    D=4.88 + 3.30*P - 0.0137*(P - 18.01)**2

    return D


Profile_funcs={ryan_lerner, mavko_debari, hill_zucca, prezzi, rasmussen}
Profile_funcs_by_name= {p.__name__: p for p in Profile_funcs}


## Two and three step profiles
def convert_pressure_depth_2step(P_kbar=None, d1=None, rho1=None, rho2=None, g=9.81):
    """ Converts Pressure to depth using a 2 step profile for int or float

    Parameters
    --------------
    P_kbar: int or float
        Pressure in kbar

    d1: int or float
        depth in km of 1st step transition in density

    rho1: int or float
        Density (kg/m3) down to step transition

    rho2: int or float
        Density (kg/m3) below step transition

    g: float
        gravitational constant

    Returns
    -------------
    float
        Depth in km

    """

    d1_SI=d1*1000
    P_step1=(g*rho1*d1_SI)/100000000
    # print('Pressure Moho in kbar')
    # print(P_Moho)
    if P_kbar<P_step1:
        depth_km=10**(-3)*((P_kbar*100000000))/(g*rho1)
    if P_kbar>=P_step1:
        P_belowstep1=P_kbar-P_step1
        # print('P below  Moho')
        # print(P_belowMoho)
        depth_km_bm=10**(-3)*((P_belowstep1*100000000)/(g*rho2))
        depth_km=d1+depth_km_bm
    if np.isnan(P_kbar):
        depth_km=np.nan

    return depth_km

def loop_pressure_depth_2step(P_kbar=None, d1=14, rho1=2800, rho2=3100, g=9.81):

    """ Converts Pressure to depth using a 2 step profile for a pandas.Series of presssures

    Parameters
    --------------
    P_kbar: pd.Series
        Pressure in kbar

    d1: int or float
        depth in km of 1st step transition in density

    rho1: int or float
        Density (kg/m3) down to step transition

    rho2: int or float
        Density (kg/m3) below step transition

    g: float
        gravitational constant

    Returns
    -------------
    pd.Series
        Depth in km

    """
    if type(P_kbar) is int or type(P_kbar) is float:
        depth_km_loop=convert_pressure_depth_2step(P_kbar,
            d1=d1, rho1=rho1, rho2=rho2, g=g)
    else:
        depth_km_loop=np.empty(len(P_kbar))
        for i in range(0, len(P_kbar)):
            depth_km_loop[i]=convert_pressure_depth_2step(P_kbar[i],
            d1=d1, rho1=rho1, rho2=rho2, g=g)
    return depth_km_loop


def convert_pressure_depth_3step(P_kbar=None, d1=5, d2=14, g=9.81,
                                 rho1=2700, rho2=3000, rho3=3100):
    """ Converts Pressure to depth using a 3 step profile for int or float

    Parameters
    --------------
    P_kbar: int or float
        Pressure in kbar

    d1: int or float
        depth in km of 1st step transition in density

    d2: int or float
        depth in km of 2nd step transition in density

    rho1: int or float
        Density (kg/m3) down to first step transition

    rho2: int or float
        Density (kg/m3) between first and second step transition


    rho3: int or float
        Density (kg/m3) below 2nd step transition

    g: float
        gravitational constant

    Returns
    -------------
    float
        Depth in km

    """

    d1_SI=d1*1000
    d2_SI=d2*1000
    P_Step1=(g*rho1*d1_SI)/100000000
    P_Step2=P_Step1+(g*(d2_SI-d1_SI)*rho2)/100000000
    # print('Pressure Moho in kbar')
    # print(P_Moho)
    if P_kbar<P_Step1:
        depth_km=10**(-3)*((P_kbar*100000000))/(g*rho1)
    if P_kbar>=P_Step1 and P_kbar<P_Step2:
        P_belowStep2=P_kbar-P_Step1
        # print('P below  Moho')
        # print(P_belowMoho)
        depth_km_bm=10**(-3)*((P_belowStep2*100000000)/(g*rho2))
        depth_km=d1+depth_km_bm
    if P_kbar>=P_Step2:
        P_belowstep2=P_kbar-P_Step2
        # print('P below  Moho')
        # print(P_belowMoho)
        depth_km_bm=10**(-3)*((P_belowstep2*100000000)/(g*rho3))
        depth_km=d2+depth_km_bm

    if np.isnan(P_kbar):
        depth_km=np.nan

    return depth_km

def loop_pressure_depth_3step(P_kbar=None,  d1=5, d2=14,
                                 rho1=2700, rho2=3000, rho3=3100, g=9.81):

    """ Converts Pressure to depth using a 3 step profile for pd.Series

    Parameters
    --------------
    P_kbar: pd.Series
        Pressure in kbar

    d1: int or float
        depth in km of 1st step transition in density

    d2: int or float
        depth in km of 2nd step transition in density

    rho1: int or float
        Density (kg/m3) down to first step transition

    rho2: int or float
        Density (kg/m3) between first and second step transition


    rho3: int or float
        Density (kg/m3) below 2nd step transition

    Returns
    -------------
    pd.Series
        Depth in km

    """
    if type(P_kbar) is int or type(P_kbar) is float:
        depth_km_loop=convert_pressure_depth_3step(P_kbar,
            d1=d1, rho1=rho1, rho2=rho2, g=g)
    else:

        depth_km_loop=np.empty(len(P_kbar))
        for i in range(0, len(P_kbar)):
            depth_km_loop[i]=convert_pressure_depth_3step(P_kbar[i],
            d1=d1, d2=d2,rho1=rho1, rho2=rho2, rho3=rho3, g=g)
    return depth_km_loop



def convert_pressure_to_depth(P_kbar=None, crust_dens_kgm3=None, g=9.81,
d1=None, d2=None,rho1=None, rho2=None, rho3=None, model=None):
    """ Converts pressure in kbar to depth in km using a variety of crustal density profiles
    or existing models for Pressure vs. depth


    Parameters
    -----------

    P_kbar: int, float, pd.Series, np.ndarray
        Pressure in kbar

    g: float
        gravitational constant, in m/s2

    Choose from:

    crust_dens_kgm3: float or str
        Crustal density in kg/m3

    OR

    model, choose from:

        ryan_lerner:
            Parameterization of Ryan 1987, actual equation from Lerner et al. 2021
            After 16.88 km (455 MPa), assume density is 2.746, as density turns around again. This profile is tweaked for Hawaii

        mavko_debari:
            Parameterization of Mavko and Thompson (1983) and DeBari and Greene (2011)
            as given in Putirka (2017) Down the Crater Elements supplement.


        hill_zucca:
            Parameterization of Hill and Zucca (1987),
            as given in Putirka (2017) Down the Crater Elements supplement

        prezzi:
            Parameterization of Prezzi et al. (2009),
            as given in Putirka (2017) Down the Crater Elements supplement. Tweaked for Andes.

        rasmussen:
            Linear fit to the supporting information of Rasmussen et al. 2022,
            overall best fit density vs. depth

        two-step:
            If two step, must also define:
                d1: Depth to first transition in km
                rho1: Density between surface and 1st transition
                d2: Depth to second transition in km (from surface)
                rho2: Density between 1st and 2nd transition

        three-step:
            If three step, must also define:
                d1: Depth to first transition in km
                rho1: Density between surface and 1st transition
                d2: Depth to second transition in km (from surface)
                rho2: Density between 1st and 2nd transition
                d3: Depth to third transition in km (from surface)
                rho3: Density between 2nd and 3rd transition depth.




    Returns
    -----------

    Depth in km as a panda series

    """


    # Check, is it an integer, If so just calculate depth

    if isinstance(crust_dens_kgm3, str):
        raise TypeError('Do not enter a string for crustal density, put it as a model instead')

    if crust_dens_kgm3 is not None:
        if type(crust_dens_kgm3)==int or type(crust_dens_kgm3)==float:
            D=10**5*P_kbar/(9.81*crust_dens_kgm3)
            model=None


        else:
            # Check if its a pandas series.
            if isinstance(crust_dens_kgm3, pd.Series):
                # Now check, is it a series of strings, or
                if type(crust_dens_kgm3[0])==str:
                    model=crust_dens_kgm3.iloc[0]
                else:
                    model=None
                    D=10**5*P_kbar/(9.81*crust_dens_kgm3)

            # Check if its just a single string
            elif type(crust_dens_kgm3)==str:
                model=crust_dens_kgm3



    elif model is not None:
        if model == "two-step":

            if d1 is None or rho1 is None or rho2 is None:
                raise Exception('You have selected the two-step model, You must enter d1 (km), rho1 and rho2 (kg/m3)')
            D=loop_pressure_depth_2step(P_kbar=P_kbar,
            d1=d1, rho1=rho1, rho2=rho2)

        if model == "three-step":
            if d1 is None or d2 is None or rho1 is None or rho2 is None or rho3 is None:
                raise Exception('You have selected the three-step model, You must enter d1 and d2 (km), rho1, rho2 and rho3 (kg/m3)')

            D=loop_pressure_depth_3step(P_kbar=P_kbar,
            d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3)

        if model !="two-step" and model != "three-step":
            try:

                func = Profile_funcs_by_name[model]
            except KeyError:
                raise ValueError(f'{model} is not a valid model') from None

            sig=inspect.signature(func)

            if isinstance(P_kbar, float) or isinstance(P_kbar, int):
                D=func(P_kbar)

            if isinstance(P_kbar, pd.Series):
                D=np.empty(len(P_kbar), float)
                for i in range(0, len(P_kbar)):
                    D[i]=func(P_kbar.iloc[i])

            if isinstance(P_kbar, np.ndarray):
                D=np.empty(len(P_kbar), float)
                for i in range(0, len(P_kbar)):
                    D[i]=func(P_kbar[i])

    else:
        raise TypeError('You need to either enter a model, or a value for crust_dens_kgm3' )


    D_series=pd.Series(D)

    if g != 9.81:
        print('You specified a g that wasnt 9.81, ive adjusted the profiles accordingly')
        D_series=D_series*(9.81/g)

    return D_series





def convert_co2_dens_press_depth_old(T_K=None,
    CO2_dens_gcm3=None,
    crust_dens_kgm3=None, output='kbar',
    g=9.81, model=None,
    d1=None, d2=None, rho1=None, rho2=None, rho3=None, EOS='SW96'):

    """ This is a now old function that isn't used, kept for backwards functionality.
    Dont use unless you have built code relying on it!
    """





    try:
        import CoolProp.CoolProp as cp
    except ImportError:
        raise RuntimeError('You havent installed CoolProp, which is required to convert FI densities to pressures. If you have python through conda, run conda install -c conda-forge coolprop in your command line')


    if type(T_K) is pd.Series:
        T_K=T_K.values
    if type(CO2_dens_gcm3) is pd.Series:
        CO2_dens_gcm3=CO2_dens_gcm3

    density_SI_units=CO2_dens_gcm3*1000
    if type(density_SI_units) is pd.Series:
        density_SI_units=density_SI_units.values


    P_Pa=cp.PropsSI('P', 'D', density_SI_units, 'T', T_K, 'CO2')
    P_kbar=P_Pa*10**(-8)
    if output=='kbar':
        return P_kbar
    if output=='MPa':
        return P_kbar*100

    if output=='df':
        if model is not None:
            Depth_km=convert_pressure_to_depth(P_kbar=P_kbar, model=model,
            d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3)


        if isinstance(CO2_dens_gcm3, np.float64) or isinstance(CO2_dens_gcm3, np.float):
            length=1
        elif isinstance(CO2_dens_gcm3, pd.Series):
            length=len(CO2_dens_gcm3)
        else:
            print(type(CO2_dens_gcm3))
        # else:
        #     if np.shape(CO2_dens_gcm3)[0]==1:
        #         length=1
        #     else:
        #         length=2

        if length==1:

            if crust_dens_kgm3 is not None:

                Depth_km=convert_pressure_to_depth(P_kbar=P_kbar, crust_dens_kgm3=crust_dens_kgm3,
                g=g, d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3)

            if rho1 is not None and rho2 is not None and rho3 is None:

                Depth_km=convert_pressure_depth_2step(P_kbar=P_kbar,
                g=g, d1=d1, rho1=rho1, rho2=rho2)

            if rho1 is not None and rho2 is not None and rho3 is not None:

                Depth_km=convert_pressure_depth_3step(P_kbar=P_kbar,
                g=g, d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3)
        else:
            if crust_dens_kgm3 is not None:

                Depth_km=convert_pressure_to_depth(P_kbar=P_kbar, crust_dens_kgm3=crust_dens_kgm3,
                g=g, d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3)

            if rho1 is not None and rho2 is not None and rho3 is None:

                Depth_km=loop_pressure_depth_2step(P_kbar=P_kbar,
                g=g, d1=d1, rho1=rho1, rho2=rho2)

            if rho1 is not None and rho2 is not None and rho3 is not None:

                Depth_km=loop_pressure_depth_3step(P_kbar=P_kbar,
                g=g, d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3)


        if type(P_kbar) is float:
        # Crustal density, using P=rho g H
            df=pd.DataFrame(data={'Pressure (kbar)': P_kbar,
                                'Pressure (MPa)': P_kbar*100,
                                'Depth (km)': Depth_km,
                                'input_crust_dens_kgm3': crust_dens_kgm3,
                                'MC_T_K': T_K,
                                'MC_CO2_dens_gcm3': CO2_dens_gcm3}, index=[0])

        else:


            df=pd.DataFrame(data={'Pressure (kbar)': P_kbar,
                                'Pressure (MPa)': P_kbar*100,
                                'Depth (km)': Depth_km,
                                'input_crust_dens_kgm3': crust_dens_kgm3,
                                'MC_T_K': T_K,
                                'MC_CO2_dens_gcm3': CO2_dens_gcm3})

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df
