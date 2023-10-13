import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.optimize import newton
import warnings

from DiadFit.density_depth_crustal_profiles import *
from DiadFit.CO2_EOS import *


## Functions to find P when the user chooses to start with a depth. It requires input of a crustal model

class config_crustalmodel:
    """
    A configuration class for specifying parameters of the crustal model. 

    Attributes:
    - crust_dens_kgm3 (float): The density of the crust in kilograms per cubic meter (kg/m^3).
    - d1 (float): The depth boundary for the first layer in kilometers (km).
    - d2 (float): The depth boundary for the second layer in kilometers (km).
    - rho1 (float): The density of the first layer in kilograms per cubic meter (kg/m^3).
    - rho2 (float): The density of the second layer in kilograms per cubic meter (kg/m^3).
    - rho3 (float): The density of the third layer in kilograms per cubic meter (kg/m^3).
    - model (str): The name of the model used for crustal calculations.
    """
    def __init__(self, crust_dens_kgm3=None,
                 d1=None, d2=None, rho1=None, rho2=None, rho3=None, model=None):
        self.crust_dens_kgm3 = crust_dens_kgm3
        self.d1 = d1
        self.d2 = d2
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.model = model


def objective_function_depth(P_kbar, target_depth, crust_dens_kgm3,
                            d1, d2, rho1, rho2, rho3, model):
    """
    Calculate the difference between the current depth and the target depth
    given pressure (P_kbar) and other parameters.

    Parameters:
    - P_kbar (float): The pressure in kilobars (kbar) to be used in the depth calculation.
    - target_depth (float): The desired depth in kilometers (km).
    - crust_dens_kgm3 (float): The density of the crust in kilograms per cubic meter (kg/m^3).
    - d1, d2 (float): Depth boundaries for different layers (km).
    - rho1, rho2, rho3 (float): Densities for different layers (kg/m^3).
    - model (str): The name of the model used for the depth calculation.

    Returns:
    - float: The difference between the current depth and the target depth.
    """

    current_depth = convert_pressure_to_depth(P_kbar=P_kbar, crust_dens_kgm3=crust_dens_kgm3, g=9.81,
                                              d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3, model=model)[0]

    return current_depth - target_depth


def find_P_for_kmdepth(target_depth, config=config_crustalmodel(), initial_P_guess=0, tolerance=0.1):
    """
    Approximate the pressure (P_kbar) based on the target depth using the Newton-Raphson method.

    Parameters:
    - target_depth (float, Pandas Series, list): The desired depth(s) in kilometers (km).
    - P_kbar (float, optional): Initial guess for the pressure in kilobars (kbar). Default is None.
    - crust_dens_kgm3 (float, optional): The density of the crust in kilograms per cubic meter (kg/m^3). Default is None.
    - d1, d2 (float, optional): Depth boundaries for different layers (km). Default is None.
    - rho1, rho2, rho3 (float, optional): Densities for different layers (kg/m^3). Default is None.
    - model (str, optional): The name of the model used for the depth calculation. Default is None.
    - tolerance (float, optional): How close the pressure estimate should be to the true value. Default is 0.1.

    Returns:
    - float or Pandas Series or list: The estimated pressure(s) (P_kbar) that correspond to the target depth(s).
    """

    if isinstance(target_depth, (float, int)):
        target_depth = [target_depth]  

    pressures = []

    for depth in target_depth:
        if all(v is None for v in [config.crust_dens_kgm3, config.d1, config.d2, config.rho1, config.rho2, config.rho3, config.model]):
            config.crust_dens_kgm3 = 2750
            warning_message = "\033[91mNo crustal parameters were provided, setting crust_dens_kgm3 to 2750. \nPlease use config_crustalmodel(...) to set your desired crustal model parameters.\033[0m"
            warnings.simplefilter("always")
            warnings.warn(warning_message, Warning, stacklevel=2)
        
        # Use the Newton-Raphson method for each target depth
        pressure = newton(objective_function_depth, initial_P_guess, args=(depth, config.crust_dens_kgm3, config.d1, config.d2, config.rho1, config.rho2, config.rho3, config.model), tol=tolerance)
        pressures.append(pressure)

    if isinstance(target_depth, (float, int)):
        return pressures[0]
    elif isinstance(target_depth, pd.Series):
        return pd.Series(pressures)
    else:
        return pressures

## Auxilliary functions for the stretching models

# Calculate decompression steps for polybaric model (Pressure, Depth, dt)

def calculate_DPdt(ascent_rate_ms,config=config_crustalmodel(),D_initial=None,D_final=None,D_step=100,initial_P_guess=0, tolerance=0.001):
    """
    Calculate the decompression rate (DP/dt) during ascent.

    Parameters:
    - ascent_rate_ms (float): Ascent rate in meters per second.
    - D_initial (float): Initial depth in kilometers. Default is 30 km.
    - D_final (float): Final depth in kilometers. Default is 0 km.
    - D_step (int): Number of depth steps for calculation. Default is 100.

    Returns:
    - D (pd.Series): Depth values in kilometers.
    - Pexternal_steps (list): Lithostatic pressure values in MPa at each depth step.
    - dt (float): Time step for the integration.
    """

    if D_initial is None or D_final is None or D_initial <= D_final:
        raise ValueError("Both D_initial and D_final must be provided, and D_initial must be larger than D_final")
    if D_initial>30 and D_step <= 80 and ascent_rate_ms <= 0.02:
        raise Warning("Your D_step is too small, the minimum recommended for ascent rates below 0.02 m/s is 80")
    D = pd.Series(list(np.linspace(D_initial, D_final, D_step)))  # km

    Pexternal_steps=find_P_for_kmdepth(D, config=config, initial_P_guess=initial_P_guess, tolerance=tolerance)
    Pexternal_steps_MPa=Pexternal_steps*100

    # Time steps of the ascent
    ascent_rate = ascent_rate_ms / 1000  # km/s
    D_change = abs(D.diff())
    time_series = D_change / ascent_rate  # calculates the time in between each step based on ascent rate
    dt = time_series.max()  # this sets the time step for the iterations later

    return D, Pexternal_steps_MPa, dt

# Olivine creep constants
class power_creep_law_constants:
    """
    Olivine power-law creep constants used in the stretching model (Wanamaker and Evans, 1989).

    Attributes:
    - A (float): Creep law constant A (default: 3.9e3).
    - n (float): Creep law constant n (default: 3.6).
    - Q (float): Activation energy for dislocation motions in J/mol (default: 523000).
    - IgasR (float): Gas constant in J/(mol*K) (default: 8.314).
    """
    def __init__(self):
        self.A = 3.9*10**3 #7.0 * 10**4
        self.n = 3.6 #3
        self.Q = 523000 # 520 Activation energy for dislocation motions in J/mol
        self.IgasR= 8.314  # Gas constant in J/(mol*K)

# Helper function to calculate change in radius over time (dR/dt)
def calculate_dR_dt(*,R, b, T,  Pinternal, Pexternal):
    """
    Calculate the rate of change of inclusion radius (dR/dt) based on power law creep.

    Parameters:
    - R (float): Inclusion radius in m.
    - b (float): Distance to the crystal defect structures, Wanamaker and Evans (1989) use R/b=1/1000
    - T (float): Temperature in Kelvin.
    - Pinternal (float): Internal pressure in MPa.
    - Pexternal (float): External pressure in MPa.

    Returns:
    - dR_dt (float): Rate of change of inclusion radius in m/s.
    """
    pl_Cs = power_creep_law_constants()
    if Pinternal<Pexternal==True:
        S=-1
    else:
        S=1
    try:
        dR_dt = 2 * (S * pl_Cs.A * math.exp(-pl_Cs.Q / (pl_Cs.IgasR * T))) * (((R * b)**3) / (((b**(3 / pl_Cs.n)) - (R**(3 / pl_Cs.n))))**pl_Cs.n) * (((3 * abs(Pinternal - Pexternal)) / (2 * pl_Cs.n))**pl_Cs.n) / R**2
        return dR_dt

    except FloatingPointError:
        return np.nan
    
# Helper function to numerically solve for R (uses Runge-Kutta method, orders 1-4)   
def get_R(R,b,T,Pinternal,Pexternal,dt,method='RK1'):
    """
    Find Radius R of an FI over a time step using Runge-Kutta numerical method. 
    Options are are order 1 to 4 RK methods as RK1 (Euler), RK2 (Huen), RK3 or RK4.

    Parameters:
    R (float): FI Radius in m
    b (float): distance to defect 
    T (float): temperature in K
    Pinternal (float): Internal pressure in MPa
    Pexternal (float): External pressure in MPa
    dt (float): The time step for integration.
    method (str, optional): The numerical integration method to use. Default is 'RK1'.

    Returns:
    tuple: A tuple containing the updated value of R and the derivative dR_dt.
    """
    if method == 'RK1'or 'Euler':
        k1 = dt * calculate_dR_dt(R=R, b=b, T=T, Pinternal=Pinternal, Pexternal=Pexternal)
        dR=k1
        dR_dt = dR / dt
        R += dR
    elif method == 'RK2' or 'Heun':
        k1 = dt * calculate_dR_dt(R=R, b=b, T=T, Pinternal=Pinternal, Pexternal=Pexternal)
        k2 = dt * calculate_dR_dt(R=R + 0.5 * k1, b=b, T=T, Pinternal=Pinternal, Pexternal=Pexternal)
        dR = ((k1 + k2) / 2)
        dR_dt = dR / dt
        R += dR
    elif method == 'RK3':
        k1 = dt * calculate_dR_dt(R=R, b=b, T=T, Pinternal=Pinternal, Pexternal=Pexternal)
        k2 = dt * calculate_dR_dt(R=R + 0.5 * k1, b=b, T=T, Pinternal=Pinternal, Pexternal=Pexternal)
        k3 = dt * calculate_dR_dt(R=R + 0.5 * k2, b=b, T=T, Pinternal=Pinternal, Pexternal=Pexternal)
        dR = ((k1 + 4 * k2 + k3) / 6)
        dR_dt = dR / dt
        R += dR
    elif method == 'RK4':
        k1 = dt * calculate_dR_dt(R=R, b=b, T=T, Pinternal=Pinternal, Pexternal=Pexternal)
        k2 = dt * calculate_dR_dt(R=R + 0.5 * k1, b=b, T=T, Pinternal=Pinternal, Pexternal=Pexternal)
        k3 = dt * calculate_dR_dt(R=R + 0.5 * k2, b=b, T=T, Pinternal=Pinternal, Pexternal=Pexternal)
        k4 = dt * calculate_dR_dt(R=R + k3, b=b, T=T, Pinternal=Pinternal, Pexternal=Pexternal)
        dR = ((k1 + 2 * k2 + 2 * k3 + k4) / 6)
        dR_dt = dR / dt
        R += dR / 6
    else:
        raise ValueError("Unsupported numerical method. Choose from 'RK1' or 'Euler', 'RK2' or 'Huen', 'RK3', 'RK4'")

    return R, dR_dt

## Functions to calculate P, CO2dens, CO2mass and V

# Calculate initial CO2 density in g/cm3 and CO2 mass in g

def get_initial_CO2(R, T, P, EOS='SW96', return_volume=False):
    """
    Calculate the initial density and mass of CO2 inside a fluid inclusion (FI).

    Parameters:
    R (float): The radius of the fluid inclusion (FI), in meters.
    T (float): The temperature, in Kelvin.
    P (float): The pressure, in MegaPascals (MPa).
    EOS (str, optional): The equation of state (EOS) to use for density calculations.
        Can be one of: 'ideal' (ideal gas), 'SW96' (Soave-Redlich-Kwong EOS 1996), or 'SP94' (Sutton and Precious EOS 1994).
        Defaults to 'SW96'.
    return_volume (bool, optional): Whether to return the volume of the FI along with density and mass. Defaults to False.

    Returns:
    tuple or float: If return_volume is True, returns a tuple containing (V, CO2_dens_initial, CO2_mass_initial), where:
    - V (float): The volume of the fluid inclusion (FI), in cubic meters (m³).
    - CO2_dens_initial (float): The initial density of CO2 within the FI, in grams per cubic centimeter (g/cm³).
    - CO2_mass_initial (float): The initial mass of CO2 within the FI, in grams (g).

    If return_volume is False, returns a tuple containing (CO2_dens_initial, CO2_mass_initial).

    Raises:
    ValueError: If an unsupported EOS is specified.
    """

    valid_EOS = ['ideal', 'SW96', 'SP94']

    try:
        if EOS not in valid_EOS:
            raise ValueError("EOS can only be 'ideal', 'SW96', or 'SP94'")
        
        if EOS == 'ideal':
            R_gas = 8.314  # J.mol/K J: kg·m²/s²
            V = 4/3 * math.pi * R**3  # m3
            P = P * 10**6  # convert MPa to Pa
            M = 44.01 / 1000  # kg/mol

            CO2_mass_kg = P * V * M / (R_gas * T)  # CO2 mass in kg
            rho = (CO2_mass_kg / V)  # rho in kg/m3

            CO2_dens_initial = rho / 1000  # CO2 density in g/cm3
            CO2_mass_initial = CO2_mass_kg / 1000  # CO2 mass in g

        else:
            R = R * 10**2  # radius in cm
            V = 4/3 * math.pi * R**3  # cm3, Volume of the FI, assume sphere
            P_kbar = P / 100  # Internal pressure of the FI, convert to kbar

            CO2_dens_initial = calculate_rho_for_P_T(EOS=EOS, P_kbar=P_kbar, T_K=T)[0]  # CO2 density in g/cm3
            CO2_mass_initial = CO2_dens_initial * V  # CO2 mass in g

        if return_volume:
            return V, CO2_dens_initial, CO2_mass_initial
        else:
            return CO2_dens_initial, CO2_mass_initial

    except ValueError as ve:
        raise ve

# Calculate CO2 density in g/cm3 and P in MPa for fixed CO2 mass in g

def get_CO2dens_P(R,T,CO2_mass,EOS='SW96',return_volume=False):
    """
    Calculate the density and pressure of CO2 inside a fluid inclusion (FI).

    Parameters:
    R (float): The radius of the fluid inclusion (FI), in meters.
    T (float): The temperature, in Kelvin.
    CO2_mass (float): The mass of CO2 within the FI, in grams (g).
    EOS (str, optional): The equation of state (EOS) to use for density calculations.
        Can be one of: 'ideal' (ideal gas), 'SW96' (Soave-Redlich-Kwong EOS 1996), or 'SP94' (Sutton and Precious EOS 1994).
        Defaults to 'SW96'.
    return_volume (bool, optional): Whether to return the volume of the FI along with density and pressure. Defaults to False.

    Returns:
    tuple or float: If return_volume is True, returns a tuple containing (V, CO2_dens, P), where:
    - V (float): The volume of the fluid inclusion (FI), in cubic meters (m³).
    - CO2_dens (float): The density of CO2 within the FI, in grams per cubic centimeter (g/cm³).
    - P (float): The pressure of CO2 within the FI, in MegaPascals (MPa).

    If return_volume is False, returns a tuple containing (CO2_dens, P).

    Raises:
    ValueError: If an unsupported EOS is specified or if the EOS calculation fails.
    """
    valid_EOS = ['ideal', 'SW96', 'SP94']

    try:
        if EOS not in valid_EOS:
            raise ValueError("EOS can only be 'ideal', 'SW96', or 'SP94'")
        
        if EOS == 'ideal':
            R_gas = 8.314  # J.mol/K J: kg·m²/s²
            V = 4/3 * math.pi * R**3  # m3
            M = 44.01 / 1000  # kg/mol

            CO2_mass_kg=CO2_mass*1000
            P=CO2_mass_kg*R_gas*T/(M*V) #P in Pa
            CO2_dens=(CO2_mass_kg/V) # CO2 density in kg/m3

            P=P/(10**6) #P in MPa
            CO2_dens=CO2_dens/1000 #rho in g/cm3

        else:
            R=R*10**2 #FI radius, convert to cm
            V=4/3*math.pi*R**3 #cm3, Volume of the FI, assume sphere

            CO2_dens=CO2_mass/V # CO2 density in g/cm3

            try:
                P=calculate_P_for_rho_T(EOS=EOS,CO2_dens_gcm3=CO2_dens, T_K=T)['P_MPa'][0] #g/cm3, CO2 density

            except ValueError:
                P=np.nan
    
        if return_volume:
            return V, CO2_dens, P
        else:
            return CO2_dens,P

    except ValueError as ve:
        raise ve
    
## Stretching Models

# This function is to model FI stretching during decompression and ascent 
def stretch_in_ascent(*, R, b, T, ascent_rate_ms, depth_path_ini_fin_step=[100, 0, 100],
                      crustal_model_config=config_crustalmodel(crust_dens_kgm3=2750),
                      EOS, plotfig=True, report_results='fullpath',
                      initial_P_guess=0, tolerance=0.001,method='RK4',update_b=False):
    """
    Simulate the stretching of a CO2 fluid inclusion (FI) during ascent through the Earth's crust.

    Parameters:
    R (float): The initial radius of the fluid inclusion (FI), in meters.
    b (float): The initial distance to the crystal rim from the FI center, in meters.
    T (float): The temperature, in Kelvin.
    ascent_rate_ms (float): The ascent rate of the FI, in meters per second (m/s).
    depth_path_ini_fin_step (list, optional): A list containing [initial_depth_km, final_depth_km, depth_step].
        Defaults to [100, 0, 100], representing the depth path from initial to final depth with a step size.
    crustal_model_config (dict, optional): Configuration parameters for the crustal model.
        Defaults to a predefined configuration with a crustal density of 2750 kg/m³.
    EOS (str): The equation of state (EOS) to use for density calculations.
        Can be one of: 'ideal' (ideal gas), 'SW96' (Soave-Redlich-Kwong EOS 1996), or 'SP94' (Sutton and Precious EOS 1994).
    plotfig (bool, optional): Whether to plot figures showing the changes in depth and CO2 density. Defaults to True.
    report_results (str, optional): The type of results to report. Can be 'fullpath', 'startendonly', or 'endonly'.
        Defaults to 'fullpath'.
    initial_P_guess (float, optional): Initial guess for internal pressure (Pinternal) in MPa. Defaults to 0.
    tolerance (float, optional): Tolerance for pressure calculations. Defaults to 0.001.
    method (str, optional): The numerical integration method to use for updating the FI. Can be 'RK1', 'RK2', 'RK3', or 'RK4'.
        Defaults to 'RK4'.
    update_b (bool, optional): Whether to update 'b' during the ascent. Defaults to False.

    Returns:
    pandas.DataFrame: A DataFrame containing the simulation results, including time, depth, pressure, radius changes, and CO2 density.

    Raises:
    ValueError: If an unsupported EOS is specified.
    """

    D, Pexternal_steps, dt = calculate_DPdt(ascent_rate_ms=ascent_rate_ms, config=crustal_model_config,
                                            D_initial=depth_path_ini_fin_step[0], D_final=depth_path_ini_fin_step[1],
                                            D_step=depth_path_ini_fin_step[2],
                                            initial_P_guess=initial_P_guess, tolerance=tolerance)
    Pinternal = Pexternal_steps[0]

    CO2_dens_initial, CO2_mass_initial = get_initial_CO2(R=R, T=T, P=Pinternal, EOS=EOS)

    
    results = pd.DataFrame([{'Time(s)': 0,
                            'Step':0,
                            'dt(s)':0,
                        'Pexternal(MPa)': Pinternal,
                        'Pinternal(MPa)': Pinternal,
                        'dR/dt(m/s)': calculate_dR_dt(R=R, b=b, Pinternal=Pinternal, Pexternal=Pinternal, T=T),
                        'Fi_radius(μm)': R*10**6,
                        'b (distance to xtal rim -μm)':b*10**6,
                        '\u0394R/R0 (fractional change in radius)':np.nan,
                        'CO2_dens_gcm3': CO2_dens_initial,
                        'Depth(km)':D.iloc[0]}], index=range(len(Pexternal_steps)))

   
    for i in range(1,len(Pexternal_steps)):
        
        Pexternal = Pexternal_steps[i]

        R,dR_dt = get_R(R=R,b=b,T=T,Pinternal=Pinternal,Pexternal=Pexternal,dt=dt,method=method)
        CO2_dens_new,P_new = get_CO2dens_P(R=R,T=T,CO2_mass=CO2_mass_initial,EOS=EOS)

        Pinternal = P_new

        if update_b==True:
            b=1000*R
        
        results.loc[i] = [dt*i, i, dt, Pexternal, Pinternal, dR_dt, R * 10 ** 6, b * 10 ** 6,
                    (R * 10 ** 6 - results.loc[0, 'Fi_radius(μm)']) / results.loc[0, 'Fi_radius(μm)'],
                    CO2_dens_new, D.iloc[i]]

    if report_results == 'startendonly':
        results.drop(index=list(range(1, results.shape[0] - 1)), inplace=True)  # Drop all rows except first and last

    if report_results == 'endonly':
        results.drop(index=list(range(0, results.shape[0] - 1)), inplace=True)  # Drop all rows except last

    if plotfig==True:
        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10,3))
        ax0.plot(-results['Depth(km)'],results['\u0394R/R0 (fractional change in radius)'],marker='s',label=f"Ascent Rate = {ascent_rate_ms} m/s")
        ax0.set_xlabel("Depth")
        ax0.set_ylabel('\u0394R/R0 (fractional change in radius)')

        ax1.plot(-results['Depth(km)'],results['CO2_dens_gcm3'],marker='s',label=f"Ascent Rate = {ascent_rate_ms} m/s")
        ax1.set_xlabel("Depth")
        ax1.set_ylabel("CO2_density_gmL")
        ax0.legend(loc='best')
        ax1.legend(loc='best')
        plt.show()

    return results

# This function is to model stretching at fixed External Pressure (e.g., during stalling or upon eruption)
def stretch_at_constant_Pext(*,R,b,T,EOS='SW96',Pinternal,Pexternal,totaltime,steps,method='Euler',report_results='fullpath',plotfig=False,update_b=False):
    """
    Simulate the stretching of a CO2 fluid inclusion (FI) under constant external pressure (i.e., quenching or storage).

    Parameters:
    R (float): The initial radius of the fluid inclusion (FI), in meters.
    b (float): The initial distance to the crystal rim from the FI center, in meters.
    T (float): The temperature, in Kelvin.
    EOS (str, optional): The equation of state (EOS) to use for density calculations.
        Can be one of: 'ideal' (ideal gas), 'SW96' (Soave-Redlich-Kwong EOS 1996), or 'SP94' (Sutton and Precious EOS 1994).
        Defaults to 'SW96'.
    Pinternal (float): The initial internal pressure of the FI, in MegaPascals (MPa).
    Pexternal (float): The constant external pressure applied to the FI, in MegaPascals (MPa).
    totaltime (float): The total simulation time, in seconds.
    steps (int): The number of simulation steps.
    method (str, optional): The numerical integration method to use for updating the FI. Can be 'Euler' or 'RK4'.
        Defaults to 'Euler'.
    report_results (str, optional): The type of results to report. Can be 'fullpath', 'startendonly', or 'endonly'.
        Defaults to 'fullpath'.
    plotfig (bool, optional): Whether to plot figures showing the changes in time, radius, and CO2 density. Defaults to False.
    update_b (bool, optional): Whether to update 'b' during the simulation. Defaults to False.

    Returns:
    pandas.DataFrame: A DataFrame containing the simulation results, including time, pressure, radius changes, and CO2 density.

    Raises:
    ValueError: If an unsupported EOS is specified.
    """

    CO2_dens_initial,CO2_mass_initial=get_initial_CO2(R=R,T=T,P=Pinternal,EOS=EOS)

    results = pd.DataFrame([{'Time(s)': 0,
                             'Step':0,
                             'dt(s)':0,
                            'Pexternal(MPa)': Pexternal,
                            'Pinternal(MPa)': Pinternal,
                            'dR/dt(m/s)': calculate_dR_dt(R=R, b=b, Pinternal=Pinternal, Pexternal=Pexternal, T=T),
                            'Fi_radius(μm)': R*10**6,
                            'b (distance to xtal rim -μm)':b*10**6,
                            '\u0394R/R0 (fractional change in radius)':0,
                            'CO2_dens_gcm3': CO2_dens_initial}], index=range(steps))

    dt=totaltime/steps
    
    for step in range(1,steps):

        R_new,dR_dt = get_R(R=R,b=b,T=T,Pinternal=Pinternal,Pexternal=Pexternal,dt=dt,method=method)

        CO2_dens_new,P_new = get_CO2dens_P(R=R_new,T=T,CO2_mass=CO2_mass_initial,EOS=EOS)

        Pinternal = P_new
        R=R_new

        if update_b==True:
            b=1000*R
        
        results.loc[step] = [step * dt, step, dt, Pexternal, Pinternal, dR_dt, R * 10 ** 6, b * 10 ** 6,
                            (R * 10 ** 6 - results.loc[0, 'Fi_radius(μm)']) / results.loc[0, 'Fi_radius(μm)'],
                                CO2_dens_new]

    if report_results == 'startendonly':
        results.drop(index=list(range(1, results.shape[0] - 1)), inplace=True)  # Drop all rows except first and last

    if report_results == 'endonly':
        results.drop(index=list(range(0, results.shape[0] - 1)), inplace=True)  # Drop all rows except last

    if plotfig==True:
        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10,3))
        ax0.plot(results['Time(s)'],results['\u0394R/R0 (fractional change in radius)'],marker='s')
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("\u0394R/R0 (fractional change in radius)")

        ax1.plot(results['Time(s)'],results['CO2_dens_gcm3'],marker='s')
        ax1.set_xlabel("Time(s)")
        ax1.set_ylabel("CO2_density_gmL")
        plt.show()

    return results

# This function can loop through different R and b value sets using stretch at constant Pext

def loop_R_b_constant_Pext(*,R_values, b_values, T, EOS, Pinternal, Pexternal, totaltime, steps, T4endcalc_PD, method='Euler',
                            plotfig=False, config=config_crustalmodel(crust_dens_kgm3=2750)):
    """
    Perform a looped simulation with varying R and b values under constant external pressure.

    Parameters:
    R_values (list): A list of initial radius values for fluid inclusions (FI), in meters.
    b_values (list): A list of initial distance values to the crystal rim from the FI center, in meters.
    T (float): The temperature, in Kelvin.
    EOS (str): The equation of state (EOS) to use for density calculations.
    Pinternal (float): The initial internal pressure of the FI, in MegaPascals (MPa).
    Pexternal (float): The constant external pressure applied to the FI, in MegaPascals (MPa).
    totaltime (float): The total simulation time, in seconds.
    steps (int): The number of simulation steps.
    T4endcalc_PD (float): The temperature at which to calculate the depths (Kelvin) during post-decompression calculations.
    method (str, optional): The numerical integration method to use for updating the FI. Can be 'Euler' or 'RK4'.
        Defaults to 'Euler'.
    plotfig (bool, optional): Whether to plot figures showing the changes in time, radius, and CO2 density. Defaults to False.
    config (dict, optional): Configuration parameters for the crustal model. Defaults to a predefined configuration with a crustal density of 2750 kg/m³.

    Returns:
    dict: A dictionary containing simulation results for varying R and b values. The keys are in the format 'R{index}_b{index}',
          where 'index' corresponds to the index of R_values and b_values, respectively. The values are DataFrames with simulation results.
    """

    results_dict = {}
    for idx_R, R in enumerate(R_values):  # Use enumerate to get the index of R_values
        R_key = f'R{idx_R}'  # Use 'R' followed by the index
        results_dict[R_key] = {}

        for idx_b, b in enumerate(b_values):  # Use enumerate to get the index of b_values
            b_key = f'b{idx_b}'  # Use 'b' followed by the index
            results = stretch_at_constant_Pext(R=R, b=b, T=T, Pinternal=Pinternal, Pexternal=Pexternal,
                                              totaltime=totaltime, steps=steps, EOS=EOS, method=method,
                                              plotfig=plotfig)
            results['Calculated depths (km)_StorageT'] = convert_pressure_to_depth(
                P_kbar=results['Pinternal(MPa)'] / 100,
                crust_dens_kgm3=config.crust_dens_kgm3, g=9.81,
                d1=config.d1, d2=config.d2,
                rho1=config.rho1, rho2=config.rho2, rho3=config.rho3,
                model=config.model)
            results['Calculated P from rho (MPa)_TrappingT'] = calculate_P_for_rho_T(
                EOS='SW96', CO2_dens_gcm3=results['CO2_dens_gcm3'], T_K=T4endcalc_PD + 273.15)['P_MPa']

            results['Calculated depths (km)_TrappingT'] = convert_pressure_to_depth(
                P_kbar=results['Calculated P from rho (MPa)_TrappingT'] / 100,
                crust_dens_kgm3=config.crust_dens_kgm3, g=9.81,
                d1=config.d1, d2=config.d2,
                rho1=config.rho1, rho2=config.rho2, rho3=config.rho3,
                model=config.model)

            results_dict[R_key][b_key] = results

    return results_dict
