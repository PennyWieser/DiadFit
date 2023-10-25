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

def objective_function_depth(P_kbar, target_depth_km, crust_dens_kgm3,
                            d1, d2, rho1, rho2, rho3, model):
    """
    Calculate the difference between the current depth and the target depth
    given pressure (P_kbar) and other parameters.

    Parameters:
    - P_kbar (float): The pressure in kilobars (kbar) to be used in the depth calculation.
    - target_depth_km (float): The desired depth in kilometers (km).
    - crust_dens_kgm3 (float): The density of the crust in kilograms per cubic meter (kg/m^3).
    - d1, d2 (float): Depth boundaries for different layers (km).
    - rho1, rho2, rho3 (float): Densities for different layers (kg/m^3).
    - model (str): The name of the model used for the depth calculation.

    Returns:
    - float: The difference between the current depth and the target depth.
    """

    current_depth = convert_pressure_to_depth(P_kbar=P_kbar, crust_dens_kgm3=crust_dens_kgm3, g=9.81,
                                              d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3, model=model)[0]

    return current_depth - target_depth_km

def find_P_for_kmdepth(target_depth_km, crustal_model_config=config_crustalmodel(), initial_P_guess_kbar=0, tolerance=0.1):
    """
    Approximate the pressure (P_kbar) based on the target depth using the Newton-Raphson method.

    Parameters:
    - target_depth_km (float, pd.Series, list): The desired depth(s) in kilometers (km).
    - initial_P_guess_kbar (float, optional): Initial guess for the pressure in kilobars (kbar). Default is 0.
    - crustal_model_config (object, optional): Configuration object containing crustal model parameters.
        - crust_dens_kgm3 (float, optional): The density of the crust in kilograms per cubic meter (kg/m^3). Default is None.
        - d1, d2 (float, optional): Depth boundaries for different layers (km). Default is None.
        - rho1, rho2, rho3 (float, optional): Densities for different layers (kg/m^3). Default is None.
        - model (str, optional): The name of the model used for depth calculation. Default is None.
    - tolerance (float, optional): Tolerance for the Newton-Raphson method. The pressure estimate should be within this tolerance of the true value. Default is 0.1.

    Returns:
    - float or pd.Series or list: The estimated pressure(s) (P_kbar) that correspond to the target depth(s).
    
    Notes:
    - If the target_depth_km is a single value, a float is returned.
    - If the target_depth_km is a Pandas Series, a Pandas Series is returned.
    - If the target_depth_km is a list, a list of floats is returned.

    If crustal parameters are not provided in the crustal_model_config object, a single step model with a crustal density = 2750 kg/cm3 will be used.

    """

    if isinstance(target_depth_km, (float, int)):
        target_depth_km = [target_depth_km]  

    pressures = []

    for depth in target_depth_km:
        if all(v is None for v in [crustal_model_config.crust_dens_kgm3, crustal_model_config.d1, crustal_model_config.d2, crustal_model_config.rho1, crustal_model_config.rho2, crustal_model_config.rho3, crustal_model_config.model]):
            crustal_model_config.crust_dens_kgm3 = 2750
            warning_message = "\033[91mNo crustal parameters were provided, setting crust_dens_kgm3 to 2750. \nPlease use config_crustalmodel(...) to set your desired crustal model parameters.\033[0m"
            warnings.simplefilter("always")
            warnings.warn(warning_message, Warning, stacklevel=2)
        
        # Use the Newton-Raphson method for each target depth
        pressure = newton(objective_function_depth, initial_P_guess_kbar, args=(depth, crustal_model_config.crust_dens_kgm3, crustal_model_config.d1, crustal_model_config.d2, crustal_model_config.rho1, crustal_model_config.rho2, crustal_model_config.rho3, crustal_model_config.model), tol=tolerance)
        pressures.append(pressure)

    if isinstance(target_depth_km, (float, int)):
        return pressures[0]
    elif isinstance(target_depth_km, pd.Series):
        return pd.Series(pressures)
    else:
        return pressures

## Auxilliary functions for the stretching models

# Calculate decompression steps for polybaric model (Pressure, Depth, dt)

def calculate_DPdt(ascent_rate_ms,crustal_model_config=config_crustalmodel(),D_initial_km=None,D_final_km=None,D_step=100,initial_P_guess_kbar=0, tolerance=0.001):
    """
    Calculate the decompression rate (dP/dt) during ascent.

    Parameters:
    - ascent_rate_ms (float): Ascent rate in meters per second.
    - D_initial_km (float, optional): Initial depth in kilometers. Default is 30 km.
    - D_final_km (float, optional): Final depth in kilometers. Default is 0 km.
    - D_step (int, optional): Number of depth steps for calculation. Default is 100.
    - initial_P_guess_kbar (float, optional): Initial guess for pressure in kilobars (kbar). Default is 0.
    - tolerance (float, optional): Tolerance for pressure estimation. Default is 0.001.

    Returns:
    - D (pd.Series): Depth values in kilometers.
    - Pexternal_steps_MPa (list): Lithostatic pressure values in megapascals (MPa) at each depth step.
    - dt (float): Time step for the integration.
    """

    if D_initial_km is None or D_final_km is None or D_initial_km <= D_final_km:
        raise ValueError("Both D_initial_km and D_final_km must be provided, and D_initial_km must be larger than D_final_km")
    if D_initial_km>30 and D_step <= 80 and ascent_rate_ms <= 0.02:
        raise Warning("Your D_step is too small, the minimum recommended for ascent rates below 0.02 m/s is 80")
    D = pd.Series(list(np.linspace(D_initial_km, D_final_km, D_step)))  # km

    Pexternal_steps=find_P_for_kmdepth(D, crustal_model_config=crustal_model_config, initial_P_guess_kbar=initial_P_guess_kbar, tolerance=tolerance)
    Pexternal_steps_MPa=Pexternal_steps*100

    # Time steps of the ascent
    ascent_rate = ascent_rate_ms / 1000  # km/s
    D_change = abs(D.diff())
    time_series = D_change / ascent_rate  # calculates the time in between each step based on ascent rate
    dt_s = time_series.max()  # this sets the time step for the iterations later

    return D, Pexternal_steps_MPa, dt_s

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
def calculate_dR_dt(*,R_m, b_m, T_K,  Pinternal_MPa, Pexternal_MPa):
    """
    Calculate the rate of change of inclusion radius (dR/dt) based on power law creep.

    Parameters:
    - R_m (float): Inclusion radius in meters.
    - b_m (float): Distance to the crystal defect structures. Wanamaker and Evans (1989) use R/b=1/1000.
    - T_K (float): Temperature in Kelvin.
    - Pinternal_MPa (float): Internal pressure in MPa.
    - Pexternal_MPa (float): External pressure in MPa.

    Returns:
    - dR_dt (float): Rate of change of inclusion radius in meters per second.
    """
    
    pl_Cs = power_creep_law_constants()
    if Pinternal_MPa<Pexternal_MPa==True:
        S=-1
    else:
        S=1
    try:
        dR_dt = 2 * (S * pl_Cs.A * math.exp(-pl_Cs.Q / (pl_Cs.IgasR * T_K))) * (((R_m * b_m)**3) / (((b_m**(3 / pl_Cs.n)) - (R_m**(3 / pl_Cs.n))))**pl_Cs.n) * (((3 * abs(Pinternal_MPa - Pexternal_MPa)) / (2 * pl_Cs.n))**pl_Cs.n) / R_m**2
        return dR_dt

    except FloatingPointError:
        return np.nan
    
# Helper function to numerically solve for R (uses Runge-Kutta method, orders 1-4)   
def get_R(R_m,b_m,T_K,Pinternal_MPa,Pexternal_MPa,dt_s,method='RK1'):
    """
    Find the radius R of an FI over a time step using the Runge-Kutta numerical method. 
    Options are order 1 to 4 RK methods, such as RK1 (Euler), RK2 (Heun), RK3, or RK4.

    Parameters:
    - R_m (float): Initial FI Radius in meters.
    - b_m (float): Distance to defect structures in meters.
    - T_K (float): Temperature in Kelvin.
    - Pinternal_MPa (float): Internal pressure in MPa.
    - Pexternal_MPa (float): External pressure in MPa.
    - dt_s (float): The time step for integration in seconds.
    - method (str, optional): The numerical integration method to use. Default is 'RK1'.

    Returns:
    - tuple: A tuple containing the updated value of R_m and the derivative dR_dt (rate of change of R).
    """
    if method == 'RK1'or 'Euler':
        k1 = dt_s * calculate_dR_dt(R_m=R_m, b_m=b_m, T_K=T_K, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa)
        dR=k1
        dR_dt = dR / dt_s
        R_m += dR
    elif method == 'RK2' or 'Heun':
        k1 = dt_s * calculate_dR_dt(R_m=R_m, b_m=b_m, T_K=T_K, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa)
        k2 = dt_s * calculate_dR_dt(R_m=R_m + 0.5 * k1, b_m=b_m, T_K=T_K, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa)
        dR = ((k1 + k2) / 2)
        dR_dt = dR / dt_s
        R_m += dR
    elif method == 'RK3':
        k1 = dt_s * calculate_dR_dt(R_m=R_m, b_m=b_m, T_K=T_K, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa)
        k2 = dt_s * calculate_dR_dt(R_m=R_m + 0.5 * k1, b_m=b_m, T_K=T_K, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa)
        k3 = dt_s * calculate_dR_dt(R_m=R_m + 0.5 * k2, b_m=b_m, T_K=T_K, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa)
        dR = ((k1 + 4 * k2 + k3) / 6)
        dR_dt = dR / dt_s
        R_m += dR
    elif method == 'RK4':
        k1 = dt_s * calculate_dR_dt(R_m=R_m, b_m=b_m, T_K=T_K, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa)
        k2 = dt_s * calculate_dR_dt(R_m=R_m + 0.5 * k1, b_m=b_m, T_K=T_K, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa)
        k3 = dt_s * calculate_dR_dt(R_m=R_m + 0.5 * k2, b_m=b_m, T_K=T_K, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa)
        k4 = dt_s * calculate_dR_dt(R_m=R_m + k3, b_m=b_m, T_K=T_K, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa)
        dR = ((k1 + 2 * k2 + 2 * k3 + k4) / 6)
        dR_dt = dR / dt_s
        R_m += dR / 6
    else:
        raise ValueError("Unsupported numerical method. Choose from 'RK1' or 'Euler', 'RK2' or 'Huen', 'RK3', 'RK4'")

    return R_m, dR_dt

## Functions to calculate P, CO2dens, CO2mass and V

# Calculate initial CO2 density in g/cm3 and CO2 mass in g

def get_initial_CO2(R_m, T_K, P_MPa, EOS='SW96', return_volume=False):
    """
    Calculate the initial density and mass of CO2 inside a fluid inclusion (FI).

    Parameters:
    - R_m (float): The radius of the fluid inclusion (FI), in meters.
    - T_K (float): The temperature, in Kelvin.
    - P_MPa (float): The pressure, in MegaPascals (MPa).
    - EOS (str, optional): The equation of state (EOS) to use for density calculations.
        Can be one of: 'ideal' (ideal gas), 'SW96' (Span and Wagner EOS 1996), or 'SP94' (Sterner and Pitzer EOS 1994).
        Defaults to 'SW96'.
    - return_volume (bool, optional): Whether to return the volume of the FI along with density and mass. Defaults to False.

    Returns:
    - tuple or float: If return_volume is True, returns a tuple containing (V, CO2_dens_initial, CO2_mass_initial), where:
      - V (float): The volume of the fluid inclusion (FI), in cubic meters (m³).
      - CO2_dens_initial (float): The initial density of CO2 within the FI, in grams per cubic centimeter (g/cm³).
      - CO2_mass_initial (float): The initial mass of CO2 within the FI, in grams (g).

    - If return_volume is False, returns a tuple containing (CO2_dens_initial, CO2_mass_initial).

    """

    valid_EOS = ['ideal', 'SW96', 'SP94']

    try:
        if EOS not in valid_EOS:
            raise ValueError("EOS can only be 'ideal', 'SW96', or 'SP94'")
        
        if EOS == 'ideal':
            R_gas = 8.314  # J.mol/K J: kg·m²/s²
            V = 4/3 * math.pi * R_m**3  # m3
            P = P_MPa * 10**6  # convert MPa to Pa
            M = 44.01 / 1000  # kg/mol

            CO2_mass_kg = P * V * M / (R_gas * T_K)  # CO2 mass in kg
            rho = (CO2_mass_kg / V)  # rho in kg/m3

            CO2_dens_initial = rho / 1000  # CO2 density in g/cm3
            CO2_mass_initial = CO2_mass_kg / 1000  # CO2 mass in g

        else:
            R_m = R_m * 10**2  # radius in cm
            V = 4/3 * math.pi * R_m**3  # cm3, Volume of the FI, assume sphere
            P_kbar = P_MPa / 100  # Internal pressure of the FI, convert to kbar

            CO2_dens_initial = calculate_rho_for_P_T(EOS=EOS, P_kbar=P_kbar, T_K=T_K)[0]  # CO2 density in g/cm3
            CO2_mass_initial = CO2_dens_initial * V  # CO2 mass in g

        if return_volume:
            return V, CO2_dens_initial, CO2_mass_initial
        else:
            return CO2_dens_initial, CO2_mass_initial

    except ValueError as ve:
        raise ve

# Calculate CO2 density in g/cm3 and P in MPa for fixed CO2 mass in g

def get_CO2dens_P(R_m,T_K,CO2_mass,EOS='SW96',return_volume=False):
    """
    Calculate the density and pressure of CO2 inside a fluid inclusion (FI).

    Parameters:
    - R_m (float): The radius of the fluid inclusion (FI), in meters.
    - T_K (float): The temperature, in Kelvin.
    - CO2_mass (float): The mass of CO2 within the FI, in grams (g).
    - EOS (str, optional): The equation of state (EOS) to use for density and pressure calculations.
        Can be one of: 'ideal' (ideal gas), 'SW96' (Span and Wagner 1996), or 'SP94' (Sterner and Pitzer 1994).
        Defaults to 'SW96'.
    - return_volume (bool, optional): Whether to return the volume of the FI along with density and pressure. Defaults to False.

    Returns:
    - tuple or float: If return_volume is True, returns a tuple containing (V, CO2_dens, P), where:
      - V (float): The volume of the fluid inclusion (FI), in cubic meters (m³).
      - CO2_dens (float): The density of CO2 within the FI, in grams per cubic centimeter (g/cm³).
      - P (float): The pressure of CO2 within the FI, in MegaPascals (MPa).

    - If return_volume is False, returns a tuple containing (CO2_dens, P).

    """
    valid_EOS = ['ideal', 'SW96', 'SP94']

    try:
        if EOS not in valid_EOS:
            raise ValueError("EOS can only be 'ideal', 'SW96', or 'SP94'")
        
        if EOS == 'ideal':
            R_gas = 8.314  # J.mol/K J: kg·m²/s²
            V = 4/3 * math.pi * R_m**3  # m3
            M = 44.01 / 1000  # kg/mol

            CO2_mass_kg=CO2_mass*1000
            P=CO2_mass_kg*R_gas*T_K/(M*V) #P in Pa
            CO2_dens=(CO2_mass_kg/V) # CO2 density in kg/m3

            P=P/(10**6) #P in MPa
            CO2_dens=CO2_dens/1000 #rho in g/cm3

        else:
            R_m=R_m*10**2 #FI radius, convert to cm
            V=4/3*math.pi*R_m**3 #cm3, Volume of the FI, assume sphere

            CO2_dens=CO2_mass/V # CO2 density in g/cm3

            try:
                P=calculate_P_for_rho_T(EOS=EOS,CO2_dens_gcm3=CO2_dens, T_K=T_K)['P_MPa'][0] #g/cm3, CO2 density

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
def stretch_in_ascent(*, R_m, b_m, T_K, ascent_rate_ms, depth_path_ini_fin_step=[100, 0, 100],
                      crustal_model_config=config_crustalmodel(crust_dens_kgm3=2750),
                      EOS, plotfig=True, report_results='fullpath',
                      initial_P_guess_kbar=0, tolerance=0.001,method='RK4',update_b=False):
    """
    Simulate the stretching of a CO2-dominated fluid inclusion (FI) during ascent.

    Parameters:
    - R_m (float): The initial radius of the fluid inclusion (FI), in meters.
    - b_m (float): The initial distance to a crystal defect (rim, crack, etc) from the FI center, in meters.
    - T_K (float): The temperature, in Kelvin.
    - ascent_rate_ms (float): The ascent rate, in meters per second (m/s).
    - depth_path_ini_fin_step (list, optional): A list containing [initial_depth_km, final_depth_km, depth_step].
      Defaults to [100, 0, 100], representing the depth path from initial to final depth in a number of steps.
    - crustal_model_config (dict, optional): Configuration parameters for the crustal model.
      Defaults to a predefined configuration with a crustal density of 2750 kg/m³.
    - EOS (str): The equation of state (EOS) to use for density calculations. Can be one of: 'ideal' (ideal gas),
      'SW96' (Span and Wagner 1996), or 'SP94' (Sterner and Pitzer 1994).
    - plotfig (bool, optional): Whether to plot figures showing the changes in depth and CO2 density. Defaults to True.
    - report_results (str, optional): The type of results to report. Can be 'fullpath', 'startendonly', or 'endonly'.
      Defaults to 'fullpath'.
    - initial_P_guess_kbar (float, optional): Initial guess for internal pressure (Pinternal_MPa) in MPa. Defaults to 0.
    - tolerance (float, optional): Tolerance for pressure calculations. Defaults to 0.001.
    - method (str, optional): The numerical integration method to use for change in FI radius. Can be 'RK1' (also 'Euler'), 'RK2' (also 'Heun'), 'RK3', or 'RK4'.
      Defaults to 'RK4'.
    - update_b (bool, optional): Whether to update 'b' during the ascent. Defaults to False.

    Returns:
    - pandas.DataFrame: A DataFrame containing the simulation results, including time, depth, pressure, radius changes,
      and CO2 density.

    """

    D, Pexternal_steps, dt_s = calculate_DPdt(ascent_rate_ms=ascent_rate_ms, crustal_model_config=crustal_model_config,
                                            D_initial_km=depth_path_ini_fin_step[0], D_final_km=depth_path_ini_fin_step[1],
                                            D_step=depth_path_ini_fin_step[2],
                                            initial_P_guess_kbar=initial_P_guess_kbar, tolerance=tolerance)
    Pinternal_MPa = Pexternal_steps[0]

    CO2_dens_initial, CO2_mass_initial = get_initial_CO2(R_m=R_m, T_K=T_K, P_MPa=Pinternal_MPa, EOS=EOS)

    
    results = pd.DataFrame([{'Time(s)': 0,
                            'Step':0,
                            'dt(s)':0,
                        'Pexternal(MPa)': Pinternal_MPa,
                        'Pinternal(MPa)': Pinternal_MPa,
                        'dR/dt(m/s)': calculate_dR_dt(R_m=R_m, b_m=b_m, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pinternal_MPa, T_K=T_K),
                        'Fi_radius(μm)': R_m*10**6,
                        'b (distance to xtal rim -μm)':b_m*10**6,
                        '\u0394R/R0 (fractional change in radius)':np.nan,
                        'CO2_dens_gcm3': CO2_dens_initial,
                        'Depth(km)':D.iloc[0]}], index=range(len(Pexternal_steps)))

   
    for i in range(1,len(Pexternal_steps)):
        
        Pexternal = Pexternal_steps[i]

        R_m,dR_dt = get_R(R_m=R_m,b_m=b_m,T_K=T_K,Pinternal_MPa=Pinternal_MPa,Pexternal_MPa=Pexternal,dt_s=dt_s,method=method)
        CO2_dens_new,P_new = get_CO2dens_P(R_m=R_m,T_K=T_K,CO2_mass=CO2_mass_initial,EOS=EOS)

        Pinternal_MPa = P_new

        if update_b==True:
            b_m=1000*R_m
        
        results.loc[i] = [dt_s*i, i, dt_s, Pexternal, Pinternal_MPa, dR_dt, R_m * 10 ** 6, b_m * 10 ** 6,
                    (R_m * 10 ** 6 - results.loc[0, 'Fi_radius(μm)']) / results.loc[0, 'Fi_radius(μm)'],
                    CO2_dens_new, D.iloc[i]]

    if report_results == 'startendonly':
        results.drop(index=list(range(1, results.shape[0] - 1)), inplace=True)  # Drop all rows except first and last

    if report_results == 'endonly':
        results.drop(index=list(range(0, results.shape[0] - 1)), inplace=True)  # Drop all rows except last

    if plotfig==True:
        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10,3))
        ax0.plot(results['Depth(km)'],results['\u0394R/R0 (fractional change in radius)'],marker='s',label=f"Ascent Rate = {ascent_rate_ms} m/s")
        ax0.set_xlim([depth_path_ini_fin_step[0],depth_path_ini_fin_step[1]])
        ax0.set_xlabel("Depth (km)")
        ax0.set_ylabel('\u0394R/R0 (fractional change in radius)')

        ax1.plot(results['Depth(km)'],results['CO2_dens_gcm3'],marker='s',label=f"Ascent Rate = {ascent_rate_ms} m/s")
        ax1.set_xlim([depth_path_ini_fin_step[0],depth_path_ini_fin_step[1]])
        ax1.set_xlabel("Depth (km)")
        ax1.set_ylabel("CO$_2$ density (g/cm$^{3}$)")
        ax0.legend(loc='best')
        ax1.legend(loc='best')
        fig.tight_layout()
        plt.show()

    return results

# This function is to model stretching at fixed External Pressure (e.g., during stalling or upon eruption)
def stretch_at_constant_Pext(*,R_m,b_m,T_K,EOS='SW96',Pinternal_MPa,Pexternal_MPa,totaltime_s,steps,method='RK4',report_results='fullpath',plotfig=False,update_b=False):
    """
    Simulate the stretching of a CO2 fluid inclusion (FI) under constant external pressure (e.g., quenching or storage).

    Parameters:
    - R_m (float): The initial radius of the fluid inclusion (FI), in meters.
    - b_m (float): The initial distance to a crystal defect (rim, crack, etc) from the FI center, in meters.
    - T_K (float): The temperature, in Kelvin.
    - Pinternal_MPa (float): The initial internal pressure of the FI, in MegaPascals (MPa).
    - Pexternal_MPa (float): The constant external pressure applied to the FI, in MegaPascals (MPa).
    - totaltime_s (float): The total simulation time, in seconds.
    - steps (int): The number of simulation steps.
    - EOS (str): The equation of state (EOS) to use for density calculations. Can be one of: 'ideal' (ideal gas),
      'SW96' (Span and Wagner 1996), or 'SP94' (Sterner and Pitzer 1994).
    - method (str, optional): The numerical integration method to use for change in FI radius. Can be 'RK1' (also 'Euler'), 'RK2' (also 'Heun'), 'RK3', or 'RK4'.
      Defaults to 'RK4'.
    - report_results (str, optional): The type of results to report. Can be 'fullpath' (all steps reported), 'startendonly' (only initial and end steps), or 'endonly' (only last step).
      Defaults to 'fullpath'.
    - plotfig (bool, optional): Whether to plot figures showing the changes in time, radius, and CO2 density. Defaults to False.
    - update_b (bool, optional): Whether to update 'b' during the simulation. Defaults to False.

    Returns:
    - pandas.DataFrame: A DataFrame containing the simulation results, including time, pressure, radius changes, and CO2 density.

    Raises:
    - ValueError: If an unsupported EOS is specified.
    """

    CO2_dens_initial,CO2_mass_initial=get_initial_CO2(R_m=R_m,T_K=T_K,P_MPa=Pinternal_MPa,EOS=EOS)

    results = pd.DataFrame([{'Time(s)': 0,
                             'Step':0,
                             'dt(s)':0,
                            'Pexternal(MPa)': Pexternal_MPa,
                            'Pinternal(MPa)': Pinternal_MPa,
                            'dR/dt(m/s)': calculate_dR_dt(R_m=R_m, b_m=b_m, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa, T_K=T_K),
                            'Fi_radius(μm)': R_m*10**6,
                            'b (distance to xtal rim -μm)':b_m*10**6,
                            '\u0394R/R0 (fractional change in radius)':0,
                            'CO2_dens_gcm3': CO2_dens_initial}], index=range(steps))

    dt_s=totaltime_s/steps
    
    for step in range(1,steps):

        R_new,dR_dt = get_R(R_m=R_m,b_m=b_m,T_K=T_K,Pinternal_MPa=Pinternal_MPa,Pexternal_MPa=Pexternal_MPa,dt_s=dt_s,method=method)

        CO2_dens_new,P_new = get_CO2dens_P(R_m=R_new,T_K=T_K,CO2_mass=CO2_mass_initial,EOS=EOS)

        Pinternal_MPa = P_new
        R_m=R_new

        if update_b==True:
            b_m=1000*R_m
        
        results.loc[step] = [step * dt_s, step, dt_s, Pexternal_MPa, Pinternal_MPa, dR_dt, R_m * 10 ** 6, b_m * 10 ** 6,
                            (R_m * 10 ** 6 - results.loc[0, 'Fi_radius(μm)']) / results.loc[0, 'Fi_radius(μm)'],
                                CO2_dens_new]

    if report_results == 'startendonly':
        results.drop(index=list(range(1, results.shape[0] - 1)), inplace=True)  # Drop all rows except first and last

    if report_results == 'endonly':
        results.drop(index=list(range(0, results.shape[0] - 1)), inplace=True)  # Drop all rows except last

    if plotfig==True:
        if totaltime_s < 60:
            x_time = results['Time(s)']
            xlabel = 'Time(s)'
        elif 60 <= totaltime_s < 3600:
            x_time = results['Time(s)'] / 60
            xlabel = 'Time(min)'
            results[xlabel]=x_time
        elif 3600 <= totaltime_s < 86400:
            x_time = results['Time(s)'] / 3600
            xlabel = 'Time(hr)'
            results[xlabel]=x_time
        elif 86400 <= totaltime_s < 31536000:
            x_time = results['Time(s)'] / (3600 * 24)
            xlabel = 'Time(days)'
            results[xlabel]=x_time
        elif totaltime_s >= 31536000:
            x_time = results['Time(s)'] / (3600 * 24 * 365)
            xlabel = 'Time(years)'
            results[xlabel]=x_time

        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10,3))
        ax0.plot(x_time,results['\u0394R/R0 (fractional change in radius)'],marker='s')
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel("\u0394R/R0 (fractional change in radius)")

        ax1.plot(x_time,results['CO2_dens_gcm3'],marker='s')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("CO2_density_gmL")
        fig.tight_layout()
        plt.show()

    return results

# This function can loop through different R and b value sets using stretch at constant Pext

def loop_R_b_constant_Pext(*,R_m_values, b_m_values, T_K, EOS, Pinternal_MPa, Pexternal_MPa, totaltime_s, steps, T4endcalc_PD, method='RK4',
                            plotfig=False, crustal_model_config=config_crustalmodel(crust_dens_kgm3=2750)):

    """
    Perform multiple simulations under constant external pressure with various R and b values.

    Parameters:
    - R_m_values (list): A list of initial radius values for fluid inclusions (FI), in meters.
    - b_m_values (list): A list of initial distance values to a crystal defect (rim, crack, etc) from the FI center, in meters.
    - T_K (float): The temperature, in Kelvin.
    - EOS (str): The equation of state (EOS) to use for density calculations. Can be one of: 'ideal' (ideal gas),
      'SW96' (Span and Wagner 1996), or 'SP94' (Sterner and Pitzer 1994).
    - Pinternal_MPa (float): The initial internal pressure of the FI, in MegaPascals (MPa).
    - Pexternal_MPa (float): The constant external pressure applied to the FI, in MegaPascals (MPa).
    - totaltime_s (float): The total simulation time, in seconds.
    - steps (int): The number of simulation steps.
    - T4endcalc_PD (float): The temperature at which to calculate the depths (Kelvin) at the end of the simulations.
    - method (str, optional): The numerical integration method to use for change in FI radius. Can be 'RK1' (also 'Euler'), 'RK2' (also 'Heun'), 'RK3', or 'RK4'.
      Defaults to 'RK4'.
    - plotfig (bool, optional): Whether to plot figures showing the changes in time, radius, and CO2 density. Defaults to False.
    - crustal_model_config (dict, optional): Configuration parameters for the crustal model. Defaults to a predefined 
        configuration with a crustal density of 2750 kg/m³.

    Returns:
    - dict: A dictionary containing simulation results for varying R and b values. The keys are in the format 'R{index}_b{index}',
          where 'index' corresponds to the index of R_values and b_values, respectively. The values are DataFrames with 
          simulation results.
    """

    results_dict = {}
    for idx_R, R in enumerate(R_m_values):  # Use enumerate to get the index of R_values
        R_key = f'R{idx_R}'  # Use 'R' followed by the index
        results_dict[R_key] = {}

        for idx_b, b in enumerate(b_m_values):  # Use enumerate to get the index of b_values
            b_key = f'b{idx_b}'  # Use 'b' followed by the index
            results = stretch_at_constant_Pext(R_m=R, b_m=b, T_K=T_K, Pinternal_MPa=Pinternal_MPa, Pexternal_MPa=Pexternal_MPa,
                                              totaltime_s=totaltime_s, steps=steps, EOS=EOS, method=method,
                                              plotfig=plotfig)
            results['Calculated depths (km)_StorageT'] = convert_pressure_to_depth(
                P_kbar=results['Pinternal(MPa)'] / 100,
                crust_dens_kgm3=crustal_model_config.crust_dens_kgm3, g=9.81,
                d1=crustal_model_config.d1, d2=crustal_model_config.d2,
                rho1=crustal_model_config.rho1, rho2=crustal_model_config.rho2, rho3=crustal_model_config.rho3,
                model=crustal_model_config.model)
            results['Calculated P from rho (MPa)_TrappingT'] = calculate_P_for_rho_T(
                EOS='SW96', CO2_dens_gcm3=results['CO2_dens_gcm3'], T_K=T4endcalc_PD + 273.15)['P_MPa']

            results['Calculated depths (km)_TrappingT'] = convert_pressure_to_depth(
                P_kbar=results['Calculated P from rho (MPa)_TrappingT'] / 100,
                crust_dens_kgm3=crustal_model_config.crust_dens_kgm3, g=9.81,
                d1=crustal_model_config.d1, d2=crustal_model_config.d2,
                rho1=crustal_model_config.rho1, rho2=crustal_model_config.rho2, rho3=crustal_model_config.rho3,
                model=crustal_model_config.model)

            results_dict[R_key][b_key] = results

    return results_dict
