
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DiadFit as pf

from DiadFit.density_depth_crustal_profiles import *
from DiadFit.CO2_EOS import *

## Microthermometry error propagation
# propagate_microthermometry_uncertainty_1sam goes to 'make_error_dist_microthermometry_1sam'

def make_error_dist_microthermometry_1sam(*, T_h_C, sample_i=0, error_T_h_C=0.3, N_dup=1000,
        error_dist_T_h_C='uniform', error_type_T_h_C='Abs', len_loop=1):
    
    """ 

    This function generates a dataset of temperature measurements for a given sample. 
    It adds random noise to the temperature measurement based on the specified distribution and error type.

    Parameters
    ----------
    T_h_C : numeric or list of numeric values
        The measured temperature(s) of the sample(s) in degrees Celsius.
    sample_i : int, optional
        The index of the sample for which the error distribution will be generated. Default value is 0.
    error_T_h_C : numeric, optional
        The amount of error to add to the temperature measurement. Default value is 0.3.
    N_dup : int, optional
        The number of duplicated samples to generate with random noise. Default value is 1000.
    error_dist_T_h_C : str, optional
        The distribution of the random noise to be added to the temperature measurement. Can be either 'normal' or 'uniform'. Default value is 'uniform'.
    error_type_T_h_C : str, optional
        The type of error to add to the temperature measurement. Can be either 'Abs' or 'Perc'. Default value is 'Abs'.
    len_loop : int, optional
        The number of samples for which the error distribution will be generated. Default value is 1.

    Returns
    -------
    numpy.ndarray
        An array of temperature measurements with random noise added to them based on the specified error 
        distribution and error type. The size of the array is (N_dup, len(T_h_C)).

    """

    if len_loop==1:
        df_c=pd.DataFrame(data={'T_h_C': T_h_C}, index=[0])
    else:
        df_c=pd.DataFrame(data={'T_h_C': T_h_C})


    # Temperature error distribution
    if error_type_T_h_C=='Abs':
        error_T_h_C=error_T_h_C
    if error_type_T_h_C =='Perc':
        error_T_h_C=df_c['T_h_C'].iloc[sample_i]*error_T_h_C/100
    if error_dist_T_h_C=='normal':
        Noise_to_add_T_h_C = np.random.normal(0, error_T_h_C, N_dup)
    if error_dist_T_h_C=='uniform':
        Noise_to_add_T_h_C = np.random.uniform(- error_T_h_C, +
                                                      error_T_h_C, N_dup)
    
    T_h_C_with_noise=Noise_to_add_T_h_C+df_c['T_h_C'].iloc[0]
    
    

    return T_h_C_with_noise


def propagate_microthermometry_uncertainty(T_h_C, Sample_ID=None,  error_T_h_C=0.3, N_dup=1000,
        error_dist_T_h_C='uniform', error_type_T_h_C='Abs', EOS='SW96',  homog_to=None, set_to_critical=False):

    """
    This function propagates the uncertainty in measured temperature values to calculate the density of gas and
     liquid CO2 using an equation of state (EOS). 
    It loops over more than 1 sample, using the function make_error_dist_microthermometry_1sam for 
    each sample to generate the variable input parameters
    It generates a dataset of temperature measurements with random noise added to them based on the 
    specified distribution and error type, 
    calculates the CO2 density for each temperature value, and returns the mean and standard deviation 
    of the density values.

    Parameters
    ----------
    T_h_C : numeric or list of numeric values
        The measured temperature(s) of the sample(s) in degrees Celsius.

    Sample_ID : str or pandas.Series
        The ID or a pandas Series of IDs of the sample(s). If not provided, the function 
        uses an index number as the ID. Default is None.

    error_T_h_C : float or pandas.Series
        The amount of error to add to the temperature measurement. If a pandas.Series is 
        provided, the function takes the right one for each loop. Default value is 0.3.

    N_dup : int, optional
        The number of duplicated samples to generate with random noise. Default value is 1000.

    error_dist_T_h_C : str, optional
        The distribution of the random noise to be added to the temperature measurement. 
        Can be either 'normal' or 'uniform'. Default value is 'uniform'.

    error_type_T_h_C : str, optional
        The type of error to add to the temperature measurement. Can be either 'Abs' or 'Perc'. 
        Default value is 'Abs'.

    EOS : str, optional
        The equation of state to use for the calculation. Can be either 'SW96' or 'SP94'. Default value is 'SW96'.


    homog_to : str, optional
        The phase to which the CO2 density is homogenized. Can be either 'Gas' or 'Liq'. Default is None.
        
    set_to_critical: bool
        Default False. If true, if you enter T_h_C which exceeds 30.9782 (the critical point of CO2) it replaces your entered Temp with that temp.
        

    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        A tuple of two pandas DataFrames. The first DataFrame contains the mean and standard 
        deviation of the gas and liquid CO2 density values. The second DataFrame contains the 
        CO2 density values for each temperature value in the input.
        The output Std_density_Gas_gcm3_from_percentiles is a more representative output of the error than the 
        standard deviation if the output distribution isn't Gaussian. 

        
    """

    if type(T_h_C) is pd.Series:
        len_loop=len(T_h_C)
    else:
        len_loop=1

    


    All_outputs=pd.DataFrame([])
    Std_density_gas=np.empty(len_loop)
    Std_density_liq=np.empty(len_loop)
    Mean_density_gas=np.empty(len_loop)
    Mean_density_liq=np.empty(len_loop)
    Std_density_gas_IQR=np.empty(len_loop)
    Std_density_Liq_IQR=np.empty(len_loop)
    Sample=np.empty(len_loop,  dtype=np.dtype('U100') )

    for i in range(0, len_loop):

        # If user has entered a pandas series for error, takes right one for each loop
        if type(error_T_h_C) is pd.Series:
            error_T_h_C=error_T_h_C.iloc[i]
        else:
            error_T_h_C=error_T_h_C

        if type(T_h_C) is pd.Series:
            T_h_C_i=T_h_C.iloc[i]
        else:
            T_h_C_i=T_h_C

        # Check of
        if Sample_ID is None:
            Sample[i]=i

        elif isinstance(Sample_ID, str):
            Sample[i]=Sample_ID
        else:
            Sample[i]=Sample_ID.iloc[i]

        Temp_MC=make_error_dist_microthermometry_1sam(T_h_C=T_h_C_i,
        sample_i=i, error_T_h_C=error_T_h_C, N_dup=N_dup,
        error_dist_T_h_C=error_dist_T_h_C, error_type_T_h_C=error_type_T_h_C, len_loop=1)

        Sample2=Sample[i]
        MC_T=calculate_CO2_density_homog_T(T_h_C=Temp_MC, Sample_ID=Sample2, EOS=EOS, homog_to=homog_to)

        # Replace critical with NaN



        # MC for each FI
        All_outputs=pd.concat([All_outputs, MC_T], axis=0)

        # get av and mean
        Std_density_gas[i]=np.nanstd(MC_T['Gas_gcm3'])
        Std_density_liq[i]=np.nanstd(MC_T['Liq_gcm3'])
        Mean_density_gas[i]=np.nanmean(MC_T['Gas_gcm3'])
        Mean_density_liq[i]=np.nanmean(MC_T['Liq_gcm3'])
        var=MC_T['Gas_gcm3']
        Std_density_gas_IQR[i]=0.5*np.abs((np.percentile(var, 84) -np.percentile(var, 16)))
        varL=MC_T['Liq_gcm3']
        Std_density_Liq_IQR[i]=0.5*np.abs((np.percentile(varL, 84) -np.percentile(varL, 16)))

    # Preferred density no MC
    Density_pref=calculate_CO2_density_homog_T(T_h_C=T_h_C, Sample_ID=Sample_ID, EOS=EOS, homog_to=homog_to, set_to_critical=set_to_critical)

    Av_outputs=pd.DataFrame(data={'Sample_ID': Sample,
                                    'Density_Gas_noMC': Density_pref['Gas_gcm3'],
                                    'Density_Liq_noMC': Density_pref['Liq_gcm3'],
                                      'Mean_density_Gas_gcm3': Mean_density_gas,
                                      'Std_density_Gas_gcm3': Std_density_gas,
                                      'Std_density_Gas_gcm3_from_percentiles': Std_density_gas_IQR,
                                      'Std_density_Liq_gcm3_from_percentiles': Std_density_Liq_IQR,
                                       'Mean_density_Liq_gcm3': Mean_density_liq,
                                      'Std_density_Liq_gcm3': Std_density_liq,
                                      'Input_temp': T_h_C,
                                      'error_T_h_C': error_T_h_C})



    return Av_outputs, All_outputs




def calculate_temperature_density_MC(sample_i=0,  N_dup=1000, 
CO2_dens_gcm3=None, error_CO2_dens=0, error_type_CO2_dens='Abs', error_dist_CO2_dens='normal',
 T_K=None, error_T_K=0, error_type_T_K='Abs', error_dist_T_K='normal',
crust_dens_kgm3=None, error_crust_dens=0, error_type_crust_dens='Abs', error_dist_crust_dens='normal',
 model=None):

    """
    This function generates the range of T_K, CO2 densities and crustal densities for 1 sample 
    for performing Monte Carlo simulations
    using the function propagate_FI_uncertainty (e.g. this function makes the range of 
    input parameters for each sample, but doesnt do the EOS calculations). 

    Parameters
    -----------------
    sample_i: int
        The index of the sample

    N_dup: int
        Number of synthetic inputs to do for each sample. 

    Required input information about CO2 densities

    CO2_dens_gcm3: pd.Series, integer or float
        CO2 densities in g/cm3 to perform calculations with. Can be a column from your dataframe (df['density_g_cm3']), or a single value (e.g.., 0.2)
    error_CO2_dens: float
        Error in CO2 fluid density
    error_type_CO2_dens: str
        Type of CO2 fluid density error. Can be 'Abs' or 'Perc'
    error_dist_CO2_dens: str
        Distribution of CO2 fluid density error. Can be 'normal' or 'uniform'.

    Required input information about fluid temperature 

    T_K: pd.Series, integer, float
        Temperature in Kelvin at which you think your fluid inclusion was trapped.
        Can be a column from your dataframe (df['T_K']), or a single value (e.g.., 1500)  
    error_T_K: float
        Error in temperature.
    error_type_T_K: str
        Type of temperature error. Can be 'Abs' or 'Perc'.
    error_dist_T_K: str
        Distribution of temperature error. Can be 'normal' or 'uniform'.

    Required input information for converting pressure to depth. Choose either:

    A fixed crustal density

        crust_dens_kgm3: float
            Density of the crust in kg/m^3.
        error_crust_dens: float, optional
            Error in crust density.
        error_type_crust_dens: str
            Type of crust density error. Can be 'Abs' or 'Perc'.
        error_dist_crust_dens: str
            Distribution of crust density error. Can be 'normal' or 'uniform'.
    

    OR a crustal density model

        model: str
            see documentation for the function 'convert_pressure_to_depth' to see more detail about model options. 
            If you select a model, it will just use this to calculate a depth, but it wont add any uncertainty to this model (as this is 
            more likely systematic than random uncertainty that can be simulated with MC methods)
            For this function, you dont need any more info like d1, rho1, that comes in the propagate_FI_uncertainty function. 

    Returns
    ------------
    df_out: pandas.DataFrame
        DataFrame containing information on temperature, CO2 density, crust density, and error.   

    """

    # print('entered T_K')
    # print(T_K)
    # print('entered CO2')
    # print(CO2_dens_gcm3)
    # If any of them are panda series or numpy nd array, you dont need an index
    if isinstance(T_K, pd.Series) or isinstance(CO2_dens_gcm3, pd.Series) or isinstance(T_K, np.ndarray) or isinstance(CO2_dens_gcm3, np.ndarray):
        df_c=pd.DataFrame(data={'T_K': T_K,
                            'CO2_dens_gcm3': CO2_dens_gcm3})
        
    # you do need an index here
    else:
        #print('here')
        df_c=pd.DataFrame(data={'T_K': T_K,
                            'CO2_dens_gcm3': CO2_dens_gcm3}, index=[0])
        


    # Temperature error distribution
    if error_type_T_K=='Abs':
        error_T_K=error_T_K
    if error_type_T_K =='Perc':
        error_T_K=df_c['T_K'].iloc[sample_i]*error_T_K/100
    if error_dist_T_K=='normal':
        Noise_to_add_T_K = np.random.normal(0, error_T_K, N_dup)
    if error_dist_T_K=='uniform':
        Noise_to_add_T_K = np.random.uniform(- error_T_K, +
                                                      error_T_K, N_dup)

    
    T_K_with_noise=Noise_to_add_T_K+df_c['T_K'].iloc[sample_i]
    T_K_with_noise[T_K_with_noise < 0.0001] = 0.0001


    # CO2 error distribution

    if error_type_CO2_dens=='Abs':
        error_CO2_dens=error_CO2_dens
    if error_type_CO2_dens =='Perc':
        error_CO2_dens=df_c['CO2_dens_gcm3'].iloc[sample_i]*error_CO2_dens/100
    if error_dist_CO2_dens=='normal':
        Noise_to_add_CO2_dens = np.random.normal(0, error_CO2_dens, N_dup)
    if error_dist_CO2_dens=='uniform':
        Noise_to_add_CO2_dens = np.random.uniform(- error_CO2_dens, +
                                                      error_CO2_dens, N_dup)

    CO2_dens_with_noise=Noise_to_add_CO2_dens+df_c['CO2_dens_gcm3'].iloc[sample_i]
    CO2_dens_with_noise[CO2_dens_with_noise < 0.0001] = 0.0001

    # Crustal density noise
    # First need to work out what crustal density is

    if type(crust_dens_kgm3) is float or type(crust_dens_kgm3) is int:
        # This is the simplicest scenario, just makes a distribution of pressures

        if error_type_crust_dens=='Abs':
            error_crust_dens=error_crust_dens
        if error_type_crust_dens =='Perc':
            error_crust_dens=crust_dens_kgm3*error_crust_dens/100
        if error_dist_crust_dens=='normal':
            Noise_to_add_crust_dens = np.random.normal(0, error_crust_dens, N_dup)
        if error_dist_crust_dens=='uniform':
            Noise_to_add_crust_dens = np.random.uniform(- error_crust_dens, +
                                                        error_crust_dens, N_dup)



        crust_dens_with_noise=Noise_to_add_crust_dens+crust_dens_kgm3
        crust_dens_with_noise[crust_dens_with_noise < 0.0001] = 0.0001

    elif model is not None:
        crust_dens_with_noise=None

    else:
        if error_crust_dens>0:
            raise Exception('You cannot use a crustal density model with an error in density due to ambiguity about how to apply this (and because errors will be systematic not random). This variable only works for a constant crustal density. Set error_crust_dens=0')
        # For all other models



    df_out=pd.DataFrame(data={'T_K_with_noise': T_K_with_noise,
                                'CO2_dens_with_noise': CO2_dens_with_noise,
                                'crust_dens_with_noise': crust_dens_with_noise,
                                'T_K': df_c['T_K'].iloc[sample_i],
                                'CO2_dens_gcm3': df_c['CO2_dens_gcm3'].iloc[sample_i],
                                'Crustal Density_kg_m3': crust_dens_kgm3,
                                'model': model,
                                'error_T_K': error_T_K,
                                'error_type_T_K': error_type_T_K,
                                'error_dist_T_K': error_dist_T_K,
                                'error_CO2_dens': error_CO2_dens,
                                'error_type_CO2_dens': error_type_CO2_dens,
                                'error_dist_CO2_dens': error_dist_CO2_dens,
                                'error_crust_dens_kgm3': error_crust_dens,
                                'error_type_crust_dens': error_type_crust_dens,
                                'error_dist_crust_dens': error_dist_crust_dens,
                                })




    return df_out


def loop_all_FI_MC(sample_ID, CO2_dens_gcm3, T_K, N_dup=1000,
crust_dens_kgm3=None, d1=None, d2=None, rho1=None, rho2=None, rho3=None,
error_crust_dens=0, error_type_crust_dens='Abs', error_dist_crust_dens='uniform',
error_T_K=0, error_type_T_K='Abs', error_dist_T_K='normal',
error_CO2_dens=0, error_type_CO2_dens='Abs', error_dist_CO2_dens='normal',
                plot_figure=False, fig_i=0):
    """ This is a redundant function, kept around for a while for backwards compatability"""

    print('Please use the new function propagate_FI_uncertainty instead as this allows you use different EOS')



def propagate_FI_uncertainty(sample_ID, CO2_dens_gcm3, T_K,   N_dup=1000, EOS='SW96', 
plot_figure=False, fig_i=0, 
error_CO2_dens=0, error_type_CO2_dens='Abs', error_dist_CO2_dens='normal',
 crust_dens_kgm3=None, model=None, d1=None, d2=None, rho1=None, rho2=None, rho3=None,
error_crust_dens=0, error_type_crust_dens='Abs', error_dist_crust_dens='uniform',
error_T_K=0, error_type_T_K='Abs', error_dist_T_K='normal',
                 ):

    """
    This function performs Monte Carlo simulations of uncertainty in CO2 density, input temperature, 
    and crustal density. 
    It uses the function 'calculate_temperature_density_MC' to make the simulated variables, 
    and then uses this to calculate a resulting 
    pressure using the equation of state of choice

    Parameters
    -----------------
    sample_ID: pd.Series
        Panda series of sample names. E.g., select a column from your dataframe (df['sample_name'])

    N_dup: int
        Number of Monte Carlo simulations to perform for each sample

    EOS: str
        'SW96' or 'SP94' for the pure CO2 EOS
    
    plot_figure: bool   
        if True, plots a figure for one sample showing the range of input parameters. If this is True, 
        also select:
        fig_i: int
        Which determins which sample is plotted (e.g. 0 is the 1st sample name, etc. )
    

    CO2_dens_gcm3: pd.Series, integer or float
        CO2 densities in g/cm3 to perform calculations with. Can be a column from your dataframe (df['density_g_cm3']), or a single value (e.g.., 0.2)
    error_CO2_dens: float
        Error in CO2 fluid density
    error_type_CO2_dens: str
        Type of CO2 fluid density error. Can be 'Abs' or 'Perc'
    error_dist_CO2_dens: str
        Distribution of CO2 fluid density error. Can be 'normal' or 'uniform'.

    T_K: pd.Series, integer, float
        Temperature in Kelvin at which you think your fluid inclusion was trapped.
        Can be a column from your dataframe (df['T_K']), or a single value (e.g.., 1500)  
    error_T_K: float
        Error in temperature.
    error_type_T_K: str
        Type of temperature error. Can be 'Abs' or 'Perc'.
    error_dist_T_K: str
        Distribution of temperature error. Can be 'normal' or 'uniform'.

    For converting pressure to depth in the crust, choose either

    A fixed crustal density

        crust_dens_kgm3: float
            Density of the crust in kg/m^3.
        error_crust_dens: float, optional
            Error in crust density.
        error_type_crust_dens: str
            Type of crust density error. Can be 'Abs' or 'Perc'.
        error_dist_crust_dens: str
            Distribution of crust density error. Can be 'normal' or 'uniform'.
    

    OR a crustal density model

        model: str
            see documentation for the function 'convert_pressure_to_depth' to see more detail about model options. 
            If you select a model, it will just use this to calculate a depth, but it wont add any uncertainty to this model (as this is 
            more likely systematic than random uncertainty that can be simulated with MC methods)

                if model is two-step:
                If two step, must also define:
                    d1: Depth to first transition in km
                    rho1: Density between surface and 1st transition
                    d2: Depth to second transition in km (from surface)
                    rho2: Density between 1st and 2nd transition

                if model is three-step:
                If three step, must also define:
                    d1: Depth to first transition in km
                    rho1: Density between surface and 1st transition
                    d2: Depth to second transition in km (from surface)
                    rho2: Density between 1st and 2nd transition
                    d3: Depth to third transition in km (from surface)
                    rho3: Density between 2nd and 3rd transition depth.

    


    Returns
    ----------------

    df_step, All_outputs,fig if plot_figure is true

    df_step: pd.DataFrame
        has 1 row for each entered sample, with mean, median, and standard deviation for each parameter

    All_outputs: pd.DataFrame
        has N_dup rows for each input, e.g. full simulation data for each sample. What is being shown in the figure
    
    fig: figure
        Figure of simulated input and output parameters for 1 sample selected by the user. 


    """
    if isinstance(T_K, float) or isinstance(T_K, int) :
        if pd.isna(T_K):
            raise TypeError("Your Input Temperature is NaN - We cant do EOS calculatoins")
    elif isinstance(T_K, pd.Series):
            if T_K.isna().any():
                raise TypeError("At least one of your Input Temperature is NaN - We cant do EOS calculatoins")
    
    if isinstance(crust_dens_kgm3, str):
        raise TypeError('Do not enter a string for crustal density, put it as a model instead')

    # Set up empty things to fill up.

    if type(CO2_dens_gcm3) is pd.Series:
        len_loop=len(CO2_dens_gcm3)
    else:
        len_loop=1



    SingleCalc_D_km = np.empty(len_loop, dtype=float)
    SingleCalc_Press_kbar = np.empty(len_loop, dtype=float)

    mean_Press_kbar = np.empty(len_loop, dtype=float)
    med_Press_kbar = np.empty(len_loop, dtype=float)
    std_Press_kbar = np.empty(len_loop, dtype=float)
    std_Press_kbar_IQR=np.empty(len_loop, dtype=float)

    mean_D_km = np.empty(len_loop, dtype=float)
    med_D_km = np.empty(len_loop, dtype=float)
    std_D_km = np.empty(len_loop, dtype=float)
    std_D_km_IQR=np.empty(len_loop, dtype=float)
    CO2_density_input=np.empty(len_loop, dtype=float)
    error_crust_dens_loop=np.empty(len_loop, dtype=float)
    error_crust_dens2_loop=np.empty(len_loop, dtype=float)
    Sample=np.empty(len_loop,  dtype=np.dtype('U100') )

    All_outputs=pd.DataFrame([])

    # Lets calculate the parameters not using any errors, to be able to plot an error bar around each point

    df_ind=convert_co2_dens_press_depth(T_K=T_K,
    CO2_dens_gcm3=CO2_dens_gcm3,
    crust_dens_kgm3=crust_dens_kgm3, output='kbar',
    g=9.81, model=model,
    d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3, EOS=EOS)



    #This loops through each fluid inclusion entered density
    for i in range(0, len_loop):
        if i % 20 == 0:
            print('working on sample number '+str(i))

        SingleCalc_D_km[i]=df_ind['Depth (km)'].iloc[i]
        SingleCalc_Press_kbar[i]=df_ind['Pressure (kbar)'].iloc[i]


        # If user has entered a pandas series for error, takes right one for each loop
        if type(error_T_K) is pd.Series:
            error_T_K=error_T_K.iloc[i]
        else:
            error_T_K=error_T_K

        if type(T_K) is pd.Series:
            T_K_i=T_K.iloc[i]
        else:
            T_K_i=T_K


        if type(CO2_dens_gcm3) is pd.Series:
            CO2_dens_gcm3_i=CO2_dens_gcm3.iloc[i]
        else:
            CO2_dens_gcm3_i=CO2_dens_gcm3


        if type(error_CO2_dens) is pd.Series:
            error_CO2_dens=error_CO2_dens.iloc[i]
        else:
            error_CO2_dens=error_CO2_dens

        if type(error_crust_dens) is pd.Series:
            error_crust_dens=error_crust_dens.iloc[i]
        else:
            error_crust_dens=error_crust_dens


        # This is the function doing the work to actually make the simulations for each variable.
        df_synthetic=calculate_temperature_density_MC(sample_i=i, N_dup=N_dup, CO2_dens_gcm3=CO2_dens_gcm3,
        T_K=T_K, error_T_K=error_T_K, error_type_T_K=error_type_T_K, error_dist_T_K=error_dist_T_K,
        error_CO2_dens=error_CO2_dens, error_type_CO2_dens=error_type_CO2_dens, error_dist_CO2_dens=error_dist_CO2_dens,
        crust_dens_kgm3=crust_dens_kgm3,  error_crust_dens=error_crust_dens, error_type_crust_dens= error_type_crust_dens, error_dist_crust_dens=error_dist_crust_dens,
    model=model)

        # Convert to densities for MC

        if model is None:
            MC_T=convert_co2_dens_press_depth(T_K=df_synthetic['T_K_with_noise'],
                                            CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                        crust_dens_kgm3=df_synthetic['crust_dens_with_noise'],
                                        d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,model=model,
                                            EOS=EOS)
        else:
            MC_T=convert_co2_dens_press_depth(T_K=df_synthetic['T_K_with_noise'],
                                            CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                            d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,
                                        model=model, EOS=EOS)





        # Check of
        if sample_ID is None:
            Sample[i]=i

        elif isinstance(sample_ID, str):
            Sample[i]=sample_ID
        else:
            Sample[i]=sample_ID.iloc[i]

        MC_T.insert(0, 'Filename', Sample[i])


        if isinstance(CO2_dens_gcm3, pd.Series):
            CO2_density_input[i]=CO2_dens_gcm3.iloc[i]
        else:
            CO2_density_input=CO2_dens_gcm3





        All_outputs=pd.concat([All_outputs, MC_T], axis=0)


        mean_Press_kbar[i]=np.nanmean(MC_T['Pressure (kbar)'])
        med_Press_kbar[i]=np.nanmedian(MC_T['Pressure (kbar)'])
        std_Press_kbar[i]=np.nanstd(MC_T['Pressure (kbar)'])
        var=MC_T['Pressure (kbar)']
        std_Press_kbar_IQR[i]=0.5*np.abs((np.percentile(var, 84) -np.percentile(var, 16)))

        mean_D_km[i]=np.nanmean(MC_T['Depth (km)'])
        med_D_km[i]=np.nanmedian(MC_T['Depth (km)'])
        std_D_km[i]=np.nanstd(MC_T['Depth (km)'])
        var=MC_T['Depth (km)']
        std_D_km_IQR[i]=0.5*np.abs((np.percentile(var, 84) -np.percentile(var, 16)))

        error_crust_dens_loop[i]=np.nanmean(df_synthetic['error_crust_dens_kgm3'])
        error_crust_dens2_loop[i]=np.nanstd(df_synthetic['error_crust_dens_kgm3'])








    df_step=pd.DataFrame(data={'Filename': Sample,
                        'CO2_dens_gcm3': CO2_density_input,
                         'SingleFI_D_km': SingleCalc_D_km,
                         'SingleFI_P_kbar': SingleCalc_Press_kbar,

                             'Mean_MC_P_kbar': mean_Press_kbar,
                         'Med_MC_P_kbar': med_Press_kbar,
                            'std_dev_MC_P_kbar': std_Press_kbar,
'std_dev_MC_P_kbar_from_percentile': std_Press_kbar_IQR,
                          'Mean_MC_D_km': mean_D_km,
                         'Med_MC_D_km': med_D_km,
                        'std_dev_MC_D_km': std_D_km,
                        'std_dev_MC_D_km_from_percentile': std_D_km_IQR,
                         'error_T_K': error_T_K,
                         'error_CO2_dens_gcm3': error_CO2_dens,
                         'error_crust_dens_kgm3': error_crust_dens_loop,
                        'T_K': T_K,
                        'CO2_dens_gcm3_input': CO2_dens_gcm3,
                        'model': model,
                        'crust_dens_kgm3':crust_dens_kgm3,
                        'EOS': EOS
                         })




    if plot_figure is True:
        df_1_sample=All_outputs.loc[All_outputs['Filename']==All_outputs['Filename'].iloc[fig_i]]
        fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, figsize=(12,8))
        fig.suptitle('Simulations for sample = ' + str(All_outputs['Filename'].iloc[fig_i]), fontsize=20)
        ax4.set_title('Calculated distribution of depths')
        ax5.set_title('Calculated distribution of pressures (MPa)')
        ax6.set_title('Calculated distribution of pressures (kbar)')


        ax1.hist(df_1_sample['MC_T_K'], color='red',  ec='k')
        ax2.hist(df_1_sample['MC_CO2_dens_gcm3'], facecolor='white', ec='k')
        ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        if model is None:
            ax3.hist(df_1_sample['input_crust_dens_kgm3'], color='salmon', ec='k')
        ax4.hist(df_1_sample['Depth (km)'], color='cornflowerblue', ec='k')
        ax5.hist(df_1_sample['Pressure (MPa)'], color='cyan', ec='k')
        ax6.hist(df_1_sample['Pressure (kbar)'], color='cyan', ec='k')

        ax4.set_xlabel('Depth (km)')
        ax5.set_xlabel('Pressure (MPa)')
        ax6.set_xlabel('Pressure (kbar)')


        if error_dist_T_K=='normal' and error_type_T_K == 'Abs':
            ax1.set_title('Input distribution Temp: Normally-distributed, 1σ =' +str(error_T_K) + ' K')
        if error_dist_T_K=='normal' and error_type_T_K == 'Perc':
            ax1.set_title('Input distribution Temp: Normally-distributed, 1σ =' +str(error_T_K) + '%')



        if error_dist_CO2_dens=='normal' and error_type_CO2_dens == 'Abs':
            ax2.set_title('Input distribution CO$_2$ density: Normally-distributed, 1σ =' +str(error_CO2_dens) + ' g/cm$^{3}$')
        if error_dist_CO2_dens=='normal' and error_type_CO2_dens == 'Perc':
            ax2.set_title('Input distribution CO$_2$ density: Normally-distributed, 1σ =' +str(error_CO2_dens) + '%')


        if error_dist_crust_dens=='normal' and error_type_crust_dens == 'Abs':
            ax3.set_title('Input Distribution Crustal density: Normally-distributed, 1σ =' +str(error_crust_dens) + 'kg/m$^{3}$')
        if error_dist_crust_dens=='normal' and error_type_crust_dens == 'Perc':
            ax3.set_title('Input distribution crustal density: Normally-distributed, 1σ =' +str(error_crust_dens) + '%')

        if error_dist_T_K=='uniform' and error_type_T_K == 'Abs':
            ax1.set_title('Input Distribution Temp=+-' +str(error_T_K))
        if error_dist_T_K=='uniform' and error_type_T_K == 'Perc':
            ax1.set_title('Input distribution Temp=+-' +str(error_T_K) + '%')





        if error_dist_CO2_dens=='uniform' and error_type_CO2_dens == 'Abs':
            ax2.set_title('Input Distribution CO$_2$ density: uniformly-distributed, +-' +str(error_CO2_dens))
        if error_dist_CO2_dens=='uniform' and error_type_CO2_dens == 'Perc':
            ax2.set_title('Input distribution CO$_2$ density=+-' +str(error_CO2_dens) + '%')


        if error_dist_crust_dens=='uniform' and error_type_crust_dens == 'Abs':
            ax3.set_title('Input distribution crustal density: uniformly-distributed, +- ' +str(error_crust_dens))
        if error_dist_crust_dens=='uniform' and error_type_crust_dens == 'Perc':
            ax3.set_title('Input distribution crustal density: uniformly-distributed, +- ' +str(error_crust_dens) + '%')



        ax1.set_xlabel('Temperature simulation (K)')
        ax2.set_xlabel('CO$_2$ density simulation (g/cm$^{3}$)')
        ax3.set_xlabel('Crustal density simulation (g/cm$^{3}$)')
        ax1.set_ylabel('# of synthetic samples')
        ax2.set_ylabel('# of synthetic samples')
        ax3.set_ylabel('# of synthetic samples')

        fig.tight_layout()

    #return df_step, All_outputs, fig

    return df_step, All_outputs,fig if 'fig' in locals() else None


## Actual functions doing the conversions
def convert_co2_dens_press_depth(EOS='SW96', T_K=None,
    CO2_dens_gcm3=None,
    crust_dens_kgm3=None, output='kbar',
    g=9.81, model=None,
    d1=None, d2=None, rho1=None, rho2=None, rho3=None, ):

    """ This function calculates pressure and depth based on input CO2 densities, 
    temperatures, and crustal density information from the user
    
    Parameters
    ------------------
    EOS: str
        'SP94' or 'SW96' - CO2 equation of state choosen

    T_K: float, pd.Series
        Temperature in Kelvin at which fluid inclusion was trapped. 

    CO2_dens_gcm3: float, pd.Series
        CO2 density of FI in g/cm3
    
    For Pressure to depth conversion choose:

    crust_dens_kgm3: float
        Crustal density in kg/m3

    OR

    model: str
        see documentation for the function 'convert_pressure_to_depth' to see more detail about model options. 
            If you select a model, it will just use this to calculate a depth, but it wont add any uncertainty to this model (as this is 
            more likely systematic than random uncertainty that can be simulated with MC methods)

                if model is two-step:
                If two step, must also define:
                    d1: Depth to first transition in km
                    rho1: Density between surface and 1st transition
                    d2: Depth to second transition in km (from surface)
                    rho2: Density between 1st and 2nd transition

                if model is three-step:
                If three step, must also define:
                    d1: Depth to first transition in km
                    rho1: Density between surface and 1st transition
                    d2: Depth to second transition in km (from surface)
                    rho2: Density between 1st and 2nd transition
                    d3: Depth to third transition in km (from surface)
                    rho3: Density between 2nd and 3rd transition depth.

    Returns
    ---------------------
    pd.DataFrame
        dataframe of pressure, depth, and input parameterss
    
    """



    # First step is to get pressure
    Pressure=calculate_P_for_rho_T(T_K=T_K,
                CO2_dens_gcm3=CO2_dens_gcm3,
                 EOS=EOS)

    # Second step is to get crustal depths

    Depth_km=convert_pressure_to_depth(P_kbar=Pressure['P_kbar'],
                crust_dens_kgm3=crust_dens_kgm3,     g=9.81, model=model,
    d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3)



    if type(Depth_km) is float:
    # Crustal density, using P=rho g H
        df=pd.DataFrame(data={'Pressure (kbar)': Pressure['P_kbar'],
                            'Pressure (MPa)': Pressure['P_MPa'],
                            'Depth (km)': Depth_km,
                            'input_crust_dens_kgm3': crust_dens_kgm3,
                            'model': model,
                            'MC_T_K': T_K,
                            'MC_CO2_dens_gcm3': CO2_dens_gcm3}, index=[0])

    else:


        df=pd.DataFrame(data={'Pressure (kbar)': Pressure['P_kbar'],
                            'Pressure (MPa)': Pressure['P_MPa'],
                            'Depth (km)': Depth_km,
                            'input_crust_dens_kgm3': crust_dens_kgm3,
                             'model': model,
                            'MC_T_K': T_K,
                            'MC_CO2_dens_gcm3': CO2_dens_gcm3})

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df




