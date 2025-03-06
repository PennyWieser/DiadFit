
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DiadFit as pf
from tqdm import tqdm
from DiadFit.density_depth_crustal_profiles import *
from DiadFit.CO2_EOS import *
# This gets us the functions for Monte Carloing
from DiadFit.CO2_in_bubble_error import *
import multiprocessing as mp
np.random.seed(42)

## Microthermometry error propagation




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
    Std_density_gas=np.zeros(len_loop)
    Std_density_liq=np.zeros(len_loop)
    Mean_density_gas=np.zeros(len_loop)
    Mean_density_liq=np.zeros(len_loop)
    Std_density_gas_IQR=np.zeros(len_loop)
    Std_density_Liq_IQR=np.zeros(len_loop)
    Sample=np.zeros(len_loop,  dtype=np.dtype('U100') )

    for i in range(0, len_loop):

        # If user has entered a pandas series for error, takes right one for each loop
        error_T_h_C_i=get_value(error_T_h_C, i)
        T_h_C_i=get_value(T_h_C, i)


        # Check of
        if Sample_ID is None:
            Sample[i]=i

        elif isinstance(Sample_ID, str):
            Sample[i]=Sample_ID
        else:
            Sample[i]=Sample_ID.iloc[i]

        Temp_MC= add_noise_to_variable(T_h_C_i, error_T_h_C_i,
        error_type_T_h_C, error_dist_T_h_C, N_dup, True, neg_threshold=0.0000000001)

        Sample2=Sample[i]
        MC_T=calculate_CO2_density_homog_T(T_h_C=Temp_MC, Sample_ID=Sample2, EOS=EOS, homog_to=homog_to, set_to_critical=set_to_critical)

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
crust_dens_kgm3=None, error_crust_dens=0, error_type_crust_dens='Abs', error_dist_crust_dens='normal', XH2O=None,
error_XH2O=None, error_type_XH2O='Abs', error_dist_XH2O='normal',
 model=None, neg_values=True):

    """
    This function generates the range of T_K, CO2 densities, XH2O and crustal densities for 1 sample 
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

    # If any of them are panda series or numpy nd array, you dont need an index
    if XH2O is None:
        if isinstance(T_K, pd.Series) or isinstance(CO2_dens_gcm3, pd.Series) or isinstance(T_K, np.ndarray) or isinstance(CO2_dens_gcm3, np.ndarray):
            df_c=pd.DataFrame(data={'T_K': T_K,
                                'CO2_dens_gcm3': CO2_dens_gcm3,
                                'XH2O': None})
        else:
            #print('here')
            df_c=pd.DataFrame(data={'T_K': T_K,
                                'CO2_dens_gcm3': CO2_dens_gcm3,
                                'XH2O': None}, index=[0])
    # IF have XH2O add here
    else:
        if isinstance(T_K, pd.Series) or isinstance(CO2_dens_gcm3, pd.Series) or isinstance(T_K, np.ndarray) or isinstance(CO2_dens_gcm3, np.ndarray) or isinstance(XH2O, np.ndarray):
            df_c=pd.DataFrame(data={'T_K': T_K,
                                'CO2_dens_gcm3': CO2_dens_gcm3,
                                'XH2O': XH2O})
        else:
            df_c=pd.DataFrame(data={'T_K': T_K,
                                'CO2_dens_gcm3': CO2_dens_gcm3,
                                'XH2O': XH2O}, index=[0])
        
        
    # you do need an index here

        
    # Temperature error distribution
    T_K_with_noise=add_noise_to_variable(T_K, error_T_K,
        error_type_T_K, error_dist_T_K,  N_dup, neg_values, neg_threshold=0.0000000001)
        
    # CO2 error distribution
    CO2_dens_with_noise=add_noise_to_variable(CO2_dens_gcm3, error_CO2_dens,
        error_type_CO2_dens, error_dist_CO2_dens,  N_dup, neg_values, neg_threshold=0.0000000001)


    
    # XH2O error distribution (if relevant)
    if XH2O is not None:
        XH2O_with_noise=add_noise_to_variable(XH2O, error_XH2O,
        error_type_XH2O, error_dist_XH2O,  N_dup, True, neg_threshold=0.0000000000)
        XH2O_with_noise[XH2O_with_noise < 0.000000] = 0.00000
        XH2O_with_noise[XH2O_with_noise > 1] = 1
        
        
        
    
    # Crustal density noise
    # First need to work out what crustal density is

    if type(crust_dens_kgm3) is float or type(crust_dens_kgm3) is int:
        crust_dens_with_noise=add_noise_to_variable(crust_dens_kgm3, error_crust_dens,
        error_type_crust_dens, error_dist_crust_dens,  N_dup, neg_values, neg_threshold=0.0000000001)
        

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
    if XH2O is not None:
        df_out['error_XH2O']=error_XH2O
        df_out['XH2O_with_noise']=XH2O_with_noise
        df_out['error_type_XH2O']=error_type_XH2O
        df_out['error_dist_XH2O']=error_dist_XH2O



    return df_out


def loop_all_FI_MC(sample_ID, CO2_dens_gcm3, T_K, N_dup=1000,
crust_dens_kgm3=None, d1=None, d2=None, rho1=None, rho2=None, rho3=None,
error_crust_dens=0, error_type_crust_dens='Abs', error_dist_crust_dens='uniform',
error_T_K=0, error_type_T_K='Abs', error_dist_T_K='normal',
error_CO2_dens=0, error_type_CO2_dens='Abs', error_dist_CO2_dens='normal',
                plot_figure=False, fig_i=0):
    """ This is a redundant function, kept around for a while for backwards compatability"""

    print('Please use the new function propagate_FI_uncertainty instead as this allows you use different EOS')

# Check for panda Series



def convert_inputs_to_series(T_K, error_T_K, CO2_dens_gcm3, error_CO2_dens_gcm3, XH2O, error_XH2O):
    # Create a list of all inputs
    inputs = [T_K, error_T_K, CO2_dens_gcm3, error_CO2_dens_gcm3, XH2O, error_XH2O]
    
    # Convert any numpy arrays in the inputs to pandas Series
    converted_inputs = [pd.Series(item) if isinstance(item, np.ndarray) else item for item in inputs]
    
    # Unpack the converted inputs back to their respective variables
    T_K, error_T_K, CO2_dens_gcm3, error_CO2_dens_gcm3, XH2O, error_XH2O = converted_inputs
    
    # Reset index only if the input is a pandas Series
    T_K = T_K.reset_index(drop=True) if isinstance(T_K, pd.Series) else T_K
    error_T_K = error_T_K.reset_index(drop=True) if isinstance(error_T_K, pd.Series) else error_T_K
    CO2_dens_gcm3 = CO2_dens_gcm3.reset_index(drop=True) if isinstance(CO2_dens_gcm3, pd.Series) else CO2_dens_gcm3
    error_CO2_dens_gcm3 = error_CO2_dens_gcm3.reset_index(drop=True) if isinstance(error_CO2_dens_gcm3, pd.Series) else error_CO2_dens_gcm3
    XH2O = XH2O.reset_index(drop=True) if isinstance(XH2O, pd.Series) else XH2O
    error_XH2O = error_XH2O.reset_index(drop=True) if isinstance(error_XH2O, pd.Series) else error_XH2O
    
    # Return the possibly converted inputs
    return T_K, error_T_K, CO2_dens_gcm3, error_CO2_dens_gcm3, XH2O, error_XH2O



def propagate_FI_uncertainty(sample_ID, CO2_dens_gcm3, T_K, multiprocess='default',  cores='default', 
EOS='SW96', N_dup=1000,
plot_figure=False, fig_i=0, 
error_CO2_dens=0, error_type_CO2_dens='Abs', error_dist_CO2_dens='normal',
 crust_dens_kgm3=None, 
 error_crust_dens=0, error_type_crust_dens='Abs', error_dist_crust_dens='uniform',
 model=None, d1=None, d2=None, rho1=None, rho2=None, rho3=None,
error_T_K=0, error_type_T_K='Abs', error_dist_T_K='normal',
XH2O=None, error_XH2O=0, error_type_XH2O='Abs', error_dist_XH2O='normal', Hloss=True,
neg_values=True, 
):

    """
    This function performs Monte Carlo simulations of uncertainty in CO2 density, input temperature, 
   crustal density and XH2O. If XH2O is specified as not None, it will use Duan and Zhang 2006 EOS
   
    It uses the function 'calculate_temperature_density_MC' to make the simulated variables, 
    and then uses this to calculate a resulting 
    pressure using the equation of state of choice

    Parameters
    -----------------
    multiprocess: 'default' or bool
        Default uses multiprocessing for Duan and Zhang (2006) but not for Span and Wanger or Sterner and Pitzer. This is because these EOS are so fast, making the multiprocess has a time penalty.
        You can override this defualt by specifying True or False.
    cores: 'default'
        By default, if multiprocess, uses default number of cores selected by multiprocess, ca noverride. 
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




    XH2O: Default Nan, else pd.Series, integer, float
        mol proportion of H2O in the fluid phase
    error_XH2O: float
        Error in XH2O
    error_type_XH2O: str
        Type of XH2O error. Can be 'Abs' or 'Perc'.
    error_dist_XH2O: str
        Distribution of XH2O error. Can be 'normal' or 'uniform'.


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

    

    neg_values: bool (default True)
        if True, negative values of input parameters  allowed, if False, makes neg values zero. 
        
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
    # Check for non pandas series as inputs 
    T_K, error_T_K, CO2_dens_gcm3, error_CO2_dens, XH2O, error_XH2O = convert_inputs_to_series(
        T_K, error_T_K, CO2_dens_gcm3, error_CO2_dens, XH2O, error_XH2O)
    
    
    if XH2O is not None:
        print('You have entered a value for XH2O, so we are now using the EOS of Duan and Zhang 200 regardless of what model you selected. If you dont want this, specify XH2O=None')
        print('Please note, the DZ2006 EOS is about 5-40X slower to run than the SP94 and SW94 EOS')
        
    if isinstance(T_K, float) or isinstance(T_K, int) :
        if pd.isna(T_K):
            raise TypeError("Your Input Temperature is NaN - We cant do EOS calculatoins")
            
    elif isinstance(T_K, pd.Series):
            if T_K.isna().any():
                raise TypeError("At least one of your Input Temperature is NaN - We cant do EOS calculatoins")
    
    if isinstance(crust_dens_kgm3, str):
        raise TypeError('Do not enter a string for crustal density, put it as a model instead')


    if type(CO2_dens_gcm3) is pd.Series:
        len_loop=len(CO2_dens_gcm3)
    else:
        len_loop=1
        
    # Setting up empty arrays filled with zeros 

    SingleCalc_D_km = np.zeros(len_loop, dtype=float)
    SingleCalc_Press_kbar = np.zeros(len_loop, dtype=float)

    mean_Press_kbar = np.zeros(len_loop, dtype=float)
    med_Press_kbar = np.zeros(len_loop, dtype=float)
    std_Press_kbar = np.zeros(len_loop, dtype=float)
    std_Press_kbar_IQR=np.zeros(len_loop, dtype=float)

    mean_D_km = np.zeros(len_loop, dtype=float)
    med_D_km = np.zeros(len_loop, dtype=float)
    std_D_km = np.zeros(len_loop, dtype=float)
    std_D_km_IQR=np.zeros(len_loop, dtype=float)
    CO2_density_input=np.zeros(len_loop, dtype=float)
    error_crust_dens_loop=np.zeros(len_loop, dtype=float)
    error_crust_dens2_loop=np.zeros(len_loop, dtype=float)
    Sample=np.zeros(len_loop,  dtype=np.dtype('U100') )

    All_outputs=pd.DataFrame([])

    # Lets calculate the parameters not using any errors, to be able to plot an error bar around each point

    df_ind=convert_co2_dens_press_depth(T_K=T_K,
    CO2_dens_gcm3=CO2_dens_gcm3,
    crust_dens_kgm3=crust_dens_kgm3, output='kbar',
    g=9.81, model=model,
    d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3, EOS=EOS, XH2O=XH2O, Hloss=Hloss)



    #This loops through each fluid inclusion entered density
    
    if multiprocess =='default':
        if XH2O is None and EOS!='DZ06':
            print('We are not using multiprocessing based on your selected EOS. You can override this by setting multiprocess=True in the function, but for SP94 and SW96 it might actually be slower')
            multiprocess=False
        else:
            if XH2O is not None:
                multiprocess=True
                print('We are using multiprocessing based on your selected EOS. You can override this by setting multiprocess=False in the function, but it might slow it down a lot')
    
    if multiprocess is False:
        


        for i in tqdm(range(0, len_loop), desc="Processing"):
    
            
            # This fills in the columns for the single calculation, e.g. no Monte-Carloing from above. 
            SingleCalc_D_km[i]=df_ind['Depth (km)'].iloc[i]
            SingleCalc_Press_kbar[i]=df_ind['Pressure (kbar)'].iloc[i]
            
            # Now lets get the value, using the function in the CO2_in_bubble_error.py file
            error_T_K_i=get_value(error_T_K, i)
            T_K_i=get_value(T_K, i)
            
            CO2_dens_gcm3_i=get_value(CO2_dens_gcm3, i)
            error_CO2_dens_i=get_value(error_CO2_dens, i)
            
            crust_dens_kgm3_i=get_value(crust_dens_kgm3, i)
            error_crust_dens_i=get_value(error_crust_dens, i)
            # Now, if XH2O was entered, and isnt None, do the same. Else keeps as None. 
            if XH2O is not None:
                error_XH2O_i=get_value(error_XH2O, i)
                XH2O_i=get_value(XH2O, i)
            
    
    
            # This is the function doing the work to actually make the simulations for each variable.
            # For each input variable, it makes a distribution. 
            
            # If XH2O is None, it doesnt return the column XH2O_with_noise, which helps ID this later. 
            if XH2O is None:
                
                df_synthetic=calculate_temperature_density_MC(N_dup=N_dup, CO2_dens_gcm3=CO2_dens_gcm3_i,
            T_K=T_K_i, error_T_K=error_T_K_i, error_type_T_K=error_type_T_K, error_dist_T_K=error_dist_T_K,
            error_CO2_dens=error_CO2_dens_i, error_type_CO2_dens=error_type_CO2_dens, error_dist_CO2_dens=error_dist_CO2_dens,
            crust_dens_kgm3=crust_dens_kgm3_i,  error_crust_dens=error_crust_dens_i, error_type_crust_dens= error_type_crust_dens, error_dist_crust_dens=error_dist_crust_dens,
        model=model, neg_values=neg_values)
        
                if model is None:
                    MC_T=convert_co2_dens_press_depth(T_K=df_synthetic['T_K_with_noise'],
                                                    CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                                crust_dens_kgm3=df_synthetic['crust_dens_with_noise'],
                                                d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,model=model,
                                                    EOS=EOS )
                else:
                    MC_T=convert_co2_dens_press_depth(T_K=df_synthetic['T_K_with_noise'],
                                                    CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                                    d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,
                                                model=model, EOS=EOS)
        
            if XH2O is not None:
                
                df_synthetic=calculate_temperature_density_MC(N_dup=N_dup, CO2_dens_gcm3=CO2_dens_gcm3_i,
            T_K=T_K_i, error_T_K=error_T_K_i, error_type_T_K=error_type_T_K, error_dist_T_K=error_dist_T_K,
            error_CO2_dens=error_CO2_dens_i, error_type_CO2_dens=error_type_CO2_dens, error_dist_CO2_dens=error_dist_CO2_dens,
            crust_dens_kgm3=crust_dens_kgm3_i,  error_crust_dens=error_crust_dens_i, error_type_crust_dens= error_type_crust_dens, error_dist_crust_dens=error_dist_crust_dens,
        model=model, XH2O=XH2O_i, error_XH2O=error_XH2O_i, error_type_XH2O=error_type_XH2O, error_dist_XH2O=error_dist_XH2O,  neg_values=neg_values)
        
                EOS='DZ06'
                

            # Convert to densities for MC
    
                if model is None:
                    MC_T=convert_co2_dens_press_depth(T_K=df_synthetic['T_K_with_noise'],
                                                    CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                                crust_dens_kgm3=df_synthetic['crust_dens_with_noise'],
                                                XH2O=df_synthetic['XH2O_with_noise'], Hloss=Hloss,
                                                d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,model=model,
                                                    EOS=EOS )
                else:
                    MC_T=convert_co2_dens_press_depth(T_K=df_synthetic['T_K_with_noise'],
                                                    CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                                XH2O=df_synthetic['XH2O_with_noise'], Hloss=Hloss,
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
    
    
    
        
        df_step=pd.DataFrame(data={'Filename': Sample,
                            'CO2_dens_gcm3': CO2_density_input,
                            'SingleCalc_D_km': SingleCalc_D_km,
                            'SingleCalc_P_kbar': SingleCalc_Press_kbar,
                                'Mean_MC_P_kbar': mean_Press_kbar,
                            'Med_MC_P_kbar': med_Press_kbar,
                                'std_dev_MC_P_kbar': std_Press_kbar,
    'std_dev_MC_P_kbar_from_percentile': std_Press_kbar_IQR,
                            'Mean_MC_D_km': mean_D_km,
                            'Med_MC_D_km': med_D_km,
                            'std_dev_MC_D_km': std_D_km,
                            'std_dev_MC_D_km_from_percentile': std_D_km_IQR,
                            'T_K_input': T_K,
                            'error_T_K': error_T_K,
                            'CO2_dens_gcm3_input': CO2_dens_gcm3,
                            'error_CO2_dens_gcm3': error_CO2_dens,
                            'crust_dens_kgm3_input':crust_dens_kgm3,
                            'error_crust_dens_kgm3': error_crust_dens_loop,
                            'model': model,
                            'EOS': EOS
                            })
                            
        if XH2O is not None:
            df_step['XH2O_input']=XH2O
            df_step['error_XH2O']=error_XH2O


            
            
    if multiprocess is True:



    # Choose number of processes
        
        if cores=='default':
            print("Number of processors: ", mp.cpu_count())
            pool = mp.Pool(mp.cpu_count())
        else: 
            pool = mp.Pool(cores)
        
        
        
        args=(sample_ID, df_ind,  CO2_dens_gcm3, T_K,  N_dup, EOS, 
    error_CO2_dens, error_type_CO2_dens, error_dist_CO2_dens,
    crust_dens_kgm3, 
    error_crust_dens, error_type_crust_dens, error_dist_crust_dens,
    model, d1, d2, rho1, rho2, rho3,
    error_T_K, error_type_T_K, error_dist_T_K,
    XH2O, error_XH2O, error_type_XH2O, error_dist_XH2O, Hloss,
    neg_values)
    
    
    
        results = [pool.apply_async(worker_function, args=(i, *args)) for i in range(len_loop)]
    
        pool.close()
        pool.join()
    
        # Collect results
        final_results = [r.get() for r in results]
    
        # Initialize DataFrames
        df_step = pd.DataFrame(data={'Filename': np.zeros(len_loop, dtype=np.dtype('U100'))})
        All_outputs = pd.DataFrame([])
    
        # Populate DataFrames
        for result, MC_T in final_results:
            i = result['i']
            for key, value in result.items():
                df_step.at[i, key] = value
        
            All_outputs = pd.concat([All_outputs, MC_T], axis=0)
                   
        
        
       




    if plot_figure is True:
        df_1_sample=All_outputs.loc[All_outputs['Filename']==df_step['Filename'].iloc[fig_i]]
        
        df_1_step=df_step.loc[df_step['Filename']==df_step['Filename'].iloc[fig_i]]
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12,8))
        
        
        fig.suptitle('Simulations for sample = ' + str(All_outputs['Filename'].iloc[fig_i]), fontsize=20)
        
        # Getting things to annotate
        # Temperature 
        if isinstance(error_T_K, pd.Series):
            error_T_K_plot = error_T_K.iloc[fig_i]
        elif isinstance(error_T_K, np.ndarray):
            error_T_K_plot = error_T_K[fig_i]
        else:
            error_T_K_plot = error_T_K
        error_T_K_plot=np.round(error_T_K_plot, 1)
        
        
        # Crustal density 
        if isinstance(error_crust_dens, pd.Series):
            error_crust_dens_plot = error_crust_dens.iloc[fig_i]
        elif isinstance(error_crust_dens, np.ndarray):
            error_crust_dens_plot = error_crust_dens[fig_i]
        else:
            error_crust_dens_plot = error_crust_dens
        error_crust_dens_plot=np.round(error_crust_dens_plot, 1)
            
        # CO2 density
        if isinstance(error_CO2_dens, pd.Series):
            error_CO2_dens_plot = error_CO2_dens.iloc[fig_i]
        elif isinstance(error_CO2_dens, np.ndarray):
            error_CO2_dens_plot = error_CO2_dens[fig_i]
        else:
            error_CO2_dens_plot = error_CO2_dens
        error_CO2_dens_plot=np.round(error_CO2_dens_plot, 3)
            
            
        # XH2O 
        if isinstance(error_XH2O, pd.Series):
            error_XH2O_plot = error_XH2O.iloc[fig_i]
        elif isinstance(error_XH2O, np.ndarray):
            error_XH2O_plot = error_XH2O[fig_i]
        else:
            error_XH2O_plot = error_XH2O
        error_XH2O_plot=np.round(error_XH2O_plot, 3)
    


        # Ax1 is temperature
        if error_dist_T_K=='normal' and error_type_T_K == 'Abs':
            ax1.set_title('Input distribution Temp: Normally-distributed, 1σ =' +str(error_T_K_plot) + ' K')
        if error_dist_T_K=='normal' and error_type_T_K == 'Perc':
            ax1.set_title('Input distribution Temp: Normally-distributed, 1σ =' +str(error_T_K_plot) + '%')
        
        if df_1_step['error_T_K'][0]!=0:
            ax1.hist(df_1_sample['MC_T_K'], color='red',  ec='k')
        else:
            ax1.plot([0, 0], [0, N_dup], '-r')
            
        # ax2 is CO2 density
        if error_dist_CO2_dens=='normal' and error_type_CO2_dens == 'Abs':
            ax2.set_title('Input distribution CO$_2$ density: Normally-distributed, 1σ =' +str(error_CO2_dens_plot) + ' g/cm$^{3}$')
        if error_dist_CO2_dens=='normal' and error_type_CO2_dens == 'Perc':
            ax2.set_title('Input distribution CO$_2$ density: Normally-distributed, 1σ =' +str(error_CO2_dens_plot) + '%')
        if df_1_step['error_CO2_dens_gcm3'][0]!=0:
            ax2.hist(df_1_sample['MC_CO2_dens_gcm3'], facecolor='white',  ec='k')
        else:
            ax2.plot([0, 0], [0, N_dup], '-r')
        ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        
        # ax3 is the crustal density error
        if error_dist_crust_dens=='normal' and error_type_crust_dens == 'Abs':
            ax3.set_title('Input Distribution Crustal density: Normally-distributed, 1σ =' +str(error_crust_dens_plot) + 'kg/m$^{3}$')
        if error_dist_crust_dens=='normal' and error_type_crust_dens == 'Perc':
            ax3.set_title('Input distribution crustal density: Normally-distributed, 1σ =' +str(error_crust_dens_plot) + '%')
        if model is None and df_1_step['error_crust_dens_kgm3'][0]!=0:
            ax3.hist(df_1_sample['MC_crust_dens_kgm3'], facecolor='white',  ec='k')
        else:
            ax3.plot([0, 0], [0, N_dup], '-r')
        ax3.ticklabel_format(useOffset=False)
            
        # ax4 is XH2O
        if error_dist_XH2O=='normal' and error_type_XH2O == 'Abs':
            ax4.set_title('Input Distribution XH2O: Normally-distributed, 1σ =' +str(error_XH2O_plot) + 'molar prop')
        if error_dist_XH2O=='normal' and error_type_XH2O == 'Perc':
            ax4.set_title('Input distribution XH2O: Normally-distributed, 1σ =' +str(error_XH2O_plot) + '%')
        if XH2O is not None and df_1_step['error_XH2O'][0]!=0:
            ax4.hist(df_1_sample['MC_XH2O'], facecolor='white',  ec='k')
        else:
            ax4.plot([0, 0], [0, N_dup], '-r')
        ax4.ticklabel_format(useOffset=False)
        
        
        # ax5 is pressure output of simulation
        ax5.hist(df_1_sample['Pressure (kbar)'], color='cyan', ec='k')
        ax5.set_xlabel('Pressure (kbar)')
            
        # ax6
        ax6.hist(df_1_sample['Depth (km)'], color='cornflowerblue', ec='k')
        ax6.set_xlabel('Depth (km)')
        
        ax1.set_ylabel('# of synthetic samples')
        ax3.set_ylabel('# of synthetic samples')
        ax5.set_ylabel('# of synthetic samples')
        
        ax1.set_xlabel('T (K)')
        ax2.set_xlabel('CO$_2$ Density (g/cm$^{3}$)')
        ax3.set_xlabel('Crustal Density (kg/m$^{3}$)')
        ax4.set_xlabel('X$_{H_{2}O}$ (molar prop)')
        
        
        fig.tight_layout()

    #return df_step, All_outputs, fig

    return df_step, All_outputs,fig if 'fig' in locals() else None
    
    




## Actual functions doing the conversions
def convert_co2_dens_press_depth(EOS='SW96', T_K=None,
    CO2_dens_gcm3=None,
    crust_dens_kgm3=None, output='kbar',
    g=9.81, model=None, XH2O=None, Hloss=True,
    d1=None, d2=None,d3=None, rho1=None, rho2=None, rho3=None, rho4=None, T_K_ambient=37+273.15 ):

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
                    rho1: Density between surface and 1st transition
                    d1: Depth to first transition in km
                    rho2: Density between 1st and 2nd transition
                    d2: Depth to second transition in km (from surface)
                    rho3: Density between 2nd and 3rd transition depth.
                    d3: Depth to third transition in km (from surface)
                    rho4: Density between below d3

    Returns
    ---------------------
    pd.DataFrame
        dataframe of pressure, depth, and input parameterss
    
    """



    # First step is to get pressure
    
    
    if XH2O is None:
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
                                        'MC_crust_dens_kgm3': crust_dens_kgm3,
                                        'model': model,
                                        'MC_T_K': T_K,
                                        'MC_CO2_dens_gcm3': CO2_dens_gcm3}, index=[0])
            
        else:
            
            
            df=pd.DataFrame(data={'Pressure (kbar)': Pressure['P_kbar'],
                                        'Pressure (MPa)': Pressure['P_MPa'],
                                        'Depth (km)': Depth_km,
                                        'MC_crust_dens_kgm3': crust_dens_kgm3,
                                        'model': model,
                                        'MC_T_K': T_K,
                                        'MC_CO2_dens_gcm3': CO2_dens_gcm3})
    
        
                    
        # Make as a dict to allow index=0 if depth is float, else not. 
        data_dict = {
            'Pressure (kbar)': [Pressure['P_kbar']],
            'Pressure (MPa)': [Pressure['P_MPa']],
            'Depth (km)': [Depth_km] if isinstance(Depth_km, float) else Depth_km,
            'MC_crust_dens_kgm3': [crust_dens_kgm3],
            'model': [model],
            'MC_T_K': [T_K],
            'MC_CO2_dens_gcm3': [CO2_dens_gcm3]
        }
        

    
        
    # If XH2O, need different outputs. 
    
    else:
        
        
        P_kbar_calc=pf.calculate_entrapment_P_XH2O(XH2O=XH2O, CO2_dens_gcm3=CO2_dens_gcm3, T_K=T_K, T_K_ambient=T_K_ambient, fast_calcs=True, Hloss=Hloss )
        
        
        Depth_km=convert_pressure_to_depth(P_kbar=P_kbar_calc,
                        crust_dens_kgm3=crust_dens_kgm3,     g=9.81, model=model,
            d1=d1, d2=d2, d3=d3, rho1=rho1, rho2=rho2, rho3=rho3, rho4=rho4)
            

            
        if type(Depth_km) is float:
                # Crustal density, using P=rho g H
            df=pd.DataFrame(data={'Pressure (kbar)': P_kbar_calc,
                                        'Pressure (MPa)': 100*P_kbar_calc,
                                        'Depth (km)': Depth_km,
                                        'MC_crust_dens_kgm3': crust_dens_kgm3,
                                        'model': model,
                                        'MC_T_K': T_K,
                                        'MC_CO2_dens_gcm3': CO2_dens_gcm3,
                                        'MC_XH2O': XH2O}, index=[0])
            
        else:
            
            
            df=pd.DataFrame(data={'Pressure (kbar)': P_kbar_calc,
                                        'Pressure (MPa)': 100*P_kbar_calc,
                                        'Depth (km)': Depth_km,
                                        'MC_crust_dens_kgm3': crust_dens_kgm3,
                                        'model': model,
                                        'MC_T_K': T_K,
                                        'MC_CO2_dens_gcm3': CO2_dens_gcm3,
                                        'MC_XH2O': XH2O})
            
            
            
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df
    
    
    ## Now lets try to parallize it

def worker_function(i, sample_ID, df_ind,  CO2_dens_gcm3, T_K,  N_dup, EOS, 
error_CO2_dens, error_type_CO2_dens, error_dist_CO2_dens,
 crust_dens_kgm3, 
 error_crust_dens, error_type_crust_dens, error_dist_crust_dens,
 model, d1, d2, rho1, rho2, rho3,
error_T_K, error_type_T_K, error_dist_T_K,
XH2O, error_XH2O, error_type_XH2O, error_dist_XH2O, Hloss,
neg_values
):
    
    
    # This fills in the columns for the single calculation, e.g. no Monte-Carloing from above. 
    SingleCalc_D_km=df_ind['Depth (km)'].iloc[i]
    SingleCalc_Press_kbar=df_ind['Pressure (kbar)'].iloc[i]
    
    # Now lets get the value, using the function in the CO2_in_bubble_error.py file
    T_K_i=get_value(T_K, i)
    error_T_K_i=get_value(error_T_K, i)
    
    
    CO2_dens_gcm3_i=get_value(CO2_dens_gcm3, i)
    error_CO2_dens_i=get_value(error_CO2_dens, i)
    
    crust_dens_kgm3_i=get_value(crust_dens_kgm3, i)
    error_crust_dens_i=get_value(error_crust_dens, i)
    # Now, if XH2O was entered, and isnt None, do the same. Else keeps as None. 
    if XH2O is not None:
        error_XH2O_i=get_value(error_XH2O, i)
        XH2O_i=get_value(XH2O, i)
    else:
        error_XH2O_i=0
    
    if sample_ID is None:
        Sample[i]=i

    elif isinstance(sample_ID, str):
        Sample=sample_ID
    else:
        Sample=sample_ID.iloc[i]
                


    # This is the function doing the work to actually make the simulations for each variable.
    # For each input variable, it makes a distribution. 
    
    # If XH2O is None, it doesnt return the column XH2O_with_noise, which helps ID this later. 
    if XH2O is None:
        
        df_synthetic=calculate_temperature_density_MC(N_dup=N_dup, CO2_dens_gcm3=CO2_dens_gcm3_i,
    T_K=T_K_i, error_T_K=error_T_K_i, error_type_T_K=error_type_T_K, error_dist_T_K=error_dist_T_K,
    error_CO2_dens=error_CO2_dens_i, error_type_CO2_dens=error_type_CO2_dens, error_dist_CO2_dens=error_dist_CO2_dens,
    crust_dens_kgm3=crust_dens_kgm3_i,  error_crust_dens=error_crust_dens_i, error_type_crust_dens= error_type_crust_dens, error_dist_crust_dens=error_dist_crust_dens,
model=model, neg_values=neg_values)

        if model is None:
            MC_T=convert_co2_dens_press_depth(T_K=df_synthetic['T_K_with_noise'],
                                            CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                        crust_dens_kgm3=df_synthetic['crust_dens_with_noise'],
                                        d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,model=model,
                                            EOS=EOS )
        else:
            MC_T=convert_co2_dens_press_depth(T_K=df_synthetic['T_K_with_noise'],
                                            CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                            d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,
                                        model=model, EOS=EOS)

    if XH2O is not None:
        
        df_synthetic=calculate_temperature_density_MC(N_dup=N_dup, CO2_dens_gcm3=CO2_dens_gcm3_i,
    T_K=T_K_i, error_T_K=error_T_K_i, error_type_T_K=error_type_T_K, error_dist_T_K=error_dist_T_K,
    error_CO2_dens=error_CO2_dens_i, error_type_CO2_dens=error_type_CO2_dens, error_dist_CO2_dens=error_dist_CO2_dens,
    crust_dens_kgm3=crust_dens_kgm3_i,  error_crust_dens=error_crust_dens_i, error_type_crust_dens= error_type_crust_dens, error_dist_crust_dens=error_dist_crust_dens,
model=model, XH2O=XH2O_i, error_XH2O=error_XH2O_i, error_type_XH2O=error_type_XH2O, error_dist_XH2O=error_dist_XH2O,  neg_values=neg_values)

        EOS='DZ06'
        

    # Convert to densities for MC

        if model is None:
            MC_T=convert_co2_dens_press_depth(T_K=df_synthetic['T_K_with_noise'],
                                            CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                        crust_dens_kgm3=df_synthetic['crust_dens_with_noise'],
                                        XH2O=df_synthetic['XH2O_with_noise'], Hloss=Hloss,
                                        d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,model=model,
                                            EOS=EOS )
        else:
            MC_T=convert_co2_dens_press_depth(T_K=df_synthetic['T_K_with_noise'],
                                            CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                        XH2O=df_synthetic['XH2O_with_noise'], Hloss=Hloss,
                                            d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,
                                        model=model, EOS=EOS)





    # Check of
    if sample_ID is None:
        Sample=i

    elif isinstance(sample_ID, str):
        Sample=sample_ID
    else:
        Sample=sample_ID.iloc[i]

    MC_T.insert(0, 'Filename', Sample)



    CO2_density_input=CO2_dens_gcm3_i




# Collect results
    result = {
        'i': i,
        'Filename': Sample,
         'CO2_density_input': CO2_dens_gcm3_i,
        'SingleCalc_D_km': df_ind['Depth (km)'].iloc[i],
        'SingleCalc_P_kbar': df_ind['Pressure (kbar)'].iloc[i],
        'Mean_MC_P_kbar': np.nanmean(MC_T['Pressure (kbar)']),
        'Med_MC_P_kbar': np.nanmedian(MC_T['Pressure (kbar)']),
        'std_dev_MC_P_kbar': np.nanstd(MC_T['Pressure (kbar)']),
        'std_dev_MC_P_kbar_from_percentile': 0.5*np.abs((np.percentile(MC_T['Pressure (kbar)'], 84) -np.percentile(MC_T['Pressure (kbar)'], 16))),
        'Mean_MC_D_km': np.nanmean(MC_T['Depth (km)']),
        'Med_MC_D_km': np.nanmedian(MC_T['Depth (km)']),
        'std_dev_MC_D_km': np.nanstd(MC_T['Depth (km)']),
        'std_dev_MC_D_km_from_percentile': 0.5*np.abs((np.percentile(MC_T['Depth (km)'], 84) -np.percentile(MC_T['Depth (km)'], 16))),
        'T_K_input': T_K_i,
        'error_T_K': error_T_K_i,
        'CO2_dens_gcm3_input': CO2_dens_gcm3_i,
        'error_CO2_dens_gcm3': error_CO2_dens_i,
        'crust_dens_kgm3_input':crust_dens_kgm3_i,
        'error_crust_dens_kgm3': error_crust_dens_i,
        'model': model,
        'EOS': EOS
                            
    }
    if XH2O is not None:
        result['XH2O_input']=XH2O_i
        result['error_XH2O']=error_XH2O_i

    return result, MC_T
    
    
        
        
        
        
        
 