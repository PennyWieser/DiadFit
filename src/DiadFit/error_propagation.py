
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DiadFit as pf


def convert_density_depth_Coolprop(T_K=None, CO2_dens_gcm3=None,
                                   Crust_dens_gcm3=None, output='kbar'):

    try:
        import CoolProp.CoolProp as cp
    except ImportError:
        raise RuntimeError('You havent installed CoolProp, which is required to convert FI densities to pressures. If you have python through conda, run conda install -c conda-forge coolprop in your command line')


    if type(T_K) is pd.Series:
        T_K=T_K.values
    if type(CO2_dens_gcm3) is pd.Series:
        CO2_dens_gcm3=CO2_dens_gcm3.values

    density_SI_units=CO2_dens_gcm3*1000
    P_Pa=cp.PropsSI('P', 'D', density_SI_units, 'T', T_K, 'CO2')
    P_kbar=P_Pa*10**(-8)
    if output=='kbar':
        return P_kbar
    if output=='MPa':
        return P_kbar*100

    if output=='df':

        Depth_km=10**(-3)*P_Pa/((Crust_dens_gcm3*1000)*9.81)
        if type(P_kbar) is float:
        # Crustal density, using P=rho g H
            df=pd.DataFrame(data={'Pressure (kbar)': P_kbar,
                                'Pressure (MPa)': P_kbar*100,
                                'Depth (km)': Depth_km,
                                'input_Crust_dens_gcm3': Crust_dens_gcm3,
                                'input_T_K': T_K,
                                'input_CO2_dens_gcm3': CO2_dens_gcm3}, index=[0])

        else:


            df=pd.DataFrame(data={'Pressure (kbar)': P_kbar,
                                'Pressure (MPa)': P_kbar*100,
                                'Depth (km)': Depth_km,
                                'input_Crust_dens_gcm3': Crust_dens_gcm3,
                                'input_T_K': T_K,
                                'input_CO2_dens_gcm3': CO2_dens_gcm3})


        return df



def calculate_temperature_density_MC(df=None, sample_i=1, crust_dens_gcm3=2.7, N_dup=1000,
error_T_K=30, error_type_T_K='Abs', error_dist_T_K='normal',
error_CO2_dens=0.01, error_type_CO2_dens='Abs', error_dist_CO2_dens='normal',
error_crust_dens=0.1, error_type_crust_dens='Abs', error_dist_crust_dens='normal',
plot_figure=True):


    """ Makes a dataframe of propagated errors for temperature, CO2 density and crustal density

    Parameters
    -----------
    df: pd.Dataframe
        dataframe with column headings 'T_K' with temperature in Kelvin, and 'Density_g_cm3'
        for CO2 fluid density in g/cm3

    for each variable, e.g. _T_K, _CO2_dens, _crust_dens:

    error_type_T_K: str
    error_type_CO2_dens: str
    error_type_CO2_dens:str

        'Abs' or 'Perc', Determins whether the error you feed in is an absolute error, or a percentage error


    error_T_K: float or int
    error_CO2_dens: float or int
    error_crust_dens: float or int
        Magnitude of error

    error_dist_T_K: str
    error_type_CO2_dens: str
    error_type_CO2_dens: str
        'normal' or 'uniform', determins whether error is normally distributed or


    crust_density: float or int
        Density of the crust in g/cm3

    Returns
    -----------
    pd.DataFrame: Columns for new simulated data, along with all input variables.


    """

    df_c=df.copy()

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

    T_K_with_noise=Noise_to_add_T_K+df['T_K'].iloc[sample_i]
    T_K_with_noise[T_K_with_noise < 0.0001] = 0.0001


    # CO2 error distribution

    if error_type_CO2_dens=='Abs':
        error_CO2_dens=error_CO2_dens
    if error_type_CO2_dens =='Perc':
        error_CO2_dens=df_c['CO2_dens'].iloc[sample_i]*error_CO2_dens/100
    if error_dist_CO2_dens=='normal':
        Noise_to_add_CO2_dens = np.random.normal(0, error_CO2_dens, N_dup)
    if error_dist_CO2_dens=='uniform':
        Noise_to_add_CO2_dens = np.random.uniform(- error_CO2_dens, +
                                                      error_CO2_dens, N_dup)

    CO2_dens_with_noise=Noise_to_add_CO2_dens+df_c['Density_g_cm3'].iloc[sample_i]
    CO2_dens_with_noise[CO2_dens_with_noise < 0.0001] = 0.0001

    # Crustal density noise

    if error_type_crust_dens=='Abs':
        error_crust_dens=error_crust_dens
    if error_type_crust_dens =='Perc':
        error_crust_dens=FIs['crust_dens'].iloc[sample_i]*error_crust_dens/100
    if error_dist_crust_dens=='normal':
        Noise_to_add_crust_dens = np.random.normal(0, error_crust_dens, N_dup)
    if error_dist_crust_dens=='uniform':
        Noise_to_add_crust_dens = np.random.uniform(- error_crust_dens, +
                                                      error_crust_dens, N_dup)

    crust_dens_with_noise=Noise_to_add_crust_dens+crust_dens_gcm3
    crust_dens_with_noise[crust_dens_with_noise < 0.0001] = 0.0001

    df_out=pd.DataFrame(data={'T_K_with_noise': T_K_with_noise,
                              'CO2_dens_with_noise': CO2_dens_with_noise,
                              'crust_dens_with_noise': crust_dens_with_noise,
                               'T_K': df['T_K'].iloc[sample_i],
                              'Density_g_cm3': df['Density_g_cm3'].iloc[sample_i],
                              'Crustal Density': crust_dens_gcm3,
                              'error_T_K': error_T_K,
                              'error_type_T_K': error_type_T_K,
                              'error_dist_T_K': error_dist_T_K,
                              'error_CO2_dens': error_CO2_dens,
                              'error_type_CO2_dens': error_type_CO2_dens,
                              'error_dist_CO2_dens': error_dist_CO2_dens,
                              'error_crust_dens': error_crust_dens,
                              'error_type_crust_dens': error_type_crust_dens,
                              'error_dist_crust_dens': error_dist_crust_dens,
                             })

    if plot_figure is True:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13,5))
        ax1.hist(T_K_with_noise, color='red',  ec='k')
        ax2.hist(CO2_dens_with_noise, color='black', ec='k')
        ax3.hist(crust_dens_with_noise, color='salmon', ec='k')

        if error_dist_T_K=='normal' and error_type_T_K == 'Abs':
            ax1.set_title('Normally-distributed, 1σ =' +str(error_T_K))
        if error_dist_T_K=='normal' and error_type_T_K == 'Perc':
            ax1.set_title('Normally-distributed, 1σ =' +str(error_T_K) + '%')



        if error_dist_CO2_dens=='normal' and error_type_CO2_dens == 'Abs':
            ax2.set_title('Normally-distributed, 1σ =' +str(error_CO2_dens))
        if error_dist_CO2_dens=='normal' and error_type_CO2_dens == 'Perc':
            ax2.set_title('Normally-distributed, 1σ =' +str(error_CO2_dens) + '%')


        if error_dist_crust_dens=='normal' and error_type_crust_dens == 'Abs':
            ax3.set_title('Normally-distributed, 1σ =' +str(error_crust_dens))
        if error_dist_crust_dens=='normal' and error_type_crust_dens == 'Perc':
            ax3.set_title('Normally-distributed, 1σ =' +str(error_crust_dens) + '%')

        if error_dist_T_K=='uniform' and error_type_T_K == 'Abs':
            ax1.set_title('+-' +str(error_T_K))
        if error_dist_T_K=='uniform' and error_type_T_K == 'Perc':
            ax1.set_title('+-' +str(error_T_K) + '%')





        if error_dist_CO2_dens=='uniform' and error_type_CO2_dens == 'Abs':
            ax2.set_title('uniformly-distributed, +-' +str(error_CO2_dens))
        if error_dist_CO2_dens=='uniform' and error_type_CO2_dens == 'Perc':
            ax2.set_title('+-' +str(error_CO2_dens) + '%')


        if error_dist_crust_dens=='uniform' and error_type_crust_dens == 'Abs':
            ax3.set_title('uniformly-distributed, +- ' +str(error_crust_dens))
        if error_dist_crust_dens=='uniform' and error_type_crust_dens == 'Perc':
            ax3.set_title('uniformly-distributed, +- ' +str(error_crust_dens) + '%')





        ax1.set_xlabel('Temperature simulation (K)')
        ax2.set_xlabel('Density simulation (g/cm3)')
        ax3.set_xlabel('Crustal density simulation (g/cm3)')
        ax1.set_ylabel('# of synthetic samples')


    return df_out


crust_dens_gcm3=2.7

def loop_all_FI_MC(df=None, sample_ID=None, crust_dens_gcm3=2.7, N_dup=1000,
                error_T_K=30, error_CO2_dens=0.005, error_crust_dens=0.1,
                error_type_T_K='Abs', error_dist_T_K='normal',
                error_type_CO2_dens='Abs', error_dist_CO2_dens='normal',
                error_type_crust_dens='Abs',
                error_dist_crust_dens='uniform',
                plot_figure=False):

    # Set up mean and standard deviation
    SingleCalc_D_km = np.empty(len(df), dtype=float)
    SingleCalc_Press_kbar = np.empty(len(df), dtype=float)

    mean_Press_kbar = np.empty(len(df), dtype=float)
    med_Press_kbar = np.empty(len(df), dtype=float)
    std_Press_kbar = np.empty(len(df), dtype=float)

    mean_D_km = np.empty(len(df), dtype=float)
    med_D_km = np.empty(len(df), dtype=float)
    std_D_km = np.empty(len(df), dtype=float)
    Sample=np.empty(len(df),  dtype=np.dtype('U100') )





    for i in range(0, len(df)):



        # Working out the errors for each loop

        if type(error_T_K) is pd.Series:
            error_T_K=error_T_K.iloc[i]
        else:
            error_T_K=error_T_K

        if type(error_CO2_dens) is pd.Series:
            error_CO2_dens=error_CO2_dens.iloc[i]
        else:
            error_CO2_dens=error_CO2_dens

        if type(error_crust_dens) is pd.Series:
            error_crust_dens=error_crust_dens.iloc[i]
        else:
            error_crust_dens=error_crust_dens


        # Make synthetic things

        df_synthetic=calculate_temperature_density_MC(df=df, sample_i=i,
        crust_dens_gcm3=crust_dens_gcm3, N_dup=N_dup,
    error_T_K=error_T_K, error_type_T_K=error_type_T_K, error_dist_T_K=error_dist_T_K,
    error_CO2_dens=error_CO2_dens, error_type_CO2_dens=error_type_CO2_dens, error_dist_CO2_dens=error_dist_CO2_dens,
    error_crust_dens=error_crust_dens, error_type_crust_dens= error_type_crust_dens,
    error_dist_crust_dens=error_dist_crust_dens, plot_figure=plot_figure)

        # Convert to densities for MC

        MC_T=convert_density_depth_Coolprop(T_K=df_synthetic['T_K_with_noise'],
                                        CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                       Crust_dens_gcm3=df_synthetic['crust_dens_with_noise'],
                                         output='df')



        # Singular density calculation

        Densities=convert_density_depth_Coolprop(T_K=df['T_K'].iloc[i],
                                       Crust_dens_gcm3=crust_dens_gcm3,
                    CO2_dens_gcm3=df['Density_g_cm3'].iloc[i], output='df')

        # Check of
        if sample_ID is None:
            Sample[i]=i
        else:
            Sample[i]=df[sample_ID].iloc[i]
        SingleCalc_D_km[i]=Densities['Depth (km)']
        SingleCalc_Press_kbar[i]=Densities['Pressure (kbar)']


        mean_Press_kbar[i]=np.nanmean(MC_T['Pressure (kbar)'])
        med_Press_kbar[i]=np.nanmedian(MC_T['Pressure (kbar)'])
        std_Press_kbar[i]=np.nanstd(MC_T['Pressure (kbar)'])

        mean_D_km[i]=np.nanmean(MC_T['Depth (km)'])
        med_D_km[i]=np.nanmedian(MC_T['Depth (km)'])
        std_D_km[i]=np.nanstd(MC_T['Depth (km)'])

    df_step=pd.DataFrame(data={'Filename': Sample,
                         'SingleFI_D_km': SingleCalc_D_km,
                        'std_dev_MC_D_km': std_D_km,
                         'SingleFI_P_kbar': SingleCalc_Press_kbar,
                            'std_dev_MC_P_kbar': std_Press_kbar,
                             'Mean_MC_P_kbar': mean_Press_kbar,
                         'Med_MC_P_kbar': med_Press_kbar,

                          'Mean_MC_D_km': mean_D_km,
                         'Med_MC_D_km': med_D_km,
                         'error_T_K': error_T_K,
                         'error_CO2_dens': error_CO2_dens,
                         'error_crust_dens': error_crust_dens

                         })
    return df_step