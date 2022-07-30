
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DiadFit as pf

from DiadFit.density_depth_crustal_profiles import *




def calculate_temperature_density_MC(sample_i=1,  N_dup=1000, T_K=None, CO2_density_gcm3=None,
crust_dens_kgm3=None, d1=None, d2=None, rho1=None, rho2=None, rho3=None,
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

    #df_c=df.copy()
    df_c=pd.DataFrame(data={'T_K': T_K,
                            'CO2_density_gcm3': CO2_density_gcm3

    })

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
        error_CO2_dens=df_c['CO2_density_gcm3'].iloc[sample_i]*error_CO2_dens/100
    if error_dist_CO2_dens=='normal':
        Noise_to_add_CO2_dens = np.random.normal(0, error_CO2_dens, N_dup)
    if error_dist_CO2_dens=='uniform':
        Noise_to_add_CO2_dens = np.random.uniform(- error_CO2_dens, +
                                                      error_CO2_dens, N_dup)

    CO2_dens_with_noise=Noise_to_add_CO2_dens+df_c['CO2_density_gcm3'].iloc[sample_i]
    CO2_dens_with_noise[CO2_dens_with_noise < 0.0001] = 0.0001

    # Crustal density noise
    # First need to work out what crustal density is

    if type(crust_dens_kgm3) is float or type(crust_dens_kgm3) is int:
        # This is the simplicest scenario, just makes a distribution of pressures

        if error_type_crust_dens=='Abs':
            error_crust_dens=error_crust_dens
        if error_type_crust_dens =='Perc':
            error_crust_dens=crust_dens_kgm3.iloc[sample_i]*error_crust_dens/100
        if error_dist_crust_dens=='normal':
            Noise_to_add_crust_dens = np.random.normal(0, error_crust_dens, N_dup)
        if error_dist_crust_dens=='uniform':
            Noise_to_add_crust_dens = np.random.uniform(- error_crust_dens, +
                                                        error_crust_dens, N_dup)

        crust_dens_with_noise=Noise_to_add_crust_dens+crust_dens_kgm3
        crust_dens_with_noise[crust_dens_with_noise < 0.0001] = 0.0001


        df_out=pd.DataFrame(data={'T_K_with_noise': T_K_with_noise,
                                'CO2_dens_with_noise': CO2_dens_with_noise,
                                'crust_dens_with_noise': crust_dens_with_noise,
                                'T_K': df_c['T_K'].iloc[sample_i],
                                'CO2_density_gcm3': df_c['CO2_density_gcm3'].iloc[sample_i],
                                'Crustal Density_kg_m3': crust_dens_kgm3,
                                'model': None,
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

    # elif crust_dens_kgm3 == "two-step":
    #     df_out=1
    #
    # elif crust_dens_kgm3 == "three-step":
    #     df_out=1

    else:
        if error_crust_dens>0:
            raise Exception('You cannot use a crustal density model with an error in density due to ambiguity about how to apply this (and because errors will be systematic not random). This variable only works for a constant crustal density. Set error_crust_dens=0')
        # For all other models



        df_out=pd.DataFrame(data={'T_K_with_noise': T_K_with_noise,
                                'CO2_dens_with_noise': CO2_dens_with_noise,
                                'crust_dens_with_noise': crust_dens_kgm3,
                                'T_K': df_c['T_K'].iloc[sample_i],
                                'CO2_density_gcm3': df_c['CO2_density_gcm3'].iloc[sample_i],
                                'Crustal Density_kg_m3': crust_dens_kgm3,
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




def loop_all_FI_MC(df=None, sample_ID=None, N_dup=1000, T_K=None, CO2_density_gcm3=None,
crust_dens_kgm3=None, d1=None, d2=None, rho1=None, rho2=None, rho3=None,
error_crust_dens=0.1, error_type_crust_dens='Abs', error_dist_crust_dens='uniform',
error_T_K=30, error_type_T_K='Abs', error_dist_T_K='normal',
error_CO2_dens=0.005, error_type_CO2_dens='Abs', error_dist_CO2_dens='normal',
                plot_figure=False):

    """ This function propagates errors in CO2 density, temperature and crustal density into pressure and depth distributions
    """

    # Set up empty things to fill up.
    SingleCalc_D_km = np.empty(len(CO2_density_gcm3), dtype=float)
    SingleCalc_Press_kbar = np.empty(len(CO2_density_gcm3), dtype=float)

    mean_Press_kbar = np.empty(len(CO2_density_gcm3), dtype=float)
    med_Press_kbar = np.empty(len(CO2_density_gcm3), dtype=float)
    std_Press_kbar = np.empty(len(CO2_density_gcm3), dtype=float)

    mean_D_km = np.empty(len(CO2_density_gcm3), dtype=float)
    med_D_km = np.empty(len(CO2_density_gcm3), dtype=float)
    std_D_km = np.empty(len(CO2_density_gcm3), dtype=float)
    Sample=np.empty(len(CO2_density_gcm3),  dtype=np.dtype('U100') )




    #This loops through each fluid inclusion
    for i in range(0, len(CO2_density_gcm3)):
        if i % 20 == 0:
            print('working on sample number '+str(i))

        # If user has entered a pandas series for error, takes right one for each loop
        if type(error_T_K) is pd.Series:
            error_T_K=error_T_K.iloc[i]
        else:
            error_T_K=error_T_K

        if type(T_K) is pd.Series:
            T_K_i=T_K.iloc[i]
        else:
            T_K_i=T_K


        if type(CO2_density_gcm3) is pd.Series:
            CO2_density_gcm3_i=CO2_density_gcm3.iloc[i]
        else:
            CO2_density_gcm3_i=CO2_density_gcm3


        if type(error_CO2_dens) is pd.Series:
            error_CO2_dens=error_CO2_dens.iloc[i]
        else:
            error_CO2_dens=error_CO2_dens

        if type(error_crust_dens) is pd.Series:
            error_crust_dens=error_crust_dens.iloc[i]
        else:
            error_crust_dens=error_crust_dens


        # This is the function doing the work to actually make the simulations for each variable.
        df_synthetic=calculate_temperature_density_MC(sample_i=i, N_dup=N_dup, CO2_density_gcm3=CO2_density_gcm3,
        T_K=T_K, error_T_K=error_T_K, error_type_T_K=error_type_T_K, error_dist_T_K=error_dist_T_K,
        error_CO2_dens=error_CO2_dens, error_type_CO2_dens=error_type_CO2_dens, error_dist_CO2_dens=error_dist_CO2_dens,
        crust_dens_kgm3=crust_dens_kgm3,  error_crust_dens=error_crust_dens, error_type_crust_dens= error_type_crust_dens, error_dist_crust_dens=error_dist_crust_dens,
        d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,
     plot_figure=plot_figure)

        # Convert to densities for MC


        MC_T=convert_co2_density_depth_Coolprop(T_K=df_synthetic['T_K_with_noise'],
                                        CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                       crust_dens_kgm3=df_synthetic['crust_dens_with_noise'],
                                       d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,
                                         output='df')



        # Singular density calculation

        Densities=convert_co2_density_depth_Coolprop(T_K=T_K_i,
                                       crust_dens_kgm3=crust_dens_kgm3,
                    CO2_dens_gcm3=CO2_density_gcm3_i, output='df', d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3)

        # Check of
        if sample_ID is None:
            Sample[i]=i
        else:
            Sample[i]=sample_ID.iloc[i]
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