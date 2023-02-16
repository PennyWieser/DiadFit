
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

    T_h_C_with_noise=Noise_to_add_T_h_C+df_c['T_h_C'].iloc[sample_i]

    return T_h_C_with_noise


def propagate_microthermometry_uncertainty(T_h_C, Sample_ID=None, sample_i=0, error_T_h_C=0.3, N_dup=1000,
        error_dist_T_h_C='uniform', error_type_T_h_C='Abs', len_loop=1, EOS='SW96', T_K=None, homog_to=None):

    # Set up empty things to fill up.

    if type(T_h_C) is pd.Series:
        len_loop=len(T_h_C)
    else:
        len_loop=1


    All_outputs=pd.DataFrame([])
    Std_density_gas=np.empty(len_loop)
    Std_density_liq=np.empty(len_loop)
    Mean_density_gas=np.empty(len_loop)
    Mean_density_liq=np.empty(len_loop)
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
        sample_i=0, error_T_h_C=error_T_h_C, N_dup=N_dup,
        error_dist_T_h_C=error_dist_T_h_C, error_type_T_h_C=error_type_T_h_C, len_loop=1)

        Sample2=Sample[i]
        MC_T=calculate_CO2_density_homog_T(T_h_C=Temp_MC, Sample_ID=Sample2, EOS=EOS, homog_to=homog_to)




        # MC for each FI
        All_outputs=pd.concat([All_outputs, MC_T], axis=0)

        # get av and mean
        Std_density_gas[i]=np.nanstd(MC_T['Gas_gcm3'])
        Std_density_liq[i]=np.nanstd(MC_T['Liq_gcm3'])
        Mean_density_gas[i]=np.nanmean(MC_T['Gas_gcm3'])
        Mean_density_liq[i]=np.nanmean(MC_T['Liq_gcm3'])



    Av_outputs=pd.DataFrame(data={'Sample_ID': Sample,
                                      'Mean_density_Gas_gcm3': Mean_density_gas,
                                      'Std_density_Gas_gcm3': Std_density_gas,
                                       'Mean_density_Liq_gcm3': Mean_density_liq,
                                      'Std_density_Liq_gcm3': Std_density_liq,
                                      'error_T_h_C': error_T_h_C})

    # if T_K is not None:
    #
    #     Press=calculate_P_for_rho_T(density_gcm3=All_outputs['Bulk_gcm3'],
    #     T_K=T_K, EOS=EOS)
    #     All_outputs['P_kbar']=Press
    #
    #     Av_outputs['Mean_P_kbar']=np.nanmean(Press)
    #     Av_outputs['std_P_kbar']=np.nanstd(Press)




    return Av_outputs, All_outputs




def calculate_temperature_density_MC(sample_i=1,  N_dup=1000, T_K=None, CO2_density_gcm3=None,
crust_dens_kgm3=None, d1=None, d2=None, rho1=None, rho2=None, rho3=None,
error_T_K=0, error_type_T_K='Abs', error_dist_T_K='normal',
error_CO2_dens=0, error_type_CO2_dens='Abs', error_dist_CO2_dens='normal',
error_crust_dens=0, error_type_crust_dens='Abs', error_dist_crust_dens='normal',
plot_figure=True, len_loop=1):

    """ Calculate temperature, CO2 density, and crustal density for a given sample using Monte Carlo simulations with added noise.

    Parameters
    ----------------

    N_dup (int, optional):
        The number of simulations to run. Default is 1000.

    T_K (float, optional):
        The temperature of the sample in degrees Kelvin.

    len_loop: float
        Number of samples you are doing, if only 1 for loop, uses an index.

    error_T_K (float, optional):
        The error in the temperature measurement. Default is 0.

    error_type_T_K (str, optional):
        The type of error in the temperature measurement, either 'Abs' for absolute error or 'Perc' for percent error. Default is 'Abs'.

    error_dist_T_K (str, optional):
        The distribution of error in the CO2 density measurement, either 'normal' or 'uniform'

    CO2_density_gcm3 (float, optional):
        The CO2 density of the CO2 fluid in g/cm^3.

    error_CO2_dens (float, optional):
        The error in the CO2 density measurement. Default is 0.

    error_type_CO2_dens (str, optional):
        The type of error in the CO2 density measurement, either 'Abs' for absolute error or 'Perc' for percent error. Default is 'Abs'.

    error_dist_CO2_dens (str, optional):
        The distribution of error in the CO2 density measurement, either 'normal' or 'uniform'


    crust_dens_kgm3 (float, optional) or str
        if float, The crustal density of the sample in kg/m^3.
        if str, either a density model ('ryan_lerner, two step etc')
        if two-step or three-step:
            rho1 - density in kg/m3 down to d1
            rho2 - density in kg/m3 between d1 and d2
            rho3 - density in kg/m3 between d2 and d3
            d1 - depth in km to first density transition
            d2 - depth in km to second density transition

    error_crust_dens (float, optional):
        The error in the crustal density measurement. Default is 0.

    error_type_crust_dens(str, optional):
        The type of error in the crustal density measurement, either 'Abs' for absolute error or 'Perc' for percent         error. Default is 'Abs'.

    error_dist_crust_dens(str, optional):
        The distribution of error in the CO2 density measurement, either 'normal' or 'uniform'.

    plot_figure (bool):
        if True, plots a figure of the distribution of different variables.

    Returns
    ----------------
    pd.DataFrame
        Dataframe with N_dup rows, and calculated T, densities, as well as input parameters

    """


    if len_loop==1:
        df_c=pd.DataFrame(data={'T_K': T_K,
                            'CO2_density_gcm3': CO2_density_gcm3}, index=[0])
    else:
        df_c=pd.DataFrame(data={'T_K': T_K,
                            'CO2_density_gcm3': CO2_density_gcm3})


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
            error_crust_dens=crust_dens_kgm3*error_crust_dens/100
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
                                'error_crust_dens_kgm3': error_crust_dens,
                                'error_type_crust_dens': error_type_crust_dens,
                                'error_dist_crust_dens': error_dist_crust_dens,
                                })



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
                                'error_crust_dens_kgm3': error_crust_dens,
                                'error_type_crust_dens': error_type_crust_dens,
                                'error_dist_crust_dens': error_dist_crust_dens,
                                })


    return df_out




def loop_all_FI_MC(sample_ID, CO2_density_gcm3, T_K, N_dup=1000,
crust_dens_kgm3=None, d1=None, d2=None, rho1=None, rho2=None, rho3=None,
error_crust_dens=0, error_type_crust_dens='Abs', error_dist_crust_dens='uniform',
error_T_K=0, error_type_T_K='Abs', error_dist_T_K='normal',
error_CO2_dens=0, error_type_CO2_dens='Abs', error_dist_CO2_dens='normal',
                plot_figure=False, fig_i=0):

    """
    Loop through all fluid inclusions in a dataset and run Monte Carlo simulations for
    temperature, CO2 density, and crustal density

    sample_ID: pd.Series
        Panda series of sample names. E.g., select a column from your dataframe (df['sample_name'])

    CO2_density_gcm3: pd.Series, integer or float
        CO2 densities in g/cm3 to perform calculations with. Can be a column from your dataframe (df['density_g_cm3']), or a single value (e.g.., 0.2)

    T_K: pd.Series, integer, float
        Temperature in Kelvin. Can be a column from your dataframe (df['T_K']), or a single value (e.g.., 1500)

    N_dup (int, optional):
        The number of simulations to run. Default is 1000.


    error_T_K (float, optional):
        The error in the temperature measurement. Default is 0.

    error_type_T_K (str, optional):
        The type of error in the temperature measurement, either 'Abs' for absolute error or 'Perc' for percent error. Default is 'Abs'.

    error_dist_T_K (str, optional):
        The distribution of error in the CO2 density measurement, either 'normal' or 'uniform'

    error_CO2_dens (float, optional):
        The error in the CO2 density measurement. Default is 0.

    error_type_CO2_dens (str, optional):
        The type of error in the CO2 density measurement, either 'Abs' for absolute error or 'Perc' for percent error. Default is 'Abs'.

    error_dist_CO2_dens (str, optional):
        The distribution of error in the CO2 density measurement, either 'normal' or 'uniform'

    crust_dens_kgm3 (float, optional) or str
        if float, The crustal density of the sample in kg/m^3.
        if str, either a density model ('ryan_lerner, two step etc')
        if two-step or three-step:
            rho1 - density in kg/m3 down to d1
            rho2 - density in kg/m3 between d1 and d2
            rho3 - density in kg/m3 between d2 and d3
            d1 - depth in km to first density transition
            d2 - depth in km to second density transition

    error_crust_dens (float, optional):
        The error in the crustal density measurement. Default is 0.

    error_type_crust_dens(str, optional):
        The type of error in the crustal density measurement, either 'Abs' for absolute error or 'Perc' for percent         error. Default is 'Abs'.

    error_dist_crust_dens(str, optional):
        The distribution of error in the CO2 density measurement, either 'normal' or 'uniform'.

    plot_figure (bool):
        if True, plots a figure of the distribution of different variables.

    Returns
    ----------------
    pd.DataFrame
        Dataframe with N_dup rows, and calculated T, densities, as well as input parameters



    """

    # Set up empty things to fill up.

    if type(CO2_density_gcm3) is pd.Series:
        len_loop=len(CO2_density_gcm3)
    else:
        len_loop=1



    SingleCalc_D_km = np.empty(len_loop, dtype=float)
    SingleCalc_Press_kbar = np.empty(len_loop, dtype=float)

    mean_Press_kbar = np.empty(len_loop, dtype=float)
    med_Press_kbar = np.empty(len_loop, dtype=float)
    std_Press_kbar = np.empty(len_loop, dtype=float)

    mean_D_km = np.empty(len_loop, dtype=float)
    med_D_km = np.empty(len_loop, dtype=float)
    std_D_km = np.empty(len_loop, dtype=float)
    CO2_density_input=np.empty(len_loop, dtype=float)
    error_crust_dens_loop=np.empty(len_loop, dtype=float)
    error_crust_dens2_loop=np.empty(len_loop, dtype=float)
    Sample=np.empty(len_loop,  dtype=np.dtype('U100') )

    All_outputs=pd.DataFrame([])



    #This loops through each fluid inclusion
    for i in range(0, len_loop):
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
     plot_figure=plot_figure, len_loop=len_loop)

        # Convert to densities for MC


        MC_T=convert_co2_dens_press_depth(T_K=df_synthetic['T_K_with_noise'],
                                        CO2_dens_gcm3=df_synthetic['CO2_dens_with_noise'],
                                       crust_dens_kgm3=df_synthetic['crust_dens_with_noise'],
                                       d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3,
                                         output='df')



        # Singular density calculation

        Densities=convert_co2_dens_press_depth(T_K=T_K_i,
                                       crust_dens_kgm3=crust_dens_kgm3,
                    CO2_dens_gcm3=CO2_density_gcm3_i, output='df', d1=d1, d2=d2, rho1=rho1, rho2=rho2, rho3=rho3)



        # Check of
        if sample_ID is None:
            Sample[i]=i

        elif isinstance(sample_ID, str):
            Sample[i]=sample_ID
        else:
            Sample[i]=sample_ID.iloc[i]

        MC_T.insert(0, 'Filename', Sample[i])


        if isinstance(CO2_density_gcm3, pd.Series):
            CO2_density_input[i]=CO2_density_gcm3.iloc[i]
        else:
            CO2_density_input=CO2_density_gcm3
        SingleCalc_D_km[i]=Densities['Depth (km)']
        SingleCalc_Press_kbar[i]=Densities['Pressure (kbar)']

        All_outputs=pd.concat([All_outputs, MC_T], axis=0)


        mean_Press_kbar[i]=np.nanmean(MC_T['Pressure (kbar)'])
        med_Press_kbar[i]=np.nanmedian(MC_T['Pressure (kbar)'])
        std_Press_kbar[i]=np.nanstd(MC_T['Pressure (kbar)'])

        mean_D_km[i]=np.nanmean(MC_T['Depth (km)'])
        med_D_km[i]=np.nanmedian(MC_T['Depth (km)'])
        std_D_km[i]=np.nanstd(MC_T['Depth (km)'])

        error_crust_dens_loop[i]=np.nanmean(df_synthetic['error_crust_dens_kgm3'])
        error_crust_dens2_loop[i]=np.nanstd(df_synthetic['error_crust_dens_kgm3'])

    df_step=pd.DataFrame(data={'Filename': Sample,
                        'CO2_density_gcm3': CO2_density_input,
                         'SingleFI_D_km': SingleCalc_D_km,
                         'SingleFI_P_kbar': SingleCalc_Press_kbar,

                             'Mean_MC_P_kbar': mean_Press_kbar,
                         'Med_MC_P_kbar': med_Press_kbar,
                            'std_dev_MC_P_kbar': std_Press_kbar,

                          'Mean_MC_D_km': mean_D_km,
                         'Med_MC_D_km': med_D_km,
                        'std_dev_MC_D_km': std_D_km,
                         'error_T_K': error_T_K,
                         'error_CO2_dens_gcm3': error_CO2_dens,
                         'error_crust_dens_kgm3': error_crust_dens_loop,

                         })

    if plot_figure is True:
        df_1_sample=All_outputs.loc[All_outputs['Filename']==All_outputs['Filename'].iloc[fig_i]]
        fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, figsize=(12,8))
        fig.suptitle('Simulations for sample = ' + str(All_outputs['Filename'].iloc[fig_i]))
        ax4.set_title('Calculated distribution of depths')
        ax5.set_title('Calculated distribution of pressures (MPa)')
        ax6.set_title('Calculated distribution of pressures (kbar)')


        ax1.hist(df_1_sample['input_T_K'], color='red',  ec='k')
        ax2.hist(df_1_sample['input_CO2_dens_gcm3'], facecolor='white', ec='k')
        ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
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

    return df_step, All_outputs, fig