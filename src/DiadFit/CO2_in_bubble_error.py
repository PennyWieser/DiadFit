import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def propagate_CO2_in_bubble(*, N_dup=1000, sample_ID, vol_perc_bub, melt_dens_kgm3, CO2_bub_dens_gcm3,
error_vol_perc_bub=0, error_type_vol_perc_bub='Abs', error_dist_vol_perc_bub='normal',
error_CO2_bub_dens_gcm3=0, error_type_CO2_bub_dens_gcm3='Abs', error_dist_CO2_bub_dens_gcm3='normal',
error_melt_dens_kgm3=0, error_type_melt_dens_kgm3='Abs', error_dist_melt_dens_kgm3='normal',
plot_figure=True, fig_i=0, neg_values=True):



    """ This function propagates uncertainty in reconstruction of melt inclusion CO2 contents
    by feeding each row into propagate_CO2_in_bubble_ind. The returned standard deviation uses the 84th-16th percentile
    rather than the true standard deviation, as this is better for skewed distributions.

    Parameters
    ----------------
    SampleID: str, pd.series
        Sample_ID (e.g. sample names) which is returned on dataframe

    N_dup: int
        Number of duplicates when generating errors for Monte Carlo simulations

    vol_perc_bub: int, float, pd.series
        Volume proportion of sulfide in melt inclusion

    melt_dens_kgm3:int, float, pd.series
        Density of the silicate melt in kg/m3, e.g. from DensityX

    CO2_bub_dens_gcm3: int, float, pd.Series
        Density of the vapour bubble in g/cm3


    error_vol_perc_bub, CO2_bub_dens_gcm3, error_melt_dens_kgm3: int, float, pd.Series
        Error for each variable, can be absolute or %

    error_type_vol_perc_bub, error_type_bub_dens_gcm3, error_type_melt_dens_kgm3: 'Abs' or 'Perc'
        whether given error is perc or absolute

    error_dist_vol_perc_bub, error_dist_bub_dens_gcm3, error_dist_melt_dens_kgm3: 'normal' or 'uniform'
        Distribution of simulated error

    plot_figure: bool
        Default true - plots a figure of the row indicated by fig_i (default 1st row, fig_i=0)

    neg_values: bool
        Default True - whether negative values are removed from MC simulations or not. False, replace all negative values with zeros.


    Returns
    ------------------
    pd.DataFrame: df_step, All_outputs
        All outputs has calculations for every simulaiton
        df_step has average and standard deviation for each sample

    """



    # Set up empty things to fill up.

    if type(vol_perc_bub) is pd.Series:
        len_loop=len(vol_perc_bub)
    else:
        len_loop=1



    mean_CO2_eq_melt = np.empty(len_loop, dtype=float)
    mean_CO2_eq_melt_ind = np.empty(len_loop, dtype=float)
    med_CO2_eq_melt  = np.empty(len_loop, dtype=float)
    std_CO2_eq_melt  = np.empty(len_loop, dtype=float)
    preferred_val_CO2_melt= np.empty(len_loop, dtype=float)
    std_IQR_CO2_eq_melt= np.empty(len_loop, dtype=float)
    Sample=np.empty(len_loop,  dtype=np.dtype('U100') )


    All_outputs=pd.DataFrame([])


    #This loops through each sample
    for i in range(0, len_loop):
        if i % 20 == 0:
            print('working on sample number '+str(i))


        # If user has entered a pandas series for error, takes right one for each loop
        # vol_perc_bub % and error

        # Checking volume right format
        if type(vol_perc_bub) is pd.Series:
            vol_perc_bub_i=vol_perc_bub.iloc[i]
        else:
            vol_perc_bub_i=vol_perc_bub


        if type(error_vol_perc_bub) is pd.Series:
            error_vol_perc_bub_i=error_vol_perc_bub.iloc[i]
        else:
            error_vol_perc_bub_i=error_vol_perc_bub

        # Checking bubble dens right form

        if type(CO2_bub_dens_gcm3) is pd.Series:
            CO2_bub_dens_gcm3_i=CO2_bub_dens_gcm3.iloc[i]
        else:
            CO2_bub_dens_gcm3_i=CO2_bub_dens_gcm3

        # Checking melt density

        if type(error_CO2_bub_dens_gcm3) is pd.Series:
            error_CO2_bub_dens_gcm3_i=error_CO2_bub_dens_gcm3.iloc[i]
        else:
            error_CO2_bub_dens_gcm3_i=error_CO2_bub_dens_gcm3

        # Checking melt density


        if type(melt_dens_kgm3) is pd.Series:
            melt_dens_kgm3_i=melt_dens_kgm3.iloc[i]
        else:
            melt_dens_kgm3_i=melt_dens_kgm3


        if type(error_melt_dens_kgm3) is pd.Series:
            error_melt_dens_kgm3_i=error_melt_dens_kgm3.iloc[i]
        else:
            error_melt_dens_kgm3_i=error_melt_dens_kgm3






        # This is the function doing the work to actually make the simulations for each variable.
        df_synthetic=propagate_CO2_in_bubble_ind(
N_dup=N_dup,
vol_perc_bub=vol_perc_bub_i,
melt_dens_kgm3=melt_dens_kgm3_i,
CO2_bub_dens_gcm3=CO2_bub_dens_gcm3_i,
error_CO2_bub_dens_gcm3=error_CO2_bub_dens_gcm3_i,
error_type_CO2_bub_dens_gcm3=error_type_CO2_bub_dens_gcm3,
 error_dist_CO2_bub_dens_gcm3=error_dist_CO2_bub_dens_gcm3,
error_vol_perc_bub=error_vol_perc_bub_i,
error_type_vol_perc_bub=error_type_vol_perc_bub,
error_dist_vol_perc_bub=error_dist_vol_perc_bub,
error_melt_dens_kgm3=error_melt_dens_kgm3_i,
error_type_melt_dens_kgm3=error_type_melt_dens_kgm3,
error_dist_melt_dens_kgm3=error_dist_melt_dens_kgm3,
 len_loop=1, neg_values=neg_values)


        # Convert to densities for MC



        df=df_synthetic






        # Check of
        if sample_ID is None:
            Sample[i]=i

        elif isinstance(sample_ID, str):
            Sample[i]=sample_ID
        else:
            Sample[i]=sample_ID.iloc[i]

        df.insert(0, 'Filename', Sample[i])







        All_outputs=pd.concat([All_outputs, df], axis=0)

        mean_CO2_eq_melt[i]=np.nanmean(df['CO2_eq_melt_ppm_MC'])
        med_CO2_eq_melt[i]=np.nanmedian(df['CO2_eq_melt_ppm_MC'])
        std_CO2_eq_melt[i]=np.nanstd(df['CO2_eq_melt_ppm_MC'])
        var=df['CO2_eq_melt_ppm_MC']
        std_IQR_CO2_eq_melt[i]=0.5*np.abs((np.percentile(var, 84) -np.percentile(var, 16)))

        preferred_val_CO2_melt[i]=np.nanmean(df['CO2_eq_melt_ppm_noMC'])


        mean_CO2_eq_melt_ind[i]=df['CO2_eq_melt_ppm_noMC'].iloc[0]





    df_step=pd.DataFrame(data={'Filename': Sample,

                        'CO2_eq_in_melt_noMC': preferred_val_CO2_melt,
                        'mean_MC_CO2_equiv_melt_ppm': mean_CO2_eq_melt,
                        'med_MC_CO2_equiv_melt_ppm': med_CO2_eq_melt,
                        'std_MC_CO2_equiv_melt_ppm': std_IQR_CO2_eq_melt,



                         })




    if plot_figure is True:
        all_sims=fig_i
        all_sims=All_outputs.loc[All_outputs['Filename']==All_outputs['Filename'].iloc[fig_i]]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8))
        ax1.hist(all_sims['vol_perc_bub_with_noise'], bins=50);
        ax1.set_xlabel('MC bubble volume (vol%)')
        ax1.set_ylabel('# of simulations')

        ax2.hist(all_sims['melt_dens_kgm3_with_noise'], bins=50);
        ax2.set_xlabel('MC Melt Density (kg/cm3)')

        ax3.hist(all_sims['CO2_bub_dens_gcm3_with_noise'], bins=50);
        ax3.set_xlabel('MC Bubble Density (g/cm3)')
        ax3.set_ylabel('# of simulations')

        ax4.hist(all_sims['CO2_eq_melt_ppm_MC'], bins=50, color='red');
        ax4.set_xlabel('CO2 equivalent in the melt held in the bubble (ppm)')
        ax4.plot([all_sims['CO2_eq_melt_ppm_noMC'].iloc[0], 	all_sims['CO2_eq_melt_ppm_noMC'].iloc[0]], [0, N_dup/20], '-k', label='Preferred value');
        ax4.plot([all_sims['CO2_eq_melt_ppm_noMC'].iloc[0]+np.std(all_sims['CO2_eq_melt_ppm_MC']), 	all_sims['CO2_eq_melt_ppm_noMC'].iloc[0]+np.std(all_sims['CO2_eq_melt_ppm_MC'])], [0, N_dup/20], ':k', label='Preferred+value+1s MC');
        ax4.plot([all_sims['CO2_eq_melt_ppm_noMC'].iloc[0]-np.std(all_sims['CO2_eq_melt_ppm_MC']), 	all_sims['CO2_eq_melt_ppm_noMC'].iloc[0]-np.std(all_sims['CO2_eq_melt_ppm_MC'])], [0, N_dup/20], ':k', label='Preferred-value+1s MC');
        ax4.legend()

    return df_step, All_outputs, fig if 'fig' in locals() else None






def propagate_CO2_in_bubble_ind(sample_i=0,  N_dup=1000, vol_perc_bub=None,
CO2_bub_dens_gcm3=None, melt_dens_kgm3=None,
error_vol_perc_bub=0, error_type_vol_perc_bub='Abs', error_dist_vol_perc_bub='normal',
error_CO2_bub_dens_gcm3=0, error_type_CO2_bub_dens_gcm3='Abs', error_dist_CO2_bub_dens_gcm3='normal',
error_melt_dens_kgm3=0, error_type_melt_dens_kgm3='Abs', error_dist_melt_dens_kgm3='normal', len_loop=1, neg_values=True):

    """ This function propagates uncertainty in reconstruction of melt inclusion bubble equivalent CO2 contents.
    and returns a dataframe. it does one sample at a time.

    Parameters
    ----------------
    sample_i: int
        if your inputs are panda series, says which row to take

    Parameters
    ----------------
    SampleID: str, pd.series
        Sample_ID (e.g. sample names) which is returned on dataframe

    N_dup: int
        Number of duplicates when generating errors for Monte Carlo simulations

    vol_perc_bub: int, float, pd.series
        Volume proportion of sulfide in melt inclusion

    melt_dens_kgm3:int, float, pd.series
        Density of the silicate melt in kg/m3, e.g. from DensityX

    CO2_bub_dens_gcm3: int, float, pd.Series
        Density of the vapour bubble in g/cm3


    error_vol_perc_bub, CO2_bub_dens_gcm3, error_melt_dens_kgm3: int, float, pd.Series
        Error for each variable, can be absolute or %

    error_type_vol_perc_bub, error_type_bub_dens_gcm3, error_type_melt_dens_kgm3: 'Abs' or 'Perc'
        whether given error is perc or absolute

    error_dist_vol_perc_bub, error_dist_bub_dens_gcm3, error_dist_melt_dens_kgm3: 'normal' or 'uniform'
        Distribution of simulated error

    plot_figure: bool
        Default true - plots a figure of the row indicated by fig_i (default 1st row, fig_i=0)

    neg_values: bool
        Default True - whether negative values are removed from MC simulations or not. False, replace all negative values with zeros.



    Returns
    ------------------
    pd.DataFrame:
        Input variable duplicated N_dup times with noise added.




    """


    if len_loop==1:
        df_c=pd.DataFrame(data={
                            'vol_perc_bub': vol_perc_bub,
                            'CO2_bub_dens_gcm3': CO2_bub_dens_gcm3,
                            'melt_dens_kgm3': melt_dens_kgm3 }, index=[0])
    else:
        df_c=pd.DataFrame(data={
                            'vol_perc_bub': vol_perc_bub,
                            'CO2_bub_dens_gcm3': CO2_bub_dens_gcm3,
                            'melt_dens_kgm3': melt_dens_kgm3})





    # Volume error distribution

    if error_type_vol_perc_bub=='Abs':
        error_Vol=error_vol_perc_bub
    if error_type_vol_perc_bub =='Perc':
        error_Vol=df_c['vol_perc_bub'].iloc[sample_i]*error_vol_perc_bub/100
    if error_dist_vol_perc_bub=='normal':
        Noise_to_add_Vol = np.random.normal(0, error_Vol, N_dup)
    if error_dist_vol_perc_bub=='uniform':
        Noise_to_add_Vol = np.random.uniform(- error_Vol, +
                                                      error_Vol, N_dup)

    Vol_with_noise=Noise_to_add_Vol+df_c['vol_perc_bub'].iloc[sample_i]
    if neg_values is False:
        Vol_with_noise[Vol_with_noise < 0.000000000000001] = 0.000000000000001

    #

    # Volume error distribution

    if error_type_CO2_bub_dens_gcm3=='Abs':
        error_CO2_bub_dens_gcm3=error_CO2_bub_dens_gcm3
    if error_type_CO2_bub_dens_gcm3 =='Perc':
        error_CO2_bub_dens_gcm3=df_c['vol_perc_bub'].iloc[sample_i]*error_CO2_bub_dens_gcm3/100
    if error_dist_CO2_bub_dens_gcm3=='normal':
        Noise_to_add_CO2_bub_dens_gcm3 = np.random.normal(0, error_CO2_bub_dens_gcm3, N_dup)
    if error_dist_CO2_bub_dens_gcm3=='uniform':
        Noise_to_add_CO2_bub_dens_gcm3 = np.random.uniform(- error_CO2_bub_dens_gcm3, +
                                                      error_CO2_bub_dens_gcm3, N_dup)

    CO2_bub_dens_gcm3_with_noise=Noise_to_add_CO2_bub_dens_gcm3+df_c['CO2_bub_dens_gcm3'].iloc[sample_i]
    if neg_values is False:
        CO2_bub_dens_gcm3_with_noise[CO2_bub_dens_gcm3_with_noise < 0.000000000000001] = 0.000000000000001

    # Volume error distribution

    if error_type_melt_dens_kgm3=='Abs':
        error_melt_dens_kgm3=error_melt_dens_kgm3
    if error_type_melt_dens_kgm3 =='Perc':
        error_melt_dens_kgm3=df_c['vol_perc_bub'].iloc[sample_i]*error_melt_dens_kgm3/100
    if error_dist_melt_dens_kgm3=='normal':
        Noise_to_add_melt_dens_kgm3 = np.random.normal(0, error_melt_dens_kgm3, N_dup)
    if error_dist_melt_dens_kgm3=='uniform':
        Noise_to_add_melt_dens_kgm3 = np.random.uniform(- error_melt_dens_kgm3, +
                                                      error_melt_dens_kgm3, N_dup)

    melt_dens_kgm3_with_noise=Noise_to_add_melt_dens_kgm3+df_c['melt_dens_kgm3'].iloc[sample_i]
    if neg_values is False:
        melt_dens_kgm3_with_noise[melt_dens_kgm3_with_noise < 0.000000000000001] = 0.000000000000001
    CO2_eq_melt_ind=10**4 * (df_c['vol_perc_bub']*df_c['CO2_bub_dens_gcm3'])/(df_c['melt_dens_kgm3']/1000)
    df_out=pd.DataFrame(data={

                                'vol_perc_bub_with_noise': Vol_with_noise,
                                'CO2_bub_dens_gcm3_with_noise': CO2_bub_dens_gcm3_with_noise,
                                'melt_dens_kgm3_with_noise': melt_dens_kgm3_with_noise,
                                'vol_perc_bub': df_c['vol_perc_bub'].iloc[sample_i],
                                'CO2_bub_dens_gcm3': CO2_bub_dens_gcm3,
                                'Absolute_error_Vol': error_Vol,
                                'error_type_vol_perc_bub': error_type_vol_perc_bub,
                                'error_dist_Vol': error_dist_vol_perc_bub,
                                'error_CO2_bub_dens_gcm3': error_CO2_bub_dens_gcm3,
                                'error_type_CO2_bub_dens_gcm3': error_type_CO2_bub_dens_gcm3,
                                'error_dist_CO2_bub_dens_gcm3': error_dist_CO2_bub_dens_gcm3,
                                })

    CO2_eq_melt=10**4*((df_out['vol_perc_bub_with_noise']*df_out['CO2_bub_dens_gcm3_with_noise']))/(df_out['melt_dens_kgm3_with_noise']/1000)

    df_out.insert(0, 'CO2_eq_melt_ppm_MC',CO2_eq_melt)

    df_out.insert(1, 'CO2_eq_melt_ppm_noMC',float(CO2_eq_melt_ind.values))







    return df_out

