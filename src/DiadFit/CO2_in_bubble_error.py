import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def propagate_CO2_in_bubble(sample_ID, vol_perc_bub, melt_dens_kgm3, CO2_bub_dens_gcm3,  N_dup=1000,
error_vol_perc_bub=0, error_type_vol_perc_bub='Abs', error_dist_vol_perc_bub='normal',
error_CO2_bub_dens_gcm3=0, error_type_CO2_bub_dens_gcm3='Abs', error_dist_CO2_bub_dens_gcm3='normal',
error_melt_dens_kgm3=0, error_type_melt_dens_kgm3='Abs', error_dist_melt_dens_kgm3='normal'):



    """ This function propagates uncertainty in reconstruction of melt inclusion CO2 contents
    by feeding each row into propagate_CO2_in_bubble_ind

    Parameters
    ----------------

    N_dup: int
        Number of duplicates when generating errors for Monte Carlo simulations


    vol_perc_bub: int, float, pd.series
        Volume proportion of sulfide in melt inclusion

    melt_dens_kgm3:int, float, pd.series
        Density of the melt in kg/m3

    error_vol_perc_bub, CO2_bub_dens_gcm3, error_melt_dens_kgm3: int, float, pd.Series
        Error for each variable, can be absolute or %

    error_type_vol_perc_bub, error_type_bub_dens_gcm3, error_type_melt_dens_kgm3: 'Abs' or 'Perc'
        whether given error is perc or absolute

    error_dist_vol_perc_bub, error_dist_bub_dens_gcm3, error_dist_melt_dens_kgm3: 'normal' or 'uniform'
        Distribution of simulated error

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
 len_loop=1)


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


        mean_CO2_eq_melt_ind[i]=df['CO2_eq_melt_ppm'].iloc[0]





    df_step=pd.DataFrame(data={'Filename': Sample,
                        'CO2_eq_melt_ppm':mean_CO2_eq_melt_ind,
                        'std_MC_CO2_equiv_melt_ppm': std_CO2_eq_melt,
                        'med_MC_CO2_equiv_melt_ppm': med_CO2_eq_melt,
                        'mean_MC_CO2_equiv_melt_ppm': mean_CO2_eq_melt,

                         })



    return df_step, All_outputs





def propagate_CO2_in_bubble_ind(sample_i=0,  N_dup=1000, vol_perc_bub=None,
CO2_bub_dens_gcm3=None, melt_dens_kgm3=None,
error_vol_perc_bub=0, error_type_vol_perc_bub='Abs', error_dist_vol_perc_bub='normal',
error_CO2_bub_dens_gcm3=0, error_type_CO2_bub_dens_gcm3='Abs', error_dist_CO2_bub_dens_gcm3='normal',
error_melt_dens_kgm3=0, error_type_melt_dens_kgm3='Abs', error_dist_melt_dens_kgm3='normal', len_loop=1):

    """ This function propagates uncertainty in reconstruction of melt inclusion -sulfide volumes for a single row in a dataframe
    and returns a dataframe

    Parameters
    ----------------
    sample_i: int
        if your inputs are panda series, says which row to take

    N_dup: int
        Number of duplicates when generating errors for Monte Carlo simulations


   vol_perc_bub: int, float, pd.series
        Volume perc of bubble in melt inclusion

    CO2_bub_dens_gcm3: int, float, pd.series
        Density of the sulfide in g/cm3

    melt_dens_kgm3:int, float, pd.series
        Density of the melt in kg/m3

    error_vol_perc_bub, error_CO2_bub_dens_gcm3, error_melt_dens_kgm3: int, float, pd.Series
        Error for each variable, can be absolute or %

    error_type_vol_perc, error_type_CO2_bub_dens_gcm3, error_type_melt_dens_kgm3: 'Abs' or 'Perc'
        whether given error is perc or absolute

    error_dist_Vol, error_dist_CO2_bub_dens_gcm3, error_dist_melt_dens_kgm3: 'normal' or 'uniform'
        Distribution of simulated error

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
    melt_dens_kgm3_with_noise[melt_dens_kgm3_with_noise < 0.000000000000001] = 0.000000000000001
    CO2_eq_melt_ind=10**4 * (df_c['vol_perc_bub']*df_c['CO2_bub_dens_gcm3'])/(df_c['melt_dens_kgm3']/1000)
    df_out=pd.DataFrame(data={

                                'vol_perc_bub_with_noise': Vol_with_noise,
                                'CO2_bub_dens_gcm3_with_noise': CO2_bub_dens_gcm3_with_noise,
                                'melt_dens_kgm3_with_noise': melt_dens_kgm3_with_noise,
                                'vol_perc_bub': df_c['vol_perc_bub'].iloc[sample_i],
                                'Crustal Density_kg_m3': CO2_bub_dens_gcm3,
                                'error_Vol': error_Vol,
                                'error_type_vol_perc_bub': error_type_vol_perc_bub,
                                'error_dist_Vol': error_dist_vol_perc_bub,
                                'error_CO2_bub_dens_gcm3': error_CO2_bub_dens_gcm3,
                                'error_type_CO2_bub_dens_gcm3': error_type_CO2_bub_dens_gcm3,
                                'error_dist_CO2_bub_dens_gcm3': error_dist_CO2_bub_dens_gcm3,
                                })

    CO2_eq_melt=10**4*((df_out['vol_perc_bub_with_noise']*df_out['CO2_bub_dens_gcm3_with_noise']))/(df_out['melt_dens_kgm3_with_noise']/1000)

    df_out.insert(1, 'CO2_eq_melt_ppm_MC',CO2_eq_melt)

    df_out.insert(2, 'CO2_eq_melt_ppm',float(CO2_eq_melt_ind.values))




    return df_out

