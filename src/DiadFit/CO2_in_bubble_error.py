import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lets use a generic function for getting a panda series value or not.

def get_value(variable, i):
    """ This function returns the value if its not a series, otherwise returns the right row in the series
    """
    if isinstance(variable, pd.Series):
        return variable.iloc[i]
    else:
        return variable

const=(4/3)*np.pi

def propagate_CO2_in_bubble(*, N_dup=1000, sample_ID, vol_perc_bub=None, error_vol_perc_bub=None, error_type_vol_perc_bub='Abs',
MI_x=None, MI_y=None,MI_z=None,VB_x=None, VB_y=None,VB_z=None,
error_MI_x=None, error_MI_y=None,error_MI_z=None,error_VB_x=None, error_VB_y=None, error_VB_z=None,
error_type_dimension='Abs', error_dist_dimension='normal',
melt_dens_kgm3, CO2_bub_dens_gcm3,
 error_dist_vol_perc_bub='normal',
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

    Now, either you know the vol% of the bubble and the associated error, in which case enter these two arguements:

    vol_perc_bub: int, float, pd.series
        Volume proportion of sulfide in melt inclusion

    error_vol_perc_bub, CO2_bub_dens_gcm3, error_melt_dens_kgm3: int, float, pd.Series
        Error for each variable, can be absolute or %

    error_type_vol_perc_bub, error_type_bub_dens_gcm3, error_type_melt_dens_kgm3: 'Abs' or 'Perc'
        whether given error is perc or absolute

    Or, you have measured the x-y-z dimensions of the MI and VB (e.g. by perpendicular polishing).

    MI_x, MI_y, MI_Z, VB_x, VB_y, VB_Z: int, float, pd.Series
        x, y, z dimensions of vapour bubble and melt inclusion

    error_MI_x, error_MI_y, error_MI_Z, error_VB_x, error_VB_y, error_VB_Z: int, float, pd.Series
        Error on x, y, z dimension meausurement

    error_type_dimension: 'Abs' or 'Perc'
        Whether errors on all x-y-z are perc or abs

    error_dist_dimension: 'normal' or 'uniform'
        Whether errors on all x-y-z are normally or uniformly distributed.


    melt_dens_kgm3:int, float, pd.series
        Density of the silicate melt in kg/m3, e.g. from DensityX

    CO2_bub_dens_gcm3: int, float, pd.Series
        Density of the vapour bubble in g/cm3


    error_dist_vol_perc_bub, error_dist_bub_dens_gcm3, error_dist_melt_dens_kgm3, error_dist_dimension: 'normal' or 'uniform'
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
    # Constant for sphere calcs


     # Lets check what they entered for volume - if they didnt enter a volume % Bubble, lets calculate it
    if vol_perc_bub is None:
        if VB_z is None:
            VB_z=(VB_x+VB_y)/2
        if MI_z is None:
            MI_z=(MI_x+MI_y)/2

        Vol_VB_sphere=const*VB_x*VB_y*VB_z*(0.5)**3
        Vol_MI_sphere=const*MI_x*MI_y*MI_z*(0.5)**3
        vol_perc_bub=100* Vol_VB_sphere/Vol_MI_sphere

    # Now lets check how they entered error in volume
    if not ((error_vol_perc_bub is not None and all(v is None for v in [error_MI_x, error_MI_y, error_MI_z, error_VB_x, error_VB_y, error_VB_z])) or (error_vol_perc_bub is None and all(v is not None for v in [error_MI_x, error_MI_y, error_MI_z, error_VB_x, error_VB_y, error_VB_z]))):
            raise ValueError('Specify either error_vol_perc_bub or non-None values for all of error_MI_x, error_MI_y, error_MI_z, error_VB_x, error_VB_y, and error_VB_z.')




    # Set up empty things to fill up.
    if type(vol_perc_bub) is pd.Series:
        len_loop=len(vol_perc_bub)

    else:
        len_loop=1



    mean_CO2_eq_melt = np.zeros(len_loop, dtype=float)
    mean_CO2_eq_melt_ind = np.zeros(len_loop, dtype=float)
    med_CO2_eq_melt  = np.zeros(len_loop, dtype=float)
    std_CO2_eq_melt  = np.zeros(len_loop, dtype=float)
    preferred_val_CO2_melt= np.zeros(len_loop, dtype=float)
    std_IQR_CO2_eq_melt= np.zeros(len_loop, dtype=float)
    Q84= np.zeros(len_loop, dtype=float)
    Q16= np.zeros(len_loop, dtype=float)
    Sample=np.zeros(len_loop,  dtype=np.dtype('U100') )


    All_outputs=pd.DataFrame([])





    #This loops through each sample
    for i in range(0, len_loop):
        if i % 20 == 0:
            print('working on sample number '+str(i))


        # If user has entered a pandas series for error, takes right one for each loop
        # vol_perc_bub % and error

        # Checking volume right format, if panda series or integer
        vol_perc_bub_i=get_value(vol_perc_bub, i)
        error_vol_perc_bub_i=get_value(error_vol_perc_bub, i)

        CO2_bub_dens_gcm3_i=get_value(CO2_bub_dens_gcm3, i)
        error_CO2_bub_dens_gcm3_i=get_value(error_CO2_bub_dens_gcm3, i)

        melt_dens_kgm3_i=get_value(melt_dens_kgm3, i)
        error_melt_dens_kgm3_i=get_value(error_melt_dens_kgm3, i)




        # This is the function doing the work to actually make the simulations for each variable.
        if error_vol_perc_bub is not None:


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



        else:


            # This is the more complex one where we have to account for x-y-z errors on all of them.
            MI_x_i = get_value(MI_x, i)
            MI_y_i = get_value(MI_y, i)



            VB_x_i = get_value(VB_x, i)
            VB_y_i = get_value(VB_y, i)




            VB_z_i = get_value(VB_z, i)
            MI_z_i = get_value(MI_z, i)



            error_MI_x_i = get_value(error_MI_x, i)
            error_MI_y_i = get_value(error_MI_y, i)
            error_MI_z_i = get_value(error_MI_z, i)
            error_VB_x_i = get_value(error_VB_x, i)
            error_VB_y_i = get_value(error_VB_y, i)
            error_VB_z_i = get_value(error_VB_z, i)



            df_synthetic=propagate_CO2_in_bubble_ind(
    N_dup=N_dup,
    vol_perc_bub=vol_perc_bub_i,
    melt_dens_kgm3=melt_dens_kgm3_i,
    CO2_bub_dens_gcm3=CO2_bub_dens_gcm3_i,
    error_CO2_bub_dens_gcm3=error_CO2_bub_dens_gcm3_i,
    error_type_CO2_bub_dens_gcm3=error_type_CO2_bub_dens_gcm3,
    error_dist_CO2_bub_dens_gcm3=error_dist_CO2_bub_dens_gcm3,
    error_vol_perc_bub=None,
MI_x=MI_x_i, MI_y=MI_y_i,MI_z=MI_z_i,VB_x=VB_x_i, VB_y=VB_y_i,VB_z=VB_z_i,
error_MI_x=error_MI_x_i, error_MI_y=error_MI_y_i,error_MI_z=error_MI_z_i,
error_VB_x=error_VB_x_i, error_VB_y=error_VB_y_i, error_VB_z=error_VB_z_i,
error_type_dimension=error_type_dimension, error_dist_dimension=error_dist_dimension,
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
        Q16[i]=np.percentile(var, 16)
        Q84[i]=np.percentile(var, 84)
        std_IQR_CO2_eq_melt[i]=0.5*np.abs((np.percentile(var, 84) -np.percentile(var, 16)))



        # Values all the same, so can just take the 1st.
        preferred_val_CO2_melt[i]=df['CO2_eq_melt_ppm_noMC'].iloc[0]







    df_step=pd.DataFrame(data={'Filename': Sample,

                        'CO2_eq_in_melt_noMC': preferred_val_CO2_melt,
                        'mean_MC_CO2_equiv_melt_ppm': mean_CO2_eq_melt,
                        'med_MC_CO2_equiv_melt_ppm': med_CO2_eq_melt,
                        'std_MC_IQR_CO2_equiv_melt_ppm': std_IQR_CO2_eq_melt,
                        '16th_quantile_melt_ppm': Q16,
                        '84th_quantile_melt_ppm': Q84,



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

# Lets set the random seed
np.random.seed(42)

def add_noise_to_variable(original_value, error, error_type, error_dist, N_dup, neg_values, neg_threshold):
        """ This function adds noise to each variable for the monte-carloing

        Parameters
        -----------------
        original_value: int, float
            Preferred value (e.g. center of distribution)

        error: int, float
            Error value

        error_type: str
            'Abs' if absolute error, 'Perc' if percent

        error_dist: str
            'normal' if normally distributed, 'uniform' if uniformly distributed.

        N_dup: int
            number of duplicates

        neg_values: bool
            whether negative values are replaced with zeros


        """

        #  Depending on the error type, allocates an error
        if error_type == 'Abs':
            calculated_error = error
        elif error_type == 'Perc':
            calculated_error = original_value * error / 100

        # Generates noise following a distribution
        if error_dist == 'normal':
            noise_to_add = np.random.normal(0, calculated_error, N_dup)
        elif error_dist == 'uniform':
            noise_to_add = np.random.uniform(-calculated_error, calculated_error, N_dup)

        # adds this noise to original value
        value_with_noise = noise_to_add + original_value

        if not neg_values:
            value_with_noise[value_with_noise < neg_threshold] = neg_threshold

        return value_with_noise







def propagate_CO2_in_bubble_ind(sample_i=0,  N_dup=1000, vol_perc_bub=None,
CO2_bub_dens_gcm3=None, melt_dens_kgm3=None,
MI_x=None, MI_y=None,MI_z=None,VB_x=None, VB_y=None,VB_z=None,
error_MI_x=None, error_MI_y=None,error_MI_z=None,error_VB_x=None, error_VB_y=None, error_VB_z=None,
error_type_dimension='Abs', error_dist_dimension='normal',
error_vol_perc_bub=None, error_type_vol_perc_bub='Abs', error_dist_vol_perc_bub='normal',
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
    For volumes, either enter vol% bubble and associated errors...

    vol_perc_bub: int, float, pd.series
        Volume proportion of sulfide in melt inclusion


    error_vol_perc_bub, CO2_bub_dens_gcm3, error_melt_dens_kgm3: int, float, pd.Series
        Error for each variable, can be absolute or %

    error_type_vol_perc_bub, error_type_bub_dens_gcm3, error_type_melt_dens_kgm3: 'Abs' or 'Perc'
        whether given error is perc or absolute

    error_dist_vol_perc_bub, error_dist_bub_dens_gcm3, error_dist_melt_dens_kgm3: 'normal' or 'uniform'
        Distribution of simulated error


    OR
    Enter melt inclusion and vapour bubble dimensions (diameter, not radii), and their errors
    MI_x, MI_y, MI_z, VB_x, VB_y, VB_z: int, float, series:
        Diameter of melt inclusions.

    error_MI_x, error_MI_y, error_MI_z, error_VB_x, error_VB_y, error_VB_z:
        Error on diameter of melt inclusions

    error_type_dimension='Abs' or 'Perc':
        Specify whether errors on these dimensions are absolute or percentage

    error_dist_dimension='normal' or 'uniform':
        Specify error distribution





    SampleID: str, pd.series
        Sample_ID (e.g. sample names) which is returned on dataframe

    N_dup: int
        Number of duplicates when generating errors for Monte Carlo simulations



    melt_dens_kgm3:int, float, pd.series
        Density of the silicate melt in kg/m3, e.g. from DensityX

    CO2_bub_dens_gcm3: int, float, pd.Series
        Density of the vapour bubble in g/cm3



    plot_figure: bool
        Default true - plots a figure of the row indicated by fig_i (default 1st row, fig_i=0)

    neg_values: bool
        Default True - whether negative values are removed from MC simulations or not. False, replace all negative values with zeros.



    Returns
    ------------------
    pd.DataFrame:
        Input variable duplicated N_dup times with noise added.




    """

    # If only a single sample, set up an output dataframe with an index.
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





    # Volume error distribution - if they give a volume percentage rather than dimensions
    if error_vol_perc_bub is not None:
        print('using error on the bubble volume percent, not the entered dimensions, as error_vol_perc_bub was not None')
        # Easy peasy
        Vol_with_noise=add_noise_to_variable(vol_perc_bub, error_vol_perc_bub,
        error_type_vol_perc_bub, error_dist_vol_perc_bub,  N_dup, neg_values, neg_threshold=0.0000000001)


    else:

        x_MI_with_noise=add_noise_to_variable(MI_x, error_MI_x,
        error_type_dimension, error_dist_dimension, N_dup, neg_values, neg_threshold=0.0000000001)

        y_MI_with_noise=add_noise_to_variable(MI_y, error_MI_y,
        error_type_dimension, error_dist_dimension, N_dup, neg_values, neg_threshold=0.0000000001)

        x_VB_with_noise=add_noise_to_variable(VB_x, error_VB_x,
        error_type_dimension, error_dist_dimension,  N_dup, neg_values, neg_threshold=0.0000000001)

        y_VB_with_noise=add_noise_to_variable(VB_y, error_VB_y,
        error_type_dimension, error_dist_dimension, N_dup, neg_values, neg_threshold=0.0000000001)


        z_MI_with_noise=add_noise_to_variable(MI_z, error_MI_z,
        error_type_dimension, error_dist_dimension, N_dup, neg_values, neg_threshold=0.0000000001)


        z_VB_with_noise=add_noise_to_variable(VB_z, error_VB_z,
        error_type_dimension, error_dist_dimension, N_dup, neg_values, neg_threshold=0.0000000001)


        Vol_VB_sphere_with_noise=const*x_VB_with_noise*y_VB_with_noise*z_VB_with_noise*(0.5)**3
        Vol_MI_sphere_with_noise=const*x_MI_with_noise*y_MI_with_noise*z_MI_with_noise*(0.5)**3
        Vol_with_noise=100* Vol_VB_sphere_with_noise/Vol_MI_sphere_with_noise


    # Bubble density
    CO2_bub_dens_gcm3_with_noise=add_noise_to_variable(CO2_bub_dens_gcm3, error_CO2_bub_dens_gcm3,
        error_type_CO2_bub_dens_gcm3, error_dist_CO2_bub_dens_gcm3, N_dup, neg_values, neg_threshold=0.0000000001)

    # Melt density
    melt_dens_kgm3_with_noise=add_noise_to_variable(melt_dens_kgm3, error_melt_dens_kgm3,
        error_type_melt_dens_kgm3, error_dist_melt_dens_kgm3,  N_dup, neg_values, neg_threshold=0.0000000001)

    # Now lets calculate the equilibrium CO2 content of the melt
    CO2_eq_melt_ind=10**4 * (df_c['vol_perc_bub']*df_c['CO2_bub_dens_gcm3'])/(df_c['melt_dens_kgm3']/1000)

    if error_vol_perc_bub is not None:
        df_out=pd.DataFrame(data={

                                'vol_perc_bub_with_noise': Vol_with_noise,
                                'CO2_bub_dens_gcm3_with_noise': CO2_bub_dens_gcm3_with_noise,
                                'melt_dens_kgm3_with_noise': melt_dens_kgm3_with_noise,
                                'vol_perc_bub': df_c['vol_perc_bub'].iloc[sample_i],
                                'CO2_bub_dens_gcm3': CO2_bub_dens_gcm3,
                                'error_type_vol_perc_bub': error_type_vol_perc_bub,
                                'error_dist_Vol': error_dist_vol_perc_bub,
                                'error_CO2_bub_dens_gcm3': error_CO2_bub_dens_gcm3,
                                'error_type_CO2_bub_dens_gcm3': error_type_CO2_bub_dens_gcm3,
                                'error_dist_CO2_bub_dens_gcm3': error_dist_CO2_bub_dens_gcm3,
                                })

    else:
        df_out=pd.DataFrame(data={

                                'vol_perc_bub_with_noise': Vol_with_noise,
                                'CO2_bub_dens_gcm3_with_noise': CO2_bub_dens_gcm3_with_noise,
                                'melt_dens_kgm3_with_noise': melt_dens_kgm3_with_noise,
                                'vol_perc_bub': df_c['vol_perc_bub'].iloc[sample_i],
                                'CO2_bub_dens_gcm3': CO2_bub_dens_gcm3,
                                'error_CO2_bub_dens_gcm3': error_CO2_bub_dens_gcm3,
                                'error_type_CO2_bub_dens_gcm3': error_type_CO2_bub_dens_gcm3,
                                'error_dist_CO2_bub_dens_gcm3': error_dist_CO2_bub_dens_gcm3,
                                'Vol_VB_with_noise': Vol_VB_sphere_with_noise,
                                'Vol_MI_with_noise': Vol_MI_sphere_with_noise,
                                'VB_x_with_noise': x_VB_with_noise,
                                'VB_y_with_noise':y_VB_with_noise,
                                'VB_z_with_noise': z_VB_with_noise,
                                'MI_x_with_noise': x_MI_with_noise,
                                'MI_y_with_noise': y_MI_with_noise,
                                'MI_z_with_noise': z_MI_with_noise,
                                })

    CO2_eq_melt=10**4*((df_out['vol_perc_bub_with_noise']*df_out['CO2_bub_dens_gcm3_with_noise']))/(df_out['melt_dens_kgm3_with_noise']/1000)

    df_out.insert(0, 'CO2_eq_melt_ppm_MC',CO2_eq_melt)

    df_out.insert(1, 'CO2_eq_melt_ppm_noMC',float(CO2_eq_melt_ind.values))


    return df_out

