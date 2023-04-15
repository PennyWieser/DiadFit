# Import useful python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# from tqdm.autonotebook import tqdm - Gave error

from tqdm import tqdm
import DiadFit as pf

encode="ISO-8859-1"

def check_pars(plot_rays, save_fig,export_cleanspec,exclude_ranges,Diad_files,diad_peaks):
    """ Checks if input parameters plot_rays, save_fig, export_cleanspec and exclude_ranges are valid. If any of them is not valid, it raises a ValueError indicating which parameter is incorrect and what the allowed values are.

    Parameters
    --------------
    plot_rays: str ('all' or 'rays_only')
        Plot all spectra or just those for which cosmic rays were found
    save_fig: str ('all' or 'rays_only')
        Save all figures or just those for spectra in which cosmic rays were found
    export_cleanspec: bool
        Indicates whether to export de-rayed spectra or not
    exclude_ranges: list of tuples
        List of tuples that defines user-specified ranges to ignore during filtering (i.e., secondary peaks)

    Returns
    -------------
    None

    """
    if not isinstance(plot_rays, str) or plot_rays not in ['all', 'rays_only']:
        raise ValueError("plot_rays can only be 'all' or 'rays_only', please correct")
    if not isinstance(save_fig, str) or save_fig not in ['all', 'rays_only']:
        raise ValueError("save_fig can only be 'all' or 'rays_only', please correct")
    if not isinstance(export_cleanspec, bool):
        raise ValueError("export_cleanspec can only be True or False, please correct")
    if not isinstance(exclude_ranges,(list,type(None))):
        raise ValueError("exclude_ranges can only be a list of tuples (i.e., [(1145,1155),(1080,1090)]), an empty list or None, please correct")
    if not isinstance(Diad_files,pd.core.series.Series):
        raise ValueError("Diad_files can only be a pandas Series, please correct")
    if not isinstance(diad_peaks,(pd.core.frame.DataFrame,type(None))):
        raise ValueError("diad_peaks can only be a pandas DataFrame or None, please correct")
## This function is the main filter, runs on one spectrum

def filter_singleray(*,path=None,Diad_files=None,i=None,diad_peaks=None, exclude_ranges=[], filetype='headless_txt',n=1,dynfact=0.01,dynfact_2=0.003,
                        export_cleanspec=True,plot_rays='all',save_fig='rays_only',figsize=(20,5), xlims=None):
    """ This function is used to filter out cosmic rays in single Raman spectra of CO2.
    It requires the input of pre-identified peaks to avoid excluding peaks of interest.
    The filter compares the intensity of each pixel in the spectrum to n surrounding pixels and calculates an intensity factor.
    Pixels that exceed a certain intensity factor threshold are removed from the spectrum.
    It repeats the process when a cosmic ray is found, so that "wide" cosmic rays can be excluded (when a cosmic ray encompasses more than a single pixel)

    Parameters
    --------------
    path: str
        The path of the spectrum file
    Diad_files: pd.Series
        'filename' column of fit_params variable output by pf.loop_approx_diad_fits
    i: int
        index number in Diad_files of the file to be filtered
    diad_peaks: pd.DataFrame
        Dataframe containing the peaks of interest for each file subset from fit_params variable output by pf.loop_approx_diad_fits(columns Diad1_pos,	Diad2_pos,HB1_pos,HB2_pos,C13_pos)
    exclude_ranges: tuple list
        List of tuples containing ranges for user-defined peaks of interest to exclude from ray-filtering.
    filetype: str ('headless_txt')
        Sets the filetype of the spectrum file
    n: int
        Neighbor pixels to consider, 1 is typically enough.
    dynfact: float or int
        Intensity factor cutoff for the first filter pass (can adjust based on y axis of intensity factor plot, i.e. the y axis on plot a) that this function produces).
        It removes any peaks above this threshold except those within +-1 pixel of any entered peak positoin (or the input ranges)
    dynfact_2: float or int
        Intensity factor cutoff for the second filter pass (can adjust based on y axis of intensity factor plot, i.e. the y axis on plot d) this function produces)
    export_cleanspec: bool
        Indicates whether to export de-rayed spectra or not
    plot_rays: str ('all' or 'rays_only')
        Plot all spectra or just those for which cosmic rays were found
    save_fig: str ('all' or 'rays_only')
        Save all figures or just those for spectra in which cosmic rays were found
    fig_size: tuple
        Sets the figure size
    xlims: list
        Sets the x axis limits of the plots

    Returns
    -------------
    record: pd.DataFrame
        Dataframe containing the filename and whether cosmic rays were found
    clean_spec_df: pd.DataFrame
        DataFrame containing the cleaned spectrum (Wavenumber and Intensity columns)

    """
    try:
        check_pars(plot_rays, save_fig,export_cleanspec,exclude_ranges,Diad_files,diad_peaks)
    except ValueError as e:
        print(str(e))
        raise e

    if type(diad_peaks)==type(None):
        diad_peaks=pd.DataFrame()

    file=Diad_files.iloc[i]
    #open the spectrum in form of array
    Diad_array=pf.get_data(path=path, filename=file, filetype=filetype)

    # Get the intensity of the next and previous pixels
    intensity_prev = np.roll(Diad_array[:,1], -n)
    intensity_next = np.roll(Diad_array[:,1], n)

    # Calculate the maxpx_fact and minpx_fact factors, and a combination of both
    maxpx_fact = (Diad_array[:,1] - intensity_next) / intensity_next
    minpx_fact = (Diad_array[:,1] - intensity_prev) / intensity_prev
    combo_fact=maxpx_fact*minpx_fact

    # Create a new DataFrame with the calculated factors
    pxdf = pd.DataFrame({'Wavenumber': Diad_array[:,0],
                        'Intensity': Diad_array[:,1],
                        'minpx_fact': minpx_fact,
                        'maxpx_fact': maxpx_fact,
                        'maxpx_fact*minpx_fact':combo_fact})

    #This is for the extra ranges defined by the user
    if exclude_ranges==None:
        exclude_ranges=[]

    exclude_mask = pd.Series(False, index=pxdf.index)
    for r in exclude_ranges:
        exclude_mask |= (pxdf.Wavenumber >= r[0]) & (pxdf.Wavenumber <= r[1])
    not_exclude_mask=~exclude_mask

    # this creates a three pixel range for the peaks around the identify peak spot, "signal region"
    res=np.round(max(np.diff(Diad_array[:,0])),1) # resolution of the spectrum - max distance between two pixels, rounded up.

    if diad_peaks.empty==True:
        query_str='maxpx_fact*minpx_fact > @dynfact & @not_exclude_mask'
    else:
        snr_diad1=[diad_peaks['Diad1_pos']-res,diad_peaks['Diad1_pos']+res]
        snr_diad2=[diad_peaks['Diad2_pos']-res,diad_peaks['Diad2_pos']+res]
        snr_hb1=[diad_peaks['HB1_pos']-res,diad_peaks['HB1_pos']+res]
        snr_hb2=[diad_peaks['HB2_pos']-res,diad_peaks['HB2_pos']+res]
        snr_c13=[diad_peaks['C13_pos']-res,diad_peaks['C13_pos']+res]



        # These identifies regions that aren't the peak regions
        not_diad1=~pxdf.Wavenumber.between(snr_diad1[0][i],snr_diad1[1][i])
        not_diad2=~pxdf.Wavenumber.between(snr_diad2[0][i],snr_diad2[1][i])

        not_hb1=~pxdf.Wavenumber.between(snr_hb1[0][i],snr_hb1[1][i])
        not_hb2=~pxdf.Wavenumber.between(snr_hb2[0][i],snr_hb2[1][i])
        not_c13=~pxdf.Wavenumber.between(snr_c13[0][i],snr_c13[1][i])


        #This filters wavenumbers, intensities and intensity factors based on the query criteria.
        query_str='maxpx_fact*minpx_fact > @dynfact & @not_diad1 & @not_diad2 & @not_hb1 & @not_hb2 & @not_c13 & @not_exclude_mask'

    rays_wavenumber=pxdf.query(query_str)['Wavenumber']
    rays_intensity=pxdf.query(query_str)['Intensity']
    rays_fact=pxdf.query(query_str)['maxpx_fact*minpx_fact']

    # Removal of cosmic rays from the main dataframe, the distinction is important for the filter.
    # Second pass will not be efficient with nan values in the spectrum

    # If rays were detected at this step, the filter applies a second pass to get rid of potential "wide" rays

    if rays_wavenumber.empty==False:
        second_pass=True
    else:
        second_pass=False

    if second_pass==False:
        all_rayswave=rays_wavenumber
        pxdf_filt=pxdf.copy()

    if second_pass==True:
        pxdf_filt=pxdf.loc[~pxdf['Wavenumber'].isin(rays_wavenumber)]
        derayed_spectrum=pxdf_filt[['Wavenumber','Intensity']].values

        # Get the intensity and wavenumber of the next and previous pixels
        intensity_pass2_prev = np.roll(derayed_spectrum[:,1], -n)
        intensity_pass2_next = np.roll(derayed_spectrum[:,1], n)

        # Calculate the maxpx_fact and minpx_fact factors, and a combination of both
        maxpx_fact_pass2 = (derayed_spectrum[:,1] - intensity_pass2_next) / intensity_pass2_next
        minpx_fact_pass2 = (derayed_spectrum[:,1] - intensity_pass2_prev) / intensity_pass2_prev
        combo_fact_pass2=maxpx_fact_pass2*minpx_fact_pass2

        # Create a new DataFrame with the calculated factors
        pxdf_pass2 = pd.DataFrame({'Wavenumber': derayed_spectrum[:,0],
                            'Intensity': derayed_spectrum[:,1],
                            'minpx_fact_pass2': minpx_fact_pass2,
                            'maxpx_fact_pass2': maxpx_fact_pass2,
                            'maxpx_fact_pass2*minpx_fact_pass2':combo_fact_pass2})
        #This is for the extra ranges defined by the user
        if exclude_ranges==None:
            exclude_ranges=[]

        exclude_mask2 = pd.Series(False, index=pxdf_pass2.index)
        for r in exclude_ranges:
            exclude_mask2 |= (pxdf_pass2.Wavenumber >= r[0]) & (pxdf_pass2.Wavenumber <= r[1])
        not_exclude_mask2=~exclude_mask2

        #
        if diad_peaks.empty==True:
            query_str2='maxpx_fact_pass2*minpx_fact_pass2 > @dynfact_2 & @not_exclude_mask2'
        else:
            not_diad1=~pxdf_pass2.Wavenumber.between(snr_diad1[0][i],snr_diad1[1][i])
            not_diad2=~pxdf_pass2.Wavenumber.between(snr_diad2[0][i],snr_diad2[1][i])

            not_hb1=~pxdf_pass2.Wavenumber.between(snr_hb1[0][i],snr_hb1[1][i])
            not_hb2=~pxdf_pass2.Wavenumber.between(snr_hb2[0][i],snr_hb2[1][i])
            not_c13=~pxdf_pass2.Wavenumber.between(snr_c13[0][i],snr_c13[1][i])

            #This filters the dataframe
            query_str2='maxpx_fact_pass2*minpx_fact_pass2 > @dynfact_2 & @not_diad1 & @not_diad2 & @not_hb1 & @not_hb2 and @not_c13 & @not_exclude_mask2'

        rays_wavenumber_pass2=pxdf_pass2.query(query_str2)['Wavenumber']
        rays_intensity_pass2=pxdf_pass2.query(query_str2)['Intensity']
        rays_fact_pass2=pxdf_pass2.query(query_str2)['maxpx_fact_pass2*minpx_fact_pass2']

        # Removal of cosmic rays from the main dataframe (replaces intensity with NaN - only for plotting!)
        all_rayswave=pd.concat([rays_wavenumber, rays_wavenumber_pass2])
        pxdf_filt_pass2=pxdf.copy()
        pxdf_filt_pass2.loc[pxdf_filt_pass2['Wavenumber'].isin(all_rayswave), 'Intensity'] = np.nan
        # Removal of cosmic rays from the main dataframe (deletes row - only for fitting!)
        pxdf_filt_pass2_4export=pxdf.loc[~pxdf['Wavenumber'].isin(all_rayswave)]


    # This exports the clean spectrum in headless txt format if True, will only output a file if cosmic rays were detected

    if export_cleanspec==True:

        if all_rayswave.empty==False:
            if second_pass==True:
                pxdf_filt_pass2_4export[['Wavenumber','Intensity']].to_csv(path+'/'+file.replace('.txt', '')+'_CRR_DiadFit.txt', sep='\t', header=False, index=False)
            if second_pass==False:
                pxdf[['Wavenumber','Intensity']].to_csv(path+'/'+file.replace('.txt', '')+'_CRR_DiadFit.txt', sep='\t', header=False, index=False)

    # This plots the results if True
    if plot_rays=='rays_only':
        if all_rayswave.empty==False:
            if second_pass==True:
                fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(1,5,figsize=figsize)

                ax0.plot(pxdf['Wavenumber'],pxdf['maxpx_fact*minpx_fact'],color='k')
                ax3.plot(pxdf_pass2['Wavenumber'],pxdf_pass2['maxpx_fact_pass2*minpx_fact_pass2'],color='r')
                ax0.scatter(rays_wavenumber,rays_fact,color='r')
                ax3.scatter(rays_wavenumber_pass2,rays_fact_pass2,color='orange')

                ax1.plot(pxdf['Wavenumber'],pxdf['Intensity'],color='k')
                ax2.plot(pxdf_filt['Wavenumber'],pxdf_filt['Intensity'],color='r')
                ax4.plot(pxdf_filt_pass2['Wavenumber'],pxdf_filt_pass2['Intensity'],color='orange')

                y0_min, y0_max = ax0.get_ylim()
                y1_min,y1_max = ax1.get_ylim()
                y2_min, y2_max = ax2.get_ylim()
                y3_min,y3_max = ax3.get_ylim()
                y4_min, y4_max = ax4.get_ylim()


                ax0.set_ylim([y0_min,y0_max])
                ax1.set_ylim([y1_min,y1_max])
                ax2.set_ylim([y2_min,y2_max])
                ax3.set_ylim([y3_min,y3_max])
                ax4.set_ylim([y4_min,y4_max])

                for r in exclude_ranges:
                    ax0.fill_between(r, y0_min, y0_max, color='blue', alpha=0.2)
                    ax1.fill_between(r, y1_min, y1_max, color='blue', alpha=0.2)
                    ax2.fill_between(r, y2_min, y2_max, color='blue', alpha=0.2)
                    ax3.fill_between(r, y3_min, y3_max, color='blue', alpha=0.2)
                    ax4.fill_between(r, y4_min, y4_max, color='blue', alpha=0.2)

                ax0.set_title('a) Intensity factor and rays')
                ax1.set_title('b) Original spectrum')
                ax2.set_title('c) De-rayed spectrum')
                ax3.set_title('d) Intensity factor and rays - pass 2')
                ax4.set_title('e) De-rayed spectrum - pass 2')
                if xlims is not None:
                    ax0.set_xlim(xlims)
                    ax1.set_xlim(xlims)
                    ax2.set_xlim(xlims)
                    ax3.set_xlim(xlims)
                    ax4.set_xlim(xlims)

                fig.suptitle(file, fontsize=12)

            if second_pass==False:

                fig, (ax0,ax1) = plt.subplots(1,2,figsize=figsize)

                ax0.plot(pxdf['Wavenumber'],pxdf['maxpx_fact*minpx_fact'],color='k')
                ax0.scatter(rays_wavenumber,rays_fact,color='r')

                ax1.plot(pxdf['Wavenumber'],pxdf['Intensity'],color='k')

                ax0.scatter(pxdf['Wavenumber'][exclude_mask],pxdf['maxpx_fact*minpx_fact'][exclude_mask],color='blue')
                ax1.scatter(pxdf['Wavenumber'][exclude_mask],pxdf['Intensity'][exclude_mask],color='blue')

                y0_min, y0_max = ax0.get_ylim()
                y1_min,y1_max = ax1.get_ylim()
                ax0.set_ylim([y0_min,y0_max])
                ax1.set_ylim([y1_min,y1_max])

                for r in exclude_ranges:
                    ax0.fill_between(r, y0_min, y0_max, color='blue', alpha=0.2)
                    ax1.fill_between(r, y1_min, y1_max, color='blue', alpha=0.2)

                ax0.set_title('Intensity factor and rays')
                ax1.set_title('Original spectrum')
                if xlims is not None:
                    ax0.set_xlim(xlims)
                    ax1.set_xlim(xlims)

                fig.suptitle(file+' - NO RAYS detected', fontsize=12)

    if plot_rays=='all':
        if second_pass==True:
            fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(1,5,figsize=figsize)

            ax0.plot(pxdf['Wavenumber'],pxdf['maxpx_fact*minpx_fact'],color='k')
            ax3.plot(pxdf_pass2['Wavenumber'],pxdf_pass2['maxpx_fact_pass2*minpx_fact_pass2'],color='r')
            ax0.scatter(rays_wavenumber,rays_fact,color='r')
            ax3.scatter(rays_wavenumber_pass2,rays_fact_pass2,color='orange')

            ax1.plot(pxdf['Wavenumber'],pxdf['Intensity'],color='k')
            ax2.plot(pxdf_filt['Wavenumber'],pxdf_filt['Intensity'],color='r')
            ax4.plot(pxdf_filt_pass2['Wavenumber'],pxdf_filt_pass2['Intensity'],color='orange')

            y0_min, y0_max = ax0.get_ylim()
            y1_min,y1_max = ax1.get_ylim()
            y2_min, y2_max = ax2.get_ylim()
            y3_min,y3_max = ax3.get_ylim()
            y4_min, y4_max = ax4.get_ylim()


            ax0.set_ylim([y0_min,y0_max])
            ax1.set_ylim([y1_min,y1_max])
            ax2.set_ylim([y2_min,y2_max])
            ax3.set_ylim([y3_min,y3_max])
            ax4.set_ylim([y4_min,y4_max])

            for r in exclude_ranges:
                ax0.fill_between(r, y0_min, y0_max, color='blue', alpha=0.2)
                ax1.fill_between(r, y1_min, y1_max, color='blue', alpha=0.2)
                ax2.fill_between(r, y2_min, y2_max, color='blue', alpha=0.2)
                ax3.fill_between(r, y3_min, y3_max, color='blue', alpha=0.2)
                ax4.fill_between(r, y4_min, y4_max, color='blue', alpha=0.2)

            ax0.set_title('Intensity factor and rays')
            ax1.set_title('Original spectrum')
            ax2.set_title('De-rayed spectrum')
            ax3.set_title('Intensity factor and rays - pass 2')
            ax4.set_title('De-rayed spectrum - pass 2')
            if xlims is not None:
                ax0.set_xlim(xlims)
                ax1.set_xlim(xlims)
                ax2.set_xlim(xlims)
                ax3.set_xlim(xlims)
                ax4.set_xlim(xlims)

            fig.suptitle(file, fontsize=12)

        if second_pass==False:

            fig, (ax0,ax1) = plt.subplots(1,2,figsize=figsize)

            ax0.plot(pxdf['Wavenumber'],pxdf['maxpx_fact*minpx_fact'],color='k')
            ax0.scatter(rays_wavenumber,rays_fact,color='r')

            ax1.plot(pxdf['Wavenumber'],pxdf['Intensity'],color='k')

            y0_min, y0_max = ax0.get_ylim()
            y1_min,y1_max = ax1.get_ylim()
            ax0.set_ylim([y0_min,y0_max])
            ax1.set_ylim([y1_min,y1_max])

            for r in exclude_ranges:
                ax0.fill_between(r, y0_min, y0_max, color='blue', alpha=0.2)
                ax1.fill_between(r, y1_min, y1_max, color='blue', alpha=0.2)

            ax0.set_title('Intensity factor and rays')
            ax1.set_title('Original spectrum')
            if xlims is not None:
                ax0.set_xlim(xlims)
                ax1.set_xlim(xlims)

            fig.suptitle(file+' - NO RAYS detected', fontsize=12)

        if save_fig=='all':
            fig_path = os.path.join(path, "CRRfigs")
            os.makedirs(fig_path, exist_ok=True)
            fig.savefig(fig_path+'/'+file.replace('.txt', '')+'_CRR_DiadFit.png')
        if save_fig=='rays_only':
            fig_path = os.path.join(path, "CRRfigs")
            os.makedirs(fig_path, exist_ok=True)
            if all_rayswave.empty==False:
                fig.savefig(fig_path+'/'+file.replace('.txt', '')+'_CRR_DiadFit.png')

    record=pd.DataFrame([])
    record.loc[file,'filename']=file
    record.loc[file,'rays_present']=not all_rayswave.empty

    if second_pass==True:
        clean_spec_df=pxdf_filt_pass2[['Wavenumber','Intensity']]
    if second_pass==False:
        clean_spec_df=pxdf[['Wavenumber','Intensity']]

    return record ,clean_spec_df

## Filter rays in a loop

def filter_raysinloop(*,spectra_path=None,Diad_files=None, diad_peaks=None,exclude_ranges=[],filetype='headless_txt',
n=1,dynfact=0.01, dynfact_2=0.0005, export_cleanspec=True,plot_rays='all', save_fig='all', xlims=None,fit_params=None, filename_col='filename'):
    """ This function is used to filter out cosmic rays in multiple Raman spectra of CO2 in a loop.

    Parameters
    --------------
    spectra_path: str
        The folder path of the spectrum files
    Diad_files: pd.Series
        'filename' column of fit_params variable output by pf.loop_approx_diad_fits
    filetype: str
        Identifies type of file
        choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'
    diad_peaks: pd.DataFrame
        Dataframe containing the peaks of interest for each file subset from fit_params variable output by pf.loop_approx_diad_fits(columns Diad1_pos,	Diad2_pos,HB1_pos,HB2_pos,C13_pos)
    fit_params: pd.DataFrame
        Dataframe containing peak fits and filenames. Used to be called fit_params, still works.
    filename_col: str
        Column name that contains the filenames in fit_params, this is used to merge the new CRR filenames
    exclude_ranges: tuple list
        List of tuples containing ranges for user-defined peaks of interest to exclude from ray-filtering.
    filetype: str ('headless_txt')
        Sets the filetype of the spectrum file
    n: int
        Neighbor pixels to consider, 1 is typically enough.
    dynfact: float or int
        Intensity factor cutoff for the first filter pass (can adjust based on y axis of intensity factor plot)
    dynfact_2: float or int
        Intensity factor cutoff for the second filter pass (can adjust based on y axis of intensity factor plot)
    export_cleanspec: bool
        Indicates whether to export de-rayed spectra or not
    plot_rays: str ('all' or 'rays_only')
        Plot all spectra or just those for which cosmic rays were found
    save_fig: str ('all' or 'rays_only')
        Save all figures or just those for spectra in which cosmic rays were found
    xlims: list
        Sets the x axis limits of the plots

    Returns
    -------------
    data_y_all_CRR_DiadFit: np.array
        Array containing all cleaned spectra for plotting
    fit_params_CRR_DiadFit: pd.DataFrame
        fit_params DataFrame output from pf.loop_approx_diad_fits with updated filenames when cosmic rays were found and a new column indicating if cosmic rays were found

    """

    try:
        check_pars(plot_rays, save_fig,export_cleanspec,exclude_ranges,Diad_files,diad_peaks)
    except ValueError as e:
        print(str(e))
        raise e

    ray_list=pd.DataFrame([])
    spectra_df=pd.DataFrame([])

    for i in tqdm(Diad_files.index.tolist()):


        filename_select=Diad_files.iloc[i]

        rays_found,spectrum=pf.filter_singleray(path=spectra_path,Diad_files=Diad_files,i=i,diad_peaks=diad_peaks,exclude_ranges=exclude_ranges,plot_rays=plot_rays,
                                export_cleanspec=export_cleanspec,save_fig=save_fig,dynfact=dynfact,dynfact_2=dynfact_2,n=n,xlims=xlims,filetype=filetype)

        ray_list=pd.concat([ray_list,rays_found])
        spectra_df=pd.concat([spectra_df,spectrum['Intensity']],axis=1)

    ray_list=ray_list.reset_index(drop=True)

    # this is the new data_y_all array, contains all intensities for the spectra, with rays removed.
    data_y_all_CRR_DiadFit=spectra_df.to_numpy()


    # # This merges the results of the CRR filtering loop back in with the fit_parameters (filenames for which CRR detected are replaced by filename_CRR_DiadFit
    fit_params_CRR_DiadFit=pd.merge(left=ray_list, right=fit_params, left_on='filename',right_on=filename_col, how='outer')
    fit_params_CRR_DiadFit.loc[fit_params_CRR_DiadFit['rays_present']==True, filename_col]=fit_params_CRR_DiadFit[filename_col].str.replace('.txt', '',regex=True)+'_CRR_DiadFit.txt'
    display(fit_params_CRR_DiadFit.head())

    return data_y_all_CRR_DiadFit,fit_params_CRR_DiadFit