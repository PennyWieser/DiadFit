# Import useful python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm_notebook
import DiadFit as pf

encode="ISO-8859-1"

## This function is the main filter, runs on one spectrum

def filter_singleray(*,path=None,Diad_files=None,i=None,diad_peaks=None,  filetype='headless_txt',n=1,dynfact=0.01,dynfact_2=0.003,
                        export_cleanspec=True,plot_rays=True,save_fig='rays_only',figsize=(20,5), xlims=None):
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

    # this creates a three pixel range for the peaks around the identify peak spot, "signal region"
    res=np.round(max(np.diff(Diad_array[:,0])),1) # resolution of the spectrum - max distance between two pixels, rounded up.
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
    query_str='maxpx_fact*minpx_fact > @dynfact & @not_diad1 & @not_diad2 & @not_hb1 & @not_hb2 & @not_c13'
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

        not_diad1=~pxdf_pass2.Wavenumber.between(snr_diad1[0][i],snr_diad1[1][i])
        not_diad2=~pxdf_pass2.Wavenumber.between(snr_diad2[0][i],snr_diad2[1][i])

        not_hb1=~pxdf_pass2.Wavenumber.between(snr_hb1[0][i],snr_hb1[1][i])
        not_hb2=~pxdf_pass2.Wavenumber.between(snr_hb2[0][i],snr_hb2[1][i])
        not_c13=~pxdf_pass2.Wavenumber.between(snr_c13[0][i],snr_c13[1][i])

        #This filters the dataframe
        query_str2='maxpx_fact_pass2*minpx_fact_pass2 > @dynfact_2 & @not_diad1 & @not_diad2 & @not_hb1 & @not_hb2 and @not_c13'
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
                pxdf_filt_pass2_4export[['Wavenumber','Intensity']].to_csv(path+'/'+file.replace('.txt', '')+'_CRR.txt', sep='\t', header=False, index=False)
            if second_pass==False:
                pxdf[['Wavenumber','Intensity']].to_csv(path+'/'+file.replace('.txt', '')+'_CRR.txt', sep='\t', header=False, index=False)

    # This plots the results if True
    if plot_rays==True:
        if second_pass==True:
            fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(1,5,figsize=figsize)

            ax0.plot(pxdf['Wavenumber'],pxdf['maxpx_fact*minpx_fact'],color='k')
            ax3.plot(pxdf_pass2['Wavenumber'],pxdf_pass2['maxpx_fact_pass2*minpx_fact_pass2'],color='r')
            ax0.scatter(rays_wavenumber,rays_fact,color='r')
            ax3.scatter(rays_wavenumber_pass2,rays_fact_pass2,color='orange')

            ax1.plot(pxdf['Wavenumber'],pxdf['Intensity'],color='k')
            ax2.plot(pxdf_filt['Wavenumber'],pxdf_filt['Intensity'],color='r')
            ax4.plot(pxdf_filt_pass2['Wavenumber'],pxdf_filt_pass2['Intensity'],color='orange')

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

            ax0.set_title('Intensity factor and rays')
            ax1.set_title('Original spectrum')
            if xlims is not None:
                ax0.set_xlim(xlims)
                ax1.set_xlim(xlims)

            fig.suptitle(file+' - NO RAYS detected', fontsize=12)

        if save_fig=='all':
            fig_path = os.path.join(path, "CRRfigs")
            os.makedirs(fig_path, exist_ok=True)
            fig.savefig(fig_path+'/'+file.replace('.txt', '')+'_CRR.png')
        if save_fig=='rays_only':
            fig_path = os.path.join(path, "CRRfigs")
            os.makedirs(fig_path, exist_ok=True)
            if all_rayswave.empty==False:
                fig.savefig(fig_path+'/'+file.replace('.txt', '')+'_CRR.png')

    record=pd.DataFrame([])
    record.loc[file,'filename']=file
    record.loc[file,'rays_present']=not all_rayswave.empty

    if second_pass==True:
        return record ,pxdf_filt_pass2[['Wavenumber','Intensity']]
    if second_pass==False:
        return record ,pxdf[['Wavenumber','Intensity']]

## Filter rays in a loop

def filter_raysinloop(*,Diad_files=None, spectra_path=None, diad_peaks=None,fit_params=None, plot_rays=False, export_cleanspec=True, save_fig='all', dynfact=0.01, dynfact_2=0.0005, n=1,xlims=None):

    ray_list=pd.DataFrame([])
    spectra_df=pd.DataFrame([])

    for i in tqdm_notebook(Diad_files.index.tolist()):

        filename_select=Diad_files.iloc[i]
        rays_found,spectrum=filter_singleray(path=spectra_path,Diad_files=Diad_files,i=i,diad_peaks=diad_peaks,plot_rays=plot_rays,
                                 export_cleanspec=export_cleanspec,save_fig=save_fig,dynfact=dynfact,dynfact_2=dynfact_2,n=n,xlims=xlims)
        ray_list=pd.concat([ray_list,rays_found])
        spectra_df=pd.concat([spectra_df,spectrum['Intensity']],axis=1)

    ray_list=ray_list.reset_index(drop=True)

    # this is the new data_y_all array, contains all intensities for the spectra, with rays removed.
    data_y_all_crr=spectra_df.to_numpy()

    # This merges the results of the CRR filtering loop back in with the fit_parameters (filenames for which CRR detected are replaced by filename_CRR
    fit_params_crr=pd.merge(ray_list, fit_params, on='filename', how='outer')
    fit_params_crr.loc[fit_params_crr['rays_present']==True, 'filename']=fit_params_crr['filename'].str.replace('.txt', '',regex=True)+'_CRR.txt'
    display(fit_params_crr.head())
    return data_y_all_crr,fit_params_crr