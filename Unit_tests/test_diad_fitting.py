import unittest
import pandas as pd
import DiadFit as pf
import os

# define things
file_ext='.txt'
prefix=True
spectra_path = os.path.dirname(os.path.realpath(__file__))
exclude_str=['Ne']
filetype='headless_txt'
i=0
# Get files
Diad_Files=pf.get_files(path=spectra_path, file_ext=file_ext, exclude_str=exclude_str)
print(Diad_Files)
# series


# Set up scipy find peaks

diad_id_config=pf.diad_id_config(prominence=30, width=1)

# now see what peaks it found

df_peaks, Diad, fig=pf.identify_diad_peaks(
config=diad_id_config,
path=spectra_path, filename=Diad_Files[0],
filetype=filetype, plot_figure=True)
df_peaks

# Get approx fit params

fit_params, data_y_all=pf.loop_approx_diad_fits(spectra_path=spectra_path, config=diad_id_config,
                       Diad_Files=Diad_Files, filetype=filetype, plot_figure=False)

Diad_files=fit_params['filename']

# cosmic ray filter

# This extract the peaks for the diads, HBs and C13 from fit_params_crr, essential to the CRR process.
diad_peaks=fit_params[['Diad1_pos','Diad2_pos','HB1_pos','HB2_pos','C13_pos']]

#Pick your settings
plot_rays='rays_only'#whether to plot the results or not
export_cleanspec=True #whether to export the spectra with the cosmic ray pixels removed
save_fig='all' #whether to save the figures, options are 'all' or the default 'rays_only'
dynfact=0.001 #dynamic intensity factor for the first pass (y axis on the leftmost plots)
dynfact_2=0.001#dynamic intensity factor for the second pass
n=1 # number of neighboring pixels being compared, typically 1 is ideal.

exclude_ranges=[(1100,1200)] # List of tuples containing ranges to exclude from filtering (i.e., for secondary peaks


rays_found,spectrum=pf.cosmicray_filter.filter_singleray(path=spectra_path,
                                        exclude_ranges=exclude_ranges,filetype=filetype,
                                        Diad_files=Diad_files,i=i,diad_peaks=diad_peaks,plot_rays=plot_rays,
                                      export_cleanspec=export_cleanspec,save_fig=save_fig,dynfact=dynfact,dynfact_2=dynfact_2,n=n)



GroupN_df=fit_params
# Set up actual peak fit

model_name='PseudoVoigtModel'

# Lets use a weak fit to start with
diad1_fit_config_weak=pf.diad1_fit_config(
    model_name=model_name, fit_peaks=2,
    N_poly_bck_diad1=1, lower_bck_diad1=(1180, 1250),
    upper_bck_diad1=(1300, 1350),
    diad_sigma=0.6,
    x_range_residual=10, x_range_baseline=30,
     y_range_baseline=100,
    HB_prom=GroupN_df['HB1_abs_prom'].iloc[i],
    diad_prom=GroupN_df['Diad1_abs_prom'].iloc[i])

Diad1_fit_weak=pf.fit_diad_1_w_bck(config1=diad1_fit_config_weak,
config2=diad_id_config,
path=spectra_path, filename='23 K23_101_FID_50X.txt',
filetype=filetype, plot_figure=True, close_figure=False,
Diad_pos=GroupN_df['Diad1_pos'].iloc[i],
HB_pos=GroupN_df['HB1_pos'].iloc[i])


diad2_fit_config_weak=pf.diad2_fit_config(model_name=model_name,
    fit_peaks=2, upper_bck_diad2=(1430, 1480),
    lower_bck_diad2=(1310, 1360), diad_sigma=0.4,  N_poly_bck_diad2=2,
    x_range_residual=30, y_range_baseline=100,
    x_range_baseline=30,
    HB_prom=GroupN_df['HB2_abs_prom'].iloc[i],
    diad_prom=GroupN_df['Diad2_abs_prom'].iloc[i])

Diad2_fit_weak=pf.fit_diad_2_w_bck(config1=diad2_fit_config_weak,
    config2=diad_id_config,
path=spectra_path, filename='23 K23_101_FID_50X.txt', filetype=filetype,
plot_figure=True, close_figure=False,
Diad_pos=GroupN_df['Diad2_pos'].iloc[i],
HB_pos=GroupN_df['HB2_pos'].iloc[i],
C13_pos=GroupN_df['C13_pos'].iloc[i])



class test_get_diad_files(unittest.TestCase):
    def test_get_diad_files(self):
        self.assertEqual(pf.get_files(path=spectra_path, file_ext=file_ext, exclude_str=exclude_str)[0], '23 K23_101_FID_50X.txt',
 "loaded filename doesnt match")

class test_scipy(unittest.TestCase):
    def test_scipy_pk1(self):
        self.assertAlmostEqual(df_peaks['Diad1_pos'].iloc[0],1285.7938023359134,
decimalPlace, "Calculated line position doesnt match")

    def test_scipy_pk2(self):
        self.assertAlmostEqual(df_peaks['Diad1_pos'].iloc[0],1389.08209899914,
decimalPlace, "Calculated line position doesnt match")


class test_cosmic_ray_filter(unittest.TestCase):
    def test_CRR_filter(self):
        self.assertEqual(rays_found['rays_present'], 'False', 'It found a ray when it shouldnt have')

class test_weak_diad_fit(unittest.TestCase):
    def test_wk_pk1_combo(self):
        self.assertAlmostEqual(Diad1_fit_weak['Diad1_Combofit_Cent'].iloc[0],1285.8203732751274,
decimalPlace, "Calculated Diad 1 position doesnt match test")

    def test_wk_pk1_area(self):
        self.assertAlmostEqual(Diad1_fit_weak['Diad1_Voigt_Area'].iloc[0],936.1206698490131,
decimalPlace, "Calculated Diad 1 position doesnt match test")

    def test_wk_pk1_err(self):
        self.assertAlmostEqual(Diad1_fit_weak['Diad1_cent_err'].iloc[0],0.007354145769786887,
decimalPlace, "Calculated Diad 1 error doesnt match test")


    def test_wk_pk2_combo(self):
        self.assertAlmostEqual(Diad2_fit_weak['Diad2_Combofit_Cent'].iloc[0],1389.0725782923641,
decimalPlace, "Calculated Diad 2 position doesnt match test")

    def test_wk_pk2_area(self):
        self.assertAlmostEqual(Diad2_fit_weak['Diad2_Voigt_Area'].iloc[0],1340.5602764421023,
decimalPlace, "Calculated Diad 2 position doesnt match test")

    def test_wk_pk2_err(self):
        self.assertAlmostEqual(Diad2_fit_weak['Diad2_cent_err'].iloc[0],0.004334933552254167,
decimalPlace, "Calculated Diad 2 error doesnt match test")




