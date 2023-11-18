import unittest
import pandas as pd
import DiadFit as pf
import os


spectra_path = os.path.dirname(os.path.realpath(__file__))


decimalPlace=4
# define things
file_ext='.txt'
prefix=True

exclude_str=['Ne', 'K23']
filetype='headless_txt'
i=0

# Get files
Diad_Files=pf.get_files(path=spectra_path, file_ext=file_ext, exclude_str=exclude_str)
print(Diad_Files)
# series
filename='POC8.txt'

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




GroupN_df=fit_params
# Set up actual peak fit

model_name='PseudoVoigtModel'

# Lets use a strong fit to start with
diad1_fit_config_strong=pf.diad1_fit_config(
    fit_gauss=True, gauss_amp= 2*GroupN_df['HB1_abs_prom'].iloc[i],
    model_name=model_name, fit_peaks=2,
    N_poly_bck_diad1=1, lower_bck_diad1=(1160, 1200),
    upper_bck_diad1=(1330, 1360),
    diad_sigma=0.6,
    x_range_residual=10, x_range_baseline=30,
     y_range_baseline=1000,
    HB_prom=GroupN_df['HB1_abs_prom'].iloc[i],
    diad_prom=GroupN_df['Diad1_abs_prom'].iloc[i])

Diad1_fit_strong=pf.fit_diad_1_w_bck(config1=diad1_fit_config_strong,
config2=diad_id_config,
path=spectra_path, filename=filename,
filetype=filetype, plot_figure=True, close_figure=False,
Diad_pos=GroupN_df['Diad1_pos'].iloc[i],
HB_pos=GroupN_df['HB1_pos'].iloc[i])


diad2_fit_config_strong=pf.diad2_fit_config(model_name=model_name,
    fit_peaks=3, fit_gauss=True, gauss_amp= 2*GroupN_df['HB2_abs_prom'].iloc[i],
    lower_bck_diad2=(1310, 1340), diad_sigma=1,  N_poly_bck_diad2=2,
    x_range_residual=30, y_range_baseline=1000,
    x_range_baseline=30,
    HB_prom=GroupN_df['HB2_abs_prom'].iloc[i],
    diad_prom=GroupN_df['Diad2_abs_prom'].iloc[i],
    C13_prom=GroupN_df['C13_abs_prom'].iloc[i])

Diad2_fit_strong=pf.fit_diad_2_w_bck(config1=diad2_fit_config_strong,
    config2=diad_id_config,
path=spectra_path, filename=filename, filetype=filetype,
plot_figure=True, close_figure=False,
Diad_pos=GroupN_df['Diad2_pos'].iloc[i],
HB_pos=GroupN_df['HB2_pos'].iloc[i],
C13_pos=GroupN_df['C13_pos'].iloc[i])





class test_strong_diad_fit(unittest.TestCase):
    def test_st_pk1_combo(self):
        self.assertAlmostEqual(Diad1_fit_strong['Diad1_Combofit_Cent'].iloc[0],1282.7310347177588,
2, "Calculated Diad 1 position doesnt match test")

    def test_st_pk1_area(self):
        self.assertAlmostEqual(Diad1_fit_strong['Diad1_Voigt_Area'].iloc[0],44923.511465629825,
0, "Calculated Diad 1 area doesnt match test")

    def test_st_pk1_err(self):
        self.assertAlmostEqual(Diad1_fit_strong['Diad1_cent_err'].iloc[0],0.001553027374488664,
4, "Calculated Diad 1 error doesnt match test")


    def test_st_pk2_combo(self):
        self.assertAlmostEqual(Diad2_fit_strong['Diad2_Combofit_Cent'].iloc[0],1387.1727902209943,
2, "Calculated Diad 2 position doesnt match test")

    def test_st_pk2_area(self):
        self.assertAlmostEqual(Diad2_fit_strong['Diad2_Voigt_Area'].iloc[0],79240.29545872404,
0, "Calculated Diad 2 area doesnt match test")

    def test_st_pk2_err(self):
        self.assertAlmostEqual(Diad2_fit_strong['Diad2_cent_err'].iloc[0],0.0006681704840915489,
4, "Calculated Diad 2 error doesnt match test")




