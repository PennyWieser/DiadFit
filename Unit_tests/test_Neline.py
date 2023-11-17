import unittest
import pandas as pd
import DiadFit as pf
import os
spectra_path = os.path.dirname(os.path.realpath(__file__))

#
prefix=True
# Lets get the files
Ne_files=pf.get_files(path=spectra_path,
file_ext='txt', ID_str='Ne',
exclude_str=['diad'], sort=False)



# Lets define the Ne set up
exclude_range_1=None
exclude_range_2=None
line_1=1117
line_2=1447
Neon_id_config=pf.Neon_id_config(height=10,  distance=1, prominence=10, 
            width=1, threshold=0.6,
            peak1_cent=line_1, peak2_cent=line_2, n_peaks=6, 
            exclude_range_1=exclude_range_1, 
            exclude_range_2=exclude_range_2)

Ne, df_fit_params=pf.identify_Ne_lines(path=spectra_path,
filename='01 Ne_lines_1.txt', filetype='headless_txt',
config=Neon_id_config, print_df=False)




# Lets set up the lmfit. 
model_name='PseudoVoigtModel'
Ne_Config_est=pf.Ne_peak_config(model_name=model_name,
 DeltaNe_ideal=330.47763434663284, peaks_1=2, LH_offset_mini=[0.5, 3],
pk1_sigma=0.6, pk2_sigma=0.3,
lower_bck_pk1=(-40, -25), upper_bck1_pk1=[40, 70], upper_bck2_pk1=[40, 70],
lower_bck_pk2=[-40, -30], upper_bck1_pk2=[10, 15], upper_bck2_pk2=[25, 40],
x_range_peak=5, x_span_pk1=[-10, 8], x_span_pk2=[-10, 10],
N_poly_pk2_baseline=2 )

# Lets do the fit
fit=pf.fit_Ne_lines(Ne=Ne, filename='01 Ne_lines_1.txt',
path=spectra_path, prefix=prefix,
config=Ne_Config_est,
    Ne_center_1=df_fit_params['Peak1_cent'].iloc[0], 
    Ne_center_2=df_fit_params['Peak2_cent'].iloc[0],
    Ne_prom_1=df_fit_params['Peak1_prom'].iloc[0],
    Ne_prom_2=df_fit_params['Peak2_prom'].iloc[0],
    const_params=False)



decimalPlace=4
class test_Ne_files(unittest.TestCase):
    def test_Ne_line_get_files(self):
        self.assertEqual(pf.get_files(path=spectra_path,
file_ext='txt', ID_str='Ne',
exclude_str=['diad'], sort=True)[0], '01 Ne_lines_1.txt',
 "loaded file doesnt match")
        
class test_Ne_line_pos(unittest.TestCase):
    def test_Ne_line_position(self):
        self.assertAlmostEqual(pf.calculate_Ne_line_positions(wavelength=532.046,
cut_off_intensity=2000)['Raman_shift (cm-1)'].iloc[0], 391.53636168859157,
decimalPlace, "Calculated line position doesnt match")
        
class test_Ne_line_distance(unittest.TestCase):
    def test_Ne_line_distance(self):
        self.assertAlmostEqual(pf.calculate_Ne_splitting(wavelength=532.046,
line1_shift=1117, line2_shift=1447,
cut_off_intensity=2000)['Ne_Split'].iloc[0], 330.47763434663284,
decimalPlace, "Calculated Ne splitting doesnt match")

class test_scipy_peaks(unittest.TestCase):       
    def test_scipy_peaks(self):
        self.assertAlmostEqual(pf.identify_Ne_lines(path=spectra_path,
filename='01 Ne_lines_1.txt', filetype='headless_txt',
config=Neon_id_config, print_df=False)[1]['Peak1_cent'].iloc[0], 1116.16583,
decimalPlace, "Calculated Pk1Center Scipy doesnt match test value")
        


class test_peak_pos(unittest.TestCase):       
    def test_fit_pk2(self):
        self.assertAlmostEqual(fit['pk2_peak_cent'].iloc[0], 1447.5834836992235,
decimalPlace, "Fitted pk2 Center doesnt match test value")

    def test_fit_pk1(self):
        self.assertAlmostEqual(fit['pk1_peak_cent'].iloc[0], 1116.3366243217165,
decimalPlace, "Fitted pk1 Center doesnt match test value")

    def test_fit_corr(self):
        self.assertAlmostEqual(fit['Ne_Corr'].iloc[0], 0.997678,
4, "Fitted pk1 Center doesnt match test value")

    def test_fit_corr_err(self):
        self.assertAlmostEqual(fit['Ne_Corr_min'].iloc[0], 0.997633,
4, "Fitted pk1 Center doesnt match test value")









