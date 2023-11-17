import unittest
import pandas as pd
import DiadFit as pf
import os
spectra_path = os.path.dirname(os.path.realpath(__file__))

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


decimalPlace=4
class test_Ne_files(unittest.TestCase):
    def test_Ne_line_get_files(self):
        self.assertEqual(pf.get_files(path=spectra_path,
file_ext='txt', ID_str='Ne',
exclude_str=['diad'], sort=False)[0], '01 Ne_lines_1.txt',
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












