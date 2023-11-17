import unittest
import pandas as pd
import DiadFit as pf
import os
spectra_path = os.path.dirname(os.path.realpath(__file__))

# Lets get the files
Ne_files=pf.get_files(path=spectra_path,
file_ext='txt', ID_str='Ne',
exclude_str=['diad'], sort=False)[0]
wavelength=532.046


decimalPlace=4
class test_Ne_line(unittest.TestCase):
    def test_Ne_line_get_files(self):
        self.assertAlmostEqual(pf.get_files(path=spectra_path,
file_ext='txt', ID_str='Ne',
exclude_str=['diad'], sort=False)[0], '01 Ne_lines_1.txt',
decimalPlace, "loaded file doesnt match")

    def test_Ne_line_position(self):
        self.assertAlmostEqual(pf.calculate_Ne_line_positions(wavelength=532.046,
cut_off_intensity=2000)['Raman_shift (cm-1)'].iloc[0], 391.53636168859157,
decimalPlace, "Calculated line position doesnt match")

    def test_Ne_line_distance(self):
        self.assertAlmostEqual(pf.calculate_Ne_splitting(wavelength=532.046,
line1_shift=1117, line2_shift=1447,
cut_off_intensity=2000)['Ne_Split'].iloc[0], 330.47763434663284,
decimalPlace, "Calculated Ne splitting doesnt match")












