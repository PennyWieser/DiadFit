import unittest
import pandas as pd
import DiadFit as pf


decimalPlace=4
class test_CO2_Bubble(unittest.TestCase):
    def test_CO2_Bubble(self):
        self.assertAlmostEqual(bub=pf.propagate_CO2_in_bubble_ind(sample_i=0,  N_dup=1,
vol_perc_bub=5,
error_vol_perc_bub=50, error_type_vol_perc_bub='Perc',
error_dist_vol_perc_bub='normal',
CO2_bub_dens_gcm3=0.1, error_CO2_bub_dens_gcm3=0.02,
melt_dens_kgm3=2700, error_melt_dens_kgm3=200,
len_loop=1, neg_values=True)['CO2_eq_melt_ppm_noMC'], 1851.851852,
decimalPlace, "Calculated CO2 in bubble (no MC) doesnt match test value")






