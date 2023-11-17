import unittest
import pandas as pd
import DiadFit as pf


decimalPlace=4
class test_SP94_EOS(unittest.TestCase):
    def test_SP94_pressure_to_density(self):
        self.assertAlmostEqual(pf.calculate_rho_for_P_T(P_kbar=1, 
T_K=1400, EOS='SP94')[0], 0.297958,
decimalPlace, "Calculated SP94 P doesnt match test value")
        



