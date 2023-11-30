
================
Change Log
================
Nov 29th, 2023 - Penny Wieser - V.0.80
-------------------------------------------
Added significant functionality for the CO2-H2O EOS of Duan and Zhang (2006)
Oct 25th, 2023 - Penny Wieser - v.0.78
-----------------------------
Major tweak of the secondary peak finding function. Instead of using a sigma filter, uses prominence_filter,
identifies peaks as if the highest peak is above the median of the two edges of the window.
Old sigma_filter syntax will return error


June 23rd, 2023 - Penny Wieser
-----------------------------------
Changed function 'plot_and_save_Ne_line_pickle' to 'generate_Ne_corr_model'

March 28th, 2023 - Penny Wieser
------------------------------------
Changed 'calculate_rho_gcm3_for_P_T' to 'calculate_rho_for_P_T' for EOS calcs

March 22nd, 2023 - Penny Wieser
-------------------------------------
__version__ = '0.0.53'
Changed arguement in diad2_fit_config to be C13_prom not C13_abs_prom for consistency

Changed cent_generic to cent for secondary peak

Changelog started March 22nd, 2023, as syntax converging across different functions
