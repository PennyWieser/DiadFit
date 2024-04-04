
================
Change Log
================

March 29th, 2024 - Penny Wieser - V.0.86
-------------------------------------------
Was an issue in the math for the CO2-H2O EOS in the situation where H2O wasnt loss - mass ratio not calculated correctly. Caused convergence issues.
Also fixed bug in MC code for FI - was using error from first sample for all samples (clear in output columns)


March 7th, 2024 - Penny Wieser - V.0.85
-------------------------------------------

Duan and Zhang (2006) inverts for pressure from density using an adapted newton raphson method. At high densities, it was finding a negative root.

Built in a look up table to get starting pressure. Helps convergence. Now gives Nans for things in the look up table that yielded negative volumes.

Feb 4th, 2024 - Penny Wieser - V.0.82
-------------------------------------------
Changes after review. Major restructuring of docs.
When loading files, options for spectra_filetype, meta_file_ext, spectra_file_ext, old code with new versions will return error for these.
No real way to make backwards compatible and get functionality for reviewer 2.

Also simplified Ne fitting to only use 1 function for baseline and peak, means can have 2 peaks for either pk1 or pk2

Dec 18th, 2023 - Penny Wieser - V.0.81
-------------------------------------------
Added options to calculate mixed fluids pressure assuming H2O is lost, and not lost.


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
