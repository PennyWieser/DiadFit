================
Fitting Fermi Diads
================


DiadFit uses a 4-5 step workflow to fit neon lines, fit diads, and inspect for secondary peaks. We provide a number of worked examples below.

Step 1 fits Neon lines to build a model of instrument drift throughout the day.
Step 2 performs a preliminary fit on diad spectra to classify them into groups (this allows tweaking of instrument parameters for each group to optimize fits)
Step 3 fits diad spectra.
Step 3b (optional) looks for secondary peaks such as SO2 and carbonate
Step 4 merges the diad and secondary peak outputs, and applies the Neon correction model to obtain corrected splitting.



Example 1a - Gas Calibration Data
-----------------------------------

This set of notebooks shows data collected on a WITEC alpha300R from a gas calibration cell (DeVitre et al. 2021). All files are headless_txt files.
Spectral files for Neon and diads are stored in the 'Spectra' folder, and metadata files in the 'MetaData' folder.
These spectra files show a wide range of spectral strengths, so are subdivided into weak, medium and strong spectra prior to fitting in step 3.
We do not include Step3b, because spectra were collected on pure CO2 gas in a calibration cell, so there are no impurities.

See :doc:`Step 1 <Examples/Fitting_Fermi_Diads/Example1a_Gas_Cell_Calibration/Step1_Fit_Your_Ne_Lines>`,
:doc:`Step 2 <Examples/Fitting_Fermi_Diads/Example1a_Gas_Cell_Calibration/Step2_Filtering_Numerical>`,
:doc:`Step 3 <Examples/Fitting_Fermi_Diads/Example1a_Gas_Cell_Calibration/Step3_FitAll_Together>`,
:doc:`Step 4 <Examples/Fitting_Fermi_Diads/Example1a_Gas_Cell_Calibration/Step4_Stitch_Outputs_Together>`

Example 1b - Natural Fluid inclusions
-----------------------------------
This set of notebooks shows data collected from natural fluid inclusions from Kilauea volcano on a WITEC alpha300R. All files are headless_txt files.
Spectral files for Neon and diads are stored in the 'Spectra' folder, and metadata files in the 'MetaData' folder.
All spectra show very similar shapes, so we classify them all as 'Weak'. Additionally, unlike example 1a, these fluid inclusions also contain SO$_2$ peaks, so we utilize step3b.

See :doc:`Step 1 <Examples/Fitting_Fermi_Diads/Example1b_CO2_Fluid_Inclusions/Step1_Fit_Your_Ne_Lines>`,
:doc:`Step 2 <Examples/Fitting_Fermi_Diads/Example1b_CO2_Fluid_Inclusions/Step2_Filtering_Numerical>`,
:doc:`Step 3 <Examples/Fitting_Fermi_Diads/Example1b_CO2_Fluid_Inclusions/Step3_FitAll_Together>`,
:doc:`Step 3b <Examples/Fitting_Fermi_Diads/Example1b_CO2_Fluid_Inclusions/Step3b(optional)_Secondary_Peaks>`,
:doc:`Step 4 <Examples/Fitting_Fermi_Diads/Example1b_CO2_Fluid_Inclusions/Step4_Stitch_Outputs_Together>`,


Example 1bb - Nasty backgrounds
-----------------------------------
:doc:`Example 1bb <Examples/Fitting_Fermi_Diads/Example1bb_highbackground_FIs/Step2_Filtering_Numerical>`  shows how to filter out spectra which have a very slanted background, which you may want to fit with a separate set of peak parameters, or exclude entirely.




Example 1c - HORIBA synthetic fluid inclusions
-----------------------------------
This set of notebooks shows data from Neon lines and qtz fluid inclusions collected on an older HORIBA instrument. The spectral resolution of this instrument is low, which makes peak fitting more of a challenge.

see :doc:`Step 1 <Examples/Fitting_Fermi_Diads/Example1c_HORIBA_Calibration/Step1_Fit_Your_Ne_Lines>`,
:doc:`Step 2 <Examples/Fitting_Fermi_Diads/Example1c_HORIBA_Calibration/Step2_Filtering_Numerical>`,
:doc:`Step 3 <Examples/Fitting_Fermi_Diads/Example1c_HORIBA_Calibration/Step3_FitAll_Together>`,
:doc:`Step 4 <Examples/Fitting_Fermi_Diads/Example1c_HORIBA_Calibration/Step4_Stitch_Outputs_Together>`


Example 1d- Newer HORIBA
-----------------------------------
Coming soon!

Example 1e - Quick Peak fitting
-----------------------------------
Sometimes when you are Raman-ing a new set of samples, you want an approximate indication of what densities you are dealing with (e..g 0.2 g/cm3, 0.5 g/cm3).
:doc:`Example 1e <Examples/Fitting_Fermi_Diads/Example1e_Quick_Peak_Fitting_While_Ramaning/Quick_Peak_fitting>`  shows how to quickly peak fit a few spectra you have grabbed off the instrument, without doing all 4-5 steps. All spectra are stored in the folder 'Spectra'. We dont worry about Metadata,
we just use an average Neon line correction factor for our instrument.




Example 1f - Quantifying Peak assymmetry
-----------------------------------
:doc:`Example 1f <Examples/Fitting_Fermi_Diads/Example1f_Diad_Peak_Assymetry/Asessing_Diad_Skewness>` follows the method of DeVitre et al. (2023), which demonstrates that peak assymetry can be used to identify the presence of both vapour and liquid phases. We use some of their spectra to demonstrate this method further.





