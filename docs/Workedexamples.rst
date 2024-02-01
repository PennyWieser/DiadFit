================
Worked Examples
================


Here, we briefly describe the different worked examples available in DiadFit.
If there is a common workflow you dont see represented, or you would like to contribute some spectra or metadata for us to make an example for another instrument, please get in touch!


Fitting Fermi Diads
=============================

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
We do not include Step3b, because spectra were collected on pure CO :sub:`2` gas in a calibration cell, so there are no impurities.

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
:doc:`Examle 1bb <Examples/Fitting_Fermi_Diads/Example1bb_highbackground_FIs/Step2_Filtering_Numerical>`example shows how to filter out spectra which have a very slanted background, which you may want to fit with a separate set of peak parameters, or exclude entirely.




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

see


Example 1f - Quantifying Peak assymmetry
-----------------------------------
:doc:`Example 1f <Examples/Fitting_Fermi_Diads/Example1f_Diad_Peak_Assymetry/Asessing_Diad_Skewness>` follows the method of DeVitre et al. (2023), which demonstrates that peak assymetry can be used to identify the presence of both vapour and liquid phases. We use some of their spectra to demonstrate this method further.


Quantifying water contents in silicate melts
===============================================
Example 4a -  Just Glasses
-----------------------------------
:doc:`Example 1f <Examples/Fitting_Fermi_Diads/Fitting_Water_Silicate_Melts\Example4a_H2OQuant_Glass>` shows how to quantify the relative peak areas of the silicate and water peaks. There are default peak positions for different compositions, and these numbers can also be fully adjusted. This Notebook allows the user to fit each spectra individually, and then the results are merged together at the end. This is useful if you want to tweak the background positions for each individual spectra to get better fits.



Example 4b, 4c, 4d -  Unmixing olivine and melt inclusions
---------------------------------------------------
Acquiring Raman spectra in unexposed melt inclusions is very helpful when performin the carbonate rehomogenization techniques of DeVitre et al. (2021) to monitor for diffusive water loss. This method relies on acquiring a spectra in the center of the melt inclusion with the strongest H2O signal, and then a signal on the host olivine next to it. We provide examples showing a variety of different workflows.

:doc:`Example 4b <Examples/Fitting_Water_Silicate_Melts\Example4b_H2OQuant_MI/H2O_Fitting_MI_AutoLoop>` loops throough a set of MI and olivine analyses, applying the same fit parameters to all of them

:doc:`Example 4b <Examples/Fitting_Water_Silicate_Melts\Example4b_H2OQuant_MI/H2O_Fitting_MI_ManualLoop>` allows the user to manually loop through files to apply different fit parameters to all of them.



Equation of state (EOS) calculations
=======================================
DiadFit includes the CO :sub:`2` equation of state of Sterner and Pitzer (1994) and Span and Wagner (1996), as well as the mixed H/ :sub:`2`O-CO :sub:`2` EOS of Duan and Zhang (2006).
These EOS can be used for a variety of different calculations, described below:

Example 5a -  Different EOS functions
--------------------------------------------------------------
:doc:`Example 5a <Examples/EOS_calculations/Example5a_Introducing_EOS_Calcs>` shows how to perform different CO2 EOS calculations in DiadFit:
    - Calc 1: Calculating P for a given T and CO2 density.
    - Calc 2: Calculating CO2 density for a given T and P
    - Calc 3: Calculating T for a given P and CO2 density.
    - Calc 4: Calculating co-existing liquid and vapour densities
    - Calc 5: Converting homogenization temperatures from microthermometry into CO2 densities, and propagating errors.



Example 5b -  Visualizing how CO$_2$ density relates to P and T
--------------------------------------------------------------
:doc:`Example 5b <Examples/EOS_calculations/Example5b_Visualizing_EOSs_Density_Pressure>`
 shows how to perform EOS calculations of CO2 density for an array of pressures at different temperatures.
The resulting plots in pressure-density space with lines for different temperatures are very helpful to demonstrate that the CO :sub:`2` EOS isn't that sensitive to temperature.

Example 5c -  Calculating fluid inclusion entrapment pressures and depths in La Palma
--------------------------------------------------------------
:doc:`Example 5c <Examples/EOS_calculations/Example5c_LaPalma_FluidInclusions>` uses CO :sub:`2` densities from Dayton et al. (2022, Science Advances) to calculate entrapment pressures, and then a 2 step density profile to calculate storage depths

Example 5d -  Calculating fluid inclusion entrapment pressures and depths for different density profiles
-----------------------------------------------------------------------------------------------------------
:doc:`Example 5d <Examples/EOS_calculations/Example5d_Fluid_Inclusion_Density_to_Depth>` shows how to convert CO :sub:`2` density to depth, and then calculate storage depths using a variety of crustal density profiles (2, 3 step, etc).

Example 5e -  Propagating uncertainties in fluid inclusion barometry
---------------------------------------------------------------------
:doc:`Example 5e <Examples/EOS_calculations/Example5e_FI_Monte_Carlo_Simulations>` shows how to propagate uncertainties in temperature, CO :sub:`2` density and crustal density using Monte Carlo methods.

Example 5f -  Calculations using CO$_2$-H$_2$O EOS
---------------------------------------------------------------------
:doc:`Example 5f <Examples/EOS_calculations/Example5f_H2O_CO2_EOS>` shows how to perform calculations using CO :sub:`2`-H/ :sub:`2`O EOs, and how to integrate XH/ :sub:`2`O measurements from melt inclusions into this correction.


Quantifying uncertainty in the CO$_2$ contents of melt inclusion vapour bubbles
================================================================================

Example 8 - Propagating CO2 Uncertainties
-----------------------------------
T:doc:`Example 8a <Examples/CO2_in_Melt_Inclusion_Vapour_Bubbles/Example8a_PropagatingCO2Uncertainties>` shows how to propagate uncertainty in bubble densities, bubble volumes and silicate melt densities into equivalent CO2 contents in glasses.




Modelling Fluid inclusion re-equilibration
===============================================
These examples show how to use the code of DeVitre and Wieser (2024) to model fluid inclusion re-equilibration during ascent towards the surface.

Example 9a -  Stretching during ascent
--------------------------------------
:doc:`Example 9a <Examples/Modelling_Fluid_Inclusion_Re-equilibration/Example9a_FI_stretching_during_ascent>`  shows how to model stretching of a 1 um radius CO2 fluid inclusion during ascent from 10 km depth to the surface.
This can be easily adapted for different starting pressures, inclusion sizes, and ascent rates

Example 9b -  Stretching during quenching
--------------------------------------
:doc:`Example 9b <Examples/Modelling_Fluid_Inclusion_Re-equilibration/Example9b_FI_stretching_slow_quenching_at_surface>` shows how to model fluid inclusion stretching that occurs during syn-eruptive quenching on the surface (e.g. in a lava flow)

Example 9c -  Stretching during stalling
--------------------------------------
:doc:`Example 9c <Examples/Modelling_Fluid_Inclusion_Re-equilibration/Example9c_FI_stretching_during_stalling>` shows how to model fluid inclusion stretching that occurs after a magma and stalls in a shallower reservoir.


Other Useful Functions
======================================

Example 10 -  Crustal Density Profiles
--------------------------------------
:doc:`Example 9c <Examples/Other_Useful_Functions/Example10_Different_Crustal_Density_Profiles>` shows how to compare different crustal density profiles you may want to use to convert fluid inclusion pressures to depths.

