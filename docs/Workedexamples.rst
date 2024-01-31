================
Worked Examples
================


Here, we briefly describe the different worked examples available in DiadFit.
If there is a common workflow you dont see represented, or you would like to contribute some spectra or metadata for us to make an example for another instrument, please get in touch!


Fitting Fermi Diads
=============================

The following examples all follow the same basic workflows (+- certain steps) for fitting Fermi diads, using 5 different jupyter notebooks.

Step 1 fits Neon lines to build a model of instrument drift throughout the day.
Step 2 performs a preliminary fit on diad spectra to classify them into groups (this allows tweaking of instrument parameters for each group to optimize fits)
Step 3 fits diad spectra.
Step 3b (optional) looks for secondary peaks such as SO2 and carbonate
Step 4 merges the diad and secondary peak outputs, and applies the Neon correction model to obtain corrected splitting.



Example 1a - Gas Calibration Data
-----------------------------------
This example shows data collected on a WITEC alpha300R from a gas calibration cell (DeVitre et al. 2021). All files are headless_txt files.
Spectral files for Neon and diads are stored in the 'Spectra' folder, and metadata files in the 'MetaData' folder.
These spectra files show a wide range of spectral strengths, so are subdivided into weak, medium and strong spectra prior to fitting in step 3.
We do not include Step3b, because spectra were collected on pure CO2 gas in a calibration cell, so there are no impurities.


Example 1b - Natural Fluid inclusions
-----------------------------------
This example shows data collected from natural fluid inclusions from Kilauea volcano on a WITEC alpha300R. All files are headless_txt files.
Spectral files for Neon and diads are stored in the 'Spectra' folder, and metadata files in the 'MetaData' folder.
All spectra show very similar shapes, so we classify them all as 'Weak'. Additionally, unlike example 1a, these fluid inclusions also contain SO$_2$ peaks, so we utilize step3b.


Example 1bb - Nasty backgrounds
-----------------------------------
This example shows how to filter out spectra which have a very slanted background, which you may want to fit with a separate set of peak parameters, or exclude entirely.


Example 1c - HORIBA synthetic fluid inclusions
-----------------------------------
This example shows data from Neon lines and qtz fluid inclusions collected on an older HORIBA instrument. The spectral resolution of this instrument is low, which makes peak fitting more of a challenge.


Example 1d- Newer HORIBA
-----------------------------------
This example shows data supplied by Lowell Moore on the Virginia Tech HORIBA. (Penny to get some newer data)


Example 1e - Quick Peak fitting
-----------------------------------
Sometimes when you are Ramining a new set of samples, you want an approximate indication of what densities you are dealing with (e..g 0.2 g/cm3, 0.5 g/cm3).
This notebook shows how to quickly peak fit a few spectra you have grabbed off the instrument, without doing all 4-5 steps. All spectra are stored in the folder 'Spectra'. We dont worry about Metadata,
we just use an average Neon line correction factor for our instrument.

Example 1f - Quantifying Peak assymmetry
-----------------------------------
DeVitre et al. (2023) show that peak assymetry can be used to identify the presence of both vapour and liquid phases. This example shows how to quantify peak assymetry on some spectra from DeVitre et al. (2023)


Quantifying water contents in silicate melts
===============================================
Example 4 -  Just Glasses
-----------------------------------

Example 4b -  Unmixing olivine and melt inclusions
-----------------------------------

Equation of state (EOS) calculations
=======================================
DiadFit includes the CO$_2$ equation of state of Sterner and Pitzer (1994) and Span and Wagner (1996), as well as the mixed H$_2$O-CO$_2$ EOS of Duan and Zhang (2006).
These EOS can be used for a variety of different calculations, described below:

Example 5a -  Different EOS functions
--------------------------------------------------------------
This example shows how to perform different CO2 EOS calculations in DiadFit:
    - Calc 1: Calculating P for a given T and CO2 density
    - Calc 2: Calculating CO2 density for a given T and P
    - Calc 3: Calculating T for a given P and CO2 density.
    - Calc 4: Calculating co-existing liquid and vapour densities
    - Calc 5: Converting homogenization temperatures from microthermometry into CO2 densities, and propagating errors.



Example 5b -  Visualizing how CO$_2$ density relates to P and T
--------------------------------------------------------------
This example shows how to perform EOS calculations of CO2 density for an array of pressures at different temperatures.
The resulting plots in pressure-density space with lines for different temperatures are very helpful to demonstrate that the CO2 EOS isn't that sensitive to temperature.

Example 5c -  Calculating fluid inclusion entrapment pressures and depths in La Palma
--------------------------------------------------------------
This example uses CO2 densities from Dayton et al. (2022, Science Advances) to calculate entrapment pressures, and then a 2 step density profile to calculate storage depths

Example 5d -  Calculating fluid inclusion entrapment pressures and depths for different density profiles
-----------------------------------------------------------------------------------------------------------
This notebook shows how to convert CO2 density to depth, and then calculate storage depths using a variety of crustal density profiles (2, 3 step, etc).

Example 5e -  Propagating uncertainties in fluid inclusion barometry
---------------------------------------------------------------------
This notebook shows how to propagate uncertainties in temperature, CO2 density and crustal density using Monte Carlo methods.

Example 5f -  Calculations using CO$_2$-H$_2$O EOS
---------------------------------------------------------------------
This notebook shows how to perform calculations using CO2-H2O EOs, and how to integrate XH2O measurements from melt inclusions into this correction.

Quantifying uncertainty in the CO$_2$ contents of melt inclusion vapour bubbles
================================================================================

Example 8 - Propagating CO2 Uncertainties
-----------------------------------
This notebook shows how to propagate uncertainty in bubble densities, bubble volumes and silicate melt densities into equivalent CO2 contents in glasses.




Modelling Fluid inclusion re-equilibration
===============================================
These examples show how to use the code of DeVitre and Wieser (2024) to model fluid inclusion re-equilibration during ascent towards the surface.

Example 9a -  Stretching during ascent
--------------------------------------
This example shows how to model stretching of a 1um radius CO2 fluid inclusion during ascent from 10 km depth to the surface.
This can be easily adapted for different starting pressures, inclusion sizes, and ascent rates

Example 9a -  Stretching during quenching
--------------------------------------
This example shows how to model fluid inclusion stretching that occurs during syn-eruptive quenching on the surface (e.g. in a lava flow)

Example 9c -  Stretching during stalling
--------------------------------------
This example shows how to model fluid inclusion stretching that occurs after a magma and stalls in a shallower reservoir.


Other Useful Functions
======================================

Example 10 -  Crustal Density Profiles
--------------------------------------
This example shows how to compare different crustal density profiles you may want to use to convert fluid inclusion pressures to depths.