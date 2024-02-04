=======================================
Equation of state (EOS) calculations
=======================================
DiadFit includes the CO2 equation of state of Sterner and Pitzer (1994) and Span and Wagner (1996), as well as the mixed H2O-CO2 EOS of Duan and Zhang (2006).
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
:doc:`Example 5b <Examples/EOS_calculations/Example5b_Visualizing_EOSs_Density_Pressure>` shows how to perform EOS calculations of CO2 density for an array of pressures at different temperatures.
The resulting plots in pressure-density space with lines for different temperatures are very helpful to demonstrate that the CO2 EOS isn't that sensitive to temperature.

Example 5c -  Calculating fluid inclusion entrapment pressures and depths in La Palma
--------------------------------------------------------------
:doc:`Example 5c <Examples/EOS_calculations/Example5c_LaPalma_FluidInclusions>` uses CO2 densities from Dayton et al. (2022, Science Advances) to calculate entrapment pressures, and then a 2 step density profile to calculate storage depths

Example 5d -  Calculating fluid inclusion entrapment pressures and depths for different density profiles
-----------------------------------------------------------------------------------------------------------
:doc:`Example 5d <Examples/EOS_calculations/Example5d_Fluid_Inclusion_Density_to_Depth>` shows how to convert CO2 density to depth, and then calculate storage depths using a variety of crustal density profiles (2, 3 step, etc).

Example 5e -  Propagating uncertainties in fluid inclusion barometry
---------------------------------------------------------------------
:doc:`Example 5e <Examples/EOS_calculations/Example5e_FI_Monte_Carlo_Simulations>` shows how to propagate uncertainties in temperature, CO2 density and crustal density using Monte Carlo methods.

Example 5f -  Calculations using CO$_2$-H$_2$O EOS
---------------------------------------------------------------------
:doc:`Example 5f <Examples/EOS_calculations/Example5f_H2O_CO2_EOS>` shows how to perform calculations using CO2-H2O EOs, and how to integrate XH2O measurements from melt inclusions into this correction.
