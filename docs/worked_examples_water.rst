
================
Quantifying melt water contents
================

Example 4a -  Just Glasses
-----------------------------------
:doc:`Example 4a <Examples/Fitting_Water_Silicate_Melts/Example4a_H2OQuant_Glass/H2O_Fitting>` shows how to quantify the relative peak areas of the silicate and water peaks. There are default peak positions for different compositions, and these numbers can also be fully adjusted. This Notebook allows the user to fit each spectra individually, and then the results are merged together at the end. This is useful if you want to tweak the background positions for each individual spectra to get better fits.



Example 4b, 4c, 4d -  Unmixing olivine and melt inclusions
---------------------------------------------------
Acquiring Raman spectra in unexposed melt inclusions is very helpful when performing the carbonate rehomogenization techniques of DeVitre et al. (2021) to monitor for diffusive water loss. This method relies on acquiring a spectra in the center of the melt inclusion with the strongest H2O signal, and then a signal on the host olivine next to it. We provide examples showing a variety of different workflows.

:doc:`Example 4b <Examples/Fitting_Water_Silicate_Melts/Example4b_H2OQuant_MI/H2O_Fitting_MI_AutoLoop>` loops throough a set of MI and olivine analyses, applying the same fit parameters to all of them

:doc:`Example 4b <Examples/Fitting_Water_Silicate_Melts/Example4b_H2OQuant_MI/H2O_Fitting_MI_ManualLoop>` allows the user to manually loop through files to apply different fit parameters to all of them.

