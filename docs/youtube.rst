#################
YouTube Tutorials
#################
.. contents::

Subscribe to `DiadFit channel on youtube <https://www.youtube.com/@diadfit3888>`_ to get notified when we create new tutorial videos.

We recommend readers go through the associated notebooks for each of their tutorials in their own time to read the detailed descriptions.


Why did we develop DiadFit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A short introduction to the motivation behind developing DiadFit

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/0u9RdE1lrYY?si=qgpyvcWFWrlo6Lzt" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
---------

How to get files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This video shows how to get files of GitHub and read the docs for people who have not done this before!

.. raw:: html
    <iframe width="560" height="315" src="https://www.youtube.com/embed/IaAdUBvlndM?si=n3eSR0Ikvf8AJB5y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

WITEC Diad Fitting Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^
This playlist shows how to fit CO2 peaks in a 5 step workflow (tweaked for WITEC file types, but could be easily altered)

.. raw:: html
    <iframe width="560" height="315" src="https://www.youtube.com/embed/an14NBzNZW0?si=hUG4cmtgrLj4X3LE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


CO2 EOS calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^
This video introduces the functions in DiadFit associated with the CO2 equation of state. If you know two of pressure, temperature, or CO2 density, you can calculate the third unknown. We show how to visualize differences between the results of the Sterner and Pitzer (1994) and Span and Wanger (1996) EOS. We also show how to convert homogenization temps from microthermometry into CO2 densities, and how to calculate the maximum permitted densities at room T of a two-phase inclusion. You will need to have installed CoolProp to use the Span and Wanger EOS. Either  pip install CoolProp or if you use anaconda, see these! https://anaconda.org/conda-forge/coolprop

**Useful files:

    * Download :download:`Python Notebook (Example5f_H2O_CO2_EOS.ipynb) <Examples/EOS_calculations/Example5f_H2O_CO2_EOS.ipynb>`
    * Download :download:`Excel Workbook (FI_densities.xlsx) <Examples/EOS_calculations/FI_densities.xlsx>`

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/sysmHrVrMR8?si=dOg0wDX9813KrB5Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


Propagating uncertainty for fluid inclusions from La Palma
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This video recreates calculations performed in Dayton et al. (2023).

.. raw:: html
    <iframe width="560" height="315" src="https://www.youtube.com/embed/pM5LfnLRySg?si=mvHVr3hJwIATNEGt" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

CO2-H2O EOS calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^
This video shows how to perform EOS calculations using the CO2-H2O EOS of Duan and Zhang (2006).

**Useful files:

    * Download :download:`Python Notebook (Example5f_H2O_CO2_EOS.ipynb) <Examples/EOS_calculations/Example5f_H2O_CO2_EOS.ipynb>`
    * Download :download:`Excel Workbook (FI_densities.xlsx) <Examples/EOS_calculations/FI_densities.xlsx>`


.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/x_ixS3HtdMc?si=zKDSnIuPNXDjU7nr" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
---------



Olivine-Liquid Thermometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^
This video shows how to calculate Olivine-Liquid temperatures, as well as:
    * various ways to assess whether Ol-Liq pairs are in equilibrium
    * calculating equilibrium Ol Fo contents using just a liquid composition for a variety of Kd models
    * plotting olivine and liquid compositions on a Rhodes diagram
    * converting buffer values to Fe3FeT proportions


**Useful files:

    * Download :download:`Python Notebook (Olivine_Liquid_thermometry.ipynb) <Examples/Liquid_Ol_Liq_Themometry/Olivine_Liquid_thermometry.ipynb>`
    * Download :download:`Excel Workbook (Liquid_only_Thermometry.xlsx) <Examples/Liquid_Ol_Liq_Themometry/Liquid_only_Thermometry.xlsx>`

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/IkSROME78IE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
---------

Clinopyroxene-Liquid Melt Matching 1 (simpler)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This video recreates the Cpx-Liq melt matching results from :cite:`gleeson2020upper`.
It shows how to calculate all possible matches between inputted Cpx and Liq compositions, and how to change the equilibrium filters for assessing equilibrium matches (Kd, EnFs, DiHd, CaTs), how to plot calculated pressures and temperatures etc.

**Useful files:

    * Download :download:`Python Notebook (Cpx_MeltMatch1_Gleeson2020.ipynb) <Examples/Cpx_Cpx_Liq_Thermobarometry/Cpx_Liquid_melt_matching/Cpx_MeltMatch1_Gleeson2020.ipynb>`
    * Download :download:`Excel Workbook (Gleeson2020JPET_Input_Pyroxene_Melts.xlsx) <Examples/Cpx_Cpx_Liq_Thermobarometry/Cpx_Liquid_melt_matching/Gleeson2020JPET_Input_Pyroxene_Melts.xlsx>`


.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/8cz37AtGSHc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
---------


Clinopyroxene-Liquid Melt Matching 2 (advanced)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This video builds on the video above, and shows how the approach of :cite:`scruggs2018eruption` can be recreated in python.

Synthetic liquid compositions are produced by adding noise and bootstrapping, and then all possible matches are considered between measured liquids + synthetic liquids + measured Cpxs.

**Useful files:

    * Download :download:`Python Notebook (Cpx_MeltMatch2_ScruggsPutirka2018.ipynb) <Examples/Cpx_Cpx_Liq_Thermobarometry/Cpx_Liquid_melt_matching/Cpx_MeltMatch2_ScruggsPutirka2018.ipynb>`
    * Download :download:`Excel Workbook (Scruggs_Input.xlsx) <Examples/Cpx_Cpx_Liq_Thermobarometry/Cpx_Liquid_melt_matching/Scruggs_Input.xlsx>`


.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/gCyFB6z5hT4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
---------


Amphibole-only and Amphibole-Liquid  (advanced)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Demonstrates amphibole-only and amphibole-Liquid thermobarometry, hygrometry and chemometry, including equilibrium tests.

**Useful files:

    * Download :download:`Python Notebook (Amphibole_Examples.ipynb) <Examples/Amphibole/Amphibole_Examples.ipynb>`
    * Download :download:`Excel Workbook (Amphibole_Liquids.xlsx) <Examples/Amphibole/Amphibole_Liquids.xlsx>`



.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/yEsPwglCN80" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
---------



Feldspar-Liquid thermobarometry and hygrometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Demonstrates plagioclase-liquid and kspar-liquid thermometry, and plagioclase-liquid hygrometry. We discuss equilibrium tests, and iteration between Temp and H2O for hygrometers.


**Useful files:

    * Download :download:`Python Notebook (Feldspar_Liquid.ipynb) <Examples/Feldspar_Thermobarometry/Feldspar_Liquid.ipynb>`
    * Download :download:`Excel Workbook (Feldspar_Liquid.xlsx) <Examples/Feldspar_Thermobarometry/Feldspar_Liquid.xlsx>`



.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/ahYGgBG4gHM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
---------



Two Feldspar thermometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Demonstrates two feldspar thermometry, along with discussion of how to apply various equilibrium filters.

**Useful files:

    * Download :download:`Python Notebook (Two_Feldspar_Example.ipynb) <Examples/Feldspar_Thermobarometry/Two_Feldspar_Example.ipynb>`
    * Download :download:`Excel Workbook (Two_Feldspar_input.xlsx) <Examples/Feldspar_Thermobarometry/Two_Feldspar_input.xlsx>`




.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/uTYdh4Y1S0Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
---------


Integration with VESIcal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Demonstrates how to combine Thermobar and VESIcal to calculate temperatures at which to calculate saturation pressures.

**Useful files:

    * Download :download:`Python Notebook (Integration_with_VESIcal.ipynb) <Examples/Integration_with_VESIcal/Combining_VESIcal_Thermobar_SatPs.ipynb>`
    * Download :download:`Excel Workbook (Ol_hosted_melt_inclusions.xlsx) <Examples/Integration_with_VESIcal/Ol_hosted_melt_inclusions.xlsx>`

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/FRpsDbouuec" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
---------