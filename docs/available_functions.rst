========================
Available Functions
========================
For a more detailed description of what DiadFit can do, please see our preprint:
https://eartharxiv.org/repository/view/5236/

Extracting Meta Data
======================

This is currently supported for WITEC ASCII files. You tell DiadFit which folder contains your metadata files, and extracts:
    1) The filename
    2) The date of acquisition
    3) The laser power
    4) The integration time
    5) The number of accumulations
    6) The objective magnification
    7) The duration of the analysis
    8) the time (24 hrs)
    9) Seconds since midnight

We are sure other Raman manufacturers have similarly weird metadata formats. We are happy to adapt DiadFit to extract this too
if you send us an example. If you dont have a designed Metadata format, you can use this function:

meta=pf.loop_convert_datastamp_to_metadata(path=spectra_path, 
files=Ne_files, creation=False,
modification=True)

It extracts the filename and the creation or modification time stamp (whichever on your instrument is done when the file is made)

Fitting Neon Lines
====================

DiadFit allows users to fit any 2 selected Ne lines. 
There is also an automated option that fits all Ne lines within a single folder, producing a png for the operator to inspect with each fit.


Fitting Fermi Diads
=====================
DiadFit was designed for the fitting of Fermi Diads. There are lots of options - e.g., fitting with and without hot bands, with and without a gaussian background, with and without a C13 peak.


Calculations involving the CO$_2$ EOS
=====================
DiadFit can be used to perform calculations using the CO$_2$ equation of state (EOS) from Sterner and Pitzer (1994) 
or Span and Wanger (1996):
This includes:
- Calculating  Pressure if CO$_2$ density and Temperature is known
- Calculating entrapment temperature if Pressure and CO$_2$ density is known
- Calculating density if Pressure and Temperature is known
- Converting a homogenization temperature from microthermometry into a CO$_2$ density


Fitting H$_2$O peaks 
=====================
DiadFit can quantify the ratio of H$_2$O to silicate peaks, both on exposed glasses and unexposed melt Fluid_Inclusion_Density_to_Depth

Propagating uncertainty
=====================
DiadFit has a number of options for propgating uncertainty:
- Full propagation of peak fitting uncertainties, and Ne line drift correction uncertainties
- Propagation of uncertainty in EOS calculations - e.g. uncertainty in CO$_2$ density and temperature when calculating entrapment Pressure
- Uncertainty when calculating the equivalent CO$_2$ contents of melt inclusion vapour bubbles (from the volume proportion, CO$_2$ density, and melt density )

