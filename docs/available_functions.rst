========================
Available Functions
========================

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


Fitting Neon Lines
====================

DiadFit allows users to fit any 2 selected Ne lines (although the code is optimised for the 1117 and 1447 lines, and may need tweaking for the others).
There is also an automated option that fits all Ne lines within a single folder, producing a png for the operator to inspect with each fit.

Fitting Fermi Diads
=====================
DiadFit is designed for the fitting of Fermi Diads. There are lots of options - e.g., fitting with and without hot bands, with and without a gaussian background, with and without a C13 peak.




