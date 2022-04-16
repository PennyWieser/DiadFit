This is a python package in progress allowing users to fit Ne lines, Fermi diads and carbonate peaks.

Requirements: lmfit

FileFormats
_________________________________________
For screenshots of the accepted file formats, please see FileFormats.ipynb. If your file isn't one of these, please send penny an example. 


Diad Spectra included as examples:
_________________________________________

1. Strong Diad with carbonate (Thanks Charlotte)
'headless_txt_FG18_6_MI1.txt': Headless txt file from Witec showing a strong diad, hot bands and C13 

'DiadFit_headlesstxt_Dense_Gaussian.ipynb' fits this using a Gaussian background + 4 voigt peaks (splitting = 103.79765)
'DiadFit_headlesstxt_Dense_Local.ipynb' fits this using 4 voigt peaks (splitting =103.798393)


2. Weak Diad no carbonate (Mine!)
'WITEC_ASCII_MS14_11_MI1_50X.txt': WITEC ASCII file (e.g. metadata + data), Diad and very weak hotbands
'Diad_Fitting_WITEC_Weak.ipynb': Shows how to fit this.  


3. Very strong Diad signal in CSV format, v large spectral range (Thanks Chelsea)
Diad_Fitting_CSV.ipynb

4. Diad in Qtz from HORIBA (Thanks Sarah!)
Diad_Fitting_HORIBA_Qtz.ipynb
Currently uses a local background, if results v different from Fityk, could add a gaussian routine in. 




Ne Lines included as example
__________________________________________________
* so far, the code is configured to fit just 2 user selected lines. For 1117, you can choose 1 peak or 2 (e.g. fits shoulder voigt).  

1. Fitting all lines at once: Spectra in folder 'Ne_Test_Loop', notebook='Ne_Line_Fitting_Loop_txt.ipynb'


2. Fitting each line individually (e.g. for weaker signals) -  
Ne_Line_Fitting_WITEC_Strong_SingleSpectra.ipynb - Strong signal where you need 2 voigt peaks for the 1117 line
Ne_Line_Fitting_WITEC_Weak_SingleSpectra.ipynb - Weaker signal where its less clear if you need 2 voigt peaks for the 1117 line



Stripping Metadata
__________________________________________________________

So far, you can link to your WITEC folder with metadata, and it will get laser power, time since midnight, acquision time etc. 
See String_Stripping_encoding_withdates_WITEC.ipynb


Planned updates
_____________________________________________________________
- Metadata stripping for Horiba
- Water fitting (cry)
- Possibility to loop for very simple spectra diads. 
- Gaussian fit to quartz background if needed vs. local background. 