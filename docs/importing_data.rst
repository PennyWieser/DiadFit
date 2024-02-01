================
Importing Data and Metadata
================


Supported Spectral File Types
==================================
DiadFit currently supports data input from the following file types. The name given between quotes is the string that should be entered as the filetype argument when loading data.
Screenshots here :doc:`Examples: <Examples/File_Formats/Extracting_HORIBA_MEtadata>`,


If your Raman exports another file type, please get in touch and I will try to add it.

Quick options:

1. "headless_txt" file:
    - txt file with no headers, wavenumber in the 1st column, intensity in the second column

2. ""WITec ASCII"" file:
    - Standard output from WITec Raman instruments. File starts with //Exported ASCII-File, then has several lines of metadata,
    before data is listed under [Data] heading

3. "Renishaw_txt":
    This is a txt file from a Renishaw instrument, first header is #Wave, second is #Intensity.

4. "HORIBA_txt":
    - This is an output from the newer HORIBAs. It has some metadata like #Acq. time, then data is listed under #Acquired

5. "headless_csv":
    This is a csv file with no headers, with wavenumber in 1 column, intensity in the second.

6. 'head_csv':
    This is a csv file with a header, wavenumber in the 1st column, intensity in the second. It doesnt matter what the header is. Python will strip it away.




Supported Metadata File Types
===============================

Many of the workflows in DiadFit use, or even rely on metadata. For example, to generate a Neon line correction model, you need the timestamp of each spectra.
There are a huge number of ways that Raman instruments generate their metadata.

The following options are currently supported:
1. WITEC instruments - WITEC instruments save a separate spectra and metadata file. Ensure your metadata and spectra file have the same name and file extension. E.g. FO1.txt for spectra, and FO1.txt for metadata. If they do not have the same name, DiadFit has no way of knowing what metadata file matches which spectra file, as there is no shared information between the two

2. Instruments which save the spectra file with the datastamp of the anlaysis - Some instruments set the datastamp of the file based on when the spectra was acquired. If this is the case, you can get the time using the function 'pf.loop_convert_datastamp_to_metadata'. See https://github.com/PennyWieser/DiadFit/blob/main/docs/Examples/Example1c_HORIBA_Calibration/Step4_Stitch_Outputs_Together_v74.ipynb for an example

Again, we are happy to add additional functionality.