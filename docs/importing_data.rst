================
Importing Data
================

Please see the python notebook https://github.com/PennyWieser/DiadFit/blob/master/FileFormats.ipynb for a summary of
currently supported file formats. If you raman exports something else, please get in contact and I can probably add this in.

Quick options:

1. "headless_txt" file:
    - txt file with no headers, wavenumber in the 1st column, intensity in the second column

2. ""WITec ASCII"" file:
    - Standard output from WITec Raman instruments. File starts with //Exported ASCII-File, then has several lines of metadata,
    before data is listed under [Data] heading

3. "HORIBA_txt":
    - This is the standard output on newer HORIBAs. It has some metadata like #Acq. time, then data is listed under #Acquired

4. "headless_csv":
    This is a csv file with no headers, with wavenumber in 1 column, intensity in the second.

5. "Renishaw_txt":
    This is a txt file from a Renishaw instrument, first header is #Wave, second is #Intensity.



