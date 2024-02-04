==========================================
Python datastructures used in DiadFit
==========================================
If you are new to python, you might find it helpful to skim through these explanations of the datatypes used in DiadFit.

What is a function?
=======================
You call different calculations using functions. For example, the function 'calculate_P_for_rho_T' calculates a pressure for a specified CO2 density and temperature. This function takes arguements (aka inputs). These inputs may be option, or required.
For example, this function requires the inputs:
-T_K: Temperature in Kelvin
-EOS: The equation of state you wish to use
-CO2_dens_gcm3: The CO2 density in g/cm3

You may wonder, how are you supposed to know what inputs to use, and what things you can enter? This is when you want to use the help functions. If after loading DiadFit as pf, you type:
help(pf.calculate_P_for_rho_T) it will return the documentation for this function which tells you all about it


What are different datatypes?
=======================
Once you look at the help functions, you will realize it says things like

CO2_dens_gcm3: int, float, pd.Series, np.array
        CO2 density in g/cm3

This is telling you that you can enter CO2_dens_gcm3 as 4 different data types. But what do these words mean?

int
------
int means integer. E.g., it could be 4

float
---------
float is a single number, but not an integer. E.g. 4.1 would be a float.

string
---------
A string is a piece of text. These are used to tell a function something about your data input, or specify a certain thing you want the function to do. For example, users must specify their filetype when using functions that load in spectra (e.g., filetype='headless_txt').

pd.DataFrame
--------------
Pandas (pd) is a package that treats data a bit like a spreadsheet. A dataframe can be thought of like a single sheet in excel, e.g. data with labelled columns. In DiadFit, users load data as dataframes (e.g., homogenization temperatures from microthermometry) and functions return dataframes with parameters (e.g. the position of diad 1, its area, its intensity, etc. )

pd.Series
-----------
Panda Series is a column of data with a heading. It is often a single column of a dataframe. For example, if you import an excel spreadsheet as a dataframe called df, and pressure is stored in the column 'Pressure (kbar)', you could get a pandas series of this data using df['Pressure (kbar)'] (i.e. the name of the dataframe and then the name of the column in square brackets within quote symbols).

dataclasses
----------------
DiadFit uses dataclasses to provide default configurations to  peak finding and fitting functions. These default configurations can be tweaked as much or as little as required for each specific Raman spectrometer. For example, the default parameters for fitting diad 1 are stored in the dataclass diad1_fit_config.

diad1_fit_config(model_name='PseudoVoigtModel',
fit_peaks=2, fit_gauss=False,
gauss_amp=1000, diad_sigma=0.2,
diad_sigma_min_allowance=0.2,
diad_sigma_max_allowance=5,
N_poly_bck_diad1=1,
lower_bck_diad1=(1180, 1220),
upper_bck_diad1=(1300, 1350),
diad_prom=100, HB_prom=20,
x_range_baseline=75, y_range_baseline=100,
plot_figure=True, dpi=200,
x_range_residual=20)
