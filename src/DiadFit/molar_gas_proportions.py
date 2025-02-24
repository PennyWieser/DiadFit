import math
import pandas as pd
import numpy as np






# def calculate_sigma(wavelength, vi_dict, T_K):
#     """
#     This function calculates the σ cross section according to wavelength based on Burke (2001) EQ(1), you must provide:
# ### 1) the peak shift of the species, 2) temp (doesn't really matter) and 3) Σ (wavelength-independent relative Raman scattering cross-sections).
#     """
#     c = 2.998 * 10**10  # cm/s light speed
#     h = 6.626 * 10**-27  # erg.s Planck constant
#     k = 1.381 * 10**-16  # erg/K Boltzmann's constant
#
#     v0 = 1 / (wavelength * 10**-7)
#
#     sigma_results = {}
#     for name, vi_info in vi_dict.items():
#         vi = vi_info["Peak_shift_cm-1"]
#         BigSigma = vi_info["Σ"]
#
#         result = BigSigma / (((v0 - vi)**-4 / (v0 - 2331)**-4) * (1 - math.exp(-h * c * vi / (k * T_K))))
#         sigma_results[name] = round(result, 2)
#
#     return sigma_results
#
#
#
#
# def calculate_mole_percent(components):
#     """ This function calculates the mole percents of the components entered, based on Burke (2001) EQ(2)
#     """
#     def partial_molec_contribution_single(A, sigma, squiggle):
#         return A / (sigma * squiggle)
#
#     def partial_molec_contribution_double(A1, sigma1, A2, sigma2, squiggle):
#         return (A1 + A2) / ((sigma1 + sigma2) * squiggle)
#
#     total_partials = 0
#     partials = []
#
#     for component in components:
#         if component['name'] == 'CO2':
#             partial = partial_molec_contribution_double(component['peak_area_1'], component['cross_section_1'],
#                                                        component['peak_area_2'], component['cross_section_2'],
#                                                        component['efficiency'])
#         else:
#             partial = partial_molec_contribution_single(component['peak_area'], component['cross_section'],
#                                                         component['efficiency'])
#
#         partials.append(partial)
#         total_partials += partial
#
#     mole_percentages = [round((partial / total_partials) * 100, 1) for partial in partials]
#
#     mole_percent_dict = {component['name']: mole_percent for component, mole_percent in zip(components, mole_percentages)}
#     mole_percent_dict['Mole_Percent_Sum'] = sum(mole_percentages)
#
#     return mole_percent_dict


#
#
# def calculate_CO2_SO2_ratio(*, peak_area_SO2, peak_area_diad1, peak_area_diad2,wavelength=532.067, T_K=37+273.15,efficiency_SO2=1, efficiency_CO2=0.5, sigma_SO2=4.03, sigma_CO2_v1=0.8, sigma_CO2_v2=1.23):
#
#   # First we need to calculate the oarameters
#
#   component_dict = {
#     "SO2": {"Peak_shift_cm-1": 1151, "Σ": sigma_SO2},
#     "CO2_v1": {"Peak_shift_cm-1": 1285, "Σ": sigma_CO2_v1},
#     "CO2_2v2": {"Peak_shift_cm-1": 1388, "Σ": sigma_CO2_v2}}
# ### "Σ" is the wavelength independent relative cross-section
#
#   sigma_results = calculate_sigma(wavelength=wavelength, vi_dict=component_dict, T_K=T_K)
#
#     # Now lets allocate these calculations
#   components = [
#         {'name': 'SO2',
#         'peak_area': peak_area_SO2,
#         'cross_section':  sigma_results['SO2'],
#         'efficiency': efficiency_SO2},
#         {'name': 'CO2',
#           'peak_area_1':peak_area_diad2, 'cross_section_1': sigma_results['CO2_2v2'],
#         'peak_area_2': peak_area_diad1, 'cross_section_2': sigma_results['CO2_v1'], 'efficiency': efficiency_CO2}
#     ]
#   mol_perc=calculate_mole_percent(components)
#
#
#   return pd.DataFrame(mol_perc)



  ## Math for converting back and forth between Raman scattering cross sections

def calculate_wavelength_dependent_cross_section(wavelength_nm, T_C, Raman_shift_cm, wavelength_independent_cross_section ):
    """ This function calculates the wavelength dependent cross section (lower case sigma) from the wavelength independent Raman scattering efficiency (Upper case sigma)

      Parameters
       ----------------
        wavelength_nm:
            laser wavelength used in nm to calculate the wavelength dependent cross section

        wavelength_independent_cross_section:
            Wavelength independent cross section

        T_K:
            absolute temperature in Kelvin

        Raman_shift_cm:
            Raman shift in cm-1 of peak of interest (e.g. 1151 for SO$_2$)

        Returns
        -----------------
        Wavelength dependent cross section

          """
    Wavelength_cm1=1/wavelength_nm*10000000
    constant =1-np.exp(((-6.626*10**-27)*(2.998*10**10)*Raman_shift_cm)/((1.381*10**-16)*(273.15+T_C)))
    Wavelength_dependent = wavelength_independent_cross_section/(((Wavelength_cm1-Raman_shift_cm)**(-4)/(Wavelength_cm1-2331)**(-4))*constant)
    return Wavelength_dependent

def calculate_wavelength_independent_cross_section(wavelength_nm, T_C, Raman_shift_cm, wavelength_dependent_cross_section ):
    """ This function calculates the wavelength independent cross section (capital Sigma) from the wavelength dependent Raman scattering efficiency (lower case sigma)

      Parameters
       ----------------
        wavelength_nm:
            laser wavelength used in nm to calculate the wavelength dependent cross section

        wavelength_dependent_cross_section:
            Wavelength dependent cross section

        T_K:
            absolute temperature in Kelvin

        Raman_shift_cm:
            Raman shift in cm-1 of peak of interest (e.g. 1151 for SO$_2$)

        Returns
        -----------------
        Wavelength independent cross section

          """
    Wavelength_cm1=1/wavelength_nm*10000000
    constant =1-np.exp(((-6.626*10**-27)*(2.998*10**10)*Raman_shift_cm)/((1.381*10**-16)*(273.15+T_C)))
    Wavelength_independent = wavelength_dependent_cross_section*((Wavelength_cm1-Raman_shift_cm)**(-4)/(Wavelength_cm1-2331)**(-4)*constant)
    return Wavelength_independent

def convert_cross_section_wavelength1_wavelength2(wavelength_nm_1,wavelength_nm_2,  Raman_shift_cm, wavelength_dependent_cross_section_wavelength1, T_C):
   """ This function calculates the wavelength dependent cross section (lower case sigma) for laser wavelength 2 from the wavelength-dependent cross section for laser wavelength 1.


      Parameters
       ----------------
        wavelength_nm_1:
            laser wavelength used in nm to calculate the wavelength dependent cross section

        wavelength_nm_2:
            laser wavelength of the system of interest you are trying to calculate the wavelength dependent cross section for

        wavelength_dependent_cross_section_wavelength1:
            Wavelength dependent cross section

        T_K:
            absolute temperature in Kelvin

        Raman_shift_cm:
            Raman shift in cm-1 of peak of interest (e.g. 1151 for SO$_2$)

        Returns
        -----------------
        Wavelength dependent cross section for wavelength2

          """

   # First calculate the independent cross section
   ind_cross_sec=calculate_wavelength_independent_cross_section(wavelength_nm=wavelength_nm_1, T_C=T_C, Raman_shift_cm=Raman_shift_cm,
                                                         wavelength_dependent_cross_section =wavelength_dependent_cross_section_wavelength1)
   print(ind_cross_sec)
   dep_cross_sec=calculate_wavelength_dependent_cross_section(wavelength_nm=wavelength_nm_2, T_C=T_C, Raman_shift_cm=Raman_shift_cm, wavelength_independent_cross_section =ind_cross_sec)

   return dep_cross_sec

def calculate_SO2_CO2_mol_prop_wave_indep(SO2_wavelength_ind, CO2_diad1_wavelength_ind, CO2_diad2_wavelength_ind, wavelength_nm, T_C,
                               A_SO2, A_CO2_Tot):
    """ Takes wavelength independnet cross sections and CO2 and SO2 peak areas and converts them into SO2 mol proportions
    Parameters
    ------------------

    SO2_wavelength_ind: Wavelength independent cross section for SO2

    CO2_diad1_wavelength_ind: Wavelength independent cross section for diad 1 (at 1285)

    CO2_diad2_wavelength_ind: Wavelength independent cross section for diad 2 (at 1388)

    wavelength_nm: Laser wavelenth of system in nm

    T_C: Temperature of analysis in C.

    Returns
    ---------------

    SO2 mol proportion


    """

    SO2_cross_sec=calculate_wavelength_dependent_cross_section(wavelength_nm=wavelength_nm, T_C=T_C, Raman_shift_cm=1151, wavelength_independent_cross_section=SO2_wavelength_ind)
    CO2_diad1_xsec=calculate_wavelength_dependent_cross_section(wavelength_nm=wavelength_nm, T_C=T_C, Raman_shift_cm=1285, wavelength_independent_cross_section=CO2_diad1_wavelength_ind)
    CO2_diad2_xsec=calculate_wavelength_dependent_cross_section(wavelength_nm=wavelength_nm, T_C=T_C, Raman_shift_cm=1388, wavelength_independent_cross_section=CO2_diad2_wavelength_ind)


    SO2_prop=(A_SO2/SO2_cross_sec)/(A_CO2_Tot /(CO2_diad1_xsec + CO2_diad2_xsec) +  (A_SO2/SO2_cross_sec) )

    return SO2_prop


def calculate_SO2_CO2_ratio(SO2_area, diad1_area, diad2_area, SO2_cross_sec=5.3, diad1_cross_sec=0.89, diad2_cross_sec=1.4):
    """ Calculates SO2:CO2 ratio using the parameters from Marie-Camille Caumons lab"""


    A_CO2_star=( diad1_area + diad2_area)/(diad2_cross_sec+diad1_cross_sec)
    A_SO2_star=(SO2_area)/(SO2_cross_sec)
    Ratio=A_SO2_star/(A_SO2_star+A_CO2_star)

    return Ratio

