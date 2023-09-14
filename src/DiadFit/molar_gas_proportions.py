import math
import pandas as pd

def calculate_sigma(wavelength, vi_dict, T_K):
    """
    This function calculates the σ cross section according to wavelength based on Burke (2001) EQ(1), you must provide:
### 1) the peak shift of the species, 2) temp (doesn't really matter) and 3) Σ (wavelength-independent relative Raman scattering cross-sections).
    """
    c = 2.998 * 10**10  # cm/s light speed
    h = 6.626 * 10**-27  # erg.s Planck constant
    k = 1.381 * 10**-16  # erg/K Boltzmann's constant

    v0 = 1 / (wavelength * 10**-7)

    sigma_results = {}
    for name, vi_info in vi_dict.items():
        vi = vi_info["Peak_shift_cm-1"]
        BigSigma = vi_info["Σ"]

        result = BigSigma / (((v0 - vi)**-4 / (v0 - 2331)**-4) * (1 - math.exp(-h * c * vi / (k * T_K))))
        sigma_results[name] = round(result, 2)

    return sigma_results




def calculate_mole_percent(components):
    """ This function calculates the mole percents of the components entered, based on Burke (2001) EQ(2)
    """
    def partial_molec_contribution_single(A, sigma, squiggle):
        return A / (sigma * squiggle)

    def partial_molec_contribution_double(A1, sigma1, A2, sigma2, squiggle):
        return (A1 + A2) / ((sigma1 + sigma2) * squiggle)

    total_partials = 0
    partials = []

    for component in components:
        if component['name'] == 'CO2':
            partial = partial_molec_contribution_double(component['peak_area_1'], component['cross_section_1'],
                                                       component['peak_area_2'], component['cross_section_2'],
                                                       component['efficiency'])
        else:
            partial = partial_molec_contribution_single(component['peak_area'], component['cross_section'],
                                                        component['efficiency'])

        partials.append(partial)
        total_partials += partial

    mole_percentages = [round((partial / total_partials) * 100, 1) for partial in partials]

    mole_percent_dict = {component['name']: mole_percent for component, mole_percent in zip(components, mole_percentages)}
    mole_percent_dict['Mole_Percent_Sum'] = sum(mole_percentages)

    return mole_percent_dict




def calculate_CO2_SO2_ratio(*, peak_area_SO2, peak_area_diad1, peak_area_diad2,wavelength=532.067, T_K=37+273.15,efficiency_SO2=1, efficiency_CO2=0.5, sigma_SO2=4.03, sigma_CO2_v1=0.8, sigma_CO2_v2=1.23):

  # First we need to calculate the oarameters

  component_dict = {
    "SO2": {"Peak_shift_cm-1": 1151, "Σ": sigma_SO2},
    "CO2_v1": {"Peak_shift_cm-1": 1285, "Σ": sigma_CO2_v1},
    "CO2_2v2": {"Peak_shift_cm-1": 1388, "Σ": sigma_CO2_v2}}
### "Σ" is the wavelength independent relative cross-section

  sigma_results = calculate_sigma(wavelength=wavelength, vi_dict=component_dict, T_K=T_K)

    # Now lets allocate these calculations
  components = [
        {'name': 'SO2',
        'peak_area': peak_area_SO2,
        'cross_section':  sigma_results['SO2'],
        'efficiency': efficiency_SO2},
        {'name': 'CO2',
          'peak_area_1':peak_area_diad2, 'cross_section_1': sigma_results['CO2_2v2'],
        'peak_area_2': peak_area_diad1, 'cross_section_2': sigma_results['CO2_v1'], 'efficiency': efficiency_CO2}
    ]
  mol_perc=calculate_mole_percent(components)


  return pd.DataFrame(mol_perc)
