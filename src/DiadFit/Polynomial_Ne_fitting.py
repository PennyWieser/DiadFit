import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import lmfit
from lmfit.models import GaussianModel, VoigtModel, LinearModel, ConstantModel, PseudoVoigtModel, SkewedVoigtModel
from scipy.signal import find_peaks
import os
import re
from os import listdir
from os.path import isfile, join
from DiadFit.importing_data_files import *
from typing import Tuple, Optional
from dataclasses import dataclass
import matplotlib.patches as patches
from tqdm.notebook import tqdm
import re
import scipy.stats as stats
import pickle
from DiadFit.ne_lines import *


allowed_models = ["VoigtModel", "PseudoVoigtModel", "Pearson4Model", "SkewedVoigtModel", "GaussianModel"]



encode="ISO-8859-1"


## Code for polynomial fit to all the neon lines

# Calculate theoretical Ne line positions


def calculate_all_Ne_lines(wavelength=532.046, lines_in_air=[565.66588, 568.98163, 571.92248, 574.82985, 576.44188, 580.44496]):
    """ This code finds the Ne line positions for all the lines"""
    lines_in_air = [
        565.66588,
        568.98163,
        571.92248,
        574.82985,
        576.44188,
        580.44496
    ]

    df_Ne = calculate_Ne_line_positions(
        wavelength=wavelength, cut_off_intensity=1000
    )

    rows = []
    tol = 0.5

    for line in lines_in_air:


        df_filtered = df_Ne[
            (df_Ne['Ne emission line in air'] >= line - tol) &
            (df_Ne['Ne emission line in air'] <= line + tol)
        ]


        max_intensity_row = df_filtered.loc[
            df_filtered['Intensity'].idxmax()
        ]


        rows.append({
            'Entered Ne line in air': line,
            'Ne emission line in air': max_intensity_row['Ne emission line in air'],
            'Raman_shift (cm-1)': max_intensity_row['Raman_shift (cm-1)'],
            'Intensity': max_intensity_row['Intensity']
        })

    df = pd.DataFrame(rows)

    return df