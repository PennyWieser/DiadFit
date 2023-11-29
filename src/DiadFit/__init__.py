__author__ = 'Penny Wieser'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This is for filtering cosmic rays
from DiadFit.cosmicray_filter import *

# This is for processing calibration pressures with the FDCA
from DiadFit.Psensor import *

# This reads different data formats, imports files, and does the string stripping for WITEC
from DiadFit.importing_data_files import *

# This does the work associated with fitting Ne lines
from DiadFit.ne_lines import *
from DiadFit.argon_lines import *

# This does the work associated with fitting Diads
from DiadFit.diads import *


# This has densimeters from different instruments
from DiadFit.densimeters import *

# This has functions to fit densimeters
from DiadFit.densimeter_fitting import *

# H2O fitting
from DiadFit.H2O_fitting import *

# Monte Carlo Error propagation
from DiadFit.error_propagation import *

from DiadFit.density_depth_crustal_profiles import *
from DiadFit.CO2_EOS import *
from DiadFit.CO2_H2O_EOS import *

from DiadFit.CO2_in_bubble_error import *

from DiadFit.molar_gas_proportions import *

from DiadFit.relaxifi import *

# version
from ._version import __version__



