import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import lmfit
from lmfit.models import GaussianModel, VoigtModel, LinearModel, ConstantModel
from scipy.signal import find_peaks
import os
import re
from os import listdir
from os.path import isfile, join

encode="ISO-8859-1"


def calculate_density_cornell(temp='SupCrit', Split=None):
    """ Calculates density for Cornell densimeters"""

    #if temp is "RoomT":
    LowD_RT=-38.34631 + 0.3732578*Split
    HighD_RT=-41.64784 + 0.4058777*Split- 0.1460339*(Split-104.653)**2

    # IF temp is 37
    LowD_SC=-38.62718 + 0.3760427*Split
    MedD_SC=-47.2609 + 0.4596005*Split+ 0.0374189*(Split-103.733)**2-0.0187173*(Split-103.733)**3
    HighD_SC=-42.52782 + 0.4144277*Split- 0.1514429*(Split-104.566)**2

    df=pd.DataFrame(data={'Preferred D': 0,
     'in range': 'Y',
                            'Notes': 'not in range',
                            'LowD_RT': LowD_RT,
                            'HighD_RT': HighD_RT,
                            'LowD_SC': LowD_SC,
                            'MedD_SC': MedD_SC,
                            'HighD_SC': HighD_SC,
                            'Temperature': temp,
                            'Splitting': Split,

                            })

    roomT=df['Temperature']=="RoomT"
    SupCrit=df['Temperature']=="SupCrit"
    # If splitting is 0
    zero=df['Splitting']==0

    # Range for SC low density
    min_lowD_SC_Split=df['Splitting']>=102.72
    max_lowD_SC_Split=df['Splitting']<=103.16
    # Range for SC med density
    min_MD_SC_Split=df['Splitting']>103.16
    max_MD_SC_Split=df['Splitting']<=104.28
    # Range for SC high density
    min_HD_SC_Split=df['Splitting']>=104.28
    max_HD_SC_Split=df['Splitting']<=104.95
    # Range for Room T low density
    min_lowD_RoomT_Split=df['Splitting']>=102.734115670188
    max_lowD_RoomT_Split=df['Splitting']<=103.350311768435
    # Range for Room T high density
    min_HD_RoomT_Split=df['Splitting']>=104.407308904012
    max_HD_RoomT_Split=df['Splitting']<=105.1
    # Impossible densities, room T
    Imposs_lower_end=(df['Splitting']>103.350311768435) & (df['Splitting']<103.88)
    # Impossible densities, room T
    Imposs_upper_end=(df['Splitting']<104.407308904012) & (df['Splitting']>103.88)
    # Too low density
    Too_Low_SC=df['Splitting']<102.72
    Too_Low_RT=df['Splitting']<102.734115670188

    df.loc[zero, 'Preferred_D']=0
    df.loc[zero, 'Notes']=0


    # If room T, low density, set as low density
    df.loc[roomT&(min_lowD_RoomT_Split&max_lowD_RoomT_Split), 'Preferred D'] = LowD_RT
    df.loc[roomT&(min_lowD_RoomT_Split&max_lowD_RoomT_Split), 'Notes']='Room T, low density'
    # If room T, high density
    df.loc[roomT&(min_HD_RoomT_Split&max_HD_RoomT_Split), 'Preferred D'] = HighD_RT
    df.loc[roomT&(min_HD_RoomT_Split&max_HD_RoomT_Split), 'Notes']='Room T, high density'

    # If SupCrit, high density
    df.loc[ SupCrit&(min_HD_SC_Split&max_HD_SC_Split), 'Preferred D'] = HighD_SC
    df.loc[ SupCrit&(min_HD_SC_Split&max_HD_SC_Split), 'Notes']='SupCrit, high density'
    # If SupCrit, Med density
    df.loc[SupCrit&(min_MD_SC_Split&max_MD_SC_Split), 'Preferred D'] = MedD_SC
    df.loc[SupCrit&(min_MD_SC_Split&max_MD_SC_Split), 'Notes']='SupCrit, Med density'

    # If SupCrit, low density
    df.loc[ SupCrit&(min_lowD_SC_Split&max_lowD_SC_Split), 'Preferred D'] = LowD_SC
    df.loc[SupCrit&(min_lowD_SC_Split&max_lowD_SC_Split), 'Notes']='SupCrit, low density'

    # If Supcritical, and too low
    df.loc[SupCrit&(Too_Low_SC), 'Preferred D']=LowD_SC
    df.loc[SupCrit&(Too_Low_SC), 'Notes']='Below lower calibration limit'
    df.loc[SupCrit&(Too_Low_SC), 'in range']='N'


    # If RoomT, and too low
    df.loc[roomT&(Too_Low_RT), 'Preferred D']=LowD_RT
    df.loc[roomT&(Too_Low_RT), 'Notes']='Below lower calibration limit'
    df.loc[roomT&(Too_Low_RT), 'in range']='N'

    #if splitting is zero
    SplitZero=df['Splitting']==0
    df.loc[SupCrit&(SplitZero), 'Preferred D']=np.nan
    df.loc[SupCrit&(SplitZero), 'Notes']='Splitting=0'
    df.loc[SupCrit&(SplitZero), 'in range']='N'

    df.loc[roomT&(SplitZero), 'Preferred D']=np.nan
    df.loc[roomT&(SplitZero), 'Notes']='Splitting=0'
    df.loc[roomT&(SplitZero), 'in range']='N'


    # If impossible density, lower end
    df.loc[roomT&Imposs_lower_end, 'Preferred D'] = LowD_RT
    df.loc[roomT&Imposs_lower_end, 'Notes']='Impossible Density, low density'
    df.loc[roomT&Imposs_lower_end, 'in range']='N'

    # If impossible density, lower end
    df.loc[roomT&Imposs_upper_end, 'Preferred D'] = HighD_RT
    df.loc[roomT&Imposs_upper_end, 'Notes']='Impossible Density, high density'
    df.loc[roomT&Imposs_upper_end, 'in range']='N'


    #df.loc[zero, 'in range']='Y'
    # If high densiy, and beyond the upper calibration limit
    Upper_Cal_RT=df['Splitting']>105.1
    Upper_Cal_SC=df['Splitting']>104.95

    df.loc[roomT&Upper_Cal_RT, 'Preferred D'] = HighD_RT
    df.loc[roomT&Upper_Cal_RT, 'Notes']='Above upper Cali Limit'
    df.loc[roomT&Upper_Cal_RT, 'in range']='N'

    df.loc[SupCrit&Upper_Cal_SC, 'Preferred D'] = HighD_SC
    df.loc[SupCrit&Upper_Cal_SC, 'Notes']='Above upper Cali Limit'
    df.loc[SupCrit&Upper_Cal_SC, 'in range']='N'

    return df