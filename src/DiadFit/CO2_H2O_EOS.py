import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inspect
from scipy.interpolate import CubicSpline
import scipy
from scipy.optimize import minimize

from pathlib import Path
from pickle import load
import pickle
import math
from DiadFit.CO2_EOS import *


DiadFit_dir=Path(__file__).parent

# Set up constants.
Tc1 = 647.25
Pc1 = 221.19
Tc2 = 301.1282
Pc2 = 73.773

# Set up low pressure and high pressure parameters for CO2.

aL1 = [0] * 16  # Assuming the array is 1-indexed like in C.
#So we dont need to adjust everything

aL1[1] = 4.38269941 / 10**2
aL1[2] = -1.68244362 / 10**1
aL1[3] = -2.36923373 / 10**1
aL1[4] = 1.13027462 / 10**2
aL1[5] = -7.67764181 / 10**2
aL1[6] = 9.71820593 / 10**2
aL1[7] = 6.62674916 / 10**5
aL1[8] = 1.06637349 / 10**3
aL1[9] = -1.23265258 / 10**3
aL1[10] = -8.93953948 / 10**6
aL1[11] = -3.88124606 / 10**5
aL1[12] = 5.61510206 / 10**5
aL1[13] = 7.51274488 / 10**3  # alpha for H2O
aL1[14] = 2.51598931  # beta for H2O
aL1[15] = 3.94 / 10**2  # gamma for H2O

# Higher pressure parameters - 0.2- 1 GPa
aH1 = [0] * 16  # Assuming the array is 1-indexed like in C

aH1[1] = 4.68071541 / 10**2
aH1[2] = -2.81275941 / 10**1
aH1[3] = -2.43926365 / 10**1
aH1[4] = 1.10016958 / 10**2
aH1[5] = -3.86603525 / 10**2
aH1[6] = 9.30095461 / 10**2
aH1[7] = -1.15747171 / 10**5
aH1[8] = 4.19873848 / 10**4
aH1[9] = -5.82739501 / 10**4
aH1[10] = 1.00936000 / 10**6
aH1[11] = -1.01713593 / 10**5
aH1[12] = 1.63934213 / 10**5
aH1[13] = -4.49505919 / 10**2  # alpha for H2O
aH1[14] = -3.15028174 / 10**1  # beta for H2O
aH1[15] = 1.25 / 10**2  # gamma for H2O


# Low presure CO2 parameters.

aL2 = [0] * 16  # Assuming the array is 1-indexed like in C

aL2[1] = 1.14400435 / 10**1
aL2[2] = -9.38526684 / 10**1
aL2[3] = 7.21857006 / 10**1
aL2[4] = 8.81072902 / 10**3
aL2[5] = 6.36473911 / 10**2
aL2[6] = -7.70822213 / 10**2
aL2[7] = 9.01506064 / 10**4
aL2[8] = -6.81834166 / 10**3
aL2[9] = 7.32364258 / 10**3
aL2[10] = -1.10288237 / 10**4
aL2[11] = 1.26524193 / 10**3
aL2[12] = -1.49730823 / 10**3
aL2[13] = 7.81940730 / 10**3  # alpha for CO2
aL2[14] = -4.22918013  # beta for CO2
aL2[15] = 1.585 / 10**1  # gamma for CO2



# High pressure CO2 parameters.
aH2 = [0] * 16  # Assuming the array is 1-indexed like in C
aH2[1] = 5.72573440 / 10**3
aH2[2] = 7.94836769
aH2[3] = -3.84236281 * 10.0
aH2[4] = 3.71600369 / 10**2
aH2[5] = -1.92888994
aH2[6] = 6.64254770
aH2[7] = -7.02203950 / 10**6
aH2[8] = 1.77093234 / 10**2
aH2[9] = -4.81892026 / 10**2
aH2[10] = 3.88344869 / 10**6
aH2[11] = -5.54833167 / 10**4
aH2[12] = 1.70489748 / 10**3
aH2[13] = -4.13039220 / 10**1  # alpha for CO2
aH2[14] = -8.47988634  # beta for CO2
aH2[15] = 2.800 / 10**2  # gamma for CO2

## This is for when you only feed a numpy array
# def ensure_series(a, b, c):
#     # Determine the target length
#     lengths = [len(a) if isinstance(a, pd.Series) else None,
#                len(b) if isinstance(b, pd.Series) else None,
#                len(c) if isinstance(c, pd.Series) else None]
#     lengths = [l for l in lengths if l is not None]
#     target_length = max(lengths) if lengths else 1
#
#     # Convert each input to a Series of the target length
#     if not isinstance(a, pd.Series):
#         a = pd.Series([a] * target_length)
#     if not isinstance(b, pd.Series):
#         b = pd.Series([b] * target_length)
#     if not isinstance(c, pd.Series):
#         c = pd.Series([c] * target_length)
#
#     return a, b, c
#
#
# def ensure_series_4(a, b, c, d):
#     # Determine the target length
#     lengths = [len(a) if isinstance(a, pd.Series) else None,
#                len(b) if isinstance(b, pd.Series) else None,
#                len(c) if isinstance(c, pd.Series) else None,
#                len(d) if isinstance(d, pd.Series) else None]
#     lengths = [l for l in lengths if l is not None]
#     target_length = max(lengths) if lengths else 1
#
#     # Convert each input to a Series of the target length
#     if not isinstance(a, pd.Series):
#         a = pd.Series([a] * target_length)
#     if not isinstance(b, pd.Series):
#         b = pd.Series([b] * target_length)
#     if not isinstance(c, pd.Series):
#         c = pd.Series([c] * target_length)
#     if not isinstance(d, pd.Series):
#         d = pd.Series([d] * target_length)
#     return a, b, c, d

import pandas as pd
import numpy as np

def ensure_series(a, b, c):
    # Determine the target length
    lengths = [len(a) if isinstance(a, (pd.Series, np.ndarray)) else None,
               len(b) if isinstance(b, (pd.Series, np.ndarray)) else None,
               len(c) if isinstance(c, (pd.Series, np.ndarray)) else None]
    lengths = [l for l in lengths if l is not None]
    target_length = max(lengths) if lengths else 1

    # Convert each input to a Series of the target length
    if not isinstance(a, (pd.Series, np.ndarray)):
        a = pd.Series([a] * target_length)
    else:
        a = pd.Series(a)

    if not isinstance(b, (pd.Series, np.ndarray)):
        b = pd.Series([b] * target_length)
    else:
        b = pd.Series(b)

    if not isinstance(c, (pd.Series, np.ndarray)):
        c = pd.Series([c] * target_length)
    else:
        c = pd.Series(c)

    return a, b, c


def ensure_series_4(a, b, c, d):
    # Determine the target length
    lengths = [len(a) if isinstance(a, (pd.Series, np.ndarray)) else None,
               len(b) if isinstance(b, (pd.Series, np.ndarray)) else None,
               len(c) if isinstance(c, (pd.Series, np.ndarray)) else None,
               len(d) if isinstance(d, (pd.Series, np.ndarray)) else None]
    lengths = [l for l in lengths if l is not None]
    target_length = max(lengths) if lengths else 1

    # Convert each input to a Series of the target length
    if not isinstance(a, (pd.Series, np.ndarray)):
        a = pd.Series([a] * target_length)
    else:
        a = pd.Series(a)

    if not isinstance(b, (pd.Series, np.ndarray)):
        b = pd.Series([b] * target_length)
    else:
        b = pd.Series(b)

    if not isinstance(c, (pd.Series, np.ndarray)):
        c = pd.Series([c] * target_length)
    else:
        c = pd.Series(c)

    if not isinstance(d, (pd.Series, np.ndarray)):
        d = pd.Series([d] * target_length)
    else:
        d = pd.Series(d)

    return a, b, c, d



## Pure EOS functions
# First, we need the pure EOS
def pureEOS(i, V, P, B, C, D, E, F, Vc, TK, b, g):
    """
    This function calculates the compressability factor for a pure EOS using the modified Lee-Kesler
    equation.

    i=0 for H2O, i=1 for CO2.

    You input a volume, and it returns the difference between the compresability factor, and that calculated at the input P, V and T_K.
    E.g. gives the residual so that you can iterate.
    """
    CF = (1.0 + (B[i] * Vc[i] / V) + (C[i] * Vc[i] *
    Vc[i] / (V * V)) + (D[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] /
     (V * V * V * V)))

    CF += ((E[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] /
    (V * V * V * V * V)))

    CF += ((F[i] * Vc[i] * Vc[i] / (V * V)) *
    (b[i] + g[i] * Vc[i] * Vc[i] / (V * V)) *
    math.exp(-g[i] * Vc[i] * Vc[i] / (V * V)))

    return CF - (P * V) / (83.14467 * TK)

def pureEOS_CF(i, V, P, B, C, D, E, F, Vc, TK, b, g):
    """
    This function calculates the compressability factor for a pure EOS using the modified Lee-Kesler
    equation.

    i=0 for H2O, i=1 for CO2.

    You input a volume, and it returns the difference between the compresability factor, and that calculated at the input P, V and T_K.
    E.g. gives the residual so that you can iterate.
    """
    CF = (1.0 + (B[i] * Vc[i] / V) + (C[i] * Vc[i] *
    Vc[i] / (V * V)) + (D[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] /
     (V * V * V * V)))

    CF += ((E[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] /
    (V * V * V * V * V)))

    CF += ((F[i] * Vc[i] * Vc[i] / (V * V)) *
    (b[i] + g[i] * Vc[i] * Vc[i] / (V * V)) *
    math.exp(-g[i] * Vc[i] * Vc[i] / (V * V)))

    return CF


# Volume iterative function using Netwon-Raphson method.
def purevolume(i, V, P, B, C, D, E, F, Vc, TK, b, g):
    """ Using the pure EOS, this function solves for the best volume using the pureEOS residual calculated above

    It returns the volume.

    """
    for iter in range(1, 51):
        # Calculate the derivative of the pureEOS function at (V, P)
        diff = (pureEOS(i, V + 0.0001, P,B, C, D, E, F, Vc, TK, b, g) - pureEOS(i, V, P, B, C, D, E, F, Vc, TK, b, g)) / 0.0001

        # Update the volume using the Newton-Raphson method
        Vnew = V - pureEOS(i, V, P, B, C, D, E, F, Vc, TK, b, g) / diff

        # Check if the update is within the tolerance (0.000001)
        if abs(Vnew - V) <= 0.000001:
            break

        # Update V for the next iteration
        V = Vnew

    # Return the final estimated volume
    return V

def purepressure(i, V, P, TK):
    """ Using the pure EOS, this function solves for the best pressure using the pureEOS residual calculated above

    It returns the pressure.

    """
    for iter in range(1, 51):
        # Calculate the derivative of the pureEOS function at (V, P)
        k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess=get_EOS_params(P, TK)

        diff = (pureEOS(i, V, P + 0.0001, B, C, D, E, F, Vc, TK, b, g) - pureEOS(i, V, P, B, C, D, E, F, Vc, TK, b, g)) / 0.0001

        # Update the pressure using the Newton-Raphson method
        Pnew = P - pureEOS(i, V, P, B, C, D, E, F, Vc, TK, b, g) / diff

        # Check if the update is within the tolerance (0.000001)
        if abs(Pnew - P) <= 0.000001:
            break

        # Update P for the next iteration
        P = Pnew

    # Return the final estimated pressure
    return P




def mol_vol_to_density(mol_vol, XH2O):
    """ Converts molar mass to molar density for a given XH2O"""
    density=((1-XH2O)*44 + (XH2O)*18)/mol_vol
    return density

def pure_lnphi(i, Z, B, Vc, V, C, D, E, F, g, b):
    """
    This function calculates the fugacity coefficient from the equation of state for a pure fluid

    """
    lnph = Z[i] - 1.0 - math.log(Z[i]) + (B[i] * Vc[i] / V[i]) + (C[i] * Vc[i] * Vc[i] / (2.0 * V[i] * V[i]))
    lnph += (D[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] / (4.0 * V[i] * V[i] * V[i] * V[i]))
    lnph += (E[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] * Vc[i] / (5.0 * V[i] * V[i] * V[i] * V[i] * V[i]))
    lnph += (F[i] / (2.0 * g[i])) * (b[i] + 1.0 - (b[i] + 1.0 + g[i] * Vc[i] * Vc[i] / (V[i] * V[i])) * math.exp(-g[i] * Vc[i] * Vc[i] / (V[i] * V[i])))

    return lnph



## Mixing between species
def cbrt_calc(x):
    """
    This function calculates the cubic root that can deal with negative numbers.
    """
    if x >= 0:
        return math.pow(x, 1/3)
    else:
        return -math.pow(-x, 1/3)


def mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK):
    """ This function is like the one for the pureEOS. It calculates the compressability factor, and
    then calculates the compressability factor based on the P-V-T you entered. It returns the residual between those two values.
    """
    CF = 1.0 + (BVc / V) + (CVc2 / (V * V)) + (DVc4 / (V * V * V * V)) + (EVc5 / (V * V * V * V * V))
    CF += (FVc2 / (V * V)) * (bmix + gVc2 / (V * V)) * np.exp(-gVc2 / (V * V))

    return CF - (P * V) / (83.14467 * TK)


def mixEOS_CF(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK):
    """ This function is like the one for the pureEOS. It calculates the compressability factor, and
    then calculates that based on the P-V-T you entered. It does not return the residual
    """
    CF = 1.0 + (BVc / V) + (CVc2 / (V * V)) + (DVc4 / (V * V * V * V)) + (EVc5 / (V * V * V * V * V))
    CF += (FVc2 / (V * V)) * (bmix + gVc2 / (V * V)) * np.exp(-gVc2 / (V * V))

    return CF


def mixvolume(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK):
    """ This function iterates in volume space to get the best match to the entered pressure using the mixEOS function above.

    """
    for iter in range(1, 51):
        diff = ((mixEOS(V + 0.0001, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK)
    - mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK)) / 0.0001)
        Vnew = V - mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK) / diff
        if abs(Vnew - V) <= 0.000001:
            break
        V = Vnew

    return V

def mixpressure(P, V, TK, Y):
    """ This function iterates in pressure space to get the best match to the entered volume using the mixEOS function above.

    """
    for iter in range(1, 51):
        k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess=get_EOS_params(P, TK)
        Bij, Vcij, BVc_prm, BVc, Cijk, Vcijk, CVc2_prm, CVc2, Dijklm, Vcijklm, DVc4_prm, DVc4, Eijklmn, Vcijklmn, EVc5_prm,  EVc5, Fij, FVc2_prm, FVc2, bmix, b_prm, gijk, gVc2_prm, gVc2=mixing_rules(B, C,D, E, F, Vc, Y, b,    g, k1_temperature, k2_temperature, k3_temperature)

        diff = ((mixEOS(V, P + 0.0001, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK)
        - mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK)) / 0.0001)
        Pnew = P - mixEOS(V, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, TK) / diff
        if abs(Pnew - P) <= 0.000001:
            break
        P = Pnew

    return P



def mix_lnphi(i, Zmix, BVc_prm, CVc2_prm, DVc4_prm, EVc5_prm, FVc2_prm, FVc2, bmix, b_prm, gVc2, gVc2_prm, Vmix):
    lnph=0

    lnph = -math.log(Zmix)
    lnph += (BVc_prm[i] / Vmix)
    lnph += (CVc2_prm[i] / (2.0 * Vmix ** 2))
    lnph += (DVc4_prm[i] / (4.0 * Vmix ** 4))
    lnph += (EVc5_prm[i] / (5.0 * Vmix ** 5))
    lnph += ((FVc2_prm[i] * bmix + b_prm[i] * FVc2) / (2 * gVc2)) * (1.0 - math.exp(-gVc2 / (Vmix ** 2)))
    lnph += ((FVc2_prm[i] * gVc2 + gVc2_prm[i] * FVc2 - FVc2 * bmix * (gVc2_prm[i] - gVc2)) / (2.0 * gVc2 ** 2)) * (1.0 - (gVc2 / (Vmix ** 2) + 1.0) * math.exp(-gVc2 / (Vmix ** 2)))
    lnph += -(((gVc2_prm[i] - gVc2) * FVc2) / (2 * gVc2 ** 2)) * (2.0 - (((gVc2 ** 2) / (Vmix ** 4)) + (2.0 * gVc2 / (Vmix ** 2)) + 2.0) * math.exp(-gVc2 / (Vmix ** 2)))


    return lnph



def mix_fugacity_ind(*, P_kbar, T_K, XH2O, Vmix):
    """ This function calculates fugacity for a single sample.
    It returns the activity of each component (fugacity/fugacity in pure component)


    """

    P=P_kbar*1000
    TK=T_K
    XCO2=1-XH2O
    Y = [0] * 2
    Y[0]=XH2O
    Y[1]=XCO2

    # Calculate the constants you neeed
    k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess=get_EOS_params(P, TK)


    lnphi2kbL = [0.0, 0.0]
    lnphi2kbH = [0.0, 0.0]

    lnphi = [0.0, 0.0]
    phi_mix = [0.0, 0.0]
    activity = [0.0, 0.0]
    f = [0.0, 0.0]

    # Calculate at pressure of interest
    Z_pure = [0.0, 0.0]
    V_pure = [0.0, 0.0]


    # Initial guess for volumne

    if P<=2000:
        Vguess=1000
    elif P>20000:
        Vguess=10
    else:
        Vguess=100

    V_pure[1]=purevolume(1, Vguess, P, B, C, D, E, F, Vc, TK, b, g)
    V_pure[0]=purevolume(0, Vguess, P, B, C, D, E, F, Vc, TK, b, g)
    Z_pure[0]=P*V_pure[0]/(83.14467*TK)
    Z_pure[1]=P*V_pure[1]/(83.14467*TK)


    #H2O pure
    lnphi0=pure_lnphi(0, Z_pure, B, Vc, V_pure, C, D, E, F, g, b)
    #CO2 pure
    lnphi1=pure_lnphi(1, Z_pure, B, Vc, V_pure, C, D, E, F, g, b)


    # Funny maths you have to do incase P>2000 bars


    # First, calculate parameters with low pressure coefficients
    k1_temperature_LP, k2_temperature_LP, k3_temperature_LP, a1_LP, a2_LP, g_LP, b_LP, Vc_LP, B_LP, C_LP, D_LP, E_LP, F_LP, Vguess=get_EOS_params(500, TK)
    Z_pure_LP_2000 = [0.0, 0.0]
    V_pure_LP_2000 = [0.0, 0.0]
    V_pure_LP_2000[0]=purevolume(0, 100, 2000, B_LP, C_LP, D_LP, E_LP, F_LP, Vc_LP, TK, b_LP, g_LP)
    V_pure_LP_2000[1]=purevolume(1, 100, 2000, B_LP, C_LP, D_LP, E_LP, F_LP, Vc_LP, TK, b_LP, g_LP)

    Z_pure_LP_2000[0]=2000.0*V_pure_LP_2000[0]/(83.14467*TK)
    Z_pure_LP_2000[1]=2000.0*V_pure_LP_2000[1]/(83.14467*TK)

    # Low pressure
    lnphi0_LP=pure_lnphi(0, Z_pure_LP_2000, B_LP, Vc_LP, V_pure_LP_2000, C_LP, D_LP, E_LP, F_LP, g_LP, b_LP)
    lnphi1_LP=pure_lnphi(1, Z_pure_LP_2000, B_LP, Vc_LP, V_pure_LP_2000, C_LP, D_LP, E_LP, F_LP, g_LP, b_LP)



    # Same with high P
    k1_temperature_HP, k2_temperature_HP, k3_temperature_HP, a1_HP, a2_HP, g_HP, b_HP, Vc_HP, B_HP, C_HP, D_HP, E_HP, F_HP, Vguess=get_EOS_params(3000, TK)
    Z_pure_HP_2000 = [0.0, 0.0]
    V_pure_HP_2000 = [0.0, 0.0]
    V_pure_HP_2000[0]=purevolume(0, 100, 2000, B_HP, C_HP, D_HP, E_HP, F_HP, Vc_HP, TK, b_HP, g_HP)
    V_pure_HP_2000[1]=purevolume(1, 100, 2000, B_HP, C_HP, D_HP, E_HP, F_HP, Vc_HP, TK, b_HP, g_HP)
    Z_pure_HP_2000[0]=2000.0*V_pure_HP_2000[0]/(83.14467*TK)
    Z_pure_HP_2000[1]=2000.0*V_pure_HP_2000[1]/(83.14467*TK)


    #pure_HP
    lnphi0_HP=pure_lnphi(0, Z_pure_HP_2000, B_HP, Vc_HP, V_pure_HP_2000, C_HP, D_HP, E_HP, F_HP, g_HP, b_HP)
    lnphi1_HP=pure_lnphi(1, Z_pure_HP_2000, B_HP, Vc_HP, V_pure_HP_2000, C_HP, D_HP, E_HP, F_HP, g_HP, b_HP)

    if P>2000:
        # This is a weird thing described on Page 6 of Yoshimura -
        lnphi0=lnphi0-lnphi0_HP+lnphi0_LP
        lnphi1=lnphi1-lnphi1_HP+lnphi1_LP

    phi0_pure=math.exp(lnphi0)
    phi1_pure=math.exp(lnphi1)



    # Now we need to do the mixed fugacity part of this
    #--------------------------------------------------------------------------
    k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess=get_EOS_params(P, TK)
    Bij, Vcij, BVc_prm, BVc, Cijk, Vcijk, CVc2_prm, CVc2, Dijklm, Vcijklm, DVc4_prm, DVc4, Eijklmn, Vcijklmn, EVc5_prm,  EVc5, Fij, FVc2_prm, FVc2, bmix, b_prm, gijk, gVc2_prm, gVc2=mixing_rules(B, C, D, E, F, Vc, Y, b, g, k1_temperature, k2_temperature, k3_temperature)
    Zmix=(P*Vmix)/(83.14467*TK)
    lnphi_mix = [0.0, 0.0]
    phi_mix = [0.0, 0.0]
    lnphi_mix[0]=mix_lnphi(0,  Zmix, BVc_prm, CVc2_prm, DVc4_prm, EVc5_prm, FVc2_prm,FVc2, bmix, b_prm, gVc2, gVc2_prm, Vmix)
    lnphi_mix[1]=mix_lnphi(1,  Zmix, BVc_prm, CVc2_prm, DVc4_prm, EVc5_prm, FVc2_prm,FVc2, bmix, b_prm, gVc2, gVc2_prm, Vmix)



    # But what if P>2000, well we need to do these calcs at low and high P
    # High P - using Parameters from up above
    Bij_HP, Vcij_HP, BVc_prm_HP, BVc_HP, Cijk_HP, Vcijk_HP, CVc2_prm_HP, CVc2_HP, Dijklm_HP, Vcijklm_HP, DVc4_prm_HP, DVc4_HP, Eijklmn_HP, Vcijklmn_HP, EVc5_prm_HP,  EVc5_HP, Fij_HP, FVc2_prm_HP, FVc2_HP, bmix_HP, b_prm_HP, gijk_HP, gVc2_prm_HP, gVc2_HP=mixing_rules(B_HP, C_HP, D_HP, E_HP, F_HP, Vc_HP, Y, b_HP, g_HP, k1_temperature_HP, k2_temperature_HP, k3_temperature_HP)
    Vmix_HP=mixvolume(100, 2000, BVc_HP, CVc2_HP, DVc4_HP, EVc5_HP, FVc2_HP, bmix_HP, gVc2_HP, TK)
    Zmix_HP=(2000*Vmix_HP)/(83.14467*TK)
    lnphi_mix_HP = [0.0, 0.0]
    lnphi_mix_HP[0]=mix_lnphi(0,  Zmix_HP, BVc_prm_HP, CVc2_prm_HP, DVc4_prm_HP, EVc5_prm_HP, FVc2_prm_HP,FVc2_HP, bmix_HP, b_prm_HP, gVc2_HP, gVc2_prm_HP, Vmix_HP)
    lnphi_mix_HP[1]=mix_lnphi(1,  Zmix_HP, BVc_prm_HP, CVc2_prm_HP, DVc4_prm_HP, EVc5_prm_HP, FVc2_prm_HP, FVc2_HP, bmix_HP, b_prm_HP, gVc2_HP, gVc2_prm_HP, Vmix_HP)


    # Same for LP
    Bij_LP, Vcij_LP, BVc_prm_LP, BVc_LP, Cijk_LP, Vcijk_LP, CVc2_prm_LP, CVc2_LP, Dijklm_LP, Vcijklm_LP, DVc4_prm_LP, DVc4_LP, Eijklmn_LP, Vcijklmn_LP, EVc5_prm_LP,  EVc5_LP, Fij_LP, FVc2_prm_LP, FVc2_LP, bmix_LP, b_prm_LP, gijk_LP, gVc2_prm_LP, gVc2_LP=mixing_rules(B_LP, C_LP, D_LP, E_LP, F_LP,Vc_LP, Y, b_LP, g_LP, k1_temperature_LP, k2_temperature_LP, k3_temperature_LP)
    Vmix_LP=mixvolume(100, 2000, BVc_LP, CVc2_LP, DVc4_LP, EVc5_LP, FVc2_LP, bmix_LP, gVc2_LP, TK)
    Zmix_LP=(2000*Vmix_LP)/(83.14467*TK)
    lnphi_mix_LP = [0.0, 0.0]
    lnphi_mix_LP[0]=mix_lnphi(0,  Zmix_LP, BVc_prm_LP, CVc2_prm_LP, DVc4_prm_LP, EVc5_prm_LP, FVc2_prm_LP, FVc2_LP, bmix_LP, b_prm_LP, gVc2_LP, gVc2_prm_LP, Vmix_LP)
    lnphi_mix_LP[1]=mix_lnphi(1,  Zmix_LP, BVc_prm_LP, CVc2_prm_LP, DVc4_prm_LP, EVc5_prm_LP, FVc2_prm_LP,FVc2_LP, bmix_LP, b_prm_LP, gVc2_LP, gVc2_prm_LP, Vmix_LP)

    if P>2000:
        lnphi_mix[0]=lnphi_mix[0]-lnphi_mix_HP[0]+lnphi_mix_LP[0]
        lnphi_mix[1]=lnphi_mix[1]-lnphi_mix_HP[1]+lnphi_mix_LP[1]


    phi_mix[0]=math.exp(lnphi_mix[0])
    phi_mix[1]=math.exp(lnphi_mix[1])







    activity[0] = phi_mix[0] * Y[0] / phi0_pure
    activity[1] = phi_mix[1] * Y[1] / phi1_pure

    f[0] = Y[0] * P * phi_mix[0] / 1000.0  # fugacity in kbar
    f[1] = Y[1] * P * phi_mix[1] / 1000.0  # fugacity in kbar




    return f[0], f[1], activity[0], activity[1], Zmix





def mixing_rules(B, C, D, E, F, Vc, Y, b, g, k1_temperature, k2_temperature, k3_temperature):
    Bij = np.zeros((2, 2))
    Vcij = np.zeros((2, 2))
    BVc_prm = np.zeros(2)
    b_prm=np.zeros(2)
    BVc = 0.0
    Cijk = np.zeros((2, 2, 2))
    Vcijk = np.zeros((2, 2, 2))
    CVc2_prm = np.zeros(2)
    CVc2 = 0.0
    Dijklm = np.zeros((2, 2, 2, 2, 2))
    Vcijklm = np.zeros((2, 2, 2, 2, 2))
    Eijklmn=np.zeros((2, 2, 2, 2, 2, 2))
    Vcijklmn=np.zeros((2, 2, 2, 2, 2, 2))
    DVc4_prm=np.zeros(2)
    DVc4=0

    EVc5_prm = np.zeros(2)
    EVc5 = 0.0
    Fij = np.zeros((2, 2))
    FVc2_prm = np.zeros(2)
    FVc2 = 0.0
    gijk = np.zeros((2, 2, 2))
    gVc2_prm = np.zeros(2)
    gVc2 = 0.0

    for i in range(2):
        for j in range(2):
            k1 = 1.0 if i == j else k1_temperature
            Bij[i, j] = pow((cbrt_calc(B[i]) + cbrt_calc(B[j]))/2, 3.0) * k1

    for i in range(2):
        for j in range(2):
            Vcij[i, j] = pow((cbrt_calc(Vc[i]) + cbrt_calc(Vc[j]))/2, 3.0)


    for i in range(2):
        for j in range(2):
            BVc_prm[i] += 2 * Y[j] * Bij[i, j] * Vcij[i, j]



    for i in range(2):
        for j in range(2):

            BVc += Y[i] * Y[j] * Bij[i, j] * Vcij[i, j]

    for i in range(2):
        for j in range(2):
            for k in range(2):
                k2 = 1.0 if i == j and j == k else k2_temperature
                Cijk[i, j, k] = pow((cbrt_calc(C[i]) + cbrt_calc(C[j]) + cbrt_calc(C[k]))/3, 3.0) * k2

    for i in range(2):
        for j in range(2):
            for k in range(2):
                Vcijk[i, j, k] = pow((cbrt_calc(Vc[i]) + cbrt_calc(Vc[j]) + cbrt_calc(Vc[k]))/3, 3.0)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                CVc2_prm[i] += 3 * Y[j] * Y[k] * Cijk[i, j, k] * Vcijk[i, j, k] * Vcijk[i, j, k]

    CVc2=0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                CVc2 += Y[i] * Y[j] * Y[k] * Cijk[i, j, k] * Vcijk[i, j, k] * Vcijk[i, j, k]


    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        Dijklm[i, j, k, l, m] = pow((cbrt_calc(D[i]) + cbrt_calc(D[j]) + cbrt_calc(D[k]) + cbrt_calc(D[l]) + cbrt_calc(D[m]))/5, 3.0)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        Vcijklm[i, j, k, l, m] = pow((cbrt_calc(Vc[i]) + cbrt_calc(Vc[j]) + cbrt_calc(Vc[k]) + cbrt_calc(Vc[l]) + cbrt_calc(Vc[m]))/5, 3.0)


    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        DVc4_prm[i] += 5.0 * Y[j] * Y[k] * Y[l] * Y[m] * Dijklm[i, j, k, l, m] * pow(Vcijklm[i, j, k, l, m], 4.0)


    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        DVc4 += Y[i] * Y[j] * Y[k] * Y[l] * Y[m] * Dijklm[i, j, k, l, m] * pow(Vcijklm[i, j, k, l, m], 4)

# Missing Eijklmn,
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            Eijklmn[i, j, k, l, m, n] = pow((cbrt_calc(E[i]) + cbrt_calc(E[j]) + cbrt_calc(E[k]) + cbrt_calc(E[l]) + cbrt_calc(E[m]) + cbrt_calc(E[n]))/6, 3.0)


    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            Vcijklmn[i, j, k, l, m, n] = pow((cbrt_calc(Vc[i]) + cbrt_calc(Vc[j]) + cbrt_calc(Vc[k]) + cbrt_calc(Vc[l]) + cbrt_calc(Vc[m]) + cbrt_calc(Vc[n]))/6, 3.0)



    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            EVc5_prm[i] += 6.0 * Y[j] * Y[k] * Y[l] * Y[m] * Y[n] * Eijklmn[i, j, k, l, m, n] * pow(Vcijklmn[i, j, k, l, m, n], 5.0)


    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            EVc5 += Y[i] * Y[j] * Y[k] * Y[l] * Y[m] * Y[n] * Eijklmn[i, j, k, l, m, n] * pow(Vcijklmn[i, j, k, l, m, n], 5.0)

    for i in range(2):
        for j in range(2):
            Fij[i, j] = pow((cbrt_calc(F[i]) + cbrt_calc(F[j]))/2, 3.0)

    for i in range(2):
        for j in range(2):
            FVc2_prm[i] += 2.0 * Y[j] * Fij[i, j] * Vcij[i, j] * Vcij[i, j]

    for i in range(2):
        for j in range(2):
            FVc2 += Y[i] * Y[j] * Fij[i, j] * Vcij[i, j] * Vcij[i, j]

    bmix = Y[0] * b[0] + Y[1] * b[1]

    b_prm[0] = b[0]
    b_prm[1] = b[1]

    for i in range(2):
        for j in range(2):
            for k in range(2):
                k3 = 1.0 if i == j and j == k else k3_temperature
                gijk[i, j, k] = pow((cbrt_calc(g[i]) + cbrt_calc(g[j]) + cbrt_calc(g[k]))/3, 3.0) * k3

    for i in range(2):
        for j in range(2):
            for k in range(2):
                gVc2_prm[i] += 3.0 * Y[j] * Y[k] * gijk[i, j, k] * Vcijk[i, j, k] * Vcijk[i, j, k]


    for i in range(2):
        for j in range(2):
            for k in range(2):
                gVc2 += Y[i] * Y[j] * Y[k] * gijk[i, j, k] * Vcijk[i, j, k] * Vcijk[i, j, k]


    return Bij, Vcij, BVc_prm, BVc, Cijk, Vcijk, CVc2_prm, CVc2, Dijklm, Vcijklm, DVc4_prm, DVc4, Eijklmn, Vcijklmn, EVc5_prm,  EVc5, Fij, FVc2_prm, FVc2, bmix, b_prm, gijk, gVc2_prm, gVc2


## Getting EOS contsants themselves

def get_EOS_params(P, TK):
    """ This function returns the EOS 'constants' if you know the pressure and temperature

    """

    a1 = np.zeros(16)
    a2 = np.zeros(16)
    b = np.zeros(2)
    g = np.zeros(2)
    Vc = np.zeros(1)
    B = np.zeros(2)
    C = np.zeros(2)
    D = np.zeros(2)
    E = np.zeros(2)
    F = np.zeros(2)
    V = np.zeros(2)
    Vc = np.zeros(2)








    # Initial guess for volumne

    if P<=2000:
        Vguess=1000
    elif P>20000:
        Vguess=10
    else:
        Vguess=100


    if P <= 2000.0:
        for i in range(16):
            a1[i] = aL1[i]
            a2[i] = aL2[i]
        # These are the binary interaction parameters
        k1_temperature = 3.131 - (5.0624 / 10**3.0) * TK + (1.8641 / 10**6) * TK**2 - 31.409 / TK
        k2_temperature = -46.646 + (4.2877 / 10**2.0) * TK - (1.0892 / 10**5) * TK**2 + 1.5782 * 10**4 / TK
        k3_temperature = 0.9
    else:
        for i in range(16):
            a1[i] = aH1[i]
            a2[i] = aH2[i]
        # Same, but for higher pressures
        k1_temperature = 9.034 - (7.9212 / 10**3) * TK + (2.3285 / 10**6) * TK**2 - 2.4221 * 10**3 / TK
        k2_temperature = -1.068 + (1.8756 / 10**3) * TK - (4.9371 / 10**7) * TK**2 + 6.6180 * 10**2 / TK
        k3_temperature = 1.0

    b[0] = a1[14]  # beta for H2O
    b[1] = a2[14]  # beta for CO2
    g[0] = a1[15]  # gamma for H2O
    g[1] = a2[15]  # gamma for CO2

    Vc[0] = 83.14467 * Tc1 / Pc1
    B[0] = a1[1] + a1[2] / ((TK / Tc1) * (TK / Tc1)) + a1[3] / ((TK / Tc1) * (TK / Tc1) * (TK / Tc1))
    C[0] = a1[4] + a1[5] / ((TK / Tc1) * (TK / Tc1)) + a1[6] / ((TK / Tc1) * (TK / Tc1) * (TK / Tc1))
    D[0] = a1[7] + a1[8] / ((TK / Tc1) * (TK / Tc1)) + a1[9] / ((TK / Tc1) * (TK / Tc1) * (TK / Tc1))
    E[0] = a1[10] + a1[11] / ((TK / Tc1) * (TK / Tc1)) + a1[12] / ((TK / Tc1) * (TK / Tc1) * (TK / Tc1))
    F[0] = a1[13] / ((TK / Tc1) * (TK / Tc1) * (TK / Tc1))

    Vc[1] = 83.14467 * Tc2 / Pc2
    B[1] = a2[1] + a2[2] / ((TK / Tc2) * (TK / Tc2)) + a2[3] / ((TK / Tc2) * (TK / Tc2) * (TK / Tc2))
    C[1] = a2[4] + a2[5] / ((TK / Tc2) * (TK / Tc2)) + a2[6] / ((TK / Tc2) * (TK / Tc2) * (TK / Tc2))
    D[1] = a2[7] + a2[8] / ((TK / Tc2) * (TK / Tc2)) + a2[9] / ((TK / Tc2) * (TK / Tc2) * (TK / Tc2))
    E[1] = a2[10] + a2[11] / ((TK / Tc2) * (TK / Tc2)) + a2[12] / ((TK / Tc2) * (TK / Tc2) * (TK / Tc2))
    F[1] = a2[13] / ((TK / Tc2) * (TK / Tc2) * (TK / Tc2))
    return k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess

## Lets wrap all these functions up.

def calculate_molar_volume_ind_DZ2006(*, P_kbar, T_K, XH2O):
    """ This function calculates molar volume for a known pressure, T in K and XH2O (mol frac) for a single value
    """

    P=P_kbar*1000
    TK=T_K

    # Calculate the constants you neeed
    k1_temperature, k2_temperature, k3_temperature, a1, a2, g, b, Vc, B, C, D, E, F, Vguess=get_EOS_params(P, TK)

    if XH2O==0:
        mol_vol=purevolume(1, Vguess, P, B, C, D, E, F, Vc, TK, b, g)

    if XH2O==1:
        mol_vol=purevolume(0, Vguess, P, B, C, D, E, F, Vc, TK, b, g)

    else:
        XCO2=1-XH2O
        Y = [0] * 2
        Y[0]=XH2O
        Y[1]=XCO2
        Bij, Vcij, BVc_prm, BVc, Cijk, Vcijk, CVc2_prm, CVc2, Dijklm, Vcijklm, DVc4_prm, DVc4, Eijklmn, Vcijklmn, EVc5_prm,  EVc5, Fij, FVc2_prm, FVc2, bmix, b_prm, gijk, gVc2_prm, gVc2=mixing_rules(B, C,D, E, F, Vc, Y, b,    g, k1_temperature, k2_temperature, k3_temperature)


        mol_vol=mixvolume(Vguess, P, BVc, CVc2, DVc4, EVc5, FVc2, bmix, gVc2, T_K)

    return mol_vol


def calculate_molar_volume_DZ2006(*, P_kbar, T_K, XH2O):
    """ Used to calculate molar volume in a loop for multiple inputs


    """

    P_kbar, T_K, XH2O=ensure_series(P_kbar, T_K, XH2O)

    # Check all the same length
    lengths = [len(P_kbar), len(T_K), len(XH2O)]
    if len(set(lengths)) != 1:
        raise ValueError("All input Pandas Series must have the same length.")

    # Set up loop
    mol_vol=np.empty(len(P_kbar), float)

    for i in range(0, len(P_kbar)):
        mol_vol[i]=calculate_molar_volume_ind_DZ2006(P_kbar=P_kbar.iloc[i].astype(float), T_K=T_K.iloc[i].astype(float), XH2O=XH2O.iloc[i].astype(float))





    return mol_vol

def calculate_Pressure_ind_DZ2006(*, mol_vol, T_K, XH2O, Pguess=None):
    """ This function calculates pressure for a known molar volume, T in K and XH2O (mol frac) for a single value
    """
    V=mol_vol
    if Pguess is None:
        if V>1000:
            Pguess=1000
        elif V<10:
            Pguess=20000
        else:
            Pguess=200

    TK=T_K

    # lets do for low pressure initially


    if XH2O==0:
        P=purepressure(1,  V, Pguess, TK)

    elif XH2O==1:
        P=purepressure(0, V, Pguess, TK)

    else:
        XCO2=1-XH2O
        Y = [0] * 2
        Y[0]=XH2O
        Y[1]=XCO2

        P=mixpressure(Pguess, V, T_K, Y)

    return P

def calculate_Pressure_DZ2006(*, mol_vol=None, density=None, T_K, XH2O):
    """ Used to calculate molar volume in a loop for multiple inputs


    """
    # Make all a panda series



    if mol_vol is None and density is not None:
        mol_vol=density_to_mol_vol(density=density, XH2O=XH2O)

    mol_vol, T_K, XH2O=ensure_series(mol_vol, T_K, XH2O)

    # Check all the same length
    lengths = [len(mol_vol), len(T_K), len(XH2O)]
    if len(set(lengths)) != 1:
        raise ValueError("All input Pandas Series must have the same length.")

    # Set up loop
    P=np.empty(len(mol_vol), float)

    for i in range(0, len(mol_vol)):
        P[i]=calculate_Pressure_ind_DZ2006(mol_vol=mol_vol.iloc[i].astype(float), T_K=T_K.iloc[i].astype(float), XH2O=XH2O.iloc[i].astype(float))



    return P


def mix_fugacity(*, P_kbar, T_K, XH2O, Vmix):

    """ Used to calculate fugacity, compressability and activities for a panda series

    """
    # Make everything a pandas series

    P_kbar, T_K, XH2O, Vmix=ensure_series_4(P_kbar, T_K, XH2O, Vmix)



    #Check all the same length
    lengths = [len(P_kbar), len(T_K), len(XH2O), len(Vmix)]
    if len(set(lengths)) != 1:
        raise ValueError("All input Pandas Series must have the same length.")

    f=np.empty(len(P_kbar), float)
    a_CO2=np.empty(len(P_kbar), float)
    a_H2O=np.empty(len(P_kbar), float)
    f_CO2=np.empty(len(P_kbar), float)
    f_H2O=np.empty(len(P_kbar), float)
    Zmix=np.empty(len(P_kbar), float)
    for i in range(0, len(P_kbar)):

        f_H2O[i], f_CO2[i], a_H2O[i], a_CO2[i], Zmix[i]=mix_fugacity_ind(P_kbar=P_kbar.iloc[i].astype(float), T_K=T_K.iloc[i].astype(float), XH2O=XH2O.iloc[i].astype(float), Vmix=Vmix.iloc[i].astype(float))

    return f_H2O, f_CO2, a_H2O,a_CO2,  Zmix


def mol_vol_to_density(*, mol_vol, XH2O):
    """ Converts molar mass to density for a given XH2O"""
    density=((1-XH2O)*44 + (XH2O)*18)/mol_vol
    return density

def density_to_mol_vol(*, density, XH2O):
    """ Converts density in g/cm3 to molar volume for a given XH2O"""
    mol_vol=((1-XH2O)*44 + (XH2O)*18)/density
    return mol_vol



def calc_prop_knownP_EOS_DZ2006(*, P_kbar=1, T_K=1200, XH2O=1):
    """ This function calculates molar volume, density, compressability factor, fugacity, and activity for mixed H2O-CO2 fluids
    using the EOS of Span and Wanger. It assumes you know P, T, and XH2O.

    Parameters
    -------------------
    P_kbar: float, np.array, pd.Series
        Pressure in kbar
    T_K: float, np.array, pd.Series
        Temperature in Kelvin
    XH2O: float, np.array, pd.Series
        Molar fraction of H2O in the fluid phase.

    Returns
    -------------------
    pd.DataFrame

    """



    # First, check all pd Series


    mol_vol=calculate_molar_volume_DZ2006(P_kbar=P_kbar, T_K=T_K, XH2O=XH2O)


    f_H2O, f_CO2, a_H2O, a_CO2, Zmix=mix_fugacity(P_kbar=P_kbar, T_K=T_K, XH2O=XH2O,
                                                      Vmix=mol_vol)
    density=mol_vol_to_density(mol_vol=mol_vol, XH2O=XH2O)
    # 'T_K': T_K,
    # 'P_kbar': P_kbar,
    # 'XH2O': XH2O,
    #


    df=pd.DataFrame(data={'P_kbar': P_kbar,
                          'T_K': T_K,
                          'XH2O': XH2O,
                          'XCO2': 1-XH2O,
                          'Molar Volume (cm3/mol)': mol_vol,
                          'Density (g/cm3)': density,
                          'Compressability_factor': Zmix,
                          'fugacity_H2O (kbar)': f_H2O,
                          'fugacity_CO2 (kbar)': f_CO2,
                          'activity_H2O': a_H2O,
                          'activity_CO2': a_CO2})

    return df



def calculate_entrapment_P_XH2O(*, XH2O, CO2_dens_gcm3, T_K):
    """" This function calculates pressure for a measured CO$_2$ density, temperature and estimate of initial XH2O.
    It first corrects the density to obtain a bulk density for a CO2-H2O mix, assuming that H2O was lost from the inclusion.
    correcting for XH2O. It assumes that H2O has been lost from the inclusion (see Hansteen and Klugel, 2008 for method). It also calculates using other
    pure CO2 equation of states for comparison

    Parameters
    ----------------------
    XH2O: float, pd.Series.
        The molar fraction of H2O in the fluid. Should be between 0 and 1. Can get an estimate from say VESical.

    CO2_dens_gcm3: float, pd.Series
        Measured CO2 density in g/cm3

    T_K: float, pd.Series
        Temperature in Kelvin.

    Returns
    -----------------------------
    pd.DataFrame:
        Columns showing:
        P_kbar_pureCO2_SW96: Pressure calculated for the measured CO$_2$ density using the pure CO2 EOS from Span and Wanger (1996)
        P_kbar_pureCO2_SP94: Pressure calculated for the measured CO$_2$ density using the pure CO2 EOS from Sterner and Pitzer (1994)
        P_kbar_pureCO2_DZ06: Pressure calculated from the measured CO$_2$ density using the pure CO2 EOs from Duan and Zhang (2006)
        P_kbar_mixCO2_DZ06: Pressure calculated from the reconstructed mixed fluid density using the mixed EOS from Duan and Zhang (2006)
        P Mix/P Pure DZ06: Correction factor - e.g. how much deeper the pressure is from the mixed EOS
        rho_mix_calc: Bulk density calculated (C+H) at time of entrapment
        CO2_dens_gcm3: Input CO2 density
        T_K: input temperature
        XH2O: input molar fraction of H2O

    """
    XH2O, rho_meas, T_K=ensure_series(a=XH2O, b=CO2_dens_gcm3, c=T_K)
    alpha=XH2O/(1-XH2O)
    # This gets the bulk density of the CO2-H2O fluid
    rho_orig=rho_meas*(1+alpha*(18/44))
    # Lets calculate the pressure using SW96
    P_SW=calculate_P_for_rho_T(T_K=T_K, CO2_dens_gcm3=rho_meas, EOS='SW96')
    P_SP=calculate_P_for_rho_T(T_K=T_K, CO2_dens_gcm3=rho_meas, EOS='SP94')
    # Same for DZ2006
    P_DZ=calculate_Pressure_DZ2006(density=rho_meas, T_K=T_K, XH2O=XH2O*0)
    # Now doing it with XH2O
    P_DZ_mix=calculate_Pressure_DZ2006(density=rho_orig, T_K=T_K, XH2O=XH2O)

    df=pd.DataFrame(data={
        'P_kbar_pureCO2_SW96': P_SW['P_kbar'],
        'P_kbar_pureCO2_SP94': P_SW['P_kbar'],
        'P_kbar_pureCO2_DZ06': P_DZ/1000,
        'P_kbar_mixCO2_DZ06': P_DZ_mix/1000,
        'P Mix/P Pure DZ06': P_DZ_mix/P_DZ,
        'rho_mix_calc': rho_orig,
        'CO2_dens_gcm3': rho_meas,
        'T_K': T_K,
        'XH2O': XH2O})

    return df








