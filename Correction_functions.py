from InputOutput import LeoPosIdx, LeoQuatIdx, SatPosIdx, SatApoIdx, SatClkIdx, SatBiaIdx
import numpy as np

from COMMON import GnssConstants as Const
from COMMON.Misc import modulo
from COMMON.Dates import convertYearDoy2JulianDay

def computeLeoComPos(Sod, LeoPosInfo):

    xPos = yPos = zPos = None

    # try:      # If LeoPosInfo is a dictionary
    #     # First we ensure Sod index
    #     sod_index = LeoPosInfo.get("SOD").index(Sod)

    #     # Iterate the keys of a given dictionary to acquire x, y and z position
    #     for key in LeoPosInfo.keys():

    #         # In case of finding one of them, it takes a list (value associated to that key) and finds the correspondent value using sod_index
    #         if "x" in key:
    #             xPos = LeoPosInfo.get(key)[sod_index]

    #         elif "y" in key:
    #             yPos = LeoPosInfo.get(key)[sod_index]

    #         elif "z" in key:
    #             zPos = LeoPosInfo.get(key)[sod_index]

    # except:
    #     pass

    try:
        LeoPosInfo = LeoPosInfo.to_numpy()
        value_position = np.where(LeoPosInfo[:,LeoPosIdx["SOD"]] == Sod)[0][0]

        xPos = LeoPosInfo[value_position:, LeoPosIdx["xCM"]][0]*1000
        yPos = LeoPosInfo[value_position:, LeoPosIdx["yCM"]][0]*1000
        zPos = LeoPosInfo[value_position:, LeoPosIdx["zCM"]][0]*1000
    except:
        pass

    return (xPos, yPos, zPos)


# -----------------------------------------------------------------------------------------------------------------------

def computeSatClkBias(Sod, SatLabel, SatClkInfo):
    # Things to discuss, there are cases where the Sod is 20 but this clk uses jump of 30 seconds
    clock_bias = 0
    close_value_down = close_value_up = 0   # In case of not having any Sod similar, interpolate between the closest values

    # Setting decimal precision for flaots
    # np.set_printoptions(suppress=True, precision=32)    # Force pandas to work in complete notation and not scientific notation
    np.set_printoptions(precision=32)       # Set precision of np to 32 to use all decimals

    # First, filter by satellite
    SatClkInfo = SatClkInfo[SatClkInfo[SatClkIdx["CONST"]] == SatLabel[:1]]
    SatClkInfo = SatClkInfo[SatClkInfo[SatClkIdx["PRN"]] == SatLabel[1:]]

    # Converting SOD from object to int
    SatClkInfo[SatClkIdx["SOD"]] = SatClkInfo[SatClkIdx["SOD"]].astype(int)     # Convert SOD Column from object type to int type

    try:
        if Sod in SatClkInfo[SatClkIdx["SOD"]].unique():                     # SatClkInfo returns object type, not int or float type
            # Get values of SatClkInfo filtered
            SatClkInfo = SatClkInfo[SatClkInfo[SatClkIdx["SOD"]] == Sod].values

            # Get a value of np in the column SatClkIdx["CLK-BIAS"] and in every row and set it to np.float64
            # set to np.float64 and getting the first value of the given array
            clock_bias = SatClkInfo[:, SatClkIdx["CLK-BIAS"]].astype(np.float64)[0]


        else:
            # First, we acquire the last element for cases lower than Sod, because un case of having more than 1 option, 
            # the closer one is the last position of the array
            close_value_down = SatClkInfo[SatClkInfo[SatClkIdx["SOD"]] < Sod][-1:]     # Getting the last element for cases lower than Sod
            close_value_down = close_value_down.values

            close_value_up = SatClkInfo[SatClkInfo[SatClkIdx["SOD"]] > Sod][:1]        # Getting the first element for cases higher than Sod
            close_value_up = close_value_up.values

            # The idea is to compute first the differences to get the pendent of the line, but doing this we will love the offset
            # Then, we have to add this offset as a sum

            clock_bias = (\
                            (close_value_up[:, SatClkIdx["CLK-BIAS"]].astype(np.float64)[0] - close_value_down[:, SatClkIdx["CLK-BIAS"]].astype(np.float64)[0])/  \
                            (close_value_up[:, SatClkIdx["SOD"]][0] - close_value_down[:, SatClkIdx["SOD"]][0]) \
                        ) * Sod \
                        + close_value_down[:, SatClkIdx["CLK-BIAS"]].astype(np.float64)[0]      # Adding the offset removed by the last expression
    except:
        pass

    return clock_bias


# -----------------------------------------------------------------------------------------------------------------------

def computeRcvrApo(Conf, Year, Doy, Sod, SatLabel, LeoQuatInfo):
    
    # STEP 1: -------------------------------------------------------------------
    # Acquiring Center of Masses, Antenna Reference Frame and Phase Center Offset
    COM = Conf['LEO_COM_POS']
    ARP = Conf['LEO_ARP_POS']

    if SatLabel[0] == 'G':
        PCO = Conf['LEO_PCO_GPS']
    elif SatLabel[0] == 'E':
        PCO = Conf['LEO_PCO_GAL']

    # STEP 2: -------------------------------------------------------------------
    # Acquiring Antenna Phase Center by using the previous data
    COM_to_ARP = np.subtract(ARP, COM)
    APC_at_SRF = np.add(COM_to_ARP, PCO)       # This is referred to the Satellite Reference Frame, use quarternials to move to ECI coordinates

    # STEP 3: -------------------------------------------------------------------
    # Apply Satellite Quaternions to rotate the Satellite Frame Reference towards the Earth Centered Inertial (ECI) by building the Rotation Matrix
    LeoQuatInfo = LeoQuatInfo[LeoQuatInfo[LeoQuatIdx["SOD"]] == Sod]

    # Converting SOD from object to int
    q0 = LeoQuatInfo[LeoQuatIdx["q0"]].iloc[0]
    q1 = LeoQuatInfo[LeoQuatIdx["q1"]].iloc[0]
    q2 = LeoQuatInfo[LeoQuatIdx["q2"]].iloc[0]
    q3 = LeoQuatInfo[LeoQuatIdx["q3"]].iloc[0]

    # Creating Rotation Matrix for later computation
    rotation_matrix = np.array([[  (1 - 2*q2**2 - 2*q3**2),  2*(q1*q2 - q0*q3),         2*(q0*q2 + q1*q3)],
                                [   2*(q1*q2 + q0*q3),       (1 - 2*q1**2 - 2*q3**2),   2*(q2*q3 - q0*q1)],
                                [   2*(q1*q3 - q0*q2),       2*(q0*q1 + q2*q3),         (1 - 2*q1**2 - 2*q2**2)]
                                ])

    APC_at_SRF = np.array(APC_at_SRF)

    APC_at_ECI = np.dot(rotation_matrix, APC_at_SRF)

    # STEP 4: -------------------------------------------------------------------
    # Convert ECI coordinates to ECEF coordinates with the simplified model for Greenwich Siderial Time

    # Acquiring Greenwich Siderial Time Reference, but need to obtain Jualian Day Number and fraction of the day first
    JDN = convertYearDoy2JulianDay(Year, Doy, Sod) - 2415020    # Julian Day Number
    fday = Sod / Const.S_IN_D                                   # fday is Fractional part of the day

    gstr = modulo(279.690983 + 0.9856473354*JDN + 360*fday + 180, 360)      # Convert o radians and rotate it over the third axis (Applying Earth Rotation)
    gstr = np.deg2rad(gstr)     # Corvert it from degree to radian


    # Build rotation matrix for ECI to ECEF conversion based on GST
    rotation_matrix = np.array([[np.cos(gstr),      np.sin(gstr),       0],
                                [-np.sin(gstr),     np.cos(gstr),       0],
                                [0,                 0,                  1]
                                ])

    APC_at_ECEF_coordinates = np.dot(rotation_matrix, APC_at_ECI)

    return APC_at_ECEF_coordinates


# -----------------------------------------------------------------------------------------------------------------------

def lagrange_basis(x, x_values, i):
    L_i = 1.0
    n = len(x_values)
    for j in range(n):
        if j != i:
            L_i *= (x - x_values[j]) / (x_values[i] - x_values[j])
    return L_i

def lagrange_interpolation(x, x_values, y_values):
    n = len(x_values)
    y = 0.0
    for i in range(n):
        y += y_values[i] * lagrange_basis(x, x_values, i)
    return y

def computeSatComPos(TransmissionTime, SatPosInfo, SatLabel): # Apply Lagrange 10 points
    
    # Filter to get 10 points of its data
    # This is not working because the step is about 300 points
    # if Sod < 50:
    #     n_min = 0
    #     n_max = (100 - Sod) / 10
    # else:
    #     n_min = 1
    #     n_min = 5

    # SatPosInfo = SatPosInfo[SatPosInfo[SatPosIdx["SOD"]].astype(int) > n_min * Sod]
    # SatPosInfo = SatPosInfo[SatPosInfo[SatPosIdx["SOD"]].astype(int) < (Sod + n_max*10) ]
    SatPosInfo = SatPosInfo[SatPosInfo[SatPosIdx["CONST"]] == SatLabel[:1]]
    SatPosInfo = SatPosInfo[SatPosInfo[SatPosIdx["PRN"]] == SatLabel[1:]]


    times = SatPosInfo[SatPosIdx["SOD"]].values
    times = times.astype(int)

    xSatPos = SatPosInfo[SatPosIdx["xCM"]].values
    xSatPos = xSatPos.astype(float)

    ySatPos = SatPosInfo[SatPosIdx["yCM"]].values
    ySatPos = ySatPos.astype(float)

    zSatPos = SatPosInfo[SatPosIdx["zCM"]].values
    zSatPos = zSatPos.astype(float)

    # Get the indices of the 10 closest times to TransmissionTime
    closest_indices = np.argsort(np.abs(times - TransmissionTime))[:10]

    # Select the corresponding times and positions for interpolation
    closest_times = times[closest_indices]
    closest_x = xSatPos[closest_indices]*1000
    closest_y = ySatPos[closest_indices]*1000
    closest_z = zSatPos[closest_indices]*1000

    x_CoM = lagrange_interpolation(TransmissionTime, closest_times, closest_x)
    y_CoM = lagrange_interpolation(TransmissionTime, closest_times, closest_y)
    z_CoM = lagrange_interpolation(TransmissionTime, closest_times, closest_z)
    
    return (x_CoM, y_CoM, z_CoM)


# -----------------------------------------------------------------------------------------------------------------------

def applySagnac(SatComPos, FlightTime):

    angle = Const.OMEGA_EARTH * FlightTime
    # angle = np.deg2rad(angle)
    rot_matrix = np.array( [[np.cos(angle),    np.sin(angle),   0],
                            [-np.sin(angle),   np.cos(angle),   0],
                            [0,                0,               1]])

    sagnac = np.dot(rot_matrix, SatComPos)

    return sagnac


# -----------------------------------------------------------------------------------------------------------------------

def computeSatApo(SatLabel, SatComPos, SunPos, SatApoInfo):
    SatApoInfo = SatApoInfo[SatApoInfo[SatApoIdx["CONST"]] == SatLabel[:1]]
    SatApoInfo = SatApoInfo[SatApoInfo[SatApoIdx["PRN"]] == SatLabel[1:]]

    if SatLabel[0] == 'G':
        L1 = Const.GPS_L1_MHZ
        L2 = Const.GPS_L2_MHZ
    else:
        L1 = Const.GAL_E1_MHZ
        L2 = Const.GAL_E5A_MHZ

    APO_L1 = np.array([float(SatApoInfo[SatApoIdx["x_f1"]].iloc[0])/1000, float(SatApoInfo[SatApoIdx["y_f1"]].iloc[0])/1000, float(SatApoInfo[SatApoIdx["z_f1"]].iloc[0])/1000])
    APO_L2 = np.array([float(SatApoInfo[SatApoIdx["x_f2"]].iloc[0])/1000, float(SatApoInfo[SatApoIdx["y_f2"]].iloc[0])/1000, float(SatApoInfo[SatApoIdx["z_f2"]].iloc[0])/1000])


    APO = (L1**2 * APO_L1 - L2**2 * APO_L2) / (L1**2 - L2**2)       # Computing Iono-Free Combination: https://gssc.esa.int/navipedia/index.php/Ionosphere-free_Combination_for_Dual_Frequency_Receivers

    # From Center of Masses, Receiver Position, Sun Position and Antenna Phase Offset, compute Antenna Phase Offset position

    norm = np.linalg.norm(SatComPos)
    k = - SatComPos / norm

    norm = np.linalg.norm(SunPos - SatComPos)
    e = (SunPos - SatComPos) / norm
    j = np.cross(k, e)

    # Normalize the j vector
    norm = np.linalg.norm(j)
    j = j / norm

    # j = np.dot(k, e)

    i = np.cross(j, k)      # As j and k are ortogonal, its product is unitary, then no normalization is needed
    # i = np.dot(j, k)

    R = np.array([i, j, k]).T


    # --------------------------------------------------------------------------------------------

    APO = np.dot(R, APO)
    
    # APO = np.dot(R, RcvrPos)

    # --------------------------------------------------------------------------------------------

    return APO


# -----------------------------------------------------------------------------------------------------------------------

def getSatBias(GammaF1F2, SatLabel, SatBiaInfo):        # https://gssc.esa.int/navipedia/index.php/Combining_pairs_of_signals_and_clock_definition
    CodeBias = -0.0000000001
    PhaseBias = -0.00000000001

    SatBiaInfo = SatBiaInfo[SatBiaInfo[SatBiaIdx["CONST"]] == SatLabel[:1]]
    SatBiaInfo = SatBiaInfo[SatBiaInfo[SatBiaIdx["PRN"]] == SatLabel[1:]]

    if SatLabel[0] == 'G':
        L1 = Const.GPS_L1_MHZ
        L2 = Const.GPS_L2_MHZ
    else:
        L1 = Const.GAL_E1_MHZ
        L2 = Const.GAL_E5A_MHZ

    # CodeBias = (SatBiaInfo[SatBiaIdx["OBS_f1_C"]].astype(float) - GammaF1F2 * SatBiaInfo[SatBiaIdx["OBS_f2_C"]].astype(float)) / (1 - GammaF1F2)
    # PhaseBias = (SatBiaInfo[SatBiaIdx["OBS_f1_P"]].astype(float) - GammaF1F2 * SatBiaInfo[SatBiaIdx["OBS_f2_P"]].astype(float)) / (1 - GammaF1F2)

    CodeBias = - (SatBiaInfo[SatBiaIdx["OBS_f1_C"]].astype(float)*L1**2 - SatBiaInfo[SatBiaIdx["OBS_f2_C"]].astype(float)*L2**2) / (L1**2 - L2**2) 
    # PhaseBias = (SatBiaInfo[SatBiaIdx["OBS_f1_P"]].astype(float)*L1**2  - SatBiaInfo[SatBiaIdx["OBS_f2_P"]].astype(float)*L2**2)  / (L1**2 - L2**2) 

    # ClockBias = (SatBiaInfo[SatBiaIdx["CLK_f1_C"]].astype(float) + GammaF1F2 * SatBiaInfo[SatBiaIdx["CLK_f2_C"]].astype(float)) / (1 + GammaF1F2)

    # NEED TO PASS FROM Nanoseconds TO Meters, then pass nanoseconds to seconds
    CodeBias = (CodeBias / 1000000000) * Const.SPEED_OF_LIGHT
    PhaseBias = (PhaseBias / 1000000000)  * Const.SPEED_OF_LIGHT

    return CodeBias, PhaseBias


# -----------------------------------------------------------------------------------------------------------------------

def computeDtr(SatComPos_1, SatComPos, Sod, Sod_1):
    # DTR = -2 * (Satellite Position * Velocity Satellite)/ Speed of Light

    velocity = (SatComPos - SatComPos_1) / (Sod - Sod_1)

    dtr = -2 * np.dot(SatComPos, velocity) / (Const.SPEED_OF_LIGHT)

    return dtr


# -----------------------------------------------------------------------------------------------------------------------

def getUERE(Conf, SatLabel):
    sigmaUERE = 0

    if SatLabel[0] == 'G':
        sigmaUERE = Conf['GPS_UERE']

    elif SatLabel[0] == 'E':
        sigmaUERE = Conf['GAL_UERE']

    return sigmaUERE


# -----------------------------------------------------------------------------------------------------------------------

def computeGeoRange(SatCopPos, RcvrPos):
    # Taking into account that both arguments have X, Y and Z coordinates

    SatCopPos = np.array(SatCopPos)
    RcvrPos = np.array(RcvrPos)
    
    diff_vector = np.subtract(SatCopPos, RcvrPos)

    # geo_range =  np.sqrt( np.power(SatCopPos[0] - RcvrPos[0], 2) + np.power(SatCopPos[1] - RcvrPos[1], 2) + np.power(SatCopPos[2] - RcvrPos[2], 2)   )
    # Calculate the geometrical range (at Euclidean distance) ????
    geo_range = np.linalg.norm(diff_vector)     # https://gssc.esa.int/navipedia/index.php/Geometric_Range_Modelling

    return geo_range


# -----------------------------------------------------------------------------------------------------------------------

# def estimateRcvrClk(CodeResidual, SigmaUERE):
#     rcvr_clk_bias_estimate = 0
#     CodeResidual = np.array(CodeResidual)
#     SigmaUERE = np.array(SigmaUERE)
    
#     # Calculate weights as the inverse of the variances (SigmaUERE squared)
#     if SigmaUERE != 0:
#         weights = 1 / (SigmaUERE ** 2)

#         # Compute the weighted average of the residuals
#         weighted_sum = np.sum(weights * CodeResidual)
#         total_weight = np.sum(weights)

#         # Calculate the receiver clock bias estimate
#         rcvr_clk_bias_estimate = weighted_sum / total_weight
    
#     return rcvr_clk_bias_estimate

def estimateRcvrClk(Residuals_Info, CorrInfo):
    gal_sum = 0
    gps_sum = 0
    n_counter = 0

    for gal_residual in Residuals_Info['E']:
        gal_sum += gal_residual
    gal_average = gal_sum / (len(Residuals_Info['E']) - 1)

    for gps_residual in Residuals_Info['G']:
        gps_sum += gps_residual
    gps_average = gps_sum / (len(Residuals_Info['G']))

    for SatLabel in Residuals_Info["SatLabel"]:
        if SatLabel[0] == 'E':
            CorrInfo[SatLabel]['RcvrClk'] = gal_average * Residuals_Info['SigmaUere'][n_counter]
        elif SatLabel[0] == 'G':
            CorrInfo[SatLabel]['RcvrClk'] = gps_average * Residuals_Info['SigmaUere'][n_counter]

        CorrInfo[SatLabel]["CodeResidual"] -= CorrInfo[SatLabel]['RcvrClk']
        CorrInfo[SatLabel]["PhaseResidual"] -= CorrInfo[SatLabel]['RcvrClk']

        n_counter += 1

    return CorrInfo