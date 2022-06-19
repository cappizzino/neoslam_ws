import numpy as np
from numpy import linalg as LA

def normc(matrix_a):
    'Norm'
    matrix_b = np.zeros((matrix_a.shape[0], matrix_a.shape[1]))
    for i in range(matrix_a.shape[1]):
        matrix_b[:, i] = matrix_a[:, i]/LA.norm(matrix_a[:, i])
    return matrix_b

def min_delta(d1, d2, max_):
    delta = np.min([np.abs(d1-d2), max_-np.abs(d1-d2)])
    return delta

def clip_rad_180(angle):
    while angle > np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle

def clip_rad_360(angle):
    while angle < 0:
        angle += 2*np.pi
    while angle >= 2*np.pi:
        angle -= 2*np.pi
    return angle

def signed_delta_rad(angle1, angle2):
    dir = clip_rad_180(angle2 - angle1)
    
    delta_angle = abs(clip_rad_360(angle1) - clip_rad_360(angle2))
    
    if (delta_angle < (2*np.pi-delta_angle)):
        if (dir>0):
            angle = delta_angle
        else:
            angle = -delta_angle
    else: 
        if (dir>0):
            angle = 2*np.pi - delta_angle
        else:
            angle = -(2*np.pi-delta_angle)
    return angle

# Grid Cells
#PC_DIM_XY = 11#100
#PC_DIM_TH = 36#18

#POSECELL_VTRANS_SCALING = 1./5.
#ODO_ROT_SCALING         = 1 #1./10. #np.pi/180./1.

# Spatial View Cells
#VT_START = 5
#VT_GLOBAL_DECAY         = 0.1
#VT_ACTIVE_DECAY         = 1.0
#VT_MATCH_THRESHOLD      = 0.8#0.75
#VT_SHIFT_MATCH          = 4#20

# Experince Map
#EXP_DELTA_PC_THRESHOLD  = 2#1.0
#EXP_CORRECTION          = 0.5
#EXP_LOOPS               = 20#100