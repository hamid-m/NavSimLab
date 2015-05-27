__author__ = 'Leonardo D Le'
# Created: October 10th, 2014
# Last Update: May 27th, 2015

# -*- coding: utf-8 -*-
import os
import sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.gridspec import GridSpec
from numpy import linalg as lina
# Clear the Console
os.system('cls')
plt.close('all')


'''
========================================================================================================================
                                                GLOBAL VARIABLE DECLARATIONS
========================================================================================================================
'''
'''
---------------------
A. Conversion Factors
---------------------
'''
global d2r, r2d, micro_g, g, R_0, ecc_o, mu, OMEGA_ie, c, dd2sec, hr2sec, mn2sec, tol
c = 2.99792458E08                           # Speed of light in (m/s)
ecc_o = 0.0818191908425                     # WGS84 Eccentricity
g = 9.80665                                 # Earth graviational acceleration (m/s^2)
mu = 3.986004418E+14                        # WGS84 Earth gravitational constant (m^3/s^2)
J_2 = 1.082627E-03                          # WGS84 Earth's second gravitational constant
d2r = np.pi / 180.0                         # Degree to radian (rad/deg)
r2d = 180.0 / np.pi                         # Radian to degree
R_0 = 6.378137E06                           # WGS84 Equatorial radius in meters
OMEGA_ie = 7.292115146E-05                  # Earth rotation rate in (rad/s)
micro_g = 9.80665E-06                       # Micro g in (m/s^2)
dd2sec = 86400.0                            # seconds per day
hr2sec = 3600.00                            # seconds per hour
mn2sec = 60.0                               # seconds per minute
tol = 1.0E-08                               # tolerance in solving Kepler's equation

# End of Conversion Factor Declaration


'''
-----------------------
B. File Directory Paths
-----------------------
'''
global finpath, foutpath
finpath = os.path.join('SimData', 'SimIn') + os.path.sep            # input data file path
foutpath = os.path.join('SimData', 'SimOut') + os.path.sep          # output data file path

'''
========================================================================================================================
                                        FUNCTIONS TO CREATE GRAPHS OF RESULTS
========================================================================================================================
'''
'''
    -----------------------------
    1. Customize Color for Graphs
    -----------------------------
'''

clr.ColorConverter.colors['c1'] = (0.0, 0.4, 0.9)
clr.ColorConverter.colors['c2'] = (0.0, 0.9, 0.4)
clr.ColorConverter.colors['c3'] = (0.9, 0.0, 0.4)
clr.ColorConverter.colors['c4'] = (0.9, 0.4, 0.0)
clr.ColorConverter.colors['c5'] = (0.4, 0.9, 0.0)
clr.ColorConverter.colors['c6'] = (0.4, 0.0, 0.9)
clr.ColorConverter.colors['c7'] = (0.7, 0.5, 0.0)
clr.ColorConverter.colors['c8'] = (0.7, 0.0, 0.5)
clr.ColorConverter.colors['c9'] = (0.0, 0.7, 0.5)
clr.ColorConverter.colors['c10'] = (0.8, 0.2, 0.6)
clr.ColorConverter.colors['c11'] = (0.5, 0.0, 0.7)
clr.ColorConverter.colors['c12'] = (0.5, 0.7, 0.0)
clr.ColorConverter.colors['c13'] = (0.2, 0.6, 0.8)
clr.ColorConverter.colors['c14'] = (0.2, 0.4, 0.6)
clr.ColorConverter.colors['c15'] = (0.8, 0.6, 0.2)
clr.ColorConverter.colors['c16'] = (0.1, 0.9, 0.6)
clr.ColorConverter.colors['c17'] = (0.6, 0.8, 0.2)
clr.ColorConverter.colors['c18'] = (0.6, 0.2, 0.8)
clr.ColorConverter.colors['c19'] = (0.1, 0.2, 0.3)

# End of Color Customization


'''
    --------------------------------------
    2. Plot Results from Single Simulation
    --------------------------------------
'''


def plot_single_profile(true_profile, est_profile):

    # 1. Create the Vehicle Scatter Profile
    tin = true_profile[:, 0] / 60.0
    dat_len = len(tin)
    spacing = 30
    dat_point = dat_len/spacing
    scatter_track = np.nan * np.ones((dat_point, 2))
    indx = 0
    for i in xrange(0, (dat_len - np.mod(dat_len, spacing)), spacing):
        scatter_track[indx, :] = true_profile[i, 1:3]
        indx += 1
    # End of For loop to collect data points for scatter plot
    lat_arr = scatter_track[:, 0] * r2d
    lat_head = lat_arr[1:]
    lat_tail = lat_arr[:-1]
    lat_mag = lat_head - lat_tail
    lon_arr = scatter_track[:, 1] * r2d
    lon_head = lon_arr[1:]
    lon_tail = lon_arr[:-1]
    lon_mag = lon_head - lon_tail

    # 2. Plot Ground Track in Geodetic Frame
    plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(true_profile[:, 2] * r2d, true_profile[:, 1] * r2d, 'c1', label='$Flight Data$')
    plt.hold('on')
    plt.grid('on')
    plt.quiver(lon_tail, lat_tail, lon_mag, lat_mag, scale_units='xy', angles='xy', scale=1.5, color='c1')
    plt.plot(est_profile[:, 2] * r2d, est_profile[:, 1] * r2d, 'c3', linestyle='--', label='$Simulation$')
    plt.xlabel('$Longitude$' + ' ' + '$(deg)$', fontsize=16)
    plt.ylabel('$Latitude$' + ' ' + '$(deg)$', fontsize=16)
    plt.title('$Ground$' + ' ' + '$Track$' + ' ' + '$in$' + ' ' + '$Geodetic$', fontsize=18)
    plt.legend(loc='best', prop={'size': 12})

    # 3. Plot Position, Velocity, and Attitude in NED:
    tout = est_profile[:, 0] / 60.0

    # 3.1 For Position Displacements:
    [in_r_ns, in_r_ew] = radii_of_curv(true_profile[0, 1])
    [out_r_ns, out_r_ew] = radii_of_curv(est_profile[0, 1])
    pfig, p = plt.subplots(3, 3, sharex=True, figsize=(12, 8))

    # 3.1.a For North Position Displacement from Its Initial Value
    p[0, 0].plot(tin, (true_profile[:, 1] - true_profile[0, 1]) * (in_r_ns + true_profile[0, 3]), 'c19',
                 linestyle='--', label='$Orig$')
    p[0, 0].hold('on')
    p[0, 0].plot(tout, (est_profile[:, 1] - est_profile[0, 1]) * (out_r_ns + est_profile[0, 3]), 'c1',
                 label='$Sim$')
    p[0, 0].set_title('$North$' + ' ' + '$Displacement$')
    p[0, 0].set_ylabel('$(m)$')

    # 3.1.b. For East Position Displacement from Its Initial Value
    p[0, 1].plot(tin, (true_profile[:, 2] - true_profile[0, 2]) * (in_r_ns + true_profile[0, 3]) *
                 np.cos(true_profile[0, 1]), 'c19', linestyle='--', label='$Orig$')
    p[0, 1].hold('on')
    p[0, 1].plot(tout, (est_profile[:, 2] - est_profile[0, 2]) * (out_r_ns + est_profile[0, 3]) *
                 np.cos(est_profile[0, 1]), 'c2', label='$Sim$')
    p[0, 1].set_title('$East$' + ' ' + '$Displacement$')

    # 3.1.c. For Down Position Displacement (Altitude) from Its Initial Value
    p[0, 2].plot(tin, true_profile[:, 3], 'c19', linestyle='--', label='$Orig$')
    p[0, 2].hold('on')
    p[0, 2].plot(tout, est_profile[:, 3], 'c3', label='$Sim$')
    p[0, 2].set_title('$Altitude$')
    p[0, 2].legend(loc='best', prop={'size': 10})

    # 3.2 For Velocity:
    # 3.2.a For North Velocity
    p[1, 0].plot(tin, true_profile[:, 4], 'c19', linestyle='--', label='$Orig$')
    p[1, 0].hold('on')
    p[1, 0].plot(tout, est_profile[:, 4], 'c4', label='$Sim$')
    p[1, 0].set_title('$v_N$')
    p[1, 0].set_ylabel('$(m/s)$')

    # 3.2.b For East Velocity
    p[1, 1].plot(tin, true_profile[:, 5], 'c19', linestyle='--', label='$Orig$')
    p[1, 1].hold('on')
    p[1, 1].plot(tout, est_profile[:, 5], 'c5', label='$Sim$')
    p[1, 1].set_title('$v_E$')

    # 3.2.c For Down Velocity
    p[1, 2].plot(tin, true_profile[:, 6], 'c19', linestyle='--', label='$Orig$')
    p[1, 2].hold('on')
    p[1, 2].plot(tout, est_profile[:, 6], 'c6', label='$Sim$')
    p[1, 2].set_title('$v_E$')

    # 3.3 For Attitude:
    # 3.3.a For Roll Angle
    p[2, 0].plot(tin, r2d * true_profile[:, 7], 'c19', linestyle='--', label='$Orig$')
    p[2, 0].hold('on')
    p[2, 0].plot(tout, r2d * est_profile[:, 7], 'c7', label='$Sim$')
    p[2, 0].set_title('$Roll$'+' '+'$Angle$')
    p[2, 0].set_ylabel('$(deg)$')
    p[2, 0].set_xlabel('$t (mn)$')

    # 3.3.b For Pitch Angle
    p[2, 1].plot(tin, r2d * true_profile[:, 8], 'c19', linestyle='--', label='$Orig$')
    p[2, 1].hold('on')
    p[2, 1].plot(tout, r2d * est_profile[:, 8], 'c8', label='$Sim$')
    p[2, 1].set_title('$Pitch$'+' '+'$Angle$')
    p[2, 1].set_xlabel('$t (mn)$')

    # 3.3.c For Yaw Angle
    p[2, 2].plot(tin, r2d * true_profile[:, 9], 'c19', linestyle='--', label='$Orig$')
    p[2, 2].hold('on')
    p[2, 2].plot(tout, r2d * est_profile[:, 9], 'c9', label='$Sim$')
    p[2, 2].set_title('$Yaw$'+' '+'$Angle$')
    p[2, 2].set_xlabel('$t (mn)$')

    plt.tight_layout()

    return

# End of Plotting Profiles


'''
    ------------------------------------
    3. Plot Results from Dual Simulation
    ------------------------------------
'''


def plot_dual_profile(true_profile, lc_est_profile, tc_est_profile):

    # 1. Create the Vehicle Scatter Profile
    tin = true_profile[:, 0] / 60.0
    dat_len = len(tin)
    spacing = 30
    dat_point = dat_len/spacing
    scatter_track = np.nan * np.ones((dat_point, 2))
    indx = 0
    for i in xrange(0, (dat_len - np.mod(dat_len, spacing)), spacing):
        scatter_track[indx, :] = true_profile[i, 1:3]
        indx += 1
    # End of For loop to collect data points for scatter plot
    lat_arr = scatter_track[:, 0] * r2d
    lat_head = lat_arr[1:]
    lat_tail = lat_arr[:-1]
    lat_mag = lat_head - lat_tail
    lon_arr = scatter_track[:, 1] * r2d
    lon_head = lon_arr[1:]
    lon_tail = lon_arr[:-1]
    lon_mag = lon_head - lon_tail

    # 2. Plot Ground Track in Geodetic Frame
    plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(true_profile[:, 2] * r2d, true_profile[:, 1] * r2d, color='c1', label='$Flight Data$')
    plt.hold('on')
    plt.grid('on')
    plt.quiver(lon_tail, lat_tail, lon_mag, lat_mag, scale_units='xy', angles='xy', scale=1.5, color='c1')
    plt.plot(lc_est_profile[:, 2] * r2d, lc_est_profile[:, 1] * r2d, color='c2', linestyle='--', label='$LC Sim$')
    plt.plot(tc_est_profile[:, 2] * r2d, tc_est_profile[:, 1] * r2d, color='c3', linestyle='-.', label='$TC Sim$')
    plt.xlabel('$Longitude$' + ' ' + '$(deg)$', fontsize=16)
    plt.ylabel('$Latitude$' + ' ' + '$(deg)$', fontsize=16)
    plt.title('$Ground$' + ' ' + '$Track$' + ' ' + '$in$' + ' ' + '$Geodetic$', fontsize=18)
    plt.legend(loc='best', prop={'size': 12})

    # 3. Plot the Displacements, Speeds, and Attitudes
    lc_tout = lc_est_profile[:, 0] / 60.0
    tc_tout = tc_est_profile[:, 0] / 60.0

    # 3.1 For Position Displacements:
    [in_r_ns, in_r_ew] = radii_of_curv(true_profile[0, 1])
    [lc_out_r_ns, lc_out_r_ew] = radii_of_curv(lc_est_profile[0, 1])
    [tc_out_r_ns, tc_out_r_ew] = radii_of_curv(tc_est_profile[0, 1])
    pfig, p = plt.subplots(3, 3, sharex=True, figsize=(12, 8))

    # 3.1.a For North Position Displacement from Its Initial Value
    p[0, 0].plot(tin, ((true_profile[:, 1] - true_profile[0, 1]) * (in_r_ns + true_profile[0, 3])), 'c19',
                 label='$Orig$')
    p[0, 0].hold('on')
    p[0, 0].plot(lc_tout, ((lc_est_profile[:, 1] - lc_est_profile[0, 1]) * (lc_out_r_ns + lc_est_profile[0, 3])), 'c1',
                 linestyle='--', label='$LC Sim$')
    p[0, 0].plot(tc_tout, ((tc_est_profile[:, 1] - tc_est_profile[0, 1]) * (tc_out_r_ns + tc_est_profile[0, 3])), 'c10',
                 linestyle='-.', label='$TC Sim$')
    p[0, 0].set_title('$North$' + ' ' + '$Displacement$')
    p[0, 0].set_ylabel('$(m)$')
    p[0, 0].legend(loc='best', prop={'size': 10})

    # 3.1.b. For East Position Displacement from Its Initial Value
    p[0, 1].plot(tin, ((true_profile[:, 2] -
                        true_profile[0, 2]) * (in_r_ns + true_profile[0, 3]) * np.cos(true_profile[0, 1])), 'c19',
                 label='$Orig$')
    p[0, 1].hold('on')
    p[0, 1].plot(lc_tout,
                 ((lc_est_profile[:, 2] - lc_est_profile[0, 2]) * (lc_out_r_ns + lc_est_profile[0, 3]) * np.cos(
                     lc_est_profile[0, 1])), 'c2', linestyle='--', label='$LC Sim$')
    p[0, 1].plot(tc_tout,
                 ((tc_est_profile[:, 2] - tc_est_profile[0, 2]) * (tc_out_r_ns + tc_est_profile[0, 3]) * np.cos(
                     tc_est_profile[0, 1])), 'c11', linestyle='-.', label='$TC Sim$')
    p[0, 1].set_title('$East$' + ' ' + '$Displacement$')
    p[0, 1].legend(loc='best', prop={'size': 10})

    # 3.1.c. For Down Position Displacement (Altitude) from Its Initial Value
    p[0, 2].plot(tin, true_profile[:, 3], 'c19', label='$Orig$')
    p[0, 2].hold('on')
    p[0, 2].plot(lc_tout, lc_est_profile[:, 3], 'c3', linestyle='--', label='$LC Sim$')
    p[0, 2].plot(tc_tout, tc_est_profile[:, 3], 'c12', linestyle='-.', label='$TC Sim$')
    p[0, 2].set_title('$Altitude$')
    p[0, 2].legend(loc='best', prop={'size': 10})

    # 3.2 For Velocity:
    # 3.2.a For North Velocity
    p[1, 0].plot(tin, true_profile[:, 4], 'c19', label='$Orig$')
    p[1, 0].hold('on')
    p[1, 0].plot(lc_tout, lc_est_profile[:, 4], 'c4', linestyle='--', label='$LC Sim$')
    p[1, 0].plot(tc_tout, tc_est_profile[:, 4], 'c13', linestyle='-.', label='$TC Sim$')
    p[1, 0].set_title('$v_N$')
    p[1, 0].set_ylabel('$(m/s)$')
    p[1, 0].legend(loc='best', prop={'size': 10})

    # 3.2.b For East Velocity
    p[1, 1].plot(tin, true_profile[:, 5], 'c19', label='$Orig$')
    p[1, 1].hold('on')
    p[1, 1].plot(lc_tout, lc_est_profile[:, 5], 'c5', linestyle='--', label='$LC Sim$')
    p[1, 1].plot(tc_tout, tc_est_profile[:, 5], 'c14', linestyle='-.', label='$TC Sim$')
    p[1, 1].set_title('$v_E$')
    p[1, 1].legend(loc='best', prop={'size': 10})

    # 3.2.c For Down Velocity
    p[1, 2].plot(tin, true_profile[:, 6], 'c19', label='$Orig$')
    p[1, 2].hold('on')
    p[1, 2].plot(lc_tout, lc_est_profile[:, 6], 'c6', linestyle='--', label='$LC Sim$')
    p[1, 2].plot(tc_tout, tc_est_profile[:, 6], 'c15', linestyle='-.', label='$TC Sim$')
    p[1, 2].set_title('$v_E$')
    p[1, 2].legend(loc='best', prop={'size': 10})

    # 3.3 For Attitude:
    # 3.3.a For Roll Angle
    p[2, 0].plot(tin, r2d * true_profile[:, 7], 'c19', label='$Orig$')
    p[2, 0].hold('on')
    p[2, 0].plot(lc_tout, r2d * lc_est_profile[:, 7], 'c7', linestyle='--', label='$LC Sim$')
    p[2, 0].plot(tc_tout, r2d * tc_est_profile[:, 7], 'c16', linestyle='-.', label='$TC Sim$')
    p[2, 0].set_title('$Roll$'+' '+'$Angle$')
    p[2, 0].set_ylabel('$(deg)$')
    p[2, 0].set_xlabel('$t (mn)$')
    p[2, 0].legend(loc='best', prop={'size': 10})

    # 3.3.b For Pitch Angle
    p[2, 1].plot(tin, r2d * true_profile[:, 8], 'c19', label='$Orig$')
    p[2, 1].hold('on')
    p[2, 1].plot(lc_tout, r2d * lc_est_profile[:, 8], 'c8', linestyle='--', label='$LC Sim$')
    p[2, 1].plot(tc_tout, r2d * tc_est_profile[:, 8], 'c17', linestyle='-.', label='$TC Sim$')
    p[2, 1].set_title('$Pitch$'+' '+'$Angle$')
    p[2, 1].set_xlabel('$t (mn)$')
    p[2, 1].legend(loc='best', prop={'size': 10})

    # 3.3.c For Yaw Angle
    p[2, 2].plot(tin, r2d * true_profile[:, 9], 'c19', label='$Orig$')
    p[2, 2].hold('on')
    p[2, 2].plot(lc_tout, r2d * lc_est_profile[:, 9], 'c9', linestyle='--', label='$LC Sim$')
    p[2, 2].plot(tc_tout, r2d * tc_est_profile[:, 9], 'c18', linestyle='-.', label='$TC Sim$')
    p[2, 2].set_title('$Yaw$'+' '+'$Angle$')
    p[2, 2].set_xlabel('$t (mn)$')
    p[2, 2].legend(loc='best', prop={'size': 10})

    plt.tight_layout()

    return

# End of plotting dual profiles


'''
    ------------------------------------
    4. Plot Errors for Single Simulation
    ------------------------------------
'''


def plot_single_error(errors, kf_sd):

    nsig = 3.0                      # Number of sigmas (standard deviation)
    tkf = kf_sd[:, 0] / 60.0        # Kalman Filter updating time history
    ter = errors[:, 0] / 60.0       # Output errors updating time history

    perfig, per = plt.subplots(3, 3, sharex=True, figsize=(12, 8))

    # 1. For Position Errors:
    [r_ns, r_ew] = radii_of_curv(kf_sd[0, 1])
    # 1.1. For North Position Error
    per[0, 0].plot(ter, errors[:, 1], 'c1', label=r'$\delta$' + '$r_N$')
    per[0, 0].hold('on')
    per[0, 0].plot(tkf, nsig * kf_sd[:, 1], 'k', label=r'$3\sigma_N$')
    per[0, 0].plot(tkf, -nsig * kf_sd[:, 1], 'k')
    per[0, 0].set_title(r'$\delta$' + '$r_N$')
    per[0, 0].set_ylabel('$(m)$')

    # 1.2. For East Position Error
    per[0, 1].plot(ter, errors[:, 2], 'c2', label=r'$\delta$' + '$r_E$')
    per[0, 1].hold('on')
    per[0, 1].plot(tkf, nsig * kf_sd[:, 2], 'k', label=r'$3\sigma_E$')
    per[0, 1].plot(tkf, -nsig * kf_sd[:, 2], 'k')
    per[0, 1].set_title(r'$\delta$' + '$r_E$')

    # 1.3. For Down Position Error
    per[0, 2].plot(ter, errors[:, 3], 'c3', label=r'$\delta$' + '$r_D$')
    per[0, 2].hold('on')
    per[0, 2].plot(tkf, nsig * kf_sd[:, 3], 'k', label=r'$3\sigma_D$')
    per[0, 2].plot(tkf, -nsig * kf_sd[:, 3], 'k')
    per[0, 2].set_title(r'$\delta$' + '$r_D$')
    per[0, 2].legend(loc='best', prop={'size': 10})

    # 2. For Velocity Errors:
    # 2.1. For North Velocity Error
    per[1, 0].plot(ter, errors[:, 4], 'c4', label=r'$\delta$' + '$v_N$')
    per[1, 0].hold('on')
    per[1, 0].plot(tkf, nsig * kf_sd[:, 4], 'k', label=r'$3\sigma_{v_N}$')
    per[1, 0].plot(tkf, -nsig * kf_sd[:, 4], 'k')
    per[1, 0].set_title(r'$\delta$' + '$v_N$')
    per[1, 0].set_ylabel('$(m/s)$')

    # 2.2. For East Velocity Error
    per[1, 1].plot(ter, errors[:, 5], 'c5', label=r'$\delta$' + '$v_E$')
    per[1, 1].hold('on')
    per[1, 1].plot(tkf, nsig * kf_sd[:, 5], 'k', label=r'$3\sigma_{v_E}$')
    per[1, 1].plot(tkf, -nsig * kf_sd[:, 5], 'k')
    per[1, 1].set_title(r'$\delta$' + '$v_E$')

    # 2.3. For Down Velocity Error
    per[1, 2].plot(ter, errors[:, 6], 'c6', label=r'$\delta$' + '$v_D$')
    per[1, 2].hold('on')
    per[1, 2].plot(tkf, nsig * kf_sd[:, 6], 'k', label=r'$3\sigma_{v_D}$')
    per[1, 2].plot(tkf, -nsig * kf_sd[:, 6], 'k')
    per[1, 2].set_title(r'$\delta$' + '$v_D$')

    # 3. For Attitude Errors:
    # 3.1. For Roll Angle Error
    per[2, 0].plot(ter, r2d * errors[:, 7], 'c7', label=r'$\delta_\phi$')
    per[2, 0].hold('on')
    per[2, 0].plot(tkf, nsig * r2d * kf_sd[:, 7], 'k', label=r'$3\sigma_\phi$')
    per[2, 0].plot(tkf, -nsig * r2d * kf_sd[:, 7], 'k')
    per[2, 0].set_title(r'$\delta\phi$')
    per[2, 0].set_ylabel('$(deg)$')
    per[2, 0].set_xlabel('$t (mn)$')

    # 3.2. For Pitch Angle Error
    per[2, 1].plot(ter, r2d * errors[:, 8], 'c8', label=r'$\delta_\theta$')
    per[2, 1].hold('on')
    per[2, 1].plot(tkf, nsig * r2d * kf_sd[:, 8], 'k', label=r'$3\sigma_\theta$')
    per[2, 1].plot(tkf, -nsig * r2d * kf_sd[:, 8], 'k')
    per[2, 1].set_title(r'$\delta\theta$')
    per[2, 1].set_xlabel('$t (mn)$')

    # 3.3. For Yaw Angle Error
    per[2, 2].plot(ter, r2d * errors[:, 9], 'c9', label=r'$\delta_\psi$')
    per[2, 2].hold('on')
    per[2, 2].plot(tkf, nsig * r2d * kf_sd[:, 9], 'k', label=r'$3\sigma_\psi$')
    per[2, 2].plot(tkf, -nsig * r2d * kf_sd[:, 9], 'k')
    per[2, 2].set_title(r'$\delta\psi$')
    per[2, 2].set_xlabel('$t (mn)$')

    # perfig.suptitle("Estimation Error Profile over Time")

    plt.tight_layout()

    return

# End of Plotting Errors for Single Simulation


'''
    ----------------------------------
    5. Plot Errors for Dual Simulation
    ----------------------------------
'''


def plot_dual_error(lc_errors, lc_kf_sd, tc_errors, tc_kf_sd):

    nsig = 3.0                         # Number of sigma (standard deviation)
    tkf = lc_kf_sd[:, 0] / 60.0        # Kalman Filter updating time history
    ter = lc_errors[:, 0] / 60.0       # Output errors updating time history

    # A. Loosely Coupled Errors
    lc_perfig, lc_per = plt.subplots(3, 3, sharex=True, figsize=(12, 8))

    # 1. For Position Errors:
    [lc_r_ns, lc_r_ew] = radii_of_curv(lc_kf_sd[0, 1])
    # 1.1. For North Position Error
    lc_per[0, 0].plot(ter, lc_errors[:, 1], 'c1', label=r'$\delta$' + '$r_N$')
    lc_per[0, 0].hold('on')
    lc_per[0, 0].plot(tkf, nsig * lc_kf_sd[:, 1], 'k', label=r'$3\sigma_N$')
    lc_per[0, 0].plot(tkf, -nsig * lc_kf_sd[:, 1], 'k')
    lc_per[0, 0].set_title('$LC$' + ' ' + r'$\delta$' + '$r_N$')
    lc_per[0, 0].set_ylabel('$(m)$')

    # 1.2. For East Position Error
    lc_per[0, 1].plot(ter, lc_errors[:, 2], 'c2', label=r'$\delta$' + '$r_E$')
    lc_per[0, 1].hold('on')
    lc_per[0, 1].plot(tkf, nsig * lc_kf_sd[:, 2], 'k', label=r'$3\sigma_E$')
    lc_per[0, 1].plot(tkf, -nsig * lc_kf_sd[:, 2], 'k')
    lc_per[0, 1].set_title('$LC$' + ' ' + r'$\delta$' + '$r_E$')

    # 1.3. For Down Position Error
    lc_per[0, 2].plot(ter, lc_errors[:, 3], 'c3', label=r'$\delta$' + '$r_D$')
    lc_per[0, 2].hold('on')
    lc_per[0, 2].plot(tkf, nsig * lc_kf_sd[:, 3], 'k', label=r'$3\sigma_D$')
    lc_per[0, 2].plot(tkf, -nsig * lc_kf_sd[:, 3], 'k')
    lc_per[0, 2].set_title('$LC$' + ' ' + r'$\delta$' + '$r_D$')
    lc_per[0, 2].legend(loc='best', prop={'size': 10})

    # 2. For Velocity Errors:
    # 2.1. For North Velocity Error
    lc_per[1, 0].plot(ter, lc_errors[:, 4], 'c4', label=r'$\delta$' + '$v_N$')
    lc_per[1, 0].hold('on')
    lc_per[1, 0].plot(tkf, nsig * lc_kf_sd[:, 4], 'k', label=r'$3\sigma_{v_N}$')
    lc_per[1, 0].plot(tkf, -nsig * lc_kf_sd[:, 4], 'k')
    lc_per[1, 0].set_title('$LC$' + ' ' + r'$\delta$' + '$v_N$')
    lc_per[1, 0].set_ylabel('$(m/s)$')

    # 2.2. For East Velocity Error
    lc_per[1, 1].plot(ter, lc_errors[:, 5], 'c5', label=r'$\delta$' + '$v_E$')
    lc_per[1, 1].hold('on')
    lc_per[1, 1].plot(tkf, nsig * lc_kf_sd[:, 5], 'k', label=r'$3\sigma_{v_E}$')
    lc_per[1, 1].plot(tkf, -nsig * lc_kf_sd[:, 5], 'k')
    lc_per[1, 1].set_title('$LC$' + ' ' + r'$\delta$' + '$v_E$')

    # 3.3. For Down Velocity Error
    lc_per[1, 2].plot(ter, lc_errors[:, 6], 'c6', label=r'$\delta$' + '$v_D$')
    lc_per[1, 2].hold('on')
    lc_per[1, 2].plot(tkf, nsig * lc_kf_sd[:, 6], 'k', label=r'$3\sigma_{v_D}$')
    lc_per[1, 2].plot(tkf, -nsig * lc_kf_sd[:, 6], 'k')
    lc_per[1, 2].set_title('$LC$' + ' ' + r'$\delta$' + '$v_D$')

    # 3. For Attitude Errors:
    # 3.1. For Roll Angle Error
    lc_per[2, 0].plot(ter, r2d * lc_errors[:, 7], 'c7', label=r'$\delta_\phi$')
    lc_per[2, 0].hold('on')
    lc_per[2, 0].plot(tkf, nsig * r2d * lc_kf_sd[:, 7], 'k', label=r'$3\sigma_\phi$')
    lc_per[2, 0].plot(tkf, -nsig * r2d * lc_kf_sd[:, 7], 'k')
    lc_per[2, 0].set_title('$LC$' + ' ' + r'$\delta\phi$')
    lc_per[2, 0].set_ylabel('$(deg)$')
    lc_per[2, 0].set_xlabel('$t (mn)$')

    # 3.2. For Pitch Angle Error
    lc_per[2, 1].plot(ter, r2d * lc_errors[:, 8], 'c8', label=r'$\delta_\theta$')
    lc_per[2, 1].hold('on')
    lc_per[2, 1].plot(tkf, nsig * r2d * lc_kf_sd[:, 8], 'k', label=r'$3\sigma_\theta$')
    lc_per[2, 1].plot(tkf, -nsig * r2d * lc_kf_sd[:, 8], 'k')
    lc_per[2, 1].set_title('$LC$' + ' ' + r'$\delta\theta$')
    lc_per[2, 1].set_xlabel('$t (mn)$')

    # 3.3. For Yaw Angle Error
    lc_per[2, 2].plot(ter, r2d * lc_errors[:, 9], 'c9', label=r'$\delta_\psi$')
    lc_per[2, 2].hold('on')
    lc_per[2, 2].plot(tkf, nsig * r2d * lc_kf_sd[:, 9], 'k', label=r'$3\sigma_\psi$')
    lc_per[2, 2].plot(tkf, -nsig * r2d * lc_kf_sd[:, 9], 'k')
    lc_per[2, 2].set_title('$LC$' + ' ' + r'$\delta\psi$')
    lc_per[2, 2].set_xlabel('$t (mn)$')

    # lc_perfig.suptitle("Estimation Error Profile over Time")

    plt.tight_layout()

    # B. Tightly Coupled Errors
    tc_perfig, tc_per = plt.subplots(3, 3, sharex=True, figsize=(12, 8))

    # 1. For Position Errors:
    [tc_r_ns, tc_r_ew] = radii_of_curv(tc_kf_sd[0, 1])
    # 1.1. For North Position Error
    tc_per[0, 0].plot(ter, tc_errors[:, 1], 'c10', label=r'$\delta$' + '$r_N$')
    tc_per[0, 0].hold('on')
    tc_per[0, 0].plot(tkf, nsig * tc_kf_sd[:, 1], 'k', label=r'$3\sigma_N$')
    tc_per[0, 0].plot(tkf, -nsig * tc_kf_sd[:, 1], 'k')
    tc_per[0, 0].set_title('$TC$' + ' ' + r'$\delta$' + '$r_N$')
    tc_per[0, 0].set_ylabel('$(m)$')

    # 1.2. For East Position Error
    tc_per[0, 1].plot(ter, tc_errors[:, 2], 'c11', label=r'$\delta$' + '$r_E$')
    tc_per[0, 1].hold('on')
    tc_per[0, 1].plot(tkf, nsig * tc_kf_sd[:, 2], 'k', label=r'$3\sigma_E$')
    tc_per[0, 1].plot(tkf, -nsig * tc_kf_sd[:, 2], 'k')
    tc_per[0, 1].set_title('$TC$' + ' ' + r'$\delta$' + '$r_E$')

    # 1.3. For Down Position Error
    tc_per[0, 2].plot(ter, tc_errors[:, 3], 'c12', label=r'$\delta$' + '$r_D$')
    tc_per[0, 2].hold('on')
    tc_per[0, 2].plot(tkf, nsig * tc_kf_sd[:, 3], 'k', label=r'$3\sigma_D$')
    tc_per[0, 2].plot(tkf, -nsig * tc_kf_sd[:, 3], 'k')
    tc_per[0, 2].set_title('$TC$' + ' ' + r'$\delta$' + '$r_D$')
    tc_per[0, 2].legend(loc='best', prop={'size': 10})

    # 2. For Velocity Errors:
    # 2.1. For North Velocity Error
    tc_per[1, 0].plot(ter, tc_errors[:, 4], 'c13', label=r'$\delta$' + '$v_N$')
    tc_per[1, 0].hold('on')
    tc_per[1, 0].plot(tkf, nsig * tc_kf_sd[:, 4], 'k', label=r'$3\sigma_{v_N}$')
    tc_per[1, 0].plot(tkf, -nsig * tc_kf_sd[:, 4], 'k')
    tc_per[1, 0].set_title('$TC$' + ' ' + r'$\delta$' + '$v_N$')
    tc_per[1, 0].set_ylabel('$(m/s)$')

    # 2.2. For East Velocity Error
    tc_per[1, 1].plot(ter, tc_errors[:, 5], 'c14', label=r'$\delta$' + '$v_E$')
    tc_per[1, 1].hold('on')
    tc_per[1, 1].plot(tkf, nsig * tc_kf_sd[:, 5], 'k', label=r'$3\sigma_{v_E}$')
    tc_per[1, 1].plot(tkf, -nsig * tc_kf_sd[:, 5], 'k')
    tc_per[1, 1].set_title('$TC$' + ' ' + r'$\delta$' + '$v_E$')

    # 2.3. For Down Velocity Error
    tc_per[1, 2].plot(ter, tc_errors[:, 6], 'c15', label=r'$\delta$' + '$v_D$')
    tc_per[1, 2].hold('on')
    tc_per[1, 2].plot(tkf, nsig * tc_kf_sd[:, 6], 'k', label=r'$3\sigma_{v_D}$')
    tc_per[1, 2].plot(tkf, -nsig * tc_kf_sd[:, 6], 'k')
    tc_per[1, 2].set_title('$TC$' + ' ' + r'$\delta$' + '$v_D$')

    # 3. For Attitude Errors:
    # 3.1. For Roll Angle Error
    tc_per[2, 0].plot(ter, r2d * tc_errors[:, 7], 'c16', label=r'$\delta_\phi$')
    tc_per[2, 0].hold('on')
    tc_per[2, 0].plot(tkf, nsig * r2d * tc_kf_sd[:, 7], 'k', label=r'$3\sigma_\phi$')
    tc_per[2, 0].plot(tkf, -nsig * r2d * tc_kf_sd[:, 7], 'k')
    tc_per[2, 0].set_title('$TC$' + ' ' + r'$\delta\phi$')
    tc_per[2, 0].set_ylabel('$(deg)$')
    tc_per[2, 0].set_xlabel('$t (mn)$')

    # 3.2. For Pitch Angle Error
    tc_per[2, 1].plot(ter, r2d * tc_errors[:, 8], 'c17', label=r'$\delta_\theta$')
    tc_per[2, 1].hold('on')
    tc_per[2, 1].plot(tkf, nsig * r2d * tc_kf_sd[:, 7], 'k', label=r'$3\sigma_\theta$')
    tc_per[2, 1].plot(tkf, -nsig * r2d * tc_kf_sd[:, 7], 'k')
    tc_per[2, 1].set_title('$TC$' + ' ' + r'$\delta\theta$')
    tc_per[2, 1].set_xlabel('$t (mn)$')

    # 3.3. For Yaw Angle Error
    tc_per[2, 2].plot(ter, r2d * tc_errors[:, 9], 'c18', label=r'$\delta_\psi$')
    tc_per[2, 2].hold('on')
    tc_per[2, 2].plot(tkf, nsig * r2d * tc_kf_sd[:, 9], 'k', label=r'$3\sigma_\psi$')
    tc_per[2, 2].plot(tkf, -nsig * r2d * tc_kf_sd[:, 9], 'k')
    tc_per[2, 2].set_title('$TC$' + ' ' + r'$\delta\psi$')
    tc_per[2, 2].set_xlabel('$t (mn)$')

    # tc_perfig.suptitle("Estimation Error Profile over Time")

    plt.tight_layout()

    # C. Loosely and Tightly Coupled Errors Together
    perfig, per = plt.subplots(3, 3, sharex=True, figsize=(12, 8))

    # 1. For Position Errors:
    # 1.1. For North Position Error
    per[0, 0].plot(ter, lc_errors[:, 1], 'c1', linestyle='--', label='$LC$')
    per[0, 0].hold('on')
    per[0, 0].plot(ter, tc_errors[:, 1], 'c10', linestyle='-.', label='$TC$')
    per[0, 0].set_title(r'$\delta$' + '$r_N$')
    per[0, 0].set_ylabel('$(m)$')
    per[0, 0].legend(loc='best', prop={'size': 10})

    # 1.2. For East Position Error
    per[0, 1].plot(ter, lc_errors[:, 2], 'c2', linestyle='--', label='$LC$')
    per[0, 1].hold('on')
    per[0, 1].plot(ter, tc_errors[:, 2], 'c11', linestyle='-.', label='$TC$')
    per[0, 1].set_title(r'$\delta$' + '$r_E$')
    per[0, 1].legend(loc='best', prop={'size': 10})

    # 1.3. For Down Position Error
    per[0, 2].plot(ter, lc_errors[:, 3], 'c3', linestyle='--', label='$LC$')
    per[0, 2].hold('on')
    per[0, 2].plot(ter, tc_errors[:, 3], 'c12', linestyle='-.', label='$TC$')
    per[0, 2].set_title(r'$\delta$' + '$r_D$')
    per[0, 2].legend(loc='best', prop={'size': 10})

    # 2. For Velocity Errors:
    # 2.1. For North Velocity Error
    per[1, 0].plot(ter, lc_errors[:, 4], 'c4', linestyle='--', label='$LC$')
    per[1, 0].hold('on')
    per[1, 0].plot(ter, tc_errors[:, 4], 'c13', linestyle='-.', label='$TC$')
    per[1, 0].set_title(r'$\delta$' + '$v_N$')
    per[1, 0].set_ylabel('$(m/s)$')
    per[1, 0].legend(loc='best', prop={'size': 10})

    # 2.2. For East Velocity Error
    per[1, 1].plot(ter, lc_errors[:, 5], 'c5', linestyle='--', label='$LC$')
    per[1, 1].hold('on')
    per[1, 1].plot(ter, tc_errors[:, 5], 'c14', linestyle='-.', label='$TC$')
    per[1, 1].set_title(r'$\delta$' + '$v_E$')
    per[1, 1].legend(loc='best', prop={'size': 10})

    # 2.3. For Down Velocity Error
    per[1, 2].plot(ter, lc_errors[:, 6], 'c6', linestyle='--', label='$LC$')
    per[1, 2].hold('on')
    per[1, 2].plot(ter, tc_errors[:, 6], 'c15', linestyle='-.', label='$TC$')
    per[1, 2].set_title(r'$\delta$' + '$v_D$')
    per[1, 2].legend(loc='best', prop={'size': 10})

    # 3. For Attitude Errors:
    # 3.1. For Roll Angle Error
    per[2, 0].plot(ter, r2d * lc_errors[:, 7], 'c7', linestyle='--', label='$LC$')
    per[2, 0].hold('on')
    per[2, 0].plot(ter, r2d * tc_errors[:, 7], 'c16', linestyle='-.', label='$TC$')
    per[2, 0].set_title(r'$\delta\phi$')
    per[2, 0].set_ylabel('$(deg)$')
    per[2, 0].set_xlabel('$t (mn)$')
    per[2, 0].legend(loc='best', prop={'size': 10})

    # 3.2. For Pitch Angle Error
    per[2, 1].plot(ter, r2d * lc_errors[:, 8], 'c8', linestyle='--', label='$LC$')
    per[2, 1].hold('on')
    per[2, 1].plot(ter, r2d * tc_errors[:, 8], 'c17', linestyle='-.', label='$TC$')
    per[2, 1].set_title(r'$\delta\theta$')
    per[2, 1].set_xlabel('$t (mn)$')
    per[2, 1].legend(loc='best', prop={'size': 10})

    # 3.3. For Yaw Angle Error
    per[2, 2].plot(ter, r2d * lc_errors[:, 9], 'c9', linestyle='--', label='$LC$')
    per[2, 2].hold('on')
    per[2, 2].plot(ter, r2d * tc_errors[:, 9], 'c18', linestyle='-.', label='$TC$')
    per[2, 2].set_title(r'$\delta\psi$')
    per[2, 2].set_xlabel('$t (mn)$')
    per[2, 2].legend(loc='best', prop={'size': 10})

    # perfig.suptitle("Estimation Error Profile over Time")

    plt.tight_layout()

    return

# End of Plotting Errors for Dual Simulation


'''
========================================================================================================================
                                                PROCESSING RAW DATA
========================================================================================================================
'''
'''
    -----------------------------------------------------------------------------------
    B. Load the raw data into workspace and process raw data into usable formatted data
    -----------------------------------------------------------------------------------
'''


def data_processing(fpath, fname):
    print 'Processing Flight Data...'
    global datlen, no_epoch
    no_epoch = 0

    # Assemble full file location
    in_file = fpath + fname

    # Load the raw data
    raw_data = sio.loadmat(in_file, struct_as_record=False, squeeze_me=True)

    # Check whether the raw data was unpacked or not
    if not 'flight_data' in raw_data:
        # If yes, process the raw data and organize the data into the profile.
        datlen = len(raw_data['time'])              # Determine the length of raw data
        gps_lock = raw_data['navValid'] == 0        # Determine the GPS lock moments
        # Determine the number of true epochs (in which GPS was locked)
        for i in xrange(0, datlen):
            if gps_lock[i]:
                no_epoch += 1
        # Declare the profile variable
        flightdata = np.nan * np.ones((no_epoch, 30))
        k = 0
        for i in xrange(0, datlen):
            if gps_lock[i]:
                # Time is seconds
                flightdata[k, 0] = raw_data['time'][i]

                # Navigation from EKF_15_State Solution
                # Latitude and longitude are in radians
                flightdata[k, 1] = raw_data['navlat'][i]              # latitude
                flightdata[k, 2] = raw_data['navlon'][i]              # longitude
                # Altitude is in meters
                flightdata[k, 3] = raw_data['navalt'][i]              # altitude

                # Velocity components are in m/s
                flightdata[k, 4] = raw_data['navvn'][i]               # North velocity
                flightdata[k, 5] = raw_data['navve'][i]               # East velocity
                flightdata[k, 6] = raw_data['navvd'][i]               # Down velocity

                # Attitude angles are in radians
                flightdata[k, 7] = raw_data['phi'][i]                 # roll angle
                if 'theta' in raw_data:
                    # If the raw data has "theta" for pitch
                    flightdata[k, 8] = raw_data['theta'][i]           # pitch angle
                else:
                    # Else the raw data has "the" for pitch
                    flightdata[k, 8] = raw_data['the'][i]
                flightdata[k, 9] = raw_data['psi'][i]                 # yaw angle

                # Gyroscope outputs (rad/s)
                flightdata[k, 10] = raw_data['p'][i]                  # roll rate
                flightdata[k, 11] = raw_data['q'][i]                  # pitch rate
                flightdata[k, 12] = raw_data['r'][i]                  # yaw rate

                # Gyro biases are in rad/s
                flightdata[k, 13] = raw_data['p_bias'][i]             # roll rate bias
                flightdata[k, 14] = raw_data['q_bias'][i]             # pitch rate bias
                flightdata[k, 15] = raw_data['r_bias'][i]             # yaw rate bias

                # Acceleronmeter outputs (m/s^2)
                flightdata[k, 16] = raw_data['ax'][i]                 # specific force in the x-axis
                flightdata[k, 17] = raw_data['ay'][i]                 # specific force in the y-axis
                flightdata[k, 18] = raw_data['az'][i]                 # specific force in the z-axis

                # Acceleronmeter biases are in m/s^2
                flightdata[k, 19] = raw_data['ax_bias'][i]            # x-acceleration bias
                flightdata[k, 20] = raw_data['ay_bias'][i]            # y-acceleration bias
                flightdata[k, 21] = raw_data['az_bias'][i]            # z-acceleration bias

                # Navigation Information from GPS
                # Latitude and longitude are in degree
                flightdata[k, 22] = raw_data['lat'][i]
                flightdata[k, 23] = raw_data['lon'][i]

                # Altitude is in meter
                flightdata[k, 24] = raw_data['alt'][i]

                # Velocity components are in m/s
                flightdata[k, 25] = raw_data['vn'][i]
                flightdata[k, 26] = raw_data['ve'][i]
                flightdata[k, 27] = raw_data['vd'][i]

                # Number of satellites used in the GPS solution
                flightdata[k, 28] = raw_data['satVisible'][i]

                # GPS TOW (GPS Time of the Week in seconds)
                flightdata[k, 29] = raw_data['GPS_TOW'][i]

                k += 1
            # End of If Statement on GPS Lock
        # End of For Loop Sweeping Through the Data
    else:
        # If not, pull the data out of the "flight_data" struct element.
        unpackdata = raw_data['flight_data']
        datlen = len(unpackdata.time)  # Determine the length of raw data
        gps_lock = unpackdata.navValid == 0  # Determine the GPS lock moments

        # Determine the number of true epochs (in which GPS was locked)
        for i in xrange(0, datlen):

            if gps_lock[i]:
                no_epoch += 1

        # Declare the profile variable
        flightdata = np.nan * np.ones((no_epoch, 30))
        k = 0
        for i in xrange(0, datlen):

            if gps_lock[i]:

                # Time is seconds
                flightdata[k, 0] = unpackdata.time[i]

                # Navigation from EKF_15_State Solution
                # Latitude and longitude are in rad
                flightdata[k, 1] = unpackdata.navlat[i]         # latitude
                flightdata[k, 2] = unpackdata.navlon[i]         # longitude
                # Altitude is in meter
                flightdata[k, 3] = unpackdata.navalt[i]         # altitude

                # Velocity components are in m/s
                flightdata[k, 4] = unpackdata.navvn[i]          # North velocity
                flightdata[k, 5] = unpackdata.navve[i]          # East velocity
                flightdata[k, 6] = unpackdata.navvd[i]          # Down velocity

                # Attitude angles are in radian
                flightdata[k, 7] = unpackdata.phi[i]            # roll angle
                # If the raw data has "theta" for pitch
                if hasattr(unpackdata, 'theta'):
                    flightdata[k, 8] = unpackdata.theta[i]      # pitch angle
                # Else the raw data has "the" for pitch
                else:
                    flightdata[k, 8] = unpackdata.the[i]
                flightdata[k, 9] = unpackdata.psi[i]            # yaw angle

                # Gyroscope outputs (rad/s)
                flightdata[k, 10] = unpackdata.p[i]           # roll rate
                flightdata[k, 11] = unpackdata.q[i]           # pitch rate
                flightdata[k, 12] = unpackdata.r[i]           # yaw rate

                # Gyroscope biases are in rad/s
                flightdata[k, 13] = unpackdata.p_bias[i]      # roll rate bias
                flightdata[k, 14] = unpackdata.q_bias[i]      # pitch rate bias
                flightdata[k, 15] = unpackdata.r_bias[i]      # yaw rate bias

                # Acceleronmeter outputs (m/s^2)
                flightdata[k, 16] = unpackdata.ax[i]          # specific force in the x-axis
                flightdata[k, 17] = unpackdata.ay[i]          # specific force in the y-axis
                flightdata[k, 18] = unpackdata.az[i]          # specific force in the z-axis

                # Acceleronmeter biases are in m/s^2
                if hasattr(unpackdata, 'ax_bias'):
                    flightdata[k, 19] = unpackdata.ax_bias[i]      # x-acceleration bias
                else:
                    flightdata[k, 19] = unpackdata.ax_bias_nav[i]  # x-acceleration bias
                if hasattr(unpackdata, 'ay_bias'):
                    flightdata[k, 20] = unpackdata.ay_bias[i]      # y-acceleration bias
                else:
                    flightdata[k, 20] = unpackdata.ay_bias_nav[i]  # y-acceleration bias
                if hasattr(unpackdata, 'az_bias'):
                    flightdata[k, 21] = unpackdata.az_bias[i]      # z-acceleration bias
                else:
                    flightdata[k, 21] = unpackdata.az_bias_nav[i]  # z-acceleration bias

                # Navigation Information from GPS
                # Latitude and longitude are in degree
                flightdata[k, 22] = unpackdata.lat[i]
                flightdata[k, 23] = unpackdata.lon[i]

                # Altitude is in meter
                flightdata[k, 24] = unpackdata.alt[i]

                # Velocity components are in m/s
                flightdata[k, 25] = unpackdata.vn[i]
                flightdata[k, 26] = unpackdata.ve[i]
                flightdata[k, 27] = unpackdata.vd[i]

                # Number of satellites used in the GPS solution
                flightdata[k, 28] = unpackdata.satVisible[i]

                # GPS TOW (GPS Time of the Week in seconds)
                flightdata[k, 29] = unpackdata.GPS_TOW[i]

                k += 1
            # End of If Statement on GPS Lock
        # End of For Loop Sweeping Through the Data
    # ------------------------------------------------------------------------------------
    # End of If Statement on Checking for Data Status

    # Off set the time so that the first GPS lock moment is the initial time, t_o = 0 s.
    flightdata[:, 0] = flightdata[:, 0] - flightdata[0, 0]
    flightdata = np.matrix(flightdata)

    return flightdata, no_epoch

# End of Data Processing


'''
    ------------------------------------------------------------------------------------
    C. Load the raw navigation and observation messages into workspace and process them
    ------------------------------------------------------------------------------------
'''


def ephem_processing(fpath, fname_n, TOW, DyOM):

    print 'Processing the Navigation Message...'

    # Assemble full file location and open the file
    nav_message = open(fpath+fname_n, 'r')

    # Place holders
    raw_ephem = np.nan*np.ones((1, 35))
    iono_alpha = np.nan*np.ones((1, 4))
    iono_beta = np.nan*np.ones((1, 4))
    alma_t_para = np.nan*np.ones((1, 4))

    # Reading navigation message file's header
    end_of_header = False
    nav_hdr_cnt = 0
    while not end_of_header:
        nav_header = nav_message.readline()
        # print 'HLine #%s:_%s' % (nav_hdr_cnt, nav_header)
        words = nav_header.strip().split()
        # Checking for the end of header
        if words == ['END', 'OF', 'HEADER']:
            end_of_header = True
        # Reading ionosphere alpha parameters (A0 - A3) of almanac
        elif words[-1] == 'ALPHA':
            nav_header = nav_header.replace('D', 'E')
            for j in xrange(0, 4):
                iono_alpha[0][j] = float(nav_header[2 + j*12:2 + (j + 1)*12])
        # Reading ionosphere beta parameters (B0 - B3) of almanac
        elif words[-1] == 'BETA':
            nav_header = nav_header.replace('D', 'E')
            for j in xrange(0, 4):
                iono_beta[0][j] = float(nav_header[2 + j*12:2 + (j + 1)*12])
        # Reading almanac parameters to compute time in UTC
        elif words[-1] == 'A0,A1,T,W':
            nav_header = nav_header.replace('D', 'E')
            alma_t_para[0][0] = float(nav_header[3:3 + 19])
            alma_t_para[0][1] = float(nav_header[3 + 19:3 + 2*19])
            alma_t_para[0][2] = int(nav_header[3 + 2*19:3 + 2*19 + 9])
            alma_t_para[0][3] = int(nav_header[3 + 2*19 + 9: 3 + 2*19 + 2*9])

        nav_hdr_cnt += 1
    # End of reading the navigation message's header

    # ******************************************************************************************************************

    # Reading navigation message content
    cnt0 = 0
    cnt2 = 0
    for nav_line in nav_message:
        # print 'Line #%s:_%s' % (cnt0, nav_line)
        # Replacing scientific notation 'D' by 'E' so that we can convert a string to a number
        nav_line = nav_line.replace('D', 'E')
        if cnt0 <= 7:
            # Reading the PRN/EPOCH/SV CLK
            if np.mod(cnt0, 8) == 0:
                cnt1 = 0
                raw_ephem[cnt2][0] = int(nav_line[0:2])                     # Col. 0: PRN
                raw_ephem[cnt2][1] = int(nav_line[3:5])                     # Col. 1: Toc - Year
                raw_ephem[cnt2][2] = int(nav_line[6:8])                     # Col. 2: Toc - Month
                raw_ephem[cnt2][3] = int(nav_line[9:11])                    # Col. 3: Toc - Day
                raw_ephem[cnt2][4] = int(nav_line[12:14])                   # Col. 4: Toc - Hour (hr)
                raw_ephem[cnt2][5] = int(nav_line[15:17])                   # Col. 5: Toc - Minute (mn)
                raw_ephem[cnt2][6] = float(nav_line[17:22])                 # Col. 6: Toc - Second (s)
                raw_ephem[cnt2][7] = float(nav_line[22:22 + 19])            # Col. 7: SV clock bias (s)
                raw_ephem[cnt2][8] = float(nav_line[22 + 19:22 + 2*19])     # Col. 8: SV clock drift (s/s)
                raw_ephem[cnt2][9] = float(nav_line[22 + 2*19:22 + 3*19])   # Col. 9: SV clock drift rate (s/s^2)
                cnt3 = 9
            else:
                cnt1 += 1
                # Reading IODE/Crs/Delta n/Mo
                if cnt1 == 1:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        raw_ephem[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                        # Col. 10: IODE - Issue of Data, Ephemeris
                        # Col. 11: Crs      (meter)
                        # Col. 12: Delta n  (rad/sec)
                        # Col. 13: Mo       (rad)
                # Reading Cuc/e Eccentricity/Cus/Sqrt(A)
                elif cnt1 == 2:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        raw_ephem[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                        # Col. 14: Cuc              (rad)
                        # Col. 15: Eccentricity, e
                        # Col. 16: Cus              (rad)
                        # Col. 17: sqrt(A)          (sqrt(meter))
                # Reading Toe/Cic/OMEGA/Cis
                elif cnt1 == 3:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        raw_ephem[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                        # Col. 18: Toe, time of ephemeris   (sec of GPS Week)
                        # Col. 19: Cic                      (rad)
                        # Col. 20: OMEGA                    (rad)
                        # Col. 21: Cis                      (rad)
                # Reading Inclination/Crc/omega/OMEGA DOT
                elif cnt1 == 4:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        raw_ephem[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                        # Col. 22: i_o, reference inclination   (rad)
                        # Col. 23: Crc                          (meter)
                        # Col. 24: omega                        (rad)
                        # Col. 25: OMEGA DOT                    (rad/sec)
                # Reading IDOT/L2 Codes/GPS Week# (to go with Toe)/L2 P data flag
                elif cnt1 == 5:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        raw_ephem[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                        # Col. 26: IDOT, inclination rate (rad/sec)
                        # Col. 27: Codes on L2 channel
                        # Col. 28: GPS Week # (to go with Toe)
                        # Col. 29: L2 P data flag
                # Reading SV accuracy/SV health/TGD/IODC
                elif cnt1 == 6:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        raw_ephem[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                        # Col. 30: SV accuracy  (meter)
                        # Col. 31: SV health
                        # Col. 32: TGD          (sec)
                        # Col. 33: IODC - Issue of Data, Clock
                # Reading Transmission time of message (sec of GPS week)
                elif cnt1 == 7:
                    raw_ephem[cnt2][34] = float(nav_line[3:3 + 19])
                    # Col. 34: Tsv - transmission time of message (sec of GPS week)
                    # Update the row count in "raw_ephem" array
                    cnt2 += 1
                    # First time declare "temp" array
                    temp = np.nan*np.ones((cnt2 + 1, 35))
                    # Cast "raw_ephem" into the first row of "temp"
                    temp[0:cnt2][:] = raw_ephem
        # Done reading the first time entry
        else:
            # Reading the PRN/EPOCH/SV CLK
            if np.mod(cnt0, 8) == 0:
                cnt1 = 0
                temp[cnt2][0] = int(nav_line[0:2])
                temp[cnt2][1] = int(nav_line[3:5])
                temp[cnt2][2] = int(nav_line[6:8])
                temp[cnt2][3] = int(nav_line[9:11])
                temp[cnt2][4] = int(nav_line[12:14])
                temp[cnt2][5] = int(nav_line[15:17])
                temp[cnt2][6] = float(nav_line[17:22])
                temp[cnt2][7] = float(nav_line[22:22 + 19])
                temp[cnt2][8] = float(nav_line[22 + 19:22 + 2*19])
                temp[cnt2][9] = float(nav_line[22 + 2*19:22 + 3*19])
                cnt3 = 9
            else:
                cnt1 += 1
                # Reading IODE/Crs/Delta n/Mo
                if cnt1 == 1:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        temp[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                # Reading Cuc/e Eccentricity/Cus/Sqrt(A)
                elif cnt1 == 2:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        temp[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                # Reading Toe/Cic/OMEGA/Cis
                elif cnt1 == 3:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        temp[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                # Reading Inclination/Crc/omega/OMEGA DOT
                elif cnt1 == 4:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        temp[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                # Reading IDOT/L2 Codes/GPS Week# (to go with Toe)/L2 P data flag
                elif cnt1 == 5:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        temp[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                # Reading SV accuracy/SV health/TGD/IODC
                elif cnt1 == 6:
                    for i in xrange(0, 4):
                        cnt3 += 1
                        temp[cnt2][cnt3] = float(nav_line[3 + i*19:3 + (i + 1)*19])
                # Reading Transmission time of message (sec of GPS week)
                elif cnt1 == 7:
                    temp[cnt2][34] = float(nav_line[3:3 + 19])
                    # Update the raw_ephem array
                    raw_ephem = temp
                    # Update the row count in raw_ephem array
                    cnt2 += 1
                    # Extend the "temp" array by one row
                    temp = np.nan*np.ones((cnt2 + 1, 35))
                    # Cast the "raw_ephem" array into the extended "temp[0:n-1][:]"
                    temp[0:cnt2][:] = raw_ephem
        # Just read another time entry
        # Update the nav_line count
        cnt0 += 1
    # End of "for" loop over the navigation message
    # Close the navigation file
    nav_message.close()

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           EXTRACTING THE EPHEMERIS OUT OF THE NAVIGATION MESSAGE
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Post processing the ephemeris:
    print 'Extracting the Ephemeris...'

    # Rounding up the time stamp base on the minute
    rndup_index = []
    rnddown_index = []
    # Loop over the entire "raw_ephem" array
    for i in xrange(0, len(raw_ephem)):
        # If the minute is not exactly zero, then round up hour, minute, and second
        if raw_ephem[i, 5] != 0:
            # If the minute is less than 30,
            if raw_ephem[i, 5] < 30:
                raw_ephem[i, 4] -= 1        # round the hour down by 1 unit
                raw_ephem[i, 5] = 0         # set the minute to zero
                raw_ephem[i, 6] = 0         # set the second to zero
                rnddown_index.append(i)
            # If the minute is greater than or equal 30,
            elif raw_ephem[i, 5] >= 30:
                raw_ephem[i, 4] += 1        # round the hour up by 1 unit
                raw_ephem[i, 5] = 0         # set the minute to zero
                raw_ephem[i, 6] = 0         # set the second to zero
                rndup_index.append(i)
    # Create the flags for different types of rounding
    if rnddown_index:
        rnddown_flag = True
    elif rndup_index:
        rndup_flag = True
    elif not (rndup_index and rnddown_index):
        rnddown_flag = False
        rndup_flag = False

    # Collect the initial and final TOWs from the flight data
    TOWo = TOW[0]       # Initial time of week (seconds)
    TOWf = TOW[-1]      # Final time of week (seconds)
    # Days of week based on the initial time of week
    DyOW = (TOWo - np.mod(TOWo, dd2sec))/dd2sec         # days
    # The remaining seconds of the day after extracting the days
    rem_sec = np.mod(TOWo, dd2sec)                      # seconds
    # Hours of the day
    HrOD = (rem_sec - np.mod(rem_sec, hr2sec))/hr2sec   # hours
    # The remaining seconds of the hour after extracting days and hours
    rem_sec = np.mod(rem_sec, hr2sec)                   # seconds
    # Minutes of the hour
    MnOH = (rem_sec - np.mod(rem_sec, mn2sec))/mn2sec   # minutes
    # Flight duration in GPS time (seconds)
    deltaTOW = TOWf - TOWo
    if (rem_sec + deltaTOW)/mn2sec <= 60:
        LkUpHr = HrOD
    elif (rem_sec + deltaTOW)/mn2sec > 60:
        LkUpHr = HrOD + 1
    print "Look-up hour: %d" % LkUpHr
    # Look up for the nearest time stamp, then collect the PRNs in this time stamp.
    ephem = np.nan*np.ones((32, 17))
    sv_clock = np.nan * np.ones((32, 5))
    search_again = False
    prev_PRN = 0
    PRN = 1
    cnt4 = 0
    # Loop over the entire "raw_ephem" array
    for i in xrange(0, len(raw_ephem)):
        # Look up the day of the month
        if raw_ephem[i, 3] == DyOM:
            # If found the day, look up the hour
            if raw_ephem[i, 4] == LkUpHr:
                # If the hour matches, check for new PRN
                if raw_ephem[i, 0] == PRN and raw_ephem[i, 0] != prev_PRN:
                    # If new PRN is found, collect the ephemeris parameters
                    for j in xrange(0, 17):
                        if j == 0:
                            ephem[cnt4, j] = raw_ephem[i, 0]
                            # Col. 0: PRN
                        else:
                            ephem[cnt4, j] = raw_ephem[i, 10 + j]
                            # Col. 1: Crs                           (meter)
                            # Col. 2: Delta n                       (rad/sec)
                            # Col. 3: Mo                            (rad)
                            # Col. 4: Cuc                           (rad)
                            # Col. 5: Eccentricity, e
                            # Col. 6: Cus                           (rad)
                            # Col. 7: sqrt(A)                       (sqrt(meter))
                            # Col. 8: Toe, time of ephemeris        (sec of GPS Week)
                            # Col. 9: Cic                           (rad)
                            # Col. 10: OMEGA                        (rad)
                            # Col. 11: Cis                          (rad)
                            # Col. 12: i_o, reference inclination   (rad)
                            # Col. 13: Crc                          (meter)
                            # Col. 14: omega                        (rad)
                            # Col. 15: OMEGA DOT                    (rad/sec)
                            # Col. 16: IDOT, inclination rate       (rad/sec)
                    # Collect the SV Clock information
                    for m in xrange(0, 5):
                        # Collect af0, af1, and af2
                        if m < 3:
                            sv_clock[cnt4, m] = raw_ephem[i, m + 7]
                        # Collect toc, time of clock                (sec of GPS Week)
                        elif m == 3:
                            sv_clock[cnt4, m] = DyOW*dd2sec + LkUpHr*hr2sec + raw_ephem[i, 5]*mn2sec + raw_ephem[i, 6]
                        # Collect TGD, group delay                  (sec)
                        else:
                            sv_clock[cnt4, m] = raw_ephem[i, 32]
                    # Remember the current PRN
                    prev_PRN = PRN
                    # Update the PRN and the "ephemeris" row index
                    PRN += 1
                    cnt4 += 1
                elif raw_ephem[i, 0] == (PRN - 1) and raw_ephem[i, 0] == prev_PRN:
                    # Else if the PRN is repeated, check for rounding condition
                    if rndup_flag:
                        # If the previous time stamp was rounded up,
                        print "Used the original time stamp instead of the rounded up on PRN %s" % prev_PRN
                        # use the current (original) time stamp instead of the rounding one.
                        for j in xrange(0, 17):
                            if j == 0:
                                ephem[cnt4 - 1, j] = raw_ephem[i, 0]
                            else:
                                ephem[cnt4 - 1, j] = raw_ephem[i, 10 + j]
                        # Collect the SV Clock information
                        for m in xrange(0, 5):
                            # Collect af0, af1, and af2
                            if m < 3:
                                sv_clock[cnt4 - 1, m] = raw_ephem[i, m + 7]
                            # Collect toc, time of clock                (sec of GPS Week)
                            elif m == 3:
                                sv_clock[cnt4 - 1, m] = DyOW * dd2sec + LkUpHr * hr2sec + raw_ephem[i, 5] * mn2sec + \
                                                     raw_ephem[i, 6]
                            # Collect TGD, group delay                  (sec)
                            else:
                                sv_clock[cnt4 - 1, m] = raw_ephem[i, 32]
                    elif rnddown_flag:
                        # If the current time stamp is rounded down, skipped it and used the previous (original) one.
                        print 'Skipped repeated rounded down time stamp on PRN %s' % prev_PRN
    # End of "raw_ephem" and did not find all 32 satellites in the time stamp,
    # or did not find any time stamp that matches the look up hour.
        elif i == (len(raw_ephem) - 1) and cnt4 < 32:
            # If the hour does not match, adjust the look up hour based on the minute of the hour.
            print "Did not find any time stamp that matches the initial TOW."
            if MnOH <= 30:
                # If the minute is less than or equal to 30,
                HrOD -= 1         # decrease the look up hour by 1 unit
                if (rem_sec + deltaTOW) / mn2sec <= 60:
                    LkUpHr = HrOD
                elif (rem_sec + deltaTOW) / mn2sec > 60:
                    LkUpHr = HrOD + 1
                print "Change look up hour to %d." % LkUpHr
            elif MnOH > 30:
                # If the minute is greater than 30,
                HrOD += 1         # increase the look up hour by 1 unit
                if (rem_sec + deltaTOW) / mn2sec <= 60:
                    LkUpHr = HrOD
                elif (rem_sec + deltaTOW) / mn2sec > 60:
                    LkUpHr = HrOD + 1
                print "Change look up hour to %d." % LkUpHr
            search_again = True

    if search_again:
        prev_PRN = 0
        PRN = 1
        cnt4 = 0
        while cnt4 < 32:
            for k in xrange(0, len(raw_ephem)):
                # Look up the day of the month
                if raw_ephem[k, 3] == DyOM:
                    # If found the day, look up the hour
                    if raw_ephem[k, 4] == LkUpHr:
                        # If the hour matches, check for new PRN
                        if raw_ephem[k, 0] == PRN and raw_ephem[k, 0] != prev_PRN:
                            # If new PRN is found, collect the ephemeris parameters
                            for j in xrange(0, 17):
                                if j == 0:
                                    ephem[cnt4, j] = raw_ephem[k, 0]
                                else:
                                    ephem[cnt4, j] = raw_ephem[k, 10 + j]
                            # Collect the SV Clock information
                            for m in xrange(0, 5):
                                # Collect af0, af1, and af2
                                if m < 3:
                                    sv_clock[cnt4, m] = raw_ephem[k, m + 7]
                                # Collect toc, time of clock                (sec of GPS Week)
                                elif m == 3:
                                    sv_clock[cnt4, m] = DyOW * dd2sec + LkUpHr * hr2sec + raw_ephem[k, 5] * mn2sec + \
                                                        raw_ephem[k, 6]
                                # Collect TGD, group delay                  (sec)
                                else:
                                    sv_clock[cnt4, m] = raw_ephem[k, 32]
                            # Remember the current PRN
                            prev_PRN = PRN
                            # Update the PRN and "ephemeris" row index
                            PRN += 1
                            cnt4 += 1
                        elif raw_ephem[k, 0] == (PRN - 1) and raw_ephem[k, 0] == prev_PRN:
                            # Else if the PRN is repeated, check for rounding condition
                            if rndup_flag:
                                # If the previous time stamp was rounded up,
                                print "Used the original time stamp instead of the rounded up on PRN %s" % prev_PRN
                                # use the current (original) time stamp instead of the rounding one.
                                for j in xrange(0, 17):
                                    if j == 0:
                                        ephem[cnt4 - 1, j] = raw_ephem[k, 0]
                                    else:
                                        ephem[cnt4 - 1, j] = raw_ephem[k, 10 + j]
                                # Collect the SV Clock information
                                for m in xrange(0, 5):
                                    # Collect af0, af1, and af2
                                    if m < 3:
                                        sv_clock[cnt4 - 1, m] = raw_ephem[k, m + 7]
                                    # Collect toc, time of clock                (sec of GPS Week)
                                    elif m == 3:
                                        sv_clock[cnt4 - 1, m] = DyOW * dd2sec + LkUpHr * hr2sec + \
                                                         raw_ephem[k, 5] * mn2sec + raw_ephem[k, 6]
                                    # Collect TGD, group delay                  (sec)
                                    else:
                                        sv_clock[cnt4 - 1, m] = raw_ephem[k, 32]
                            elif rnddown_flag:
                                # If the current time stamp is rounded down,
                                # skipped it and used the previous (original) one
                                print 'Skipped repeated rounded down time stamp on PRN %s' % prev_PRN

    # Finish reading and processing navigation message

    return iono_alpha, iono_beta, alma_t_para, sv_clock, raw_ephem, ephem

# End of Ephemeris Processing


'''
========================================================================================================================
                                                SYSTEM CONFIGURATIONS
========================================================================================================================
'''
'''
    ---------------------------
    A. IMU Configuration Struct
    ---------------------------
'''


class ImuConfigStruct:

    def __init__(self):
        self.b_a = np.nan * np.ones((3, 1))
        self.b_g = np.nan * np.ones((3, 1))
        self.M_a = np.nan * np.matrix(np.ones((3, 3)))
        self.M_g = np.nan * np.matrix(np.ones((3, 3)))
        self.G_g = np.nan * np.matrix(np.ones((3, 3)))
        self.accel_noise_root_PSD = np.nan
        self.gyro_noise_root_PSD = np.nan
        self.accel_quant_level = np.nan
        self.gyro_quant_level = np.nan


'''
    ----------------------------
    B. GNSS Configuration Struct
    ----------------------------
'''


class GnssConfigStruct:

    def __init__(self):
        self.epoch_interval = np.nan
        self.init_est_r_ea_e = np.nan * np.ones((3, 1))
        self.init_est_v_ea_e = np.nan * np.ones((3, 1))
        self.no_sat = np.nan
        self.r_os = np.nan
        self.inclination = np.nan
        self.const_delta_lambda = np.nan
        self.const_t_offset = np.nan
        self.mask_angle = np.nan
        self.SIS_err_SD = np.nan
        self.zenith_iono_err_SD = np.nan
        self.zenith_trop_err_SD = np.nan
        self.code_track_err_ID = np.nan
        self.rate_track_err_ID = np.nan
        self.rx_clock_offset = np.nan
        self.rx_clock_drift = np.nan


'''
    ---------------------------
    C. EKF Configuration Struct
    ---------------------------
'''


class EkfConfigStruct:

    def __init__(self):
        self.init_att_unc = np.nan
        self.init_vel_unc = np.nan
        self.init_pos_unc = np.nan
        self.init_b_a_unc = np.nan
        self.init_b_g_unc = np.nan
        self.gyro_clock_offset_unc = np.nan
        self.init_clock_drift_unc = np.nan
        self.gyro_noise_PSD = np.nan
        self.accel_noise_PSD = np.nan
        self.accel_bias_PSD = np.nan
        self.gyro_bias_PSD = np.nan
        self.clock_freq_PSD = np.nan
        self.clock_phase_PSD = np.nan
        self.pseudo_range_SD = np.nan
        self.range_rate_SD = np.nan
        self.pos_meas_SD = np.nan
        self.vel_meas_SD = np.nan


'''
    -------------------------------------------------------------------
    D. Initial Errors of Attitude Angles (radian) Resolved in NED Frame
    -------------------------------------------------------------------
'''


def att_init_error(r_err, p_err, y_err, unit):

    print ' Set Initial Errors...'

    if unit == 'radian':
        eul_err_nb_n = np.array([[r_err], [p_err], [y_err]])
    elif unit == 'degree':
        eul_err_nb_n = d2r * np.array([[r_err], [p_err], [y_err]])

    return eul_err_nb_n

# End of Attitude Error Initialization


'''
    -----------------------------
    E. IMU Configuration Function
    -----------------------------
'''


def imu_configuration(imugrade):

    print ' Setup IMU Configuraton...'
    imu_config = ImuConfigStruct()

    if imugrade == 'aviation':

        # Accelerometer biases (micro-g --> m/s^2; body axes)
        imu_config.b_a = micro_g * np.matrix([[30.0], [-45.0], [26.0]])

        # Gyro biases (deg/hr --> rad/s; body axes)
        imu_config.b_g = (d2r / 3600.0) * np.matrix([[-0.0009], [0.0013], [-0.0008]])

        # Accelerometer scale factor and cross coupling errors (ppm --> unitless; body axes)
        imu_config.M_a = 1.0e-06 * np.matrix([[100.0, -120.0, 80.0],
                                             [-60.0, -120.0, 100.0],
                                             [-100.0, 40.0, 90.0]])

        # Gyro scale factor and cross coupling errors (ppm --> unitless; body axes)
        imu_config.M_g = 1.0e-06 * np.matrix([[8.0, -120.0, 100.0],
                                             [0.0, -6.0, -60.0],
                                             [0.0, 0.0, -7.0]])

        # Gyro g-dependent biases (deg/hr/g --> rad-s/m; body axes)
        imu_config.G_g = -(d2r / (3600.0 * g)) * np.matrix(np.zeros(3))

        # Accelerometer noise sqrt(PSD) (micro-g/sqrt(Hz) --> m/s^1.5)
        imu_config.accel_noise_root_PSD = 20.0 * micro_g

        # Gyro noise root PSD (deg/sqrt(hr) --> rad/sqrt(s))
        imu_config.gyro_noise_root_PSD = 0.002 * d2r / 60.0

        # Accelerometer quantization level (m/s^2)
        imu_config.accel_quant_level = 5.0e-05

        # Gyro quantization level (rad/s)
        imu_config.gyro_quant_level = 1.0e-06

    elif imugrade == 'consumer':

        # Accelerometer biases (micro-g --> m/s^2; body axes)
        imu_config.b_a = micro_g * np.array([[9.0e03], [-1.3e04], [8.0e03]])

        # Gyro biases (deg/hr --> rad/s; body axes)
        imu_config.b_g = (d2r / 3600.0) * np.array([[-1.8e02], [2.6e02], [-1.6e02]])

        # Accelerometer scale factor and cross coupling errors (ppm --> unitless; body axes)
        imu_config.M_a = 1.0e-06 * np.matrix([[5.0e4,  -1.5e04,  1.0e04],
                                             [-7.5e03, -6.0e04, 1.25e04],
                                             [-1.25e04, 5.0e03,  2.0e04]])

        # Gyro scale factor and cross coupling errors (ppm --> unitless; body axes)
        imu_config.M_g = 1.0e-06 * np.matrix([[4.0e04, -1.4e04,  1.25e04],
                                              [0.0,    -3.0e04,  -7.5e03],
                                              [0.0,        0.0, -1.75e04]])

        # Gyro g-dependent biases (deg/hr/g --> rad-s/m; body axes)
        imu_config.G_g = (d2r / (3600.0 * g)) * np.matrix([[9.0e01, -1.1e02, -6.0e01],
                                                          [-5.0e01,  1.9e02, -1.6e02],
                                                           [3.0e01,  1.1e02, -1.3e02]])

        # Accelerometer noise sqrt(PSD) (micro-g/sqrt(Hz) --> m/s^1.5)
        imu_config.accel_noise_root_PSD = 1000.0 * micro_g

        # Gyro noise root PSD (deg/sqrt(hr) --> rad/sqrt(s))
        imu_config.gyro_noise_root_PSD = 1.0 * d2r / 60.0

        # Accelerometer quantization level (m/s^2)
        imu_config.accel_quant_level = 1.0e-01

        # Gyro quantization level (rad/s)
        imu_config.gyro_quant_level = 2.0e-03

    elif imugrade == 'tactical':

        # Accelerometer biases (micro-g --> m/s^2; body axes)
        imu_config.b_a = micro_g * np.array([[9.0e02], [-13.0e02], [8.0e02]])

        # Gyro biases (deg/hr --> rad/s; body axes)
        imu_config.b_g = (d2r / 3600.0) * np.array([[-9.0], [13.0], [-8.0]])

        # Accelerometer scale factor and cross coupling errors (ppm --> unitless; body axes)
        imu_config.M_a = 1.0e-06 * np.matrix([[5.0e02, -3.0e02, 2.0e02],
                                             [-1.5e02, -6.0e02, 2.5e02],
                                             [-2.5e02,  1.0e02, 4.5e02]])

        # Gyro scale factor and cross coupling errors (ppm --> unitless; body axes)
        imu_config.M_g = 1.0e-06 * np.matrix([[4.0e02, -3.0e02,  2.5e02],
                                              [0.0,    -3.0e02, -1.5e02],
                                              [0.0,        0.0, -3.5e02]])

        # Gyro g-dependent biases (deg/hr/g --> rad-s/m; body axes)
        imu_config.G_g = (d2r / (3600.0 * g)) * np.matrix([[0.9, -1.1, -0.6],
                                                          [-0.5,  1.9, -1.6],
                                                           [0.3,  1.1, -1.3]])

        # Accelerometer noise root PSD (micro-g/sqrt(Hz) --> m/s^1.5)
        imu_config.accel_noise_root_PSD = 100.0 * micro_g

        # Gyro noise root PSD (deg/sqrt(hr) --> rad/sqrt(s))
        imu_config.gyro_noise_root_PSD = 0.01 * d2r / 60.0

        # Accelerometer quantization level (m/s^2)
        imu_config.accel_quant_level = 1.0e-02

        # Gyro quantization level (rad/s)
        imu_config.gyro_quant_level = 2.0e-04

    # End of "If" Statement

    return imu_config

# End of IMU Configuration


'''
    ---------------------
    F. GNSS Configuration
    ---------------------
'''


def gnss_configuration(frequence, constellation, tow):

    print ' Setup GNSS Constellation...'
    gnss_config = GnssConfigStruct()

    if constellation == 'gps':

        gnss_config.epoch_interval = 1.0 / frequence            # GNSS updating frequency (Hz --> sec)
        gnss_config.init_est_r_ea_e = np.zeros((3, 1))          # Initial estimated position (m; ECEF)
        gnss_config.init_est_v_ea_e = np.zeros((3, 1))          # Initial estimated velocity (m/s; ECEF)
        gnss_config.no_sat = 32                                 # Number of satellites in constellation
        gnss_config.r_os = 2.656175e+07                         # Orbital radius of satellites (m)
        gnss_config.inclination = 55.0                          # Inclination angle of satellites (deg)
        gnss_config.const_delta_lambda = 0.0                    # Longitude offset of constellation (deg)
        gnss_config.const_t_offset = tow[0, 0]                  # Timing offset of constellation (s)
        gnss_config.mask_angle = 10.0                           # Mask angle (deg)
        gnss_config.SIS_err_SD = 1.0                            # Signal in space error SD (m)
        gnss_config.zenith_iono_err_SD = 2.0                    # Zenith ionosphere error SD (m)
        gnss_config.zenith_trop_err_SD = 0.2                    # Zenith troposphere error SD (m)
        gnss_config.code_track_err_SD = 1.0                     # Code tracking error SD (m)
        gnss_config.rate_track_err_SD = 0.02                    # Range rate tracking error SD (m/s)
        gnss_config.rx_clock_offset = 10000.0                   # Receiver clock offset at time = 0 (m)
        gnss_config.rx_clock_drift = 100.0                      # Receiver clock drift at time = 0 (m/s)

    # End of "If" Statement

    return gnss_config

# End of GNSS Configuration


'''
    -----------------------------------------
    G. EKF Initialization for Single Coupling
    -----------------------------------------
'''


def single_ekf_configuration(imugrade, tightness):
    print ' Setup Single EKF...'
    tc_ekf_config = EkfConfigStruct()
    lc_ekf_config = EkfConfigStruct()

    if imugrade == 'aviation' and tightness == 'tight':
        tc_ekf_config.init_att_unc = 0.01*d2r                       # Initial attitude uncertainty per axis (deg-->rad)
        tc_ekf_config.init_vel_unc = 0.1                            # Initial velocity uncertainty per axis (m/s)
        tc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        tc_ekf_config.init_b_a_unc = 30.0*micro_g
        # Initial gyroscope bias uncertainty per instrument (deg/hr --> rad/s)
        tc_ekf_config.init_b_g_unc = 0.001*d2r/3600.0
        tc_ekf_config.init_clock_offset_unc = 10.0                  # Initial clock offset uncertainty per axis (m)
        tc_ekf_config.init_clock_drift_unc = 0.1                    # Initial clock drift uncertainty per axis (m/s)
        tc_ekf_config.gyro_noise_PSD = (0.004*d2r/60.0)**2          # Gyro noise PSD (deg^2/hr --> rad^2/s)
        tc_ekf_config.accel_noise_PSD = (40.0*micro_g)**2           # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        tc_ekf_config.accel_bias_PSD = 3.0E-9                       # Accelerometer bias random walk PSD (m^2/s^5)
        tc_ekf_config.gyro_bias_PSD = 2.0E-16                       # Gyro bias random walk PSD (rad^2/s^3)
        tc_ekf_config.clock_freq_PSD = 1.0                          # Receiver clock frequency-drift PSD (m^2/s^3)
        tc_ekf_config.clock_phase_PSD = 1.0                         # Receiver clock phase-drift PSD (m^2/s)
        tc_ekf_config.pseudo_range_SD = 2.5                         # Pseudo-range measurement noise SD (m)
        tc_ekf_config.range_rate_SD = 0.1                           # Pseudo-range rate measurement noise SD (m/s)

    elif imugrade == 'consumer' and tightness == 'tight':
        tc_ekf_config.init_att_unc = 2.0*d2r                        # Initial attitude uncertainty per axis (deg-->rad)
        tc_ekf_config.init_vel_unc = 0.1                            # Initial velocity uncertainty per axis (m/s)
        tc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        tc_ekf_config.init_b_a_unc = 10000.0*micro_g
        # Initial gyroscope bias uncertainty per instrument (deg/hr --> rad/s)
        tc_ekf_config.init_b_g_unc = 200.0*d2r/3600.0
        tc_ekf_config.init_clock_offset_unc = 10.0                  # Initial clock offset uncertainty per axis (m)
        tc_ekf_config.init_clock_drift_unc = 0.1                    # Initial clock drift uncertainty per axis (m/s)
        tc_ekf_config.gyro_noise_PSD = 1.0E-4                       # Gyro noise PSD (deg^2/hr --> rad^2/s)
        tc_ekf_config.accel_noise_PSD = 0.04                        # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        tc_ekf_config.accel_bias_PSD = 1.0E-5                       # Accelerometer bias random walk PSD (m^2/s^5)
        tc_ekf_config.gyro_bias_PSD = 4.0E-11                       # Gyro bias random walk PSD (rad^2/s^3)
        tc_ekf_config.clock_freq_PSD = 1.0                          # Receiver clock frequency-drift PSD (m^2/s^3)
        tc_ekf_config.clock_phase_PSD = 1.0                         # Receiver clock phase-drift PSD (m^2/s)
        tc_ekf_config.pseudo_range_SD = 2.5                         # Pseudo-range measurement noise SD (m)
        tc_ekf_config.range_rate_SD = 0.1                           # Pseudo-range rate measurement noise SD (m/s)

    elif imugrade == 'tactical' and tightness == 'tight':
        tc_ekf_config.init_att_unc = 1.0*d2r                        # Initial attitude uncertainty per axis (deg-->rad)
        tc_ekf_config.init_vel_unc = 0.1                            # Initial velocity uncertainty per axis (m/s)
        tc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        tc_ekf_config.init_b_a_unc = 1000.0 * micro_g
        # Initial gyroscope bias uncertainty per instrument (deg/hr --> rad/s)
        tc_ekf_config.init_b_g_unc = 10.0 * d2r / 3600.0
        tc_ekf_config.init_clock_offset_unc = 10.0                  # Initial clock offset uncertainty per axis (m)
        tc_ekf_config.init_clock_drift_unc = 0.1                    # Initial clock drift uncertainty per axis (m/s)
        tc_ekf_config.gyro_noise_PSD = (0.02 * d2r / 60) ** 2       # Gyro noise PSD (deg^2/hr --> rad^2/s)
        tc_ekf_config.accel_noise_PSD = (200.0*micro_g)**2          # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        tc_ekf_config.accel_bias_PSD = 1.0E-7                       # Accelerometer bias random walk PSD (m^2/s^5)
        tc_ekf_config.gyro_bias_PSD = 2.0E-12                       # Gyro bias random walk PSD (rad^2/s^3)
        tc_ekf_config.clock_freq_PSD = 1.0                          # Receiver clock frequency-drift PSD (m^2/s^3)
        tc_ekf_config.clock_phase_PSD = 1.0                         # Receiver clock phase-drift PSD (m^2/s)
        tc_ekf_config.pseudo_range_SD = 2.5                         # Pseudo-range measurement noise SD (m)
        tc_ekf_config.range_rate_SD = 0.1                           # Pseudo-range rate measurement noise SD (m/s)

    elif imugrade == 'aviation' and tightness == 'loose':
        lc_ekf_config.init_att_unc = 0.01 * d2r                     # Initial attitude uncertainty per axis (deg-->rad)
        lc_ekf_config.init_vel_unc = 0.2                            # Initial velocity uncertainty per axis (m/s)
        lc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        lc_ekf_config.init_b_a_unc = 30.0 * micro_g
        # Initial gyro bias uncertainty per instrument (deg/hr --> rad/s)
        lc_ekf_config.init_b_g_unc = 0.001 * d2r / 3600.0
        lc_ekf_config.gyro_noise_PSD = (0.004 * d2r / 60.0) ** 2    # Gyro noise PSD (deg^2/hr --> rad^2/s)
        lc_ekf_config.accel_noise_PSD = (40.0*micro_g)**2           # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        lc_ekf_config.accel_bias_PSD = 3.0E-9                       # Accelerometer bias random walk PSD (m^2/s^5)
        lc_ekf_config.gyro_bias_PSD = 2.0E-16                       # Gyro bias random walk PSD (rad^2/s^3)
        lc_ekf_config.pos_meas_SD = 2.5                             # Position measurement noise SD per axis (m)
        lc_ekf_config.vel_meas_SD = 0.1                             # Velocity measurement noise SD per axis (m/s)

    elif imugrade == 'consumer' and tightness == 'loose':

        lc_ekf_config.init_att_unc = 2.0 * d2r                      # Initial attitude uncertainty per axis (deg-->rad)
        lc_ekf_config.init_vel_unc = 0.2                            # Initial velocity uncertainty per axis (m/s)
        lc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        lc_ekf_config.init_b_a_unc = 10000.0 * micro_g
        # Initial gyro bias uncertainty per instrument (deg/hr --> rad/s)
        lc_ekf_config.init_b_g_unc = 200.0 * d2r / 3600.0
        lc_ekf_config.gyro_noise_PSD = 0.01 ** 2                    # Gyro noise PSD (deg^2/hr --> rad^2/s)
        lc_ekf_config.accel_noise_PSD = 0.2 ** 2                    # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        lc_ekf_config.accel_bias_PSD = 1.0E-5                       # Accelerometer bias random walk PSD (m^2/s^5)
        lc_ekf_config.gyro_bias_PSD = 4.0E-11                       # Gyro bias random walk PSD (rad^2/s^3)
        lc_ekf_config.pos_meas_SD = 2.5                             # Position measurement noise SD per axis (m)
        lc_ekf_config.vel_meas_SD = 0.1                             # Velocity measurement noise SD per axis (m/s)

    elif imugrade == 'tactical' and tightness == 'loose':

        lc_ekf_config.init_att_unc = 1.0 * d2r                      # Initial attitude uncertainty per axis (deg-->rad)
        lc_ekf_config.init_vel_unc = 0.2                            # Initial velocity uncertainty per axis (m/s)
        lc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        lc_ekf_config.init_b_a_unc = 1000.0 * micro_g
        # Initial gyro bias uncertainty per instrument (deg/hr --> rad/s)
        lc_ekf_config.init_b_g_unc = 10.0 * d2r / 3600.0
        lc_ekf_config.gyro_noise_PSD = (0.02 * d2r / 60) ** 2       # Gyro noise PSD (deg^2/hr --> rad^2/s)
        lc_ekf_config.accel_noise_PSD = (200.0 * micro_g) ** 2      # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        lc_ekf_config.accel_bias_PSD = 1.0E-7                       # Accelerometer bias random walk PSD (m^2/s^5)
        lc_ekf_config.gyro_bias_PSD = 2.0E-12                       # Gyro bias random walk PSD (rad^2/s^3)
        lc_ekf_config.pos_meas_SD = 2.5                             # Position measurement noise SD per axis (m)
        lc_ekf_config.vel_meas_SD = 0.1                             # Velocity measurement noise SD per axis (m/s)

    # End of If statement for initialization

    if tightness == 'loose':
        return lc_ekf_config
    elif tightness == 'tight':
        return tc_ekf_config

    # End of If statement for returning

# End of EKF Initialization


'''
    ---------------------------------------
    H. EKF Initialization for Dual Coupling
    ---------------------------------------
'''


def dual_ekf_configuration(imugrade):
    print 'Setup Dual EKF...'
    tc_ekf_config = EkfConfigStruct()
    lc_ekf_config = EkfConfigStruct()

    if imugrade == 'aviation':

        # Loosely coupling
        lc_ekf_config.init_att_unc = 0.01 * d2r                     # Initial attitude uncertainty per axis (deg-->rad)
        lc_ekf_config.init_vel_unc = 0.1                            # Initial velocity uncertainty per axis (m/s)
        lc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        lc_ekf_config.init_b_a_unc = 30.0 * micro_g
        # Initial gyro bias uncertainty per instrument (deg/hr --> rad/s)
        lc_ekf_config.init_b_g_unc = 0.001 * d2r / 3600.0
        lc_ekf_config.gyro_noise_PSD = (0.004 * d2r / 60.0) ** 2    # Gyro noise PSD (deg^2/hr --> rad^2/s)
        lc_ekf_config.accel_noise_PSD = (40.0 * micro_g) ** 2       # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        lc_ekf_config.accel_bias_PSD = 3.0E-9                       # Accelerometer bias random walk PSD (m^2/s^5)
        lc_ekf_config.gyro_bias_PSD = 2.0E-16                       # Gyro bias random walk PSD (rad^2/s^3)
        lc_ekf_config.pos_meas_SD = 2.5                             # Position measurement noise SD per axis (m)
        lc_ekf_config.vel_meas_SD = 0.1                             # Velocity measurement noise SD per axis (m/s)

        # Tightly coupling
        tc_ekf_config.init_att_unc = 0.01 * d2r                     # Initial attitude uncertainty per axis (deg-->rad)
        tc_ekf_config.init_vel_unc = 0.1                            # Initial velocity uncertainty per axis (m/s)
        tc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        tc_ekf_config.init_b_a_unc = 30.0 * micro_g
        # Initial gyro bias uncertainty per instrument (deg/hr --> rad/s)
        tc_ekf_config.init_b_g_unc = 0.001 * d2r / 3600.0
        tc_ekf_config.init_clock_offset_unc = 10.0                  # Initial clock offset uncertainty per axis (m)
        tc_ekf_config.init_clock_drift_unc = 0.1                    # Initial clock drift uncertainty per axis (m/s)
        tc_ekf_config.gyro_noise_PSD = (0.004 * d2r / 60.0) ** 2    # Gyro noise PSD (deg^2/hr --> rad^2/s)
        tc_ekf_config.accel_noise_PSD = (40.0 * micro_g) ** 2       # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        tc_ekf_config.accel_bias_PSD = 3.0E-9                       # Accelerometer bias random walk PSD (m^2/s^5)
        tc_ekf_config.gyro_bias_PSD = 2.0E-16                       # Gyro bias random walk PSD (rad^2/s^3)
        tc_ekf_config.clock_freq_PSD = 1.0                          # Receiver clock frequency-drift PSD (m^2/s^3)
        tc_ekf_config.clock_phase_PSD = 1.0                         # Receiver clock phase-drift PSD (m^2/s)
        tc_ekf_config.pseudo_range_SD = 2.5                         # Pseudo-range measurement noise SD (m)
        tc_ekf_config.range_rate_SD = 0.1                           # Pseudo-range rate measurement noise SD (m/s)

    elif imugrade == 'consumer':

        # Loosely coupling
        lc_ekf_config.init_att_unc = 2.0 * d2r                      # Initial attitude uncertainty per axis (deg-->rad)
        lc_ekf_config.init_vel_unc = 0.1                            # Initial velocity uncertainty per axis (m/s)
        lc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        lc_ekf_config.init_b_a_unc = 10000.0 * micro_g
        # Initial gyro bias uncertainty per instrument (deg/hr --> rad/s)
        lc_ekf_config.init_b_g_unc = 200.0 * d2r / 3600.0
        lc_ekf_config.gyro_noise_PSD = 0.01 ** 2                    # Gyro noise PSD (deg^2/hr --> rad^2/s)
        lc_ekf_config.accel_noise_PSD = 0.2 ** 2                    # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        lc_ekf_config.accel_bias_PSD = 1.0E-5                       # Accelerometer bias random walk PSD (m^2/s^5)
        lc_ekf_config.gyro_bias_PSD = 4.0E-11                       # Gyro bias random walk PSD (rad^2/s^3)
        lc_ekf_config.pos_meas_SD = 2.5                             # Position measurement noise SD per axis (m)
        lc_ekf_config.vel_meas_SD = 0.1                             # Velocity measurement noise SD per axis (m/s)

        # Tightly coupling
        tc_ekf_config.init_att_unc = 2.0 * d2r                      # Initial attitude uncertainty per axis (deg-->rad)
        tc_ekf_config.init_vel_unc = 0.1                            # Initial velocity uncertainty per axis (m/s)
        tc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        tc_ekf_config.init_b_a_unc = 10000.0 * micro_g
        # Initial gyro bias uncertainty per instrument (deg/hr --> rad/s)
        tc_ekf_config.init_b_g_unc = 200.0 * d2r / 3600.0
        tc_ekf_config.init_clock_offset_unc = 10.0                  # Initial clock offset uncertainty per axis (m)
        tc_ekf_config.init_clock_drift_unc = 0.1                    # Initial clock drift uncertainty per axis (m/s)
        tc_ekf_config.gyro_noise_PSD = 0.01 ** 2                    # Gyro noise PSD (deg^2/hr --> rad^2/s)
        tc_ekf_config.accel_noise_PSD = 0.2 ** 2                    # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        tc_ekf_config.accel_bias_PSD = 1.0E-5                       # Accelerometer bias random walk PSD (m^2/s^5)
        tc_ekf_config.gyro_bias_PSD = 4.0E-11                       # Gyro bias random walk PSD (rad^2/s^3)
        tc_ekf_config.clock_freq_PSD = 1.0                          # Receiver clock frequency-drift PSD (m^2/s^3)
        tc_ekf_config.clock_phase_PSD = 1.0                         # Receiver clock phase-drift PSD (m^2/s)
        tc_ekf_config.pseudo_range_SD = 2.5                         # Pseudo-range measurement noise SD (m)
        tc_ekf_config.range_rate_SD = 0.1                           # Pseudo-range rate measurement noise SD (m/s)

    elif imugrade == 'tactical':

        # Loosely coupling
        lc_ekf_config.init_att_unc = 1.0 * d2r                      # Initial attitude uncertainty per axis (deg-->rad)
        lc_ekf_config.init_vel_unc = 0.1                            # Initial velocity uncertainty per axis (m/s)
        lc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        lc_ekf_config.init_b_a_unc = 1000.0 * micro_g
        # Initial gyro bias uncertainty per instrument (deg/hr --> rad/s)
        lc_ekf_config.init_b_g_unc = 10.0 * d2r / 3600.0
        lc_ekf_config.gyro_noise_PSD = (0.02 * d2r / 60) ** 2       # Gyro noise PSD (deg^2/hr --> rad^2/s)
        lc_ekf_config.accel_noise_PSD = (200.0 * micro_g) ** 2      # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        lc_ekf_config.accel_bias_PSD = 1.0E-7                       # Accelerometer bias random walk PSD (m^2/s^5)
        lc_ekf_config.gyro_bias_PSD = 2.0E-12                       # Gyro bias random walk PSD (rad^2/s^3)
        lc_ekf_config.pos_meas_SD = 2.5                             # Position measurement noise SD per axis (m)
        lc_ekf_config.vel_meas_SD = 0.1                             # Velocity measurement noise SD per axis (m/s)

        # Tightly coupling
        tc_ekf_config.init_att_unc = 1.0 * d2r                      # Initial attitude uncertainty per axis (deg-->rad)
        tc_ekf_config.init_vel_unc = 0.1                            # Initial velocity uncertainty per axis (m/s)
        tc_ekf_config.init_pos_unc = 2.0                            # Initial position uncertainty per axis (m)
        # Initial accelerometer bias uncertainty per instrument (micro-g --> m/s^2)
        tc_ekf_config.init_b_a_unc = 1000.0 * micro_g
        # Initial gyro bias uncertainty per instrument (deg/hr --> rad/s)
        tc_ekf_config.init_b_g_unc = 10.0 * d2r / 3600.0
        tc_ekf_config.init_clock_offset_unc = 10.0                  # Initial clock offset uncertainty per axis (m)
        tc_ekf_config.init_clock_drift_unc = 0.1                    # Initial clock drift uncertainty per axis (m/s)
        tc_ekf_config.gyro_noise_PSD = (0.02 * d2r / 60) ** 2       # Gyro noise PSD (deg^2/hr --> rad^2/s)
        tc_ekf_config.accel_noise_PSD = (200.0 * micro_g) ** 2      # Accelerometer noise PSD (micro-g^2/Hz --> m^2/s^3)
        tc_ekf_config.accel_bias_PSD = 1.0E-7                       # Accelerometer bias random walk PSD (m^2/s^5)
        tc_ekf_config.gyro_bias_PSD = 2.0E-12                       # Gyro bias random walk PSD (rad^2/s^3)
        tc_ekf_config.clock_freq_PSD = 1.0                          # Receiver clock frequency-drift PSD (m^2/s^3)
        tc_ekf_config.clock_phase_PSD = 1.0                         # Receiver clock phase-drift PSD (m^2/s)
        tc_ekf_config.pseudo_range_SD = 2.5                         # Pseudo-range measurement noise SD (m)
        tc_ekf_config.range_rate_SD = 0.1                           # Pseudo-range rate measurement noise SD (m/s)

    # End of "If" statement for initialization

    return lc_ekf_config, tc_ekf_config

# End of Dual EKF Initialization


'''
========================================================================================================================
                                                UTILITY FUNCTIONS
========================================================================================================================
'''
'''
    ---------------------------------------------------------
    1. Euler Angles to Coordinate Transformation Matrix (CTM)
    ---------------------------------------------------------
'''


def euler_to_ctm(euler):

    # Pre-calculate np.sines and np.cosines of the Euler angles
    sin_phi = np.sin(euler[0, 0])
    cos_phi = np.cos(euler[0, 0])
    sin_theta = np.sin(euler[1, 0])
    cos_theta = np.cos(euler[1, 0])
    sin_psi = np.sin(euler[2, 0])
    cos_psi = np.cos(euler[2, 0])

    # Establish the coordinate transformation matrix using (2.22)
    c_matrix = np.nan * np.matrix(np.ones((3, 3)))
    c_matrix[0, 0] = cos_theta * cos_psi
    c_matrix[0, 1] = cos_theta * sin_psi
    c_matrix[0, 2] = -sin_theta
    c_matrix[1, 0] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
    c_matrix[1, 1] = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
    c_matrix[1, 2] = sin_phi * cos_theta
    c_matrix[2, 0] = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi
    c_matrix[2, 1] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi
    c_matrix[2, 2] = cos_phi * cos_theta

    return c_matrix

# End of Euler Angle to CTM Conversion


'''
    ----------------------
    2. CTM to Euler Angles
    ----------------------
'''


def ctm_to_euler(c_matrix):

    euler = np.zeros((3, 1))
    euler[0, 0] = np.arctan2(c_matrix[1, 2], c_matrix[2, 2])
    euler[1, 0] = -np.arcsin(c_matrix[0, 2])
    euler[2, 0] = np.arctan2(c_matrix[0, 1], c_matrix[0, 0])

    return euler

# End of CTM to Euler Angle Conversion


'''
    -----------------------------------------------------
    3. ECEF to NED Coordinate Transformation Matrix (CTM)
    -----------------------------------------------------
'''


def ecef_to_ned_ctm(lat, lon, trig):

    # Calculate ECEF to NED coordinate transformation matrix
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)
    c_e_n_matrix = np.matrix([[-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                              [-sin_lon,                cos_lon,             0],
                              [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]])

    if trig == 'yes':
        trig = np.array([cos_lat, sin_lat, cos_lon, sin_lon])
        return c_e_n_matrix, trig
    elif trig == 'no':
        return c_e_n_matrix

# End of ECEF to NED CTM Conversion


'''
    -----------------------------------------------------
    4. ECEF to ECI Coordinate Transformation Matrix (CTM)
    -----------------------------------------------------
'''


def ecef_to_eci_ctm(omega, approx_range, c_speed):

    # Calculate ECEF to ECI CTM
    c_e_i_matrix = np.matrix([[1, omega*approx_range/c_speed, 0],
                              [-omega*approx_range/c_speed, 1, 0],
                              [0, 0, 1]])

    return c_e_i_matrix

# End of ECEF to ECI CTM Conversion


'''
    ------------------------
    5. Skew-Symmetric Matrix
    ------------------------
'''


def skew_sym(vector):

    s_matrix = np.matrix(np.zeros((3, 3)))
    s_matrix[0, 1] = -vector[2, 0]
    s_matrix[0, 2] = vector[1, 0]
    s_matrix[1, 0] = vector[2, 0]
    s_matrix[1, 2] = -vector[0, 0]
    s_matrix[2, 0] = -vector[1, 0]
    s_matrix[2, 1] = vector[0, 0]

    return s_matrix

# End of Creating Skew Symmetric Matrix


'''
    --------------------------------------------------------------
    6. Convert Position, Velocity, and CTM in NED to Those in ECEF
    --------------------------------------------------------------
'''


def lla_to_ecef(lat_b, lambda_b, h_b, v_eb_n, c_b_n_matrix):

    # Calculate transverse radius of curvature
    r_ew = R_0 / np.sqrt(1 - (ecc_o * np.sin(lat_b)) ** 2)

    # Calculate ECEF to NED CTM using ECEF_to_NED_CTM()
    [c_e_n_matrix, trig] = ecef_to_ned_ctm(lat_b, lambda_b, trig='yes')

    # Convert position
    cos_lat = trig[0]
    sin_lat = trig[1]
    cos_lon = trig[2]
    sin_lon = trig[3]
    r_eb_e = np.matrix([[(r_ew + h_b) * cos_lat * cos_lon],
                        [(r_ew + h_b) * cos_lat * sin_lon],
                        [((1 - ecc_o ** 2) * r_ew + h_b) * sin_lat]])

    # Transform velocity
    v_eb_e = c_e_n_matrix.T * v_eb_n

    # Transform attitude
    c_b_e_matrix = c_e_n_matrix.T * c_b_n_matrix

    return r_eb_e, v_eb_e, c_b_e_matrix

# End of NED to ECEF Convertion


'''
    --------------------------------------------------------------
    7. Convert Position, Volocity, and CTM in ECEF to Those in NED
    --------------------------------------------------------------
'''


def ecef_to_lla(r_eb_e, v_eb_e, c_b_e_matrix):

    # Compute the Longitude is straight forward
    lambda_b = np.arctan2(r_eb_e[1, 0], r_eb_e[0, 0])

    # Convert position using Borkowski closed-form exact solution in order to avoid while loop never exits. If doing
    # this by iteration, we can't ensure while loop convergence. Refer to Appendix C (Paul Grove) or Borkowski, K.M.,
    # "Accurate Algorithms to Transform Geocentric to Geodetic Coordinates", Bull. Geod. 63, pp.50 - 56, 1989.
    k1 = np.sqrt(1 - ecc_o ** 2) * abs(r_eb_e[2, 0])
    k2 = (ecc_o ** 2) * R_0
    beta = np.sqrt(r_eb_e[0, 0] ** 2 + r_eb_e[1, 0] ** 2)
    e_term = (k1 - k2) / beta
    f_term = (k1 + k2) / beta
    p_term = (4 / 3.0) * (e_term * f_term + 1)
    q_term = 2 * (e_term ** 2 - f_term ** 2)
    d_term = p_term ** 3 + q_term ** 2
    v_term = (np.sqrt(d_term) - q_term) ** (1 / 3.0) - (np.sqrt(d_term) + q_term) ** (1 / 3.0)
    g_term = 0.5 * (np.sqrt(e_term ** 2 + v_term) + e_term)
    t_term = np.sqrt(g_term ** 2 + (f_term - v_term * g_term) / (2 * g_term - e_term)) - g_term
    lat_b = np.sign(r_eb_e[2, 0]) * np.arctan((1 - t_term ** 2) / (2 * t_term * np.sqrt(1 - ecc_o ** 2)))
    h_b = (beta - R_0 * t_term) * np.cos(lat_b) + (r_eb_e[2, 0] -
                                                np.sign(r_eb_e[2, 0]) * R_0 * np.sqrt(1 - ecc_o ** 2)) * np.sin(lat_b)

    # Calculate ECEF to NED coordinate transformation matrix
    c_e_n_matrix = ecef_to_ned_ctm(lat_b, lambda_b, trig='no')

    # Transform velocity
    v_eb_n = c_e_n_matrix * v_eb_e

    # Transform attitude
    c_b_n_matrix = c_e_n_matrix * c_b_e_matrix

    return lat_b, lambda_b, h_b, v_eb_n, c_b_n_matrix

# End of Converting Position, Velocity, and CTM from ECEF to NED


'''
    ---------------------------------
    8. Convert Position in LLA to XYZ
    ---------------------------------
'''


def lla_to_xyz(lat_b, lambda_b, h_b):

    # Calculate transverse radius of curvature
    r_ew = R_0 / np.sqrt(1 - (ecc_o * np.sin(lat_b)) ** 2)

    # Calculate ECEF to NED CTM using ECEF_to_NED_CTM()
    [c_e_n_matrix, trig] = ecef_to_ned_ctm(lat_b, lambda_b, trig='yes')

    # Convert position
    cos_lat = trig[0]
    sin_lat = trig[1]
    cos_lon = trig[2]
    sin_lon = trig[3]
    r_eb_e = np.matrix([[(r_ew + h_b) * cos_lat * cos_lon],
                        [(r_ew + h_b) * cos_lat * sin_lon],
                        [((1 - ecc_o ** 2) * r_ew + h_b) * sin_lat]])

    return r_eb_e

# End of LLA to XYZ


'''
    ---------------------------------
    9. Convert Position in XYZ to NED
    ---------------------------------
'''


def xyz_to_ned(r_eb_e, lat_b_ref, lambda_b_ref, h_b_ref):

    # Convert referenced position in LLA to ECEF
    r_eb_e_ref = lla_to_xyz(lat_b_ref, lambda_b_ref, h_b_ref)

    # Compute the relative position vector in ECEF
    delta_r_eb_e = r_eb_e - r_eb_e_ref

    # Calculate ECEF to NED CTM using ECEF_to_NED_CTM()
    c_e_n_matrix = ecef_to_ned_ctm(lat_b_ref, lambda_b_ref, trig='no')

    # Convert the relative position vector in ECEF to NED
    r_eb_ned = c_e_n_matrix*delta_r_eb_e

    return r_eb_ned

# End of XYZ to NED

'''
    --------------------------------------------------------------
    10. Convert Position, Volocity, and CTM in ECEF to Those in NED
    --------------------------------------------------------------
'''


def ecef_to_ned_ekfsd(lat_b_ref, lambda_b_ref, h_b_ref, rsd_eb_e, vsd_eb_e, c_b_e_matrix):

    # Transform position using xyz_to_ned
    rsd_eb_n = xyz_to_ned(rsd_eb_e, lat_b_ref, lambda_b_ref, h_b_ref)

    # Calculate ECEF to NED coordinate transformation matrix
    c_e_n_matrix = ecef_to_ned_ctm(lat_b_ref, lambda_b_ref, trig='no')

    # Transform velocity using (2.73)
    vsd_eb_n = c_e_n_matrix * vsd_eb_e

    # Transform attitude using (2.15)
    c_b_n_matrix = c_e_n_matrix * c_b_e_matrix

    return rsd_eb_n, vsd_eb_n, c_b_n_matrix

# End of Converting Position, Velocity, and CTM from ECEF to NED


'''
    -------------------------------------------------
    11. Convert Position and Velocity from ECEF to NED
    -------------------------------------------------
'''


def pv_ecef_to_lla(r_eb_e, v_eb_e):

    # Compute the Longitude
    lambda_b = np.arctan2(r_eb_e[1, 0], r_eb_e[0, 0])

    # Convert position using Borkowski closed-form exact solution in order to avoid while loop never exits. If doing
    # this by iteration, we can't ensure while loop convergence. Refer to Appendix C (Paul Grove) or Borkowski, K.M.,
    # "Accurate Algorithms to Transform Geocentric to Geodetic Coordinates", Bull. Geod. 63, pp.50 - 56, 1989.
    k1 = np.sqrt(1 - ecc_o ** 2) * abs(r_eb_e[2, 0])
    k2 = (ecc_o ** 2) * R_0
    beta = np.sqrt(r_eb_e[0, 0] ** 2 + r_eb_e[1, 0] ** 2)
    e_term = (k1 - k2) / beta
    f_term = (k1 + k2) / beta
    p_term = (4 / 3.0) * (e_term * f_term + 1)
    q_term = 2 * (e_term ** 2 - f_term ** 2)
    d_term = p_term ** 3 + q_term ** 2
    v_term = (np.sqrt(d_term) - q_term) ** (1 / 3.0) - (np.sqrt(d_term) + q_term) ** (1 / 3.0)
    g_term = 0.5 * (np.sqrt(e_term ** 2 + v_term) + e_term)
    t_term = np.sqrt(g_term ** 2 + (f_term - v_term * g_term) / (2 * g_term - e_term)) - g_term
    lat_b = np.sign(r_eb_e[2, 0]) * np.arctan((1 - t_term ** 2) / (2 * t_term * np.sqrt(1 - ecc_o ** 2)))
    h_b = (beta - R_0 * t_term) * np.cos(lat_b) + (r_eb_e[2, 0] - np.sign(r_eb_e[2, 0]) * R_0 *
                                                   np.sqrt(1 - ecc_o ** 2)) * np.sin(lat_b)

    # Calculate ECEF to NED coordinate transformation matrix
    c_e_n_matrix = ecef_to_ned_ctm(lat_b, lambda_b, trig='no')

    # Transform velocity
    v_eb_n = c_e_n_matrix * v_eb_e

    return lat_b, lambda_b, h_b, v_eb_n

# End of Converting Position an Velocity from ECEF to NED


'''
    ------------------------------
    12. Initialized Attitude in NED
    ------------------------------
'''


def init_ned_att(c_b_n_matrix, eul_err_nb_n):

    # Attitude initialization
    delta_c_b_n_matrix = euler_to_ctm(-eul_err_nb_n)
    est_c_b_n_matrix = delta_c_b_n_matrix * c_b_n_matrix

    return est_c_b_n_matrix

# End of Initializing Attitude in NED


'''
    ------------------------------------------
    13. Calculate the Radii of Earth Curvature
    ------------------------------------------
'''


def radii_of_curv(latitude):

    # Calculate meridian radius of curvature
    temp = 1 - (ecc_o * np.sin(latitude)) ** 2
    r_ns = R_0 * (1 - ecc_o ** 2) / (temp ** 1.5)

    # Calculate transverse radius of curvature
    r_ew = R_0 / np.sqrt(temp)

    return r_ns, r_ew

# End of Calculate Radii of Earth Curvature


'''
    ------------------------------------------------------------
    14. Progress Bar: Displays or Updates a Console Progress Bar
    ------------------------------------------------------------
'''


def progressbar(progress):

    # Accepts "progress" as a float percentage between 0 and 1.
    barlength = 25  # Modify this to change the length of the progress bar
    status = " "
    block = int(round(barlength * progress))
    text = "\r NavSim: [{0}] {1}% {2}".format(">" * block + "-" * (barlength - block), int(round(progress * 100)),
                                              status)
    sys.stdout.write(text)
    sys.stdout.flush()

# End of Progress Bar


'''
    ----------------------------------------------------------
    15. Calculate the Earth Gravitational Force Vector in ECEF
    ----------------------------------------------------------
'''


def gravity_ecef(r_eb_e):

    # Calculate distance from center of the Earth
    mag_r = np.sqrt(r_eb_e.T * r_eb_e)

    # If the input position is [0,0,0], produce a dummy output
    if mag_r == 0:
        gravity_vec = np.matrix(np.zeros((3, 1)))

    else:
        # Calculate gravitational acceleration
        gravity_vec = np.nan * np.matrix(np.ones((3, 1)))
        gamma = np.nan * np.matrix(np.ones((3, 1)))
        z_scale = 5.0 * (r_eb_e[2, 0] / mag_r) ** 2
        gamma[0, 0] = (-mu / mag_r ** 3) * (r_eb_e[0.0] + 1.5 * J_2 * (R_0 / mag_r) ** 2 * (1.0 - z_scale) *
                                            r_eb_e[0, 0])
        gamma[1, 0] = (-mu / mag_r ** 3) * (r_eb_e[1.0] + 1.5 * J_2 * (R_0 / mag_r) ** 2 * (1.0 - z_scale) *
                                            r_eb_e[1, 0])
        gamma[2, 0] = (-mu / mag_r ** 3) * (r_eb_e[2.0] + 1.5 * J_2 * (R_0 / mag_r) ** 2 * (1.0 - z_scale) *
                                            r_eb_e[2, 0])

        # Add centripetal acceleration
        gravity_vec[0:2, 0] = gamma[0:2, 0] + OMEGA_ie ** 2 * r_eb_e[0:2, 0]
        gravity_vec[2, 0] = gamma[2, 0]

    return gravity_vec

# End of Calculating Earth Gravitation Force in ECEF


'''
    -------------------------------------------
    16. Earth Rotation Over the Update Interval
    -------------------------------------------
'''


def c_earth(tau_i):

    # Determine the Earth rotation over the update interval
    alpha_ie = OMEGA_ie * tau_i
    c_earth_matrix = np.matrix([[np.cos(alpha_ie), np.sin(alpha_ie), 0.0],
                                [-np.sin(alpha_ie), np.cos(alpha_ie), 0.0],
                                [0, 0, 1]])

    return c_earth_matrix, alpha_ie

# End of Calculating the Earth Rotational Matrix


'''
    -----------------------------------------------------------------------
    17. Solve Kepler's Equation for Eccentric Anomaly Using Newton's Method
    -----------------------------------------------------------------------
'''


def kepler(Mk, ecc, tol):

    # Determine the initial guess for eccentric anomaly, Ek
    # From Paul Grove
    Ek = Mk + (ecc*np.sin(Mk))/(1.0 - np.sin(Mk + ecc) + np.sin(Mk))

    # From Howard Curtis
    # if Mk < np.pi:
    #     Ek = Mk + ecc/2.0
    # else:
    #     Ek = Mk - ecc/2.0

    # Define ratio = f(Ei)/f'(Ei) to be the conditioner
    # Hence, ratio = (E_i - e*sin(E_i) - M)/(1 - e*cos(E_i))
    ratio = 1
    # Iterate over E_(i+1) = E_i - (E_i - e*sin(E_i) - M)/(1 - e*cos(E_i))
    while abs(ratio) > tol:
        ratio = (Ek - ecc*np.sin(Ek) - Mk)/(1.0 - ecc*np.cos(Ek))
        Ek -= ratio
    # End of while

    return Ek

# End of Kepler's equation solver


'''
    ---------------------------
    18. Determine the Leap Year
    ---------------------------
'''


def is_leap_year(yyyy):

    # Function is_leap_year() determines whether a given year "yyyy" is a leap year.
    # A leap year year is defined as a year that is divisible by 4 but not by 100,
    # unless it is divisible by 400 as every 100 years a leap year is skipped.
    if yyyy % 4 == 0:
        if yyyy % 100 == 0:
            if yyyy % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

# End of leap year checker


'''
    --------------------------------------------------
    19. Calculate the Number of Days between Two Dates
    --------------------------------------------------
'''


def num_days(yyyy1, mm1, dd1, yyyy2, mm2, dd2):

    # Function numdays() calculates the number of days between two given dates.
    # Date #1: year = yyyy1, month = mm1, and day = dd1
    # Date #2: year = yyyy2, month = mm2, and day = dd2
    # Usually, date #2 is after date #1 counting from the past to the future.

    # Cumulative days by month for a non-leap year (up to the beginning of the month)
    cum_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    # Cumulative days by month for a leap year (up to the beginning of the month)
    leap_cum_days = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

    # Total days
    totdays = 0

    # Checking for the same year (if the two dates are in the same year).
    if yyyy1 == yyyy2:
        # If the two dates are in the same year, check for the leap year
        if is_leap_year(yyyy1):
            # If the year is a leap year, count the cumulative days for each date by
            # using the cumulative days in the year for a leap year.
            # Then calculate the difference in the number of days of year between them.
            return (leap_cum_days[mm2 - 1] + dd2) - (leap_cum_days[mm1 - 1] + dd1)
        else:
            # If the year is not a leap year, count the cumulative days for each date by
            # using the cumulative days in the year for a non-leap year.
            return (cum_days[mm2 - 1] + dd2) - (cum_days[mm1 - 1] + dd1)
    else:
        # If the two years are different, check for leap year on year #1
        if is_leap_year(yyyy1):
            # A leap year has 366 days.
            totdays = totdays + 366 - (leap_cum_days[mm1 - 1] + dd1)
        else:
            # A non-leap year has 365 days.
            totdays = totdays + 365 - (cum_days[mm1 - 1] + dd1)
        # Counting the number of years different between year #1 and year #2.
        # Increase year #1 by 1 unit,
        year = yyyy1 + 1
        # check if the incremented year is equal to the year #2
        while year < yyyy2:
            # For each leap year in difference,
            if is_leap_year(year):
                # the total days is accumulated by 366 days
                totdays += 366
            # Otherwise,
            else:
                # the total days is accumulated by 365 days
                totdays += 365
            # Increase year #1 by another unit
            year += 1
        # Check for leap year on year #2
        if is_leap_year(yyyy2):
            # If year #2 is a leap year, the total days equals days of year for date #1
            # plus number of days for number of years in difference between two years,
            # then plus days of year for date #2 by cumulative days of a leap year.
            totdays = totdays + (leap_cum_days[mm2 - 1] + dd2)
        else:
            # Otherwise, the total days equals days of year for date #1 plus number of
            # days for number of years in difference between two years, then plus days
            # of year for date #2 by cumulative days of a non-leap year.
            totdays = totdays + (cum_days[mm2 - 1] + dd2)
        return totdays
# End of days between two dates


'''
    ------------------------------------------------------
    20. Calculate the Number of Days of the Year of a Date
    ------------------------------------------------------
'''


def days_of_year(yyyy, mm, dd):

    # Cumulative days by month
    cum_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    # Cumulative days by month for leap year
    leap_cum_days = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

    # Check for leap year on year (yyyy)
    if is_leap_year(yyyy):
        # If yyyy is a leap year, then count the cumulative days of year by leap year.
        return leap_cum_days[mm - 1] + dd
    else:
        # Otherwise, count the cumulative days of year by non-leap year.
        return cum_days[mm - 1] + dd

# End of days of year of a date


'''
    --------------------------------------------------------
    21. Convert GPS TOW to UTC Time (Day/Hour/Minute/Second)
    --------------------------------------------------------
'''


def tow_to_utc(tow):

    # Days of week based on the initial time of week
    DyOW = (tow - np.mod(tow, dd2sec)) / dd2sec  # days

    # The remaining seconds of the day after extracting the days
    rem_sec = np.mod(tow, dd2sec)  # seconds

    # Hours of the day
    HrOD = (rem_sec - np.mod(rem_sec, hr2sec)) / hr2sec  # hours

    # The remaining seconds of the hour after extracting days and hours
    rem_sec = np.mod(rem_sec, hr2sec)  # seconds

    # Minutes of the hour
    MnOH = (rem_sec - np.mod(rem_sec, mn2sec)) / mn2sec  # minutes

    # The remaining seconds of the minute after extracting days, hours, and minutes
    ScOM = np.mod(rem_sec, mn2sec)

    return DyOW, HrOD, MnOH, ScOM

# End of TOW to UTC


'''
    ----------------------------------------------------------------
    22. Determine Flgith Duration Based on TOWs from the Flight Data
    ----------------------------------------------------------------
'''


def flight_duration(towo, towf):

    # Flight duration in GPS time (seconds)
    delta_tow = towf - towo

    # Flight duration in minutes
    deltaMnOW = (delta_tow - np.mod(delta_tow, mn2sec)) / mn2sec

    # Remaining flight duration in seconds
    rem_deltaTOW = np.mod(delta_tow, mn2sec)

    return deltaMnOW, rem_deltaTOW

# End of Flight Duration


'''
========================================================================================================================
                                                    MAIN FUNCTIONS
========================================================================================================================
    ----------------------------------------------------------------------------------
    1. Generate Satellite Positions and Velocities by Simulating Virtual Constellation
    ----------------------------------------------------------------------------------
'''


def sat_pv_sim(t_i, gnss_config):

    # Convert inclination angle to degrees
    inclination = d2r * gnss_config.inclination

    # Determine orbital angular rate
    omega_is = np.sqrt(mu / (gnss_config.r_os ** 3))

    # Determine constellation time
    tgps = t_i + gnss_config.const_t_offset

    # Allocate the position and the velocity arrays for all satellites
    sat_r_es_e = np.nan * np.matrix(np.ones((gnss_config.no_sat, 3)))
    sat_v_es_e = np.nan * np.matrix(np.ones((gnss_config.no_sat, 3)))

    # Loop over the satellites
    for i in xrange(0, gnss_config.no_sat):

        # Corrected argument of latitude
        u_os_o = 2 * np.pi * i / gnss_config.no_sat + omega_is * tgps

        # Satellite position in the orbital plane
        r_os_o = gnss_config.r_os * np.matrix([np.cos(u_os_o), np.sin(u_os_o), 0]).T

        # Longitude of the ascending node
        omega = (np.pi * np.mod(i + 1, 6) / 3) + d2r * gnss_config.const_delta_lambda - OMEGA_ie * tgps

        # ECEF Satellite Position
        sat_r_es_e[i, 0] = r_os_o[0, 0] * np.cos(omega) - r_os_o[1, 0] * np.cos(inclination) * np.sin(omega)
        sat_r_es_e[i, 1] = r_os_o[0, 0] * np.sin(omega) + r_os_o[1, 0] * np.cos(inclination) * np.cos(omega)
        sat_r_es_e[i, 2] = r_os_o[1, 0] * np.sin(inclination)

        # Satellite velocity in the orbital frame, noting that with a circular orbit r_os_o is constant and
        # the time derivative of u_os_o is omega_is.
        v_os_o = gnss_config.r_os * omega_is * np.matrix([-np.sin(u_os_o), np.cos(u_os_o), 0]).T

        # ECEF Satellite velocity
        sat_v_es_e[i, 0] = v_os_o[0, 0] * np.cos(omega) - v_os_o[1, 0] * np.cos(inclination) * np.sin(omega) + \
                           (OMEGA_ie * sat_r_es_e[i, 1])
        sat_v_es_e[i, 1] = v_os_o[0, 0] * np.sin(omega) + v_os_o[1, 0] * np.cos(inclination) * np.cos(omega) - \
                           (OMEGA_ie * sat_r_es_e[i, 0])
        sat_v_es_e[i, 2] = v_os_o[1, 0] * np.sin(inclination)

    # End of For Loop

    return sat_r_es_e, sat_v_es_e

# End of Generating Satellite Positions and Velocities


'''
    -----------------------------------------------------------------
    2. Generate Satellite Positions and Velocities from the Ephemeris
    -----------------------------------------------------------------
'''


def sat_pv_ephem(ephem, gnss_config, t_i, tol):

    # Ephemeris Array Structure:
    # Col. 0: PRN
    # Col. 1: Crs                           (meter)
    # Col. 2: Delta n                       (rad/sec)
    # Col. 3: Mo                            (rad)
    # Col. 4: Cuc                           (rad)
    # Col. 5: Eccentricity, e
    # Col. 6: Cus                           (rad)
    # Col. 7: sqrt(A)                       (sqrt(meter))
    # Col. 8: Toe, time of ephemeris        (sec of GPS Week)
    # Col. 9: Cic                           (rad)
    # Col. 10: OMEGA_o                      (rad)
    # Col. 11: Cis                          (rad)
    # Col. 12: i_o, reference inclination   (rad)
    # Col. 13: Crc                          (meter)
    # Col. 14: omega                        (rad)
    # Col. 15: OMEGA DOT                    (rad/sec)
    # Col. 16: IDOT, inclination rate       (rad/sec)
    # Col. 17: Toc, time of clock           (sec of GPS Week)
    # Col. 18: TGD, group delay             (sec)

    # Determine constellation time
    tgps = t_i + gnss_config.const_t_offset

    # Create frames for the position and the velocity arrays of all the satellites in the constellation
    sat_r_es_e = np.nan * np.matrix(np.ones((gnss_config.no_sat, 3)))
    sat_v_es_e = np.nan * np.matrix(np.ones((gnss_config.no_sat, 3)))

    # Loop over all satellites
    for i in xrange(0, gnss_config.no_sat):
        # Compute the semi-major axis
        A = ephem[i, 7]**2

        # Compute the mean motion (rd/sec)
        n_o = np.sqrt(mu/(A**3))

        # Calculate the time from ephemeris reference epoch
        tk = tgps - ephem[i, 8]

        # Correct the beginning or end of week crossovers
        if tk > 3.024E+05:
            tk -= 6.048E+05
        elif tk < -3.024E+05:
            tk += 6.048E+05

        # Correct the mean motion
        n = n_o + ephem[i, 2]

        # Determine the mean anomaly
        Mk = ephem[i, 3] + n*tk

        # Solve Kepler's Equation for eccentric anomaly
        ecc = ephem[i, 5]
        Ek = kepler(Mk, ecc, tol)

        # Determine the true anomaly
        sin_nuk = np.sqrt(1.0 - ecc)*np.sin(Ek)/(1.0 - ecc*np.cos(Ek))
        cos_nuk = (np.cos(Ek) - ecc)/(1.0 - ecc*np.cos(Ek))
        nuk = np.arctan2(sin_nuk, cos_nuk)

        # Calculate the argument of latitude
        Phik = nuk + ephem[i, 14]

        # Compute the 2nd harmonic perturbations
        delt_uk = ephem[i, 6]*np.sin(2.0*Phik) + ephem[i, 4]*np.cos(2.0*Phik)    # Argument of latitude correction
        delt_rk = ephem[i, 1]*np.sin(2.0*Phik) + ephem[i, 13]*np.cos(2.0*Phik)   # Radius correction
        delt_ik = ephem[i, 11]*np.sin(2.0*Phik) + ephem[i, 9]*np.cos(2.0*Phik)   # Inclination correction

        # Corrected argument of latitude
        uk = Phik + delt_uk

        # Corrected radius
        rk = A*(1 - ecc*np.cos(Ek)) + delt_rk

        # Corrected inclination
        ik = ephem[i, 12] + delt_ik + ephem[i, 16]*tk

        # Satellite's position in orbital plane
        xk_prime = rk*np.cos(uk)        # x-component
        yk_prime = rk*np.sin(uk)        # y-component

        # Corrected longitude of ascending node
        OMEGAk = ephem[i, 10] + (ephem[i, 15] - OMEGA_ie)*tk - OMEGA_ie*ephem[i, 8]

        # Satellite's position in ECEF
        sat_r_es_e[i, 0] = xk_prime*np.cos(OMEGAk) - yk_prime*np.cos(ik)*np.sin(OMEGAk)     # x-component
        sat_r_es_e[i, 1] = xk_prime*np.sin(OMEGAk) + yk_prime*np.cos(ik)*np.cos(OMEGAk)     # y-component
        sat_r_es_e[i, 2] = yk_prime*np.sin(ik)                                              # z-component

        # Calculate the time derivative of the eccentric anomaly
        Ek_dot = n/(1.0 - ecc*np.cos(Ek))

        # Compute the time derivative of the argument of latitude
        Phik_dot = (np.sin(nuk)/np.sin(Ek))*Ek_dot

        # Determine the time derivative of the corrected radius
        rk_dot = A*ecc*np.sin(Ek)*Ek_dot + 2.0*(ephem[i, 1]*np.cos(2.0*Phik) - ephem[i, 13]*np.sin(2.0*Phik))*Phik_dot

        # Determine the time derivative of the corrected argument of latitude
        uk_dot = (1.0 + 2.0*ephem[i, 6]*np.cos(2.0*Phik) - 2.0*ephem[i, 4]*np.sin(2.0*Phik))*Phik_dot

        # Satellite's velocity in orbital plane
        xk_prime_dot = rk_dot*np.cos(uk) - rk*uk_dot*np.sin(uk)       # x-component
        yk_prime_dot = rk_dot*np.sin(uk) + rk*uk_dot*np.cos(uk)       # y-component

        # Calculate the time derivative of corrected longitude of ascending node
        OMEGAk_dot = ephem[i, 15] - OMEGA_ie

        # Calculate the time derivative of corrected inclination
        ik_dot = ephem[i, 16] + 2.0*(ephem[i, 11]*np.cos(2.0*Phik) - ephem[i, 9]*np.sin(2.0*Phik))*Phik_dot

        # Satellite's velocity in ECEF
        vx_term1 = xk_prime_dot*np.cos(OMEGAk) - yk_prime_dot*np.cos(ik)*np.sin(OMEGAk) + \
                   ik_dot*yk_prime*np.sin(ik)*np.sin(OMEGAk)

        vx_term2 = -OMEGAk_dot*(xk_prime*np.sin(OMEGAk) + yk_prime*np.cos(ik)*np.cos(OMEGAk))

        sat_v_es_e[i, 0] = vx_term1 + vx_term2                                      # x-component

        vy_term1 = xk_prime_dot*np.sin(OMEGAk) + yk_prime_dot*np.cos(ik)*np.cos(OMEGAk) - \
                  ik_dot*yk_prime*np.sin(ik)*np.cos(OMEGAk)

        vy_term2 = -OMEGAk_dot*(-xk_prime*np.cos(OMEGAk) + yk_prime*np.cos(ik)*np.sin(OMEGAk))

        sat_v_es_e[i, 1] = vy_term1 + vy_term2                                      # y-component

        sat_v_es_e[i, 2] = yk_prime_dot*np.sin(ik) + ik_dot*yk_prime*np.cos(ik)     # z-component

    return sat_r_es_e, sat_v_es_e

# End of Determine Satellite Position and Velocity in ECEF Using Ephemeris


'''
    -----------------------------------------------------------------
    3. Initialize GNSS Biases by Simulating the Virtual Constellation
    -----------------------------------------------------------------
'''


def init_gnss_bias_sim(sat_r_es_e, r_ea_e, lat_a, lambda_a, gnss_config):

    # Calculate ECEF to NED CTM using ECEF_to_NED_CTM()
    ctm_e_n = ecef_to_ned_ctm(lat_a, lambda_a, trig='no')

    # Loop satellites
    gnss_biases = np.nan * np.matrix(np.ones((gnss_config.no_sat, 1)))

    for i in xrange(0, gnss_config.no_sat):

        # Determine ECEF line-of-sight vector
        delta_r = sat_r_es_e[i, 0:3].T - r_ea_e
        u_as_e = delta_r / np.sqrt(delta_r.T * delta_r)

        # Convert line-of-sight vector to NED and determine elevation
        elevation = -np.arcsin(ctm_e_n[2, :] * u_as_e)

        # Limit the minimum elevation angle to the masking angle
        elevation = max(elevation, d2r * gnss_config.mask_angle)

        # Calculate ionosphere and troposphere error SDs
        iono_sd = gnss_config.zenith_iono_err_SD / np.sqrt(1 - 0.899 * np.cos(elevation) ** 2)
        trop_sd = gnss_config.zenith_trop_err_SD / np.sqrt(1 - 0.998 * np.cos(elevation) ** 2)

        # Determine range bias
        gnss_biases[i, 0] = gnss_config.SIS_err_SD * rnd.randn() + iono_sd * rnd.randn() + trop_sd * rnd.randn()

    # End of For Loop

    return gnss_biases
# End of Initializing GNSS Biases


'''
    --------------------------------------------------------------
    4. Initialize GNSS Biases: Ionospheric and Tropospheric Delays
    --------------------------------------------------------------
'''


def init_gnss_bias_ephem(t_i, doy, ephem, iono_alpha, iono_beta, lat_a, lon_a, alt_a, sat_r_es_e, r_ea_e, gnss_config):

    # Calculate ECEF to NED CTM using ECEF_to_NED_CTM()
    ctm_e_n = ecef_to_ned_ctm(lat_a, lon_a, trig='no')

    # Determine constellation time
    tgps = t_i + gnss_config.const_t_offset

    # Loop satellites
    gnss_biases = np.nan * np.matrix(np.ones((gnss_config.no_sat, 1)))

    for i in xrange(0, gnss_config.no_sat):

        # Determine ECEF line-of-sight vector
        delta_r = sat_r_es_e[i, 0:3].T - r_ea_e
        u_as_e = delta_r / np.sqrt(delta_r.T * delta_r)

        # Convert line-of-sight vector to NED and determine the elevation and the azimuth
        elevation = -np.arcsin(ctm_e_n[2, :] * u_as_e)
        azimuth = np.arctan2(ctm_e_n[1, :] * u_as_e, ctm_e_n[0, :] * u_as_e)

        # Limit the minimum elevation angle to the masking angle
        elevation = max(elevation, d2r * gnss_config.mask_angle)

        # --------------------------------------------------------
        # Calculate the ionospheric bias using the Klobuchar model
        # --------------------------------------------------------

        # 1.1 Compute the Earth-central angle and the sub-ionospheric latitude
        Psi_E_s = (0.0137*(np.pi**2))/(elevation + 0.11*np.pi) - 0.022*np.pi        # (rad)
        L_I_s = lat_a + Psi_E_s*np.cos(azimuth)                                     # (rad)
        # 1.2 Apply limits to L_I_s, [-1.307, 1.307] rad
        if L_I_s < -1.307:
            L_I_s  = -1.307
        elif L_I_s > 1.307:
            L_I_s = 1.307

        # 2. Calculate the sub-ionospheric longitude and the geomagnetic latitude
        lambda_I_s = lon_a + Psi_E_s*np.sin(azimuth)/np.cos(L_I_s)                  # (rad)
        L_m_s = L_I_s + (0.064*np.pi)*np.cos(lambda_I_s - 1.617*np.pi)              # (rad)

        # 3.1 Compute the time of flight, delta_t = t_gps - t_oe
        delt_t_s = tgps - ephem[i, 8]                                               # (sec)

        # 3.2 Apply limits to delta_t_s, [0, 86400) sec
        if delt_t_s > 8.64E+04:
            delt_t_s -= 8.64E+04
        elif delt_t_s <= 0:
            delt_t_s += 8.64E+04
        # 3.3 Compute the time at the sub-ionopsheric point
        t_I_s = delt_t_s + ((4.32E+04)/np.pi)*lambda_I_s                            # (sec)

        # 4.1 Calculate PER
        PER = 0.0
        for n in xrange(0, 4):
            PER += iono_beta[0, n]*(L_m_s/np.pi)**n                                 # (sec)
        # 4.2 Apply minimum limit to PER
        if PER < 7.2E+04:
            PER = 7.2E+04                                                           # (sec)
        # 4.3 Calculate the phase X_s, as the ionospheric formula's conditioner
        X_s = 2*np.pi*(t_I_s - 5.04E+04)/PER

        # 5. Estimate the ionosphere propagation delay (for GPS L1 band signal)
        # 5.1 Compute the obliquity factor
        F = 1.0 + (16.0/(np.pi**3))*(0.53*np.pi - elevation)**3
        # 5.2 Condition 1 on X
        if np.abs(X_s) >= 1.57:
            iono_delay = 5.0E-09*F                                                              # (sec)
        # 5.3 Condition 2 on X
        elif np.abs(X_s) < 1.57:
            # 5.3.1 Compute AMP
            AMP = 0.0
            for n in xrange(0, 4):
                AMP += iono_alpha[0, n]*(L_m_s/np.pi)**n                                        # (sec)
            # 5.3.2 Apply minimum limit to AMP
            if AMP < 0.0:
                AMP = 0.0                                                                       # (sec)
            # 5.3.3 Compute the ionospheric delay
            iono_delay = F*(5.0E-09 + AMP*(1 - (X_s**2)/2.0 + (X_s**4)/24.0))                   # (sec)
        # Estimate the ionospheric psuedorange bias, (m)
        iono_bias = iono_delay*c

        # -----------------------------------------------------
        # Calculate the trophospheric bias using the WAAS Model
        # -----------------------------------------------------

        # Estimate the surface refractivity based on the user's altitude and latitude
        # User's in the Northen hemisphere
        if lat_a > 0.0:
            cosine1 = np.cos(2.0 * np.pi * ((doy - 152) / 365))
            cosine2 = np.cos(2.0 * np.pi * ((doy - 213) / 365))
            delta_N = 3.61E-03 * alt_a * cosine1 + (0.1 * cosine2 - 0.8225) * np.abs(lat_a)
        # User's in the Southern hemisphere
        elif lat_a < 0.0:
            cosine1 = np.cos(2.0 * np.pi * ((doy - 335) / 365))
            cosine2 = np.cos(2.0 * np.pi * ((doy - 30) / 365))
            delta_N = 3.61E-03 * alt_a * cosine1 + (0.1 * cosine2 - 0.8225) * np.abs(lat_a)
        # User's lower than 1,500 m
        den = np.sin(elevation + 6.11E-03)
        if alt_a <= 1.5E+03:
            num = 2.506*(1.0 + 1.25E-03*delta_N)*(1.0 - 1.264E-04*alt_a)
            tropo_bias = num/den    # (m)
        # User's higher than 1,500 m
        elif alt_a > 1.5E+03:
            num = 2.484*(1.0 + 1.5363E-03*np.exp(-2.133E-04*alt_a)*delta_N)*np.exp(-1.509E-04*alt_a)
            tropo_bias = num/den    # (m)

        # Assemble the ionopsheric bias and the tropospheric bias into the initial GNSS bias matrix
        gnss_biases[i, 0] = iono_bias + tropo_bias

    # End of "for" loop

    return gnss_biases
# End of Initializing GNSS Biases


'''
    -----------------------------------------------------------------------------------------------------
    5. Generate GNSS Measurements (pseudorange, range rate, etc.) by Simulating the Virtual Constellation
    -----------------------------------------------------------------------------------------------------
'''


def gnss_meas_gen_sim(t, sat_r_es_e, sat_v_es_e, r_ea_e, lat_a, lon_a, v_ea_e, gnss_biases, gnss_config):

    # Allocate necessary place holders before generating GNSS measurements
    no_gnss_meas = 0
    prn = []
    pseudorange = []
    pseudorange_rate = []
    rx = []
    ry = []
    rz = []
    vx = []
    vy = []
    vz = []

    # Calculate ECEF to NED CTM using ECEF_to_NED_CTM()
    ctm_e_n = ecef_to_ned_ctm(lat_a, lon_a, trig='no')

    # Skew symmetric matrix of Earth rate
    omega_ie_matrix = skew_sym(np.matrix([[0], [0], [OMEGA_ie]]))

    # Loop over all satellites
    for i in xrange(0, gnss_config.no_sat):

        # Determine ECEF line-of-sight vector
        delta_r = sat_r_es_e[i, :].T - r_ea_e
        approx_range = np.sqrt(delta_r.T * delta_r)
        u_as_e = delta_r / approx_range

        # Convert line-of-sight vector to NED and determine elevation
        elevation = -np.arcsin(ctm_e_n[2, :] * u_as_e)

        # Determine if satellite is above the masking angle
        if elevation >= d2r * gnss_config.mask_angle:

            # Record the PRN
            prn.append(i + 1)

            # Increment number of measurements
            no_gnss_meas += 1

            # Calculate frame rotation during signal transit time
            ctm_e_i = ecef_to_eci_ctm(OMEGA_ie, approx_range, c)

            # Calculate geometric range
            delta_r = ctm_e_i * sat_r_es_e[i, :].T - r_ea_e
            rangex = np.sqrt(delta_r.T * delta_r)

            # Calculate geometric range rate
            range_rate = u_as_e.T * (ctm_e_i * (sat_v_es_e[i, :].T + omega_ie_matrix * sat_r_es_e[i, :].T) -
                                     (v_ea_e + omega_ie_matrix * r_ea_e))

            # Calculate pseudo-range measurement
            psdrel1 = gnss_config.rx_clock_offset
            psdrel2 = gnss_config.rx_clock_drift * t
            psdrel3 = gnss_config.code_track_err_SD * rnd.randn()
            psdr = rangex + gnss_biases[i] + psdrel1 + psdrel2 + psdrel3
            pseudorange.append(psdr[0, 0])

            # Calculate pseudo-range rate measurement
            rateel1 = gnss_config.rx_clock_drift
            rateel2 = gnss_config.rate_track_err_SD * rnd.randn()
            psdr_rate = range_rate + rateel1 + rateel2
            pseudorange_rate.append(psdr_rate[0, 0])

            # Append satellite position and velocity to output data
            rx.append(sat_r_es_e[i, 0])
            ry.append(sat_r_es_e[i, 1])
            rz.append(sat_r_es_e[i, 2])
            vx.append(sat_v_es_e[i, 0])
            vy.append(sat_v_es_e[i, 1])
            vz.append(sat_v_es_e[i, 2])

            # End of "If" Statement to Determine the Number of Satellites in View

    # End of For Loop Sweeping Through All Satellites in the Constellation

    # Forming the GNSS Measurement Output matrix
    gnss_meas = np.matrix([pseudorange, pseudorange_rate, rx, ry, rz, vx, vy, vz]).T

    return gnss_meas, no_gnss_meas, prn

# End of GNSS Measurement Generation


'''
    --------------------------------------------------------------------------------
    6. Generate GNSS Measurements (pseudorange, range rate, etc.) from the Ephemeris
    --------------------------------------------------------------------------------
'''


def gnss_meas_gen_ephem(t_i, sat_r_es_e, sat_v_es_e, r_ea_e, lat_a, lon_a, v_ea_e, gnss_biases, gnss_config, sv_clock,
                        ephem, alma_t_para):

    # Allocate necessary place holders before generating GNSS measurements
    no_gnss_meas = 0
    prn = []
    pseudorange = []
    pseudorange_rate = []
    rx = []
    ry = []
    rz = []
    vx = []
    vy = []
    vz = []

    # Calculate ECEF to NED CTM using ECEF_to_NED_CTM()
    ctm_e_n = ecef_to_ned_ctm(lat_a, lon_a, trig='no')

    # Determine constellation time
    tgps = t_i + gnss_config.const_t_offset

    # Skew symmetric matrix of Earth rate
    omega_ie_matrix = skew_sym(np.matrix([[0], [0], [OMEGA_ie]]))

    # Loop over all satellites
    for i in xrange(0, gnss_config.no_sat):
        # Determine ECEF line-of-sight vector
        delta_r = sat_r_es_e[i, :].T - r_ea_e
        approx_range = np.sqrt(delta_r.T * delta_r)
        u_as_e = delta_r / approx_range

        # Convert line-of-sight vector to NED and determine elevation
        elevation = -np.arcsin(ctm_e_n[2, :] * u_as_e)

        # Determine if satellite is above the masking angle
        if elevation >= d2r * gnss_config.mask_angle:

            # Record the PRN
            prn.append(ephem[i, 0])

            # Increment number of measurements
            no_gnss_meas += 1

            # Calculate frame rotation during signal transit time
            ctm_e_i = ecef_to_eci_ctm(OMEGA_ie, approx_range, c)

            # Calculate geometric range
            delta_r = ctm_e_i * sat_r_es_e[i, :].T - r_ea_e
            rangex = np.sqrt(delta_r.T * delta_r)

            # Calculate geometric range rate
            range_rate = u_as_e.T * (ctm_e_i * (sat_v_es_e[i, :].T + omega_ie_matrix * sat_r_es_e[i, :].T) -
                                     (v_ea_e + omega_ie_matrix * r_ea_e))

            # Calculate pseudo-range measurement: geometric range + SV clock errors + I and T delay + Epsilon
            delta_t = tgps - sv_clock[i, 3]        # delta_t = tgps - toc (sec)

            # Inter-signal timing biases for GPS L1 band signal, (sec)
            delta_a_is_L1 = (gnss_config.SIS_err_SD/c)*rnd.randn() - sv_clock[i, 4]

            # SV clock bias, (m)
            sv_clock_bias = delta_a_is_L1 * c

            # Relativistic correction, (m)
            rel_corr = -2.0*(sat_r_es_e[i, :]*sat_v_es_e[i, :].T)/c

            # SV clock errors, (m)
            sv_clock_err = sv_clock_bias + rel_corr

            # Code track error, (m)
            epsilon = gnss_config.code_track_err_SD * rnd.randn()

            # Pseudo-range, (m)
            psdrel1 = gnss_config.rx_clock_offset
            psdrel2 = gnss_config.rx_clock_drift * t_i
            psdr = rangex + gnss_biases[i] + psdrel1 + psdrel2 + sv_clock_err + epsilon
            pseudorange.append(psdr[0, 0])

            # Calculate pseudo-range measurement: geometric range rate + SV clock drift error + Epsilon rate
            # SV clock drift error, (m/s)
            sv_clock_drift = (sv_clock[i, 1] + sv_clock[i, 2]*delta_t)*c

            # Code tracker error rate, (m/s)
            epsilon_rate = gnss_config.rate_track_err_SD * rnd.randn()

            # Pseudo-range rate, (m/s)
            rx_clock_drift = gnss_config.rx_clock_drift
            psdr_rate = range_rate + rx_clock_drift + sv_clock_drift + epsilon_rate
            pseudorange_rate.append(psdr_rate[0, 0])

            # Append satellite position and velocity to output data
            rx.append(sat_r_es_e[i, 0])
            ry.append(sat_r_es_e[i, 1])
            rz.append(sat_r_es_e[i, 2])
            vx.append(sat_v_es_e[i, 0])
            vy.append(sat_v_es_e[i, 1])
            vz.append(sat_v_es_e[i, 2])

    # End of If Statement to Determine the Number of Satellites in View

    # Forming the GNSS Measurement Output matrix
    gnss_meas = np.matrix([pseudorange, pseudorange_rate, rx, ry, rz, vx, vy, vz]).T

    return gnss_meas, no_gnss_meas, prn

# End of GNSS Measurement Generation


'''
    --------------------------------------------------
    7. Determine the LS Position & Velocity of Vehicle
    --------------------------------------------------
'''


def gnss_ls_pos_vel(gnss_meas, no_gnss_meas, pred_r_ea_e, pred_v_ea_e):

    # ******************************************************************************************************************
    #                                           POSITION AND CLOCK BIAS
    # ******************************************************************************************************************

    # Setup predicted state
    x_pred = np.nan * np.matrix(np.ones((4, 1)))
    x_pred[0:3, 0] = pred_r_ea_e
    x_pred[3, 0] = 0
    pred_meas = np.nan * np.matrix(np.ones((no_gnss_meas, 1)))
    geo_matrix = np.nan * np.matrix(np.ones((no_gnss_meas, 4)))
    est_clock = np.nan * np.ones((1, 2))
    est_r_ea_e = np.nan * np.matrix([1, 1, 1]).T
    est_v_ea_e = np.nan * np.matrix([1, 1, 1]).T
    tolerance = 1

    # Repeat until convergence
    while tolerance > 1.0E-04:

        # Loop measurements
        for i in xrange(0, no_gnss_meas):

            # Predict approximated geometric range
            delta_r = gnss_meas[i, 2:5].T - x_pred[0:3, 0]
            approx_range = np.sqrt(delta_r.T * delta_r)

            # Calculate frame rotation during signal transit time
            ctm_e_i = ecef_to_eci_ctm(OMEGA_ie, approx_range, c)

            # Predict pseudo-range
            delta_r = (ctm_e_i * gnss_meas[i, 2:5].T) - x_pred[0:3, 0]
            rangex = np.sqrt(delta_r.T * delta_r)
            pred_meas[i, 0] = rangex + x_pred[3, 0]

            # Predict line of sight and deploy in measurement matrix
            geo_matrix[i, 0:3] = -delta_r.T / rangex
            geo_matrix[i, 3] = 1

        # End of For Loop to Compute the Geometric matrix

        # Unweighted least-squares solution
        x_est = x_pred + (geo_matrix.T * geo_matrix).I * geo_matrix.T * (gnss_meas[:, 0] - pred_meas)

        # Test convergence
        tolerance = np.sqrt((x_est - x_pred).T * (x_est - x_pred))

        # Set predictions to estimates for next iteration
        x_pred = x_est

    # End of While Loop (LS Estimation Converged)

    # Set outputs to estimates
    est_r_ea_e[:, 0] = x_est[0:3, 0]
    est_clock[0, 0] = x_est[3, 0]

    # ******************************************************************************************************************
    #                                               VELOCITY AND CLOCK DRIFT
    # ******************************************************************************************************************

    # Skew symmetric matrix of Earth rate
    omega_ie_matrix = skew_sym(np.matrix([[0], [0], [OMEGA_ie]]))

    # Setup predicted state
    x_pred[0:3, 0] = pred_v_ea_e
    x_pred[3, 0] = 0
    tolerance = 1

    # Repeat until convergence
    while tolerance > 1.0E-04:

        # Loop measurements
        for i in xrange(0, no_gnss_meas):

            # Predict approximated geometric range
            delta_r = gnss_meas[i, 2:5].T - est_r_ea_e
            approx_range = np.sqrt(delta_r.T * delta_r)

            # Calculate frame rotation during signal transit time
            ctm_e_i = ecef_to_eci_ctm(OMEGA_ie, approx_range, c)

            # Calculate geometric range
            delta_r = ctm_e_i * gnss_meas[i, 2:5].T - est_r_ea_e
            rangex = np.sqrt(delta_r.T * delta_r)

            # Calculate line of sight
            u_as_e = delta_r / rangex

            # Predict pseudo-range rate
            range_rate = u_as_e.T * (ctm_e_i * (gnss_meas[i, 5:8].T + omega_ie_matrix * gnss_meas[i, 2:5].T) -
                                     (x_pred[0:3, 0] + omega_ie_matrix * est_r_ea_e))
            pred_meas[i, 0] = range_rate + x_pred[3, 0]

            # Predict line of sight and deploy in measurement matrix
            geo_matrix[i, 0:3] = -u_as_e.T
            geo_matrix[i, 3] = 1

        # End of For Loop to Compute the Geometric matrix

        # Unweighted least-squares solution
        x_est = x_pred + (geo_matrix.T * geo_matrix).I * geo_matrix.T * (gnss_meas[:, 1] - pred_meas)

        # Test convergence
        tolerance = np.sqrt((x_est - x_pred).T * (x_est - x_pred))

        # Set predictions to estimates for next iteration
        x_pred = x_est

    # End of While Loop (LS Estimation Converged)

    # Set outputs to estimates
    est_v_ea_e[:, 0] = x_est[0:3, 0]
    est_clock[0, 1] = x_est[3, 0]

    return est_r_ea_e, est_v_ea_e, est_clock

# End of Computing GNSS Least Square Positions and Velocities


'''
    ---------------------------------
    8. Calculate Output Errors in NED
    ---------------------------------
'''


def cal_err_ned(est_lat_b, est_lambda_b, est_alt_b, est_v_eb_n, est_ctm_b_n, true_lat_b, true_lambda_b,
                true_alt_b, true_v_eb_n, true_ctm_b_n):

    # Position error calculation
    delta_r_eb_n = np.nan * np.ones((3, 1))
    [r_ns, r_ew] = radii_of_curv(true_lat_b)
    delta_r_eb_n[0, 0] = (est_lat_b - true_lat_b) * (r_ns + true_alt_b)
    delta_r_eb_n[1, 0] = (est_lambda_b - true_lambda_b) * (r_ew + true_alt_b) * np.cos(true_lat_b)
    delta_r_eb_n[2, 0] = -(est_alt_b - true_alt_b)

    # Velocity error calculation
    delta_v_eb_n = est_v_eb_n - true_v_eb_n

    # Attitude error calculation
    delta_ctm_b_n = est_ctm_b_n * true_ctm_b_n.T
    eul_err_nb_n = -ctm_to_euler(delta_ctm_b_n)

    return delta_r_eb_n, delta_v_eb_n, eul_err_nb_n

# End of Calculating Errors in NED


'''
    ---------------------------------------------------------------------------
    9. Initialize the State Estimate Covariance Matrix, P for LC_EKF and TC_EKF
    ---------------------------------------------------------------------------
'''


def init_p_matrix(tightness, ekf_config):

    if tightness == 'loose':

        # Initialize error covariance matrix
        p_matrix = np.zeros((15, 15))

        # Determine each element of the covariance matrix
        p_matrix[0:3, 0:3] = np.eye(3) * ekf_config.init_pos_unc ** 2
        p_matrix[3:6, 3:6] = np.eye(3) * ekf_config.init_vel_unc ** 2
        p_matrix[6:9, 6:9] = np.eye(3) * ekf_config.init_att_unc ** 2
        p_matrix[9:12, 9:12] = np.eye(3) * ekf_config.init_b_a_unc ** 2
        p_matrix[12:15, 12:15] = np.eye(3) * ekf_config.init_b_g_unc ** 2

    elif tightness == 'tight':

        # Initialize error covariance matrix
        p_matrix = np.zeros((17, 17))

        # Determine each element of the covariance matrix
        p_matrix[0:3, 0:3] = np.eye(3) * ekf_config.init_pos_unc ** 2
        p_matrix[3:6, 3:6] = np.eye(3) * ekf_config.init_vel_unc ** 2
        p_matrix[6:9, 6:9] = np.eye(3) * ekf_config.init_att_unc ** 2
        p_matrix[9:12, 9:12] = np.eye(3) * ekf_config.init_b_a_unc ** 2
        p_matrix[12:15, 12:15] = np.eye(3) * ekf_config.init_b_g_unc ** 2
        p_matrix[15, 15] = ekf_config.init_clock_offset_unc ** 2
        p_matrix[16, 16] = ekf_config.init_clock_drift_unc ** 2

    return p_matrix

# End of Initializing Single EKF Matrix


'''
    -------------------------------------------------------------------
    10. Initialize the State Estimate Covariance Matrix, P for Dual EKF
    -------------------------------------------------------------------
'''


def init_dual_p_matrix(lc_ekf_config, tc_ekf_config):

    # Initialize error covariance matrix
    lc_p_matrix = np.zeros((15, 15))
    tc_p_matrix = np.zeros((17, 17))

    # Determine each element of the covariance matrix
    # For loosely coupled EKF
    lc_p_matrix[0:3, 0:3] = np.eye(3) * lc_ekf_config.init_pos_unc ** 2
    lc_p_matrix[3:6, 3:6] = np.eye(3) * lc_ekf_config.init_vel_unc ** 2
    lc_p_matrix[6:9, 6:9] = np.eye(3) * lc_ekf_config.init_att_unc ** 2
    lc_p_matrix[9:12, 9:12] = np.eye(3) * lc_ekf_config.init_b_a_unc ** 2
    lc_p_matrix[12:15, 12:15] = np.eye(3) * lc_ekf_config.init_b_g_unc ** 2

    # For tightly coupled EKF
    tc_p_matrix[0:3, 0:3] = np.eye(3) * tc_ekf_config.init_pos_unc ** 2
    tc_p_matrix[3:6, 3:6] = np.eye(3) * tc_ekf_config.init_vel_unc ** 2
    tc_p_matrix[6:9, 6:9] = np.eye(3) * tc_ekf_config.init_att_unc ** 2
    tc_p_matrix[9:12, 9:12] = np.eye(3) * tc_ekf_config.init_b_a_unc ** 2
    tc_p_matrix[12:15, 12:15] = np.eye(3) * tc_ekf_config.init_b_g_unc ** 2
    tc_p_matrix[15, 15] = tc_ekf_config.init_clock_offset_unc ** 2
    tc_p_matrix[16, 16] = tc_ekf_config.init_clock_drift_unc ** 2

    return lc_p_matrix, tc_p_matrix

# End of Initializing Dual EKF Covariance Matrices


'''
    -----------------------------------------------------------
    11. Calculate Specific Forces and Angular Rates from the IMU
    -----------------------------------------------------------
'''


def kinematics_ecef(tau_i, ctm_b_e, old_ctm_b_e, v_eb_e, old_v_eb_e, r_eb_e):

    # Allocate the alpha_ib_b vector
    alpha_ib_b = np.nan * np.matrix([[1], [1], [1]])

    if tau_i > 0:

        # Determine the Earth rotation over the update interval
        [ctm_earth, alpha_ie] = c_earth(tau_i)

        # Obtain coordinate transformation matrix from the old attitude (w.r.t. an inertial frame) to the new
        ctm_old_new = ctm_b_e.T * ctm_earth * old_ctm_b_e

        # Calculate the approximate angular rate w.r.t. an inertial frame
        alpha_ib_b[0, 0] = 0.5 * (ctm_old_new[1, 2] - ctm_old_new[2, 1])
        alpha_ib_b[1, 0] = 0.5 * (ctm_old_new[2, 0] - ctm_old_new[0, 2])
        alpha_ib_b[2, 0] = 0.5 * (ctm_old_new[0, 1] - ctm_old_new[1, 0])

        # Calculate and apply the scaling factor
        scale = np.arccos(0.5 * (ctm_old_new[0, 0] + ctm_old_new[1, 1] + ctm_old_new[2, 2] - 1.0))

        if scale > 2E-05:  # scaling factor is 1 if "scale" is less than this minimum limit.

            alpha_ib_b = alpha_ib_b * scale / np.sin(scale)

        # Calculate the angular rate using
        omega_ib_b = alpha_ib_b / tau_i

        # Calculate the specific force resolved about ECEF-frame axes
        f_ib_e = ((v_eb_e - old_v_eb_e) / tau_i) - gravity_ecef(r_eb_e) + 2 * skew_sym(
            np.matrix([[0], [0], [OMEGA_ie]])) * old_v_eb_e

        # Calculate the average body-to-ECEF-frame coordinate transformation matrix over the update interval
        mag_alpha = np.sqrt(alpha_ib_b.T * alpha_ib_b)
        alpha_ib_b_matrix = skew_sym(alpha_ib_b)

        if mag_alpha > 1.0E-8:

            term_1 = ((1 - np.cos(mag_alpha[0, 0])) / (mag_alpha[0, 0] ** 2)) * alpha_ib_b_matrix

            term_2 = ((1 - np.sin(mag_alpha[0, 0]) / mag_alpha[0, 0]) / (mag_alpha[0, 0] ** 2)) * (alpha_ib_b_matrix *
                                                                                                   alpha_ib_b_matrix)
            term_3 = 0.5 * skew_sym(np.matrix([[0], [0], [alpha_ie]])) * tau_i

            ave_ctm_b_e = old_ctm_b_e * (np.matrix(np.eye(3)) + term_1 + term_2) + term_3 * old_ctm_b_e

        else:

            ave_ctm_b_e = old_ctm_b_e - 0.5 * skew_sym(np.matrix([[0], [0], [alpha_ie]])) * old_ctm_b_e * tau_i

        # End of "if" mag_alpha

        # Transform specific force to body-frame resolving axes
        f_ib_b = ave_ctm_b_e.I * f_ib_e

    else:

        # If time interval is zero, set angular rate and specific force to zero
        omega_ib_b = np.matrix(np.zeros((3, 1)))
        f_ib_b = np.matrix(np.zeros((3, 1)))

    # End of "if" tau_i

    return f_ib_b, omega_ib_b

# End of Calculating Specific Forces and Angular Rates


'''
    ---------------------------
    12. Simulating the IMU Model
    ---------------------------
'''


def imu_model(tau_i, true_f_ib_b, true_omega_ib_b, imu_config, old_quant_residuals):

    # Generate noise for accelerometer and gyroscope
    if tau_i > 0:
        accel_noise = np.matrix(rnd.randn(3, 1)) * imu_config.accel_noise_root_PSD / np.sqrt(tau_i)
        gyro_noise = np.matrix(rnd.randn(3, 1)) * imu_config.gyro_noise_root_PSD / np.sqrt(tau_i)

    else:
        accel_noise = np.matrix([0, 0, 0]).T
        gyro_noise = np.matrix([0, 0, 0]).T
    # End  of If tau_i

    # Calculate accelerometer and gyro outputs
    uq_f_ib_b = imu_config.b_a + (np.matrix(np.eye(3)) + imu_config.M_a) * true_f_ib_b + accel_noise
    uq_omega_ib_b = imu_config.b_g + imu_config.G_g * true_f_ib_b + gyro_noise + (np.matrix(np.eye(3)) +
                                                                                  imu_config.M_g) * true_omega_ib_b
    # Quantize accelerometer outputs
    quant_residuals = np.nan * np.matrix(np.ones((6, 1)))

    if imu_config.accel_quant_level > 0:
        meas_f_ib_b = imu_config.accel_quant_level * np.round((uq_f_ib_b + old_quant_residuals[0:3, 0]) /
                                                              imu_config.accel_quant_level)
        quant_residuals[0:3, 0] = uq_f_ib_b + old_quant_residuals[0:3, 0] - meas_f_ib_b

    else:
        meas_f_ib_b = uq_f_ib_b
        quant_residuals[0:3, 0] = np.matrix([0, 0, 0]).T
    # End  of If IMU_errors.accel_quant_level

    # Quantize gyro outputs
    if imu_config.gyro_quant_level > 0:
        meas_omega_ib_b = imu_config.gyro_quant_level * np.round((uq_omega_ib_b + old_quant_residuals[3:6, 0]) /
                                                                 imu_config.gyro_quant_level)
        quant_residuals[3:6, 0] = uq_omega_ib_b + old_quant_residuals[3:6, 0] - meas_omega_ib_b

    else:
        meas_omega_ib_b = uq_omega_ib_b
        quant_residuals[3:6, 0] = np.matrix([0, 0, 0]).T

    return meas_f_ib_b, meas_omega_ib_b, quant_residuals

# End of Simulating IMU Model


'''
    -----------------------------------------
    13. Update Estimated Navigation Solutions
    -----------------------------------------
'''


def nav_eqs_ecef(tau_i, old_r_eb_e, old_v_eb_e, old_ctm_b_e, f_ib_b, omega_ib_b):

    # ******************************************************************************************************************
    #                                               UPDATE ATTITUDE
    # ******************************************************************************************************************
    [ctm_earth, alpha_ie] = c_earth(tau_i)

    # Calculate attitude increment, magnitude, and skew-symmetric matrix
    alpha_ib_b = omega_ib_b * tau_i
    mag_alpha = np.sqrt(alpha_ib_b.T * alpha_ib_b)
    alpha_ib_b_matrix = skew_sym(alpha_ib_b)

    # Obtain coordinate transformation matrix from the new attitude w.r.t. an inertial frame to the old using
    # Rodrigues' formula
    if mag_alpha > 1.0E-08:
        c_term_1 = (np.sin(mag_alpha[0, 0]) / mag_alpha[0, 0]) * alpha_ib_b_matrix
        c_term_2 = ((1 - np.cos(mag_alpha[0, 0])) / (mag_alpha[0, 0] ** 2)) * (alpha_ib_b_matrix * alpha_ib_b_matrix)
        ctm_new_old = np.matrix(np.eye(3)) + c_term_1 + c_term_2
    else:
        ctm_new_old = np.matrix(np.eye(3)) + alpha_ib_b_matrix
    # End  of "if" mag_alpha

    # Update attitude
    ctm_b_e = ctm_earth * old_ctm_b_e * ctm_new_old

    # ******************************************************************************************************************
    #                                   SPECIFIC FORCE FRAME TRANSFORMATION
    # ******************************************************************************************************************

    # Calculate the average body-to-ECEF-frame coordinate transformation matrix over the update interval
    if mag_alpha > 1.0E-08:

        a_term_1 = ((1 - np.cos(mag_alpha[0, 0])) / (mag_alpha[0, 0] ** 2)) * alpha_ib_b_matrix

        a_term_2 = ((1 - np.sin(mag_alpha[0, 0]) / mag_alpha[0, 0]) / (mag_alpha[0, 0] ** 2)) * (alpha_ib_b_matrix *
                                                                                                 alpha_ib_b_matrix)
        a_term_3 = 0.5 * skew_sym(np.matrix([[0], [0], [alpha_ie]])) * tau_i

        ave_ctm_b_e = old_ctm_b_e * (np.matrix(np.eye(3)) + a_term_1 + a_term_2) - a_term_3 * old_ctm_b_e

    else:

        ave_ctm_b_e = old_ctm_b_e - 0.5 * skew_sym(np.matrix([[0], [0], [alpha_ie]])) * old_ctm_b_e * tau_i

    # End of "if" mag_alpha

    # Transform specific force to ECEF-frame resolving axes
    f_ib_e = ave_ctm_b_e * f_ib_b

    # ******************************************************************************************************************
    #                                           UPDATE VELOCITY
    # ******************************************************************************************************************

    v_eb_e = old_v_eb_e + tau_i * (f_ib_e + gravity_ecef(old_r_eb_e) -
                                   2 * skew_sym(np.matrix([[0], [0], [OMEGA_ie]])) * old_v_eb_e)

    # ******************************************************************************************************************
    #                                       UPDATE CARTESIAN POSITION
    # ******************************************************************************************************************
    # From (5.38),
    r_eb_e = old_r_eb_e + (v_eb_e + old_v_eb_e) * 0.5 * tau_i

    return r_eb_e, v_eb_e, ctm_b_e

# End of Updating Estimated Navigation Solutions


'''
    --------------------------------------------------------------
    14. Loosely Coupled INS/GNSS EKF Integration in a Single Epoch
    --------------------------------------------------------------
'''


def lc_ekf_epoch(gnss_r_eb_e, gnss_v_eb_e, tau_s, est_ctm_b_e_old, est_v_eb_e_old, est_r_eb_e_old, est_imu_bias_old,
                 p_matrix_old, meas_f_ib_b, est_lat_b_old, lc_kf_config):

    # 0. Skew symmetric matrix of Earth rate
    omega_ie_matrix = skew_sym(np.matrix([[0], [0], [OMEGA_ie]]))

    # ******************************************************************************************************************
    #                                           SYSTEM PROPAGATION PHASE
    # ******************************************************************************************************************

    # 1. Build the System Matrix F in ECEF Frame
    f_matrix = np.matrix(np.zeros((15, 15)))

    # Calculate the meridian radius and transverse radius
    [r_ns, r_ew] = radii_of_curv(est_lat_b_old)

    # Calculate the geocentric radius at current latitude
    geo_radius = r_ew * np.sqrt(np.cos(est_lat_b_old) ** 2.0 + ((1.0 - ecc_o ** 2.0) ** 2.0) *
                                np.sin(est_lat_b_old) ** 2.0)

    # For position vector
    f_matrix[0:3, 3:6] = np.eye(3)

    # For velocity vector
    f_matrix[3:6, 0:3] = -(2.0 * gravity_ecef(est_r_eb_e_old) / geo_radius) * (est_r_eb_e_old.T / np.sqrt(
        est_r_eb_e_old.T * est_r_eb_e_old))
    f_matrix[3:6, 3:6] = -2.0 * omega_ie_matrix
    f_matrix[3:6, 6:9] = -skew_sym(est_ctm_b_e_old * meas_f_ib_b)
    f_matrix[3:6, 9:12] = est_ctm_b_e_old

    # For attitude vector
    f_matrix[6:9, 0:3] = -omega_ie_matrix
    f_matrix[6:9, 12:15] = est_ctm_b_e_old

    # 2. Determine the State Transition Matrix (first-order approximate), PHI = I + F*tau
    phi_matrix = np.matrix(np.eye(15)) + f_matrix * tau_s

    # 3. Determine approximate system noise covariance matrix
    q_prime_matrix = np.matrix(np.zeros((15, 15)))
    q_prime_matrix[3:6, 3:6] = np.eye(3) * lc_kf_config.accel_noise_PSD * tau_s
    q_prime_matrix[6:9, 6:9] = np.eye(3) * lc_kf_config.gyro_noise_PSD * tau_s
    q_prime_matrix[9:12, 9:12] = np.eye(3) * lc_kf_config.accel_bias_PSD * tau_s
    q_prime_matrix[12:15, 12:15] = np.eye(3) * lc_kf_config.gyro_bias_PSD * tau_s

    # 4. Propagate state estimates, noting that only the clock states are non-zero due to closed-loop correction
    x_est_propa = np.matrix(np.zeros((15, 1)))

    # 5. Propagate state estimation error covariance matrix
    p_matrix_propa = phi_matrix * (p_matrix_old + 0.5 * q_prime_matrix) * phi_matrix.T + 0.5 * q_prime_matrix
    # Check for NaNs
    nans_indx = np.isnan(p_matrix_propa)
    if nans_indx.any():
        for i in xrange(0, 15):
            for j in xrange(0, 15):
                if nans_indx[i, j]:
                    p_matrix_propa[i, j] = 0.0
    # Check for negative infinity
    neginf_indx = np.isneginf(p_matrix_propa)
    if neginf_indx.any():
        for i in xrange(0, 15):
            for j in xrange(0, 15):
                if neginf_indx[i, j]:
                    p_matrix_propa[i, j] = -1.0E+08
    # Check for positive infinity
    posinf_indx = np.isposinf(p_matrix_propa)
    if posinf_indx.any():
        for i in xrange(0, 15):
            for j in xrange(0, 15):
                if posinf_indx[i, j]:
                    p_matrix_propa[i, j] = 1.0E+08

    # ******************************************************************************************************************
    #                                           MEASUREMENT UPDATE PHASE
    # ******************************************************************************************************************

    # 6. Set-up measurement matrix
    h_matrix = np.matrix(np.zeros((6, 15)))
    h_matrix[0:3, 0:3] = -np.eye(3)
    h_matrix[3:6, 3:6] = -np.eye(3)

    # 7. Set-up measurement noise covariance matrix assuming all measurements are independent and have equal variance
    #    for a given measurement type
    r_matrix = np.matrix(np.zeros((6, 6)))
    r_matrix[0:3, 0:3] = np.eye(3) * lc_kf_config.pos_meas_SD ** 2
    r_matrix[3:6, 3:6] = np.eye(3) * lc_kf_config.vel_meas_SD ** 2

    # 7. Calculate Kalman gain
    k_matrix = p_matrix_propa * h_matrix.T * (h_matrix * p_matrix_propa * h_matrix.T + r_matrix).I

    # 8. Formulate measurement innovations, noting that zero lever arm is assumed here
    delta_z = np.matrix(np.zeros((6, 1)))
    delta_z[0:3, 0] = gnss_r_eb_e - est_r_eb_e_old
    delta_z[3:6, 0] = gnss_v_eb_e - est_v_eb_e_old

    # 9. Update state estimates
    x_est_new = x_est_propa + k_matrix * delta_z

    # 10. Update state estimation error covariance matrix
    p_matrix_new = (np.eye(15) - k_matrix * h_matrix) * p_matrix_propa

    # ******************************************************************************************************************
    #                                           CLOSED-LOOP CORRECTION
    # ******************************************************************************************************************

    # 11. Correct attitude, velocity, and position
    est_ctm_b_e_new = (np.eye(3) - skew_sym(x_est_new[6:9])) * est_ctm_b_e_old
    est_v_eb_e_new = est_v_eb_e_old - x_est_new[3:6]
    est_r_eb_e_new = est_r_eb_e_old - x_est_new[0:3]

    # 12. Update IMU bias estimates
    est_imu_bias_new = est_imu_bias_old + x_est_new[9:15, 0]

    return est_ctm_b_e_new, est_v_eb_e_new, est_r_eb_e_new, est_imu_bias_new, p_matrix_new

# End of Loosely Coupled INS/GNSS EKF Integration for a Single Epoch


'''
    ---------------------------------------------------------------
    15. Main Function to Run the Loosely Coupled INS/GPS EKF Fusion
    ---------------------------------------------------------------
'''


def lc_ins_gps_ekf_fusion(simtype, tightness, true_profile, no_t_steps, eul_err_nb_n, imu_config, gnss_config,
                          lc_kf_config, DyOM, doy, fin_nav, gps_tow):

    print 'Starting Loosely EKF Fusion...'

    # 1. Initialize true navigation solution
    old_t = true_profile[0, 0]                          # starting epoch (s)
    true_lat_b = true_profile[0, 1]                     # initial true latitude (rad)
    true_lon_b = true_profile[0, 2]                     # initial true longitude (rad)
    true_alt_b = true_profile[0, 3]                     # initial true altitude (m)
    true_v_eb_n = true_profile[0, 4:7].T                # initial true velocity vector (m/s)
    true_eul_nb = true_profile[0, 7:10].T               # initial true attitude (rad)
    true_ctm_b_n = euler_to_ctm(true_eul_nb).T          # coordinate transfer matrix from body frame to NED frame

    # 2. Convert all the above parameters to ECEF frame
    [old_true_r_eb_e, old_true_v_eb_e, old_true_ctm_b_e] = lla_to_ecef(true_lat_b, true_lon_b, true_alt_b,
                                                                       true_v_eb_n, true_ctm_b_n)

    # Conditioning the GNSS simulation by the "simtype"
    if simtype == 'simulation':

        # 3. Determine satellite positions and velocities
        [sat_r_es_e, sat_v_es_e] = sat_pv_sim(old_t, gnss_config)

        # 4. Initialize the GNSS biases
        gnss_biases = init_gnss_bias_sim(sat_r_es_e, old_true_r_eb_e, true_lat_b, true_lon_b, gnss_config)

        # 5. Generate GNSS measurements
        [gnss_meas, no_gnss_meas, prn] = gnss_meas_gen_sim(old_t, sat_r_es_e, sat_v_es_e, old_true_r_eb_e, true_lat_b,
                                                           true_lon_b, old_true_v_eb_e, gnss_biases, gnss_config)

    elif simtype == 'play back':

        # 3.1 Process the Ephemeris only in the "play back" Mode
        [iono_alpha, iono_beta, alma_t_para, sv_clock, navigation, ephemeris] = ephem_processing(finpath, fin_nav,
                                                                                                 gps_tow, DyOM)
        # 3.2 Save the Ephemris Data
        navigation_fname = foutpath + 'navigation_message.txt'
        np.savetxt(navigation_fname, navigation)
        ephem_fname = foutpath + 'ephemeris.txt'
        np.savetxt(ephem_fname, ephemeris)

        # 3.3 Determine satellite positions and velocities in ECEF
        [sat_r_es_e, sat_v_es_e] = sat_pv_ephem(ephemeris, gnss_config, old_t, tol)

        # 4. Initialize the GNSS biases
        gnss_biases = init_gnss_bias_ephem(old_t, doy, ephemeris, iono_alpha, iono_beta, true_lat_b, true_lon_b,
                                           true_alt_b, sat_r_es_e, old_true_r_eb_e, gnss_config)

        # 5. Generate GNSS measurements
        [gnss_meas, no_gnss_meas, prn] = \
            gnss_meas_gen_ephem(old_t, sat_r_es_e, sat_v_es_e, old_true_r_eb_e, true_lat_b, true_lon_b,
                                old_true_v_eb_e, gnss_biases, gnss_config, sv_clock, ephemeris, alma_t_para)

    # Array to hold GNSS generated measurements for output
    out_gnss_time = [old_t]
    # out_gnss_time.append(old_t)
    out_gnss_gen = np.nan * np.ones((1, 16, 9))
    out_gnss_gen[0, 0:len(prn), 0] = prn
    out_gnss_gen[0, 0:len(prn), 1:9] = gnss_meas

    # 6. Determine Least-Square GNSS position solutions
    [gnss_r_eb_e, gnss_v_eb_e, est_clock] = gnss_ls_pos_vel(gnss_meas, no_gnss_meas, gnss_config.init_est_r_ea_e,
                                                            gnss_config.init_est_v_ea_e)

    old_est_r_eb_e = gnss_r_eb_e
    old_est_v_eb_e = gnss_v_eb_e

    # 7. Convert Position and Velocity from ECEF to NED
    [old_est_lat_b, old_est_lon_b, old_est_alt_b, old_est_v_eb_n] = pv_ecef_to_lla(old_est_r_eb_e, old_est_v_eb_e)
    est_lat_b = old_est_lat_b
    # Save the least-square gnss navigation solution to the outprofile

    # 8. Initialize estimated attitude solution
    old_est_ctm_b_n = init_ned_att(true_ctm_b_n, eul_err_nb_n)

    # 9. Compute the CTM from NED to ECEF
    [temp1, temp2, old_est_ctm_b_e] = lla_to_ecef(old_est_lat_b, old_est_lon_b, old_est_alt_b, old_est_v_eb_n,
                                                  old_est_ctm_b_n)

    # 10. Initialize loosely coupled output profile and error profile
    est_profile = np.nan * np.ones((no_t_steps, 10))
    est_errors = np.nan * np.ones((no_t_steps, 10))

    # 11. Generate loosely coupled initial output profile
    est_profile[0, 0] = old_t
    est_profile[0, 1] = old_est_lat_b
    est_profile[0, 2] = old_est_lon_b
    est_profile[0, 3] = old_est_alt_b
    est_profile[0, 4:7] = old_est_v_eb_n.T
    est_profile[0, 7:10] = ctm_to_euler(old_est_ctm_b_n.T).T

    # 12. Determine errors and generate output record
    [delta_r_eb_n, delta_v_eb_n, eul_err_nb_n] = \
        cal_err_ned(old_est_lat_b, old_est_lon_b, old_est_alt_b, old_est_v_eb_n, old_est_ctm_b_n, true_lat_b,
                    true_lon_b, true_alt_b, true_v_eb_n, true_ctm_b_n)

    # 13. Generate loosely coupled initial error profile
    est_errors[0, 0] = old_t
    est_errors[0, 1:4] = delta_r_eb_n.T
    est_errors[0, 4:7] = delta_v_eb_n.T
    est_errors[0, 7:10] = eul_err_nb_n.T

    # 15. Initialize loosely coupled Kalman filter P matrix
    p_matrix = init_p_matrix(tightness, lc_kf_config)

    # 16. Initialize IMU bias states
    est_imu_bias = np.matrix(np.zeros((6, 1)))

    # 17. Initialize IMU quantization residuals
    quant_resid = np.matrix(np.zeros((6, 1)))

    # 18. Generate IMU bias and clock output records
    out_imu_gen = np.nan * np.matrix(np.ones((no_t_steps, 7)))
    out_imu_gen[0, 0] = old_t
    out_imu_gen[0, 1:7] = 0.0
    out_imu_bias_est = np.nan * np.matrix(np.ones((1, 7)))
    out_imu_bias_est[0, 0] = old_t
    out_imu_bias_est[0, 1:7] = est_imu_bias.T
    output_clock = np.nan * np.matrix(np.ones((1, 3)))
    output_clock[0, 0] = old_t
    output_clock[0, 1:3] = est_clock

    # 19. Generate KF uncertainty record in ECEF
    output_kf_sd = np.nan * np.matrix(np.ones((1, 16)))
    output_kf_sd[0, 0] = old_t
    eig_value = lina.eigvals(p_matrix)
    for i in xrange(0, 15):
        output_kf_sd[0, i + 1] = np.sqrt(eig_value[i])
    # End of For Loop

    # 20. Initialize GNSS model timing
    t_last_gnss = old_t
    gnss_epoch = 1

    # 21. Initialize Progress Bar
    print 'Simulation is in Progress. Please Wait!'

    # ******************************************************************************************************************
    #                                                   MAIN LOOP
    # ******************************************************************************************************************

    for epoch in xrange(1, no_t_steps):

        # 22. Input data from motion profile
        t = true_profile[epoch, 0]                      # current epoch (s)
        true_lat_b = true_profile[epoch, 1]             # current true latitude (rad)
        true_lon_b = true_profile[epoch, 2]             # current true longitude (rad)
        true_alt_b = true_profile[epoch, 3]             # current true altitude (m)
        true_v_eb_n = true_profile[epoch, 4:7].T        # current true velocity vector (m/s)
        true_eul_nb = true_profile[epoch, 7:10].T
        true_ctm_b_n = euler_to_ctm(true_eul_nb).T
        [true_r_eb_e, true_v_eb_e, true_ctm_b_e] = lla_to_ecef(true_lat_b, true_lon_b, true_alt_b, true_v_eb_n,
                                                               true_ctm_b_n)
        tau_i = t - old_t

        # Conditioning the IMU simulation by the "simtype"
        if simtype == 'simulation':

            # 23. Calculate specific force and angular rate
            [true_f_ib_b, true_omega_ib_b] = kinematics_ecef(tau_i, true_ctm_b_e, old_true_ctm_b_e, true_v_eb_e,
                                                             old_true_v_eb_e, old_true_r_eb_e)

            # 24. Simulate IMU errors
            [meas_f_ib_b, meas_omega_ib_b, quant_resid] = imu_model(tau_i, true_f_ib_b, true_omega_ib_b, imu_config,
                                                                    quant_resid)
            out_imu_gen[epoch, 0] = t
            out_imu_gen[epoch, 1:4] = meas_f_ib_b.T
            out_imu_gen[epoch, 4:7] = meas_omega_ib_b.T

        elif simtype == 'play back':

            # 23. Calculate specific force and angular rate
            true_f_ib_b = true_profile[epoch, 16:19].T                          # accelerometer reading (error free)
            true_omega_ib_b = true_profile[epoch, 10:13].T                      # gyroscope reading (error free)

            # 24. Simulate IMU errors
            meas_f_ib_b = true_f_ib_b + true_profile[epoch, 19:22].T            # accelerometer reading
            meas_omega_ib_b = true_omega_ib_b + true_profile[epoch, 13:16].T    # gyroscope reading

            out_imu_gen[epoch, 0] = t
            out_imu_gen[epoch, 1:4] = meas_f_ib_b.T
            out_imu_gen[epoch, 4:7] = meas_omega_ib_b.T

        # 25. Correct IMU errors
        meas_f_ib_b = meas_f_ib_b - est_imu_bias[0:3, 0]
        meas_omega_ib_b = meas_omega_ib_b - est_imu_bias[3:6, 0]

        # 26. Update estimated navigation solution
        [est_r_eb_e, est_v_eb_e, est_ctm_b_e] = nav_eqs_ecef(tau_i, old_est_r_eb_e, old_est_v_eb_e, old_est_ctm_b_e,
                                                             meas_f_ib_b, meas_omega_ib_b)

        # 27. Determine whether to update GNSS simulation and run Kalman filter
        if (t - t_last_gnss) >= gnss_config.epoch_interval:

            gnss_epoch += 1             # update epoch (time) index
            tau_s = t - t_last_gnss     # KF time interval
            t_last_gnss = t             # update the last epoch

            # Conditioning the GNSS simulation by the "simtype"
            if simtype == 'simulation':

                # 28. Determine satellite positions and velocities
                [sat_r_es_e, sat_v_es_e] = sat_pv_sim(t, gnss_config)

                # 29. Generate GNSS measurements
                [gnss_meas, no_gnss_meas, prn] = gnss_meas_gen_sim(t, sat_r_es_e, sat_v_es_e, true_r_eb_e, true_lat_b,
                                                                   true_lon_b, true_v_eb_e, gnss_biases, gnss_config)

            elif simtype == 'play back':

                # 28.1 Determine satellite positions and velocities in ECEF
                [sat_r_es_e, sat_v_es_e] = sat_pv_ephem(ephemeris, gnss_config, t, tol)

                # 28.2 Initialize the GNSS biases
                gnss_biases = init_gnss_bias_ephem(t, doy, ephemeris, iono_alpha, iono_beta, true_lat_b, true_lon_b,
                                                   true_alt_b, sat_r_es_e, old_true_r_eb_e, gnss_config)

                # 29. Generate GNSS measurements
                [gnss_meas, no_gnss_meas, prn] = \
                    gnss_meas_gen_ephem(t, sat_r_es_e, sat_v_es_e, old_true_r_eb_e, true_lat_b, true_lon_b,
                                        old_true_v_eb_e, gnss_biases, gnss_config, sv_clock, ephemeris, alma_t_para)

            # Array to hold GNSS generated measurements for output
            out_gnss_time.append(t)
            out_gnss_gen_new = np.nan * np.ones((gnss_epoch, 16, 9))
            out_gnss_gen_new[0:gnss_epoch - 1, :, :] = out_gnss_gen
            out_gnss_gen_new[gnss_epoch - 1, 0:len(prn), 0] = prn
            out_gnss_gen_new[gnss_epoch - 1, 0:len(prn), 1:9] = gnss_meas
            out_gnss_gen = out_gnss_gen_new

            # 30. Determine Least-Square GNSS position solutions
            [gnss_r_eb_e, gnss_v_eb_e, est_clock] = gnss_ls_pos_vel(gnss_meas, no_gnss_meas, gnss_r_eb_e, gnss_v_eb_e)

            # 31. Run Integration Kalman filter
            [est_ctm_b_e, est_v_eb_e, est_r_eb_e, est_imu_bias, p_matrix] = \
                lc_ekf_epoch(gnss_r_eb_e, gnss_v_eb_e, tau_s, est_ctm_b_e, est_v_eb_e, est_r_eb_e, est_imu_bias,
                             p_matrix, meas_f_ib_b, est_lat_b, lc_kf_config)

            # 32. Generate IMU Bias and Clock Output Records Recursively
            # 32.1 IMU Bias
            out_imu_bias_est_new = np.nan * np.matrix(np.ones((gnss_epoch, 7)))
            out_imu_bias_est_new[0:gnss_epoch - 1, 0] = out_imu_bias_est[0:gnss_epoch - 1, 0]
            out_imu_bias_est_new[gnss_epoch - 1, 0] = t
            out_imu_bias_est_new[0:gnss_epoch - 1, 1:7] = out_imu_bias_est[0:gnss_epoch - 1, 1:7]
            out_imu_bias_est_new[gnss_epoch - 1, 1:7] = est_imu_bias.T
            out_imu_bias_est = out_imu_bias_est_new

            # 32.2 Clock Bias
            out_clock_new = np.nan * np.matrix(np.ones((gnss_epoch, 3)))
            out_clock_new[0:gnss_epoch - 1, 0] = output_clock[0:gnss_epoch - 1, 0]
            out_clock_new[gnss_epoch - 1, 0] = t
            out_clock_new[0:gnss_epoch - 1, 1:3] = output_clock[0:gnss_epoch - 1, 1:3]
            out_clock_new[gnss_epoch - 1, 1:3] = est_clock
            output_clock = out_clock_new

            # 33. Generate KF uncertainty output record recursively
            out_kf_sd_new = np.nan * np.matrix(np.ones((gnss_epoch, 16)))
            out_kf_sd_new[0:gnss_epoch - 1, 0] = output_kf_sd[0:gnss_epoch - 1, 0]
            out_kf_sd_new[gnss_epoch - 1, 0] = t
            out_kf_sd_new[0:gnss_epoch - 1, 1:16] = output_kf_sd[0:gnss_epoch - 1, 1:16]
            eig_value = lina.eigvals(p_matrix)
            for i in xrange(0, 15):
                out_kf_sd_new[gnss_epoch - 1, i + 1] = np.sqrt(eig_value[i])
            # End of For out_kf_sd update

            output_kf_sd = out_kf_sd_new

        # End of "if" on checking for GNSS update

        # 34. Convert navigation solution to NED
        [est_lat_b, est_lon_b, est_alt_b, est_v_eb_n, est_ctm_b_n] = ecef_to_lla(est_r_eb_e, est_v_eb_e, est_ctm_b_e)

        # 35. Generate output profile record
        est_profile[epoch, 0] = t
        est_profile[epoch, 1] = est_lat_b
        est_profile[epoch, 2] = est_lon_b
        est_profile[epoch, 3] = est_alt_b
        est_profile[epoch, 4:7] = est_v_eb_n.T
        est_profile[epoch, 7:10] = ctm_to_euler(est_ctm_b_n.T).T

        # 36. Determine Errors
        [delta_r_eb_n, delta_v_eb_n, eul_err_nb_n] = \
            cal_err_ned(est_lat_b, est_lon_b, est_alt_b, est_v_eb_n, est_ctm_b_n, true_lat_b, true_lon_b, true_alt_b,
                        true_v_eb_n, true_ctm_b_n)

        # 37. Generate Error Records
        est_errors[epoch, 0] = t
        est_errors[epoch, 1:4] = delta_r_eb_n.T
        est_errors[epoch, 4:7] = delta_v_eb_n.T
        est_errors[epoch, 7:10] = eul_err_nb_n.T

        # 38. Reset old values
        old_t = t
        old_true_r_eb_e = true_r_eb_e
        old_true_v_eb_e = true_v_eb_e
        old_true_ctm_b_e = true_ctm_b_e
        old_est_r_eb_e = est_r_eb_e
        old_est_v_eb_e = est_v_eb_e
        old_est_ctm_b_e = est_ctm_b_e

        # 39. Updating Progress Bar
        progressbar(epoch / float(no_t_steps))

    # End of For Main Loop

    print '\n NavSim Completed!'

    return est_profile, est_errors, output_kf_sd, out_imu_gen, out_imu_bias_est, output_clock, out_gnss_gen, \
           out_gnss_time

# End of Main Loosely Coupled INS/GNSS Fusion


'''
    --------------------------------------------------------------
    16. Tightly Coupled INS/GNSS EKF Integration in a Single Epoch
    --------------------------------------------------------------
'''


def tc_ekf_epoch(gnss_meas, no_meas, tau_s, est_ctm_b_e_old, est_v_eb_e_old, est_r_eb_e_old, est_imu_bias_old,
                 est_clock_old, p_matrix_old, meas_f_ib_b, est_lat_b_old, tc_kf_config):

    # 0. Skew symmetric matrix of Earth rate
    omega_ie_matrix = skew_sym(np.matrix([[0], [0], [OMEGA_ie]]))

    # ******************************************************************************************************************
    #                                           SYSTEM PROPAGATION PHASE
    # ******************************************************************************************************************

    # 1. Build the System Matrix F in ECEF Frame
    f_matrix = np.matrix(np.zeros((17, 17)))

    # Calculate the meridian radius and transverse radius
    [r_ns, r_ew] = radii_of_curv(est_lat_b_old)

    # Calculate the geocentric radius at current latitude
    geo_radius = r_ew * np.sqrt(np.cos(est_lat_b_old) ** 2.0 + ((1.0 - ecc_o ** 2.0) ** 2.0) *
                                np.sin(est_lat_b_old) ** 2.0)

    # For position vector
    f_matrix[0:3, 3:6] = np.eye(3)

    # For velocity vector
    f_matrix[3:6, 0:3] = -(2.0 * gravity_ecef(est_r_eb_e_old) / geo_radius) * (est_r_eb_e_old.T / np.sqrt(
        est_r_eb_e_old.T * est_r_eb_e_old))
    f_matrix[3:6, 3:6] = -2.0 * omega_ie_matrix
    f_matrix[3:6, 6:9] = -skew_sym(est_ctm_b_e_old * meas_f_ib_b)
    f_matrix[3:6, 9:12] = est_ctm_b_e_old

    # For attitude vector
    f_matrix[6:9, 0:3] = -omega_ie_matrix
    f_matrix[6:9, 12:15] = est_ctm_b_e_old

    # For user's clock drift (bias rate)
    f_matrix[15, 16] = 1.0

    # 2. Determine the State Transition Matrix (first-order approximate), PHI = I + F*tau
    phi_matrix = np.matrix(np.eye(17)) + f_matrix * tau_s

    # 3. Determine approximate system noise covariance matrix
    q_prime_matrix = np.matrix(np.zeros((17, 17)))
    q_prime_matrix[3:6, 3:6] = np.eye(3) * tc_kf_config.accel_noise_PSD * tau_s
    q_prime_matrix[6:9, 6:9] = np.eye(3) * tc_kf_config.gyro_noise_PSD * tau_s
    q_prime_matrix[9:12, 9:12] = np.eye(3) * tc_kf_config.accel_bias_PSD * tau_s
    q_prime_matrix[12:15, 12:15] = np.eye(3) * tc_kf_config.gyro_bias_PSD * tau_s
    q_prime_matrix[15, 15] = tc_kf_config.clock_phase_PSD * tau_s
    q_prime_matrix[16, 16] = tc_kf_config.clock_freq_PSD * tau_s

    # 3. Propagate state estimates, noting that only the clock states are non-zero due to closed-loop correction
    x_est_propa = np.nan * np.matrix(np.ones((17, 1)))
    x_est_propa[0:15, 0] = 0.0
    x_est_propa[15, 0] = est_clock_old[0, 0] + est_clock_old[0, 1] * tau_s
    x_est_propa[16, 0] = est_clock_old[0, 1]

    # 4. Propagate state estimation error covariance matrix
    p_matrix_propa = phi_matrix * (p_matrix_old + 0.5 * q_prime_matrix) * phi_matrix.T + 0.5 * q_prime_matrix
    # Check for NaNs
    nans_indx = np.isnan(p_matrix_propa)
    if nans_indx.any():
        for i in xrange(0, 17):
            for j in xrange(0, 17):
                if nans_indx[i, j]:
                    p_matrix_propa[i, j] = 0.0
    # Check for negative infinity
    neginf_indx = np.isneginf(p_matrix_propa)
    if neginf_indx.any():
        for i in xrange(0, 17):
            for j in xrange(0, 17):
                if neginf_indx[i, j]:
                    p_matrix_propa[i, j] = -1.0E+08
    # Check for positive infinity
    posinf_indx = np.isposinf(p_matrix_propa)
    if posinf_indx.any():
        for i in xrange(0, 17):
            for j in xrange(0, 17):
                if posinf_indx[i, j]:
                    p_matrix_propa[i, j] = 1.0E+08

    # ******************************************************************************************************************
    #                                           MEASUREMENT UPDATE PHASE
    # ******************************************************************************************************************
    u_as_e_trps = np.matrix(np.zeros((no_meas, 3)))
    v_as_e_trps = np.matrix(np.zeros((no_meas, 3)))
    pred_meas = np.matrix(np.zeros((no_meas, 2)))

    # 5. Loop over measurements
    for i in xrange(0, no_meas):

        # 5.1 Predict approximated geometric range
        delta_r = gnss_meas[i, 2:5].T - est_r_eb_e_old
        approx_range = np.sqrt(delta_r.T * delta_r)

        # 5.2 Calculate frame rotation during signal transit time
        ctm_e_i = ecef_to_eci_ctm(OMEGA_ie, approx_range, c)

        # 5.3 Predict pseudo-range in ECEF
        delta_r = ctm_e_i * gnss_meas[i, 2:5].T - est_r_eb_e_old
        rangex = np.sqrt(delta_r.T * delta_r)
        pred_meas[i, 0] = rangex + x_est_propa[15, 0]

        # 5.4 Predict line of sight
        u_as_e_trps[i, 0:3] = delta_r.T / rangex

        # 5.5 For pseudorange-range rate position dependent unit vector
        delta_v = ctm_e_i * gnss_meas[i, 5:8].T - est_v_eb_e_old
        v_as_e_trps[i, 0] = (delta_r[0, 0] ** 2 - rangex ** 2) * delta_v[0, 0] / (rangex ** 3)
        v_as_e_trps[i, 1] = (delta_r[1, 0] ** 2 - rangex ** 2) * delta_v[1, 0] / (rangex ** 3)
        v_as_e_trps[i, 2] = (delta_r[2, 0] ** 2 - rangex ** 2) * delta_v[2, 0] / (rangex ** 3)

        # 5.6 Predict pseudo-range rate in ECEF
        rate_term_1 = ctm_e_i * (gnss_meas[i, 5:8].T + omega_ie_matrix * gnss_meas[i, 2:5].T)
        rate_term_2 = est_v_eb_e_old + omega_ie_matrix * est_r_eb_e_old
        range_rate = u_as_e_trps[i, 0:3] * (rate_term_1 - rate_term_2)
        pred_meas[i, 1] = range_rate + x_est_propa[16, 0]

    # End of For Loop on Measurements

    # 6. Set-up measurement matrix
    h_matrix = np.matrix(np.zeros((2 * no_meas, 17)))
    h_matrix[0:no_meas, 0:3] = u_as_e_trps[0:no_meas, 0:3]
    h_matrix[0:no_meas, 15] = np.ones((no_meas, 1))
    h_matrix[no_meas:2 * no_meas, 0:3] = v_as_e_trps[0:no_meas, 0:3]
    h_matrix[no_meas:2 * no_meas, 3:6] = u_as_e_trps[0:no_meas, 0:3]
    h_matrix[no_meas:2 * no_meas, 16] = np.ones((no_meas, 1))

    # 7. Set-up measurement noise covariance matrix assuming all measurements are independent and have equal variance
    #    for a given measurement type
    r_matrix = np.matrix(np.zeros((2 * no_meas, 2 * no_meas)))
    r_matrix[0:no_meas, 0:no_meas] = np.eye(no_meas) * tc_kf_config.pseudo_range_SD ** 2
    r_matrix[no_meas:2 * no_meas, no_meas:2 * no_meas] = np.eye(no_meas) * tc_kf_config.range_rate_SD ** 2

    # 8. Calculate Kalman gain
    k_matrix = p_matrix_propa * h_matrix.T * (h_matrix * p_matrix_propa * h_matrix.T + r_matrix).I

    # 9. Formulate measurement innovations
    delta_z = np.matrix(np.zeros((2 * no_meas, 1)))
    delta_z[0:no_meas, 0] = gnss_meas[0:no_meas, 0] - pred_meas[0:no_meas, 0]
    delta_z[no_meas:2 * no_meas, 0] = gnss_meas[0:no_meas, 1] - pred_meas[0:no_meas, 1]

    # 10. Update state estimates
    x_est_new = x_est_propa + k_matrix * delta_z

    # 11. Update state estimation error covariance matrix
    p_matrix_new = (np.eye(17) - k_matrix * h_matrix) * p_matrix_propa

    # ******************************************************************************************************************
    #                                           CLOSED-LOOP CORRECTION
    # ******************************************************************************************************************

    # 12. Correct attitude, velocity, and position
    est_ctm_b_e_new = (np.eye(3) - skew_sym(x_est_new[6:9])) * est_ctm_b_e_old
    est_v_eb_e_new = est_v_eb_e_old - x_est_new[3:6]
    est_r_eb_e_new = est_r_eb_e_old - x_est_new[0:3]

    # 13. Update IMU bias and GNSS receiver clock estimates
    est_imu_bias_new = est_imu_bias_old + x_est_new[9:15]
    est_clock_new = x_est_new[15:17].T

    return est_ctm_b_e_new, est_v_eb_e_new, est_r_eb_e_new, est_imu_bias_new, est_clock_new, p_matrix_new

# End of Tightly Coupled INS/GNSS EKF Integration for a Single Epoch


'''
    ---------------------------------------------------------------
    17. Main Function to Run the Tightly Coupled INS/GPS EKF Fusion
    ---------------------------------------------------------------
'''


def tc_ins_gps_ekf_fusion(simtype, tightness, true_profile, no_t_steps, eul_err_nb_n, imu_config, gnss_config,
                          tc_kf_config, DyOM, doy, fin_nav, gps_tow):

    print 'Starting Tightly EKF Fusion...'

    # 1. Initialize true navigation solution
    old_t = true_profile[0, 0]                          # starting epoch (s)
    true_lat_b = true_profile[0, 1]                     # initial true latitude (rad)
    true_lon_b = true_profile[0, 2]                     # initial true longitude (rad)
    true_alt_b = true_profile[0, 3]                     # initial true altitude (m)
    true_v_eb_n = true_profile[0, 4:7].T                # initial true velocity vector (m/s)
    true_eul_nb = true_profile[0, 7:10].T               # initial true attitude (rad)
    true_ctm_b_n = euler_to_ctm(true_eul_nb).T          # coordinate transfer matrix from body frame to NED frame

    # 2. Convert all the above parameters to ECEF frame
    [old_true_r_eb_e, old_true_v_eb_e, old_true_ctm_b_e] = lla_to_ecef(true_lat_b, true_lon_b, true_alt_b,
                                                                       true_v_eb_n, true_ctm_b_n)

    # Conditioning the GNSS simulation by the "simtype"
    if simtype == 'simulation':

        # 3. Determine satellite positions and velocities
        [sat_r_es_e, sat_v_es_e] = sat_pv_sim(old_t, gnss_config)

        # 4. Initialize the GNSS biases
        gnss_biases = init_gnss_bias_sim(sat_r_es_e, old_true_r_eb_e, true_lat_b, true_lon_b, gnss_config)

        # 5. Generate GNSS measurements
        [gnss_meas, no_gnss_meas, prn] = gnss_meas_gen_sim(old_t, sat_r_es_e, sat_v_es_e, old_true_r_eb_e, true_lat_b,
                                                           true_lon_b, old_true_v_eb_e, gnss_biases, gnss_config)

    elif simtype == 'play back':

        # 3.1 Process the Ephemeris only in the "play back" Mode
        [iono_alpha, iono_beta, alma_t_para, sv_clock, navigation, ephemeris] = ephem_processing(finpath, fin_nav,
                                                                                                 gps_tow, DyOM)

        # 3.2 Save the Ephemris Data
        navigation_fname = foutpath + 'navigation_message.txt'
        np.savetxt(navigation_fname, navigation)
        ephem_fname = foutpath + 'ephemeris.txt'
        np.savetxt(ephem_fname, ephemeris)

        # 3.3 Determine satellite positions and velocities in ECEF
        [sat_r_es_e, sat_v_es_e] = sat_pv_ephem(ephemeris, gnss_config, old_t, tol)

        # 4. Initialize the GNSS biases
        gnss_biases = init_gnss_bias_ephem(old_t, doy, ephemeris, iono_alpha, iono_beta, true_lat_b, true_lon_b,
                                           true_alt_b, sat_r_es_e, old_true_r_eb_e, gnss_config)

        # 5. Generate GNSS measurements
        [gnss_meas, no_gnss_meas, prn] = \
            gnss_meas_gen_ephem(old_t, sat_r_es_e, sat_v_es_e, old_true_r_eb_e, true_lat_b, true_lon_b,
                                old_true_v_eb_e, gnss_biases, gnss_config, sv_clock, ephemeris, alma_t_para)

    # Array to hold GNSS generated measurements for output
    out_gnss_time = [old_t]
    out_gnss_gen = np.nan * np.ones((1, 16, 9))
    out_gnss_gen[0, 0:len(prn), 0] = prn
    out_gnss_gen[0, 0:len(prn), 1:9] = gnss_meas

    # 6. Determine Least-square GNSS position solutions
    [old_est_r_eb_e, old_est_v_eb_e, est_clock] = gnss_ls_pos_vel(gnss_meas, no_gnss_meas, gnss_config.init_est_r_ea_e,
                                                                  gnss_config.init_est_v_ea_e)

    # 7. Convert Position and Velocity from ECEF to NED
    [old_est_lat_b, old_est_lon_b, old_est_alt_b, old_est_v_eb_n] = pv_ecef_to_lla(old_est_r_eb_e, old_est_v_eb_e)
    est_lat_b = old_est_lat_b

    # 8. Initialize estimated attitude solution
    old_est_ctm_b_n = init_ned_att(true_ctm_b_n, eul_err_nb_n)

    # 9. Compute the CTM from NED to ECEF
    [temp1, temp2, old_est_ctm_b_e] = lla_to_ecef(old_est_lat_b, old_est_lon_b, old_est_alt_b, old_est_v_eb_n,
                                                  old_est_ctm_b_n)

    # 10. Initialize tightly coupled output profile and error profile
    est_profile = np.nan * np.ones((no_t_steps, 10))
    est_errors = np.nan * np.ones((no_t_steps, 10))

    # 11. Generate tightly coupled initial output profile
    est_profile[0, 0] = old_t
    est_profile[0, 1] = old_est_lat_b
    est_profile[0, 2] = old_est_lon_b
    est_profile[0, 3] = old_est_alt_b
    est_profile[0, 4:7] = old_est_v_eb_n.T
    est_profile[0, 7:10] = ctm_to_euler(old_est_ctm_b_n.T).T

    # 13. Determine errors and generate output record
    [delta_r_eb_n, delta_v_eb_n, eul_err_nb_n] = \
        cal_err_ned(old_est_lat_b, old_est_lon_b, old_est_alt_b, old_est_v_eb_n, old_est_ctm_b_n, true_lat_b,
                    true_lon_b, true_alt_b, true_v_eb_n, true_ctm_b_n)

    # 14. Generate tightly coupled initial error profile
    est_errors[0, 0] = old_t
    est_errors[0, 1:4] = delta_r_eb_n.T
    est_errors[0, 4:7] = delta_v_eb_n.T
    est_errors[0, 7:10] = eul_err_nb_n.T

    # 15. Initialize tightly coupled Kalman filter P matrix
    p_matrix = init_p_matrix(tightness, tc_kf_config)

    # 16. Initialize IMU bias states
    est_imu_bias = np.matrix(np.zeros((6, 1)))

    # 17. Initialize IMU quantization residuals
    quant_resid = np.matrix(np.zeros((6, 1)))

    # 18. Generate IMU bias and clock output records
    out_imu_gen = np.nan * np.matrix(np.ones((no_t_steps, 7)))
    out_imu_gen[0, 0] = old_t
    out_imu_gen[0, 1:7] = 0.0
    out_imu_bias_est = np.nan * np.matrix(np.ones((1, 7)))
    out_imu_bias_est[0, 0] = old_t
    out_imu_bias_est[0, 1:7] = est_imu_bias.T
    output_clock = np.nan * np.matrix(np.ones((1, 3)))
    output_clock[0, 0] = old_t
    output_clock[0, 1:3] = est_clock

    # 19. Generate KF uncertainty record
    output_kf_sd = np.nan * np.matrix(np.ones((1, 18)))
    output_kf_sd[0, 0] = old_t
    eig_value = lina.eigvals(p_matrix)
    for i in xrange(0, 17):
        output_kf_sd[0, i + 1] = np.sqrt(eig_value[i])
    # End of For Loop

    # 20. Initialize GNSS model timing
    t_last_gnss = old_t
    gnss_epoch = 1

    # 21. Initialize Progress Bar
    print 'Simulation is in Progress. Please Wait!'

    # ******************************************************************************************************************
    #                                                   MAIN LOOP
    # ******************************************************************************************************************

    for epoch in xrange(1, no_t_steps):

        # 22. Input data from motion profile
        t = true_profile[epoch, 0]                      # current epoch (s)
        true_lat_b = true_profile[epoch, 1]             # current true latitude (rad)
        true_lon_b = true_profile[epoch, 2]             # current true longitude (rad)
        true_alt_b = true_profile[epoch, 3]             # current true altitude (m)
        true_v_eb_n = true_profile[epoch, 4:7].T        # current true velocity vector (m/s)
        true_eul_nb = true_profile[epoch, 7:10].T
        true_ctm_b_n = euler_to_ctm(true_eul_nb).T
        [true_r_eb_e, true_v_eb_e, true_ctm_b_e] = lla_to_ecef(true_lat_b, true_lon_b, true_alt_b, true_v_eb_n,
                                                               true_ctm_b_n)
        tau_i = t - old_t

        # Conditioning the IMU simulation by the "simtype"
        if simtype == 'simulation':

            # 23. Calculate specific force and angular rate
            [true_f_ib_b, true_omega_ib_b] = kinematics_ecef(tau_i, true_ctm_b_e, old_true_ctm_b_e, true_v_eb_e,
                                                             old_true_v_eb_e, old_true_r_eb_e)

            # 24. Simulate IMU errors
            [meas_f_ib_b, meas_omega_ib_b, quant_resid] = imu_model(tau_i, true_f_ib_b, true_omega_ib_b, imu_config,
                                                                    quant_resid)

            out_imu_gen[epoch, 0] = t
            out_imu_gen[epoch, 1:4] = meas_f_ib_b.T
            out_imu_gen[epoch, 4:7] = meas_omega_ib_b.T

        elif simtype == 'play back':

            # 23. Calculate specific force and angular rate
            true_f_ib_b = true_profile[epoch, 16:19].T                          # accelerometer reading (error free)
            true_omega_ib_b = true_profile[epoch, 10:13].T                      # gyroscope reading (error free)

            # 24. Simulate IMU errors
            meas_f_ib_b = true_f_ib_b + true_profile[epoch, 19:22].T            # accelerometer reading
            meas_omega_ib_b = true_omega_ib_b + true_profile[epoch, 13:16].T    # gyroscope reading

            out_imu_gen[epoch, 0] = t
            out_imu_gen[epoch, 1:4] = meas_f_ib_b.T
            out_imu_gen[epoch, 4:7] = meas_omega_ib_b.T

        # 25. Correct IMU errors
        meas_f_ib_b = meas_f_ib_b - est_imu_bias[0:3, 0]
        meas_omega_ib_b = meas_omega_ib_b - est_imu_bias[3:6, 0]

        # 26. Update estimated navigation solution
        [est_r_eb_e, est_v_eb_e, est_ctm_b_e] = nav_eqs_ecef(tau_i, old_est_r_eb_e, old_est_v_eb_e, old_est_ctm_b_e,
                                                             meas_f_ib_b, meas_omega_ib_b)

        # 27. Determine whether to update GNSS simulation and run Kalman filter
        if (t - t_last_gnss) >= gnss_config.epoch_interval:

            gnss_epoch += 1             # update epoch (time) index
            tau_s = t - t_last_gnss     # KF time interval
            t_last_gnss = t             # update the last epoch

            # Conditioning the GNSS simulation by the "simtype"
            if simtype == 'simulation':

                # 28. Determine satellite positions and velocities
                [sat_r_es_e, sat_v_es_e] = sat_pv_sim(t, gnss_config)

                # 29. Generate GNSS measurements
                [gnss_meas, no_gnss_meas, prn] = gnss_meas_gen_sim(t, sat_r_es_e, sat_v_es_e, true_r_eb_e, true_lat_b,
                                                                   true_lon_b, true_v_eb_e, gnss_biases, gnss_config)

            elif simtype == 'play back':

                # 28.1 Determine satellite positions and velocities in ECEF
                [sat_r_es_e, sat_v_es_e] = sat_pv_ephem(ephemeris, gnss_config, t, tol)

                # 28.2 Initialize the GNSS biases
                gnss_biases = init_gnss_bias_ephem(t, doy, ephemeris, iono_alpha, iono_beta, true_lat_b, true_lon_b,
                                                   true_alt_b, sat_r_es_e, old_true_r_eb_e, gnss_config)

                # 29. Generate GNSS measurements
                [gnss_meas, no_gnss_meas, prn] = \
                    gnss_meas_gen_ephem(t, sat_r_es_e, sat_v_es_e, old_true_r_eb_e, true_lat_b, true_lon_b,
                                        old_true_v_eb_e, gnss_biases, gnss_config, sv_clock, ephemeris, alma_t_para)

            # Array to hold GNSS generated measurements for output
            out_gnss_time.append(t)
            out_gnss_gen_new = np.nan * np.ones((gnss_epoch, 16, 9))
            out_gnss_gen_new[0:gnss_epoch - 1, :, :] = out_gnss_gen
            out_gnss_gen_new[gnss_epoch - 1, 0:len(prn), 0] = prn
            out_gnss_gen_new[gnss_epoch - 1, 0:len(prn), 1:9] = gnss_meas
            out_gnss_gen = out_gnss_gen_new

            # 30. Run Integration Kalman filter
            [est_ctm_b_e, est_v_eb_e, est_r_eb_e, est_imu_bias, est_clock, p_matrix] = \
                tc_ekf_epoch(gnss_meas, no_gnss_meas, tau_s, est_ctm_b_e, est_v_eb_e, est_r_eb_e, est_imu_bias,
                             est_clock, p_matrix, meas_f_ib_b, est_lat_b, tc_kf_config)

            # 31. Generate IMU bias and clock output records recursively
            # 31.1 IMU Bias
            out_imu_bias_est_new = np.nan * np.matrix(np.ones((gnss_epoch, 7)))
            out_imu_bias_est_new[0:gnss_epoch - 1, 0] = out_imu_bias_est[0:gnss_epoch - 1, 0]
            out_imu_bias_est_new[gnss_epoch - 1, 0] = t
            out_imu_bias_est_new[0:gnss_epoch - 1, 1:7] = out_imu_bias_est[0:gnss_epoch - 1, 1:7]
            out_imu_bias_est_new[gnss_epoch - 1, 1:7] = est_imu_bias.T
            out_imu_bias_est = out_imu_bias_est_new

            # 31.2 Clock Bias
            out_clock_new = np.nan * np.matrix(np.ones((gnss_epoch, 3)))
            out_clock_new[0:gnss_epoch - 1, 0] = output_clock[0:gnss_epoch - 1, 0]
            out_clock_new[gnss_epoch - 1, 0] = t
            out_clock_new[0:gnss_epoch - 1, 1:3] = output_clock[0:gnss_epoch - 1, 1:3]
            out_clock_new[gnss_epoch - 1, 1:3] = est_clock
            output_clock = out_clock_new

            # 32. Generate KF uncertainty output record recursively
            out_kf_sd_new = np.nan * np.matrix(np.ones((gnss_epoch, 18)))
            out_kf_sd_new[0:gnss_epoch - 1, 0] = output_kf_sd[0:gnss_epoch - 1, 0]
            out_kf_sd_new[gnss_epoch - 1, 0] = t
            out_kf_sd_new[0:gnss_epoch - 1, 1:18] = output_kf_sd[0:gnss_epoch - 1, 1:18]
            eig_value = lina.eigvals(p_matrix)
            for i in xrange(0, 17):
                out_kf_sd_new[gnss_epoch - 1, i + 1] = np.sqrt(eig_value[i])
            # End of For out_kf_sd update

            output_kf_sd = out_kf_sd_new

        # End of If on checking for GNSS update

        # 33. Convert navigation solution to NED
        [est_lat_b, est_lon_b, est_alt_b, est_v_eb_n, est_ctm_b_n] = ecef_to_lla(est_r_eb_e, est_v_eb_e, est_ctm_b_e)

        # 34. Generate output profile record
        est_profile[epoch, 0] = t
        est_profile[epoch, 1] = est_lat_b
        est_profile[epoch, 2] = est_lon_b
        est_profile[epoch, 3] = est_alt_b
        est_profile[epoch, 4:7] = est_v_eb_n.T
        est_profile[epoch, 7:10] = ctm_to_euler(est_ctm_b_n.T).T

        # 35. Determine Errors
        [delta_r_eb_n, delta_v_eb_n, eul_err_nb_n] = cal_err_ned(est_lat_b, est_lon_b, est_alt_b, est_v_eb_n,
                                                                 est_ctm_b_n, true_lat_b, true_lon_b, true_alt_b,
                                                                 true_v_eb_n, true_ctm_b_n)
        # 36. Generate Error Records
        est_errors[epoch, 0] = t
        est_errors[epoch, 1:4] = delta_r_eb_n.T
        est_errors[epoch, 4:7] = delta_v_eb_n.T
        est_errors[epoch, 7:10] = eul_err_nb_n.T

        # 37. Reset old values
        old_t = t
        old_true_r_eb_e = true_r_eb_e
        old_true_v_eb_e = true_v_eb_e
        old_true_ctm_b_e = true_ctm_b_e
        old_est_r_eb_e = est_r_eb_e
        old_est_v_eb_e = est_v_eb_e
        old_est_ctm_b_e = est_ctm_b_e

        # 38. Updating Progress Bar
        progressbar(epoch / float(no_t_steps))

    # End of For Main Loop

    print '\n NavSim Completed!'

    return est_profile, est_errors, output_kf_sd, out_imu_gen, out_imu_bias_est, output_clock, out_gnss_gen, \
           out_gnss_time

# End of Main Tightly Coupled INS/GNSS Fusion


'''
    ---------------------------------------------
    18. Main Function to Run the Dual INS/GPS EKF
    ---------------------------------------------
'''


def dual_ins_gps_ekf_fusion(simtype, true_profile, no_t_steps, eul_err_nb_n, imu_config, gnss_config, lc_kf_config,
                            tc_kf_config, DyOM, doy, fin_nav, gps_tow):

    print 'Starting Dual EKF Fusion...'

    # 1. Initialize true navigation solution
    old_t = true_profile[0, 0]                          # starting epoch (s)
    true_lat_b = true_profile[0, 1]                     # initial true latitude (rad)
    true_lon_b = true_profile[0, 2]                     # initial true longitude (rad)
    true_alt_b = true_profile[0, 3]                     # initial true altitude (m)
    true_v_eb_n = true_profile[0, 4:7].T                # initial true velocity vector (m/s)
    true_eul_nb = true_profile[0, 7:10].T               # initial true attitude (rad)
    true_ctm_b_n = euler_to_ctm(true_eul_nb).T          # coordinate transfer matrix from body frame to NED frame

    # 2. Convert all the above parameters to ECEF frame
    [old_true_r_eb_e, old_true_v_eb_e, old_true_ctm_b_e] = lla_to_ecef(true_lat_b, true_lon_b, true_alt_b,
                                                                       true_v_eb_n, true_ctm_b_n)
    # Conditioning the GNSS simulation by the "simtype"
    if simtype == 'simulation':

        # 3. Determine satellite positions and velocities
        [sat_r_es_e, sat_v_es_e] = sat_pv_sim(old_t, gnss_config)

        # 4. Initialize the GNSS biases
        gnss_biases = init_gnss_bias_sim(sat_r_es_e, old_true_r_eb_e, true_lat_b, true_lon_b, gnss_config)

        # 5. Generate GNSS measurements
        [gnss_meas, no_gnss_meas, prn] = \
            gnss_meas_gen_sim(old_t, sat_r_es_e, sat_v_es_e, old_true_r_eb_e, true_lat_b, true_lon_b, old_true_v_eb_e,
                              gnss_biases, gnss_config)

    elif simtype == 'play back':

        # 3.1 Process the Ephemeris only in the "play back" Mode
        [iono_alpha, iono_beta, alma_t_para, sv_clock, navigation, ephemeris] = ephem_processing(finpath, fin_nav,
                                                                                                 gps_tow, DyOM)

        # 3.2 Save the Ephemris Data
        navigation_fname = foutpath + 'navigation_message.txt'
        np.savetxt(navigation_fname, navigation)
        ephem_fname = foutpath + 'ephemeris.txt'
        np.savetxt(ephem_fname, ephemeris)

        # 3.3 Determine satellite positions and velocities in ECEF
        [sat_r_es_e, sat_v_es_e] = sat_pv_ephem(ephemeris, gnss_config, old_t, tol)

        # 4. Initialize the GNSS biases
        gnss_biases = init_gnss_bias_ephem(old_t, doy, ephemeris, iono_alpha, iono_beta, true_lat_b, true_lon_b,
                                           true_alt_b, sat_r_es_e, old_true_r_eb_e, gnss_config)

        # 5. Generate GNSS measurements
        [gnss_meas, no_gnss_meas, prn] = \
            gnss_meas_gen_ephem(old_t, sat_r_es_e, sat_v_es_e, old_true_r_eb_e, true_lat_b, true_lon_b,
                                old_true_v_eb_e, gnss_biases, gnss_config, sv_clock, ephemeris, alma_t_para)

    # Array to hold GNSS generated measurements for output
    out_gnss_time = [old_t]
    out_gnss_gen = np.nan * np.ones((1, 16, 9))
    out_gnss_gen[0, 0:len(prn), 0] = prn
    out_gnss_gen[0, 0:len(prn), 1:9] = gnss_meas

    # 6. Determine Least-Square GNSS position solutions
    [gnss_r_eb_e, gnss_v_eb_e, est_clock] = gnss_ls_pos_vel(gnss_meas, no_gnss_meas, gnss_config.init_est_r_ea_e,
                                                            gnss_config.init_est_v_ea_e)
    old_est_r_eb_e = gnss_r_eb_e
    old_est_v_eb_e = gnss_v_eb_e

    # 6.1 Initialize loosely coupled pos, vel estimations
    lc_old_est_r_eb_e = gnss_r_eb_e
    lc_old_est_v_eb_e = gnss_v_eb_e

    # 6.2 Initialize tightly coupled pos, vel estimations
    tc_old_est_r_eb_e = gnss_r_eb_e
    tc_old_est_v_eb_e = gnss_v_eb_e
    tc_est_clock = est_clock

    # 7. Convert Position and Velocity from ECEF to NED
    [old_est_lat_b, old_est_lon_b, old_est_alt_b, old_est_v_eb_n] = pv_ecef_to_lla(old_est_r_eb_e, old_est_v_eb_e)

    # 7.1 Initialize loosely coupled latitude estimation
    lc_est_lat_b = old_est_lat_b

    # 7.2 Initialize tightly coupled latitude estimation
    tc_est_lat_b = old_est_lat_b

    # 8. Initialize estimated attitude solution
    old_est_ctm_b_n = init_ned_att(true_ctm_b_n, eul_err_nb_n)

    # 9. Compute the CTM from NED to ECEF
    [temp1, temp2, old_est_ctm_b_e] = lla_to_ecef(old_est_lat_b, old_est_lon_b, old_est_alt_b, old_est_v_eb_n,
                                                  old_est_ctm_b_n)

    # 9.1 Initialize loosely coupled CTM from NED to ECEF
    lc_old_est_ctm_b_e = old_est_ctm_b_e

    # 9.2 Initialize tightly coupled CTM from NED to ECEF
    tc_old_est_ctm_b_e = old_est_ctm_b_e

    # 10. Initialize loosely and tightly coupled output profiles
    lc_est_profile = np.nan * np.ones((no_t_steps, 10))
    tc_est_profile = np.nan * np.ones((no_t_steps, 10))

    # 11. Generate loosely coupled output profile
    lc_est_profile[0, 0] = old_t
    lc_est_profile[0, 1] = old_est_lat_b
    lc_est_profile[0, 2] = old_est_lon_b
    lc_est_profile[0, 3] = old_est_alt_b
    lc_est_profile[0, 4:7] = old_est_v_eb_n.T
    lc_est_profile[0, 7:10] = ctm_to_euler(old_est_ctm_b_n.T).T

    # 12. Generate tightly coupled output profile
    tc_est_profile[0, 0] = old_t
    tc_est_profile[0, 1] = old_est_lat_b
    tc_est_profile[0, 2] = old_est_lon_b
    tc_est_profile[0, 3] = old_est_alt_b
    tc_est_profile[0, 4:7] = old_est_v_eb_n.T
    tc_est_profile[0, 7:10] = ctm_to_euler(old_est_ctm_b_n.T).T

    # 13. Determine errors and generate output record
    [delta_r_eb_n, delta_v_eb_n, eul_err_nb_n] = \
        cal_err_ned(old_est_lat_b, old_est_lon_b, old_est_alt_b, old_est_v_eb_n, old_est_ctm_b_n, true_lat_b,
                    true_lon_b, true_alt_b, true_v_eb_n, true_ctm_b_n)

    # 14. Initialize loosely and tightly coupled error profiles
    lc_est_errors = np.nan * np.ones((no_t_steps, 10))
    tc_est_errors = np.nan * np.ones((no_t_steps, 10))

    # 15. Loosely coupled errors
    lc_est_errors[0, 0] = old_t
    lc_est_errors[0, 1:4] = delta_r_eb_n.T
    lc_est_errors[0, 4:7] = delta_v_eb_n.T
    lc_est_errors[0, 7:10] = eul_err_nb_n.T

    # 16. Tightly coupled errors
    tc_est_errors[0, 0] = old_t
    tc_est_errors[0, 1:4] = delta_r_eb_n.T
    tc_est_errors[0, 4:7] = delta_v_eb_n.T
    tc_est_errors[0, 7:10] = eul_err_nb_n.T

    # 17. Dual coupled Kalman filter
    [lc_p_matrix, tc_p_matrix] = init_dual_p_matrix(lc_kf_config, tc_kf_config)

    # 18. Initialize IMU bias states
    lc_est_imu_bias = np.matrix(np.zeros((6, 1)))
    tc_est_imu_bias = np.matrix(np.zeros((6, 1)))

    # 19. Initialize IMU quantization residuals
    quant_resid = np.matrix(np.zeros((6, 1)))

    # 20. Generate IMU bias and clock output records
    out_imu_gen = np.nan * np.matrix(np.ones((no_t_steps, 7)))
    out_imu_gen[0, 0] = old_t
    out_imu_gen[0, 1:7] = 0.0
    # 20.1 Loosely coupled
    lc_out_imu_bias_est = np.nan * np.matrix(np.ones((1, 7)))
    lc_out_imu_bias_est[0, 0] = old_t
    lc_out_imu_bias_est[0, 1:7] = lc_est_imu_bias.T
    lc_output_clock = np.nan * np.matrix(np.ones((1, 3)))
    lc_output_clock[0, 0] = old_t
    lc_output_clock[0, 1:3] = est_clock

    # 20.2 Tightly coupled
    tc_out_imu_bias_est = np.nan * np.matrix(np.ones((1, 7)))
    tc_out_imu_bias_est[0, 0] = old_t
    tc_out_imu_bias_est[0, 1:7] = tc_est_imu_bias.T
    tc_output_clock = np.nan * np.matrix(np.ones((1, 3)))
    tc_output_clock[0, 0] = old_t
    tc_output_clock[0, 1:3] = est_clock

    # 21. Generate KF uncertainty record
    # 21.1 Loosely coupled EKF
    lc_output_kf_sd = np.nan * np.matrix(np.ones((1, 16)))
    lc_output_kf_sd[0, 0] = old_t
    lc_eig_value = lina.eigvals(lc_p_matrix)
    for i in xrange(0, 15):
        lc_output_kf_sd[0, i + 1] = np.sqrt(lc_eig_value[i])
    # End of For Loop

    # 21.2 Tightly coupled EKF
    tc_output_kf_sd = np.nan * np.matrix(np.ones((1, 18)))
    tc_output_kf_sd[0, 0] = old_t
    tc_eig_value = lina.eigvals(tc_p_matrix)
    for i in xrange(0, 17):
        tc_output_kf_sd[0, i + 1] = np.sqrt(tc_eig_value[i])
    # End of For Loop

    # 22. Initialize GNSS model timing
    t_last_gnss = old_t
    gnss_epoch = 1

    # 23. Initialize Progress Bar
    print 'Simulation is in progress. Please wait!'

    # ******************************************************************************************************************
    #                                                   MAIN LOOP
    # ******************************************************************************************************************

    for epoch in xrange(1, no_t_steps):

        # 24. Input data from motion profile
        t = true_profile[epoch, 0]                      # current epoch (s)
        true_lat_b = true_profile[epoch, 1]             # current true latitude (rad)
        true_lon_b = true_profile[epoch, 2]             # current true longitude (rad)
        true_alt_b = true_profile[epoch, 3]             # current true altitude (m)
        true_v_eb_n = true_profile[epoch, 4:7].T        # current true velocity vector (m/s)
        true_eul_nb = true_profile[epoch, 7:10].T
        true_ctm_b_n = euler_to_ctm(true_eul_nb).T
        [true_r_eb_e, true_v_eb_e, true_ctm_b_e] = lla_to_ecef(true_lat_b, true_lon_b, true_alt_b, true_v_eb_n,
                                                               true_ctm_b_n)
        # 25. Calculate the time interval
        tau_i = t - old_t

        # Conditioning the IMU simulation by the "simtype"
        if simtype == 'simulation':

            # 26. Calculate specific force and angular rate
            [true_f_ib_b, true_omega_ib_b] = kinematics_ecef(tau_i, true_ctm_b_e, old_true_ctm_b_e, true_v_eb_e,
                                                             old_true_v_eb_e, old_true_r_eb_e)

            # 27. Simulate IMU errors
            [meas_f_ib_b, meas_omega_ib_b, quant_resid] = imu_model(tau_i, true_f_ib_b, true_omega_ib_b, imu_config,
                                                                    quant_resid)

            out_imu_gen[epoch, 0] = t
            out_imu_gen[epoch, 1:4] = meas_f_ib_b.T
            out_imu_gen[epoch, 4:7] = meas_omega_ib_b.T

        elif simtype == 'play back':

            # 26. Calculate specific force and angular rate
            true_f_ib_b = true_profile[epoch, 16:19].T          # accelerometer reading (error free)
            true_omega_ib_b = true_profile[epoch, 10:13].T      # gyroscope reading (error free)

            # 27. Simulate IMU errors
            meas_f_ib_b = true_f_ib_b + true_profile[epoch, 19:22].T            # accelerometer reading
            meas_omega_ib_b = true_omega_ib_b + true_profile[epoch, 13:16].T    # gyroscope reading

            out_imu_gen[epoch, 0] = t
            out_imu_gen[epoch, 1:4] = meas_f_ib_b.T
            out_imu_gen[epoch, 4:7] = meas_omega_ib_b.T

        # 28. Correct IMU errors
        # 28.1 Loosely coupled IMU error corrections
        lc_meas_f_ib_b = meas_f_ib_b - lc_est_imu_bias[0:3, 0]
        lc_meas_omega_ib_b = meas_omega_ib_b - lc_est_imu_bias[3:6, 0]

        # 28.2 Tightly coupled IMU error corrections
        tc_meas_f_ib_b = meas_f_ib_b - tc_est_imu_bias[0:3, 0]
        tc_meas_omega_ib_b = meas_omega_ib_b - tc_est_imu_bias[3:6, 0]

        # 29. Update estimated navigation solution
        # 29.1 Loosely coupled estimated navigation solution update
        [lc_est_r_eb_e, lc_est_v_eb_e, lc_est_ctm_b_e] = nav_eqs_ecef(
            tau_i, lc_old_est_r_eb_e, lc_old_est_v_eb_e, lc_old_est_ctm_b_e, lc_meas_f_ib_b, lc_meas_omega_ib_b)

        # 29.2 Tightly coupled estimated navigation solution update
        [tc_est_r_eb_e, tc_est_v_eb_e, tc_est_ctm_b_e] = nav_eqs_ecef(
            tau_i, tc_old_est_r_eb_e, tc_old_est_v_eb_e, tc_old_est_ctm_b_e, tc_meas_f_ib_b, tc_meas_omega_ib_b)

        # 30. Determine whether to update GNSS simulation and run Kalman filter
        if (t - t_last_gnss) >= gnss_config.epoch_interval:

            gnss_epoch += 1             # update epoch (time) index
            tau_s = t - t_last_gnss     # KF time interval
            t_last_gnss = t             # update the last epoch

            # Conditioning the GNSS simulation by the "simtype"
            if simtype == 'simulation':

                # 31. Determine satellite positions and velocities in ECEF
                [sat_r_es_e, sat_v_es_e] = sat_pv_sim(t, gnss_config)

                # 32. Generate GNSS measurements
                [gnss_meas, no_gnss_meas, prn] = gnss_meas_gen_sim(t, sat_r_es_e, sat_v_es_e, true_r_eb_e, true_lat_b,
                                                                   true_lon_b, true_v_eb_e, gnss_biases, gnss_config)

            elif simtype == 'play back':

                # 31.1 Determine satellite positions and velocities in ECEF
                [sat_r_es_e, sat_v_es_e] = sat_pv_ephem(ephemeris, gnss_config, t, tol)

                # 31.2 Initialize the GNSS biases
                gnss_biases = init_gnss_bias_ephem(t, doy, ephemeris, iono_alpha, iono_beta, true_lat_b, true_lon_b,
                                                   true_alt_b, sat_r_es_e, old_true_r_eb_e, gnss_config)

                # 32. Generate GNSS measurements
                [gnss_meas, no_gnss_meas, prn] = \
                    gnss_meas_gen_ephem(t, sat_r_es_e, sat_v_es_e, old_true_r_eb_e, true_lat_b, true_lon_b,
                                        old_true_v_eb_e, gnss_biases, gnss_config, sv_clock, ephemeris, alma_t_para)

            # Array to hold GNSS generated measurements for output
            out_gnss_time.append(t)
            out_gnss_gen_new = np.nan * np.ones((gnss_epoch, 16, 9))
            out_gnss_gen_new[0:gnss_epoch - 1, :, :] = out_gnss_gen
            out_gnss_gen_new[gnss_epoch - 1, 0:len(prn), 0] = prn
            out_gnss_gen_new[gnss_epoch - 1, 0:len(prn), 1:9] = gnss_meas
            out_gnss_gen = out_gnss_gen_new

            # 33. Determine Least-Square GNSS position solutions for loosely coupling
            [gnss_r_eb_e, gnss_v_eb_e, lc_est_clock] = gnss_ls_pos_vel(gnss_meas, no_gnss_meas, gnss_r_eb_e,
                                                                       gnss_v_eb_e)

            # 34. Run Loosely Coupled Integration Kalman filter
            [lc_est_ctm_b_e, lc_est_v_eb_e, lc_est_r_eb_e, lc_est_imu_bias, lc_p_matrix] = \
                lc_ekf_epoch(gnss_r_eb_e, gnss_v_eb_e, tau_s, lc_est_ctm_b_e, lc_est_v_eb_e, lc_est_r_eb_e,
                             lc_est_imu_bias, lc_p_matrix, lc_meas_f_ib_b, lc_est_lat_b, lc_kf_config)

            # 35. Run Tightly Coupled Integration Kalman filter
            [tc_est_ctm_b_e, tc_est_v_eb_e, tc_est_r_eb_e, tc_est_imu_bias, tc_est_clock, tc_p_matrix] = \
                tc_ekf_epoch(gnss_meas, no_gnss_meas, tau_s, tc_est_ctm_b_e, tc_est_v_eb_e, tc_est_r_eb_e,
                             tc_est_imu_bias, tc_est_clock, tc_p_matrix, tc_meas_f_ib_b, tc_est_lat_b, tc_kf_config)

            # 36. Generate IMU bias and clock output records recursively
            # 36.1 Loosely coupled IMU bias
            lc_out_imu_bias_est_new = np.nan * np.matrix(np.ones((gnss_epoch, 7)))
            lc_out_imu_bias_est_new[0:gnss_epoch - 1, 0] = lc_out_imu_bias_est[0:gnss_epoch - 1, 0]
            lc_out_imu_bias_est_new[gnss_epoch - 1, 0] = t
            lc_out_imu_bias_est_new[0:gnss_epoch - 1, 1:7] = lc_out_imu_bias_est[0:gnss_epoch - 1, 1:7]
            lc_out_imu_bias_est_new[gnss_epoch - 1, 1:7] = lc_est_imu_bias.T

            lc_out_imu_bias_est = lc_out_imu_bias_est_new

            # 36.2 Tightly coupled IMU bias
            tc_out_imu_bias_est_new = np.nan * np.matrix(np.ones((gnss_epoch, 7)))
            tc_out_imu_bias_est_new[0:gnss_epoch - 1, 0] = tc_out_imu_bias_est[0:gnss_epoch - 1, 0]
            tc_out_imu_bias_est_new[gnss_epoch - 1, 0] = t
            tc_out_imu_bias_est_new[0:gnss_epoch - 1, 1:7] = tc_out_imu_bias_est[0:gnss_epoch - 1, 1:7]
            tc_out_imu_bias_est_new[gnss_epoch - 1, 1:7] = tc_est_imu_bias.T

            tc_out_imu_bias_est = tc_out_imu_bias_est_new

            # 36.3 Loosely coupled clock bias
            lc_out_clock_new = np.nan * np.matrix(np.ones((gnss_epoch, 3)))
            lc_out_clock_new[0:gnss_epoch - 1, 0] = lc_output_clock[0:gnss_epoch - 1, 0]
            lc_out_clock_new[gnss_epoch - 1, 0] = t
            lc_out_clock_new[0:gnss_epoch - 1, 1:3] = lc_output_clock[0:gnss_epoch - 1, 1:3]
            lc_out_clock_new[gnss_epoch - 1, 1:3] = lc_est_clock

            lc_output_clock = lc_out_clock_new

            # 36.4 Tightly coupled clock bias
            tc_out_clock_new = np.nan * np.matrix(np.ones((gnss_epoch, 3)))
            tc_out_clock_new[0:gnss_epoch - 1, 0] = tc_output_clock[0:gnss_epoch - 1, 0]
            tc_out_clock_new[gnss_epoch - 1, 0] = t
            tc_out_clock_new[0:gnss_epoch - 1, 1:3] = tc_output_clock[0:gnss_epoch - 1, 1:3]
            tc_out_clock_new[gnss_epoch - 1, 1:3] = tc_est_clock

            tc_output_clock = tc_out_clock_new

            # 37. Generate EKF uncertainty output record recursively
            # 37.1 Loosely coupled EKF standard deviation
            lc_out_kf_sd_new = np.nan * np.matrix(np.ones((gnss_epoch, 16)))
            lc_out_kf_sd_new[0:gnss_epoch - 1, 0] = lc_output_kf_sd[0:gnss_epoch - 1, 0]
            lc_out_kf_sd_new[gnss_epoch - 1, 0] = t
            lc_out_kf_sd_new[0:gnss_epoch - 1, 1:16] = lc_output_kf_sd[0:gnss_epoch - 1, 1:16]
            lc_eig_value = lina.eigvals(lc_p_matrix)
            for i in xrange(0, 15):
                lc_out_kf_sd_new[gnss_epoch - 1, i + 1] = np.sqrt(lc_eig_value[i])
            # End of For out_kf_sd update

            lc_output_kf_sd = lc_out_kf_sd_new

            # 37.2 Tightly coupled EKF standard deviation
            tc_out_kf_sd_new = np.nan * np.matrix(np.ones((gnss_epoch, 18)))
            tc_out_kf_sd_new[0:gnss_epoch - 1, 0] = tc_output_kf_sd[0:gnss_epoch - 1, 0]
            tc_out_kf_sd_new[gnss_epoch - 1, 0] = t
            tc_out_kf_sd_new[0:gnss_epoch - 1, 1:18] = tc_output_kf_sd[0:gnss_epoch - 1, 1:18]
            tc_eig_value = lina.eigvals(tc_p_matrix)
            for i in xrange(0, 17):
                tc_out_kf_sd_new[gnss_epoch - 1, i + 1] = np.sqrt(tc_eig_value[i])
            # End of For out_kf_sd update

            tc_output_kf_sd = tc_out_kf_sd_new

        # End of "If" on checking for GNSS update

        # 38. Convert navigation solution to NED
        # 38.1 Loosely coupled
        [lc_est_lat_b, lc_est_lon_b, lc_est_alt_b, lc_est_v_eb_n, lc_est_ctm_b_n] = \
            ecef_to_lla(lc_est_r_eb_e, lc_est_v_eb_e, lc_est_ctm_b_e)

        # 38.2 Tightly coupled
        [tc_est_lat_b, tc_est_lon_b, tc_est_alt_b, tc_est_v_eb_n, tc_est_ctm_b_n] = \
            ecef_to_lla(tc_est_r_eb_e, tc_est_v_eb_e, tc_est_ctm_b_e)

        # 39. Generate output profile record
        # 39.1 Loosely coupled EKF outputs
        lc_est_profile[epoch, 0] = t
        lc_est_profile[epoch, 1] = lc_est_lat_b
        lc_est_profile[epoch, 2] = lc_est_lon_b
        lc_est_profile[epoch, 3] = lc_est_alt_b
        lc_est_profile[epoch, 4:7] = lc_est_v_eb_n.T
        lc_est_profile[epoch, 7:10] = ctm_to_euler(lc_est_ctm_b_n.T).T

        # 39.2 Tightly coupled EKF outputs
        tc_est_profile[epoch, 0] = t
        tc_est_profile[epoch, 1] = tc_est_lat_b
        tc_est_profile[epoch, 2] = tc_est_lon_b
        tc_est_profile[epoch, 3] = tc_est_alt_b
        tc_est_profile[epoch, 4:7] = tc_est_v_eb_n.T
        tc_est_profile[epoch, 7:10] = ctm_to_euler(tc_est_ctm_b_n.T).T

        # 40. Determine Errors
        # 40.1 Loosely coupled
        [lc_delta_r_eb_n, lc_delta_v_eb_n, lc_eul_err_nb_n] = \
            cal_err_ned(lc_est_lat_b, lc_est_lon_b, lc_est_alt_b, lc_est_v_eb_n, lc_est_ctm_b_n, true_lat_b,
                        true_lon_b, true_alt_b, true_v_eb_n, true_ctm_b_n)

        # 40.2 Tightly coupled
        [tc_delta_r_eb_n, tc_delta_v_eb_n, tc_eul_err_nb_n] = \
            cal_err_ned(tc_est_lat_b, tc_est_lon_b, tc_est_alt_b, tc_est_v_eb_n, tc_est_ctm_b_n, true_lat_b,
                        true_lon_b, true_alt_b, true_v_eb_n, true_ctm_b_n)

        # 41. Generate Error Records
        # 41.1 Loosely coupled error records
        lc_est_errors[epoch, 0] = t
        lc_est_errors[epoch, 1:4] = lc_delta_r_eb_n.T
        lc_est_errors[epoch, 4:7] = lc_delta_v_eb_n.T
        lc_est_errors[epoch, 7:10] = lc_eul_err_nb_n.T

        # 41.2 Tightly coupled error records
        tc_est_errors[epoch, 0] = t
        tc_est_errors[epoch, 1:4] = tc_delta_r_eb_n.T
        tc_est_errors[epoch, 4:7] = tc_delta_v_eb_n.T
        tc_est_errors[epoch, 7:10] = tc_eul_err_nb_n.T

        # 42. Reset old values
        # 42.1 Time and True Pos, Vel, Att
        old_t = t
        old_true_r_eb_e = true_r_eb_e
        old_true_v_eb_e = true_v_eb_e
        old_true_ctm_b_e = true_ctm_b_e

        # 42.2 Loosely coupled Pos, Vel, Att
        lc_old_est_r_eb_e = lc_est_r_eb_e
        lc_old_est_v_eb_e = lc_est_v_eb_e
        lc_old_est_ctm_b_e = lc_est_ctm_b_e

        # 42.3 Tightly coupled Pos, Vel, Att
        tc_old_est_r_eb_e = tc_est_r_eb_e
        tc_old_est_v_eb_e = tc_est_v_eb_e
        tc_old_est_ctm_b_e = tc_est_ctm_b_e

        # 43. Updating Progress Bar
        progressbar(epoch / float(no_t_steps))

    # End of For Main Loop

    print '\n NavSim Completed!'

    return lc_est_profile, lc_est_errors, lc_output_kf_sd, tc_est_profile, tc_est_errors, tc_output_kf_sd, out_imu_gen,\
           lc_out_imu_bias_est, tc_out_imu_bias_est, lc_output_clock, tc_output_clock, out_gnss_gen, out_gnss_time

# End of Main Dual INS/GNSS Fusion


'''
========================================================================================================================
                                        INS/GNSS FUSION SIMULATION DRIVER
========================================================================================================================
'''


def nav_sim_driver(fin_data, fin_nav, DyOM, doy, imugrade, constellation, frequency, simtype, simmode,
                   tightness):

    # 1.1 Process the Flight Data for All Modes
    [in_profile, epochs] = data_processing(finpath, fin_data)
    gps_tow = in_profile[:, 29]
    [DyOW, HrOD, MnOH, ScOM] = tow_to_utc(gps_tow[0])
    print "Initial TOW in UTC: %d Days %d Hours %d Minutes %.2f Seconds" % (DyOW, HrOD, MnOH, ScOM)
    [deltaMnOW, rem_deltaTOW] = flight_duration(gps_tow[0], gps_tow[-1])
    print "Flight Duration: %d Minutes %.2f Seconds" % (deltaMnOW, rem_deltaTOW)

    # 1.2 Save the Flight Data to Simout Directory
    words = fin_data.split('.')
    flightdata_fname = foutpath + words[0] + '_in_profile.txt'
    np.savetxt(flightdata_fname, in_profile)

    # 2. Initialize the Navigation Simulation (NavSim) System
    print 'Initializing NavSim...'
    delta_eul_nb_n = att_init_error(-0.01, 0.008, 0.01, unit='degree')
    imu_config = imu_configuration(imugrade)
    gnss_config = gnss_configuration(frequency, constellation, gps_tow)
    if simmode == 'alone':
        ekf_config = single_ekf_configuration(imugrade, tightness)
    elif simmode == 'dual':
        [lc_ekf_config, tc_ekf_config] = dual_ekf_configuration(imugrade)

    # End of If on initialization

    # 3. Call Main Fusion Function to Start Simulation
    if simmode == 'alone' and tightness == 'loose':
        [out_profile, out_errors, out_kf_sd, out_imu_gen, out_imu_bias_est, output_clock, out_gnss_gen,
         out_gnss_time] = lc_ins_gps_ekf_fusion(simtype, tightness, in_profile, epochs, delta_eul_nb_n, imu_config,
                                                gnss_config, ekf_config, DyOM, doy, fin_nav, gps_tow)

        # Save simulation outputs to Simout directory
        outprofile_fname = foutpath + words[0] + '_out_profile_lc.txt'
        np.savetxt(outprofile_fname, out_profile)
        outerrors_fname = foutpath + words[0] + '_out_errors_lc.txt'
        np.savetxt(outerrors_fname, out_errors)
        outekf_sds_fname = foutpath + words[0] + '_out_EKF_SDs_lc.txt'
        np.savetxt(outekf_sds_fname, out_kf_sd)
        outimu_fname = foutpath + words[0] + '_out_IMU_Gen_lc.txt'
        np.savetxt(outimu_fname, out_imu_gen)
        outimu_bias_fname = foutpath + words[0] + '_out_IMU_Bias_lc.txt'
        np.savetxt(outimu_bias_fname, out_imu_bias_est)
        outlock_bias_fname = foutpath + words[0] + '_out_Rx_Lock_Bias_lc.txt'
        np.savetxt(outlock_bias_fname, output_clock)
        outgnss_fname = foutpath + 'gnss_measurements.txt'
        with file(outgnss_fname, 'w') as outfile:
            outfile.write('GNSS Measurement array shape: {0}\n'.format(out_gnss_gen.shape))
            epoch = 0
            for gnss_slice in out_gnss_gen:
                string = 'Epoch (sec): ' + str(out_gnss_time[epoch]) + '\n'
                outfile.write(string)
                np.savetxt(outfile, gnss_slice)
                epoch += 1

        # Plot Output Data for Graphical Analysis
        print ' Begin Plotting Results...'
        plot_single_profile(in_profile, out_profile)
        plot_single_error(out_errors, out_kf_sd)
        print 'Finish!'
        plt.show()

    elif simmode == 'alone' and tightness == 'tight':
        [out_profile, out_errors, out_kf_sd, out_imu_gen, out_imu_bias_est, output_clock, out_gnss_gen,
         out_gnss_time] = tc_ins_gps_ekf_fusion(simtype, tightness, in_profile, epochs, delta_eul_nb_n, imu_config,
                                                gnss_config, ekf_config, DyOM, doy, fin_nav, gps_tow)

        # Save simulation outputs
        outprofile_fname = foutpath + words[0] + '_out_profile_tc.txt'
        np.savetxt(outprofile_fname, out_profile)
        outerrors_fname = foutpath + words[0] + '_out_errors_tc.txt'
        np.savetxt(outerrors_fname, out_errors)
        outekf_sds_fname = foutpath + words[0] + '_out_EKF_SDs_tc.txt'
        np.savetxt(outekf_sds_fname, out_kf_sd)
        outimu_fname = foutpath + words[0] + '_out_IMU_Gen_tc.txt'
        np.savetxt(outimu_fname, out_imu_gen)
        outimu_bias_fname = foutpath + words[0] + '_out_IMU_Bias_tc.txt'
        np.savetxt(outimu_bias_fname, out_imu_bias_est)
        outlock_bias_fname = foutpath + words[0] + '_out_Rx_Lock_Bias_tc.txt'
        np.savetxt(outlock_bias_fname, output_clock)
        outgnss_fname = foutpath + 'gnss_measurements.txt'
        with file(outgnss_fname, 'w') as outfile:
            outfile.write('GNSS Measurement array shape: {0}\n'.format(out_gnss_gen.shape))
            epoch = 0
            for gnss_slice in out_gnss_gen:
                string = 'Epoch (sec): ' + str(out_gnss_time[epoch]) + '\n'
                outfile.write(string)
                np.savetxt(outfile, gnss_slice)
                epoch += 1

        # Plot Output Data for Graphical Analysis
        print ' Begin Plotting Results...'
        plot_single_profile(in_profile, out_profile)
        plot_single_error(out_errors, out_kf_sd)
        print 'Finish!'
        plt.show()

    elif simmode == 'dual':
        [lc_out_profile, lc_out_errors, lc_out_kf_sd, tc_out_profile, tc_out_errors, tc_out_kf_sd, out_imu_gen,
         lc_out_imu_bias_est, tc_out_imu_bias_est, lc_output_clock, tc_output_clock, out_gnss_gen, out_gnss_time] = \
            dual_ins_gps_ekf_fusion(simtype, in_profile, epochs, delta_eul_nb_n, imu_config, gnss_config, lc_ekf_config,
                                    tc_ekf_config, DyOM, doy, fin_nav, gps_tow)

        # Save simulation outputs from LC
        lc_outprofile_fname = foutpath + words[0] + '_out_profile_lc.txt'
        np.savetxt(lc_outprofile_fname, lc_out_profile)
        lc_outerrors_fname = foutpath + words[0] + '_out_errors_lc.txt'
        np.savetxt(lc_outerrors_fname, lc_out_errors)
        lc_outekf_sds_fname = foutpath + words[0] + '_out_EKF_SDs_lc.txt'
        np.savetxt(lc_outekf_sds_fname, lc_out_kf_sd)
        outimu_fname = foutpath + words[0] + '_out_IMU_Gen.txt'
        np.savetxt(outimu_fname, out_imu_gen)
        lc_outimu_bias_fname = foutpath + words[0] + '_out_IMU_Bias_lc.txt'
        np.savetxt(lc_outimu_bias_fname, lc_out_imu_bias_est)
        lc_outlock_bias_fname = foutpath + words[0] + '_out_Rx_Lock_Bias_lc.txt'
        np.savetxt(lc_outlock_bias_fname, lc_output_clock)

        # Save simulation outputs from TC
        tc_outprofile_fname = foutpath + words[0] + '_out_profile_tc.txt'
        np.savetxt(tc_outprofile_fname, tc_out_profile)
        tc_outerrors_fname = foutpath + words[0] + '_out_errors_tc.txt'
        np.savetxt(tc_outerrors_fname, tc_out_errors)
        tc_outekf_sds_fname = foutpath + words[0] + '_out_EKF_SDs_tc.txt'
        np.savetxt(tc_outekf_sds_fname, tc_out_kf_sd)
        tc_outimu_bias_fname = foutpath + words[0] + '_out_IMU_Bias_tc.txt'
        np.savetxt(tc_outimu_bias_fname, tc_out_imu_bias_est)
        tc_outlock_bias_fname = foutpath + words[0] + '_out_Rx_Lock_Bias_tc.txt'
        np.savetxt(tc_outlock_bias_fname, tc_output_clock)
        outgnss_fname = foutpath + 'gnss_measurements.txt'
        with file(outgnss_fname, 'w') as outfile:
            outfile.write('GNSS Measurement array shape: {0}\n'.format(out_gnss_gen.shape))
            epoch = 0
            for gnss_slice in out_gnss_gen:
                string = 'Epoch (sec): ' + str(out_gnss_time[epoch]) + '\n'
                outfile.write(string)
                np.savetxt(outfile, gnss_slice)
                epoch += 1

        # Plot Output Data for Graphical Analysis
        print ' Begin Plotting Results...'
        plot_dual_profile(in_profile, lc_out_profile, tc_out_profile)
        plot_dual_error(lc_out_errors, lc_out_kf_sd, tc_out_errors, tc_out_kf_sd)
        print 'Finish!'
        plt.show()

    # End of If on calling main fusion runner

# End of Navigation Simulation Driver


# **********************************************************************************************************************
#                                                  SET UP THE SIMULATION
# **********************************************************************************************************************

# 1. Specify the flight data file and the ephemeris file
data_fname = raw_input('Enter the flight data file name (.mat file): ')
nav_fname = raw_input('\nEnter the navigation message file name (.xxn file, include the file extension): ')
nav_words = nav_fname.split('.')
while len(nav_words) != 2:
    nav_fname = raw_input('\nEnter the navigation message file name (.xxn file, include the file extension): ')
    nav_words = nav_fname.split('.')

# 2. Specify the date of the flight
date = raw_input('\nPlease enter the date when the flight test was carried out (mm/dd/yyyy): ')
num_chars = len(date)
while num_chars != 10:
    date = raw_input('\nPlease enter the date when the flight test was carried out (mm/dd/yyyy): ')
    num_chars = len(date)
numbers = date.split('/')
month = int(numbers[0])
# The number of days of the month
num_dom = int(numbers[1])
year = int(numbers[2])
# Calculate the number of days of the year
num_doy = days_of_year(year, month, num_dom)

# 3. Specify the grade of the IMU
imu_grade = raw_input('\nPlease specify the grade of the IMU ("aviation", "consumer", and "tactical"): ')
while imu_grade != 'aviation' and imu_grade != 'consumer' and imu_grade != 'tactical':
    imu_grade = raw_input('\nPlease specify the grade of the IMU ("aviation", "consumer", and "tactical"): ')

# 4. Specify the GNSS constellation
print '\nCurrently, NavSim Lab has only the GPS option for GNSS constellation. Please enter "gps" at prompt.'
gnss = raw_input('Please specify the GNSS constellation ("gps", "glonass", galileo"): ')
while gnss != 'gps':
    gnss = raw_input('\nPlease specify the GNSS constellation as "gps": ')

# 5. Specify the integration update rate
freq = input('\nPlease enter the integration rate (GNSS update rate) in Hz: ')

# 6. Select the simulation type
print '\nNavSim Lab has two simulation types:\n\n' \
      '     Type #1 == "simulation": performs the flight by the simulating the virtual GNSS constellation and the ' \
      'virtual IMU model based on the flight trajectory, the vehicle dynamics, the vehicle attitude from the '\
      'flight data.\n\n' \
      '     Type #2 == "play back": performs the flight by playing back the entire flight profile in which the GNSS ' \
      'measurements are calculated using the real ephemeris, the real specific forces and the real angular rates ' \
      'from the flight data.\n'

sim_type = raw_input('Please select the simulation type by entering either "simulation" or "play back": ')
while sim_type != 'simulation' and sim_type != 'play back':
    sim_type = raw_input('\nPlease select the simulation type by entering either "simulation" or "play back": ')

# 7. Specify the simulation mode
print '\nEach simulation type has two modes:\n\n' \
      '     Mode #1 == "stand alone": performs either the "loosely coupled" integration ' \
      'or the "tightly coupled" integration.\n\n' \
      '     Mode #2 == "dual": performs both the "loosely coupled" integration and ' \
      'the "tightly coupled" integration simultaneously.\n'

sim_mode = raw_input('Please select the simulation mode by entering either "alone" or "dual": ')
while sim_mode != 'alone' and sim_mode != 'dual':
    sim_mode = raw_input('\nPlease select the simulation mode by entering either "alone" or "dual": ')

# 8. Specify the tightness
if sim_mode == 'alone':
    print '\nThere are two integration schemes in the "stand-alone" mode:\n\n' \
          '     Scheme #1 == "Loose": to loosely integrate the INS and the GNSS solutions.\n\n' \
          '     Scheme #2 == "Tight": to tightly integrate the INS and the GNSS solutions.\n'
    scheme = raw_input('Specify your choice by entering either "loose" or "tight": ')
    while scheme != 'loose' and scheme != 'tight':
        scheme = raw_input('\nSpecify your choice by entering either "loose" or "tight": ')
elif sim_mode == 'dual':
    scheme = 'loose'   # default integration scheme

# 9. Specify the random seed
seed = input('\nPlease choose your random seed int[0, inf): ')
while seed < 0:
    print '\nRandom seed must be an non-negative integer.'
    seed = input('Please choose your random seed int[0, inf): ')
rnd = np.random.RandomState(seed)

# 10. Run the simulation by calling the nav_sim_driver with appropriate options
nav_sim_driver(data_fname, nav_fname, num_dom, num_doy, imu_grade, gnss, freq, sim_type, sim_mode, scheme)

# **********************************************************************************************************************
# *******************************************  REACH THE PYTHON'S TAIL  ************************************************
# **********************************************************************************************************************