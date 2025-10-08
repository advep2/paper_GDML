#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:59:49 2020

@author: adrian
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MultipleLocator, AutoMinorLocator
from scipy import interpolate
from contour_2D import contour_2D
from streamlines_function import streamplot, streamline_2D
from HET_sims_read_df import HET_sims_read_df
from HET_sims_mean_df import HET_sims_mean_df
from HET_sims_plotvars import HET_sims_cp_vars_df
import pylab
import scipy.io as sio
import pickle

from erosion_Ftheta import erosion_Ftheta
from erosion_Y0 import erosion_Y0

# ---- Deactivate/activate all types of python warnings ----
import warnings
warnings.filterwarnings("ignore") # Deactivate all types of warnings
#    warnings.simplefilter('always')   # Activate all types of warnings
# -------------------------------------------------



# Close all existing figures
plt.close("all")


################################ INPUTS #######################################
# Print out time step
timestep = 'last'
#timestep = 13
#timestep = 400
if timestep == 'last':
    timestep = -1

# Printing results flag and last number of steps for averaging for printing results
print_flag = 0
last_steps = 5

save_flag = 1

    
# Plots save flag
#figs_format = ".eps"
figs_format = ".png"
#figs_format = ".pdf"

# Plots to produce
df_plots            = 1

#path_out = "CHEOPS_Final_figs/"
path_out = "VHT_LP_US/testcase2/comp_2c_2f_2h_2hKr/"


if save_flag == 1 and os.path.isdir(path_out) != 1:  
    sys.exit("ERROR: path_out is not an existing directory")


# Set options for LaTeX font
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
font_size           = 25
font_size_legend    = font_size - 10
ticks_size          = 25
ticks_size_isolines = ticks_size - 15
text_size           = 25
levels_step         = 100
cbar_nticks         = 10

line_width             = 1.5
line_width_boundary    = line_width + 0.5
line_width_grid        = 1
marker_every           = 1
marker_every_time      = 10
marker_every_time_fast = 3000
marker_every_FFT       = 2
marker_every_mesh      = 3
marker_size            = 7
marker_size_cath       = 5
xticks_pad             = 6.0

# Inputs for streamlines plots
flag_start        = 1
plot_start_points = 1
arrow_size        = 1.5
arrow_style       = '-|>'
streamline_width  = 0.8
streamline_color  = 'b'
min_length        = 0.065

# Extra colors
orange ='#FD6A02'            
gold   ='#F9A602'
brown  ='#8B4000'
silver ='#C0C0C0'
grey   ='#808080'


# Physical constants
# Van der Walls radius for Xe
r_Xe = 216e-12
e    = 1.6021766E-19
me   = 9.1093829E-31
g0   = 9.80665

def fmt_func_exponent_cbar(x,pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${}\cdot$10$^{{{}}}$'.format(a, b)
    
def fmt_func_exponent_lines(x):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${}\cdot$10$^{{{}}}$'.format(a, b)
    
    
def plot_MFAM(faces,nodes,line_width):
    nfaces = np.shape(faces)[1]
    for i in range(0,nfaces):
        if faces[2,i] == 2:     # face type >> sigma = const. (red)
            plt.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'r-',linewidth = line_width)
        elif faces[2,i] == 1:   # face type >> lambda = const. (blue)
            plt.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'b-',linewidth = line_width)
        else:                   # any other face type (black)  
            plt.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'k-',linewidth = line_width)
            
def plot_MFAM_ax(ax,faces,nodes,line_width):
    nfaces = np.shape(faces)[1]
    for i in range(0,nfaces):
        if faces[2,i] == 2:     # face type >> sigma = const. (red)
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'r-',linewidth = line_width)
        elif faces[2,i] == 1:   # face type >> lambda = const. (blue)
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'b-',linewidth = line_width)
        else:                   # any other face type (black)  
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'k-',linewidth = line_width)
            
    return
###############################################################################
    

if df_plots == 1:
    print("######## df_plots ########")
          
    # Print out time steps
    timestep = 'last'
    
    marker_size  = 5
    marker_every = 1
  
    allsteps_flag   = 1
    
    mean_vars       = 1
    mean_type       = 0
#    last_steps      = 700
    last_steps      = 24
    step_i          = 100
    step_f          = 500
    plot_mean_vars  = 1
    
    # For plotting profiles along the boundary walls
    plot_Dwall     = 1
    plot_Awall     = 0
    
    plot_curr      = 1
    plot_angle     = 1
    plot_imp_ene   = 1
    plot_q         = 0
    plot_erosion   = 1
    
    # Erosion inputs
    t_op      = 300          # Operation time in [h]
    # --- For Y0(E)
    c         = 3.75e-5      # [mm3/C-eV]  [*0.5,*1,*1.5]
    E_th      = 60           # [eV]  [40,60,80]
    # --- For F(theta)
    Fmax      = 3            # [-]   [2,3,4]
    theta_max = 60           # [deg] [50,60,70]
    a         = 8            # [-]
    # Eroded profile last point of chamber approach
    # 0 - Old approach: moving last picM node in chamfer according to the panel normal
    # 1 - Old approach: moving last picM node in chamfer according to a vertical normal
    # 2 - New approach: last point in eroded profile is the crossing point between the 
    #                   vertical plume wall and the straight line defined by the two 
    #                   previous nodes of the eroded profile
    erprof_app = 2
    
    # For plotting distribution functions at given points
    plot_angle_df  = 0
    plot_ene_df    = 0
    plot_normv_df  = 0

    normalized_df  = 0
    log_yaxis      = 1
    log_yaxis_tol  = 1E0
    
    # Enter the number of points and the points coorrdinates in cm at which plot the
    # distribution function. The closest surface elements will be chosen to plot
    # the distribution functions. Enter the boundary type to which each point
    # belongs according to:
    # Dwall_bot >> Bottom dielectric woll
    # Dwall_top >> Top dielectric wll
    # Awall     >> Anode wall
#    npoints_plot_df = 7
#    bpoints_plot_df = ["Dwall_bot","Dwall_top","Dwall_bot","Dwall_top","Dwall_bot","Dwall_top","Awall"]
#    zpoints_plot_df = np.array([1.0,1.0,2.0,2.0,2.81,2.81,0.0],dtype=float)
#    rpoints_plot_df = np.array([5.247,7.5,5.247,7.5,5.247,7.5,6.37],dtype=float)
    
    npoints_plot_df = 4
    bpoints_plot_df = ["Dwall_bot","Dwall_bot","Dwall_bot","Dwall_bot","Dwall_bot","Dwall_bot","Dwall_bot"]
    zpoints_plot_df = np.array([0.02283654,0.02465,0.02645,0.02825,0.03005,0.03185],dtype=float)*1E2
    rpoints_plot_df = np.array([0.052475,0.0518448,0.0505844,0.049324,0.0480636,0.0468032],dtype=float)*1E2

    

    if timestep == 'last':
        timestep = -1  
    if allsteps_flag == 0:
        mean_vars = 0
        
        
    # Simulation names
    nsims = 4
    oldpost_sim      = np.array([6,6,6,6,3,3,3,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([20,20,20,20,8,8,8,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)         
    
    
    sim_names = [
#                 "../../../Sr_sims_files/T1N2_pm1em1_cat313_tm615_te2_tq12_71d0dcb",
#                 "../../../Sr_sims_files/T1N1_pm1em1_cat313_tm515_te1_tq21_71d0dcb",
                 
#                 "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_71d0dcb",
#                 "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_71d0dcb_Es50",
    
#                 "../../../Ca_sims_files/T2N4_pm1em1_cat1200_tm15_te1_tq125_71d0dcb",
#                 "../../../Ca_sims_files/T2N4_pm1em1_cat1200_tm15_te1_tq125_71d0dcb_Es50",
#                 "../../../Ca_sims_files/T2N4_pm1em1_cat1200_tm15_te1_tq125_71d0dcb_olddf_Es50",
            
#                 "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd300",
#                 "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd250",
#                 "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd200",
#                 "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd150",
                 
                 "../../../Ca_hyphen/sim/sims/PPSX00_em2_OP2c_tmte08_2_tq1_fcat4656_CEX_VDF",
#                 "../../../Ca_hyphen/sim/sims/PPSX00_em2_OP2c_tmte08_2_tq1_fcat4656_CEX_Kr_VDF",
                 "../../../Ca_hyphen/sim/sims/PPSX00_em2_OP2f_tmte08_2_tq1_fcat4656_CEX_VDF",
                 "../../../Ca_hyphen/sim/sims/PPSX00_em2_OP2h_tmte06_2_tq1_fcat4656_CEX_VDF",
                 "../../../Ca_hyphen/sim/sims/PPSX00_em2_OP2h_tmte06_2_tq1_fcat4656_CEX_Kr_VDF",
                 
#                 "../../../Ca_hyphen/sim/sims/PPSX00_em2_OP2f_tmte08_2_tq1_fcat4656_CEX",
#                 "../../../Ca_hyphen/sim/sims/PPSX00_em2_OP2h_tmte06_2_tq1_fcat4656_CEX",
#                 "../../../Ca_hyphen/sim/sims/PPSX00_em2_OP2h_tmte06_2_tq1_fcat4656_CEX_Kr",
                 
                 ]

    
    
    topo_case = 0
    if topo_case == 1:
        PIC_mesh_file_name = ["PIC_mesh_topo1_refined4.hdf5",
                              "PIC_mesh_topo1_refined4.hdf5",
                              "PIC_mesh_topo1_refined4.hdf5",
                              "PIC_mesh_topo1_refined4.hdf5",
                              "PIC_mesh_topo1_refined4.hdf5",
                              "PIC_mesh_topo1_refined4.hdf5",
                              "PIC_mesh_topo1_refined4.hdf5",
                              "PIC_mesh_topo1_refined4.hdf5",
                             ]
    elif topo_case == 2:
        PIC_mesh_file_name = [
                              "PIC_mesh_topo2_refined4.hdf5",
                              "PIC_mesh_topo2_refined4.hdf5",
                              "PIC_mesh_topo2_refined4.hdf5",
                              "PIC_mesh_topo2_refined4.hdf5",
                              "PIC_mesh_topo2_refined4.hdf5",
                              "PIC_mesh_topo2_refined4.hdf5",
                              "PIC_mesh_topo2_refined4.hdf5"
                              ]
    elif topo_case == 0:    
        PIC_mesh_file_name = [
                              "PIC_mesh.hdf5", # VHT_US
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                
                
                              "SPT100_picM.hdf5",
                              "SPT100_picM.hdf5",
                              "SPT100_picM.hdf5",
                              "SPT100_picM.hdf5",
                              
#                              "SPT100_picM_Reference1500points.hdf5",
                              "SPT100_picM_Reference1500points_rm.hdf5",
                              "SPT100_picM_Reference1500points_rm.hdf5",
                              "SPT100_picM.hdf5",
                              "SPT100_picM.hdf5",
                              "SPT100_picM.hdf5",
                              ]

    # Labels                          
    labels = [
            
               r"OP1",
               r"OP2",
               r"OP4",
               r"OP5",
            
               r"",
               r"",
               r"",
               r"",
               r"",
               r"",
            
#              r"T1N1 REF",
#              r"T1N2 REF",

#              r"T2N3-REF",
              r"T2N4-REF",
            
              r"REF",
              r"V1",
              r"V2",
              r"V3",
              ]

    
    # Line colors
    colors = ['k','r','g','b','m','c','y',orange,brown]
#    colors = ['k','m',orange,brown]
    # Markers
    markers = ['s','o','v','^','<', '>','D','p','*']
#    markers = ['s','<','D','p']
    markers = ['','','','','','','']
    # Line style
    linestyles = ['-','--','-.', ':','-','--','-.']
    linestyles = ['-','-','-','-','-','-','-']
    
    
    xaxis_label   = r"$s/L_\mathrm{c}$"
    xaxis_label_z = r"$z/L_\mathrm{c}$"
    yaxis_label_r = r"$r/H_\mathrm{c}$"


    # Axial profile plots
    if plot_Dwall == 1:
        if plot_curr == 1:
            plt.figure(r'ji1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_\mathrm{n,i1}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_\mathrm{n,i2}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji3 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_\mathrm{n,i3}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji4 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_\mathrm{n,i4}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji_tot_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_\mathrm{ni}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'egn1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$eg_\mathrm{n,n1}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'egn2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$eg_\mathrm{n,n2}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'egn3 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$eg_\mathrm{n,n3}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'egn Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$eg_\mathrm{nn}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_angle == 1:
            plt.figure(r'angle_i1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\theta_\mathrm{i1,W}$ (deg)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'angle_i2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\theta_\mathrm{i2,W}$ (deg)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'angle_i3 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\theta_\mathrm{i3,W}$ (deg)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'angle_i4 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\theta_\mathrm{i4,W}$ (deg)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'angle_ion Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\theta_\mathrm{iW}$ (deg)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'angle_n1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\theta_\mathrm{n1,W}$ (deg)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'angle_n2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\theta_\mathrm{n2,W}$ (deg)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'angle_n3 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\theta_\mathrm{n3,W}$ (deg)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'angle_neu Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\theta_\mathrm{nW}$ (deg)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_imp_ene == 1:
            plt.figure(r'imp_ene_i1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{i1,W}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_i2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{i2,W}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_i3 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{i3,W}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_i4 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{i4,W}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_ion Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{iW}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_n1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{n1,W}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_n2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{n2,W}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_n3 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{n3,W}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_neu Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{nW}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_erosion == 1:
            plt.figure(r'dhdt_i1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$dh/dt|_\mathrm{i1}$ ($\mu$m/h)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dhdt_i2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$dh/dt|_\mathrm{i2}$ ($\mu$m/h)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dhdt_i3 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$dh/dt|_\mathrm{i3}$ ($\mu$m/h)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dhdt_i4 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$dh/dt|_\mathrm{i4}$ ($\mu$m/h)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dhdt_n1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$dh/dt|_\mathrm{n1}$ ($\mu$m/h)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dhdt_n2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$dh/dt|_\mathrm{n2}$ ($\mu$m/h)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dhdt_n3 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$dh/dt|_\mathrm{n3}$ ($\mu$m/h)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dhdt Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $dh/dt$ (mm/s)",fontsize = font_size)
            plt.title(r"$dh/dt$ ($\mu$m/h)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'h(z) Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $dh/dt$ (mm/s)",fontsize = font_size)
            plt.title(r"$h$ ($\mu$m)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'eroded chamber Dwall_bot_top')
            plt.xlabel(xaxis_label_z,fontsize = font_size)
            plt.ylabel(yaxis_label_r,fontsize = font_size)
#            plt.title(r"(c) $dh/dt$ (mm/s)",fontsize = font_size)
#            plt.title(r"$h$ ($\mu$m)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'points eroded chamber Dwall_bot_top')
            plt.xlabel(xaxis_label_z,fontsize = font_size-5)
            plt.ylabel(yaxis_label_r,fontsize = font_size-5)
#            plt.title(r"(c) $dh/dt$ (mm/s)",fontsize = font_size)
#            plt.title(r"$h$ ($\mu$m)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size-5) 
            plt.yticks(fontsize = ticks_size-5)
            
    if plot_Awall == 1:
        if plot_curr == 1:
            plt.figure(r'ji_tot_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $j_{in}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_imp_ene == 1:
            plt.figure(r'imp_ene_i1 Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $E_{imp,i1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_i2 Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $E_{imp,i2}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'imp_ene_ion Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $E_{imp,i}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_n1 Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $E_{imp,n1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
    
        
        
    ind  = 0
    ind2 = 0
    ind3 = 0
    for k in range(0,nsims):
        ind_ini_letter = sim_names[k].rfind('/') + 1
        print("##### CASE "+str(k+1)+": "+sim_names[k][ind_ini_letter::]+" #####")
        ######################## READ INPUT/OUTPUT FILES ##########################
        # Obtain paths to simulation files
        path_picM         = sim_names[k]+"/SET/inp/"+PIC_mesh_file_name[k]
        path_simstate_inp = sim_names[k]+"/CORE/inp/SimState.hdf5"
        path_simstate_out = sim_names[k]+"/CORE/out/SimState.hdf5"
        path_postdata_out = sim_names[k]+"/CORE/out/PostData.hdf5"
        path_simparams_inp = sim_names[k]+"/CORE/inp/sim_params.inp"
        print("Reading results...")
        [num_ion_spe,num_neu_spe,points,zs,rs,zscells,rscells,dims,nodes_flag,
           cells_flag,cells_vol,volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,
           eta_max,eta_min,xi_top,xi_bottom,time,steps,dt,nsteps,sc_bot,sc_top,
           Lplume_bot,Lplume_top,Lchamb_bot,Lchamb_top,Lanode,Lfreeloss_ver,
           Lfreeloss_lat,Lfreeloss,Laxis,
           
           nnodes_Dwall_bot,nnodes_Dwall_top,nnodes_Awall,nnodes_FLwall_ver,
           nnodes_FLwall_lat,nnodes_Axis,nnodes_bound,inodes_Dwall_bot,
           jnodes_Dwall_bot,inodes_Dwall_top,jnodes_Dwall_top,inodes_Awall,
           jnodes_Awall,inodes_FLwall_ver,jnodes_FLwall_ver,inodes_FLwall_lat,
           jnodes_FLwall_lat,inodes_Axis,jnodes_Axis,sDwall_bot_nodes,
           sDwall_top_nodes,sAwall_nodes,sFLwall_ver_nodes,sFLwall_lat_nodes,
           sAxis_nodes,
           
           nsurf_Dwall_bot,nsurf_Dwall_top,nsurf_Awall,nsurf_FLwall_ver,
           nsurf_FLwall_lat,nsurf_bound,indsurf_Dwall_bot,zsurf_Dwall_bot,
           rsurf_Dwall_bot,indsurf_Dwall_top,zsurf_Dwall_top,rsurf_Dwall_top,
           indsurf_Awall,zsurf_Awall,rsurf_Awall,indsurf_FLwall_ver,
           zsurf_FLwall_ver,rsurf_FLwall_ver,indsurf_FLwall_lat,
           zsurf_FLwall_lat,rsurf_FLwall_lat,sDwall_bot_surf,sDwall_top_surf,
           sAwall_surf,sFLwall_ver_surf,sFLwall_lat_surf,imp_elems,norm_vers,
           
           nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,
           MkQ1_surf,ji1_surf,ji2_surf,ji3_surf,ji4_surf,ji_surf,
           gn1_tw_surf,gn1_fw_surf,gn2_tw_surf,gn2_fw_surf,gn3_tw_surf,gn3_fw_surf,
           gn_tw_surf,qi1_tot_wall_surf,qi2_tot_wall_surf,qi3_tot_wall_surf,
           qi4_tot_wall_surf,qi_tot_wall_surf,qn1_tw_surf,qn1_fw_surf,qn2_tw_surf,
           qn2_fw_surf,qn3_tw_surf,qn3_fw_surf,qn_tot_wall_surf,imp_ene_i1_surf,
           imp_ene_i2_surf,imp_ene_i3_surf,imp_ene_i4_surf,imp_ene_n1_surf,
           imp_ene_n2_surf,imp_ene_n3_surf,
           
           angle_bins_i1,ene_bins_i1,normv_bins_i1,angle_bins_i2,ene_bins_i2,
           normv_bins_i2,angle_bins_i3,ene_bins_i3,normv_bins_i3,angle_bins_i4,
           ene_bins_i4,normv_bins_i4,angle_bins_n1,ene_bins_n1,normv_bins_n1,
           angle_bins_n2,ene_bins_n2,normv_bins_n2,angle_bins_n3,ene_bins_n3,
           normv_bins_n3,nbins_angle,nbins_ene,nbins_normv,
           
           angle_df_i1,ene_df_i1,normv_df_i1,ene_angle_df_i1,
           angle_df_i2,ene_df_i2,normv_df_i2,ene_angle_df_i2,
           angle_df_i3,ene_df_i3,normv_df_i3,ene_angle_df_i3,
           angle_df_i4,ene_df_i4,normv_df_i4,ene_angle_df_i4,
           angle_df_n1,ene_df_n1,normv_df_n1,ene_angle_df_n1,
           angle_df_n2,ene_df_n2,normv_df_n2,ene_angle_df_n2,
           angle_df_n3,ene_df_n3,normv_df_n3,ene_angle_df_n3] = HET_sims_read_df(path_simstate_inp,path_simstate_out,path_postdata_out,path_simparams_inp,
                                                                                 path_picM,allsteps_flag,timestep,oldpost_sim[k],oldsimparams_sim[k])
            
        
        
        if mean_vars == 1:        
            print("Averaging variables...")                                                                              
            [nQ1_inst_surf_mean,nQ1_surf_mean,nQ2_inst_surf_mean,
               nQ2_surf_mean,dphi_kbc_surf_mean,MkQ1_surf_mean,ji1_surf_mean,
               ji2_surf_mean,ji3_surf_mean,ji4_surf_mean,ji_surf_mean,
               gn1_tw_surf_mean,gn1_fw_surf_mean,gn2_tw_surf_mean,gn2_fw_surf_mean,
               gn3_tw_surf_mean,gn3_fw_surf_mean,gn_tw_surf_mean,
               qi1_tot_wall_surf_mean,qi2_tot_wall_surf_mean,
               qi3_tot_wall_surf_mean,qi4_tot_wall_surf_mean,qi_tot_wall_surf_mean,
               qn1_tw_surf_mean,qn1_fw_surf_mean,qn2_tw_surf_mean,qn2_fw_surf_mean,
               qn3_tw_surf_mean,qn3_fw_surf_mean,qn_tot_wall_surf_mean,
               imp_ene_i1_surf_mean,imp_ene_i2_surf_mean,imp_ene_i3_surf_mean,
               imp_ene_i4_surf_mean,imp_ene_n1_surf_mean,imp_ene_n2_surf_mean,
               imp_ene_n3_surf_mean,
               
               angle_df_i1_mean,ene_df_i1_mean,normv_df_i1_mean,ene_angle_df_i1_mean,
               angle_df_i2_mean,ene_df_i2_mean,normv_df_i2_mean,ene_angle_df_i2_mean,
               angle_df_i3_mean,ene_df_i3_mean,normv_df_i3_mean,ene_angle_df_i3_mean,
               angle_df_i4_mean,ene_df_i4_mean,normv_df_i4_mean,ene_angle_df_i4_mean,
               angle_df_n1_mean,ene_df_n1_mean,normv_df_n1_mean,ene_angle_df_n1_mean,
               angle_df_n2_mean,ene_df_n2_mean,normv_df_n2_mean,ene_angle_df_n2_mean,
               angle_df_n3_mean,ene_df_n3_mean,normv_df_n3_mean,ene_angle_df_n3_mean] = HET_sims_mean_df(nsteps,mean_type,last_steps,step_i,step_f,
                                                                                                         nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,
                                                                                                         MkQ1_surf,ji1_surf,ji2_surf,ji3_surf,ji4_surf,ji_surf,
                                                                                                         gn1_tw_surf,gn1_fw_surf,gn2_tw_surf,gn2_fw_surf,gn3_tw_surf,
                                                                                                         gn3_fw_surf,gn_tw_surf,
                                                                                                         qi1_tot_wall_surf,qi2_tot_wall_surf,qi3_tot_wall_surf,
                                                                                                         qi4_tot_wall_surf,qi_tot_wall_surf,qn1_tw_surf,qn1_fw_surf,
                                                                                                         qn2_tw_surf,qn2_fw_surf,qn3_tw_surf,qn3_fw_surf,
                                                                                                         qn_tot_wall_surf,imp_ene_i1_surf,imp_ene_i2_surf,
                                                                                                         imp_ene_i3_surf,imp_ene_i4_surf,imp_ene_n1_surf,
                                                                                                         imp_ene_n2_surf,imp_ene_n3_surf,
                                                                                                         
                                                                                                         angle_df_i1,ene_df_i1,normv_df_i1,ene_angle_df_i1,
                                                                                                         angle_df_i2,ene_df_i2,normv_df_i2,ene_angle_df_i2,
                                                                                                         angle_df_i3,ene_df_i3,normv_df_i3,ene_angle_df_i3,
                                                                                                         angle_df_i4,ene_df_i4,normv_df_i4,ene_angle_df_i4,
                                                                                                         angle_df_n1,ene_df_n1,normv_df_n1,ene_angle_df_n1,
                                                                                                         angle_df_n2,ene_df_n2,normv_df_n2,ene_angle_df_n2,
                                                                                                         angle_df_n3,ene_df_n3,normv_df_n3,ene_angle_df_n3)
                                                                                            
                                                                                            
        print("Obtaining final variables for plotting...") 
        if mean_vars == 1 and plot_mean_vars == 1:
            print("Plotting variables are time-averaged")
            [nQ1_inst_surf_plot,nQ1_surf_plot,nQ2_inst_surf_plot,
               nQ2_surf_plot,dphi_kbc_surf_plot,MkQ1_surf_plot,ji1_surf_plot,
               ji2_surf_plot,ji3_surf_plot,ji4_surf_plot,ji_surf_plot,
               gn1_tw_surf_plot,gn1_fw_surf_plot,gn2_tw_surf_plot,gn2_fw_surf_plot,
               gn3_tw_surf_plot,gn3_fw_surf_plot,gn_tw_surf_plot,
               
               qi1_tot_wall_surf_plot,qi2_tot_wall_surf_plot,qi3_tot_wall_surf_plot,
               qi4_tot_wall_surf_plot,qi_tot_wall_surf_plot,
               qn1_tw_surf_plot,qn1_fw_surf_plot,qn2_tw_surf_plot,qn2_fw_surf_plot,
               qn3_tw_surf_plot,qn3_fw_surf_plot,qn_tot_wall_surf_plot,
               
               imp_ene_i1_surf_plot,imp_ene_i2_surf_plot,imp_ene_i3_surf_plot,
               imp_ene_i4_surf_plot,imp_ene_n1_surf_plot,imp_ene_n2_surf_plot,
               imp_ene_n3_surf_plot,
               
               angle_df_i1_plot,ene_df_i1_plot,normv_df_i1_plot,ene_angle_df_i1_plot,
               angle_df_i2_plot,ene_df_i2_plot,normv_df_i2_plot,ene_angle_df_i2_plot,
               angle_df_i3_plot,ene_df_i3_plot,normv_df_i3_plot,ene_angle_df_i3_plot,
               angle_df_i4_plot,ene_df_i4_plot,normv_df_i4_plot,ene_angle_df_i4_plot,
               angle_df_n1_plot,ene_df_n1_plot,normv_df_n1_plot,ene_angle_df_n1_plot,
               angle_df_n2_plot,ene_df_n2_plot,normv_df_n2_plot,ene_angle_df_n2_plot,
               angle_df_n3_plot,ene_df_n3_plot,normv_df_n3_plot,ene_angle_df_n3_plot] = HET_sims_cp_vars_df(nQ1_inst_surf_mean,nQ1_surf_mean,nQ2_inst_surf_mean,
                                                                                                           nQ2_surf_mean,dphi_kbc_surf_mean,MkQ1_surf_mean,ji1_surf_mean,
                                                                                                           ji2_surf_mean,ji3_surf_mean,ji4_surf_mean,ji_surf_mean,
                                                                                                           gn1_tw_surf_mean,gn1_fw_surf_mean,gn2_tw_surf_mean,gn2_fw_surf_mean,
                                                                                                           gn3_tw_surf_mean,gn3_fw_surf_mean,gn_tw_surf_mean,
                                                                                                           qi1_tot_wall_surf_mean,qi2_tot_wall_surf_mean,
                                                                                                           qi3_tot_wall_surf_mean,qi4_tot_wall_surf_mean,qi_tot_wall_surf_mean,
                                                                                                           qn1_tw_surf_mean,qn1_fw_surf_mean,qn2_tw_surf_mean,qn2_fw_surf_mean,
                                                                                                           qn3_tw_surf_mean,qn3_fw_surf_mean,qn_tot_wall_surf_mean,
                                                                                                           imp_ene_i1_surf_mean,imp_ene_i2_surf_mean,imp_ene_i3_surf_mean,
                                                                                                           imp_ene_i4_surf_mean,imp_ene_n1_surf_mean,imp_ene_n2_surf_mean,
                                                                                                           imp_ene_n3_surf_mean,
                                                                                                           
                                                                                                           angle_df_i1_mean,ene_df_i1_mean,normv_df_i1_mean,ene_angle_df_i1_mean,
                                                                                                           angle_df_i2_mean,ene_df_i2_mean,normv_df_i2_mean,ene_angle_df_i2_mean,
                                                                                                           angle_df_i3_mean,ene_df_i3_mean,normv_df_i3_mean,ene_angle_df_i3_mean,
                                                                                                           angle_df_i4_mean,ene_df_i4_mean,normv_df_i4_mean,ene_angle_df_i4_mean,
                                                                                                           angle_df_n1_mean,ene_df_n1_mean,normv_df_n1_mean,ene_angle_df_n1_mean,
                                                                                                           angle_df_n2_mean,ene_df_n2_mean,normv_df_n2_mean,ene_angle_df_n2_mean,
                                                                                                           angle_df_n3_mean,ene_df_n3_mean,normv_df_n3_mean,ene_angle_df_n3_mean)
            
        else:
            [nQ1_inst_surf_plot,nQ1_surf_plot,nQ2_inst_surf_plot,
               nQ2_surf_plot,dphi_kbc_surf_plot,MkQ1_surf_plot,ji1_surf_plot,
               ji2_surf_plot,ji3_surf_plot,ji4_surf_plot,ji_surf_plot,
               gn1_tw_surf_plot,gn1_fw_surf_plot,gn2_tw_surf_plot,gn2_fw_surf_plot,
               gn3_tw_surf_plot,gn3_fw_surf_plot,gn_tw_surf_plot,
               
               qi1_tot_wall_surf_plot,qi2_tot_wall_surf_plot,qi3_tot_wall_surf_plot,
               qi4_tot_wall_surf_plot,qi_tot_wall_surf_plot,
               qn1_tw_surf_plot,qn1_fw_surf_plot,qn2_tw_surf_plot,qn2_fw_surf_plot,
               qn3_tw_surf_plot,qn3_fw_surf_plot,qn_tot_wall_surf_plot,
               
               imp_ene_i1_surf_plot,imp_ene_i2_surf_plot,imp_ene_i3_surf_plot,
               imp_ene_i4_surf_plot,imp_ene_n1_surf_plot,imp_ene_n2_surf_plot,
               imp_ene_n3_surf_plot,
               
               angle_df_i1_plot,ene_df_i1_plot,normv_df_i1_plot,ene_angle_df_i1_plot,
               angle_df_i2_plot,ene_df_i2_plot,normv_df_i2_plot,ene_angle_df_i2_plot,
               angle_df_i3_plot,ene_df_i3_plot,normv_df_i3_plot,ene_angle_df_i3_plot,
               angle_df_i4_plot,ene_df_i4_plot,normv_df_i4_plot,ene_angle_df_i4_plot,
               angle_df_n1_plot,ene_df_n1_plot,normv_df_n1_plot,ene_angle_df_n1_plot,
               angle_df_n2_plot,ene_df_n2_plot,normv_df_n2_plot,ene_angle_df_n2_plot,
               angle_df_n3_plot,ene_df_n3_plot,normv_df_n3_plot,ene_angle_df_n3_plot] = HET_sims_cp_vars_df(nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,
                                                                                                             MkQ1_surf,ji1_surf,ji2_surf,ji3_surf,ji4_surf,ji_surf,
                                                                                                             gn1_tw_surf,gn1_fw_surf,gn2_tw_surf,gn2_fw_surf,gn3_tw_surf,
                                                                                                             gn3_fw_surf,gn_tw_surf,
                                                                                                             qi1_tot_wall_surf,qi2_tot_wall_surf,qi3_tot_wall_surf,
                                                                                                             qi4_tot_wall_surf,qi_tot_wall_surf,qn1_tw_surf,qn1_fw_surf,
                                                                                                             qn2_tw_surf,qn2_fw_surf,qn3_tw_surf,qn3_fw_surf,
                                                                                                             qn_tot_wall_surf,imp_ene_i1_surf,imp_ene_i2_surf,
                                                                                                             imp_ene_i3_surf,imp_ene_i4_surf,imp_ene_n1_surf,
                                                                                                             imp_ene_n2_surf,imp_ene_n3_surf,
                                                                                                             
                                                                                                             angle_df_i1,ene_df_i1,normv_df_i1,ene_angle_df_i1,
                                                                                                             angle_df_i2,ene_df_i2,normv_df_i2,ene_angle_df_i2,
                                                                                                             angle_df_i3,ene_df_i3,normv_df_i3,ene_angle_df_i3,
                                                                                                             angle_df_i4,ene_df_i4,normv_df_i4,ene_angle_df_i4,
                                                                                                             angle_df_n1,ene_df_n1,normv_df_n1,ene_angle_df_n1,
                                                                                                             angle_df_n2,ene_df_n2,normv_df_n2,ene_angle_df_n2,
                                                                                                             angle_df_n3,ene_df_n3,normv_df_n3,ene_angle_df_n3)
                                                                                                            
        
        # Obtain the surface elements in Dwall that are inside the chamber
        indsurf_inC_Dwall_bot = np.where(sDwall_bot_surf <= sc_bot)[0][:]
        indsurf_inC_Dwall_top = np.where(sDwall_top_surf <= sc_top)[0][:]
        nsurf_inC_Dwall_bot   = len(indsurf_inC_Dwall_bot)
        nsurf_inC_Dwall_top   = len(indsurf_inC_Dwall_top)
            
        # Compute currents from the angle distribution function, mean impact angles and energies at each surface element 
        df_ji1_surf_plot        = np.zeros(nsurf_bound,dtype=float)
        df_ji2_surf_plot        = np.zeros(nsurf_bound,dtype=float)
        df_ji3_surf_plot        = np.zeros(nsurf_bound,dtype=float)
        df_ji4_surf_plot        = np.zeros(nsurf_bound,dtype=float)
        df_gn1_tw_surf_plot     = np.zeros(nsurf_bound,dtype=float)
        df_gn2_tw_surf_plot     = np.zeros(nsurf_bound,dtype=float)
        df_gn3_tw_surf_plot     = np.zeros(nsurf_bound,dtype=float)
        df_angle_i1_surf_plot   = np.zeros(nsurf_bound,dtype=float)
        df_angle_i2_surf_plot   = np.zeros(nsurf_bound,dtype=float)
        df_angle_i3_surf_plot   = np.zeros(nsurf_bound,dtype=float)
        df_angle_i4_surf_plot   = np.zeros(nsurf_bound,dtype=float)
        df_angle_n1_surf_plot   = np.zeros(nsurf_bound,dtype=float)
        df_angle_n2_surf_plot   = np.zeros(nsurf_bound,dtype=float)
        df_angle_n3_surf_plot   = np.zeros(nsurf_bound,dtype=float)
        df_imp_ene_i1_surf_plot = np.zeros(nsurf_bound,dtype=float)
        df_imp_ene_i2_surf_plot = np.zeros(nsurf_bound,dtype=float)
        df_imp_ene_i3_surf_plot = np.zeros(nsurf_bound,dtype=float)
        df_imp_ene_i4_surf_plot = np.zeros(nsurf_bound,dtype=float)
        df_imp_ene_n1_surf_plot = np.zeros(nsurf_bound,dtype=float)
        df_imp_ene_n2_surf_plot = np.zeros(nsurf_bound,dtype=float)
        df_imp_ene_n3_surf_plot = np.zeros(nsurf_bound,dtype=float)
        for ind_surf in range(0,nsurf_bound):
            df_ji1_surf_plot[ind_surf]        = np.trapz(angle_df_i1_plot[ind_surf,:])*e
            df_ji2_surf_plot[ind_surf]        = np.trapz(angle_df_i2_plot[ind_surf,:])*2*e
            df_gn1_tw_surf_plot[ind_surf]     = np.trapz(angle_df_n1_plot[ind_surf,:])
            df_angle_i1_surf_plot[ind_surf]   = np.trapz(angle_bins_i1*angle_df_i1_plot[ind_surf,:])/np.trapz(angle_df_i1_plot[ind_surf,:])
            df_angle_i2_surf_plot[ind_surf]   = np.trapz(angle_bins_i2*angle_df_i2_plot[ind_surf,:])/np.trapz(angle_df_i2_plot[ind_surf,:])
            df_angle_n1_surf_plot[ind_surf]   = np.trapz(angle_bins_n1*angle_df_n1_plot[ind_surf,:])/np.trapz(angle_df_n1_plot[ind_surf,:])
            df_imp_ene_i1_surf_plot[ind_surf] = np.trapz(ene_bins_i1*ene_df_i1_plot[ind_surf,:])/np.trapz(ene_df_i1_plot[ind_surf,:])
            df_imp_ene_i2_surf_plot[ind_surf] = np.trapz(ene_bins_i2*ene_df_i2_plot[ind_surf,:])/np.trapz(ene_df_i2_plot[ind_surf,:])
            df_imp_ene_n1_surf_plot[ind_surf] = np.trapz(ene_bins_n1*ene_df_n1_plot[ind_surf,:])/np.trapz(ene_df_n1_plot[ind_surf,:])
        if num_ion_spe == 4 and num_neu_spe == 3:
            for ind_surf in range(0,nsurf_bound):
                df_ji3_surf_plot[ind_surf]        = np.trapz(angle_df_i3_plot[ind_surf,:])*e
                df_ji4_surf_plot[ind_surf]        = np.trapz(angle_df_i4_plot[ind_surf,:])*2*e
                df_gn2_tw_surf_plot[ind_surf]     = np.trapz(angle_df_n2_plot[ind_surf,:])
                df_gn3_tw_surf_plot[ind_surf]     = np.trapz(angle_df_n3_plot[ind_surf,:])
                df_angle_i3_surf_plot[ind_surf]   = np.trapz(angle_bins_i3*angle_df_i3_plot[ind_surf,:])/np.trapz(angle_df_i3_plot[ind_surf,:])
                df_angle_i4_surf_plot[ind_surf]   = np.trapz(angle_bins_i4*angle_df_i4_plot[ind_surf,:])/np.trapz(angle_df_i4_plot[ind_surf,:])
                df_angle_n2_surf_plot[ind_surf]   = np.trapz(angle_bins_n2*angle_df_n2_plot[ind_surf,:])/np.trapz(angle_df_n2_plot[ind_surf,:])
                df_angle_n3_surf_plot[ind_surf]   = np.trapz(angle_bins_n3*angle_df_n3_plot[ind_surf,:])/np.trapz(angle_df_n3_plot[ind_surf,:])
                df_imp_ene_i3_surf_plot[ind_surf] = np.trapz(ene_bins_i3*ene_df_i3_plot[ind_surf,:])/np.trapz(ene_df_i3_plot[ind_surf,:])
                df_imp_ene_i4_surf_plot[ind_surf] = np.trapz(ene_bins_i4*ene_df_i4_plot[ind_surf,:])/np.trapz(ene_df_i4_plot[ind_surf,:])
                df_imp_ene_n2_surf_plot[ind_surf] = np.trapz(ene_bins_n2*ene_df_n2_plot[ind_surf,:])/np.trapz(ene_df_n2_plot[ind_surf,:])
                df_imp_ene_n3_surf_plot[ind_surf] = np.trapz(ene_bins_n3*ene_df_n3_plot[ind_surf,:])/np.trapz(ene_df_n3_plot[ind_surf,:])
            
        # Obtain magnitudes for the whole ion and neu population from the magnitudes of different ion and neu lists
        df_ji_surf_plot          = df_ji1_surf_plot + df_ji2_surf_plot + df_ji3_surf_plot + df_ji4_surf_plot
        den                      = df_ji1_surf_plot/e + df_ji2_surf_plot/(2*e) + df_ji3_surf_plot/e + df_ji4_surf_plot/(2*e)
        df_angle_ion_surf_plot   = df_angle_i1_surf_plot*(df_ji1_surf_plot/e/den) + df_angle_i2_surf_plot*(df_ji2_surf_plot/(2*e)/den) + df_angle_i3_surf_plot*(df_ji3_surf_plot/e/den) + df_angle_i4_surf_plot*(df_ji4_surf_plot/(2*e)/den)
        df_imp_ene_ion_surf_plot = df_imp_ene_i1_surf_plot*(df_ji1_surf_plot/e/den) + df_imp_ene_i2_surf_plot*(df_ji2_surf_plot/(2*e)/den) + df_imp_ene_i3_surf_plot*(df_ji3_surf_plot/e/den) + df_imp_ene_i4_surf_plot*(df_ji4_surf_plot/(2*e)/den)
        df_gn_tw_surf_plot       = df_gn1_tw_surf_plot + df_gn2_tw_surf_plot + df_gn3_tw_surf_plot
        df_angle_neu_surf_plot   = df_angle_n1_surf_plot*(df_gn1_tw_surf_plot/df_gn_tw_surf_plot) + df_angle_n2_surf_plot*(df_gn2_tw_surf_plot/df_gn_tw_surf_plot) + df_angle_n3_surf_plot*(df_gn3_tw_surf_plot/df_gn_tw_surf_plot)
        df_imp_ene_n_surf_plot   = df_imp_ene_n1_surf_plot*(df_gn1_tw_surf_plot/df_gn_tw_surf_plot) + df_imp_ene_n2_surf_plot*(df_gn2_tw_surf_plot/df_gn_tw_surf_plot) + df_imp_ene_n3_surf_plot*(df_gn3_tw_surf_plot/df_gn_tw_surf_plot)

#        #####################################
#        Iene_i1 = np.zeros((nsurf_bound,nsteps),dtype=float)
#        df_imp_ene_i1_surf_plot_2 = np.zeros((nsurf_bound,nsteps),dtype=float)
#        for kstep in range(0,nsteps):
#            for ind_elem in range(0,nsurf_bound):
#                for kind in range(0,nbins_ene-1):
#                    dene = ene_bins_i1[kind+1]-ene_bins_i1[kind]
#                    Iene_i1[ind_elem,kstep] = Iene_i1[ind_elem,kstep] + 0.5*(ene_df_i1[ind_elem,kind,kstep] + ene_df_i1[ind_elem,kind+1,kstep])*dene
#                    df_imp_ene_i1_surf_plot_2[ind_elem,kstep] = df_imp_ene_i1_surf_plot_2[ind_elem,kstep] + 0.5*(ene_bins_i1[kind]*ene_df_i1[ind_elem,kind,kstep] + ene_bins_i1[kind+1]*ene_df_i1[ind_elem,kind+1,kstep])*dene
#                df_imp_ene_i1_surf_plot_2[ind_elem,kstep] = df_imp_ene_i1_surf_plot_2[ind_elem,kstep]/Iene_i1[ind_elem,kstep]
#        df_imp_ene_i1_surf_plot_2_avg = np.nanmean(df_imp_ene_i1_surf_plot_2[:,nsteps-last_steps::],axis=1)
#        ####################################
        
        # Obtain the erosion rate [mm/s] o [mum/h]. Yields Y [mm3/C], fluxes are [1/m2-s]    
        # Case 1D distribution functions
        Y0_E_i1   = erosion_Y0(c,E_th,imp_ene_i1_surf_plot/e,nsurf_bound)
        Y0_E_i2   = erosion_Y0(c,E_th,imp_ene_i2_surf_plot/e,nsurf_bound)
        Y0_E_i3   = erosion_Y0(c,E_th,imp_ene_i3_surf_plot/e,nsurf_bound)
        Y0_E_i4   = erosion_Y0(c,E_th,imp_ene_i4_surf_plot/e,nsurf_bound)
        Y0_E_n1   = erosion_Y0(c,E_th,imp_ene_n1_surf_plot/e,nsurf_bound)
        Y0_E_n2   = erosion_Y0(c,E_th,imp_ene_n2_surf_plot/e,nsurf_bound)
        Y0_E_n3   = erosion_Y0(c,E_th,imp_ene_n3_surf_plot/e,nsurf_bound)
        Ftheta_i1 = erosion_Ftheta(Fmax,theta_max,a,df_angle_i1_surf_plot,nsurf_bound)
        Ftheta_i2 = erosion_Ftheta(Fmax,theta_max,a,df_angle_i2_surf_plot,nsurf_bound)
        Ftheta_i3 = erosion_Ftheta(Fmax,theta_max,a,df_angle_i3_surf_plot,nsurf_bound)
        Ftheta_i4 = erosion_Ftheta(Fmax,theta_max,a,df_angle_i4_surf_plot,nsurf_bound)
        Ftheta_n1 = erosion_Ftheta(Fmax,theta_max,a,df_angle_n1_surf_plot,nsurf_bound)
        Ftheta_n2 = erosion_Ftheta(Fmax,theta_max,a,df_angle_n2_surf_plot,nsurf_bound)
        Ftheta_n3 = erosion_Ftheta(Fmax,theta_max,a,df_angle_n3_surf_plot,nsurf_bound)
        Y_i1      = Y0_E_i1*Ftheta_i1
        Y_i2      = Y0_E_i2*Ftheta_i2
        Y_i3      = Y0_E_i3*Ftheta_i3
        Y_i4      = Y0_E_i4*Ftheta_i4        
        Y_n1      = Y0_E_n1*Ftheta_n1
        Y_n2      = Y0_E_n2*Ftheta_n2
        Y_n3      = Y0_E_n3*Ftheta_n3
        dhdt_i1   = (ji1_surf_plot/e)*e*Y_i1*1E-6                                       # [mm/s]
        dhdt_i2   = (ji2_surf_plot/(2*e))*e*Y_i2*1E-6                                   # [mm/s]
        dhdt_i3   = (ji3_surf_plot/(e))*e*Y_i3*1E-6                                     # [mm/s]
        dhdt_i4   = (ji4_surf_plot/(2*e))*e*Y_i4*1E-6                                   # [mm/s]
        dhdt_n1   = (gn1_tw_surf_plot)*e*Y_n1*1E-6                                      # [mm/s]
        dhdt_n2   = (gn2_tw_surf_plot)*e*Y_n2*1E-6                                      # [mm/s]
        dhdt_n3   = (gn3_tw_surf_plot)*e*Y_n3*1E-6                                      # [mm/s]
#        dhdt      = (ji1_surf_plot/e)*e*Y_i1*1E-6 + (ji2_surf_plot/(2*e))*e*Y_i2*1E-6  # [mm/s]
        dhdt      = dhdt_i1 + dhdt_i2 + dhdt_i3 + dhdt_i4 + dhdt_n1 + dhdt_n2 + dhdt_n3 # [mm/s]
        dhdt_mumh = dhdt*1E3*3600                                                       # [mum/h]
        dhdt_i1_mumh = dhdt_i1*1E3*3600                                                 # [mum/h]
        dhdt_i2_mumh = dhdt_i2*1E3*3600                                                 # [mum/h]
        dhdt_i3_mumh = dhdt_i3*1E3*3600                                                 # [mum/h]
        dhdt_i4_mumh = dhdt_i4*1E3*3600                                                 # [mum/h]
        dhdt_n1_mumh = dhdt_n1*1E3*3600                                                 # [mum/h]
        dhdt_n2_mumh = dhdt_n2*1E3*3600                                                 # [mum/h]
        dhdt_n3_mumh = dhdt_n3*1E3*3600                                                 # [mum/h]
        h_mum     = dhdt_mumh*t_op # h(z) [mum] erosion profile after a given operation time in hours
        h_m       = h_mum*1E-6     # h(z) [m] for performing operations
        
        
        # Case 2D distribution functions
        Y0_E_i1_2D   = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Y0_E_i2_2D   = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Y0_E_i3_2D   = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Y0_E_i4_2D   = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Y0_E_n1_2D   = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Y0_E_n2_2D   = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Y0_E_n3_2D   = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Ftheta_i1_2D = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Ftheta_i2_2D = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Ftheta_i3_2D = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Ftheta_i4_2D = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Ftheta_n1_2D = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Ftheta_n2_2D = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        Ftheta_n3_2D = np.zeros((nsurf_bound,nbins_ene,nbins_angle),dtype=float)
        
        dhdt_i1_2D = np.zeros(nsurf_bound,dtype=float)
        dhdt_i2_2D = np.zeros(nsurf_bound,dtype=float)
        dhdt_i3_2D = np.zeros(nsurf_bound,dtype=float)
        dhdt_i4_2D = np.zeros(nsurf_bound,dtype=float)
        dhdt_n1_2D = np.zeros(nsurf_bound,dtype=float)
        dhdt_n2_2D = np.zeros(nsurf_bound,dtype=float)
        dhdt_n3_2D = np.zeros(nsurf_bound,dtype=float)
        
        dene_i1   = ene_bins_i1[1]-ene_bins_i1[0]
        dangle_i1 = angle_bins_i1[1]-angle_bins_i1[0]
        dene_i2   = ene_bins_i2[1]-ene_bins_i2[0]
        dangle_i2 = angle_bins_i2[1]-angle_bins_i2[0]
        dene_i3   = ene_bins_i3[1]-ene_bins_i3[0]
        dangle_i3 = angle_bins_i3[1]-angle_bins_i3[0]
        dene_i4   = ene_bins_i4[1]-ene_bins_i4[0]
        dangle_i4 = angle_bins_i4[1]-angle_bins_i4[0]
        dene_n1   = ene_bins_n1[1]-ene_bins_n1[0]
        dangle_n1 = angle_bins_n1[1]-angle_bins_n1[0]
        dene_n2   = ene_bins_n2[1]-ene_bins_n2[0]
        dangle_n2 = angle_bins_n2[1]-angle_bins_n2[0]
        dene_n3   = ene_bins_n3[1]-ene_bins_n3[0]
        dangle_n3 = angle_bins_n3[1]-angle_bins_n3[0]
        
        for E in range (0,nbins_ene):
            for angle in range(0,nbins_angle):
                Y0_E_i1_2D[:,E,angle]   = erosion_Y0(c,E_th,ene_bins_i1[E]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Y0_E_i2_2D[:,E,angle]   = erosion_Y0(c,E_th,ene_bins_i2[E]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Y0_E_i3_2D[:,E,angle]   = erosion_Y0(c,E_th,ene_bins_i3[E]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Y0_E_i4_2D[:,E,angle]   = erosion_Y0(c,E_th,ene_bins_i4[E]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Y0_E_n1_2D[:,E,angle]   = erosion_Y0(c,E_th,ene_bins_n1[E]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Y0_E_n2_2D[:,E,angle]   = erosion_Y0(c,E_th,ene_bins_n2[E]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Y0_E_n3_2D[:,E,angle]   = erosion_Y0(c,E_th,ene_bins_n3[E]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Ftheta_i1_2D[:,E,angle] = erosion_Ftheta(Fmax,theta_max,a,angle_bins_i1[angle]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Ftheta_i2_2D[:,E,angle] = erosion_Ftheta(Fmax,theta_max,a,angle_bins_i2[angle]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Ftheta_i3_2D[:,E,angle] = erosion_Ftheta(Fmax,theta_max,a,angle_bins_i3[angle]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Ftheta_i4_2D[:,E,angle] = erosion_Ftheta(Fmax,theta_max,a,angle_bins_i4[angle]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Ftheta_n1_2D[:,E,angle] = erosion_Ftheta(Fmax,theta_max,a,angle_bins_n1[angle]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Ftheta_n2_2D[:,E,angle] = erosion_Ftheta(Fmax,theta_max,a,angle_bins_n2[angle]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                Ftheta_n3_2D[:,E,angle] = erosion_Ftheta(Fmax,theta_max,a,angle_bins_n3[angle]*np.ones(nsurf_bound,dtype=float),nsurf_bound)
                
                # Erosion rates for each species in [mm/s]
                dhdt_i1_2D = dhdt_i1_2D + dene_i1*dangle_i1*Y0_E_i1_2D[:,E,angle]*Ftheta_i1_2D[:,E,angle]*ene_angle_df_i1_plot[:,E,angle]*e*1E-6 
                dhdt_i2_2D = dhdt_i2_2D + dene_i2*dangle_i2*Y0_E_i2_2D[:,E,angle]*Ftheta_i2_2D[:,E,angle]*ene_angle_df_i2_plot[:,E,angle]*e*1E-6 
                dhdt_i3_2D = dhdt_i3_2D + dene_i3*dangle_i3*Y0_E_i3_2D[:,E,angle]*Ftheta_i3_2D[:,E,angle]*ene_angle_df_i3_plot[:,E,angle]*e*1E-6 
                dhdt_i4_2D = dhdt_i4_2D + dene_i4*dangle_i4*Y0_E_i4_2D[:,E,angle]*Ftheta_i4_2D[:,E,angle]*ene_angle_df_i4_plot[:,E,angle]*e*1E-6 
                dhdt_n1_2D = dhdt_n1_2D + dene_n1*dangle_n1*Y0_E_n1_2D[:,E,angle]*Ftheta_n1_2D[:,E,angle]*ene_angle_df_n1_plot[:,E,angle]*e*1E-6 
                dhdt_n2_2D = dhdt_n2_2D + dene_n2*dangle_n2*Y0_E_n2_2D[:,E,angle]*Ftheta_n2_2D[:,E,angle]*ene_angle_df_n2_plot[:,E,angle]*e*1E-6 
                dhdt_n3_2D = dhdt_n3_2D + dene_n3*dangle_n3*Y0_E_n3_2D[:,E,angle]*Ftheta_n3_2D[:,E,angle]*ene_angle_df_n3_plot[:,E,angle]*e*1E-6 
                
        dhdt_2D      = dhdt_i1_2D + dhdt_i2_2D + dhdt_i3_2D + dhdt_i4_2D + dhdt_n1_2D + dhdt_n2_2D + dhdt_n3_2D
        # Erosion rates in [mum/h]
        dhdt_mumh_2D    = dhdt_2D*1E3*3600 
        dhdt_i1_2D_mumh = dhdt_i1_2D*1E3*3600     
        dhdt_i2_2D_mumh = dhdt_i2_2D*1E3*3600    
        dhdt_i3_2D_mumh = dhdt_i3_2D*1E3*3600   
        dhdt_i4_2D_mumh = dhdt_i4_2D*1E3*3600   
        dhdt_n1_2D_mumh = dhdt_n1_2D*1E3*3600     
        dhdt_n2_2D_mumh = dhdt_n2_2D*1E3*3600    
        dhdt_n3_2D_mumh = dhdt_n3_2D*1E3*3600                                      
        h_mum_2D        = dhdt_mumh_2D*t_op  # h(z) [mum] erosion profile after a given operation time in hours
        h_m_2D          = h_mum_2D*1E-6      # h(z) [m] for performing operations
              
        Y0_E_i1_mean   = np.sum(np.sum(Y0_E_i1_2D*ene_angle_df_i1_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_i1_plot,axis=1),axis=1)  
        Y0_E_i2_mean   = np.sum(np.sum(Y0_E_i2_2D*ene_angle_df_i2_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_i2_plot,axis=1),axis=1)
        Y0_E_i3_mean   = np.sum(np.sum(Y0_E_i3_2D*ene_angle_df_i3_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_i3_plot,axis=1),axis=1)
        Y0_E_i4_mean   = np.sum(np.sum(Y0_E_i4_2D*ene_angle_df_i4_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_i4_plot,axis=1),axis=1)
        Y0_E_n1_mean   = np.sum(np.sum(Y0_E_n1_2D*ene_angle_df_n1_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_n1_plot,axis=1),axis=1)  
        Y0_E_n2_mean   = np.sum(np.sum(Y0_E_n2_2D*ene_angle_df_n2_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_n2_plot,axis=1),axis=1)
        Y0_E_n3_mean   = np.sum(np.sum(Y0_E_n3_2D*ene_angle_df_n3_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_n3_plot,axis=1),axis=1)
        Ftheta_i1_mean = np.sum(np.sum(Ftheta_i1_2D*ene_angle_df_i1_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_i1_plot,axis=1),axis=1)  
        Ftheta_i2_mean = np.sum(np.sum(Ftheta_i2_2D*ene_angle_df_i2_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_i2_plot,axis=1),axis=1) 
        Ftheta_i3_mean = np.sum(np.sum(Ftheta_i3_2D*ene_angle_df_i3_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_i3_plot,axis=1),axis=1) 
        Ftheta_i4_mean = np.sum(np.sum(Ftheta_i4_2D*ene_angle_df_i4_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_i4_plot,axis=1),axis=1) 
        Ftheta_n1_mean = np.sum(np.sum(Ftheta_n1_2D*ene_angle_df_n1_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_n1_plot,axis=1),axis=1)  
        Ftheta_n2_mean = np.sum(np.sum(Ftheta_n2_2D*ene_angle_df_n2_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_n2_plot,axis=1),axis=1) 
        Ftheta_n3_mean = np.sum(np.sum(Ftheta_n3_2D*ene_angle_df_n3_plot,axis=1),axis=1)/np.sum(np.sum(ene_angle_df_n3_plot,axis=1),axis=1) 
        
        
        # Obtain the chamber Dwalls eroded profile for 1D distribution functions
#        np_erosion_bot = 2+nsurf_inC_Dwall_bot  # Taking inner points along lateral walls as centers of panels
#        np_erosion_top = 2+nsurf_inC_Dwall_top 
        np_erosion_bot = 1+nsurf_inC_Dwall_bot   # Taking inner points along lateral walls as PIC mesh nodes
        np_erosion_top = 1+nsurf_inC_Dwall_top 
        re_Dwall_bot   = np.zeros(np_erosion_bot,dtype=float)
        ze_Dwall_bot   = np.zeros(np_erosion_bot,dtype=float)
        re_Dwall_top   = np.zeros(np_erosion_top,dtype=float)
        ze_Dwall_top   = np.zeros(np_erosion_top,dtype=float)
        # Dwall_bot
        for ind_er in range(0,np_erosion_bot):
            if ind_er == 0:
                # First point is the first picM node along Dwall_bot
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_bot[0]][0],imp_elems[indsurf_Dwall_bot[0]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_bot[0]][0],imp_elems[indsurf_Dwall_bot[0]][1],1] 
                re_Dwall_bot[ind_er] = rs[eta_min,0] + h_m[indsurf_Dwall_bot[0]]*norm_r
                ze_Dwall_bot[ind_er] = zs[eta_min,0] + h_m[indsurf_Dwall_bot[0]]*norm_z
#                re_Dwall_bot[ind_er] = rs[eta_min,0] + 0.5*h_m[indsurf_Dwall_bot[0]]*norm_r
#                ze_Dwall_bot[ind_er] = zs[eta_min,0] + 0.5*h_m[indsurf_Dwall_bot[0]]*norm_z
            elif ind_er == np_erosion_bot-1:
                # Last point is the last picM node along Dwall_bot
                if erprof_app == 0:
                    norm_r = -norm_vers[imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][0],imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][1],0] 
                    norm_z = -norm_vers[imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][0],imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][1],1] 
                    re_Dwall_bot[ind_er] = rs[eta_min,xi_bottom] + h_m[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*norm_r
                    ze_Dwall_bot[ind_er] = zs[eta_min,xi_bottom] + h_m[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*norm_z
                elif erprof_app == 1:
                    norm_r = -1.0
                    norm_z = 0.0
                    re_Dwall_bot[ind_er] = rs[eta_min,xi_bottom] + h_m[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*norm_r
                    ze_Dwall_bot[ind_er] = zs[eta_min,xi_bottom] + h_m[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*norm_z
                elif erprof_app == 2:
                    m_st_line = (re_Dwall_bot[ind_er-2] - re_Dwall_bot[ind_er-1]) /(ze_Dwall_bot[ind_er-2] - ze_Dwall_bot[ind_er-1])
                    n_st_line =  re_Dwall_bot[ind_er-1] - m_st_line*ze_Dwall_bot[ind_er-1]
                    re_Dwall_bot[ind_er] = m_st_line*zs[eta_min,xi_bottom] + n_st_line
                    ze_Dwall_bot[ind_er] = zs[eta_min,xi_bottom]
            else:
                # Points in the middle 
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_bot[ind_er-1]][0],imp_elems[indsurf_Dwall_bot[ind_er-1]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_bot[ind_er-1]][0],imp_elems[indsurf_Dwall_bot[ind_er-1]][1],1] 
                # Old approach: points in the middle correspond to central points of picS surface elements along Dwall_bot
#                re_Dwall_bot[ind_er] = rsurf_Dwall_bot[ind_er-1] + h_m[indsurf_Dwall_bot[ind_er-1]]*norm_r
#                ze_Dwall_bot[ind_er] = zsurf_Dwall_bot[ind_er-1] + h_m[indsurf_Dwall_bot[ind_er-1]]*norm_z
                # New approach: points in the middle correspond to picM nodes along Dwall_bot
                re_Dwall_bot[ind_er] = rs[eta_min,ind_er] + 0.5*(h_m[indsurf_Dwall_bot[ind_er-1]]+h_m[indsurf_Dwall_bot[ind_er]])*norm_r
                ze_Dwall_bot[ind_er] = zs[eta_min,ind_er] + 0.5*(h_m[indsurf_Dwall_bot[ind_er-1]]+h_m[indsurf_Dwall_bot[ind_er]])*norm_z
        # Dwall_top
        for ind_er in range(0,np_erosion_top):
            if ind_er == 0:
                # First point is the first picM node along Dwall_top
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_top[0]][0],imp_elems[indsurf_Dwall_top[0]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_top[0]][0],imp_elems[indsurf_Dwall_top[0]][1],1] 
                re_Dwall_top[ind_er] = rs[eta_max,0] + h_m[indsurf_Dwall_top[0]]*norm_r
                ze_Dwall_top[ind_er] = zs[eta_max,0] + h_m[indsurf_Dwall_top[0]]*norm_z
            elif ind_er == np_erosion_top-1:
                # Last point is the last picM node along Dwall_top
                if erprof_app == 0:
                    norm_r = -norm_vers[imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][0],imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][1],0] 
                    norm_z = -norm_vers[imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][0],imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][1],1] 
                    re_Dwall_top[ind_er] = rs[eta_max,xi_top] + h_m[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*norm_r
                    ze_Dwall_top[ind_er] = zs[eta_max,xi_top] + h_m[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*norm_z
                elif erprof_app == 1:
                    norm_r = 1.0
                    norm_z = 0.0
                    re_Dwall_top[ind_er] = rs[eta_max,xi_top] + h_m[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*norm_r
                    ze_Dwall_top[ind_er] = zs[eta_max,xi_top] + h_m[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*norm_z
                elif erprof_app == 2:
                    m_st_line = (re_Dwall_top[ind_er-2] - re_Dwall_top[ind_er-1]) /(ze_Dwall_top[ind_er-2] - ze_Dwall_top[ind_er-1])
                    n_st_line =  re_Dwall_top[ind_er-1] - m_st_line*ze_Dwall_top[ind_er-1]
                    re_Dwall_top[ind_er] = m_st_line*zs[eta_max,xi_top] + n_st_line
                    ze_Dwall_top[ind_er] = zs[eta_max,xi_top]
            else:
                # Points in the middle 
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_top[ind_er-1]][0],imp_elems[indsurf_Dwall_top[ind_er-1]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_top[ind_er-1]][0],imp_elems[indsurf_Dwall_top[ind_er-1]][1],1] 
                # Old approach: points in the middle correspond to central points of picS surface elements along Dwall_top
#                re_Dwall_top[ind_er] = rsurf_Dwall_top[ind_er-1] + h_m[indsurf_Dwall_top[ind_er-1]]*norm_r
#                ze_Dwall_top[ind_er] = zsurf_Dwall_top[ind_er-1] + h_m[indsurf_Dwall_top[ind_er-1]]*norm_z
                # New approach: points in the middle correspond to picM nodes along Dwall_top
                re_Dwall_top[ind_er] = rs[eta_max,ind_er] + 0.5*(h_m[indsurf_Dwall_top[ind_er-1]]+h_m[indsurf_Dwall_top[ind_er]])*norm_r
                ze_Dwall_top[ind_er] = zs[eta_max,ind_er] + 0.5*(h_m[indsurf_Dwall_top[ind_er-1]]+h_m[indsurf_Dwall_top[ind_er]])*norm_z
                

        # Obtain the chamber Dwalls eroded profile for 2D distribution functions
        np_erosion_bot = 1+nsurf_inC_Dwall_bot 
        np_erosion_top = 1+nsurf_inC_Dwall_top 
        re_Dwall_bot_2D   = np.zeros(np_erosion_bot,dtype=float)
        ze_Dwall_bot_2D   = np.zeros(np_erosion_bot,dtype=float)
        re_Dwall_top_2D   = np.zeros(np_erosion_top,dtype=float)
        ze_Dwall_top_2D   = np.zeros(np_erosion_top,dtype=float)
        # Dwall_bot
        for ind_er in range(0,np_erosion_bot):
            if ind_er == 0:
                # First point is the first picM node along Dwall_bot
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_bot[0]][0],imp_elems[indsurf_Dwall_bot[0]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_bot[0]][0],imp_elems[indsurf_Dwall_bot[0]][1],1] 
                re_Dwall_bot_2D[ind_er] = rs[eta_min,0] + h_m_2D[indsurf_Dwall_bot[0]]*norm_r
                ze_Dwall_bot_2D[ind_er] = zs[eta_min,0] + h_m_2D[indsurf_Dwall_bot[0]]*norm_z
            elif ind_er == np_erosion_bot-1:
                # Last point is the last picM node along Dwall_bot
                if erprof_app == 0:
                    norm_r = -norm_vers[imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][0],imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][1],0] 
                    norm_z = -norm_vers[imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][0],imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][1],1] 
                    re_Dwall_bot_2D[ind_er] = rs[eta_min,xi_bottom] + h_m_2D[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*norm_r
                    ze_Dwall_bot_2D[ind_er] = zs[eta_min,xi_bottom] + h_m_2D[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*norm_z
                elif erprof_app == 1:
                    norm_r = -1.0
                    norm_z = 0.0
                    re_Dwall_bot_2D[ind_er] = rs[eta_min,xi_bottom] + h_m_2D[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*norm_r
                    ze_Dwall_bot_2D[ind_er] = zs[eta_min,xi_bottom] + h_m_2D[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*norm_z
                elif erprof_app == 2:
                    m_st_line = (re_Dwall_bot_2D[ind_er-2] - re_Dwall_bot_2D[ind_er-1]) /(ze_Dwall_bot_2D[ind_er-2] - ze_Dwall_bot_2D[ind_er-1])
                    n_st_line =  re_Dwall_bot_2D[ind_er-1] - m_st_line*ze_Dwall_bot_2D[ind_er-1]
                    re_Dwall_bot_2D[ind_er] = m_st_line*zs[eta_min,xi_bottom] + n_st_line
                    ze_Dwall_bot_2D[ind_er] = zs[eta_min,xi_bottom]
            else:
                # Points in the middle
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_bot[ind_er-1]][0],imp_elems[indsurf_Dwall_bot[ind_er-1]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_bot[ind_er-1]][0],imp_elems[indsurf_Dwall_bot[ind_er-1]][1],1] 
                # Old approach: points in the middle correspond to central points of picS surface elements along Dwall_bot
#                re_Dwall_bot_2D[ind_er] = rsurf_Dwall_bot[ind_er-1] + h_m_2D[indsurf_Dwall_bot[ind_er-1]]*norm_r
#                ze_Dwall_bot_2D[ind_er] = zsurf_Dwall_bot[ind_er-1] + h_m_2D[indsurf_Dwall_bot[ind_er-1]]*norm_z
                # New approach: points in the middle correspond to picM nodes along Dwall_bot
                re_Dwall_bot_2D[ind_er] = rs[eta_min,ind_er] + 0.5*(h_m_2D[indsurf_Dwall_bot[ind_er-1]]+h_m_2D[indsurf_Dwall_bot[ind_er]])*norm_r
                ze_Dwall_bot_2D[ind_er] = zs[eta_min,ind_er] + 0.5*(h_m_2D[indsurf_Dwall_bot[ind_er-1]]+h_m_2D[indsurf_Dwall_bot[ind_er]])*norm_z
        # Dwall_top
        for ind_er in range(0,np_erosion_top):
            if ind_er == 0:
                # First point is the first picM node along Dwall_top
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_top[0]][0],imp_elems[indsurf_Dwall_top[0]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_top[0]][0],imp_elems[indsurf_Dwall_top[0]][1],1] 
                re_Dwall_top_2D[ind_er] = rs[eta_max,0] + h_m_2D[indsurf_Dwall_top[0]]*norm_r
                ze_Dwall_top_2D[ind_er] = zs[eta_max,0] + h_m_2D[indsurf_Dwall_top[0]]*norm_z
            elif ind_er == np_erosion_top-1:
                # Last point is the last picM node along Dwall_top
                if erprof_app == 0:
                    norm_r = -norm_vers[imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][0],imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][1],0] 
                    norm_z = -norm_vers[imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][0],imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][1],1] 
                    re_Dwall_top_2D[ind_er] = rs[eta_max,xi_top] + h_m_2D[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*norm_r
                    ze_Dwall_top_2D[ind_er] = zs[eta_max,xi_top] + h_m_2D[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*norm_z
                elif erprof_app == 1:
                    norm_r = 1.0
                    norm_z = 0.0
                    re_Dwall_top_2D[ind_er] = rs[eta_max,xi_top] + h_m_2D[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*norm_r
                    ze_Dwall_top_2D[ind_er] = zs[eta_max,xi_top] + h_m_2D[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*norm_z
                elif erprof_app == 2:
                    m_st_line = (re_Dwall_top_2D[ind_er-2] - re_Dwall_top_2D[ind_er-1]) /(ze_Dwall_top_2D[ind_er-2] - ze_Dwall_top_2D[ind_er-1])
                    n_st_line =  re_Dwall_top_2D[ind_er-1] - m_st_line*ze_Dwall_top_2D[ind_er-1]
                    re_Dwall_top_2D[ind_er] = m_st_line*zs[eta_max,xi_top] + n_st_line
                    ze_Dwall_top_2D[ind_er] = zs[eta_max,xi_top]
            else:
                # Points in the middle 
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_top[ind_er-1]][0],imp_elems[indsurf_Dwall_top[ind_er-1]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_top[ind_er-1]][0],imp_elems[indsurf_Dwall_top[ind_er-1]][1],1]
                # Old approach: points in the middle correspond to central points of picS surface elements along Dwall_top
#                re_Dwall_top_2D[ind_er] = rsurf_Dwall_top[ind_er-1] + h_m_2D[indsurf_Dwall_top[ind_er-1]]*norm_r
#                ze_Dwall_top_2D[ind_er] = zsurf_Dwall_top[ind_er-1] + h_m_2D[indsurf_Dwall_top[ind_er-1]]*norm_z
                # New approach: points in the middle correspond to picM nodes along Dwall_bot
                re_Dwall_top_2D[ind_er] = rs[eta_max,ind_er] + 0.5*(h_m_2D[indsurf_Dwall_top[ind_er-1]]+h_m_2D[indsurf_Dwall_top[ind_er]])*norm_r
                ze_Dwall_top_2D[ind_er] = zs[eta_max,ind_er] + 0.5*(h_m_2D[indsurf_Dwall_top[ind_er-1]]+h_m_2D[indsurf_Dwall_top[ind_er]])*norm_z


        # Currents in A/cm2
        ji_surf_plot     = ji_surf_plot*1E-4
        ji1_surf_plot    = ji1_surf_plot*1E-4
        ji2_surf_plot    = ji2_surf_plot*1E-4
        ji3_surf_plot    = ji3_surf_plot*1E-4
        ji4_surf_plot    = ji4_surf_plot*1E-4
        jn_surf_plot     = gn_tw_surf_plot*e*1E-4
        jn1_surf_plot    = gn1_tw_surf_plot*e*1E-4
        jn2_surf_plot    = gn2_tw_surf_plot*e*1E-4
        jn3_surf_plot    = gn3_tw_surf_plot*e*1E-4
        
        df_ji_surf_plot  = df_ji_surf_plot*1E-4
        df_ji1_surf_plot = df_ji1_surf_plot*1E-4
        df_ji2_surf_plot = df_ji2_surf_plot*1E-4
        df_ji3_surf_plot = df_ji3_surf_plot*1E-4
        df_ji4_surf_plot = df_ji4_surf_plot*1E-4
        df_jn_surf_plot  = df_gn_tw_surf_plot*e*1E-4
        df_jn1_surf_plot = df_gn1_tw_surf_plot*e*1E-4
        df_jn2_surf_plot = df_gn2_tw_surf_plot*e*1E-4
        df_jn3_surf_plot = df_gn3_tw_surf_plot*e*1E-4
        
        qi_tot_wall_surf_plot   = qi_tot_wall_surf_plot*1E-4
        qn_tot_wall_surf_plot   = qn_tot_wall_surf_plot*1E-4

        # Impact energies in eV
        imp_ene_i1_surf_plot     = imp_ene_i1_surf_plot/e
        imp_ene_i2_surf_plot     = imp_ene_i2_surf_plot/e
        imp_ene_i3_surf_plot     = imp_ene_i3_surf_plot/e
        imp_ene_i4_surf_plot     = imp_ene_i4_surf_plot/e
#        imp_ene_ion_surf_plot    = imp_ene_i1_surf_plot*(ji1_surf_plot/ji_surf_plot) + imp_ene_i2_surf_plot*(ji2_surf_plot/ji_surf_plot) + imp_ene_i3_surf_plot*(ji3_surf_plot/ji_surf_plot) + imp_ene_i4_surf_plot*(ji4_surf_plot/ji_surf_plot)
        den                      = ji1_surf_plot/e + ji2_surf_plot/(2*e) + ji3_surf_plot/e + ji4_surf_plot/(2*e)
        imp_ene_ion_surf_plot    = imp_ene_i1_surf_plot*(ji1_surf_plot/e/den) + imp_ene_i2_surf_plot*(ji2_surf_plot/(2*e)/den) + imp_ene_i3_surf_plot*(ji3_surf_plot/e/den) + imp_ene_i4_surf_plot*(ji4_surf_plot/(2*e)/den)
        imp_ene_ion_surf_plot_v2 = qi_tot_wall_surf_plot/(ji_surf_plot/e)/e
        imp_ene_n1_surf_plot     = imp_ene_n1_surf_plot/e
        imp_ene_n2_surf_plot     = imp_ene_n2_surf_plot/e
        imp_ene_n3_surf_plot     = imp_ene_n3_surf_plot/e
        imp_ene_n_surf_plot      = imp_ene_n1_surf_plot*(jn1_surf_plot/jn_surf_plot) + imp_ene_n2_surf_plot*(jn2_surf_plot/jn_surf_plot) + imp_ene_n3_surf_plot*(jn3_surf_plot/jn_surf_plot)
        imp_ene_n_surf_plot_v2   = qn_tot_wall_surf_plot/(jn_surf_plot/e)/e

        
        # Arc lengths in cm
        sc_bot           = sc_bot*1E2
        sc_top           = sc_top*1E2
        sDwall_bot_surf  = sDwall_bot_surf*1E2
        sDwall_top_surf  = sDwall_top_surf*1E2
        sAwall_surf      = sAwall_surf*1E2
        sFLwall_ver_surf = sFLwall_ver_surf*1E2
        sFLwall_lat_surf = sFLwall_lat_surf*1E2
        
        # Coordinates in cm
        zsurf_Dwall_bot = zsurf_Dwall_bot*1E2
        rsurf_Dwall_bot = rsurf_Dwall_bot*1E2
        zsurf_Dwall_top = zsurf_Dwall_top*1E2
        rsurf_Dwall_top = rsurf_Dwall_top*1E2
        zsurf_Awall     = zsurf_Awall*1E2
        rsurf_Awall     = rsurf_Awall*1E2
        re_Dwall_bot    = re_Dwall_bot*1E2
        ze_Dwall_bot    = ze_Dwall_bot*1E2
        re_Dwall_top    = re_Dwall_top*1E2
        ze_Dwall_top    = ze_Dwall_top*1E2
        re_Dwall_bot_2D = re_Dwall_bot_2D*1E2
        ze_Dwall_bot_2D = ze_Dwall_bot_2D*1E2
        re_Dwall_top_2D = re_Dwall_top_2D*1E2
        ze_Dwall_top_2D = ze_Dwall_top_2D*1E2
        zs              = zs*1E2
        rs              = rs*1E2
        
        max_dhdt_mumh_bot        = np.max(dhdt_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]])
        max_dhdt_mumh_2D_bot     = np.max(dhdt_mumh_2D[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]])
        pos_max_dhdt_mumh_bot    = np.where(dhdt_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]] == max_dhdt_mumh_bot)[0][0]
        pos_max_dhdt_mumh_2D_bot = np.where(dhdt_mumh_2D[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]] == max_dhdt_mumh_2D_bot)[0][0]
        s_max_dhdt_mumh_bot      = sDwall_bot_surf[indsurf_inC_Dwall_bot][pos_max_dhdt_mumh_bot]
        s_max_dhdt_mumh_2D_bot   = sDwall_bot_surf[indsurf_inC_Dwall_bot][pos_max_dhdt_mumh_2D_bot]
        
        max_dhdt_mumh_top      = np.max(dhdt_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]])
        max_dhdt_mumh_2D_top   = np.max(dhdt_mumh_2D[indsurf_Dwall_top[indsurf_inC_Dwall_top]])
        pos_max_dhdt_mumh_top    = np.where(dhdt_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]] == max_dhdt_mumh_top)[0][0]
        pos_max_dhdt_mumh_2D_top = np.where(dhdt_mumh_2D[indsurf_Dwall_top[indsurf_inC_Dwall_top]] == max_dhdt_mumh_2D_top)[0][0]
        s_max_dhdt_mumh_top      = sDwall_top_surf[indsurf_inC_Dwall_top][pos_max_dhdt_mumh_top]
        s_max_dhdt_mumh_2D_top   = sDwall_top_surf[indsurf_inC_Dwall_top][pos_max_dhdt_mumh_2D_top]
        
        
        
        #    # Do not plot units in axes
    #    # SAFRAN CHEOPS 1: units in cm
    ##    L_c = 3.725
    ##    H_c = (0.074995-0.052475)*100
    #    # HT5k: units in cm
    #    L_c = 2.53
    #    H_c = (0.0785-0.0565)*100
        # VHT_US (IEPC 2022)
    #    L_c = 2.9
    #    H_c = 2.22    
        # VHT_US PPSX00 testcase1 LP (TFM Alejandro)
    #    L_c = 2.5
    #    H_c = 1.1
        # PPSX00 testcase2 LP
        L_c = 2.5
        H_c = 1.5
        
     #   L_c = 1.0
     #   H_c = 1.0
        
        
        print("Operation time                           [h] = "+str(t_op))
        print("h at chamber exit bot                   [mm] = %15.8e" %(h_mum[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*1E-3))
        print("2D h at chamber exit bot                [mm] = %15.8e" %(h_mum_2D[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*1E-3))
        print("h at chamber exit top                   [mm] = %15.8e" %(h_mum[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*1E-3))
        print("2D h at chamber exit top                [mm] = %15.8e" %(h_mum_2D[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*1E-3))
        print("Average h at chamber bot                [mm] = %15.8e" %(np.mean(h_mum[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]])*1E-3))
        print("2D Average h at chamber bot             [mm] = %15.8e" %(np.mean(h_mum_2D[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]])*1E-3))
        print("Average h at chamber top                [mm] = %15.8e" %(np.mean(h_mum[indsurf_Dwall_top[indsurf_inC_Dwall_top]])*1E-3))
        print("2D Average h at chamber top             [mm] = %15.8e" %(np.mean(h_mum_2D[indsurf_Dwall_top[indsurf_inC_Dwall_top]])*1E-3))
        
        print("max dhdt at chamber exit bot         [mum/h] = %15.8e" %(max_dhdt_mumh_bot))
        print("max 2D dhdt at chamber exit bot      [mum/h] = %15.8e" %(max_dhdt_mumh_2D_bot))
        print("s/Lc max dhdt at chamber exit bot        [-] = %15.8e" %(s_max_dhdt_mumh_bot/L_c))
        print("s/Lc max 2D dhdt at chamber exit bot     [-] = %15.8e" %(s_max_dhdt_mumh_2D_bot/L_c))
        
        print("max dhdt at chamber exit top         [mum/h] = %15.8e" %(max_dhdt_mumh_top))
        print("max 2D dhdt at chamber exit top      [mum/h] = %15.8e" %(max_dhdt_mumh_2D_top))
        print("s/Lc max dhdt at chamber exit top        [-] = %15.8e" %(s_max_dhdt_mumh_top/L_c))
        print("s/Lc max 2D dhdt at chamber exit top     [-] = %15.8e" %(s_max_dhdt_mumh_2D_top/L_c))
        
        print("r orig at chamber exit bot              [mm] = %15.8e" %(rs[eta_min,xi_bottom]*1E1))
        print("r eroded at chamber exit bot            [mm] = %15.8e" %(re_Dwall_bot[-1]*1E1))
        print("2D r eroded at chamber exit bot         [mm] = %15.8e" %(re_Dwall_bot_2D[-1]*1E1))
        print("r orig at chamber exit top              [mm] = %15.8e" %(rs[eta_max,xi_top]*1E1))
        print("r eroded at chamber exit top            [mm] = %15.8e" %(re_Dwall_top[-1]*1E1))
        print("2D r eroded at chamber exit top         [mm] = %15.8e" %(re_Dwall_top_2D[-1]*1E1))
        print("dr eroded at chamber exit bot           [mm] = %15.8e" %(np.abs(re_Dwall_bot[-1] - rs[eta_min,xi_bottom])*1E1))
        print("2D dr eroded at chamber exit bot        [mm] = %15.8e" %(np.abs(re_Dwall_bot_2D[-1] - rs[eta_min,xi_bottom])*1E1))
        print("dr eroded at chamber exit top           [mm] = %15.8e" %(np.abs(re_Dwall_top[-1] - rs[eta_max,xi_top])*1E1))
        print("2D dr eroded at chamber exit top        [mm] = %15.8e" %(np.abs(re_Dwall_top_2D[-1] - rs[eta_max,xi_top])*1E1))
                    
        
    
     
        sc_bot           = sc_bot/L_c
        sc_top           = sc_top/L_c
        sDwall_bot_surf  = sDwall_bot_surf/L_c
        sDwall_top_surf  = sDwall_top_surf/L_c
        sAwall_surf      = sAwall_surf/L_c
        sFLwall_ver_surf = sFLwall_ver_surf/L_c
        sFLwall_lat_surf = sFLwall_lat_surf/L_c
        
        # Coordinates in cm
        zsurf_Dwall_bot = zsurf_Dwall_bot/L_c
        rsurf_Dwall_bot = rsurf_Dwall_bot/H_c
        zsurf_Dwall_top = zsurf_Dwall_top/L_c
        rsurf_Dwall_top = rsurf_Dwall_top/H_c
        zsurf_Awall     = zsurf_Awall/L_c
        rsurf_Awall     = rsurf_Awall/H_c
        re_Dwall_bot    = re_Dwall_bot/H_c
        ze_Dwall_bot    = ze_Dwall_bot/L_c
        re_Dwall_top    = re_Dwall_top/H_c
        ze_Dwall_top    = ze_Dwall_top/L_c
        re_Dwall_bot_2D = re_Dwall_bot_2D/H_c
        ze_Dwall_bot_2D = ze_Dwall_bot_2D/L_c
        re_Dwall_top_2D = re_Dwall_top_2D/H_c
        ze_Dwall_top_2D = ze_Dwall_top_2D/L_c
        zs              = zs/L_c
        rs              = rs/H_c
     
        
        # Check for plots of df at points
        if plot_angle_df == 1 or plot_ene_df == 1 or plot_normv_df == 1:
            indsurf_points = np.zeros(npoints_plot_df,dtype=int)
            npoints_Dwall_bot = 0
            npoints_Dwall_top = 0
            npoints_Awall     = 0
            indsurf_points_Dwall_bot = np.zeros(0,dtype=int)
            indsurf_points_Dwall_top = np.zeros(0,dtype=int)
            indsurf_points_Awall     = np.zeros(0,dtype=int)
            for ind_point in range(0,npoints_plot_df):
                if bpoints_plot_df[ind_point] == "Dwall_bot":
                    dmin = 1E2
                    ind_Dwall_bot = 0
                    for cont in range(0,nsurf_Dwall_bot):
                        if np.sqrt((zpoints_plot_df[ind_point] - zsurf_Dwall_bot[cont])**2 + (rpoints_plot_df[ind_point] - rsurf_Dwall_bot[cont])**2) < dmin:
                            dmin = np.sqrt((zpoints_plot_df[ind_point] - zsurf_Dwall_bot[cont])**2 + (rpoints_plot_df[ind_point] - rsurf_Dwall_bot[cont])**2)
                            indsurf_points[ind_point] = indsurf_Dwall_bot[cont]
                            ind_Dwall_bot = cont
                            
                    if dmin == 1E2:
                        print("Point number "+str(ind_point+1)+" not found in boundary Dwall_bot")
                    else:
                        indsurf_points_Dwall_bot = np.append(indsurf_points_Dwall_bot,np.array([ind_Dwall_bot],dtype=int),axis=0)
                        npoints_Dwall_bot = npoints_Dwall_bot + 1
                    
                elif bpoints_plot_df[ind_point] == "Dwall_top":
                    dmin = 1E2
                    ind_Dwall_top = 0
                    for cont in range(0,nsurf_Dwall_top):
                        if np.sqrt((zpoints_plot_df[ind_point] - zsurf_Dwall_top[cont])**2 + (rpoints_plot_df[ind_point] - rsurf_Dwall_top[cont])**2) < dmin:
                            dmin = np.sqrt((zpoints_plot_df[ind_point] - zsurf_Dwall_top[cont])**2 + (rpoints_plot_df[ind_point] - rsurf_Dwall_top[cont])**2)
                            indsurf_points[ind_point] = indsurf_Dwall_top[cont]
                            ind_Dwall_top = cont
                    if dmin == 1E2:
                        print("Point number "+str(ind_point+1)+" not found in boundary Dwall_top")
                    else:
                        indsurf_points_Dwall_top = np.append(indsurf_points_Dwall_top,np.array([ind_Dwall_top],dtype=int),axis=0)
                        npoints_Dwall_top = npoints_Dwall_top + 1
                    
                elif bpoints_plot_df[ind_point] == "Awall":
                    dmin = 1E2
                    ind_Awall = 0
                    for cont in range(0,nsurf_Awall):
                        if np.sqrt((zpoints_plot_df[ind_point] - zsurf_Awall[cont])**2 + (rpoints_plot_df[ind_point] - rsurf_Awall[cont])**2) < dmin:
                            dmin = np.sqrt((zpoints_plot_df[ind_point] - zsurf_Awall[cont])**2 + (rpoints_plot_df[ind_point] - rsurf_Awall[cont])**2)
                            indsurf_points[ind_point] = indsurf_Awall[cont]
                            ind_Awall = cont
                    if dmin == 1E2:
                        print("Point number "+str(ind_point+1)+" not found in boundary Awall")
                    else:
                        indsurf_points_Awall = np.append(indsurf_points_Awall,np.array([ind_Awall],dtype=int),axis=0)
                        npoints_Awall = npoints_Awall + 1
                        
            print("N. points found in Dwall_bot = "+str(npoints_Dwall_bot))
            counter = 0
            for cont in range(0,npoints_plot_df):
                if bpoints_plot_df[cont] == "Dwall_bot":
                    print("Point "+str(cont+1)+" (z,r) coords (cm) = ("+str(zsurf_Dwall_bot[indsurf_points_Dwall_bot[counter]])+", "+str(rsurf_Dwall_bot[indsurf_points_Dwall_bot[counter]])+")")
                    counter = counter + 1
            print("N. points found in Dwall_top = "+str(npoints_Dwall_top))
            counter = 0
            for cont in range(0,npoints_plot_df):
                if bpoints_plot_df[cont] == "Dwall_top":
                    print("Point "+str(cont+1)+" (z,r) coords (cm) = ("+str(zsurf_Dwall_top[indsurf_points_Dwall_top[counter]])+", "+str(rsurf_Dwall_top[indsurf_points_Dwall_top[counter]])+")")
                    counter = counter + 1
            print("N. points found in Awall     = "+str(npoints_Awall))
            counter = 0
            for cont in range(0,npoints_plot_df):
                if bpoints_plot_df[cont] == "Awall":
                    print("Point "+str(cont+1)+" (z,r) coords (cm) = ("+str(zsurf_Awall[indsurf_points_Awall[counter]])+", "+str(rsurf_Awall[indsurf_points_Awall[counter]])+")")
                    counter = counter + 1
            
        
        # Distribution function plots at points
        if plot_angle_df == 1:
            counter_Dwall_bot = 0
            counter_Dwall_top = 0
            counter_Awall     = 0
            if log_yaxis == 1:
                angle_df_i1_plot[angle_df_i1_plot < log_yaxis_tol] = np.nan
                angle_df_i2_plot[angle_df_i2_plot < log_yaxis_tol] = np.nan
                angle_df_n1_plot[angle_df_n1_plot < log_yaxis_tol] = np.nan
            
            for cont in range(0,npoints_plot_df):
                if bpoints_plot_df[cont] == "Dwall_bot":
                    plt.figure("angle_df_i1 at points")
#                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_bot_surf[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"+" Dwall bot"
                    text1 = labels[k]+" P"+str(cont+1)+" $z = $"+"{:.2f}".format(zsurf_Dwall_bot[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(angle_bins_i1,angle_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(angle_bins_i1,angle_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(angle_bins_i1,angle_df_i1_plot[indsurf_points[cont]]/np.max(angle_df_i1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    plt.figure("angle_df_i2 at points")
#                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_bot_surf[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"+" Dwall bot"
                    text1 = labels[k]+" P"+str(cont+1)+" $z = $"+"{:.2f}".format(zsurf_Dwall_bot[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(angle_bins_i2,angle_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(angle_bins_i2,angle_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(angle_bins_i2,angle_df_i2_plot[indsurf_points[cont]]/np.max(angle_df_i2_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("angle_df_n1 at points")
#                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_bot_surf[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"+" Dwall bot"
                    text1 = labels[k]+" P"+str(cont+1)+" $z = $"+"{:.2f}".format(zsurf_Dwall_bot[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(angle_bins_n1,angle_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(angle_bins_n1,angle_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(angle_bins_n1,angle_df_n1_plot[indsurf_points[cont]]/np.max(angle_df_n1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    counter_Dwall_bot = counter_Dwall_bot + 1
                    
                elif bpoints_plot_df[cont] == "Dwall_top":
                    plt.figure("angle_df_i1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_top_surf[indsurf_points_Dwall_top[counter_Dwall_top]])+" cm"+" Dwall top"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(angle_bins_i1,angle_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(angle_bins_i1,angle_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(angle_bins_i1,angle_df_i1_plot[indsurf_points[cont]]/np.max(angle_df_i1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    plt.figure("angle_df_i2 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_top_surf[indsurf_points_Dwall_top[counter_Dwall_top]])+" cm"+" Dwall top"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(angle_bins_i2,angle_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0: 
                            plt.plot(angle_bins_i2,angle_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(angle_bins_i1,angle_df_i2_plot[indsurf_points[cont]]/np.max(angle_df_i2_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("angle_df_n1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_top_surf[indsurf_points_Dwall_top[counter_Dwall_top]])+" cm"+" Dwall top"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(angle_bins_n1,angle_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0: 
                            plt.plot(angle_bins_n1,angle_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(angle_bins_n1,angle_df_n1_plot[indsurf_points[cont]]/np.max(angle_df_n1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    counter_Dwall_top = counter_Dwall_top + 1
                    
                elif bpoints_plot_df[cont] == "Awall":
                    plt.figure("angle_df_i1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sAwall_surf[indsurf_points_Awall[counter_Awall]])+" cm"+" Awall"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(angle_bins_i1,angle_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0: 
                            plt.plot(angle_bins_i1,angle_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(angle_bins_i1,angle_df_i1_plot[indsurf_points[cont]]/np.max(angle_df_i1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("angle_df_i2 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sAwall_surf[indsurf_points_Awall[counter_Awall]])+" cm"+" Awall"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(angle_bins_i2,angle_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(angle_bins_i2,angle_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(angle_bins_i2,angle_df_i2_plot[indsurf_points[cont]]/np.max(angle_df_i2_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("angle_df_n1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sAwall_surf[indsurf_points_Awall[counter_Awall]])+" cm"+" Awall"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(angle_bins_n1,angle_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(angle_bins_n1,angle_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(angle_bins_n1,angle_df_n1_plot[indsurf_points[cont]]/np.max(angle_df_n1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    counter_Awall = counter_Awall + 1


        if plot_ene_df == 1:
            counter_Dwall_bot = 0
            counter_Dwall_top = 0
            counter_Awall     = 0
            if log_yaxis == 1:
                ene_df_i1_plot[ene_df_i1_plot < log_yaxis_tol] = np.nan
                ene_df_i2_plot[ene_df_i2_plot < log_yaxis_tol] = np.nan
                ene_df_n1_plot[ene_df_n1_plot < log_yaxis_tol] = np.nan
                
            for cont in range(0,npoints_plot_df):
                if bpoints_plot_df[cont] == "Dwall_bot":
                    plt.figure("ene_df_i1 at points")
#                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_bot_surf[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"+" Dwall bot"
                    text1 = labels[k]+" P"+str(cont+1)+" $z = $"+"{:.2f}".format(zsurf_Dwall_bot[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(ene_bins_i1,ene_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        if log_yaxis == 0:
                            plt.plot(ene_bins_i1,ene_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(ene_bins_i1,ene_df_i1_plot[indsurf_points[cont]]/np.max(ene_df_i1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    plt.figure("ene_df_i2 at points")
#                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_bot_surf[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"+" Dwall bot"
                    text1 = labels[k]+" P"+str(cont+1)+" $z = $"+"{:.2f}".format(zsurf_Dwall_bot[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(ene_bins_i2,ene_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(ene_bins_i2,ene_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(ene_bins_i2,ene_df_i2_plot[indsurf_points[cont]]/np.max(ene_df_i2_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("ene_df_n1 at points")
#                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_bot_surf[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"+" Dwall bot"
                    text1 = labels[k]+" P"+str(cont+1)+" $z = $"+"{:.2f}".format(zsurf_Dwall_bot[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(ene_bins_n1,ene_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(ene_bins_n1,ene_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(ene_bins_n1,ene_df_n1_plot[indsurf_points[cont]]/np.max(ene_df_n1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    counter_Dwall_bot = counter_Dwall_bot + 1
                    
                elif bpoints_plot_df[cont] == "Dwall_top":
                    plt.figure("ene_df_i1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_top_surf[indsurf_points_Dwall_top[counter_Dwall_top]])+" cm"+" Dwall top"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(ene_bins_i1,ene_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(ene_bins_i1,ene_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(ene_bins_i1,ene_df_i1_plot[indsurf_points[cont]]/np.max(ene_df_i1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    plt.figure("ene_df_i2 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_top_surf[indsurf_points_Dwall_top[counter_Dwall_top]])+" cm"+" Dwall top"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(ene_bins_i2,ene_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(ene_bins_i2,ene_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(ene_bins_i1,ene_df_i2_plot[indsurf_points[cont]]/np.max(ene_df_i2_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("ene_df_n1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_top_surf[indsurf_points_Dwall_top[counter_Dwall_top]])+" cm"+" Dwall top"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(ene_bins_n1,ene_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(ene_bins_n1,ene_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(ene_bins_n1,ene_df_n1_plot[indsurf_points[cont]]/np.max(ene_df_n1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    counter_Dwall_top = counter_Dwall_top + 1
                    
                elif bpoints_plot_df[cont] == "Awall":
                    plt.figure("ene_df_i1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sAwall_surf[indsurf_points_Awall[counter_Awall]])+" cm"+" Awall"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(ene_bins_i1,ene_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(ene_bins_i1,ene_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(ene_bins_i1,ene_df_i1_plot[indsurf_points[cont]]/np.max(ene_df_i1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("ene_df_i2 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sAwall_surf[indsurf_points_Awall[counter_Awall]])+" cm"+" Awall"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(ene_bins_i2,ene_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(ene_bins_i2,ene_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(ene_bins_i2,ene_df_i2_plot[indsurf_points[cont]]/np.max(ene_df_i2_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("ene_df_n1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sAwall_surf[indsurf_points_Awall[counter_Awall]])+" cm"+" Awall"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(ene_bins_n1,ene_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(ene_bins_n1,ene_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(ene_bins_n1,ene_df_n1_plot[indsurf_points[cont]]/np.max(ene_df_n1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    counter_Awall = counter_Awall + 1
                    
        
        if plot_normv_df == 1:
            counter_Dwall_bot = 0
            counter_Dwall_top = 0
            counter_Awall     = 0
            if log_yaxis == 1:
                normv_df_i1_plot[normv_df_i1_plot < log_yaxis_tol] = np.nan
                normv_df_i2_plot[normv_df_i2_plot < log_yaxis_tol] = np.nan
                normv_df_n1_plot[normv_df_n1_plot < log_yaxis_tol] = np.nan
            
            for cont in range(0,npoints_plot_df):
                if bpoints_plot_df[cont] == "Dwall_bot":
                    plt.figure("normv_df_i1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_bot_surf[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"+" Dwall bot"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(normv_bins_i1,normv_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(normv_bins_i1,normv_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(normv_bins_i1,normv_df_i1_plot[indsurf_points[cont]]/np.max(normv_df_i1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    plt.figure("normv_df_i2 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_bot_surf[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"+" Dwall bot"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(normv_bins_i2,normv_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(normv_bins_i2,normv_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(normv_bins_i2,normv_df_i2_plot[indsurf_points[cont]]/np.max(normv_df_i2_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("normv_df_n1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_bot_surf[indsurf_points_Dwall_bot[counter_Dwall_bot]])+" cm"+" Dwall bot"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(normv_bins_n1,normv_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 1:
                            plt.plot(normv_bins_n1,normv_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(normv_bins_n1,normv_df_n1_plot[indsurf_points[cont]]/np.max(normv_df_n1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    counter_Dwall_bot = counter_Dwall_bot + 1
                    
                elif bpoints_plot_df[cont] == "Dwall_top":
                    plt.figure("normv_df_i1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_top_surf[indsurf_points_Dwall_top[counter_Dwall_top]])+" cm"+" Dwall top"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(normv_bins_i1,normv_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(normv_bins_i1,normv_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(normv_bins_i1,normv_df_i1_plot[indsurf_points[cont]]/np.max(normv_df_i1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    plt.figure("normv_df_i2 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_top_surf[indsurf_points_Dwall_top[counter_Dwall_top]])+" cm"+" Dwall top"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(normv_bins_i2,normv_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:   
                            plt.plot(normv_bins_i2,normv_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(normv_bins_i1,normv_df_i2_plot[indsurf_points[cont]]/np.max(normv_df_i2_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("normv_df_n1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sDwall_top_surf[indsurf_points_Dwall_top[counter_Dwall_top]])+" cm"+" Dwall top"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(normv_bins_n1,normv_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(normv_bins_n1,normv_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(normv_bins_n1,normv_df_n1_plot[indsurf_points[cont]]/np.max(normv_df_n1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)

                    counter_Dwall_top = counter_Dwall_top + 1
                    
                elif bpoints_plot_df[cont] == "Awall":
                    plt.figure("normv_df_i1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sAwall_surf[indsurf_points_Awall[counter_Awall]])+" cm"+" Awall"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(normv_bins_i1,normv_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(normv_bins_i1,normv_df_i1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(normv_bins_i1,normv_df_i1_plot[indsurf_points[cont]]/np.max(normv_df_i1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("normv_df_i2 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sAwall_surf[indsurf_points_Awall[counter_Awall]])+" cm"+" Awall"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(normv_bins_i2,normv_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(normv_bins_i2,normv_df_i2_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(normv_bins_i2,normv_df_i2_plot[indsurf_points[cont]]/np.max(normv_df_i2_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    plt.figure("normv_df_n1 at points")
                    text1 = labels[k]+" P"+str(cont+1)+" $s = $"+"{:.2f}".format(sAwall_surf[indsurf_points_Awall[counter_Awall]])+" cm"+" Awall"
                    if normalized_df == 0:
                        if log_yaxis == 1:
                            plt.semilogy(normv_bins_n1,normv_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                        elif log_yaxis == 0:
                            plt.plot(normv_bins_n1,normv_df_n1_plot[indsurf_points[cont]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    elif normalized_df == 1:
                        plt.plot(normv_bins_n1,normv_df_n1_plot[indsurf_points[cont]]/np.max(normv_df_n1_plot[indsurf_points[cont]]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[cont], markeredgecolor = 'k', label=text1)
                    
                    counter_Awall = counter_Awall + 1
        
        # Axial profile plots
        if plot_Dwall == 1:
            if plot_curr == 1:
                plt.figure(r'ji1 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],ji1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],ji1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'ji2 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],ji2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],ji2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'ji3 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],ji3_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],ji3_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")

                plt.figure(r'ji4 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],ji4_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],ji4_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'ji_tot_b Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],ji_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf,df_ji_surf_plot[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],ji_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf,df_ji_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")

                plt.figure(r'egn1 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],jn1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],jn1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'egn2 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],jn2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],jn2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'egn3 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],jn3_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],jn3_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'egn Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],jn_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],jn_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
            if plot_angle == 1:
                plt.figure(r'angle_i1 Dwall_bot_top')
#                plt.plot(sDwall_bot_surf,df_angle_i1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.plot(sDwall_top_surf,df_angle_i1_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top df")
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_angle_i1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],df_angle_i1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'angle_i2 Dwall_bot_top')
#                plt.plot(sDwall_bot_surf,df_angle_i2_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.plot(sDwall_top_surf,df_angle_i2_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top df")
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_angle_i2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],df_angle_i2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'angle_i3 Dwall_bot_top')
#                plt.plot(sDwall_bot_surf,df_angle_i3_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.plot(sDwall_top_surf,df_angle_i3_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top df")
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_angle_i3_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],df_angle_i3_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'angle_i4 Dwall_bot_top')
#                plt.plot(sDwall_bot_surf,df_angle_i4_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.plot(sDwall_top_surf,df_angle_i4_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top df")
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_angle_i4_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],df_angle_i4_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")

                plt.figure(r'angle_ion Dwall_bot_top')
#                plt.plot(sDwall_bot_surf,df_angle_ion_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.plot(sDwall_top_surf,df_angle_ion_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top df")
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_angle_ion_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],df_angle_ion_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
               
                plt.figure(r'angle_n1 Dwall_bot_top')
#                plt.plot(sDwall_bot_surf,df_angle_n1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.plot(sDwall_top_surf,df_angle_n1_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top df")
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_angle_n1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],df_angle_n1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                
                plt.figure(r'angle_n2 Dwall_bot_top')
#                plt.plot(sDwall_bot_surf,df_angle_n2_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.plot(sDwall_top_surf,df_angle_n2_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top df")
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_angle_n2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],df_angle_n2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")

                plt.figure(r'angle_n3 Dwall_bot_top')
#                plt.plot(sDwall_bot_surf,df_angle_n3_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.plot(sDwall_top_surf,df_angle_n3_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top df")
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_angle_n3_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],df_angle_n3_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'angle_neu Dwall_bot_top')
#                plt.plot(sDwall_bot_surf,df_angle_neu_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.plot(sDwall_top_surf,df_angle_neu_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top df")
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_angle_neu_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],df_angle_neu_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
               
                
            if plot_imp_ene == 1:
                plt.figure(r'imp_ene_i1 Dwall_bot_top')
#                plt.semilogy(sDwall_bot_surf,imp_ene_i1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf,df_imp_ene_i1_surf_plot[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
##                plt.semilogy(sDwall_bot_surf,df_imp_ene_i1_surf_plot_2_avg[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='g', markeredgecolor = 'k', label=labels[k]+" bot df 2")
#                plt.semilogy(sDwall_top_surf,imp_ene_i1_surf_plot[indsurf_Dwall_top], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf,df_imp_ene_i1_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
##                plt.semilogy(sDwall_top_surf,df_imp_ene_i1_surf_plot_2_avg[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='g', markeredgecolor = 'k', label=labels[k]+" top df 2")
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_i1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_imp_ene_i1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.semilogy(sDwall_bot_surf,df_imp_ene_i1_surf_plot_2_avg[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='g', markeredgecolor = 'k', label=labels[k]+" bot df 2")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_i1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],df_imp_ene_i1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
#                plt.semilogy(sDwall_top_surf,df_imp_ene_i1_surf_plot_2_avg[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='g', markeredgecolor = 'k', label=labels[k]+" top df 2")

                plt.figure(r'imp_ene_i2 Dwall_bot_top')
#                plt.semilogy(sDwall_bot_surf,imp_ene_i2_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf,df_imp_ene_i2_surf_plot[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.semilogy(sDwall_top_surf,imp_ene_i2_surf_plot[indsurf_Dwall_top], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf,df_imp_ene_i2_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_i2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_imp_ene_i2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_i2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],df_imp_ene_i2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                
                plt.figure(r'imp_ene_i3 Dwall_bot_top')
#                plt.semilogy(sDwall_bot_surf,imp_ene_i3_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf,df_imp_ene_i3_surf_plot[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.semilogy(sDwall_top_surf,imp_ene_i3_surf_plot[indsurf_Dwall_top], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf,df_imp_ene_i3_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_i3_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_imp_ene_i3_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_i3_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],df_imp_ene_i3_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                
                plt.figure(r'imp_ene_i4 Dwall_bot_top')
#                plt.semilogy(sDwall_bot_surf,imp_ene_i4_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf,df_imp_ene_i4_surf_plot[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.semilogy(sDwall_top_surf,imp_ene_i4_surf_plot[indsurf_Dwall_top], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf,df_imp_ene_i4_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_i4_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_imp_ene_i4_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_i4_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],df_imp_ene_i4_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")

                plt.figure(r'imp_ene_ion Dwall_bot_top')
#                plt.semilogy(sDwall_bot_surf,imp_ene_ion_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf,df_imp_ene_ion_surf_plot[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.semilogy(sDwall_top_surf,imp_ene_ion_surf_plot[indsurf_Dwall_top], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf,df_imp_ene_ion_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_ion_surf_plot_v2[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_imp_ene_ion_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_ion_surf_plot_v2[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],df_imp_ene_ion_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                
                plt.figure(r'imp_ene_n1 Dwall_bot_top')
#                plt.semilogy(sDwall_bot_surf,imp_ene_n1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf,df_imp_ene_n1_surf_plot[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.semilogy(sDwall_top_surf,imp_ene_n1_surf_plot[indsurf_Dwall_top], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf,df_imp_ene_n1_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_n1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_imp_ene_n1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_n1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],df_imp_ene_n1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
            
                plt.figure(r'imp_ene_n2 Dwall_bot_top')
#                plt.semilogy(sDwall_bot_surf,imp_ene_n2_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf,df_imp_ene_n2_surf_plot[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.semilogy(sDwall_top_surf,imp_ene_n2_surf_plot[indsurf_Dwall_top], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf,df_imp_ene_n2_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_n2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_imp_ene_n2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_n2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],df_imp_ene_n2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
            
                plt.figure(r'imp_ene_n3 Dwall_bot_top')
#                plt.semilogy(sDwall_bot_surf,imp_ene_n3_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf,df_imp_ene_n3_surf_plot[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.semilogy(sDwall_top_surf,imp_ene_n3_surf_plot[indsurf_Dwall_top], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf,df_imp_ene_n3_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_n3_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_imp_ene_n3_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_n3_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],df_imp_ene_n3_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
            
                plt.figure(r'imp_ene_neu Dwall_bot_top')
#                plt.semilogy(sDwall_bot_surf,imp_ene_n_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf,df_imp_n_neu_surf_plot[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
#                plt.semilogy(sDwall_top_surf,imp_ene_n_surf_plot[indsurf_Dwall_top], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf,df_imp_ene_n_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_n_surf_plot_v2[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],df_imp_ene_n_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot df")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_n_surf_plot_v2[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],df_imp_ene_n_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top df")
                
            if plot_erosion == 1:
                plt.figure(r'dhdt_i1 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_i1_2D_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_i1_2D_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
            
                plt.figure(r'dhdt_i2 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_i2_2D_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_i2_2D_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
            
                plt.figure(r'dhdt_i3 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_i3_2D_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_i3_2D_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
            
                plt.figure(r'dhdt_i4 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_i4_2D_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_i4_2D_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'dhdt_n1 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_n1_2D_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_n1_2D_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'dhdt_n2 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_n2_2D_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_n2_2D_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
            
                plt.figure(r'dhdt_n3 Dwall_bot_top')
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_n3_2D_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_n3_2D_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
            
                plt.figure(r'dhdt Dwall_bot_top')
#                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_mumh_2D[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_mumh_2D[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label="")
#                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_i1_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot i1")
#                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_i1_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top i1")
#                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_i2_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k]+" bot i2")
#                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_i2_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k]+" top i2")
                
            
                plt.figure(r'h(z) Dwall_bot_top')
#                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],h_mum[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],h_mum[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],h_mum_2D[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],h_mum_2D[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'eroded chamber Dwall_bot_top')
                # Chamber walls before erosion
                np_ptop = 2
                np_pbot = 1
#                legend_label = labels[k]+" $t_\mathrm{op} = 0$"
#                legend_label_2 = labels[k]+" $t_\mathrm{op} = $"+str(t_op)+" h"
                legend_label = labels[k]
                legend_label_2 = labels[k]
                plt.plot(zs[eta_min:eta_max+1,0],rs[eta_min:eta_max+1,0], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=brown, markeredgecolor = 'k', label=legend_label)
                plt.plot(zs[eta_min,0:xi_bottom+1],rs[eta_min,0:xi_bottom+1], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=brown, markeredgecolor = 'k', label="")
                plt.plot(zs[eta_max,0:xi_top+1],rs[eta_max,0:xi_top+1], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=brown, markeredgecolor = 'k', label="")
                plt.plot(zs[eta_max:eta_max+np_ptop,xi_top],rs[eta_max:eta_max+np_ptop,xi_top], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=brown, markeredgecolor = 'k', label="")
                plt.plot(zs[eta_min-np_pbot:eta_min+1,xi_bottom],rs[eta_min-np_pbot:eta_min+1,xi_bottom], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=brown, markeredgecolor = 'k', label="")
#                plt.plot(ze_Dwall_bot,re_Dwall_bot, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=legend_label_2)
#                plt.plot(ze_Dwall_top,re_Dwall_top, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label="")
                plt.plot(ze_Dwall_bot_2D,re_Dwall_bot_2D, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=legend_label_2)
                plt.plot(ze_Dwall_top_2D,re_Dwall_top_2D, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label="")
                
                
        if plot_Awall == 1:
            if plot_curr == 1:
                plt.figure(r'ji_tot_b Awall')

            if plot_imp_ene == 1:
                plt.figure(r'imp_ene_i1 Awall')

                
                plt.figure(r'imp_ene_i2 Awall')

    
                plt.figure(r'imp_ene_ion Awall')

                
                plt.figure(r'imp_ene_n1 Awall')
                
        
        ind = ind + 1
        if ind > 6:
            ind = 0
            ind2 = ind2 + 1
            if ind2 > 6:
                ind = 0
                ind2 = 0
                ind3 = ind3 + 1



    if plot_angle_df == 1:
        plt.figure("angle_df_i1 at points")
        plt.xlabel(r"$\theta$ (deg)",fontsize = font_size)
        if normalized_df == 0:
            plt.title(r"$f_{i1}(\theta)$ (parts/m$^2$-s-deg)",fontsize = font_size)
        elif normalized_df == 1:
            plt.title(r"$f_{i1}(\theta)$ (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = font_size_legend-3,loc=3)      
        
        plt.figure("angle_df_i2 at points")
        plt.xlabel(r"$\theta_{imp}$ (deg)",fontsize = font_size)
        if normalized_df == 0:
            plt.title(r"$f_{i2}(\theta_{imp})$ (parts/m$^2$-s-deg)",fontsize = font_size)
        elif normalized_df == 1:
            plt.title(r"$f_{i2}(\theta_{imp})$ (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = font_size_legend,loc=3)  
        
        plt.figure("angle_df_n1 at points")
        plt.xlabel(r"$\theta_{imp}$ (deg)",fontsize = font_size)
        if normalized_df == 0:
            plt.title(r"$f_{n1}(\theta_{imp})$ (parts/m$^2$-s-deg)",fontsize = font_size)
        elif normalized_df == 1:
            plt.title(r"$f_{n1}(\theta_{imp})$ (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = font_size_legend,loc=2) 
        
    if plot_ene_df == 1:
        plt.figure("ene_df_i1 at points")
        plt.xlabel(r"$E$ (eV)",fontsize = font_size)
        if normalized_df == 0:
            plt.title(r"$f_{i1}(E)$ (parts/m$^2$-s-eV)",fontsize = font_size)
        elif normalized_df == 1:
            plt.title(r"$f_{i1}(E)$ (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = font_size_legend-3,loc=1)      
        
        plt.figure("ene_df_i2 at points")
        plt.xlabel(r"$E_{imp}$ (eV)",fontsize = font_size)
        if normalized_df == 0:
            plt.title(r"$f_{i2}(E_{imp})$ (parts/m$^2$-s-eV)",fontsize = font_size)
        elif normalized_df == 1:
            plt.title(r"$f_{i2}(E_{imp})$ (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = font_size_legend,loc=1)  
        
        plt.figure("ene_df_n1 at points")
        plt.xlabel(r"$E_{imp}$ (eV)",fontsize = font_size)
        if normalized_df == 0:
            plt.title(r"$f_{n1}(E_{imp})$ (parts/m$^2$-s-eV)",fontsize = font_size)
        elif normalized_df == 1:
            plt.title(r"$f_{n1}(E_{imp})$ (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = font_size_legend,loc=1) 
        
    if plot_normv_df == 1:
        plt.figure("normv_df_i1 at points")
        plt.xlabel(r"$v_{n}$ (m/s)",fontsize = font_size)
        if normalized_df == 0:
            plt.title(r"$f_{i1}(v_{n})$ (parts/m$^3$)",fontsize = font_size)
        elif normalized_df == 1:
            plt.title(r"$f_{i1}(v_{n})$ (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = font_size_legend,loc=2)      
        
        plt.figure("normv_df_i2 at points")
        plt.xlabel(r"$v_{n}$ (m/s)",fontsize = font_size)
        if normalized_df == 0:
            plt.title(r"$f_{i2}(v_{n})$ (parts/m$^3$)",fontsize = font_size)
        elif normalized_df == 1:
            plt.title(r"$f_{i2}(v_{n})$ (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = font_size_legend,loc=2)  
        
        plt.figure("normv_df_n1 at points")
        plt.xlabel(r"$v_{n}$ (m/s)",fontsize = font_size)
        if normalized_df == 0:
            plt.title(r"$f_{n1}(v_{n})$ (parts/m$^3$)",fontsize = font_size)
        elif normalized_df == 1:
            plt.title(r"$f_{n1}(v_{n})$ (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = font_size_legend,loc=1) 


    if plot_Dwall == 1:
        if plot_curr == 1:
            plt.figure(r'ji1 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E-7,1E0)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'ji2 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E-7,1E0)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'ji3 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E-10,1E0)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'ji4 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E-10,1E0)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'ji_tot_b Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E-7,1E0)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'egn1 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E-7,1E0)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'egn2 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E-8,1E0)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'egn3 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E-8,1E0)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'egn Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E-7,1E0)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
        if plot_angle == 1:
            plt.figure(r'angle_i1 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(0,65)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'angle_i2 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(0,65)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'angle_i3 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(0,65)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'angle_i4 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(0,65)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'angle_ion Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(0,65)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=4) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'angle_n1 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(0,80)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'angle_n2 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(0,80)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'angle_n3 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(0,80)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'angle_neu Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(0,80)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=4) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
        if plot_imp_ene == 1:
            plt.figure(r'imp_ene_i1 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E0,1E4)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'imp_ene_i2 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E0,1E4)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'imp_ene_i3 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E0,1E4)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'imp_ene_i4 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E0,1E4)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'imp_ene_ion Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E0,1E4)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'imp_ene_n1 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(0.14,0.24)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'imp_ene_n2 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E-1,1E4)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'imp_ene_n3 Dwall_bot_top')
            ax = plt.gca()
            ax.set_ylim(1E0,1E4)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            plt.figure(r'imp_ene_neu Dwall_bot_top')
            ax = plt.gca()
#            ax.set_ylim(0.1,0.25)
#            ax.set_ylim(1E0,1E4)
            ylims = ax.get_ylim()
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
        if plot_erosion == 1:
            plt.figure(r'dhdt Dwall_bot_top')
            ax = plt.gca()
#            ax.set_ylim(1E-8,1E1)
#            ax.set_ylim(1E-7,1E0)
#            ax.set_ylim(1.4075698916637696e-08,3*4.350901430865441)
#            ax.set_ylim(1.0001e-08,3*4.350901430865441)
            ax.set_ylim(1.00001e-08,1E1)
#            ax.set_yticks([1E-8,1E-7,1E-6,1E-5,1E-4,1E-3,1E-2,1E-1,1E0,1E1])
            ylims_dhdt = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=0.8, linestyle=':',color='k')
            plt.legend(fontsize = font_size_legend,loc=10)
            xlims = ax.get_xlim()
#            ax.set_xlim(xlims[0],3.5)
            
            plt.figure(r'dhdt_i1 Dwall_bot_top')
            ax = plt.gca()
#            ax.set_ylim(ylims_dhdt[0],ylims_dhdt[1])
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
#            ax.set_xlim(xlims[0],3.5)
            plt.figure(r'dhdt_i2 Dwall_bot_top')
            ax = plt.gca()
#            ax.set_ylim(ylims_dhdt[0],ylims_dhdt[1])
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
#            ax.set_xlim(xlims[0],3.5)
            plt.figure(r'dhdt_i3 Dwall_bot_top')
            ax = plt.gca()
#            ax.set_ylim(ylims_dhdt[0],ylims_dhdt[1])
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
#            ax.set_xlim(xlims[0],3.5)
            plt.figure(r'dhdt_i4 Dwall_bot_top')
            ax = plt.gca()
#            ax.set_ylim(ylims_dhdt[0],ylims_dhdt[1])
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
#            ax.set_xlim(xlims[0],3.5)
            plt.figure(r'dhdt_n1 Dwall_bot_top')
            ax = plt.gca()
#            ax.set_ylim(ylims_dhdt[0],ylims_dhdt[1])
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
#            ax.set_xlim(xlims[0],3.5)
            plt.figure(r'dhdt_n2 Dwall_bot_top')
            ax = plt.gca()
#            ax.set_ylim(ylims_dhdt[0],ylims_dhdt[1])
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
#            ax.set_xlim(xlims[0],3.5)
            plt.figure(r'dhdt_n3 Dwall_bot_top')
            ax = plt.gca()
#            ax.set_ylim(ylims_dhdt[0],ylims_dhdt[1])
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
#            ax.set_xlim(xlims[0],3.5)

            plt.figure(r'h(z) Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=0.8, linestyle=':',color='k')
#            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
#            ax.set_xlim(xlims[0],3.5)
            plt.figure(r'eroded chamber Dwall_bot_top')
#            plt.legend(fontsize = font_size_legend,loc=10)
            ax = plt.gca()
            ylims = ax.get_ylim()
            xlims = ax.get_xlim()
#            plt.gca().set_aspect('equal', adjustable='box')
#            plt.gca().set_aspect('equal')
#            ax.set_xlim(xlims[0],3.5)
#            ax.set_ylim(ylims[0],9.0)
            ax.set_xlim(xlims[0],zs[0,xi_bottom+1])
            ax.set_ylim(ylims[0],rs[eta_max+1,xi_bottom+1])
#            ax.set_xticks([0.0,0.5,1.0,1.5,2,2.5,3.0,3.5])
#            ax.set_yticks([4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.xaxis.grid(True, which='minor')
            ax.yaxis.grid(True, which='minor')
#            ax.set_xlim(0.2,1.2)
            
            plt.figure(r'points eroded chamber Dwall_bot_top')
            np_ptop = 3
            np_pbot = 2
            npx_bot = 10
            tol_x   = 0.98
            tol_y   = tol_x 
            text_size = 15
#            plt.plot(zs[eta_min:eta_max+1,0],rs[eta_min:eta_max+1,0], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker='s', color='k', markeredgecolor = 'k', label="")
            plt.plot(zs[eta_min,npx_bot:xi_bottom+1],rs[eta_min,npx_bot:xi_bottom+1], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker='s', color='k', markeredgecolor = 'k', label="")
#            plt.plot(zs[eta_max,0:xi_top+1],rs[eta_max,0:xi_top+1], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker='s', color='k', markeredgecolor = 'k', label="")
#            plt.plot(zs[eta_max:eta_max+np_ptop,xi_top],rs[eta_max:eta_max+np_ptop,xi_top], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker='s', color='k', markeredgecolor = 'k', label="")
            plt.plot(zs[eta_min-np_pbot:eta_min+1,xi_bottom],rs[eta_min-np_pbot:eta_min+1,xi_bottom], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker='s', color='k', markeredgecolor = 'k', label="")
            plt.plot(zsurf_Dwall_bot[indsurf_inC_Dwall_bot[12:16]],rsurf_Dwall_bot[indsurf_inC_Dwall_bot[12:16]], linestyle='', linewidth = line_width, markevery=marker_every, markersize=marker_size+2, marker='x', color='r', markeredgecolor = 'r', label="")           
            plt.text(tol_x*zsurf_Dwall_bot[indsurf_inC_Dwall_bot[12]],tol_y*rsurf_Dwall_bot[indsurf_inC_Dwall_bot[12]],"P1",fontsize = text_size,color='k',ha='center',va='center')
            plt.text(tol_x*zsurf_Dwall_bot[indsurf_inC_Dwall_bot[13]],tol_y*rsurf_Dwall_bot[indsurf_inC_Dwall_bot[13]],"P2",fontsize = text_size,color='k',ha='center',va='center')
            plt.text(tol_x*zsurf_Dwall_bot[indsurf_inC_Dwall_bot[14]],tol_y*rsurf_Dwall_bot[indsurf_inC_Dwall_bot[14]],"P3",fontsize = text_size,color='k',ha='center',va='center')
            plt.text(tol_x*zsurf_Dwall_bot[indsurf_inC_Dwall_bot[15]],tol_y*rsurf_Dwall_bot[indsurf_inC_Dwall_bot[15]],"P4",fontsize = text_size,color='k',ha='center',va='center')
            plt.text((2.03-tol_x)*zs[eta_min,npx_bot+1],(2-tol_y)*rs[eta_min,npx_bot+1],"Bottom chamber wall",fontsize = text_size,color='k',ha='center',va='center')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],6.0)
    
    if plot_Awall == 1:
        if plot_curr == 1:
            plt.figure(r'ji_tot_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_imp_ene == 1:
            plt.figure(r'imp_ene_i1 Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_i2 Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_ion Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_n1 Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            
    if save_flag == 1:
        if plot_angle_df == 1:
            plt.figure("angle_df_i1 at points")
            plt.savefig(path_out+"angle_df_i1"+figs_format,bbox_inches='tight')
            plt.close()
            plt.figure("angle_df_i2 at points")
            plt.savefig(path_out+"angle_df_i2"+figs_format,bbox_inches='tight')
            plt.close()
            plt.figure("angle_df_n1 at points")
            plt.savefig(path_out+"angle_df_n1"+figs_format,bbox_inches='tight')
            plt.close()
        if plot_ene_df == 1:
            plt.figure("ene_df_i1 at points")
            plt.savefig(path_out+"ene_df_i1"+figs_format,bbox_inches='tight')
            plt.close()
            plt.figure("ene_df_i2 at points")
            plt.savefig(path_out+"ene_df_i2"+figs_format,bbox_inches='tight')
            plt.close()
            plt.figure("ene_df_n1 at points")
            plt.savefig(path_out+"ene_df_n1"+figs_format,bbox_inches='tight')
            plt.close()
        if plot_normv_df == 1:
            plt.figure("normv_df_i1 at points")
            plt.savefig(path_out+"normv_df_i1"+figs_format,bbox_inches='tight')
            plt.close()
            plt.figure("normv_df_i2 at points")
            plt.savefig(path_out+"normv_df_i2"+figs_format,bbox_inches='tight')
            plt.close()
            plt.figure("normv_df_n1 at points")
            plt.savefig(path_out+"normv_df_n1"+figs_format,bbox_inches='tight')
            plt.close()

        if plot_Dwall == 1:
            if plot_curr == 1:
                plt.figure(r'ji1 Dwall_bot_top')
                plt.savefig(path_out+"df_ji1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji2 Dwall_bot_top')
                plt.savefig(path_out+"df_ji2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji3 Dwall_bot_top')
                plt.savefig(path_out+"df_ji3_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji4 Dwall_bot_top')
                plt.savefig(path_out+"df_ji4_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji_tot_b Dwall_bot_top')
                plt.savefig(path_out+"df_ji_tot_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'egn1 Dwall_bot_top')
                plt.savefig(path_out+"df_egn1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'egn2 Dwall_bot_top')
                plt.savefig(path_out+"df_egn2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'egn3 Dwall_bot_top')
                plt.savefig(path_out+"df_egn3_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'egn Dwall_bot_top')
                plt.savefig(path_out+"df_egn_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
            if plot_angle == 1:
                plt.figure(r'angle_i1 Dwall_bot_top')
                plt.savefig(path_out+"df_angle_i1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'angle_i2 Dwall_bot_top')
                plt.savefig(path_out+"df_angle_i2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'angle_i3 Dwall_bot_top')
                plt.savefig(path_out+"df_angle_i3_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'angle_i4 Dwall_bot_top')
                plt.savefig(path_out+"df_angle_i4_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'angle_ion Dwall_bot_top')
                plt.savefig(path_out+"df_angle_ion_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'angle_n1 Dwall_bot_top')
                plt.savefig(path_out+"df_angle_n1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'angle_n2 Dwall_bot_top')
                plt.savefig(path_out+"df_angle_n2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'angle_n3 Dwall_bot_top')
                plt.savefig(path_out+"df_angle_n3_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'angle_neu Dwall_bot_top')
                plt.savefig(path_out+"df_angle_neu_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
            if plot_imp_ene == 1:
                plt.figure(r'imp_ene_i1 Dwall_bot_top')
                plt.savefig(path_out+"df_imp_ene_i1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_i2 Dwall_bot_top')
                plt.savefig(path_out+"df_imp_ene_i2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_i3 Dwall_bot_top')
                plt.savefig(path_out+"df_imp_ene_i3_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_i4 Dwall_bot_top')
                plt.savefig(path_out+"df_imp_ene_i4_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_ion Dwall_bot_top')
                plt.savefig(path_out+"df_imp_ene_ion_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_n1 Dwall_bot_top')
                plt.savefig(path_out+"df_imp_ene_n1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_n2 Dwall_bot_top')
                plt.savefig(path_out+"df_imp_ene_n2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_n3 Dwall_bot_top')
                plt.savefig(path_out+"df_imp_ene_n3_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_neu Dwall_bot_top')
                plt.savefig(path_out+"df_imp_ene_neu_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
            if plot_erosion == 1:
                plt.figure(r'dhdt_i1 Dwall_bot_top')
                plt.savefig(path_out+"dhdt_i1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'dhdt_i2 Dwall_bot_top')
                plt.savefig(path_out+"dhdt_i2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'dhdt_i3 Dwall_bot_top')
                plt.savefig(path_out+"dhdt_i3_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'dhdt_i4 Dwall_bot_top')
                plt.savefig(path_out+"dhdt_i4_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'dhdt_n1 Dwall_bot_top')
                plt.savefig(path_out+"dhdt_n1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'dhdt_n2 Dwall_bot_top')
                plt.savefig(path_out+"dhdt_n2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'dhdt_n3 Dwall_bot_top')
                plt.savefig(path_out+"dhdt_n3_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'dhdt Dwall_bot_top')
                plt.savefig(path_out+"dhdt_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'h(z) Dwall_bot_top')
                plt.savefig(path_out+"hzerosion_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'eroded chamber Dwall_bot_top')
                plt.savefig(path_out+"eroded_chamber_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'points eroded chamber Dwall_bot_top')
                plt.savefig(path_out+"points_eroded_chamber_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                
        if plot_Awall == 1:
            if plot_curr == 1:
                plt.figure(r'ji_tot_b Awall')
                plt.savefig(path_out+"df_ji_tot_b_Awall"+figs_format,bbox_inches='tight')
                plt.close()
            if plot_imp_ene == 1:
                plt.figure(r'imp_ene_i1 Awall')
                plt.savefig(path_out+"df_imp_ene_i1_Awall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_i2 Awall')
                plt.savefig(path_out+"df_imp_ene_i2_Awall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_ion Awall')
                plt.savefig(path_out+"df_imp_ene_ion_Awall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_n1 Awall')
                plt.savefig(path_out+"df_imp_ene_n1_Awall"+figs_format,bbox_inches='tight')
                plt.close()
                
