#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:30:07 2023

@author: adrian
"""


# Clear all variables
#from IPython import get_ipython
#get_ipython().magic('reset -sf')

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
from contour_2D import contour_2D
from streamlines_function import streamplot, streamline_2D
from HET_sims_read import HET_sims_read
from HET_sims_mean import HET_sims_mean
from HET_sims_plotvars import HET_sims_plotvars
from HET_sims_plotvars import HET_sims_cp_vars
from HET_sims_post import max_min_mean_vals, comp_phase_shift, comp_FFT, comp_Boltz, domain_average
from FFT import FFT
from find_firstmax import find_firstmax
from scipy.signal import correlate
import pylab
import scipy.io as sio
import pickle
import pandas as pd


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

save_flag = 0

    
# Plots save flag
#figs_format = ".eps"
figs_format = ".png"
#figs_format = ".pdf"

# Plots to produce
time_plots          = 1

path_out = "CHALMERS_CHEOPS_LP_MP_data/"


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



# Physical constants
# Van der Walls radius for Xe
r_Xe = 216e-12
e    = 1.6021766E-19
me   = 9.1093829E-31
g0   = 9.80665
eps0 = 8.854188e-12   # Vacuum permitivity [A2s4kg−1m−3] in SI base units,
                      # or [C2N−1m−2] or [CV−1m−1] using other SI coherent units

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

def plot_MFAM_ax_nosig(ax,faces,nodes,line_width):
    nfaces = np.shape(faces)[1]
    for i in range(0,nfaces):
        if faces[2,i] == 1:     # face type >> lambda = const. (cyan)
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'k-',linewidth = line_width)
            
def plot_MFAM_ax_nosig_lambda(ax,faces,nodes,line_width,face_geom):
    nfaces = np.shape(faces)[1]
    count = 0
    count_max = 20
    for i in range(0,nfaces):
        if faces[2,i] == 1:     # face type >> lambda = const. (cyan)
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'k-',linewidth = line_width)
            count = count + 1
            if count == count_max:
                ax.text(np.mean(nodes[0,faces[0:2,i]-1]),np.mean(nodes[1,faces[0:2,i]-1]),str(face_geom[2,i]))
                count = 0
    return
###############################################################################


# Create the dictionary
dic_data = {
        'Vd (V)':           [],
        'mA (mg/s)':        [],
        'mC (mg/s)':        [],
        'Id (A)':           [],
        'fd_max (kHz)':     [],
        'fd_2 (kHz)':       [],
        'fd_3 (kHz)':       [],
        'fd_4 (kHz)':       [],
        'fd_5 (kHz)':       [],
        'fd_6 (kHz)':       [],
        'Id half amp. (%)': [],
        'F (mN)':           [],
        'Isp (s)':          [],     
        'eta':              [], 
        'eta_ene':          [], 
        'eta_div':          [], 
        'eta_disp':         [], 
        'eta_cur':          [],
        'eta_vol':          [],
        'eta_u':            [],
        'eta_ch':           [],
        'eta_prod':         [],
        'P (W)':            [],
        'P_A/P':            [],
        'P_D/P':            [],
        'Iprod (A)':        [], 
        'I_A/Iprod':        [],
        'I_D/I_prod':       []
        }


if time_plots == 1:
    print("######## CHALMERS data time_plots ########")
    
    marker_size = 4
#    marker_size_ID = marker_size + 2
#    marker_size = 0
    marker_size_ID = 10
    
    font_size_legend = 15
    font_size_legend = 8
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'k',alpha=1) 
    
    timestep        = 0
    allsteps_flag   = 1
    read_inst_data  = 1
    read_part_lists = 0
#    read_flag       = 0
    read_flag       = 1
    
    ###########################################################################
    # NOTE: only mean_type = 2 must be used since average values and FFTs are 
#           computed considering, for each signal, a time interval containing
#           an integer number of cycles. The computations for mean_type = 0
#           are wrong since the average and FFTs are not computed in intervals
#           containing an integer number of periods for each signal. The same 
#           happens with mean_type = 1. The approach mean_type = 2 should be 
#           extended in the future for considering partial time series between
#           two given steps (step_i,step_f) and (step_i_fast,step_f_fast) 
#           (as done in mean_type = 1). Up to now is only available for a
#           partial time series containing the last number of
#           steps given in last_steps and last_steps_fast. In the approach
#           mean_type = 2,  the implemented functions max_min_mean_vals,
#           comp_phase_shift and comp_FFT consider a time interval containing
#           an integer number of cycles. This interval is contained within the
#           given time serie of the signal (as mentioned above, currently only  
#           the series containing the last number of steps in last_steps or
#           last_steps_fast are available).     
    ###########################################################################
    make_mean       = 1  # 0 Do not compute nor print mean values; 1 Compute and print mean values
    mean_type       = 2
    order           = 50
    order_fast      = 500
    

#    last_steps      = 700
#    last_steps      = 670
#    last_steps      = 600
#    last_steps      = 1670
    last_steps      = 1000
    last_steps      = 600
    last_steps      = 800
    step_i          = 350
    step_f          = 1000
    step_i          = 400
    step_f          = 1000
    last_steps_fast = 33500
#    last_steps_fast = 60000
#    last_steps_fast = 16750 # For 2dt
#    last_steps_fast = 10000 # For 5dt
    step_i_fast     = int(step_i*50)
    step_f_fast     = int(step_f*50)
    print_mean_vars = 1
    num_firstmax       = 20 # Number of first maximum values searched for in the FFTs of Id
    num_firstmax_print = 5 # Number of first maximum values above to be printed
    
    # Define PIC mesh (i,j) indeces for the point at which we want to plot Te and phi at the free loss boundary
    # Cheops 1
#    i_plot_ver = 17
#    j_plot_ver = 42
#    i_plot_lat = 44
#    j_plot_lat = 32
    # Cheops LP
#    i_plot_ver = 15
#    j_plot_ver = 44
#    i_plot_lat = 30
#    j_plot_lat = 36
    # VHT_US MP
    i_plot_ver = 15
    j_plot_ver = -1
    i_plot_lat = 38
    j_plot_lat = 29
    # VHT_US_LP
    i_plot_ver = 15
    j_plot_ver = -1
    i_plot_lat = 35
    j_plot_lat = 30
    
    i_plot_ver = 20
    j_plot_ver = 40
    i_plot_lat = 37
    j_plot_lat = -1
    
    plot_mass = 0
    plot_dens = 0
    plot_Nmp  = 0
    plot_eff  = 0
    plot_thr  = 0
    plot_Te   = 0
    plot_Id   = 0
    plot_Vd   = 0
    plot_Pd   = 0
    plot_cath = 0
    plot_mbal = 0
    plot_Pbal = 0
    plot_Pbal_inistep = 300
    plot_FLvars = 0
    
    Nke_effects = 0
    
    time2steps_axis  = 0
    prntstepsID_axis = 0 
    
    prntstep_IDs         = [996, 1010, 1050, 1095]
    fast_prntstep_IDs    = [50*996, 50*1010, 50*1050, 50*1095]
    prntstep_IDs_text    = [r"A",r"B",r"C",r"D"]
    prntstep_IDs_colors  = ['r','g','b','m']
    prntstep_IDs_markers = ['^','>','v','<']
    plot_tol = 3
    fact_x = np.array([0.97,1.03,1.0,0.98])
    fact_y = np.array([1.05,0.99,0.60,1.01])
    
    prntstep_IDs = []
    prntstep_IDs_text = []
    fast_prntstep_IDs = []
    
    #### READ EXPERIMENTAL DATA IF REQUIRED
    exp_data_time_plots = 0
#    exp_datafile_name   = "TOPO1_n1_UC3M.CSV"
#    exp_datafile_name   = "TOPO1_n2_UC3M.CSV"
#    exp_datafile_name   = "TOPO2_n3_UC3M.CSV"
    exp_datafile_name   = "TOPO2_n4_UC3M.CSV"
    exp_data = np.genfromtxt(exp_datafile_name, delimiter='\t')
    exp_time = exp_data[:,0]
    exp_Vd   = exp_data[:,1]  # Vd experimental values
    exp_Id   = exp_data[:,4]  # Id experimental values
    exp_Pd   = exp_Id*exp_Vd
    
    exp_nsteps     = len(exp_time)
    exp_last_steps = exp_nsteps
    exp_order      = 20
    
    ########################################
        
    # Simulation names
    nsims = 16
    
    oldpost_sim      = np.array([6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],dtype = int)
    oldsimparams_sim = np.array([18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18],dtype = int)  
    
    


    
    sim_names = [
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume10_OP3_global_CEX_Np",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume20_OP3_global_CEX_Np",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume30_OP3_global_CEX_Np",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume40_OP3_global_CEX_Np",
                
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume10_OP3_local_CEX_Np",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume20_OP3_local_CEX_Np",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume30_OP3_local_CEX_Np",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume40_OP3_local_CEX_Np",
    
#                "../../../Mg_hyphen_alejandro/sim/sims/VLP_300_35_Kr",
#                "../../../Mg_hyphen_alejandro/sim/sims/VLP_300_35_VDF",
                
#                "../../../Mg_hyphen_alejandro/sim/sims/VLP_9L_refined",
#                "../../../Mg_hyphen_alejandro/sim/sims/Neutral_injection_VLP",
                  
#                "../../../sim/sims/CHEOPS_LP_OP1",
#                "../../../Mg_hyphen_alejandro/sim/sims/VLP_300_25",
                
#                "../../../Mg_hyphen_alejandro/sim/sims/VLP_300_25",
#                 "../../../Mg_hyphen_alejandro/sim/sims/VLP_6L_CEX",
    
#                 # CHEOPS LP CHALMERS sims Xe
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_250_20",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_250_25",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_250_30",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_250_35",
#                 
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_300_20",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_300_25",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_300_30",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_300_35",
#                 
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_350_20",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_350_25",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_350_30",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_350_35",
#                 
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_400_20",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_400_25",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_400_30",
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_400_35",
                 
                 
                 # CHEOPS LP CHALMERS sims Kr
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_250_20_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_250_25_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_250_30_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_250_35_Kr",
                 
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_300_20_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_300_25_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_300_30_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_300_35_Kr",
                 
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_350_20_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_350_25_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_350_30_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_350_35_Kr",
                 
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_400_20_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_400_25_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_400_30_Kr",
                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/VLP_400_35_Kr",
                 
                
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
    elif topo_case == 3:
        PIC_mesh_file_name = [
                              "PIC_mesh_LP.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                             ]
    elif topo_case == 0:    
        PIC_mesh_file_name = [
#                              "PIC_mesh_MP.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "SPT100_picM_Reference1500points_rm2.hdf5",
                              "SPT100_picM_Reference1500points_rm2.hdf5",
                              "SPT100_picM_Reference1500points_rm2.hdf5",
                              "SPT100_picM_Reference1500points_rm2.hdf5",
                              "SPT100_picM_Reference1500points_rm2.hdf5",
                              "SPT100_picM.hdf5",
                              "SPT100_picM.hdf5",
                              "SPT100_picM.hdf5",
                              "SPT100_picM.hdf5",
                              "SPT100_picM_Reference1500points_rm.hdf5",
                              "SPT100_picM_Reference1500points_rm.hdf5",
                              "SPT100_picM_Reference1500points_rm.hdf5",
                              "SPT100_picM.hdf5",
                              "SPT100_picM.hdf5"
                              ]

    # Labels  

    labels = [ 
            
#               r"PPSX00 OP1 Xe",
#               r"VHT LP OP1 Xe 6L",
               
               r"No CEX",
               r"CEX",
               
               r"$\dot{m} = 0$ mg/s",
               r"$\dot{m} = 0.25$ mg/s",
               
               
               r"",
               r"",
               r"",
               r"",
               r"",
               r"",
               

               
               r"Local zero current",
               r"Global DML",

              ]
              

    # Titles for the reference case and S case
    titles_Id_fig = [r"(a) $I_d$ (A)",
                     r"(e) $I_d$ instantaneous (A)",
                     r"(b) $I_d$ normalized amplitude (-)",
                     r"(e) Instantaneous $I_d$ normalized amplitude (-)",
                     r"(g) $I_{i \infty}$ (A)",
                     r"(h) $I_{i \infty}$ normalized amplitude (-)",
                     r"(c) $I_d$, $I_{i \infty}$ (A)",
                     r"(d) $I_d$, $I_{i \infty}$ normalized amplitude (-)"]
                     
    titles_dens_fig = [r"(c) $\bar{n}_e$ (m$^{-3}$)",
                       r"(a) $\bar{n}_{n}$ (m$^{-3}$)",
                       r"(d) $\bar{n}_e$ normalized amplitude (-)",
                       r"(b) $\bar{n}_n$ normalized amplitude (-)",
                       r"(a) $\bar{n}_e$, $\bar{n}_n$ (m$^{-3}$)",
                       r"(b) $\bar{n}_e$, $\bar{n}_n$ normalized amplitude (-)"]
                       
    titles_eff_fig  = [r"(e) $\eta_{u}$ (-)",
                       r"(f) $\eta_{thr}$ (-)"]
                       
                       
    # Titles for the cathode cases
    titles_Id_fig = [r"$I_d$ (A)",
                     r"$I_d$ instantaneous (A)",
                     r"$I_d$ normalized amplitude (-)",
                     r"Instantaneous $I_d$ normalized amplitude (-)",
                     r"$I_{i \infty}$ (A)",
                     r"$I_{i \infty}$ normalized amplitude (-)",
                     r"$I_d$, $I_{i \infty}$ (A)",
                     r"$I_d$, $I_{i \infty}$ normalized amplitude (-)",
                     r"$I_{cat}$ (A)",
                     r"$I_{cat}$ normalized amplitude (-)",
                     r"$I_{cond}$ (A)",
                     r"$I_{cond}$ normalized amplitude (-)",
                     r"$\epsilon_{I}$ (-)",
                     r"$I_{cond}$ + $I_d$ (A)",
                     ]
                     
    titles_dens_fig = [r"$\bar{n}_e$ (m$^{-3}$)",
                       r"$\bar{n}_{n}$ (m$^{-3}$)",
                       r"$\bar{n}_e$ normalized amplitude (-)",
                       r"$\bar{n}_n$ normalized amplitude (-)",
                       r"$\bar{n}_e$, $\bar{n}_n$ (m$^{-3}$)",
                       r"$\bar{n}_e$, $\bar{n}_n$ normalized amplitude (-)"]

    # Line colors
#    colors = ['r','g','b','k','c','m','y',orange,brown]
    colors = ['k','r','g','b','m','c','m','y',orange,brown]
    colors = ['m','k','b','r','m','c','m','y',orange,brown]
    colors = ['k','b','g','b','m','c','m','y',orange,brown]
#    colors = ['k','m',orange,brown]
    # Markers
    markers = ['s','o','v', '^', '<', '>','*']
#    markers = ['s','','D','p']
    markers = ['','','','','','','']
#    markers = ['s','o','v','^','<', '>','D','p','*']
    # Line style
    linestyles = ['-','--','-.', ':','-','--','-.']
#    linestyles = ['-','-','--','-.', ':','-','--','-.']
    linestyles = ['-','-','-','-','-','-','-']
    linestyles = ['--','-','-.',':','-','-','-']
    
    dashList = [(None,None),(None,None),(12,6,12,6,3,6),(12,6,3,6,3,6),(5,2,20,2)] 
              
              
    if plot_mass == 1:
        # Plot the time evolution of the ions 1 mass
        plt.figure(r'mi1(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(r"(a) $m_{i1}$ ($10^{-11}$ kg)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the ions 2 mass
        plt.figure(r'mi2(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(r"(b) $m_{i2}$ ($10^{-11}$ kg)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the total ion mass
        plt.figure(r'mitot(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(r"(b) $m_{i} = m_{i1} + m_{i2}$ ($10^{-11}$ kg)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the neutral mass
        plt.figure(r'mn(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(r"(b) $m_{n}$ ($10^{-11}$ kg)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
    if plot_dens == 1:
        # Plot the time evolution of the average plasma density in the domain
        plt.figure(r'dens_e(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(titles_dens_fig[0],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the average neutral density in the domain
        plt.figure(r'dens_n(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(titles_dens_fig[1],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the average plasma density in the domain
        plt.figure(r'FFT dens_e(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(titles_dens_fig[2],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the average neutral density in the domain
        plt.figure(r'FFT dens_n(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(titles_dens_fig[3],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of both the average plasma and neutral density in the domain
        plt.figure(r'dens_e_dens_n(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(titles_dens_fig[4],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for both the average plasma and neutral density in the domain
        plt.figure(r'FFT dens_e_dens_n(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(titles_dens_fig[5],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of both the average plasma and neutral density in the domain (normalized)
        plt.figure(r'norm_dens_e_dens_n(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(titles_dens_fig[4]+" (normalized)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
    if plot_Nmp == 1:
        # Plot the time evolution of the ions 1 number of particles
        plt.figure(r'Nmpi1(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(r"(a) $N_{mp,i1}$ ($10^{6}$ -)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the ions 2 number of particles
        plt.figure(r'Nmpi2(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(r"(b) $N_{mp,i2}$ ($10^{6}$ -)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the total ion number of particles
        plt.figure(r'Nmpitot(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(r"(b) $N_{mp,i} = N_{mp,i1} + N_{mp,i2}$ ($10^{6}$ -)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the neutral number of particles
        plt.figure(r'Nmpn(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        plt.title(r"(b) $N_{mp,n}$ ($10^{6}$ -)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
    if plot_eff == 1:
        # Plot the time evolution of the utilization efficiency
        plt.figure(r'eta_u(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_eff_fig[0],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the production efficiency
        plt.figure(r'eta_prod(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(e) $\eta_{prod}$ (-)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the current efficiency
        plt.figure(r'eta_cur(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(e) $\eta_{cur}$ (-)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the divergence efficiency
        plt.figure(r'eta_div(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(e) $\eta_{div}$ (-)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the thrust efficiency
        plt.figure(r'eta_thr(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_eff_fig[1],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
    if plot_thr == 1:
        # Plot the time evolution of the total thrust
        plt.figure(r'T(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(h) $F$ (mN)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the ions 1 thrust
        plt.figure(r'Ti1(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(e) $F_{i1}$ (mN)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the ions 2 thrust
        plt.figure(r'Ti2(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(e) $F_{i2}$ (mN)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the neutrals 1 thrust
        plt.figure(r'Tn(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(e) $F_{n}$ (mN)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
    if plot_Te == 1:
        # Plot the time evolution of the average Te in the domain
        plt.figure(r'Te(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(e) $T_e$ (eV)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the average Te in the domain
        plt.figure(r'FFT Te(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(r"(b) $T_e$ normalized amplitude (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of both the normalized Id and Te_mean_dom            
        plt.figure(r'Te_Id(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(e) $T_e/\bar{T}_e$, $I_d/\bar{I}_d$ (-)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
    if plot_Id == 1:
        # Plot the time evolution of the discharge current
        plt.figure(r'Id(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_Id_fig[0],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the instantaneous discharge current
        plt.figure(r'Id_inst(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_Id_fig[1],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the discharge current
        plt.figure(r'FFT Id(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(titles_Id_fig[2],fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the instantaneous discharge current
        plt.figure(r'FFT Id_inst(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(titles_Id_fig[3],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the ion beam current
        plt.figure(r'I_beam(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_Id_fig[4],fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the ion beam current
        plt.figure(r'FFT I_beam(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(titles_Id_fig[5],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of both the discharge and the beam current
        plt.figure(r'Id_Ibeam(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_Id_fig[6],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for both the discharge and the ion beam current
        plt.figure(r'FFT Id_Ibeam(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_Id_fig[7],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of both the discharge and the beam current (normalized)
        plt.figure(r'norm_Id_Ibeam(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_Id_fig[6]+" (normalized)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the cathode current
        plt.figure(r'Icath(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_Id_fig[8],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the cathode current            
        plt.figure(r'FFT Icath(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(titles_Id_fig[9],fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the conducting wall current
        plt.figure(r'Icond(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_Id_fig[10],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size) 
        # Plot the FFT plot for the conducting wall current            
        plt.figure(r'FFT Icond(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(titles_Id_fig[11],fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the error in currents of the external circuit
        plt.figure(r'err_I(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_Id_fig[12],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the conducting wall current plus the discharge current
        plt.figure(r'Icond+Id(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(titles_Id_fig[13],fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size) 

    if plot_Vd == 1:
        # Plot the time evolution of the discharge voltage
        plt.figure(r'Vd(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$V_d$ (V)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the conductig walls voltage
        plt.figure(r'Vcond(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$V_{cond}$ (V)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the conducting wall voltage
        plt.figure(r'FFT Vcond(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(r"$V_{cond}$ normalized amplitude (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
    if plot_Pd == 1:
        # Plot the time evolution of the input power
        plt.figure(r'Pd(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_d$ (kW)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the power deposited to material (dielectric) walls
        plt.figure(r'P_mat(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_{W,D}$ (W)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the power deposited to the injection (anode) wall
        plt.figure(r'P_inj(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_{W,A}$ (W)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the power deposited to the free loss wall
        plt.figure(r'P_inf(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_{\infty}$ (W)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the power spent in ionization
        plt.figure(r'P_ion(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_{ion}$ (W)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the power spent in excitation
        plt.figure(r'P_ex(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_{ex}$ (W)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the total ion and neutral power deposited to the free loss wall
        plt.figure(r'P_use_tot ion plus neu (t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_{\infty,hs}$ (W)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the axial ion and neutral power deposited to the free loss wall
        plt.figure(r'P_use_z ion plus neu (t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_{z,\infty,hs}$ (W)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the power deposited to material (dielectric) walls by the heavy species
        plt.figure(r'P_mat_hs(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_{W,D,hs}$ (W)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the power deposited to the injection (anode) wall by the heavy species
        plt.figure(r'P_inj_hs(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_{W,A,hs}$ (W)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the power deposited to the injection (anode) wall faces by the electrons
        plt.figure(r'P_inj_faces_e(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_{W,A,faces,e}$ (W)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
    if plot_cath == 1:
        # Plot the time evolution of the cathode equivalent emission frequency
        plt.figure(r'nu_cat(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$\nu_{cat}$ (Hz)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the cathode equivalent emission frequency
        plt.figure(r'FFT nu_cat(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(r'$\nu_{cat}$ normalized amplitude (-)',fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the cathode emission power
        plt.figure(r'P_cat(t)')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$P_{cat}$ (W)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the cathode emission power
        plt.figure(r'FFT P_cat(t)')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(r'$P_{cat}$ normalized amplitude (-)',fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
    if plot_mbal == 1:
        # Plot the time evolution of the neutrals 1 mass balance
        plt.figure("n1 mass bal")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(f) $dM_{n1}/dt$ (mgs$^{-1}$)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the ions 1 mass balance
        plt.figure("i1 mass bal")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(f) $dM_{i1}/dt$ (mgs$^{-1}$)",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the ions 2 mass balance
        plt.figure("i2 mass bal")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(f) $dM_{i2}/dt$ (mgs$^{-1}$)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the total mass balance
        plt.figure("Total mass bal")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(f) $dM_{tot}/dt$ (mgs$^{-1}$)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the neutrals 1 mass balance error 
        plt.figure("err n1 mass bal")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(f) $\epsilon_{M,n1}$ (-)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the ions 1 mass balance error 
        plt.figure("err i1 mass bal")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(f) $\epsilon_{M,i1}$ (-)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the ions 2 mass balance error 
        plt.figure("err i2 mass bal")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(f) $\epsilon_{M,i2}$ (-)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the total mass balance error 
        plt.figure("err total mass bal")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(f) $\epsilon_{M}$ (-)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        if nsims == 1:
            # Plot the time evolution of the species and the total mass balance
            plt.figure("All mass bal")
            plt.xlabel(r'$t$ (ms)', fontsize = font_size)
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
            elif time2steps_axis == 1 and prntstepsID_axis == 1:
                plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
            plt.title(r"(a) Heavy species mass balances (mgs$^{-1}$)", fontsize = font_size,y=1.02)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            # Plot the time evolution of the species and the total mass balance errors
            plt.figure("All err mass bal")
            plt.xlabel(r'$t$ (ms)', fontsize = font_size)
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
            elif time2steps_axis == 1 and prntstepsID_axis == 1:
                plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
            plt.title(r"(c) Heavy species mass balances errors (-)", fontsize = font_size,y=1.02)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            # Plot the time evolution of the contributions to the total mass balance
            plt.figure("Contributions on total mass bal")
            plt.xlabel(r'$t$ (ms)', fontsize = font_size)
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
            elif time2steps_axis == 1 and prntstepsID_axis == 1:
                plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
            plt.title(r"(e) Heavy species total mass balance contr. (-)", fontsize = font_size,y=1.02)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            # Plot the time evolution of the contributions to the n1 mass balance
            plt.figure("Contributions on n1 mass bal")
            plt.xlabel(r'$t$ (ms)', fontsize = font_size)
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
            elif time2steps_axis == 1 and prntstepsID_axis == 1:
                plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
            plt.title(r"$\varepsilon_{M,n1}^{coll}$, $\varepsilon_{M,n1}^{tw}$, $\varepsilon_{M,n1}^{fw}$ (-)", fontsize = font_size,y=1.02)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            # Plot the time evolution of the contributions to the i1 mass balance
            plt.figure("Contributions on i1 mass bal")
            plt.xlabel(r'$t$ (ms)', fontsize = font_size)
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
            elif time2steps_axis == 1 and prntstepsID_axis == 1:
                plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
            plt.title(r"$\varepsilon_{M,i1}^{coll}$, $\varepsilon_{M,i1}^{tw}$ (-)", fontsize = font_size,y=1.02)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            # Plot the time evolution of the contributions to the i2 mass balance
            plt.figure("Contributions on i2 mass bal")
            plt.xlabel(r'$t$ (ms)', fontsize = font_size)
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
            elif time2steps_axis == 1 and prntstepsID_axis == 1:
                plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
            plt.title(r"$\varepsilon_{M,i2}^{coll}$, $\varepsilon_{M,i2}^{tw}$ (-)", fontsize = font_size,y=1.02)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)     
    if plot_Pbal == 1:
        # Plot the time evolution of the total energy balance
        plt.figure("P balance")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(b) (kW)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the total energy balance error
        plt.figure("P balance error")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(d) $\epsilon_E$ (-)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the contributions to the total energy balance
        plt.figure("Contributions on P balance")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"(f) Total energy balance contr. (-)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)   
    if plot_FLvars == 1:
        # Plot the time evolution of the phi infinity at free loss
        plt.figure("phi_inf FL")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$\phi_{\infty}$ (V)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the phi infinity at free loss
        plt.figure(r'FFT phi_inf')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(r"$\phi_\infty$ normalized amplitude (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the I infinity at free loss
        plt.figure("I_inf FL")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$I_{\infty}$ (A)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the I infinity at free loss
        plt.figure(r'FFT I_inf')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(r"$I_\infty$ normalized amplitude (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the sum of Id and I infinity at free loss
        plt.figure("I_inf+Id FL")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$I_{\infty}$ + $I_d$ (A)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the current balance error
        plt.figure(r'err_I_inf FL')
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$\epsilon_{I_\infty}$ (-)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the electron temperatures at free loss
        plt.figure("Te FL")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$T_e$ (eV)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the electron temperatures at vertical free loss
        plt.figure("Te FL ver")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$T_e$ (eV)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the electron temperatures at lateral free loss
        plt.figure("Te FL lat")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$T_e$ (eV)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the electron temperatures at vertical free loss
        plt.figure(r'FFT Te FL ver')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(r"$T_e$ normalized amplitude (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        
        # Plot the time evolution of the electric potential at free loss
        plt.figure("phi FL")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$\phi$ (V)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the electric potential at vertical free loss
        plt.figure("phi FL ver")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$\phi$ (V)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the electric potential at lateral free loss
        plt.figure("phi FL lat")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$\phi$ (V)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the electric potential at vertical free loss
        plt.figure(r'FFT phi FL ver')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(r"$\phi$ normalized amplitude (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        
        # Plot the time evolution of the dphi/Te at free loss
        plt.figure("dphi/Te FL")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$e\Delta\phi_\infty/T_e$ (-)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the dphi/Te at vertical free loss
        plt.figure("dphi/Te FL ver")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$e\Delta\phi_\infty/T_e$ (-)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the time evolution of the dphi/Te at lateral free loss
        plt.figure("dphi/Te FL lat")
        plt.xlabel(r'$t$ (ms)', fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$e\Delta\phi_\infty/T_e$ (-)", fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        # Plot the FFT plot for the dphi/Te at vertical free loss
        plt.figure(r'FFT dphi/Te FL ver')
        plt.xlabel(r"$f$ (Hz)",fontsize = font_size)
        plt.title(r"$e\Delta\phi_\infty/T_e$ normalized amplitude (-)",fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        

    ind = 0
    ind2 = 0
    ind3 = 0
    
    for k in range(0,nsims):
        ind_ini_letter = sim_names[k].rfind('/') + 1
        print("##### CASE "+str(k+1)+": "+sim_names[k][ind_ini_letter::]+" #####")
        print("##### oldsimparams_sim = "+str(oldsimparams_sim[k])+" #####")
        print("##### oldpost_sim      = "+str(oldpost_sim[k])+" #####")
        print("##### last_steps       = "+str(last_steps)+" #####")
        ######################## READ INPUT/OUTPUT FILES ##########################
        # Obtain paths to simulation files
        path_picM         = sim_names[k]+"/SET/inp/"+PIC_mesh_file_name[k]
        path_simstate_inp = sim_names[k]+"/CORE/inp/SimState.hdf5"
        path_simstate_out = sim_names[k]+"/CORE/out/SimState.hdf5"
        path_postdata_out = sim_names[k]+"/CORE/out/PostData.hdf5"
        path_simparams_inp = sim_names[k]+"/CORE/inp/sim_params.inp"
        print("Reading results...")
        [num_ion_spe,num_neu_spe,Z_ion_spe,n_mp_cell_i,n_mp_cell_n,n_mp_cell_i_min,
       n_mp_cell_i_max,n_mp_cell_n_min,n_mp_cell_n_max,min_ion_plasma_density,
       m_A,spec_refl_prob,ene_bal,points,zs,rs,zscells,rscells,dims,
       nodes_flag,cells_flag,cells_vol,volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,
       eta_max,eta_min,xi_top,xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,
       nsteps,nsteps_fast,nsteps_eFld,faces,nodes,elem_n,boundary_f,face_geom,elem_geom,
       n_faces,n_elems,n_faces_boundary,bIDfaces_Dwall,bIDfaces_Awall,
       bIDfaces_FLwall,IDfaces_Dwall,IDfaces_Awall,IDfaces_FLwall,zfaces_Dwall,
       rfaces_Dwall,Afaces_Dwall,zfaces_Awall,rfaces_Awall,Afaces_Awall,
       zfaces_FLwall,rfaces_FLwall,Afaces_FLwall,zfaces_Cwall,rfaces_Cwall,Afaces_Cwall,
       cath_elem,z_cath,r_cath,V_cath,mass,ssIons1,ssIons2,ssNeutrals1,ssNeutrals2,
       n_mp_i1_list,n_mp_i2_list,n_mp_n1_list,n_mp_n2_list,
       alpha_ano,alpha_ano_e,alpha_ano_q,alpha_ine,alpha_ine_q,
       phi,phi_elems,Ez,Er,Efield,Bz,Br,Bfield,Te,Te_elems,je_mag_elems,je_perp_elems,
       je_theta_elems,je_para_elems,cs01,cs02,cs03,cs04,nn1,nn2,nn3,ni1,ni2,ni3,ni4,
       ne,ne_elems,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fn3_x,fn3_y,fn3_z,
       fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,
       un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,
       ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,ui4_z,
       ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,ji3_z,ji4_x,ji4_y,ji4_z,
       je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,
       ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,
       avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
       ion_gen_weights1,ion_gen_weights2,surf_elems,n_imp_elems,imp_elems,
       imp_elems_kbc,imp_elems_MkQ1,imp_elems_Te,imp_elems_dphi_kbc,
       imp_elems_dphi_sh,imp_elems_nQ1,imp_elems_nQ2,imp_elems_ion_flux_in1,
       imp_elems_ion_flux_out1,imp_elems_ion_ene_flux_in1,
       imp_elems_ion_ene_flux_out1,imp_elems_ion_imp_ene1,
       imp_elems_ion_flux_in2,imp_elems_ion_flux_out2,
       imp_elems_ion_ene_flux_in2,imp_elems_ion_ene_flux_out2,
       imp_elems_ion_imp_ene2,imp_elems_neu_flux_in1,imp_elems_neu_flux_out1,
       imp_elems_neu_ene_flux_in1,imp_elems_neu_ene_flux_out1,
       imp_elems_neu_imp_ene1,imp_elems_neu_flux_in2,imp_elems_neu_flux_out2,
       imp_elems_neu_ene_flux_in2,imp_elems_neu_ene_flux_out2,
       imp_elems_neu_imp_ene2,tot_mass_mp_neus,tot_mass_mp_ions,tot_num_mp_neus,
       tot_num_mp_ions,tot_mass_exit_neus,tot_mass_exit_ions,mass_mp_neus,
       mass_mp_ions,num_mp_neus,num_mp_ions,avg_dens_mp_neus,avg_dens_mp_ions,
       eta_u,eta_prod,eta_thr,eta_div,eta_cur,thrust,thrust_ion,thrust_neu,thrust_e,
       Id_inst,Id,Vd_inst,Vd,I_beam,I_tw_tot,Pd,Pd_inst,P_mat,P_inj,P_inf,P_ion,
       P_ex,P_use_tot_i,P_use_tot_n,P_use_tot,P_use_z_i,P_use_z_n,P_use_z_e,P_use_z,
       qe_wall,qe_wall_inst,Pe_faces_Dwall,Pe_faces_Awall,Pe_faces_FLwall,
       Pe_faces_Dwall_inst,Pe_faces_Awall_inst,Pe_faces_FLwall_inst,
       Pe_Dwall,Pe_Awall,Pe_FLwall,Pe_Dwall_inst,Pe_Awall_inst,Pe_FLwall_inst, 
       Pe_Cwall,Pe_Cwall_inst,
       Pi_Dwall,Pi_Awall,Pi_FLwall,Pi_FLwall_nonz,Pi_Cwall,Pn_Dwall,Pn_Awall,Pn_FLwall,
       Pn_FLwall_nonz,Pn_Cwall,P_Dwall,P_Awall,P_FLwall,Pwalls,Pionex,Ploss,Psource,Pthrust,
       Pnothrust,Pnothrust_walls,Pturb,balP,err_balP,ctr_Pd,ctr_Ploss,ctr_Pwalls,
       ctr_Pionex,ctr_P_DAwalls,ctr_P_FLwalls,ctr_P_FLwalls_in,ctr_P_FLwalls_i,
       ctr_P_FLwalls_n,ctr_P_FLwalls_e,balP_Pthrust,err_balP_Pthrust,
       ctr_balPthrust_Pd,ctr_balPthrust_Pnothrust,ctr_balPthrust_Pthrust,
       ctr_balPthrust_Pnothrust_walls,ctr_balPthrust_Pnothrust_ionex,
       err_def_balP,Isp_s,Isp_ms,
       dMdt_i1,dMdt_i2,dMdt_i3,dMdt_i4,dMdt_n1,dMdt_n2,dMdt_n3,dMdt_tot,
       mflow_coll_i1,mflow_coll_i2,mflow_coll_i3,mflow_coll_i4,mflow_coll_n1,
       mflow_coll_n2,mflow_coll_n3,mflow_fw_i1,mflow_fw_i2,mflow_fw_i3,
       mflow_fw_i4,mflow_fw_n1,mflow_fw_n2,mflow_fw_n3,mflow_tw_i1,mflow_tw_i2,
       mflow_tw_i3,mflow_tw_i4,mflow_tw_n1,mflow_tw_n2,mflow_tw_n3,
       mflow_ircmb_picS_n1,mflow_ircmb_picS_n2,mflow_ircmb_picS_n3,
       mflow_inj_i1,mflow_fwinf_i1,mflow_fwmat_i1,mflow_fwcat_i1,
       mflow_inj_i2,mflow_fwinf_i2,mflow_fwmat_i2,mflow_fwcat_i2,
       mflow_inj_i3,mflow_fwinf_i3,mflow_fwmat_i3,mflow_fwcat_i3,
       mflow_inj_i4,mflow_fwinf_i4,mflow_fwmat_i4,mflow_fwcat_i4,
       mflow_inj_n1,mflow_fwinf_n1,mflow_fwmat_n1,mflow_fwcat_n1,
       mflow_inj_n2,mflow_fwinf_n2,mflow_fwmat_n2,mflow_fwcat_n2,
       mflow_inj_n3,mflow_fwinf_n3,mflow_fwmat_n3,mflow_fwcat_n3,
       mflow_twa_i1,mflow_twinf_i1,mflow_twmat_i1,mflow_twcat_i1,
       mflow_twa_i2,mflow_twinf_i2,mflow_twmat_i2,mflow_twcat_i2,
       mflow_twa_i3,mflow_twinf_i3,mflow_twmat_i3,mflow_twcat_i3,
       mflow_twa_i4,mflow_twinf_i4,mflow_twmat_i4,mflow_twcat_i4,
       mflow_twa_n1,mflow_twinf_n1,mflow_twmat_n1,mflow_twcat_n1,
       mflow_twa_n2,mflow_twinf_n2,mflow_twmat_n2,mflow_twcat_n2,
       mflow_twa_n3,mflow_twinf_n3,mflow_twmat_n3,mflow_twcat_n3,
       mbal_n1,mbal_n2,mbal_n3,mbal_i1,mbal_i2,mbal_i3,mbal_i4,mbal_tot,
       err_mbal_n1,err_mbal_n2,err_mbal_n3,err_mbal_i1,err_mbal_i2,
       err_mbal_i3,err_mbal_i4,err_mbal_tot,ctr_mflow_coll_n1,
       ctr_mflow_fw_n1,ctr_mflow_tw_n1,ctr_mflow_coll_i1,ctr_mflow_fw_i1,
       ctr_mflow_tw_i1,ctr_mflow_coll_i2,ctr_mflow_fw_i2,ctr_mflow_tw_i2,
       ctr_mflow_coll_tot,ctr_mflow_fw_tot,ctr_mflow_tw_tot,
       dEdt_i1,dEdt_i2,dEdt_i3,dEdt_i4,dEdt_n1,dEdt_n2,dEdt_n3,
       eneflow_coll_i1,eneflow_coll_i2,eneflow_coll_i3,eneflow_coll_i4,
       eneflow_coll_n1,eneflow_coll_n2,eneflow_coll_n3,eneflow_fw_i1,
       eneflow_fw_i2,eneflow_fw_i3,eneflow_fw_i4,eneflow_fw_n1,eneflow_fw_n2,
       eneflow_fw_n3,eneflow_tw_i1,eneflow_tw_i2,eneflow_tw_i3,eneflow_tw_i4,
       eneflow_tw_n1,eneflow_tw_n2,eneflow_tw_n3,Pfield_i1,Pfield_i2,
       Pfield_i3,Pfield_i4,eneflow_inj_i1,eneflow_fwinf_i1,eneflow_fwmat_i1,
       eneflow_inj_i2,eneflow_fwinf_i2,eneflow_fwmat_i2,
       eneflow_inj_i3,eneflow_fwinf_i3,eneflow_fwmat_i3,
       eneflow_inj_i4,eneflow_fwinf_i4,eneflow_fwmat_i4,
       eneflow_inj_n1,eneflow_fwinf_n1,eneflow_fwmat_n1,
       eneflow_inj_n2,eneflow_fwinf_n2,eneflow_fwmat_n2,
       eneflow_inj_n3,eneflow_fwinf_n3,eneflow_fwmat_n3,
       eneflow_twa_i1,eneflow_twinf_i1,eneflow_twmat_i1,
       eneflow_twa_i2,eneflow_twinf_i2,eneflow_twmat_i2,
       eneflow_twa_i3,eneflow_twinf_i3,eneflow_twmat_i3,
       eneflow_twa_i4,eneflow_twinf_i4,eneflow_twmat_i4,
       eneflow_twa_n1,eneflow_twinf_n1,eneflow_twmat_n1,
       eneflow_twa_n2,eneflow_twinf_n2,eneflow_twmat_n2,
       eneflow_twa_n3,eneflow_twinf_n3,eneflow_twmat_n3,
       ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,
       ndot_ion02_n2,ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,
       ndot_CEX01_i3,ndot_CEX02_i4,
       cath_type,ne_cath,Te_cath,
       nu_cath,ndot_cath,Q_cath,P_cath,V_cath_tot,ne_cath_avg,F_theta,Hall_par,
       Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,
       nu_ex,Boltz,Boltz_dim,Pfield_e,Ebal_e,ge_b,ge_b_acc,ge_sb_b,ge_sb_b_acc,
       delta_see,delta_see_acc,err_interp_n,n_cond_wall,Icond,Vcond,Icath,phi_inf,I_inf,
       f_split,f_split_adv,f_split_qperp,f_split_qpara,f_split_qb,f_split_Pperp,
       f_split_Ppara,f_split_ecterm,f_split_inel] = HET_sims_read(path_simstate_inp,path_simstate_out,
                                                                      path_postdata_out,path_simparams_inp,
                                                                      path_picM,allsteps_flag,timestep,read_inst_data,
                                                                      read_part_lists,read_flag,oldpost_sim[k],oldsimparams_sim[k])
            
        # Set initial time to zero
        time = time - time[0]
        time_fast = time_fast - time_fast[0]
        # Domain averaged (spatially averaged) variables using the nodal weighting volumes
        [Te_mean_dom,_] = domain_average(Te,time,vol)
        
        # Check indeces at free loss boundary and change them if necessary to avoid error (send message if indeces are changed)
        if i_plot_ver != (eta_min+eta_max)/2:
            i_plot_ver_old = i_plot_ver
            i_plot_ver = int((eta_min+eta_max)/2)
            print("WARNING: FL var i_plot_ver = %d changed to i_plot_ver = %d" %(i_plot_ver_old,i_plot_ver))
        if i_plot_ver >= dims[0]:
            i_plot_ver_old = i_plot_ver
            i_plot_ver = dims[0]-1
            print("WARNING: FL var i_plot_ver = %d changed to i_plot_ver = %d" %(i_plot_ver_old,i_plot_ver))
        if j_plot_ver != dims[1]-1 and j_plot_ver !=-1:
            j_plot_ver_old = j_plot_ver
            j_plot_ver = dims[1]-1
            print("WARNING: FL var j_plot_ver = %d changed to j_plot_ver = %d" %(j_plot_ver_old,j_plot_ver))  
        
        if i_plot_lat != dims[0]-1 and i_plot_lat != -1:
            i_plot_lat_old = i_plot_lat
            i_plot_lat = dims[0]-1
            print("WARNING: FL var i_plot_lat = %d changed to i_plot_lat = %d" %(i_plot_lat_old,i_plot_lat))  
        if j_plot_lat >= dims[1] or j_plot_lat <= xi_top:
            j_plot_lat_old = j_plot_lat
            j_plot_lat = int((xi_top + dims[1]-1)/2)
            print("WARNING: FL var j_plot_lat = %d changed to j_plot_lat = %d" %(j_plot_lat_old,j_plot_lat))  
        
        
        # Average electron temperature and potential at free loss boundary
        Te_FL_lat  = np.mean(Te[-1,int(xi_bottom)::,:],axis=0)
        Te_FL_ver  = np.mean(Te[:,-1,:],axis=0)
        Te_FL      = 0.5*(Te_FL_lat+Te_FL_ver)
        phi_FL_lat = np.mean(phi[-1,int(xi_bottom)::,:],axis=0)
        phi_FL_ver = np.mean(phi[:,-1,:],axis=0)
        phi_FL     = 0.5*(phi_FL_lat+phi_FL_ver)
        
        Hall_par_eff_FL_lat = np.mean(Hall_par_eff[-1,int(xi_bottom)::,:],axis=0)
        Hall_par_eff_FL_ver = np.mean(Hall_par_eff[:,-1,:],axis=0)
        Hall_par_eff_FL     = 0.5*(Hall_par_eff_FL_lat+Hall_par_eff_FL_ver)
        Hall_par_FL_lat     = np.mean(Hall_par[-1,int(xi_bottom)::,:],axis=0)
        Hall_par_FL_ver     = np.mean(Hall_par[:,-1,:],axis=0)
        Hall_par_FL         = 0.5*(Hall_par_FL_lat+Hall_par_FL_ver)
        
        Te_FL_lat_int  = np.zeros(np.shape(Te_FL_lat),dtype=float)
        Te_FL_ver_int  = np.zeros(np.shape(Te_FL_ver),dtype=float)
        phi_FL_lat_int = np.zeros(np.shape(phi_FL_lat),dtype=float)
        phi_FL_ver_int = np.zeros(np.shape(phi_FL_ver),dtype=float)
        
        Hall_par_eff_FL_lat_int = np.zeros(np.shape(phi_FL_lat),dtype=float)
        Hall_par_eff_FL_ver_int = np.zeros(np.shape(phi_FL_ver),dtype=float)
        Hall_par_FL_lat_int     = np.zeros(np.shape(phi_FL_lat),dtype=float)
        Hall_par_FL_ver_int     = np.zeros(np.shape(phi_FL_ver),dtype=float)
        
        vec_z2 = zs[0,int(xi_bottom+1)::]
        vec_z1 = zs[0,int(xi_bottom):-1:1]
        vec_r2 = rs[-1,int(xi_bottom+1)::]
        vec_r1 = rs[-1,int(xi_bottom):-1:1]
        surf_lat = np.zeros(len(vec_z2),dtype=float)
        surf_lat_tot = 0.0
        for index_node in range(0,len(surf_lat)):
            Lpanel = np.sqrt( (vec_z2[index_node]-vec_z1[index_node])**2 + (vec_r2[index_node] - vec_r1[index_node])**2 )
            rpanel = 0.5*(vec_r2[index_node]+vec_r1[index_node])
            surf_lat[index_node] = 2.0*np.pi*rpanel*Lpanel
            surf_lat_tot = surf_lat_tot + surf_lat[index_node]
            
        for istep in range(0,nsteps):
#            vec_z = zs[0,int(xi_bottom)::]
#            Te_FL_lat_int[istep] = np.trapz(Te[-1,int(xi_bottom)::,istep],x=vec_z)
#            phi_FL_lat_int[istep] = np.trapz(phi[-1,int(xi_bottom)::,istep],x=vec_z)
#            Hall_par_eff_FL_lat_int[istep] = np.trapz(Hall_par_eff[-1,int(xi_bottom)::,istep],x=vec_z)
#            Hall_par_FL_lat_int[istep] = np.trapz(Hall_par[-1,int(xi_bottom)::,istep],x=vec_z)
            
            Te_FL_lat_int[istep]           = np.dot(0.5*(Te[-1,int(xi_bottom+1)::,istep] + Te[-1,int(xi_bottom):-1:1,istep]),surf_lat)
            phi_FL_lat_int[istep]          = np.dot(0.5*(phi[-1,int(xi_bottom+1)::,istep] + phi[-1,int(xi_bottom):-1:1,istep]),surf_lat)
            Hall_par_eff_FL_lat_int[istep] = np.dot(0.5*(Hall_par_eff[-1,int(xi_bottom+1)::,istep] + Hall_par_eff[-1,int(xi_bottom):-1:1,istep]),surf_lat)
            Hall_par_FL_lat_int[istep]     = np.dot(0.5*(Hall_par[-1,int(xi_bottom+1)::,istep] + Hall_par[-1,int(xi_bottom):-1:1,istep]),surf_lat)

            vec_r = rs[:,-1]
            Te_FL_ver_int[istep]           = 2*np.pi*np.trapz(np.multiply(Te[:,-1,istep],rs[:,-1]),x=vec_r)
            phi_FL_ver_int[istep]          = 2*np.pi*np.trapz(np.multiply(phi[:,-1,istep],rs[:,-1]),x=vec_r)
            Hall_par_eff_FL_ver_int[istep] = 2*np.pi*np.trapz(np.multiply(Hall_par_eff[:,-1,istep],rs[:,-1]),x=vec_r)
            Hall_par_FL_ver_int[istep]     = 2*np.pi*np.trapz(np.multiply(Hall_par[:,-1,istep],rs[:,-1]),x=vec_r)
            
#        Te_FL_lat_int = Te_FL_lat_int/(vec_z[-1] - vec_z[0])
#        phi_FL_lat_int = phi_FL_lat_int/(vec_z[-1] - vec_z[0])
        
        Te_FL_lat_int           = Te_FL_lat_int/surf_lat_tot
        phi_FL_lat_int          = phi_FL_lat_int/surf_lat_tot
        Hall_par_eff_FL_lat_int = Hall_par_eff_FL_lat_int/surf_lat_tot
        Hall_par_FL_lat_int     = Hall_par_FL_lat_int/surf_lat_tot
        
        Te_FL_ver_int           = Te_FL_ver_int/(np.pi*(rs[-1,-1]**2-rs[0,-1]**2))
        phi_FL_ver_int          = phi_FL_ver_int/(np.pi*(rs[-1,-1]**2-rs[0,-1]**2))
        Hall_par_eff_FL_ver_int = Hall_par_eff_FL_ver_int/(np.pi*(rs[-1,-1]**2-rs[0,-1]**2))
        Hall_par_FL_ver_int     = Hall_par_FL_ver_int/(np.pi*(rs[-1,-1]**2-rs[0,-1]**2))
        
#        Te_FL_int = 0.5*(Te_FL_lat_int+Te_FL_ver_int)
#        Te_FL_int = (Te_FL_lat_int*2*np.pi*rs[-1,-1]*(zs[-1,-1]-zs[-1,int(xi_top)]) + Te_FL_ver_int*np.pi*(rs[-1,-1]**2-rs[0,-1]**2))/(2*np.pi*rs[-1,-1]*(zs[-1,-1]-zs[-1,int(xi_top)])+np.pi*(rs[-1,-1]**2-rs[0,-1]**2))
#        phi_FL_int = 0.5*(phi_FL_lat_int+phi_FL_ver_int) 
#        phi_FL_int = (phi_FL_lat_int*2*np.pi*rs[-1,-1]*(zs[-1,-1]-zs[-1,int(xi_top)]) + phi_FL_ver_int*np.pi*(rs[-1,-1]**2-rs[0,-1]**2))/(2*np.pi*rs[-1,-1]*(zs[-1,-1]-zs[-1,int(xi_top)])+np.pi*(rs[-1,-1]**2-rs[0,-1]**2))
        Te_FL_int           = (Te_FL_lat_int*surf_lat_tot + Te_FL_ver_int*np.pi*(rs[-1,-1]**2-rs[0,-1]**2))/(surf_lat_tot+np.pi*(rs[-1,-1]**2-rs[0,-1]**2))
        phi_FL_int          = (phi_FL_lat_int*surf_lat_tot + phi_FL_ver_int*np.pi*(rs[-1,-1]**2-rs[0,-1]**2))/(surf_lat_tot+np.pi*(rs[-1,-1]**2-rs[0,-1]**2))
        Hall_par_eff_FL_int = (Hall_par_eff_FL_lat_int*surf_lat_tot + Hall_par_eff_FL_ver_int*np.pi*(rs[-1,-1]**2-rs[0,-1]**2))/(surf_lat_tot+np.pi*(rs[-1,-1]**2-rs[0,-1]**2))
        Hall_par_FL_int     = (Hall_par_FL_lat_int*surf_lat_tot + Hall_par_FL_ver_int*np.pi*(rs[-1,-1]**2-rs[0,-1]**2))/(surf_lat_tot+np.pi*(rs[-1,-1]**2-rs[0,-1]**2))

        ratio_DphiTe_FL_lat     = (phi_FL_lat - phi_inf)/Te_FL_lat
        ratio_DphiTe_FL_ver     = (phi_FL_ver - phi_inf)/Te_FL_ver
        ratio_DphiTe_FL         = (phi_FL - phi_inf)/Te_FL
        ratio_DphiTe_FL_lat_int = (phi_FL_lat_int - phi_inf)/Te_FL_lat_int
        ratio_DphiTe_FL_ver_int = (phi_FL_ver_int - phi_inf)/Te_FL_ver_int
        ratio_DphiTe_FL_int     = (phi_FL_int - phi_inf)/Te_FL_int
                                                        
        # Obtain the FFT of the discharge current and the beam current
        if make_mean == 1 and mean_type == 0:
            time_vector   = time[nsteps-last_steps::]
            Id_vector     = Id[nsteps-last_steps::]
            I_beam_vector = I_beam[nsteps-last_steps::]
            nsamples = len(time_vector)
            [fft_Id,freq_Id,max_fft_Id,max_freq_Id]                     = FFT(time[1]-time[0],time[nsteps-last_steps::],Id[nsteps-last_steps::])
            [fft_Id_inst,freq_Id_inst,max_fft_Id_inst,max_freq_Id_inst] = FFT(time[1]-time[0],time[nsteps-last_steps::],Id_inst[nsteps-last_steps::])
            [fft_I_beam,freq_I_beam,max_fft_I_beam,max_freq_I_beam]     = FFT(time[1]-time[0],time[nsteps-last_steps::],I_beam[nsteps-last_steps::])
            [fft_avg_dens_mp_ions,freq_avg_dens_mp_ions,
             max_fft_avg_dens_mp_ions,max_freq_avg_dens_mp_ions]        = FFT(time_fast[1]-time_fast[0],time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_ions[nsteps_fast-last_steps_fast::])  
            [fft_avg_dens_mp_neus,freq_avg_dens_mp_neus,
             max_fft_avg_dens_mp_neus,max_freq_avg_dens_mp_neus]        = FFT(time_fast[1]-time_fast[0],time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_neus[nsteps_fast-last_steps_fast::])  
            # Obtain the phase shift of the signals Id and I_beam from the time between max peaks
            [_,_,time_shift_IdIbeam,phase_shift_IdIbeam_deg] = comp_phase_shift(time,Id,I_beam,time[nsteps-last_steps::],Id[nsteps-last_steps::],
                                                                                time[nsteps-last_steps::],I_beam[nsteps-last_steps::],order)        
        elif make_mean == 1 and mean_type == 1:
            time_vector   = time[step_i:step_f+1]
            Id_vector     = Id[step_i:step_f+1]
            I_beam_vector = I_beam[step_i:step_f+1]
            nsamples = len(time_vector)
            if exp_data_time_plots == 1:
                [fft_exp_Id,freq_exp_Id,max_fft_exp_Id,max_freq_exp_Id] = comp_FFT(exp_time,exp_Id,exp_time[exp_nsteps-exp_last_steps::],exp_Id[exp_nsteps-exp_last_steps::],exp_order)
                [maxs_fft_exp_Id,maxs_freq_exp_Id] = find_firstmax(freq_exp_Id[1:],np.abs(fft_exp_Id[1:]),num_firstmax)
            [fft_Te_mean_dom,freq_Te_mean_dom,max_fft_Te_mean_dom,max_freq_Te_mean_dom] = FFT(time[1]-time[0],time[step_i:step_f+1],Te_mean_dom[step_i:step_f+1])
            [fft_Id,freq_Id,max_fft_Id,max_freq_Id]                     = FFT(time[1]-time[0],time[step_i:step_f+1],Id[step_i:step_f+1])
            [maxs_fft_Id,maxs_freq_Id] = find_firstmax(freq_Id[1:],np.abs(fft_Id[1:]),num_firstmax)
            [fft_Id_inst,freq_Id_inst,max_fft_Id_inst,max_freq_Id_inst] = FFT(time[1]-time[0],time[step_i:step_f+1],Id_inst[step_i:step_f+1])
            [fft_I_beam,freq_I_beam,max_fft_I_beam,max_freq_I_beam]     = FFT(time[1]-time[0],time[step_i:step_f+1],I_beam[step_i:step_f+1])
            [fft_avg_dens_mp_ions,freq_avg_dens_mp_ions,
             max_fft_avg_dens_mp_ions,max_freq_avg_dens_mp_ions]        = FFT(time_fast[1]-time_fast[0],time_fast[step_i_fast:step_f_fast+1],avg_dens_mp_ions[step_i_fast:step_f_fast+1])  
            [fft_avg_dens_mp_neus,freq_avg_dens_mp_neus,
             max_fft_avg_dens_mp_neus,max_freq_avg_dens_mp_neus]        = FFT(time_fast[1]-time_fast[0],time_fast[step_i_fast:step_f_fast+1],avg_dens_mp_neus[step_i_fast:step_f_fast+1])  
            [fft_nu_cath,freq_nu_cath,max_fft_nu_cath,max_freq_nu_cath] = FFT(time[1]-time[0],time[step_i:step_f+1],nu_cath[step_i:step_f+1])
            [fft_P_cath,freq_P_cath,max_fft_P_cath,max_freq_P_cath]     = FFT(time[1]-time[0],time[step_i:step_f+1],P_cath[step_i:step_f+1])
            # Obtain the phase shift of the signals Id and Te_mean_dom from the time between max peaks
            [_,_,time_shift_IdTe_mean_dom,phase_shift_IdTe_mean_dom_deg] = comp_phase_shift(time,Id,Te_mean_dom,time[step_i:step_f+1],Id[step_i:step_f+1],
                                                                                            time[step_i:step_f+1],Te_mean_dom[step_i:step_f+1],order)
            # Obtain the phase shift of the signals Id and I_beam from the time between max peaks
            [_,_,time_shift_IdIbeam,phase_shift_IdIbeam_deg] = comp_phase_shift(time,Id,I_beam,time[step_i:step_f+1],Id[step_i:step_f+1],
                                                                                time[step_i:step_f+1],I_beam[step_i:step_f+1],order)  
            # Obtain the phase shift of the signals avg_dens_mp_neus and avg_dens_mp_ions from the time between max peaks
            [_,_,time_shift_avg_dens_mp_neusions,phase_shift_avg_dens_mp_neusions_deg] = comp_phase_shift(time_fast,avg_dens_mp_neus,avg_dens_mp_ions,time_fast[step_i_fast:step_f_fast+1],avg_dens_mp_neus[step_i_fast:step_f_fast+1],
                                                                                                          time_fast[step_i_fast:step_f_fast+1],avg_dens_mp_ions[step_i_fast:step_f_fast+1],order_fast)
            # Obtain the phase shift of the signals representing the contributions to the total heavy species mass balance
            [_,_,time_shift_ctr_mbal_tot,phase_shift_ctr_mbal_tot_deg] = comp_phase_shift(time,ctr_mflow_fw_tot,ctr_mflow_tw_tot,time[step_i:step_f+1],ctr_mflow_fw_tot[step_i:step_f+1],
                                                                                          time[step_i:step_f+1],ctr_mflow_tw_tot[step_i:step_f+1],order)   
        elif make_mean == 1 and mean_type == 2:
            if exp_data_time_plots == 1:
                # Obtain FFT for Id considering an integer number of periods
                [fft_exp_Id,freq_exp_Id,max_fft_exp_Id,max_freq_exp_Id] = comp_FFT(exp_time,exp_Id,exp_time[exp_nsteps-exp_last_steps::],exp_Id[exp_nsteps-exp_last_steps::],exp_order)
                [maxs_fft_exp_Id,maxs_freq_exp_Id] = find_firstmax(freq_exp_Id[1:],np.abs(fft_exp_Id[1:]),num_firstmax)
            # Obtain FFT for Te_mean_dom considering an integer number of periods
            [fft_Te_mean_dom,freq_Te_mean_dom,max_fft_Te_mean_dom,max_freq_Te_mean_dom] = comp_FFT(time,Te_mean_dom,time[nsteps-last_steps::],Te_mean_dom[nsteps-last_steps::],order)
            # Obtain FFT for Id considering an integer number of periods
            [fft_Id,freq_Id,max_fft_Id,max_freq_Id] = comp_FFT(time,Id,time[nsteps-last_steps::],Id[nsteps-last_steps::],order)
            [maxs_fft_Id,maxs_freq_Id] = find_firstmax(freq_Id[1:],np.abs(fft_Id[1:]),num_firstmax)
            # Obtain FFT for Id_inst considering an integer number of periods
            [fft_Id_inst,freq_Id_inst,max_fft_Id_inst,max_freq_Id_inst] = comp_FFT(time,Id_inst,time[nsteps-last_steps::],Id_inst[nsteps-last_steps::],order)
            # Obtain FFT for I_beam considering an integer number of periods
            [fft_I_beam,freq_I_beam,max_fft_I_beam,max_freq_I_beam] = comp_FFT(time,I_beam,time[nsteps-last_steps::],I_beam[nsteps-last_steps::],order)
            # Obtain FFT for avg_dens_mp_ions considering an integer number of periods
            [fft_avg_dens_mp_ions,freq_avg_dens_mp_ions,max_fft_avg_dens_mp_ions,max_freq_avg_dens_mp_ions] = comp_FFT(time_fast,avg_dens_mp_ions,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_ions[nsteps_fast-last_steps_fast::],order_fast)
            # Obtain FFT for avg_dens_mp_neus considering an integer number of periods
            [fft_avg_dens_mp_neus,freq_avg_dens_mp_neus,max_fft_avg_dens_mp_neus,max_freq_avg_dens_mp_neus] = comp_FFT(time_fast,avg_dens_mp_neus,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_neus[nsteps_fast-last_steps_fast::],order_fast)
            
       
            # Obtain FFT for Icond considering an integer number of periods
            if n_cond_wall > 0:
                for i in range(0,n_cond_wall):
                    [fft_Icond,freq_Icond,max_fft_Icond,max_freq_Icond] = comp_FFT(time,Icond[:,i],time[nsteps-last_steps::],Icond[nsteps-last_steps::,i],order)
                    if np.any(Vcond != 0):
                        if Vcond[1,0] != Vcond[2,0]:
                            [fft_Vcond,freq_Vcond,max_fft_Vcond,max_freq_Vcond] = comp_FFT(time,Vcond[:,i],time[nsteps-last_steps::],Vcond[nsteps-last_steps::,i],order)
            if np.any(Icath != 0):
                [fft_Icath,freq_Icath,max_fft_Icath,max_freq_Icath] = comp_FFT(time,Icath,time[nsteps-last_steps::],Icath[nsteps-last_steps::],order)
            
            if cath_type == 2:
                # Obtain FFT for nu_cath considering an integer number of periods
                [fft_nu_cath,freq_nu_cath,max_fft_nu_cath,max_freq_nu_cath] = comp_FFT(time,nu_cath,time[nsteps-last_steps::],nu_cath[nsteps-last_steps::],order)
                # Obtain FFT for P_cath considering an integer number of periods
                [fft_P_cath,freq_P_cath,max_fft_P_cath,max_freq_P_cath] = comp_FFT(time,P_cath,time[nsteps-last_steps::],P_cath[nsteps-last_steps::],order)
            elif cath_type == 1:
                fft_nu_cath      = 0
                freq_nu_cath     = 0
                max_fft_nu_cath  = 0
                max_freq_nu_cath = 0
                # Obtain FFT for P_cath considering an integer number of periods
                [fft_P_cath,freq_P_cath,max_fft_P_cath,max_freq_P_cath] = comp_FFT(time,P_cath,time[nsteps-last_steps::],P_cath[nsteps-last_steps::],order)
            
            if np.any(np.diff(phi_inf != 0)):
                # Obtain FFT for phi_inf considering an integer number of periods
                [fft_phi_inf,freq_phi_inf,max_fft_phi_inf,max_freq_phi_inf] = comp_FFT(time,phi_inf,time[nsteps-last_steps::],phi_inf[nsteps-last_steps::],order)
            else:
                fft_phi_inf      = 0
                freq_phi_inf     = 0
                max_fft_phi_inf  = 0
                max_freq_phi_inf = 0

            if np.any(I_inf != 0):
                # Obtain FFT for I_inf considering an integer number of periods
                [fft_I_inf,freq_I_inf,max_fft_I_inf,max_freq_I_inf] = comp_FFT(time,I_inf,time[nsteps-last_steps::],I_inf[nsteps-last_steps::],order)
            else:
                fft_I_inf      = 0
                freq_I_inf     = 0
                max_fft_I_inf  = 0
                max_freq_I_inf = 0
                
            # Obtain FFT for Te at vertical free loss
            [fft_Te_FL_pver,freq_Te_FL_pver,max_fft_Te_FL_pver,max_freq_Te_FL_pver] = comp_FFT(time,Te[i_plot_ver,j_plot_ver,:],time[nsteps-last_steps::],Te[i_plot_ver,j_plot_ver,nsteps-last_steps::],order)
            # Obtain FFT for phi at vertical free loss
            [fft_phi_FL_pver,freq_phi_FL_pver,max_fft_phi_FL_pver,max_freq_phi_FL_pver] = comp_FFT(time,phi[i_plot_ver,j_plot_ver,:],time[nsteps-last_steps::],phi[i_plot_ver,j_plot_ver,nsteps-last_steps::],order)

            if np.any(phi_inf != 0):
                # Obtain FFT for Dphi/Te at vertical free loss
                [fft_DphiTe_FL_pver,freq_DphiTe_FL_pver,max_fft_DphiTe_FL_pver,max_freq_DphiTe_FL_pver] = comp_FFT(time,(phi[i_plot_lat,j_plot_lat,:]-phi_inf[:])/Te[i_plot_lat,j_plot_lat,:],time[nsteps-last_steps::],(phi[i_plot_lat,j_plot_lat,nsteps-last_steps::]-phi_inf[nsteps-last_steps::])/Te[i_plot_lat,j_plot_lat,nsteps-last_steps::],order)
            else:
                fft_DphiTe_FL_pver      = 0
                freq_DphiTe_FL_pver     = 0 
                max_fft_DphiTe_FL_pver  = 0
                max_freq_DphiTe_FL_pver = 0
                

            
            # Obtain the phase shift of the signals Id and Te_mean_dom from the time between max peaks
            [_,_,time_shift_IdTe_mean_dom,phase_shift_IdTe_mean_dom_deg] = comp_phase_shift(time,Id,Te_mean_dom,time[nsteps-last_steps::],Id[nsteps-last_steps::],
                                                                                            time[nsteps-last_steps::],Te_mean_dom[nsteps-last_steps::],order)
            # Obtain the phase shift of the signals Id and I_beam from the time between max peaks
            [_,_,time_shift_IdIbeam,phase_shift_IdIbeam_deg] = comp_phase_shift(time,Id,I_beam,time[nsteps-last_steps::],Id[nsteps-last_steps::],
                                                                                time[nsteps-last_steps::],I_beam[nsteps-last_steps::],order)
            # Obtain the phase shift of the signals avg_dens_mp_neus and avg_dens_mp_ions from the time between max peaks
            [_,_,time_shift_avg_dens_mp_neusions,phase_shift_avg_dens_mp_neusions_deg] = comp_phase_shift(time_fast,avg_dens_mp_neus,avg_dens_mp_ions,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_neus[nsteps_fast-last_steps_fast::],
                                                                                                          time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_ions[nsteps_fast-last_steps_fast::],order_fast)
            
            phase_shift_avg_dens_mp_neusions_deg = 0.0
            # Obtain the phase shift of the signals representing the contributions to the total heavy species mass balance
            [_,_,time_shift_ctr_mbal_tot,phase_shift_ctr_mbal_tot_deg] = comp_phase_shift(time,ctr_mflow_fw_tot,ctr_mflow_tw_tot,time[nsteps-last_steps::],ctr_mflow_fw_tot[nsteps-last_steps::],
                                                                                          time[nsteps-last_steps::],ctr_mflow_tw_tot[nsteps-last_steps::],order)     
        ######################### OLD CODE ####################################
#        # Obtain the cross-correlation of Id and I_beam signals to obtaing their phase shift
#        if mean_type == 2:
#            time_vector   = time[nsteps-last_steps::]
#            Id_vector     = Id[nsteps-last_steps::]
#            I_beam_vector = I_beam[nsteps-last_steps::]
#            nsamples = len(time_vector)
#        xcorr = correlate(Id_vector, I_beam_vector)
##        xcorr = correlate(I_beam_vector, Id_vector)
#        # The peak of the cross-correlation gives the shift between the two signals
#        # The xcorr array goes from -nsamples to nsamples
#        dt = np.linspace(-time_vector[-1], time_vector[-1], 2*nsamples-1)
#        time_shift_IdIbeam = dt[xcorr.argmax()]
#        # force the phase shift to be in [-pi:pi]
#        period = 1.0/max_freq_Id
#        phase_shift_IdIbeam = 2*np.pi*(((0.5 + time_shift_IdIbeam/period) % 1.0) - 0.5)
#        # Convert to degrees
#        phase_shift_IdIbeam_deg = phase_shift_IdIbeam*180.0/np.pi
        #######################################################################
        
        # Obtain the utilization efficiency from the actual flows
        eta_u_bis = (mflow_twinf_i1 + mflow_twinf_i2)/(mflow_inj_n1-(mflow_twa_i1+mflow_twa_i2+mflow_twa_n1))   
        
        # Obtain the total net power of the heavy species deposited to the injection (anode) wall
#        P_inj_hs = eneflow_twa_i1 + eneflow_twa_i2 + eneflow_twa_n1 - (eneflow_inj_i1 + eneflow_inj_i2 + eneflow_inj_n1)
        P_inj_hs = Pi_Awall + Pn_Awall
        # Obtain the total net power of the heavy species deposited to the dielectric walls
#        P_mat_hs = eneflow_twmat_i1 + eneflow_twmat_i2 + eneflow_twmat_n1 - (eneflow_fwmat_i1 + eneflow_fwmat_i2 + eneflow_fwmat_n1)
        P_mat_hs = Pi_Dwall + Pn_Dwall
        
        # Obtain mean values
        if make_mean == 1 and mean_type == 0:  
            mass_mp_ions1_mean       = np.mean(mass_mp_ions[nsteps_fast-last_steps_fast::,0])            
            mass_mp_ions2_mean       = np.mean(mass_mp_ions[nsteps_fast-last_steps_fast::,1])   
            tot_mass_mp_ions_mean    = np.mean(tot_mass_mp_ions[nsteps_fast-last_steps_fast::])   
            tot_mass_mp_neus_mean    = np.mean(tot_mass_mp_neus[nsteps_fast-last_steps_fast::])  
            Isp_s_mean               = np.mean(Isp_s[nsteps-last_steps::])
            Isp_ms_mean              = np.mean(Isp_ms[nsteps-last_steps::])
            eta_u_mean               = np.mean(eta_u[nsteps-last_steps::])           
            eta_u_bis_mean           = np.mean(eta_u_bis[nsteps-last_steps::]) 
            eta_prod_mean            = np.mean(eta_prod[nsteps-last_steps::])   
            eta_cur_mean             = np.mean(eta_cur[nsteps-last_steps::]) 
            eta_div_mean             = np.mean(eta_div[nsteps-last_steps::])  
            eta_thr_mean             = np.mean(eta_thr[nsteps-last_steps::])  
            thrust_mean              = np.mean(thrust[nsteps-last_steps::]) 
            thrust_i1_mean           = np.mean(thrust_ion[nsteps-last_steps::,0]) 
            thrust_i2_mean           = np.mean(thrust_ion[nsteps-last_steps::,1]) 
            thrust_n_mean            = np.mean(thrust_neu[nsteps-last_steps::])  
            Id_mean                  = np.mean(Id[nsteps-last_steps::])  
            Id_inst_mean             = np.mean(Id_inst[nsteps-last_steps::])  
            I_beam_mean              = np.mean(I_beam[nsteps-last_steps::]) 
            avg_dens_mp_ions_mean    = np.mean(avg_dens_mp_ions[nsteps_fast-last_steps_fast::]) 
            avg_dens_mp_neus_mean    = np.mean(avg_dens_mp_neus[nsteps_fast-last_steps_fast::]) 
            Pd_mean                  = np.mean(Pd[nsteps-last_steps::])
            P_mat_mean               = np.mean(P_mat[nsteps-last_steps::])
            P_inj_mean               = np.mean(P_inj[nsteps-last_steps::])
            P_inf_mean               = np.mean(P_inf[nsteps-last_steps::])
            P_ion_mean               = np.mean(P_ion[nsteps-last_steps::])
            P_ex_mean                = np.mean(P_ex[nsteps-last_steps::])
            P_inj_hs_mean            = np.mean(P_inj_hs[nsteps-last_steps::])
            P_mat_hs_mean            = np.mean(P_mat_hs[nsteps-last_steps::])
            P_use_tot_i_mean         = np.mean(P_use_tot_i[nsteps-last_steps::])
            P_use_tot_n_mean         = np.mean(P_use_tot_n[nsteps-last_steps::])
            P_use_tot_mean           = np.mean(P_use_tot[nsteps-last_steps::])
            P_use_z_i_mean           = np.mean(P_use_z_i[nsteps-last_steps::])
            P_use_z_n_mean           = np.mean(P_use_z_n[nsteps-last_steps::])
            P_use_z_mean             = np.mean(P_use_z[nsteps-last_steps::])
            P_cath_mean              = np.mean(P_cath[nsteps-last_steps::])
            nu_cath_mean             = np.mean(nu_cath[nsteps-last_steps::])
            I_tw_tot_mean            = np.mean(I_tw_tot[nsteps-last_steps::])
            mflow_twinf_i1_mean      = np.mean(mflow_twinf_i1[nsteps-last_steps::])
            mflow_twinf_i2_mean      = np.mean(mflow_twinf_i2[nsteps-last_steps::])
            mflow_twinf_n1_mean      = np.mean(mflow_twinf_n1[nsteps-last_steps::])
            mflow_inj_n1_mean        = np.mean(mflow_inj_n1[nsteps-last_steps::])
            mflow_twa_i1_mean        = np.mean(mflow_twa_i1[nsteps-last_steps::])
            mflow_twa_i2_mean        = np.mean(mflow_twa_i2[nsteps-last_steps::])
            mflow_twa_n1_mean        = np.mean(mflow_twa_n1[nsteps-last_steps::])
            err_mbal_n1_mean         = np.mean(err_mbal_n1[nsteps-last_steps::])
            err_mbal_i1_mean         = np.mean(err_mbal_i1[nsteps-last_steps::])
            err_mbal_i2_mean         = np.mean(err_mbal_i2[nsteps-last_steps::])
            Pe_Dwall_mean            = np.mean(Pe_Dwall[nsteps-last_steps::])
            Pe_Awall_mean            = np.mean(Pe_Awall[nsteps-last_steps::])
            Pe_FLwall_mean           = np.mean(Pe_FLwall[nsteps-last_steps::])
            Pi_Dwall_mean            = np.mean(Pi_Dwall[nsteps-last_steps::])
            Pi_Awall_mean            = np.mean(Pi_Awall[nsteps-last_steps::])
            Pi_FLwall_mean           = np.mean(Pi_FLwall[nsteps-last_steps::])
            Pn_Dwall_mean            = np.mean(Pn_Dwall[nsteps-last_steps::])
            Pn_Awall_mean            = np.mean(Pn_Awall[nsteps-last_steps::])
            Pn_FLwall_mean           = np.mean(Pn_FLwall[nsteps-last_steps::])
            P_Dwall_mean             = np.mean(P_Dwall[nsteps-last_steps::])
            P_Awall_mean             = np.mean(P_Awall[nsteps-last_steps::])
            P_FLwall_mean            = np.mean(P_FLwall[nsteps-last_steps::])
            Pfield_e_mean            = np.mean(Pfield_e[nsteps-last_steps::])
        elif make_mean == 1 and mean_type == 1:      
            mass_mp_ions1_mean       = np.mean(mass_mp_ions[step_i_fast:step_f_fast+1,0])            
            mass_mp_ions2_mean       = np.mean(mass_mp_ions[step_i_fast:step_f_fast+1,1])   
            tot_mass_mp_ions_mean    = np.mean(tot_mass_mp_ions[step_i_fast:step_f_fast+1])   
            tot_mass_mp_neus_mean    = np.mean(tot_mass_mp_neus[step_i_fast:step_f_fast+1])  
            Isp_s_mean               = np.mean(Isp_s[step_i:step_f+1])
            Isp_ms_mean              = np.mean(Isp_ms[step_i:step_f+1])
            eta_u_mean               = np.mean(eta_u[step_i:step_f+1])   
            eta_u_bis_mean           = np.mean(eta_u_bis[step_i:step_f+1]) 
            eta_prod_mean            = np.mean(eta_prod[step_i:step_f+1])   
            eta_cur_mean             = np.mean(eta_cur[step_i:step_f+1])  
            eta_div_mean             = np.mean(eta_div[step_i:step_f+1])  
            eta_thr_mean             = np.mean(eta_thr[step_i:step_f+1])  
            thrust_mean              = np.mean(thrust[step_i:step_f+1]) 
            thrust_i1_mean           = np.mean(thrust_ion[step_i:step_f+1,0]) 
            thrust_i2_mean           = np.mean(thrust_ion[step_i:step_f+1,1]) 
            thrust_n_mean            = np.mean(thrust_neu[step_i:step_f+1])                                     
            Id_mean                  = np.mean(Id[step_i:step_f+1])
            Id_inst_mean             = np.mean(Id_inst[step_i:step_f+1])
            I_beam_mean              = np.mean(I_beam[step_i:step_f+1])
            avg_dens_mp_ions_mean    = np.mean(avg_dens_mp_ions[step_i_fast:step_f_fast+1]) 
            avg_dens_mp_neus_mean    = np.mean(avg_dens_mp_neus[step_i_fast:step_f_fast+1]) 
            Pd_mean                  = np.mean(Pd[step_i:step_f+1])
            P_mat_mean               = np.mean(P_mat[step_i:step_f+1])
            P_inj_mean               = np.mean(P_inj[step_i:step_f+1])
            P_inf_mean               = np.mean(P_inf[step_i:step_f+1])
            P_ion_mean               = np.mean(P_ion[step_i:step_f+1])
            P_ex_mean                = np.mean(P_ex[step_i:step_f+1])
            P_inj_hs_mean            = np.mean(P_inj_hs[step_i:step_f+1])
            P_mat_hs_mean            = np.mean(P_mat_hs[step_i:step_f+1])
            P_use_tot_i_mean         = np.mean(P_use_tot_i[step_i:step_f+1])
            P_use_tot_n_mean         = np.mean(P_use_tot_n[step_i:step_f+1])
            P_use_tot_mean           = np.mean(P_use_tot[step_i:step_f+1])
            P_use_z_i_mean           = np.mean(P_use_z_i[step_i:step_f+1])
            P_use_z_n_mean           = np.mean(P_use_z_n[step_i:step_f+1])
            P_use_z_mean             = np.mean(P_use_z[step_i:step_f+1])
            P_cath_mean              = np.mean(P_cath[step_i:step_f+1])
            nu_cath_mean             = np.mean(nu_cath[step_i:step_f+1])
            I_tw_tot_mean            = np.mean(I_tw_tot[step_i:step_f+1])
            mflow_twinf_i1_mean      = np.mean(mflow_twinf_i1[step_i:step_f+1])
            mflow_twinf_i2_mean      = np.mean(mflow_twinf_i2[step_i:step_f+1])
            mflow_twinf_n1_mean      = np.mean(mflow_twinf_n1[step_i:step_f+1])
            mflow_inj_n1_mean        = np.mean(mflow_inj_n1[step_i:step_f+1])
            mflow_twa_i1_mean        = np.mean(mflow_twa_i1[step_i:step_f+1])
            mflow_twa_i2_mean        = np.mean(mflow_twa_i2[step_i:step_f+1])
            mflow_twa_n1_mean        = np.mean(mflow_twa_n1[step_i:step_f+1])
            err_mbal_n1_mean         = np.mean(err_mbal_n1[step_i:step_f+1])
            err_mbal_i1_mean         = np.mean(err_mbal_i1[step_i:step_f+1])
            err_mbal_i2_mean         = np.mean(err_mbal_i2[step_i:step_f+1])
            err_mbal_tot_mean        = np.mean(err_mbal_tot[step_i:step_f+1])  
            Pe_Dwall_mean            = np.mean(Pe_Dwall[step_i:step_f+1])
            Pe_Awall_mean            = np.mean(Pe_Awall[step_i:step_f+1])
            Pe_FLwall_mean           = np.mean(Pe_FLwall[step_i:step_f+1])
            Pi_Dwall_mean            = np.mean(Pi_Dwall[step_i:step_f+1])
            Pi_Awall_mean            = np.mean(Pi_Awall[step_i:step_f+1])
            Pi_FLwall_mean           = np.mean(Pi_FLwall[step_i:step_f+1])
            Pn_Dwall_mean            = np.mean(Pn_Dwall[step_i:step_f+1])
            Pn_Awall_mean            = np.mean(Pn_Awall[step_i:step_f+1])
            Pn_FLwall_mean           = np.mean(Pn_FLwall[step_i:step_f+1])
            P_Dwall_mean             = np.mean(P_Dwall[step_i:step_f+1])
            P_Awall_mean             = np.mean(P_Awall[step_i:step_f+1])
            P_FLwall_mean            = np.mean(P_FLwall[step_i:step_f+1])
            Pionex_mean              = np.mean(Pionex[step_i:step_f+1])               
            Pnothrust_walls_mean     = np.mean(Pnothrust_walls[step_i:step_f+1])       
            Pnothrust_mean           = np.mean(Pnothrust[step_i:step_f+1])   
            Pthrust_mean             = np.mean(Pthrust[step_i:step_f+1])   
            Ploss_mean               = np.mean(Ploss[step_i:step_f+1])                
            Pfield_e_mean            = np.mean(Pfield_e[step_i:step_f+1])
        if make_mean == 1 and mean_type == 2:
            [_,_,_,_,_,_,
             mean_min_exp_Id,mean_max_exp_Id,exp_Id_mean,
             max2mean_exp_Id,min2mean_exp_Id,amp_exp_Id,
             mins_ind_comp_exp_Id,maxs_ind_comp_exp_Id]                                 = max_min_mean_vals(exp_time,exp_time[exp_nsteps-exp_last_steps::],exp_Id[exp_nsteps-exp_last_steps::],exp_order)
            [_,_,_,_,_,_,
             mean_min_exp_Vd,mean_max_exp_Vd,exp_Vd_mean,
             max2mean_exp_Vd,min2mean_exp_Vd,amp_exp_Vd,
             mins_ind_comp_exp_Vd,maxs_ind_comp_exp_Vd]                                 = max_min_mean_vals(exp_time,exp_time[exp_nsteps-exp_last_steps::],exp_Vd[exp_nsteps-exp_last_steps::],exp_order)
            [_,_,_,_,_,_,
             mean_min_exp_Pd,mean_max_exp_Pd,exp_Pd_mean,
             max2mean_exp_Pd,min2mean_exp_Pd,amp_exp_Pd,
             mins_ind_comp_exp_Pd,maxs_ind_comp_exp_Pd]                                 = max_min_mean_vals(exp_time,exp_time[exp_nsteps-exp_last_steps::],exp_Pd[exp_nsteps-exp_last_steps::],exp_order)
            [_,_,_,_,_,_,
             mean_min_mass_mp_ions1,mean_max_mass_mp_ions1,mass_mp_ions1_mean,
             max2mean_mass_mp_ions1,min2mean_mass_mp_ions1,amp_mass_mp_ions1,
             mins_ind_comp_mass_mp_ions1,maxs_ind_comp_mass_mp_ions1]                   = max_min_mean_vals(time_fast,time_fast[nsteps_fast-last_steps_fast::],mass_mp_ions[nsteps_fast-last_steps_fast::,0],order)
            if num_ion_spe > 1:
                [_,_,_,_,_,_,
                 mean_min_mass_mp_ions2,mean_max_mass_mp_ions2,mass_mp_ions2_mean,
                 max2mean_mass_mp_ions2,min2mean_mass_mp_ions2,amp_mass_mp_ions2,
                 mins_ind_comp_mass_mp_ions2,maxs_ind_comp_mass_mp_ions2]               = max_min_mean_vals(time_fast,time_fast[nsteps_fast-last_steps_fast::],mass_mp_ions[nsteps_fast-last_steps_fast::,1],order)
            else:
                mean_min_mass_mp_ions2 = 0.0
                mean_max_mass_mp_ions2 = 0.0
                mass_mp_ions2_mean     = 0.0
                max2mean_mass_mp_ions2 = 0.0
                min2mean_mass_mp_ions2 = 0.0
                amp_mass_mp_ions2      = 0.0
                mins_ind_comp_mass_mp_ions2 = np.zeros(np.shape(maxs_ind_comp_mass_mp_ions1),dtype=int)
                maxs_ind_comp_mass_mp_ions2 = np.zeros(np.shape(maxs_ind_comp_mass_mp_ions1),dtype=int)
                
            [_,_,_,_,_,_,
             mean_min_tot_mass_mp_ions,mean_max_tot_mass_mp_ions,tot_mass_mp_ions_mean,
             max2mean_tot_mass_mp_ions,min2mean_tot_mass_mp_ions,amp_tot_mass_mp_ions,
             mins_ind_comp_tot_mass_mp_ions,maxs_ind_comp_tot_mass_mp_ions]             = max_min_mean_vals(time_fast,time_fast[nsteps_fast-last_steps_fast::],tot_mass_mp_ions[nsteps_fast-last_steps_fast::],order)
            [_,_,_,_,_,_,
             mean_min_tot_mass_mp_neus,mean_max_tot_mass_mp_neus,tot_mass_mp_neus_mean,
             max2mean_tot_mass_mp_neus,min2mean_tot_mass_mp_neus,amp_tot_mass_mp_neus,
             mins_ind_comp_tot_mass_mp_neus,maxs_ind_comp_tot_mass_mp_neus]             = max_min_mean_vals(time_fast,time_fast[nsteps_fast-last_steps_fast::],tot_mass_mp_neus[nsteps_fast-last_steps_fast::],order)            
            [_,_,_,_,_,_,
             mean_min_avg_dens_mp_ions,mean_max_avg_dens_mp_ions,avg_dens_mp_ions_mean,
             max2mean_avg_dens_mp_ions,min2mean_avg_dens_mp_ions,amp_avg_dens_mp_ions,
             mins_ind_comp_avg_dens_mp_ions,maxs_ind_comp_avg_dens_mp_ions]             = max_min_mean_vals(time_fast,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_ions[nsteps_fast-last_steps_fast::],order)
            [_,_,_,_,_,_,
             mean_min_avg_dens_mp_neus,mean_max_avg_dens_mp_neus,avg_dens_mp_neus_mean,
             max2mean_avg_dens_mp_neus,min2mean_avg_dens_mp_neus,amp_avg_dens_mp_neus,
             mins_ind_comp_avg_dens_mp_neus,maxs_ind_comp_avg_dens_mp_neus]             = max_min_mean_vals(time_fast,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_neus[nsteps_fast-last_steps_fast::],order)            
            [_,_,_,_,_,_,
             mean_min_Isp_s,mean_max_Isp_s,Isp_s_mean,
             max2mean_Isp_s,min2mean_Isp_s,amp_Isp_s,
             mins_ind_comp_Isp_s,maxs_ind_comp_Isp_s]                                   = max_min_mean_vals(time,time[nsteps-last_steps::],Isp_s[nsteps-last_steps::],order) 
            [_,_,_,_,_,_,
             mean_min_Isp_ms,mean_max_Isp_ms,Isp_ms_mean,
             max2mean_Isp_ms,min2mean_Isp_ms,amp_Isp_ms,
             mins_ind_comp_Isp_ms,maxs_ind_comp_Isp_ms]                                 = max_min_mean_vals(time,time[nsteps-last_steps::],Isp_ms[nsteps-last_steps::],order) 
            [_,_,_,_,_,_,
             mean_min_eta_u,mean_max_eta_u,eta_u_mean,
             max2mean_eta_u,min2mean_eta_u,amp_eta_u,
             mins_ind_comp_eta_u,maxs_ind_comp_eta_u]                                   = max_min_mean_vals(time,time[nsteps-last_steps::],eta_u[nsteps-last_steps::],order) 
            [_,_,_,_,_,_,
             mean_min_eta_u_bis,mean_max_eta_u_bis,eta_u_bis_mean,
             max2mean_eta_u_bis,min2mean_eta_u_bis,amp_eta_u_bis,
             mins_ind_comp_eta_u_bis,maxs_ind_comp_eta_u_bis]                           = max_min_mean_vals(time,time[nsteps-last_steps::],eta_u_bis[nsteps-last_steps::],order) 
            [_,_,_,_,_,_,
             mean_min_eta_prod,mean_max_eta_prod,eta_prod_mean,
             max2mean_eta_prod,min2mean_eta_prod,amp_eta_prod,
             mins_ind_comp_eta_prod,maxs_ind_comp_eta_prod]                             = max_min_mean_vals(time,time[nsteps-last_steps::],eta_prod[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_eta_cur,mean_max_eta_cur,eta_cur_mean,
             max2mean_eta_cur,min2mean_eta_cur,amp_eta_cur,
             mins_ind_comp_eta_cur,maxs_ind_comp_eta_cur]                               = max_min_mean_vals(time,time[nsteps-last_steps::],eta_cur[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_eta_div,mean_max_eta_div,eta_div_mean,
             max2mean_eta_div,min2mean_eta_div,amp_eta_div,
             mins_ind_comp_eta_div,maxs_ind_comp_eta_div]                               = max_min_mean_vals(time,time[nsteps-last_steps::],eta_div[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_eta_thr,mean_max_eta_thr,eta_thr_mean,
             max2mean_eta_thr,min2mean_eta_thr,amp_eta_thr,
             mins_ind_comp_eta_thr,maxs_ind_comp_eta_thr]                               = max_min_mean_vals(time,time[nsteps-last_steps::],eta_thr[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_thrust,mean_max_thrust,thrust_mean,
             max2mean_thrust,min2mean_thrust,amp_thrust,
             mins_ind_comp_thrust,maxs_ind_comp_thrust]                                 = max_min_mean_vals(time,time[nsteps-last_steps::],thrust[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_thrust_i1,mean_max_thrust_i1,thrust_i1_mean,
             max2mean_thrust_i1,min2mean_thrust_i1,amp_thrust_i1,
             mins_ind_comp_thrust_i1,maxs_ind_comp_thrust_i1]                           = max_min_mean_vals(time,time[nsteps-last_steps::],thrust_ion[nsteps-last_steps::,0],order)
            if num_ion_spe == 2:
                [_,_,_,_,_,_,
                 mean_min_thrust_i2,mean_max_thrust_i2,thrust_i2_mean,
                 max2mean_thrust_i2,min2mean_thrust_i2,amp_thrust_i2,
                 mins_ind_comp_thrust_i2,maxs_ind_comp_thrust_i2]                           = max_min_mean_vals(time,time[nsteps-last_steps::],thrust_ion[nsteps-last_steps::,1],order)
                mean_min_thrust_i3 = 0.0
                mean_max_thrust_i3 = 0.0
                thrust_i3_mean     = 0.0
                max2mean_thrust_i3 = 0.0
                min2mean_thrust_i3 = 0.0
                amp_thrust_i3      = 0.0
                mins_ind_comp_thrust_i3 = np.zeros(np.shape(maxs_ind_comp_thrust_i1),dtype=int)
                maxs_ind_comp_thrust_i3 = np.zeros(np.shape(maxs_ind_comp_thrust_i1),dtype=int)
                mean_min_thrust_i4 = 0.0
                mean_max_thrust_i4 = 0.0
                thrust_i4_mean     = 0.0
                max2mean_thrust_i4 = 0.0
                min2mean_thrust_i4 = 0.0
                amp_thrust_i4      = 0.0
                mins_ind_comp_thrust_i4 = np.zeros(np.shape(maxs_ind_comp_thrust_i1),dtype=int)
                maxs_ind_comp_thrust_i4 = np.zeros(np.shape(maxs_ind_comp_thrust_i1),dtype=int)
            elif num_ion_spe == 4:
                [_,_,_,_,_,_,
                 mean_min_thrust_i2,mean_max_thrust_i2,thrust_i2_mean,
                 max2mean_thrust_i2,min2mean_thrust_i2,amp_thrust_i2,
                 mins_ind_comp_thrust_i2,maxs_ind_comp_thrust_i2]                           = max_min_mean_vals(time,time[nsteps-last_steps::],thrust_ion[nsteps-last_steps::,1],order)
                [_,_,_,_,_,_,
                 mean_min_thrust_i3,mean_max_thrust_i3,thrust_i3_mean,
                 max2mean_thrust_i3,min2mean_thrust_i3,amp_thrust_i3,
                 mins_ind_comp_thrust_i3,maxs_ind_comp_thrust_i3]                           = max_min_mean_vals(time,time[nsteps-last_steps::],thrust_ion[nsteps-last_steps::,2],order)
                [_,_,_,_,_,_,
                 mean_min_thrust_i4,mean_max_thrust_i4,thrust_i4_mean,
                 max2mean_thrust_i4,min2mean_thrust_i4,amp_thrust_i4,
                 mins_ind_comp_thrust_i4,maxs_ind_comp_thrust_i4]                           = max_min_mean_vals(time,time[nsteps-last_steps::],thrust_ion[nsteps-last_steps::,3],order)
            
            else:
                mean_min_thrust_i2 = 0.0
                mean_max_thrust_i2 = 0.0
                thrust_i2_mean     = 0.0
                max2mean_thrust_i2 = 0.0
                min2mean_thrust_i2 = 0.0
                amp_thrust_i2      = 0.0
                mins_ind_comp_thrust_i2 = np.zeros(np.shape(maxs_ind_comp_thrust_i1),dtype=int)
                maxs_ind_comp_thrust_i2 = np.zeros(np.shape(maxs_ind_comp_thrust_i1),dtype=int)
                mean_min_thrust_i3 = 0.0
                mean_max_thrust_i3 = 0.0
                thrust_i3_mean     = 0.0
                max2mean_thrust_i3 = 0.0
                min2mean_thrust_i3 = 0.0
                amp_thrust_i3      = 0.0
                mins_ind_comp_thrust_i3 = np.zeros(np.shape(maxs_ind_comp_thrust_i1),dtype=int)
                maxs_ind_comp_thrust_i3 = np.zeros(np.shape(maxs_ind_comp_thrust_i1),dtype=int)
                mean_min_thrust_i4 = 0.0
                mean_max_thrust_i4 = 0.0
                thrust_i4_mean     = 0.0
                max2mean_thrust_i4 = 0.0
                min2mean_thrust_i4 = 0.0
                amp_thrust_i4      = 0.0
                mins_ind_comp_thrust_i4 = np.zeros(np.shape(maxs_ind_comp_thrust_i1),dtype=int)
                maxs_ind_comp_thrust_i4 = np.zeros(np.shape(maxs_ind_comp_thrust_i1),dtype=int)
            [_,_,_,_,_,_,
             mean_min_thrust_n1,mean_max_thrust_n1,thrust_n1_mean,
             max2mean_thrust_n1,min2mean_thrust_n1,amp_thrust_n1,
             mins_ind_comp_thrust_n1,maxs_ind_comp_thrust_n1]                             = max_min_mean_vals(time,time[nsteps-last_steps::],thrust_neu[nsteps-last_steps::,0],order)
            if num_neu_spe == 3:
                [_,_,_,_,_,_,
                 mean_min_thrust_n2,mean_max_thrust_n2,thrust_n2_mean,
                 max2mean_thrust_n2,min2mean_thrust_n2,amp_thrust_n2,
                 mins_ind_comp_thrust_n2,maxs_ind_comp_thrust_n2]                         = max_min_mean_vals(time,time[nsteps-last_steps::],thrust_neu[nsteps-last_steps::,1],order)
                [_,_,_,_,_,_,
                 mean_min_thrust_n3,mean_max_thrust_n3,thrust_n3_mean,
                 max2mean_thrust_n3,min2mean_thrust_n1,amp_thrust_n3,
                 mins_ind_comp_thrust_n3,maxs_ind_comp_thrust_n3]                         = max_min_mean_vals(time,time[nsteps-last_steps::],thrust_neu[nsteps-last_steps::,2],order)
                
            else:
                mean_min_thrust_n2 = 0.0
                mean_max_thrust_n2 = 0.0
                thrust_n2_mean     = 0.0
                max2mean_thrust_n2 = 0.0
                min2mean_thrust_n2 = 0.0
                amp_thrust_n2      = 0.0
                mins_ind_comp_thrust_n2 = np.zeros(np.shape(maxs_ind_comp_thrust_n1),dtype=int)
                maxs_ind_comp_thrust_n2 = np.zeros(np.shape(maxs_ind_comp_thrust_n1),dtype=int)
                mean_min_thrust_n3 = 0.0
                mean_max_thrust_n3 = 0.0
                thrust_n3_mean     = 0.0
                max2mean_thrust_n3 = 0.0
                min2mean_thrust_n3 = 0.0
                amp_thrust_n3      = 0.0
                mins_ind_comp_thrust_n3 = np.zeros(np.shape(maxs_ind_comp_thrust_n1),dtype=int)
                maxs_ind_comp_thrust_n3 = np.zeros(np.shape(maxs_ind_comp_thrust_n1),dtype=int)
                
            [_,_,_,_,_,_,
             mean_min_thrust_e,mean_max_thrust_e,thrust_e_mean,
             max2mean_thrust_e,min2mean_thrust_e,amp_thrust_e,
             mins_ind_comp_thrust_e,maxs_ind_comp_thrust_e]                             = max_min_mean_vals(time,time[nsteps-last_steps::],thrust_e[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Te_mean_dom,mean_max_Te_mean_dom,Te_mean_dom_mean,
             max2mean_Te_mean_dom,min2mean_Te_mean_dom,amp_Te_mean_dom,
             mins_ind_comp_Te_mean_dom,maxs_ind_comp_Te_mean_dom]                       = max_min_mean_vals(time,time[nsteps-last_steps::],Te_mean_dom[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Id,mean_max_Id,Id_mean,
             max2mean_Id,min2mean_Id,amp_Id,
             mins_ind_comp_Id,maxs_ind_comp_Id]                                         = max_min_mean_vals(time,time[nsteps-last_steps::],Id[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Id_inst,mean_max_Id_inst,Id_inst_mean,
             max2mean_Id_inst,min2mean_Id_inst,amp_Id_inst,
             mins_ind_comp_Id_inst,maxs_ind_comp_Id_inst]                               = max_min_mean_vals(time,time[nsteps-last_steps::],Id_inst[nsteps-last_steps::],order)
            
            if n_cond_wall > 0:
                for i in range(0,n_cond_wall):
                    [_,_,_,_,_,_,
                     mean_min_Icond,mean_max_Icond,Icond_mean,
                     max2mean_Icond,min2mean_Icond,amp_Icond,
                     mins_ind_comp_Icond,maxs_ind_comp_Icond]                           = max_min_mean_vals(time,time[nsteps-last_steps::],Icond[nsteps-last_steps::,i],order)
                    if np.any(Vcond != 0):
                        if Vcond[1,0] != Vcond[2,0]:
                            [_,_,_,_,_,_,
                             mean_min_Vcond,mean_max_Vcond,Vcond_mean,
                             max2mean_Vcond,min2mean_Vcond,amp_Vcond,
                             mins_ind_comp_Vcond,maxs_ind_comp_Vcond]                   = max_min_mean_vals(time,time[nsteps-last_steps::],Vcond[nsteps-last_steps::,i],order)
                        else:
                            mean_min_Vcond      = 0
                            mean_max_Vcond      = 0
                            Vcond_mean          = 0
                            max2mean_Vcond      = 0
                            min2mean_Vcond      = 0
                            amp_Vcond           = 0
                            mins_ind_comp_Vcond = 0
                            maxs_ind_comp_Vcond = 0
            else:
                mean_min_Icond      = 0
                mean_max_Icond      = 0
                Icond_mean          = 0
                max2mean_Icond      = 0
                min2mean_Icond      = 0
                amp_Icond           = 0
                mins_ind_comp_Icond = 0
                maxs_ind_comp_Icond = 0
                
                mean_min_Vcond      = 0
                mean_max_Vcond      = 0
                Vcond_mean          = 0
                max2mean_Vcond      = 0
                min2mean_Vcond      = 0
                amp_Vcond           = 0
                mins_ind_comp_Vcond = 0
                maxs_ind_comp_Vcond = 0
            
            if np.any(Icath != 0):
                [_,_,_,_,_,_,
                 mean_min_Icath,mean_max_Icath,Icath_mean,
                 max2mean_Icath,min2mean_Icath,amp_Icath,
                 mins_ind_comp_Icath,maxs_ind_comp_Icath]                                 = max_min_mean_vals(time,time[nsteps-last_steps::],Icath[nsteps-last_steps::],order)
            
            else:
                mean_min_Icath      = 0
                mean_max_Icath      = 0
                Icath_mean          = 0
                max2mean_Icath      = 0
                min2mean_Icath      = 0
                amp_Icath           = 0
                mins_ind_comp_Icath = 0
                maxs_ind_comp_Icath = 0
            
            [_,_,_,_,_,_,
             mean_min_I_beam,mean_max_I_beam,I_beam_mean,
             max2mean_I_beam,min2mean_I_beam,amp_I_beam,
             mins_ind_comp_I_beam,maxs_ind_comp_I_beam]                                 = max_min_mean_vals(time,time[nsteps-last_steps::],I_beam[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Pd,mean_max_Pd,Pd_mean,
             max2mean_Pd,min2mean_Pd,amp_Pd,
             mins_ind_comp_Pd,maxs_ind_comp_Pd]                                         = max_min_mean_vals(time,time[nsteps-last_steps::],Pd[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_mat,mean_max_P_mat,P_mat_mean,
             max2mean_P_mat,min2mean_P_mat,amp_P_mat,
             mins_ind_comp_P_mat,maxs_ind_comp_P_mat]                                   = max_min_mean_vals(time,time[nsteps-last_steps::],P_mat[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_inj,mean_max_P_inj,P_inj_mean,
             max2mean_P_inj,min2mean_P_inj,amp_P_inj,
             mins_ind_comp_P_inj,maxs_ind_comp_P_inj]                                   = max_min_mean_vals(time,time[nsteps-last_steps::],P_inj[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_inf,mean_max_P_inf,P_inf_mean,
             max2mean_P_inf,min2mean_P_inf,amp_P_inf,
             mins_ind_comp_P_inf,maxs_ind_comp_P_inf]                                   = max_min_mean_vals(time,time[nsteps-last_steps::],P_inf[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_ion,mean_max_P_ion,P_ion_mean,
             max2mean_P_ion,min2mean_P_ion,amp_P_ion,
             mins_ind_comp_P_ion,maxs_ind_comp_P_ion]                                   = max_min_mean_vals(time,time[nsteps-last_steps::],P_ion[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_ex,mean_max_P_ex,P_ex_mean,
             max2mean_P_ex,min2mean_P_ex,amp_P_ex,
             mins_ind_comp_P_ex,maxs_ind_comp_P_ex]                                     = max_min_mean_vals(time,time[nsteps-last_steps::],P_ex[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_inj_hs,mean_max_P_inj_hs,P_inj_hs_mean,
             max2mean_P_inj_hs,min2mean_P_inj_hs,amp_P_inj_hs,
             mins_ind_comp_P_inj_hs,maxs_ind_comp_P_inj_hs]                             = max_min_mean_vals(time,time[nsteps-last_steps::],P_inj_hs[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_mat_hs,mean_max_P_mat_hs,P_mat_hs_mean,
             max2mean_P_mat_hs,min2mean_P_mat_hs,amp_P_mat_hs,
             mins_ind_comp_P_mat_hs,maxs_ind_comp_P_mat_hs]                             = max_min_mean_vals(time,time[nsteps-last_steps::],P_mat_hs[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_use_tot_i,mean_max_P_use_tot_i,P_use_tot_i_mean,
             max2mean_P_use_tot_i,min2mean_P_use_tot_i,amp_P_use_tot_i,
             mins_ind_comp_P_use_tot_i,maxs_ind_comp_P_use_tot_i]                       = max_min_mean_vals(time,time[nsteps-last_steps::],P_use_tot_i[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_use_tot_n,mean_max_P_use_tot_n,P_use_tot_n_mean,
             max2mean_P_use_tot_n,min2mean_P_use_tot_n,amp_P_use_tot_n,
             mins_ind_comp_P_use_tot_n,maxs_ind_comp_P_use_tot_n]                       = max_min_mean_vals(time,time[nsteps-last_steps::],P_use_tot_n[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_use_tot,mean_max_P_use_tot,P_use_tot_mean,
             max2mean_P_use_tot,min2mean_P_use_tot,amp_P_use_tot,
             mins_ind_comp_P_use_tot,maxs_ind_comp_P_use_tot]                           = max_min_mean_vals(time,time[nsteps-last_steps::],P_use_tot[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_use_z_i,mean_max_P_use_z_i,P_use_z_i_mean,
             max2mean_P_use_z_i,min2mean_P_use_z_i,amp_P_use_z_i,
             mins_ind_comp_P_use_z_i,maxs_ind_comp_P_use_z_i]                           = max_min_mean_vals(time,time[nsteps-last_steps::],P_use_z_i[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_use_z_n,mean_max_P_use_z_n,P_use_z_n_mean,
             max2mean_P_use_z_n,min2mean_P_use_z_n,amp_P_use_z_n,
             mins_ind_comp_P_use_z_n,maxs_ind_comp_P_use_z_n]                           = max_min_mean_vals(time,time[nsteps-last_steps::],P_use_z_n[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_use_z,mean_max_P_use_z,P_use_z_mean,
             max2mean_P_use_z,min2mean_P_use_z,amp_P_use_z,
             mins_ind_comp_P_use_z,maxs_ind_comp_P_use_z]                               = max_min_mean_vals(time,time[nsteps-last_steps::],P_use_z[nsteps-last_steps::],order)
            
            if cath_type == 2:
                # NOTE: for cath_type = 1 (wall cathode) we have:
                # nu_cath = 0
                # P_cath  = power inputed from the wall cathode (needs to be obtained)
                [_,_,_,_,_,_,
                 mean_min_P_cath,mean_max_P_cath,P_cath_mean,
                 max2mean_P_cath,min2mean_P_cath,amp_P_cath,
                 mins_ind_comp_P_cath,maxs_ind_comp_P_cath]                                 = max_min_mean_vals(time,time[nsteps-last_steps::],P_cath[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_nu_cath,mean_max_nu_cath,nu_cath_mean,
                 max2mean_nu_cath,min2mean_nu_cath,amp_nu_cath,
                 mins_ind_comp_nu_cath,maxs_ind_comp_nu_cath]                               = max_min_mean_vals(time,time[nsteps-last_steps::],nu_cath[nsteps-last_steps::],order)
            elif cath_type == 1:
                [_,_,_,_,_,_,
                 mean_min_P_cath,mean_max_P_cath,P_cath_mean,
                 max2mean_P_cath,min2mean_P_cath,amp_P_cath,
                 mins_ind_comp_P_cath,maxs_ind_comp_P_cath]                                 = max_min_mean_vals(time,time[nsteps-last_steps::],P_cath[nsteps-last_steps::],order)
                mean_min_nu_cath      = 0
                mean_max_nu_cath      = 0
                nu_cath_mean          = 0
                max2mean_nu_cath      = 0
                min2mean_nu_cath      = 0
                amp_nu_cath           = 0
                mins_ind_comp_nu_cath = 0
                maxs_ind_comp_nu_cath = 0
            
            [_,_,_,_,_,_,
             mean_min_I_tw_tot,mean_max_I_tw_tot,I_tw_tot_mean,
             max2mean_I_tw_tot,min2mean_I_tw_tot,amp_I_tw_tot,
             mins_ind_comp_I_tw_tot,maxs_ind_comp_I_tw_tot]                             = max_min_mean_vals(time,time[nsteps-last_steps::],I_tw_tot[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_mflow_twinf_i1,mean_max_mflow_twinf_i1,mflow_twinf_i1_mean,
             max2mean_mflow_twinf_i1,min2mean_mflow_twinf_i1,amp_mflow_twinf_i1,
             mins_ind_comp_mflow_twinf_i1,maxs_ind_comp_mflow_twinf_i1]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twinf_i1[nsteps-last_steps::],order)
            if num_ion_spe == 2:
                [_,_,_,_,_,_,
                 mean_min_mflow_twinf_i2,mean_max_mflow_twinf_i2,mflow_twinf_i2_mean,
                 max2mean_mflow_twinf_i2,min2mean_mflow_twinf_i2,amp_mflow_twinf_i2,
                 mins_ind_comp_mflow_twinf_i2,maxs_ind_comp_mflow_twinf_i2]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twinf_i2[nsteps-last_steps::],order)
                mean_min_mflow_twinf_i3 = 0.0
                mean_max_mflow_twinf_i3 = 0.0
                mflow_twinf_i3_mean     = 0.0
                max2mean_mflow_twinf_i3 = 0.0
                min2mean_mflow_twinf_i3 = 0.0
                amp_mflow_twinf_i3      = 0.0
                mins_ind_comp_mflow_twinf_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_i1),dtype=int)
                maxs_ind_comp_mflow_twinf_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_i1),dtype=int)
                mean_min_mflow_twinf_i4 = 0.0
                mean_max_mflow_twinf_i4 = 0.0
                mflow_twinf_i4_mean     = 0.0
                max2mean_mflow_twinf_i4 = 0.0
                min2mean_mflow_twinf_i4 = 0.0
                amp_mflow_twinf_i4      = 0.0
                mins_ind_comp_mflow_twinf_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_i1),dtype=int)
                maxs_ind_comp_mflow_twinf_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_i1),dtype=int)
            elif num_ion_spe == 4:
                [_,_,_,_,_,_,
                 mean_min_mflow_twinf_i2,mean_max_mflow_twinf_i2,mflow_twinf_i2_mean,
                 max2mean_mflow_twinf_i2,min2mean_mflow_twinf_i2,amp_mflow_twinf_i2,
                 mins_ind_comp_mflow_twinf_i2,maxs_ind_comp_mflow_twinf_i2]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twinf_i2[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_mflow_twinf_i3,mean_max_mflow_twinf_i3,mflow_twinf_i3_mean,
                 max2mean_mflow_twinf_i3,min2mean_mflow_twinf_i3,amp_mflow_twinf_i3,
                 mins_ind_comp_mflow_twinf_i3,maxs_ind_comp_mflow_twinf_i3]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twinf_i3[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_mflow_twinf_i4,mean_max_mflow_twinf_i4,mflow_twinf_i4_mean,
                 max2mean_mflow_twinf_i4,min2mean_mflow_twinf_i4,amp_mflow_twinf_i4,
                 mins_ind_comp_mflow_twinf_i4,maxs_ind_comp_mflow_twinf_i4]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twinf_i4[nsteps-last_steps::],order)
                
            else:
                mean_min_mflow_twinf_i2 = 0.0
                mean_max_mflow_twinf_i2 = 0.0
                mflow_twinf_i2_mean     = 0.0
                max2mean_mflow_twinf_i2 = 0.0
                min2mean_mflow_twinf_i2 = 0.0
                amp_mflow_twinf_i2      = 0.0
                mins_ind_comp_mflow_twinf_i2 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_i1),dtype=int)
                maxs_ind_comp_mflow_twinf_i2 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_i1),dtype=int)
                mean_min_mflow_twinf_i3 = 0.0
                mean_max_mflow_twinf_i3 = 0.0
                mflow_twinf_i3_mean     = 0.0
                max2mean_mflow_twinf_i3 = 0.0
                min2mean_mflow_twinf_i3 = 0.0
                amp_mflow_twinf_i3      = 0.0
                mins_ind_comp_mflow_twinf_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_i1),dtype=int)
                maxs_ind_comp_mflow_twinf_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_i1),dtype=int)
                mean_min_mflow_twinf_i4 = 0.0
                mean_max_mflow_twinf_i4 = 0.0
                mflow_twinf_i4_mean     = 0.0
                max2mean_mflow_twinf_i4 = 0.0
                min2mean_mflow_twinf_i4 = 0.0
                amp_mflow_twinf_i4      = 0.0
                mins_ind_comp_mflow_twinf_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_i1),dtype=int)
                maxs_ind_comp_mflow_twinf_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_i1),dtype=int)
            [_,_,_,_,_,_,
             mean_min_mflow_twinf_n1,mean_max_mflow_twinf_n1,mflow_twinf_n1_mean,
             max2mean_mflow_twinf_n1,min2mean_mflow_twinf_n1,amp_mflow_twinf_n1,
             mins_ind_comp_mflow_twinf_n1,maxs_ind_comp_mflow_twinf_n1]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twinf_n1[nsteps-last_steps::],order)
            if num_neu_spe == 3:
                [_,_,_,_,_,_,
                 mean_min_mflow_twinf_n2,mean_max_mflow_twinf_n2,mflow_twinf_n2_mean,
                 max2mean_mflow_twinf_n2,min2mean_mflow_twinf_n2,amp_mflow_twinf_n2,
                 mins_ind_comp_mflow_twinf_n2,maxs_ind_comp_mflow_twinf_n2]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twinf_n2[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_mflow_twinf_n3,mean_max_mflow_twinf_n3,mflow_twinf_n3_mean,
                 max2mean_mflow_twinf_n3,min2mean_mflow_twinf_n3,amp_mflow_twinf_n3,
                 mins_ind_comp_mflow_twinf_n3,maxs_ind_comp_mflow_twinf_n3]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twinf_n3[nsteps-last_steps::],order)
            else:
                mean_min_mflow_twinf_n2 = 0.0
                mean_max_mflow_twinf_n2 = 0.0
                mflow_twinf_n2_mean     = 0.0
                max2mean_mflow_twinf_n2 = 0.0
                min2mean_mflow_twinf_n2 = 0.0
                amp_mflow_twinf_n2      = 0.0
                mins_ind_comp_mflow_twinf_n2 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_n1),dtype=int)
                maxs_ind_comp_mflow_twinf_n2 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_n1),dtype=int)
                mean_min_mflow_twinf_n3 = 0.0
                mean_max_mflow_twinf_n3 = 0.0
                mflow_twinf_n3_mean     = 0.0
                max2mean_mflow_twinf_n3 = 0.0
                min2mean_mflow_twinf_n3 = 0.0
                amp_mflow_twinf_n3      = 0.0
                mins_ind_comp_mflow_twinf_n3 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_n1),dtype=int)
                maxs_ind_comp_mflow_twinf_n3 = np.zeros(np.shape(maxs_ind_comp_mflow_twinf_n1),dtype=int)
            
            [_,_,_,_,_,_,
             mean_min_mflow_inj_n1,mean_max_mflow_inj_n1,mflow_inj_n1_mean,
             max2mean_mflow_inj_n1,min2mean_mflow_inj_n1,amp_mflow_inj_n1,
             mins_ind_comp_mflow_inj_n1,maxs_ind_comp_mflow_inj_n1]                     = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_inj_n1[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_mflow_twa_i1,mean_max_mflow_twa_i1,mflow_twa_i1_mean,
             max2mean_mflow_twa_i1,min2mean_mflow_twa_i1,amp_mflow_twa_i1,
             mins_ind_comp_mflow_twa_i1,maxs_ind_comp_mflow_twa_i1]                     = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twa_i1[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_mflow_twmat_i1,mean_max_mflow_twmat_i1,mflow_twmat_i1_mean,
             max2mean_mflow_twmat_i1,min2mean_mflow_twmat_i1,amp_mflow_twmat_i1,
             mins_ind_comp_mflow_twmat_i1,maxs_ind_comp_mflow_twmat_i1]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twmat_i1[nsteps-last_steps::],order)

            if cath_type == 1 and np.any(mflow_twcat_i1) != 0:
                [_,_,_,_,_,_,
                 mean_min_mflow_twcat_i1,mean_max_mflow_twcat_i1,mflow_twcat_i1_mean,
                 max2mean_mflow_twcat_i1,min2mean_mflow_twcat_i1,amp_mflow_twcat_i1,
                 mins_ind_comp_mflow_twcat_i1,maxs_ind_comp_mflow_twcat_i1]             = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twcat_i1[nsteps-last_steps::],order)
            else:
                 mean_min_mflow_twcat_i1 = 0.0
                 mean_max_mflow_twcat_i1 = 0.0
                 mflow_twcat_i1_mean     = 0.0
                 max2mean_mflow_twcat_i1 = 0.0
                 min2mean_mflow_twcat_i1 = 0.0
                 amp_mflow_twcat_i1      = 0.0
                 mins_ind_comp_mflow_twcat_i1 = 0.0
                 maxs_ind_comp_mflow_twcat_i1 = 0.0
                
            if num_ion_spe == 2:
                [_,_,_,_,_,_,
                 mean_min_mflow_twa_i2,mean_max_mflow_twa_i2,mflow_twa_i2_mean,
                 max2mean_mflow_twa_i2,min2mean_mflow_twa_i2,amp_mflow_twa_i2,
                 mins_ind_comp_mflow_twa_i2,maxs_ind_comp_mflow_twa_i2]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twa_i2[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_mflow_twmat_i2,mean_max_mflow_twmat_i2,mflow_twmat_i2_mean,
                 max2mean_mflow_twmat_i2,min2mean_mflow_twmat_i2,amp_mflow_twmat_i2,
                 mins_ind_comp_mflow_twmat_i2,maxs_ind_comp_mflow_twmat_i2]             = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twmat_i2[nsteps-last_steps::],order)
                if cath_type == 1 and np.any(mflow_twcat_i2) != 0:
                    [_,_,_,_,_,_,
                     mean_min_mflow_twcat_i2,mean_max_mflow_twcat_i2,mflow_twcat_i2_mean,
                     max2mean_mflow_twcat_i2,min2mean_mflow_twcat_i2,amp_mflow_twcat_i2,
                     mins_ind_comp_mflow_twcat_i2,maxs_ind_comp_mflow_twcat_i2]         = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twcat_i2[nsteps-last_steps::],order)
                else:
                    mean_min_mflow_twcat_i2 = 0.0
                    mean_max_mflow_twcat_i2 = 0.0
                    mflow_twcat_i2_mean     = 0.0
                    max2mean_mflow_twcat_i2 = 0.0
                    min2mean_mflow_twcat_i2 = 0.0
                    amp_mflow_twcat_i2      = 0.0
                    mins_ind_comp_mflow_twcat_i2 = 0.0
                    maxs_ind_comp_mflow_twcat_i2 = 0.0
                
                mean_min_mflow_twa_i3 = 0.0
                mean_max_mflow_twa_i3 = 0.0
                mflow_twa_i3_mean     = 0.0
                max2mean_mflow_twa_i3 = 0.0
                min2mean_mflow_twa_i3 = 0.0
                amp_mflow_twa_i3      = 0.0
                mins_ind_comp_mflow_twa_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_i1),dtype=int)
                maxs_ind_comp_mflow_twa_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_i1),dtype=int)
                mean_min_mflow_twmat_i3 = 0.0
                mean_max_mflow_twmat_i3 = 0.0
                mflow_twmat_i3_mean     = 0.0
                max2mean_mflow_twmat_i3 = 0.0
                min2mean_mflow_twmat_i3 = 0.0
                amp_mflow_twmat_i3      = 0.0
                mins_ind_comp_mflow_twmat_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twmat_i1),dtype=int)
                maxs_ind_comp_mflow_twmat_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twmat_i1),dtype=int)
                mean_min_mflow_twcat_i3 = 0.0
                mean_max_mflow_twcat_i3 = 0.0
                mflow_twcat_i3_mean     = 0.0
                max2mean_mflow_twcat_i3 = 0.0
                min2mean_mflow_twcat_i3 = 0.0
                amp_mflow_twcat_i3      = 0.0
                mins_ind_comp_mflow_twcat_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twcat_i1),dtype=int)
                maxs_ind_comp_mflow_twcat_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twcat_i1),dtype=int)
                mean_min_mflow_twa_i4 = 0.0
                mean_max_mflow_twa_i4 = 0.0
                mflow_twa_i4_mean     = 0.0
                max2mean_mflow_twa_i4 = 0.0
                min2mean_mflow_twa_i4 = 0.0
                amp_mflow_twa_i4      = 0.0
                mins_ind_comp_mflow_twa_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_i1),dtype=int)
                maxs_ind_comp_mflow_twa_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_i1),dtype=int)
                mean_min_mflow_twmat_i4 = 0.0
                mean_max_mflow_twmat_i4 = 0.0
                mflow_twmat_i4_mean     = 0.0
                max2mean_mflow_twmat_i4 = 0.0
                min2mean_mflow_twmat_i4 = 0.0
                amp_mflow_twmat_i4      = 0.0
                mins_ind_comp_mflow_twmat_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twmat_i1),dtype=int)
                maxs_ind_comp_mflow_twmat_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twmat_i1),dtype=int)
                mean_min_mflow_twcat_i4 = 0.0
                mean_max_mflow_twcat_i4 = 0.0
                mflow_twcat_i4_mean     = 0.0
                max2mean_mflow_twcat_i4 = 0.0
                min2mean_mflow_twcat_i4 = 0.0
                amp_mflow_twcat_i4      = 0.0
                mins_ind_comp_mflow_twcat_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twcat_i1),dtype=int)
                maxs_ind_comp_mflow_twcat_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twcat_i1),dtype=int)

                    
            elif num_ion_spe == 4:
                [_,_,_,_,_,_,
                 mean_min_mflow_twa_i2,mean_max_mflow_twa_i2,mflow_twa_i2_mean,
                 max2mean_mflow_twa_i2,min2mean_mflow_twa_i2,amp_mflow_twa_i2,
                 mins_ind_comp_mflow_twa_i2,maxs_ind_comp_mflow_twa_i2]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twa_i2[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_mflow_twmat_i2,mean_max_mflow_twmat_i2,mflow_twmat_i2_mean,
                 max2mean_mflow_twmat_i2,min2mean_mflow_twmat_i2,amp_mflow_twmat_i2,
                 mins_ind_comp_mflow_twmat_i2,maxs_ind_comp_mflow_twmat_i2]             = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twmat_i2[nsteps-last_steps::],order)
                if cath_type == 1 and np.any(mflow_twcat_i2) != 0:
                    [_,_,_,_,_,_,
                     mean_min_mflow_twcat_i2,mean_max_mflow_twcat_i2,mflow_twcat_i2_mean,
                     max2mean_mflow_twcat_i2,min2mean_mflow_twcat_i2,amp_mflow_twcat_i2,
                     mins_ind_comp_mflow_twcat_i2,maxs_ind_comp_mflow_twcat_i2]         = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twcat_i2[nsteps-last_steps::],order)
                else:
                    mean_min_mflow_twcat_i2 = 0.0
                    mean_max_mflow_twcat_i2 = 0.0
                    mflow_twcat_i2_mean     = 0.0
                    max2mean_mflow_twcat_i2 = 0.0
                    min2mean_mflow_twcat_i2 = 0.0
                    amp_mflow_twcat_i2      = 0.0
                    mins_ind_comp_mflow_twcat_i2 = 0.0
                    maxs_ind_comp_mflow_twcat_i2 = 0.0
                    
                [_,_,_,_,_,_,
                 mean_min_mflow_twa_i3,mean_max_mflow_twa_i3,mflow_twa_i3_mean,
                 max2mean_mflow_twa_i3,min2mean_mflow_twa_i3,amp_mflow_twa_i3,
                 mins_ind_comp_mflow_twa_i3,maxs_ind_comp_mflow_twa_i3]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twa_i3[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_mflow_twmat_i3,mean_max_mflow_twmat_i3,mflow_twmat_i3_mean,
                 max2mean_mflow_twmat_i3,min2mean_mflow_twmat_i3,amp_mflow_twmat_i3,
                 mins_ind_comp_mflow_twmat_i3,maxs_ind_comp_mflow_twmat_i3]             = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twmat_i3[nsteps-last_steps::],order)
                if cath_type == 1 and np.any(mflow_twcat_i3) != 0:
                    [_,_,_,_,_,_,
                     mean_min_mflow_twcat_i3,mean_max_mflow_twcat_i3,mflow_twcat_i3_mean,
                     max2mean_mflow_twcat_i3,min2mean_mflow_twcat_i3,amp_mflow_twcat_i3,
                     mins_ind_comp_mflow_twcat_i3,maxs_ind_comp_mflow_twcat_i3]             = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twcat_i3[nsteps-last_steps::],order)
                else:
                    mean_min_mflow_twcat_i3 = 0.0
                    mean_max_mflow_twcat_i3 = 0.0
                    mflow_twcat_i3_mean     = 0.0
                    max2mean_mflow_twcat_i3 = 0.0
                    min2mean_mflow_twcat_i3 = 0.0
                    amp_mflow_twcat_i3      = 0.0
                    mins_ind_comp_mflow_twcat_i3 = 0.0
                    maxs_ind_comp_mflow_twcat_i3 = 0.0
                    
                [_,_,_,_,_,_,
                 mean_min_mflow_twa_i4,mean_max_mflow_twa_i4,mflow_twa_i4_mean,
                 max2mean_mflow_twa_i4,min2mean_mflow_twa_i4,amp_mflow_twa_i4,
                 mins_ind_comp_mflow_twa_i4,maxs_ind_comp_mflow_twa_i4]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twa_i4[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_mflow_twmat_i4,mean_max_mflow_twmat_i4,mflow_twmat_i4_mean,
                 max2mean_mflow_twmat_i4,min2mean_mflow_twmat_i4,amp_mflow_twmat_i4,
                 mins_ind_comp_mflow_twmat_i4,maxs_ind_comp_mflow_twmat_i4]             = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twmat_i4[nsteps-last_steps::],order)
                if cath_type == 1 and np.any(mflow_twcat_i4) != 0:
                    [_,_,_,_,_,_,
                     mean_min_mflow_twcat_i4,mean_max_mflow_twcat_i4,mflow_twcat_i4_mean,
                     max2mean_mflow_twcat_i4,min2mean_mflow_twcat_i4,amp_mflow_twcat_i4,
                     mins_ind_comp_mflow_twcat_i4,maxs_ind_comp_mflow_twcat_i4]             = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twcat_i4[nsteps-last_steps::],order)
                else:
                    mean_min_mflow_twcat_i4 = 0.0
                    mean_max_mflow_twcat_i4 = 0.0
                    mflow_twcat_i4_mean     = 0.0
                    max2mean_mflow_twcat_i4 = 0.0
                    min2mean_mflow_twcat_i4 = 0.0
                    amp_mflow_twcat_i4      = 0.0
                    mins_ind_comp_mflow_twcat_i4 = 0.0
                    maxs_ind_comp_mflow_twcat_i4 = 0.0
                
            else:
                mean_min_mflow_twa_i2 = 0.0
                mean_max_mflow_twa_i2 = 0.0
                mflow_twa_i2_mean     = 0.0
                max2mean_mflow_twa_i2 = 0.0
                min2mean_mflow_twa_i2 = 0.0
                amp_mflow_twa_i2      = 0.0
                mins_ind_comp_mflow_twa_i2 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_i1),dtype=int)
                maxs_ind_comp_mflow_twa_i2 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_i1),dtype=int)
                mean_min_mflow_twmat_i2 = 0.0
                mean_max_mflow_twmat_i2 = 0.0
                mflow_twmat_i2_mean     = 0.0
                max2mean_mflow_twmat_i2 = 0.0
                min2mean_mflow_twmat_i2 = 0.0
                amp_mflow_twmat_i2      = 0.0
                mins_ind_comp_mflow_twmat_i2 = np.zeros(np.shape(maxs_ind_comp_mflow_twmat_i1),dtype=int)
                maxs_ind_comp_mflow_twmat_i2 = np.zeros(np.shape(maxs_ind_comp_mflow_twmat_i1),dtype=int)
                mean_min_mflow_twcat_i2 = 0.0
                mean_max_mflow_twcat_i2 = 0.0
                mflow_twcat_i2_mean     = 0.0
                max2mean_mflow_twcat_i2 = 0.0
                min2mean_mflow_twcat_i2 = 0.0
                amp_mflow_twcat_i2      = 0.0
                mins_ind_comp_mflow_twcat_i2 = np.zeros(np.shape(maxs_ind_comp_mflow_twcat_i1),dtype=int)
                maxs_ind_comp_mflow_twcat_i2 = np.zeros(np.shape(maxs_ind_comp_mflow_twcat_i1),dtype=int)
                mean_min_mflow_twa_i3 = 0.0
                mean_max_mflow_twa_i3 = 0.0
                mflow_twa_i3_mean     = 0.0
                max2mean_mflow_twa_i3 = 0.0
                min2mean_mflow_twa_i3 = 0.0
                amp_mflow_twa_i3      = 0.0
                mins_ind_comp_mflow_twa_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_i1),dtype=int)
                maxs_ind_comp_mflow_twa_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_i1),dtype=int)
                mean_min_mflow_twmat_i3 = 0.0
                mean_max_mflow_twmat_i3 = 0.0
                mflow_twmat_i3_mean     = 0.0
                max2mean_mflow_twmat_i3 = 0.0
                min2mean_mflow_twmat_i3 = 0.0
                amp_mflow_twmat_i3      = 0.0
                mins_ind_comp_mflow_twmat_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twmat_i1),dtype=int)
                maxs_ind_comp_mflow_twmat_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twmat_i1),dtype=int)
                mean_min_mflow_twcat_i3 = 0.0
                mean_max_mflow_twcat_i3 = 0.0
                mflow_twcat_i3_mean     = 0.0
                max2mean_mflow_twcat_i3 = 0.0
                min2mean_mflow_twcat_i3 = 0.0
                amp_mflow_twcat_i3      = 0.0
                mins_ind_comp_mflow_twcat_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twcat_i1),dtype=int)
                maxs_ind_comp_mflow_twcat_i3 = np.zeros(np.shape(maxs_ind_comp_mflow_twcat_i1),dtype=int)
                mean_min_mflow_twa_i4 = 0.0
                mean_max_mflow_twa_i4 = 0.0
                mflow_twa_i4_mean     = 0.0
                max2mean_mflow_twa_i4 = 0.0
                min2mean_mflow_twa_i4 = 0.0
                amp_mflow_twa_i4      = 0.0
                mins_ind_comp_mflow_twa_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_i1),dtype=int)
                maxs_ind_comp_mflow_twa_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_i1),dtype=int)
                mean_min_mflow_twmat_i4 = 0.0
                mean_max_mflow_twmat_i4 = 0.0
                mflow_twmat_i4_mean     = 0.0
                max2mean_mflow_twmat_i4 = 0.0
                min2mean_mflow_twmat_i4 = 0.0
                amp_mflow_twmat_i4      = 0.0
                mins_ind_comp_mflow_twmat_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twmat_i1),dtype=int)
                maxs_ind_comp_mflow_twmat_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twmat_i1),dtype=int)
                mean_min_mflow_twcat_i4 = 0.0
                mean_max_mflow_twcat_i4 = 0.0
                mflow_twcat_i4_mean     = 0.0
                max2mean_mflow_twcat_i4 = 0.0
                min2mean_mflow_twcat_i4 = 0.0
                amp_mflow_twcat_i4      = 0.0
                mins_ind_comp_mflow_twcat_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twcat_i1),dtype=int)
                maxs_ind_comp_mflow_twcat_i4 = np.zeros(np.shape(maxs_ind_comp_mflow_twcat_i1),dtype=int)
                
                
            [_,_,_,_,_,_,
             mean_min_mflow_twa_n1,mean_max_mflow_twa_n1,mflow_twa_n1_mean,
             max2mean_mflow_twa_n1,min2mean_mflow_twa_n1,amp_mflow_twa_n1,
             mins_ind_comp_mflow_twa_n1,maxs_ind_comp_mflow_twa_n1]                     = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twa_n1[nsteps-last_steps::],order)
            if num_neu_spe == 3:
                [_,_,_,_,_,_,
                 mean_min_mflow_twa_n2,mean_max_mflow_twa_n2,mflow_twa_n2_mean,
                 max2mean_mflow_twa_n2,min2mean_mflow_twa_n2,amp_mflow_twa_n2,
                 mins_ind_comp_mflow_twa_n2,maxs_ind_comp_mflow_twa_n2]                     = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twa_n2[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_mflow_twa_n3,mean_max_mflow_twa_n3,mflow_twa_n3_mean,
                 max2mean_mflow_twa_n3,min2mean_mflow_twa_n3,amp_mflow_twa_n3,
                 mins_ind_comp_mflow_twa_n3,maxs_ind_comp_mflow_twa_n3]                     = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twa_n3[nsteps-last_steps::],order)
            else:
                mean_min_mflow_twa_n2 = 0.0
                mean_max_mflow_twa_n2 = 0.0
                mflow_twa_n2_mean     = 0.0
                max2mean_mflow_twa_n2 = 0.0
                min2mean_mflow_twa_n2 = 0.0
                amp_mflow_twa_n2      = 0.0
                mins_ind_comp_mflow_twa_n2 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_n1),dtype=int)
                maxs_ind_comp_mflow_twa_n2 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_n1),dtype=int)
                mean_min_mflow_twa_n3 = 0.0
                mean_max_mflow_twa_n3 = 0.0
                mflow_twa_n3_mean     = 0.0
                max2mean_mflow_twa_n3 = 0.0
                min2mean_mflow_twa_n3 = 0.0
                amp_mflow_twa_n3      = 0.0
                mins_ind_comp_mflow_twa_n3 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_n1),dtype=int)
                maxs_ind_comp_mflow_twa_n3 = np.zeros(np.shape(maxs_ind_comp_mflow_twa_n1),dtype=int)
                    
            
            [_,_,_,_,_,_,
             mean_min_err_mbal_n1,mean_max_err_mbal_n1,err_mbal_n1_mean,
             max2mean_err_mbal_n1,min2mean_err_mbal_n1,amp_err_mbal_n1,
             mins_ind_comp_err_mbal_n1,maxs_ind_comp_err_mbal_n1]                     = max_min_mean_vals(time,time[nsteps-last_steps::],err_mbal_n1[nsteps-last_steps::],order)
            if num_neu_spe == 3:
                [_,_,_,_,_,_,
                 mean_min_err_mbal_n2,mean_max_err_mbal_n2,err_mbal_n2_mean,
                 max2mean_err_mbal_n2,min2mean_err_mbal_n2,amp_err_mbal_n2,
                 mins_ind_comp_err_mbal_n2,maxs_ind_comp_err_mbal_n2]                     = max_min_mean_vals(time,time[nsteps-last_steps::],err_mbal_n2[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_err_mbal_n3,mean_max_err_mbal_n3,err_mbal_n3_mean,
                 max2mean_err_mbal_n3,min2mean_err_mbal_n3,amp_err_mbal_n3,
                 mins_ind_comp_err_mbal_n3,maxs_ind_comp_err_mbal_n3]                     = max_min_mean_vals(time,time[nsteps-last_steps::],err_mbal_n3[nsteps-last_steps::],order)
            else:
                mean_min_err_mbal_n2 = 0.0
                mean_max_err_mbal_n2 = 0.0
                err_mbal_n2_mean     = 0.0
                max2mean_err_mbal_n2 = 0.0
                min2mean_err_mbal_n2 = 0.0
                amp_err_mbal_n2      = 0.0
                mins_ind_comp_err_mbal_n2 = np.zeros(np.shape(mins_ind_comp_err_mbal_n1),dtype=int)
                maxs_ind_comp_err_mbal_n2 = np.zeros(np.shape(maxs_ind_comp_err_mbal_n1),dtype=int)
                mean_min_err_mbal_n3 = 0.0
                mean_max_err_mbal_n3 = 0.0
                err_mbal_n3_mean     = 0.0
                max2mean_err_mbal_n3 = 0.0
                min2mean_err_mbal_n3 = 0.0
                amp_err_mbal_n3      = 0.0
                mins_ind_comp_err_mbal_n3 = np.zeros(np.shape(mins_ind_comp_err_mbal_n1),dtype=int)
                maxs_ind_comp_err_mbal_n3 = np.zeros(np.shape(maxs_ind_comp_err_mbal_n1),dtype=int)
            
            [_,_,_,_,_,_,
             mean_min_err_mbal_i1,mean_max_err_mbal_i1,err_mbal_i1_mean,
             max2mean_err_mbal_i1,min2mean_err_mbal_i1,amp_err_mbal_i1,
             mins_ind_comp_err_mbal_i1,maxs_ind_comp_err_mbal_i1]                     = max_min_mean_vals(time,time[nsteps-last_steps::],err_mbal_i1[nsteps-last_steps::],order)
            if num_ion_spe == 2:
                [_,_,_,_,_,_,
                 mean_min_err_mbal_i2,mean_max_err_mbal_i2,err_mbal_i2_mean,
                 max2mean_err_mbal_i2,min2mean_err_mbal_i2,amp_err_mbal_i2,
                 mins_ind_comp_err_mbal_i2,maxs_ind_comp_err_mbal_i2]                     = max_min_mean_vals(time,time[nsteps-last_steps::],err_mbal_i2[nsteps-last_steps::],order)
                mean_min_err_mbal_i3 = 0.0
                mean_max_err_mbal_i3 = 0.0
                err_mbal_i3_mean     = 0.0
                max2mean_err_mbal_i3 = 0.0
                min2mean_err_mbal_i3 = 0.0
                amp_err_mbal_i3      = 0.0
                mins_ind_comp_err_mbal_i3 = np.zeros(np.shape(mins_ind_comp_err_mbal_i1),dtype=int)
                maxs_ind_comp_err_mbal_i3 = np.zeros(np.shape(maxs_ind_comp_err_mbal_i1),dtype=int)
                mean_min_err_mbal_i4 = 0.0
                mean_max_err_mbal_i4 = 0.0
                err_mbal_i4_mean     = 0.0
                max2mean_err_mbal_i4 = 0.0
                min2mean_err_mbal_i4 = 0.0
                amp_err_mbal_i4      = 0.0
                mins_ind_comp_err_mbal_i4 = np.zeros(np.shape(mins_ind_comp_err_mbal_i1),dtype=int)
                maxs_ind_comp_err_mbal_i4 = np.zeros(np.shape(maxs_ind_comp_err_mbal_i1),dtype=int)
            elif num_ion_spe == 4:
                [_,_,_,_,_,_,
                 mean_min_err_mbal_i2,mean_max_err_mbal_i2,err_mbal_i2_mean,
                 max2mean_err_mbal_i2,min2mean_err_mbal_i2,amp_err_mbal_i2,
                 mins_ind_comp_err_mbal_i2,maxs_ind_comp_err_mbal_i2]                     = max_min_mean_vals(time,time[nsteps-last_steps::],err_mbal_i2[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_err_mbal_i3,mean_max_err_mbal_i3,err_mbal_i3_mean,
                 max2mean_err_mbal_i3,min2mean_err_mbal_i3,amp_err_mbal_i3,
                 mins_ind_comp_err_mbal_i3,maxs_ind_comp_err_mbal_i3]                     = max_min_mean_vals(time,time[nsteps-last_steps::],err_mbal_i3[nsteps-last_steps::],order)
                [_,_,_,_,_,_,
                 mean_min_err_mbal_i4,mean_max_err_mbal_i4,err_mbal_i4_mean,
                 max2mean_err_mbal_i4,min2mean_err_mbal_i4,amp_err_mbal_i4,
                 mins_ind_comp_err_mbal_i4,maxs_ind_comp_err_mbal_i4]                     = max_min_mean_vals(time,time[nsteps-last_steps::],err_mbal_i4[nsteps-last_steps::],order)
            else:
                mean_min_err_mbal_i2 = 0.0
                mean_max_err_mbal_i2 = 0.0
                err_mbal_i2_mean     = 0.0
                max2mean_err_mbal_i2 = 0.0
                min2mean_err_mbal_i2 = 0.0
                amp_err_mbal_i2      = 0.0
                mins_ind_comp_err_mbal_i2 = np.zeros(np.shape(mins_ind_comp_err_mbal_i1),dtype=int)
                maxs_ind_comp_err_mbal_i2 = np.zeros(np.shape(maxs_ind_comp_err_mbal_i1),dtype=int)
                mean_min_err_mbal_i3 = 0.0
                mean_max_err_mbal_i3 = 0.0
                err_mbal_i3_mean     = 0.0
                max2mean_err_mbal_i3 = 0.0
                min2mean_err_mbal_i3 = 0.0
                amp_err_mbal_i3      = 0.0
                mins_ind_comp_err_mbal_i3 = np.zeros(np.shape(mins_ind_comp_err_mbal_i1),dtype=int)
                maxs_ind_comp_err_mbal_i3 = np.zeros(np.shape(maxs_ind_comp_err_mbal_i1),dtype=int)
                mean_min_err_mbal_i4 = 0.0
                mean_max_err_mbal_i4 = 0.0
                err_mbal_i4_mean     = 0.0
                max2mean_err_mbal_i4 = 0.0
                min2mean_err_mbal_i4 = 0.0
                amp_err_mbal_i4      = 0.0
                mins_ind_comp_err_mbal_i4 = np.zeros(np.shape(mins_ind_comp_err_mbal_i1),dtype=int)
                maxs_ind_comp_err_mbal_i4 = np.zeros(np.shape(maxs_ind_comp_err_mbal_i1),dtype=int)
            [_,_,_,_,_,_,
             mean_min_err_mbal_tot,mean_max_err_mbal_tot,err_mbal_tot_mean,
             max2mean_err_mbal_tot,min2mean_err_mbal_tot,amp_err_mbal_tot,
             mins_ind_comp_err_mbal_tot,maxs_ind_comp_err_mbal_tot]                     = max_min_mean_vals(time,time[nsteps-last_steps::],err_mbal_tot[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Pe_Dwall,mean_max_Pe_Dwall,Pe_Dwall_mean,
             max2mean_Pe_Dwall,min2mean_Pe_Dwall,amp_Pe_Dwall,
             mins_ind_comp_Pe_Dwall,maxs_ind_comp_Pe_Dwall]                             = max_min_mean_vals(time,time[nsteps-last_steps::],Pe_Dwall[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Pe_Awall,mean_max_Pe_Awall,Pe_Awall_mean,
             max2mean_Pe_Awall,min2mean_Pe_Awall,amp_Pe_Awall,
             mins_ind_comp_Pe_Awall,maxs_ind_comp_Pe_Awall]                             = max_min_mean_vals(time,time[nsteps-last_steps::],Pe_Awall[nsteps-last_steps::],order)
            if cath_type == 1:
                [_,_,_,_,_,_,
                 mean_min_Pe_Cwall,mean_max_Pe_Cwall,Pe_Cwall_mean,
                 max2mean_Pe_Cwall,min2mean_Pe_Cwall,amp_Pe_Cwall,
                 mins_ind_comp_Pe_Cwall,maxs_ind_comp_Pe_Cwall]                         = max_min_mean_vals(time,time[nsteps-last_steps::],Pe_Cwall[nsteps-last_steps::],order)
            elif cath_type == 2:
                mean_min_Pe_Cwall = 0.0
                mean_max_Pe_Cwall = 0.0
                Pe_Cwall_mean     = 0.0
                max2mean_Pe_Cwall = 0.0
                min2mean_Pe_Cwall = 0.0
                amp_Pe_Cwall      = 0.0
                mins_ind_comp_Pe_Cwall = 0.0
                maxs_ind_comp_Pe_Cwall = 0.0  
            
            [_,_,_,_,_,_,
             mean_min_Pe_FLwall,mean_max_Pe_FLwall,Pe_FLwall_mean,
             max2mean_Pe_FLwall,min2mean_Pe_FLwall,amp_Pe_FLwall,
             mins_ind_comp_Pe_FLwall,maxs_ind_comp_Pe_FLwall]                           = max_min_mean_vals(time,time[nsteps-last_steps::],Pe_FLwall[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Pi_Dwall,mean_max_Pi_Dwall,Pi_Dwall_mean,
             max2mean_Pi_Dwall,min2mean_Pi_Dwall,amp_Pi_Dwall,
             mins_ind_comp_Pi_Dwall,maxs_ind_comp_Pi_Dwall]                             = max_min_mean_vals(time,time[nsteps-last_steps::],Pi_Dwall[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Pi_Awall,mean_max_Pi_Awall,Pi_Awall_mean,
             max2mean_Pi_Awall,min2mean_Pi_Awall,amp_Pi_Awall,
             mins_ind_comp_Pi_Awall,maxs_ind_comp_Pi_Awall]                             = max_min_mean_vals(time,time[nsteps-last_steps::],Pi_Awall[nsteps-last_steps::],order)
            if cath_type == 1 and np.any(Pi_Cwall) != 0:
                [_,_,_,_,_,_,
                 mean_min_Pi_Cwall,mean_max_Pi_Cwall,Pi_Cwall_mean,
                 max2mean_Pi_Cwall,min2mean_Pi_Cwall,amp_Pi_Cwall,
                 mins_ind_comp_Pi_Cwall,maxs_ind_comp_Pi_Cwall]                         = max_min_mean_vals(time,time[nsteps-last_steps::],Pi_Cwall[nsteps-last_steps::],order)
            else:
                mean_min_Pi_Cwall = 0.0
                mean_max_Pi_Cwall = 0.0
                Pi_Cwall_mean     = 0.0
                max2mean_Pi_Cwall = 0.0
                min2mean_Pi_Cwall = 0.0
                amp_Pi_Cwall      = 0.0
                mins_ind_comp_Pi_Cwall = 0.0
                maxs_ind_comp_Pi_Cwall = 0.0  
            [_,_,_,_,_,_,
             mean_min_Pi_FLwall,mean_max_Pi_FLwall,Pi_FLwall_mean,
             max2mean_Pi_FLwall,min2mean_Pi_FLwall,amp_Pi_FLwall,
             mins_ind_comp_Pi_FLwall,maxs_ind_comp_Pi_FLwall]                           = max_min_mean_vals(time,time[nsteps-last_steps::],Pi_FLwall[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Pn_Dwall,mean_max_Pn_Dwall,Pn_Dwall_mean,
             max2mean_Pn_Dwall,min2mean_Pn_Dwall,amp_Pn_Dwall,
             mins_ind_comp_Pn_Dwall,maxs_ind_comp_Pn_Dwall]                             = max_min_mean_vals(time,time[nsteps-last_steps::],Pn_Dwall[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Pn_Awall,mean_max_Pn_Awall,Pn_Awall_mean,
             max2mean_Pn_Awall,min2mean_Pn_Awall,amp_Pn_Awall,
             mins_ind_comp_Pn_Awall,maxs_ind_comp_Pn_Awall]                             = max_min_mean_vals(time,time[nsteps-last_steps::],Pn_Awall[nsteps-last_steps::],order)
            if cath_type == 1 and np.any(Pn_Cwall) != 0:
                [_,_,_,_,_,_,
                 mean_min_Pn_Cwall,mean_max_Pn_Cwall,Pn_Cwall_mean,
                 max2mean_Pn_Cwall,min2mean_Pn_Cwall,amp_Pn_Cwall,
                 mins_ind_comp_Pn_Cwall,maxs_ind_comp_Pn_Cwall]                         = max_min_mean_vals(time,time[nsteps-last_steps::],Pn_Cwall[nsteps-last_steps::],order)
            else:
                mean_min_Pn_Cwall = 0.0
                mean_max_Pn_Cwall = 0.0
                Pn_Cwall_mean     = 0.0
                max2mean_Pn_Cwall = 0.0
                min2mean_Pn_Cwall = 0.0
                amp_Pn_Cwall      = 0.0
                mins_ind_comp_Pn_Cwall = 0.0
                maxs_ind_comp_Pn_Cwall = 0.0
            [_,_,_,_,_,_,
             mean_min_Pn_FLwall,mean_max_Pn_FLwall,Pn_FLwall_mean,
             max2mean_Pn_FLwall,min2mean_Pn_FLwall,amp_Pn_FLwall,
             mins_ind_comp_Pn_FLwall,maxs_ind_comp_Pn_FLwall]                           = max_min_mean_vals(time,time[nsteps-last_steps::],Pn_FLwall[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_Dwall,mean_max_P_Dwall,P_Dwall_mean,
             max2mean_P_Dwall,min2mean_P_Dwall,amp_P_Dwall,
             mins_ind_comp_P_Dwall,maxs_ind_comp_P_Dwall]                               = max_min_mean_vals(time,time[nsteps-last_steps::],P_Dwall[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_Awall,mean_max_P_Awall,P_Awall_mean,
             max2mean_P_Awall,min2mean_P_Awall,amp_P_Awall,
             mins_ind_comp_P_Awall,maxs_ind_comp_P_Awall]                               = max_min_mean_vals(time,time[nsteps-last_steps::],P_Awall[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_P_FLwall,mean_max_P_FLwall,P_FLwall_mean,
             max2mean_P_FLwall,min2mean_P_FLwall,amp_P_FLwall,
             mins_ind_comp_P_FLwall,maxs_ind_comp_P_FLwall]                             = max_min_mean_vals(time,time[nsteps-last_steps::],P_FLwall[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Pionex,mean_max_Pionex,Pionex_mean,
             max2mean_Pionex,min2mean_Pionex,amp_Pionex,
             mins_ind_comp_Pionex,maxs_ind_comp_Pionex]                                 = max_min_mean_vals(time,time[nsteps-last_steps::],Pionex[nsteps-last_steps::],order)
            [_,_,_,_,_,_,
             mean_min_Pnothrust_walls,mean_max_Pnothrust_walls,Pnothrust_walls_mean,
             max2mean_Pnothrust_walls,min2mean_Pnothrust_walls,amp_Pnothrust_walls,
             mins_ind_comp_Pnothrust_walls,maxs_ind_comp_Pnothrust_walls]               = max_min_mean_vals(time,time[nsteps-last_steps::],Pnothrust_walls[nsteps-last_steps::],order)         
            [_,_,_,_,_,_,
             mean_min_Pnothrust,mean_max_Pnothrust,Pnothrust_mean,
             max2mean_Pnothrust,min2mean_Pnothrust,amp_Pnothrust,
             mins_ind_comp_Pnothrust,maxs_ind_comp_Pnothrust]                           = max_min_mean_vals(time,time[nsteps-last_steps::],Pnothrust[nsteps-last_steps::],order) 
            [_,_,_,_,_,_,
             mean_min_Pthrust,mean_max_Pthrust,Pthrust_mean,
             max2mean_Pthrust,min2mean_Pthrust,amp_Pthrust,
             mins_ind_comp_Pthrust,maxs_ind_comp_Pthrust]                               = max_min_mean_vals(time,time[nsteps-last_steps::],Pthrust[nsteps-last_steps::],order) 
            [_,_,_,_,_,_,
             mean_min_Ploss,mean_max_Ploss,Ploss_mean,
             max2mean_Ploss,min2mean_Ploss,amp_Ploss,
             mins_ind_comp_Ploss,maxs_ind_comp_Ploss]                                   = max_min_mean_vals(time,time[nsteps-last_steps::],Ploss[nsteps-last_steps::],order)          
            [_,_,_,_,_,_,
             mean_min_Pfield_e,mean_max_Pfield_e,Pfield_e_mean,
             max2mean_Pfield_e,min2mean_Pfield_e,amp_Pfield_e,
             mins_ind_comp_Pfield_e,maxs_ind_comp_Pfield_e]                             = max_min_mean_vals(time,time[nsteps-last_steps::],Pfield_e[nsteps-last_steps::],order)      
            if np.any(Pturb != 0):
                [_,_,_,_,_,_,
                 mean_min_Pturb,mean_max_Pturb,Pturb_mean,
                 max2mean_Pturb,min2mean_Pturb,amp_Pturb,
                 mins_ind_comp_Pturb,maxs_ind_comp_Pturb]                                   = max_min_mean_vals(time,time[nsteps-last_steps::],Pturb[nsteps-last_steps::],order)    
            else:
                mean_min_Pturb      = 0.0 
                mean_max_Pturb      = 0.0 
                Pturb_mean          = 0.0 
                max2mean_Pturb      = 0.0 
                min2mean_Pturb      = 0.0
                amp_Pturb           = 0.0  
                mins_ind_comp_Pturb = 0.0
                maxs_ind_comp_Pturb = 0.0
            [_,_,_,_,_,_,
             mean_min_Psource,mean_max_Psource,Psource_mean,
             max2mean_Psource,min2mean_Psource,amp_Psource,
             mins_ind_comp_Psource,maxs_ind_comp_Psource]                               = max_min_mean_vals(time,time[nsteps-last_steps::],Psource[nsteps-last_steps::],order)    
            
#            if np.any(phi_inf != 0):
            if np.any(np.diff(phi_inf != 0)):
                [_,_,_,_,_,_,
                 mean_min_phi_inf,mean_max_phi_inf,phi_inf_mean,
                 max2mean_phi_inf,min2mean_phi_inf,amp_phi_inf,
                 mins_ind_comp_phi_inf,maxs_ind_comp_phi_inf]                               = max_min_mean_vals(time,time[nsteps-last_steps::],phi_inf[nsteps-last_steps::],order)    
            else:
                mean_min_phi_inf      = 0.0 
                mean_max_phi_inf      = 0.0 
#                phi_inf_mean          = 0.0 
                phi_inf_mean          = phi_inf[-1]
                max2mean_phi_inf      = 0.0 
                min2mean_phi_inf      = 0.0
                amp_phi_inf           = 0.0  
                mins_ind_comp_phi_inf = 0.0
                maxs_ind_comp_phi_inf = 0.0
            
            if np.any(I_inf != 0):
                [_,_,_,_,_,_,
                 mean_min_I_inf,mean_max_I_inf,I_inf_mean,
                 max2mean_I_inf,min2mean_I_inf,amp_I_inf,
                 mins_ind_comp_I_inf,maxs_ind_comp_I_inf]                               = max_min_mean_vals(time,time[nsteps-last_steps::],I_inf[nsteps-last_steps::],order)    
            else:
                mean_min_I_inf      = 0.0 
                mean_max_I_inf      = 0.0 
                I_inf_mean          = 0.0 
                max2mean_I_inf      = 0.0 
                min2mean_I_inf      = 0.0
                amp_I_inf           = 0.0  
                mins_ind_comp_I_inf = 0.0
                maxs_ind_comp_I_inf = 0.0
            
            [_,_,_,_,_,_,
             mean_min_phi_FL_lat,mean_max_phi_FL_lat,phi_FL_lat_mean,
             max2mean_phi_FL_lat,min2mean_phi_FL_lat,amp_phi_FL_lat,
             mins_ind_comp_phi_FL_lat,maxs_ind_comp_phi_FL_lat]                         = max_min_mean_vals(time,time[nsteps-last_steps::],phi_FL_lat[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_phi_FL_ver,mean_max_phi_FL_ver,phi_FL_ver_mean,
             max2mean_phi_FL_ver,min2mean_phi_FL_ver,amp_phi_FL_ver,
             mins_ind_comp_phi_FL_ver,maxs_ind_comp_phi_FL_ver]                         = max_min_mean_vals(time,time[nsteps-last_steps::],phi_FL_ver[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_phi_FL,mean_max_phi_FL,phi_FL_mean,
             max2mean_phi_FL,min2mean_phi_FL,amp_phi_FL,
             mins_ind_comp_phi_FL,maxs_ind_comp_phi_FL]                                 = max_min_mean_vals(time,time[nsteps-last_steps::],phi_FL[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_phi_FL_lat_int,mean_max_phi_FL_lat_int,phi_FL_lat_int_mean,
             max2mean_phi_FL_lat_int,min2mean_phi_FL_lat_int,amp_phi_FL_lat_int,
             mins_ind_comp_phi_FL_lat_int,maxs_ind_comp_phi_FL_lat_int]                 = max_min_mean_vals(time,time[nsteps-last_steps::],phi_FL_lat_int[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_phi_FL_ver_int,mean_max_phi_FL_ver_int,phi_FL_ver_int_mean,
             max2mean_phi_FL_ver_int,min2mean_phi_FL_ver_int,amp_phi_FL_ver_int,
             mins_ind_comp_phi_FL_ver_int,maxs_ind_comp_phi_FL_ver_int]                 = max_min_mean_vals(time,time[nsteps-last_steps::],phi_FL_ver_int[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_phi_FL_int,mean_max_phi_FL_int,phi_FL_int_mean,
             max2mean_phi_FL_int,min2mean_phi_FL_int,amp_phi_FL_int,
             mins_ind_comp_phi_FL_int,maxs_ind_comp_phi_FL_int]                         = max_min_mean_vals(time,time[nsteps-last_steps::],phi_FL_int[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_phi_FL_plat,mean_max_phi_FL_plat,phi_FL_plat_mean,
             max2mean_phi_FL_plat,min2mean_phi_FL_plat,amp_phi_FL_plat,
             mins_ind_comp_phi_FL_plat,maxs_ind_comp_phi_FL_plat]                         = max_min_mean_vals(time,time[nsteps-last_steps::],phi[i_plot_lat,j_plot_lat,nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_phi_FL_pver,mean_max_phi_FL_pver,phi_FL_pver_mean,
             max2mean_phi_FL_pver,min2mean_phi_FL_pver,amp_phi_FL_pver,
             mins_ind_comp_phi_FL_pver,maxs_ind_comp_phi_FL_pver]                         = max_min_mean_vals(time,time[nsteps-last_steps::],phi[i_plot_ver,j_plot_ver,nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Te_FL_lat,mean_max_Te_FL_lat,Te_FL_lat_mean,
             max2mean_Te_FL_lat,min2mean_Te_FL_lat,amp_Te_FL_lat,
             mins_ind_comp_Te_FL_lat,maxs_ind_comp_Te_FL_lat]                           = max_min_mean_vals(time,time[nsteps-last_steps::],Te_FL_lat[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Te_FL_ver,mean_max_Te_FL_ver,Te_FL_ver_mean,
             max2mean_Te_FL_ver,min2mean_phi_FL_ver,amp_Te_FL_ver,
             mins_ind_comp_Te_FL_ver,maxs_ind_comp_Te_FL_ver]                           = max_min_mean_vals(time,time[nsteps-last_steps::],Te_FL_ver[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Te_FL,mean_max_Te_FL,Te_FL_mean,
             max2mean_Te_FL,min2mean_Te_FL,amp_Te_FL,
             mins_ind_comp_Te_FL,maxs_ind_comp_Te_FL]                                   = max_min_mean_vals(time,time[nsteps-last_steps::],Te_FL[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Te_FL_lat_int,mean_max_Te_FL_lat_int,Te_FL_lat_int_mean,
             max2mean_Te_FL_lat_int,min2mean_Te_FL_lat_int,amp_Te_FL_lat_int,
             mins_ind_comp_Te_FL_lat_int,maxs_ind_comp_Te_FL_lat_int]                   = max_min_mean_vals(time,time[nsteps-last_steps::],Te_FL_lat_int[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Te_FL_ver_int,mean_max_Te_FL_ver_int,Te_FL_ver_int_mean,
             max2mean_Te_FL_ver_int,min2mean_Te_FL_ver_int,amp_Te_FL_ver_int,
             mins_ind_comp_Te_FL_ver_int,maxs_ind_comp_Te_FL_ver_int]                   = max_min_mean_vals(time,time[nsteps-last_steps::],Te_FL_ver_int[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Te_FL_int,mean_max_Te_FL_int,Te_FL_int_mean,
             max2mean_Te_FL_int,min2mean_Te_FL_int,amp_Te_FL_int,
             mins_ind_comp_Te_FL_int,maxs_ind_comp_Te_FL_int]                           = max_min_mean_vals(time,time[nsteps-last_steps::],Te_FL_int[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Te_FL_plat,mean_max_Te_FL_plat,Te_FL_plat_mean,
             max2mean_Te_FL_plat,min2mean_Te_FL_plat,amp_Te_FL_plat,
             mins_ind_comp_Te_FL_plat,maxs_ind_comp_Te_FL_plat]                         = max_min_mean_vals(time,time[nsteps-last_steps::],Te[i_plot_lat,j_plot_lat,nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Te_FL_pver,mean_max_Te_FL_pver,Te_FL_pver_mean,
             max2mean_Te_FL_pver,min2mean_Te_FL_pver,amp_Te_FL_pver,
             mins_ind_comp_Te_FL_pver,maxs_ind_comp_Te_FL_pver]                         = max_min_mean_vals(time,time[nsteps-last_steps::],Te[i_plot_ver,j_plot_ver,nsteps-last_steps::],order)    
        
        
            [_,_,_,_,_,_,
             mean_min_Hall_par_eff_FL_lat_int,mean_max_Hall_par_eff_FL_lat_int,Hall_par_eff_FL_lat_int_mean,
             max2mean_Hall_par_eff_FL_lat_int,min2mean_Hall_par_eff_FL_lat_int,amp_Hall_par_eff_FL_lat_int,
             mins_ind_comp_Hall_par_eff_FL_lat_int,maxs_ind_comp_Hall_par_eff_FL_lat_int]                   = max_min_mean_vals(time,time[nsteps-last_steps::],Hall_par_eff_FL_lat_int[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Hall_par_eff_FL_ver_int,mean_max_Hall_par_eff_FL_ver_int,Hall_par_eff_FL_ver_int_mean,
             max2mean_Hall_par_eff_FL_ver_int,min2mean_Hall_par_eff_FL_ver_int,amp_Hall_par_eff_FL_ver_int,
             mins_ind_comp_Hall_par_eff_FL_ver_int,maxs_ind_comp_Hall_par_eff_FL_ver_int]                   = max_min_mean_vals(time,time[nsteps-last_steps::],Hall_par_eff_FL_ver_int[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Hall_par_eff_FL_int,mean_max_Hall_par_eff_FL_int,Hall_par_eff_FL_int_mean,
             max2mean_Hall_par_eff_FL_int,min2mean_Hall_par_eff_FL_int,amp_Hall_par_eff_FL_int,
             mins_ind_comp_Hall_par_eff_FL_int,maxs_ind_comp_Hall_par_eff_FL_int]                           = max_min_mean_vals(time,time[nsteps-last_steps::],Hall_par_eff_FL_int[nsteps-last_steps::],order)    
            
            [_,_,_,_,_,_,
             mean_min_Hall_par_FL_lat_int,mean_max_Hall_par_FL_lat_int,Hall_par_FL_lat_int_mean,
             max2mean_Hall_par_FL_lat_int,min2mean_Hall_par_FL_lat_int,amp_Hall_par_FL_lat_int,
             mins_ind_comp_Hall_par_FL_lat_int,maxs_ind_comp_Hall_par_FL_lat_int]                   = max_min_mean_vals(time,time[nsteps-last_steps::],Hall_par_FL_lat_int[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Hall_par_FL_ver_int,mean_max_Hall_par_FL_ver_int,Hall_par_FL_ver_int_mean,
             max2mean_Hall_par_FL_ver_int,min2mean_Hall_par_FL_ver_int,amp_Hall_par_FL_ver_int,
             mins_ind_comp_Hall_par_FL_ver_int,maxs_ind_comp_Hall_par_FL_ver_int]                   = max_min_mean_vals(time,time[nsteps-last_steps::],Hall_par_FL_ver_int[nsteps-last_steps::],order)    
            [_,_,_,_,_,_,
             mean_min_Hall_par_FL_int,mean_max_Hall_par_FL_int,Hall_par_FL_int_mean,
             max2mean_Hall_par_FL_int,min2mean_Hall_par_FL_int,amp_Hall_par_FL_int,
             mins_ind_comp_Hall_par_FL_int,maxs_ind_comp_Hall_par_FL_int]                           = max_min_mean_vals(time,time[nsteps-last_steps::],Hall_par_FL_int[nsteps-last_steps::],order)    
            
        
        
        
        
        
        # Obtain time-averaged dphi/Te at free loss
        ratio_DphiTe_FL_lat_mean     = (phi_FL_lat_mean - phi_inf_mean)/Te_FL_lat_mean
        ratio_DphiTe_FL_ver_mean     = (phi_FL_ver_mean - phi_inf_mean)/Te_FL_ver_mean
        ratio_DphiTe_FL_mean         = (phi_FL_mean - phi_inf_mean)/Te_FL_mean
        ratio_DphiTe_FL_lat_int_mean = (phi_FL_lat_int_mean - phi_inf_mean)/Te_FL_lat_int_mean
        ratio_DphiTe_FL_ver_int_mean = (phi_FL_ver_int_mean - phi_inf_mean)/Te_FL_ver_int_mean
        ratio_DphiTe_FL_int_mean     = (phi_FL_int_mean - phi_inf_mean)/Te_FL_int_mean
        ratio_DphiTe_FL_plat_mean    = (phi_FL_plat_mean-phi_inf_mean)/Te_FL_plat_mean
        ratio_DphiTe_FL_pver_mean    = (phi_FL_pver_mean-phi_inf_mean)/Te_FL_pver_mean
        # Obtain time-averaged effective Hall parameters at free loss
        Hall_effect_FL_lat_int_mean  = np.sqrt(Hall_par_eff_FL_lat_int_mean*Hall_par_FL_lat_int_mean)
        Hall_effect_FL_ver_int_mean  = np.sqrt(Hall_par_eff_FL_ver_int_mean*Hall_par_FL_ver_int_mean)
        Hall_effect_FL_int_mean      = np.sqrt(Hall_par_eff_FL_int_mean*Hall_par_FL_int_mean)
        
        
        # Obtain time-averaged performances from time-averaged Id, Ibeam, thrust and ion fluxes
        I_iD = (e/mass)*(mflow_twmat_i1_mean) + (2.0*e/mass)*(mflow_twmat_i2_mean) + (e/mass)*(mflow_twmat_i3_mean) + (2.0*e/mass)*(mflow_twmat_i4_mean)
        I_iA = (e/mass)*(mflow_twa_i1_mean) + (2.0*e/mass)*(mflow_twa_i2_mean) + (e/mass)*(mflow_twa_i3_mean) + (2.0*e/mass)*(mflow_twa_i4_mean)
        I_iC = (e/mass)*(mflow_twcat_i1_mean) + (2.0*e/mass)*(mflow_twcat_i2_mean) + (e/mass)*(mflow_twcat_i3_mean) + (2.0*e/mass)*(mflow_twcat_i4_mean)
        mflow_twinf_itot_mean = mflow_twinf_i1_mean + mflow_twinf_i2_mean + mflow_twinf_i3_mean + mflow_twinf_i4_mean
        mflow_twa_tot_mean    = mflow_twa_i1_mean + mflow_twa_i2_mean + mflow_twa_i3_mean + mflow_twa_i4_mean + mflow_twa_n1_mean + mflow_twa_n2_mean + mflow_twa_n3_mean
        m_A_mean              = mflow_inj_n1_mean - mflow_twa_tot_mean
        eta_u_avg             = mflow_twinf_itot_mean/m_A
        eta_u_avg_2           = mflow_twinf_itot_mean/m_A_mean
        eta_u_avg_3           = mflow_twinf_itot_mean/mflow_inj_n1_mean
        eta_prod_avg          = I_beam_mean/I_tw_tot_mean
        eta_cur_avg           = I_beam_mean/Id_mean 
        eta_ch_avg            = (e/mass)*mflow_twinf_itot_mean/I_beam_mean     
#	    eta_div_avg           = P_use_z_i_mean/P_use_tot_i_mean
        # Using Pd
        eta_div_avg           = P_use_z_mean/P_use_tot_mean
        eta_thr_avg           = thrust_mean**2/(2*m_A*Pd_mean)
        eta_ene_avg           = P_use_tot_mean/Pd_mean
        eta_disp_avg          = thrust_mean**2/(2*m_A*P_use_z_mean)
        # Using total input power (Psource)
        eta_thr_avg_source    = thrust_mean**2/(2*m_A*Psource_mean)
        eta_ene_avg_source    = P_use_tot_mean/Psource_mean
        # Compute eta_div and eta_disp considering only heavy species contributions downstream. This is equivalent to considering that
        # electrons have completely expanded (i.e. Te = 0)
        # eta_ene should remain constant when increasing the plume, since along the expansion, electrons give their energy to ions,
        # so that the total plasma power at crossing the FL boundary is the same
        # With the eta_div and eta_disp values computed only with heavy species, and considering the same eta_ene, we can obtain a 
        # new eta_thr, and from it a new F corresponding to that of a plume with completely expanded electrons
        eta_div_avg_hs        = (P_use_z_i_mean+P_use_z_n_mean)/(P_use_tot_i_mean + P_use_tot_n_mean)
        eta_disp_avg_hs       = (thrust_mean-thrust_e_mean)**2/(2*m_A*(P_use_z_i_mean+P_use_z_n_mean))
        eta_thr_avg_hs        = eta_ene_avg*eta_div_avg_hs*eta_disp_avg_hs
        eta_thr_avg_hs_source = eta_ene_avg_source*eta_div_avg_hs*eta_disp_avg_hs
        F_avg_hs              = np.sqrt(eta_thr_avg_hs*2*m_A*Pd_mean)
        F_avg_hs_source       = np.sqrt(eta_thr_avg_hs_source*2*m_A*Psource_mean)
        
        Isp_s_avg             = thrust_mean/(g0*m_A_mean)
        Isp_ms_avg            = thrust_mean/(m_A_mean)
        Isp_s_avg_mA          = thrust_mean/(g0*m_A)
        Isp_ms_avg_mA         = thrust_mean/(m_A)

        err_balP_avg          = np.abs(Psource_mean - Ploss_mean)/Psource_mean
        err_balP_Pthrust_avg  = np.abs(Psource_mean -  (Pnothrust_mean + Pthrust_mean))/Psource_mean
        ctr_balPthrust_Pd_avg              = (Psource_mean)/(Psource_mean + Pthrust_mean + Pnothrust_mean)
        ctr_balPthrust_Pthrust_avg         = (Pthrust_mean)/(Psource_mean + Pthrust_mean + Pnothrust_mean)
        ctr_balPthrust_Pnothrust_avg       = (Pnothrust_mean)/(Psource_mean + Pthrust_mean + Pnothrust_mean)
        ctr_balPthrust_Pnothrust_walls_avg = (Pnothrust_walls_mean)/(Psource_mean + Pthrust_mean + Pnothrust_mean)
        ctr_balPthrust_Pionex_avg          = (Pionex_mean)/(Psource_mean + Pthrust_mean + Pnothrust_mean)
        ctr_balPthrust_total_avg           = ctr_balPthrust_Pd_avg + ctr_balPthrust_Pthrust_avg + ctr_balPthrust_Pnothrust_avg
#        err_balP_avg          = np.abs(Pd_mean - Ploss_mean)/Pd_mean
#        err_balP_Pthrust_avg  = np.abs(Pd_mean -  (Pnothrust_mean + Pthrust_mean))/Pd_mean
#        ctr_balPthrust_Pd_avg              = (Pd_mean)/(Pd_mean + Pthrust_mean + Pnothrust_mean)
#        ctr_balPthrust_Pthrust_avg         = (Pthrust_mean)/(Pd_mean + Pthrust_mean + Pnothrust_mean)
#        ctr_balPthrust_Pnothrust_avg       = (Pnothrust_mean)/(Pd_mean + Pthrust_mean + Pnothrust_mean)
#        ctr_balPthrust_Pnothrust_walls_avg = (Pnothrust_walls_mean)/(Pd_mean + Pthrust_mean + Pnothrust_mean)
#        ctr_balPthrust_Pionex_avg          = (Pionex_mean)/(Pd_mean + Pthrust_mean + Pnothrust_mean)
#        ctr_balPthrust_total_avg           = ctr_balPthrust_Pd_avg + ctr_balPthrust_Pthrust_avg + ctr_balPthrust_Pnothrust_avg
        
	
	    # Electron energy balance
        if cath_type == 2:
            if Pturb_mean < 0:
                err_Ebal_e    = np.abs(Pfield_e_mean + P_cath_mean + Pturb_mean - (Pionex_mean+Pe_Dwall_mean+Pe_Awall_mean+Pe_FLwall_mean))/(Pfield_e_mean + P_cath_mean)
            else:
                err_Ebal_e    = np.abs(Pfield_e_mean + P_cath_mean + Pturb_mean - (Pionex_mean+Pe_Dwall_mean+Pe_Awall_mean+Pe_FLwall_mean))/(Pfield_e_mean + P_cath_mean + Pturb_mean)
            ctr_Pfield_e  = Pfield_e_mean/(Pfield_e_mean + P_cath_mean + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_P_cath    = P_cath_mean/(Pfield_e_mean + P_cath_mean + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_Pe_Dwall  = Pe_Dwall_mean/(Pfield_e_mean + P_cath_mean + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_Pe_Awall  = Pe_Awall_mean/(Pfield_e_mean + P_cath_mean + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_Pe_FLwall = Pe_FLwall_mean/(Pfield_e_mean + P_cath_mean + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_Pionex    = Pionex_mean/(Pfield_e_mean + P_cath_mean + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_Pturb     = np.abs(Pturb_mean)/(Pfield_e_mean + P_cath_mean + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
        elif cath_type == 1:
            if Pturb_mean < 0:
                err_Ebal_e    = np.abs(Pfield_e_mean + np.abs(Pe_Cwall_mean) + Pturb_mean - (Pionex_mean+Pe_Dwall_mean+Pe_Awall_mean+Pe_FLwall_mean))/(Pfield_e_mean + np.abs(Pe_Cwall_mean))
            else:
                err_Ebal_e    = np.abs(Pfield_e_mean + np.abs(Pe_Cwall_mean) + Pturb_mean - (Pionex_mean+Pe_Dwall_mean+Pe_Awall_mean+Pe_FLwall_mean))/(Pfield_e_mean + np.abs(Pe_Cwall_mean) + Pturb_mean)
            ctr_Pfield_e  = Pfield_e_mean/(Pfield_e_mean + np.abs(Pe_Cwall_mean) + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_P_cath    = np.abs(Pe_Cwall_mean)/(Pfield_e_mean + np.abs(Pe_Cwall_mean) + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_Pe_Dwall  = Pe_Dwall_mean/(Pfield_e_mean + np.abs(Pe_Cwall_mean) + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_Pe_Awall  = Pe_Awall_mean/(Pfield_e_mean + np.abs(Pe_Cwall_mean) + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_Pe_FLwall = Pe_FLwall_mean/(Pfield_e_mean + np.abs(Pe_Cwall_mean) + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_Pionex    = Pionex_mean/(Pfield_e_mean + np.abs(Pe_Cwall_mean) + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
            ctr_Pturb     = np.abs(Pturb_mean)/(Pfield_e_mean + np.abs(Pe_Cwall_mean) + np.abs(Pturb_mean) + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)

#        err_Ebal_e    = np.abs(Pfield_e_mean + P_cath_mean - (Pionex_mean+Pe_Dwall_mean+Pe_Awall_mean+Pe_FLwall_mean))/(Pionex_mean+Pe_Dwall_mean+Pe_Awall_mean+Pe_FLwall_mean)
#        ctr_Pfield_e  = Pfield_e_mean/(Pfield_e_mean + P_cath_mean + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
#        ctr_P_cath    = P_cath_mean/(Pfield_e_mean + P_cath_mean + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
#        ctr_Pe_Dwall  = Pe_Dwall_mean/(Pfield_e_mean + P_cath_mean + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
#        ctr_Pe_Awall  = Pe_Awall_mean/(Pfield_e_mean + P_cath_mean + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
#        ctr_Pe_FLwall = Pe_FLwall_mean/(Pfield_e_mean + P_cath_mean + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
#        ctr_Pionex    = Pionex_mean/(Pfield_e_mean + P_cath_mean + Pionex_mean + Pe_Dwall_mean + Pe_Awall_mean + Pe_FLwall_mean)
        
        # Current balance error in case we have a conducting wall
        if n_cond_wall > 0:
            # Current balance error
            err_I = np.abs(Id + Icond[:,0] - Icath)/np.abs(Icath)
            err_I[err_I == np.Inf] = np.NaN
#            err_I = np.zeros(np.shape(Id))
        else:
            err_I = np.zeros(np.shape(Id))
            
        # Current balance error in case we have GDML at given infinite potential
        if np.any(I_inf != 0):
            err_I_inf = np.abs(Id + I_inf - Icath)/np.abs(Icath)
            err_I_inf[err_I_inf == np.Inf] = np.NaN
        else:
            err_I_inf = np.zeros(np.shape(Id))

            
        # Obtain neutrals mean residence time (using average magnitudes)
        nn1[np.where(nodes_flag == 0)]   = np.nan
        fn1_z[np.where(nodes_flag == 0)] = np.nan
        if make_mean == 1 and mean_type == 0:
            nn1_mean   = np.nanmean(nn1[:,:,nsteps-last_steps::],axis=2)
            fn1_z_mean = np.nanmean(fn1_z[:,:,nsteps-last_steps::],axis=2)
        elif make_mean == 1 and mean_type == 1:
            nn1_mean   = np.nanmean(nn1[:,:,step_i:step_f+1],axis=2)
            fn1_z_mean = np.nanmean(fn1_z[:,:,step_i:step_f+1],axis=2)
        elif make_mean == 1 and mean_type == 2:
            # Obtain the print-out steps corresponding to the fast print-out
            # steps for averaging the neutral density and axial flux. Use the
            # steps obtained for the variable avg_dens_mp_neus
            print_out_mins_ind_avg_dens_mp_neus_real =  mins_ind_comp_avg_dens_mp_neus/50.0
            print_out_mins_ind_avg_dens_mp_neus      = print_out_mins_ind_avg_dens_mp_neus_real.astype(int)
            nn1_mean   = np.nanmean(nn1[:,:,print_out_mins_ind_avg_dens_mp_neus[0]:print_out_mins_ind_avg_dens_mp_neus[-1]+1],axis=2)
            fn1_z_mean = np.nanmean(fn1_z[:,:,print_out_mins_ind_avg_dens_mp_neus[0]:print_out_mins_ind_avg_dens_mp_neus[-1]+1],axis=2)
        if make_mean == 1:
            un1_z_mean = np.divide(fn1_z_mean,nn1_mean)
            eta_min   = int(eta_min)
            eta_max   = int(eta_max)
            xi_bottom = int(xi_bottom)
            xi_top    = int(xi_top)
            # Approach 1: integrate uz radially first
            un1_z_Ir = np.zeros(xi_bottom+1,dtype='float')
            for j in range(0,xi_bottom+1):
                for i in range(eta_min,eta_max):
                    dr = rs[i+1,j] - rs[i,j]
                    un1_z_Ir[j] = un1_z_Ir[j] + 0.5*(un1_z_mean[i,j] + un1_z_mean[i+1,j])*dr
            un1_z_Ir = un1_z_Ir/(rs[eta_max,0]-rs[eta_min,0])
            res_time1 = 0.0
            for j in range(0,xi_bottom):
                dz = zs[eta_min,j+1] - zs[eta_min,j]
                res_time1 = res_time1 + 0.5*(1/un1_z_Ir[j+1] + 1/un1_z_Ir[j])*dz
            res_freq1 = 1.0/res_time1
        
            # Approach 2: integrate uz in the whole chamber
            un1_z_Irz = 0.0
            for j in range(0,xi_bottom):
                dz = zs[eta_min,j+1] - zs[eta_min,j]
                un1_z_Irz = un1_z_Irz + 0.5*(un1_z_Ir[j+1] + un1_z_Ir[j])*dz
            un1_z_Irz = un1_z_Irz/(zs[eta_min,xi_bottom]-zs[eta_min,0])
            res_time2 = (zs[eta_min,xi_bottom]-zs[eta_min,0])/un1_z_Irz
            res_freq2 = 1.0/res_time2
        

     
        ###########################################################################
        print("Plotting...")
        ############################ GENERATING PLOTS #############################
        print("Last step ID    (-)  = %3d" %np.where(eta_u > 0)[0][-1])
        if make_mean == 1 and print_mean_vars == 1:
            print("Mean mi1              (kg)           = %15.8e" %mass_mp_ions1_mean)            
            print("Mean mi2              (kg)           = %15.8e" %mass_mp_ions2_mean)   
            print("Mean mitot            (kg)           = %15.8e" %tot_mass_mp_ions_mean)   
            print("Mean mn               (kg)           = %15.8e" %tot_mass_mp_neus_mean)  
            print("Mean dens_e           (1/m3)         = %15.8e" %avg_dens_mp_ions_mean)
            print("Mean dens_n           (1/m3)         = %15.8e" %avg_dens_mp_neus_mean)
            print("Mean mflow ion inf    (kg/s)         = %15.8e" %mflow_twinf_itot_mean)
            print("mA (input)            (kg/s)         = %15.8e" %m_A)
            print("mA_mean               (kg/s)         = %15.8e" %m_A_mean)
            print("Mean mflow_inj_n      (kg/s)         = %15.8e" %mflow_inj_n1_mean)
            print("Mean mflow_twa (n+i)  (kg/s)         = %15.8e" %mflow_twa_tot_mean)
            print("Mean err_mbal_n1      (-)            = %15.8e" %err_mbal_n1_mean)     
            print("Mean err_mbal_n2      (-)            = %15.8e" %err_mbal_n2_mean)        
            print("Mean err_mbal_n3      (-)            = %15.8e" %err_mbal_n3_mean)        
            print("Mean err_mbal_i1      (-)            = %15.8e" %err_mbal_i1_mean)  
            print("Mean err_mbal_i2      (-)            = %15.8e" %err_mbal_i2_mean) 
            print("Mean err_mbal_i3      (-)            = %15.8e" %err_mbal_i3_mean)  
            print("Mean err_mbal_i4      (-)            = %15.8e" %err_mbal_i4_mean) 
            print("Mean err_mbal_tot     (-)            = %15.8e" %err_mbal_tot_mean) 
            print("Mean Isp              (s)            = %15.8e" %Isp_s_mean)
            print("Mean Isp              (m/s)          = %15.8e" %Isp_ms_mean)
            print("Mean eta_u (input mA) (-)            = %15.8e" %eta_u_mean)   
            print("Mean eta_u_bis(mflows)(-)            = %15.8e" %eta_u_bis_mean)  
            print("Mean eta_prod         (-)            = %15.8e" %eta_prod_mean)   
            print("Mean eta_cur          (-)            = %15.8e" %eta_cur_mean)  
            print("Mean eta_div          (-)            = %15.8e" %eta_div_mean)  
            print("Mean eta_thr          (-)            = %15.8e" %eta_thr_mean)  
            print("Mean thrust           (N)            = %15.8e" %thrust_mean) 
            print("Mean thrust i1        (N)            = %15.8e" %thrust_i1_mean) 
            print("Mean thrust i2        (N)            = %15.8e" %thrust_i2_mean) 
            print("Mean thrust i3        (N)            = %15.8e" %thrust_i3_mean) 
            print("Mean thrust i4        (N)            = %15.8e" %thrust_i4_mean) 
            print("Mean thrust n1        (N)            = %15.8e" %thrust_n1_mean) 
            print("Mean thrust n2        (N)            = %15.8e" %thrust_n2_mean) 
            print("Mean thrust n3        (N)            = %15.8e" %thrust_n3_mean) 
            print("Mean thrust e         (N)            = %15.8e" %thrust_e_mean) 
            print("Mean Pd               (W)            = %15.8e" %Pd_mean) 
            print("Mean P_mat            (W)            = %15.8e" %P_mat_mean) 
            print("Mean P_mat_hs         (W)            = %15.8e" %P_mat_hs_mean)
            print("Mean P_inj            (W)            = %15.8e" %P_inj_mean) 
            print("Mean P_inj_hs         (W)            = %15.8e" %P_inj_hs_mean) 
            print("Mean P_inf            (W)            = %15.8e" %P_inf_mean) 
            print("Mean P_use_tot        (W)            = %15.8e" %P_use_tot_mean) 
            print("Mean P_use_z          (W)            = %15.8e" %P_use_z_mean)
            print("Mean P_use_tot_i      (W)            = %15.8e" %P_use_tot_i_mean) 
            print("Mean P_use_tot_n      (W)            = %15.8e" %P_use_tot_n_mean) 
            print("Mean P_use_z_i        (W)            = %15.8e" %P_use_z_i_mean)
            print("Mean P_use_z_n        (W)            = %15.8e" %P_use_z_n_mean)
            print("Mean P_ion            (W)            = %15.8e" %P_ion_mean) 
            print("Mean P_ex             (W)            = %15.8e" %P_ex_mean)
            print("Mean P_cath           (W)            = %15.8e" %P_cath_mean)
            print("Mean Pturb            (W)            = %15.8e" %Pturb_mean)
            print("Mean nu_cath          (Hz)           = %15.8e" %nu_cath_mean)
            print("Mean Id               (A)            = %15.8e" %Id_mean) 
            print("Mean Id inst          (A)            = %15.8e" %Id_inst_mean) 
            print("Mean I_beam           (A)            = %15.8e" %I_beam_mean) 
            print("Mean I_prod (I_tw_tot) (A)           = %15.8e" %I_tw_tot_mean)
            print("Mean I_wi (I_tw_tot-I_beam) (A)      = %15.8e" %(I_tw_tot_mean-I_beam_mean))
            print("Mean res_time1        (s)            = %15.8e" %res_time1)
            print("Mean res_time2        (s)            = %15.8e" %res_time2)
            print("Mean res_freq1        (Hz)           = %15.8e" %res_freq1)
            print("Mean res_freq2        (Hz)           = %15.8e" %res_freq2)
            print("Max. Te_mean_dom FFT freq (Hz)       = %15.8e" %max_freq_Te_mean_dom)
            print("Max. Id FFT freq      (Hz)           = %15.8e" %max_freq_Id)
            print("Max. Id inst FFT freq (Hz)           = %15.8e" %max_freq_Id_inst)
            print("Max. I_beam FFT freq  (Hz)           = %15.8e" %max_freq_I_beam)
            print("Max. dens_e FFT freq  (Hz)           = %15.8e" %max_freq_avg_dens_mp_ions)
            print("Max. dens_n FFT freq  (Hz)           = %15.8e" %max_freq_avg_dens_mp_neus)
            print("Max. nu_cath FFT freq (Hz)           = %15.8e" %max_freq_nu_cath)
            print("Max. P_cath FFT freq  (Hz)           = %15.8e" %max_freq_P_cath)
            print("Phase shift Id-Te_mean_dom  (deg)    = %15.8e" %phase_shift_IdTe_mean_dom_deg)
            print("Phase shift Id-Ibeam  (deg)          = %15.8e" %phase_shift_IdIbeam_deg)
            print("Phase shift avg_dens neus-ions (deg) = %15.8e" %phase_shift_avg_dens_mp_neusions_deg)
            print("Phase shift ctr_mbal_tot (deg)       = %15.8e" %phase_shift_ctr_mbal_tot_deg)
            print("##### Time-average eff. #####")
            print("Avg Isp (input mA)    (s)            = %15.8e" %Isp_s_avg_mA)
            print("Avg Isp (input mA)    (ms)           = %15.8e" %Isp_ms_avg_mA)
            print("Avg Isp (mean mA)     (s)            = %15.8e" %Isp_s_avg)
            print("Avg Isp (mean mA)     (m/s)          = %15.8e" %Isp_ms_avg)
            print("Avg eta_u (input mA)  (-)            = %15.8e" %eta_u_avg)   
            print("Avg eta_u (mean mA)   (-)            = %15.8e" %eta_u_avg_2)
            print("Avg eta_u (mflow_inj) (-)            = %15.8e" %eta_u_avg_3) 
            print("Avg eta_prod          (-)            = %15.8e" %eta_prod_avg)   
            print("Avg eta_cur           (-)            = %15.8e" %eta_cur_avg) 
            print("Avg eta_ene           (-)            = %15.8e" %eta_ene_avg) 
            print("Avg eta_disp          (-)            = %15.8e" %eta_disp_avg) 
            print("Avg eta_div           (-)            = %15.8e" %eta_div_avg) 
            print("Avg eta_thr           (-)            = %15.8e" %eta_thr_avg) 
            print("#### Total power balance ####")
            print("Mean Psource          (W)            = %15.8e" %Psource_mean)
            print("Mean Ploss            (W)            = %15.8e" %Ploss_mean) 
#            print("Mean Ploss            (W)            = %15.8e" %(P_Dwall_mean + P_Awall_mean + P_FLwall_mean + P_ion_mean + P_ex_mean)) 
#            print("Mean err Pbal         (-)            = %15.8e" %(np.abs(Pd_mean - (P_Dwall_mean + P_Awall_mean + P_FLwall_mean + P_ion_mean + P_ex_mean))/Pd_mean)) 
            print("err_balP_avg          (-)            = %15.8e" %err_balP_avg) 
            print("err_balP_Pthrust_avg  (-)            = %15.8e" %err_balP_Pthrust_avg)   
            print("Mean Pd               (W)            = %15.8e" %Pd_mean) 
            print("Mean P_cath           (W)            = %15.8e" %P_cath_mean)
            print("Mean Pturb            (W)            = %15.8e" %Pturb_mean) 
            print("Mean Pwalls           (W)            = %15.8e" %(P_Dwall_mean + P_Awall_mean + P_FLwall_mean)) 
            print("Mean P_ion_ex         (W)            = %15.8e" %(P_ion_mean + P_ex_mean)) 
            print("Mean Pnothrust        (W)            = %15.8e" %Pnothrust_mean)
            print("Mean Pnothrust_walls  (W)            = %15.8e" %Pnothrust_walls_mean)
            print("Mean Pthrust          (W)            = %15.8e" %Pthrust_mean)
            print("Puse tot (i+n)        (W)            = %15.8e" %(P_use_tot_i_mean + P_use_tot_n_mean))
            print("Puse axial (i+n)      (W)            = %15.8e" %(P_use_z_i_mean + P_use_z_n_mean))
            print("P_Dwall e             (W)            = %15.8e" %(Pe_Dwall_mean))
            print("P_Dwall hs            (W)            = %15.8e" %(Pi_Dwall_mean + Pn_Dwall_mean))
            print("P_Awall e             (W)            = %15.8e" %(Pe_Awall_mean))
            print("P_Awall hs            (W)            = %15.8e" %(Pi_Awall_mean + Pn_Awall_mean))
            print("P_Cwall e             (W)            = %15.8e" %(Pe_Cwall_mean))
            print("P_Cwall hs            (W)            = %15.8e" %(Pi_Cwall_mean + Pn_Cwall_mean))
            print("P_FLwall e            (W)            = %15.8e" %(P_inf_mean - (Pi_FLwall_mean + Pn_FLwall_mean)))
            print("P_FLwall hs           (W)            = %15.8e" %(Pi_FLwall_mean + Pn_FLwall_mean))
            print("Pthrust_mean/Pd_mean  (-)            = %15.8e" %(Pthrust_mean/Pd_mean))
            print("ctr_balPthrust_Pd_avg              (-) = %15.8e" %ctr_balPthrust_Pd_avg)
            print("ctr_balPthrust_Pthrust_avg         (-) = %15.8e" %ctr_balPthrust_Pthrust_avg)
            print("ctr_balPthrust_Pnothrust_avg       (-) = %15.8e" %ctr_balPthrust_Pnothrust_avg)
            print("ctr_balPthrust_Pnothrust_walls_avg (-) = %15.8e" %ctr_balPthrust_Pnothrust_walls_avg)
            print("ctr_balPthrust_Pionex_avg          (-) = %15.8e" %ctr_balPthrust_Pionex_avg)
            print("ctr_balPthrust_total_avg           (-) = %15.8e" %ctr_balPthrust_total_avg)
            print("##### Signals max,min,ratios,amp #####")
            print("mean_min_ne        (1/m3) = %15.8e" %mean_min_avg_dens_mp_ions)
            print("mean_max_ne        (1/m3) = %15.8e" %mean_max_avg_dens_mp_ions)
            print("ne_mean            (1/m3) = %15.8e" %avg_dens_mp_ions_mean)
            print("max2mean_ne           (-) = %15.8e" %max2mean_avg_dens_mp_ions)
            print("min2mean_ne           (-) = %15.8e" %min2mean_avg_dens_mp_ions)
            print("amp_ne                (-) = %15.8e" %amp_avg_dens_mp_ions)
            print("mean_min_nn        (1/m3) = %15.8e" %mean_min_avg_dens_mp_neus)
            print("mean_max_nn        (1/m3) = %15.8e" %mean_max_avg_dens_mp_neus)
            print("nn_mean            (1/m3) = %15.8e" %avg_dens_mp_neus_mean)
            print("max2mean_nn           (-) = %15.8e" %max2mean_avg_dens_mp_neus)
            print("min2mean_nn           (-) = %15.8e" %min2mean_avg_dens_mp_neus)
            print("amp_nn                (-) = %15.8e" %amp_avg_dens_mp_neus)
            print("mean_min_Te_mean_dom (eV) = %15.8e" %mean_min_Te_mean_dom)
            print("mean_max_Te_mean_dom (eV) = %15.8e" %mean_max_Te_mean_dom)
            print("Te_mean_dom_mean     (eV) = %15.8e" %Te_mean_dom_mean)
            print("max2mean_Te_mean_dom  (-) = %15.8e" %max2mean_Te_mean_dom)
            print("min2mean_Te_mean_dom  (-) = %15.8e" %min2mean_Te_mean_dom)
            print("amp_Te_mean_dom       (-) = %15.8e" %amp_Te_mean_dom)
            print("mean_min_Id           (A) = %15.8e" %mean_min_Id)
            print("mean_max_Id           (A) = %15.8e" %mean_max_Id)
            print("Id_mean               (A) = %15.8e" %Id_mean)
            print("max2mean_Id           (-) = %15.8e" %max2mean_Id)
            print("min2mean_Id           (-) = %15.8e" %min2mean_Id)
            print("amp_Id                (-) = %15.8e" %amp_Id)
            print("mean_min_I_beam       (A) = %15.8e" %mean_min_I_beam)
            print("mean_max_I_beam       (A) = %15.8e" %mean_max_I_beam)
            print("I_beam_mean           (A) = %15.8e" %I_beam_mean)
            print("max2mean_I_beam       (-) = %15.8e" %max2mean_I_beam)
            print("min2mean_I_beam       (-) = %15.8e" %min2mean_I_beam)
            print("amp_I_beam            (-) = %15.8e" %amp_I_beam)

            print("##### Electron energy balance #####")
            print("err_Ebal_e            (-)            = %15.8e" %(err_Ebal_e)) 
            if cath_type == 2:
                if Pturb_mean < 0:
                    print("Psources (f+c)        (W)            = %15.8e" %(Pfield_e_mean+P_cath_mean)) 
                    print("Psinks (i+w+t)        (W)            = %15.8e" %(Pionex_mean+Pe_Dwall_mean+Pe_Awall_mean+Pe_FLwall_mean+np.abs(Pturb_mean))) 
                else:
                    print("Psources (f+c+t)      (W)            = %15.8e" %(Pfield_e_mean+P_cath_mean+Pturb_mean)) 
                    print("Psinks (i+w)          (W)            = %15.8e" %(Pionex_mean+Pe_Dwall_mean+Pe_Awall_mean+Pe_FLwall_mean)) 
            elif cath_type == 1:
                if Pturb_mean < 0:
                    print("Psources (f+c)        (W)            = %15.8e" %(Pfield_e_mean+np.abs(Pe_Cwall_mean))) 
                    print("Psinks (i+w+t)        (W)            = %15.8e" %(Pionex_mean+Pe_Dwall_mean+Pe_Awall_mean+Pe_FLwall_mean+np.abs(Pturb_mean))) 
                else:
                    print("Psources (f+c+t)      (W)            = %15.8e" %(Pfield_e_mean+np.abs(Pe_Cwall_mean)+Pturb_mean)) 
                    print("Psinks (i+w)          (W)            = %15.8e" %(Pionex_mean+Pe_Dwall_mean+Pe_Awall_mean+Pe_FLwall_mean)) 
            print("Pfield_e              (W)            = %15.8e" %(Pfield_e_mean)) 
            if cath_type == 2:
                print("P_cath                (W)            = %15.8e" %(P_cath_mean))
            elif cath_type == 1:
                print("P_cath                (W)            = %15.8e" %(np.abs(Pe_Cwall_mean)))
            print("Pturb                 (W)            = %15.8e" %(Pturb_mean)) 
            print("P_Dwall e             (W)            = %15.8e" %(Pe_Dwall_mean))
            print("P_Awall e             (W)            = %15.8e" %(Pe_Awall_mean))
            print("P_Cwall e             (W)            = %15.8e" %(Pe_Cwall_mean))
            print("P_FLwall e            (W)            = %15.8e" %(Pe_FLwall_mean))
            print("P_ionex               (W)            = %15.8e" %(Pionex_mean)) 
            print("ctr_Pfield_e          (-)            = %15.8e" %(ctr_Pfield_e)) 
            print("ctr_P_cath            (-)            = %15.8e" %(ctr_P_cath)) 
            print("ctr_Pe_Dwall          (-)            = %15.8e" %(ctr_Pe_Dwall)) 
            print("ctr_Pe_Awall          (-)            = %15.8e" %(ctr_Pe_Awall)) 
            print("ctr_Pe_FLwall         (-)            = %15.8e" %(ctr_Pe_FLwall)) 
            print("ctr_Pionex            (-)            = %15.8e" %(ctr_Pionex)) 
            print("ctr_Pturb             (-)            = %15.8e" %(ctr_Pturb)) 
            print("sum_ctr               (-)            = %15.8e" %(ctr_Pfield_e+ctr_P_cath+ctr_Pe_Dwall+ctr_Pe_Awall+ctr_Pe_FLwall+ctr_Pionex+ctr_Pturb)) 
            print("##### Summary of performances #####")
            print("Id_mean               (A) = %15.8e" %Id_mean)
            print("Min Id                (A) = %15.8e" %mean_min_Id)
            print("Max Id                (A) = %15.8e" %mean_max_Id)
            print("max2mean_Id           (-) = %15.8e" %max2mean_Id)
            print("min2mean_Id           (-) = %15.8e" %min2mean_Id)
            print("amp_Id                (-) = %15.8e" %amp_Id)
            print("Max. Id FFT freq      (Hz)= %15.8e" %max_freq_Id)
            for pindex in range(0, num_firstmax_print):
                print(str(pindex+2)+" Max. Id FFT freq    (Hz)= %15.8e" %maxs_freq_Id[pindex+1])
            print("Mean Pd               (W) = %15.8e" %Pd_mean)
            print("Min Pd                (W) = %15.8e" %mean_min_Pd)
            print("Max Pd                (W) = %15.8e" %mean_max_Pd)
            print("Mean Psource          (W) = %15.8e" %Psource_mean)
            print("Min Psource           (W) = %15.8e" %mean_min_Psource)
            print("Max Psource           (W) = %15.8e" %mean_max_Psource)
            print("Avg Isp (input mA)    (s) = %15.8e" %Isp_s_avg_mA)
            print("Min Isp               (s) = %15.8e" %(mean_min_thrust/(m_A*g0)))
            print("Max Isp               (s) = %15.8e" %(mean_max_thrust/(m_A*g0)))
            print("Avg eta_u (input mA)  (-) = %15.8e" %eta_u_avg)
            print("Avg eta_prod          (-) = %15.8e" %eta_prod_avg)   
            print("Avg eta_cur           (-) = %15.8e" %eta_cur_avg) 
            print("Avg eta_ch            (-) = %15.8e" %eta_ch_avg) 
            print("Avg eta_ene           (-) = %15.8e" %eta_ene_avg) 
            print("Avg eta_ene_source    (-) = %15.8e" %eta_ene_avg_source) 
            print("Avg eta_disp          (-) = %15.8e" %eta_disp_avg) 
            print("Avg eta_div           (-) = %15.8e" %eta_div_avg) 
            print("Avg eta_thr           (-) = %15.8e" %eta_thr_avg) 
            print("Avg eta_thr_source    (-) = %15.8e" %eta_thr_avg_source)      
            print("Avg eta_div_hs        (-) = %15.8e" %eta_div_avg_hs) 
            print("Avg eta_disp_hs       (-) = %15.8e" %eta_disp_avg_hs) 
            print("Avg eta_thr_hs        (-) = %15.8e" %eta_thr_avg_hs)
            print("Avg eta_thr_hs_source (-) = %15.8e" %eta_thr_avg_hs_source)
            print("Mean thrust hs        (mN)= %15.8e" %(F_avg_hs*1e3)) 
            print("Mean thrust hs source (mN)= %15.8e" %(F_avg_hs_source*1e3)) 
            print("Pthrust/Pd            (-) = %15.8e" %(Pthrust_mean/Pd_mean))
            print("Pthrust/Psource       (-) = %15.8e" %(Pthrust_mean/Psource_mean))
            print("Mean thrust           (mN)= %15.8e" %(thrust_mean*1e3)) 
            print("Min thrust            (mN)= %15.8e" %(mean_min_thrust*1e3)) 
            print("Max thrust            (mN)= %15.8e" %(mean_max_thrust*1e3)) 
            print("Icath_mean            (A) = %15.8e" %Icath_mean)
            print("Icond_mean            (A) = %15.8e" %Icond_mean)
            print("Vcond_mean            (V) = %15.8e" %Vcond_mean)
            print("I_iD/I_tw_tot         (-) = %15.8e" %(I_iD/I_tw_tot_mean))
            print("I_iA/I_tw_tot         (-) = %15.8e" %(I_iA/I_tw_tot_mean))
            print("I_iC/I_tw_tot         (-) = %15.8e" %(I_iC/I_tw_tot_mean))
            print("I_iinf/I_tw_tot       (-) = %15.8e" %(I_beam_mean/I_tw_tot_mean))
            print("sum(I)/I_tw_tot       (-) = %15.8e" %((I_iD+I_iA+I_iC+I_beam_mean)/I_tw_tot_mean))
            print("P_ion_ex/P_source     (-) = %15.8e" %((P_ion_mean + P_ex_mean)/Psource_mean)) 
            print("P_D/P_source          (-) = %15.8e" %(P_Dwall_mean/Psource_mean)) 
            print("P_A/P_source          (-) = %15.8e" %(P_Awall_mean/Psource_mean))
            print("P_FL/P_source         (-) = %15.8e" %(P_FLwall_mean/Psource_mean)) 
            print("Pturb/P_source        (-) = %15.8e" %(np.abs(Pturb_mean)/Psource_mean)) 
            print("sum(P)/P_source       (-) = %15.8e" %((P_ion_mean+P_ex_mean+P_Dwall_mean+P_Awall_mean+P_FLwall_mean+np.abs(Pturb_mean))/Psource_mean))
            print("##### Variables at FL #####")
            print("Dphi/Te               (-) = %15.8e" %ratio_DphiTe_FL_mean)
            print("Dphi/Te int           (-) = %15.8e" %ratio_DphiTe_FL_int_mean)
            print("Dphi/Te ver           (-) = %15.8e" %ratio_DphiTe_FL_ver_mean)
            print("Dphi/Te ver int       (-) = %15.8e" %ratio_DphiTe_FL_ver_int_mean)
            print("Dphi/Te pver          (-) = %15.8e" %ratio_DphiTe_FL_pver_mean)
            print("Dphi/Te lat           (-) = %15.8e" %ratio_DphiTe_FL_lat_mean)
            print("Dphi/Te lat int       (-) = %15.8e" %ratio_DphiTe_FL_lat_int_mean)
            print("Dphi/Te plat          (-) = %15.8e" %ratio_DphiTe_FL_plat_mean)
            print("Te                    (eV)= %15.8e" %Te_FL_mean)
            print("Te int                (eV)= %15.8e" %Te_FL_int_mean)
            print("Te ver                (eV)= %15.8e" %Te_FL_ver_mean)
            print("Te ver int            (eV)= %15.8e" %Te_FL_ver_int_mean)
            print("Te pver (i,j)=(%d,%d) (eV)= %15.8e" %(i_plot_ver,j_plot_ver,Te_FL_pver_mean))
            print("Te lat                (eV)= %15.8e" %Te_FL_lat_mean)
            print("Te lat int            (eV)= %15.8e" %Te_FL_lat_int_mean)
            print("Te plat (i,j)=(%d,%d) (eV)= %15.8e" %(i_plot_lat,j_plot_lat,Te_FL_plat_mean))
            print("phi                   (V) = %15.8e" %phi_FL_mean)
            print("phi int               (V) = %15.8e" %phi_FL_int_mean)
            print("phi ver               (V) = %15.8e" %phi_FL_ver_mean)
            print("phi ver int           (V) = %15.8e" %phi_FL_ver_int_mean)
            print("phi pver (i,j)=(%d,%d)(V) = %15.8e" %(i_plot_ver,j_plot_ver,phi_FL_pver_mean))
            print("phi lat               (V) = %15.8e" %phi_FL_lat_mean)
            print("phi lat int           (V) = %15.8e" %phi_FL_lat_int_mean)
            print("phi plat (i,j)=(%d,%d)(V) = %15.8e" %(i_plot_lat,j_plot_lat,phi_FL_plat_mean))
            print("phi_inf               (V) = %15.8e" %phi_inf_mean)
            print("I_inf                 (A) = %15.8e" %I_inf_mean)
            print("Hall eff int          (-) = %15.8e" %Hall_par_eff_FL_int_mean)
            print("Hall eff ver int      (-) = %15.8e" %Hall_par_eff_FL_ver_int_mean)
            print("Hall eff lat int      (-) = %15.8e" %Hall_par_eff_FL_lat_int_mean)
            print("Hall int              (-) = %15.8e" %Hall_par_FL_int_mean)
            print("Hall ver int          (-) = %15.8e" %Hall_par_FL_ver_int_mean)
            print("Hall lat int          (-) = %15.8e" %Hall_par_FL_lat_int_mean)
            print("Hall effect int       (-) = %15.8e" %Hall_effect_FL_int_mean)
            print("Hall effect ver int   (-) = %15.8e" %Hall_effect_FL_ver_int_mean)
            print("Hall effect lat int   (-) = %15.8e" %Hall_effect_FL_lat_int_mean)

            
            if exp_data_time_plots == 1:
                print("##### Experimental data #####")
                print("Id_mean               (A) = %15.8e" %exp_Id_mean)
                print("Min Id                (A) = %15.8e" %mean_min_exp_Id)
                print("Max Id                (A) = %15.8e" %mean_max_exp_Id)
                print("max2mean_Id           (-) = %15.8e" %max2mean_exp_Id)
                print("min2mean_Id           (-) = %15.8e" %min2mean_exp_Id)
                print("amp_Id                (-) = %15.8e" %amp_exp_Id)
                print("Max. Id FFT freq      (Hz)= %15.8e" %max_freq_exp_Id)
                for pindex in range(0, num_firstmax_print):
                    print(str(pindex+2)+" Max. Id FFT freq    (Hz)= %15.8e" %maxs_freq_exp_Id[pindex+1])
                print("Pd_mean               (W) = %15.8e" %exp_Pd_mean)
                print("Min Pd                (W) = %15.8e" %mean_min_exp_Pd)
                print("Max Pd                (W) = %15.8e" %mean_max_exp_Pd)
        
        
        # Update dictionary
        dic_data['Vd (V)'].append(Vd[-1])
        dic_data['mA (mg/s)'].append(m_A/1.12*1E6)
        dic_data['mC (mg/s)'].append((m_A - m_A/1.12)*1E6)
        dic_data['Id (A)'].append(Id_mean)
        
#        if max_freq_Id < 1E5:
#            fd_max = max_freq_Id
#        else:
#            for pindex in range(0,num_firstmax_print):
#                if maxs_freq_Id[pindex+1] < 1E5 and maxs_freq_Id[pindex+1] > 1.5E4:
#                    fd_max = maxs_freq_Id[pindex+1]
#        dic_data['fd_max (kHz)'].append(fd_max*1E-3)
        
        dic_data['fd_max (kHz)'].append(max_freq_Id*1E-3)
        dic_data['fd_2 (kHz)'].append(maxs_freq_Id[2]*1E-3)
        dic_data['fd_3 (kHz)'].append(maxs_freq_Id[3]*1E-3)
        dic_data['fd_4 (kHz)'].append(maxs_freq_Id[4]*1E-3)
        dic_data['fd_5 (kHz)'].append(maxs_freq_Id[5]*1E-3)
        dic_data['fd_6 (kHz)'].append(maxs_freq_Id[6]*1E-3)
        dic_data['Id half amp. (%)'].append(100*amp_Id/2.0)
        dic_data['F (mN)'].append(thrust_mean*1E3)
        dic_data['Isp (s)'].append(Isp_s_avg_mA) 
        dic_data['eta'].append(eta_thr_avg_source)
        dic_data['eta_ene'].append(eta_ene_avg_source)
        dic_data['eta_div'].append(eta_div_avg)
        dic_data['eta_disp'].append(eta_disp_avg)
        dic_data['eta_cur'].append(eta_cur_avg)
        dic_data['eta_vol'].append(eta_ene_avg_source/eta_cur_avg)
        dic_data['eta_u'].append(eta_u_avg)
        dic_data['eta_ch'].append(eta_ch_avg)
        dic_data['eta_prod'].append(eta_prod_avg)
        dic_data['P (W)'].append(Psource_mean)
        dic_data['P_A/P'].append(P_Awall_mean/Psource_mean)
        dic_data['P_D/P'].append(P_Dwall_mean/Psource_mean)
        dic_data['Iprod (A)'].append(I_tw_tot_mean)
        dic_data['I_A/Iprod'].append(I_iA/I_tw_tot_mean)
        dic_data['I_D/I_prod'].append(I_iD/I_tw_tot_mean)

        # Time in ms
        time       = time*1e3
        time_fast  = time_fast*1e3
        steps      = steps*1e-3
        steps_fast = steps_fast*1e-3
        prntsteps_ID = np.linspace(0,nsteps-1,nsteps)
        tot_mass_mp_ions   = tot_mass_mp_ions*1e11
        tot_mass_mp_neus   = tot_mass_mp_neus*1e11
        tot_mass_exit_ions = tot_mass_exit_ions*1e11
        tot_mass_exit_neus = tot_mass_exit_neus*1e11
        mass_mp_ions       = mass_mp_ions*1e11
        mass_mp_neus       = mass_mp_neus*1e11
        tot_num_mp_ions    = tot_num_mp_ions*1e-6
        tot_num_mp_neus    = tot_num_mp_neus*1e-6
        num_mp_ions        = num_mp_ions*1e-6
        num_mp_neus        = num_mp_neus*1e-6
        thrust             = thrust*1e3
        thrust_ion         = thrust_ion*1e3
        thrust_neu         = thrust_neu*1e3
        mbal_n1            = mbal_n1*1E6
        dMdt_n1            = dMdt_n1*1E6
        mbal_i1            = mbal_i1*1E6
        dMdt_i1            = dMdt_i1*1E6
        mbal_i2            = mbal_i2*1E6
        dMdt_i2            = dMdt_i2*1E6
        
        
        if plot_mass == 1:
            # Plot the time evolution of the ions 1 mass
            plt.figure(r'mi1(t)')
            if time2steps_axis == 1:
                plt.plot(steps_fast, mass_mp_ions[:,0], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time_fast, mass_mp_ions[:,0], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the ions 2 mass
            plt.figure(r'mi2(t)')
            if time2steps_axis == 1:
                plt.plot(steps_fast, mass_mp_ions[:,1], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:   
                plt.plot(time_fast, mass_mp_ions[:,1], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the ion total mass
            plt.figure(r'mitot(t)')
            if time2steps_axis == 1:
                plt.plot(steps_fast, tot_mass_mp_ions, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
            else:
                plt.plot(time_fast, tot_mass_mp_ions, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
            # Plot the time evolution of the neutral mass
            plt.figure(r'mn(t)')
            if time2steps_axis == 1:
                plt.plot(steps_fast, tot_mass_mp_neus, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:     
                plt.plot(time_fast, tot_mass_mp_neus, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        if plot_dens == 1:
            # Plot the time evolution of the average plasma density in the domain
            plt.figure(r'dens_e(t)')
            if time2steps_axis == 1:
                plt.semilogy(steps_fast, avg_dens_mp_ions, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
            else:
#                plt.semilogy(time_fast, avg_dens_mp_ions, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])      
                plt.semilogy(time_fast, avg_dens_mp_ions, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the average neutral density in the domain
            plt.figure(r'dens_n(t)')
            if time2steps_axis == 1:
                plt.semilogy(steps_fast, avg_dens_mp_neus, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
            else:
#                plt.semilogy(time_fast, avg_dens_mp_neus, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])  
                plt.semilogy(time_fast, avg_dens_mp_neus, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k]) 
            # Plot the FFT plot for the average plasma density in the domain
            plt.figure(r'FFT dens_e(t)')
            #plt.semilogx(freq_avg_dens_mp_ions[1:], np.abs(fft_avg_dens_mp_ions[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the FFT plot for the average neutral density in the domain
            plt.figure(r'FFT dens_n(t)')     
            #plt.semilogx(freq_avg_dens_mp_neus[1:], np.abs(fft_avg_dens_mp_neus[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of both the average plasma and neutral density in the domain
            plt.figure(r'dens_e_dens_n(t)')
            if time2steps_axis == 1:
                plt.semilogy(steps_fast, avg_dens_mp_ions, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
                plt.semilogy(steps_fast, avg_dens_mp_neus, linestyle='--', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            
            
            else:
                plt.semilogy(time_fast, avg_dens_mp_ions, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
                plt.semilogy(time_fast, avg_dens_mp_neus, linestyle='--', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            
#                for i in range(0,len(fast_prntstep_IDs)):
#                    plt.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_ions[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#                    plt.text(fact_x[i]*time_fast[fast_prntstep_IDs[i]-plot_tol], fact_y[i]*avg_dens_mp_ions[fast_prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')     
#                    plt.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_neus[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#                    plt.text(fact_x[i]*time_fast[fast_prntstep_IDs[i]-plot_tol], fact_y[i]*avg_dens_mp_neus[fast_prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')                         
#                    print(prntstep_IDs_text[i]+" time_fast = "+str(time_fast[fast_prntstep_IDs[i]])+", time = "+str(time[prntstep_IDs[i]]))
            # Plot the FFT plot for both the average plasma and neutral density in the domain
            plt.figure(r'FFT dens_e_dens_n(t)')   
            #plt.semilogx(freq_avg_dens_mp_ions[1:], np.abs(fft_avg_dens_mp_ions[1:]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            #plt.semilogx(freq_avg_dens_mp_neus[1:], np.abs(fft_avg_dens_mp_neus[1:]), linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")
            # Plot the time evolution of both the average plasma and neutral density in the domain (normalized)
            plt.figure(r'norm_dens_e_dens_n(t)')  
            #if time2steps_axis == 1:
                #plt.semilogy(steps_fast, avg_dens_mp_ions/avg_dens_mp_ions_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
                #plt.semilogy(steps_fast, avg_dens_mp_neus/avg_dens_mp_neus_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            
            
            #else:
                #plt.semilogy(time_fast, avg_dens_mp_ions/avg_dens_mp_ions_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
                #plt.semilogy(time_fast, avg_dens_mp_neus/avg_dens_mp_neus_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            

        if plot_Nmp == 1:
            # Plot the time evolution of the ions 1 number of particles
            plt.figure(r'Nmpi1(t)')
            if time2steps_axis == 1:
                plt.plot(steps_fast, num_mp_ions[:,0], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time_fast, num_mp_ions[:,0], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the ions 2 number of particles
            plt.figure(r'Nmpi2(t)')
            if time2steps_axis == 1:
                plt.plot(steps_fast, num_mp_ions[:,1], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time_fast, num_mp_ions[:,1], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the total ion number of particles
            plt.figure(r'Nmpitot(t)')
            if time2steps_axis == 1:
                plt.plot(steps_fast, tot_num_mp_ions, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time_fast, tot_num_mp_ions, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the neutral number of particles
            plt.figure(r'Nmpn(t)')
            if time2steps_axis == 1:
                plt.plot(steps_fast, tot_num_mp_neus, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time_fast, tot_num_mp_neus, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        if plot_eff == 1:
            # Plot the time evolution of the utilization efficiency
            plt.figure(r'eta_u(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, eta_u, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:    
                plt.semilogy(prntsteps_ID, eta_u, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.semilogy(time, eta_u, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the production efficiency
            plt.figure(r'eta_prod(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, eta_prod, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, eta_prod, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.semilogy(time, eta_prod, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the current efficiency
            plt.figure(r'eta_cur(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, eta_cur, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, eta_cur, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.semilogy(time, eta_cur, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                plt.semilogy(time[nsteps-last_steps::], eta_cur_mean*np.ones(np.shape(time[nsteps-last_steps::])), linestyle=linestyles[ind3], linewidth = line_width-1, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")              
            # Plot the time evolution of the divergence efficiency
            plt.figure(r'eta_div(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, eta_div, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, eta_div, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.semilogy(time, eta_div, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the thrust efficiency
            plt.figure(r'eta_thr(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, eta_thr, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, eta_thr, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.semilogy(time, eta_thr, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        if plot_thr == 1:
            # Plot the time evolution of the total thrust
            plt.figure(r'T(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, thrust, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, thrust, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.semilogy(time, thrust, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
            # Plot the time evolution of the ions 1 thrust
            plt.figure(r'Ti1(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, thrust_ion[:,0], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, thrust_ion[:,0], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, thrust_ion[:,0], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the ions 2 thrust
            plt.figure(r'Ti2(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, thrust_ion[:,1], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, thrust_ion[:,1], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, thrust_ion[:,1], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the neutral thrust
            plt.figure(r'Tn(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, thrust_neu, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, thrust_neu, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, thrust_neu, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        if plot_Te == 1:
            # Plot the time evolution of the average Te in the domain
            plt.figure(r'Te(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, Te_mean_dom, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, Te_mean_dom, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                for i in range(0,len(prntstep_IDs)):
                    plt.semilogy(prntstep_IDs[i], Te_mean_dom[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
            else:
                plt.semilogy(time, Te_mean_dom, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the FFT plot for the average Te in the domain
            plt.figure(r'FFT Te(t)')
            plt.semilogx(freq_Te_mean_dom[1:], np.abs(fft_Te_mean_dom[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of both the normalized Id and Te_mean_dom            
            plt.figure(r'Te_Id(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, Te_mean_dom/Te_mean_dom_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(steps, Id/Id_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")    
                for i in range(0,len(prntstep_IDs)):                
#                    plt.text(steps[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)          
                    plt.text(steps[prntstep_IDs[i]-plot_tol], Te_mean_dom[prntstep_IDs[i]-plot_tol]/Te_mean_dom_mean,prntstep_IDs_text[i],fontsize = text_size,color='r',ha='center',va='center')             
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, Te_mean_dom/Te_mean_dom_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(prntsteps_ID, Id/Id_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")
                plt.text(time[prntstep_IDs[i]-plot_tol], Te_mean_dom[prntstep_IDs[i]-plot_tol]/Te_mean_dom_mean,prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
                for i in range(0,len(prntstep_IDs)):
                    plt.semilogy(prntstep_IDs[i], Te_mean_dom[prntstep_IDs[i]]/Te_mean_dom_mean, linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                    plt.semilogy(prntstep_IDs[i], Id[prntstep_IDs[i]]/Id_mean, linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
            else:
                plt.semilogy(time, Te_mean_dom/Te_mean_dom_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(time, Id/Id_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")  
                for i in range(0,len(prntstep_IDs)):
                    plt.semilogy(time[prntstep_IDs[i]], Te_mean_dom[prntstep_IDs[i]]/Te_mean_dom_mean, linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#                    plt.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
                    plt.text(fact_x[i]*time[prntstep_IDs[i]-plot_tol], fact_y[i]*Te_mean_dom[prntstep_IDs[i]-plot_tol]/Te_mean_dom_mean,prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')                
#                plt.semilogy(time, Id_mean*np.ones(np.shape(time)), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            

        if plot_Id == 1:
            # Plot the time evolution of the discharge current
            plt.figure(r'Id(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, Id, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(steps, Id_mean*np.ones(np.shape(steps)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
                for i in range(0,len(prntstep_IDs)): 
                    plt.semilogy(steps[prntstep_IDs[i]], Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color='r', markeredgecolor = 'k', label="")
#                    plt.text(steps[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)            
                    plt.text(steps[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='r',ha='center',va='center')            
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, Id, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(prntsteps_ID, Id_mean*np.ones(np.shape(prntsteps_ID)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
                for i in range(0,len(prntstep_IDs)):
                    plt.semilogy(prntstep_IDs[i], Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color='r', markeredgecolor = 'k', label="")
#                    plt.text(prntstep_IDs[i]+plot_tol, Id[prntstep_IDs[i]],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
                    plt.text(prntstep_IDs[i]+plot_tol, Id[prntstep_IDs[i]],prntstep_IDs_text[i],fontsize = text_size,color='r',ha='center',va='center')
            else:
#                plt.semilogy(time, Id, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                plt.plot(time, Id, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.plot(time, Id, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind3], color=colors[ind], markeredgecolor = 'k', label=labels[ind3])

#                if k == 0 or k == 1:
#                    plt.semilogy(time, Id, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                elif k == 2 or k == 3:
#                    plt.semilogy(time, Id, linestyle=linestyles[k], dashes=dashList[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])

#                plt.semilogy(time, Id_mean*np.ones(np.shape(time)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")              
#                plt.semilogy(time[nsteps-last_steps::], Id_mean*np.ones(np.shape(time[nsteps-last_steps::])), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")                              
                for i in range(0,len(prntstep_IDs)):
                    plt.semilogy(time[prntstep_IDs[i]], Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color='r', markeredgecolor = 'k', label="")            
#                    plt.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
                    plt.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='r',ha='center',va='center')
                
            # Plot the time evolution of the instantaneous discharge current
            plt.figure(r'Id_inst(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, Id_inst, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, Id_inst, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                for i in range(0,len(prntstep_IDs)):
                    plt.semilogy(prntstep_IDs[i], Id_inst[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
            else:
#                plt.semilogy(time, Id_inst, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                plt.semilogy(time[nsteps-last_steps::], Id_mean*np.ones(np.shape(time[nsteps-last_steps::])), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")                          

                plt.plot(time, Id_inst, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.plot(time[nsteps-last_steps::], Id_mean*np.ones(np.shape(time[nsteps-last_steps::])), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")                          

            # Plot the FFT plot for the discharge current  
            if make_mean == 1:           
                plt.figure(r'FFT Id(t)')
#                plt.semilogx(freq_Id[1:], np.abs(fft_Id[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogx(freq_Id[1:], np.abs(fft_Id[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
    
                # Plot the FFT plot for the instantaneous discharge current
                plt.figure(r'FFT Id_inst(t)')
                plt.semilogx(freq_Id_inst[1:], np.abs(fft_Id_inst[1:]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the ion beam current
            plt.figure(r'I_beam(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, I_beam, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(steps, I_beam_mean*np.ones(np.shape(steps)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, I_beam, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(prntsteps_ID, I_beam_mean*np.ones(np.shape(prntsteps_ID)), linestyle=linestyles[ind3], linewidth = line_width-1, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
                for i in range(0,len(prntstep_IDs)):
                    plt.semilogy(prntstep_IDs[i], I_beam[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
            else:
                plt.semilogy(time, I_beam, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                plt.semilogy(time, I_beam_mean*np.ones(np.shape(time)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
                plt.plot(time, I_beam, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            if make_mean == 1:
                # Plot the FFT plot for the ion beam current            
                plt.figure(r'FFT I_beam(t)')
                plt.semilogx(freq_I_beam[1:], np.abs(fft_I_beam[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of both the discharge and the beam current            
            plt.figure(r'Id_Ibeam(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, Id, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(steps, I_beam, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")    
                for i in range(0,len(prntstep_IDs)):                
#                    plt.text(steps[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)          
                    plt.text(steps[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='r',ha='center',va='center')             
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, Id, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(prntsteps_ID, I_beam, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")
                plt.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
                for i in range(0,len(prntstep_IDs)):
                    plt.semilogy(prntstep_IDs[i], Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                    plt.semilogy(prntstep_IDs[i], I_beam[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
            else:
                plt.semilogy(time, Id, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(time, I_beam, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")  
                for i in range(0,len(prntstep_IDs)):
                    plt.semilogy(time[prntstep_IDs[i]], Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#                    plt.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
                    plt.text(fact_x[i]*time[prntstep_IDs[i]-plot_tol], fact_y[i]*Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')                
                plt.semilogy(time, Id_mean*np.ones(np.shape(time)), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            
            # Plot the FFT plot for both the discharge and the ion beam current
            plt.figure(r'FFT Id_Ibeam(t)')       
            plt.semilogx(freq_Id[1:], np.abs(fft_Id[1:]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            plt.semilogx(freq_I_beam[1:], np.abs(fft_I_beam[1:]), linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")        
            # Plot the time evolution of both the discharge and the beam current (normalized)            
            plt.figure(r'norm_Id_Ibeam(t)')     
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, Id/Id_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(steps, I_beam/I_beam_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")    
                for i in range(0,len(prntstep_IDs)):                
#                    plt.text(steps[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)          
                    plt.text(steps[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol]/Id_mean,prntstep_IDs_text[i],fontsize = text_size,color='r',ha='center',va='center')             
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, Id/Id_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(prntsteps_ID, I_beam/I_beam_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")
                plt.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol]/Id_mean,prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
                for i in range(0,len(prntstep_IDs)):
                    plt.semilogy(prntstep_IDs[i], Id[prntstep_IDs[i]]/Id_mean, linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                    plt.semilogy(prntstep_IDs[i], I_beam[prntstep_IDs[i]]/I_beam_mean, linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
            else:
                plt.semilogy(time, Id/Id_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.semilogy(time, I_beam/I_beam_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")  
                for i in range(0,len(prntstep_IDs)):
                    plt.semilogy(time[prntstep_IDs[i]], Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#                    plt.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
                    plt.text(fact_x[i]*time[prntstep_IDs[i]-plot_tol], fact_y[i]*Id[prntstep_IDs[i]-plot_tol]/Id_mean,prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')                
                plt.semilogy(time, Id_mean*np.ones(np.shape(time))/Id_mean, linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            
        
            # Plot the time evolution of the cathode current
            plt.figure(r'Icath(t)')
            if np.any(Icath != 0):
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.semilogy(steps, Icath, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.semilogy(steps, Icath_mean*np.ones(np.shape(steps)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                    plt.semilogy(prntsteps_ID, Icath, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.semilogy(prntsteps_ID, Icath_mean*np.ones(np.shape(prntsteps_ID)), linestyle=linestyles[ind3], linewidth = line_width-1, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
                    for i in range(0,len(prntstep_IDs)):
                        plt.semilogy(prntstep_IDs[i], Icath[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                else:
                    plt.plot(time, Icath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                    plt.semilogy(time, Icath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
    #                plt.semilogy(time, Icath_mean*np.ones(np.shape(time)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
                if make_mean == 1:
                    # Plot the FFT plot for the cathode current            
                    plt.figure(r'FFT Icath(t)')
                    plt.semilogx(freq_Icath[1:], np.abs(fft_Icath[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
            # Plot the time evolution of the conducting wall current
            plt.figure(r'Icond(t)')
            if n_cond_wall > 0:
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.semilogy(steps, Icond, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.semilogy(steps, Icond_mean*np.ones(np.shape(steps)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                    plt.semilogy(prntsteps_ID, Icond, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.semilogy(prntsteps_ID, Icond_mean*np.ones(np.shape(prntsteps_ID)), linestyle=linestyles[ind3], linewidth = line_width-1, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
                    for i in range(0,len(prntstep_IDs)):
                        plt.semilogy(prntstep_IDs[i], Icond[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                else:
                    plt.semilogy(time, Icond[:,0], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
    #                plt.semilogy(time, Icond_mean*np.ones(np.shape(time)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
                if make_mean == 1:
                    # Plot the FFT plot for the conducting wall current            
                    plt.figure(r'FFT Icond(t)')
                    plt.semilogx(freq_Icond[1:], np.abs(fft_Icond[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
            # Plot the time evolution of the error in currents of the external circuit
            plt.figure(r'err_I(t)')
            if np.any(err_I != 0):
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.semilogy(steps, err_I, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                    plt.semilogy(prntsteps_ID, err_I, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    for i in range(0,len(prntstep_IDs)):
                        plt.semilogy(prntstep_IDs[i], err_I[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                else:
                    plt.semilogy(time, err_I, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])

            # Plot the time evolution of the conducting wall current
            plt.figure(r'Icond+Id(t)')
            if n_cond_wall > 0:
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.semilogy(steps, Icond+Id, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                    plt.semilogy(prntsteps_ID, Icond+Id, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    for i in range(0,len(prntstep_IDs)):
                        plt.semilogy(prntstep_IDs[i], Icond[prntstep_IDs[i]]+Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                else:
                    plt.semilogy(time, Icond[:,0]+Id, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.semilogy(time, Icath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k', label=labels[k])
   
            
        
        
        if plot_Vd == 1:
            # Plot the time evolution of the discharge voltage
            plt.figure(r'Vd(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, Vd, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, Vd, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                for i in range(0,len(prntstep_IDs)):
                    plt.plot(prntstep_IDs[i], Vd[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
            else:
                plt.plot(time, Vd, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
            # Plot the time evolution of the discharge voltage
            plt.figure(r'Vcond(t)')
            if np.any(Vcond != 0):
                if Vcond[1,0] != Vcond[2,0]:
                    if time2steps_axis == 1 and prntstepsID_axis == 0:
                        plt.plot(steps, Vcond, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                        plt.plot(prntsteps_ID, Vcond, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        for i in range(0,len(prntstep_IDs)):
                            plt.plot(prntstep_IDs[i], Vd[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                    else:
                        plt.plot(time, Vcond, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    # Plot the FFT plot for the discharge current  
                    if make_mean == 1:           
                        plt.figure(r'FFT Vcond(t)')
        #                plt.semilogx(freq_Vcond[1:], np.abs(fft_Vcond[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogx(freq_Vcond[1:], np.abs(fft_Vcond[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        
        
        if plot_Pd == 1:
            # Plot the time evolution of the input power
            plt.figure(r'Pd(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, Pd*1e-3, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, Pd*1e-3, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                for i in range(0,len(prntstep_IDs)):
                    plt.plot(prntstep_IDs[i], Pd[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
            else:
                plt.plot(time, Pd*1e-3, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the power deposited to material (dielectric) walls
            plt.figure(r'P_mat(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, P_mat, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, P_mat, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, P_mat, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the power deposited to the injection (anode) wall
            plt.figure(r'P_inj(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, P_inj, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, P_inj, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, P_inj, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the power deposited to the free loss wall
            plt.figure(r'P_inf(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, P_inf, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, P_inf, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, P_inf, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the power spent in ionization
            plt.figure(r'P_ion(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, P_ion, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, P_ion, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, P_ion, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the power spent in excitation
            plt.figure(r'P_ex(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, P_ex, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, P_ex, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, P_ex, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the total ion and neutral power deposited to the free loss wall
            plt.figure(r'P_use_tot ion plus neu (t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, P_use_tot, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, P_use_tot, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, P_use_tot, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the axial ion and neutral power deposited to the free loss wall
            plt.figure(r'P_use_z ion plus neu (t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, P_use_z, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, P_use_z, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, P_use_z, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the power deposited to material (dielectric) walls by the heavy species
            plt.figure(r'P_mat_hs(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, P_mat_hs, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, P_mat_hs, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, P_mat_hs, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the power deposited to the injection (anode) wall by the heavy species
            plt.figure(r'P_inj_hs(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, P_inj_hs, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, P_inj_hs, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.plot(time, P_inj_hs, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the power deposited to the injection (anode) wall faces by the electrons
            plt.figure(r'P_inj_faces_e(t)')        
        
        if plot_cath == 1:
            # Plot the time evolution of the cathode equivalent emission frequency
            plt.figure(r'nu_cat(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, nu_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, nu_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.semilogy(time, nu_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the time evolution of the cathode emission power
            plt.figure(r'P_cat(t)')
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, P_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, P_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                plt.semilogy(time, P_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the FFT plot for the cathode equivalent emission frequency
            plt.figure(r'FFT nu_cat(t)')
            plt.semilogx(freq_nu_cath[1:], np.abs(fft_nu_cath[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # Plot the FFT plot for the cathode emission power
            plt.figure(r'FFT P_cat(t)')        
            plt.semilogx(freq_P_cath[1:], np.abs(fft_P_cath[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        
        if plot_mbal == 1:
            # Plot the time evolution of the neutrals 1 mass balance
            plt.figure("n1 mass bal")
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(steps, dMdt_n1, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(prntsteps_ID, dMdt_n1, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            else:
                plt.plot(time, mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(time, dMdt_n1, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            # Plot the time evolution of the ions 1 mass balance
            plt.figure("i1 mass bal")
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(steps, dMdt_i1, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(prntsteps_ID, dMdt_i1, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            else:
                plt.plot(time, mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(time, dMdt_i1, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            # Plot the time evolution of the ions 2 mass balance
            plt.figure("i2 mass bal")
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(steps, dMdt_i2, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(prntsteps_ID, dMdt_i2, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            else:
                plt.plot(time, mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(time, dMdt_i2, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            # Plot the time evolution of the total mass balance
            plt.figure("Total mass bal")
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps, mbal_n1 + mbal_i1 + mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(steps, dMdt_n1 + dMdt_i1 + dMdt_i2, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID, mbal_n1 + mbal_i1 + mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(prntsteps_ID, dMdt_n1 + dMdt_i1 + dMdt_i2, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            else:
                plt.plot(time, mbal_n1 + mbal_i1 + mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
                plt.plot(time, dMdt_n1 + dMdt_i1 + dMdt_i2, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind+1], markeredgecolor = 'k',label=labels[k]+" dM/dt")
            # Plot the time evolution of the neutrals 1 mass balance error 
            plt.figure("err n1 mass bal")
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, err_mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, err_mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            else:
                plt.semilogy(time, err_mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            # Plot the time evolution of the ions 1 mass balance error 
            plt.figure("err i1 mass bal")
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, err_mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, err_mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            else:
                plt.semilogy(time, err_mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            # Plot the time evolution of the ions 2 mass balance error 
            plt.figure("err i2 mass bal")
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, err_mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, err_mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            else:
                plt.semilogy(time, err_mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            # Plot the time evolution of the total mass balance error 
            plt.figure("err total mass bal")
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps, err_mbal_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID, err_mbal_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            else:
                plt.semilogy(time, err_mbal_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=labels[k])
            
            if nsims == 1:
                # Plot the time evolution of the species and the total mass balance
                plt.figure("All mass bal")
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.plot(steps, mbal_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"Total")
                    plt.plot(steps, mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$n1$")
                    plt.plot(steps, mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$i1$")
                    plt.plot(steps, mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$i2$")                    
                elif time2steps_axis == 1 and prntstepsID_axis == 1: 
                    plt.plot(prntsteps_ID, mbal_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"Total")
                    plt.plot(prntsteps_ID, mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$n1$")
                    plt.plot(prntsteps_ID, mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$i1$")
                    plt.plot(prntsteps_ID, mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$i2$")    
                else:
                    plt.plot(time, mbal_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"Total")
                    plt.plot(time, mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$n1$")
                    plt.plot(time, mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$i1$")
                    plt.plot(time, mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$i2$")    
                # Plot the time evolution of the species and the total mass balance errors
                plt.figure("All err mass bal")
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.semilogy(steps, err_mbal_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"\epsilon_{M}")
                    plt.semilogy(steps, err_mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\epsilon_{M,n1}$")
                    plt.semilogy(steps, err_mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\epsilon_{M,i1}$")
                    plt.semilogy(steps, err_mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\epsilon_{M,i2}$")                    
                elif time2steps_axis == 1 and prntstepsID_axis == 1: 
                    plt.semilogy(prntsteps_ID, err_mbal_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"\epsilon_{M}")
                    plt.semilogy(prntsteps_ID, err_mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\epsilon_{M,n1}$")
                    plt.semilogy(prntsteps_ID, err_mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\epsilon_{M,i1}$")
                    plt.semilogy(prntsteps_ID, err_mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\epsilon_{M,i2}$")    
                else:
                    plt.semilogy(time, err_mbal_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"\epsilon_{M}")
                    plt.semilogy(time, err_mbal_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\epsilon_{M,n1}$")
                    plt.semilogy(time, err_mbal_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\epsilon_{M,i1}$")
                    plt.semilogy(time, err_mbal_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\epsilon_{M,i2}$") 
                # Plot the time evolution of the contributions to the total mass balance
                plt.figure("Contributions on total mass bal")
                if time2steps_axis == 1 and prntstepsID_axis == 0:
#                    plt.plot(steps, ctr_mflow_coll_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M}^{coll}$")
                    plt.plot(steps, ctr_mflow_tw_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M}^{tw}$")
                    plt.plot(steps, ctr_mflow_fw_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{M}^{fw}$")                    
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
#                    plt.plot(prntsteps_ID, ctr_mflow_coll_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M}^{coll}$")
                    plt.plot(prntsteps_ID, ctr_mflow_tw_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M}^{tw}$")
                    plt.plot(prntsteps_ID, ctr_mflow_fw_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{M}^{fw}$") 
                else:
#                    plt.plot(time, ctr_mflow_coll_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M}^{coll}$")
                    plt.plot(time, ctr_mflow_tw_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M}^{tw}$")
                    plt.plot(time, ctr_mflow_fw_tot, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{M}^{fw}$")                    
                # Plot the time evolution of the contributions to the n1 mass balance
                plt.figure("Contributions on n1 mass bal")
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.plot(steps, ctr_mflow_coll_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M,n1}^{coll}$")
                    plt.plot(steps, ctr_mflow_tw_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M,n1}^{tw}$")
                    plt.plot(steps, ctr_mflow_fw_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{M,n1}^{fw}$")                    
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                    plt.plot(prntsteps_ID, ctr_mflow_coll_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M,n1}^{coll}$")
                    plt.plot(prntsteps_ID, ctr_mflow_tw_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M,n1}^{tw}$")
                    plt.plot(prntsteps_ID, ctr_mflow_fw_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{M,n1}^{fw}$") 
                else:
                    plt.plot(time, ctr_mflow_coll_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M,n1}^{coll}$")
                    plt.plot(time, ctr_mflow_tw_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M,n1}^{tw}$")
                    plt.plot(time, ctr_mflow_fw_n1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{M,n1}^{fw}$")                    
                # Plot the time evolution of the contributions to the i1 mass balance
                plt.figure("Contributions on i1 mass bal")
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.plot(steps, ctr_mflow_coll_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M,i1}^{coll}$")
                    plt.plot(steps, ctr_mflow_tw_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M,i1}^{tw}$")
#                    plt.plot(steps, ctr_mflow_fw_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k',label=r"$\varepsilon_{M,i1}^{fw}$")                    
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                    plt.plot(prntsteps_ID, ctr_mflow_coll_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M,i1}^{coll}$")
                    plt.plot(prntsteps_ID, ctr_mflow_tw_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M,i1}^{tw}$")
#                    plt.plot(prntsteps_ID, ctr_mflow_fw_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{M,i1}^{fw}$") 
                else:
                    plt.plot(time, ctr_mflow_coll_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M,i1}^{coll}$")
                    plt.plot(time, ctr_mflow_tw_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M,i1}^{tw}$")
#                    plt.plot(time, ctr_mflow_fw_i1, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{M,i1}^{fw}$")                    
                # Plot the time evolution of the contributions to the i2 mass balance
                plt.figure("Contributions on i2 mass bal")
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.plot(steps, ctr_mflow_coll_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M,i2}^{coll}$")
                    plt.plot(steps, ctr_mflow_tw_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M,i2}^{tw}$")
#                    plt.plot(steps, ctr_mflow_fw_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{M,i2}^{fw}$")                    
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                    plt.plot(prntsteps_ID, ctr_mflow_coll_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M,i2}^{coll}$")
                    plt.plot(prntsteps_ID, ctr_mflow_tw_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M,i2}^{tw}$")
#                    plt.plot(prntsteps_ID, ctr_mflow_fw_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{M,i2}^{fw}$") 
                else:
                    plt.plot(time, ctr_mflow_coll_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{M,i2}^{coll}$")
                    plt.plot(time, ctr_mflow_tw_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{M,i2}^{tw}$")
#                    plt.plot(time, ctr_mflow_fw_i2, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{M,i2}^{fw}$")                    
        if plot_Pbal == 1:
            # Plot the time evolution of the total energy balance
            fact = 1E-3 
            plt.figure("P balance")
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps[plot_Pbal_inistep::], balP[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"Balance")
                plt.plot(steps[plot_Pbal_inistep::], Pd[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$P_d$")
                plt.plot(steps[plot_Pbal_inistep::], -Pthrust[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$P_{use}$")                    
                plt.plot(steps[plot_Pbal_inistep::], -Pnothrust[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$P_{loss}$")
                plt.plot(steps[plot_Pbal_inistep::], -Pnothrust_walls[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=brown, markeredgecolor = 'k',label=r"$P_{walls}$") 
                plt.plot(steps[plot_Pbal_inistep::], -Pionex[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=orange, markeredgecolor = 'k',label=r"$P_{ion,ex}$")                                
#                plt.plot(steps, -(Pionex+Pnothrust_walls)*fact, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"$P_{walls}+P_{ion,ex}$")                                
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID[plot_Pbal_inistep::], balP[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"Balance")
                plt.plot(prntsteps_ID[plot_Pbal_inistep::], Pd[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$P_d$")
                plt.plot(prntsteps_ID[plot_Pbal_inistep::], -Pthrust[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$P_{use}$")                    
                plt.plot(prntsteps_ID[plot_Pbal_inistep::], -Pnothrust[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$P_{loss}$")
                plt.plot(prntsteps_ID[plot_Pbal_inistep::], -Pnothrust_walls[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=brown, markeredgecolor = 'k',label=r"$P_{walls}$") 
                plt.plot(prntsteps_ID[plot_Pbal_inistep::], -Pionex[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=orange, markeredgecolor = 'k',label=r"$P_{ion,ex}$")                                
#                plt.plot(prntsteps_ID, -(Pionex+Pnothrust_walls)*fact, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"$P_{walls}+P_{ion,ex}$")                                 
            else:
                plt.plot(time[plot_Pbal_inistep::], balP[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"Balance")
                plt.plot(time[plot_Pbal_inistep::], Pd[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$P_d$")
                plt.plot(time[plot_Pbal_inistep::], -Pthrust[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$P_{use}$")                    
                plt.plot(time[plot_Pbal_inistep::], -Pnothrust[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$P_{loss}$")
                plt.plot(time[plot_Pbal_inistep::], -Pnothrust_walls[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=brown, markeredgecolor = 'k',label=r"$P_{walls}$") 
                plt.plot(time[plot_Pbal_inistep::], -Pionex[plot_Pbal_inistep::]*fact, linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=orange, markeredgecolor = 'k',label=r"$P_{ion,ex}$")                                
#                plt.plot(time, -(Pionex+Pnothrust_walls)*fact, linestyle='--', linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='k', markeredgecolor = 'k',label=r"$P_{walls}+P_{ion,ex}$")                                
            # Plot the time evolution of the total energy balance error
            plt.figure("P balance error")
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.semilogy(steps[plot_Pbal_inistep::], err_balP_Pthrust[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=labels[k])
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.semilogy(prntsteps_ID[plot_Pbal_inistep::], err_balP_Pthrust[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=labels[k])
            else:
                plt.semilogy(time[plot_Pbal_inistep::], err_balP_Pthrust[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=labels[k]) 
            # Plot the time evolution of the contributions to the total energy balance
            plt.figure("Contributions on P balance")
            if time2steps_axis == 1 and prntstepsID_axis == 0:
                plt.plot(steps[plot_Pbal_inistep::], ctr_balPthrust_Pd[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_d}$")
                plt.plot(steps[plot_Pbal_inistep::], ctr_balPthrust_Pthrust[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{use}}$")                    
                plt.plot(steps[plot_Pbal_inistep::], ctr_balPthrust_Pnothrust[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{loss}}$")
                plt.plot(steps[plot_Pbal_inistep::], ctr_balPthrust_Pnothrust_walls[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=brown, markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{walls}}$") 
                plt.plot(steps[plot_Pbal_inistep::], ctr_balPthrust_Pnothrust_ionex[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=orange, markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{ion,ex}}$")                                
            elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                plt.plot(prntsteps_ID[plot_Pbal_inistep::], ctr_balPthrust_Pd[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_d}$")
                plt.plot(prntsteps_ID[plot_Pbal_inistep::], ctr_balPthrust_Pthrust[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{use}}$")                    
                plt.plot(prntsteps_ID[plot_Pbal_inistep::], ctr_balPthrust_Pnothrust[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{loss}}$")
                plt.plot(prntsteps_ID[plot_Pbal_inistep::], ctr_balPthrust_Pnothrust_walls[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=brown, markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{walls}}$") 
                plt.plot(prntsteps_ID[plot_Pbal_inistep::], ctr_balPthrust_Pnothrust_ionex[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=orange, markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{ion,ex}}$")                                
            else:
                plt.plot(time[plot_Pbal_inistep::], ctr_balPthrust_Pd[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='b', markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_d}$")
                plt.plot(time[plot_Pbal_inistep::], ctr_balPthrust_Pthrust[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='g', markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{use}}$")                    
                plt.plot(time[plot_Pbal_inistep::], ctr_balPthrust_Pnothrust[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color='r', markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{loss}}$")
                plt.plot(time[plot_Pbal_inistep::], ctr_balPthrust_Pnothrust_walls[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=brown, markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{walls}}$") 
                plt.plot(time[plot_Pbal_inistep::], ctr_balPthrust_Pnothrust_ionex[plot_Pbal_inistep::], linestyle=linestyles[ind3], linewidth=line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=orange, markeredgecolor = 'k',label=r"$\varepsilon_{E}^{P_{ion,ex}}$")                                
        if plot_FLvars == 1:
            find = 5
            # Plot the time evolution of the phi infinity at free loss
            plt.figure("phi_inf FL")
            if np.any(phi_inf != 0):
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.semilogy(steps[find::], phi_inf[find::], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                    plt.semilogy(prntsteps_ID[find::], phi_inf[find::], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    for i in range(0,len(prntstep_IDs)):
                        plt.semilogy(prntstep_IDs[i], phi_inf[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                else:
                    plt.plot(time[find::], phi_inf[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])

            
                # Plot the FFT plot for the phi infinity at free loss
                if make_mean == 1: 
                    if np.any(np.diff(phi_inf) != 0):
                        plt.figure(r'FFT phi_inf')
                        plt.semilogx(freq_phi_inf[1:], np.abs(fft_phi_inf[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])

            
            # Plot the time evolution of the I infinity at free loss
            plt.figure("I_inf FL")
            if np.any(I_inf != 0):
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.semilogy(steps[find::], I_inf[find::], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                    plt.semilogy(prntsteps_ID[find::], I_inf[find::], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    for i in range(0,len(prntstep_IDs)):
                        plt.semilogy(prntstep_IDs[i], I_inf[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                else:
                    plt.plot(time[find::], I_inf[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    
                # Plot the FFT plot for the I infinity at free loss
                if make_mean == 1:         
                    plt.figure(r'FFT I_inf')
                    plt.semilogx(freq_I_inf[1:], np.abs(fft_I_inf[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])

            
            # Plot the time evolution of the sum of Id and I infinity at free loss
            plt.figure("I_inf+Id FL")
            if np.any(I_inf != 0):
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.semilogy(steps[find::], I_inf[find::]+Id[find::], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                    plt.semilogy(prntsteps_ID[find::], I_inf[find::]+Id[find::], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    for i in range(0,len(prntstep_IDs)):
                        plt.semilogy(prntstep_IDs[i], I_inf[prntstep_IDs[i]]+Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                else:
                    plt.plot(time[find::], I_inf[find::]+Id[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
            # Plot the time evolution of the current balance error
            plt.figure(r'err_I_inf FL')
            if np.any(err_I_inf != 0):
                if time2steps_axis == 1 and prntstepsID_axis == 0:
                    plt.semilogy(steps[find::], err_I_inf[find::], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif time2steps_axis == 1 and prntstepsID_axis == 1:  
                    plt.semilogy(prntsteps_ID[find::], err_I_inf[find::], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    for i in range(0,len(prntstep_IDs)):
                        plt.semilogy(prntstep_IDs[i], err_I_inf[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
                else:
                    plt.semilogy(time[find::], err_I_inf[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])


            # Plot the time evolution of the electron temperatures at free loss
            plt.figure("Te FL")
#            plt.plot(time[find::], Te_FL[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" mean")
            plt.plot(time[find::], Te_FL_int[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" int")
            # Plot the time evolution of the electron temperatures at vertical  free loss
            plt.figure("Te FL ver")
#            plt.plot(time[find::], Te_FL_ver[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" mean")
#            plt.plot(time[find::], Te_FL_ver_int[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="s", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" int")
            plt.plot(time[find::], Te[i_plot_ver,j_plot_ver,find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ("+str(i_plot_lat)+","+str(j_plot_lat)+")")
            # Plot the time evolution of the electron temperatures at lateral free loss
            plt.figure("Te FL lat")
#            plt.plot(time[find::], Te_FL_lat[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" mean")
#            plt.plot(time[find::], Te_FL_lat_int[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="s", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" int")
            plt.plot(time[find::], Te[i_plot_lat,j_plot_lat,find::],linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ("+str(i_plot_lat)+","+str(j_plot_lat)+")")
            
            if make_mean == 1:
                # Plot the FFT plot for the electron temperatures at vertical free loss
                plt.figure(r'FFT Te FL ver')
                plt.semilogx(freq_Te_FL_pver[1:], np.abs(fft_Te_FL_pver[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])

            
            # Plot the time evolution of the electric potential at free loss
            plt.figure("phi FL")
#            plt.plot(time[find::], phi_FL[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" mean")
            plt.plot(time[find::], phi_FL_int[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" int")
            # Plot the time evolution of the electric potential at vertical free loss
            plt.figure("phi FL ver")
#            plt.plot(time[find::], phi_FL_ver[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" mean")
#            plt.plot(time[find::], phi_FL_ver_int[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="s", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" int")
            plt.plot(time[find::], phi[i_plot_ver,j_plot_ver,find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ("+str(i_plot_lat)+","+str(j_plot_lat)+")")
            # Plot the time evolution of the electric potential at lateral free loss
            plt.figure("phi FL lat")
#            plt.plot(time[find::], phi_FL_lat[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" mean")
#            plt.plot(time[find::], phi_FL_lat_int[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="s", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" int")
            plt.plot(time[find::], phi[i_plot_lat,j_plot_lat,find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ("+str(i_plot_lat)+","+str(j_plot_lat)+")")
            
            if make_mean == 1:
                # Plot the FFT plot for the electric potential at vertical free loss
                plt.figure(r'FFT phi FL ver')
                plt.semilogx(freq_phi_FL_pver[1:], np.abs(fft_phi_FL_pver[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
            
            
            # Plot the time evolution of the dphi/Te at free loss
            plt.figure("dphi/Te FL")
            if np.any(phi_inf != 0):
    #            plt.plot(time[find::], ratio_DphiTe_FL[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" mean")
                plt.plot(time[find::], ratio_DphiTe_FL_int[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" int")
            # Plot the time evolution of the electric potential at vertical free loss
            plt.figure("dphi/Te FL ver")
            if np.any(phi_inf != 0):
    #            plt.plot(time[find::], ratio_DphiTe_FL_ver[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" mean")
    #            plt.plot(time[find::], ratio_DphiTe_FL_ver_int[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="s", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" int")
                plt.plot(time[find::], (phi[i_plot_ver,j_plot_ver,find::]-phi_inf[find::])/Te[i_plot_ver,j_plot_ver,find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ("+str(i_plot_lat)+","+str(j_plot_lat)+")")
            # Plot the time evolution of the electric potential at lateral free loss
            plt.figure("dphi/Te FL lat")
            if np.any(phi_inf != 0):
    #            plt.plot(time[find::], ratio_DphiTe_FL_lat[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" mean")
    #            plt.plot(time[find::], ratio_DphiTe_FL_lat_int[find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="s", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" int")
                plt.plot(time[find::], (phi[i_plot_lat,j_plot_lat,find::]-phi_inf[find::])/Te[i_plot_lat,j_plot_lat,find::], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ("+str(i_plot_lat)+","+str(j_plot_lat)+")")
            
            if np.any(phi_inf != 0):
                if make_mean == 1:
                    # Plot the FFT plot for the dphi/Te at vertical free loss
                    plt.figure(r'FFT dphi/Te FL ver')
                    plt.semilogx(freq_DphiTe_FL_pver[1:], np.abs(fft_DphiTe_FL_pver[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        
        ind = ind + 1
        if ind > 8:
            ind = 0
            ind2 = ind2 + 1
            if ind2 > 6:
                ind = 0
                ind2 = 0
                ind3 = ind3 + 1
     
               
    if exp_data_time_plots == 1:
        exp_time = exp_time*1e3
        last_ind_exp = np.where(exp_time <= time[-1])[0][-1]
        plt.figure(r'Id_exp(t)')
        ax = plt.gca()
        #ax.set_ylim(2e1,3e1)
        plt.semilogy(exp_time[0:last_ind_exp+1], exp_Id[0:last_ind_exp+1], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='D', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
        #plt.plot(exp_time[0:last_ind_exp+1], exp_Id[0:last_ind_exp+1], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='D', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
        plt.xlabel(r"$t$ (ms)",fontsize = font_size)
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
        elif time2steps_axis == 1 and prntstepsID_axis == 1:
            plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
        plt.title(r"$I_d$ exp. data",fontsize = font_size,y=1.02)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        if save_flag == 1:
            plt.figure(r'Id_exp(t)')
            plt.savefig(path_out+"Id_exp_t"+figs_format,bbox_inches='tight') 
            plt.close()

    if plot_mass == 1:
        plt.figure(r'mi1(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'mi2(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'mitot(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'mn(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
    if plot_dens == 1:
        plt.figure(r'dens_e(t)')
        ax = plt.gca()
        ax.set_ylim(10**15,10**19)
        plt.legend(fontsize = font_size_legend,loc=4,ncol=2)
        plt.figure(r'dens_n(t)')
        ax = plt.gca()
        ax.set_ylim(10**17,10**19)
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'FFT dens_e(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        ax = plt.gca()
        ax.set_xlim(10**3,10**6)
#        ax.grid(which='both',axis='x',linestyle=':',linewidth=line_width_grid,color='k')
        plt.figure(r'FFT dens_n(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        ax = plt.gca()
        ax.set_xlim(10**3,10**6)
#        ax.grid(which='both',axis='x',linestyle=':',linewidth=line_width_grid,color='k')
        plt.figure(r'dens_e_dens_n(t)')
        ax = plt.gca()
        ax.set_ylim(10**15,10**19)
#        ax.set_xlim(0.0,0.9)
        ylims = ax.get_ylim()
        fact_x = np.array([0.97,1.02,1.02,1.02])
        marker_size_ID = 6
        for i in range(0,len(fast_prntstep_IDs)):
            plt.semilogy(time_fast[fast_prntstep_IDs[i]]*np.ones(2), np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker="", color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
            plt.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_ions[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
            plt.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_neus[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
            plt.text(fact_x[i]*time_fast[fast_prntstep_IDs[i]-plot_tol], 1.5*ylims[0],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')     
            print(prntstep_IDs_text[i]+" time_fast = "+str(time_fast[fast_prntstep_IDs[i]])+", time = "+str(time[prntstep_IDs[i]]))
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'FFT dens_e_dens_n(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        ax = plt.gca()
        ax.set_xlim(10**3,10**6)
        ax.grid(which='both',axis='x',linestyle=':',linewidth=line_width_grid,color='k')
        plt.figure(r'norm_dens_e_dens_n(t)')  
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
    if plot_Nmp == 1:
        plt.figure(r'Nmpi1(t)')
        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
        plt.figure(r'Nmpi2(t)')
        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
        plt.figure(r'Nmpitot(t)')
        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
        plt.figure(r'Nmpn(t)')
        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
    if plot_eff == 1:
        plt.figure(r'eta_u(t)')
        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
        ax = plt.gca()
        ax.set_ylim(1E-2,1E1)
        plt.figure(r'eta_prod(t)')
        plt.legend(fontsize = font_size_legend,loc=2,ncol=2)
        ax = plt.gca()
        ax.set_ylim(1E-2,1E0)
        plt.figure(r'eta_cur(t)')
        plt.legend(fontsize = font_size_legend,loc=2,ncol=2)
        ax = plt.gca()
        ax.set_ylim(1E-1,1E1)
        plt.figure(r'eta_div(t)')
        plt.legend(fontsize = font_size_legend,loc=2,ncol=2)
        plt.figure(r'eta_thr(t)')
        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
    if plot_thr == 1:
        plt.figure(r'T(t)')
        ax = plt.gca()
        ax.set_ylim(1E-4,1E4)
        plt.legend(fontsize = font_size_legend,loc=4,ncol=2)    
        plt.figure(r'Ti1(t)')
        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
        plt.figure(r'Ti2(t)')
        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
        plt.figure(r'Tn(t)')
        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
    if plot_Te == 1:
        plt.figure(r'Te(t)')
        plt.legend(fontsize = font_size_legend,loc=2,ncol=1) 
        plt.figure(r'FFT Te(t)')
        plt.legend(fontsize = font_size_legend,loc=2,ncol=1)     
        plt.figure(r'Te_Id(t)')
        plt.legend(fontsize = font_size_legend,loc=2,ncol=1)  
    if plot_Id == 1:
        plt.figure(r'Id(t)')
        if exp_data_time_plots == 1:
            #plt.semilogy(exp_time[0:last_ind_exp+1], exp_Id[0:last_ind_exp+1], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='D', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
            #plt.semilogy(exp_time[0:last_ind_exp+1], exp_Id[0:last_ind_exp+1], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
            plt.plot(exp_time[0:last_ind_exp+1], exp_Id[0:last_ind_exp+1], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
#        plt.legend(fontsize = font_size_legend-2,loc=3,ncol=2) 
#        plt.legend(fontsize = font_size_legend-2,loc=2,ncol=2) 
        plt.legend(fontsize = font_size_legend-2,loc=4,ncol=2) 
#        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
        ax = plt.gca()
#        ax.set_ylim(1E0,1E2)
#        ax.set_ylim(1E-2,1E2)
        #ax.set_ylim(1E-1,1E2)
#        ax.set_ylim(5E0,1E2)
#        ax.set_ylim(0,25)
#        ax.set_ylim(0,20)
#        ax.set_ylim(0,5)
#        ax.set_ylim(0,35)
#        ax.set_ylim(0,40)
#        ax.set_xlim(0.3,0.8)
        plt.figure(r'Id_inst(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        ax = plt.gca()
#        ax.set_ylim(1E-2,1E2)
#        ax.set_ylim(0,25)
        plt.figure(r'FFT Id(t)')
        ax = plt.gca()
        ax.set_xlim(3E2,8E5)
        if exp_data_time_plots == 1:
            #plt.semilogx(freq_exp_Id[1:], np.abs(fft_exp_Id[1:]), linestyle='-', linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker='D', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
            plt.semilogx(freq_exp_Id[1:], np.abs(fft_exp_Id[1:]), linestyle='-', linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker='', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
#            plt.semilogx(maxs_freq_Id[6], np.abs(maxs_fft_Id[6]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size+5, marker='*', color='b', markeredgecolor = 'b', label='')

        plt.legend(fontsize = font_size_legend,loc=2,ncol=1)
        ax = plt.gca()
#        ax.grid(which='both',axis='x',linestyle=':',linewidth=line_width_grid,color='k')
        plt.figure(r'FFT Id_inst(t)')
        plt.legend(fontsize = font_size_legend,loc=2,ncol=1)
        ax = plt.gca()
        ax.grid(which='both',axis='x',linestyle=':',linewidth=line_width_grid,color='k')
        plt.figure(r'I_beam(t)')
        plt.legend(fontsize = font_size_legend,loc=3,ncol=1)
        ax = plt.gca()
#        ax.set_ylim(1E-2,1E2)
        plt.figure(r'FFT I_beam(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        ax = plt.gca()
        ax.grid(which='both',axis='x',linestyle=':',linewidth=line_width_grid,color='k')
        plt.figure(r'Id_Ibeam(t)')
        plt.legend(fontsize = font_size_legend,loc=3,ncol=1)
        ax = plt.gca()
        ax.set_ylim(1E-2,1E2)
        plt.figure(r'FFT Id_Ibeam(t)') 
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        ax = plt.gca()
        ax.grid(which='both',axis='x',linestyle=':',linewidth=line_width_grid,color='k')  
        plt.figure(r'norm_Id_Ibeam(t)')  
        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
        ax = plt.gca()
        
        plt.figure(r'Icath(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        ax = plt.gca()
#        ax.set_ylim(0,20)
        
        plt.figure(r'FFT Icath(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        
        plt.figure(r'Icond(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        
        plt.figure(r'FFT Icond(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        
        plt.figure(r'err_I(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        
        plt.figure(r'Icond+Id(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        
        
        
    if plot_Vd == 1:
        plt.figure(r'Vd(t)')
        if exp_data_time_plots == 1:
            #plt.plot(exp_time[0:last_ind_exp+1], exp_Vd[0:last_ind_exp+1], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='D', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
            plt.plot(exp_time[0:last_ind_exp+1], exp_Vd[0:last_ind_exp+1], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
        plt.legend(fontsize = font_size_legend,loc=4,ncol=1)
        
        plt.figure(r'Vcond(t)')
        plt.legend(fontsize = font_size_legend,loc=4,ncol=1)
        
        plt.figure(r'FFT Vcond(t)')
        plt.legend(fontsize = font_size_legend,loc=4,ncol=1)
        
    if plot_Pd == 1:
        plt.figure(r'Pd(t)')
        if exp_data_time_plots == 1:
            #plt.plot(exp_time[0:last_ind_exp+1], exp_Vd[0:last_ind_exp+1], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='D', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
            #plt.plot(exp_time[0:last_ind_exp+1], exp_Id[0:last_ind_exp+1]*exp_Vd[0:last_ind_exp+1]*1e-3, linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
            plt.semilogy(exp_time[0:last_ind_exp+1], exp_Id[0:last_ind_exp+1]*exp_Vd[0:last_ind_exp+1]*1e-3, linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=1)
        plt.figure(r'P_mat(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'P_inj(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'P_inf(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'P_ion(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'P_ex(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'P_use_tot ion plus neu (t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'P_use_z ion plus neu (t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'P_mat_hs(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure(r'P_inj_hs(t)')
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)    
    if plot_cath == 1:
        plt.figure(r'nu_cat(t)')
        plt.legend(fontsize = font_size_legend,loc=2,ncol=1)    
        plt.figure(r'P_cat(t)')
        plt.legend(fontsize = font_size_legend,loc=2,ncol=1) 
        plt.figure(r'FFT nu_cat(t)')
        plt.legend(fontsize = font_size_legend,loc=2,ncol=1) 
        plt.figure(r'FFT P_cat(t)') 
        plt.legend(fontsize = font_size_legend,loc=2,ncol=1) 
        
    if plot_mbal == 1:
        plt.figure("n1 mass bal")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure("i1 mass bal")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure("i2 mass bal")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure("Total mass bal")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure("err n1 mass bal")
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure("err i1 mass bal")
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure("err i2 mass bal")
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        plt.figure("err total mass bal")
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        if nsims == 1:
            plt.figure("All mass bal")
            plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
            plt.figure("All err mass bal")
            plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
            ax = plt.gca()
            ax.set_ylim(1E-9,1E-1)
            plt.figure("Contributions on total mass bal")
            plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
            plt.figure("Contributions on n1 mass bal")
            plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
            plt.figure("Contributions on i1 mass bal")
            plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
            plt.figure("Contributions on i2 mass bal")
            plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
    if plot_Pbal == 1:
        plt.figure("P balance")
        ax = plt.gca()
        ylims = ax.set_ylim()
        fact_x = np.array([0.98,1.02,1.02,1.02])
        for i in range(0,len(prntstep_IDs)):
            plt.plot(time[prntstep_IDs[i]]*np.ones(2), np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker="", color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
            plt.text(fact_x[i]*time[prntstep_IDs[i]-plot_tol], 0.9*ylims[0],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')     
        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)        
        plt.figure("P balance error")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
        # Plot the time evolution of the contributions to the total energy balance
        plt.figure("Contributions on P balance")
        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)        
        
    if plot_FLvars == 1:
        plt.figure("phi_inf FL")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)  
        plt.figure(r'FFT phi_inf')
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3) 
        plt.figure("I_inf FL")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)  
        plt.figure(r'FFT I_inf')
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3) 
        plt.figure("I_inf+Id FL")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)  
        ax = plt.gca()
        ax.set_ylim(0,20)
        plt.figure(r'err_I_inf FL')
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)         
        plt.figure("Te FL")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)   
        plt.figure("Te FL ver")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)   
        plt.figure("Te FL lat")    
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)   
        plt.figure(r'FFT Te FL ver')
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)   
        plt.figure("phi FL")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)   
        plt.figure("phi FL ver")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)   
        plt.figure("phi FL lat")    
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)
        plt.figure(r'FFT phi FL ver')
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)       
        plt.figure("dphi/Te FL")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)
        plt.figure("dphi/Te FL ver")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)
        plt.figure("dphi/Te FL lat")
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)
        plt.figure(r'FFT dphi/Te FL ver')
#        plt.legend(fontsize = font_size_legend,loc=1,ncol=3)
        
      

    
    if save_flag == 1:
        if plot_mass == 1:
            plt.figure(r'mi1(t)')
            plt.savefig(path_out+"mi1_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure(r'mi2(t)')
            plt.savefig(path_out+"mi2_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure(r'mitot(t)')
            plt.savefig(path_out+"mitot_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure(r'mn(t)')
            plt.savefig(path_out+"mn_t"+figs_format,bbox_inches='tight') 
            plt.close() 
        if plot_dens == 1:
            plt.figure(r'dens_e(t)')
            plt.savefig(path_out+"avg_dens_e_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'dens_n(t)')
            plt.savefig(path_out+"avg_dens_n_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT dens_e(t)')
            plt.savefig(path_out+"FFT_avg_dens_e"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT dens_n(t)')
            plt.savefig(path_out+"FFT_avg_dens_n"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'dens_e_dens_n(t)')
            plt.savefig(path_out+"avg_dens_e_dens_n_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT dens_e_dens_n(t)')
            plt.savefig(path_out+"FFT_avg_dens_e_dens_n"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'norm_dens_e_dens_n(t)')  
            plt.savefig(path_out+"norm_avg_dens_e_dens_n_t"+figs_format,bbox_inches='tight') 
            plt.close()
        if plot_Nmp == 1:
            plt.figure(r'Nmpi1(t)')
            plt.savefig(path_out+"Nmpi1_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure(r'Nmpi2(t)')
            plt.savefig(path_out+"Nmpi2_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure(r'Nmpitot(t)')
            plt.savefig(path_out+"Nmpitot_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure(r'Nmpn(t)')
            plt.savefig(path_out+"Nmpn_t"+figs_format,bbox_inches='tight') 
            plt.close() 
        if plot_eff == 1:
            plt.figure(r'eta_u(t)')
            plt.savefig(path_out+"eta_u_t"+figs_format,bbox_inches='tight') 
            plt.close()   
            plt.figure(r'eta_prod(t)')
            plt.savefig(path_out+"eta_prod_t"+figs_format,bbox_inches='tight') 
            plt.close()  
            plt.figure(r'eta_cur(t)')
            plt.savefig(path_out+"eta_cur_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'eta_div(t)')
            plt.savefig(path_out+"eta_div_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'eta_thr(t)')
            plt.savefig(path_out+"eta_thr_t"+figs_format,bbox_inches='tight') 
            plt.close()
        if plot_thr == 1:
            plt.figure(r'T(t)')
            plt.savefig(path_out+"T_t"+figs_format,bbox_inches='tight') 
            plt.close()    
            plt.figure(r'Ti1(t)')
            plt.savefig(path_out+"Ti1_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'Ti2(t)')
            plt.savefig(path_out+"Ti2_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'Tn(t)')
            plt.savefig(path_out+"Tn_t"+figs_format,bbox_inches='tight') 
            plt.close()
        if plot_Te == 1:
            plt.figure(r'Te(t)')
            plt.savefig(path_out+"Te_mean_dom_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT Te(t)')
            plt.savefig(path_out+"FFT_Te_mean_dom"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'Te_Id(t)')
            plt.savefig(path_out+"Te_dom_mean_Id_t"+figs_format,bbox_inches='tight') 
            plt.close()
        if plot_Id == 1:
            plt.figure(r'Id(t)')
            plt.savefig(path_out+"Id_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'Id_inst(t)')
            plt.savefig(path_out+"Id_inst_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT Id(t)')
            plt.savefig(path_out+"FFT_Id"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT Id_inst(t)')
            plt.savefig(path_out+"FFT_Id_inst"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'I_beam(t)')
            plt.savefig(path_out+"I_beam_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT I_beam(t)')
            plt.savefig(path_out+"FFT_I_beam"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'Id_Ibeam(t)')
            plt.savefig(path_out+"Id_Ibeam_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT Id_Ibeam(t)') 
            plt.savefig(path_out+"FFT_Id_Ibeam"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'norm_Id_Ibeam(t)') 
            plt.savefig(path_out+"norm_Id_Ibeam_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'Icath(t)')
            plt.savefig(path_out+"Icath_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT Icath(t)')
            plt.savefig(path_out+"FFT_Icath"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'Icond(t)')
            plt.savefig(path_out+"Icond_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT Icond(t)')
            plt.savefig(path_out+"FFT_Icond"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'err_I(t)')
            plt.savefig(path_out+"errI_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'Icond+Id(t)')
            plt.savefig(path_out+"Icond_Id_t"+figs_format,bbox_inches='tight') 
            plt.close()
            
        if plot_Vd == 1:
            plt.figure(r'Vd(t)')
            plt.savefig(path_out+"Vd_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'Vcond(t)')
            plt.savefig(path_out+"Vcond_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT Vcond(t)')
            plt.savefig(path_out+"FFT_Vcond"+figs_format,bbox_inches='tight') 
            plt.close()
            
        if plot_Pd == 1:
            plt.figure(r'Pd(t)')
            plt.savefig(path_out+"Pd_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'P_mat(t)')
            plt.savefig(path_out+"P_mat_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'P_inj(t)')
            plt.savefig(path_out+"P_inj_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'P_inf(t)')
            plt.savefig(path_out+"P_inf_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'P_ion(t)')
            plt.savefig(path_out+"P_ion_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'P_ex(t)')
            plt.savefig(path_out+"P_ex_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'P_use_tot ion plus neu (t)')
            plt.savefig(path_out+"P_use_tot_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'P_use_z ion plus neu (t)')
            plt.savefig(path_out+"P_use_z_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'P_mat_hs(t)')
            plt.savefig(path_out+"P_mat_hs_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'P_inj_hs(t)')
            plt.savefig(path_out+"P_inj_hs_t"+figs_format,bbox_inches='tight') 
            plt.close()
        if plot_cath == 1:
            plt.figure(r'nu_cat(t)')
            plt.savefig(path_out+"nu_cat_t"+figs_format,bbox_inches='tight') 
            plt.close()   
            plt.figure(r'P_cat(t)')
            plt.savefig(path_out+"P_cat_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT nu_cat(t)')
            plt.savefig(path_out+"FFT_nu_cat"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT P_cat(t)')
            plt.savefig(path_out+"FFT_P_cat"+figs_format,bbox_inches='tight') 
            plt.close()
        if plot_mbal == 1:
            plt.figure("n1 mass bal")
            plt.savefig(path_out+"n1_mass_bal_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("i1 mass bal")
            plt.savefig(path_out+"i1_mass_bal_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("i2 mass bal")
            plt.savefig(path_out+"i2_mass_bal_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("Total mass bal")
            plt.savefig(path_out+"tot_mass_bal_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("err n1 mass bal")
            plt.savefig(path_out+"n1_err_mass_bal_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("err i1 mass bal")
            plt.savefig(path_out+"i1_err_mass_bal_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("err i2 mass bal")
            plt.savefig(path_out+"i2_err_mass_bal_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("err total mass bal")
            plt.savefig(path_out+"tot_err_mass_bal_t"+figs_format,bbox_inches='tight') 
            plt.close()
            if nsims == 1:
                plt.figure("All mass bal")
                plt.savefig(path_out+"all_mass_bal_t"+figs_format,bbox_inches='tight') 
                plt.close()
                plt.figure("All err mass bal")
                plt.savefig(path_out+"all_err_mass_bal_t"+figs_format,bbox_inches='tight') 
                plt.close()
                plt.figure("Contributions on total mass bal")
                plt.savefig(path_out+"contr_tot_mass_bal_t"+figs_format,bbox_inches='tight') 
                plt.close()
                plt.figure("Contributions on n1 mass bal")
                plt.savefig(path_out+"contr_n1_mass_bal_t"+figs_format,bbox_inches='tight') 
                plt.close()
                plt.figure("Contributions on i1 mass bal")
                plt.savefig(path_out+"contr_i1_mass_bal_t"+figs_format,bbox_inches='tight') 
                plt.close()
                plt.figure("Contributions on i2 mass bal")
                plt.savefig(path_out+"contr_i2_mass_bal_t"+figs_format,bbox_inches='tight') 
                plt.close()
        if plot_Pbal == 1:
            plt.figure("P balance")
            plt.savefig(path_out+"balP_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("P balance error")
            plt.savefig(path_out+"err_balP_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("Contributions on P balance")
            plt.savefig(path_out+"contr_balP_t"+figs_format,bbox_inches='tight') 
            plt.close()
        if plot_FLvars == 1:
            plt.figure("phi_inf FL")
            plt.savefig(path_out+"phi_inf_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT phi_inf')
            plt.savefig(path_out+"FFT_phi_inf"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("I_inf FL")
            plt.savefig(path_out+"I_inf_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'FFT I_inf')
            plt.savefig(path_out+"FFT_I_inf"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("I_inf+Id FL")
            plt.savefig(path_out+"I_inf_Id_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure(r'err_I_inf FL')
            plt.savefig(path_out+"err_I_inf_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("Te FL")
            plt.savefig(path_out+"Te_FL_t"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("Te FL ver")
            plt.savefig(path_out+"Te_FL_ver_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure("Te FL lat")    
            plt.savefig(path_out+"Te_FL_lat_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure(r'FFT Te FL ver')
            plt.savefig(path_out+"FFT_Te_FL_ver"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure("phi FL")
            plt.savefig(path_out+"phi_FL_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure("phi FL ver")
            plt.savefig(path_out+"phi_FL_ver_t"+figs_format,bbox_inches='tight') 
            plt.close()   
            plt.figure("phi FL lat")    
            plt.savefig(path_out+"phi_FL_lat_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure(r'FFT phi FL ver')
            plt.savefig(path_out+"FFT_phi_FL_ver"+figs_format,bbox_inches='tight') 
            plt.close()
            plt.figure("dphi/Te FL")
            plt.savefig(path_out+"dphiTe_FL_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure("dphi/Te FL ver")
            plt.savefig(path_out+"dphiTe_FL_ver_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure("dphi/Te FL lat")
            plt.savefig(path_out+"dphiTe_FL_lat_t"+figs_format,bbox_inches='tight') 
            plt.close() 
            plt.figure(r'FFT dphi/Te FL ver')
            plt.savefig(path_out+"FFT_dphiTe_FL_ver"+figs_format,bbox_inches='tight') 
            plt.close() 
    ###########################################################################   
    

df = pd.DataFrame(dic_data)
print(df)
#writer = pd.ExcelWriter(path_out+'export_dataframe.xlsx', engine='xlsxwriter')
#df.to_excel(writer,
#            index=False,startrow=1,
#            columns=['Vd (V)','mA (mg/s)','mC (mg/s)','Id (A)','fd (kHz)',
#                     'Id half amp. (%)','F (mN)','Isp (s)','eta','eta_ene',
#                     'eta_div','eta_disp','eta_cur','eta_vol','eta_u','eta_ch',
#                     'eta_prod','P (W)','P_A/P','P_D/P','Iprod (A)','I_A/Iprod',
#                     'I_D/I_prod']) 

#df.to_excel(path_out+'export_dataframe.xlsx',
#            index=False,startrow=1,
#            columns=['Vd (V)','mA (mg/s)','mC (mg/s)','Id (A)','fd (kHz)',
#                     'Id half amp. (%)','F (mN)','Isp (s)','eta','eta_ene',
#                     'eta_div','eta_disp','eta_cur','eta_vol','eta_u','eta_ch',
#                     'eta_prod','P (W)','P_A/P','P_D/P','Iprod (A)','I_A/Iprod',
#                     'I_D/I_prod'])

df.to_excel(path_out+'export_dataframe.xlsx',
            index=False,startrow=1,
            columns=['Vd (V)','mA (mg/s)','mC (mg/s)','Id (A)','fd_max (kHz)',
                     'fd_2 (kHz)','fd_3 (kHz)','fd_4 (kHz)','fd_5 (kHz)','fd_6 (kHz)',
                     'Id half amp. (%)','F (mN)','Isp (s)','eta','eta_ene',
                     'eta_div','eta_disp','eta_cur','eta_vol','eta_u','eta_ch',
                     'eta_prod','P (W)','P_A/P','P_D/P','Iprod (A)','I_A/Iprod',
                     'I_D/I_prod'],float_format="%.6f") 


