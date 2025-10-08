# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:02:15 2018

@author: adrian

###############################################################################
Description:    This Python function carries out the post processing of the 
                CORE resutls for the HET sims
###############################################################################
Inputs:         No inputs
###############################################################################
Outputs:        Plots and outputs for the simulations
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
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedFormatter
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
from HET_sims_interp_MFAM_picM_plot import HET_sims_interp_MFAM_picM_plot
from HET_sims_interp_scan import HET_sims_interp_scan
from HET_sims_interp_zprof import HET_sims_interp_zprof
from find_firstmax import find_firstmax
from scipy.signal import correlate
import pylab
import scipy.io as sio
import pickle

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
# figs_format = ".png"
figs_format = ".pdf"

# Plots to produce
prof_plots          = 1


# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/fig5_paper/"
path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/fig5_paper/new_sims/"


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
marker_every_mesh      = 30
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
streamline_color  = 'k'
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

def fmt_func_exponent_lines2(x):
    a, b = '{:.1e}'.format(x).split('e')
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
def plot_MFAM_ax_color(ax,faces,nodes,line_width,lambda_color,sigma_color,boundary_color):
    nfaces = np.shape(faces)[1]
    for i in range(0,nfaces):
        if faces[2,i] == 2:     # face type >> sigma = const. (red)
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],color=sigma_color,linestyle='-',linewidth = line_width)
        elif faces[2,i] == 1:   # face type >> lambda = const. (blue)
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],color=lambda_color,linestyle='-',linewidth = line_width)
        else:                   # any other face type (black)  
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],color=boundary_color,linestyle='-',linewidth = line_width)
            
    return
###############################################################################
    

if prof_plots == 1:
    print("######## prof_plots ########")
    
    marker_size  = 6
    marker_size_cath = 14
    cathode_marker = '*'
    cathode_color  = orange
    ref_color      = 'c'
    marker_every = 3
    marker_every_2 = 4
    marker_every_mesh = 180
    text_size           = 20
    ticks_size          = 20
    font_size           = 20
#    font_size_legend    = font_size - 15
    font_size_legend    = font_size
    # font_size_legend    = 8
    font_size_legend    = 15
    font_size_legend    = 11
    
    line_width = 2.5
    
    marker_every_mesh_P1 = marker_every_mesh
    marker_every_mesh_P2 = marker_every_mesh + 50
    marker_every_mesh_P3 = marker_every_mesh + 100
    marker_every_mesh_P4 = marker_every_mesh
    
    marker_every_P1 = marker_every
    marker_every_P2 = marker_every + 1
    marker_every_P3 = marker_every + 2
    marker_every_P4 = marker_every + 3
    
    # Radial index for axial profiles
#    rind = 21   # New picM for SPT100
#    rind = 32   # New picM for SPT100 (picrm)
#    rind = 17   # Safran T1,T2 cases
#    rind = 17   # VHT_US MP coarse mesh
#    rind = 20   # VHT_US MP fine mesh
#    rind = 15    # VHT_US MP fine mesh Np
#    rind = 36   # Safran T1,T2 cases (picrm)
#    rind = 19   # picM for SPT100 thesis
#    rind = 29    # HT5k rm6
#    rind = 15   # Safran PPSX00 Cheops LP
#    rind = 15    # VHT_US LP (TFM Alejandro)
    rind = 15    # PPSX00 testcase2

    # Axial coordinate for radial profiles in cm (closest zindex is obtained automatically)
    z_rprof = 10 # PPSX00 testcase2
    # z_rprof = 22 # VHET US MP
    z_rprof = 22.9 # VHET US MP (boundary of P2)
    # z_rprof = 18.61 # VHET US MP (P2G j=0 point axial position)
    # z_rprof = 22.301 # VHET US MP (P4G j=0 point axial position)
    z_rprof = 4.35 # VHET US MP (at z/Lc = 1.5 to see cathode near plume in cathode cases)
    z_rprof = 3.625 # VHET US MP (at z/Lc = 1.25 to see cathode near plume in cathode cases)
    
    # Decide log scale (1) or linear scale (0) in y-axis for rprof plots of currents
    # When plotting with log scale, negative currents (for example in z,r components) appear in absolute value
    log_curr_rprof = 0
    
    
    # Options for plotting angular profiles in plume (Faraday probe scans)
    # Decide log scale (1) or linear scale (0) in y-axis for scan plots of currents
    # When plotting with log scale, negative currents (for example in z,r components) appear in absolute value
    log_curr_scan = 0
    # Settings for scans from mid radius of the VHET US MP 
    z_offset    = -2.9 # Distance (cm) from anode to axial position of the axis of Faraday probe scan
    r_offset    = 6.56  # Offset radius (cm) of the axis of the Faraday probe scan
    rscan       = 6.56  # Radius (cm) of the Faraday probe scan 
    ang_min     = -90   # Minimum angle for the profile (deg)
    ang_max     = 90  # Maximum angle for the profile (deg)
    Npoints_ang = 200 # Number of points for the profile 
    # Settings for scans from axis at left bottom corner of the plume for the VHET US MP
    z_offset    = -2.9 # Distance (cm) from anode to axial position of the axis of Faraday probe scan
    r_offset    = 0.00 # Offset radius (cm) of the axis of the Faraday probe scan
    #rscan       = 20   # Radius (cm) of the Faraday probe scan 
    # rscan       = 8.7   # Radius (cm) of the Faraday probe scan 
    rscan       = 18.85   # Radius (cm) of the Faraday probe scan 
    ang_min     = -90    # Minimum angle for the profile (deg)
    ang_max     = 90   # Maximum angle for the profile (deg)
    Npoints_ang = 200  # Number of points for the profile 

    # Cathode plotting flag and cathode position in cm (for plot_zcath_012 = 2,3) for axial profiles
    plot_zcath_012 = 2                  # 0 - Deactivated
                                        # 1 - Plot z_cath (cathode z position from eFld mesh)
                                        # 2 - Plot zcat_pos indicated below (cross Cathode Bline with axial profile)
                                        # 3 - Plot zcat_pos_2 (additional cross Cathode Bline with axial profile)
    zcat_pos       = 5.9394542444501024 # z coordinate of crossing point of cathode C1, C2 and C3 Bline with rind = 19
#    zcat_pos_2     = 9.6917             # z coordinate of crossing point of cathode C5 (C4 thesis) Bline with rind=19
    # TOPO 2 ------------------------------------------------------------------
    plot_zcath_012 = 2
#    zcat_pos       = 15.55106875         # z coordinate of crossing point of cathode topo2 3298 and 3283 Bline with rind = 17
    zcat_pos       = 7.47248             # z coordinate of crossing point of cathode topo2 1200 Bline with rind = 17
#    zcat_pos       = 5.902074            # z coordinate of crossing point of cathode topo2 2853 Bline with rind = 17
    # TOPO 1 ------------------------------------------------------------------
    plot_zcath_012 = 2
#    zcat_pos       = 12.14428            # z coordinate of crossing point of cathode topo1 699 Bline with rind = 17
#    zcat_pos       = 7.3422075           # z coordinate of crossing point of cathode topo1 313 Bline with rind = 17   
#    zcat_pos       = 5.688635            # z coordinate of crossing point of cathode topo1 251 Bline with rind = 17 
    # VHT_US MP (IEPC2022) ----------------------------------------------------
    zcat_pos = 15.55
    zcat_pos = 15.71  # paper GDML
    zcat_pos2 = 16.104 # paper GDML: Plot cross with MS (magnetic separatrix) instead of cross with cathode line
#    # VHT_US LP (TFM Alejandro) -----------------------------------------------
#    zcat_pos = 6.25
#    # PPSX00 testcase1 (D4.1 LP) ----------------------------------------------
#    zcat_pos = 5.74805
#    # PPSX00 testcase2 em1 cathode face ID = 5803 -----------------------------
#    zcat_pos = 5.678125
#    # PPSX00 testcase2 em2 cathode face ID = 6613 -----------------------------
#    zcat_pos = 5.777875
#    # PPSX00 testcase2 em2 cathode face ID = 4656 -----------------------------
#    zcat_pos = 5.76525
    
    
    
    elems_cath_Bline    = range(407-1,483-1+2,2) # Elements along the cathode B line for cases C1, C2 and C3
    elems_cath_Bline_2  = range(875-1,951-1+2,2) # Elements along the cathode B line for case C5 (C4 thesis)
    elems_Bline         = range(330-1,406-1+2,2) # Elements along a B line
    elems_cath_Bline    = list(range(994-1,926-1+2,-2)) + list([923-1,922-1,920-1,917-1,916-1,914-1,912-1,909-1,907-1,906-1,904-1]) # Elements along the cathode B line for HT5k rm6 cathode at volume 922 or face 1637
#    elems_cath_Bline    = list(range(1968-1,1922-1+2,-2)) + list([1925-1]) + list(range(1922-1,1908-1+2,-2)) +list([1911-1]) + list(range(1908-1,1894-1+2,-2)) + list([1896-1,1895-1,1892-1,1890-1,1888-1,1886-1,1884-1,1882-1,1880-1,1971-1]) # Elements along the cathode B line for HT5k rm4 cathode at volume 1966 or face 3464
    ref_elem            = elems_Bline[int(len(elems_Bline)/2)]
#    ref_elem            = elems_Bline[0]
    ref_elem            = elems_cath_Bline[1]
    elems_cath_Bline   = []
    elems_cath_Bline_2 = []
    plot_Bline_cathBline = 1          # Only used for plots activated when plot_cath_Bline_prof = 1
    
    # Common reference potential PIC mesh node Python indeces
    phi_ref  = 0
    iphi_ref = 24
    jphi_ref = 28
    
    # Print out time steps
#    timestep = 'last'
    timestep = 300
  
    allsteps_flag   = 1
    read_inst_data  = 0
    read_part_lists = 0
    read_flag       = 1
    
    mean_vars       = 1
    mean_type       = 0
    # last_steps      = 600
    # last_steps      = 700
    # last_steps      = 1000
#    last_steps      = 500
    last_steps      = 1200
#    last_steps      = 100
#    last_steps      = 40
    step_i          = 700
    step_f          = 960
    plot_mean_vars  = 1

        
    # Flag to decide if interpolate from MFAM to a finer picM for plotting phi, Te and je components (Recommended = 1)
    interp_MFAM_picM_plot = 1
    
    
    plot_B_scan      = 0
    plot_fields_scan = 0
    plot_dens_scan   = 0
    plot_temp_scan   = 0
    plot_curr_scan   = 0
    plot_freq_scan   = 0
    



    if timestep == 'last':
        timestep = -1  
    if allsteps_flag == 0:
        mean_vars = 0

    
    # Simulation names
    nsims = 20
    # nsims = 10
    # nsims = 6
    # nsims = 1

    # Flag for old sims (1: old sim files, 0: new sim files)
    oldpost_sim      = np.array([6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],dtype = int)
    oldsimparams_sim = np.array([20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20],dtype = int)  
    oldsimparams_sim = np.array([21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21],dtype = int)  


    
    sim_names = [
                    # "../../../sim/sims/P1L",
                    # "../../../sim/sims/P2L",
                    # "../../../sim/sims/P3L",
                    # "../../../sim/sims/P4L",
                    # "../../../sim/sims/P2L_fcat3198",
                    # "../../../sim/sims/P3L_fcat1962",
                    # "../../../sim/sims/P4L_fcat7610_Fz",
                    # "../../../sim/sims/P1G",
                    # "../../../sim/sims/P2G",
                    # "../../../sim/sims/P3G",
                    # "../../../sim/sims/P4G",
                    # "../../../sim/sims/P2G_fcat3198",
                    # "../../../sim/sims/P3G_fcat1962",
                    # "../../../sim/sims/P4G_fcat7610_Fz",
                    
                    # "../../../sim/sims/P1L_Tcath_new",
                    # "../../../sim/sims/P2L_Tcath_new",
                    # "../../../sim/sims/P3L_Tcath_new",
                    # "../../../sim/sims/P4L_Fz_Tcath_new",
                    # "../../../sim/sims/P2L_fcat3198_Tcath_new",
                    # "../../../sim/sims/P3L_fcat1962_Tcath_new",
                    # "../../../sim/sims/P4L_fcat7610_Fz_Tcath_new",
                    # "../../../sim/sims/P2L_fcat2543_2542_Tcath_new",
                    # "../../../sim/sims/P3L_fcat6259_5993_Tcath_new",
                    # "../../../sim/sims/P4L_fcat6266_2356_Fz_Tcath_new",
                    # "../../../sim/sims/P1G_Tcath_new",
                    # "../../../sim/sims/P2G_Tcath_new",
                    # "../../../sim/sims/P3G_Tcath_new",
                    # "../../../sim/sims/P4G_Fz_Tcath_new",
                    # "../../../sim/sims/P2G_fcat3198_Tcath_new",
                    # "../../../sim/sims/P3G_fcat1962_Tcath_new",
                    # "../../../sim/sims/P4G_fcat7610_Fz_Tcath_new",
                    # "../../../sim/sims/P2G_fcat2543_2542_Tcath_new",
                    # "../../../sim/sims/P3G_fcat6259_5993_Tcath_new",
                    # "../../../sim/sims/P4G_fcat6266_2356_Fz_Tcath_new",
        
        
                    "../../../sim/sims/P4L_fcat6266_2356_Fz_Tcath_new",
                    "../../../sim/sims/P3L_fcat6259_5993_Tcath_new",
                    "../../../sim/sims/P2L_fcat2543_2542_Tcath_new",
                    "../../../sim/sims/P4L_fcat7610_Fz_Tcath_new",
                    "../../../sim/sims/P3L_fcat1962_Tcath_new",
                    "../../../sim/sims/P2L_fcat3198_Tcath_new",
                    "../../../sim/sims/P4L_Fz_Tcath_new",
                    "../../../sim/sims/P3L_Tcath_new",
                    "../../../sim/sims/P2L_Tcath_new",
                    "../../../sim/sims/P1L_Tcath_new",
                    "../../../sim/sims/P4G_fcat6266_2356_Fz_Tcath_new",
                    "../../../sim/sims/P3G_fcat6259_5993_Tcath_new",
                    "../../../sim/sims/P2G_fcat2543_2542_Tcath_new",
                    "../../../sim/sims/P4G_fcat7610_Fz_Tcath_new",
                    "../../../sim/sims/P3G_fcat1962_Tcath_new",
                    "../../../sim/sims/P2G_fcat3198_Tcath_new",
                    "../../../sim/sims/P4G_Fz_Tcath_new",
                    "../../../sim/sims/P3G_Tcath_new",
                    "../../../sim/sims/P2G_Tcath_new",
                    "../../../sim/sims/P1G_Tcath_new",
                    

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
#                              "aspire_picM_rm6.hdf5",
#                              "PIC_mesh_LP.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                             ]
    elif topo_case == 0:    
        PIC_mesh_file_name = [
                              "PIC_mesh.hdf5", # VHT_US
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
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              ]


    PIC_mesh_plot_file_name = [
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               "PIC_mesh_plot.hdf5",
                               ]
    

    labels = [
        
               # r"P1L1",
               # r"P2L1",
               # r"P3L1",
               # r"P4L1",
               # r"P2L2",
               # r"P3L2",
               # r"P4L2",
               # r"P1G1",
               # r"P2G1",
               # r"P3G1",
               # r"P4G1",
               # r"P2G2",
               # r"P3G2",
               # r"P4G2",
               
               # r"LP1C1",
               # r"LP2C1",
               # r"LP3C1",
               # r"LP4C1",
               # r"LP2C2",
               # r"LP3C2",
               # r"LP4C2",
               # r"LP2C3",
               # r"LP3C3",
               # r"LP4C3",
               # r"GP1C1",
               # r"GP2C1",
               # r"GP3C1",
               # r"GP4C1",
               # r"GP2C2",
               # r"GP3C2",
               # r"GP4C2",
               # r"GP2C3",
               # r"GP3C3",
               # r"GP4C3",
               
               r"LP4C3",
               r"LP3C3",
               r"LP2C3",
               r"LP4C2",
               r"LP3C2",
               r"LP2C2",
               r"LP4C1",
               r"LP3C1",
               r"LP2C1",
               r"LP1C1",
               r"GP4C3",
               r"GP3C3",
               r"GP2C3",
               r"GP4C2",
               r"GP3C2",
               r"GP2C2",
               r"GP4C1",
               r"GP3C1",
               r"GP2C1",
               r"GP1C1",

              ]

    
    # Line colors
#    colors = ['k','r','g','b','m','c','m','y',orange,brown]
#    colors = ['r','k','g','b','m','c','m','y',orange,brown]
    # colors = ['m','b','k','r','g','b','m','c','m','y',orange,brown] # P1G-P4G (comp with alejandro)
    colors = ['m','k','b','r',grey,'c',orange,'m','k','b','r',grey,'c',orange] # P1G-P4G, P1L-P4L (paper)
    # colors = ['k','b','r','m','g','c','m','c','m','y',orange,brown] # P1G-P4G, P1L-P4L (paper) cathode cases
    # colors = ['k','r','k','r','k','r',orange,brown]
    # colors = ['k','r','g','b','m','c',orange,brown]
    colors = ['m','k','b','r',grey,'c',orange,brown,'g','y','m','k','b','r',grey,'c',orange,brown,'g','y']
    colors = ['m','k','b','r','k','b','r','k','b','r','m','k','b','r','k','b','r','k','b','r']
    
    # Different color for different plume size
    colors = ['r','b','k',
              'r','b','k',
              'r','b','k','m',
              'r','b','k',
              'r','b','k',
              'r','b','k','m']
    # Different color for different plume size and cathode
    colors = ['salmon','red','darkred',
              'royalblue','blue','darkblue',
              'lightgrey','darkgrey','dimgrey','black',
              'salmon','red','darkred',
              'royalblue','blue','darkblue',
              'lightgrey','darkgrey','dimgrey','black']
    
    # Markers
    markers = ['s','o','v','^','<', '>','D','p','*']
    markers = ['','','','','s','o','v','','','','','s','o','v']
    markers = ['','','','','s','o','','','','','s','o','v','p','D']
    markers = ['v','D','s','o','D','s','o','D','s','o','v','D','s','o','D','s','o','D','s','o']
    # Different marker for different plume size
    markers = ['o','s','D', 
               'o','s','D', 
               'o','s','D','v',
               'o','s','D', 
               'o','s','D', 
               'o','s','D','v']
    # Different marker for different cathode
    markers = ['o','o','o', 
               's','s','s', 
               'v','v','v','v',
               'o','o','o', 
               's','s','s', 
               'v','v','v','v',]
    
#    markers = ['s','<','D','p']
    marker_size_vec = [marker_size,marker_size,marker_size,marker_size,
                       marker_size,marker_size,marker_size+1,
                       marker_size,marker_size,marker_size,marker_size,
                       marker_size,marker_size,marker_size+1]
    marker_size_vec = [marker_size,marker_size,marker_size,marker_size,
                       marker_size,marker_size,marker_size+1,
                       marker_size,marker_size,marker_size,marker_size,
                       marker_size,marker_size,marker_size+1,marker_size+1,marker_size]
    marker_size_vec = [marker_size,marker_size,marker_size,marker_size,marker_size,
                       marker_size,marker_size,marker_size,
                       marker_size,marker_size,marker_size,
                       marker_size,marker_size,marker_size,marker_size,marker_size,
                       marker_size,marker_size,marker_size,
                       marker_size,marker_size,marker_size]
    marker_size_vec = [marker_size,marker_size,marker_size,
                       marker_size,marker_size,marker_size,
                       marker_size+1,marker_size+1,marker_size+1,marker_size+1,
                       marker_size,marker_size,marker_size,
                       marker_size,marker_size,marker_size,
                       marker_size+1,marker_size+1,marker_size+1,marker_size+1]
    

    marker_every_mesh_vec = [marker_every_mesh_P1,marker_every_mesh_P2,marker_every_mesh_P3,marker_every_mesh_P4,
                             marker_every_mesh_P2,marker_every_mesh_P3,marker_every_mesh_P4,
                             marker_every_mesh_P2,marker_every_mesh_P3,marker_every_mesh_P4,
                             marker_every_mesh_P1,marker_every_mesh_P2,marker_every_mesh_P3,marker_every_mesh_P4,
                             marker_every_mesh_P2,marker_every_mesh_P3,marker_every_mesh_P4,
                             marker_every_mesh_P2,marker_every_mesh_P3,marker_every_mesh_P4]
    marker_every_mesh_vec = [marker_every_mesh_P4,marker_every_mesh_P3,marker_every_mesh_P2,
                             marker_every_mesh_P4,marker_every_mesh_P3,marker_every_mesh_P2,
                             marker_every_mesh_P4,marker_every_mesh_P3,marker_every_mesh_P2,marker_every_mesh_P1,
                             marker_every_mesh_P4,marker_every_mesh_P3,marker_every_mesh_P2,
                             marker_every_mesh_P4,marker_every_mesh_P3,marker_every_mesh_P2,
                             marker_every_mesh_P4,marker_every_mesh_P3,marker_every_mesh_P2,marker_every_mesh_P1]
    marker_every_mesh_vec = [marker_every_mesh_P4+50,marker_every_mesh_P3+50,marker_every_mesh_P2+50,
                             marker_every_mesh_P4-50,marker_every_mesh_P3-50,marker_every_mesh_P2-50,
                             marker_every_mesh_P4,marker_every_mesh_P3,marker_every_mesh_P2,marker_every_mesh_P1,
                             marker_every_mesh_P4+50,marker_every_mesh_P3+50,marker_every_mesh_P2+50,
                             marker_every_mesh_P4-50,marker_every_mesh_P3-50,marker_every_mesh_P2-50,
                             marker_every_mesh_P4,marker_every_mesh_P3,marker_every_mesh_P2,marker_every_mesh_P1]

    marker_every_vec = [marker_every_P1,marker_every_P2,marker_every_P3,marker_every_P4,
                        marker_every_P2-1,marker_every_P3-1,marker_every_P4-1,
                        marker_every_P2+1,marker_every_P3+1,marker_every_P4+1,
                        marker_every_P1,marker_every_P2-2,marker_every_P3-2,marker_every_P4-2,
                        marker_every_P2+2,marker_every_P3+2,marker_every_P4+2,
                        marker_every_P2+3,marker_every_P3+3,marker_every_P4+3]
    
    marker_every_vec = [marker_every_P4+2,marker_every_P3+2,marker_every_P2+2,
                        marker_every_P4-2,marker_every_P3-2,marker_every_P2-2,
                        marker_every_P4,marker_every_P3,marker_every_P2,marker_every_P1,
                        marker_every_P4+2,marker_every_P3+2,marker_every_P2+2,
                        marker_every_P4-2,marker_every_P3-2,marker_every_P2-2,
                        marker_every_P4,marker_every_P3,marker_every_P2,marker_every_P1]
    
    marker_every_vec = [marker_every_P4+1,marker_every_P3+2,marker_every_P2+2,
                        marker_every_P4-1,marker_every_P3-1,marker_every_P2-1,
                        marker_every_P4,marker_every_P3,marker_every_P2,marker_every_P1,
                        marker_every_P4+1,marker_every_P3+2,marker_every_P2+2,
                        marker_every_P4-1,marker_every_P3-1,marker_every_P2-1,
                        marker_every_P4,marker_every_P3,marker_every_P2,marker_every_P1]
                        
    # Line style
    linestyles = ['-','--','-.',':','-','-','-','-','--','-.',':','-','-','-']
    linestyles = ['-','--','-.',':','-','-','-','-','--','-.',':','-','-','-','-','-']
    linestyles = ['-','-','-','-','--','--','--','-.','-.','-.','-','-','-','-','--','--','--','-.','-.','-.']
    
    # Different linestyle for different cathode 
    linestyles = ['-.','-.','-.',
                  '--','--','--',
                  '-','-','-','-',
                  '-.','-.','-.',
                  '--','--','--',
                  '-','-','-','-']
    # Different linestyle for different plume size 
    linestyles = ['-','--','-.',
                  '-','--','-.',
                  '-','--','-.',':',
                  '-','--','-.',
                  '-','--','-.',
                  '-','--','-.',':']
    # linestyles = ['-','-.',':', ':','-','--','-.']
    # linestyles = ['-','-','-','-','-','-','-']
#    linestyles = ['--','-','-.',':','-','--','-.'] # P1G-P4G, P1L-P4L
    # linestyles = ['-','-','--','--',':',':']
    # linestyles = ['-',':','-',':','-',':']

    
    # zmax for profiles (cm)
#    xmax = 18  # FOR CHEOPS-1 Topo1 Topo2 sims
#    xmax = 12 # FOR SPT sims
#    xmax = 17.67 # FOR HT5k
#    xmax = 7.5  # FOR cheops LP sims
#    xmax = 12.9  # FOR VHT_US MP sims plume 10
    # xmax = 22.9  # FOR VHT_US MP sims plume 20
    # xmax = 32.9  # FOR VHT_US MP sims plume 30
    xmax = 42.9  # FOR VHT_US MP sims plume 40
#    xmax = 15 # For VHT_US LP case 6L (TFM Alejandro)
#    xmax = 22.5 # For VHT_US LP case 9L (TFM Alejandro)
    
    # For radial profiles (cm)
    rmax = 30   # FOR VHT_US MP sims plume 20 (at boundary)
    # rmax = 28.3 # For P2G at j=0 point
    # rmax = 29.705 # For P4G at j=0 point
    rmax = 22.2 # For z/Lc = 1.5 at the cathode near plume 
    
    
    # Do not plot units in axes (comment if we want units in axes)
#    # SAFRAN CHEOPS 1: units in cm
#    L_c = 3.725
#    H_c = (0.074995-0.052475)*100
     # HT5k
#    L_c = 2.53
#    H_c = (0.0785-0.0565)*100
    # VHT_US MP
    L_c = 2.9
    H_c = 2.22
    # VHT_US LP (TFM Alejandro) and PPSX00 testcase1 LP D4.1
#    L_c = 2.5
#    H_c = 1.1
    # PPSX00 testcase2
#    L_c = 2.5
#    H_c = 1.5

    xmax = xmax/L_c
    rmax = rmax/H_c
    zcat_pos = zcat_pos/L_c
    zcat_pos2 = zcat_pos2/L_c
    
#    prof_xlabel = r"$z$ (cm)"
    prof_xlabel = r"$z/L_\mathrm{c}$"

#    rprof_xlabel = r"$r$ (cm)"
    rprof_xlabel = r"$r/H_\mathrm{c}$"
    
    
    scan_xlabel = r"$\theta$ (deg)"
    
    
    
    # [fig1, axes1] = plt.subplots(nrows=3, ncols=2, figsize=(15,18))
    
    [fig1, axes1] = plt.subplots(nrows=3, ncols=2, figsize=(15,9))
    
    axes1[0,0].set_ylabel(r"$\phi$ (V)",fontsize = font_size)
    # axes1[0,0].set_xlabel(prof_xlabel,fontsize = font_size)    
    axes1[0,0].set_xlim(3,15)
    axes1[0,0].set_xticks(np.arange(3,15+1,1))
    # axes1[0,0].set_ylim(0,70)
    # axes1[0,0].set_yticks([0,10,20,30,40,50,60,70])
    axes1[0,0].set_ylim(0,70)
    axes1[0,0].set_yticks([0,10,20,30,40,50,60,70])
    text = '(a)'
    zstext = 0.15
    rstext = 0.925
    axes1[0,0].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[0,0].transAxes)  
    zstext_phi_LC2 = 0.29
    rstext_phi_LC2 = 0.73
    zstext_phi_LC1 = 0.17
    rstext_phi_LC1 = 0.34
    zstext_phi_LC3 = 0.05
    rstext_phi_LC3 = 0.10
    text_C1 = 'C1'
    text_C2 = 'C2'
    text_C3 = 'C3'
    axes1[0,0].text(zstext_phi_LC1,rstext_phi_LC1,text_C1,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[0,0].transAxes)  
    axes1[0,0].text(zstext_phi_LC2,rstext_phi_LC2,text_C2,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[0,0].transAxes)  
    axes1[0,0].text(zstext_phi_LC3,rstext_phi_LC3,text_C3,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[0,0].transAxes)  
    axes1[0,0].tick_params(labelsize = ticks_size,labelbottom=False)
    
    axes1[1,0].set_ylabel(r"$T_\mathrm{e}$ (eV)",fontsize = font_size)
    # axes1[1,0].set_xlabel(prof_xlabel,fontsize = font_size)
    axes1[1,0].set_xlim(3,15)
    axes1[1,0].set_xticks(np.arange(3,15+1,1))
    # axes1[1,0].set_ylim(2,16)
    # axes1[1,0].set_yticks([2,4,6,8,10,12,14,16])
    axes1[1,0].set_ylim(0,16)
    axes1[1,0].set_yticks([0,2,4,6,8,10,12,14,16])
    text = '(c)'
    # zstext = 1.1*3
    # rstext = 0.9*14
    axes1[1,0].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[1,0].transAxes)  
    zstext_Te_LC2 = 0.5
    rstext_Te_LC2 = 0.77
    zstext_Te_LC1 = 0.15
    rstext_Te_LC1 = 0.48
    zstext_Te_LC3 = 0.08
    rstext_Te_LC3 = 0.15
    axes1[1,0].text(zstext_Te_LC1,rstext_Te_LC1,text_C1,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[1,0].transAxes)  
    axes1[1,0].text(zstext_Te_LC2,rstext_Te_LC2,text_C2,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[1,0].transAxes)  
    axes1[1,0].text(zstext_Te_LC3,rstext_Te_LC3,text_C3,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[1,0].transAxes)
    axes1[1,0].tick_params(labelsize = ticks_size,labelbottom=False)
    
    axes1[2,0].set_ylabel(r"$n_\mathrm{e}$ (10$^{16}$ m$^{-3}$)",fontsize = font_size)
    axes1[2,0].set_xlabel(prof_xlabel,fontsize = font_size)
    axes1[2,0].set_xlim(3,15)
    axes1[2,0].set_xticks(np.arange(3,15+1,1))
    # axes1[2,0].set_ylim(1E16,3E17)
    # axes1[2,0].set_yticks([1E16,5E16,1E17,3E17])
    # axes1[2,0].set_ylim(2E16,2E17)
    # axes1[2,0].set_yticks([2E16,4E16,6E16,8E16,1E17,1.2E17,1.4E17,1.6E17,1.8E17,2E17])
    # axes1[2,0].set_ylim(0.2,2.0)
    # axes1[2,0].set_yticks([0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0])
    axes1[2,0].set_ylim(2.0,23.0)
    axes1[2,0].set_yticks([2,5,8,11,14,17,20,23])
    text = '(e)'
    # zstext = 1.1*3
    # rstext = 0.9*3E17
    axes1[2,0].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[2,0].transAxes)  
    axes1[2,0].tick_params(labelsize = ticks_size)
    
    
    axes1[0,1].set_ylabel(r"$\phi$ (V)",fontsize = font_size)
    # axes1[0,1].set_xlabel(prof_xlabel,fontsize = font_size)    
    axes1[0,1].set_xlim(3,15)
    axes1[0,1].set_xticks(np.arange(3,15+1,1))
    # axes1[0,1].set_ylim(0,70)
    # axes1[0,1].set_yticks([0,10,20,30,40,50,60,70])
    axes1[0,1].set_ylim(0,70)
    axes1[0,1].set_yticks([0,10,20,30,40,50,60,70])
    text = '(b)'
    # zstext = 1.1*3
    # rstext = 0.9*40
    axes1[0,1].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[0,1].transAxes)  
    zstext_phi_GC2 = 0.29
    rstext_phi_GC2 = 0.6
    zstext_phi_GC1 = 0.17
    rstext_phi_GC1 = 0.32
    zstext_phi_GC3 = 0.05
    rstext_phi_GC3 = 0.10
    text_C1 = 'C1'
    text_C2 = 'C2'
    text_C3 = 'C3'
    axes1[0,1].text(zstext_phi_GC1,rstext_phi_GC1,text_C1,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[0,1].transAxes)  
    axes1[0,1].text(zstext_phi_GC2,rstext_phi_GC2,text_C2,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[0,1].transAxes)  
    axes1[0,1].text(zstext_phi_GC3,rstext_phi_GC3,text_C3,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[0,1].transAxes) 
    axes1[0,1].tick_params(labelsize = ticks_size,labelbottom=False)
    
    axes1[1,1].set_ylabel(r"$T_\mathrm{e}$ (eV)",fontsize = font_size)
    # axes1[1,1].set_xlabel(prof_xlabel,fontsize = font_size)
    axes1[1,1].set_xlim(3,15)
    axes1[1,1].set_xticks(np.arange(3,15+1,1))
    # axes1[1,1].set_ylim(2,16)
    # axes1[1,1].set_yticks([2,4,6,8,10,12,14,16])
    axes1[1,1].set_ylim(0,16)
    axes1[1,1].set_yticks([0,2,4,6,8,10,12,14,16])
    text = '(d)'
    # zstext = 1.1*3
    # rstext = 0.9*14
    axes1[1,1].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[1,1].transAxes)  
    zstext_Te_GC2 = 0.5
    rstext_Te_GC2 = 0.64
    zstext_Te_GC1 = 0.15
    rstext_Te_GC1 = 0.44
    zstext_Te_GC3 = 0.08
    rstext_Te_GC3 = 0.15
    axes1[1,1].text(zstext_Te_GC1,rstext_Te_GC1,text_C1,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[1,1].transAxes)  
    axes1[1,1].text(zstext_Te_GC2,rstext_Te_GC2,text_C2,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[1,1].transAxes)  
    axes1[1,1].text(zstext_Te_GC3,rstext_Te_GC3,text_C3,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[1,1].transAxes)
    axes1[1,1].tick_params(labelsize = ticks_size,labelbottom=False)
    
    axes1[2,1].set_ylabel(r"$n_\mathrm{e}$ (10$^{16}$ m$^{-3}$)",fontsize = font_size)
    axes1[2,1].set_xlabel(prof_xlabel,fontsize = font_size)
    axes1[2,1].set_xlim(3,15)
    axes1[2,1].set_xticks(np.arange(3,15+1,1))
    # axes1[2,1].set_ylim(1E16,3E17)
    # axes1[2,1].set_yticks([1E16,5E16,1E17,3E17])
    # axes1[2,1].set_ylim(2E16,2E17)
    # axes1[2,1].set_yticks([2E16,4E16,6E16,8E16,1E17,1.2E17,1.4E17,1.6E17,1.8E17,2E17])
    # axes1[2,1].set_ylim(0.2,2.0)
    # axes1[2,1].set_yticks([0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0])
    axes1[2,1].set_ylim(2.0,23.0)
    axes1[2,1].set_yticks([2,5,8,11,14,17,20,23])
    text = '(f)'
    # zstext = 1.1*3
    # rstext = 0.9*3E17
    axes1[2,1].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[2,1].transAxes)  
    axes1[2,1].tick_params(labelsize = ticks_size)
    
    
    ind  = 0
    ind2 = 0
    ind3 = 0
    for k in range(0,nsims):
        ind_ini_letter = sim_names[k].rfind('/') + 1
        print("##### CASE "+str(k+1)+": "+sim_names[k][ind_ini_letter::]+" #####")
        
        # if k==3:
        #     print("WARNING: last_steps modified for CASE = "+str(k+1))
        #     last_steps = 800
        
        print("##### oldsimparams_sim = "+str(oldsimparams_sim[k])+" #####")
        print("##### oldpost_sim      = "+str(oldpost_sim[k])+" #####")
        print("##### last_steps       = "+str(last_steps)+" #####")
        ######################## READ INPUT/OUTPUT FILES ##########################
        # Obtain paths to simulation files
        path_picM         = sim_names[k]+"/SET/inp/"+PIC_mesh_file_name[k]
        path_picM_plot    = sim_names[k]+"/SET/inp/"+PIC_mesh_plot_file_name[k]
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
           versors_e,versors_f,n_faces,n_elems,n_faces_boundary,bIDfaces_Dwall,bIDfaces_Awall,
           bIDfaces_FLwall,IDfaces_Dwall,IDfaces_Awall,IDfaces_FLwall,zfaces_Dwall,
           rfaces_Dwall,Afaces_Dwall,zfaces_Awall,rfaces_Awall,Afaces_Awall,
           zfaces_FLwall,rfaces_FLwall,Afaces_FLwall,zfaces_Cwall,rfaces_Cwall,Afaces_Cwall,
           cath_elem,z_cath,r_cath,V_cath,mass,ssIons1,ssIons2,ssNeutrals1,ssNeutrals2,
           n_mp_i1_list,n_mp_i2_list,n_mp_n1_list,n_mp_n2_list,
           alpha_ano,alpha_ano_e,alpha_ano_q,alpha_ine,alpha_ine_q,
           alpha_ano_elems,alpha_ano_e_elems,alpha_ano_q_elems,alpha_ine_elems,
           alpha_ine_q_elems,alpha_ano_faces,alpha_ano_e_faces,alpha_ano_q_faces,
           alpha_ine_faces,alpha_ine_q_faces,
           phi,phi_elems,phi_faces,Ez,Er,Efield,Bz,Br,Bfield,Te,Te_elems,Te_faces,
           je_mag_elems,je_perp_elems,je_theta_elems,je_para_elems,je_z_elems,je_r_elems,
           je_mag_faces,je_perp_faces,je_theta_faces,je_para_faces,je_z_faces,je_r_faces,
           cs01,cs02,cs03,cs04,nn1,nn2,nn3,ni1,ni2,ni3,ni4,
           ne,ne_elems,ne_faces,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fn3_x,fn3_y,fn3_z,
           fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,
           un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,
           ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,ui4_z,
           ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,ji3_z,ji4_x,ji4_y,ji4_z,
           je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,
           ue_perp,ue_para,uthetaExB,Tn1,Tn2,Tn3,Ti1,Ti2,Ti3,Ti4,
           n_mp_n1,n_mp_n2,n_mp_n3,n_mp_i1,n_mp_i2,n_mp_i3,n_mp_i4,
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
           thrust_m,thrust_pres,Id_inst,Id,Vd_inst,Vd,I_beam,I_tw_tot,Pd,Pd_inst,P_mat,
           P_inj,P_inf,P_ion,P_ex,P_use_tot_i,P_use_tot_n,P_use_tot,P_use_z_i,P_use_z_n,
           P_use_z_e,P_use_z,qe_wall,qe_wall_inst,Pe_faces_Dwall,Pe_faces_Awall,
           Pe_faces_FLwall,Pe_faces_Dwall_inst,Pe_faces_Awall_inst,Pe_faces_FLwall_inst,
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
           nu_cath,ndot_cath,Q_cath,P_cath,V_cath_tot,ne_cath_avg,
           F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,
           nu_ei2,nu_i01,nu_i02,nu_i12,nu_ex,
           F_theta_elems,Hall_par_elems,Hall_par_eff_elems,nu_e_tot_elems,
           nu_e_tot_eff_elems,F_theta_faces,Hall_par_faces,Hall_par_eff_faces,
           nu_e_tot_faces,nu_e_tot_eff_faces,nu_en_elems,nu_ei1_elems,
           nu_ei2_elems,nu_i01_elems,nu_i02_elems,nu_i12_elems,nu_ex_elems,
           nu_en_faces,nu_ei1_faces,nu_ei2_faces,nu_i01_faces,nu_i02_faces,
           nu_i12_faces,nu_ex_faces, 
           felec_para_elems,felec_para_faces,felec_perp_elems,felec_perp_faces,
           felec_z_elems,felec_z_faces,felec_r_elems,felec_r_faces,
           Boltz,Boltz_dim,Pfield_e,Ebal_e,
           dphi_sh_b,dphi_sh_b_Te,imp_ene_e_b,imp_ene_e_b_Te,imp_ene_e_wall,
           imp_ene_e_wall_Te,ge_b,ge_b_acc,ge_sb_b,ge_sb_b_acc,delta_see,
           delta_see_acc,err_interp_n,n_cond_wall,Icond,Vcond,Icath,phi_inf,
           I_inf,f_split,f_split_adv,f_split_qperp,f_split_qpara,f_split_qb,
           f_split_Pperp,f_split_Ppara,f_split_ecterm,f_split_inel] = HET_sims_read(path_simstate_inp,path_simstate_out,
                                                                                    path_postdata_out,path_simparams_inp,
                                                                                    path_picM,allsteps_flag,timestep,read_inst_data,
                                                                                    read_part_lists,read_flag,oldpost_sim[k],oldsimparams_sim[k])
                    
        
        #### NOTE: After change in eFld collisions, ionization collisions are 
        #          not multiplied by the charge number jump (as before). 
        #          We do it here
        if oldpost_sim[k] >= 3:
            nu_i02 = 2.0*nu_i02
        #######################################################################
        
        
        print("Generating plotting variables (NaN in ghost nodes)...")                                                                                                      
        [Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,
      nn2,nn3,ni1,ni2,ni3,ni4,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,
      fn2_z,fn3_x,fn3_y,fn3_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,
      fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,un1_x,un1_y,
      un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,ui1_x,ui1_y,
      ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,
      ui4_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,
      ji3_z,ji4_x,ji4_y,ji4_z,je_r,je_t,je_z,je_perp,je_para,
      ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Tn3,
      Ti1,Ti2,Ti3,Ti4,Te,n_mp_n1,n_mp_n2,n_mp_n3,
      n_mp_i1,n_mp_i2,n_mp_i3,n_mp_i4,avg_w_n1,avg_w_n2,
      avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
      ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,
      ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
      ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,ndot_CEX01_i3,ndot_CEX02_i4,
      F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,
      nu_ei2,nu_i01,nu_i02,nu_i12,nu_ex,err_interp_n,f_split_adv,
      f_split_qperp,f_split_qpara,f_split_qb,f_split_Pperp,
      f_split_Ppara,f_split_ecterm,f_split_inel,alpha_ano,alpha_ano_e,
      alpha_ano_q,alpha_ine,alpha_ine_q] = HET_sims_plotvars(nodes_flag,cells_flag,Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,
                                                             nn2,nn3,ni1,ni2,ni3,ni4,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,
                                                             fn2_z,fn3_x,fn3_y,fn3_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,
                                                             fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,un1_x,un1_y,
                                                             un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,ui1_x,ui1_y,
                                                             ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,
                                                             ui4_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,
                                                             ji3_z,ji4_x,ji4_y,ji4_z,je_r,je_t,je_z,je_perp,je_para,
                                                             ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Tn3,
                                                             Ti1,Ti2,Ti3,Ti4,Te,n_mp_n1,n_mp_n2,n_mp_n3,
                                                             n_mp_i1,n_mp_i2,n_mp_i3,n_mp_i4,avg_w_n1,avg_w_n2,
                                                             avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
                                                             ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,
                                                             ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
                                                             ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,ndot_CEX01_i3,
                                                             ndot_CEX02_i4,F_theta,Hall_par,Hall_par_eff,nu_e_tot,
                                                             nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,nu_ex,
                                                             err_interp_n,f_split_adv,f_split_qperp,f_split_qpara,
                                                             f_split_qb,f_split_Pperp,f_split_Ppara,f_split_ecterm,
                                                             f_split_inel,alpha_ano,alpha_ano_e,alpha_ano_q,alpha_ine,
                                                             alpha_ine_q)
        if mean_vars == 1:        
            print("Averaging variables...")                                                                              
            [phi_mean,Er_mean,Ez_mean,Efield_mean,nn1_mean,nn2_mean,nn3_mean,
           ni1_mean,ni2_mean,ni3_mean,ni4_mean,ne_mean,fn1_x_mean,fn1_y_mean,
           fn1_z_mean,fn2_x_mean,fn2_y_mean,fn2_z_mean,fn3_x_mean,fn3_y_mean,
           fn3_z_mean,fi1_x_mean,fi1_y_mean,fi1_z_mean,fi2_x_mean,fi2_y_mean,
           fi2_z_mean,fi3_x_mean,fi3_y_mean,fi3_z_mean,fi4_x_mean,fi4_y_mean,
           fi4_z_mean,un1_x_mean,un1_y_mean,un1_z_mean,un2_x_mean,un2_y_mean,
           un2_z_mean,un3_x_mean,un3_y_mean,un3_z_mean,ui1_x_mean,ui1_y_mean,
           ui1_z_mean,ui2_x_mean,ui2_y_mean,ui2_z_mean,ui3_x_mean,ui3_y_mean,
           ui3_z_mean,ui4_x_mean,ui4_y_mean,ui4_z_mean,ji1_x_mean,ji1_y_mean,
           ji1_z_mean,ji2_x_mean,ji2_y_mean,ji2_z_mean,ji3_x_mean,ji3_y_mean,
           ji3_z_mean,ji4_x_mean,ji4_y_mean,ji4_z_mean,je_r_mean,je_t_mean,
           je_z_mean,je_perp_mean,je_para_mean,ue_r_mean,ue_t_mean,ue_z_mean,
           ue_perp_mean,ue_para_mean,uthetaExB_mean,Tn1_mean,Tn2_mean,Tn3_mean,
           Ti1_mean,Ti2_mean,Ti3_mean,Ti4_mean,Te_mean,n_mp_n1_mean,n_mp_n2_mean,
           n_mp_n3_mean,n_mp_i1_mean,n_mp_i2_mean,n_mp_i3_mean,n_mp_i4_mean,
           avg_w_n1_mean,avg_w_n2_mean,avg_w_i1_mean,avg_w_i2_mean,
           neu_gen_weights1_mean,neu_gen_weights2_mean,ion_gen_weights1_mean,
           ion_gen_weights2_mean,ndot_ion01_n1_mean,ndot_ion02_n1_mean,
           ndot_ion12_i1_mean,ndot_ion01_n2_mean,ndot_ion02_n2_mean,
           ndot_ion01_n3_mean,ndot_ion02_n3_mean,ndot_ion12_i3_mean,
           ndot_CEX01_i3_mean,ndot_CEX02_i4_mean,
           ne_cath_mean,Te_cath_mean,nu_cath_mean,ndot_cath_mean,
           F_theta_mean,Hall_par_mean,Hall_par_eff_mean,nu_e_tot_mean,
           nu_e_tot_eff_mean,nu_en_mean,nu_ei1_mean,nu_ei2_mean,nu_i01_mean,
           nu_i02_mean,nu_i12_mean,nu_ex_mean,
           Boltz_mean,Boltz_dim_mean,phi_elems_mean,phi_faces_mean,ne_elems_mean,
           ne_faces_mean,Te_elems_mean,Te_faces_mean,err_interp_n_mean,f_split_adv_mean,
           f_split_qperp_mean,f_split_qpara_mean,f_split_qb_mean,f_split_Pperp_mean,
           f_split_Ppara_mean,f_split_ecterm_mean,f_split_inel_mean,
           je_perp_elems_mean,je_theta_elems_mean,je_para_elems_mean,
           je_z_elems_mean,je_r_elems_mean,je_perp_faces_mean,je_theta_faces_mean,
           je_para_faces_mean,je_z_faces_mean,je_r_faces_mean,
           F_theta_elems_mean,Hall_par_elems_mean,Hall_par_eff_elems_mean,
           nu_e_tot_elems_mean,nu_e_tot_eff_elems_mean,F_theta_faces_mean,
           Hall_par_faces_mean,Hall_par_eff_faces_mean,nu_e_tot_faces_mean,
           nu_e_tot_eff_faces_mean,nu_en_elems_mean,nu_ei1_elems_mean,
           nu_ei2_elems_mean,nu_i01_elems_mean,nu_i02_elems_mean,
           nu_i12_elems_mean,nu_ex_elems_mean,nu_en_faces_mean,
           nu_ei1_faces_mean,nu_ei2_faces_mean,nu_i01_faces_mean,
           nu_i02_faces_mean,nu_i12_faces_mean,nu_ex_faces_mean] = HET_sims_mean(nsteps,mean_type,last_steps,step_i,step_f,Z_ion_spe,
                                                                               num_ion_spe,num_neu_spe,phi,Er,Ez,
                                                                               Efield,Br,Bz,Bfield,nn1,nn2,nn3,ni1,ni2,ni3,ni4,ne,fn1_x,
                                                                               fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fn3_x,fn3_y,fn3_z,fi1_x,fi1_y,
                                                                               fi1_z,fi2_x,fi2_y,fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,
                                                                               un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,ui1_x,
                                                                               ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,
                                                                               ui4_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,ji3_z,
                                                                               ji4_x,ji4_y,ji4_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,
                                                                               ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Tn3,Ti1,Ti2,Ti3,Ti4,Te,
                                                                               n_mp_n1,n_mp_n2,n_mp_n3,n_mp_i1,n_mp_i2,n_mp_i3,n_mp_i4,
                                                                               avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                                                               neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                                                               ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
                                                                               ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,ndot_CEX01_i3,ndot_CEX02_i4,
                                                                               ne_cath,Te_cath,nu_cath,ndot_cath,
                                                                               F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,
                                                                               nu_ei2,nu_i01,nu_i02,nu_i12,nu_ex,
                                                                               Boltz,Boltz_dim,phi_elems,phi_faces,ne_elems,ne_faces,
                                                                               Te_elems,Te_faces,err_interp_n,
                                                                               f_split_adv,f_split_qperp,f_split_qpara,f_split_qb,f_split_Pperp,
                                                                               f_split_Ppara,f_split_ecterm,f_split_inel,
                                                                               je_perp_elems,je_theta_elems,je_para_elems,je_z_elems,je_r_elems,
                                                                               je_perp_faces,je_theta_faces,je_para_faces,je_z_faces,je_r_faces,
                                                                               F_theta_elems,Hall_par_elems,Hall_par_eff_elems,nu_e_tot_elems,
                                                                               nu_e_tot_eff_elems,F_theta_faces,Hall_par_faces,Hall_par_eff_faces,
                                                                               nu_e_tot_faces,nu_e_tot_eff_faces,nu_en_elems,nu_ei1_elems,nu_ei2_elems,
                                                                               nu_i01_elems,nu_i02_elems,nu_i12_elems,nu_ex_elems,nu_en_faces,
                                                                               nu_ei1_faces,nu_ei2_faces,nu_i01_faces,nu_i02_faces,nu_i12_faces,
                                                                               nu_ex_faces)
        print("Obtaining final variables for plotting...") 
        if mean_vars == 1 and plot_mean_vars == 1:
            print("Plotting variables are time-averaged")
            [Br_plot,Bz_plot,Bfield_plot,phi_plot,Er_plot,Ez_plot,Efield_plot,
           nn1_plot,nn2_plot,nn3_plot,ni1_plot,ni2_plot,ni3_plot,ni4_plot,
           ne_plot,fn1_x_plot,fn1_y_plot,fn1_z_plot,fn2_x_plot,fn2_y_plot,
           fn2_z_plot,fn3_x_plot,fn3_y_plot,fn3_z_plot,fi1_x_plot,fi1_y_plot,
           fi1_z_plot,fi2_x_plot,fi2_y_plot,fi2_z_plot,fi3_x_plot,fi3_y_plot,
           fi3_z_plot,fi4_x_plot,fi4_y_plot,fi4_z_plot,un1_x_plot,un1_y_plot,
           un1_z_plot,un2_x_plot,un2_y_plot,un2_z_plot,un3_x_plot,un3_y_plot,
           un3_z_plot,ui1_x_plot,ui1_y_plot,ui1_z_plot,ui2_x_plot,ui2_y_plot,
           ui2_z_plot,ui3_x_plot,ui3_y_plot,ui3_z_plot,ui4_x_plot,ui4_y_plot,
           ui4_z_plot,ji1_x_plot,ji1_y_plot,ji1_z_plot,ji2_x_plot,ji2_y_plot,
           ji2_z_plot,ji3_x_plot,ji3_y_plot,ji3_z_plot,ji4_x_plot,ji4_y_plot,
           ji4_z_plot,je_r_plot,je_t_plot,je_z_plot,je_perp_plot,je_para_plot,
           ue_r_plot,ue_t_plot,ue_z_plot,ue_perp_plot,ue_para_plot,uthetaExB_plot,
           Tn1_plot,Tn2_plot,Tn3_plot,Ti1_plot,Ti2_plot,Ti3_plot,Ti4_plot,
           Te_plot,n_mp_n1_plot,n_mp_n2_plot,n_mp_n3_plot,
           n_mp_i1_plot,n_mp_i2_plot,n_mp_i3_plot,n_mp_i4_plot,
           avg_w_n1_plot,avg_w_n2_plot,avg_w_i1_plot,
           avg_w_i2_plot,neu_gen_weights1_plot,neu_gen_weights2_plot,
           ion_gen_weights1_plot,ion_gen_weights2_plot,ndot_ion01_n1_plot,
           ndot_ion02_n1_plot,ndot_ion12_i1_plot,ndot_ion01_n2_plot,
           ndot_ion02_n2_plot,ndot_ion01_n3_plot,ndot_ion02_n3_plot,
           ndot_ion12_i3_plot,ndot_CEX01_i3_plot,ndot_CEX02_i4_plot,ne_cath_plot,
           nu_cath_plot,ndot_cath_plot,
           F_theta_plot,Hall_par_plot,Hall_par_eff_plot,nu_e_tot_plot,
           nu_e_tot_eff_plot,nu_en_plot,nu_ei1_plot,nu_ei2_plot,nu_i01_plot,
           nu_i02_plot,nu_i12_plot,nu_ex_plot,err_interp_n_plot,f_split_adv_plot,
           f_split_qperp_plot,f_split_qpara_plot,f_split_qb_plot,
           f_split_Pperp_plot,f_split_Ppara_plot,f_split_ecterm_plot,
           f_split_inel_plot] = HET_sims_cp_vars(Br,Bz,Bfield,phi_mean,Er_mean,Ez_mean,Efield_mean,nn1_mean,nn2_mean,
                                                 nn3_mean,ni1_mean,ni2_mean,ni3_mean,ni4_mean,ne_mean,fn1_x_mean,
                                                 fn1_y_mean,fn1_z_mean,fn2_x_mean,fn2_y_mean,fn2_z_mean,
                                                 fn3_x_mean,fn3_y_mean,fn3_z_mean,fi1_x_mean,fi1_y_mean,fi1_z_mean,
                                                 fi2_x_mean,fi2_y_mean,fi2_z_mean,fi3_x_mean,fi3_y_mean,fi3_z_mean,
                                                 fi4_x_mean,fi4_y_mean,fi4_z_mean,un1_x_mean,un1_y_mean,un1_z_mean,
                                                 un2_x_mean,un2_y_mean,un2_z_mean,un3_x_mean,un3_y_mean,un3_z_mean,
                                                 ui1_x_mean,ui1_y_mean,ui1_z_mean,ui2_x_mean,ui2_y_mean,ui2_z_mean,
                                                 ui3_x_mean,ui3_y_mean,ui3_z_mean,ui4_x_mean,ui4_y_mean,ui4_z_mean,
                                                 ji1_x_mean,ji1_y_mean,ji1_z_mean,ji2_x_mean,ji2_y_mean,ji2_z_mean,
                                                 ji3_x_mean,ji3_y_mean,ji3_z_mean,ji4_x_mean,ji4_y_mean,ji4_z_mean,
                                                 je_r_mean,je_t_mean,je_z_mean,
                                                 je_perp_mean,je_para_mean,ue_r_mean,ue_t_mean,ue_z_mean,ue_perp_mean,
                                                 ue_para_mean,uthetaExB_mean,Tn1_mean,Tn2_mean,Tn3_mean,Ti1_mean,Ti2_mean,
                                                 Ti3_mean,Ti4_mean,Te_mean,n_mp_n1_mean,n_mp_n2_mean,n_mp_n3_mean,
                                                 n_mp_i1_mean,n_mp_i2_mean,n_mp_i3_mean,n_mp_i4_mean,avg_w_n1_mean,
                                                 avg_w_n2_mean,avg_w_i1_mean,avg_w_i2_mean,neu_gen_weights1_mean,
                                                 neu_gen_weights2_mean,ion_gen_weights1_mean,ion_gen_weights2_mean,
                                                 ndot_ion01_n1_mean,ndot_ion02_n1_mean,ndot_ion12_i1_mean,
                                                 ndot_ion01_n2_mean,ndot_ion02_n2_mean,ndot_ion01_n3_mean,
                                                 ndot_ion02_n3_mean,ndot_ion12_i3_mean,ndot_CEX01_i3_mean,
                                                 ndot_CEX02_i4_mean,ne_cath_mean,
                                                 nu_cath_mean,ndot_cath_mean,F_theta_mean,Hall_par_mean,Hall_par_eff_mean,
                                                 nu_e_tot_mean,nu_e_tot_eff_mean,nu_en_mean,nu_ei1_mean,nu_ei2_mean,nu_i01_mean,
                                                 nu_i02_mean,nu_i12_mean,nu_ex_mean,err_interp_n_mean,f_split_adv_mean,f_split_qperp_mean,
                                                 f_split_qpara_mean,f_split_qb_mean,f_split_Pperp_mean,f_split_Ppara_mean,
                                                 f_split_ecterm_mean,f_split_inel_mean)
            
            
        else:
            [Br_plot,Bz_plot,Bfield_plot,phi_plot,Er_plot,Ez_plot,Efield_plot,
           nn1_plot,nn2_plot,nn3_plot,ni1_plot,ni2_plot,ni3_plot,ni4_plot,
           ne_plot,fn1_x_plot,fn1_y_plot,fn1_z_plot,fn2_x_plot,fn2_y_plot,
           fn2_z_plot,fn3_x_plot,fn3_y_plot,fn3_z_plot,fi1_x_plot,fi1_y_plot,
           fi1_z_plot,fi2_x_plot,fi2_y_plot,fi2_z_plot,fi3_x_plot,fi3_y_plot,
           fi3_z_plot,fi4_x_plot,fi4_y_plot,fi4_z_plot,un1_x_plot,un1_y_plot,
           un1_z_plot,un2_x_plot,un2_y_plot,un2_z_plot,un3_x_plot,un3_y_plot,
           un3_z_plot,ui1_x_plot,ui1_y_plot,ui1_z_plot,ui2_x_plot,ui2_y_plot,
           ui2_z_plot,ui3_x_plot,ui3_y_plot,ui3_z_plot,ui4_x_plot,ui4_y_plot,
           ui4_z_plot,ji1_x_plot,ji1_y_plot,ji1_z_plot,ji2_x_plot,ji2_y_plot,
           ji2_z_plot,ji3_x_plot,ji3_y_plot,ji3_z_plot,ji4_x_plot,ji4_y_plot,
           ji4_z_plot,je_r_plot,je_t_plot,je_z_plot,je_perp_plot,je_para_plot,
           ue_r_plot,ue_t_plot,ue_z_plot,ue_perp_plot,ue_para_plot,uthetaExB_plot,
           Tn1_plot,Tn2_plot,Tn3_plot,Ti1_plot,Ti2_plot,Ti3_plot,Ti4_plot,
           Te_plot,n_mp_n1_plot,n_mp_n2_plot,n_mp_n3_plot,
           n_mp_i1_plot,n_mp_i2_plot,n_mp_i3_plot,n_mp_i4_plot,
           avg_w_n1_plot,avg_w_n2_plot,avg_w_i1_plot,
           avg_w_i2_plot,neu_gen_weights1_plot,neu_gen_weights2_plot,
           ion_gen_weights1_plot,ion_gen_weights2_plot,ndot_ion01_n1_plot,
           ndot_ion02_n1_plot,ndot_ion12_i1_plot,ndot_ion01_n2_plot,
           ndot_ion02_n2_plot,ndot_ion01_n3_plot,ndot_ion02_n3_plot,
           ndot_ion12_i3_plot,ndot_CEX01_i3_plot,ndot_CEX02_i4_plot,ne_cath_plot,
           nu_cath_plot,ndot_cath_plot,
           F_theta_plot,Hall_par_plot,Hall_par_eff_plot,nu_e_tot_plot,
           nu_e_tot_eff_plot,nu_en_plot,nu_ei1_plot,nu_ei2_plot,nu_i01_plot,
           nu_i02_plot,nu_i12_plot,nu_ex_plot,err_interp_n_plot,f_split_adv_plot,
           f_split_qperp_plot,f_split_qpara_plot,f_split_qb_plot,
           f_split_Pperp_plot,f_split_Ppara_plot,f_split_ecterm_plot,
           f_split_inel_plot] = HET_sims_cp_vars(Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,
                                                 nn2,nn3,ni1,ni2,ni3,ni4,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,
                                                 fn2_z,fn3_x,fn3_y,fn3_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,
                                                 fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,un1_x,un1_y,
                                                 un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,ui1_x,ui1_y,
                                                 ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,
                                                 ui4_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,
                                                 ji3_z,ji4_x,ji4_y,ji4_z,je_r,je_t,je_z,je_perp,je_para,
                                                 ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Tn3,Ti1,Ti2,
                                                 Ti3,Ti4,Te,n_mp_n1,n_mp_n2,n_mp_n3,n_mp_i1,n_mp_i2,
                                                 n_mp_i3,n_mp_i4,avg_w_n1,avg_w_n2,
                                                 avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
                                                 ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,
                                                 ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
                                                 ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,ndot_CEX01_i3,
                                                 ndot_CEX02_i4,ne_cath,nu_cath,
                                                 ndot_cath,F_theta,Hall_par,Hall_par_eff,nu_e_tot,
                                                 nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,nu_ex,
                                                 err_interp_n,f_split_adv,f_split_qperp,f_split_qpara,
                                                 f_split_qb,f_split_Pperp,f_split_Ppara,f_split_ecterm,
                                                 f_split_inel)
                                                                                                          
         
        # Setting a common reference potential point
        if phi_ref == 1:
            phi_plot = phi_plot - phi_plot[iphi_ref,jphi_ref]
            
        # Interpolate variables directly computed at the MFAM from the MFAM to a finer PIC mesh for plotting
        # These variables include: phi, Te and je components
        if interp_MFAM_picM_plot == 1:
            if mean_vars == 1 and plot_mean_vars == 1:
                
                ji_x_mean               = ji1_x_mean + ji2_x_mean + ji3_x_mean + ji4_x_mean
                ji_y_mean               = ji1_y_mean + ji2_y_mean + ji3_y_mean + ji4_y_mean
                ji_z_mean               = ji1_z_mean + ji2_z_mean + ji3_z_mean + ji4_z_mean
                [zs_mp,rs_mp,dims_mp,nodes_flag_mp,cells_vol_mp,xi_bottom_mp,
                 xi_top_mp,eta_min_mp,eta_max_mp,phi_mp,Te_mp,je_perp_mp,je_theta_mp,
                 je_para_mp,je_z_mp,je_r_mp,je_2D_mp,ji_x_mp,ji_y_mp,ji_z_mp,
                 ji_2D_mp,j_r_mp,j_t_mp,j_z_mp,j_2D_mp,ne_mp,Bfield_mp,Br_mp,Bz_mp,
                 alpha_ano_mp,alpha_ano_e_mp,alpha_ano_q_mp,alpha_ine_mp,
                 alpha_ine_q_mp] = HET_sims_interp_MFAM_picM_plot(path_picM_plot,n_elems,n_faces,elem_geom,
                                                                  face_geom,versors_e,versors_f,phi_elems_mean,phi_faces_mean,Te_elems_mean,
                                                                  Te_faces_mean,je_perp_elems_mean,je_theta_elems_mean,
                                                                  je_para_elems_mean,je_z_elems_mean,je_r_elems_mean,
                                                                  je_perp_faces_mean,je_theta_faces_mean,je_para_faces_mean,
                                                                  je_z_faces_mean,je_r_faces_mean,zs,rs,ji_x_mean,ji_y_mean,
                                                                  ji_z_mean,ne_mean,
                                                                  alpha_ano_elems,alpha_ano_e_elems,alpha_ano_q_elems,alpha_ine_elems,
                                                                  alpha_ine_q_elems,alpha_ano_faces,alpha_ano_e_faces,alpha_ano_q_faces,
                                                                  alpha_ine_faces,alpha_ine_q_faces)
                
            else:
                ji_x                    = ji1_x + ji2_x + ji3_x + ji4_x
                ji_y                    = ji1_y + ji2_y + ji3_y + ji4_y
                ji_z                    = ji1_z + ji2_z + ji3_z + ji4_z
                [zs_mp,rs_mp,dims_mp,nodes_flag_mp,cells_vol_mp,xi_bottom_mp,
                 xi_top_mp,eta_min_mp,eta_max_mp,phi_mp,Te_mp,je_perp_mp,je_theta_mp,
                 je_para_mp,je_z_mp,je_r_mp,je_2D_mp,ji_x_mp,ji_y_mp,ji_z_mp,
                 ji_2D_mp,j_r_mp,j_t_mp,j_z_mp,j_2D_mp,ne_mp,Bfield_mp,Br_mp,Bz_mp,
                 alpha_ano_mp,alpha_ano_e_mp,alpha_ano_q_mp,alpha_ine_mp,
                 alpha_ine_q_mp] = HET_sims_interp_MFAM_picM_plot(path_picM_plot,n_elems,n_faces,elem_geom,
                                                                  face_geom,versors_e,versors_f,phi_elems,phi_faces,Te_elems,
                                                                  Te_faces,je_perp_elems,je_theta_elems,
                                                                  je_para_elems,je_z_elems,je_r_elems,
                                                                  je_perp_faces,je_theta_faces,je_para_faces,
                                                                  je_z_faces,je_r_faces,zs,rs,ji_x,ji_y,ji_z,ne,
                                                                  alpha_ano_elems,alpha_ano_e_elems,alpha_ano_q_elems,alpha_ine_elems,
                                                                  alpha_ine_q_elems,alpha_ano_faces,alpha_ano_e_faces,alpha_ano_q_faces,
                                                                  alpha_ine_faces,alpha_ine_q_faces)

        # Obtain auxiliar average variables
        if interp_MFAM_picM_plot == 1:
            ue_perp_mp         = -je_perp_mp/(e*ne_mp)
            ue_theta_mp        = -je_theta_mp/(e*ne_mp)
            ue_para_mp         = -je_para_mp/(e*ne_mp)
            ue_z_mp            = -je_z_mp/(e*ne_mp)
            ue_r_mp            = -je_r_mp/(e*ne_mp)
            ue_mp              = np.sqrt(ue_r_mp**2 +ue_theta_mp**2 + ue_z_mp**2)
            Ekin_e_mp          = 0.5*me*ue_mp**2/e
            ratio_Ekin_Te_mp   = Ekin_e_mp/Te_mp
            ratio_je_t_perp_mp = je_theta_mp/je_perp_mp
            je_mp              = np.sqrt(je_r_mp**2 + je_theta_mp**2 + je_z_mp**2)
            j_mp               = np.sqrt(j_r_mp**2 + j_t_mp**2 + j_z_mp**2)
            
            
            
        # Obtain angular profiles (Faraday probe scan) if required
        if plot_B_scan == 1 or plot_fields_scan == 1 or plot_dens_scan == 1 or plot_temp_scan  == 1 or plot_curr_scan == 1 or plot_freq_scan == 1:
            if mean_vars == 1 and plot_mean_vars == 1:
                nn_mean = nn1_mean + nn2_mean + nn3_mean
                [ang_scan,r_scan,z_scan,
                 B_scan,Br_scan,Bz_scan,phi_scan,Te_scan,je_perp_scan,je_theta_scan,
                 je_para_scan,je_z_scan,je_r_scan,je_2D_scan,je_scan,
                 ji_x_scan,ji_y_scan,ji_z_scan,ji_2D_scan,ji_scan,ne_scan,nn_scan,
                 Hall_par_scan,Hall_par_eff_scan,
                 j_r_scan,j_t_scan,j_z_scan,j_2D_scan,j_scan] = HET_sims_interp_scan(z_offset,r_offset,rscan,ang_min,ang_max,Npoints_ang,
                                                                                      n_elems,n_faces,elem_geom,face_geom,versors_e,versors_f,
                                                                                      phi_elems_mean,phi_faces_mean,Te_elems_mean,Te_faces_mean,
                                                                                      je_perp_elems_mean,je_theta_elems_mean,je_para_elems_mean,
                                                                                      je_z_elems_mean,je_r_elems_mean,
                                                                                      je_perp_faces_mean,je_theta_faces_mean,je_para_faces_mean,
                                                                                      je_z_faces_mean,je_r_faces_mean,zs,rs,ji_x_mean,ji_y_mean,
                                                                                      ji_z_mean,ne_mean,nn_mean,Hall_par_mean,Hall_par_eff_mean)
            else:
                nn = nn1 + nn2 + nn3
                [ang_scan,r_scan,z_scan,
                 B_scan,Br_scan,Bz_scan,phi_scan,Te_scan,je_perp_scan,je_theta_scan,
                 je_para_scan,je_z_scan,je_r_scan,je_2D_scan,je_scan,
                 ji_x_scan,ji_y_scan,ji_z_scan,ji_2D_scan,ji_scan,ne_scan,nn_scan,
                 Hall_par_scan,Hall_par_eff_scan,
                 j_r_scan,j_t_scan,j_z_scan,j_2D_scan,j_scan] = HET_sims_interp_scan(z_offset,r_offset,rscan,ang_min,ang_max,Npoints_ang,
                                                                                      n_elems,n_faces,elem_geom,face_geom,versors_e,versors_f,
                                                                                      phi_elems,phi_faces,Te_elems,Te_faces,je_perp_elems,
                                                                                      je_theta_elems,je_para_elems,je_z_elems,je_r_elems,
                                                                                      je_perp_faces,je_theta_faces,je_para_faces,je_z_faces,
                                                                                      je_r_faces,zs,rs,ji_x,ji_y,ji_z,ne,nn,Hall_par,
                                                                                      Hall_par_eff)
            
            Hall_par_effect_scan = np.sqrt(Hall_par_scan*Hall_par_eff_scan)
            ue_scan              = je_scan/(e*ne_scan)
            Ekin_e_scan          = 0.5*me*ue_scan**2/e
            ratio_Ekin_Te_scan   = Ekin_e_scan/Te_scan
            
    
        
 
        # Obtain the isothermal Boltzmann relation along the cathode magnetic line
        if len(elems_cath_Bline) > 0:
            if k == 3:
                elems_cath_Bline = elems_cath_Bline_2
                
    #        [cath_Bline_phi, cath_phi, cath_Bline_Te, cath_Te, cath_Bline_ne,
    #         cath_ne, cath_Bline_nodim_Boltz, cath_nodim_Boltz,
    #         cath_Bline_dim_Boltz, cath_dim_Boltz] = comp_Boltz(elems_cath_Bline,cath_elem,V_cath,V_cath_tot,zs,rs,elem_geom,phi_plot,Te_plot,ne_plot)
             
            [cath_Bline_phi, cath_phi, cath_Bline_Te, cath_Te, cath_Bline_ne,
             cath_ne, cath_Bline_nodim_Boltz, cath_nodim_Boltz,
             cath_Bline_dim_Boltz, cath_dim_Boltz] = comp_Boltz(elems_cath_Bline,cath_elem,V_cath,V_cath_tot,zs,rs,elem_geom,face_geom,phi_plot,Te_plot,ne_plot,cath_type)
            
            # Obtain the isothermal Boltzmann relation along the given magnetic line
            [Bline_phi, ref_phi, Bline_Te, ref_Te, Bline_ne,
             ref_ne, Bline_nodim_Boltz, ref_nodim_Boltz,
             Bline_dim_Boltz, ref_dim_Boltz] = comp_Boltz(elems_Bline,np.array([ref_elem],dtype=int),V_cath,V_cath_tot,zs,rs,elem_geom,face_geom,phi_plot,Te_plot,ne_plot,cath_type)
         
        # Obtain zindex for radial profiles
        zind = np.where(zs[rind,:]<z_rprof*1E-2)[0][-1]
        if interp_MFAM_picM_plot == 1:
            zind_mp = np.where(zs_mp[rind,:]<z_rprof*1E-2)[0][-1]
            rind_mp = np.where(rs_mp[:,0] == rs[rind,0])[0][0]

        ue_plot                 = np.sqrt(ue_r_plot**2 +ue_t_plot**2 + ue_z_plot**2)
        ue2_plot                = np.sqrt(ue_perp_plot**2 +ue_t_plot**2 + ue_para_plot**2)
        ui1_plot                = np.sqrt(ui1_x_plot**2 + ui1_y_plot**2 + ui1_z_plot**2)
        ui2_plot                = np.sqrt(ui2_x_plot**2 + ui2_y_plot**2 + ui2_z_plot**2)
        cs01_plot               = np.sqrt(e*Te_plot/mass)
        cs02_plot               = np.sqrt(2*e*Te_plot/mass)
        Mi1_plot                = np.divide(ui1_plot,cs01_plot)
        Mi2_plot                = np.divide(ui2_plot,cs02_plot) 
        Mi1_z_plot              = np.divide(ui1_z_plot,cs01_plot)
        Mi2_z_plot              = np.divide(ui2_z_plot,cs02_plot)
        Ekin_e_plot             = 0.5*me*ue_plot**2/e
        Ekin_i1_plot            = 0.5*mass*ui1_plot**2/e
        Ekin_i2_plot            = 0.5*mass*ui2_plot**2/e
        ratio_Ekin_Te_plot      = Ekin_e_plot/Te_plot
        ratio_Ekin_Ti1_plot     = Ekin_i1_plot/Ti1_plot
        ratio_Ekin_Ti2_plot     = Ekin_i2_plot/Ti2_plot
        je_plot                 = np.sqrt(je_r_plot**2 + je_t_plot**2 + je_z_plot**2)
        je2_plot                = np.sqrt(je_perp_plot**2 + je_t_plot**2 + je_para_plot**2)
        ji_x_plot               = ji1_x_plot + ji2_x_plot + ji3_x_plot + ji4_x_plot
        ji_y_plot               = ji1_y_plot + ji2_y_plot + ji3_y_plot + ji4_y_plot
        ji_z_plot               = ji1_z_plot + ji2_z_plot + ji3_z_plot + ji4_z_plot
        ui_x_plot               = ji_x_plot/(e*ne_plot)
        ui_y_plot               = ji_y_plot/(e*ne_plot)
        ui_z_plot               = ji_z_plot/(e*ne_plot)
        ui_plot                 = np.sqrt(ui_x_plot**2 +ui_y_plot**2 + ui_z_plot**2)
        Z_avg                   = 1.0*ni1_plot/ne_plot + 2.0*ni2_plot/ne_plot + 1.0*ni3_plot/ne_plot + 2.0*ni4_plot/ne_plot
        cs_plot                 = np.sqrt(Z_avg*e*Te_plot/mass)
        Mi_plot                 = ui_plot/cs_plot
        Mi_z_plot               = ui_z_plot/cs_plot 
        ji_plot                 = np.sqrt( ji_x_plot**2 + ji_y_plot**2 +ji_z_plot**2 )
        ji1_plot                = np.sqrt( ji1_x_plot**2 + ji1_y_plot**2 + ji1_z_plot**2 )
        ji2_plot                = np.sqrt( ji2_x_plot**2 + ji2_y_plot**2 + ji2_z_plot**2 )
        j_r_plot                = ji_x_plot + je_r_plot
        j_t_plot                = ji_y_plot + je_t_plot
        j_z_plot                = ji_z_plot + je_z_plot
        j_plot                  = np.sqrt(j_r_plot**2 + j_t_plot**2 + j_z_plot**2)
        erel_je_plot            = np.abs(je2_plot-je_plot)/np.abs(je_plot)
        erel_ue_plot            = np.abs(ue2_plot-ue_plot)/np.abs(ue_plot)
        erel_jeji_plot          = np.abs(je_plot-ji_plot)/np.abs(ji_plot)
        ratio_ue_t_perp_plot    = ue_t_plot/ue_perp_plot
        ratio_ue_t_para_plot    = ue_t_plot/ue_para_plot
        ratio_ue_perp_para_plot = ue_perp_plot/ue_para_plot
        ratio_je_t_perp_plot    = je_t_plot/je_perp_plot
        je2D_plot               = np.sqrt(je_r_plot**2 + je_z_plot**2)
        ji2D_plot               = np.sqrt(ji_x_plot**2 + ji_z_plot**2)
        j2D_plot                = np.sqrt(j_r_plot**2 + j_z_plot**2)
        nu_ei_el_tot_plot       = nu_ei1_plot + nu_ei2_plot
        nu_ion_tot_plot         = nu_i01_plot + nu_i02_plot + nu_i12_plot
        nu_t_plot               = nu_e_tot_eff_plot - nu_e_tot_plot
        lambdaD_plot            = np.sqrt(eps0*(e*Te_plot)/(ne_plot*e**2))
        nn_plot                 = nn1_plot + nn2_plot + nn3_plot
        pn1_plot                = nn1_plot*Tn1_plot*e*1E-2 # Neutral pressure in mbar (1Pa = 1E-2 mbar)
        pn2_plot                = nn2_plot*Tn2_plot*e*1E-2 # Neutral pressure in mbar (1Pa = 1E-2 mbar)
        pn3_plot                = nn3_plot*Tn3_plot*e*1E-2 # Neutral pressure in mbar (1Pa = 1E-2 mbar)
        pn_plot                 = pn1_plot + pn2_plot + pn3_plot
        ndot_e_plot             = ndot_ion01_n1_plot[0:-1,0:-1] + ndot_ion01_n2_plot[0:-1,0:-1] + ndot_ion01_n3_plot[0:-1,0:-1] + 2.0*ndot_ion02_n1_plot[0:-1,0:-1] + 2.0*ndot_ion02_n2_plot[0:-1,0:-1] + 2.0*ndot_ion02_n3_plot[0:-1,0:-1] + ndot_ion12_i1_plot[0:-1,0:-1] + ndot_ion12_i3_plot[0:-1,0:-1]
        ratio_ni1_ni2_plot      = np.divide(ni2_plot,ni1_plot)
        ratio_ni1_ni3_plot      = np.divide(ni3_plot,ni1_plot)
        ratio_ni1_ni4_plot      = np.divide(ni4_plot,ni1_plot)
        ratio_ne_neCEX_plot     = np.divide(ni3_plot + 2*ni4_plot,ne_plot)
        ratio_nn1_nn2_plot      = np.divide(nn2_plot,nn1_plot)
        ratio_nn1_nn3_plot      = np.divide(nn3_plot,nn1_plot)
        ratio_nn1_nnCEX_plot    = np.divide(nn2_plot+nn3_plot,nn1_plot)
        ratio_nn_nnCEX_plot     = np.divide(nn2_plot+nn3_plot,nn_plot)
        ratio_nu_en_nu_e_tot_eff_plot      = np.divide(nu_en_plot,nu_e_tot_eff_plot)         # en elastic / eff tot
        ratio_nu_ion_tot_nu_e_tot_eff_plot = np.divide(nu_ion_tot_plot,nu_e_tot_eff_plot)    # ion tot / eff tot
        ratio_nu_ei_nu_e_tot_eff_plot      = np.divide(nu_ei_el_tot_plot,nu_e_tot_eff_plot)  # ei el (coulomb) / eff tot
        ratio_nu_ex_nu_e_tot_eff_plot      = np.divide(nu_ex_plot,nu_e_tot_eff_plot)         # excitation / eff tot
        ratio_nu_e_tot_nu_e_tot_eff_plot   = np.divide(nu_e_tot_plot,nu_e_tot_eff_plot)      # e tot / eff tot
        ratio_nu_t_nu_e_tot_eff_plot       = np.divide(nu_t_plot,nu_e_tot_eff_plot)          # t / eff tot
        ratio_nu_e_tot_nu_t_plot           = np.divide(nu_e_tot_plot,nu_t_plot)              # e tot / t
        ratio_nu_ei_tot_nu_t_plot          = np.divide(nu_ei_el_tot_plot,nu_t_plot)          # ei el (coulomb) / t 
        
        
        # Compute the total mass flow of all heavy species at each z section within the chamber
        xi_bottom = int(xi_bottom)
        xi_top    = int(xi_top)
        eta_min   = int(eta_min)
        eta_max   = int(eta_max)
        mA_Ch = np.zeros(int(xi_bottom)+1,dtype=float)
        mA_Ch2 = np.zeros(int(xi_bottom)+1,dtype=float)
        mA_Ch3 = np.zeros(int(xi_bottom)+1,dtype=float)
        I_Ch   = np.zeros(int(xi_bottom)+1,dtype=float)
        flux = fn1_z_plot + fn2_z_plot + fn3_z_plot + fi1_z_plot + fi2_z_plot + fi3_z_plot + fi4_z_plot
        for j in range(0,int(xi_bottom)+1):
#        for j in range(19,20):
            if j == 24:
#            for i in range(int(eta_min),int(eta_max)):
                for i in range(0,dims[0]-1):
                    dr = rs[i+1,j] - rs[i,j]
                    dz = rs[i+1,j]**2 - rs[i,j]**2
    #                print(dr,rs[i+1,j],rs[i,j])
                    mA_Ch[j] = mA_Ch[j] + 0.5*(flux[i,j]*rs[i,j] + flux[i+1,j]*rs[i+1,j])*dr
                    mA_Ch3[j] = mA_Ch3[j] + 0.5*(flux[i,j] + flux[i+1,j])*dz
                    I_Ch[j] = I_Ch[j] + 0.5*(j_z_plot[i,j]*rs[i,j] + j_z_plot[i+1,j]*rs[i+1,j])*dr
                    vec = np.multiply(rs[int(eta_min):int(eta_max)+1,j],flux[int(eta_min):int(eta_max)+1,j])
                    vec_r = rs[int(eta_min):int(eta_max)+1,j]
                    mA_Ch2[j] = 2.0*np.pi*mass*np.trapz(vec,x=vec_r)
            else:
                for i in range(int(eta_min),int(eta_max)):
#                for i in range(0,dims[0]-1):
                    dr = rs[i+1,j] - rs[i,j]
                    dz = rs[i+1,j]**2 - rs[i,j]**2
    #                print(dr,rs[i+1,j],rs[i,j])
                    mA_Ch[j] = mA_Ch[j] + 0.5*(flux[i,j]*rs[i,j] + flux[i+1,j]*rs[i+1,j])*dr
                    mA_Ch3[j] = mA_Ch3[j] + 0.5*(flux[i,j] + flux[i+1,j])*dz
                    I_Ch[j] = I_Ch[j] + 0.5*(j_z_plot[i,j]*rs[i,j] + j_z_plot[i+1,j]*rs[i+1,j])*dr
                    vec = np.multiply(rs[int(eta_min):int(eta_max)+1,j],flux[int(eta_min):int(eta_max)+1,j])
                    vec_r = rs[int(eta_min):int(eta_max)+1,j]
                    mA_Ch2[j] = 2.0*np.pi*mass*np.trapz(vec,x=vec_r)
        mA_Ch = mA_Ch*2.0*np.pi*mass
        mA_Ch3 = mA_Ch3*np.pi*mass
        I_Ch = 2*np.pi*I_Ch
#        print(mA_Ch) 
#        print(mA_Ch2[19])
#        print(mA_Ch3[19])
##        print(mA_Ch4)
#        print(I_Ch)
        
#        plt.figure("mdot")
#        plt.plot(zs[rind,0:int(xi_bottom)],mA_Ch[0:int(xi_bottom)],'b')
#        plt.plot(zs[rind,0:int(xi_bottom)],mA_Ch2[0:int(xi_bottom)],'r')
#        plt.plot(zs[rind,0:int(xi_bottom)],mA_Ch3[0:int(xi_bottom)],'g')
#        plt.figure("I_Ch")
#        plt.plot(zs[rind,0:int(xi_bottom)],I_Ch[0:int(xi_bottom)],'g')
       
        
        # Obtain effective Hall parameter and background pressure in plume volume
        Hall_par_effect_plot = np.sqrt(Hall_par_plot*Hall_par_eff_plot)
        hall_effect_mean     = 0.0
        hall_eff_mean        = 0.0
        hall_mean            = 0.0
        pn_mean              = 0.0
        cells_vol_tot = 0.0
        for icell in range(0,dims[0]-1):
            for jcell in range(int(xi_bottom),dims[1]-1):
                hall_effect_cell = 0.25*(Hall_par_effect_plot[icell,jcell] + Hall_par_effect_plot[icell,jcell+1] + Hall_par_effect_plot[icell+1,jcell+1] +Hall_par_effect_plot[icell+1,jcell])
                hall_eff_cell    = 0.25*(Hall_par_eff_plot[icell,jcell] + Hall_par_eff_plot[icell,jcell+1] + Hall_par_eff_plot[icell+1,jcell+1] +Hall_par_eff_plot[icell+1,jcell])
                hall_cell        = 0.25*(Hall_par_plot[icell,jcell] + Hall_par_plot[icell,jcell+1] + Hall_par_plot[icell+1,jcell+1] +Hall_par_plot[icell+1,jcell])
                pn_cell          = 0.25*(pn_plot[icell,jcell] + pn_plot[icell,jcell+1] + pn_plot[icell+1,jcell+1] +pn_plot[icell+1,jcell])
                hall_effect_mean = hall_effect_mean + hall_effect_cell*cells_vol[icell,jcell]
                hall_eff_mean    = hall_eff_mean + hall_eff_cell*cells_vol[icell,jcell]
                hall_mean        = hall_mean + hall_cell*cells_vol[icell,jcell]
                pn_mean          = pn_mean + pn_cell*cells_vol[icell,jcell]
                cells_vol_tot    = cells_vol_tot + cells_vol[icell,jcell]
        hall_effect_mean = hall_effect_mean/cells_vol_tot
        hall_eff_mean    = hall_eff_mean/cells_vol_tot
        hall_mean        = hall_mean/cells_vol_tot
        pn_mean          = pn_mean/cells_vol_tot
    
        
        # if interp_MFAM_picM_plot == 1:
        #     # Copy j matrix and set NaNs where we are not looking for j=0 point
        #     # For P3 and P4 (paper GDML)
        #     pos_tol_z1 = 1000
        #     pos_tol_z2 = 1000
        #     pos_tol_r1 = 600
        #     # pos_tol_r2 = 2000
        #     pos_tol_r2 = 1000
        #     # For P2 (paper GDML)
        #     # pos_tol_z1 = 1000
        #     # pos_tol_z2 = 200
        #     # pos_tol_r1 = 600
        #     # pos_tol_r2 = 1000
        #     copy_j_2D_mp = np.copy(j_2D_mp)
        #     copy_j_2D_mp[0:pos_tol_r1,:]                   = np.nan
        #     copy_j_2D_mp[dims_mp[0]-pos_tol_r2::,:]        = np.nan
        #     copy_j_2D_mp[:,0:int(xi_bottom_mp)+pos_tol_z1] = np.nan
        #     copy_j_2D_mp[:,dims_mp[1]-pos_tol_z2::]        = np.nan
        #     # pos_null_j2D_point = np.where(j_2D_mp == np.nanmin(np.nanmin(j_2D_mp[pos_tol::,pos_tol::])))
        #     pos_null_j2D_point = np.where(j_2D_mp == np.nanmin(np.nanmin(copy_j_2D_mp)))
        #     z_null_j2D_point = zs_mp[pos_null_j2D_point][0]
        #     r_null_j2D_point = rs_mp[pos_null_j2D_point][0]
        #     j2D_null_point   = j_2D_mp[pos_null_j2D_point][0]
        # else:
        #     pos_tol = 5
        #     pos_null_j2D_point = np.where(j2D_plot == np.nanmin(np.nanmin(j2D_plot[pos_tol:dims[0]-pos_tol,int(xi_bottom)+pos_tol:-pos_tol:1])))
        #     z_null_j2D_point = zs[pos_null_j2D_point][0]
        #     r_null_j2D_point = rs[pos_null_j2D_point][0]
        #     j2D_null_point   = j2D_plot[pos_null_j2D_point][0]
        
    
        
        # Obtain axial profiles at the midline if lateral plume boundary is 
        # tilted, so that the PIC mesh points along thruster midradius in the plume
        # increase r position along the plume
        if mean_vars == 1 and plot_mean_vars == 1:
            [z_prof,r_prof,z_prof_mp,r_prof_mp,
             B_prof,Br_prof,Bz_prof,phi_prof,Te_prof,je_perp_prof,je_theta_prof,
             je_para_prof,je_z_prof,je_r_prof,je_2D_prof,je_prof,
             j_r_prof,j_t_prof,j_z_prof,j_2D_prof,j_prof,
             B_prof_mp,Br_prof_mp,Bz_prof_mp,phi_prof_mp,Te_prof_mp,
             je_perp_prof_mp,je_theta_prof_mp,je_para_prof_mp,je_z_prof_mp,
             je_r_prof_mp,je_2D_prof_mp,je_prof_mp,
             j_r_prof_mp,j_t_prof_mp,j_z_prof_mp,j_2D_prof_mp,j_prof_mp,
        
             ji_x_prof,ji_y_prof,ji_z_prof,ji_2D_prof,ji_prof,ne_prof,nn_prof,
             Hall_par_prof,Hall_par_eff_prof,Hall_par_effect_prof] = HET_sims_interp_zprof(interp_MFAM_picM_plot,rs[rind,0],
                                                                                           n_elems,n_faces,elem_geom,face_geom,versors_e,versors_f,
                                                                                           phi_elems_mean,phi_faces_mean,Te_elems_mean,Te_faces_mean,
                                                                                           je_perp_elems_mean,je_theta_elems_mean,je_para_elems_mean,
                                                                                           je_z_elems_mean,je_r_elems_mean,je_perp_faces_mean,
                                                                                           je_theta_faces_mean,je_para_faces_mean,je_z_faces_mean,
                                                                                           je_r_faces_mean,
                                                                                           zs,rs,zs_mp,rs_mp,
                                                                                           ji_x_plot,ji_y_plot,ji_z_plot,ne_plot,nn_plot,Hall_par_plot,
                                                                                           Hall_par_eff_plot)
        else:
            [z_prof,r_prof,z_prof_mp,r_prof_mp,
             B_prof,Br_prof,Bz_prof,phi_prof,Te_prof,je_perp_prof,je_theta_prof,
             je_para_prof,je_z_prof,je_r_prof,je_2D_prof,je_prof,
             j_r_prof,j_t_prof,j_z_prof,j_2D_prof,j_prof,
             B_prof_mp,Br_prof_mp,Bz_prof_mp,phi_prof_mp,Te_prof_mp,
             je_perp_prof_mp,je_theta_prof_mp,je_para_prof_mp,je_z_prof_mp,
             je_r_prof_mp,je_2D_prof_mp,je_prof_mp,
             j_r_prof_mp,j_t_prof_mp,j_z_prof_mp,j_2D_prof_mp,j_prof_mp,
       
             ji_x_prof,ji_y_prof,ji_z_prof,ji_2D_prof,ji_prof,ne_prof,nn_prof,
             Hall_par_prof,Hall_par_eff_prof,Hall_par_effect_prof] = HET_sims_interp_zprof(interp_MFAM_picM_plot,rs[rind,0],
                                                                                           n_elems,n_faces,elem_geom,face_geom,versors_e,versors_f,
                                                                                           phi_elems,phi_faces,Te_elems,Te_faces,
                                                                                           je_perp_elems,je_theta_elems,je_para_elems,
                                                                                           je_z_elems,je_r_elems,je_perp_faces,
                                                                                           je_theta_faces,je_para_faces,je_z_faces,
                                                                                           je_r_faces,
                                                                                           zs,rs,zs_mp,rs_mp,
                                                                                           ji_x_plot,ji_y_plot,ji_z_plot,ne_plot,nn_plot,Hall_par_plot,
                                                                                           Hall_par_eff_plot)
        
        ###########################################################################
        print("Plotting...")
        ############################ GENERATING PLOTS #############################
        print("interp_MFAM_picM_plot = "+str(interp_MFAM_picM_plot))        
        print("r zprofs              = %15.8e (cm)" %( rs[rind,0]*1E2 ) )
        if interp_MFAM_picM_plot == 1:
            print("r_mp zprofs           = %15.8e (cm)" %( rs_mp[rind_mp,0]*1E2 ) )
        print("z rprofs              = %15.8e (cm)" %( zs[0,zind]*1E2 ) )
        if interp_MFAM_picM_plot == 1:
            print("z_mp rprofs           = %15.8e (cm)" %( zs_mp[0,zind_mp]*1E2 ) )
            
        print("ne_min                = %15.8e (1/m3)" %( ne_plot[rind,:].min() ) )
        print("ne_max                = %15.8e (1/m3)" %( ne_plot[rind,:].max() ) )
        
        
        if interp_MFAM_picM_plot == 1:
            zs_mp                = zs_mp*1E2
            rs_mp                = rs_mp*1E2
        zs                = zs*1E2
        rs                = rs*1E2
        zscells           = zscells*1E2
        rscells           = rscells*1E2
        z_prof            = z_prof*1E2
        r_prof            = r_prof*1E2
        z_prof_mp         = z_prof_mp*1E2
        r_prof_mp         = r_prof_mp*1E2
        points            = points*1E2
        z_cath            = z_cath*1E2
        r_cath            = r_cath*1E2
        elem_geom[3,:]    = elem_geom[3,:]*1E4
        elem_geom[0,:]    = elem_geom[0,:]*1E2
        elem_geom[1,:]    = elem_geom[1,:]*1E2
        Efield_plot       = Efield_plot*1E-3
        Er_plot           = Er_plot*1E-3
        Ez_plot           = Ez_plot*1E-3
        Bfield_plot       = Bfield_plot*1E4
        Br_plot           = Br_plot*1E4
        Bz_plot           = Bz_plot*1E4
        je_r_plot         = je_r_plot*1E-4      # This is A/cm2
        je_t_plot         = je_t_plot*1E-4      # This is A/cm2
        je_z_plot         = je_z_plot*1E-4      # This is A/cm2
        je_para_plot      = je_para_plot*1E-4   # This is A/cm2
        je_perp_plot      = je_perp_plot*1E-4   # This is A/cm2
        ji_x_plot         = ji_x_plot*1E-4      # This is A/cm2
        ji_y_plot         = ji_y_plot*1E-4      # This is A/cm2
        ji_z_plot         = ji_z_plot*1E-4      # This is A/cm2
        je2D_plot         = je2D_plot*1E-4      # This is A/cm2
        ji2D_plot         = ji2D_plot*1E-4      # This is A/cm2
        j2D_plot          = j2D_plot*1E-4       # This is A/cm2
        ji1_plot          = ji1_plot*1E-4       # This is A/cm2
        ji2_plot          = ji2_plot*1E-4       # This is A/cm2
        j_plot            = j_plot*1E-4         # This is A/cm2
        je_plot           = je_plot*1E-4        # This is A/cm2
        ji_plot           = ji_plot*1E-4        # This is A/cm2
        if interp_MFAM_picM_plot == 1: 
            je_perp_mp        = je_perp_mp*1E-4   # This is A/cm2
            je_theta_mp       = je_theta_mp*1E-4  # This is A/cm2
            je_para_mp        = je_para_mp*1E-4   # This is A/cm2
            je_z_mp           = je_z_mp*1E-4      # This is A/cm2
            je_r_mp           = je_r_mp*1E-4      # This is A/cm2
            je_2D_mp          = je_2D_mp*1E-4     # This is A/cm2
            ji_x_mp           = ji_x_mp*1E-4      # This is A/cm2
            ji_y_mp           = ji_y_mp*1E-4      # This is A/cm2
            ji_z_mp           = ji_z_mp*1E-4      # This is A/cm2
            ji_2D_mp          = ji_2D_mp*1E-4     # This is A/cm2
            j_r_mp            = j_r_mp*1E-4       # This is A/cm2
            j_t_mp            = j_t_mp*1E-4       # This is A/cm2
            j_z_mp            = j_z_mp*1E-4       # This is A/cm2
            j_2D_mp           = j_2D_mp*1E-4      # This is A/cm2
            je_mp             = je_mp*1E-4        # This is A/cm2
            j_mp              = j_mp*1E-4         # This is A/cm2
            
        if plot_B_scan == 1 or plot_fields_scan == 1 or plot_dens_scan == 1 or plot_temp_scan  == 1 or plot_curr_scan == 1 or plot_freq_scan == 1:
            z_scan = z_scan*1E2
            r_scan = r_scan*1E2
            je_perp_scan  = je_perp_scan*1E-4  # This is A/cm2
            je_theta_scan = je_theta_scan*1E-4  # This is A/cm2
            je_para_scan  = je_para_scan*1E-4   # This is A/cm2
            je_z_scan     = je_z_scan*1E-4      # This is A/cm2
            je_r_scan     = je_r_scan*1E-4      # This is A/cm2
            je_2D_scan    = je_2D_scan*1E-4     # This is A/cm2
            je_scan       = je_scan*1E-4        # This is A/cm2
            ji_x_scan     = ji_x_scan*1E-4      # This is A/cm2
            ji_y_scan     = ji_y_scan*1E-4      # This is A/cm2
            ji_z_scan     = ji_z_scan*1E-4      # This is A/cm2
            ji_2D_scan    = ji_2D_scan*1E-4     # This is A/cm2
            ji_scan       = ji_scan*1E-4        # This is A/cm2
            j_r_scan      = j_r_scan*1E-4       # This is A/cm2
            j_t_scan      = j_t_scan*1E-4       # This is A/cm2
            j_z_scan      = j_z_scan*1E-4       # This is A/cm2
            j_2D_scan     = j_2D_scan*1E-4      # This is A/cm2
            j_scan        = j_scan*1E-4         # This is A/cm2
        
            
#        nu_e_tot_plot     = nu_e_tot_plot*1E-6
#        nu_e_tot_eff_plot = nu_e_tot_eff_plot*1E-6
#        nu_en_plot        = nu_en_plot*1E-6
#        nu_ei1_plot       = nu_ei1_plot*1E-6
#        nu_ei2_plot       = nu_ei2_plot*1E-6
#        nu_i01_plot       = nu_i01_plot*1E-6
#        nu_i02_plot       = nu_i02_plot*1E-6
#        nu_i12_plot       = nu_i12_plot*1E-6
        lambdaD_plot = lambdaD_plot*1E3
        
        ne_plot = ne_plot*1E-16
        ne_prof = ne_prof*1E-16
        
        
        # Comment the following lines if we want units in axes
        if interp_MFAM_picM_plot == 1:
            zs_mp = zs_mp/L_c
            rs_mp = rs_mp/H_c
        zs = zs/L_c
        rs = rs/H_c
        z_prof = z_prof/L_c
        r_prof = r_prof/H_c
        z_prof_mp = z_prof_mp/L_c
        r_prof_mp = r_prof_mp/H_c
        points[:,0] = points[:,0]/L_c
        points[:,1] = points[:,1]/H_c
        z_cath = z_cath/L_c
        r_cath = r_cath/H_c
        zscells = zscells/L_c
        rscells = rscells/H_c
        if plot_B_scan == 1 or plot_fields_scan == 1 or plot_dens_scan == 1 or plot_temp_scan  == 1 or plot_curr_scan == 1 or plot_freq_scan == 1:
            z_scan = z_scan/L_c
            r_scan = r_scan/H_c
        
        if k <= 9:
            if interp_MFAM_picM_plot == 1:
                # axes1[0,0].plot(zs_mp[rind_mp,:],phi_mp[rind_mp,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_mesh, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                # axes1[1,0].plot(zs_mp[rind_mp,:],Te_mp[rind_mp,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_mesh, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                axes1[0,0].plot(z_prof_mp,phi_prof_mp, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_mesh_vec[k], markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                axes1[1,0].plot(z_prof_mp,Te_prof_mp, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_mesh_vec[k], markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            else:
                # axes1[0,0].plot(zs[rind,:],phi_plot[rind,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                # axes1[1,0].plot(zs[rind,:],Te_plot[rind,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                axes1[0,0].plot(z_prof,phi_prof, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_mesh, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                axes1[1,0].plot(z_prof,Te_prof, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_mesh, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
            # axes1[2,0].plot(zs[rind,:],ne_plot[rind,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # # axes1[2,0].semilogy(zs[rind,:],ne_plot[rind,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            axes1[2,0].plot(z_prof,ne_prof, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_vec[k], markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # axes1[2,0].semilogy(z_prof,ne_prof, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif k > 9:
            if interp_MFAM_picM_plot == 1:
                # axes1[0,1].plot(zs_mp[rind_mp,:],phi_mp[rind_mp,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_mesh, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                # axes1[1,1].plot(zs_mp[rind_mp,:],Te_mp[rind_mp,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_mesh, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                axes1[0,1].plot(z_prof_mp,phi_prof_mp, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_mesh_vec[k], markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                axes1[1,1].plot(z_prof_mp,Te_prof_mp, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_mesh_vec[k], markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
            else:
                # axes1[0,1].plot(zs[rind,:],phi_plot[rind,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                # axes1[1,1].plot(zs[rind,:],Te_plot[rind,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                axes1[0,1].plot(z_prof,phi_prof, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                axes1[1,1].plot(z_prof,Te_prof, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # axes1[2,1].plot(zs[rind,:],ne_plot[rind,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # axes1[2,1].semilogy(zs[rind,:],ne_plot[rind,:], linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            axes1[2,1].plot(z_prof,ne_prof, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_vec[k], markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            # axes1[2,1].semilogy(z_prof,ne_prof, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every, markersize=marker_size_vec[ind], marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
            
            
    
        ind = ind + 1
        if ind > 25:
            ind = 0
            ind2 = ind2 + 1
            if ind2 > 6:
                ind = 0
                ind2 = 0
                ind3 = ind3 + 1
    
    if plot_zcath_012 == 2:
        # axes1[0,0].axvline(x=zcat_pos, linestyle='-.',color='k', linewidth = line_width)
        # axes1[1,0].axvline(x=zcat_pos, linestyle='-.',color='k', linewidth = line_width)
        # axes1[2,0].axvline(x=zcat_pos, linestyle='-.',color='k', linewidth = line_width)
        # axes1[0,1].axvline(x=zcat_pos, linestyle='-.',color='k', linewidth = line_width)
        # axes1[1,1].axvline(x=zcat_pos, linestyle='-.',color='k', linewidth = line_width)
        # axes1[2,1].axvline(x=zcat_pos, linestyle='-.',color='k', linewidth = line_width)
        
        # axes1[0,0].axvline(x=zcat_pos2, linestyle=':',color='k', linewidth = line_width)
        # axes1[1,0].axvline(x=zcat_pos2, linestyle=':',color='k', linewidth = line_width)
        # axes1[2,0].axvline(x=zcat_pos2, linestyle=':',color='k', linewidth = line_width)
        # axes1[0,1].axvline(x=zcat_pos2, linestyle=':',color='k', linewidth = line_width)
        # axes1[1,1].axvline(x=zcat_pos2, linestyle=':',color='k', linewidth = line_width)
        # axes1[2,1].axvline(x=zcat_pos2, linestyle=':',color='k', linewidth = line_width)
        
        axes1[0,0].axvline(x=zcat_pos2, linestyle='-.',color='k', linewidth = line_width)
        axes1[1,0].axvline(x=zcat_pos2, linestyle='-.',color='k', linewidth = line_width)
        axes1[2,0].axvline(x=zcat_pos2, linestyle='-.',color='k', linewidth = line_width)
        axes1[0,1].axvline(x=zcat_pos2, linestyle='-.',color='k', linewidth = line_width)
        axes1[1,1].axvline(x=zcat_pos2, linestyle='-.',color='k', linewidth = line_width)
        axes1[2,1].axvline(x=zcat_pos2, linestyle='-.',color='k', linewidth = line_width)
        
        
    
    
    # axes1[2,0].yaxis.set_minor_locator(MultipleLocator(5))locator=ticker.LogLocator()
    # axes1[2,0].tick_params(axis='y', which='minor')
    # axes1[2,0].yaxis.set_major_formatter('{y:.0f}')
    # axes1[2,0].set_ylim(1E16,3E17)
    # axes1[2,0].set_yticks([1E16,5E16,1E17,3E17])
    # axes1[2,0].yaxis.set_minor_locator(AutoMinorLocator())
    # axes1[2,0].yaxis.set_minor_formatter(FixedFormatter([r"$10^{16}$", r"$5\cdot 10^{16}$", r"$10^{17}$", r"$3\cdot 10^{17}$"]))
        
    
    # axes1[0,0].legend(fontsize = font_size_legend,loc='best',ncol=1,frameon = True) 
    # axes1[0,1].legend(fontsize = font_size_legend,loc='best',ncol=1,frameon = True) 
    
    
    # # Set two different legends
    # lines1 = axes1[0,0].get_lines()
    # lines2 = axes1[0,1].get_lines()
    
    # legend1 = axes1[0,0].legend([lines1[i] for i in [0,1,2,3]], [lines1[i].get_label() for i in [0,1,2,3]], bbox_to_anchor=(0.60, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True) 
    # legend2 = axes1[0,0].legend([lines1[i] for i in [4,5,6]], [lines1[i].get_label() for i in [4,5,6]], bbox_to_anchor=(0.79, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True) 
    # axes1[0,0].add_artist(legend1)
    # axes1[0,0].add_artist(legend2)
    
    # legend3 = axes1[0,1].legend([lines2[i] for i in [0,1,2,3]], [lines2[i].get_label() for i in [0,1,2,3]], bbox_to_anchor=(0.57, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True) 
    # legend4 = axes1[0,1].legend([lines2[i] for i in [4,5,6]], [lines2[i].get_label() for i in [4,5,6]], bbox_to_anchor=(0.77, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True) 
    # axes1[0,1].add_artist(legend3)
    # axes1[0,1].add_artist(legend4)
        
    # Set three different legends on each plot
    lines1 = axes1[0,0].get_lines()
    lines2 = axes1[0,1].get_lines()
    
    lines1_flip = np.flip(lines1)
    lines2_flip = np.flip(lines2)
    
    # print(len(lines1_flip))
    # print(lines1_flip)
    
    # for i in range(0,11):
    #     print(i)
    #     print(lines1_flip[i].get_label())
    
    lines1_flip1 = lines1_flip[1::]
    lines2_flip2 = lines2_flip[1::]
    
    # legend1 = axes1[0,0].legend([lines1_flip1[i] for i in [0,1,2,3]], [lines1_flip1[i].get_label() for i in [0,1,2,3]], bbox_to_anchor=(0.41, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True,numpoints=2,handlelength=4.0,borderpad=0.15) 
    # legend2 = axes1[0,0].legend([lines1_flip1[i] for i in [4,5,6]], [lines1_flip1[i].get_label() for i in [4,5,6]], bbox_to_anchor=(0.60, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True,numpoints=2,handlelength=4.0,borderpad=0.15) 
    # legend3 = axes1[0,0].legend([lines1_flip1[i] for i in [7,8,9]], [lines1_flip1[i].get_label() for i in [7,8,9]], bbox_to_anchor=(0.79, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True,numpoints=2,handlelength=4.0,borderpad=0.15)
    legend1 = axes1[0,0].legend([lines1_flip1[i] for i in [0,1,2,3]], [lines1_flip1[i].get_label() for i in [0,1,2,3]], bbox_to_anchor=(0.39, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True,numpoints=2,handlelength=4.0,borderpad=0.15) 
    legend2 = axes1[0,0].legend([lines1_flip1[i] for i in [4,5,6]], [lines1_flip1[i].get_label() for i in [4,5,6]], bbox_to_anchor=(0.585, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True,numpoints=2,handlelength=4.0,borderpad=0.15) 
    legend3 = axes1[0,0].legend([lines1_flip1[i] for i in [7,8,9]], [lines1_flip1[i].get_label() for i in [7,8,9]], bbox_to_anchor=(0.78, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True,numpoints=2,handlelength=4.0,borderpad=0.15)
    axes1[0,0].add_artist(legend1)
    axes1[0,0].add_artist(legend2)
    axes1[0,0].add_artist(legend3)
    
    legend4 = axes1[0,1].legend([lines2_flip2[i] for i in [0,1,2,3]], [lines2_flip2[i].get_label() for i in [0,1,2,3]], bbox_to_anchor=(0.38, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True,numpoints=2,handlelength=4.0,borderpad=0.15) 
    legend5 = axes1[0,1].legend([lines2_flip2[i] for i in [4,5,6]], [lines2_flip2[i].get_label() for i in [4,5,6]], bbox_to_anchor=(0.58, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True,numpoints=2,handlelength=4.0,borderpad=0.15)  
    legend6 = axes1[0,1].legend([lines2_flip2[i] for i in [7,8,9]], [lines2_flip2[i].get_label() for i in [7,8,9]], bbox_to_anchor=(0.78, 1.0),loc='upper left',fontsize = font_size_legend,ncol=1,frameon = True,numpoints=2,handlelength=4.0,borderpad=0.15)  
    axes1[0,1].add_artist(legend4)
    axes1[0,1].add_artist(legend5)
    axes1[0,1].add_artist(legend6)
    
    
    fig1.tight_layout()
    if save_flag == 1:
        fig1.savefig(path_out+"fig5"+figs_format,bbox_inches='tight')
        plt.close(fig1)
