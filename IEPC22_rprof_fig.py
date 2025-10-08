#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:41:10 2022

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
prof_plots          = 1


path_out = "VHT_US_plume_sims/1Dr_comp_Gcases/"
#path_out = "VHT_US_plume_sims/1Dr_comp_Lcases/"
#path_out = "VHT_US_plume_sims/comp_Np/"


#path_out = "removing_it_matching/60000steps_after_changes_it_matching_modified_iterations/"


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

# Define the reshape variable function ####################################
def reshape_var(f,name,vartype,dims1,dims2,nsteps,timestep):
    
    import numpy as np
    
    if timestep == 'all':
        var = np.zeros((dims1,dims2,nsteps),dtype=vartype)
        for i in range(0,nsteps):
            var[:,:,i] = np.reshape(f[name][i,0:],(dims1,dims2),order='F')
    else:
        var = np.zeros((dims1,dims2),dtype=vartype)
        var = np.reshape(f[name][timestep,0:],(dims1,dims2),order='F')
        
    return var

###############################################################################



if prof_plots == 1:
    print("######## prof_plots ########")
    
    marker_size  = 4
    marker_size_cath = 14
    cathode_marker = '*'
    cathode_color  = orange
    ref_color      = 'c'
    marker_every = 5
#    marker_every = 1
#    font_size_legend    = font_size - 15
    font_size_legend    = 15
#    font_size_legend    = 8
    
    # Radial index for axial profiles

#    rind = 21   # New picM for SPT100
#    rind = 32   # New picM for SPT100 (picrm)
#    rind = 17   # Safran T1,T2 cases
#    rind = 17   # VHT_US coarse mesh
#    rind = 20   # VHT_US fine mesh
    rind = 15   # VHT_US fine mesh Np
#    rind = 36   # Safran T1,T2 cases (picrm)
#    rind = 19   # picM for SPT100 thesis
#    rind = 29    # HT5k rm6
#    rind = 15   # Safran PPSX00 Cheops LP
    # Cathode plotting flag and cathode position in cm (for plot_zcath_012 = 2,3) 
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
    # VHT_US ------------------------------------------------------------------
    zcat_pos = 15.55
    
    elems_cath_Bline    = range(407-1,483-1+2,2) # Elements along the cathode B line for cases C1, C2 and C3
    elems_cath_Bline_2  = range(875-1,951-1+2,2) # Elements along the cathode B line for case C5 (C4 thesis)
    elems_Bline         = range(330-1,406-1+2,2) # Elements along a B line
    elems_cath_Bline    = list(range(994-1,926-1+2,-2)) + list([923-1,922-1,920-1,917-1,916-1,914-1,912-1,909-1,907-1,906-1,904-1]) # Elements along the cathode B line for HT5k rm6 cathode at volume 922 or face 1637
#    elems_cath_Bline    = list(range(1968-1,1922-1+2,-2)) + list([1925-1]) + list(range(1922-1,1908-1+2,-2)) +list([1911-1]) + list(range(1908-1,1894-1+2,-2)) + list([1896-1,1895-1,1892-1,1890-1,1888-1,1886-1,1884-1,1882-1,1880-1,1971-1]) # Elements along the cathode B line for HT5k rm4 cathode at volume 1966 or face 3464
    ref_elem            = elems_Bline[len(elems_Bline)/2]
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
    last_steps      = 600
    last_steps      = 700
    last_steps      = 1000
    last_steps      = 1200
#    last_steps      = 800
#    last_steps      = 40
    step_i          = 200
    step_f          = 700
    plot_mean_vars  = 1
    
    plot_r_prof_IEPC22   = 1
    plot_r_prof          = 1
    
    
    

    if timestep == 'last':
        timestep = -1  
    if allsteps_flag == 0:
        mean_vars = 0

    
    # Simulation names
    nsims = 3

    
    
    oldpost_sim      = np.array([6,6,6,6,6,6,6,6,5,4,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([17,17,17,17,17,17,17,17,15,13,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)  
    


    
    sim_names = [
#                 "../../../Mg_hyphen_alejandro/sim/sims/Plume20_OP3_local_CEX_Np",
#                 "../../../Mg_hyphen_alejandro/sim/sims/Plume30_OP3_local_CEX_Np",
#                 "../../../Mg_hyphen_alejandro/sim/sims/Plume40_OP3_local_CEX_Np",
                 
                 "../../../Mg_hyphen_alejandro/sim/sims/Plume20_OP3_global_CEX_Np",
                 "../../../Mg_hyphen_alejandro/sim/sims/Plume30_OP3_global_CEX_Np",
                 "../../../Mg_hyphen_alejandro/sim/sims/Plume40_OP3_global_CEX_Np",

#                "../../../Mg_hyphen_alejandro/sim/sims/Plume20_OP3_global_Np",
            
            
#                "../../../H_sims/Mg/hyphen/sims/Plume30_OP3_global_CEX",
#                "../../../H_sims/Mg/hyphen/sims/Plume40_OP3_global_CEX",    
#                "../../../H_sims/Mg/hyphen/sims/Plume30_OP3_local_CEX",
#                "../../../H_sims/Mg/hyphen/sims/Plume40_OP3_local_CEX",
    
##                "../../../H_sims/Mg/hyphen/sims/Plume10_OP3_global_CEX",
#                "../../../H_sims/Mg/hyphen/sims/Plume20_OP3_global_CEX",
#                "../../../H_sims/Mg/hyphen/sims/Plume30_OP3_global_CEX",
#                "../../../H_sims/Mg/hyphen/sims/Plume40_OP3_global_CEX",
                
                
##                "../../../H_sims/Mg/hyphen/sims/Plume10_OP3_local_CEX",
#                "../../../H_sims/Mg/hyphen/sims/Plume20_OP3_local_CEX",
#                "../../../H_sims/Mg/hyphen/sims/Plume30_OP3_local_CEX",
#                "../../../H_sims/Mg/hyphen/sims/Plume40_OP3_local_CEX",
                
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume10_OP3_local_CEX",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume10_OP3_global_CEX",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume20_OP3_local_CEX",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume20_OP3_global_CEX",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume30_OP3_local_CEX",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume30_OP3_global_CEX",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume40_OP3_local_CEX",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume40_OP3_global_CEX",
                

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
                              "aspire_picM_rm6.hdf5",
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
                              "PIC_mesh.hdf5", # VHT_US
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
                              
#                              "SPT100_picM_Reference1500points.hdf5",
                              "SPT100_picM_Reference1500points_rm.hdf5",
                              "SPT100_picM_Reference1500points_rm.hdf5",
                              "SPT100_picM.hdf5",
                              "SPT100_picM.hdf5",
                              "SPT100_picM.hdf5",
                              ]

                        
    labels = [
#              r"P2G no CEX",
#              r"P2G no CEX Np",
    
#               r"P1G",
               r"P2G",
               r"P3G",
               r"P4G",
               
##               r"P1L",
#               r"P2L",
#               r"P3L",
#               r"P4L",
            
               r"",
               r"",
               r"",
               r"",
               r"",
               r"",
               r"",
               r"",
            
              ]

    
    # Line colors
    colors = ['k','r','g','b','m','c','m','y',orange,brown]
    colors = ['m','b','k','r','g','b','m','c','m','y',orange,brown] # P1G-P4G
    colors = ['m','b','r','g','b','m','c','m','y',orange,brown]     # P1L, P2L and P4L
#    colors = ['k','m',orange,brown]
    # Markers
    markers = ['s','o','v','^','<', '>','D','p','*']
    markers = ['','','','','','','','','']
#    markers = ['s','<','D','p']
    # Line style
    linestyles = ['-','--','-.', ':','-','--','-.']
    linestyles = ['-','-','-','-','-','-','-']
    linestyles = ['--','-','-.', ':','-','--','-.'] # P1G-P4G
    linestyles = ['--','-',':','-','--','-.'] # P1L, P2L and P4L
    
#    xmax = 18  # FOR CHEOPS-1 Topo1 Topo2 sims
#    xmax = 12 # FOR SPT sims
#    xmax = 7.5  # FOR cheops LP sims
#    xmax = 12.9  # FOR VHT_US sims plume 10
#    xmax = 22.9  # FOR VHT_US sims plume 20
#    xmax = 32.9  # FOR VHT_US sims plume 30
    xmax = 42.9  # FOR VHT_US sims plume 40
    
    
    # Do not plot units in axes (comment if we want units in axes)
#    # SAFRAN CHEOPS 1: units in cm
#    L_c = 3.725
#    H_c = (0.074995-0.052475)*100
     # HT5k
#    L_c = 2.53
#    H_c = (0.0785-0.0565)*100
    # VHT_US
    L_c = 2.9
    H_c = 2.22

    xmax = xmax/L_c
    zcat_pos = zcat_pos/L_c
    
#    prof_xlabel = r"$z$ (cm)"
    prof_xlabel = r"$z/L_\mathrm{c}$"
    

    # Radial profile plots
    if plot_r_prof_IEPC22 == 1:    
        plt.figure('je_z r prof1')
        plt.xlabel(r"$r/H_\mathrm{c}$",fontsize = font_size)
        plt.ylabel(r"$-j_\mathrm{ze}$ (A/cm$^{2}$)", fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.figure('ji_z r prof1')
        plt.xlabel(r"$r/H_\mathrm{c}$",fontsize = font_size)
        plt.ylabel(r"$j_\mathrm{zi}$ (A/cm$^{2}$)", fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        
        plt.figure('je_z r prof2')
        plt.xlabel(r"$r/H_\mathrm{c}$",fontsize = font_size)
        plt.ylabel(r"$-j_\mathrm{ze}$ (A/cm$^{2}$)", fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.figure('ji_z r prof2')
        plt.xlabel(r"$r/H_\mathrm{c}$",fontsize = font_size)
        plt.ylabel(r"$j_\mathrm{zi}$ (A/cm$^{2}$)", fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
    if plot_r_prof == 1:
        plt.figure('je_z r prof')
        plt.xlabel(r"$r/H_\mathrm{c}$",fontsize = font_size)
        plt.ylabel(r"$-j_\mathrm{ze}$ (A/cm$^{2}$)", fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.figure('ji_z r prof')
        plt.xlabel(r"$r/H_\mathrm{c}$",fontsize = font_size)
        plt.ylabel(r"$j_\mathrm{zi}$ (A/cm$^{2}$)", fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)
        plt.figure('ne r prof')
        plt.xlabel(r"$r/H_\mathrm{c}$",fontsize = font_size)
        plt.ylabel(r"$n_\mathrm{e}$ (m$^{-3}$)", fontsize = font_size)
        plt.xticks(fontsize = ticks_size) 
        plt.yticks(fontsize = ticks_size)

        
    ind  = 0
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
        if k == 0:
            path_simstate_out = sim_names[k]+"/CORE/out/SimState.hdf5"
            path_postdata_out = sim_names[k]+"/CORE/out/PostData.hdf5"
            path_simparams_inp = sim_names[k]+"/CORE/inp/sim_params.inp"

        elif k >= 1:
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
        
        
        #### NOTE: After change in eFld collisions, ionization collisions are 
        #          not multiplied by the charge number jump (as before). 
        #          We do it here
        if oldpost_sim[k] >= 3:
            nu_i02 = 2.0*nu_i02
        #######################################################################
        # Open the PostData.hdf5 file
        h5_post = h5py.File(path_postdata_out,"r+")
        
        je_b_nodes   = reshape_var(h5_post,"/picM_data/je_b_acc","float",dims[0],dims[1],nsteps,"all")
        je_b_nodes[np.where(nodes_flag == 0)]     = np.nan
        
        je_b_nodes_mean = np.nanmean(je_b_nodes[:,:,nsteps-last_steps::],axis=2)
        je_b_nodes_plot = np.copy(je_b_nodes_mean)
        
        print("Generating plotting variables (NaN in ghost nodes)...")                                                                                                      
        [Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,
          nn2,nn3,ni1,ni2,ni3,ni4,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,
          fn2_z,fn3_x,fn3_y,fn3_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,
          fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,un1_x,un1_y,
          un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,ui1_x,ui1_y,
          ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,
          ui4_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,
          ji3_z,ji4_x,ji4_y,ji4_z,je_r,je_t,je_z,je_perp,je_para,
          ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,
          Te,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,
          avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
          ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,
          ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
          ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,F_theta,
          Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,
          nu_ei2,nu_i01,nu_i02,nu_i12,err_interp_n,f_split_adv,
          f_split_qperp,f_split_qpara,f_split_qb,f_split_Pperp,
          f_split_Ppara,f_split_ecterm,f_split_inel] = HET_sims_plotvars(nodes_flag,cells_flag,Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,
                                                                         nn2,nn3,ni1,ni2,ni3,ni4,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,
                                                                         fn2_z,fn3_x,fn3_y,fn3_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,
                                                                         fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,un1_x,un1_y,
                                                                         un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,ui1_x,ui1_y,
                                                                         ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,
                                                                         ui4_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,
                                                                         ji3_z,ji4_x,ji4_y,ji4_z,je_r,je_t,je_z,je_perp,je_para,
                                                                         ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,
                                                                         Te,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,
                                                                         avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
                                                                         ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,
                                                                         ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
                                                                         ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,F_theta,
                                                                         Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,
                                                                         nu_ei2,nu_i01,nu_i02,nu_i12,err_interp_n,f_split_adv,
                                                                         f_split_qperp,f_split_qpara,f_split_qb,f_split_Pperp,
                                                                         f_split_Ppara,f_split_ecterm,f_split_inel)
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
               ue_perp_mean,ue_para_mean,uthetaExB_mean,Tn1_mean,Tn2_mean,Ti1_mean,
               Ti2_mean,Te_mean,n_mp_n1_mean,n_mp_n2_mean,n_mp_i1_mean,n_mp_i2_mean,
               avg_w_n1_mean,avg_w_n2_mean,avg_w_i1_mean,avg_w_i2_mean,
               neu_gen_weights1_mean,neu_gen_weights2_mean,ion_gen_weights1_mean,
               ion_gen_weights2_mean,ndot_ion01_n1_mean,ndot_ion02_n1_mean,
               ndot_ion12_i1_mean,ndot_ion01_n2_mean,ndot_ion02_n2_mean,
               ndot_ion01_n3_mean,ndot_ion02_n3_mean,ndot_ion12_i3_mean,
               ne_cath_mean,Te_cath_mean,nu_cath_mean,ndot_cath_mean,F_theta_mean,
               Hall_par_mean,Hall_par_eff_mean,nu_e_tot_mean,nu_e_tot_eff_mean,
               nu_en_mean,nu_ei1_mean,nu_ei2_mean,nu_i01_mean,nu_i02_mean,nu_i12_mean,
               Boltz_mean,Boltz_dim_mean,phi_elems_mean,ne_elems_mean,Te_elems_mean,
               err_interp_n_mean,f_split_adv_mean,f_split_qperp_mean,f_split_qpara_mean,
               f_split_qb_mean,f_split_Pperp_mean,f_split_Ppara_mean,
               f_split_ecterm_mean,f_split_inel_mean] = HET_sims_mean(nsteps,mean_type,last_steps,step_i,step_f,Z_ion_spe,
                                                                      num_ion_spe,num_neu_spe,phi,Er,Ez,
                                                                      Efield,Br,Bz,Bfield,nn1,nn2,nn3,ni1,ni2,ni3,ni4,ne,fn1_x,
                                                                      fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fn3_x,fn3_y,fn3_z,fi1_x,fi1_y,
                                                                      fi1_z,fi2_x,fi2_y,fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,
                                                                      un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,ui1_x,
                                                                      ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,
                                                                      ui4_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,ji3_z,
                                                                      ji4_x,ji4_y,ji4_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,
                                                                      ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
                                                                      n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                                                      neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                                                      ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
                                                                      ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,ne_cath,Te_cath,
                                                                      nu_cath,ndot_cath,F_theta,Hall_par,Hall_par_eff,nu_e_tot,
                                                                      nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,Boltz,
                                                                      Boltz_dim,phi_elems,ne_elems,Te_elems,err_interp_n,
                                                                      f_split_adv,f_split_qperp,f_split_qpara,f_split_qb,f_split_Pperp,
                                                                      f_split_Ppara,f_split_ecterm,f_split_inel)
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
               Tn1_plot,Tn2_plot,Ti1_plot,Ti2_plot,Te_plot,n_mp_n1_plot,n_mp_n2_plot,
               n_mp_i1_plot,n_mp_i2_plot,avg_w_n1_plot,avg_w_n2_plot,avg_w_i1_plot,
               avg_w_i2_plot,neu_gen_weights1_plot,neu_gen_weights2_plot,
               ion_gen_weights1_plot,ion_gen_weights2_plot,ndot_ion01_n1_plot,
               ndot_ion02_n1_plot,ndot_ion12_i1_plot,ndot_ion01_n2_plot,
               ndot_ion02_n2_plot,ndot_ion01_n3_plot,ndot_ion02_n3_plot,
               ndot_ion12_i3_plot,ne_cath_plot,nu_cath_plot,ndot_cath_plot,
               F_theta_plot,Hall_par_plot,Hall_par_eff_plot,nu_e_tot_plot,
               nu_e_tot_eff_plot,nu_en_plot,nu_ei1_plot,nu_ei2_plot,nu_i01_plot,
               nu_i02_plot,nu_i12_plot,err_interp_n_plot,f_split_adv_plot,
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
                                                     ue_para_mean,uthetaExB_mean,Tn1_mean,Tn2_mean,Ti1_mean,Ti2_mean,Te_mean,
                                                     n_mp_n1_mean,n_mp_n2_mean,n_mp_i1_mean,n_mp_i2_mean,avg_w_n1_mean,
                                                     avg_w_n2_mean,avg_w_i1_mean,avg_w_i2_mean,neu_gen_weights1_mean,
                                                     neu_gen_weights2_mean,ion_gen_weights1_mean,ion_gen_weights2_mean,
                                                     ndot_ion01_n1_mean,ndot_ion02_n1_mean,ndot_ion12_i1_mean,
                                                     ndot_ion01_n2_mean,ndot_ion02_n2_mean,ndot_ion01_n3_mean,
                                                     ndot_ion02_n3_mean,ndot_ion12_i3_mean,ne_cath_mean,
                                                     nu_cath_mean,ndot_cath_mean,F_theta_mean,Hall_par_mean,Hall_par_eff_mean,
                                                     nu_e_tot_mean,nu_e_tot_eff_mean,nu_en_mean,nu_ei1_mean,nu_ei2_mean,nu_i01_mean,
                                                     nu_i02_mean,nu_i12_mean,err_interp_n_mean,f_split_adv_mean,f_split_qperp_mean,
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
               Tn1_plot,Tn2_plot,Ti1_plot,Ti2_plot,Te_plot,n_mp_n1_plot,n_mp_n2_plot,
               n_mp_i1_plot,n_mp_i2_plot,avg_w_n1_plot,avg_w_n2_plot,avg_w_i1_plot,
               avg_w_i2_plot,neu_gen_weights1_plot,neu_gen_weights2_plot,
               ion_gen_weights1_plot,ion_gen_weights2_plot,ndot_ion01_n1_plot,
               ndot_ion02_n1_plot,ndot_ion12_i1_plot,ndot_ion01_n2_plot,
               ndot_ion02_n2_plot,ndot_ion01_n3_plot,ndot_ion02_n3_plot,
               ndot_ion12_i3_plot,ne_cath_plot,nu_cath_plot,ndot_cath_plot,
               F_theta_plot,Hall_par_plot,Hall_par_eff_plot,nu_e_tot_plot,
               nu_e_tot_eff_plot,nu_en_plot,nu_ei1_plot,nu_ei2_plot,nu_i01_plot,
               nu_i02_plot,nu_i12_plot,err_interp_n_plot,f_split_adv_plot,
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
                                                     ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,
                                                     Te,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,
                                                     avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
                                                     ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,
                                                     ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
                                                     ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,ne_cath,nu_cath,
                                                     ndot_cath,F_theta,Hall_par,Hall_par_eff,nu_e_tot,
                                                     nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,
                                                     err_interp_n,f_split_adv,f_split_qperp,f_split_qpara,
                                                     f_split_qb,f_split_Pperp,f_split_Ppara,f_split_ecterm,
                                                     f_split_inel)
                                                                                                          
         
        # Setting a common reference potential point
        if phi_ref == 1:
            phi_plot = phi_plot - phi_plot[iphi_ref,jphi_ref]
        
 
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
         

        ratio_ni1_ni2_plot      = np.divide(ni2_plot,ni1_plot)
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
        lambdaD_plot            = np.sqrt(eps0*(e*Te_plot)/(ne_plot*e**2))
        nn_plot                 = nn1_plot + nn2_plot + nn3_plot
        
        
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
       
        
    
        ###########################################################################
        print("Plotting...")
        ############################ GENERATING PLOTS #############################
#        print("erel_ue max         = %15.8e; erel_ue min         = %15.8e (-)" %( np.nanmax(erel_ue_plot[rind,:]), np.nanmin(erel_ue_plot[rind,:]) ) )
#        print("erel_je max         = %15.8e; erel_je min         = %15.8e (-)" %( np.nanmax(erel_je_plot[rind,:]), np.nanmin(erel_je_plot[rind,:]) ) )
#        print("erel_jeji max       = %15.8e; erel_jeji min       = %15.8e (-)" %( np.nanmax(erel_jeji_plot[rind,:]), np.nanmin(erel_jeji_plot[rind,:]) ) )
#        print("phi max             = %15.8e; phi min             = %15.8e (V)" %( np.nanmax(phi_plot[rind,:]), np.nanmin(phi_plot[rind,:]) ) )
#        print("Efield max          = %15.8e; Efield min          = %15.8e (V/m)" %( np.nanmax(Efield_plot[rind,:]), np.nanmin(Efield_plot[rind,:]) ) )
#        print("Bfield max          = %15.8e; Bfield min          = %15.8e (G)" %( np.nanmax(Bfield_plot[rind,:]*1E4), np.nanmin(Bfield_plot[rind,:]*1E4) ) )
#        print("Er max              = %15.8e; Er min              = %15.8e (V/m)" %( np.nanmax(Er_plot[rind,:]), np.nanmin(Er_plot[rind,:]) ) )
#        print("Ez max              = %15.8e; Ez min              = %15.8e (V/m)" %( np.nanmax(Ez_plot[rind,:]), np.nanmin(Ez_plot[rind,:]) ) )
#        print("ne max              = %15.8e; ne min              = %15.8e (1/m3)" %( np.nanmax(ne_plot[rind,:]), np.nanmin(ne_plot[rind,:]) ) )
#        print("ni1 max             = %15.8e; ni1 min             = %15.8e (1/m3)" %( np.nanmax(ni1_plot[rind,:]), np.nanmin(ni1_plot[rind,:]) ) )
#        print("ni2 max             = %15.8e; ni2 min             = %15.8e (1/m3)" %( np.nanmax(ni2_plot[rind,:]), np.nanmin(ni2_plot[rind,:]) ) )
#        print("ni1/ni2 max         = %15.8e; ni1/ni2 min         = %15.8e (-)" %( np.nanmax(ratio_ni1_ni2_plot[rind,:]), np.nanmin(ratio_ni1_ni2_plot[rind,:]) ) )
#        print("nn1 max             = %15.8e; nn1 min             = %15.8e (1/m3)" %( np.nanmax(nn1_plot[rind,:]), np.nanmin(nn1_plot[rind,:]) ) )
#        print("Te max              = %15.8e; Te min              = %15.8e (eV)" %( np.nanmax(Te_plot[rind,:]), np.nanmin(Te_plot[rind,:]) ) )
#        print("Ti1 max             = %15.8e; Ti1 min             = %15.8e (eV)" %( np.nanmax(Ti1_plot[rind,:]), np.nanmin(Ti1_plot[rind,:]) ) )
#        print("Ti2 max             = %15.8e; Ti2 min             = %15.8e (eV)" %( np.nanmax(Ti2_plot[rind,:]), np.nanmin(Ti2_plot[rind,:]) ) )
#        print("Tn1 max             = %15.8e; Tn1 min             = %15.8e (eV)" %( np.nanmax(Tn1_plot[rind,:]), np.nanmin(Tn1_plot[rind,:]) ) )
#        print("Ekin_e max          = %15.8e; Ekin_e min          = %15.8e (eV)" %( np.nanmax(Ekin_e_plot[rind,:]), np.nanmin(Ekin_e_plot[rind,:]) ) )
#        print("Ekin_i1 max         = %15.8e; Ekin_i1 min         = %15.8e (eV)" %( np.nanmax(Ekin_i1_plot[rind,:]), np.nanmin(Ekin_i1_plot[rind,:]) ) )
#        print("Ekin_i2 max         = %15.8e; Ekin_i2 min         = %15.8e (eV)" %( np.nanmax(Ekin_i2_plot[rind,:]), np.nanmin(Ekin_i2_plot[rind,:]) ) )
#        print("Ekin/Te max         = %15.8e; Ekin/Te min         = %15.8e (-)" %( np.nanmax(ratio_Ekin_Te_plot[rind,:]), np.nanmin(ratio_Ekin_Te_plot[rind,:]) ) )
#        print("Ekin/Ti1 max        = %15.8e; Ekin/Ti1 min        = %15.8e (-)" %( np.nanmax(ratio_Ekin_Ti1_plot[rind,:]), np.nanmin(ratio_Ekin_Ti1_plot[rind,:]) ) )
#        print("Ekin/Ti2 max        = %15.8e; Ekin/Ti2 min        = %15.8e (-)" %( np.nanmax(ratio_Ekin_Ti2_plot[rind,:]), np.nanmin(ratio_Ekin_Ti2_plot[rind,:]) ) )
#        print("Mi1 max             = %15.8e; Mi1 min             = %15.8e (-)" %( np.nanmax(Mi1_plot[rind,:]), np.nanmin(Mi1_plot[rind,:]) ) )
#        print("Mi2 max             = %15.8e; Mi2 min             = %15.8e (-)" %( np.nanmax(Mi2_plot[rind,:]), np.nanmin(Mi2_plot[rind,:]) ) )
#        print("Mi max              = %15.8e; Mi min              = %15.8e (-)" %( np.nanmax(Mi_plot[rind,:]), np.nanmin(Mi_plot[rind,:]) ) )
#        print("Mi1_z max           = %15.8e; Mi1_z min           = %15.8e (-)" %( np.nanmax(Mi1_z_plot[rind,:]), np.nanmin(Mi1_z_plot[rind,:]) ) )
#        print("Mi2_z max           = %15.8e; Mi2_z min           = %15.8e (-)" %( np.nanmax(Mi2_z_plot[rind,:]), np.nanmin(Mi2_z_plot[rind,:]) ) )
#        print("Mi_z max            = %15.8e; Mi_z min            = %15.8e (-)" %( np.nanmax(Mi_z_plot[rind,:]), np.nanmin(Mi_z_plot[rind,:]) ) )
#        print("ui_x max            = %15.8e; ui_x min            = %15.8e (m/s)" %( np.nanmax(ui_x_plot[rind,:]), np.nanmin(ui_x_plot[rind,:]) ) )
#        print("ui_y max            = %15.8e; ui_y min            = %15.8e (m/s)" %( np.nanmax(ui_y_plot[rind,:]), np.nanmin(ui_y_plot[rind,:]) ) )
#        print("ui_z max            = %15.8e; ui_z min            = %15.8e (m/s)" %( np.nanmax(ui_z_plot[rind,:]), np.nanmin(ui_z_plot[rind,:]) ) )
#        print("ue_r max            = %15.8e; ue_r min            = %15.8e (m/s)" %( np.nanmax(ue_r_plot[rind,:]), np.nanmin(ue_r_plot[rind,:]) ) )
#        print("ue_t max            = %15.8e; ue_t min            = %15.8e (m/s)" %( np.nanmax(ue_t_plot[rind,:]), np.nanmin(ue_t_plot[rind,:]) ) )
#        print("ue_z max            = %15.8e; ue_z min            = %15.8e (m/s)" %( np.nanmax(ue_z_plot[rind,:]), np.nanmin(ue_z_plot[rind,:]) ) )
#        print("ue_perp max         = %15.8e; ue_perp min         = %15.8e (m/s)" %( np.nanmax(ue_perp_plot[rind,:]), np.nanmin(ue_perp_plot[rind,:]) ) )
#        print("ue_para max         = %15.8e; ue_para min         = %15.8e (m/s)" %( np.nanmax(ue_para_plot[rind,:]), np.nanmin(ue_para_plot[rind,:]) ) )
#        print("ue_t/ue_perp max    = %15.8e; ue_t/ue_perp min    = %15.8e (m/s)" %( np.nanmax(ratio_ue_t_perp_plot[rind,:]), np.nanmin(ratio_ue_t_perp_plot[rind,:]) ) )
#        print("ue_t/ue_para max    = %15.8e; ue_t/ue_para min    = %15.8e (m/s)" %( np.nanmax(ratio_ue_t_para_plot[rind,:]), np.nanmin(ratio_ue_t_para_plot[rind,:]) ) )
#        print("ue_perp/ue_para max = %15.8e; ue_perp/ue_para min = %15.8e (m/s)" %( np.nanmax(ratio_ue_perp_para_plot[rind,:]), np.nanmin(ratio_ue_perp_para_plot[rind,:]) ) )
#        print("uthetaExB max       = %15.8e; uthetaExB min       = %15.8e (m/s)" %( np.nanmax(uthetaExB_plot[rind,:]), np.nanmin(uthetaExB_plot[rind,:]) ) )        
#        print("je_r max            = %15.8e; je_r min            = %15.8e (A/m2)" %( np.nanmax(je_r_plot[rind,:]), np.nanmin(je_r_plot[rind,:]) ) )
#        print("je_t max            = %15.8e; je_t min            = %15.8e (A/m2)" %( np.nanmax(je_t_plot[rind,:]), np.nanmin(je_t_plot[rind,:]) ) )
#        print("je_z max            = %15.8e; je_z min            = %15.8e (A/m2)" %( np.nanmax(je_z_plot[rind,:]), np.nanmin(je_z_plot[rind,:]) ) )
#        print("je_perp max         = %15.8e; je_perp min         = %15.8e (A/m2)" %( np.nanmax(je_perp_plot[rind,:]), np.nanmin(je_perp_plot[rind,:]) ) )
#        print("je_para max         = %15.8e; je_para min         = %15.8e (A/m2)" %( np.nanmax(je_para_plot[rind,:]), np.nanmin(je_para_plot[rind,:]) ) )
#        print("je_t/je_perp max    = %15.8e; je_t/je_perp min    = %15.8e (-)" %( np.nanmax(ratio_je_t_perp_plot[rind,:]), np.nanmin(ratio_je_t_perp_plot[rind,:]) ) )
#        print("F_theta max         = %15.8e; F_theta min         = %15.8e (A/m2)" %( np.nanmax(F_theta_plot[rind,:]), np.nanmin(F_theta_plot[rind,:]) ) )        
#        print("je max              = %15.8e; je min              = %15.8e (A/m2)" %( np.nanmax(je_plot[rind,:]), np.nanmin(je_plot[rind,:]) ) )
#        print("ji max              = %15.8e; ji min              = %15.8e (A/m2)" %( np.nanmax(ji_plot[rind,:]), np.nanmin(ji_plot[rind,:]) ) )
#        print("ji1 max             = %15.8e; ji1 min             = %15.8e (A/m2)" %( np.nanmax(ji1_plot[rind,:]), np.nanmin(ji1_plot[rind,:]) ) )
#        print("ji2 max             = %15.8e; ji2 min             = %15.8e (A/m2)" %( np.nanmax(ji2_plot[rind,:]), np.nanmin(ji2_plot[rind,:]) ) )
#        print("j max               = %15.8e; j min               = %15.8e (A/m2)" %( np.nanmax(j_plot[rind,:]), np.nanmin(j_plot[rind,:]) ) )
#        print("je2D max            = %15.8e; je2D min            = %15.8e (A/m2)" %( np.nanmax(je2D_plot[rind,:]), np.nanmin(je2D_plot[rind,:]) ) )
#        print("ji2D max            = %15.8e; ji2D min            = %15.8e (A/m2)" %( np.nanmax(ji2D_plot[rind,:]), np.nanmin(ji2D_plot[rind,:]) ) )
#        print("j2D max             = %15.8e; j2D min             = %15.8e (A/m2)" %( np.nanmax(j2D_plot[rind,:]), np.nanmin(j2D_plot[rind,:]) ) )
#        print("Hall_par max        = %15.8e; Hall_par min        = %15.8e (-)" %( np.nanmax(Hall_par_plot[rind,:]), np.nanmin(Hall_par_plot[rind,:]) ) )
#        print("Hall_par_eff max    = %15.8e; Hall_par_eff min    = %15.8e (-)" %( np.nanmax(Hall_par_eff_plot[rind,:]), np.nanmin(Hall_par_eff_plot[rind,:]) ) )
#        print("nu_e_tot max        = %15.8e; nu_e_tot min        = %15.8e (Hz)" %( np.nanmax(nu_e_tot_plot[rind,:]), np.nanmin(nu_e_tot_plot[rind,:]) ) )
#        print("nu_e_tot_eff max    = %15.8e; nu_e_tot_eff min    = %15.8e (Hz)" %( np.nanmax(nu_e_tot_eff_plot[rind,:]), np.nanmin(nu_e_tot_eff_plot[rind,:]) ) )
#        print("nu_en max           = %15.8e; nu_en min           = %15.8e (Hz)" %( np.nanmax(nu_en_plot[rind,:]), np.nanmin(nu_en_plot[rind,:]) ) )
#        print("nu_ei1 max          = %15.8e; nu_ei1 min          = %15.8e (Hz)" %( np.nanmax(nu_ei1_plot[rind,:]), np.nanmin(nu_ei1_plot[rind,:]) ) )
#        print("nu_ei2 max          = %15.8e; nu_ei2 min          = %15.8e (Hz)" %( np.nanmax(nu_ei2_plot[rind,:]), np.nanmin(nu_ei2_plot[rind,:]) ) )
#        print("nu_i01 max          = %15.8e; nu_i01 min          = %15.8e (Hz)" %( np.nanmax(nu_i01_plot[rind,:]), np.nanmin(nu_i01_plot[rind,:]) ) )
#        print("nu_i02 max          = %15.8e; nu_i02 min          = %15.8e (Hz)" %( np.nanmax(nu_i02_plot[rind,:]), np.nanmin(nu_i02_plot[rind,:]) ) )
#        print("nu_i12 max          = %15.8e; nu_i12 min          = %15.8e (Hz)" %( np.nanmax(nu_i12_plot[rind,:]), np.nanmin(nu_i12_plot[rind,:]) ) )
#        print("nu_ei_el_tot max    = %15.8e; nu_ei_el_tot min    = %15.8e (Hz)" %( np.nanmax(nu_ei_el_tot_plot[rind,:]), np.nanmin(nu_ei_el_tot_plot[rind,:]) ) )
#        print("nu_ion_tot max      = %15.8e; nu_ion_tot min      = %15.8e (Hz)" %( np.nanmax(nu_ion_tot_plot[rind,:]), np.nanmin(nu_ion_tot_plot[rind,:]) ) )
#        print("lambdaD max         = %15.8e; lambdaD min         = %15.8e (mm)" %( np.nanmax(lambdaD_plot*1E3), np.nanmin(lambdaD_plot*1E3) ) )
#        print("###### Values at the cathode #######")     
#        if len(elems_cath_Bline) > 0:
#            print("cath_ne        = %15.8e" %cath_ne)
#            print("cath_Te        = %15.8e" %cath_Te)
#        print("ne_cath_mean        = %15.8e" %np.mean(ne_cath_mean))
#        print("Te_cath_mean        = %15.8e" %np.mean(Te_cath_mean))
        
        
        
        
        zs                = zs*1E2
        rs                = rs*1E2
        zscells           = zscells*1E2
        rscells           = rscells*1E2
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
        je_r_plot         = je_r_plot*1E-3
        je_t_plot         = je_t_plot*1E-4
        je_z_plot         = je_z_plot*1E-3
        je_para_plot      = je_para_plot*1E-3
        je_perp_plot      = je_perp_plot*1E-3
        ji_x_plot         = ji_x_plot*1E-2
        ji_z_plot         = ji_z_plot*1E-2
        je2D_plot         = je2D_plot
        ji2D_plot         = ji2D_plot
        j2D_plot          = j2D_plot
#        nu_e_tot_plot     = nu_e_tot_plot*1E-6
#        nu_e_tot_eff_plot = nu_e_tot_eff_plot*1E-6
#        nu_en_plot        = nu_en_plot*1E-6
#        nu_ei1_plot       = nu_ei1_plot*1E-6
#        nu_ei2_plot       = nu_ei2_plot*1E-6
#        nu_i01_plot       = nu_i01_plot*1E-6
#        nu_i02_plot       = nu_i02_plot*1E-6
#        nu_i12_plot       = nu_i12_plot*1E-6
        lambdaD_plot = lambdaD_plot*1E3
        je_b_nodes_plot = je_b_nodes_plot*1E-3
        
        
        
        # Comment the following lines if we want units in axes
        zs = zs/L_c
        rs = rs/H_c
        points[:,0] = points[:,0]/L_c
        points[:,1] = points[:,1]/H_c
        z_cath = z_cath/L_c
        r_cath = r_cath/H_c
        zscells = zscells/L_c
        rscells = rscells/H_c
        
        print("###### Values for IEPC22 ######")
        print("TeP (eV)      = %15.8e" %Te_plot[rind,-1])
        print("phiP (V)      = %15.8e" %phi_plot[rind,-1])
        print("jzeP (A/cm2)  = %15.8e" %(je_z_plot[rind,-1]*1E3*1E-4))
        print("jebP (A/cm2)  = %15.8e" %(je_b_nodes_plot[rind,-1]*1E3*1E-4))
        print("jziP (A/cm2)  = %15.8e" %(ji_z_plot[rind,-1]*1E2*1E-4))
        if zs[rind,-1] > zcat_pos:
            pos = np.where(zs[rind,:]>=zcat_pos)[0][0]
        else:
            pos = -1
        print("Vcoupling (V) = %15.8e" %phi_plot[rind,pos])
        
        
#        print("i1 Np axis cell 0 = "+str(n_mp_i1_plot[0,int(xi_bottom):-1:1]))
#        print("i2 Np axis cell 0 = "+str(n_mp_i2_plot[0,int(xi_bottom):-1:1]))
#        print("n1 Np axis cell 0 = "+str(n_mp_n1_plot[0,int(xi_bottom):-1:1]))
#        print("i1 Np axis cell 1 = "+str(n_mp_i1_plot[1,int(xi_bottom):-1:1]))
#        print("i2 Np axis cell 1 = "+str(n_mp_i2_plot[1,int(xi_bottom):-1:1]))
#        print("n1 Np axis cell 1 = "+str(n_mp_n1_plot[1,int(xi_bottom):-1:1]))
        
        print("i1 Np axis cell 0 = "+str(n_mp_i1_plot[0,dims[1]-6:dims[1]-1]))
        print("i2 Np axis cell 0 = "+str(n_mp_i2_plot[0,dims[1]-6:dims[1]-1]))
        print("n1 Np axis cell 0 = "+str(n_mp_n1_plot[0,dims[1]-6:dims[1]-1]))
        print("i1 Np axis cell 1 = "+str(n_mp_i1_plot[1,dims[1]-6:dims[1]-1]))
        print("i2 Np axis cell 1 = "+str(n_mp_i2_plot[1,dims[1]-6:dims[1]-1]))
        print("n1 Np axis cell 1 = "+str(n_mp_n1_plot[1,dims[1]-6:dims[1]-1]))
        
        
        # Modify the value of je_b_nodes at the node on the symmetry axis
        je_b_nodes_plot[0,-1] = (je_b_nodes_plot[2,-1] - je_b_nodes_plot[1,-1])/(rs[2,-1]-rs[1,-1])*(0.0 - rs[1,-1]) + je_b_nodes_plot[1,-1]
        
        
        if plot_r_prof_IEPC22 == 1:  
            # Obtain the z index for radial profile
            z_rprof = np.array([22.9/L_c,32.9/L_c],dtype=float)
            if k == 0:
                zind_prof = -1
                r_prof    = rs[:,zind_prof]
                jiz_rprof = ji_z_plot[:,zind_prof]*1E2*1E-4  # in A/cm2
#                jez_rprof = je_z_plot[:,zind_prof]*1E3*1E-4  # in A/cm2
                jez_rprof = je_b_nodes_plot[:,zind_prof]*1E3*1E-4  # in A/cm2
                
                plt.figure('je_z r prof1')
                plt.semilogy(r_prof,-jez_rprof, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker='o', color='k', markeredgecolor = 'k', label="")
    
                plt.figure('ji_z r prof1')
                plt.semilogy(r_prof,jiz_rprof, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker='o', color='k', markeredgecolor = 'k', label="")
        
            elif k == 1:
                z_rprof = 22.9/L_c
                posr1 = np.where(zs[rind,:]<z_rprof)[0][-1]
                posr2 = np.where(zs[rind,:]>z_rprof)[0][0]
                w1 = np.abs(zs[rind,posr2] - z_rprof)/np.abs(zs[rind,posr2] - zs[rind,posr1])
                w2 = np.abs(zs[rind,posr1] - z_rprof)/np.abs(zs[rind,posr2] - zs[rind,posr1])
                r_prof = w1*rs[:,posr1] + w2*rs[:,posr2]
                jiz_rprof = (w1*ji_z_plot[:,posr1] + w2*ji_z_plot[:,posr2])*1E2*1E-4  # in A/cm2
                jez_rprof = (w1*je_z_plot[:,posr1] + w2*je_z_plot[:,posr2])*1E3*1E-4  # in A/cm2
                
                plt.figure('je_z r prof1')
                plt.semilogy(r_prof,-jez_rprof, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker='o', color='b', markeredgecolor = 'b', label="")
    
                plt.figure('ji_z r prof1')
                plt.semilogy(r_prof,jiz_rprof, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker='o', color='b', markeredgecolor = 'b', label="")
                
                zind_prof = -1
                r_prof    = rs[:,zind_prof]
                jiz_rprof = ji_z_plot[:,zind_prof]*1E2*1E-4  # in A/cm2
                jez_rprof = je_z_plot[:,zind_prof]*1E3*1E-4  # in A/cm2
                
                plt.figure('je_z r prof1')
                plt.semilogy(r_prof,-jez_rprof, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker='s', color='b', markeredgecolor = 'b', label="")
    
                plt.figure('ji_z r prof1')
                plt.semilogy(r_prof,jiz_rprof, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker='s', color='b', markeredgecolor = 'b', label="")
                
            
            elif k == 2:
                z_rprof_arr = np.array([22.9/L_c,32.9/L_c],dtype=float)
                for iprof in range(0,2):
                    z_rprof = z_rprof_arr[iprof]
                    posr1 = np.where(zs[rind,:]<z_rprof)[0][-1]
                    posr2 = np.where(zs[rind,:]>z_rprof)[0][0]
                    w1 = np.abs(zs[rind,posr2] - z_rprof)/np.abs(zs[rind,posr2] - zs[rind,posr1])
                    w2 = np.abs(zs[rind,posr1] - z_rprof)/np.abs(zs[rind,posr2] - zs[rind,posr1])
                    r_prof = w1*rs[:,posr1] + w2*rs[:,posr2]
                    jiz_rprof = (w1*ji_z_plot[:,posr1] + w2*ji_z_plot[:,posr2])*1E2*1E-4  # in A/cm2
                    jez_rprof = (w1*je_z_plot[:,posr1] + w2*je_z_plot[:,posr2])*1E3*1E-4  # in A/cm2
                    if iprof == 0:
                        mark_type = 'o' 
                    else:
                        mark_type= 's'
                    plt.figure('je_z r prof1')
                    plt.semilogy(r_prof,-jez_rprof, linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=mark_type, color='r', markeredgecolor = 'r', label="")
        
                    plt.figure('ji_z r prof1')
                    plt.semilogy(r_prof,jiz_rprof, linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=mark_type, color='r', markeredgecolor = 'r', label="")
            
            # Obtain the z index for radial profile
            z_rprof = np.array([22.9/L_c,32.9/L_c],dtype=float)
            if k == 0:
                zind_prof = -1
                r_prof    = rs[:,zind_prof]
                jiz_rprof = ji_z_plot[:,zind_prof]*1E2*1E-4  # in A/cm2
#                jez_rprof = je_z_plot[:,zind_prof]*1E3*1E-4  # in A/cm2
                jez_rprof = je_b_nodes_plot[:,zind_prof]*1E3*1E-4  # in A/cm2
                ne_rprof = ne_plot[:,zind_prof]
                nn_rprof = nn_plot[:,zind_prof]
                phi_rprof = phi_plot[:,zind_prof]
                
                # Store the ion current profile as the reference one
                jiz_rprof_ref = jiz_rprof
                
                mark_type = 'o'
                mark_type = ""
                plt.figure('je_z r prof2')
                plt.semilogy(r_prof,np.abs(-jez_rprof), linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=mark_type, color='k', markeredgecolor = 'k', label="")
#                plt.semilogy(r_prof,-jez_rprof, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=mark_type, color='k', markeredgecolor = 'k', label="")

                plt.figure('ji_z r prof2')
                plt.semilogy(r_prof,jiz_rprof, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=mark_type, color='k', markeredgecolor = 'k', label=labels[k])
        
            elif k == 1:
                z_rprof = 22.9/L_c
                posr1 = np.where(zs[rind,:]<z_rprof)[0][-1]
                posr2 = np.where(zs[rind,:]>z_rprof)[0][0]
                w1 = np.abs(zs[rind,posr2] - z_rprof)/np.abs(zs[rind,posr2] - zs[rind,posr1])
                w2 = np.abs(zs[rind,posr1] - z_rprof)/np.abs(zs[rind,posr2] - zs[rind,posr1])
                r_prof = w1*rs[:,posr1] + w2*rs[:,posr2]
                jiz_rprof = (w1*ji_z_plot[:,posr1] + w2*ji_z_plot[:,posr2])*1E2*1E-4  # in A/cm2
                jez_rprof = (w1*je_z_plot[:,posr1] + w2*je_z_plot[:,posr2])*1E3*1E-4  # in A/cm2
                
                # Modify the value of jez_rprof at the node on the symmetry axis
                jez_rprof[0] = (jez_rprof[2] - jez_rprof[1])/(r_prof[2]-r_prof[1])*(0.0 - r_prof[1]) + jez_rprof[1]
                # Modify the jiz_rprof at the two closest nodes to the symmetry axis using the reference ion profile
                jiz_rprof[0:2] = jiz_rprof_ref[0:2]
                
                print("k = 1 jez_rprof = ",np.abs(-jez_rprof))

                mark_type = 'o'
                mark_type = ""
                plt.figure('je_z r prof2')
                plt.semilogy(r_prof,np.abs(-jez_rprof), linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=mark_type, color='b', markeredgecolor = 'b', label="")
#                plt.semilogy(r_prof,-jez_rprof, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=mark_type, color='b', markeredgecolor = 'b', label="")

                plt.figure('ji_z r prof2')
                plt.semilogy(r_prof,jiz_rprof, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=mark_type, color='b', markeredgecolor = 'b', label=labels[k])
            
            elif k == 2:
                z_rprof_arr = np.array([22.9/L_c,32.9/L_c],dtype=float)
                for iprof in range(0,1):
                    z_rprof = z_rprof_arr[iprof]
                    posr1 = np.where(zs[rind,:]<z_rprof)[0][-1]
                    posr2 = np.where(zs[rind,:]>z_rprof)[0][0]
                    w1 = np.abs(zs[rind,posr2] - z_rprof)/np.abs(zs[rind,posr2] - zs[rind,posr1])
                    w2 = np.abs(zs[rind,posr1] - z_rprof)/np.abs(zs[rind,posr2] - zs[rind,posr1])
                    r_prof = w1*rs[:,posr1] + w2*rs[:,posr2]
                    jiz_rprof = (w1*ji_z_plot[:,posr1] + w2*ji_z_plot[:,posr2])*1E2*1E-4  # in A/cm2
                    jez_rprof = (w1*je_z_plot[:,posr1] + w2*je_z_plot[:,posr2])*1E3*1E-4  # in A/cm2
                    ne_rprof  = w1*ne_plot[:,posr1] + w2*ne_plot[:,posr2]
                    nn_rprof  = w1*nn_plot[:,posr1] + w2*nn_plot[:,posr2]
                    phi_rprof = w1*phi_plot[:,posr1] + w2*phi_plot[:,posr2]
                    if iprof == 0:
                        mark_type = 'o' 
                    else:
                        mark_type= 's'
                        
                    # Modify the value of jez_rprof at the node on the symmetry axis
                    jez_rprof[0] = (jez_rprof[2] - jez_rprof[1])/(r_prof[2]-r_prof[1])*(0.0 - r_prof[1]) + jez_rprof[1]
                    # Modify the jiz_rprof at the two closest nodes to the symmetry axis using the reference ion profile
                    jiz_rprof[0:2] = jiz_rprof_ref[0:2]
                    
                    print("k = 2 jez_rprof = ",np.abs(-jez_rprof))

                    
                    mark_type = ""
                    plt.figure('je_z r prof2')
                    plt.semilogy(r_prof,np.abs(-jez_rprof), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=mark_type, color='r', markeredgecolor = 'r', label="")
#                    plt.semilogy(r_prof,-jez_rprof, linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=mark_type, color='r', markeredgecolor = 'r', label="")
                    
                    plt.figure('ji_z r prof2')
                    plt.semilogy(r_prof,jiz_rprof, linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=mark_type, color='r', markeredgecolor = 'r', label=labels[k])
        
        if plot_r_prof == 1:
            plt.figure('je_z r prof')
#            plt.semilogy(rs[:,-1],-je_z_plot[:,-1]*1E3*1E-4, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color=colors[k], markeredgecolor = 'r', label=labels[k])
            plt.semilogy(rs[:,-1],-je_b_nodes_plot[:,-1]*1E3*1E-4, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color=colors[k], markeredgecolor = 'r', label=labels[k])
            plt.semilogy(rs[:,38],-je_z_plot[:,38]*1E3*1E-4, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='r', markeredgecolor = 'r', label=labels[k])
            plt.semilogy(rs[:,30],-je_z_plot[:,30]*1E3*1E-4, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='b', markeredgecolor = 'r', label=labels[k])


            plt.figure('ji_z r prof')
            plt.semilogy(rs[:,-1],ji_z_plot[:,-1]*1E2*1E-4, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color=colors[k], markeredgecolor = 'r', label=labels[k])
            plt.semilogy(rs[:,38],ji_z_plot[:,38]*1E2*1E-4, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='r', markeredgecolor = 'r', label=labels[k])
            plt.semilogy(rs[:,30],ji_z_plot[:,30]*1E2*1E-4, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='b', markeredgecolor = 'r', label=labels[k])

            plt.figure('ne r prof')
            plt.semilogy(rs[:,-1],ne_plot[:,-1], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color=colors[k], markeredgecolor = 'r', label=labels[k])
            plt.semilogy(rs[:,38],ne_plot[:,38], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='r', markeredgecolor = 'r', label=labels[k])
            plt.semilogy(rs[:,30],ne_plot[:,30], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='b', markeredgecolor = 'r', label=labels[k])

        print("r/Hc je neg = "+str(r_prof[np.where(-jez_rprof < 0)]))
        
        
        ind = ind + 1
        if ind > 6:
            ind = 0
            ind2 = ind2 + 1
            if ind2 > 6:
                ind = 0
                ind2 = 0
                ind3 = ind3 + 1
                
                
    xi_top = int(xi_top)
   
    if plot_r_prof_IEPC22 == 1:    
        plt.figure('je_z r prof1')
        ax = plt.gca()
#        ax.set_xlim(0,xmax)
#        ax.set_xticks(np.arange(0,zs[0,-1]+1,1))
#        ax.set_xlim(0,12)
#        ax.set_ylim(0,300)
        ylims = ax.get_ylim()
        plt.plot(rs[rind,xi_top]*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
        ax.set_ylim(ylims[0],ylims[1])

        plt.figure('ji_z r prof1')
        ax = plt.gca()
        ylims = ax.get_ylim()
        plt.plot(rs[rind,xi_top]*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
        ax.set_ylim(ylims[0],ylims[1])
        
        plt.figure('je_z r prof2')
        ax = plt.gca()
#        ax.set_xlim(0,xmax)
#        ax.set_xticks(np.arange(0,zs[0,-1]+1,1))
#        ax.set_xlim(0,12)
        ax.set_ylim(3E-6,2.1E-1)
        ylims = ax.get_ylim()
        plt.plot(rs[rind,xi_top]*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
        ax.set_ylim(ylims[0],ylims[1])

        plt.figure('ji_z r prof2')
        if labels[0] != '':
            plt.legend(fontsize = font_size_legend,loc=1,frameon = True) 
        ax = plt.gca()
        ax.set_ylim(3E-4,2.1E-1)
        ylims = ax.get_ylim()
        plt.plot(rs[rind,xi_top]*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
        ax.set_ylim(ylims[0],ylims[1])
        
    if plot_r_prof == 1:   
        plt.figure('je_z r prof')
        ax = plt.gca()
        ylims = ax.get_ylim()
        plt.plot(rs[rind,xi_top]*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
        ax.set_ylim(ylims[0],ylims[1])
        
        plt.figure('ji_z r prof')
        ax = plt.gca()
        ylims = ax.get_ylim()
        plt.plot(rs[rind,xi_top]*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
        ax.set_ylim(ylims[0],ylims[1])

        plt.figure('ne r prof')
        ax = plt.gca()
        ylims = ax.get_ylim()
        plt.plot(rs[rind,xi_top]*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
        ax.set_ylim(ylims[0],ylims[1])
        


    if save_flag == 1:
        if plot_r_prof_IEPC22 == 1:    
            plt.figure('je_z r prof1')
            plt.savefig(path_out+"je_z_rprof1"+figs_format,bbox_inches='tight')
            plt.close()
            plt.figure('ji_z r prof1')
            plt.savefig(path_out+"ji_z_rprof1"+figs_format,bbox_inches='tight')
            plt.close()
            
            plt.figure('je_z r prof2')
            plt.savefig(path_out+"je_z_rprof2"+figs_format,bbox_inches='tight')
            plt.close()
            plt.figure('ji_z r prof2')
            plt.savefig(path_out+"ji_z_rprof2"+figs_format,bbox_inches='tight')
            plt.close()
        if plot_r_prof == 1:   
            plt.figure('je_z r prof')
            plt.savefig(path_out+"je_z_rprof"+figs_format,bbox_inches='tight')
            plt.close()
            plt.figure('ji_z r prof')
            plt.savefig(path_out+"ji_z_rprof"+figs_format,bbox_inches='tight')
            plt.close()
            plt.figure('ne r prof')
            plt.savefig(path_out+"ne_rprof"+figs_format,bbox_inches='tight')
            plt.close()

    ###########################################################################
        
