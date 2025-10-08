#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:12:47 2020

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
from matplotlib.ticker import LinearLocator, FormatStrFormatter, AutoMinorLocator, LogFormatter
from scipy import interpolate
from contour_2D import contour_2D
from streamlines_function import streamplot, streamline_2D
from HET_sims_read_bound import HET_sims_read_bound
from HET_sims_mean_bound import HET_sims_mean_bound
from HET_sims_plotvars import HET_sims_cp_vars_bound
from HET_sims_post import max_min_mean_vals
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
bf_plots            = 1

path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/fig4_paper/new_sims/"

if save_flag == 1 and os.path.isdir(path_out) != 1:  
    sys.exit("ERROR: path_out is not an existing directory")


# Set options for LaTeX font
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
font_size           = 25
#font_size_legend    = font_size - 10 - 2
font_size_legend    = font_size - 10 - 5
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
marker_size            = 5
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
    

    
    
if bf_plots == 1:
    print("######## bf_plots ########")
          
    order           = 50
    order_fast      = 500
    
    # Print out time steps
    timestep = 'last'
    
    marker_size         = 7
    marker_every        = 8
    marker_every_mfambf = 5
    
    # marker_every_C1     = 8
    # marker_every_C2     = 6
    # marker_every_C3     = 5
    
    marker_every_C1     = 8+4
    marker_every_C2     = 6+4 
    marker_every_C3     = 5+4
    
    text_size           = 20
    ticks_size          = 20
    font_size           = 20
    font_size_legend    = 18
    font_size_legend    = 15
    font_size_legend    = 12
    
    line_width          = 2.5

  
    allsteps_flag   = 1
    
    mean_vars       = 1
    mean_type       = 0
    # last_steps      = 1000
    last_steps      = 1200
    step_i          = 1
    step_f          = 6
    plot_mean_vars  = 1
    
    # Select the type of plot
    # 0 - Plot magnitudes at the MFAM boundary faces
    # 1 - Plot magnitudes at the PIC mesh boundary nodes or surface elements (if available)
    # 2 - Plot magnitudes at the MFAM boundary faces and at the PIC mesh boundary nodes or surface elements (if available)
    # -------------------------------------------------------------------------
    # NOTE: This only applies to some plots. Up to now (04/06/2020) only to:
    # 1 - Density plots (not available at PIC mesh surface elements)
    # 2 - Current plots
    # -------------------------------------------------------------------------
#    plot_type = 2
    plot_type = 2 # For paper GDML plot_down figures
    
    # Select if plotting variables at the PIC mesh nodes or at the PIC mesh
    # surface elements (except for the picM to picS comparison plots):
    # 1 - Plot variables at PIC mesh nodes
    # 2 - Plot variables at PIC mesh surface elements
#    picvars_plot = 2
    picvars_plot = 2 # For paper GDML plot_down figures
    
    
    # Select if doing plots only inside the chamber for Dwalls (only avalilable
    # for plots at MFAM boundary faces and picS surface elements)
    inC_Dwalls = 1
    
    # Select the minimum ratio je_min/je_max at dielectric walls. 
    # At dielectric walls MFAM boundary faces Values of ion/electron current 
    # with ratio below this value will be set to
    # zero and corresponding electron impact energy values will be set to zero. 
    # This avoids unphysically large electron impact energies in regions with 
    # very low current collected
    # Set the ratio to zero to deactivate the limiter
    min_curr_ratio = 0.00  # Set to zero ion/electron currents lower than 5% the maximum value at dielectric walls
                           # Impact energies are set to zero correspondingly
                           
    # i index of the channel midline
    rind = 15              # VHT_US (IEPC 2022) and paper GDML
    
    print("plot_type    = "+str(plot_type))
    print("picvars_plot = "+str(picvars_plot))
    

    plot_down             = 1
    factor_divide_ene_ion = 10   # Plotted ion energy at P is divided by this factor

    
#    plot_B               = 0
#    plot_dens            = 1
#    plot_deltas          = 1
#    plot_dphi_Te         = 1
#    plot_curr            = 1
#    plot_q               = 1
#    plot_imp_ene         = 1
#    plot_err_interp_mfam = 0
#    plot_err_interp_pic  = 0
#    plot_picM_picS_comp  = 0
    
    plot_B               = 1
    plot_dens            = 1
    plot_deltas          = 0
    plot_dphi_Te         = 1
    plot_curr            = 1
    plot_q               = 1
    plot_imp_ene         = 0
    plot_err_interp_mfam = 0
    plot_err_interp_pic  = 0
    plot_picM_picS_comp  = 0
    
#    plot_B               = 1
#    plot_dens            = 0
#    plot_deltas          = 0
#    plot_dphi_Te         = 0
#    plot_curr            = 0
#    plot_q               = 0
#    plot_imp_ene         = 0
#    plot_err_interp_mfam = 0
#    plot_err_interp_pic  = 0
#    plot_picM_picS_comp  = 0
    
#    plot_B               = 0
#    plot_dens            = 0
#    plot_deltas          = 0
#    plot_dphi_Te         = 0
#    plot_curr            = 1
#    plot_q               = 0
#    plot_imp_ene         = 0
#    plot_err_interp_mfam = 0
#    plot_err_interp_pic  = 0
#    plot_picM_picS_comp  = 0
    
#    plot_B               = 0
#    plot_dens            = 0
#    plot_deltas          = 0
#    plot_dphi_Te         = 1
#    plot_curr            = 0
#    plot_q               = 0
#    plot_imp_ene         = 0
#    plot_err_interp_mfam = 0
#    plot_err_interp_pic  = 0
#    plot_picM_picS_comp  = 0
    

    if timestep == 'last':
        timestep = -1  
    if allsteps_flag == 0:
        mean_vars = 0
        
        
    # Simulation names
    nsims = 6
    oldpost_sim      = np.array([3,5,3,3,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([8,15,8,8,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)         
   
    oldpost_sim      = np.array([6,6,5,5,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([16,17,15,15,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)   
    
    oldpost_sim      = np.array([5,5,5,5,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([15,15,15,15,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)   
    
    oldpost_sim      = np.array([6,6,6,6,6,6,6,6,5,5,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([21,21,21,21,21,21,20,20,17,17,15,15,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)   
    
    
    sim_names = [
        
                    "../../../sim/sims/P3G_Tcath_new",
                    "../../../sim/sims/P3L_Tcath_new",
                    
                    "../../../sim/sims/P3G_fcat1962_Tcath_new",
                    "../../../sim/sims/P3L_fcat1962_Tcath_new",
                    
                    "../../../sim/sims/P3G_fcat6259_5993_Tcath_new",
                    "../../../sim/sims/P3L_fcat6259_5993_Tcath_new",
    
                    # "../../../sim/sims/P3G",
                    # "../../../sim/sims/P3L",
                    
                    # "../../../sim/sims/P3G_fcat1962",
                    # "../../../sim/sims/P3L_fcat1962",
                    
                    # "../../../sim/sims/P3G_fcat6259_5993",
                    # "../../../sim/sims/P3L_fcat6259_5993",
                 
                 ]
    
    sim_names = np.flip(sim_names)

    
    
    topo_case = 3
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
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "PIC_mesh.hdf5",
                              "aspire_picM_rm6.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                             ]
    elif topo_case == 0:    
        PIC_mesh_file_name = [
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
    labels1 = [ 
               r"\phantom{-} $j_\mathrm{niP}$ GP3C1",
               r"$-j_\mathrm{neP}$ GP3C1",
               r"$j_\mathrm{niP}$ LP3C1",
               r"$-j_\mathrm{neP} = j_\mathrm{niP}$ LP3C1",
               
               r"\phantom{-} $j_\mathrm{niP}$ GP3C2",
               r"$-j_\mathrm{neP}$ GP3C2",
               r"$j_\mathrm{niP}$ LP3C2",
               r"$-j_\mathrm{neP} = j_\mathrm{niP}$ LP3C2",
               
               r"\phantom{-} $j_\mathrm{niP}$ GP3C3",
               r"$-j_\mathrm{neP}$ GP3C3",
               r"$j_\mathrm{niP}$ LP3C3",
               r"$-j_\mathrm{neP} = j_\mathrm{niP}$ LP3C3",
               ]                        
    labels2 = [

               r"GP3C1",
               r"LP3C1",
               
               r"GP3C2",
               r"LP3C2",
               
               r"GP3C3",
               r"LP3C3",
        
              ]
    labels3 = [ 
               r"$\mathcal{E}_\mathrm{niP}/10$ GP3C1",
               r"$\mathcal{E}_\mathrm{neP}$ GP3C1",
               r"$\mathcal{E}_\mathrm{niP}/10$ LP3C1",
               r"$\mathcal{E}_\mathrm{neP}$ LP3C1",
               
               r"$\mathcal{E}_\mathrm{niP}/10$ GP3C2",
               r"$\mathcal{E}_\mathrm{neP}$ GP3C2",
               r"$\mathcal{E}_\mathrm{niP}/10$ LP3C2",
               r"$\mathcal{E}_\mathrm{neP}$ LP3C2",
               
               r"$\mathcal{E}_\mathrm{niP}/10$ GP3C3",
               r"$\mathcal{E}_\mathrm{neP}$ GP3C3",
               r"$\mathcal{E}_\mathrm{niP}/10$ LP3C3",
               r"$\mathcal{E}_\mathrm{neP}$ LP3C3",
               ] 
    
    labels1 = np.flip(labels1)
    labels2 = np.flip(labels2)
    labels3 = np.flip(labels3)

    
    # Line colors
    # colors = ['k','r','g','b','m','c','m','y',orange,brown]
    colors = ['k','r','g','b','m','c',orange,silver]
    # colors = ['k','b','r','m','g','c','m','c','m','y',orange,brown] # P1G-P4G, P1L-P4L (paper) cathode cases
#    colors = ['k','m',orange,brown]
    # Different color for different model
    # colors = ['k','r','k','r','k','r']
    colors = ['g','r','g','r','g','r']
    colors = ['black','darkgrey',
              'darkblue','royalblue',
              'darkred','salmon']
    colors1 = ['black','dimgrey','darkgrey','darkgrey',
               'darkblue','blue','royalblue','royalblue',
               'darkred','red','salmon','salmon']
    colors2 = ['black','dimgrey','darkgrey','silver',
               'darkblue','blue','royalblue','lightskyblue',
               'darkred','red','salmon','pink']
    colors = np.flip(colors)
    colors1 = np.flip(colors1)
    colors2 = np.flip(colors2)
    
    
    # Markers
    # markers = ['s','o','v','^','<', '>','D','p','*']
    markers = ['','','s','o','^','v','s','o','v','^','<', '>','D','p','*']
#    markers = ['s','<','D','p']
    # Different marker for different cathode
    markers = ['v','v','s','s','o','o']
    marker_every_vec = [marker_every_C1,marker_every_C1+2,
                        marker_every_C2,marker_every_C2+3,
                        marker_every_C3,marker_every_C3+2]
    marker_every_vec_j = [marker_every_C1-5,marker_every_C1,
                          marker_every_C1-6,marker_every_C1,
                          marker_every_C2-5,marker_every_C2,
                          marker_every_C2-6,marker_every_C2,
                          marker_every_C3-5,marker_every_C3,
                          marker_every_C3-6,marker_every_C3]
    markers = np.flip(markers)
    marker_every_vec = np.flip(marker_every_vec)
    marker_every_vec_j = np.flip(marker_every_vec_j)
    
    # Line style
    linestyles = ['-','--','-.',':','-','--','-.']
#    linestyles = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-']
    # linestyles = ['-','-','-','-',':',':',':',':','-','-','-','-','-','-']
    # Different linestyle for different species
    linestyles = ['-','--','-','--','-','--','-.']

#    xaxis_label = r"$s$ (cm)"
    xaxis_label =  r"$s/L_\mathrm{c}$"
#    xaxis_label_down =  r"$s/L_\mathrm{c}$"
    xaxis_label_down =  r"$s/H_\mathrm{c}$"
    prof_xlabel = xaxis_label_down
    
    # Axial profile plots        
    if plot_down == 1:
        
        # [fig1, axes1] = plt.subplots(nrows=3, ncols=2, figsize=(15,18))
        # [fig1, axes1] = plt.subplots(nrows=3, ncols=2, figsize=(15,9))
        # [fig1, axes1] = plt.subplots(nrows=6, ncols=1, figsize=(7.5,21.5))

        [fig1, axes1] = plt.subplots(nrows=4, ncols=1, figsize=(8,16))  
        xlim = 31
        axes1[0].set_ylabel(r"$j_\mathrm{niP}$, $-j_\mathrm{neP}$ (Acm$^{-2}$)",fontsize = font_size)
        # axes1[0].set_xlabel(prof_xlabel,fontsize = font_size)    
        axes1[0].set_xlim(0,xlim)
        axes1[0].set_xticks(np.arange(0,30+5,5))        
        axes1[0].set_ylim(3E-5,2.5E-1)
        # axes1[0].set_ylim(1E-4,1E0)
        text = '(a)'
        # zstext = 0.48
        # rstext = 0.9
        # zstext = 0.15
        zstext = 0.13
        rstext = 0.9
        axes1[0].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[0].transAxes)  
        axes1[0].tick_params(labelsize = ticks_size,labelbottom=False)
        

        axes1[1].set_ylabel(r"$\phi_\mathrm{P}$ (V)",fontsize = font_size)
        # axes1[1].set_xlabel(prof_xlabel,fontsize = font_size)
        axes1[1].set_xlim(0,xlim)
        axes1[1].set_xticks(np.arange(0,30+5,5))
        axes1[1].set_ylim(-5,40)
        axes1[1].set_yticks(np.arange(-5,40+5,5))
        text = '(b)'
        axes1[1].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[1].transAxes)  
        axes1[1].tick_params(labelsize = ticks_size,labelbottom=False)
        
        
        axes1[2].set_ylabel(r"$T_\mathrm{eP}$ (eV)",fontsize = font_size)
        # axes1[2].set_xlabel(prof_xlabel,fontsize = font_size)
        axes1[2].set_xlim(0,xlim)
        axes1[2].set_xticks(np.arange(0,30+5,5))
        axes1[2].set_ylim(0,14)
        axes1[2].set_yticks(np.arange(0,14+2,2))
        text = '(c)'
        axes1[2].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[2].transAxes)  
        axes1[2].tick_params(labelsize = ticks_size,labelbottom=False)
        
        # Figure of energy in linear y scale
        # axes1[3].set_ylabel(r"$\mathcal{E}_\mathrm{niP}/10$, $\mathcal{E}_\mathrm{neP}$ (eV)",fontsize = font_size)
        axes1[3].set_ylabel(r"(eV)",fontsize = font_size)
        axes1[3].set_xlabel(prof_xlabel,fontsize = font_size)
        axes1[3].set_xlim(0,xlim)
        axes1[3].set_xticks(np.arange(0,30+5,5))
        axes1[3].set_ylim(0,70)
        axes1[3].set_yticks(np.arange(0,70+10,10))
        text = '(d)'
        axes1[3].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[3].transAxes)  
        axes1[3].tick_params(labelsize = ticks_size)
        
        # Figure of energy in semilogy
        # axes1[3].set_ylabel(r"$\mathcal{E}_\mathrm{niP}$, $\mathcal{E}_\mathrm{neP}$ (eV)",fontsize = font_size)
        # axes1[3].set_xlabel(prof_xlabel,fontsize = font_size)
        # axes1[3].set_xlim(0,xlim)
        # axes1[3].set_xticks(np.arange(0,30+5,5))
        # axes1[3].set_ylim(0.5E0,5E2)
        # # axes1[3].set_yticks([1E16,5E16,1E17,3E17])
        # text = '(d)'
        # # zstext = 1.1*3
        # # rstext = 0.9*3E17
        # axes1[3].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[3].transAxes)  
        # axes1[3].tick_params(labelsize = ticks_size)
        
        
        # Original size (8,18)=(cols,rows) ------------------------------------
        # [fig1, axes1] = plt.subplots(nrows=4, ncols=1, figsize=(8,18)) 
        # xlim = 31
        # axes1[0].set_ylabel(r"$j_\mathrm{niP}$, $-j_\mathrm{neP}$ (Acm$^{-2}$)",fontsize = font_size)
        # # axes1[0].set_xlabel(prof_xlabel,fontsize = font_size)    
        # axes1[0].set_xlim(0,xlim)
        # axes1[0].set_xticks(np.arange(0,30+5,5))        
        # axes1[0].set_ylim(1E-5,1E0)
        # # axes1[0].set_ylim(1E-4,1E0)
        # text = '(a)'
        # zstext = 0.48
        # rstext = 0.9
        # axes1[0].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[0].transAxes)  
        # axes1[0].tick_params(labelsize = ticks_size,labelbottom=False)
        

        # axes1[1].set_ylabel(r"$\phi_\mathrm{P}$ (V)",fontsize = font_size)
        # # axes1[1].set_xlabel(prof_xlabel,fontsize = font_size)
        # axes1[1].set_xlim(0,xlim)
        # axes1[1].set_xticks(np.arange(0,30+5,5))
        # axes1[1].set_ylim(0,12)
        # # axes1[1].set_yticks([2,4,6,8,10,12,14])
        # text = '(b)'
        # # zstext = 1.1*3
        # # rstext = 0.9*14
        # axes1[1].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[1].transAxes)  
        # axes1[1].tick_params(labelsize = ticks_size,labelbottom=False)
        
        
        # axes1[2].set_ylabel(r"$T_\mathrm{eP}$ (eV)",fontsize = font_size)
        # # axes1[2].set_xlabel(prof_xlabel,fontsize = font_size)
        # axes1[2].set_xlim(0,xlim)
        # axes1[2].set_xticks(np.arange(0,30+5,5))
        # axes1[2].set_ylim(0,5)
        # text = '(c)'
        # # zstext = 1.1*3
        # # rstext = 0.9*14
        # axes1[2].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[2].transAxes)  
        # axes1[2].tick_params(labelsize = ticks_size,labelbottom=False)
        
        # # Figure of energy in linear y scale
        # axes1[3].set_ylabel(r"$\mathcal{E}_\mathrm{niP}/10$, $\mathcal{E}_\mathrm{neP}$ (eV)",fontsize = font_size)
        # axes1[3].set_xlabel(prof_xlabel,fontsize = font_size)
        # axes1[3].set_xlim(0,xlim)
        # axes1[3].set_xticks(np.arange(0,30+5,5))
        # axes1[3].set_ylim(0,30)
        # # axes1[3].set_yticks([1E16,5E16,1E17,3E17])
        # text = '(d)'
        # # zstext = 1.1*3
        # # rstext = 0.9*3E17
        # axes1[3].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[3].transAxes)  
        # axes1[3].tick_params(labelsize = ticks_size)
        
        # # Figure of energy in semilogy
        # # axes1[3].set_ylabel(r"$\mathcal{E}_\mathrm{niP}$, $\mathcal{E}_\mathrm{neP}$ (eV)",fontsize = font_size)
        # # axes1[3].set_xlabel(prof_xlabel,fontsize = font_size)
        # # axes1[3].set_xlim(0,xlim)
        # # axes1[3].set_xticks(np.arange(0,30+5,5))
        # # axes1[3].set_ylim(0.5E0,5E2)
        # # # axes1[3].set_yticks([1E16,5E16,1E17,3E17])
        # # text = '(d)'
        # # # zstext = 1.1*3
        # # # rstext = 0.9*3E17
        # # axes1[3].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[3].transAxes)  
        # # axes1[3].tick_params(labelsize = ticks_size)
        # ---------------------------------------------------------------------
         
        
    ind  = 0
    ind2 = 0
    ind3 = 0
    for k in range(0,nsims):
        ind_ini_letter = sim_names[k].rfind('/') + 1
        print("##### CASE "+str(k+1)+": "+sim_names[k][ind_ini_letter::]+" #####")
        print("##### last_steps       = "+str(last_steps)+" #####")
        ######################## READ INPUT/OUTPUT FILES ##########################
        # Obtain paths to simulation files
        path_picM         = sim_names[k]+"/SET/inp/"+PIC_mesh_file_name[k]
        path_simstate_inp = sim_names[k]+"/CORE/inp/SimState.hdf5"
        path_simstate_out = sim_names[k]+"/CORE/out/SimState.hdf5"
        path_postdata_out = sim_names[k]+"/CORE/out/PostData.hdf5"
        path_simparams_inp = sim_names[k]+"/CORE/inp/sim_params.inp"
        print("Reading results...")
        [num_ion_spe,num_neu_spe,points,zs,rs,zscells,rscells,dims,nodes_flag,
           cells_flag,cells_vol,volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,eta_max,
           eta_min,xi_top,xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,nsteps,
           nsteps_fast,nsteps_eFld,faces,nodes,elem_n,boundary_f,face_geom,
           elem_geom,n_faces,n_elems,n_faces_boundary,
           nfaces_Dwall_bot,nfaces_Dwall_top,nfaces_Awall,nfaces_FLwall_ver,
           nfaces_FLwall_lat,nfaces_Axis,bIDfaces_Dwall_bot,bIDfaces_Dwall_top,
           bIDfaces_Awall,bIDfaces_FLwall_ver,bIDfaces_FLwall_lat,
           bIDfaces_Axis,IDfaces_Dwall_bot,IDfaces_Dwall_top,
           IDfaces_Awall,IDfaces_FLwall_ver,IDfaces_FLwall_lat,IDfaces_Axis,
           zfaces_Dwall_bot,rfaces_Dwall_bot,zfaces_Dwall_top,rfaces_Dwall_top,
           Afaces_Dwall_bot,Afaces_Dwall_top,zfaces_Awall,rfaces_Awall,
           Afaces_Awall,zfaces_FLwall_ver,rfaces_FLwall_ver,zfaces_FLwall_lat,
           rfaces_FLwall_lat,Afaces_FLwall_ver,Afaces_FLwall_lat,zfaces_Axis,
           rfaces_Axis,Afaces_Axis,sDwall_bot,sDwall_top,sAwall,sFLwall_ver,
           sFLwall_lat,sAxis,sc_bot,sc_top,sc_bot_nodes,sc_top_nodes,sc_bot_surf,
           sc_top_surf,Lplume_bot,Lplume_top,Lchamb_bot,Lchamb_top,Lanode,
           Lfreeloss_ver,Lfreeloss_lat,Lfreeloss,Laxis,
           
           nnodes_Dwall_bot,nnodes_Dwall_top,nnodes_Awall,nnodes_FLwall_ver,
           nnodes_FLwall_lat,nnodes_Axis,nnodes_bound,inodes_Dwall_bot,
           jnodes_Dwall_bot,inodes_Dwall_top,jnodes_Dwall_top,inodes_Awall,
           jnodes_Awall,inodes_FLwall_ver,jnodes_FLwall_ver,inodes_FLwall_lat,
           jnodes_FLwall_lat,inodes_Axis,jnodes_Axis,sDwall_bot_nodes,
           sDwall_top_nodes,sAwall_nodes,sFLwall_ver_nodes,sFLwall_lat_nodes,
           sAxis_nodes,
           
           imp_elems,surf_areas,nsurf_Dwall_bot,nsurf_Dwall_top,nsurf_Awall,
           nsurf_FLwall_ver,nsurf_FLwall_lat,nsurf_bound,indsurf_Dwall_bot,
           zsurf_Dwall_bot,rsurf_Dwall_bot,indsurf_Dwall_top,zsurf_Dwall_top,
           rsurf_Dwall_top,indsurf_Awall,zsurf_Awall,rsurf_Awall,
           indsurf_FLwall_ver,zsurf_FLwall_ver,rsurf_FLwall_ver,
           indsurf_FLwall_lat,zsurf_FLwall_lat,rsurf_FLwall_lat,sDwall_bot_surf,
           sDwall_top_surf,sAwall_surf,sFLwall_ver_surf,sFLwall_lat_surf,
           
           delta_r,delta_s,delta_s_csl,dphi_sh_b,je_b,ji_tot_b,gp_net_b,ge_sb_b,
           relerr_je_b,qe_tot_wall,qe_tot_s_wall,qe_tot_b,qe_b,qe_b_bc,qe_b_fl,
           imp_ene_e_wall,imp_ene_e_b,relerr_qe_b,relerr_qe_b_cons,Te,phi,
           err_interp_phi,err_interp_Te,err_interp_jeperp,err_interp_jetheta,
           err_interp_jepara,err_interp_jez,err_interp_jer,n_inst,ni1_inst,
           ni2_inst,nn1_inst,Bfield,inst_dphi_sh_b_Te,inst_imp_ene_e_b,
           inst_imp_ene_e_b_Te,inst_imp_ene_e_wall,inst_imp_ene_e_wall_Te,
           
           delta_r_nodes,delta_s_nodes,delta_s_csl_nodes,dphi_sh_b_nodes,
           je_b_nodes,gp_net_b_nodes,ge_sb_b_nodes,relerr_je_b_nodes,
           qe_tot_wall_nodes,qe_tot_s_wall_nodes,qe_tot_b_nodes,qe_b_nodes,
           qe_b_bc_nodes,qe_b_fl_nodes,imp_ene_e_wall_nodes,imp_ene_e_b_nodes,
           relerr_qe_b_nodes,relerr_qe_b_cons_nodes,Te_nodes,phi_nodes,
           err_interp_n_nodes,n_inst_nodes,ni1_inst_nodes,ni2_inst_nodes,
           nn1_inst_nodes,n_nodes,ni1_nodes,ni2_nodes,nn1_nodes,dphi_kbc_nodes,
           MkQ1_nodes,ji1_nodes,ji2_nodes,ji3_nodes,ji4_nodes,ji_nodes,
           gn1_tw_nodes,gn1_fw_nodes,gn2_tw_nodes,gn2_fw_nodes,gn3_tw_nodes,
           gn3_fw_nodes,gn_tw_nodes,qi1_tot_wall_nodes,qi2_tot_wall_nodes,
           qi3_tot_wall_nodes,qi4_tot_wall_nodes,qi_tot_wall_nodes,qn1_tw_nodes,
           qn1_fw_nodes,qn2_tw_nodes,qn2_fw_nodes,qn3_tw_nodes,qn3_fw_nodes,
           qn_tot_wall_nodes,imp_ene_i1_nodes,imp_ene_i2_nodes,imp_ene_i3_nodes,
           imp_ene_i4_nodes,imp_ene_ion_nodes,imp_ene_ion_nodes_v2,
           imp_ene_n1_nodes,imp_ene_n2_nodes,imp_ene_n3_nodes,imp_ene_n_nodes,
           imp_ene_n_nodes_v2,Bfield_nodes,inst_dphi_sh_b_Te_nodes,
           inst_imp_ene_e_b_nodes,inst_imp_ene_e_b_Te_nodes,
           inst_imp_ene_e_wall_nodes,inst_imp_ene_e_wall_Te_nodes,
           
           delta_r_surf,delta_s_surf,delta_s_csl_surf,dphi_sh_b_surf,je_b_surf,
           gp_net_b_surf,ge_sb_b_surf,relerr_je_b_surf,qe_tot_wall_surf,
           qe_tot_s_wall_surf,qe_tot_b_surf,qe_b_surf,qe_b_bc_surf,qe_b_fl_surf,
           imp_ene_e_wall_surf,imp_ene_e_b_surf,relerr_qe_b_surf,
           relerr_qe_b_cons_surf,Te_surf,phi_surf,nQ1_inst_surf,nQ1_surf,
           nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,MkQ1_surf,ji1_surf,ji2_surf,
           ji3_surf,ji4_surf,ji_surf,gn1_tw_surf,gn1_fw_surf,gn2_tw_surf,
           gn2_fw_surf,gn3_tw_surf,gn3_fw_surf,gn_tw_surf,qi1_tot_wall_surf,
           qi2_tot_wall_surf,qi3_tot_wall_surf,qi4_tot_wall_surf,qi_tot_wall_surf,
           qn1_tw_surf,qn1_fw_surf,qn2_tw_surf,qn2_fw_surf,qn3_tw_surf,qn3_fw_surf,
           qn_tot_wall_surf,imp_ene_i1_surf,imp_ene_i2_surf,imp_ene_i3_surf,
           imp_ene_i4_surf,imp_ene_ion_surf,imp_ene_ion_surf_v2,
           imp_ene_n1_surf,imp_ene_n2_surf,imp_ene_n3_surf,imp_ene_n_surf,
           imp_ene_n_surf_v2,phi_inf,inst_dphi_sh_b_Te_surf,
           inst_imp_ene_e_b_surf,inst_imp_ene_e_b_Te_surf,
           inst_imp_ene_e_wall_surf,inst_imp_ene_e_wall_Te_surf] = HET_sims_read_bound(path_simstate_inp,path_simstate_out,path_postdata_out,path_simparams_inp,
                                                                                       path_picM,allsteps_flag,timestep,oldpost_sim[k],oldsimparams_sim[k])
            
        
        if mean_vars == 1:        
            print("Averaging variables...")                                                                              
            [delta_r_mean,delta_s_mean,delta_s_csl_mean,dphi_sh_b_mean,je_b_mean,
               ji_tot_b_mean,gp_net_b_mean,ge_sb_b_mean,relerr_je_b_mean,qe_tot_wall_mean,
               qe_tot_s_wall_mean,qe_tot_b_mean,qe_b_mean,qe_b_bc_mean,qe_b_fl_mean,
               imp_ene_e_wall_mean,imp_ene_e_b_mean,
               relerr_qe_b_mean,relerr_qe_b_cons_mean,Te_mean,phi_mean,
               err_interp_phi_mean,err_interp_Te_mean,err_interp_jeperp_mean,
               err_interp_jetheta_mean,err_interp_jepara_mean,err_interp_jez_mean,
               err_interp_jer_mean,n_inst_mean,ni1_inst_mean,ni2_inst_mean,nn1_inst_mean,
               inst_dphi_sh_b_Te_mean,inst_imp_ene_e_b_mean,inst_imp_ene_e_b_Te_mean,
               inst_imp_ene_e_wall_mean,inst_imp_ene_e_wall_Te_mean,
               
               delta_r_nodes_mean,delta_s_nodes_mean,delta_s_csl_nodes_mean,
               dphi_sh_b_nodes_mean,je_b_nodes_mean,gp_net_b_nodes_mean,
               ge_sb_b_nodes_mean,relerr_je_b_nodes_mean,qe_tot_wall_nodes_mean,
               qe_tot_s_wall_nodes_mean,qe_tot_b_nodes_mean,qe_b_nodes_mean,
               qe_b_bc_nodes_mean,qe_b_fl_nodes_mean,relerr_qe_b_nodes_mean,
               imp_ene_e_wall_nodes_mean,imp_ene_e_b_nodes_mean,
               relerr_qe_b_cons_nodes_mean,Te_nodes_mean,phi_nodes_mean,
               err_interp_n_nodes_mean,n_inst_nodes_mean,ni1_inst_nodes_mean,
               ni2_inst_nodes_mean,nn1_inst_nodes_mean,n_nodes_mean,ni1_nodes_mean,
               ni2_nodes_mean,nn1_nodes_mean,dphi_kbc_nodes_mean,MkQ1_nodes_mean,
               ji1_nodes_mean,ji2_nodes_mean,ji3_nodes_mean,ji4_nodes_mean,
               ji_nodes_mean,gn1_tw_nodes_mean,gn1_fw_nodes_mean,gn2_tw_nodes_mean,
               gn2_fw_nodes_mean,gn3_tw_nodes_mean,gn3_fw_nodes_mean,gn_tw_nodes_mean,
               qi1_tot_wall_nodes_mean,qi2_tot_wall_nodes_mean,qi3_tot_wall_nodes_mean,
               qi4_tot_wall_nodes_mean,qi_tot_wall_nodes_mean,qn1_tw_nodes_mean,
               qn1_fw_nodes_mean,qn2_tw_nodes_mean,qn2_fw_nodes_mean,qn3_tw_nodes_mean,
               qn3_fw_nodes_mean,qn_tot_wall_nodes_mean,imp_ene_i1_nodes_mean,
               imp_ene_i2_nodes_mean,imp_ene_i3_nodes_mean,imp_ene_i4_nodes_mean,
               imp_ene_ion_nodes_mean,imp_ene_ion_nodes_v2_mean,
               imp_ene_n1_nodes_mean,imp_ene_n2_nodes_mean,imp_ene_n3_nodes_mean,
               imp_ene_n_nodes_mean,imp_ene_n_nodes_v2_mean,inst_dphi_sh_b_Te_nodes_mean, 
               inst_imp_ene_e_b_nodes_mean,inst_imp_ene_e_b_Te_nodes_mean,
               inst_imp_ene_e_wall_nodes_mean,inst_imp_ene_e_wall_Te_nodes_mean,
               
               delta_r_surf_mean,delta_s_surf_mean,delta_s_csl_surf_mean,
               dphi_sh_b_surf_mean,je_b_surf_mean,gp_net_b_surf_mean,ge_sb_b_surf_mean,
               relerr_je_b_surf_mean,qe_tot_wall_surf_mean,qe_tot_s_wall_surf_mean,
               qe_tot_b_surf_mean,qe_b_surf_mean,qe_b_bc_surf_mean,qe_b_fl_surf_mean,
               imp_ene_e_wall_surf_mean,imp_ene_e_b_surf_mean,
               relerr_qe_b_surf_mean,relerr_qe_b_cons_surf_mean,Te_surf_mean,
               phi_surf_mean,nQ1_inst_surf_mean,nQ1_surf_mean,nQ2_inst_surf_mean,
               nQ2_surf_mean,dphi_kbc_surf_mean,MkQ1_surf_mean,ji1_surf_mean,
               ji2_surf_mean,ji3_surf_mean,ji4_surf_mean,ji_surf_mean,gn1_tw_surf_mean,
               gn1_fw_surf_mean,gn2_tw_surf_mean,gn2_fw_surf_mean,gn3_tw_surf_mean,
               gn3_fw_surf_mean,gn_tw_surf_mean,qi1_tot_wall_surf_mean,
               qi2_tot_wall_surf_mean,qi3_tot_wall_surf_mean,qi4_tot_wall_surf_mean,
               qi_tot_wall_surf_mean,qn1_tw_surf_mean,qn1_fw_surf_mean,qn2_tw_surf_mean,
               qn2_fw_surf_mean,qn3_tw_surf_mean,qn3_fw_surf_mean,qn_tot_wall_surf_mean,
               imp_ene_i1_surf_mean,imp_ene_i2_surf_mean,imp_ene_i3_surf_mean,
               imp_ene_i4_surf_mean,imp_ene_ion_surf_mean,imp_ene_ion_surf_v2_mean,
               imp_ene_n1_surf_mean,imp_ene_n2_surf_mean,imp_ene_n3_surf_mean,
               imp_ene_n_surf_mean,imp_ene_n_surf_v2_mean,inst_dphi_sh_b_Te_surf_mean,
               inst_imp_ene_e_b_surf_mean,inst_imp_ene_e_b_Te_surf_mean,
               inst_imp_ene_e_wall_surf_mean,inst_imp_ene_e_wall_Te_surf_mean] = HET_sims_mean_bound(nsteps,mean_type,last_steps,step_i,step_f,delta_r,
                                                                                                    delta_s,delta_s_csl,dphi_sh_b,je_b,ji_tot_b,gp_net_b,ge_sb_b,
                                                                                                    relerr_je_b,qe_tot_wall,qe_tot_s_wall,qe_tot_b,qe_b,
                                                                                                    qe_b_bc,qe_b_fl,imp_ene_e_wall,imp_ene_e_b,relerr_qe_b,
                                                                                                    relerr_qe_b_cons,Te,phi,err_interp_phi,err_interp_Te,
                                                                                                    err_interp_jeperp,err_interp_jetheta,err_interp_jepara,
                                                                                                    err_interp_jez,err_interp_jer,n_inst,ni1_inst,ni2_inst,nn1_inst,
                                                                                                    inst_dphi_sh_b_Te,inst_imp_ene_e_b,inst_imp_ene_e_b_Te,
                                                                                                    inst_imp_ene_e_wall,inst_imp_ene_e_wall_Te,
                                                                                                    
                                                                                                    delta_r_nodes,delta_s_nodes,delta_s_csl_nodes,dphi_sh_b_nodes,je_b_nodes,
                                                                                                    gp_net_b_nodes,ge_sb_b_nodes,relerr_je_b_nodes,qe_tot_wall_nodes,
                                                                                                    qe_tot_s_wall_nodes,qe_tot_b_nodes,qe_b_nodes,qe_b_bc_nodes,
                                                                                                    qe_b_fl_nodes,imp_ene_e_wall_nodes,imp_ene_e_b_nodes,
                                                                                                    relerr_qe_b_nodes,relerr_qe_b_cons_nodes,Te_nodes,
                                                                                                    phi_nodes,err_interp_n_nodes,n_inst_nodes,ni1_inst_nodes,ni2_inst_nodes,
                                                                                                    nn1_inst_nodes,n_nodes,ni1_nodes,ni2_nodes,nn1_nodes,dphi_kbc_nodes,
                                                                                                    MkQ1_nodes,ji1_nodes,ji2_nodes,ji3_nodes,ji4_nodes,ji_nodes,gn1_tw_nodes,
                                                                                                    gn1_fw_nodes,gn2_tw_nodes,gn2_fw_nodes,gn3_tw_nodes,gn3_fw_nodes,gn_tw_nodes,                        
                                                                                                    qi1_tot_wall_nodes,qi2_tot_wall_nodes,qi3_tot_wall_nodes,
                                                                                                    qi4_tot_wall_nodes,qi_tot_wall_nodes,qn1_tw_nodes,qn1_fw_nodes,
                                                                                                    qn2_tw_nodes,qn2_fw_nodes,qn3_tw_nodes,qn3_fw_nodes,qn_tot_wall_nodes,
                                                                                                    imp_ene_i1_nodes,imp_ene_i2_nodes,imp_ene_i3_nodes,imp_ene_i4_nodes,
                                                                                                    imp_ene_ion_nodes,imp_ene_ion_nodes_v2,imp_ene_n1_nodes,imp_ene_n2_nodes,
                                                                                                    imp_ene_n3_nodes,imp_ene_n_nodes,imp_ene_n_nodes_v2,
                                                                                                    inst_dphi_sh_b_Te_nodes,inst_imp_ene_e_b_nodes,
                                                                                                    inst_imp_ene_e_b_Te_nodes,inst_imp_ene_e_wall_nodes,
                                                                                                    inst_imp_ene_e_wall_Te_nodes,
                                                                                                    
                                                                                                    delta_r_surf,delta_s_surf,delta_s_csl_surf,dphi_sh_b_surf,je_b_surf,gp_net_b_surf,
                                                                                                    ge_sb_b_surf,relerr_je_b_surf,qe_tot_wall_surf,qe_tot_s_wall_surf,
                                                                                                    qe_tot_b_surf,qe_b_surf,qe_b_bc_surf,qe_b_fl_surf,
                                                                                                    imp_ene_e_wall_surf,imp_ene_e_b_surf,relerr_qe_b_surf,
                                                                                                    relerr_qe_b_cons_surf,Te_surf,phi_surf,nQ1_inst_surf,nQ1_surf,
                                                                                                    nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,MkQ1_surf,ji1_surf,ji2_surf,
                                                                                                    ji3_surf,ji4_surf,ji_surf,gn1_tw_surf,gn1_fw_surf,gn2_tw_surf,gn2_fw_surf,
                                                                                                    gn3_tw_surf,gn3_fw_surf,gn_tw_surf,
                                                                                                    qi1_tot_wall_surf,qi2_tot_wall_surf,qi3_tot_wall_surf,qi4_tot_wall_surf,
                                                                                                    qi_tot_wall_surf,qn1_tw_surf,qn1_fw_surf,qn2_tw_surf,qn2_fw_surf,
                                                                                                    qn3_tw_surf,qn3_fw_surf,qn_tot_wall_surf,imp_ene_i1_surf,imp_ene_i2_surf,
                                                                                                    imp_ene_i3_surf,imp_ene_i4_surf,imp_ene_ion_surf,imp_ene_ion_surf_v2,
                                                                                                    imp_ene_n1_surf,imp_ene_n2_surf,imp_ene_n3_surf,imp_ene_n_surf,imp_ene_n_surf_v2,
                                                                                                    inst_dphi_sh_b_Te_surf,inst_imp_ene_e_b_surf,inst_imp_ene_e_b_Te_surf,
                                                                                                    inst_imp_ene_e_wall_surf,inst_imp_ene_e_wall_Te_surf)

            if np.any(np.diff(phi_inf != 0)) and plot_down == 1:
                [_,_,_,_,_,_,
                 mean_min_phi_inf,mean_max_phi_inf,phi_inf_mean,
                 max2mean_phi_inf,min2mean_phi_inf,amp_phi_inf,
                 mins_ind_comp_phi_inf,maxs_ind_comp_phi_inf]    = max_min_mean_vals(time,time[nsteps-last_steps::],phi_inf[nsteps-last_steps::],order)    
            else:
                mean_min_phi_inf      = 0.0 
                mean_max_phi_inf      = 0.0 
                phi_inf_mean          = 0.0
                phi_inf_mean          = phi_inf[-1] 
                max2mean_phi_inf      = 0.0 
                min2mean_phi_inf      = 0.0
                amp_phi_inf           = 0.0  
                mins_ind_comp_phi_inf = 0.0
                maxs_ind_comp_phi_inf = 0.0
                                                                                            
                                                                                            
        print("Obtaining final variables for plotting...") 
        if mean_vars == 1 and plot_mean_vars == 1:
            print("Plotting variables are time-averaged")
            [delta_r_plot,delta_s_plot,delta_s_csl_plot,dphi_sh_b_plot,je_b_plot,
               ji_tot_b_plot,gp_net_b_plot,ge_sb_b_plot,relerr_je_b_plot,qe_tot_wall_plot,
               qe_tot_s_wall_plot,qe_tot_b_plot,qe_b_plot,qe_b_bc_plot,
               qe_b_fl_plot,imp_ene_e_wall_plot,imp_ene_e_b_plot,
               relerr_qe_b_plot,relerr_qe_b_cons_plot,Te_plot,
               phi_plot,err_interp_phi_plot,err_interp_Te_plot,
               err_interp_jeperp_plot,err_interp_jetheta_plot,
               err_interp_jepara_plot,err_interp_jez_plot,err_interp_jer_plot,
               n_inst_plot,ni1_inst_plot,ni2_inst_plot,nn1_inst_plot,
               inst_dphi_sh_b_Te_plot,inst_imp_ene_e_b_plot,inst_imp_ene_e_b_Te_plot,
               inst_imp_ene_e_wall_plot,inst_imp_ene_e_wall_Te_plot,
               
               delta_r_nodes_plot,delta_s_nodes_plot,delta_s_csl_nodes_plot,
               dphi_sh_b_nodes_plot,je_b_nodes_plot,gp_net_b_nodes_plot,
               ge_sb_b_nodes_plot,relerr_je_b_nodes_plot,qe_tot_wall_nodes_plot,
               qe_tot_s_wall_nodes_plot,qe_tot_b_nodes_plot,qe_b_nodes_plot,
               qe_b_bc_nodes_plot,qe_b_fl_nodes_plot,imp_ene_e_wall_nodes_plot,
               imp_ene_e_b_nodes_plot,relerr_qe_b_nodes_plot,
               relerr_qe_b_cons_nodes_plot,Te_nodes_plot,phi_nodes_plot,
               err_interp_n_nodes_plot,n_inst_nodes_plot,ni1_inst_nodes_plot,
               ni2_inst_nodes_plot,nn1_inst_nodes_plot,n_nodes_plot,
               ni1_nodes_plot,ni2_nodes_plot,nn1_nodes_plot,dphi_kbc_nodes_plot,
               MkQ1_nodes_plot,ji1_nodes_plot,ji2_nodes_plot,ji3_nodes_plot,
               ji4_nodes_plot,ji_nodes_plot,gn1_tw_nodes_plot,gn1_fw_nodes_plot,
               gn2_tw_nodes_plot,gn2_fw_nodes_plot,gn3_tw_nodes_plot,
               gn3_fw_nodes_plot,gn_tw_nodes_plot,qi1_tot_wall_nodes_plot,
               qi2_tot_wall_nodes_plot,qi3_tot_wall_nodes_plot,qi4_tot_wall_nodes_plot,
               qi_tot_wall_nodes_plot,qn1_tw_nodes_plot,qn1_fw_nodes_plot,
               qn2_tw_nodes_plot,qn2_fw_nodes_plot,qn3_tw_nodes_plot,
               qn3_fw_nodes_plot,qn_tot_wall_nodes_plot,imp_ene_i1_nodes_plot,
               imp_ene_i2_nodes_plot,imp_ene_i3_nodes_plot,imp_ene_i4_nodes_plot,
               imp_ene_ion_nodes_plot,imp_ene_ion_nodes_v2_plot,
               imp_ene_n1_nodes_plot,imp_ene_n2_nodes_plot,imp_ene_n3_nodes_plot,
               imp_ene_n_nodes_plot,imp_ene_n_nodes_v2_plot,
               inst_dphi_sh_b_Te_nodes_plot,inst_imp_ene_e_b_nodes_plot,
               inst_imp_ene_e_b_Te_nodes_plot,inst_imp_ene_e_wall_nodes_plot,
               inst_imp_ene_e_wall_Te_nodes_plot,
               
               delta_r_surf_plot,delta_s_surf_plot,delta_s_csl_surf_plot,
               dphi_sh_b_surf_plot,je_b_surf_plot,gp_net_b_surf_plot,ge_sb_b_surf_plot,
               relerr_je_b_surf_plot,qe_tot_wall_surf_plot,qe_tot_s_wall_surf_plot,
               qe_tot_b_surf_plot,qe_b_surf_plot,qe_b_bc_surf_plot,qe_b_fl_surf_plot,
               imp_ene_e_wall_surf_plot,imp_ene_e_b_surf_plot,
               relerr_qe_b_surf_plot,relerr_qe_b_cons_surf_plot,Te_surf_plot,
               phi_surf_plot,nQ1_inst_surf_plot,nQ1_surf_plot,nQ2_inst_surf_plot,
               nQ2_surf_plot,dphi_kbc_surf_plot,MkQ1_surf_plot,ji1_surf_plot,
               ji2_surf_plot,ji3_surf_plot,ji4_surf_plot,ji_surf_plot,gn1_tw_surf_plot,
               gn1_fw_surf_plot,gn2_tw_surf_plot,gn2_fw_surf_plot,gn3_tw_surf_plot,
               gn3_fw_surf_plot,gn_tw_surf_plot,qi1_tot_wall_surf_plot,
               qi2_tot_wall_surf_plot,qi3_tot_wall_surf_plot,qi4_tot_wall_surf_plot,
               qi_tot_wall_surf_plot,qn1_tw_surf_plot,qn1_fw_surf_plot,
               qn2_tw_surf_plot,qn2_fw_surf_plot,qn3_tw_surf_plot,qn3_fw_surf_plot,
               qn_tot_wall_surf_plot,imp_ene_i1_surf_plot,imp_ene_i2_surf_plot,
               imp_ene_i3_surf_plot,imp_ene_i4_surf_plot,imp_ene_ion_surf_plot,
               imp_ene_ion_surf_v2_plot,imp_ene_n1_surf_plot,imp_ene_n2_surf_plot,
               imp_ene_n3_surf_plot,imp_ene_n_surf_plot,imp_ene_n_surf_v2_plot,
               inst_dphi_sh_b_Te_surf_plot,inst_imp_ene_e_b_surf_plot,
               inst_imp_ene_e_b_Te_surf_plot,inst_imp_ene_e_wall_surf_plot,
               inst_imp_ene_e_wall_Te_surf_plot] = HET_sims_cp_vars_bound(delta_r_mean,delta_s_mean,delta_s_csl_mean,dphi_sh_b_mean,je_b_mean,ji_tot_b_mean,
                                                                        gp_net_b_mean,ge_sb_b_mean,relerr_je_b_mean,qe_tot_wall_mean,
                                                                        qe_tot_s_wall_mean,qe_tot_b_mean,qe_b_mean,qe_b_bc_mean,qe_b_fl_mean,
                                                                        imp_ene_e_wall_mean,imp_ene_e_b_mean,relerr_qe_b_mean,relerr_qe_b_cons_mean,Te_mean,phi_mean,
                                                                        err_interp_phi_mean,err_interp_Te_mean,err_interp_jeperp_mean,
                                                                        err_interp_jetheta_mean,err_interp_jepara_mean,err_interp_jez_mean,
                                                                        err_interp_jer_mean,n_inst_mean,ni1_inst_mean,ni2_inst_mean,nn1_inst_mean,
                                                                        inst_dphi_sh_b_Te_mean,inst_imp_ene_e_b_mean,inst_imp_ene_e_b_Te_mean,
                                                                        inst_imp_ene_e_wall_mean,inst_imp_ene_e_wall_Te_mean,
                                                                         
                                                                        delta_r_nodes_mean,delta_s_nodes_mean,delta_s_csl_nodes_mean,
                                                                        dphi_sh_b_nodes_mean,je_b_nodes_mean,gp_net_b_nodes_mean,
                                                                        ge_sb_b_nodes_mean,relerr_je_b_nodes_mean,qe_tot_wall_nodes_mean,
                                                                        qe_tot_s_wall_nodes_mean,qe_tot_b_nodes_mean,qe_b_nodes_mean,
                                                                        qe_b_bc_nodes_mean,qe_b_fl_nodes_mean,
                                                                        imp_ene_e_wall_nodes_mean,imp_ene_e_b_nodes_mean,relerr_qe_b_nodes_mean,
                                                                        relerr_qe_b_cons_nodes_mean,Te_nodes_mean,phi_nodes_mean,
                                                                        err_interp_n_nodes_mean,n_inst_nodes_mean,ni1_inst_nodes_mean,
                                                                        ni2_inst_nodes_mean,nn1_inst_nodes_mean,n_nodes_mean,ni1_nodes_mean,
                                                                        ni2_nodes_mean,nn1_nodes_mean,dphi_kbc_nodes_mean,MkQ1_nodes_mean,
                                                                        ji1_nodes_mean,ji2_nodes_mean,ji3_nodes_mean,ji4_nodes_mean,
                                                                        ji_nodes_mean,gn1_tw_nodes_mean,gn1_fw_nodes_mean,gn2_tw_nodes_mean,
                                                                        gn2_fw_nodes_mean,gn3_tw_nodes_mean,gn3_fw_nodes_mean,gn_tw_nodes_mean,
                                                                        qi1_tot_wall_nodes_mean,qi2_tot_wall_nodes_mean,qi3_tot_wall_nodes_mean,
                                                                        qi4_tot_wall_nodes_mean,qi_tot_wall_nodes_mean,qn1_tw_nodes_mean,
                                                                        qn1_fw_nodes_mean,qn2_tw_nodes_mean,qn2_fw_nodes_mean,qn3_tw_nodes_mean,
                                                                        qn3_fw_nodes_mean,qn_tot_wall_nodes_mean,imp_ene_i1_nodes_mean,
                                                                        imp_ene_i2_nodes_mean,imp_ene_i3_nodes_mean,imp_ene_i4_nodes_mean,
                                                                        imp_ene_ion_nodes_mean,imp_ene_ion_nodes_v2_mean,
                                                                        imp_ene_n1_nodes_mean,imp_ene_n2_nodes_mean,imp_ene_n3_nodes_mean,
                                                                        imp_ene_n_nodes_mean,imp_ene_n_nodes_v2_mean,
                                                                        inst_dphi_sh_b_Te_nodes_mean,inst_imp_ene_e_b_nodes_mean,
                                                                        inst_imp_ene_e_b_Te_nodes_mean,inst_imp_ene_e_wall_nodes_mean,
                                                                        inst_imp_ene_e_wall_Te_nodes_mean,
                                                                         
                                                                        delta_r_surf_mean,delta_s_surf_mean,delta_s_csl_surf_mean,
                                                                        dphi_sh_b_surf_mean,je_b_surf_mean,gp_net_b_surf_mean,ge_sb_b_surf_mean,
                                                                        relerr_je_b_surf_mean,qe_tot_wall_surf_mean,qe_tot_s_wall_surf_mean,
                                                                        qe_tot_b_surf_mean,qe_b_surf_mean,qe_b_bc_surf_mean,qe_b_fl_surf_mean,
                                                                        imp_ene_e_wall_surf_mean,imp_ene_e_b_surf_mean,
                                                                        relerr_qe_b_surf_mean,relerr_qe_b_cons_surf_mean,Te_surf_mean,
                                                                        phi_surf_mean,nQ1_inst_surf_mean,nQ1_surf_mean,nQ2_inst_surf_mean,
                                                                        nQ2_surf_mean,dphi_kbc_surf_mean,MkQ1_surf_mean,ji1_surf_mean,
                                                                        ji2_surf_mean,ji3_surf_mean,ji4_surf_mean,ji_surf_mean,gn1_tw_surf_mean,
                                                                        gn1_fw_surf_mean,gn2_tw_surf_mean,gn2_fw_surf_mean,gn3_tw_surf_mean,
                                                                        gn3_fw_surf_mean,gn_tw_surf_mean,qi1_tot_wall_surf_mean,
                                                                        qi2_tot_wall_surf_mean,qi3_tot_wall_surf_mean,qi4_tot_wall_surf_mean,
                                                                        qi_tot_wall_surf_mean,qn1_tw_surf_mean,qn1_fw_surf_mean,qn2_tw_surf_mean,
                                                                        qn2_fw_surf_mean,qn3_tw_surf_mean,qn3_fw_surf_mean,qn_tot_wall_surf_mean,
                                                                        imp_ene_i1_surf_mean,imp_ene_i2_surf_mean,imp_ene_i3_surf_mean,
                                                                        imp_ene_i4_surf_mean,imp_ene_ion_surf_mean,imp_ene_ion_surf_v2_mean,
                                                                        imp_ene_n1_surf_mean,imp_ene_n2_surf_mean,imp_ene_n3_surf_mean,
                                                                        imp_ene_n_surf_mean,imp_ene_n_surf_v2_mean,inst_dphi_sh_b_Te_surf_mean,
                                                                        inst_imp_ene_e_b_surf_mean,inst_imp_ene_e_b_Te_surf_mean,
                                                                        inst_imp_ene_e_wall_surf_mean,inst_imp_ene_e_wall_Te_surf_mean)
            
        else:
            [delta_r_plot,delta_s_plot,delta_s_csl_plot,dphi_sh_b_plot,je_b_plot,
               ji_tot_b_plot,gp_net_b_plot,ge_sb_b_plot,relerr_je_b_plot,qe_tot_wall_plot,
               qe_tot_s_wall_plot,qe_tot_b_plot,qe_b_plot,qe_b_bc_plot,
               qe_b_fl_plot,imp_ene_e_wall_plot,imp_ene_e_b_plot,
               relerr_qe_b_plot,relerr_qe_b_cons_plot,Te_plot,
               phi_plot,err_interp_phi_plot,err_interp_Te_plot,
               err_interp_jeperp_plot,err_interp_jetheta_plot,
               err_interp_jepara_plot,err_interp_jez_plot,err_interp_jer_plot,
               n_inst_plot,ni1_inst_plot,ni2_inst_plot,nn1_inst_plot,
               inst_dphi_sh_b_Te_plot,inst_imp_ene_e_b_plot,inst_imp_ene_e_b_Te_plot,
               inst_imp_ene_e_wall_plot,inst_imp_ene_e_wall_Te_plot,
               
               delta_r_nodes_plot,delta_s_nodes_plot,delta_s_csl_nodes_plot,
               dphi_sh_b_nodes_plot,je_b_nodes_plot,gp_net_b_nodes_plot,
               ge_sb_b_nodes_plot,relerr_je_b_nodes_plot,qe_tot_wall_nodes_plot,
               qe_tot_s_wall_nodes_plot,qe_tot_b_nodes_plot,qe_b_nodes_plot,
               qe_b_bc_nodes_plot,qe_b_fl_nodes_plot,imp_ene_e_wall_nodes_plot,
               imp_ene_e_b_nodes_plot,relerr_qe_b_nodes_plot,
               relerr_qe_b_cons_nodes_plot,Te_nodes_plot,phi_nodes_plot,
               err_interp_n_nodes_plot,n_inst_nodes_plot,ni1_inst_nodes_plot,
               ni2_inst_nodes_plot,nn1_inst_nodes_plot,n_nodes_plot,
               ni1_nodes_plot,ni2_nodes_plot,nn1_nodes_plot,dphi_kbc_nodes_plot,
               MkQ1_nodes_plot,ji1_nodes_plot,ji2_nodes_plot,ji3_nodes_plot,
               ji4_nodes_plot,ji_nodes_plot,gn1_tw_nodes_plot,gn1_fw_nodes_plot,
               gn2_tw_nodes_plot,gn2_fw_nodes_plot,gn3_tw_nodes_plot,
               gn3_fw_nodes_plot,gn_tw_nodes_plot,qi1_tot_wall_nodes_plot,
               qi2_tot_wall_nodes_plot,qi3_tot_wall_nodes_plot,qi4_tot_wall_nodes_plot,
               qi_tot_wall_nodes_plot,qn1_tw_nodes_plot,qn1_fw_nodes_plot,
               qn2_tw_nodes_plot,qn2_fw_nodes_plot,qn3_tw_nodes_plot,
               qn3_fw_nodes_plot,qn_tot_wall_nodes_plot,imp_ene_i1_nodes_plot,
               imp_ene_i2_nodes_plot,imp_ene_i3_nodes_plot,imp_ene_i4_nodes_plot,
               imp_ene_ion_nodes_plot,imp_ene_ion_nodes_v2_plot,
               imp_ene_n1_nodes_plot,imp_ene_n2_nodes_plot,imp_ene_n3_nodes_plot,
               imp_ene_n_nodes_plot,imp_ene_n_nodes_v2_plot,
               inst_dphi_sh_b_Te_nodes_plot,inst_imp_ene_e_b_nodes_plot,
               inst_imp_ene_e_b_Te_nodes_plot,inst_imp_ene_e_wall_nodes_plot,
               inst_imp_ene_e_wall_Te_nodes_plot,
               
               delta_r_surf_plot,delta_s_surf_plot,delta_s_csl_surf_plot,
               dphi_sh_b_surf_plot,je_b_surf_plot,gp_net_b_surf_plot,ge_sb_b_surf_plot,
               relerr_je_b_surf_plot,qe_tot_wall_surf_plot,qe_tot_s_wall_surf_plot,
               qe_tot_b_surf_plot,qe_b_surf_plot,qe_b_bc_surf_plot,qe_b_fl_surf_plot,
               imp_ene_e_wall_surf_plot,imp_ene_e_b_surf_plot,
               relerr_qe_b_surf_plot,relerr_qe_b_cons_surf_plot,Te_surf_plot,
               phi_surf_plot,nQ1_inst_surf_plot,nQ1_surf_plot,nQ2_inst_surf_plot,
               nQ2_surf_plot,dphi_kbc_surf_plot,MkQ1_surf_plot,ji1_surf_plot,
               ji2_surf_plot,ji3_surf_plot,ji4_surf_plot,ji_surf_plot,gn1_tw_surf_plot,
               gn1_fw_surf_plot,gn2_tw_surf_plot,gn2_fw_surf_plot,gn3_tw_surf_plot,
               gn3_fw_surf_plot,gn_tw_surf_plot,qi1_tot_wall_surf_plot,
               qi2_tot_wall_surf_plot,qi3_tot_wall_surf_plot,qi4_tot_wall_surf_plot,
               qi_tot_wall_surf_plot,qn1_tw_surf_plot,qn1_fw_surf_plot,
               qn2_tw_surf_plot,qn2_fw_surf_plot,qn3_tw_surf_plot,qn3_fw_surf_plot,
               qn_tot_wall_surf_plot,imp_ene_i1_surf_plot,imp_ene_i2_surf_plot,
               imp_ene_i3_surf_plot,imp_ene_i4_surf_plot,imp_ene_ion_surf_plot,
               imp_ene_ion_surf_v2_plot,imp_ene_n1_surf_plot,imp_ene_n2_surf_plot,
               imp_ene_n3_surf_plot,imp_ene_n_surf_plot,imp_ene_n_surf_v2_plot,
               inst_dphi_sh_b_Te_surf_plot,inst_imp_ene_e_b_surf_plot,
               inst_imp_ene_e_b_Te_surf_plot,inst_imp_ene_e_wall_surf_plot,
               inst_imp_ene_e_wall_Te_surf_plot] = HET_sims_cp_vars_bound(delta_r,delta_s,delta_s_csl,dphi_sh_b,je_b,ji_tot_b,
                                                                        gp_net_b,ge_sb_b,relerr_je_b,qe_tot_wall,qe_tot_s_wall,
                                                                        qe_tot_b,qe_b,qe_b_bc,qe_b_fl,imp_ene_e_wall,imp_ene_e_b,
                                                                        relerr_qe_b,relerr_qe_b_cons,Te,phi,err_interp_phi,
                                                                        err_interp_Te,err_interp_jeperp,err_interp_jetheta,
                                                                        err_interp_jepara,err_interp_jez,err_interp_jer,
                                                                        n_inst,ni1_inst,ni2_inst,nn1_inst,inst_dphi_sh_b_Te,
                                                                        inst_imp_ene_e_b,inst_imp_ene_e_b_Te,
                                                                        inst_imp_ene_e_wall,inst_imp_ene_e_wall_Te,
                                                                        
                                                                        delta_r_nodes,delta_s_nodes,delta_s_csl_nodes,
                                                                        dphi_sh_b_nodes,je_b_nodes,gp_net_b_nodes,ge_sb_b_nodes,
                                                                        relerr_je_b_nodes,qe_tot_wall_nodes,qe_tot_s_wall_nodes,
                                                                        qe_tot_b_nodes,qe_b_nodes,qe_b_bc_nodes,qe_b_fl_nodes,
                                                                        imp_ene_e_wall_nodes,imp_ene_e_b_nodes,
                                                                        relerr_qe_b_nodes,relerr_qe_b_cons_nodes,Te_nodes,
                                                                        phi_nodes,err_interp_n_nodes,n_inst_nodes,ni1_inst_nodes,
                                                                        ni2_inst_nodes,nn1_inst_nodes,n_nodes,ni1_nodes,ni2_nodes,
                                                                        nn1_nodes,dphi_kbc_nodes,MkQ1_nodes,ji1_nodes,ji2_nodes,
                                                                        ji3_nodes,ji4_nodes,ji_nodes,gn1_tw_nodes,gn1_fw_nodes,
                                                                        gn2_tw_nodes,gn2_fw_nodes,gn3_tw_nodes,gn3_fw_nodes,
                                                                        gn_tw_nodes,qi1_tot_wall_nodes,qi2_tot_wall_nodes,
                                                                        qi3_tot_wall_nodes,qi4_tot_wall_nodes,qi_tot_wall_nodes,
                                                                        qn1_tw_nodes,qn1_fw_nodes,qn2_tw_nodes,qn2_fw_nodes,
                                                                        qn3_tw_nodes,qn3_fw_nodes,qn_tot_wall_nodes,
                                                                        imp_ene_i1_nodes,imp_ene_i2_nodes,imp_ene_i3_nodes,
                                                                        imp_ene_i4_nodes,imp_ene_ion_nodes,imp_ene_ion_nodes_v2,
                                                                        imp_ene_n1_nodes,imp_ene_n2_nodes,imp_ene_n3_nodes,
                                                                        imp_ene_n_nodes,imp_ene_n_nodes_v2,
                                                                        inst_dphi_sh_b_Te_nodes,inst_imp_ene_e_b_nodes,
                                                                        inst_imp_ene_e_b_Te_nodes,inst_imp_ene_e_wall_nodes,
                                                                        inst_imp_ene_e_wall_Te_nodes,
                                                                        
                                                                        delta_r_surf,delta_s_surf,delta_s_csl_surf,dphi_sh_b_surf,
                                                                        je_b_surf,gp_net_b_surf,ge_sb_b_surf,relerr_je_b_surf,
                                                                        qe_tot_wall_surf,qe_tot_s_wall_surf,qe_tot_b_surf,
                                                                        qe_b_surf,qe_b_bc_surf,qe_b_fl_surf,imp_ene_e_wall_surf,
                                                                        imp_ene_e_b_surf,relerr_qe_b_surf,relerr_qe_b_cons_surf,
                                                                        Te_surf,phi_surf,nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,
                                                                        nQ2_surf,dphi_kbc_surf,MkQ1_surf,ji1_surf,ji2_surf,
                                                                        ji3_surf,ji4_surf,ji_surf,gn1_tw_surf,gn1_fw_surf,
                                                                        gn2_tw_surf,gn2_fw_surf,gn3_tw_surf,gn3_fw_surf,
                                                                        gn_tw_surf,qi1_tot_wall_surf,qi2_tot_wall_surf,
                                                                        qi3_tot_wall_surf,qi4_tot_wall_surf,qi_tot_wall_surf,
                                                                        qn1_tw_surf,qn1_fw_surf,qn2_tw_surf,qn2_fw_surf,
                                                                        qn3_tw_surf,qn3_fw_surf,qn_tot_wall_surf,
                                                                        imp_ene_i1_surf,imp_ene_i2_surf,imp_ene_i3_surf,
                                                                        imp_ene_i4_surf,imp_ene_ion_surf,imp_ene_ion_surf_v2,
                                                                        imp_ene_n1_surf,imp_ene_n2_surf,imp_ene_n3_surf,
                                                                        imp_ene_n_surf,imp_ene_n_surf_v2,
                                                                        inst_dphi_sh_b_Te_surf,inst_imp_ene_e_b_surf,
                                                                        inst_imp_ene_e_b_Te_surf,inst_imp_ene_e_wall_surf,
                                                                        inst_imp_ene_e_wall_Te_surf)
        
        
        if oldpost_sim[k] <= 3:
            # Compute the delta_s for old sims (PROVISIONAL)
            delta_s_plot = ge_sb_b_plot/(ge_sb_b_plot + ji_tot_b_plot/e)
        
#        if inC_Dwalls == 1:
#            # Obtain the surface elements in D wall that are inside the chamber
##            indsurf_inC_Dwall_bot = np.where(sDwall_bot_surf <= sc_bot_surf)[0][:] using sc_bot_surf,sc_top_surf we miss more surface elements
##            indsurf_inC_Dwall_top = np.where(sDwall_top_surf <= sc_top_surf)[0][:]
#            indsurf_inC_Dwall_bot = np.where(sDwall_bot_surf <= Lchamb_bot)[0][:] 
#            indsurf_inC_Dwall_top = np.where(sDwall_top_surf <= Lchamb_top)[0][:]
#            
#            # Obtain the ID of MFAM faces that are inside the chamber
#            IDfaces_inC_Dwall_bot = np.where(sDwall_bot <= sc_bot)[0][:]
#            IDfaces_inC_Dwall_top = np.where(sDwall_top <= sc_top)[0][:]
            
            
        indsurf_inC_Dwall_bot = np.where(sDwall_bot_surf <= Lchamb_bot)[0][:] 
        indsurf_inC_Dwall_top = np.where(sDwall_top_surf <= Lchamb_top)[0][:]
        
        # Obtain the ID of MFAM faces that are inside the chamber
        IDfaces_inC_Dwall_bot = np.where(sDwall_bot <= sc_bot)[0][:]
        IDfaces_inC_Dwall_top = np.where(sDwall_top <= sc_top)[0][:]
        
        # Obtain the total plasma current
        j_b_plot       = je_b_plot + ji_tot_b_plot
        j_b_nodes_plot = je_b_nodes_plot + ji_nodes_plot
        j_b_surf_plot  = je_b_surf_plot + ji_surf_plot
        # Obtain the flux of internal electron energy due to convection (advection flux)
        qe_adv_b_plot       = qe_tot_b_plot - qe_b_plot
        qe_adv_b_nodes_plot = qe_tot_b_nodes_plot - qe_b_nodes_plot
        qe_adv_b_surf_plot  = qe_tot_b_surf_plot - qe_b_surf_plot
        
        
        
        # IEPC22 GDML paper: computation of currents through lateral and vertical plume surfaces
        IeP_lat       = np.dot(je_b_plot[bIDfaces_FLwall_lat],Afaces_FLwall_lat)
        IeP_ver       = np.dot(je_b_plot[bIDfaces_FLwall_ver],Afaces_FLwall_ver)
        IiP_lat       = np.dot(ji_tot_b_plot[bIDfaces_FLwall_lat],Afaces_FLwall_lat)
        IiP_ver       = np.dot(ji_tot_b_plot[bIDfaces_FLwall_ver],Afaces_FLwall_ver)
        IiP_lat_picS  = np.dot(ji_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        IiP_ver_picS  = np.dot(ji_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        Ii1P_lat_picS = np.dot(ji1_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        Ii1P_ver_picS = np.dot(ji1_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        Ii2P_lat_picS = np.dot(ji2_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        Ii2P_ver_picS = np.dot(ji2_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        Ii3P_lat_picS = np.dot(ji3_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        Ii3P_ver_picS = np.dot(ji3_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        Ii4P_lat_picS = np.dot(ji4_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        Ii4P_ver_picS = np.dot(ji4_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        IiP           = IiP_lat + IiP_ver
        IeP           = IeP_lat + IeP_ver
        IP            = IiP + IeP
        IP_lat        = IiP_lat + IeP_lat
        IP_ver        = IiP_ver + IeP_ver
        IiP_picS      = IiP_lat_picS + IiP_ver_picS
        IP_picS       = IiP_picS + IeP
        IP_lat_picS   = IiP_lat_picS + IeP_lat
        IP_ver_picS   = IiP_ver_picS + IeP_ver
                
        PeP_lat       = np.dot(qe_tot_wall_plot[bIDfaces_FLwall_lat],Afaces_FLwall_lat)
        PeP_ver       = np.dot(qe_tot_wall_plot[bIDfaces_FLwall_ver],Afaces_FLwall_ver)
        PiP_lat       = np.dot(qi_tot_wall_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        PiP_ver       = np.dot(qi_tot_wall_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        PnP_lat       = np.dot(qn_tot_wall_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        PnP_ver       = np.dot(qn_tot_wall_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        Pi1P_lat      = np.dot(qi1_tot_wall_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        Pi1P_ver      = np.dot(qi1_tot_wall_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        Pi2P_lat      = np.dot(qi2_tot_wall_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        Pi2P_ver      = np.dot(qi2_tot_wall_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        Pi3P_lat      = np.dot(qi3_tot_wall_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        Pi3P_ver      = np.dot(qi3_tot_wall_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        Pi4P_lat      = np.dot(qi4_tot_wall_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        Pi4P_ver      = np.dot(qi4_tot_wall_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        Pn1P_lat      = np.dot(qn1_tw_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        Pn1P_ver      = np.dot(qn1_tw_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        Pn2P_lat      = np.dot(qn2_tw_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        Pn2P_ver      = np.dot(qn2_tw_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        Pn3P_lat      = np.dot(qn3_tw_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        Pn3P_ver      = np.dot(qn3_tw_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        PeP           = PeP_lat + PeP_ver
        PiP           = PiP_lat + PiP_ver
        PnP           = PnP_lat + PnP_ver
        PP            = PeP + PiP + PnP
        
        print("Currents at P boundary")
        print("IP                          (A) = %15.8e " %IP)
        print("IP_picS                     (A) = %15.8e " %IP_picS)
        print("IeP                         (A) = %15.8e " %IeP)
        print("IiP                         (A) = %15.8e " %IiP)
        print("IiP_picS                    (A) = %15.8e " %IiP_picS)
        print("IiP_lat/IiP                (%%) = %15.8e " %(100.0*IiP_lat/IiP))
        print("IiP_ver/IiP                (%%) = %15.8e " %(100.0*IiP_ver/IiP))
        print("IeP_lat/IeP                (%%) = %15.8e " %(100.0*IeP_lat/IeP))
        print("IeP_ver/IeP                (%%) = %15.8e " %(100.0*IeP_ver/IeP))
        print("IiP_lat_picS/IiP_picS      (%%) = %15.8e " %(100.0*IiP_lat_picS/IiP_picS))
        print("IiP_ver_picS/IiP_picS      (%%) = %15.8e " %(100.0*IiP_ver_picS/IiP_picS))
        print("Contributions of ion species at lateral P boundary using IiP_lat_picS")
        print("Ii1P_lat_picS/IiP_lat_picS (%%) = %15.8e " %(100.0*Ii1P_lat_picS/IiP_lat_picS))
        print("Ii2P_lat_picS/IiP_lat_picS (%%) = %15.8e " %(100.0*Ii2P_lat_picS/IiP_lat_picS))
        print("Ii3P_lat_picS/IiP_lat_picS (%%) = %15.8e " %(100.0*Ii3P_lat_picS/IiP_lat_picS))
        print("Ii4P_lat_picS/IiP_lat_picS (%%) = %15.8e " %(100.0*Ii4P_lat_picS/IiP_lat_picS))
        print("sum                        (%%) = %15.8e " %(100.0*(Ii1P_lat_picS+Ii2P_lat_picS+Ii3P_lat_picS+Ii4P_lat_picS)/IiP_lat_picS))
        print("Contributions of ion species at lateral P boundary using IiP_lat")
        print("Ii1P_lat_picS/IiP_lat      (%%) = %15.8e " %(100.0*Ii1P_lat_picS/IiP_lat))
        print("Ii2P_lat_picS/IiP_lat      (%%) = %15.8e " %(100.0*Ii2P_lat_picS/IiP_lat))
        print("Ii3P_lat_picS/IiP_lat      (%%) = %15.8e " %(100.0*Ii3P_lat_picS/IiP_lat))
        print("Ii4P_lat_picS/IiP_lat      (%%) = %15.8e " %(100.0*Ii4P_lat_picS/IiP_lat))
        print("sum                        (%%) = %15.8e " %(100.0*(Ii1P_lat_picS+Ii2P_lat_picS+Ii3P_lat_picS+Ii4P_lat_picS)/IiP_lat))
        
        print("Powers at P boundary")
        print("PP                          (W) = %15.8e " %PP)
        print("PeP                         (W) = %15.8e " %PeP)
        print("PiP                         (W) = %15.8e " %PiP)
        print("PnP                         (W) = %15.8e " %PnP)
        print("PeP/PP                     (%%) = %15.8e " %(100.0*PeP/PP))
        print("PiP/PP                     (%%) = %15.8e " %(100.0*PiP/PP))
        print("PnP/PP                     (%%) = %15.8e " %(100.0*PnP/PP))
        print("PeP_lat/PeP                (%%) = %15.8e " %(100.0*PeP_lat/PeP))
        print("PeP_ver/PeP                (%%) = %15.8e " %(100.0*PeP_ver/PeP))
        print("PiP_lat/PiP                (%%) = %15.8e " %(100.0*PiP_lat/PiP))
        print("PiP_ver/PiP                (%%) = %15.8e " %(100.0*PiP_ver/PiP))
        print("PnP_lat/PnP                (%%) = %15.8e " %(100.0*PnP_lat/PnP))
        print("PnP_ver/PnP                (%%) = %15.8e " %(100.0*PnP_ver/PnP))
        print("Contributions of ion species at lateral P boundary")
        print("Pi1P_lat/PiP_lat           (%%) = %15.8e " %(100.0*Pi1P_lat/PiP_lat))
        print("Pi2P_lat/PiP_lat           (%%) = %15.8e " %(100.0*Pi2P_lat/PiP_lat))
        print("Pi3P_lat/PiP_lat           (%%) = %15.8e " %(100.0*Pi3P_lat/PiP_lat))
        print("Pi4P_lat/PiP_lat           (%%) = %15.8e " %(100.0*Pi4P_lat/PiP_lat))
        print("sum                        (%%) = %15.8e " %(100.0*(Pi1P_lat+Pi2P_lat+Pi3P_lat+Pi4P_lat)/PiP_lat))
        print("Contributions of neutral species at lateral P boundary")
        print("Pn1P_lat/PnP_lat           (%%) = %15.8e " %(100.0*Pn1P_lat/PnP_lat))
        print("Pn2P_lat/PnP_lat           (%%) = %15.8e " %(100.0*Pn2P_lat/PnP_lat))
        print("Pn3P_lat/PnP_lat           (%%) = %15.8e " %(100.0*Pn3P_lat/PnP_lat))
        print("sum                        (%%) = %15.8e " %(100.0*(Pn1P_lat+Pn2P_lat+Pn3P_lat)/PnP_lat))
        
        # Currents in A/cm2
        je_b_plot       = je_b_plot*1E-4
        je_b_nodes_plot = je_b_nodes_plot*1E-4
        je_b_surf_plot  = je_b_surf_plot*1E-4
        ji_tot_b_plot   = ji_tot_b_plot*1E-4
        ji_nodes_plot   = ji_nodes_plot*1E-4
        ji1_nodes_plot  = ji1_nodes_plot*1E-4
        ji2_nodes_plot  = ji2_nodes_plot*1E-4
        ji3_nodes_plot  = ji3_nodes_plot*1E-4
        ji4_nodes_plot  = ji4_nodes_plot*1E-4
        ji_surf_plot    = ji_surf_plot*1E-4
        ji1_surf_plot   = ji1_surf_plot*1E-4
        ji2_surf_plot   = ji2_surf_plot*1E-4
        ji3_surf_plot   = ji3_surf_plot*1E-4
        ji4_surf_plot   = ji4_surf_plot*1E-4
        j_b_plot        = j_b_plot*1E-4
        j_b_nodes_plot  = j_b_nodes_plot*1E-4
        j_b_surf_plot   = j_b_surf_plot*1E-4
        # Equivalent current collected for neutrals in A/cm2
        jn1_nodes_plot  = gn1_tw_nodes_plot*e*1E-4
        jn2_nodes_plot  = gn2_tw_nodes_plot*e*1E-4
        jn3_nodes_plot  = gn3_tw_nodes_plot*e*1E-4
        jn_nodes_plot   = gn_tw_nodes_plot*e*1E-4
        jn1_surf_plot   = gn1_tw_surf_plot*e*1E-4
        jn2_surf_plot   = gn2_tw_surf_plot*e*1E-4
        jn3_surf_plot   = gn3_tw_surf_plot*e*1E-4
        jn_surf_plot    = gn_tw_surf_plot*e*1E-4
        
        # Energy fluxes in W/cm2
        qe_tot_wall_plot         = qe_tot_wall_plot*1E-4
        qe_tot_s_wall_plot       = qe_tot_s_wall_plot*1E-4
        qe_tot_b_plot            = qe_tot_b_plot*1E-4
        qe_b_plot                = qe_b_plot*1E-4
        qe_adv_b_plot            = qe_adv_b_plot*1E-4
        qe_tot_s_wall_nodes_plot = qe_tot_s_wall_nodes_plot*1E-4
        qe_tot_b_nodes_plot      = qe_tot_b_nodes_plot*1E-4
        qe_b_nodes_plot          = qe_b_nodes_plot*1E-4
        qe_adv_b_nodes_plot      = qe_adv_b_nodes_plot*1E-4
        qe_tot_wall_surf_plot    = qe_tot_wall_surf_plot*1E-4
        qe_tot_s_wall_surf_plot  = qe_tot_s_wall_surf_plot*1E-4
        qe_tot_b_surf_plot       = qe_tot_b_surf_plot*1E-4
        qe_b_surf_plot           = qe_b_surf_plot*1E-4
        qe_adv_b_surf_plot       = qe_adv_b_surf_plot*1E-4
        qi1_tot_wall_nodes_plot  = qi1_tot_wall_nodes_plot*1E-4
        qi2_tot_wall_nodes_plot  = qi2_tot_wall_nodes_plot*1E-4
        qi3_tot_wall_nodes_plot  = qi3_tot_wall_nodes_plot*1E-4
        qi4_tot_wall_nodes_plot  = qi4_tot_wall_nodes_plot*1E-4
        qi_tot_wall_nodes_plot   = qi_tot_wall_nodes_plot*1E-4
        qi1_tot_wall_surf_plot   = qi1_tot_wall_surf_plot*1E-4
        qi2_tot_wall_surf_plot   = qi2_tot_wall_surf_plot*1E-4
        qi3_tot_wall_surf_plot   = qi3_tot_wall_surf_plot*1E-4
        qi4_tot_wall_surf_plot   = qi4_tot_wall_surf_plot*1E-4
        qi_tot_wall_surf_plot    = qi_tot_wall_surf_plot*1E-4
        qn1_tw_nodes_plot        = qn1_tw_nodes_plot*1E-4
        qn2_tw_nodes_plot        = qn2_tw_nodes_plot*1E-4
        qn3_tw_nodes_plot        = qn3_tw_nodes_plot*1E-4
        qn_tot_wall_nodes_plot   = qn_tot_wall_nodes_plot*1E-4
        qn1_tw_surf_plot         = qn1_tw_surf_plot*1E-4
        qn2_tw_surf_plot         = qn2_tw_surf_plot*1E-4
        qn3_tw_surf_plot         = qn3_tw_surf_plot*1E-4
        qn_tot_wall_surf_plot    = qn_tot_wall_surf_plot*1E-4
        
        
        # Bfield in G
        Bfield = Bfield*1E4
        Bfield_nodes = Bfield_nodes*1E4
        
        # Arc lengths in cm
        sc_bot      = sc_bot*1E2
        sc_top      = sc_top*1E2
        sDwall_bot  = sDwall_bot*1E2
        sDwall_top  = sDwall_top*1E2
        sAwall      = sAwall*1E2
        sFLwall_ver = sFLwall_ver*1E2
        sFLwall_lat = sFLwall_lat*1E2
        sAxis       = sAxis*1E2
        sDwall_bot_nodes  = sDwall_bot_nodes*1E2
        sDwall_top_nodes  = sDwall_top_nodes*1E2
        sAwall_nodes      = sAwall_nodes*1E2
        sFLwall_ver_nodes = sFLwall_ver_nodes*1E2
        sFLwall_lat_nodes = sFLwall_lat_nodes*1E2
        sAxis_nodes       = sAxis_nodes*1E2
        sDwall_bot_surf   = sDwall_bot_surf*1E2
        sDwall_top_surf   = sDwall_top_surf*1E2
        sAwall_surf       = sAwall_surf*1E2
        sFLwall_ver_surf  = sFLwall_ver_surf*1E2
        sFLwall_lat_surf  = sFLwall_lat_surf*1E2
        Lchamb_bot        = Lchamb_bot*1E2
        Lchamb_top        = Lchamb_top*1E2
        Lplume_bot        = Lplume_bot*1E2
        Lplume_top        = Lplume_top*1E2
        Lanode            = Lanode*1E2
        Lfreeloss_ver     = Lfreeloss_ver*1E2
        Lfreeloss_lat     = Lfreeloss_lat*1E2
        Lfreeloss         = Lfreeloss*1E2
        Laxis             = Laxis*1E2     
        rs                = rs*1E2
        zs                = zs*1E2
        
        if inC_Dwalls == 1:
            print("delta_s_max = "+str(0.5*(delta_s_plot[bIDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]].max() + delta_s_plot[bIDfaces_Dwall_top[IDfaces_inC_Dwall_top]].max())))
        elif inC_Dwalls == 0:
            print("delta_s_max = "+str(0.5*(delta_s_plot[bIDfaces_Dwall_bot].max() + delta_s_plot[bIDfaces_Dwall_top].max())))


        # Obtain electron and ion power losses [W] to dielectric bottom and top walls (complete and inside the chamber)
        PeDwall_bot_inC = np.dot(qe_tot_wall_plot[bIDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]]*1E4,Afaces_Dwall_bot[IDfaces_inC_Dwall_bot])
        PeDwall_bot     = np.dot(qe_tot_wall_plot[bIDfaces_Dwall_bot]*1E4,Afaces_Dwall_bot)
        PeDwall_top_inC = np.dot(qe_tot_wall_plot[bIDfaces_Dwall_top[IDfaces_inC_Dwall_top]]*1E4,Afaces_Dwall_top[IDfaces_inC_Dwall_top])
        PeDwall_top     = np.dot(qe_tot_wall_plot[bIDfaces_Dwall_top]*1E4,Afaces_Dwall_top)
        
        PiDwall_bot_inC = np.dot(qi_tot_wall_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]]*1E4,surf_areas[imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]][:,0],imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]][:,1]])
        PiDwall_bot     = np.dot(qi_tot_wall_surf_plot[indsurf_Dwall_bot]*1E4,surf_areas[imp_elems[indsurf_Dwall_bot][:,0],imp_elems[indsurf_Dwall_bot][:,1]])
        PiDwall_top_inC = np.dot(qi_tot_wall_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]]*1E4,surf_areas[imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top]][:,0],imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top]][:,1]])
        PiDwall_top     = np.dot(qi_tot_wall_surf_plot[indsurf_Dwall_top]*1E4,surf_areas[imp_elems[indsurf_Dwall_top][:,0],imp_elems[indsurf_Dwall_top][:,1]])
        
        PDwall_bot_inC = PeDwall_bot_inC + PiDwall_bot_inC
        PDwall_bot     = PeDwall_bot + PiDwall_bot
        PDwall_top_inC = PeDwall_top_inC + PiDwall_top_inC
        PDwall_top     = PeDwall_top + PiDwall_top
        
        print("PeDwall_bot_inC (W) = "+str(PeDwall_bot_inC))
        print("PeDwall_top_inC (W) = "+str(PeDwall_top_inC))
        print("PiDwall_bot_inC (W) = "+str(PiDwall_bot_inC))
        print("PiDwall_top_inC (W) = "+str(PiDwall_top_inC))
        print("PDwall_bot_inC  (W) = "+str(PDwall_bot_inC))
        print("PDwall_top_inC  (W) = "+str(PDwall_top_inC))
        
        
#        print("PeDwall_bot     (W) = "+str(PeDwall_bot))
#        print("PeDwall_top     (W) = "+str(PeDwall_top))
#        print("PiDwall_bot     (W) = "+str(PiDwall_bot))
#        print("PiDwall_top     (W) = "+str(PiDwall_top))
#        print("PDwall_bot      (W) = "+str(PDwall_bot))
#        print("PDwall_top      (W) = "+str(PDwall_top))

        
        
        # Obtain arc length variables and indeces for plots along complete boundary
        Lbot = Lchamb_bot + Lplume_bot
        Ltop = Lchamb_top + Lplume_top
        # s along the complete wall
        swall_1 = Lbot - np.flip(sDwall_bot)
#        swall_2 = Lanode - np.flip(sAwall)
        swall_2 = sAwall
#        swall_3 = Ltop - np.flip(sDwall_top)
        swall_3 = sDwall_top
        swall_2_bis = swall_2 + Lbot
        swall_3_bis = swall_3 + Lbot + Lanode
        swall = np.concatenate((swall_1,swall_2_bis,swall_3_bis),axis=0)
        bIDfaces_wall = np.concatenate((np.flip(bIDfaces_Dwall_bot),bIDfaces_Awall,bIDfaces_Dwall_top),axis=0)
        IDfaces_wall  = np.concatenate((np.flip(IDfaces_Dwall_bot),IDfaces_Awall,IDfaces_Dwall_top),axis=0)
        # s along only chamber wall only
        swall_1_inC = Lchamb_bot - np.flip(sDwall_bot[IDfaces_inC_Dwall_bot])
#        swall_3_inC = Lchamb_top - np.flip(sDwall_top[IDfaces_inC_Dwall_top])
        swall_3_inC = sDwall_top[IDfaces_inC_Dwall_top]
        swall_2_bis_inC = swall_2 + Lchamb_bot
        swall_3_bis_inC = swall_3_inC + Lchamb_bot + Lanode
        swall_inC = np.concatenate((swall_1_inC,swall_2_bis_inC,swall_3_bis_inC),axis=0)
        bIDfaces_wall_inC = np.concatenate((np.flip(bIDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]),bIDfaces_Awall,bIDfaces_Dwall_top[IDfaces_inC_Dwall_top]),axis=0)
        IDfaces_wall_inC  = np.concatenate((np.flip(IDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]),IDfaces_Awall,IDfaces_Dwall_top[IDfaces_inC_Dwall_top]),axis=0)
        
        # ssurf along the complete wall
        swall_1_surf = Lbot - np.flip(sDwall_bot_surf)
#        swall_2_surf = Lanode - np.flip(sAwall_surf)
        swall_2_surf = sAwall_surf
#        swall_3_surf = Ltop - np.flip(sDwall_top_surf)
        swall_3_surf = sDwall_top_surf
        swall_2_bis_surf = swall_2_surf + Lbot
        swall_3_bis_surf = swall_3_surf + Lbot + Lanode
        swall_surf = np.concatenate((swall_1_surf,swall_2_bis_surf,swall_3_bis_surf),axis=0)
        indsurf_wall = np.concatenate((np.flip(indsurf_Dwall_bot),indsurf_Awall,indsurf_Dwall_top),axis=0)
        # ssurf along only chamber wall only
        swall_1_inC_surf = Lchamb_bot - np.flip(sDwall_bot_surf[indsurf_inC_Dwall_bot])
#        swall_3_inC_surf = Lchamb_top - np.flip(sDwall_top_surf[indsurf_inC_Dwall_top])
        swall_3_inC_surf = sDwall_top_surf[indsurf_inC_Dwall_top]
        swall_2_bis_inC_surf = swall_2_surf + Lchamb_bot
        swall_3_bis_inC_surf = swall_3_inC_surf + Lchamb_bot + Lanode
        swall_inC_surf = np.concatenate((swall_1_inC_surf,swall_2_bis_inC_surf,swall_3_bis_inC_surf),axis=0)
        indsurf_inC_wall = np.concatenate((np.flip(indsurf_Dwall_bot[indsurf_inC_Dwall_bot]),indsurf_Awall,indsurf_Dwall_top[indsurf_inC_Dwall_top]),axis=0)
        
        # s along the complete freeloss, starting from lateral wall
#        sdown_1 = sFLwall_lat
#        sdown_2 = Lfreeloss_ver - np.flip(sFLwall_ver)
#        sdown_2_bis = sdown_2 + Lfreeloss_lat
#        sdown = np.concatenate((sdown_1,sdown_2_bis),axis=0)
#        bIDfaces_down = np.concatenate((bIDfaces_FLwall_lat,np.flip(bIDfaces_FLwall_ver)),axis=0)
#        IDfaces_down  = np.concatenate((IDfaces_FLwall_lat,np.flip(IDfaces_FLwall_ver)),axis=0)
#        # ssurf along the complete freeloss, starting from lateral wall
#        sdown_1_surf = sFLwall_lat_surf
#        sdown_2_surf = Lfreeloss_ver - np.flip(sFLwall_ver_surf)
#        sdown_2_bis_surf = sdown_2_surf + Lfreeloss_lat
#        sdown_surf = np.concatenate((sdown_1_surf,sdown_2_bis_surf),axis=0)
#        indsurf_down = np.concatenate((indsurf_FLwall_lat,np.flip(indsurf_FLwall_ver)),axis=0)
#        # snodes along the complete freeloss, starting from lateral wall
#        sdown_1_nodes = sFLwall_lat_nodes
#        sdown_2_nodes = Lfreeloss_ver - np.flip(sFLwall_ver_nodes)
#        sdown_2_bis_nodes = sdown_2_nodes + Lfreeloss_lat
#        sdown_nodes = np.concatenate((sdown_1_nodes,sdown_2_bis_nodes),axis=0)
#        inodes_down = np.concatenate((inodes_FLwall_lat,np.flip(inodes_FLwall_ver)),axis=0)
#        jnodes_down = np.concatenate((jnodes_FLwall_lat,np.flip(jnodes_FLwall_ver)),axis=0)
        
        # s along the complete freeloss, starting from vertical wall
        sdown_1 = sFLwall_ver
        sdown_2 = Lfreeloss_lat - np.flip(sFLwall_lat)
        sdown_2_bis = sdown_2 + Lfreeloss_ver
        sdown = np.concatenate((sdown_1,sdown_2_bis),axis=0)
        bIDfaces_down = np.concatenate((bIDfaces_FLwall_ver,np.flip(bIDfaces_FLwall_lat)),axis=0)
        IDfaces_down  = np.concatenate((IDfaces_FLwall_ver,np.flip(IDfaces_FLwall_lat)),axis=0)
        # ssurf along the complete freeloss, starting from vertical wall
        sdown_1_surf = sFLwall_ver_surf
        sdown_2_surf = Lfreeloss_lat - np.flip(sFLwall_lat_surf)
        sdown_2_bis_surf = sdown_2_surf + Lfreeloss_ver
        sdown_surf = np.concatenate((sdown_1_surf,sdown_2_bis_surf),axis=0)
        indsurf_down = np.concatenate((indsurf_FLwall_ver,np.flip(indsurf_FLwall_lat)),axis=0)
        # snodes along the complete freeloss, starting from vertical wall
        sdown_1_nodes = sFLwall_ver_nodes
        sdown_2_nodes = Lfreeloss_lat - np.flip(sFLwall_lat_nodes)
        sdown_2_bis_nodes = sdown_2_nodes + Lfreeloss_ver
        sdown_nodes = np.concatenate((sdown_1_nodes,sdown_2_bis_nodes),axis=0)
        inodes_down = np.concatenate((inodes_FLwall_ver,np.flip(inodes_FLwall_lat)),axis=0)
        jnodes_down = np.concatenate((jnodes_FLwall_ver,np.flip(jnodes_FLwall_lat)),axis=0)
        
        
        # Since we have BM oscillations, time average of non-linear operations is not equal to the non-linear operation computed 
        # with time-averaged variables.
        # As for efficiencies, since they are defined after time-averaging current and/or power balances, it makes sense to compute them
        # with time average quantities. 
        # However, for wall-impact energies (ene = P/g), it is more correct to do the time average of the instantaneous impact energies.
        # These values have been already computed in HET_sims_mean_bound.
        # We shall compute here wall-impact energies using time-averaged quantities:
        avg_imp_ene_i1_nodes_plot  = (qi1_tot_wall_nodes_plot/(ji1_nodes_plot/e))/e
        avg_imp_ene_i2_nodes_plot  = (qi2_tot_wall_nodes_plot/(ji2_nodes_plot/(2*e)))/e
        if num_ion_spe == 4:
            avg_imp_ene_i3_nodes_plot  = (qi3_tot_wall_nodes_plot/(ji3_nodes_plot/e))/e
            avg_imp_ene_i4_nodes_plot  = (qi4_tot_wall_nodes_plot/(ji4_nodes_plot/(2*e)))/e
            den = ji1_nodes_plot/e + ji2_nodes_plot/(2*e) + ji3_nodes_plot/e + ji4_nodes_plot/(2*e)
            avg_imp_ene_ion_nodes_plot  = avg_imp_ene_i1_nodes_plot*(ji1_nodes_plot/e/den)     + \
                                          avg_imp_ene_i2_nodes_plot*(ji2_nodes_plot/(2*e)/den) + \
                                          avg_imp_ene_i3_nodes_plot*(ji3_nodes_plot/e/den)     + \
                                          avg_imp_ene_i4_nodes_plot*(ji4_nodes_plot/(2*e)/den)
        elif num_ion_spe == 2:
            den = ji1_nodes_plot/e + ji2_nodes_plot/(2*e)
            avg_imp_ene_ion_nodes_plot  = avg_imp_ene_i1_nodes_plot*(ji1_nodes_plot/e/den) + avg_imp_ene_i2_nodes_plot*(ji2_nodes_plot/(2*e)/den)
        avg_imp_ene_ion_nodes_v2_plot = qi_tot_wall_nodes_plot/(ji_nodes_plot/e)/e
        
        avg_imp_ene_n1_nodes_plot  = qn1_tw_nodes_plot/(jn1_nodes_plot/e)/e
        if num_neu_spe == 1:
            avg_imp_ene_n_nodes_plot    = avg_imp_ene_n1_nodes_plot
            avg_imp_ene_n_nodes_v2_plot = avg_imp_ene_n_nodes_plot
        if num_neu_spe == 3:
            avg_imp_ene_n2_nodes_plot   = qn2_tw_nodes_plot/(jn2_nodes_plot/e)/e
            avg_imp_ene_n3_nodes_plot   = qn3_tw_nodes_plot/(jn3_nodes_plot/e)/e
            avg_imp_ene_n_nodes_plot    = avg_imp_ene_n1_nodes_plot*(jn1_nodes_plot/jn_nodes_plot) + \
                                          avg_imp_ene_n2_nodes_plot*(jn2_nodes_plot/jn_nodes_plot) + \
                                          avg_imp_ene_n3_nodes_plot*(jn3_nodes_plot/jn_nodes_plot)
            avg_imp_ene_n_nodes_v2_plot = qn_tot_wall_nodes_plot/(jn_nodes_plot/e)/e
            
        avg_imp_ene_i1_surf_plot   = (qi1_tot_wall_surf_plot/(ji1_surf_plot/e))/e
        avg_imp_ene_i2_surf_plot   = (qi2_tot_wall_surf_plot/(ji2_surf_plot/(2*e)))/e
        if num_ion_spe == 4:
            avg_imp_ene_i3_surf_plot   = (qi3_tot_wall_surf_plot/(ji3_surf_plot/e))/e
            avg_imp_ene_i4_surf_plot   = (qi4_tot_wall_surf_plot/(ji4_surf_plot/(2*e)))/e
            den = ji1_surf_plot/e + ji2_surf_plot/(2*e) + ji3_surf_plot/e + ji4_surf_plot/(2*e)
            avg_imp_ene_ion_surf_plot  = avg_imp_ene_i1_surf_plot*(ji1_surf_plot/e/den)     + \
                                         avg_imp_ene_i2_surf_plot*(ji2_surf_plot/(2*e)/den) + \
                                         avg_imp_ene_i3_surf_plot*(ji3_surf_plot/e/den)     + \
                                         avg_imp_ene_i4_surf_plot*(ji4_surf_plot/(2*e)/den)
        elif num_ion_spe == 2:
            den = ji1_surf_plot/e + ji2_surf_plot/(2*e)
            avg_imp_ene_ion_surf_plot  = avg_imp_ene_i1_surf_plot*(ji1_surf_plot/e/den) + avg_imp_ene_i2_surf_plot*(ji2_surf_plot/(2*e)/den)
        avg_imp_ene_ion_surf_v2_plot = qi_tot_wall_surf_plot/(ji_surf_plot/e)/e
        
        avg_imp_ene_n1_surf_plot   = qn1_tw_surf_plot/(jn1_surf_plot/e)/e
        if num_neu_spe == 1:
            avg_imp_ene_n_surf_plot    = avg_imp_ene_n1_surf_plot
            avg_imp_ene_n_surf_v2_plot = avg_imp_ene_n_surf_plot
        if num_neu_spe == 3:
            avg_imp_ene_n2_surf_plot   = qn2_tw_surf_plot/(jn2_surf_plot/e)/e
            avg_imp_ene_n3_surf_plot   = qn3_tw_surf_plot/(jn3_surf_plot/e)/e
            avg_imp_ene_n_surf_plot    = avg_imp_ene_n1_surf_plot*(jn1_surf_plot/jn_surf_plot) + \
                                         avg_imp_ene_n2_surf_plot*(jn2_surf_plot/jn_surf_plot) + \
                                         avg_imp_ene_n3_surf_plot*(jn3_surf_plot/jn_surf_plot)
            avg_imp_ene_n_surf_v2_plot = qn_tot_wall_surf_plot/(jn_surf_plot/e)/e
        
        
        avg_imp_ene_e_wall_plot       = (qe_tot_wall_plot/(-je_b_plot/e))/e
        avg_imp_ene_e_wall_nodes_plot = (qe_tot_wall_nodes_plot/(-je_b_nodes_plot/e))/e
        avg_imp_ene_e_wall_surf_plot  = (qe_tot_wall_surf_plot/(-je_b_surf_plot/e))/e
        avg_imp_ene_e_b_plot          = (qe_tot_b_plot/(-je_b_plot/e))/e
        avg_imp_ene_e_b_nodes_plot    = (qe_tot_b_nodes_plot/(-je_b_nodes_plot/e))/e
        avg_imp_ene_e_b_surf_plot     = (qe_tot_b_surf_plot/(-je_b_surf_plot/e))/e
        
        # Compute the time-averaged value of the ratio dphi_sh/Te performing 
        # the average of the instantaneous ratio (instead of computing the ratio
        # with time-averaged values)
        dphi_infty_down       = np.zeros(np.shape(phi),dtype=float)
        dphi_infty_nodes_down = np.zeros(np.shape(phi_nodes),dtype=float)
        for istep in range(0,nsteps):
            dphi_infty_down[istep,IDfaces_down]                  = phi[istep,IDfaces_down] - phi_inf[istep]
            dphi_infty_nodes_down[inodes_down,jnodes_down,istep] = phi_nodes[inodes_down,jnodes_down,istep] - phi_inf[istep]
        if mean_type == 0:
            avg_inst_dphi_sh_b_Te_down_plot          = np.nanmean((dphi_sh_b[nsteps-last_steps::,bIDfaces_down])/Te[nsteps-last_steps::,IDfaces_down],axis=0)
            avg_inst_dphi_sh_b_Te_down_nodes_plot    = np.nanmean((dphi_sh_b_nodes[inodes_down,jnodes_down,nsteps-last_steps::])/Te_nodes[inodes_down,jnodes_down,nsteps-last_steps::],axis=1)
            avg_inst_dphi_sh_b_Te_down_plot_v2       = np.nanmean((dphi_infty_down[nsteps-last_steps::,IDfaces_down])/Te[nsteps-last_steps::,IDfaces_down],axis=0)
            avg_inst_dphi_sh_b_Te_down_nodes_plot_v2 = np.nanmean((dphi_infty_nodes_down[inodes_down,jnodes_down,nsteps-last_steps::])/Te_nodes[inodes_down,jnodes_down,nsteps-last_steps::],axis=1)
            avg_inst_dphi_infty_down                 = np.nanmean(dphi_infty_down[nsteps-last_steps::,IDfaces_down],axis=0)
            avg_inst_dphi_infty_nodes_down           = np.nanmean(dphi_infty_nodes_down[inodes_down,jnodes_down,nsteps-last_steps::],axis=1)
        elif mean_type == 1:
            avg_inst_dphi_sh_b_Te_down_plot          = np.nanmean((dphi_sh_b[step_i:step_f+1,bIDfaces_down])/Te[step_i:step_f+1,IDfaces_down],axis=0)
            avg_inst_dphi_sh_b_Te_down_nodes_plot    = np.nanmean((dphi_sh_b_nodes[inodes_down,jnodes_down,step_i:step_f+1])/Te_nodes[inodes_down,jnodes_down,step_i:step_f+1],axis=1)
            avg_inst_dphi_sh_b_Te_down_plot_v2       = np.nanmean((dphi_infty_down[step_i:step_f+1,IDfaces_down])/Te[step_i:step_f+1,IDfaces_down],axis=0)
            avg_inst_dphi_sh_b_Te_down_nodes_plot_v2 = np.nanmean((dphi_infty_nodes_down[inodes_down,jnodes_down,step_i:step_f+1])/Te_nodes[inodes_down,jnodes_down,step_i:step_f+1],axis=1)
            avg_inst_dphi_infty_down                 = np.nanmean(dphi_infty_down[step_i:step_f+1,IDfaces_down],axis=0)
            avg_inst_dphi_infty_nodes_down           = np.nanmean(dphi_infty_nodes_down[inodes_down,jnodes_down,step_i:step_f+1],axis=1)
        
        
        
        # Compute surface-averaged values of dphi_sh/Te on plume boundary
        surf_avg_dphi_infty_down = (np.dot((phi_plot[IDfaces_FLwall_ver]-phi_inf_mean),Afaces_FLwall_ver) + \
                                    np.dot((phi_plot[IDfaces_FLwall_lat]-phi_inf_mean),Afaces_FLwall_lat))/(np.sum(Afaces_FLwall_ver)+np.sum(Afaces_FLwall_lat))
        surf_avg_Te_down         = (np.dot(Te_plot[IDfaces_FLwall_ver],Afaces_FLwall_ver) + \
                                    np.dot(Te_plot[IDfaces_FLwall_lat],Afaces_FLwall_lat))/(np.sum(Afaces_FLwall_ver)+np.sum(Afaces_FLwall_lat))
        surf_avg_dphi_infty_Te_down = surf_avg_dphi_infty_down/surf_avg_Te_down
        print("surf avg dphi/Te infty  (-) = "+str(surf_avg_dphi_infty_Te_down))
        
        print("phi_down[-1]  (V) = %15.8e " %phi_plot[IDfaces_down[-1]])
        print("Te_down[-1]  (eV) = %15.8e " %Te_plot[IDfaces_down[-1]])
        
        
    #    # Do not plot units in axes
    #    # SAFRAN CHEOPS 1: units in cm
    ##    L_c = 3.725
    ##    H_c = (0.074995-0.052475)*100
    #    # HT5k: units in cm
    #    L_c = 2.53
    #    H_c = (0.0785-0.0565)*100
        # VHT_US (IEPC 2022) and paper GDML
        L_c = 2.9
        H_c = 2.22    
        # VHT_US PPSX00 testcase1 LP (TFM Alejandro)
    #    L_c = 2.5
    #    H_c = 1.1
        # PPSX00 testcase2 LP
#        L_c = 2.5
#        H_c = 1.5
        
     #   L_c = 1.0
     #   H_c = 1.0
        
        rs             = rs/H_c
        zs             = zs/L_c
        swall          = swall/L_c
        swall_inC      = swall_inC/L_c
        swall_surf     = swall_surf/L_c
        swall_inC_surf = swall_inC_surf/L_c
        Lbot           = Lbot/L_c
        Lchamb_bot     = Lchamb_bot/L_c
        Lchamb_top     = Lchamb_top/L_c
        Lanode         = Lanode/L_c
        sDwall_bot       = sDwall_bot/L_c
        sDwall_bot_nodes = sDwall_bot_nodes/L_c
        sDwall_bot_surf  = sDwall_bot_surf/L_c        
        sDwall_top       = sDwall_top/L_c
        sDwall_top_nodes = sDwall_top_nodes/L_c
        sDwall_top_surf  = sDwall_top_surf/L_c
        
        
#        sdown            = sdown/L_c
#        sdown_surf       = sdown_surf/L_c
#        sdown_nodes      = sdown_nodes/L_c
#        Lfreeloss_lat    = Lfreeloss_lat/L_c
#        Lfreeloss_ver    = Lfreeloss_ver/L_c
        
        sdown            = sdown/H_c
        sdown_surf       = sdown_surf/H_c
        sdown_nodes      = sdown_nodes/H_c
        Lfreeloss_lat    = Lfreeloss_lat/H_c
        Lfreeloss_ver    = Lfreeloss_ver/H_c
        
        
        # PROVISIONAL (04/03/2022) --------------------------------------------
        # Obtain here qe_tot_wall at freeloss downstream boundary because in code it is set to qe_tot_b. This is no longer true for commit 6f12bef on
#        qe_tot_wall_plot[bIDfaces_down] = qe_tot_b_plot[bIDfaces_down] - (-je_b_plot[bIDfaces_down]*dphi_sh_b_plot[bIDfaces_down])
        
        
        # In case delta_s_csl is zero, set the CSL value the code was using in the previous treatment of the CSL condition (before 29/03/2022)
        if np.all(delta_s_csl_plot == 0):
            delta_s_csl_plot = 0.985*np.ones(np.shape(delta_s_plot),dtype=float)
            
            
        # PROVISIONAL: in case qe_b is negative inside the chamber, correct with average surounding value
        for nneg in range(0,len(bIDfaces_wall_inC)):
            if qe_b_plot[bIDfaces_wall_inC[nneg]] < 0:
                print("----")
                print("Negative qe_b = "+str(qe_b_plot[bIDfaces_wall_inC[nneg]]))
                print("nneg = "+str(nneg)+"; bIDface = "+str(bIDfaces_wall_inC[nneg])+"; IDface = "+str(IDfaces_wall_inC[nneg]))
                qe_b_plot[bIDfaces_wall_inC[nneg]] = 0.5*(qe_b_plot[bIDfaces_wall_inC[nneg-1]] + qe_b_plot[bIDfaces_wall_inC[nneg+1]])
                print("Imposed qe_b = "+str(qe_b_plot[bIDfaces_wall_inC[nneg]]))
                print("zface [m] = "+str(face_geom[0,IDfaces_wall_inC[nneg]]))
                print("rface [m] = "+str(face_geom[1,IDfaces_wall_inC[nneg]]))
            
            
        # Apply the limiter for ion/electron currents and ion/electron impact energies at dielectric walls
        if min_curr_ratio != 0:
            je_max = np.max(np.abs(je_b_plot[bIDfaces_wall_inC]))
            je_min = np.min(np.abs(je_b_plot[bIDfaces_wall_inC]))
            pos = np.where(np.abs(je_b_plot[bIDfaces_wall]) < min_curr_ratio*je_max)[0][:]
            vec_je_b_plot      = je_b_plot[bIDfaces_wall]
            vec_ji_tot_b_plot  = ji_tot_b_plot[bIDfaces_wall]
            vec_imp_ene_e_plot = imp_ene_e_plot[bIDfaces_wall]
            for i in range(0,len(pos)):
                vec_je_b_plot[pos[i]]      = 0.0
                vec_ji_tot_b_plot[pos[i]]  = 0.0
                vec_imp_ene_e_plot[pos[i]] = 0.0
#                ji_tot_b_plot[bIDfaces_wall][i]  = 0.0
            je_b_plot[bIDfaces_wall]      = vec_je_b_plot
            ji_tot_b_plot[bIDfaces_wall]  = vec_ji_tot_b_plot
            imp_ene_e_plot[bIDfaces_wall] = vec_imp_ene_e_plot
            
            
        if plot_down == 1:
            
            # Since we have BM oscillations, time average of non-linear operations is not equal to the operation with time-averaged variables.
            if phi_inf_mean != 0:
                # Operation with average values
                imp_ene_e_b_down_check          = (2 + dphi_sh_b_plot[bIDfaces_down]/Te_plot[IDfaces_down])*Te_plot[IDfaces_down]
                imp_ene_e_b_down_check_v2       = (2 + (phi_plot[IDfaces_down]-phi_inf_mean)/Te_plot[IDfaces_down])*Te_plot[IDfaces_down]
                # Average of the operation with instantaneous values
                if mean_type == 0:
                    imp_ene_e_b_down_check_inst_avg    = np.nanmean((2 + dphi_sh_b[nsteps-last_steps::,bIDfaces_down]/Te[nsteps-last_steps::,IDfaces_down])*Te[nsteps-last_steps::,IDfaces_down],axis=0)
                    imp_ene_e_b_down_check_inst_avg_v2 = np.nanmean((2 + dphi_infty_down[nsteps-last_steps::,IDfaces_down]/Te[nsteps-last_steps::,IDfaces_down])*Te[nsteps-last_steps::,IDfaces_down],axis=0)
                elif mean_type == 1:
                    imp_ene_e_b_down_check_inst_avg    = np.nanmean((2 + dphi_sh_b[step_i:step_f+1,bIDfaces_down]/Te[step_i:step_f+1,IDfaces_down])*Te[step_i:step_f+1,IDfaces_down],axis=0)
                    imp_ene_e_b_down_check_inst_avg_v2 = np.nanmean((2 + dphi_infty_down[step_i:step_f+1,IDfaces_down]/Te[step_i:step_f+1,IDfaces_down])*Te[step_i:step_f+1,IDfaces_down],axis=0)
            else:
                # Operation with average values (in this case is equal to the average of the operation with instantaneous values)
                imp_ene_e_b_down_check          = 4.5*Te_plot[IDfaces_down]
                imp_ene_e_b_down_check_v2       = imp_ene_e_b_down_check
                # Average of the operation with instantaneous values
                imp_ene_e_b_down_check_inst_avg    = imp_ene_e_b_down_check
                imp_ene_e_b_down_check_inst_avg_v2 = imp_ene_e_b_down_check_inst_avg

            
           
            if plot_type == 0 or plot_type == 2:
                if k == 0:
                    axes1[0].semilogy(sdown,-je_b_plot[bIDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec_j[k], markersize=marker_size, marker=markers[ind], color=colors1[k], markeredgecolor = 'k', label=labels1[k])                
                    # axes1[0].semilogy(sdown,ji_tot_b_plot[bIDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every_vec_j[k+1], markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels1[k+1])                
                elif k == 1:
                    axes1[0].semilogy(sdown,-je_b_plot[bIDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec_j[k+1], markersize=marker_size, marker=markers[ind], color=colors1[k+1], markeredgecolor = 'k', label=labels1[k+1])   
                    axes1[0].semilogy(sdown,ji_tot_b_plot[bIDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every_vec_j[k+2], markersize=marker_size, marker=markers[ind], color=colors1[k+2], markeredgecolor = 'k', label=labels1[k+2])                
                elif k == 2:
                    axes1[0].semilogy(sdown,-je_b_plot[bIDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec_j[k+2], markersize=marker_size, marker=markers[ind], color=colors1[k+2], markeredgecolor = 'k', label=labels1[k+2])
                    # axes1[0].semilogy(sdown,ji_tot_b_plot[bIDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every_vec_j[k+1+2], markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels1[k+1+2])                            
                elif k == 3:
                    axes1[0].semilogy(sdown,-je_b_plot[bIDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec_j[k+1+2], markersize=marker_size, marker=markers[ind], color=colors1[k+1+2], markeredgecolor = 'k', label=labels1[k+1+2])
                    axes1[0].semilogy(sdown,ji_tot_b_plot[bIDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every_vec_j[k+2+2], markersize=marker_size, marker=markers[ind], color=colors1[k+2+2], markeredgecolor = 'k', label=labels1[k+2+2])                
                elif k == 4:
                    axes1[0].semilogy(sdown,-je_b_plot[bIDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec_j[k+2+2], markersize=marker_size, marker=markers[ind], color=colors1[k+2+2], markeredgecolor = 'k', label=labels1[k+2+2])
                    # axes1[0].semilogy(sdown,ji_tot_b_plot[bIDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every_vec_j[k+1+2+2], markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels1[k+1+2+2])                                                   
                elif k == 5:
                    axes1[0].semilogy(sdown,-je_b_plot[bIDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec_j[k+1+2+2], markersize=marker_size, marker=markers[ind], color=colors1[k+1+2+2], markeredgecolor = 'k', label=labels1[k+1+2+2])
                    axes1[0].semilogy(sdown,ji_tot_b_plot[bIDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every_vec_j[k+2+2+2], markersize=marker_size, marker=markers[ind], color=colors1[k+2+2+2], markeredgecolor = 'k', label=labels1[k+2+2+2])                
                       
                
                
                if k == 0 or k == 1:    
                    axes1[1].plot(sdown,phi_plot[IDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels2[k])
                    # if k == 0:
                    #     axes1[1].plot(sdown_nodes,phi_nodes_plot[inodes_down,jnodes_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=orange, markeredgecolor = 'k', label="")
                    # elif k == 1:
                    #     axes1[1].plot(sdown_nodes,phi_nodes_plot[inodes_down,jnodes_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=brown, markeredgecolor = 'k', label="")
                # if k == 0:
                    # axes1[1].plot(sdown,(phi_plot[IDfaces_down]-phi_inf_mean)/Te_plot[IDfaces_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels2[k])
                    axes1[2].plot(sdown,Te_plot[IDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels2[k])
                elif k == 2 or k == 3:
                    axes1[1].plot(sdown,phi_plot[IDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels2[k])
                    # if k == 2:
                    #     axes1[1].plot(sdown_nodes,phi_nodes_plot[inodes_down,jnodes_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='c', markeredgecolor = 'k', label="")
                    # elif k == 3:
                    #     axes1[1].plot(sdown_nodes,phi_nodes_plot[inodes_down,jnodes_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='y', markeredgecolor = 'k', label="")
                    axes1[2].plot(sdown,Te_plot[IDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels2[k])
                elif k == 4 or k == 5:
                    axes1[1].plot(sdown,phi_plot[IDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels2[k])
                    # if k == 2:
                    #     axes1[1].plot(sdown_nodes,phi_nodes_plot[inodes_down,jnodes_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='c', markeredgecolor = 'k', label="")
                    # elif k == 3:
                    #     axes1[1].plot(sdown_nodes,phi_nodes_plot[inodes_down,jnodes_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='y', markeredgecolor = 'k', label="")
                    axes1[2].plot(sdown,Te_plot[IDfaces_down], linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels2[k])
                
                # Figure of energy in linear y scale
                if k == 0:
                    axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k], markeredgecolor = 'k', label=labels3[k])
                    axes1[3].plot(sdown_surf,avg_imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k+1], markeredgecolor = 'k', label=labels3[k+1])
                    # axes1[3].plot(sdown_surf,imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='y', markeredgecolor = 'k', label="")
                    # axes1[3].plot(sdown,avg_imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+1], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels3[k+1])
                    # axes1[3].plot(sdown,imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+1], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels3[k+1])
                    # axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'c', label=r"$T_\mathrm{eP}\left(2+\frac{\phi_{\infty P}}{T_\mathrm{eP}}\right)$")
                elif k == 1:
                    axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k+1], markeredgecolor = 'k', label=labels3[k+1])
                    axes1[3].plot(sdown_surf,avg_imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k+2], markeredgecolor = 'k', label=labels3[k+2])
                    # axes1[3].plot(sdown_surf,imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='c', markeredgecolor = 'k', label="")
                    # axes1[3].plot(sdown,avg_imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+2], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+2], markeredgecolor = 'k', label=labels3[k+2])
                    # axes1[3].plot(sdown,imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+2], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+2], markeredgecolor = 'k', label=labels3[k+2])
                    # axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=r"4.5$T_\mathrm{eP}$")
                elif k == 2:
                    axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k+2], markeredgecolor = 'k', label=labels3[k+2])
                    axes1[3].plot(sdown_surf,avg_imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k+1+2], markeredgecolor = 'k', label=labels3[k+1+2])
                    # axes1[3].plot(sdown_surf,imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='y', markeredgecolor = 'k', label="")
                    # axes1[3].plot(sdown,avg_imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+1], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels3[k+1])
                    # axes1[3].plot(sdown,imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+1], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels3[k+1])
                    # axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'c', label=r"$T_\mathrm{eP}\left(2+\frac{\phi_{\infty P}}{T_\mathrm{eP}}\right)$")
                elif k == 3:
                    axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k+1+2], markeredgecolor = 'k', label=labels3[k+1+2])
                    axes1[3].plot(sdown_surf,avg_imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k+2+2], markeredgecolor = 'k', label=labels3[k+2+2])
                    # axes1[3].plot(sdown_surf,imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='c', markeredgecolor = 'k', label="")
                    # axes1[3].plot(sdown,avg_imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+2], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+2], markeredgecolor = 'k', label=labels3[k+2])
                    # axes1[3].plot(sdown,imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+2], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+2], markeredgecolor = 'k', label=labels3[k+2])
                    # axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=r"4.5$T_\mathrm{eP}$")
                elif k == 4:
                    axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k+2+2], markeredgecolor = 'k', label=labels3[k+2+2])
                    axes1[3].plot(sdown_surf,avg_imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k+1+2+2], markeredgecolor = 'k', label=labels3[k+1+2+2])
                    # axes1[3].plot(sdown_surf,imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='y', markeredgecolor = 'k', label="")
                    # axes1[3].plot(sdown,avg_imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+1], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels3[k+1])
                    # axes1[3].plot(sdown,imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+1], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels3[k+1])
                    # axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'c', label=r"$T_\mathrm{eP}\left(2+\frac{\phi_{\infty P}}{T_\mathrm{eP}}\right)$")
                elif k == 5:
                    axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='-', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k+1+2+2], markeredgecolor = 'k', label=labels3[k+1+2+2])
                    axes1[3].plot(sdown_surf,avg_imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every_vec[ind], markersize=marker_size, marker=markers[ind], color=colors2[k+2+2+2], markeredgecolor = 'k', label=labels3[k+2+2+2])
                    # axes1[3].plot(sdown_surf,imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='c', markeredgecolor = 'k', label="")
                    # axes1[3].plot(sdown,avg_imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+2], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+2], markeredgecolor = 'k', label=labels3[k+2])
                    # axes1[3].plot(sdown,imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+2], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+2], markeredgecolor = 'k', label=labels3[k+2])
                    # axes1[3].plot(sdown,imp_ene_e_b_down_check_v2, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=r"4.5$T_\mathrm{eP}$")

                # Figure of energy in semilogy
                # if k == 0:
                #     axes1[3].semilogy(sdown_surf,avg_imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels3[k])
                #     # axes1[3].semilogy(sdown_surf,imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='y', markeredgecolor = 'k', label="")
                #     axes1[3].semilogy(sdown,imp_ene_e_b_down_check_v2, linestyle=linestyles[ind+1], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels3[k+1])
                #     # axes1[3].semilogy(sdown,avg_imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+1], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels3[k+1])
                #     # axes1[3].semilogy(sdown,imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+1], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels3[k+1])
                #     # axes1[3].semilogy(sdown,imp_ene_e_b_down_check_v2, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'c', label=r"$T_\mathrm{eP}\left(2+\frac{\phi_{\infty P}}{T_\mathrm{eP}}\right)$")
                # elif k == 1:
                #     axes1[3].semilogy(sdown_surf,avg_imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle=linestyles[ind+1], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels3[k+1])
                #     # axes1[3].semilogy(sdown_surf,imp_ene_ion_surf_v2_plot[indsurf_down]/factor_divide_ene_ion, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='c', markeredgecolor = 'k', label="")
                #     axes1[3].semilogy(sdown,imp_ene_e_b_down_check_v2, linestyle=linestyles[ind+2], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+2], markeredgecolor = 'k', label=labels3[k+2])
                #     # axes1[3].semilogy(sdown,avg_imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+2], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+2], markeredgecolor = 'k', label=labels3[k+2])
                #     # axes1[3].semilogy(sdown,imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind+2], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+2], markeredgecolor = 'k', label=labels3[k+2])
                #     # axes1[3].semilogy(sdown,imp_ene_e_b_down_check_v2, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=r"4.5$T_\mathrm{eP}$")

            
                
                
        ind = ind + 1
        if ind > 8:
            ind = 0
            ind2 = ind2 + 1
            if ind2 > 6:
                ind = 0
                ind2 = 0
                ind3 = ind3 + 1
        

    if plot_down == 1:
        axes1[0].axvline(x=Lfreeloss_ver, linestyle=':',color='k', linewidth = line_width)
        axes1[1].axvline(x=Lfreeloss_ver, linestyle=':',color='k', linewidth = line_width)
        axes1[2].axvline(x=Lfreeloss_ver, linestyle=':',color='k', linewidth = line_width)
        axes1[3].axvline(x=Lfreeloss_ver, linestyle=':',color='k', linewidth = line_width)
        # axes1[4].axvline(x=Lfreeloss_ver, linestyle=':',color='k', linewidth = line_width)
        # axes1[5].axvline(x=Lfreeloss_ver, linestyle=':',color='k', linewidth = line_width)
        
        axes1[0].axvline(x=rs[rind,0], linestyle='--',color='k', linewidth = line_width)
        axes1[1].axvline(x=rs[rind,0], linestyle='--',color='k', linewidth = line_width)
        axes1[2].axvline(x=rs[rind,0], linestyle='--',color='k', linewidth = line_width)
        axes1[3].axvline(x=rs[rind,0], linestyle='--',color='k', linewidth = line_width)
        # axes1[4].axvline(x=rs[rind,0], linestyle='--',color='k', linewidth = line_width)
        # axes1[5].axvline(x=rs[rind,0], linestyle='--',color='k', linewidth = line_width)
        
        lines1 = axes1[0].get_lines()
        lines2 = axes1[1].get_lines()
        lines3 = axes1[2].get_lines()
        lines4 = axes1[3].get_lines()
        
        lines1_flip = np.flip(lines1)
        lines2_flip = np.flip(lines2)
        lines3_flip = np.flip(lines3)
        lines4_flip = np.flip(lines4)
        
        lines1_flip1 = lines1_flip[1::]
        lines2_flip2 = lines2_flip[1::]
        lines3_flip3 = lines3_flip[1::]
        lines4_flip4 = lines4_flip[1::]
        
        axes1[0].legend([lines1_flip1[i] for i in range(0,len(lines1_flip1))], [lines1_flip1[i].get_label() for i in range(0,len(lines1_flip1))],fontsize = font_size_legend-1,loc='best',ncol=1,frameon = True,numpoints=2,handlelength=2.9,borderpad=0.25,columnspacing=1) 
        axes1[1].legend([lines2_flip2[i] for i in range(0,len(lines2_flip2))], [lines2_flip2[i].get_label() for i in range(0,len(lines2_flip2))],fontsize = font_size_legend,loc='best',ncol=3,frameon = True,numpoints=2,handlelength=2.9,borderpad=0.25,columnspacing=1) 
        axes1[2].legend([lines3_flip3[i] for i in range(0,len(lines3_flip3))], [lines3_flip3[i].get_label() for i in range(0,len(lines3_flip3))],fontsize = font_size_legend,loc='best',ncol=3,frameon = True,numpoints=2,handlelength=2.9,borderpad=0.25,columnspacing=1) 
        axes1[3].legend([lines4_flip4[i] for i in range(0,len(lines4_flip4))], [lines4_flip4[i].get_label() for i in range(0,len(lines4_flip4))],fontsize = font_size_legend,loc='best',ncol=3,frameon = True,numpoints=2,handlelength=2.9,borderpad=0.25,columnspacing=1) 
            
        # axes1[0].legend(fontsize = font_size_legend,loc='best',ncol=3,frameon = True,numpoints=2,handlelength=2.9,borderpad=0.2,columnspacing=1) 
        # axes1[1].legend(fontsize = font_size_legend,loc='best',ncol=3,frameon = True,numpoints=2,handlelength=2.9,borderpad=0.2,columnspacing=1) 
        # axes1[2].legend(fontsize = font_size_legend,loc='best',ncol=3,frameon = True,numpoints=2,handlelength=2.9,borderpad=0.2,columnspacing=1) 
        # axes1[3].legend(fontsize = font_size_legend,loc='best',ncol=3,frameon = True,numpoints=2,handlelength=2.9,borderpad=0.2,columnspacing=1) 

        # for t in legend.get_texts():
        #     # t.set_ha('right')
        #     t.set_horizontalalignment("left")

    fig1.tight_layout()
    if save_flag == 1:
        # fig1.savefig(path_out+"fig4_1Ds_1col_merged_linewidth2_ene"+figs_format,bbox_inches='tight')
        fig1.savefig(path_out+"fig4_1Ds_P3_C1_C3_noji_colors"+figs_format,bbox_inches='tight')
        plt.close(fig1)

        
    