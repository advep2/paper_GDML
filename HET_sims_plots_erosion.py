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

save_flag = 0

    
# Plots save flag
#figs_format = ".eps"
figs_format = ".png"
#figs_format = ".pdf"

# Plots to produce
erosion_parametric_plots = 1

path_out = "CHEOPS_Final_figs/"


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
    

if erosion_parametric_plots == 1:
    print("######## erosion_parametric_plots ########")
          
    # Print out time steps
    timestep = 'last'
    
    marker_size  = 5
    marker_every = 1
    marker_every_erosion = 100
  
    allsteps_flag   = 1
    
    mean_vars       = 1
    mean_type       = 0
    last_steps      = 700
    step_i          = 100
    step_f          = 500
    plot_mean_vars  = 1
    
    # For plotting profiles along the boundary walls
    plot_Dwall     = 1
    plot_Awall     = 0
    
    plot_sp_yield  = 1
    plot_erosion   = 1
    
    # Erosion inputs
    t_op          = 300     # Operation time in [h]
    # --- For Y0(E)
    c_ref         = 3.75e-5 # [mm3/C-eV]  [*0.5,*1,*1.5]
    E_th_ref      = 60      # [eV]  [40,60,80]
    # --- For F(theta)
    Fmax_ref      = 3       # [-]   [2,3,4]
    theta_max_ref = 60      # [deg] [50,60,70]
    a_ref         = 8       # [-]
    # Eroded profile last point of chamber approach
    # 0 - Old approach: moving last picM node in chamfer according to the panel normal
    # 1 - Old approach: moving last picM node in chamfer according to a vertical normal
    # 2 - New approach: last point in eroded profile is the crossing point between the 
    #                   vertical plume wall and the straight line defined by the two 
    #                   previous nodes of the eroded profile
    erprof_app = 2
    
    # X-vectors for energy and angle
    nxvec      = 1000 
    xvec_E     = np.linspace(0,350,nxvec,dtype=float)
    xvec_theta = np.linspace(0,90,nxvec,dtype=float)
    
    ncases = 3
    
    # Reference values --------------------------------------------------------
#    c_vec         = np.array([c_ref,c_ref,c_ref],dtype=float)
#    E_th_vec      = np.array([E_th_ref,E_th_ref,E_th_ref],dtype=float)
#    Fmax_vec      = np.array([Fmax_ref,Fmax_ref,Fmax_ref],dtype=float)
#    theta_max_vec = np.array([theta_max_ref,theta_max_ref,theta_max_ref],dtype=float)
#    a_vec         = np.array([a_ref,a_ref,a_ref],dtype=float)
    
    
#    # Chaning c (remaining parameters at reference values) --------------------
#    c_vec         = np.array([0.5*c_ref,1.0*c_ref,1.5*c_ref],dtype=float)
#    E_th_vec      = np.array([E_th_ref,E_th_ref,E_th_ref],dtype=float)
#    Fmax_vec      = np.array([Fmax_ref,Fmax_ref,Fmax_ref],dtype=float)
#    theta_max_vec = np.array([theta_max_ref,theta_max_ref,theta_max_ref],dtype=float)
#    a_vec         = np.array([a_ref,a_ref,a_ref],dtype=float)
#    # Chaning Eth (remaining parameters at reference values) ------------------
#    c_vec         = np.array([c_ref,c_ref,c_ref],dtype=float)
#    E_th_vec      = np.array([40,E_th_ref,80],dtype=float)
#    Fmax_vec      = np.array([Fmax_ref,Fmax_ref,Fmax_ref],dtype=float)
#    theta_max_vec = np.array([theta_max_ref,theta_max_ref,theta_max_ref],dtype=float)
#    a_vec         = np.array([a_ref,a_ref,a_ref],dtype=float)
    # Chaning Fmax (remaining parameters at reference values) -----------------
    c_vec         = np.array([c_ref,c_ref,c_ref],dtype=float)
    E_th_vec      = np.array([E_th_ref,E_th_ref,E_th_ref],dtype=float)
    Fmax_vec      = np.array([2,Fmax_ref,4],dtype=float)
    theta_max_vec = np.array([theta_max_ref,theta_max_ref,theta_max_ref],dtype=float)
    a_vec         = np.array([a_ref,a_ref,a_ref],dtype=float)
#    # Chaning thetamax (remaining parameters at reference values) -------------
#    c_vec         = np.array([c_ref,c_ref,c_ref],dtype=float)
#    E_th_vec      = np.array([E_th_ref,E_th_ref,E_th_ref],dtype=float)
#    Fmax_vec      = np.array([Fmax_ref,Fmax_ref,Fmax_ref],dtype=float)
#    theta_max_vec = np.array([50,theta_max_ref,70],dtype=float)
#    a_vec         = np.array([a_ref,a_ref,a_ref],dtype=float)

    log_yaxis      = 1
    log_yaxis_tol  = 1E0
    
    

    if timestep == 'last':
        timestep = -1  
    if allsteps_flag == 0:
        mean_vars = 0
        
        
    # Simulation name (only one simulation)
    nsims = 1
    oldpost_sim      = 3
    oldsimparams_sim = 8        
    
    
    sim_name = "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_71d0dcb"
#    sim_name = "../../../Ca_sims_files/T2N4_pm1em1_cat1200_tm15_te1_tq125_71d0dcb"


    topo_case = 2
    if topo_case == 1:
        PIC_mesh_file_name = "PIC_mesh_topo1_refined4.hdf5"
    elif topo_case == 2:
        PIC_mesh_file_name = "PIC_mesh_topo2_refined4.hdf5"
    elif topo_case == 0:    
        PIC_mesh_file_name = "SPT100_picM.hdf5"

    # Labels                          
    labels = [
#              # Chaning c (remaining parameters at reference values) ----------
#              r"C1  $\, c = 1.875\cdot 10^{-5}$ (mm$^3$/C-eV)",
#              r"REF $c = 3.750\cdot 10^{-5}$ (mm$^3$/C-eV)",
#              r"C2  $\, c = 5.625\cdot 10^{-5}$ (mm$^3$/C-eV)",
##              r"$c_1$",
##              r"$c_2$",
##              r"$c_3$",
              
#              # Chaning Eth (remaining parameters at reference values) --------
#              r"E1 $E_{th} = 40$ (eV)",
#              r"REF $E_{th} = 60$ (eV)",
#              r"E2 $E_{th} = 80$ (eV)",
              
              # Chaning Fmax (remaining parameters at reference values) -------
              r"F1 $F_{max} = 2$ (-)",
              r"REF $F_{max} = 3$ (-)",
              r"F2 $F_{max} = 4$ (-)",
              
#              # Chaning thetamax (remaining parameters at reference values) ---
#              r"A1 $\theta_{max} = 50$ (deg)",
#              r"REF $\theta_{max} = 60$ (deg)",
#              r"A2 $\theta_{max} = 70$ (deg)",
#            
              ]

    
    # Line colors
    colors = ['k','r','g','b','m','c','y',orange,brown]
#    colors = ['k','m',orange,brown]
    # Markers
    markers = ['s','o','v','^','<', '>','D','p','*']
#    markers = ['s','<','D','p']
    # Line style
    linestyles = ['-','--','-.', ':','-','--','-.']
    linestyles = ['-','-','-','-','-','-','-']


    # Axial profile plots
    if plot_Dwall == 1:
        if plot_sp_yield == 1:
            plt.figure(r'Y0(E) Dwall_bot_top')
            plt.xlabel(r"$E$ (eV)",fontsize = font_size)
            plt.title(r"$Y_0(E)$ (mm$^3$/C)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'F(theta) Dwall_bot_top')
            plt.xlabel(r"$\theta$ (deg)",fontsize = font_size)
            plt.title(r"$F(\theta)$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

        if plot_erosion == 1:
            plt.figure(r'dhdt Dwall_bot_top')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
#            plt.title(r"(c) $dh/dt$ (mm/s)",fontsize = font_size)
            plt.title(r"$dh/dt$ ($\mu$m/h)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'h(z) Dwall_bot_top')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
#            plt.title(r"$h$ ($\mu$m)",fontsize = font_size)
            plt.title(r"$h$ (mm)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'h(z)/hexit Dwall_bot_top')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"$h/h_{exit}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'eroded chamber Dwall_bot_top')
            plt.xlabel(r"$z$ (cm)",fontsize = font_size)
            plt.ylabel(r"$r$ (cm)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'points eroded chamber Dwall_bot_top')
            plt.xlabel(r"$z$ (cm)",fontsize = font_size-5)
            plt.ylabel(r"$r$ (cm)",fontsize = font_size-5)
            plt.xticks(fontsize = ticks_size-5) 
            plt.yticks(fontsize = ticks_size-5)
            
    ind_ini_letter = sim_name.rfind('/') + 1
    print("SIM NAME: "+str(sim_name[ind_ini_letter::]))    
    ind  = 0
    ind2 = 0
    ind3 = 0
    for k in range(0,ncases):
#        ind_ini_letter = sim_name.rfind('/') + 1
#        print("##### CASE "+str(k+1)+": "+sim_name[ind_ini_letter::]+" #####")
        print("##### CASE "+str(k+1)+" ################################")
        print("c [mm3/C-eV]        = "+str(c_vec[k]))
        print("E_th [eV]           = "+str(E_th_vec[k]))
        print("Fmax [-]            = "+str(Fmax_vec[k]))
        print("theta_max_vec [deg] = "+str(theta_max_vec[k]))
        ######################## READ INPUT/OUTPUT FILES ##########################
        # Obtain paths to simulation files
        path_picM         = sim_name+"/SET/inp/"+PIC_mesh_file_name
        path_simstate_inp = sim_name+"/CORE/inp/SimState.hdf5"
        path_simstate_out = sim_name+"/CORE/out/SimState.hdf5"
        path_postdata_out = sim_name+"/CORE/out/PostData.hdf5"
        path_simparams_inp = sim_name+"/CORE/inp/sim_params.inp"
        print("Reading results...")
        [points,zs,rs,zscells,rscells,dims,nodes_flag,cells_flag,cells_vol,
           volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,eta_max,eta_min,xi_top,
           xi_bottom,time,steps,dt,nsteps,sc_bot,sc_top,
           
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
           MkQ1_surf,ji1_surf,ji2_surf,ji_surf,gn1_tw_surf,gn1_fw_surf,
           qi1_tot_wall_surf,qi2_tot_wall_surf,qi_tot_wall_surf,qn1_tw_surf,
           qn1_fw_surf,imp_ene_i1_surf,imp_ene_i2_surf,imp_ene_n1_surf,
           
           angle_bins_i1,ene_bins_i1,normv_bins_i1,angle_bins_i2,ene_bins_i2,
           normv_bins_i2,angle_bins_n1,ene_bins_n1,normv_bins_n1,nbins_angle,
           nbins_ene,nbins_normv,
           
           angle_df_i1,ene_df_i1,normv_df_i1,angle_df_i2,ene_df_i2,normv_df_i2,
           angle_df_n1,ene_df_n1,normv_df_n1] = HET_sims_read_df(path_simstate_inp,path_simstate_out,path_postdata_out,path_simparams_inp,
                                                                 path_picM,allsteps_flag,timestep,oldpost_sim,oldsimparams_sim)
            
        
        
        if mean_vars == 1:        
            print("Averaging variables...")                                                                              
            [nQ1_inst_surf_mean,nQ1_surf_mean,nQ2_inst_surf_mean,
               nQ2_surf_mean,dphi_kbc_surf_mean,MkQ1_surf_mean,ji1_surf_mean,
               ji2_surf_mean,ji_surf_mean,gn1_tw_surf_mean,gn1_fw_surf_mean,
               qi1_tot_wall_surf_mean,qi2_tot_wall_surf_mean,qi_tot_wall_surf_mean,
               qn1_tw_surf_mean,qn1_fw_surf_mean,imp_ene_i1_surf_mean,
               imp_ene_i2_surf_mean,imp_ene_n1_surf_mean,
               
               angle_df_i1_mean,ene_df_i1_mean,normv_df_i1_mean,angle_df_i2_mean,
               ene_df_i2_mean,normv_df_i2_mean,angle_df_n1_mean,ene_df_n1_mean,
               normv_df_n1_mean] = HET_sims_mean_df(nsteps,mean_type,last_steps,step_i,step_f,
                                                    nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,nQ2_surf,
                                                    dphi_kbc_surf,MkQ1_surf,ji1_surf,ji2_surf,ji_surf,
                                                    gn1_tw_surf,gn1_fw_surf,qi1_tot_wall_surf,
                                                    qi2_tot_wall_surf,qi_tot_wall_surf,qn1_tw_surf,
                                                    qn1_fw_surf,imp_ene_i1_surf,imp_ene_i2_surf,imp_ene_n1_surf,
                                                    
                                                    angle_df_i1,ene_df_i1,normv_df_i1,angle_df_i2,ene_df_i2,
                                                    normv_df_i2,angle_df_n1,ene_df_n1,normv_df_n1)
                                                                                            
                                                                                            
        print("Obtaining final variables for plotting...") 
        if mean_vars == 1 and plot_mean_vars == 1:
            print("Plotting variables are time-averaged")
            [nQ1_inst_surf_plot,nQ1_surf_plot,nQ2_inst_surf_plot,
               nQ2_surf_plot,dphi_kbc_surf_plot,MkQ1_surf_plot,ji1_surf_plot,
               ji2_surf_plot,ji_surf_plot,gn1_tw_surf_plot,gn1_fw_surf_plot,
               qi1_tot_wall_surf_plot,qi2_tot_wall_surf_plot,qi_tot_wall_surf_plot,
               qn1_tw_surf_plot,qn1_fw_surf_plot,imp_ene_i1_surf_plot,
               imp_ene_i2_surf_plot,imp_ene_n1_surf_plot,
               
               angle_df_i1_plot,ene_df_i1_plot,normv_df_i1_plot,angle_df_i2_plot,
               ene_df_i2_plot,normv_df_i2_plot,angle_df_n1_plot,ene_df_n1_plot,
               normv_df_n1_plot] = HET_sims_cp_vars_df(nQ1_inst_surf_mean,nQ1_surf_mean,nQ2_inst_surf_mean,
                                                       nQ2_surf_mean,dphi_kbc_surf_mean,MkQ1_surf_mean,ji1_surf_mean,
                                                       ji2_surf_mean,ji_surf_mean,gn1_tw_surf_mean,gn1_fw_surf_mean,
                                                       qi1_tot_wall_surf_mean,qi2_tot_wall_surf_mean,qi_tot_wall_surf_mean,
                                                       qn1_tw_surf_mean,qn1_fw_surf_mean,imp_ene_i1_surf_mean,
                                                       imp_ene_i2_surf_mean,imp_ene_n1_surf_mean,
                                                       
                                                       angle_df_i1_mean,ene_df_i1_mean,normv_df_i1_mean,angle_df_i2_mean,
                                                       ene_df_i2_mean,normv_df_i2_mean,angle_df_n1_mean,ene_df_n1_mean,
                                                       normv_df_n1_mean)
            
        else:
            [nQ1_inst_surf_plot,nQ1_surf_plot,nQ2_inst_surf_plot,
               nQ2_surf_plot,dphi_kbc_surf_plot,MkQ1_surf_plot,ji1_surf_plot,
               ji2_surf_plot,ji_surf_plot,gn1_tw_surf_plot,gn1_fw_surf_plot,
               qi1_tot_wall_surf_plot,qi2_tot_wall_surf_plot,qi_tot_wall_surf_plot,
               qn1_tw_surf_plot,qn1_fw_surf_plot,imp_ene_i1_surf_plot,
               imp_ene_i2_surf_plot,imp_ene_n1_surf_plot,
               
               angle_df_i1_plot,ene_df_i1_plot,normv_df_i1_plot,angle_df_i2_plot,
               ene_df_i2_plot,normv_df_i2_plot,angle_df_n1_plot,ene_df_n1_plot,
               normv_df_n1_plot] = HET_sims_cp_vars_df(nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,nQ2_surf,
                                                       dphi_kbc_surf,MkQ1_surf,ji1_surf,ji2_surf,ji_surf,
                                                       gn1_tw_surf,gn1_fw_surf,qi1_tot_wall_surf,
                                                       qi2_tot_wall_surf,qi_tot_wall_surf,qn1_tw_surf,
                                                       qn1_fw_surf,imp_ene_i1_surf,imp_ene_i2_surf,imp_ene_n1_surf,
                                                        
                                                       angle_df_i1,ene_df_i1,normv_df_i1,angle_df_i2,ene_df_i2,
                                                       normv_df_i2,angle_df_n1,ene_df_n1,normv_df_n1)
        
        # Obtain the surface elements in Dwall that are inside the chamber
        indsurf_inC_Dwall_bot = np.where(sDwall_bot_surf <= sc_bot)[0][:]
        indsurf_inC_Dwall_top = np.where(sDwall_top_surf <= sc_top)[0][:]
        nsurf_inC_Dwall_bot   = len(indsurf_inC_Dwall_bot)
        nsurf_inC_Dwall_top   = len(indsurf_inC_Dwall_top)
            
        # Compute currents from the angle distribution function, mean impact angles and energies at each surface element 
        df_ji1_surf_plot        = np.zeros(nsurf_bound,dtype=float)
        df_ji2_surf_plot        = np.zeros(nsurf_bound,dtype=float)
        df_gn1_tw_surf_plot     = np.zeros(nsurf_bound,dtype=float)
        df_angle_i1_surf_plot   = np.zeros(nsurf_bound,dtype=float)
        df_angle_i2_surf_plot   = np.zeros(nsurf_bound,dtype=float)
        df_angle_n1_surf_plot   = np.zeros(nsurf_bound,dtype=float)
        df_imp_ene_i1_surf_plot = np.zeros(nsurf_bound,dtype=float)
        df_imp_ene_i2_surf_plot = np.zeros(nsurf_bound,dtype=float)
        df_imp_ene_n1_surf_plot = np.zeros(nsurf_bound,dtype=float)
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
        # Obtain magnitudes for the whole ion population from the magnitudes of i1 and i2
        df_ji_surf_plot          = df_ji1_surf_plot + df_ji2_surf_plot
        df_angle_ion_surf_plot   = df_angle_i1_surf_plot*(df_ji1_surf_plot/df_ji_surf_plot) + df_angle_i2_surf_plot*(df_ji2_surf_plot/df_ji_surf_plot)
        df_imp_ene_ion_surf_plot = df_imp_ene_i1_surf_plot*(df_ji1_surf_plot/df_ji_surf_plot) + df_imp_ene_i2_surf_plot*(df_ji2_surf_plot/df_ji_surf_plot)
        
        # Obtain the erosion rate [mm/s] o [mum/h]. Yields Y [mm3/C], fluxes are [1/m2-s]        
        Y0_E_i1   = erosion_Y0(c_vec[k],E_th_vec[k],imp_ene_i1_surf_plot/e,nsurf_bound)
        Y0_E_i2   = erosion_Y0(c_vec[k],E_th_vec[k],imp_ene_i2_surf_plot/e,nsurf_bound)
        Y0_E_n1   = erosion_Y0(c_vec[k],E_th_vec[k],imp_ene_n1_surf_plot/e,nsurf_bound)
        Ftheta_i1 = erosion_Ftheta(Fmax_vec[k],theta_max_vec[k],a_vec[k],df_angle_i1_surf_plot,nsurf_bound)
        Ftheta_i2 = erosion_Ftheta(Fmax_vec[k],theta_max_vec[k],a_vec[k],df_angle_i2_surf_plot,nsurf_bound)
        Ftheta_n1 = erosion_Ftheta(Fmax_vec[k],theta_max_vec[k],a_vec[k],df_angle_n1_surf_plot,nsurf_bound)
        Y_i1      = Y0_E_i1*Ftheta_i1
        Y_i2      = Y0_E_i2*Ftheta_i2
        Y_n1      = Y0_E_n1*Ftheta_n1
        dhdt_i1   = (ji1_surf_plot/e)*e*Y_i1*1E-6                                      # [mm/s]
        dhdt_i2   = (ji2_surf_plot/(2*e))*e*Y_i2*1E-6                                  # [mm/s]
        dhdt      = (ji1_surf_plot/e)*e*Y_i1*1E-6 + (ji2_surf_plot/(2*e))*e*Y_i2*1E-6  # [mm/s]
        dhdt_mumh = dhdt*1E3*3600                                                      # [mum/h]
        dhdt_i1_mumh = dhdt_i1*1E3*3600                                                # [mum/h]
        dhdt_i2_mumh = dhdt_i2*1E3*3600                                                # [mum/h]
        h_mum     = dhdt_mumh*t_op # h(z) [mum] erosion profile after a given operation time in hours
        h_m       = h_mum*1E-6    # h(z) [m] for performing operations
        # Obtain the erosion rate using magnitudes for the whole ion population
        Y0_E_ion   = erosion_Y0(c_vec[k],E_th_vec[k],(imp_ene_i1_surf_plot*(ji1_surf_plot/ji_surf_plot) + imp_ene_i2_surf_plot*(ji2_surf_plot/ji_surf_plot))/e,nsurf_bound)
        Ftheta_ion = erosion_Ftheta(Fmax_vec[k],theta_max_vec[k],a_vec[k],df_angle_ion_surf_plot,nsurf_bound)
        Y_ion      = Y0_E_ion*Ftheta_ion
        dhdt_ion   = ((ji1_surf_plot/e+ji2_surf_plot/(2*e)))*e*Y_ion*1E-6                    # [mm/s]
        dhdt_ion_mumh = dhdt_ion*1E3*3600   
        h_ion_mum     = dhdt_ion_mumh*t_op # h(z) [mum] erosion profile after a given operation time in hours
        h_ion_m       = h_ion_mum*1E-6     # h(z) [m] for performing operations
        
        # Obtain the chamber Dwalls eroded profile
        np_erosion_bot = 2+nsurf_inC_Dwall_bot 
        np_erosion_top = 2+nsurf_inC_Dwall_top 
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
                # Points in the middle correspond to central points of picS surface elements along Dwall_bot
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_bot[ind_er-1]][0],imp_elems[indsurf_Dwall_bot[ind_er-1]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_bot[ind_er-1]][0],imp_elems[indsurf_Dwall_bot[ind_er-1]][1],1] 
                re_Dwall_bot[ind_er] = rsurf_Dwall_bot[ind_er-1] + h_m[indsurf_Dwall_bot[ind_er-1]]*norm_r
                ze_Dwall_bot[ind_er] = zsurf_Dwall_bot[ind_er-1] + h_m[indsurf_Dwall_bot[ind_er-1]]*norm_z
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
                # Points in the middle correspond to central points of picS surface elements along Dwall_top
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_top[ind_er-1]][0],imp_elems[indsurf_Dwall_top[ind_er-1]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_top[ind_er-1]][0],imp_elems[indsurf_Dwall_top[ind_er-1]][1],1] 
                re_Dwall_top[ind_er] = rsurf_Dwall_top[ind_er-1] + h_m[indsurf_Dwall_top[ind_er-1]]*norm_r
                ze_Dwall_top[ind_er] = zsurf_Dwall_top[ind_er-1] + h_m[indsurf_Dwall_top[ind_er-1]]*norm_z
                
        # Obtain the chamber Dwalls eroded profile considering the erosion rate obtained for the whole ion population
        np_erosion_bot   = 2+nsurf_inC_Dwall_bot 
        np_erosion_top   = 2+nsurf_inC_Dwall_top 
        re_Dwall_bot_ion = np.zeros(np_erosion_bot,dtype=float)
        ze_Dwall_bot_ion = np.zeros(np_erosion_bot,dtype=float)
        re_Dwall_top_ion = np.zeros(np_erosion_top,dtype=float)
        ze_Dwall_top_ion = np.zeros(np_erosion_top,dtype=float)
        # Dwall_bot
        for ind_er in range(0,np_erosion_bot):
            if ind_er == 0:
                # First point is the first picM node along Dwall_bot
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_bot[0]][0],imp_elems[indsurf_Dwall_bot[0]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_bot[0]][0],imp_elems[indsurf_Dwall_bot[0]][1],1] 
                re_Dwall_bot_ion[ind_er] = rs[eta_min,0] + h_ion_m[indsurf_Dwall_bot[0]]*norm_r
                ze_Dwall_bot_ion[ind_er] = zs[eta_min,0] + h_ion_m[indsurf_Dwall_bot[0]]*norm_z
            elif ind_er == np_erosion_bot-1:
                # Last point is the last picM node along Dwall_bot
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][0],imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][0],imp_elems[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]][1],1] 
                norm_r = -1.0
                norm_z = 0.0
                re_Dwall_bot_ion[ind_er] = rs[eta_min,xi_bottom] + h_ion_m[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*norm_r
                ze_Dwall_bot_ion[ind_er] = zs[eta_min,xi_bottom] + h_ion_m[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*norm_z
            else:
                # Points in the middle correspond to central points of picS surface elements along Dwall_bot
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_bot[ind_er-1]][0],imp_elems[indsurf_Dwall_bot[ind_er-1]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_bot[ind_er-1]][0],imp_elems[indsurf_Dwall_bot[ind_er-1]][1],1] 
                re_Dwall_bot_ion[ind_er] = rsurf_Dwall_bot[ind_er-1] + h_ion_m[indsurf_Dwall_bot[ind_er-1]]*norm_r
                ze_Dwall_bot_ion[ind_er] = zsurf_Dwall_bot[ind_er-1] + h_ion_m[indsurf_Dwall_bot[ind_er-1]]*norm_z
        # Dwall_top
        for ind_er in range(0,np_erosion_top):
            if ind_er == 0:
                # First point is the first picM node along Dwall_top
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_top[0]][0],imp_elems[indsurf_Dwall_top[0]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_top[0]][0],imp_elems[indsurf_Dwall_top[0]][1],1] 
                re_Dwall_top_ion[ind_er] = rs[eta_max,0] + h_ion_m[indsurf_Dwall_top[0]]*norm_r
                ze_Dwall_top_ion[ind_er] = zs[eta_max,0] + h_ion_m[indsurf_Dwall_top[0]]*norm_z
            elif ind_er == np_erosion_top-1:
                # Last point is the last picM node along Dwall_top
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][0],imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][0],imp_elems[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]][1],1] 
                norm_r = 1.0
                norm_z = 0.0
                re_Dwall_top_ion[ind_er] = rs[eta_max,xi_top] + h_ion_m[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*norm_r
                ze_Dwall_top_ion[ind_er] = zs[eta_max,xi_top] + h_ion_m[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*norm_z
            else:
                # Points in the middle correspond to central points of picS surface elements along Dwall_top
                norm_r = -norm_vers[imp_elems[indsurf_Dwall_top[ind_er-1]][0],imp_elems[indsurf_Dwall_top[ind_er-1]][1],0] 
                norm_z = -norm_vers[imp_elems[indsurf_Dwall_top[ind_er-1]][0],imp_elems[indsurf_Dwall_top[ind_er-1]][1],1] 
                re_Dwall_top_ion[ind_er] = rsurf_Dwall_top[ind_er-1] + h_ion_m[indsurf_Dwall_top[ind_er-1]]*norm_r
                ze_Dwall_top_ion[ind_er] = zsurf_Dwall_top[ind_er-1] + h_ion_m[indsurf_Dwall_top[ind_er-1]]*norm_z
                
                
        # Obtain the erosion functions for plotting
        Y0_E_plot   = erosion_Y0(c_vec[k],E_th_vec[k],xvec_E,nxvec)
        Ftheta_plot = erosion_Ftheta(Fmax_vec[k],theta_max_vec[k],a_vec[k],xvec_theta,nxvec)

        # Currents in A/cm2
        ji_surf_plot     = ji_surf_plot*1E-4
        ji1_surf_plot    = ji1_surf_plot*1E-4
        ji2_surf_plot    = ji2_surf_plot*1E-4
        df_ji1_surf_plot = df_ji1_surf_plot*1E-4
        df_ji2_surf_plot = df_ji2_surf_plot*1E-4
        df_ji_surf_plot  = df_ji_surf_plot*1E-4
        # Impact energies in eV
        imp_ene_i1_surf_plot  = imp_ene_i1_surf_plot/e
        imp_ene_i2_surf_plot  = imp_ene_i2_surf_plot/e
        imp_ene_n1_surf_plot  = imp_ene_n1_surf_plot/e
        imp_ene_ion_surf_plot = imp_ene_i1_surf_plot*(ji1_surf_plot/ji_surf_plot) + imp_ene_i2_surf_plot*(ji2_surf_plot/ji_surf_plot)
        
        
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
        zs              = zs*1E2
        rs              = rs*1E2
        re_Dwall_bot_ion    = re_Dwall_bot_ion*1E2
        ze_Dwall_bot_ion    = ze_Dwall_bot_ion*1E2
        re_Dwall_top_ion    = re_Dwall_top_ion*1E2
        ze_Dwall_top_ion    = ze_Dwall_top_ion*1E2
        
        print("Operation time [h]                 = "+str(t_op))
        print("h at chamber exit bot [mm]         = %15.8e" %(h_mum[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]]*1E-3))
        print("h at chamber exit top [mm]         = %15.8e" %(h_mum[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]]*1E-3))
        print("r orig at chamber exit bot [mm]    = %15.8e" %(rs[eta_min,xi_bottom]*1E1))
        print("r orig at chamber exit top [mm]    = %15.8e" %(rs[eta_max,xi_top]*1E1))
        print("r eroded at chamber exit bot [mm]  = %15.8e" %(re_Dwall_bot[-1]*1E1))
        print("r eroded at chamber exit top [mm]  = %15.8e" %(re_Dwall_top[-1]*1E1))
        print("dr eroded at chamber exit bot [mm] = %15.8e" %(np.abs(re_Dwall_bot[-1] - rs[eta_min,xi_bottom])*1E1))
        print("dr eroded at chamber exit top [mm] = %15.8e" %(np.abs(re_Dwall_top[-1] - rs[eta_max,xi_top])*1E1))
        
        
        # Axial profile plots
        if plot_Dwall == 1:
            if plot_sp_yield == 1:
                plt.figure(r'Y0(E) Dwall_bot_top')
                plt.plot(xvec_E,Y0_E_plot, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_erosion, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'F(theta) Dwall_bot_top')
                plt.plot(xvec_theta,Ftheta_plot, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_erosion, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            if plot_erosion == 1:
                plt.figure(r'dhdt Dwall_bot_top')
#                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_i1_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" bot i1")
#                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_i1_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top i1")
#                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_i2_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k]+" bot i2")
#                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_i2_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k]+" top i2")
#                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],dhdt_ion_mumh[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='g', markeredgecolor = 'k', label=labels[k]+" bot ion")
#                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],dhdt_ion_mumh[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='g', markeredgecolor = 'k', label=labels[k]+" top ion")
                
                plt.figure(r'h(z) Dwall_bot_top')
                val = 1E-3
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],val*h_mum[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],val*h_mum[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'h(z)/hexit Dwall_bot_top')
                plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],h_mum[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]]/h_mum[indsurf_Dwall_bot[indsurf_inC_Dwall_bot[-1]]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],h_mum[indsurf_Dwall_top[indsurf_inC_Dwall_top]]/h_mum[indsurf_Dwall_top[indsurf_inC_Dwall_top[-1]]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                
                plt.figure(r'eroded chamber Dwall_bot_top')
                # Chamber walls before erosion
                np_ptop = 3
                np_pbot = 2
                if k == 0:
#                    plt.plot(zs[eta_min:eta_max+1,0],rs[eta_min:eta_max+1,0], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='k', markeredgecolor = 'k', label=labels[k]+" $t_{op} = 0$")
                    plt.plot(zs[eta_min:eta_max+1,0],rs[eta_min:eta_max+1,0], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='k', markeredgecolor = 'k', label=" $t_{op} = 0$")
                    plt.plot(zs[eta_min,0:xi_bottom+1],rs[eta_min,0:xi_bottom+1], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='k', markeredgecolor = 'k', label="")
                    plt.plot(zs[eta_max,0:xi_top+1],rs[eta_max,0:xi_top+1], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='k', markeredgecolor = 'k', label="")
                    plt.plot(zs[eta_max:eta_max+np_ptop,xi_top],rs[eta_max:eta_max+np_ptop,xi_top], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='k', markeredgecolor = 'k', label="")
                    plt.plot(zs[eta_min-np_pbot:eta_min+1,xi_bottom],rs[eta_min-np_pbot:eta_min+1,xi_bottom], linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='k', markeredgecolor = 'k', label="")
#                plt.plot(ze_Dwall_bot,re_Dwall_bot, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels[k]+" $t_{op} = $"+str(t_op)+" h")
                plt.plot(ze_Dwall_bot,re_Dwall_bot, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels[k])
                plt.plot(ze_Dwall_top,re_Dwall_top, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label="")
#                plt.plot(ze_Dwall_bot_ion,re_Dwall_bot_ion, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+2], markeredgecolor = 'k', label=labels[k]+" $t_{op} = $"+str(t_op)+" h")
#                plt.plot(ze_Dwall_top_ion,re_Dwall_top_ion, linestyle='-', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+2], markeredgecolor = 'k', label="")

                
        
        ind = ind + 1
        if ind > 6:
            ind = 0
            ind2 = ind2 + 1
            if ind2 > 6:
                ind = 0
                ind2 = 0
                ind3 = ind3 + 1


    if plot_Dwall == 1:
        if plot_sp_yield == 1:
            plt.figure(r'Y0(E) Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
#            ax.set_xlim(xlims[0],3.5)

            plt.figure(r'F(theta) Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
            ax.set_xlim(0,90)
        if plot_erosion == 1:
            plt.figure(r'dhdt Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
            ax.set_xlim(xlims[0],3.5)
            
            plt.figure(r'h(z) Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
            ax.set_xlim(xlims[0],3.5)
            
            plt.figure(r'h(z)/hexit Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
#            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2)
            xlims = ax.get_xlim()
            ax.set_xlim(xlims[0],3.5)
            
            plt.figure(r'eroded chamber Dwall_bot_top')
            plt.legend(fontsize = font_size_legend,loc=10)
            ax = plt.gca()
            ylims = ax.get_ylim()
            xlims = ax.get_xlim()
#            plt.gca().set_aspect('equal', adjustable='box')
#            plt.gca().set_aspect('equal')
            ax.set_xlim(xlims[0],3.5)
            ax.set_ylim(ylims[0],9.0)
            ax.set_xticks([0.0,0.5,1.0,1.5,2,2.5,3.0,3.5])
            ax.set_yticks([4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.xaxis.grid(True, which='minor')
            ax.yaxis.grid(True, which='minor')
            ax.set_xlim(1.5,3.5)
            
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
    

            
    if save_flag == 1:
        if plot_Dwall == 1:
            if plot_sp_yield == 1:
                plt.figure(r'Y0(E) Dwall_bot_top')
                plt.savefig(path_out+"Y0E_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'F(theta) Dwall_bot_top')
                plt.savefig(path_out+"Ftheta_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
            if plot_erosion == 1:
                plt.figure(r'dhdt Dwall_bot_top')
                plt.savefig(path_out+"dhdt_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'h(z) Dwall_bot_top')
                plt.savefig(path_out+"hzerosion_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'h(z)/hexit Dwall_bot_top')
                plt.savefig(path_out+"h_hmax_erosion_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'eroded chamber Dwall_bot_top')
                plt.savefig(path_out+"eroded_chamber_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'points eroded chamber Dwall_bot_top')
                plt.savefig(path_out+"points_eroded_chamber_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()

                
