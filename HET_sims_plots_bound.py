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
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
mesh_plots          = 0
bf_plots            = 1

#path_out = "CHEOPS_Final_figs/"
#path_out = "CHEOPS_second_progress_meeting/T2N4_comparison_boundaryalgorithms_Ec65/cases_71d0dcb_c/bound_plots/"
#path_out = "CHEOPS_second_progress_meeting/T2N4_comparison_REFfloating_Ec65/new_large_wall_case/bound_plots/"
#path_out = "CHEOPS_second_progress_meeting/T2N4_comparison_REF_GDML_GDMLWC_Ec65/WC_cases_with_energy_eq_type_2/GDML_with_free_qe_fact_2/bound_plots/"
#path_out = "CHEOPS_second_progress_meeting/T2N4_comparison_REF_GDML_GDMLWCmc0_GDMLWC_Ec65/WC_cases_with_energy_eq_type_2/GDML_with_free_qe_fact_2/bound_plots/"
#path_out = "CHEOPS_second_progress_meeting/T2N4_comparison_GDMLWC_phiinf_Ec65/bound_plots/"
#path_out = "CHEOPS_second_progress_meeting/T2N4_comparison_GDMLWC_phiinfminus_Ec65/"
#path_out = "CHEOPS_second_progress_meeting/T2N4_comparison_GDMLWC_phiinfplus_Ec65/"
#path_out = "T2N4_CSL_29032022/dphi_sh_Te_ratio_csl_1/bound_plots/"
#path_out = "VHT_LP_US/testcase2/PPSX00_em1_OP2c_tmte08_2_tq1_fcat203_CEX/"
#path_out = "VHT_LP_US/testcase2/comp_OP2c_CEX_Xe_Kr_Kr2/"
#path_out = "VHT_LP_US/testcase2/comp_OP2c_CEX_Xe_em2_em3/"
#path_out = "VHT_LP_US/testcase2/comp_2c_2f_2h_2hKr/"
#path_out = "VHT_LP_US/testcase2/comp_2c_2a_2b/"
# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/bound_plots_P1/"
# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/bound_plots_P2/"
path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/bound_plots_P3/"
# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/bound_plots_P4/"
# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/comparison_P3G_fcat1962_alphat/bound_plots_sigma_freeqefact_SEEfl/"
# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/cathode_cases_P2P3/P3_bound_plots/"
# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/cathode_cases_P2P3/P2_bound_plots_fcat3198/"
# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/bound_plots_P1P4/"
# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/cathode_cases_P4/P4_bound_plots_fcat7610/"
# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/cathode_cases_P3_Tcath_new/bound_plots_fcat1962/"
# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/cathode_cases_P3_Tcath_new/bound_plots_fcat6259_5993/"


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
marker_size            = 2
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
    


if mesh_plots == 1:
    print("######## mesh_plots ########")
    # Print out time step
    timestep = 'last'
    allsteps_flag   = 1

    if timestep == 'last':
        timestep = -1  
    if allsteps_flag == 0:
        mean_vars = 0
        
    line_width_boundary = 1
    line_width_bf       = 0.7
    line_width          = 1
    marker_size         = 3
    marker_size_bf      = 5
    marker_size_pic_bn  = 6
    
    # Simulation names
    nsims = 1
    oldpost_sim      = 3
    oldsimparams_sim = 8
    
#    sim_names = ["../../../Sr_sims_files/SPT100_orig_tmtetq2_Vd300_test_rel"]   
    sim_names = [
#                 "../../../Sr_sims_files/Topo1_n1_l100s100_cat313_tm515_te1_tq21_last",
#                 "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_Kr",
                 
#                 "../../../Ca_sims_files/T2N4_pm1em1_cat1200_tm15_te1_tq125_71d0dcb_Es50",
                 "../../../Rb_sims_files/Topo2_n3_l200s200_cat1200_tm15_te1_tq125",
                 "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_71d0dcb",

                 
                 ]

    
    
#    path_picM         = sim_names[0]+"/SET/inp/SPT100_picM.hdf5"
#    path_picM         = sim_names[0]+"/SET/inp/PIC_mesh_topo1_refined4.hdf5"
    path_picM         = sim_names[0]+"/SET/inp/PIC_mesh_topo2_refined4.hdf5"
#    path_picM         = sim_names[0]+"/SET/inp/SPT100_picM_Reference1500points_rm.hdf5"
    path_simstate_inp = sim_names[0]+"/CORE/inp/SimState.hdf5"
    path_simstate_out = sim_names[0]+"/CORE/out/SimState.hdf5"
    path_postdata_out = sim_names[0]+"/CORE/out/PostData.hdf5"
    path_simparams_inp = sim_names[0]+"/CORE/inp/sim_params.inp"
    
    [points,zs,rs,zscells,rscells,dims,nodes_flag,cells_flag,cells_vol,
       volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,eta_max,eta_min,xi_top,
       xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,nsteps,
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
       sFLwall_lat,sAxis,sc_bot,sc_top,
       
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
       sAwall_surf,sFLwall_ver_surf,sFLwall_lat_surf,
       
       delta_r,delta_s,dphi_sh_b,je_b,ji_tot_b,gp_net_b,ge_sb_b,relerr_je_b,
       qe_tot_wall,qe_tot_s_wall,qe_tot_b,qe_b,qe_b_bc,qe_b_fl,relerr_qe_b,
       relerr_qe_b_cons,Te,phi,err_interp_phi,err_interp_Te,
       err_interp_jeperp,err_interp_jetheta,err_interp_jepara,
       err_interp_jez,err_interp_jer,n_inst,ni1_inst,ni2_inst,nn1_inst,
       
       delta_r_nodes,delta_s_nodes,dphi_sh_b_nodes,je_b_nodes,
       gp_net_b_nodes,ge_sb_b_nodes,relerr_je_b_nodes,qe_tot_wall_nodes,
       qe_tot_s_wall_nodes,qe_tot_b_nodes,qe_b_nodes,qe_b_bc_nodes,
       qe_b_fl_nodes,relerr_qe_b_nodes,relerr_qe_b_cons_nodes,Te_nodes,
       phi_nodes,err_interp_n_nodes,n_inst_nodes,ni1_inst_nodes,ni2_inst_nodes,
       nn1_inst_nodes,n_nodes,ni1_nodes,ni2_nodes,nn1_nodes,dphi_kbc_nodes,
       MkQ1_nodes,ji1_nodes,ji2_nodes,ji_nodes,gn1_tw_nodes,gn1_fw_nodes,
       qi1_tot_wall_nodes,qi2_tot_wall_nodes,qi_tot_wall_nodes,qn1_tw_nodes,
       qn1_fw_nodes,imp_ene_i1_nodes,imp_ene_i2_nodes,imp_ene_n1_nodes,
       
       delta_r_surf,delta_s_surf,dphi_sh_b_surf,je_b_surf,gp_net_b_surf,
       ge_sb_b_surf,relerr_je_b_surf,qe_tot_wall_surf,qe_tot_s_wall_surf,
       qe_tot_b_surf,qe_b_surf,qe_b_bc_surf,qe_b_fl_surf,relerr_qe_b_surf,
       relerr_qe_b_cons_surf,Te_surf,phi_surf,nQ1_inst_surf,nQ1_surf,
       nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,MkQ1_surf,ji1_surf,ji2_surf,
       ji_surf,gn1_tw_surf,gn1_fw_surf,qi1_tot_wall_surf,qi2_tot_wall_surf,
       qi_tot_wall_surf,qn1_tw_surf,qn1_fw_surf,imp_ene_i1_surf,
       imp_ene_i2_surf,imp_ene_n1_surf] = HET_sims_read_bound(path_simstate_inp,path_simstate_out,path_postdata_out,path_simparams_inp,
                                                              path_picM,allsteps_flag,timestep,oldpost_sim,oldsimparams_sim)
        
        
    if (n_faces_boundary == nfaces_Dwall_bot+nfaces_Dwall_top+nfaces_Awall+nfaces_FLwall_ver+nfaces_FLwall_lat+nfaces_Axis):
        print("Boundary faces read correctly")
        print("n_boundary_faces  = "+str(n_faces_boundary))
        print("nfaces_Dwall_bot  = "+str(nfaces_Dwall_bot))
        print("nfaces_Dwall_top  = "+str(nfaces_Dwall_top))
        print("nfaces_Awall      = "+str(nfaces_Awall))
        print("nfaces_FLwall_ver = "+str(nfaces_FLwall_ver))
        print("nfaces_FLwall_lat = "+str(nfaces_FLwall_lat))
        print("nfaces_Axis       = "+str(nfaces_Axis))
        
    
    # Plotting part -----------------------------------------------------------
    
    # Dimensions in cm
    zs = zs*1E2
    rs = rs*1E2
    points            = points*1E2
    zfaces_Dwall_bot  = zfaces_Dwall_bot*1E2
    rfaces_Dwall_bot  = rfaces_Dwall_bot*1E2
    zfaces_Dwall_top  = zfaces_Dwall_top*1E2
    rfaces_Dwall_top  = rfaces_Dwall_top*1E2
    zfaces_Awall      = zfaces_Awall*1E2
    rfaces_Awall      = rfaces_Awall*1E2
    zfaces_FLwall_ver = zfaces_FLwall_ver*1E2
    rfaces_FLwall_ver = rfaces_FLwall_ver*1E2
    zfaces_FLwall_lat = zfaces_FLwall_lat*1E2
    rfaces_FLwall_lat = rfaces_FLwall_lat*1E2
    zfaces_Axis       = zfaces_Axis*1E2
    rfaces_Axis       = rfaces_Axis*1E2
    zsurf_Dwall_bot   = zsurf_Dwall_bot*1E2
    rsurf_Dwall_bot   = rsurf_Dwall_bot*1E2
    zsurf_Dwall_top   = zsurf_Dwall_top*1E2
    rsurf_Dwall_top   = rsurf_Dwall_top*1E2
    zsurf_Awall       = zsurf_Awall*1E2
    rsurf_Awall       = rsurf_Awall*1E2
    zsurf_FLwall_ver  = zsurf_FLwall_ver*1E2
    rsurf_FLwall_ver  = rsurf_FLwall_ver*1E2
    zsurf_FLwall_lat  = zsurf_FLwall_lat*1E2
    rsurf_FLwall_lat  = rsurf_FLwall_lat*1E2
    
    zs_plot = np.copy(zs)
    rs_plot = np.copy(rs)
    for i in range(0,dims[0]):
        for j in range(0,dims[1]):
            if nodes_flag[i,j] == 0:
                rs_plot[i,j] = np.NaN
                zs_plot[i,j] = np.NaN
        
    plt.figure("PIC mesh and MFAM boundary faces")
    plt.xlabel(r"$z$ (cm)",fontsize = font_size)
    plt.ylabel(r"$r$ (cm)",fontsize = font_size)
    plt.title(r"PIC mesh",fontsize = font_size)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    plt.plot(zs_plot,rs_plot,'ko-',linewidth=line_width,markersize = marker_size)
    plt.plot(zs_plot.transpose(),rs_plot.transpose(),'ko-',linewidth=line_width,markersize = marker_size)
    # Plot points defining mesh boundary
    # For case without chamfer 
    if len(points) == 8:
        # Inner wall
        plt.plot(points[0:3,0],points[0:3,1],'ro-',linewidth=line_width_boundary,markersize = marker_size)
        # Axis r = 0
        plt.plot(points[2:4,0],points[2:4,1],'mo-',linewidth=line_width_boundary,markersize = marker_size)
        # Free loss
        plt.plot(points[3:6,0],points[3:6,1],'bo-',linewidth=line_width_boundary,markersize = marker_size)
        # Outer wall
        plt.plot(points[5::,0],points[5::,1],'ro-',linewidth=line_width_boundary,markersize = marker_size)
        # Injection
        plt.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'go-',linewidth=line_width_boundary,markersize = marker_size)
    # For case with chamfer in bottom and top walls
    if len(points) == 10:
        # Inner wall
        plt.plot(points[0:4,0],points[0:4,1],'ro-',linewidth=line_width_boundary,markersize = marker_size)
        # Axis r = 0
        plt.plot(points[3:5,0],points[3:5,1],'mo-',linewidth=line_width_boundary,markersize = marker_size)
        # Free loss
        plt.plot(points[4:7,0],points[4:7,1],'bo-',linewidth=line_width_boundary,markersize = marker_size)
        # Outer wall
        plt.plot(points[6::,0],points[6::,1],'ro-',linewidth=line_width_boundary,markersize = marker_size)
        # Injection
        plt.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'go-',linewidth=line_width_boundary,markersize = marker_size)
    # Plot all mesh boundary points and domain points
    for i in range(0,dims[0]):
        for j in range(0,dims[1]):
            if nodes_flag[i,j] == -1:
                if i == 0:
                    # Axis r = 0
                    plt.plot(zs[i,j],rs[i,j],'mo',markeredgecolor='m',markersize = marker_size)
                elif j == dims[1] - 1 or i == dims[0] - 1:
                    # Free loss
                    plt.plot(zs[i,j],rs[i,j],'bo',markeredgecolor='b',markersize = marker_size)
                elif j == xi_top or i == eta_max or j == xi_bottom or i == eta_min:
                    # Material wall
                    plt.plot(zs[i,j],rs[i,j],'ro',markeredgecolor='r',markersize = marker_size)
                elif j == 0:
                    # Injection
                    plt.plot(zs[i,j],rs[i,j],'go',markeredgecolor='g',markersize = marker_size)
            if nodes_flag[i,j] == 1:
                plt.plot(zs[i,j],rs[i,j],'ko',markersize = marker_size)
#    # Plot the boundary faces centers for each type of boundary
#    # Dwall_bot
#    plt.plot(zfaces_Dwall_bot,rfaces_Dwall_bot,'cx-',linewidth = line_width_bf,markersize = marker_size_bf)
#    # Dwall_top
#    plt.plot(zfaces_Dwall_top,rfaces_Dwall_top,'cx-',linewidth = line_width_bf,markersize = marker_size_bf)
#    # Awall
#    plt.plot(zfaces_Awall,rfaces_Awall,'mx-',linewidth = line_width_bf,markersize = marker_size_bf)
#    # FLwall_ver
#    plt.plot(zfaces_FLwall_ver,rfaces_FLwall_ver,'yx-',linewidth = line_width_bf,markersize = marker_size_bf)
#    # FLwall_lat
#    plt.plot(zfaces_FLwall_lat,rfaces_FLwall_lat,'x-',color = orange,linewidth = line_width_bf,markersize = marker_size_bf)
#    # Axis
#    plt.plot(zfaces_Axis,rfaces_Axis,'x-',color = brown,linewidth = line_width_bf,markersize = marker_size_bf)
#    # Plot the PIC mesh boundary nodes for each type of boundary
#    # Dwall_bot
#    plt.plot(zs[inodes_Dwall_bot,jnodes_Dwall_bot],rs[inodes_Dwall_bot,jnodes_Dwall_bot],'rD',markersize = marker_size_pic_bn)
#    # Dwall_top
#    plt.plot(zs[inodes_Dwall_top,jnodes_Dwall_top],rs[inodes_Dwall_top,jnodes_Dwall_top],'rD',markersize = marker_size_pic_bn)
#    # Awall
#    plt.plot(zs[inodes_Awall,jnodes_Awall],rs[inodes_Awall,jnodes_Awall],'gP',markersize = marker_size_pic_bn)
#    # FLwall_ver
#    plt.plot(zs[inodes_FLwall_ver,jnodes_FLwall_ver],rs[inodes_FLwall_ver,jnodes_FLwall_ver],'bp',markersize = marker_size_pic_bn)
#    # FLwall_lat
#    plt.plot(zs[inodes_FLwall_lat,jnodes_FLwall_lat],rs[inodes_FLwall_lat,jnodes_FLwall_lat],'cp',markersize = marker_size_pic_bn)
#    # Axis
#    plt.plot(zs[inodes_Axis,jnodes_Axis],rs[inodes_Axis,jnodes_Axis],'mp',markersize = marker_size_pic_bn)
    # Plot the PIC mesh boundary surface elements centers for each type of boundary
    # Dwall_bot
    plt.plot(zsurf_Dwall_bot,rsurf_Dwall_bot,'ko',markersize = marker_size)
    # Dwall_top
    plt.plot(zsurf_Dwall_top,rsurf_Dwall_top,'ko',markersize = marker_size)
    # Awall
    plt.plot(zsurf_Awall,rsurf_Awall,'ko',markersize = marker_size)
    # FLwall_ver
    plt.plot(zsurf_FLwall_ver,rsurf_FLwall_ver,'ko',markersize = marker_size)
    # FLwall_lat
    plt.plot(zsurf_FLwall_lat,rsurf_FLwall_lat,'ko',markersize = marker_size)
    
    ax = plt.gca()
    ax.set_xlim(1.5,3.5)
    ax.set_ylim(3,6.5)
    
    if save_flag == 1:
        plt.figure("PIC mesh and MFAM boundary faces")
        plt.savefig(path_out+"pic_mfam_bound"+figs_format,bbox_inches='tight')
        plt.close() 
        
    
    
if bf_plots == 1:
    print("######## bf_plots ########")
          
    order           = 50
    order_fast      = 500
    
    # Print out time steps
    timestep = 'last'
    
    marker_size         = 5
    marker_every        = 1
    marker_every_mfambf = 5
  
    allsteps_flag   = 1
    
    mean_vars       = 1
    mean_type       = 0
    last_steps      = 1200
    # last_steps      = 1000
    step_i          = 1
    step_f          = 2
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
    
    plot_wall      = 0
    plot_down      = 1
    plot_Dwall     = 0
    plot_Awall     = 0
    plot_FLwall    = 0
    plot_Axis      = 0
    
    plot_B               = 0
    plot_dens            = 1
    plot_deltas          = 1
    plot_dphi_Te         = 1
    plot_curr            = 1
    plot_q               = 1
    plot_imp_ene         = 1
    plot_err_interp_mfam = 0
    plot_err_interp_pic  = 0
    plot_picM_picS_comp  = 0
    
    # plot_B               = 1
    # plot_dens            = 1
    # plot_deltas          = 0
    # plot_dphi_Te         = 1
    # plot_curr            = 1
    # plot_q               = 1
    # plot_imp_ene         = 0
    # plot_err_interp_mfam = 0
    # plot_err_interp_pic  = 0
    # plot_picM_picS_comp  = 0
    
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
    
    # plot_B               = 0
    # plot_dens            = 0
    # plot_deltas          = 0
    # plot_dphi_Te         = 1
    # plot_curr            = 0
    # plot_q               = 1
    # plot_imp_ene         = 1
    # plot_err_interp_mfam = 0
    # plot_err_interp_pic  = 0
    # plot_picM_picS_comp  = 0
    

    if timestep == 'last':
        timestep = -1  
    if allsteps_flag == 0:
        mean_vars = 0
        
        
    # Simulation names
    nsims = 2
    oldpost_sim      = np.array([3,5,3,3,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([8,15,8,8,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)         
   
    oldpost_sim      = np.array([6,6,5,5,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([16,17,15,15,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)   
    
    oldpost_sim      = np.array([5,5,5,5,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([15,15,15,15,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)   
    
    oldpost_sim      = np.array([6,6,6,6,6,6,6,6,5,5,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([21,21,20,20,20,20,20,20,17,17,15,15,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)   
    
    
    sim_names = [
        
                # "../../../sim/sims/P4G_fcat7610_Fz",
                # "../../../sim/sims/P4L_fcat7610_Fz",
        
                "../../../sim/sims/P3G_Tcath_new",
                "../../../sim/sims/P3L_Tcath_new",

                # "../../../sim/sims/P3G_fcat1962_Tcath_new",
                # "../../../sim/sims/P3L_fcat1962_Tcath_new",
                
                # "../../../sim/sims/P3G_fcat6259_5993_Tcath_new",
                # "../../../sim/sims/P3L_fcat6259_5993_Tcath_new",
        
                # "../../../sim/sims/P2G_fcat3198",
                # "../../../sim/sims/P2L_fcat3198",
                
                # "../../../sim/sims/P2G_fcat905",
                # "../../../sim/sims/P2L_fcat905",
                
                # "../../../sim/sims/P1G",
                # "../../../sim/sims/P2G",
                # "../../../sim/sims/P3G",
                # "../../../sim/sims/P4G",
                
                # "../../../sim/sims/P1L",
                # "../../../sim/sims/P2L",
                # "../../../sim/sims/P3L",
                # "../../../sim/sims/P4L",
                

        
        
                # "../../../sim/sims/P3G",
                # # "../../../sim/sims/P3G_fcat3608",
                # # "../../../sim/sims/P3G_fcat1003",
                # "../../../sim/sims/P3G_fcat1962",
                # # "../../../sim/sims/P3G_fcat1962_alphat2",
                # "../../../sim/sims/P3G_fcat1962_alphat5",
                # # "../../../sim/sims/P3G_fcat1962_alphat10",
                # "../../../sim/sims/P3G_fcat1962_alphat5_sig03",
                # "../../../sim/sims/P3G_fcat1962_alphat5_freeqefact4",
                # "../../../sim/sims/P3G_fcat1962_alphat5_SEEfl",

                # "../../../sim/sims/P1G",
                # "../../../sim/sims/P1L",
                
                # "../../../sim/sims/P2G",
                # "../../../sim/sims/P2L",
        
                    # "../../../sim/sims/P3G",
                    # "../../../sim/sims/P3L",
                 
                  # "../../../sim/sims/P4G",
                  # "../../../sim/sims/P4L",
        
                # "../../../Mg_hyphen/sim/sims/Plume10_OP3_global_CEX_Np_new",
                # "../../../Mg_hyphen/sim/sims/Plume10_OP3_local_CEX_Np_new",
            
#                "../../../Mg_hyphen/sim/sims/Plume20_OP3_global_CEX_Np_new",
#                "../../../Mg_hyphen/sim/sims/Plume20_OP3_local_CEX_Np_new",
            
#                "../../../Mg_hyphen/sim/sims/Plume30_OP3_global_CEX_Np_new",
#                "../../../Mg_hyphen/sim/sims/Plume30_OP3_local_CEX_Np_new",
                
#                "../../../Mg_hyphen/sim/sims/Plume40_OP3_global_CEX_Np_new",
#                "../../../Mg_hyphen/sim/sims/Plume40_OP3_local_CEX_Np_new",
            
            
#                 "../../../Ca_hyphen/sim/sims/PPSX00_em2_OP2c_tmte08_2_tq1_fcat4656_CEX",
#                 "../../../Ca_hyphen/sim/sims/PPSX00_em2_OP2f_tmte08_2_tq1_fcat4656_CEX",
#                 "../../../Mg_hyphen_borja/sim/sims/PPSX00_em2_OP2h_tmte06_2_tq1_fcat4656_CEX",
#                 "../../../Mg_hyphen_borja/sim/sims/PPSX00_em2_OP2h_tmte06_2_tq1_fcat4656_CEX_Kr",
                 
                 "../../../Ca_hyphen/sim/sims/PPSX00_em2_OP2c_tmte08_2_tq1_fcat4656_CEX",
                 "../../../Mg_hyphen_borja/sim/sims/PPSX00_em2_OP2a_tm2_2_te52_tq1_fcat4656_CEX",
                 "../../../Mg_hyphen_borja/sim/sims/PPSX00_em2_OP2b_tm2_2_te3_tq1_fcat4656_CEX",
                 
                 
#                 "../../../H_sims/Mg/hyphen/sims/CHEOPS_MP/VUS_OP3",
#                 "../../../Mg_hyphen_borja/sim/sims/PPSX00_em1_OP2c_tmte08_2_tq1_fcat203_CEX",
                 "../../../Mg_hyphen_borja/sim/sims/PPSX00_em2_OP2c_tmte08_2_tq1_fcat6113_CEX",
                 "../../../Mg_hyphen_borja/sim/sims/PPSX00_em2_OP2c_tmte08_2_tq1_fcat6113_CEX_Kr",
                 "../../../Mg_hyphen_borja/sim/sims/PPSX00_em2_OP2c_tmte08_2_tq1_fcat6113_CEX_Kr2",
                 
                 "../../../Mg_hyphen_borja/sim/sims/PPSX00_em2_OP2c_tmte08_2_tq1_fcat6113_CEX",
                 "../../../Mg_hyphen_borja/sim/sims/PPSX00_em3_OP2c_tmte08_2_tq1_fcat3384_CEX",
            
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume10_OP3_local_CEX_Np",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume10_OP3_global_CEX_Np",
#                "../../../Mg_hyphen_alejandro/sim/sims/Plume20_OP3_local_CEX_Np",
                "../../../Mg_hyphen_alejandro/sim/sims/Plume20_OP3_global_CEX_Np",
                "../../../Mg_hyphen_alejandro/sim/sims/Plume30_OP3_local_CEX_Np",
                "../../../Mg_hyphen_alejandro/sim/sims/Plume30_OP3_global_CEX_Np",
                "../../../Mg_hyphen_alejandro/sim/sims/Plume40_OP3_local_CEX_Np",
                "../../../Mg_hyphen_alejandro/sim/sims/Plume40_OP3_global_CEX_Np",
                
                
                "../../../sim/sims/HT20k_tm1.4_23tq25s01_V300_rm8_rm6_global_new",
            
#                  "../../../H_sims/Ca/hyphen/sims/T2N4_pm1em1_cat1200_tm15_te1_tq125_71d0dcb",
#                  "../../../Sr_hyphen/sim/sims/T2N4_pm1em1_cat1200_tm15_te1_tq125_0438e2a",
#                "../../../Sr_hyphen/sim/sims/T2N4_pm1em1_cat1200_tm15_te1_tq125_0438e2a_Es65",
                "../../../Sr_hyphen/sim/sims/T2N4_pm1em1_cat1200_tm15_te1_tq125_REF",
#                "../../../Sr_hyphen/sim/sims/T2N4_pm1em1_cat1200_tm15_te1_tq125_floating",
                "../../../Sr_hyphen/sim/sims/T2N4_pm1em1_cat1200_tm15_te1_tq125_GDML",
                "../../../Sr_hyphen/sim/sims/T2N4_pm1em1_WC1959_tm15_te1_tq125_GDML_mc0",
                "../../../Sr_hyphen/sim/sims/T2N4_pm1em1_WC1959_tm15_te1_tq125_GDML",
#                "../../../Ca_hyphen/sim/sims/T2N4_pm1em1_WC1959_tm15_te1_tq125_GDML_mc0_Temin",
#                "../../../Ca_hyphen/sim/sims/T2N4_pm1em1_WC1959_tmte15_tq125_GDML_mc0_Temin",
#                "../../../Ca_hyphen/sim/sims/T2N4_pm1em1_WC1959_tm15_te1_tq125_GDML_phiinf",
#                "../../../Ca_hyphen/sim/sims/T2N4_pm1em1_WC1959_tm15_te1_tq125_GDML_phiinfplus",
#                "../../../Ca_hyphen/sim/sims/T2N4_pm1em1_WC1959_tm15_te1_tq125_GDML_phiinfminus",
#                "../../../sim/sims/T2N4_pm1em1_WC1959_tm15_te1_tq125_GDML_tests",
                "../../../Ca_hyphen/sim/sims/T2N4_pm1em1_WC1959_tm15_te1_tq125_GDML_CSL",
            
                  "../../../Sr_hyphen/sim/sims/HT5k_pm3em4_cat1966_tm16te1tq25_sig01_7189590_orig",
#                  "../../../Sr_hyphen/sim/sims/HT5k_pm3em4_cat1966_tmtetq25_sig03",
            
            
#                 "../../../Rb_sims_files/Topo1_n1_l100s100_cat313_tm515_te1_tq21",
#                 "../../../Sr_sims_files/T1N1_pm1em1_cat313_tm515_te1_tq21_71d0dcb",
    
#                 "../../../Rb_sims_files/Topo1_n2_l100s100_cat313_tm615_te2_tq12",
#                 "../../../Sr_sims_files/T1N2_pm1em1_cat313_tm615_te2_tq12_71d0dcb",
                 
#                 "../../../Rb_sims_files/Topo2_n3_l200s200_cat1200_tm15_te1_tq125",
                 "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_71d0dcb",
            
#                 "../../../Rb_sims_files/Topo2_n4_l200s200_cat1200_tm15_te1_tq125",
                 "../../../Ca_sims_files/T2N4_pm1em1_cat1200_tm15_te1_tq125_71d0dcb",
            

            
                 "../../../Sr_sims_files/T1N2_pm1em1_cat313_tm615_te2_tq12_71d0dcb",
                 "../../../Sr_sims_files/T1N1_pm1em1_cat313_tm515_te1_tq21_71d0dcb",
            
                 "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_71d0dcb",
#                 "../../../Ca_sims_files/T2N4_pm1em1_cat1200_tm15_te1_tq125_71d0dcb",
                 
                 "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_71d0dcb_Es50",
                 
                 "../../../Ca_sims_files/T2N4_pm1em1_cat1200_tm15_te1_tq125_71d0dcb_Es50",
            
                 "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd300",
                 "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd250",
                 "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd200",
                 "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd150",
                 
                 ]

    
    
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
    labels = [

#                 r"$r_\mathrm{C0}$",
# #                r"$r_\mathrm{C}/r_\mathrm{C0} = 1.2\%$",
# #                r"$r_\mathrm{C}/r_\mathrm{C0} = 2.5\%$",
#                 r"$r_\mathrm{C}/r_\mathrm{C0} = 5.5\%$",
# #                r"$r_\mathrm{C}/r_\mathrm{C0} = 5.5\%$, $2\alpha_\mathrm{t2}$ ",
#                 r"$r_\mathrm{C}/r_\mathrm{C0} = 5.5\%$, $5\alpha_\mathrm{t2}$ ",
# #                r"$r_\mathrm{C}/r_\mathrm{C0} = 5.5\%$, $10\alpha_\mathrm{t2}$ ",
#                 r"$r_\mathrm{C}/r_\mathrm{C0} = 5.5\%$, $5\alpha_\mathrm{t2}$, $\sigma_\mathrm{rp} = 0.3$",
#                 r"$r_\mathrm{C}/r_\mathrm{C0} = 5.5\%$, $5\alpha_\mathrm{t2}$, $P''_\mathrm{ne \infty} = -4T_\mathrm{eP}j_\mathrm{neP}/e$",
#                 r"$r_\mathrm{C}/r_\mathrm{C0} = 5.5\%$, $5\alpha_\mathrm{t2}$, SEE $(T_\mathrm{s},E_\mathrm{s0})= (0.2,25)$ eV",

                r"GP3C1",
                r"LP3C1",
        
               r"P1G",
               r"P2G",
               r"P3G",
               r"P4G",
               r"P1L",
               r"P2L",
               r"P3L",
               r"P4L",
            
#               r"OP1",
#               r"OP2",
#               r"OP4",
#               r"OP5",
               
               r"OP1",
               r"OP6",
               r"OP7",
              
               r"Xe",
               r"Kr",
               r"Kr2",
               
               r"OP2c em2",
               r"OP2c em3",
               r"",
               r"",
               r"",
               r"",
              
               r"HT5k tm1-6te1tq2.5 s0.1 7189590",
#              r"T1N2-REF $E_c = 50$ eV",
#              r"T1N2-REF $E_c = 65$ eV",
              
#              r"T1N1-REF $E_c = 50$ eV",
#              r"T1N1-REF $E_c = 65$ eV",
            
#              r"T2N4-REF $E_c = 50$ eV",
#              r"T2N4-REF $E_c = 65$ eV",
            
#              r"T2N3-REF $E_c = 50$ eV",
#              r"T2N3-REF $E_c = 65$ eV",
              
              

#              r"T1N1 REF",
#              r"T1N2 REF",
              r"T2N3-REF",
              r"T2N4-REF",

    
              r"T2N3-REF, Es50",
              r"T2N4-REF, Es50",
            
              r"T2N3-REF, Es50",
              r"T2N3-REF, Es65",
              r"V1",
              r"V2",
              r"V3",
              ]

    
    # Line colors
    # colors = ['k','r','g','b','m','c','m','y',orange,brown]
    colors = ['k','r','g','b','m','c',orange,silver]
    # colors = ['k','b','r','m','g','c','m','c','m','y',orange,brown] # P1G-P4G, P1L-P4L (paper) cathode cases
#    colors = ['k','m',orange,brown]
    # Markers
#    markers = ['s','o','v','^','<', '>','D','p','*']
    markers = ['','','','','','','','','<', '>','D','p','*']
#    markers = ['s','<','D','p']
    # Line style
    linestyles = ['-','--','-.', ':','-','--','-.']
#    linestyles = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-']
    # linestyles = ['-','-','-','-',':',':',':',':','-','-','-','-','-','-']

#    xaxis_label = r"$s$ (cm)"
    xaxis_label =  r"$s/L_\mathrm{c}$"
#    xaxis_label_down =  r"$s/L_\mathrm{c}$"
    xaxis_label_down =  r"$s/H_\mathrm{c}$"
    
    # Axial profile plots
    if plot_wall == 1:
        if plot_deltas == 1:
            plt.figure(r'delta_r wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\delta_\mathrm{r}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'delta_s wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\delta_\mathrm{s}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'delta_s_csl wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\delta_\mathrm{s,CSL}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
        if plot_dphi_Te == 1:
            plt.figure(r'dphi_sh wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\Delta\phi_\mathrm{sh}$ (V)",fontsize = font_size)
            plt.title(r"$\phi_\mathrm{WQ}$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dphi_sh_Te wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$e\Delta\phi_\mathrm{sh}/T_\mathrm{e}$",fontsize = font_size)
            plt.title(r"$e\phi_\mathrm{WQ}/T_\mathrm{e}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'Te wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$T_\mathrm{e}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'phi wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\phi$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

        if plot_curr == 1:
            plt.figure(r'je_b wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$-j_\mathrm{ne}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji_tot_b wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_\mathrm{ni}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'egn wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$eg_\mathrm{nn}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
#            plt.figure(r'ji1 wall')
#            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$j_{ni1}$ (A/cm$^2$)",fontsize = font_size)
#            plt.xticks(fontsize = ticks_size) 
#            plt.yticks(fontsize = ticks_size)
            
#            plt.figure(r'ji2 wall')
#            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$j_{ni2}$ (A/cm$^2$)",fontsize = font_size)
#            plt.xticks(fontsize = ticks_size) 
#            plt.yticks(fontsize = ticks_size)
            
#            plt.figure(r'ji2/ji1 wall')
#            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$j_{ni2}/j_{ni1}$ (-)",fontsize = font_size)
#            plt.xticks(fontsize = ticks_size) 
#            plt.yticks(fontsize = ticks_size)

#            plt.figure(r'ji1/ji_tot_b wall')
#            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$j_{ni1}/j_{ni}$ (-)",fontsize = font_size)
#            plt.xticks(fontsize = ticks_size) 
#            plt.yticks(fontsize = ticks_size)

#            plt.figure(r'ji2/ji_tot_b wall')
#            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$j_{ni2}/j_{ni}$ (-)",fontsize = font_size)
#            plt.xticks(fontsize = ticks_size) 

            plt.figure(r'j_b wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_\mathrm{n}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji_tot_b je_b wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_\mathrm{ni},\, -j_\mathrm{ne}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
#            plt.figure(r'je_b_gp_net_b wall')
#            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$-j_{ne}/eg_{np}$ (-)",fontsize = font_size)
#            plt.xticks(fontsize = ticks_size) 
#            plt.yticks(fontsize = ticks_size)

            plt.figure(r'relerr_je_b wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\epsilon_{j_\mathrm{ne}}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_imp_ene == 1:
            plt.figure(r'imp_ene_ion wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$\varepsilon_\mathrm{i,wall}$ (eV)",fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{iW}$ (eV)",fontsize = font_size)
#            plt.title(r"$E_\mathrm{iw}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_neu wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$\varepsilon_\mathrm{i,wall}$ (eV)",fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{nW}$ (eV)",fontsize = font_size)
#            plt.title(r"$E_\mathrm{nw}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_e wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$\varepsilon_\mathrm{e,wall}$ (eV)",fontsize = font_size)
            plt.title(r"$\mathcal{E}_\mathrm{eW}$ (eV)",fontsize = font_size)
#            plt.title(r"$E_\mathrm{ew}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
        if plot_q == 1:
            plt.figure(r'qi_tot_wall wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$P''_{ni,wall}$ (W/cm$^2$)",fontsize = font_size)
#            plt.title(r"$P''_{i,wall}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$P''_\mathrm{niW}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qn_tot_wall wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$P''_{nn,wall}$ (W/cm$^2$)",fontsize = font_size)
#            plt.title(r"$P''_{n,wall}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$P''_\mathrm{nnW}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_tot_wall wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$P''_{ne,wall}$ (W/cm$^2$)",fontsize = font_size)
#            plt.title(r"$P''_{e,wall}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$P''_\mathrm{neW}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_tot_b wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$P''_{ne,Q}$ (W/cm$^2$)",fontsize = font_size)
#            plt.title(r"$P''_{e,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$P''_\mathrm{neQ}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'qe_b wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$q_{ne,Q}$ (W/cm$^2$)",fontsize = font_size)
#            plt.title(r"$q_{e,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$q_\mathrm{neQ}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'qe_adv_b wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$-\frac{5}{2}T_e\frac{j_{ne}}{e}|_Q$ (W/cm$^2$)",fontsize = font_size)
#            plt.title(r"$-\frac{5}{2}T_e\frac{j_{e,wall}}{e}|_Q$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$-\frac{5}{2}T_\mathrm{e}\frac{j_\mathrm{ne}}{e}|_\mathrm{Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_qe_b wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\epsilon_{q_\mathrm{ne,Q}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_qe_b_cons wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\epsilon_{q_\mathrm{ne,Q},2}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qi_tot_wall qe_tot_wall wall')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"$P''_{i,wall} \, P''_{e,wall}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$P''_\mathrm{ni} \, P''_\mathrm{ne}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        
    if plot_down == 1:
        if plot_B == 1:
            plt.figure(r'Bfield down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
            plt.ylabel(r"$B$ (G)",fontsize = font_size)
            # plt.title(r"$B$ (G)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
        if plot_dens == 1:
            plt.figure(r'ne down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
            # plt.title(r"$n_\mathrm{e}$ (m$^{-3}$)",fontsize = font_size)
            plt.ylabel(r"$n_\mathrm{e}$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
        if plot_dphi_Te == 1:
            plt.figure(r'dphi_sh down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$\Delta\phi_{\infty}$ (V)",fontsize = font_size)
            # plt.title(r"$\phi_\mathrm{\infty P}$ (V)",fontsize = font_size)
            plt.ylabel(r"$\phi_\mathrm{\infty P}$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dphi_sh_Te down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$e\Delta\phi_{\infty}/T_e$ (-)",fontsize = font_size)
            # plt.title(r"$e\phi_\mathrm{\infty P}/T_\mathrm{eP}$",fontsize = font_size)
            plt.ylabel(r"$e\phi_\mathrm{\infty P}/T_\mathrm{eP}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'Te down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$T_e$ (eV)",fontsize = font_size)
            # plt.title(r"$T_\mathrm{eP}$ (eV)",fontsize = font_size)
            plt.ylabel(r"$T_\mathrm{eP}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'phi down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$\phi$ (V)",fontsize = font_size)
            # plt.title(r"$\phi_\mathrm{P}$ (V)",fontsize = font_size)
            plt.ylabel(r"$\phi_\mathrm{P}$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dphi_inf down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$(\phi-\phi_\infty)$ (V)",fontsize = font_size)
            # plt.title(r"$(\phi_\mathrm{P}-\phi_\infty)$ (V)",fontsize = font_size)
            plt.ylabel(r"$(\phi_\mathrm{P}-\phi_\infty)$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dphi_inf_Te down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$e(\phi-\phi_\infty)/T_e$ (-)",fontsize = font_size)
            # plt.title(r"$e(\phi_\mathrm{P}-\phi_\infty)/T_\mathrm{eP}$",fontsize = font_size)
            plt.ylabel(r"$e(\phi_\mathrm{P}-\phi_\infty)/T_\mathrm{eP}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
        if plot_curr == 1:
            plt.figure(r'j_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$j_{n}$ (A/cm$^2$)",fontsize = font_size)
            # plt.title(r"$j_\mathrm{nP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$j_\mathrm{nP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'je_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$-j_{ne}$ (A/cm$^2$)",fontsize = font_size)
            # plt.title(r"$-j_\mathrm{neP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$-j_\mathrm{neP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
                 
            plt.figure(r'ji_tot_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$j_{ni}$ (A/cm$^2$)",fontsize = font_size)
            # plt.title(r"$j_\mathrm{niP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$j_\mathrm{niP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji_tot_b je_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$j_{ni},\, -j_{ne}$ (A/cm$^2$)",fontsize = font_size)
            # plt.title(r"$j_\mathrm{niP},\, -j_\mathrm{neP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$j_\mathrm{niP},\, -j_\mathrm{neP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'relerr_je_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$\epsilon_{j_{ne}}$ (-)",fontsize = font_size)
            # plt.title(r"$\epsilon_{j_\mathrm{neP}}$",fontsize = font_size)
            plt.ylabel(r"$\epsilon_{j_\mathrm{neP}}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji1 down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$j_{ni}$ (A/cm$^2$)",fontsize = font_size)
            # plt.title(r"$j_\mathrm{niP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$j_\mathrm{ni1P}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji2 down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$j_{ni}$ (A/cm$^2$)",fontsize = font_size)
            # plt.title(r"$j_\mathrm{niP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$j_\mathrm{ni2P}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji3 down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$j_{ni}$ (A/cm$^2$)",fontsize = font_size)
            # plt.title(r"$j_\mathrm{niP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$j_\mathrm{ni3P}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji4 down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$j_{ni}$ (A/cm$^2$)",fontsize = font_size)
            # plt.title(r"$j_\mathrm{niP}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$j_\mathrm{ni4P}$ (Acm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
        if plot_q == 1:
            plt.figure(r'qi_tot_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$P''_{ni,Q}$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$P''_\mathrm{niP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$P''_\mathrm{niP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_tot_wall down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$P''_{ne,wall}$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$P''_\mathrm{ne\infty}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$P''_\mathrm{ne\infty}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_tot_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$P''_{ne,Q}$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$P''_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$P''_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'qe_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$q_{ne,Q}$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$q_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$q_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'qe_adv_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$-\frac{5}{2}T_e\frac{j_{ne}}{e}|_Q$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$-\frac{5}{2}T_\mathrm{e}\frac{j_\mathrm{ne}}{e}|_\mathrm{P}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$-\frac{5}{2}T_\mathrm{e}\frac{j_\mathrm{ne}}{e}|_\mathrm{P}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'ratio_qe_b_qe_adv_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$q_{ne,Q}$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$q_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$q_\mathrm{neP}/\left(-\frac{5}{2}T_\mathrm{e}\frac{j_\mathrm{ne}}{e}|_\mathrm{P}\right)$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'ratio_qe_b_qe_tot_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$q_{ne,Q}$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$q_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$q_\mathrm{neP}/P''_\mathrm{neP}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'ratio_qe_adv_b_qe_tot_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$q_{ne,Q}$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$q_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$-\frac{5}{2}T_\mathrm{e}\frac{j_\mathrm{ne}}{e}|_\mathrm{P}/P''_\mathrm{neP}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'je_dphi_sh down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$-\Delta\phi_{\infty}j_{ne}$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$-\phi_\mathrm{\infty P}j_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$-\phi_\mathrm{\infty P}j_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'ratio_je_dphi_sh_qe_tot_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$-\Delta\phi_{\infty}j_{ne}$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$-\phi_\mathrm{\infty P}j_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$-\phi_\mathrm{\infty P}j_\mathrm{neP}/P''_\mathrm{neP}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'ratio_qe_tot_wall_qe_tot_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$-\Delta\phi_{\infty}j_{ne}$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$-\phi_\mathrm{\infty P}j_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$P''_\mathrm{ne\infty}/P''_\mathrm{neP}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'ratio_eqe_tot_b_je_bTe down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$-\Delta\phi_{\infty}j_{ne}$ (W/cm$^2$)",fontsize = font_size)
            # plt.title(r"$-\phi_\mathrm{\infty P}j_\mathrm{neP}$ (Wcm$^{-2}$)",fontsize = font_size)
            plt.ylabel(r"$-\frac{eP''_\mathrm{neP}}{j_\mathrm{neP}T_\mathrm{eP}}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_qe_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$\epsilon_{q_{ne,Q}}$ (-)",fontsize = font_size)
            # plt.title(r"$\epsilon_{q_\mathrm{neP}}$",fontsize = font_size)
            plt.ylabel(r"$\epsilon_{q_\mathrm{neP}}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_qe_b_cons down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
#            plt.title(r"$\epsilon_{q_{ne,Q},2}$ (-)",fontsize = font_size)
            # plt.title(r"$\epsilon_{q_\mathrm{neP},2}$",fontsize = font_size)
            plt.ylabel(r"$\epsilon_{q_\mathrm{neP},2}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_imp_ene == 1:
            plt.figure(r'imp_ene_ion down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
            plt.ylabel(r"$\mathcal{E}_\mathrm{niP}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_ion1 down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
            plt.ylabel(r"$\mathcal{E}_\mathrm{ni1P}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_ion2 down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
            plt.ylabel(r"$\mathcal{E}_\mathrm{ni2P}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_ion3 down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
            plt.ylabel(r"$\mathcal{E}_\mathrm{ni3P}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_ion4 down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
            plt.ylabel(r"$\mathcal{E}_\mathrm{ni4P}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_e_b down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
            plt.ylabel(r"$\mathcal{E}_\mathrm{neP}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_e_b_Te down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
            plt.ylabel(r"$\mathcal{E}_\mathrm{neP}/T_\mathrm{eP}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_e_wall down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
            plt.ylabel(r"$\mathcal{E}_\mathrm{ne\infty}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_e_wall_Te down')
            plt.xlabel(xaxis_label_down,fontsize = font_size)
            plt.ylabel(r"$\mathcal{E}_\mathrm{ne\infty}/T_\mathrm{eP}$",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        
            
        
    if plot_Dwall == 1:
        if plot_dens == 1:
            plt.figure(r'ne Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$n_e$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ne Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$n_e$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ne Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$n_e$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'nn Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $n_n$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'nn Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $n_n$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'nn Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $n_n$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_deltas == 1:
            plt.figure(r'delta_r Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\delta_r$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'delta_r Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\delta_r$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'delta_r Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\delta_r$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'delta_s Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\delta_s$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'delta_s Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\delta_s$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'delta_s Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\delta_s$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_dphi_Te == 1:
            plt.figure(r'dphi_sh Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\Delta\phi_{sh}$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'dphi_sh Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\Delta\phi_{sh}$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'dphi_sh Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\Delta\phi_{sh}$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dphi_sh_Te Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $e\Delta\phi_{sh}/T_e$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'dphi_sh_Te Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $e\Delta\phi_{sh}/T_e$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'dphi_sh_Te Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$e\Delta\phi_{sh}/T_e$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'Te Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $T_e$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'Te Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $T_e$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'Te Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$T_e$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'phi Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\phi$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'phi Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\phi$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'phi Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$\phi$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_curr == 1:
            plt.figure(r'je_b Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $-j_{en}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'je_b Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $-j_{en}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'je_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $-j_{en}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji_tot_b Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $j_{iW}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji_tot_b Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $j_{iW}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji_tot_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{iW}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji1 Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $j_{i1W}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji1 Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $j_{i1W}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{i1W}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji2 Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $j_{i2W}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji2 Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $j_{i2W}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{i2W}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji2/ji1 Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{i2W}/j_{i1W}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji2/ji1 Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{i2W}/j_{i1W}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji2/ji1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{i2W}/j_{i1W}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji1/ji_tot_b Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{i1W}/j_{iW}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji1/ji_tot_b Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{i1W}/j_{iW}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji1/ji_tot_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{i1W}/j_{iW}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji2/ji_tot_b Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{i2W}/j_{iW}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji2/ji_tot_b Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{i2W}/j_{iW}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji2/ji_tot_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{i2W}/j_{iW}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'j_b Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $j_{n}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'j_b Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $j_{n}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'j_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"$j_{n}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'je_b_gp_net_b Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $-j_{en}/eg_{pn}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'je_b_gp_net_b Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $-j_{en}/eg_{pn}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'je_b_gp_net_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $-j_{en}/eg_{pn}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_je_b Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{en}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_je_b Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{en}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_je_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{en}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_q == 1:
            plt.figure(r'qe_tot_wall Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $e_{en,W}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"(c) $q_{eW}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_tot_wall Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $e_{en,W}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"(c) $q_{eW}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_tot_wall Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $e_{en,W}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$q_{eW}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_tot_s_wall Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $e_{sn,W}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_tot_s_wall Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $e_{sn,W}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_tot_s_wall Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $e_{sn,W}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_tot_b Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $e_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_tot_b Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $e_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_tot_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $e_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_b Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $q_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_b Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $q_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $q_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_adv_b Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $5p_eu_{en}/2e|_Q$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_adv_b Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $5p_eu_{en}/2e|_Q$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_adv_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $5p_eu_{en}/2e|_Q$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_qe_b Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_qe_b Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_qe_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_qe_b_cons Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q},2}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_qe_b_cons Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q},2}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_qe_b_cons Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q},2}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qi1 Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $q_{i1,n}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"(c) $q_{i1W}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qi1 Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $q_{i1,n}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"(c) $q_{i1W}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qi1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $q_{i1,n}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$q_{i1W}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qi2 Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $q_{i2,n}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"(c) $q_{i2W}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qi2 Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $q_{i2,n}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$q_{i2W}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qi2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $q_{i2,n}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$q_{i2W}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qion Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $q_{i,n}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"(c) $q_{iW}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qion Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $q_{i,n}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"(c) $q_{iW}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qion Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
#            plt.title(r"(c) $q_{i,n}$ (W/cm$^2$)",fontsize = font_size)
            plt.title(r"$q_{iW}^{tot}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_imp_ene == 1:
            plt.figure(r'imp_ene_i1 Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_i1 Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_i1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_i2 Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i2}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_i2 Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i2}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_i2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i2}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_ion Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_ion Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_ion Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'imp_ene_n1 Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,n1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_n1 Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,n1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'imp_ene_n1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,n1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
        if plot_err_interp_mfam == 1:
            plt.figure(r'err_interp_phi Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{\phi}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_phi Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{\phi}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_phi Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{\phi}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_Te Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{T_e}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_Te Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{T_e}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_Te Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{T_e}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_jeperp Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\top e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jeperp Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\top e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jeperp Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\top e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_jetheta Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\theta e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jetheta Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\theta e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jetheta Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\theta e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_jepara Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\parallel e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jepara Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\parallel e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jepara Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\parallel e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_jez Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{ze}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jez Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{ze}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jez Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{ze}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_jer Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{re}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)      
            plt.figure(r'err_interp_jer Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{re}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)  
            plt.figure(r'err_interp_jer Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{re}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)  
        if plot_err_interp_pic == 1:
            plt.figure(r'err_interp_n Dwall_bot')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{n}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)      
            plt.figure(r'err_interp_n Dwall_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{n}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)  
            plt.figure(r'err_interp_n Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $\epsilon_{n}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size) 
        if plot_picM_picS_comp == 1:
            plt.figure(r'comp picMpicS imp_ene_i1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'comp picMpicS imp_ene_i2 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i2}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'comp picMpicS imp_ene_ion Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,i}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'comp picMpicS imp_ene_n1 Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $E_{imp,n1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'comp picMpicS ji_tot_b Dwall_bot_top')
            plt.xlabel(xaxis_label,fontsize = font_size)
            plt.title(r"(c) $j_{in}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
    if plot_Awall == 1:
        if plot_dens == 1:
            plt.figure(r'ne Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $n_e$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'nn Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $n_n$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_dphi_Te == 1:
            plt.figure(r'dphi_sh Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\Delta\phi_{sh}$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'dphi_sh_Te Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $e\Delta\phi_{sh}/T_e$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'Te Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $T_e$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'phi Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\phi$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_curr == 1:
            plt.figure(r'je_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $-j_{en}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'ji_tot_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $j_{in}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji1 Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"$j_{i1W}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji2 Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $j_{i2W}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji2/ji1 Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"$j_{i2W}/j_{i1W}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji1/ji_tot_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"$j_{i1W}/j_{iW}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji2/ji_tot_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"$j_{i2W}/j_{iW}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji_tot_b/j_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"$|j_{iW}/j_{W}|$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'je_b/j_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"$|j_{eW}/j_{W}|$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'j_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $-j_{n}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'je_b_gp_net_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $-j_{en}/eg_{pn}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)

            plt.figure(r'relerr_je_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{en}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_q == 1:
            plt.figure(r'qe_tot_wall Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $e_{en,W}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_tot_s_wall Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $e_{sn,W}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_tot_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $e_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $q_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_adv_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $5p_eu_{en}/2e|_Q$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_qe_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_qe_b_cons Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q},2}$ (-)",fontsize = font_size)
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
        if plot_err_interp_mfam == 1:
            plt.figure(r'err_interp_phi Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{\phi}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_Te Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{T_e}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jeperp Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\top e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jetheta Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\theta e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jepara Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_e{\parallel e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jez Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{ze}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jer Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{re}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_err_interp_pic == 1:
            plt.figure(r'err_interp_n Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{n}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)   
        if plot_picM_picS_comp == 1:
            plt.figure(r'comp picMpicS imp_ene_i1 Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $E_{imp,i1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'comp picMpicS imp_ene_i2 Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $E_{imp,i2}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'comp picMpicS imp_ene_ion Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $E_{imp,i}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'comp picMpicS imp_ene_n1 Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $E_{imp,n1}$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'comp picMpicS ji_tot_b Awall')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $j_{in}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
    if plot_FLwall == 1:
        if plot_dens == 1:
            plt.figure(r'ne FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $n_e$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ne FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $n_e$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ne FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $n_e$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'nn FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $n_n$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'nn FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $n_n$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'nn FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $n_n$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_dphi_Te == 1:            
            plt.figure(r'Te FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $T_e$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'Te FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $T_e$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'Te FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $T_e$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'phi FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\phi$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'phi FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\phi$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'phi FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\phi$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_curr == 1:
            plt.figure(r'je_b FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $-j_{en}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'je_b FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $-j_{en}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'je_b FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $-j_{en}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'ji_tot_b FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $j_{in}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji_tot_b FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $j_{in}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji_tot_b FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $j_{in}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'j_b FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $j_{n}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'j_b FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $j_{n}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'j_b FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $j_{n}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_je_b FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{en}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_je_b FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{en}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_je_b FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{en}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_q == 1:
            plt.figure(r'qe_tot_wall FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $e_{en,W}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_tot_wall FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $e_{en,W}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_tot_wall FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $e_{en,W}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_tot_b FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $e_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_tot_b FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $e_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_tot_b FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $e_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_b FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $q_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_b FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $q_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_b FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $q_{en,Q}$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'qe_adv_b FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $5p_eu_{en}/2e|_Q$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_adv_b FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $5p_eu_{en}/2e|_Q$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'qe_adv_b FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $5p_eu_{en}/2e|_Q$ (W/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_qe_b FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_qe_b FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_qe_b FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'relerr_qe_b_cons FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q},2}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_qe_b_cons FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q},2}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_qe_b_cons FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q},2}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_err_interp_mfam == 1:
            plt.figure(r'err_interp_phi FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{\phi}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_phi FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{\phi}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_phi FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{\phi}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_Te FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{T_e}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_Te FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{T_e}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_Te FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{T_e}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_jeperp FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\top e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jeperp FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\top e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jeperp FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\top e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_jetheta FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\theta e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jetheta FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\theta e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jetheta FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\theta e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_jepara FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\parallel e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jepara FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\parallel e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jepara FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\parallel e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_jez FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{ze}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jez FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{ze}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jez FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{ze}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'err_interp_jer FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{re}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)      
            plt.figure(r'err_interp_jer FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{re}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)  
            plt.figure(r'err_interp_jer FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{re}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)  
        if plot_err_interp_pic == 1:
            plt.figure(r'err_interp_n FLwall_ver')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{n}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)      
            plt.figure(r'err_interp_n FLwall_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{n}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)  
            plt.figure(r'err_interp_n FLwall_ver_lat')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{n}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size) 
            
    if plot_Axis == 1:
        if plot_dens == 1:
            plt.figure(r'ne Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $n_e$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'nn Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $n_n$ (m$^{-3}$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_dphi_Te == 1:
            plt.figure(r'Te Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $T_e$ (eV)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            
            plt.figure(r'phi Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\phi$ (V)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_curr == 1:
            plt.figure(r'je_b Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $-j_{en}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'ji_tot_b Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $j_{in}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'j_b Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $-j_{n}$ (A/cm$^2$)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_je_b Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{en}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_q == 1:
            plt.figure(r'relerr_qe_b Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'relerr_qe_b_cons Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{q_{en,Q},2}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_err_interp_mfam == 1:
            plt.figure(r'err_interp_phi Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{\phi}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_Te Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{T_e}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jeperp Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\top e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jetheta Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{\theta e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jepara Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_e{\parallel e}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jez Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{ze}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
            plt.figure(r'err_interp_jer Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{j_{re}}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)
        if plot_err_interp_pic == 1:
            plt.figure(r'err_interp_n Axis')
            plt.xlabel(r"$s$ (cm)",fontsize = font_size)
            plt.title(r"(c) $\epsilon_{n}$ (-)",fontsize = font_size)
            plt.xticks(fontsize = ticks_size) 
            plt.yticks(fontsize = ticks_size)      
        
        
        
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
        Ii1P_picS     = Ii1P_lat_picS + Ii1P_ver_picS
        Ii2P_picS     = Ii2P_lat_picS + Ii2P_ver_picS
        Ii3P_picS     = Ii3P_lat_picS + Ii3P_ver_picS
        Ii4P_picS     = Ii4P_lat_picS + Ii4P_ver_picS
                
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
        print("Contributions of ion species at P boundary using IiP_picS")
        print("Ii1P_picS/IiP_picS         (%%) = %15.8e " %(100.0*Ii1P_picS/IiP_picS))
        print("Ii2P_picS/IiP_picS         (%%) = %15.8e " %(100.0*Ii2P_picS/IiP_picS))
        print("Ii3P_picS/IiP_picS         (%%) = %15.8e " %(100.0*Ii3P_picS/IiP_picS))
        print("Ii4P_picS/IiP_picS         (%%) = %15.8e " %(100.0*Ii4P_picS/IiP_picS))
        print("CEX (Ii3P + Ii4P)          (%%) = %15.8e " %(100.0*(Ii3P_picS+Ii4P_picS)/IiP_picS))
        print("sum                        (%%) = %15.8e " %(100.0*(Ii1P_picS+Ii2P_picS+Ii3P_picS+Ii4P_picS)/IiP_picS))
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
        

        
        # Compute average energies at P boundary   
        # ene_i1P_lat = Pi1P_lat/Ii1P_lat_picS
        # ene_i1P_ver = Pi1P_ver/Ii1P_ver_picS
        # ene_i2P_lat = Pi2P_lat/Ii2P_lat_picS/2
        # ene_i2P_ver = Pi2P_ver/Ii2P_ver_picS/2
        # ene_i3P_lat = Pi3P_lat/Ii3P_lat_picS
        # ene_i3P_ver = Pi3P_ver/Ii3P_ver_picS
        # ene_i4P_lat = Pi4P_lat/Ii4P_lat_picS/2
        # ene_i4P_ver = Pi4P_ver/Ii4P_ver_picS/2 
    
        mean_ene_i1P_lat_picS = np.dot(imp_ene_i1_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        mean_ene_i1P_ver_picS = np.dot(imp_ene_i1_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        mean_ene_i2P_lat_picS = np.dot(imp_ene_i2_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        mean_ene_i2P_ver_picS = np.dot(imp_ene_i2_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        mean_ene_i3P_lat_picS = np.dot(imp_ene_i3_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        mean_ene_i3P_ver_picS = np.dot(imp_ene_i3_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        mean_ene_i4P_lat_picS = np.dot(imp_ene_i4_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        mean_ene_i4P_ver_picS = np.dot(imp_ene_i4_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        
        avg_ene_i1P_lat_picS  = np.dot(avg_imp_ene_i1_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        avg_ene_i1P_ver_picS  = np.dot(avg_imp_ene_i1_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        avg_ene_i2P_lat_picS  = np.dot(avg_imp_ene_i2_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        avg_ene_i2P_ver_picS  = np.dot(avg_imp_ene_i2_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        avg_ene_i3P_lat_picS  = np.dot(avg_imp_ene_i3_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        avg_ene_i3P_ver_picS  = np.dot(avg_imp_ene_i3_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        avg_ene_i4P_lat_picS  = np.dot(avg_imp_ene_i4_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        avg_ene_i4P_ver_picS  = np.dot(avg_imp_ene_i4_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        
        avg_ene_ion_lat_picS  = np.dot(avg_imp_ene_ion_surf_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        avg_ene_ion_ver_picS  = np.dot(avg_imp_ene_ion_surf_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        avg_ene_ion_picS      = (avg_ene_ion_lat_picS*np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])   + \
                                 avg_ene_ion_ver_picS*np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]]) ) / \
                                (np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]]) + np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]]) )

        avg_ene_ion_lat_picS_v2  = np.dot(avg_imp_ene_ion_surf_v2_plot[indsurf_FLwall_lat],surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])
        avg_ene_ion_ver_picS_v2  = np.dot(avg_imp_ene_ion_surf_v2_plot[indsurf_FLwall_ver],surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])/np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]])
        avg_ene_ion_picS_v2      = (avg_ene_ion_lat_picS_v2*np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]])   + \
                                    avg_ene_ion_ver_picS_v2*np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]]) ) / \
                                   (np.sum(surf_areas[imp_elems[indsurf_FLwall_lat][:,0],imp_elems[indsurf_FLwall_lat][:,1]]) + np.sum(surf_areas[imp_elems[indsurf_FLwall_ver][:,0],imp_elems[indsurf_FLwall_ver][:,1]]) )

       
        print("Energies at P boundary")
        # print("ene_i1P_lat                (eV) = %15.8e " %ene_i1P_lat)
        # print("ene_i1P_ver                (eV) = %15.8e " %ene_i1P_ver)
        # print("ene_i2P_lat                (eV) = %15.8e " %ene_i2P_lat)
        # print("ene_i2P_ver                (eV) = %15.8e " %ene_i2P_ver)
        # print("ene_i3P_lat                (eV) = %15.8e " %ene_i3P_lat)
        # print("ene_i3P_ver                (eV) = %15.8e " %ene_i3P_ver)
        # print("ene_i4P_lat                (eV) = %15.8e " %ene_i4P_lat)
        # print("ene_i4P_ver                (eV) = %15.8e " %ene_i4P_ver)
        print("mean_ene_i1P_lat            (eV) = %15.8e " %mean_ene_i1P_lat_picS)
        print("mean_ene_i1P_ver            (eV) = %15.8e " %mean_ene_i1P_ver_picS)
        print("mean_ene_i2P_lat            (eV) = %15.8e " %mean_ene_i2P_lat_picS)
        print("mean_ene_i2P_ver            (eV) = %15.8e " %mean_ene_i2P_ver_picS)
        print("mean_ene_i3P_lat            (eV) = %15.8e " %mean_ene_i3P_lat_picS)
        print("mean_ene_i3P_ver            (eV) = %15.8e " %mean_ene_i3P_ver_picS)
        print("mean_ene_i4P_lat            (eV) = %15.8e " %mean_ene_i4P_lat_picS)
        print("mean_ene_i4P_ver            (eV) = %15.8e " %mean_ene_i4P_ver_picS)
        print("avg_ene_i1P_lat             (eV) = %15.8e " %avg_ene_i1P_lat_picS)
        print("avg_ene_i1P_ver             (eV) = %15.8e " %avg_ene_i1P_ver_picS)
        print("avg_ene_i2P_lat             (eV) = %15.8e " %avg_ene_i2P_lat_picS)
        print("avg_ene_i2P_ver             (eV) = %15.8e " %avg_ene_i2P_ver_picS)
        print("avg_ene_i3P_lat             (eV) = %15.8e " %avg_ene_i3P_lat_picS)
        print("avg_ene_i3P_ver             (eV) = %15.8e " %avg_ene_i3P_ver_picS)
        print("avg_ene_i4P_lat             (eV) = %15.8e " %avg_ene_i4P_lat_picS)
        print("avg_ene_i4P_ver             (eV) = %15.8e " %avg_ene_i4P_ver_picS)
        print("avg_ene_ion_picS_lat        (eV) = %15.8e " %avg_ene_ion_lat_picS)
        print("avg_ene_ion_picS_ver        (eV) = %15.8e " %avg_ene_ion_ver_picS)
        print("avg_ene_ion_picS            (eV) = %15.8e " %avg_ene_ion_picS)
        print("avg_ene_ion_picS_lat_v2     (eV) = %15.8e " %avg_ene_ion_lat_picS_v2)
        print("avg_ene_ion_picS_ver_v2     (eV) = %15.8e " %avg_ene_ion_ver_picS_v2)
        print("avg_ene_ion_picS_v2         (eV) = %15.8e " %avg_ene_ion_picS_v2)
        
        
        
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
        
        
        # Compute surface-averaged values of dphi_sh/Te on plume boundary
        surf_avg_dphi_infty_down = (np.dot((phi_plot[IDfaces_FLwall_ver]-phi_inf_mean),Afaces_FLwall_ver) + \
                                    np.dot((phi_plot[IDfaces_FLwall_lat]-phi_inf_mean),Afaces_FLwall_lat))/(np.sum(Afaces_FLwall_ver)+np.sum(Afaces_FLwall_lat))
        surf_avg_Te_down         = (np.dot(Te_plot[IDfaces_FLwall_ver],Afaces_FLwall_ver) + \
                                    np.dot(Te_plot[IDfaces_FLwall_lat],Afaces_FLwall_lat))/(np.sum(Afaces_FLwall_ver)+np.sum(Afaces_FLwall_lat))
        surf_avg_dphi_infty_Te_down = surf_avg_dphi_infty_down/surf_avg_Te_down
        
        print("surf avg dphi/Te infty  () = "+str(surf_avg_dphi_infty_Te_down))
        
        
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
            
        
        if plot_wall == 1:
            if plot_deltas == 1:
                plt.figure(r'delta_r wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(swall,delta_r_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.plot(swall_inC,delta_r_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'delta_s wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(swall,delta_s_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.plot(swall_inC,delta_s_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'delta_s_csl wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(swall,delta_s_csl_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        if k == nsims-1:
#                            plt.plot(swall,0.985*np.ones(np.shape(swall),dtype=float), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='k', markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.plot(swall_inC,delta_s_csl_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        if k == nsims-1:
#                            plt.plot(swall_inC,0.985*np.ones(np.shape(swall_inC),dtype=float), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='k', markeredgecolor = 'k', label=labels[k])
                
            if plot_dphi_Te == 1:
                plt.figure(r'dphi_sh wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(swall,dphi_sh_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.plot(swall_inC,dphi_sh_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'dphi_sh_Te wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(swall,dphi_sh_b_plot[bIDfaces_wall]/Te_plot[IDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.plot(swall_inC,dphi_sh_b_plot[bIDfaces_wall_inC]/Te_plot[IDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'Te wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(swall,Te_plot[IDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.plot(swall_inC,Te_plot[IDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])


                plt.figure(r'phi wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(swall,phi_plot[IDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.plot(swall_inC,phi_plot[IDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

            if plot_curr == 1:
                
                plt.figure(r'je_b wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
#                        plt.plot(swall,-je_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall,-je_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC,-je_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_inC,-je_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
                plt.figure(r'ji_tot_b wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
#                        plt.plot(swall,ji_tot_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall,ji_tot_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC,ji_tot_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_inC,ji_tot_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
                plt.figure(r'egn wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.semilogy(swall_surf,jn_surf_plot[indsurf_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.semilogy(swall_inC_surf,jn_surf_plot[indsurf_inC_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        plt.semilogy(swall_inC_surf,jn1_surf_plot[indsurf_inC_wall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        plt.semilogy(swall_inC_surf,jn2_surf_plot[indsurf_inC_wall], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        plt.semilogy(swall_inC_surf,jn3_surf_plot[indsurf_inC_wall], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'j_b wall') 
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(swall,j_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.plot(swall_inC,j_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        
                plt.figure(r'ji_tot_b je_b wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
#                        plt.plot(swall,-je_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $-j_{ne}$")
#                        plt.plot(swall,ji_tot_b_plot[bIDfaces_wall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $j_{ni}$")
                        plt.semilogy(swall,-je_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $-j_{ne}$")
                        plt.semilogy(swall,ji_tot_b_plot[bIDfaces_wall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $j_{ni}$")
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC,-je_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $-j_{ne}$")
#                        plt.plot(swall_inC,ji_tot_b_plot[bIDfaces_wall_inC], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $j_{ni}$")
                        plt.semilogy(swall_inC,-je_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $-j_{ne}$")
                        plt.semilogy(swall_inC,ji_tot_b_plot[bIDfaces_wall_inC], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $j_{ni}$")
                
                plt.figure(r'relerr_je_b wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.semilogy(swall,relerr_je_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.semilogy(swall_inC,relerr_je_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

            if plot_imp_ene == 1:
                plt.figure(r'imp_ene_ion wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(swall_surf,imp_ene_ion_surf_plot[indsurf_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        plt.semilogy(swall_surf,imp_ene_ion_surf_plot[indsurf_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        # plt.semilogy(swall_surf,imp_ene_ion_surf_v2_plot[indsurf_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.plot(swall_inC_surf,imp_ene_ion_surf_plot[indsurf_inC_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        plt.semilogy(swall_inC_surf,imp_ene_ion_surf_plot[indsurf_inC_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        # plt.semilogy(swall_inC_surf,imp_ene_ion_surf_v2_plot[indsurf_inC_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'imp_ene_neu wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
#                        plt.plot(swall_surf,imp_ene_n_surf_plot[indsurf_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_surf,imp_ene_n_surf_plot[indsurf_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC_surf,imp_ene_n_surf_plot[indsurf_inC_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.plot(swall_inC_surf,imp_ene_n_surf_plot[indsurf_inC_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        plt.semilogy(swall_inC_surf,imp_ene_n1_surf_plot[indsurf_inC_wall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        plt.semilogy(swall_inC_surf,imp_ene_n2_surf_plot[indsurf_inC_wall], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        plt.semilogy(swall_inC_surf,imp_ene_n3_surf_plot[indsurf_inC_wall], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'imp_ene_e wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:                
#                        plt.plot(swall,imp_ene_e_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall,imp_ene_e_wall_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC,imp_ene_e_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_inC,imp_ene_e_wall_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
            if plot_q == 1:
                plt.figure(r'qi_tot_wall wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
#                        plt.plot(swall_surf,qi_tot_wall_surf_plot[indsurf_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_surf,qi_tot_wall_surf_plot[indsurf_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC_surf,qi_tot_wall_surf_plot[indsurf_inC_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_inC_surf,qi_tot_wall_surf_plot[indsurf_inC_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'qn_tot_wall wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
#                        plt.plot(swall_surf,qn_tot_wall_surf_plot[indsurf_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_surf,qn_tot_wall_surf_plot[indsurf_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC_surf,qn_tot_wall_surf_plot[indsurf_inC_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_inC_surf,qn_tot_wall_surf_plot[indsurf_inC_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                
                plt.figure(r'qe_tot_wall wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
#                        plt.plot(swall,qe_tot_wall_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall,qe_tot_wall_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC,qe_tot_wall_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_inC,qe_tot_wall_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'qe_tot_b wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
#                        plt.plot(swall,qe_tot_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall,qe_tot_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC,qe_tot_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_inC,qe_tot_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'qe_b wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(swall,qe_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        plt.semilogy(swall,qe_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC,qe_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_inC,qe_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'qe_adv_b wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
#                        plt.plot(swall,qe_adv_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall,qe_adv_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC,qe_adv_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_inC,qe_adv_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'relerr_qe_b wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.semilogy(swall,relerr_qe_b_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.semilogy(swall_inC,relerr_qe_b_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'relerr_qe_b_cons wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.semilogy(swall,relerr_qe_b_cons_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.semilogy(swall_inC,relerr_qe_b_cons_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'qi_tot_wall qe_tot_wall wall')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
#                        plt.plot(swall,qe_tot_wall_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        plt.plot(swall_surf,qi_tot_wall_surf_plot[indsurf_wall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall,qe_tot_wall_plot[bIDfaces_wall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_surf,qi_tot_wall_surf_plot[indsurf_wall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
#                        plt.plot(swall_inC,qe_tot_wall_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                        plt.plot(swall_inC_surf,qi_tot_wall_surf_plot[indsurf_inC_wall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_inC,qe_tot_wall_plot[bIDfaces_wall_inC], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.semilogy(swall_inC_surf,qi_tot_wall_surf_plot[indsurf_inC_wall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                
        if plot_down == 1:
            if plot_B == 1:
                plt.figure(r'Bfield down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,Bfield[IDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,Bfield_nodes[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

            if plot_dens == 1:
                plt.figure(r'ne down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown,n_inst_plot[IDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,n_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

            if plot_dphi_Te == 1:
                plt.figure(r'dphi_sh down')
                if plot_type == 0 or plot_type == 2:
                    if phi_inf_mean != 0:
                        plt.plot(sdown,dphi_sh_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        # plt.plot(sdown,phi_plot[IDfaces_down]-phi_inf_mean, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                        # plt.plot(sdown,avg_inst_dphi_infty_down, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    if phi_inf_mean != 0:
                        plt.plot(sdown_nodes,dphi_sh_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'dphi_sh_Te down')
                # Since we have BM oscillations, time average of non-linear operations is not equal to the operation with time-averaged variables.
                # We plot here the dphi_infty/Te computed with average values, which is not equal to the average of instantaneous values of
                # dphi/Te. The latter should be used later when comparing the e impact energy energy at P with corresponding 
                # expression involving Te and dphi_infty (in GPC) (see below the plot of e impact energy at P)
                if plot_type == 0 or plot_type == 2:
                    if phi_inf_mean != 0:
                        plt.plot(sdown,dphi_sh_b_plot[bIDfaces_down]/Te_plot[IDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        # plt.plot(sdown,avg_inst_dphi_sh_b_Te_down_plot, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                        # plt.plot(sdown,(phi_plot[IDfaces_down]-phi_inf_mean)/Te_plot[IDfaces_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        # plt.plot(sdown,avg_inst_dphi_infty_down/Te_plot[IDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k])
                        # plt.plot(sdown,avg_inst_dphi_sh_b_Te_down_plot_v2, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k])
                        # plt.plot(sdown,inst_dphi_sh_b_Te_plot[bIDfaces_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    if phi_inf_mean != 0:
                        # plt.plot(sdown_nodes,avg_inst_dphi_sh_b_Te_down_nodes_plot, linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        plt.plot(sdown_nodes,dphi_sh_b_nodes_plot[inodes_down,jnodes_down]/Te_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'Te down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,Te_plot[IDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,Te_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'phi down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,phi_plot[IDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,phi_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'dphi_inf down')
                if plot_type == 0 or plot_type == 2:
                    if phi_inf_mean != 0:
                        plt.plot(sdown,phi_plot[IDfaces_down]-phi_inf_mean, linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    if phi_inf_mean != 0:
                        plt.plot(sdown_nodes,phi_nodes_plot[inodes_down,jnodes_down]-phi_inf_mean, linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'dphi_inf_Te down')
                if plot_type == 0 or plot_type == 2:
                    if phi_inf_mean != 0:
                        plt.plot(sdown,(phi_plot[IDfaces_down]-phi_inf_mean)/Te_plot[IDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        # plt.plot(sdown,dphi_sh_b_plot[bIDfaces_down]/Te_plot[IDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind+1], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    if phi_inf_mean != 0:
                        plt.plot(sdown_nodes,(phi_nodes_plot[inodes_down,jnodes_down]-phi_inf_mean)/Te_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

            if plot_curr == 1:
                plt.figure(r'j_b down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,j_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,j_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'je_b down')
                if plot_type == 0 or plot_type == 2:
#                    plt.plot(sdown,-je_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])                
                    plt.semilogy(sdown,-je_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])                
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,-je_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'ji_tot_b down')
                if plot_type == 0 or plot_type == 2:
#                    plt.plot(sdown,ji_tot_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])                
                    plt.semilogy(sdown,ji_tot_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])                
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,ji_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'ji_tot_b je_b down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,-je_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $-j_{ne}$")
                    plt.plot(sdown,ji_tot_b_plot[bIDfaces_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $j_{ni}$")
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,-je_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $-j_{ne}$")
                    plt.plot(sdown_nodes,ji_nodes_plot[inodes_down,jnodes_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $j_{ni}$")

                plt.figure(r'relerr_je_b down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown,relerr_je_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,relerr_je_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" $j_{ni}$")
            
                plt.figure(r'ji1 down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown_surf,ji1_surf_plot[indsurf_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,ji1_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'ji2 down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown_surf,ji2_surf_plot[indsurf_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,ji2_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'ji3 down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown_surf,ji3_surf_plot[indsurf_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,ji3_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'ji4 down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown_surf,ji4_surf_plot[indsurf_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,ji4_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
            
            if plot_q == 1:
                plt.figure(r'qi_tot_b down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown_surf,qi_tot_wall_surf_plot[indsurf_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,qi_tot_wall_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'qe_tot_wall down')
                # ratio_check_qe_tot_wall_PLC = qe_tot_wall_plot[bIDfaces_down]/(-4.5*Te_plot[IDfaces_down]*je_b_plot[bIDfaces_down])
                # ratio_check_qe_tot_wall_PLC_v2 = qe_tot_wall[-1,bIDfaces_down]/(-4.5*Te[-1,IDfaces_down]*je_b[-1,bIDfaces_down])
                # ratio_check_qe_tot_wall_PGC = qe_tot_wall_plot[bIDfaces_down]/(-2.0*Te_plot[IDfaces_down]*je_b_plot[bIDfaces_down])
                # ratio_check_qe_tot_wall_PGC_v2 = qe_tot_wall[-1,bIDfaces_down]/(-2.0*Te[-1,IDfaces_down]*je_b[-1,bIDfaces_down])
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown,qe_tot_wall_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    # plt.semilogy(sdown,-4.5*Te_plot[IDfaces_down]*je_b_plot[bIDfaces_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                    # plt.semilogy(sdown,-2.0*Te_plot[IDfaces_down]*je_b_plot[bIDfaces_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,qe_tot_wall_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'qe_tot_b down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown,qe_tot_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,qe_tot_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'qe_b down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown,qe_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    # plt.semilogy(sdown,-2.0*Te_plot[IDfaces_down]*je_b_plot[bIDfaces_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,qe_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'qe_adv_b down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown,qe_adv_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,qe_adv_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'ratio_qe_b_qe_adv_b down')
                ratio_qeb_qe_adv_b_down = qe_b_plot[bIDfaces_down]/qe_adv_b_plot[bIDfaces_down]
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,ratio_qeb_qe_adv_b_down, linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,qe_b_nodes_plot[inodes_down,jnodes_down]/qe_adv_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'ratio_qe_b_qe_tot_b down')
                ratio_qe_b_qe_tot_b_down = qe_b_plot[bIDfaces_down]/qe_tot_b_plot[bIDfaces_down]
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,ratio_qe_b_qe_tot_b_down, linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,qe_b_nodes_plot[inodes_down,jnodes_down]/qe_tot_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'ratio_qe_adv_b_qe_tot_b down')
                ratio_qe_adv_b_eq_tot_b_down = qe_adv_b_plot[bIDfaces_down]/qe_tot_b_plot[bIDfaces_down]
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,ratio_qe_adv_b_eq_tot_b_down, linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,qe_adv_b_nodes_plot[inodes_down,jnodes_down]/qe_tot_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'je_dphi_sh down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,qe_tot_b_plot[bIDfaces_down]-qe_tot_wall_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])        
                    # plt.plot(sdown,-je_b_plot[bIDfaces_down]*dphi_sh_b_plot[bIDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])        
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,qe_tot_b_nodes_plot[inodes_down,jnodes_down] - qe_tot_wall_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'ratio_je_dphi_sh_qe_tot_b down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,(qe_tot_b_plot[bIDfaces_down]-qe_tot_wall_plot[bIDfaces_down])/qe_tot_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])              
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,(qe_tot_b_nodes_plot[inodes_down,jnodes_down] - qe_tot_wall_nodes_plot[inodes_down,jnodes_down])/qe_tot_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'ratio_qe_tot_wall_qe_tot_b down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,qe_tot_wall_plot[bIDfaces_down]/qe_tot_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])              
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,qe_tot_wall_nodes_plot[inodes_down,jnodes_down]/qe_tot_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'ratio_eqe_tot_b_je_bTe down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,-qe_tot_b_plot[bIDfaces_down]/(je_b_plot[bIDfaces_down]*Te_plot[IDfaces_down]), linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])         
                    # plt.plot(sdown,-(qe_tot_b_plot[bIDfaces_down]-qe_tot_wall_plot[bIDfaces_down])/(je_b_plot[bIDfaces_down]*Te_plot[IDfaces_down]), linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])        
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,-qe_tot_b_nodes_plot[inodes_down,jnodes_down]/(je_b_nodes_plot[inodes_down,jnodes_down]*Te_nodes_plot[inodes_down,jnodes_down]), linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'relerr_qe_b down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown,relerr_qe_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,relerr_qe_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'relerr_qe_b_cons down')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sdown,relerr_qe_b_cons_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.semilogy(sdown_nodes,relerr_qe_b_cons_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
            if plot_imp_ene == 1:
                # Since we have BM oscillations, time average of non-linear operations is not equal to the operation with time-averaged variables.
                plt.figure(r'imp_ene_ion down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown_surf,imp_ene_ion_surf_plot[indsurf_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown_surf,avg_imp_ene_ion_surf_plot[indsurf_down], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,imp_ene_ion_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'imp_ene_ion1 down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown_surf,imp_ene_i1_surf_plot[indsurf_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,imp_ene_i1_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'imp_ene_ion2 down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown_surf,imp_ene_i2_surf_plot[indsurf_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,imp_ene_i2_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'imp_ene_ion3 down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown_surf,imp_ene_i3_surf_plot[indsurf_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,imp_ene_i3_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'imp_ene_ion4 down')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown_surf,imp_ene_i4_surf_plot[indsurf_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,imp_ene_i4_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'imp_ene_e_b down')
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
                    imp_ene_e_b_down_check_inst_avg_v2 = imp_ene_e_b_down_check_v2
                # The following ratio will deviate from 1 due to non-linear operations performed with average values both in num and den
                ratio_imp_ene_check          = imp_ene_e_b_down_check/avg_imp_ene_e_b_plot[bIDfaces_down]
                # The following ratio will deviate from 1 due to non-linear operations performed with average values in den
                ratio_imp_ene_check_v2       = imp_ene_e_b_down_check_inst_avg/avg_imp_ene_e_b_plot[bIDfaces_down]
                # The following ratio will not deviate from 1 because num and den are averages of non-linear operations
                ratio_imp_ene_inst_avg_check = imp_ene_e_b_down_check_inst_avg/imp_ene_e_b_plot[bIDfaces_down]
                
                # We plot here the e impact energy computed with average values, although to compare with corresponding 
                # expression involving Te and dphi_infty (in GPC) we should compute the e impact energy as the average
                # of the corresponding operation with instantaneous values
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,avg_imp_ene_e_b_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_b_plot[bIDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_b_down_check, linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='c', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_b_down_check_v2, linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='g', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_b_down_check_inst_avg, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_b_down_check_inst_avg_v2, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='g', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,inst_imp_ene_e_b_plot[bIDfaces_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,imp_ene_e_b_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'imp_ene_e_b_Te down')
                # Since we have BM oscillations, time average of non-linear operations is not equal to the operation with time-averaged variables.
                if phi_inf_mean != 0:
                    # Operation with average values
                    imp_ene_e_b_Te_down_check    = (2 + dphi_sh_b_plot[bIDfaces_down]/Te_plot[IDfaces_down])
                    imp_ene_e_b_Te_down_check_v2 = (2 + (phi_plot[IDfaces_down]-phi_inf_mean)/Te_plot[IDfaces_down])
                    # Average of the operation with instantaneous values
                    if mean_type == 0:
                        imp_ene_e_b_Te_down_check_inst_avg    = np.nanmean((2 + dphi_sh_b[nsteps-last_steps::,bIDfaces_down]/Te[nsteps-last_steps::,IDfaces_down]),axis=0)
                        imp_ene_e_b_Te_down_check_inst_avg_v2 = np.nanmean((2 + dphi_infty_down[nsteps-last_steps::,IDfaces_down]/Te[nsteps-last_steps::,IDfaces_down]),axis=0)
                    elif mean_type == 1:
                        imp_ene_e_b_Te_down_check_inst_avg    = np.nanmean((2 + dphi_sh_b[step_i:step_f+1,bIDfaces_down]/Te[step_i:step_f+1,IDfaces_down]),axis=0)
                        imp_ene_e_b_Te_down_check_inst_avg_v2 = np.nanmean((2 + dphi_infty_down[step_i:step_f+1,IDfaces_down]/Te[step_i:step_f+1,IDfaces_down]),axis=0)
                else:
                    # Operation with average values (in this case is equal to the average of the operation with instantaneous values)
                    imp_ene_e_b_Te_down_check          = 4.5*np.ones(np.shape(imp_ene_e_b_plot[bIDfaces_down]))
                    imp_ene_e_b_Te_down_check_v2       = imp_ene_e_b_Te_down_check
                    # Average of the operation with instantaneous values
                    imp_ene_e_b_Te_down_check_inst_avg    = imp_ene_e_b_Te_down_check
                    imp_ene_e_b_Te_down_check_inst_avg_v2 = imp_ene_e_b_Te_down_check_inst_avg
                if mean_type == 0:
                    imp_ene_e_b_Te_down_inst_avg    = np.nanmean((qe_tot_b[nsteps-last_steps::,bIDfaces_down]/(-je_b[nsteps-last_steps::,bIDfaces_down]/e))/e/Te[nsteps-last_steps::,IDfaces_down],axis=0)
                elif mean_type == 1:
                    imp_ene_e_b_Te_down_inst_avg    = np.nanmean((qe_tot_b[step_i:step_f+1,bIDfaces_down]/(-je_b[step_i:step_f+1,bIDfaces_down]/e))/e/Te[step_i:step_f+1,IDfaces_down],axis=0)
                # The following ratio will deviate from 1 due to non-linear operations performed with average values both in num and den
                ratio_imp_ene_Te_check          = imp_ene_e_b_Te_down_check/(imp_ene_e_b_plot[bIDfaces_down]/Te_plot[IDfaces_down])
                # The following ratio will deviate from 1 due to non-linear operations performed with average values in den
                ratio_imp_ene_Te_check_v2       = imp_ene_e_b_Te_down_check_inst_avg/(imp_ene_e_b_plot[bIDfaces_down]/Te_plot[IDfaces_down])
                # The following ratio will not deviate from 1 because num and den are averages of non-linear operations
                ratio_imp_ene_Te_inst_avg_check = imp_ene_e_b_Te_down_check_inst_avg/imp_ene_e_b_Te_down_inst_avg
                
                # We plot here the ratio e impact energy/Te computed with average values, although to compare corresponding 
                # expression involving Te and dphi_infty (in GPC) we should compute it as the average of the operation with 
                # instantaneous values (imp_ene_e_b_Te_down_inst_avg)
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,avg_imp_ene_e_b_plot[bIDfaces_down]/Te_plot[IDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_b_plot[bIDfaces_down]/Te_plot[IDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_b_Te_down_inst_avg, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,2+avg_inst_dphi_sh_b_Te_down_plot, linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='g', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,2+avg_inst_dphi_sh_b_Te_down_plot_v2, linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='y', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_b_Te_down_check_inst_avg, linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,2+dphi_sh_b_plot[bIDfaces_down]/Te_plot[IDfaces_down], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,inst_imp_ene_e_b_Te_plot[bIDfaces_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,imp_ene_e_b_nodes_plot[inodes_down,jnodes_down]/Te_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'imp_ene_e_wall down')
                # Since we have BM oscillations, time average of non-linear operations is not equal to the operation with time-averaged variables.
                if phi_inf_mean != 0:
                    # Operation with average values
                    imp_ene_e_wall_down_check          = 2.0*Te_plot[IDfaces_down]
                    # Average of the operation with instantaneous values
                    if mean_type == 0:
                        imp_ene_e_wall_down_check_inst_avg = np.nanmean(2.0*Te[nsteps-last_steps::,IDfaces_down],axis=0)
                    elif mean_type == 1:
                        imp_ene_e_wall_down_check_inst_avg = np.nanmean(2.0*Te[step_i:step_f+1,IDfaces_down],axis=0)
                else:
                    # Operation with average values (in this case is equal to the average of the operation with instantaneous values)
                    imp_ene_e_wall_down_check       = 4.5*Te_plot[IDfaces_down]
                    # Average of the operation with instantaneous values
                    imp_ene_e_wall_down_check_inst_avg = imp_ene_e_b_down_check
                
                # We plot here the e impact energy computed with average values, although to compare with corresponding 
                # expression involving Te (in GPC) we should compute the e impact energy as the average
                # of the corresponding operation with instantaneous values
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,avg_imp_ene_e_wall_plot[bIDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_wall_plot[bIDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_wall_down_check_inst_avg, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_wall_down_check, linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='g', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,inst_imp_ene_e_wall_plot[bIDfaces_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,imp_ene_e_wall_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'imp_ene_e_wall_Te down')
                # Since we have BM oscillations, time average of non-linear operations is not equal to the operation with time-averaged variables.
                if mean_type == 0:
                    imp_ene_e_wall_Te_down_inst_avg    = np.nanmean((qe_tot_wall[nsteps-last_steps::,bIDfaces_down]/(-je_b[nsteps-last_steps::,bIDfaces_down]/e))/e/Te[nsteps-last_steps::,IDfaces_down],axis=0)
                    # imp_ene_e_wall_Te_down_inst_avg    = np.nanmean(imp_ene_e_wall[nsteps-last_steps::,bIDfaces_down]/Te[nsteps-last_steps::,IDfaces_down],axis=0)
                elif mean_type == 1:
                    imp_ene_e_wall_Te_down_inst_avg    = np.nanmean((qe_tot_wall[step_i:step_f+1,bIDfaces_down]/(-je_b[step_i:step_f+1,bIDfaces_down]/e))/e/Te[step_i:step_f+1,IDfaces_down],axis=0)
                    # imp_ene_e_wall_Te_down_inst_avg    = np.nanmean(imp_ene_e_wall[step_i:step_f+1,bIDfaces_down]/Te[step_i:step_f+1,IDfaces_down],axis=0)
                
                # We plot here the ratio e impact energy/Te computed with average values, although to compare corresponding 
                # expression involving Te and dphi_infty (in GPC) we should compute it as the average of the operation with 
                # instantaneous values (imp_ene_e_b_Te_down_inst_avg)
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sdown,avg_imp_ene_e_wall_plot[bIDfaces_down]/Te_plot[IDfaces_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_wall_plot[bIDfaces_down]/Te_plot[IDfaces_down], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,imp_ene_e_wall_Te_down_inst_avg, linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='m', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,2*np.ones(np.shape(sdown)), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='g', markeredgecolor = 'k', label=labels[k])
                    # plt.plot(sdown,inst_imp_ene_e_wall_Te_plot[bIDfaces_down], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color='b', markeredgecolor = 'k', label=labels[k])
                if plot_type == 1 and picvars_plot == 1:
                    plt.plot(sdown_nodes,avg_imp_ene_e_wall_nodes_plot[inodes_down,jnodes_down]/Te_nodes_plot[inodes_down,jnodes_down], linestyle=linestyles[ind], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                
                
        
        if plot_Dwall == 1:
            if plot_dens == 1:
                plt.figure(r'ne Dwall_bot')
                if plot_type == 0:
                    plt.semilogy(sDwall_bot,n_inst_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.semilogy(sDwall_bot_nodes,n_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                    plt.semilogy(sDwall_bot_nodes,n_inst_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.semilogy(sDwall_bot,n_inst_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.semilogy(sDwall_bot_nodes,n_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")
#                    plt.semilogy(sDwall_bot_nodes,n_inst_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

                plt.figure(r'ne Dwall_top')
                if plot_type == 0: 
                    plt.semilogy(sDwall_top,n_inst_plot[IDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.semilogy(sDwall_top_nodes,n_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                    plt.semilogy(sDwall_top_nodes,n_inst_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.semilogy(sDwall_top,n_inst_plot[IDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.semilogy(sDwall_top_nodes,n_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")
#                    plt.semilogy(sDwall_top_nodes,n_inst_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

                plt.figure(r'ne Dwall_bot_top')
                if plot_type == 0:
                    plt.semilogy(sDwall_bot,n_inst_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top,n_inst_plot[IDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                elif plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,n_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.semilogy(sDwall_top_nodes,n_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.semilogy(sDwall_bot_surf,nQ2_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                            plt.semilogy(sDwall_top_surf,nQ2_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                        elif inC_Dwalls == 1:
                            plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],nQ2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                            plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],nQ2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'nn Dwall_bot')
                if plot_type == 0:
                    plt.semilogy(sDwall_bot,nn1_inst_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.semilogy(sDwall_bot_nodes,nn1_inst_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.semilogy(sDwall_bot,nn1_inst_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.semilogy(sDwall_bot_nodes,nn1_inst_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")
                
                plt.figure(r'nn Dwall_top')
                if plot_type == 0:
                    plt.semilogy(sDwall_top,nn1_inst_plot[IDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.semilogy(sDwall_top_nodes,nn1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.semilogy(sDwall_top,nn1_inst_plot[IDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.semilogy(sDwall_top_nodes,nn1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")
                
                plt.figure(r'nn Dwall_bot_top')
                if plot_type == 1:
                    plt.semilogy(sDwall_bot_nodes,nn1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top_nodes,nn1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                elif plot_type == 0 or plot_type == 2:
                   plt.semilogy(sDwall_bot,nn1_inst_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                   plt.semilogy(sDwall_top,nn1_inst_plot[IDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
            if plot_deltas == 1:
                plt.figure(r'delta_r Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,delta_r_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'delta_r Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,delta_r_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'delta_r Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,delta_r_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.plot(sDwall_top,delta_r_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'delta_s Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,delta_s_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'delta_s Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,delta_s_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'delta_s Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(sDwall_bot,delta_s_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top,delta_s_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif inC_Dwalls == 1:
                        plt.plot(sDwall_bot[IDfaces_inC_Dwall_bot],delta_s_plot[bIDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top[IDfaces_inC_Dwall_top],delta_s_plot[bIDfaces_Dwall_top[IDfaces_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
            if plot_dphi_Te == 1:
                plt.figure(r'dphi_sh Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,dphi_sh_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'dphi_sh Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,dphi_sh_b_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'dphi_sh Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(sDwall_bot,dphi_sh_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top,dphi_sh_b_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif inC_Dwalls == 1:
                        plt.plot(sDwall_bot[IDfaces_inC_Dwall_bot],dphi_sh_b_plot[bIDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top[IDfaces_inC_Dwall_top],dphi_sh_b_plot[bIDfaces_Dwall_top[IDfaces_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
    
                plt.figure(r'dphi_sh_Te Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,dphi_sh_b_plot[bIDfaces_Dwall_bot]/Te_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'dphi_sh_Te Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,dphi_sh_b_plot[bIDfaces_Dwall_top]/Te_plot[IDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'dphi_sh_Te Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(sDwall_bot,dphi_sh_b_plot[bIDfaces_Dwall_bot]/Te_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top,dphi_sh_b_plot[bIDfaces_Dwall_top]/Te_plot[IDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif inC_Dwalls == 1:
                        plt.plot(sDwall_bot[IDfaces_inC_Dwall_bot],dphi_sh_b_plot[bIDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]]/Te_plot[IDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top[IDfaces_inC_Dwall_top],dphi_sh_b_plot[bIDfaces_Dwall_top[IDfaces_inC_Dwall_top]]/Te_plot[IDfaces_Dwall_top[IDfaces_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
    
                plt.figure(r'Te Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,Te_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'Te Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,Te_plot[IDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'Te Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(sDwall_bot,Te_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top,Te_plot[IDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif inC_Dwalls == 1:
                        plt.plot(sDwall_bot[IDfaces_inC_Dwall_bot],Te_plot[IDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top[IDfaces_inC_Dwall_top],Te_plot[IDfaces_Dwall_top[IDfaces_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
#                        plt.plot(sDwall_top[IDfaces_inC_Dwall_top],Te_plot[IDfaces_wall_inC[40::]], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color='r', markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'phi Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,phi_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'phi Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,phi_plot[IDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'phi Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,phi_plot[IDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.plot(sDwall_top,phi_plot[IDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
            
            if plot_curr == 1:
                plt.figure(r'je_b Dwall_bot')
                if plot_type == 0 or plot_type == 2:
#                    plt.plot(sDwall_bot,-je_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.semilogy(sDwall_bot,-je_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'je_b Dwall_top')
                if plot_type == 0 or plot_type == 2:
#                    plt.plot(sDwall_top,-je_b_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.semilogy(sDwall_top,-je_b_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'je_b Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
#                    plt.plot(sDwall_bot,-je_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
#                    plt.plot(sDwall_top,-je_b_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    plt.plot(sDwall_bot,-je_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.plot(sDwall_top,-je_b_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    
                plt.figure(r'ji_tot_b Dwall_bot')
                if plot_type == 0:
                    plt.semilogy(sDwall_bot,ji_tot_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,ji_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.semilogy(sDwall_bot_surf,ji_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.semilogy(sDwall_bot,ji_tot_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,ji_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")
                    elif picvars_plot == 2:
                        plt.semilogy(sDwall_bot_surf,ji_surf_plot[indsurf_Dwall_bot], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bs")
                plt.figure(r'ji_tot_b Dwall_top')
                if plot_type == 0:
                    plt.semilogy(sDwall_top,ji_tot_b_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_top_nodes,ji_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.semilogy(sDwall_top_surf,ji_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.semilogy(sDwall_top,ji_tot_b_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                    plt.semilogy(sDwall_top[IDfaces_inC_Dwall_top],ji_tot_b_plot[bIDfaces_Dwall_top[IDfaces_inC_Dwall_top]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_top_nodes,ji_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")
                    elif picvars_plot == 2:
                        plt.semilogy(sDwall_top_surf,ji_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bs")
#                        plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],ji_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bs")
                plt.figure(r'ji_tot_b Dwall_bot_top')
                if plot_type == 0:
                    plt.plot(sDwall_bot,ji_tot_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.plot(sDwall_top,ji_tot_b_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                elif plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,ji_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.semilogy(sDwall_top_nodes,ji_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,ji_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                            plt.plot(sDwall_top_surf,ji_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],ji_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],ji_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")

                plt.figure(r'ji1 Dwall_bot')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,ji1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.semilogy(sDwall_bot_surf,ji1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'ji1 Dwall_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_top_nodes,ji1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.semilogy(sDwall_top_surf,ji1_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
                plt.figure(r'ji1 Dwall_bot_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,ji1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.semilogy(sDwall_top_nodes,ji1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,ji1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf,ji1_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],ji1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],ji1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                
                plt.figure(r'ji2 Dwall_bot')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,ji2_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.semilogy(sDwall_bot_surf,ji2_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'ji2 Dwall_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_top_nodes,ji2_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.semilogy(sDwall_top_surf,ji2_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
                plt.figure(r'ji2 Dwall_bot_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,ji2_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.semilogy(sDwall_top_nodes,ji2_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,ji2_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf,ji2_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],ji2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],ji2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
    
                plt.figure(r'ji2/ji1 Dwall_bot')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_bot_nodes,ji2_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot]/ji1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.plot(sDwall_bot_surf,ji2_surf_plot[indsurf_Dwall_bot]/ji1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'ji2/ji1 Dwall_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_top_nodes,ji2_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top]/ji1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.plot(sDwall_top_surf,ji2_surf_plot[indsurf_Dwall_top]/ji1_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'ji2/ji1 Dwall_bot_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,ji2_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot]/ji1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.semilogy(sDwall_top_nodes,ji2_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top]/ji1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,ji2_surf_plot[indsurf_Dwall_bot]/ji1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf,ji2_surf_plot[indsurf_Dwall_top]/ji1_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],ji2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]]/ji1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],ji2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]]/ji1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
    
                plt.figure(r'ji1/ji_tot_b Dwall_bot')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_bot_nodes,ji1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot]/ji_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.plot(sDwall_bot_surf,ji1_surf_plot[indsurf_Dwall_bot]/ji_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'ji1/ji_tot_b Dwall_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_top_nodes,ji1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top]/ji_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.plot(sDwall_top_surf,ji1_surf_plot[indsurf_Dwall_top]/ji_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'ji1/ji_tot_b Dwall_bot_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,ji1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot]/ji_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.semilogy(sDwall_top_nodes,ji1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top]/ji_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,ji1_surf_plot[indsurf_Dwall_bot]/ji_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf,ji1_surf_plot[indsurf_Dwall_top]/ji_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],ji1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]]/ji_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],ji1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]]/ji_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   

                plt.figure(r'ji2/ji_tot_b Dwall_bot')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_bot_nodes,ji2_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot]/ji_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.plot(sDwall_bot_surf,ji2_surf_plot[indsurf_Dwall_bot]/ji_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'ji2/ji_tot_b Dwall_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_top_nodes,ji2_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top]/ji_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.plot(sDwall_top_surf,ji2_surf_plot[indsurf_Dwall_top]/ji_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'ji2/ji_tot_b Dwall_bot_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,ji2_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot]/ji_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.semilogy(sDwall_top_nodes,ji2_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top]/ji_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,ji2_surf_plot[indsurf_Dwall_bot]/ji_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf,ji2_surf_plot[indsurf_Dwall_top]/ji_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],ji2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]]/ji_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],ji2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]]/ji_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
    
                plt.figure(r'j_b Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,j_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'j_b Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,j_b_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'j_b Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(sDwall_bot,j_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top,j_b_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif inC_Dwalls == 1:
                        plt.plot(sDwall_bot[IDfaces_inC_Dwall_bot],j_b_plot[bIDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top[IDfaces_inC_Dwall_top],j_b_plot[bIDfaces_Dwall_top[IDfaces_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                        
                plt.figure(r'je_b_gp_net_b Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,-je_b_plot[bIDfaces_Dwall_bot]/(e*gp_net_b_plot[bIDfaces_Dwall_bot]*1E-4), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'je_b_gp_net_b Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,-je_b_plot[bIDfaces_Dwall_top]/(e*gp_net_b_plot[bIDfaces_Dwall_top]*1E-4), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'je_b_gp_net_b Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,-je_b_plot[bIDfaces_Dwall_bot]/(e*gp_net_b_plot[bIDfaces_Dwall_bot]*1E-4), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.plot(sDwall_top,-je_b_plot[bIDfaces_Dwall_top]/(e*gp_net_b_plot[bIDfaces_Dwall_top]*1E-4), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")

                plt.figure(r'relerr_je_b Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,relerr_je_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_je_b Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_top,relerr_je_b_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_je_b Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,relerr_je_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top,relerr_je_b_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
            
            if plot_q == 1:
                plt.figure(r'qe_tot_wall Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(sDwall_bot,qe_tot_wall_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.plot(sDwall_bot[IDfaces_inC_Dwall_bot],qe_tot_wall_plot[bIDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_wall Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(sDwall_top,qe_tot_wall_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif inC_Dwalls == 1:
                        plt.plot(sDwall_top[IDfaces_inC_Dwall_top],qe_tot_wall_plot[bIDfaces_Dwall_top[IDfaces_inC_Dwall_top]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_wall Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    if inC_Dwalls == 0:
                        plt.plot(sDwall_bot,qe_tot_wall_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top,qe_tot_wall_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif inC_Dwalls == 1:
                        plt.plot(sDwall_bot[IDfaces_inC_Dwall_bot],qe_tot_wall_plot[bIDfaces_Dwall_bot[IDfaces_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top[IDfaces_inC_Dwall_top],qe_tot_wall_plot[bIDfaces_Dwall_top[IDfaces_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every_mfambf, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")

                plt.figure(r'qe_tot_s_wall Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,qe_tot_s_wall_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_s_wall Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,qe_tot_s_wall_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_s_wall Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,qe_tot_s_wall_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.plot(sDwall_top,qe_tot_s_wall_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")

                plt.figure(r'qe_tot_b Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,qe_tot_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_b Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,qe_tot_b_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_b Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,qe_tot_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.plot(sDwall_top,qe_tot_b_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")

                plt.figure(r'qe_b Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,qe_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_b Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,qe_b_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_b Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,qe_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.plot(sDwall_top,qe_b_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")

                plt.figure(r'qe_adv_b Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,qe_adv_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_adv_b Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_top,qe_adv_b_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_adv_b Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sDwall_bot,qe_adv_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.plot(sDwall_top,qe_adv_b_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")

                plt.figure(r'relerr_qe_b Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,relerr_qe_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_qe_b Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_top,relerr_qe_b_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_qe_b Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,relerr_qe_b_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top,relerr_qe_b_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
            
                plt.figure(r'relerr_qe_b_cons Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,relerr_qe_b_cons_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_qe_b_cons Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_top,relerr_qe_b_cons_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_qe_b_cons Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,relerr_qe_b_cons_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top,relerr_qe_b_cons_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
            
                plt.figure(r'qi1 Dwall_bot')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,qi1_tot_wall_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.semilogy(sDwall_bot_surf,qi1_tot_wall_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],qi1_tot_wall_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qi1 Dwall_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_top_nodes,qi1_tot_wall_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.semilogy(sDwall_top_surf,qi1_tot_wall_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],qi1_tot_wall_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qi1 Dwall_bot_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,qi1_tot_wall_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.semilogy(sDwall_top_nodes,qi1_tot_wall_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.semilogy(sDwall_bot_surf,qi1_tot_wall_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.semilogy(sDwall_top_surf,qi1_tot_wall_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],qi1_tot_wall_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],qi1_tot_wall_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                        

                plt.figure(r'qi2 Dwall_bot')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,qi2_tot_wall_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.semilogy(sDwall_bot_surf,qi2_tot_wall_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],qi2_tot_wall_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qi2 Dwall_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_top_nodes,qi2_tot_wall_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.semilogy(sDwall_top_surf,qi2_tot_wall_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],qi2_tot_wall_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qi2 Dwall_bot_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,qi2_tot_wall_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.semilogy(sDwall_top_nodes,qi2_tot_wall_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,qi2_tot_wall_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf,qi2_tot_wall_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],qi2_tot_wall_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],qi2_tot_wall_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                        
                   
                plt.figure(r'qion Dwall_bot')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,qi_tot_wall_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.semilogy(sDwall_bot_surf,qi_tot_wall_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.semilogy(sDwall_bot_surf[indsurf_inC_Dwall_bot],qi_tot_wall_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qion Dwall_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_top_nodes,qi_tot_wall_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.semilogy(sDwall_top_surf,qi_tot_wall_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.semilogy(sDwall_top_surf[indsurf_inC_Dwall_top],qi_tot_wall_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qion Dwall_bot_top')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.semilogy(sDwall_bot_nodes,qi_tot_wall_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.semilogy(sDwall_top_nodes,qi_tot_wall_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,qi_tot_wall_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf,qi_tot_wall_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],qi_tot_wall_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")   
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],qi_tot_wall_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")   
                        
            
            if plot_imp_ene == 1:
                plt.figure(r'imp_ene_i1 Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_bot_nodes,imp_ene_i1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,imp_ene_i1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_i1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'imp_ene_i1 Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_top_nodes,imp_ene_i1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_top_surf,imp_ene_i1_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_i1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'imp_ene_i1 Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_bot_nodes,imp_ene_i1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top_nodes,imp_ene_i1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,imp_ene_i1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                            plt.plot(sDwall_top_surf,imp_ene_i1_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_i1_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_i1_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                        
                plt.figure(r'imp_ene_i2 Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_bot_nodes,imp_ene_i2_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,imp_ene_i2_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_i2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'imp_ene_i2 Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_top_nodes,imp_ene_i2_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_top_surf,imp_ene_i2_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_i2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'imp_ene_i2 Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_bot_nodes,imp_ene_i2_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top_nodes,imp_ene_i2_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,imp_ene_i2_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                            plt.plot(sDwall_top_surf,imp_ene_i2_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_i2_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_i2_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'imp_ene_ion Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_bot_nodes,imp_ene_ion_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,imp_ene_ion_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_ion_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'imp_ene_ion Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_top_nodes,imp_ene_ion_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_top_surf,imp_ene_ion_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_ion_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'imp_ene_ion Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_bot_nodes,imp_ene_ion_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top_nodes,imp_ene_ion_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        if inC_Dwalls == 0:
                            plt.plot(sDwall_bot_surf,imp_ene_ion_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                            plt.plot(sDwall_top_surf,imp_ene_ion_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                        elif inC_Dwalls == 1:
                            plt.plot(sDwall_bot_surf[indsurf_inC_Dwall_bot],imp_ene_ion_surf_plot[indsurf_Dwall_bot[indsurf_inC_Dwall_bot]], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                            plt.plot(sDwall_top_surf[indsurf_inC_Dwall_top],imp_ene_ion_surf_plot[indsurf_Dwall_top[indsurf_inC_Dwall_top]], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    
                plt.figure(r'imp_ene_n1 Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_bot_nodes,imp_ene_n1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                       plt.plot(sDwall_bot_surf,imp_ene_n1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])      
                plt.figure(r'imp_ene_n1 Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_top_nodes,imp_ene_n1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.plot(sDwall_top_surf,imp_ene_n1_surf_plot[indsurf_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])      
                plt.figure(r'imp_ene_n1 Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sDwall_bot_nodes,imp_ene_n1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top_nodes,imp_ene_n1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    elif picvars_plot == 2:
                        plt.plot(sDwall_bot_surf,imp_ene_n1_surf_plot[indsurf_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                        plt.plot(sDwall_top_surf,imp_ene_n1_surf_plot[indsurf_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                    
            if plot_err_interp_mfam == 1:
                plt.figure(r'err_interp_phi Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_phi_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_phi Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_top,err_interp_phi_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_phi Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_phi_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top,err_interp_phi_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")

                plt.figure(r'err_interp_Te Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_Te_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_Te Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_top,err_interp_Te_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_Te Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_Te_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top,err_interp_Te_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'err_interp_jeperp Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_jeperp_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jeperp Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_top,err_interp_jeperp_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jeperp Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_jeperp_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top,err_interp_jeperp_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'err_interp_jetheta Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_jetheta_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jetheta Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_top,err_interp_jetheta_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jetheta Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_jetheta_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top,err_interp_jetheta_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'err_interp_jepara Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_jepara_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jepara Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_top,err_interp_jepara_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jepara Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_jepara_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top,err_interp_jepara_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'err_interp_jez Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_jez_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jez Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_top,err_interp_jez_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jez Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_jez_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top,err_interp_jez_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                
                plt.figure(r'err_interp_jer Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_jer_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jer Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_top,err_interp_jer_plot[bIDfaces_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jer Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot,err_interp_jer_plot[bIDfaces_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top,err_interp_jer_plot[bIDfaces_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
                        
            if plot_err_interp_pic == 1:
                plt.figure(r'err_interp_n Dwall_bot')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot_nodes,err_interp_n_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_n Dwall_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_top_nodes,err_interp_n_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_n Dwall_bot_top')
                if plot_type == 0 or plot_type == 2:
                    plt.semilogy(sDwall_bot_nodes,err_interp_n_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot")
                    plt.semilogy(sDwall_top_nodes,err_interp_n_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top")
            
            if plot_picM_picS_comp == 1:
                plt.figure(r'comp picMpicS imp_ene_i1 Dwall_bot_top')
                plt.plot(sDwall_bot_nodes,imp_ene_i1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot picM")
                plt.plot(sDwall_top_nodes,imp_ene_i1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top picM")
                plt.plot(sDwall_bot_surf,imp_ene_i1_surf_plot[indsurf_Dwall_bot], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot picS")
                plt.plot(sDwall_top_surf,imp_ene_i1_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top picS")

                plt.figure(r'comp picMpicS imp_ene_i2 Dwall_bot_top')
                plt.plot(sDwall_bot_nodes,imp_ene_i2_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot picM")
                plt.plot(sDwall_top_nodes,imp_ene_i2_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top picM")
                plt.plot(sDwall_bot_surf,imp_ene_i2_surf_plot[indsurf_Dwall_bot], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot picS")
                plt.plot(sDwall_top_surf,imp_ene_i2_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top picS")

                plt.figure(r'comp picMpicS imp_ene_ion Dwall_bot_top')
                plt.plot(sDwall_bot_nodes,imp_ene_ion_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot picM")
                plt.plot(sDwall_top_nodes,imp_ene_ion_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top picM")
                plt.plot(sDwall_bot_surf,imp_ene_ion_surf_plot[indsurf_Dwall_bot], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot picS")
                plt.plot(sDwall_top_surf,imp_ene_ion_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top picS")
    
                plt.figure(r'comp picMpicS imp_ene_n1 Dwall_bot_top')
                plt.plot(sDwall_bot_nodes,imp_ene_n1_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot picM")
                plt.plot(sDwall_top_nodes,imp_ene_n1_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top picM")
                plt.plot(sDwall_bot_surf,imp_ene_n1_surf_plot[indsurf_Dwall_bot], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot picS")
                plt.plot(sDwall_top_surf,imp_ene_n1_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top picS")
    
                plt.figure(r'comp picMpicS ji_tot_b Dwall_bot_top')
                plt.semilogy(sDwall_bot_nodes,ji_nodes_plot[inodes_Dwall_bot,jnodes_Dwall_bot], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot picM")
                plt.semilogy(sDwall_top_nodes,ji_nodes_plot[inodes_Dwall_top,jnodes_Dwall_top], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top picM")
                plt.semilogy(sDwall_bot_surf,ji_surf_plot[indsurf_Dwall_bot], linestyle='-.', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" bot picS")
                plt.semilogy(sDwall_top_surf,ji_surf_plot[indsurf_Dwall_top], linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" top picS")

        
        if plot_Awall == 1:
            if plot_dens == 1:
                plt.figure(r'ne Awall')
                if plot_type == 0:
                    plt.plot(sAwall,n_inst_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sAwall_nodes,n_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                    plt.plot(sAwall_nodes,n_inst_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sAwall,n_inst_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sAwall_nodes,n_nodes_plot[inodes_Awall,jnodes_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picM")
#                    plt.plot(sAwall_nodes,n_inst_nodes_plot[inodes_Awall,jnodes_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picM")
                
                plt.figure(r'nn Awall')
                if plot_type == 0:
                    plt.plot(sAwall,nn1_inst_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sAwall_nodes,nn1_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
#                    plt.plot(sAwall_nodes,nn1_inst_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
                elif plot_type == 2:
                    plt.plot(sAwall,nn1_inst_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sAwall_nodes,nn1_nodes_plot[inodes_Awall,jnodes_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picM")
#                    plt.plot(sAwall_nodes,nn1_inst_nodes_plot[inodes_Awall,jnodes_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picM")

            if plot_dphi_Te == 1:
                plt.figure(r'dphi_sh Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,dphi_sh_b_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'dphi_sh_Te Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,dphi_sh_b_plot[bIDfaces_Awall]/Te_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'Te Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,Te_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'phi Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,phi_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
            if plot_curr == 1:
                plt.figure(r'je_b Awall')
                if plot_type == 0:
                    plt.plot(sAwall,-je_b_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sAwall_nodes,-je_b_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
                elif plot_type == 2:
                    plt.plot(sAwall,-je_b_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sAwall_nodes,-je_b_nodes_plot[inodes_Awall,jnodes_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picM")

                plt.figure(r'ji_tot_b Awall')
                if plot_type == 0:
                    plt.plot(sAwall,ji_tot_b_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    if picvars_plot == 1:
                        plt.plot(sAwall_nodes,ji_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
                    elif picvars_plot == 2:
                        plt.plot(sAwall_surf,ji_surf_plot[indsurf_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
                elif plot_type == 2:
                    plt.plot(sAwall,ji_tot_b_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    if picvars_plot == 1:
                        plt.plot(sAwall_nodes,ji_nodes_plot[inodes_Awall,jnodes_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picM")
                    elif picvars_plot == 2:
                        plt.plot(sAwall_surf,ji_surf_plot[indsurf_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picS") 
                
                plt.figure(r'ji1 Awall')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sAwall_nodes,ji1_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
                    elif picvars_plot == 2:
                        plt.plot(sAwall_surf,ji1_surf_plot[indsurf_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   

                plt.figure(r'ji2 Awall')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sAwall_nodes,ji2_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
                    elif picvars_plot == 2:
                        plt.plot(sAwall_surf,ji2_surf_plot[indsurf_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   

                plt.figure(r'ji2/ji1 Awall')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sAwall_nodes,ji2_nodes_plot[inodes_Awall,jnodes_Awall]/ji1_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
                    elif picvars_plot == 2:
                        plt.plot(sAwall_surf,ji2_surf_plot[indsurf_Awall]/ji1_surf_plot[indsurf_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   

                plt.figure(r'ji1/ji_tot_b Awall')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sAwall_nodes,ji1_nodes_plot[inodes_Awall,jnodes_Awall]/ji_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
                    elif picvars_plot == 2:
                        plt.plot(sAwall_surf,ji1_surf_plot[indsurf_Awall]/ji_surf_plot[indsurf_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   

                plt.figure(r'ji2/ji_tot_b Awall')
                if plot_type == 1 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sAwall_nodes,ji2_nodes_plot[inodes_Awall,jnodes_Awall]/ji_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   
                    elif picvars_plot == 2:
                        plt.plot(sAwall_surf,ji2_surf_plot[indsurf_Awall]/ji_surf_plot[indsurf_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])   

                plt.figure(r'ji_tot_b/j_b Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,ji_tot_b_plot[bIDfaces_Awall]/(-j_b_plot[bIDfaces_Awall]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'je_b/j_b Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,(-je_b_plot[bIDfaces_Awall])/(-j_b_plot[bIDfaces_Awall]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    
                plt.figure(r'j_b Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,-j_b_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'je_b_gp_net_b Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,-je_b_plot[bIDfaces_Awall]/(e*gp_net_b_plot[bIDfaces_Awall]*1E-4), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
                plt.figure(r'relerr_je_b Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,relerr_je_b_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
            if plot_q == 1:
                plt.figure(r'qe_tot_wall Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,qe_tot_wall_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_s_wall Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,qe_tot_s_wall_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_b Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,qe_tot_b_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_b Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,qe_b_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_adv_b Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,qe_adv_b_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_qe_b Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,relerr_qe_b_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_qe_b_cons Awall')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAwall,relerr_qe_b_cons_plot[bIDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
            if plot_imp_ene == 1:
                plt.figure(r'imp_ene_i1 Awall')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sAwall_nodes,imp_ene_i1_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.plot(sAwall_surf,imp_ene_i1_surf_plot[indsurf_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'imp_ene_i2 Awall')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sAwall_nodes,imp_ene_i2_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.plot(sAwall_surf,imp_ene_i2_surf_plot[indsurf_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'imp_ene_ion Awall')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sAwall_nodes,imp_ene_ion_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.plot(sAwall_surf,imp_ene_ion_surf_plot[indsurf_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'imp_ene_n1 Awall')
                if plot_type == 0 or plot_type == 2:
                    if picvars_plot == 1:
                        plt.plot(sAwall_nodes,imp_ene_n1_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    elif picvars_plot == 2:
                        plt.plot(sAwall_surf,imp_ene_ion_surf_plot[indsurf_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                
            if plot_err_interp_mfam == 1:
                plt.figure(r'err_interp_phi Awall')
                plt.plot(sAwall,err_interp_phi_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_Te Awall')
                plt.plot(sAwall,err_interp_Te_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jeperp Awall')
                plt.plot(sAwall,err_interp_jeperp_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jetheta Awall')
                plt.plot(sAwall,err_interp_jetheta_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jepara Awall')
                plt.plot(sAwall,err_interp_jepara_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jez Awall')
                plt.plot(sAwall,err_interp_jez_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jer Awall')
                plt.plot(sAwall,err_interp_jer_plot[IDfaces_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])                
            
            if plot_err_interp_pic == 1:
                plt.figure(r'err_interp_n Awall')
                plt.semilogy(sAwall_nodes,err_interp_n_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
            if plot_picM_picS_comp == 1:
                plt.figure(r'comp picMpicS imp_ene_i1 Awall')
                plt.plot(sAwall_nodes,imp_ene_i1_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picM")
                plt.plot(sAwall_surf,imp_ene_i1_surf_plot[indsurf_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picS")
                plt.figure(r'comp picMpicS imp_ene_i2 Awall')
                plt.plot(sAwall_nodes,imp_ene_i2_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picM")
                plt.plot(sAwall_surf,imp_ene_i2_surf_plot[indsurf_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picS")
                plt.figure(r'comp picMpicS imp_ene_ion Awall')
                plt.plot(sAwall_nodes,imp_ene_ion_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picM")
                plt.plot(sAwall_surf,imp_ene_ion_surf_plot[indsurf_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picS")
                plt.figure(r'comp picMpicS imp_ene_n1 Awall')
                plt.plot(sAwall_nodes,imp_ene_n1_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picM")
                plt.plot(sAwall_surf,imp_ene_n1_surf_plot[indsurf_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picS")
                plt.figure(r'comp picMpicS ji_tot_b Awall')
                plt.semilogy(sAwall_nodes,ji_nodes_plot[inodes_Awall,jnodes_Awall], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picM")
                plt.semilogy(sAwall_surf,ji_surf_plot[indsurf_Awall], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" picS")
                
                
        if plot_FLwall == 1:
            if plot_dens == 1:                                
                plt.figure(r'ne FLwall_ver')
                if plot_type == 0:
                    plt.plot(sFLwall_ver,n_inst_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sFLwall_ver_nodes,n_nodes_plot[inodes_FLwall_ver,jnodes_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sFLwall_ver,n_inst_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sFLwall_ver_nodes,n_nodes_plot[inodes_FLwall_ver,jnodes_FLwall_ver], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

                plt.figure(r'ne FLwall_lat')
                if plot_type == 0:
                    plt.plot(sFLwall_lat,n_inst_plot[IDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sFLwall_lat_nodes,n_nodes_plot[inodes_FLwall_lat,jnodes_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sFLwall_lat,n_inst_plot[IDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sFLwall_lat_nodes,n_nodes_plot[inodes_FLwall_lat,jnodes_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")
                
                plt.figure(r'ne FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,n_inst_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sFLwall_lat,n_inst_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

                plt.figure(r'nn FLwall_ver')
                if plot_type == 0: 
                    plt.plot(sFLwall_ver,nn1_inst_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sFLwall_ver_nodes,nn1_nodes_plot[inodes_FLwall_ver,jnodes_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sFLwall_ver,nn1_inst_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sFLwall_ver_nodes,nn1_nodes_plot[inodes_FLwall_ver,jnodes_FLwall_ver], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

                plt.figure(r'nn FLwall_lat')
                if plot_type == 0:
                    plt.plot(sFLwall_lat,nn1_inst_plot[IDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sFLwall_lat_nodes,nn1_nodes_plot[inodes_FLwall_lat,jnodes_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sFLwall_lat,nn1_inst_plot[IDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sFLwall_lat_nodes,nn1_nodes_plot[inodes_FLwall_lat,jnodes_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

                plt.figure(r'nn FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,nn1_inst_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sFLwall_lat,nn1_inst_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

            if plot_dphi_Te == 1:            
                plt.figure(r'Te FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,Te_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'Te FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,Te_plot[IDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'Te FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,Te_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,Te_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

                plt.figure(r'phi FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,phi_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'phi FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,phi_plot[IDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'phi FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,phi_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,phi_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

            if plot_curr == 1:
                plt.figure(r'je_b FLwall_ver')
                if plot_type == 0:
                    plt.plot(sFLwall_ver,-je_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sFLwall_ver_nodes,-je_b_nodes_plot[inodes_FLwall_ver,jnodes_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sFLwall_ver,-je_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sFLwall_ver_nodes,-je_b_nodes_plot[inodes_FLwall_ver,jnodes_FLwall_ver], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

                plt.figure(r'je_b FLwall_lat')
                if plot_type == 0:
                    plt.plot(sFLwall_lat,-je_b_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sFLwall_lat_nodes,-je_b_nodes_plot[inodes_FLwall_lat,jnodes_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sFLwall_lat,-je_b_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sFLwall_lat_nodes,-je_b_nodes_plot[inodes_FLwall_lat,jnodes_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

                plt.figure(r'je_b FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,-je_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,-je_b_plot[bIDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")
                
                plt.figure(r'ji_tot_b FLwall_ver')
                if plot_type == 0:
                    plt.plot(sFLwall_ver,ji_tot_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sFLwall_ver_nodes,ji_nodes_plot[inodes_FLwall_ver,jnodes_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sFLwall_ver,ji_tot_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sFLwall_ver_nodes,ji_nodes_plot[inodes_FLwall_ver,jnodes_FLwall_ver], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")
                
                plt.figure(r'ji_tot_b FLwall_lat')
                if plot_type == 0:
                    plt.plot(sFLwall_lat,ji_tot_b_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sFLwall_lat_nodes,ji_nodes_plot[inodes_FLwall_lat,jnodes_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sFLwall_lat,ji_tot_b_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sFLwall_lat_nodes,ji_nodes_plot[inodes_FLwall_lat,jnodes_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

                plt.figure(r'ji_tot_b FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,ji_tot_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,ji_tot_b_plot[bIDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

                plt.figure(r'j_b FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,j_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'j_b FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,j_b_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'j_b FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,j_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,j_b_plot[bIDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

                plt.figure(r'relerr_je_b FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,relerr_je_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_je_b FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,relerr_je_b_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_je_b FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,relerr_je_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,relerr_je_b_plot[bIDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

            if plot_q == 1:
                plt.figure(r'qe_tot_wall FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,qe_tot_wall_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_wall FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,qe_tot_wall_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_wall FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,qe_tot_wall_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,qe_tot_wall_plot[bIDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

                plt.figure(r'qe_tot_b FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,qe_tot_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_b FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,qe_tot_b_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_tot_b FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,qe_tot_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,qe_tot_b_plot[bIDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

                plt.figure(r'qe_b FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,qe_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_b FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,qe_b_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_b FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,qe_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,qe_b_plot[bIDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")
                
                plt.figure(r'qe_adv_b FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,qe_adv_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_adv_b FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,qe_adv_b_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'qe_adv_b FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,qe_adv_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,qe_adv_b_plot[bIDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")
                
                plt.figure(r'relerr_qe_b FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,relerr_qe_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_qe_b FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,relerr_qe_b_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_qe_b FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,relerr_qe_b_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,relerr_qe_b_plot[bIDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")
               
                plt.figure(r'relerr_qe_b_cons FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,relerr_qe_b_cons_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_qe_b_cons FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,relerr_qe_b_cons_plot[bIDfaces_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_qe_b_cons FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,relerr_qe_b_cons_plot[bIDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,relerr_qe_b_cons_plot[bIDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")
            
            if plot_err_interp_mfam == 1:
                plt.figure(r'err_interp_phi FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_phi_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_phi FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,err_interp_phi_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_phi FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_phi_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,err_interp_phi_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

                plt.figure(r'err_interp_Te FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_Te_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_Te FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,err_interp_Te_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_Te FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_Te_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,err_interp_Te_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

                plt.figure(r'err_interp_jeperp FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_jeperp_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jeperp FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,err_interp_jeperp_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jeperp FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_jeperp_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,err_interp_jeperp_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

                plt.figure(r'err_interp_jetheta FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_jetheta_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jetheta FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,err_interp_jetheta_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jetheta FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_jetheta_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,err_interp_jetheta_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

                plt.figure(r'err_interp_jepara FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_jepara_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jepara FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,err_interp_jepara_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jepara FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_jepara_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,err_interp_jepara_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

                plt.figure(r'err_interp_jez FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_jez_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jez FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,err_interp_jez_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jez FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_jez_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,err_interp_jez_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

                plt.figure(r'err_interp_jer FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_jer_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jer FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat,err_interp_jer_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jer FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver,err_interp_jer_plot[IDfaces_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat,err_interp_jer_plot[IDfaces_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")

            if plot_err_interp_pic == 1:
                plt.figure(r'err_interp_n FLwall_ver')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver_nodes,err_interp_n_nodes_plot[inodes_FLwall_ver,jnodes_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_n FLwall_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_lat_nodes,err_interp_n_nodes_plot[inodes_FLwall_lat,jnodes_FLwall_lat], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_n FLwall_ver_lat')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sFLwall_ver_nodes,err_interp_n_nodes_plot[inodes_FLwall_ver,jnodes_FLwall_ver], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" ver")
                    plt.plot(sFLwall_lat_nodes,err_interp_n_nodes_plot[inodes_FLwall_lat,jnodes_FLwall_lat], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" lat")
                
        if plot_Axis == 1:
            if plot_dens == 1:
                plt.figure(r'ne Axis')
                if plot_type == 0:
                    plt.plot(sAxis,n_inst_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sAxis_nodes,n_nodes_plot[inodes_Axis,jnodes_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                    plt.plot(sAxis_nodes,n_inst_nodes_plot[inodes_Axis,jnodes_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sAxis,n_inst_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sAxis_nodes,n_nodes_plot[inodes_Axis,jnodes_Axis], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")
#                    plt.plot(sAxis_nodes,n_inst_nodes_plot[inodes_Axis,jnodes_Axis], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

                plt.figure(r'nn Axis')
                if plot_type == 0:
                    plt.plot(sAxis,nn1_inst_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sAxis_nodes,nn1_nodes_plot[inodes_Axis,jnodes_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                    plt.plot(sAxis_nodes,nn1_inst_nodes_plot[inodes_Axis,jnodes_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sAxis,nn1_inst_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sAxis_nodes,nn1_nodes_plot[inodes_Axis,jnodes_Axis], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")
#                    plt.plot(sAxis_nodes,nn1_inst_nodes_plot[inodes_Axis,jnodes_Axis], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

            if plot_dphi_Te == 1:
                plt.figure(r'Te Axis')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAxis,Te_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'phi Axis')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAxis,phi_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            if plot_curr == 1:
                plt.figure(r'je_b Axis')
                if plot_type == 0:
                    plt.plot(sAxis,-je_b_plot[bIDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sAxis_nodes,-je_b_nodes_plot[inodes_Axis,jnodes_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sAxis,-je_b_plot[bIDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sAxis_nodes,-je_b_nodes_plot[inodes_Axis,jnodes_Axis], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

                plt.figure(r'ji_tot_b Axis')
                if plot_type == 0:
                    plt.plot(sAxis,ji_tot_b_plot[bIDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:   
                    plt.plot(sAxis_nodes,ji_nodes_plot[inodes_Axis,jnodes_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sAxis,ji_tot_b_plot[bIDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sAxis_nodes,ji_nodes_plot[inodes_Axis,jnodes_Axis], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")
                
                plt.figure(r'j_b Axis')
                if plot_type == 0:
                    plt.plot(sAxis,-j_b_plot[bIDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 1:
                    plt.plot(sAxis_nodes,-j_b_nodes_plot[inodes_Axis,jnodes_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                elif plot_type == 2:
                    plt.plot(sAxis,-j_b_plot[bIDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                    plt.plot(sAxis_nodes,-j_b_nodes_plot[inodes_Axis,jnodes_Axis], linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k]+" pic bn")

                plt.figure(r'relerr_je_b Axis')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAxis,relerr_je_b_plot[bIDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            if plot_q == 1:
                plt.figure(r'relerr_qe_b Axis')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAxis,relerr_qe_b_plot[bIDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'relerr_qe_b_cons Axis')
                if plot_type == 0 or plot_type == 2:
                    plt.plot(sAxis,relerr_qe_b_cons_plot[bIDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            if plot_err_interp_mfam == 1:
                plt.figure(r'err_interp_phi Axis')
                plt.plot(sAxis,err_interp_phi_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_Te Axis')
                plt.plot(sAxis,err_interp_Te_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jeperp Axis')
                plt.plot(sAxis,err_interp_jeperp_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jetheta Axis')
                plt.plot(sAxis,err_interp_jetheta_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jepara Axis')
                plt.plot(sAxis,err_interp_jepara_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jez Axis')
                plt.plot(sAxis,err_interp_jez_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
                plt.figure(r'err_interp_jer Axis')
                plt.plot(sAxis,err_interp_jer_plot[IDfaces_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            if plot_err_interp_pic == 1:
                plt.figure(r'err_interp_n Axis')
                plt.plot(sAxis_nodes,err_interp_n_nodes_plot[inodes_Axis,jnodes_Axis], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every, markersize=marker_size, marker=markers[ind], color=colors[ind], markeredgecolor = 'k', label=labels[k])

        
        ind = ind + 1
        if ind > 8:
            ind = 0
            ind2 = ind2 + 1
            if ind2 > 6:
                ind = 0
                ind2 = 0
                ind3 = ind3 + 1
        
    if plot_wall == 1:
        if plot_deltas == 1:
            plt.figure(r'delta_r wall')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
            plt.figure(r'delta_s wall')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=3) 
            plt.figure(r'delta_s_csl wall')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=3) 

        if plot_dphi_Te == 1:
            plt.figure(r'dphi_sh wall')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=3) 
            plt.figure(r'dphi_sh_Te wall')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'Te wall')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'phi wall')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=3) 
                
        if plot_curr == 1:
            plt.figure(r'je_b wall')
            ax = plt.gca()
            ax.set_ylim(1E-7,1E0)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=3) 
            plt.figure(r'ji_tot_b wall')
            ax = plt.gca()
            ax.set_ylim(1E-7,1E0)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'egn wall')
            ax = plt.gca()
            ax.set_ylim(1E-7,1E0)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'j_b wall') 
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
#            ax.set_xlim(7.0,8.0)
#            ax.set_ylim(-0.01,0.0025)
            plt.figure(r'ji_tot_b je_b wall')
            ax = plt.gca()
            ax.set_ylim(1E-7,1E0)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=2) 
#            ax.set_xlim(7.0,8.0)
#            ax.set_ylim(-0.0025,0.0150)
            plt.figure(r'relerr_je_b wall')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
        
        if plot_imp_ene == 1:
            plt.figure(r'imp_ene_ion wall')
            ax = plt.gca()
            ax.set_ylim(1E0,1E4)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_neu wall')
            ax = plt.gca()
            ax.set_ylim(1E0,1E4)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_e wall')
            ax = plt.gca()
            ax.set_ylim(1E0,1E4)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            
        if plot_q == 1:
            plt.figure(r'qi_tot_wall wall')
            ax = plt.gca()
            ax.set_ylim(1E-5,1E2)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qn_tot_wall wall')
            ax = plt.gca()
            ax.set_ylim(1E-5,1E2)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_wall wall')
            ax = plt.gca()
            ax.set_ylim(1E-5,1E2)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_b wall')
            ax = plt.gca()
            ax.set_ylim(1E-5,1E2)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_b wall')
            ax = plt.gca()
            ax.set_ylim(1E-5,1E2)
            ylims = ax.get_ylim()
#            ax.set_ylim(ylims[0],ylims[1])
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_adv_b wall')
            ax = plt.gca()
            ax.set_ylim(1E-5,1E2)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b wall')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b_cons wall')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qi_tot_wall qe_tot_wall wall')
            ax = plt.gca()
            ax.set_ylim(1E-5,1E2)
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(Lbot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lbot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            elif inC_Dwalls == 1:
                plt.plot(Lchamb_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot((Lchamb_bot+Lanode)*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=1) 
    if plot_down == 1:
        if plot_B == 1:
            plt.figure(r'Bfield down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_dens == 1:
            plt.figure(r'ne down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_dphi_Te == 1:
            plt.figure(r'dphi_sh down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'dphi_sh_Te down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'Te down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            # ax.set_ylim(0,5)
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'phi down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            # ax.set_ylim(0,12)
            # ax.set_ylim(0,16)
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'dphi_inf down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'dphi_inf_Te down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            
        if plot_curr == 1:
            plt.figure(r'j_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'je_b down')
            ax = plt.gca()
            # ax.set_ylim(1E-5,1E1)
            ax.set_ylim(1E-5,1E0)
            ylims = ax.get_ylim()
#            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji_tot_b down')
            ax = plt.gca()
            # ax.set_ylim(1E-5,1E1)
            ax.set_ylim(1E-5,1E0)
            ylims = ax.get_ylim()
#            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji_tot_b je_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=2) 
            plt.figure(r'relerr_je_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji1 down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji2 down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji3 down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji4 down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            
            
        if plot_q == 1:
            plt.figure(r'qi_tot_b down')
            ax = plt.gca()
            # ax.set_ylim(1E-4,2E1)
            ax.set_ylim(1E-5,1E2)
            ylims = ax.get_ylim()
#            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_wall down')
            ax = plt.gca()
            # ax.set_ylim(1E-4,2E1)
            ax.set_ylim(1E-5,1E2)
            ylims = ax.get_ylim()
#            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_b down')
            ax = plt.gca()
            # ax.set_ylim(1E-4,2E1)
            ax.set_ylim(1E-5,1E2)
            ylims = ax.get_ylim()
#            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            # ax.set_ylim(ylims[0],ylims[1])
            ax.set_ylim(1E-5,1E2)
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_adv_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            # ax.set_ylim(ylims[0],ylims[1])
            ax.set_ylim(1E-5,1E2)
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ratio_qe_b_qe_adv_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ratio_qe_b_qe_tot_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            # ax.set_ylim(ylims[0],ylims[1])
            ax.set_ylim(0,1)
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ratio_qe_adv_b_qe_tot_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            # ax.set_ylim(ylims[0],ylims[1])
            ax.set_ylim(0,1)
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'je_dphi_sh down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            # ax.set_ylim(ylims[0],ylims[1])
            # ax.set_ylim(1E-5,1E2)
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ratio_je_dphi_sh_qe_tot_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            # ax.set_ylim(ylims[0],ylims[1])
            ax.set_ylim(0,1)
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ratio_qe_tot_wall_qe_tot_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            # ax.set_ylim(ylims[0],ylims[1])
            ax.set_ylim(0,1)
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ratio_eqe_tot_b_je_bTe down')            
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b_cons down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_imp_ene == 1:
            plt.figure(r'imp_ene_ion down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_ion1 down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_ion2 down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_ion3 down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_ion4 down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_e_b down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1)  
            plt.figure(r'imp_ene_e_b_Te down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_e_wall down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1)  
            plt.figure(r'imp_ene_e_wall_Te down')
            ax = plt.gca()
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(0.0,np.ceil(sdown[-1]))
#            plt.plot(Lfreeloss_lat*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            # plt.plot(Lfreeloss_ver*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.axvline(x=Lfreeloss_ver, linestyle=':',color='k')
            plt.axvline(x=rs[rind,0], linestyle='--',color='k')
            if labels[0] != '':
                plt.legend(fontsize = font_size_legend,loc=1) 

                
    if plot_Dwall == 1:
        if plot_dens == 1:
            plt.figure(r'ne Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ne Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ne Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=4) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)

            plt.figure(r'nn Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'nn Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'nn Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_deltas == 1:
            plt.figure(r'delta_r Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            plt.figure(r'delta_r Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            plt.figure(r'delta_r Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
    
            plt.figure(r'delta_s Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'delta_s Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'delta_s Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend-2,loc=4) 
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
        if plot_dphi_Te == 1:
            plt.figure(r'dphi_sh Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'dphi_sh Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'dphi_sh Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            
            plt.figure(r'dphi_sh_Te Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'dphi_sh_Te Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'dphi_sh_Te Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
#            plt.legend(fontsize = font_size_legend,loc=3) 
            plt.legend(fontsize = font_size_legend-1,loc=1) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            
            plt.figure(r'Te Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'Te Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'Te Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
        
            plt.figure(r'phi Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'phi Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'phi Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1)
        if plot_curr == 1:
            plt.figure(r'je_b Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'je_b Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'je_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'ji_tot_b Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji_tot_b Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji_tot_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            
            plt.figure(r'ji1 Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji1 Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji1 Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            
            plt.figure(r'ji2 Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji2 Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji2 Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            
            plt.figure(r'ji2/ji1 Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji2/ji1 Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji2/ji1 Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            
            plt.figure(r'ji1/ji_tot_b Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1)
            plt.figure(r'ji1/ji_tot_b Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji1/ji_tot_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            
            plt.figure(r'ji2/ji_tot_b Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1)
            plt.figure(r'ji2/ji_tot_b Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji2/ji_tot_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
    
            plt.figure(r'j_b Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'j_b Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'j_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'je_b_gp_net_b Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'je_b_gp_net_b Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'je_b_gp_net_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'relerr_je_b Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_je_b Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_je_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_q == 1:
            plt.figure(r'qe_tot_wall Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            plt.figure(r'qe_tot_wall Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            plt.figure(r'qe_tot_wall Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
            
            plt.figure(r'qe_tot_s_wall Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_s_wall Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_s_wall Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'qe_tot_b Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_b Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'qe_b Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_b Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'qe_adv_b Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_adv_b Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_adv_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'relerr_qe_b Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'relerr_qe_b_cons Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b_cons Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b_cons Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'qi1 Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            plt.figure(r'qi1 Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            plt.figure(r'qi1 Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)

            plt.figure(r'qi2 Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            plt.figure(r'qi2 Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2)
            plt.figure(r'qi2 Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)

            plt.figure(r'qion Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            plt.figure(r'qion Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2)
            plt.figure(r'qion Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            if inC_Dwalls == 0:
                plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=2) 
            xlim = ax.get_xlim()
#            ax.set_xlim(xlim[0],3.5)
        if plot_imp_ene == 1:
            plt.figure(r'imp_ene_i1 Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_i1 Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_i1 Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'imp_ene_i2 Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_i2 Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_i2 Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'imp_ene_ion Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_ion Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_ion Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'imp_ene_n1 Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_n1 Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'imp_ene_n1 Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_err_interp_mfam == 1:
            plt.figure(r'err_interp_phi Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1)
            plt.figure(r'err_interp_phi Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_phi Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'err_interp_Te Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1)
            plt.figure(r'err_interp_Te Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_Te Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'err_interp_jeperp Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1)
            plt.figure(r'err_interp_jeperp Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jeperp Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'err_interp_jetheta Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1)
            plt.figure(r'err_interp_jetheta Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jetheta Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'err_interp_jepara Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1)
            plt.figure(r'err_interp_jepara Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jepara Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'err_interp_jez Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1)
            plt.figure(r'err_interp_jez Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jez Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'err_interp_jer Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1)
            plt.figure(r'err_interp_jer Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jer Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_err_interp_pic == 1:
            plt.figure(r'err_interp_n Dwall_bot')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1)
            plt.figure(r'err_interp_n Dwall_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_n Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_picM_picS_comp == 1:
            plt.figure(r'comp picMpicS imp_ene_i1 Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'comp picMpicS imp_ene_i2 Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'comp picMpicS imp_ene_ion Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'comp picMpicS imp_ene_n1 Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'comp picMpicS ji_tot_b Dwall_bot_top')
            ax = plt.gca()
            ylims = ax.get_ylim()
            plt.plot(sc_bot*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.plot(sc_top*np.ones(2),np.array([ylims[0],ylims[1]]), linestyle='--', linewidth = line_width, markevery=marker_every, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
            plt.legend(fontsize = font_size_legend,loc=1) 

    if plot_Awall == 1:
        if plot_dens == 1:
            plt.figure(r'ne Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'nn Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_dphi_Te == 1:
            plt.figure(r'dphi_sh Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'dphi_sh_Te Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'Te Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'phi Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_curr == 1:
            plt.figure(r'je_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji_tot_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji1 Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji2 Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji2/ji1 Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji1/ji_tot_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji2/ji_tot_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji_tot_b/j_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'je_b/j_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'j_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'je_b_gp_net_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_je_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_q == 1:
            plt.figure(r'qe_tot_wall Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_s_wall Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_adv_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b_cons Awall')
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
        if plot_err_interp_mfam == 1:
            plt.figure(r'err_interp_phi Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_Te Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jeperp Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jetheta Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jepara Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jez Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jer Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_err_interp_pic == 1:
            plt.figure(r'err_interp_n Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_picM_picS_comp == 1:
            plt.figure(r'comp picMpicS imp_ene_i1 Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'comp picMpicS imp_ene_i2 Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'comp picMpicS imp_ene_ion Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'comp picMpicS imp_ene_n1 Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'comp picMpicS ji_tot_b Awall')
            plt.legend(fontsize = font_size_legend,loc=1) 
        
    if plot_FLwall == 1:
        if plot_dens == 1:
            plt.figure(r'ne FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ne FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ne FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'nn FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'nn FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'nn FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_dphi_Te == 1:            
            plt.figure(r'Te FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'Te FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'Te FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'phi FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'phi FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'phi FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_curr == 1:
            plt.figure(r'je_b FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'je_b FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'je_b FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'ji_tot_b FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji_tot_b FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji_tot_b FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'j_b FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'j_b FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'j_b FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'relerr_je_b FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_je_b FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_je_b FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_q == 1:
            plt.figure(r'qe_tot_wall FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_wall FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_wall FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'qe_tot_b FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_b FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_tot_b FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'qe_b FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_b FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_b FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'qe_adv_b FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_adv_b FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'qe_adv_b FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'relerr_qe_b FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 

            plt.figure(r'relerr_qe_b_cons FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b_cons FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b_cons FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_err_interp_mfam == 1:
            plt.figure(r'err_interp_phi FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_phi FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_phi FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'err_interp_Te FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_Te FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_Te FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'err_interp_jeperp FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jeperp FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jeperp FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'err_interp_jetheta FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jetheta FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jetheta FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'err_interp_jepara FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jepara FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jepara FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'err_interp_jez FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jez FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jez FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            
            plt.figure(r'err_interp_jer FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1)  
            plt.figure(r'err_interp_jer FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jer FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1)   
        if plot_err_interp_pic == 1:
            plt.figure(r'err_interp_n FLwall_ver')
            plt.legend(fontsize = font_size_legend,loc=1)  
            plt.figure(r'err_interp_n FLwall_lat')
            plt.legend(fontsize = font_size_legend,loc=1)  
            plt.figure(r'err_interp_n FLwall_ver_lat')
            plt.legend(fontsize = font_size_legend,loc=1)  
            
    if plot_Axis == 1:
        if plot_dens == 1:
            plt.figure(r'ne Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'nn Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_dphi_Te == 1:
            plt.figure(r'Te Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'phi Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_curr == 1:
            plt.figure(r'je_b Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'ji_tot_b Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'j_b Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_je_b Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_q == 1:
            plt.figure(r'relerr_qe_b Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'relerr_qe_b_cons Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_err_interp_mfam == 1:
            plt.figure(r'err_interp_phi Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_Te Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jeperp Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jetheta Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jepara Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jez Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
            plt.figure(r'err_interp_jer Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
        if plot_err_interp_pic == 1:
            plt.figure(r'err_interp_n Axis')
            plt.legend(fontsize = font_size_legend,loc=1) 
     
    if save_flag == 1:
        if plot_wall == 1:
            if plot_deltas == 1:
                plt.figure(r'delta_r wall')
                plt.savefig(path_out+"delta_r_wall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'delta_s wall')
                plt.savefig(path_out+"delta_s_wall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'delta_s_csl wall')
                plt.savefig(path_out+"delta_s_csl_wall"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_dphi_Te == 1:
                plt.figure(r'dphi_sh wall')
                plt.savefig(path_out+"dphi_sh_b_wall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'dphi_sh_Te wall')
                plt.savefig(path_out+"dphi_sh_b_Te_wall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'Te wall')
                plt.savefig(path_out+"Te_wall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'phi wall')
                plt.savefig(path_out+"phi_wall"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_curr == 1:                
                plt.figure(r'je_b wall')
                plt.savefig(path_out+"je_b_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji_tot_b wall')
                plt.savefig(path_out+"ji_tot_b_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'egn wall')
                plt.savefig(path_out+"egn_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'j_b wall') 
                plt.savefig(path_out+"j_b_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji_tot_b je_b wall')
                plt.savefig(path_out+"ji_tot_b_je_b_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'relerr_je_b wall')
                plt.savefig(path_out+"relerr_je_b_wall"+figs_format,bbox_inches='tight')
                plt.close()
            if plot_imp_ene == 1:
                plt.figure(r'imp_ene_ion wall')
                plt.savefig(path_out+"imp_ene_ion_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_neu wall')
                plt.savefig(path_out+"imp_ene_neu_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_e wall')
                plt.savefig(path_out+"imp_ene_e_wall"+figs_format,bbox_inches='tight')
                plt.close()
            if plot_q == 1:
                plt.figure(r'qi_tot_wall wall')
                plt.savefig(path_out+"qi_tot_wall_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'qn_tot_wall wall')
                plt.savefig(path_out+"qn_tot_wall_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'qe_tot_wall wall')
                plt.savefig(path_out+"qe_tot_wall_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'qe_tot_b wall')
                plt.savefig(path_out+"qe_tot_b_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'qe_b wall')
                plt.savefig(path_out+"qe_b_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'qe_adv_b wall')
                plt.savefig(path_out+"qe_adv_b_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'relerr_qe_b wall')
                plt.savefig(path_out+"relerr_qe_b_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'relerr_qe_b_cons wall')
                plt.savefig(path_out+"relerr_qe_b_cons_wall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'qi_tot_wall qe_tot_wall wall')
                plt.savefig(path_out+"qi_tot_wall_qe_tot_wall_wall"+figs_format,bbox_inches='tight')
                plt.close()
                
        if plot_down == 1:
            if plot_B == 1:
                plt.figure(r'Bfield down')
                plt.savefig(path_out+"Bfield_down"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_dens == 1:
                plt.figure(r'ne down')
                plt.savefig(path_out+"ne_down"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_dphi_Te == 1:
                plt.figure(r'dphi_sh down')
                plt.savefig(path_out+"dphi_sh_b_down"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'dphi_sh_Te down')
                plt.savefig(path_out+"dphi_sh_b_Te_down"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'Te down')
                plt.savefig(path_out+"Te_down"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'phi down')
                plt.savefig(path_out+"phi_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'dphi_inf down')
                plt.savefig(path_out+"dphi_inf_down"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'dphi_inf_Te down')
                plt.savefig(path_out+"dphi_inf_Te_down"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_curr == 1:
                plt.figure(r'j_b down')
                plt.savefig(path_out+"j_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'je_b down')
                plt.savefig(path_out+"je_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji_tot_b down')
                plt.savefig(path_out+"ji_tot_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji_tot_b je_b down')
                plt.savefig(path_out+"ji_tot_b_je_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'relerr_je_b down')
                plt.savefig(path_out+"relerr_je_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji1 down')
                plt.savefig(path_out+"ji1_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji2 down')
                plt.savefig(path_out+"ji2_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji3 down')
                plt.savefig(path_out+"ji3_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ji4 down')
                plt.savefig(path_out+"ji4_down"+figs_format,bbox_inches='tight')
                plt.close()
                
            if plot_q == 1:
                plt.figure(r'qi_tot_b down')
                plt.savefig(path_out+"qi_tot_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'qe_tot_wall down')
                plt.savefig(path_out+"qe_tot_wall_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'qe_tot_b down')
                plt.savefig(path_out+"qe_tot_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'qe_b down')
                plt.savefig(path_out+"qe_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'qe_adv_b down')
                plt.savefig(path_out+"qe_adv_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ratio_qe_b_qe_adv_b down')
                plt.savefig(path_out+"ratio_qe_b_qe_adv_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ratio_qe_b_qe_tot_b down')
                plt.savefig(path_out+"ratio_qe_b_qe_tot_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ratio_qe_adv_b_qe_tot_b down')
                plt.savefig(path_out+"ratio_qe_adv_b_qe_tot_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'je_dphi_sh down')
                plt.savefig(path_out+"je_dphi_sh_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ratio_je_dphi_sh_qe_tot_b down')
                plt.savefig(path_out+"ratio_je_dphi_sh_qe_tot_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ratio_qe_tot_wall_qe_tot_b down')
                plt.savefig(path_out+"ratio_qe_tot_wall_qe_tot_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'ratio_eqe_tot_b_je_bTe down') 
                plt.savefig(path_out+"ratio_eqe_tot_b_je_bTe_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'relerr_qe_b down')
                plt.savefig(path_out+"relerr_qe_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'relerr_qe_b_cons down')
                plt.savefig(path_out+"relerr_qe_b_cons_down"+figs_format,bbox_inches='tight')
                plt.close()
            if plot_imp_ene == 1:
                plt.figure(r'imp_ene_ion down')
                plt.savefig(path_out+"imp_ene_ion_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_ion1 down')
                plt.savefig(path_out+"imp_ene_i1_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_ion2 down')
                plt.savefig(path_out+"imp_ene_i2_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_ion3 down')
                plt.savefig(path_out+"imp_ene_i3_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_ion4 down')
                plt.savefig(path_out+"imp_ene_i4_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_e_b down')
                plt.savefig(path_out+"imp_ene_e_b_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_e_b_Te down')
                plt.savefig(path_out+"imp_ene_e_b_Te_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_e_wall down')
                plt.savefig(path_out+"imp_ene_e_wall_down"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_e_wall_Te down')
                plt.savefig(path_out+"imp_ene_e_wall_Te_down"+figs_format,bbox_inches='tight')
                plt.close()

        if plot_Dwall == 1:
            if plot_dens == 1:
#                plt.figure(r'ne Dwall_bot')
#                plt.savefig(path_out+"ne_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'ne Dwall_top')
#                plt.savefig(path_out+"ne_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close() 
                plt.figure(r'ne Dwall_bot_top')
                plt.savefig(path_out+"ne_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 
                
#                plt.figure(r'nn Dwall_bot')
#                plt.savefig(path_out+"nn_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'nn Dwall_top')
#                plt.savefig(path_out+"nn_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'nn Dwall_bot_top')
#                plt.savefig(path_out+"nn_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close() 
            if plot_deltas == 1:
#                plt.figure(r'delta_r Dwall_bot')
#                plt.savefig(path_out+"delta_r_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()     
#                plt.figure(r'delta_r Dwall_top')
#                plt.savefig(path_out+"delta_r_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()  
#                plt.figure(r'delta_r Dwall_bot_top')
#                plt.savefig(path_out+"delta_r_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close()     
                
#                plt.figure(r'delta_s Dwall_bot')
#                plt.savefig(path_out+"delta_s_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()     
#                plt.figure(r'delta_s Dwall_top')
#                plt.savefig(path_out+"delta_s_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()  
                plt.figure(r'delta_s Dwall_bot_top')
                plt.savefig(path_out+"delta_s_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_dphi_Te == 1:
#                plt.figure(r'dphi_sh Dwall_bot')
#                plt.savefig(path_out+"dphi_sh_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'dphi_sh Dwall_top')
#                plt.savefig(path_out+"dphi_sh_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close() 
                plt.figure(r'dphi_sh Dwall_bot_top')
                plt.savefig(path_out+"dphi_sh_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 
                
#                plt.figure(r'dphi_sh_Te Dwall_bot')
#                plt.savefig(path_out+"dphi_sh_b_Te_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'dphi_sh_Te Dwall_top')
#                plt.savefig(path_out+"dphi_sh_b_Te_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close() 
                plt.figure(r'dphi_sh_Te Dwall_bot_top')
                plt.savefig(path_out+"dphi_sh_b_Te_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 
                
#                plt.figure(r'Te Dwall_bot')
#                plt.savefig(path_out+"Te_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'Te Dwall_top')
#                plt.savefig(path_out+"Te_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close() 
                plt.figure(r'Te Dwall_bot_top')
                plt.savefig(path_out+"Te_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 
                
#                plt.figure(r'phi Dwall_bot')
#                plt.savefig(path_out+"phi_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'phi Dwall_top')
#                plt.savefig(path_out+"phi_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'phi Dwall_bot_top')
#                plt.savefig(path_out+"phi_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close() 
            if plot_curr == 1:
#                plt.figure(r'je_b Dwall_bot')
#                plt.savefig(path_out+"je_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'je_b Dwall_top')
#                plt.savefig(path_out+"je_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'je_b Dwall_bot_top')
#                plt.savefig(path_out+"je_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close()
    
#                plt.figure(r'ji_tot_b Dwall_bot')
#                plt.savefig(path_out+"ji_tot_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'ji_tot_b Dwall_top')
#                plt.savefig(path_out+"ji_tot_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
                plt.figure(r'ji_tot_b Dwall_bot_top')
                plt.savefig(path_out+"ji_tot_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                
#                plt.figure(r'ji1 Dwall_bot')
#                plt.savefig(path_out+"ji1_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'ji1 Dwall_top')
#                plt.savefig(path_out+"ji1_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
                plt.figure(r'ji1 Dwall_bot_top')
                plt.savefig(path_out+"ji1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                
#                plt.figure(r'ji2 Dwall_bot')
#                plt.savefig(path_out+"ji2_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'ji2 Dwall_top')
#                plt.savefig(path_out+"ji2_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
                plt.figure(r'ji2 Dwall_bot_top')
                plt.savefig(path_out+"ji2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                
#                plt.figure(r'ji2/ji1 Dwall_bot')
#                plt.savefig(path_out+"ratio_ji2ji1_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'ji2/ji1 Dwall_top')
#                plt.savefig(path_out+"ratio_ji2ji1_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
                plt.figure(r'ji2/ji1 Dwall_bot_top')
                plt.savefig(path_out+"ratio_ji2ji1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                
#                plt.figure(r'ji1/ji_tot_b Dwall_bot')
#                plt.savefig(path_out+"ratio_ji1ji_tot_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'ji1/ji_tot_b Dwall_top')
#                plt.savefig(path_out+"ratio_ji1ji_tot_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
                plt.figure(r'ji1/ji_tot_b Dwall_bot_top')
                plt.savefig(path_out+"ratio_ji1ji_tot_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
    
#                plt.figure(r'ji2/ji_tot_b Dwall_bot')
#                plt.savefig(path_out+"ratio_ji2ji_tot_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'ji2/ji_tot_b Dwall_top')
#                plt.savefig(path_out+"ratio_ji2ji_tot_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
                plt.figure(r'ji2/ji_tot_b Dwall_bot_top')
                plt.savefig(path_out+"ratio_ji2ji_tot_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                
#                plt.figure(r'j_b Dwall_bot')
#                plt.savefig(path_out+"j_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'j_b Dwall_top')
#                plt.savefig(path_out+"j_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'j_b Dwall_bot_top')
#                plt.savefig(path_out+"j_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close() 
                
#                plt.figure(r'je_b_gp_net_b Dwall_bot')
#                plt.savefig(path_out+"je_b_gp_net_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'je_b_gp_net_b Dwall_top')
#                plt.savefig(path_out+"je_b_gp_net_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'je_b_gp_net_b Dwall_bot_top')
#                plt.savefig(path_out+"je_b_gp_net_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close() 
                
#                plt.figure(r'relerr_je_b Dwall_bot')
#                plt.savefig(path_out+"relerr_je_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'relerr_je_b Dwall_top')
#                plt.savefig(path_out+"relerr_je_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'relerr_je_b Dwall_bot_top')
#                plt.savefig(path_out+"relerr_je_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close()
            if plot_q == 1:
#                plt.figure(r'qe_tot_wall Dwall_bot')
#                plt.savefig(path_out+"qe_tot_wall_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qe_tot_wall Dwall_top')
#                plt.savefig(path_out+"qe_tot_wall_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
                plt.figure(r'qe_tot_wall Dwall_bot_top')
                plt.savefig(path_out+"qe_tot_wall_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                
#                plt.figure(r'qe_tot_s_wall Dwall_bot')
#                plt.savefig(path_out+"qe_tot_s_wall_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qe_tot_s_wall Dwall_top')
#                plt.savefig(path_out+"qe_tot_s_wall_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qe_tot_s_wall Dwall_bot_top')
#                plt.savefig(path_out+"qe_tot_s_wall_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close()
                
#                plt.figure(r'qe_tot_b Dwall_bot')
#                plt.savefig(path_out+"qe_tot_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qe_tot_b Dwall_top')
#                plt.savefig(path_out+"qe_tot_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qe_tot_b Dwall_bot_top')
#                plt.savefig(path_out+"qe_tot_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close()
                
#                plt.figure(r'qe_b Dwall_bot')
#                plt.savefig(path_out+"qe_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qe_b Dwall_top')
#                plt.savefig(path_out+"qe_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qe_b Dwall_bot_top')
#                plt.savefig(path_out+"qe_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close()
                
#                plt.figure(r'qe_adv_b Dwall_bot')
#                plt.savefig(path_out+"qe_adv_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qe_adv_b Dwall_top')
#                plt.savefig(path_out+"qe_adv_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qe_adv_b Dwall_bot_top')
#                plt.savefig(path_out+"qe_adv_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close()
                
#                plt.figure(r'relerr_qe_b Dwall_bot')
#                plt.savefig(path_out+"relerr_qe_b_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'relerr_qe_b Dwall_top')
#                plt.savefig(path_out+"relerr_qe_b_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'relerr_qe_b Dwall_bot_top')
#                plt.savefig(path_out+"relerr_qe_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close()
                
#                plt.figure(r'relerr_qe_b_cons Dwall_bot')
#                plt.savefig(path_out+"relerr_qe_b_cons_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'relerr_qe_b_cons Dwall_top')
#                plt.savefig(path_out+"relerr_qe_b_cons_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close() 
#                plt.figure(r'relerr_qe_b_cons Dwall_bot_top')
#                plt.savefig(path_out+"relerr_qe_b_cons_Dwall_bot_top"+figs_format,bbox_inches='tight')
#                plt.close()  
                
#                plt.figure(r'qi1 Dwall_bot')
#                plt.savefig(path_out+"qi1_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qi1 Dwall_top')
#                plt.savefig(path_out+"qi1_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
                plt.figure(r'qi1 Dwall_bot_top')
                plt.savefig(path_out+"qi1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
    
#                plt.figure(r'qi2 Dwall_bot')
#                plt.savefig(path_out+"qi2_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qi2 Dwall_top')
#                plt.savefig(path_out+"qi2_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
                plt.figure(r'qi2 Dwall_bot_top')
                plt.savefig(path_out+"qi2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
    
#                plt.figure(r'qion Dwall_bot')
#                plt.savefig(path_out+"qion_Dwall_bot"+figs_format,bbox_inches='tight')
#                plt.close()
#                plt.figure(r'qion Dwall_top')
#                plt.savefig(path_out+"qion_Dwall_top"+figs_format,bbox_inches='tight')
#                plt.close()
                plt.figure(r'qion Dwall_bot_top')
                plt.savefig(path_out+"qion_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
            if plot_imp_ene == 1:
                plt.figure(r'imp_ene_i1 Dwall_bot')
                plt.savefig(path_out+"imp_ene_i1_Dwall_bot"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_i1 Dwall_top')
                plt.savefig(path_out+"imp_ene_i1_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_i1 Dwall_bot_top')
                plt.savefig(path_out+"imp_ene_i1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                
                plt.figure(r'imp_ene_i2 Dwall_bot')
                plt.savefig(path_out+"imp_ene_i2_Dwall_bot"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_i2 Dwall_top')
                plt.savefig(path_out+"imp_ene_i2_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_i2 Dwall_bot_top')
                plt.savefig(path_out+"imp_ene_i2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                
                plt.figure(r'imp_ene_ion Dwall_bot')
                plt.savefig(path_out+"imp_ene_ion_Dwall_bot"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_ion Dwall_top')
                plt.savefig(path_out+"imp_ene_ion_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_ion Dwall_bot_top')
                plt.savefig(path_out+"imp_ene_ion_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()

                plt.figure(r'imp_ene_n1 Dwall_bot')
                plt.savefig(path_out+"imp_ene_n1_Dwall_bot"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'imp_ene_n1 Dwall_top')
                plt.savefig(path_out+"imp_ene_n1_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'imp_ene_n1 Dwall_bot_top')
                plt.savefig(path_out+"imp_ene_n1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                
            if plot_err_interp_mfam == 1:
                plt.figure(r'err_interp_phi Dwall_bot')
                plt.savefig(path_out+"err_interp_phi_Dwall_bot"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_phi Dwall_top')
                plt.savefig(path_out+"err_interp_phi_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_phi Dwall_bot_top')
                plt.savefig(path_out+"err_interp_phi_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()    

                plt.figure(r'err_interp_Te Dwall_bot')
                plt.savefig(path_out+"err_interp_Te_Dwall_bot"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_Te Dwall_top')
                plt.savefig(path_out+"err_interp_Te_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_Te Dwall_bot_top')
                plt.savefig(path_out+"err_interp_Te_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 

                plt.figure(r'err_interp_jeperp Dwall_bot')
                plt.savefig(path_out+"err_interp_jeperp_Dwall_bot"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jeperp Dwall_top')
                plt.savefig(path_out+"err_interp_jeperp_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jeperp Dwall_bot_top')
                plt.savefig(path_out+"err_interp_jeperp_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 

                plt.figure(r'err_interp_jetheta Dwall_bot')
                plt.savefig(path_out+"err_interp_jetheta_Dwall_bot"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jetheta Dwall_top')
                plt.savefig(path_out+"err_interp_jetheta_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jetheta Dwall_bot_top')
                plt.savefig(path_out+"err_interp_jetheta_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 

                plt.figure(r'err_interp_jepara Dwall_bot')
                plt.savefig(path_out+"err_interp_jepara_Dwall_bot"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jepara Dwall_top')
                plt.savefig(path_out+"err_interp_jepara_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jepara Dwall_bot_top')
                plt.savefig(path_out+"err_interp_jepara_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 

                plt.figure(r'err_interp_jez Dwall_bot')
                plt.savefig(path_out+"err_interp_jez_Dwall_bot"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jez Dwall_top')
                plt.savefig(path_out+"err_interp_jez_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jez Dwall_bot_top')
                plt.savefig(path_out+"err_interp_jez_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 
                
                plt.figure(r'err_interp_jer Dwall_bot')
                plt.savefig(path_out+"err_interp_jer_Dwall_bot"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jer Dwall_top')
                plt.savefig(path_out+"err_interp_jer_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jer Dwall_bot_top')
                plt.savefig(path_out+"err_interp_jer_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_err_interp_pic == 1:
                plt.figure(r'err_interp_n Dwall_bot')
                plt.savefig(path_out+"err_interp_n_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_n Dwall_top')
                plt.savefig(path_out+"err_interp_n_Dwall_top"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_n Dwall_bot_top')
                plt.savefig(path_out+"err_interp_n_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_picM_picS_comp == 1:
                plt.figure(r'comp picMpicS imp_ene_i1 Dwall_bot_top')
                plt.savefig(path_out+"comp_picMpicS_imp_ene_i1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'comp picMpicS imp_ene_i2 Dwall_bot_top')
                plt.savefig(path_out+"comp_picMpicS_imp_ene_i2_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'comp picMpicS imp_ene_ion Dwall_bot_top')
                plt.savefig(path_out+"comp_picMpicS_imp_ene_ion_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'comp picMpicS imp_ene_n1 Dwall_bot_top')
                plt.savefig(path_out+"comp_picMpicS_imp_ene_n1_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'comp picMpicS ji_tot_b Dwall_bot_top')
                plt.savefig(path_out+"comp_picMpicS_ji_tot_b_Dwall_bot_top"+figs_format,bbox_inches='tight')
                plt.close() 
    
        if plot_Awall == 1:
            if plot_dens == 1:
                plt.figure(r'ne Awall')
                plt.savefig(path_out+"ne_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'nn Awall')
                plt.savefig(path_out+"nn_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_dphi_Te == 1:
                plt.figure(r'dphi_sh Awall')
                plt.savefig(path_out+"dphi_sh_b_Awall"+figs_format,bbox_inches='tight')
                plt.close()   
                plt.figure(r'dphi_sh_Te Awall')
                plt.savefig(path_out+"dphi_sh_b_Te_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'Te Awall')
                plt.savefig(path_out+"Te_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'phi Awall')
                plt.savefig(path_out+"phi_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
            if plot_curr == 1:
                plt.figure(r'je_b Awall')
                plt.savefig(path_out+"je_b_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'ji_tot_b Awall')
                plt.savefig(path_out+"ji_tot_b_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'ji1 Awall')
                plt.savefig(path_out+"ji1_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'ji2 Awall')
                plt.savefig(path_out+"ji2_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'ji2/ji1 Awall')
                plt.savefig(path_out+"ratio_ji2ji1_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'ji1/ji_tot_b Awall')
                plt.savefig(path_out+"ratio_ji1ji_tot_b_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'ji2/ji_tot_b Awall')
                plt.savefig(path_out+"ratio_ji2ji_tot_b_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'j_b Awall')
                plt.savefig(path_out+"j_b_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'ji_tot_b/j_b Awall')
                plt.savefig(path_out+"ratio_ji_tot_bj_b_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'je_b/j_b Awall')
                plt.savefig(path_out+"ratio_je_bj_b_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'je_b_gp_net_b Awall')
                plt.savefig(path_out+"je_b_gp_net_b_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'relerr_je_b Awall')
                plt.savefig(path_out+"relerr_je_b_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_q == 1:
                plt.figure(r'qe_tot_wall Awall')
                plt.savefig(path_out+"qe_tot_wall_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'qe_tot_s_wall Awall')
                plt.savefig(path_out+"qe_tot_s_wall_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'qe_tot_b Awall')
                plt.savefig(path_out+"qe_tot_b_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'qe_b Awall')
                plt.savefig(path_out+"qe_b_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'qe_adv_b Awall')
                plt.savefig(path_out+"qe_adv_b_Awall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'relerr_qe_b Awall')
                plt.savefig(path_out+"relerr_qe_b_Awall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'relerr_qe_b_cons Awall')
                plt.savefig(path_out+"relerr_qe_b_cons_Awall"+figs_format,bbox_inches='tight')
                plt.close()
            if plot_imp_ene == 1:
                plt.figure(r'imp_ene_i1 Awall')
                plt.savefig(path_out+"imp_ene_i1_Awall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_i2 Awall')
                plt.savefig(path_out+"imp_ene_i2_Awall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_ion Awall')
                plt.savefig(path_out+"imp_ene_ion_Awall"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'imp_ene_n1 Awall')
                plt.savefig(path_out+"imp_ene_n1_Awall"+figs_format,bbox_inches='tight')
                plt.close()
            if plot_err_interp_mfam == 1:
                plt.figure(r'err_interp_phi Awall')
                plt.savefig(path_out+"err_interp_phi_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_Te Awall')
                plt.savefig(path_out+"err_interp_Te_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jeperp Awall')
                plt.savefig(path_out+"err_interp_jeperp_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jetheta Awall')
                plt.savefig(path_out+"err_interp_jetheta_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jepara Awall')
                plt.savefig(path_out+"err_interp_jepara_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jez Awall')
                plt.savefig(path_out+"err_interp_jez_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jer Awall')
                plt.savefig(path_out+"err_interp_jer_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_err_interp_pic == 1:
                plt.figure(r'err_interp_n Awall')
                plt.savefig(path_out+"err_interp_n_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_picM_picS_comp == 1:
                plt.figure(r'comp picMpicS imp_ene_i1 Awall')
                plt.savefig(path_out+"comp_picMpicS_imp_ene_i1_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'comp picMpicS imp_ene_i2 Awall')
                plt.savefig(path_out+"comp_picMpicS_imp_ene_i2_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'comp picMpicS imp_ene_ion Awall')
                plt.savefig(path_out+"comp_picMpicS_imp_ene_ion_Awall"+figs_format,bbox_inches='tight')
                plt.close()  
                plt.figure(r'comp picMpicS imp_ene_n1 Awall')
                plt.savefig(path_out+"comp_picMpicS_imp_ene_n1_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'comp picMpicS ji_tot_b Awall')
                plt.savefig(path_out+"comp_picMpicS_ji_tot_b_Awall"+figs_format,bbox_inches='tight')
                plt.close() 
                 
        if plot_FLwall == 1:
            if plot_dens == 1:
                plt.figure(r'ne FLwall_ver')
                plt.savefig(path_out+"ne_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'ne FLwall_lat')
                plt.savefig(path_out+"ne_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'ne FLwall_ver_lat')
                plt.savefig(path_out+"ne_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                
                plt.figure(r'nn FLwall_ver')
                plt.savefig(path_out+"nn_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'nn FLwall_lat')
                plt.savefig(path_out+"nn_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'nn FLwall_ver_lat')
                plt.savefig(path_out+"nn_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_dphi_Te == 1:            
                plt.figure(r'Te FLwall_ver')
                plt.savefig(path_out+"Te_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'Te FLwall_lat')
                plt.savefig(path_out+"Te_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'Te FLwall_ver_lat')
                plt.savefig(path_out+"Te_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                
                plt.figure(r'phi FLwall_ver')
                plt.savefig(path_out+"phi_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'phi FLwall_lat')
                plt.savefig(path_out+"phi_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'phi FLwall_ver_lat')
                plt.savefig(path_out+"phi_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_curr == 1:
                plt.figure(r'je_b FLwall_ver')
                plt.savefig(path_out+"je_b_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'je_b FLwall_lat')
                plt.savefig(path_out+"je_b_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'je_b FLwall_ver_lat')
                plt.savefig(path_out+"je_b_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
    
                plt.figure(r'ji_tot_b FLwall_ver')
                plt.savefig(path_out+"ji_tot_b_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'ji_tot_b FLwall_lat')
                plt.savefig(path_out+"ji_tot_b_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'ji_tot_b FLwall_ver_lat')
                plt.savefig(path_out+"ji_tot_b_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                
                plt.figure(r'j_b FLwall_ver')
                plt.savefig(path_out+"j_b_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'j_b FLwall_lat')
                plt.savefig(path_out+"j_b_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'j_b FLwall_ver_lat')
                plt.savefig(path_out+"j_b_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
    
                plt.figure(r'relerr_je_b FLwall_ver')
                plt.savefig(path_out+"relerr_je_b_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'relerr_je_b FLwall_lat')
                plt.savefig(path_out+"relerr_je_b_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'relerr_je_b FLwall_ver_lat')
                plt.savefig(path_out+"relerr_je_b_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_q == 1:
                plt.figure(r'qe_tot_wall FLwall_ver')
                plt.savefig(path_out+"qe_tot_wall_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'qe_tot_wall FLwall_lat')
                plt.savefig(path_out+"qe_tot_wall_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'qe_tot_wall FLwall_ver_lat')
                plt.savefig(path_out+"qe_tot_wall_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
    
                plt.figure(r'qe_tot_b FLwall_ver')
                plt.savefig(path_out+"qe_tot_b_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'qe_tot_b FLwall_lat')
                plt.savefig(path_out+"qe_tot_b_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'qe_tot_b FLwall_ver_lat')
                plt.savefig(path_out+"qe_tot_b_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
    
                plt.figure(r'qe_b FLwall_ver')
                plt.savefig(path_out+"qe_b_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'qe_b FLwall_lat')
                plt.savefig(path_out+"qe_b_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'qe_b FLwall_ver_lat')
                plt.savefig(path_out+"qe_b_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                
                plt.figure(r'qe_adv_b FLwall_ver')
                plt.savefig(path_out+"qe_adv_b_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'qe_adv_b FLwall_lat')
                plt.savefig(path_out+"qe_adv_b_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'qe_adv_b FLwall_ver_lat')
                plt.savefig(path_out+"qe_adv_b_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
    
                plt.figure(r'relerr_qe_b FLwall_ver')
                plt.savefig(path_out+"relerr_qe_b_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'relerr_qe_b FLwall_lat')
                plt.savefig(path_out+"relerr_qe_b_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'relerr_qe_b FLwall_ver_lat')
                plt.savefig(path_out+"relerr_qe_b_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                
                plt.figure(r'relerr_qe_b_cons FLwall_ver')
                plt.savefig(path_out+"relerr_qe_b_cons_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'relerr_qe_b_cons FLwall_lat')
                plt.savefig(path_out+"relerr_qe_b_cons_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'relerr_qe_b_cons FLwall_ver_lat')
                plt.savefig(path_out+"relerr_qe_b_cons_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_err_interp_mfam == 1:
                plt.figure(r'err_interp_phi FLwall_ver')
                plt.savefig(path_out+"err_interp_phi_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_phi FLwall_lat')
                plt.savefig(path_out+"err_interp_phi_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_phi FLwall_ver_lat')
                plt.savefig(path_out+"err_interp_phi_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
    
                plt.figure(r'err_interp_Te FLwall_ver')
                plt.savefig(path_out+"err_interp_Te_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_Te FLwall_lat')
                plt.savefig(path_out+"err_interp_Te_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_Te FLwall_ver_lat')
                plt.savefig(path_out+"err_interp_Te_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
    
                plt.figure(r'err_interp_jeperp FLwall_ver')
                plt.savefig(path_out+"err_interp_jeperp_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'err_interp_jeperp FLwall_lat')
                plt.savefig(path_out+"err_interp_jeperp_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close()
                plt.figure(r'err_interp_jeperp FLwall_ver_lat')
                plt.savefig(path_out+"err_interp_jeperp_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
    
                plt.figure(r'err_interp_jetheta FLwall_ver')
                plt.savefig(path_out+"err_interp_jetheta_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jetheta FLwall_lat')
                plt.savefig(path_out+"err_interp_jetheta_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jetheta FLwall_ver_lat')
                plt.savefig(path_out+"err_interp_jetheta_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
    
                plt.figure(r'err_interp_jepara FLwall_ver')
                plt.savefig(path_out+"err_interp_jepara_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jepara FLwall_lat')
                plt.savefig(path_out+"err_interp_jepara_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jepara FLwall_ver_lat')
                plt.savefig(path_out+"err_interp_jepara_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                
                plt.figure(r'err_interp_jez FLwall_ver')
                plt.savefig(path_out+"err_interp_jez_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jez FLwall_lat')
                plt.savefig(path_out+"err_interp_jez_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jez FLwall_ver_lat')
                plt.savefig(path_out+"err_interp_jez_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
    
                plt.figure(r'err_interp_jer FLwall_ver')
                plt.savefig(path_out+"err_interp_jer_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jer FLwall_lat')
                plt.savefig(path_out+"err_interp_jer_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jer FLwall_ver_lat')
                plt.savefig(path_out+"err_interp_jer_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_err_interp_pic == 1:
                plt.figure(r'err_interp_n FLwall_ver')
                plt.savefig(path_out+"err_interp_n_FLwall_ver"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_n FLwall_lat')
                plt.savefig(path_out+"err_interp_n_FLwall_lat"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_n FLwall_ver_lat')
                plt.savefig(path_out+"err_interp_n_FLwall_ver_lat"+figs_format,bbox_inches='tight')
                plt.close()
                 
        if plot_Axis == 1:
            if plot_dens == 1:
                plt.figure(r'ne Axis')
                plt.savefig(path_out+"ne_Axis"+figs_format,bbox_inches='tight')
                plt.figure(r'nn Axis')
                plt.savefig(path_out+"nn_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_dphi_Te == 1:
                plt.figure(r'Te Axis')
                plt.savefig(path_out+"Te_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'phi Axis')
                plt.savefig(path_out+"phi_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_curr == 1:
                plt.figure(r'je_b Axis')
                plt.savefig(path_out+"je_b_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'ji_tot_b Axis')
                plt.savefig(path_out+"ji_tot_b_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'j_b Axis')
                plt.savefig(path_out+"j_b_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'relerr_je_b Axis')
                plt.savefig(path_out+"relerr_je_b_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_q == 1:
                plt.figure(r'relerr_qe_b Axis')
                plt.savefig(path_out+"relerr_qe_b_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'relerr_qe_b_cons Axis')
                plt.savefig(path_out+"relerr_qe_b_cons_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_err_interp_mfam == 1:
                plt.figure(r'err_interp_phi Axis')
                plt.savefig(path_out+"err_interp_phi_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_Te Axis')
                plt.savefig(path_out+"err_interp_Te_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jeperp Axis')
                plt.savefig(path_out+"err_interp_jeperp_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jetheta Axis')
                plt.savefig(path_out+"err_interp_jetheta_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jepara Axis')
                plt.savefig(path_out+"err_interp_jepara_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jez Axis')
                plt.savefig(path_out+"err_interp_jez_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                plt.figure(r'err_interp_jer Axis')
                plt.savefig(path_out+"err_interp_jer_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
            if plot_err_interp_pic == 1:
                plt.figure(r'err_interp_n Axis')
                plt.savefig(path_out+"err_interp_n_Axis"+figs_format,bbox_inches='tight')
                plt.close() 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
