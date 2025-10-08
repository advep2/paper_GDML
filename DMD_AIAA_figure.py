#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 13:08:05 2021

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
ref_case_plots      = 1

path_out = "HET_DMD_figs/"


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
line_width_Blines      = 0.75
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
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'m-',linewidth = line_width)
        elif faces[2,i] == 1:   # face type >> lambda = const. (blue)
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'c-',linewidth = line_width)
        else:                   # any other face type (black)  
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'k-',linewidth = line_width)
            
def plot_MFAM_ax_nosig(ax,faces,nodes,line_width):
    nfaces = np.shape(faces)[1]
    for i in range(0,nfaces):
        if faces[2,i] == 1:     # face type >> lambda = const. (cyan)
            ax.plot(nodes[0,faces[0:2,i]-1],nodes[1,faces[0:2,i]-1],'k-',linewidth = line_width)
            
    return
###############################################################################
    

if ref_case_plots == 1:
    print("######## ref_case_plots ########")

#    ticks_size_isolines = 20
    ticks_size_isolines = 10
    marker_every = 3
    
    nlevels_2Dcontour = 150

    
#    rind       = 19
    rind       = 17
    rind       = 21
#   rind       = 32
    rind_anode1 = rind
    rind_anode2 = 17
    zind_anode  = 8
#    elems_cath_Bline   = range(407-1,483-1+2,2) # Elements along the cathode B line for cases C1, C2 and C3
#    elems_cath_Bline   = range(875-1,951-1+2,2) # Elements along the cathode B line for case C5
#    elems_cath_Bline   = range(2057-1,2064-1+1,1) # Elements along the cathode B line for topo2 cat 2059
#    elems_cath_Bline   = range(3212-1,3300-1+1,1) # Elements along the cathode B line for topo2 cat 3298
#    elems_cath_Bline   = range(639-1,701-1+2,2) # Elements along the cathode B line for topo1 cat 699
#    elems_cath_Bline = range(1115-1,1202-1+1,1) # Elements along the cathode B line for topo2 cat 1200
#    elems_cath_Bline   = range(257-1,316-1+2,2) # Elements along the cathode B line for topo1 cat 313
    elems_cath_Bline = []
    
    # Cathode plotting flag and cathode position in cm (for plot_zcath_012 = 2) 
    plot_zcath_012     = 1
    zcat_pos           = 5.9394542444501024 # In prof figures if plot_zcath_012 = 2
    plot_cath_contours = 1
    
    # Print out time step
    timestep = 0
    timesteps = [996,1010,1050,1095]
    timesteps = []
    
    allsteps_flag   = 1
    read_inst_data  = 0
    read_part_lists = 0
    read_flag       = 1
    
    mean_vars       = 1
    mean_type       = 0
#    last_steps      = 600
#    last_steps      = 700
    last_steps      = 2000
    step_i          = 1
    step_f          = 78
    plot_mean_vars  = 1
    
    
    plot_fields_ref     = 1
    plot_dens_ref       = 1
    plot_temp_ref       = 1
    plot_vel_ref        = 0
    plot_curr_ref       = 1
    plot_nmp_ref        = 0
    plot_freq_ref       = 0
    plot_anode_ref      = 0    
    plot_err_interp_ref = 1
    plot_lambdaD_ref    = 0
    
    Bline_all2Dplots = 1
    cath2D_plots     = 0
    cath2D_title     = r"(b)"
    

    if allsteps_flag == 0:
        mean_vars = 0

    
    # Simulation names
    nsims = 1    
    
    # Flag for old sims (1: old sim files, 0: new sim files)
#    oldpost_sim      = np.array([4,0,0,0,0,0],dtype = int)
#    oldsimparams_sim = np.array([13,0,0,0,0,0],dtype = int)   
    
    # These below are the flags for reading the case "../../../Sr_sims_files/SPT100_DMD_pm2em2_cat3328_tmtetq2_Vd300" (used for figure in paper):
    oldpost_sim      = np.array([3,0,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([11,0,0,0,0,0],dtype = int) 
    
#    oldpost_sim      = np.array([1,0,0,0,0,0],dtype = int)
#    oldsimparams_sim = np.array([0,0,0,0,0,0],dtype = int)   
    
    
#    sim_names = ["../../../Rb_hyphen/sim/sims/SPT100_al0025_Ne5_C1"]
#    sim_names = ["../../Rb_hyphen/sim/sims/SPT100_al0025_Ne100_C1"]
#    sim_names = ["../../Rb_hyphen/sim/sims/SPT100_also1510_V1"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n4_l200s200"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n4_l200s200_cat3298"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n4_l200s200_cat3298_WLSQb"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo1_n1_l100s100_cat699"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n4_l200s200_cat3283_relaunch"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n4_l200s200_cat1200_relaunch"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n4_l200s200_cat1200_tm110_tetq125_RLC"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n4_l200s200_cat1200_tm110_tetq125_RLC_Coll"]
#    sim_names = ["../../../sim/sims/Topo2_n4_l200s200_cat1200_tm110_tetq125_ECath2"]
#    sim_names = ["../../../sim/sims/Topo2_n4_l200s200_cat1200_tm110_tetq125_ECath_explicit"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n4_l200s200_tm110_tetq125_ExCat12001107"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo1_n1_l100s100_cat313_tm110_tetq125"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n3_l200s200_cat1200_tm110_tetq125"]
    
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n4_l200s200_cat1200_tm110_tetq125_RLC_Coll_CHECK2"]
    
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n4_l200s200_tm110_tetq125_ExCat12002942"]
#    sim_names = ["../../../Rb_hyphen/sim/sims/Topo2_n4_l200s200_tm110_tetq125_ExCat12001199"]

#    sim_names = ["../../../sim/sims/Topo1_n1_l100s100_cat313_tm515_te1_tq21"] # T1N1-REF
#    sim_names = ["../../../sim/sims/Topo1_n2_l100s100_cat313_tm615_te2_tq12"] # T1N2-REF
#    sim_names = ["../../../sim/sims/Topo2_n3_l200s200_cat1200_tm15_te1_tq125"] # T2N3-REF
#    sim_names = ["../../../sim/sims/Topo2_n4_l200s200_cat1200_tm15_te1_tq125"] # T2N4-REF
    
#    sim_names = ["../../../Ca_sims_files/SPT100_thesis_REF_MFAMjesus_rm"] 
    sim_names = [
#                 "../../../Ca_sims_files/SPT100_thesis_REF_MFAMjesus_rm3_picrm_oldsheath"
#                 "../../../Ca_sims_files/SPT100_thesis_REF_MFAMjesus_rm2_picrm_aljpara"
#                 "../../../Ca_sims_files/SPT100_thesis_REF_MFAMjesus_rm2_picrm_CHECKinterpalphate"
#                 "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd300"  
#                 "../../../Sr_sims_files/Topo2_n4_l200s200_cat1200_tm15_te1_tq125_last"
#                  "../../../Sr_sims_files/Topo1_n1_l100s100_cat313_tm515_te1_tq21_last"
#                 "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_71d0dcb",
#                 "../../../Sr_sims_files/SPT100_pm2em1_cat481_tmtetq25_RLC"
#                 "../../../sim/sims/SPT100_pm2em2_cat3328_PHIchanges"
#                 "../../../sim/sims/SPT100_pm2em2_cat3328_PHIchanges_newSET"
                 
                 "../../../Sr_sims_files/SPT100_DMD_pm2em2_cat3328_tmtetq2_Vd300"
                 
                 ] 

    
    PIC_mesh_file_name = ["PIC_mesh_topo2_refined4.hdf5"]
#    PIC_mesh_file_name = ["PIC_mesh_topo1_refined4.hdf5"]
#    PIC_mesh_file_name = ["SPT100_picM.hdf5"]
#    PIC_mesh_file_name = ["SPT100_picM_Reference1500points.hdf5"]
#    PIC_mesh_file_name = ["SPT100_picM_Reference1500points_rm.hdf5"]
    PIC_mesh_file_name = ["SPT100_picM_Reference1500points_rm2.hdf5"]



    # Labels             
    labels = [r"A",r"B",r"C",r"D",r"Average"]
    
    # Line colors
    colors = ['r','g','b','m','k','m','y']
    # Markers
    markers = ['^','>','v', '<', 's', 'o','*']
    # Line style
#    linestyles = ['-','--','-.', ':','-','--','-.']
    linestyles = ['-','-','-', '-','-','-','-']
    
              

        
   
    ######################## READ INPUT/OUTPUT FILES ##########################
    k = 0
    ind_ini_letter = sim_names[k].rfind('/') + 1
    print("##### CASE "+str(k+1)+": "+sim_names[k][ind_ini_letter::]+" #####")
    # Obtain paths to simulation files
    path_picM         = sim_names[k]+"/SET/inp/"+PIC_mesh_file_name[k]
    path_simstate_inp = sim_names[k]+"/CORE/inp/SimState.hdf5"
    # --------------
    path_simstate_out = sim_names[k]+"/CORE/out/SimState.hdf5"
    path_postdata_out = sim_names[k]+"/CORE/out/PostData.hdf5"
    path_simparams_inp = sim_names[k]+"/CORE/inp/sim_params.inp"
     # --------------
#    path_simstate_out = sim_names[k]+"/CORE/out/60000steps_after_changes_it_matching_modified_iterations/SimState.hdf5"
#    path_postdata_out = sim_names[k]+"/CORE/out/60000steps_after_changes_it_matching_modified_iterations/PostData.hdf5"
#    path_simparams_inp = sim_names[k]+"/CORE/inp/sim_params.inp"
    # --------------
#    path_simstate_out = sim_names[k]+"/CORE/out/60000steps_cond_wall_connect1_hefunc1_jefl1_280V/SimState.hdf5"
#    path_postdata_out = sim_names[k]+"/CORE/out/60000steps_cond_wall_connect1_hefunc1_jefl1_280V/PostData.hdf5"
#    path_simparams_inp = sim_names[k]+"/CORE/inp/sim_params.inp"
    # --------------
#    path_simstate_out = sim_names[k]+"/CORE/out/60000steps_cond_wall_connect1_hefunc1_jefl1_70V_plume/SimState.hdf5"
#    path_postdata_out = sim_names[k]+"/CORE/out/60000steps_cond_wall_connect1_hefunc1_jefl1_70V_plume/PostData.hdf5"
#    path_simparams_inp = sim_names[k]+"/CORE/inp/sim_params.inp"
    # --------------
    
    print("Reading results...")
#    [num_ion_spe,num_neu_spe,n_mp_cell_i,n_mp_cell_n,n_mp_cell_i_min,
#       n_mp_cell_i_max,n_mp_cell_n_min,n_mp_cell_n_max,min_ion_plasma_density,
#       m_A,spec_refl_prob,ene_bal,points,zs,rs,zscells,rscells,dims,
#       nodes_flag,cells_flag,cells_vol,volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,
#       eta_max,eta_min,xi_top,xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,
#       nsteps,nsteps_fast,nsteps_eFld,faces,nodes,elem_n,boundary_f,face_geom,elem_geom,
#       n_faces,n_elems,n_faces_boundary,bIDfaces_Dwall,bIDfaces_Awall,
#       bIDfaces_FLwall,IDfaces_Dwall,IDfaces_Awall,IDfaces_FLwall,zfaces_Dwall,
#       rfaces_Dwall,Afaces_Dwall,zfaces_Awall,rfaces_Awall,Afaces_Awall,
#       zfaces_FLwall,rfaces_FLwall,Afaces_FLwall,
#       cath_elem,z_cath,r_cath,V_cath,mass,ssIons1,ssIons2,ssNeutrals1,ssNeutrals2,
#       n_mp_i1_list,n_mp_i2_list,n_mp_n1_list,n_mp_n2_list,phi,phi_elems,Ez,Er,Efield,
#       Bz,Br,Bfield,Te,Te_elems,je_mag_elems,je_perp_elems,je_theta_elems,je_para_elems,
#       cs01,cs02,nn1,nn2,ni1,ni2,ne,ne_elems,fn1_x,fn1_y,
#       fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,
#       un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,
#       ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,
#       ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,
#       avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
#       ion_gen_weights1,ion_gen_weights2,surf_elems,n_imp_elems,imp_elems,
#       imp_elems_kbc,imp_elems_MkQ1,imp_elems_Te,imp_elems_dphi_kbc,
#       imp_elems_dphi_sh,imp_elems_nQ1,imp_elems_nQ2,imp_elems_ion_flux_in1,
#       imp_elems_ion_flux_out1,imp_elems_ion_ene_flux_in1,
#       imp_elems_ion_ene_flux_out1,imp_elems_ion_imp_ene1,
#       imp_elems_ion_flux_in2,imp_elems_ion_flux_out2,
#       imp_elems_ion_ene_flux_in2,imp_elems_ion_ene_flux_out2,
#       imp_elems_ion_imp_ene2,imp_elems_neu_flux_in1,imp_elems_neu_flux_out1,
#       imp_elems_neu_ene_flux_in1,imp_elems_neu_ene_flux_out1,
#       imp_elems_neu_imp_ene1,imp_elems_neu_flux_in2,imp_elems_neu_flux_out2,
#       imp_elems_neu_ene_flux_in2,imp_elems_neu_ene_flux_out2,
#       imp_elems_neu_imp_ene2,tot_mass_mp_neus,tot_mass_mp_ions,tot_num_mp_neus,
#       tot_num_mp_ions,tot_mass_exit_neus,tot_mass_exit_ions,mass_mp_neus,
#       mass_mp_ions,num_mp_neus,num_mp_ions,avg_dens_mp_neus,avg_dens_mp_ions,
#       eta_u,eta_prod,eta_thr,eta_div,eta_cur,thrust,thrust_ion,thrust_neu,
#       Id_inst,Id,Vd_inst,Vd,I_beam,I_tw_tot,Pd,Pd_inst,P_mat,P_inj,P_inf,P_ion,
#       P_ex,P_use_tot_i,P_use_tot_n,P_use_tot,P_use_z_i,P_use_z_n,P_use_z,
#       qe_wall,qe_wall_inst,Pe_faces_Dwall,Pe_faces_Awall,Pe_faces_FLwall,
#       Pe_faces_Dwall_inst,Pe_faces_Awall_inst,Pe_faces_FLwall_inst,
#       Pe_Dwall,Pe_Awall,Pe_FLwall,Pe_Dwall_inst,Pe_Awall_inst,Pe_FLwall_inst, 
#       Pi_Dwall,Pi_Awall,Pi_FLwall,Pi_FLwall_nonz,Pn_Dwall,Pn_Awall,Pn_FLwall,
#       Pn_FLwall_nonz,P_Dwall,P_Awall,P_FLwall,Pwalls,Pionex,Ploss,Pthrust,
#       Pnothrust,Pnothrust_walls,balP,err_balP,ctr_Pd,ctr_Ploss,ctr_Pwalls,
#       ctr_Pionex,ctr_P_DAwalls,ctr_P_FLwalls,ctr_P_FLwalls_in,ctr_P_FLwalls_i,
#       ctr_P_FLwalls_n,ctr_P_FLwalls_e,balP_Pthrust,err_balP_Pthrust,
#       ctr_balPthrust_Pd,ctr_balPthrust_Pnothrust,ctr_balPthrust_Pthrust,
#       ctr_balPthrust_Pnothrust_walls,ctr_balPthrust_Pnothrust_ionex,
#       err_def_balP,Isp_s,Isp_ms,dMdt_i1,dMdt_i2,dMdt_n1,dMdt_n2,dMdt_tot,
#       mflow_coll_i1,mflow_coll_i2,mflow_coll_n1,mflow_coll_n2,mflow_fw_i1,
#       mflow_fw_i2,mflow_fw_n1,mflow_fw_n2,mflow_tw_i1,mflow_tw_i2,mflow_tw_n1,
#       mflow_tw_n2,mflow_ircmb_picS_n1,mflow_ircmb_picS_n2,mflow_inj_i1,mflow_inj_i2,
#       mflow_fwmat_i1,mflow_fwmat_i2,mflow_inj_n1,mflow_fwmat_n1,mflow_inj_n2,
#       mflow_fwmat_n2,mflow_twmat_i1,mflow_twinf_i1,mflow_twa_i1,mflow_twmat_i2,
#       mflow_twinf_i2,mflow_twa_i2,mflow_twmat_n1,mflow_twinf_n1,mflow_twa_n1,
#       mflow_twmat_n2,mflow_twinf_n2,mflow_twa_n2,mbal_n1,mbal_i1,mbal_i2,mbal_tot,
#       err_mbal_n1,err_mbal_i1,err_mbal_i2,err_mbal_tot,ctr_mflow_coll_n1,
#       ctr_mflow_fw_n1,ctr_mflow_tw_n1,ctr_mflow_coll_i1,ctr_mflow_fw_i1,
#       ctr_mflow_tw_i1,ctr_mflow_coll_i2,ctr_mflow_fw_i2,ctr_mflow_tw_i2,
#       ctr_mflow_coll_tot,ctr_mflow_fw_tot,ctr_mflow_tw_tot,dEdt_i1,dEdt_i2,
#       dEdt_n1,dEdt_n2,eneflow_coll_i1,eneflow_coll_i2,eneflow_coll_n1,
#       eneflow_coll_n2,eneflow_fw_i1,eneflow_fw_i2,eneflow_fw_n1,eneflow_fw_n2,
#       eneflow_tw_i1,eneflow_tw_i2,eneflow_tw_n1,eneflow_tw_n2,Pfield_i1,
#       Pfield_i2,eneflow_inj_i1,eneflow_fwmat_i1,eneflow_inj_i2,
#       eneflow_fwmat_i2,eneflow_inj_n1,eneflow_fwmat_n1,eneflow_inj_n2,
#       eneflow_fwmat_n2,eneflow_twmat_i1,eneflow_twinf_i1,eneflow_twa_i1,
#       eneflow_twmat_i2,eneflow_twinf_i2,eneflow_twa_i2,eneflow_twmat_n1,
#       eneflow_twinf_n1,eneflow_twa_n1,eneflow_twmat_n2,eneflow_twinf_n2,
#       eneflow_twa_n2,ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,cath_type,ne_cath,Te_cath,
#       nu_cath,ndot_cath,Q_cath,P_cath,V_cath_tot,ne_cath_avg,F_theta,Hall_par,
#       Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,
#       nu_ex,Boltz,Boltz_dim,Pfield_e,Ebal_e,ge_b,ge_b_acc,ge_sb_b,ge_sb_b_acc,
#       delta_see,delta_see_acc,err_interp_n,n_cond_wall,Icond,Vcond,Icath] = HET_sims_read(path_simstate_inp,path_simstate_out,
#                                                                                           path_postdata_out,path_simparams_inp,
#                                                                                           path_picM,allsteps_flag,timestep,read_inst_data,
#                                                                                           read_part_lists,read_flag,oldpost_sim[k],oldsimparams_sim[k])
        
    
    [num_ion_spe,num_neu_spe,n_mp_cell_i,n_mp_cell_n,n_mp_cell_i_min,
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
           je_theta_elems,je_para_elems,cs01,cs02,nn1,nn2,ni1,ni2,ne,ne_elems,fn1_x,fn1_y,
           fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,
           un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,
           ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,
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
           eta_u,eta_prod,eta_thr,eta_div,eta_cur,thrust,thrust_ion,thrust_neu,
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
           err_def_balP,Isp_s,Isp_ms,dMdt_i1,dMdt_i2,dMdt_n1,dMdt_n2,dMdt_tot,
           mflow_coll_i1,mflow_coll_i2,mflow_coll_n1,mflow_coll_n2,mflow_fw_i1,
           mflow_fw_i2,mflow_fw_n1,mflow_fw_n2,mflow_tw_i1,mflow_tw_i2,mflow_tw_n1,
           mflow_tw_n2,mflow_ircmb_picS_n1,mflow_ircmb_picS_n2,
           mflow_inj_i1,mflow_fwinf_i1,mflow_fwmat_i1,mflow_fwcat_i1,mflow_inj_i2,
           mflow_fwinf_i2,mflow_fwmat_i2,mflow_fwcat_i2,mflow_inj_n1,mflow_fwinf_n1,
           mflow_fwmat_n1,mflow_fwcat_n1,mflow_inj_n2,mflow_fwinf_n2,mflow_fwmat_n2,
           mflow_fwcat_n2,
           mflow_twa_i1,mflow_twinf_i1,mflow_twmat_i1,mflow_twcat_i1,mflow_twa_i2,
           mflow_twinf_i2,mflow_twmat_i2,mflow_twcat_i2,mflow_twa_n1,mflow_twinf_n1,
           mflow_twmat_n1,mflow_twcat_n1,mflow_twa_n2,mflow_twinf_n2,mflow_twmat_n2,
           mflow_twcat_n2,
           mbal_n1,mbal_i1,mbal_i2,mbal_tot,
           err_mbal_n1,err_mbal_i1,err_mbal_i2,err_mbal_tot,ctr_mflow_coll_n1,
           ctr_mflow_fw_n1,ctr_mflow_tw_n1,ctr_mflow_coll_i1,ctr_mflow_fw_i1,
           ctr_mflow_tw_i1,ctr_mflow_coll_i2,ctr_mflow_fw_i2,ctr_mflow_tw_i2,
           ctr_mflow_coll_tot,ctr_mflow_fw_tot,ctr_mflow_tw_tot,dEdt_i1,dEdt_i2,
           dEdt_n1,dEdt_n2,eneflow_coll_i1,eneflow_coll_i2,eneflow_coll_n1,
           eneflow_coll_n2,eneflow_fw_i1,eneflow_fw_i2,eneflow_fw_n1,eneflow_fw_n2,
           eneflow_tw_i1,eneflow_tw_i2,eneflow_tw_n1,eneflow_tw_n2,Pfield_i1,Pfield_i2,
           eneflow_inj_i1,eneflow_fwinf_i1,eneflow_fwmat_i1,eneflow_inj_i2,
           eneflow_fwinf_i2,eneflow_fwmat_i2,eneflow_inj_n1,eneflow_fwinf_n1,
           eneflow_fwmat_n1,eneflow_inj_n2,eneflow_fwinf_n2,eneflow_fwmat_n2,
           eneflow_twa_i1,eneflow_twinf_i1,eneflow_twmat_i1,eneflow_twa_i2,
           eneflow_twinf_i2,eneflow_twmat_i2,eneflow_twa_n1,eneflow_twinf_n1,
           eneflow_twmat_n1,eneflow_twa_n2,eneflow_twinf_n2,eneflow_twmat_n2,
           ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,cath_type,ne_cath,Te_cath,
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
    if oldpost_sim[k] == 3:
        nu_i02 = 2.0*nu_i02
    #######################################################################
        
    print("Generating plotting variables (NaN in ghost nodes)...")                                                                                                      
    [Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,fn1_x,fn1_y,fn1_z,
       fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,
       un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,ji1_z,
       ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,ue_perp,
       ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,
       avg_w_n2,avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
       ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,
       F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,
       nu_i01,nu_i02,nu_i12,err_interp_n,f_split_adv,f_split_qperp,f_split_qpara,
       f_split_qb,f_split_Pperp,f_split_Ppara,f_split_ecterm,f_split_inel] = HET_sims_plotvars(nodes_flag,cells_flag,Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,
                                                                                               fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,
                                                                                               un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,
                                                                                               ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,
                                                                                               ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
                                                                                               n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                                                                               neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                                                                               ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,F_theta,Hall_par,Hall_par_eff,
                                                                                               nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,err_interp_n,
                                                                                               f_split_adv,f_split_qperp,f_split_qpara,f_split_qb,f_split_Pperp,
                                                                                               f_split_Ppara,f_split_ecterm,f_split_inel)
    
    # We look here for the point with max ne and Ez along the thruster center line 1Dz profile
    ne_1Dz = ne[rind,:,:]
    Ez_1Dz = Ez[rind,:,:]
    max_ne_1Dz  = np.zeros(nsteps,dtype=float)
    max_Ez_1Dz  = np.zeros(nsteps,dtype=float)
    zmax_ne_1Dz = np.zeros(nsteps,dtype=float)
    zmax_Ez_1Dz = np.zeros(nsteps,dtype=float)
    for ind_k in range(0,nsteps):
        for ind_j in range(0,dims[1]):
            pos_ne = np.where(ne_1Dz[:,ind_k] == np.nanmax(ne_1Dz[:,ind_k]))[0][0]
            pos_Ez = np.where(Ez_1Dz[:,ind_k] == np.nanmax(Ez_1Dz[:,ind_k]))[0][0]
            max_ne_1Dz[ind_k] = ne_1Dz[pos_ne,ind_k]
            max_Ez_1Dz[ind_k] = Ez_1Dz[pos_Ez,ind_k]
            zmax_ne_1Dz[ind_k] = zs[rind,pos_ne]
            zmax_Ez_1Dz[ind_k] = zs[rind,pos_Ez]
    
    zmax_Ez_1Dz = zmax_Ez_1Dz*1E2
    zmax_ne_1Dz = zmax_ne_1Dz*1E2
        
    if mean_vars == 1:        
        print("Averaging variables...")                                                                              
        [phi_mean,Er_mean,Ez_mean,Efield_mean,nn1_mean,nn2_mean,
           ni1_mean,ni2_mean,ne_mean,fn1_x_mean,fn1_y_mean,fn1_z_mean,
           fn2_x_mean,fn2_y_mean,fn2_z_mean,fi1_x_mean,fi1_y_mean,fi1_z_mean,
           fi2_x_mean,fi2_y_mean,fi2_z_mean,un1_x_mean,un1_y_mean,un1_z_mean,
           un2_x_mean,un2_y_mean,un2_z_mean,ui1_x_mean,ui1_y_mean,ui1_z_mean,
           ui2_x_mean,ui2_y_mean,ui2_z_mean,ji1_x_mean,ji1_y_mean,ji1_z_mean,
           ji2_x_mean,ji2_y_mean,ji2_z_mean,je_r_mean,je_t_mean,je_z_mean,
           je_perp_mean,je_para_mean,ue_r_mean,ue_t_mean,ue_z_mean,ue_perp_mean,
           ue_para_mean,uthetaExB_mean,Tn1_mean,Tn2_mean,Ti1_mean,Ti2_mean,Te_mean,
           n_mp_n1_mean,n_mp_n2_mean,n_mp_i1_mean,n_mp_i2_mean,avg_w_n1_mean,
           avg_w_n2_mean,avg_w_i1_mean,avg_w_i2_mean,neu_gen_weights1_mean,
           neu_gen_weights2_mean,ion_gen_weights1_mean,ion_gen_weights2_mean,
           ndot_ion01_n1_mean,ndot_ion02_n1_mean,ndot_ion12_i1_mean,
           ne_cath_mean,nu_cath_mean,ndot_cath_mean,F_theta_mean,Hall_par_mean,
           Hall_par_eff_mean,nu_e_tot_mean,nu_e_tot_eff_mean,nu_en_mean,
           nu_ei1_mean,nu_ei2_mean,nu_i01_mean,nu_i02_mean,nu_i12_mean,
           Boltz_mean,Boltz_dim_mean,phi_elems_mean,ne_elems_mean,Te_elems_mean,
           err_interp_n_mean,f_split_adv_mean,f_split_qperp_mean,f_split_qpara_mean,
           f_split_qb_mean,f_split_Pperp_mean,f_split_Ppara_mean,f_split_ecterm_mean,
           f_split_inel_mean] = HET_sims_mean(nsteps,mean_type,last_steps,step_i,step_f,phi,Er,Ez,Efield,Br,Bz,Bfield,
                                              nn1,nn2,ni1,ni2,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,
                                              fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,
                                              ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,
                                              je_perp,je_para,ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
                                              n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                              neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                              ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,nu_cath,ndot_cath,F_theta,
                                              Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,
                                              nu_i02,nu_i12,Boltz,Boltz_dim,phi_elems,ne_elems,Te_elems,err_interp_n,
                                              f_split_adv,f_split_qperp,f_split_qpara,f_split_qb,f_split_Pperp,
                                              f_split_Ppara,f_split_ecterm,f_split_inel)
                                                                                        
                                                                                        
    print("Obtaining final variables for plotting...") 
    if mean_vars == 1 and plot_mean_vars == 1:
        print("Plotting variables are time-averaged")
        [Br_plot,Bz_plot,Bfield_plot,phi_plot,Er_plot,Ez_plot,Efield_plot,
           nn1_plot,nn2_plot,ni1_plot,ni2_plot,ne_plot,
           fn1_x_plot,fn1_y_plot,fn1_z_plot,fn2_x_plot,fn2_y_plot,fn2_z_plot,
           fi1_x_plot,fi1_y_plot,fi1_z_plot,fi2_x_plot,fi2_y_plot,fi2_z_plot,
           un1_x_plot,un1_y_plot,un1_z_plot,un2_x_plot,un2_y_plot,un2_z_plot,
           ui1_x_plot,ui1_y_plot,ui1_z_plot,ui2_x_plot,ui2_y_plot,ui2_z_plot,
           ji1_x_plot,ji1_y_plot,ji1_z_plot,ji2_x_plot,ji2_y_plot,ji2_z_plot,
           je_r_plot,je_t_plot,je_z_plot,je_perp_plot,je_para_plot,
           ue_r_plot,ue_t_plot,ue_z_plot,ue_perp_plot,ue_para_plot,uthetaExB_plot,
           Tn1_plot,Tn2_plot,Ti1_plot,Ti2_plot,Te_plot,n_mp_n1_plot,n_mp_n2_plot,
           n_mp_i1_plot,n_mp_i2_plot,avg_w_n1_plot,avg_w_n2_plot,avg_w_i1_plot,
           avg_w_i2_plot,neu_gen_weights1_plot,neu_gen_weights2_plot,
           ion_gen_weights1_plot,ion_gen_weights2_plot,ndot_ion01_n1_plot,
           ndot_ion02_n1_plot,ndot_ion12_i1_plot,ne_cath_plot,nu_cath_plot,ndot_cath_plot,
           F_theta_plot,Hall_par_plot,Hall_par_eff_plot,nu_e_tot_plot,nu_e_tot_eff_plot,
           nu_en_plot,nu_ei1_plot,nu_ei2_plot,nu_i01_plot,nu_i02_plot,nu_i12_plot,
           err_interp_n_plot,f_split_adv_plot,f_split_qperp_plot,f_split_qpara_plot,
           f_split_qb_plot,f_split_Pperp_plot,f_split_Ppara_plot,f_split_ecterm_plot,
           f_split_inel_plot] = HET_sims_cp_vars(Br,Bz,Bfield,phi_mean,Er_mean,Ez_mean,Efield_mean,nn1_mean,nn2_mean,
                                                 ni1_mean,ni2_mean,ne_mean,fn1_x_mean,fn1_y_mean,fn1_z_mean,
                                                 fn2_x_mean,fn2_y_mean,fn2_z_mean,fi1_x_mean,fi1_y_mean,fi1_z_mean,
                                                 fi2_x_mean,fi2_y_mean,fi2_z_mean,un1_x_mean,un1_y_mean,un1_z_mean,
                                                 un2_x_mean,un2_y_mean,un2_z_mean,ui1_x_mean,ui1_y_mean,ui1_z_mean,
                                                 ui2_x_mean,ui2_y_mean,ui2_z_mean,ji1_x_mean,ji1_y_mean,ji1_z_mean,
                                                 ji2_x_mean,ji2_y_mean,ji2_z_mean,je_r_mean,je_t_mean,je_z_mean,
                                                 je_perp_mean,je_para_mean,ue_r_mean,ue_t_mean,ue_z_mean,ue_perp_mean,
                                                 ue_para_mean,uthetaExB_mean,Tn1_mean,Tn2_mean,Ti1_mean,Ti2_mean,Te_mean,
                                                 n_mp_n1_mean,n_mp_n2_mean,n_mp_i1_mean,n_mp_i2_mean,avg_w_n1_mean,
                                                 avg_w_n2_mean,avg_w_i1_mean,avg_w_i2_mean,neu_gen_weights1_mean,
                                                 neu_gen_weights2_mean,ion_gen_weights1_mean,ion_gen_weights2_mean,
                                                 ndot_ion01_n1_mean,ndot_ion02_n1_mean,ndot_ion12_i1_mean,ne_cath_mean,
                                                 nu_cath_mean,ndot_cath_mean,F_theta_mean,Hall_par_mean,Hall_par_eff_mean,
                                                 nu_e_tot_mean,nu_e_tot_eff_mean,nu_en_mean,nu_ei1_mean,nu_ei2_mean,nu_i01_mean,
                                                 nu_i02_mean,nu_i12_mean,err_interp_n_mean,f_split_adv_mean,f_split_qperp_mean,
                                                 f_split_qpara_mean,f_split_qb_mean,f_split_Pperp_mean,f_split_Ppara_mean,
                                                 f_split_ecterm_mean,f_split_inel_mean)
    else:
        [Br_plot,Bz_plot,Bfield_plot,phi_plot,Er_plot,Ez_plot,Efield_plot,
           nn1_plot,nn2_plot,ni1_plot,ni2_plot,ne_plot,
           fn1_x_plot,fn1_y_plot,fn1_z_plot,fn2_x_plot,fn2_y_plot,fn2_z_plot,
           fi1_x_plot,fi1_y_plot,fi1_z_plot,fi2_x_plot,fi2_y_plot,fi2_z_plot,
           un1_x_plot,un1_y_plot,un1_z_plot,un2_x_plot,un2_y_plot,un2_z_plot,
           ui1_x_plot,ui1_y_plot,ui1_z_plot,ui2_x_plot,ui2_y_plot,ui2_z_plot,
           ji1_x_plot,ji1_y_plot,ji1_z_plot,ji2_x_plot,ji2_y_plot,ji2_z_plot,
           je_r_plot,je_t_plot,je_z_plot,je_perp_plot,je_para_plot,
           ue_r_plot,ue_t_plot,ue_z_plot,ue_perp_plot,ue_para_plot,uthetaExB_plot,
           Tn1_plot,Tn2_plot,Ti1_plot,Ti2_plot,Te_plot,n_mp_n1_plot,n_mp_n2_plot,
           n_mp_i1_plot,n_mp_i2_plot,avg_w_n1_plot,avg_w_n2_plot,avg_w_i1_plot,
           avg_w_i2_plot,neu_gen_weights1_plot,neu_gen_weights2_plot,
           ion_gen_weights1_plot,ion_gen_weights2_plot,ndot_ion01_n1_plot,
           ndot_ion02_n1_plot,ndot_ion12_i1_plot,ne_cath_plot,nu_cath_plot,ndot_cath_plot,
           F_theta_plot,Hall_par_plot,Hall_par_eff_plot,nu_e_tot_plot,nu_e_tot_eff_plot,
           nu_en_plot,nu_ei1_plot,nu_ei2_plot,nu_i01_plot,nu_i02_plot,nu_i12_plot,
           err_interp_n_plot,f_split_adv_plot,f_split_qperp_plot,f_split_qpara_plot,
           f_split_qb_plot,f_split_Pperp_plot,f_split_Ppara_plot,f_split_ecterm_plot,
           f_split_inel_plot] = HET_sims_cp_vars(Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,
                                                  fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,
                                                  un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,
                                                  ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,
                                                  ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
                                                  n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                                  neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                                  ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,nu_cath,ndot_cath,F_theta,
                                                  Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,
                                                  nu_i02,nu_i12,err_interp_n,f_split_adv,f_split_qperp,f_split_qpara,f_split_qb,
                                                  f_split_Pperp,f_split_Ppara,f_split_ecterm,f_split_inel)
                                                                                                                                                               
    # Obtain auxiliar average variables
    ratio_ni1_ni2_plot      = np.divide(ni1_plot,ni2_plot)
    ue_plot                 = np.sqrt(ue_r_plot**2 +ue_t_plot**2 + ue_z_plot**2)
    ue2_plot                = np.sqrt(ue_perp_plot**2 +ue_t_plot**2 + ue_para_plot**2)
    ui1_plot                = np.sqrt(ui1_x_plot**2 + ui1_y_plot**2 + ui1_z_plot**2)
    ui2_plot                = np.sqrt(ui2_x_plot**2 + ui2_y_plot**2 + ui2_z_plot**2)
    cs01_plot               = np.sqrt(e*Te_plot/mass)
    cs02_plot               = np.sqrt(2*e*Te_plot/mass)
    Mi1_plot                = np.divide(ui1_plot,cs01_plot)
    Mi2_plot                = np.divide(ui2_plot,cs02_plot) 
    Ekin_e_plot             = 0.5*me*ue_plot**2/e
    Ekin_i1_plot            = 0.5*mass*ui1_plot**2/e
    Ekin_i2_plot            = 0.5*mass*ui2_plot**2/e
    ratio_Ekin_Te_plot      = Ekin_e_plot/Te_plot
    ratio_Ekin_Ti1_plot     = Ekin_i1_plot/Ti1_plot
    ratio_Ekin_Ti2_plot     = Ekin_i2_plot/Ti2_plot
    je_plot                 = np.sqrt(je_r_plot**2 + je_t_plot**2 + je_z_plot**2)
    je2_plot                = np.sqrt(je_perp_plot**2 + je_t_plot**2 + je_para_plot**2)
    ji_x_plot               = ji1_x_plot + ji2_x_plot
    ji_y_plot               = ji1_y_plot + ji2_y_plot
    ji_z_plot               = ji1_z_plot + ji2_z_plot
    ji_plot                 = np.sqrt( ji_x_plot**2 + ji_y_plot**2 + ji_z_plot**2 )
    ji1_plot                = np.sqrt( ji1_x_plot**2 + ji1_y_plot**2 + ji1_z_plot**2 )
    ji2_plot                = np.sqrt( ji2_x_plot**2 + ji2_y_plot**2 + ji2_z_plot**2 )
    uimean_x_plot           = ji_x_plot/(e*ne_plot)
    uimean_y_plot           = ji_y_plot/(e*ne_plot)
    uimean_z_plot           = ji_z_plot/(e*ne_plot)
    uimean_plot             = np.sqrt( uimean_x_plot**2 + uimean_y_plot**2 + uimean_z_plot**2 )
    j_r_plot                = ji_x_plot + je_r_plot
    j_t_plot                = ji_y_plot + je_t_plot
    j_z_plot                = ji_z_plot + je_z_plot
    j_plot                  = np.sqrt(j_r_plot**2 + j_t_plot**2 + j_z_plot**2)
    je2D_plot               = np.sqrt(je_r_plot**2 + je_z_plot**2)
    je2D2_plot              = np.sqrt(je_para_plot**2 + je_perp_plot**2)
    ji2D_plot               = np.sqrt(ji_x_plot**2 + ji_z_plot**2)
    j2D_plot                = np.sqrt(j_r_plot**2 + j_z_plot**2)
    erel_je_plot            = np.abs(je2_plot-je_plot)/np.abs(je_plot)
    erel_je2D_plot          = np.abs(je2D2_plot-je2D_plot)/np.abs(je2D_plot)
    erel_ue_plot            = np.abs(ue2_plot-ue_plot)/np.abs(ue_plot)
    erel_jeji_plot          = np.abs(je_plot-ji_plot)/np.abs(ji_plot)
    erel_jz_plot            = np.abs(je_z_plot+ji_z_plot)/np.abs(ji_z_plot)
    erel_jr_plot            = np.abs(je_r_plot+ji_x_plot)/np.abs(ji_x_plot)
    ratio_ue_t_perp_plot    = ue_t_plot/ue_perp_plot
    ratio_ue_t_para_plot    = ue_t_plot/ue_para_plot
    ratio_ue_perp_para_plot = ue_perp_plot/ue_para_plot
    lambdaD_plot            = np.sqrt(eps0*(e*Te_plot)/(ne_plot*e**2))
    
    ue                      = np.sqrt(ue_r**2 +ue_t**2 + ue_z**2)
    ratio_Ekin_Te           = (0.5*me*ue**2/e)/Te
    ji_x                    = ji1_x + ji2_x
    ji_y                    = ji1_y + ji2_y
    ji_z                    = ji1_z + ji2_z
    j_r                     = ji_x + je_r
    j_t                     = ji_y + je_t
    j_z                     = ji_z + je_z
    je2D                    = np.sqrt(je_r**2 + je_z**2)
    ji2D                    = np.sqrt(ji_x**2 + ji_z**2)
    j2D                     = np.sqrt(j_r**2 + j_z**2)
    lambdaD                 = np.sqrt(eps0*(e*Te)/(ne*e**2))
    ###########################################################################
    print("Printing average variables values...")
    ###################### PRINTING VARIABLES (AVG) ###########################
    print("erel_ue max         = %15.8e; erel_ue min         = %15.8e (-)" %( np.nanmax(erel_ue_plot), np.nanmin(erel_ue_plot) ) )
    print("erel_je max         = %15.8e; erel_je min         = %15.8e (-)" %( np.nanmax(erel_je_plot), np.nanmin(erel_je_plot) ) )
    print("erel_je2D max       = %15.8e; erel_je2D min       = %15.8e (-)" %( np.nanmax(erel_je2D_plot), np.nanmin(erel_je2D_plot) ) )
    print("erel_jeji max       = %15.8e; erel_jeji min       = %15.8e (-)" %( np.nanmax(erel_jeji_plot), np.nanmin(erel_jeji_plot) ) )
    print("erel_jz rgt inf max = %15.8e; erel_jz rgt inf min = %15.8e (-)" %( np.nanmax(erel_jz_plot[1:-1,-1]), np.nanmin(erel_jz_plot[1:-1,-1]) ) )
    print("erel_jz D bot p max = %15.8e; erel_jz D bot p min = %15.8e (-)" %( np.nanmax(erel_jz_plot[1:int(eta_min),int(xi_bottom)]), np.nanmin(erel_jz_plot[1:int(eta_min),int(xi_bottom)]) ) )
    print("erel_jz D top p max = %15.8e; erel_jz D top p min = %15.8e (-)" %( np.nanmax(erel_jz_plot[int(eta_max)+1:-1,int(xi_top)]), np.nanmin(erel_jz_plot[int(eta_max)+1:-1,int(xi_top)]) ) )        
    print("erel_jr top inf max = %15.8e; erel_jr top inf min = %15.8e (-)" %( np.nanmax(erel_jr_plot[-1,int(xi_top)+1:-1]), np.nanmin(erel_jr_plot[-1,int(xi_top)+1:-1]) ) )
    print("erel_jr D bot max   = %15.8e; erel_jr D bot min   = %15.8e (-)" %( np.nanmax(erel_jr_plot[int(eta_min),1:int(xi_bottom)]), np.nanmin(erel_jr_plot[int(eta_min),1:int(xi_bottom)]) ) )        
    print("erel_jr D top max   = %15.8e; erel_jr D top min   = %15.8e (-)" %( np.nanmax(erel_jr_plot[int(eta_max),1:int(xi_top)]), np.nanmin(erel_jr_plot[int(eta_max),1:int(xi_top)]) ) )                
#        print("erel_jr axi top max = %15.8e; erel_jr axi top min = %15.8e (-)" %( np.nanmax(erel_jr_plot[0,xi_top::]), np.nanmin(erel_jr_plot[0,xi_top::]) ) )                        
    print("phi max             = %15.8e; phi min             = %15.8e (V)" %( np.nanmax(phi_plot), np.nanmin(phi_plot) ) )
    print("Efield max          = %15.8e; Efield min          = %15.8e (V/m)" %( np.nanmax(Efield_plot), np.nanmin(Efield_plot) ) )
    print("Er max              = %15.8e; Er min              = %15.8e (V/m)" %( np.nanmax(Er_plot), np.nanmin(Er_plot) ) )
    print("Ez max              = %15.8e; Ez min              = %15.8e (V/m)" %( np.nanmax(Ez_plot), np.nanmin(Ez_plot) ) )
    print("ne max              = %15.8e; ne min              = %15.8e (1/m3)" %( np.nanmax(ne_plot), np.nanmin(ne_plot) ) )
    print("ni1 max             = %15.8e; ni1 min             = %15.8e (1/m3)" %( np.nanmax(ni1_plot), np.nanmin(ni1_plot) ) )
    print("ni2 max             = %15.8e; ni2 min             = %15.8e (1/m3)" %( np.nanmax(ni2_plot), np.nanmin(ni2_plot) ) )
    print("ni1/ni2 max         = %15.8e; ni1/ni2 min         = %15.8e (-)" %( np.nanmax(ratio_ni1_ni2_plot), np.nanmin(ratio_ni1_ni2_plot) ) )
    print("nn1 max             = %15.8e; nn1 min             = %15.8e (1/m3)" %( np.nanmax(nn1_plot), np.nanmin(nn1_plot) ) )
    print("Te max              = %15.8e; nn1 min             = %15.8e (eV)" %( np.nanmax(Te_plot), np.nanmin(Te_plot) ) )
    print("Ti1 max             = %15.8e; Ti1 min             = %15.8e (eV)" %( np.nanmax(Ti1_plot), np.nanmin(Ti1_plot) ) )
    print("Ti2 max             = %15.8e; Ti2 min             = %15.8e (eV)" %( np.nanmax(Ti2_plot), np.nanmin(Ti2_plot) ) )
    print("Tn1 max             = %15.8e; Tn1 min             = %15.8e (eV)" %( np.nanmax(Tn1_plot), np.nanmin(Tn1_plot) ) )
    print("Ekin_e max          = %15.8e; Ekin_e min          = %15.8e (eV)" %( np.nanmax(Ekin_e_plot), np.nanmin(Ekin_e_plot) ) )
    print("Ekin_i1 max         = %15.8e; Ekin_i1 min         = %15.8e (eV)" %( np.nanmax(Ekin_i1_plot), np.nanmin(Ekin_i1_plot) ) )
    print("Ekin_i2 max         = %15.8e; Ekin_i2 min         = %15.8e (eV)" %( np.nanmax(Ekin_i2_plot), np.nanmin(Ekin_i2_plot) ) )
    print("Ekin/Te max         = %15.8e; Ekin/Te min         = %15.8e (-)" %( np.nanmax(ratio_Ekin_Te_plot), np.nanmin(ratio_Ekin_Te_plot) ) )
    print("Ekin/Ti1 max        = %15.8e; Ekin/Ti1 min        = %15.8e (-)" %( np.nanmax(ratio_Ekin_Ti1_plot), np.nanmin(ratio_Ekin_Ti1_plot) ) )
    print("Ekin/Ti2 max        = %15.8e; Ekin/Ti2 min        = %15.8e (-)" %( np.nanmax(ratio_Ekin_Ti2_plot), np.nanmin(ratio_Ekin_Ti2_plot) ) )
    print("Mi1 max             = %15.8e; Mi1 min             = %15.8e (-)" %( np.nanmax(Mi1_plot), np.nanmin(Mi1_plot) ) )
    print("Mi2 max             = %15.8e; Mi2 min             = %15.8e (-)" %( np.nanmax(Mi2_plot), np.nanmin(Mi2_plot) ) )
    print("uimean max          = %15.8e; uimean min          = %15.8e (m/s)" %( np.nanmax(uimean_plot), np.nanmin(uimean_plot) ) )        
    print("ue max              = %15.8e; ue min              = %15.8e (m/s)" %( np.nanmax(ue_plot), np.nanmin(ue_plot) ) )
    print("ue_r max            = %15.8e; ue_r min            = %15.8e (m/s)" %( np.nanmax(ue_r_plot), np.nanmin(ue_r_plot) ) )
    print("ue_t max            = %15.8e; ue_t min            = %15.8e (m/s)" %( np.nanmax(ue_t_plot), np.nanmin(ue_t_plot) ) )
    print("ue_z max            = %15.8e; ue_z min            = %15.8e (m/s)" %( np.nanmax(ue_z_plot), np.nanmin(ue_z_plot) ) )
    print("ue_perp max         = %15.8e; ue_perp min         = %15.8e (m/s)" %( np.nanmax(ue_perp_plot), np.nanmin(ue_perp_plot) ) )
    print("ue_para max         = %15.8e; ue_para min         = %15.8e (m/s)" %( np.nanmax(ue_para_plot), np.nanmin(ue_para_plot) ) )
    print("ue_t/ue_perp max    = %15.8e; ue_t/ue_perp min    = %15.8e (m/s)" %( np.nanmax(ratio_ue_t_perp_plot), np.nanmin(ratio_ue_t_perp_plot) ) )
    print("ue_t/ue_para max    = %15.8e; ue_t/ue_para min    = %15.8e (m/s)" %( np.nanmax(ratio_ue_t_para_plot), np.nanmin(ratio_ue_t_para_plot) ) )
    print("ue_perp/ue_para max = %15.8e; ue_perp/ue_para min = %15.8e (m/s)" %( np.nanmax(ratio_ue_perp_para_plot), np.nanmin(ratio_ue_perp_para_plot) ) )
    print("je_r max            = %15.8e; je_r min            = %15.8e (A/m2)" %( np.nanmax(je_r_plot), np.nanmin(je_r_plot) ) )
    print("je_t max            = %15.8e; je_t min            = %15.8e (A/m2)" %( np.nanmax(je_t_plot), np.nanmin(je_t_plot) ) )
    print("je_z max            = %15.8e; je_z min            = %15.8e (A/m2)" %( np.nanmax(je_z_plot), np.nanmin(je_z_plot) ) )
    print("je_perp max         = %15.8e; je_perp min         = %15.8e (A/m2)" %( np.nanmax(je_perp_plot), np.nanmin(je_perp_plot) ) )
    print("je_para max         = %15.8e; je_para min         = %15.8e (A/m2)" %( np.nanmax(je_para_plot), np.nanmin(je_para_plot) ) )
    print("je max              = %15.8e; je min              = %15.8e (A/m2)" %( np.nanmax(je_plot), np.nanmin(je_plot) ) )
    print("ji max              = %15.8e; ji min              = %15.8e (A/m2)" %( np.nanmax(ji_plot), np.nanmin(ji_plot) ) )
    print("je 2D max           = %15.8e; je 2D min           = %15.8e (A/m2)" %( np.nanmax(je2D_plot), np.nanmin(je2D_plot) ) )
    print("ji 2D max           = %15.8e; ji 2D min           = %15.8e (A/m2)" %( np.nanmax(ji2D_plot), np.nanmin(ji2D_plot) ) )
    print("j max               = %15.8e; j min               = %15.8e (A/m2)" %( np.nanmax(j_plot), np.nanmin(j_plot) ) )
    print("F_theta max         = %15.8e; F_theta min         = %15.8e (A/m2)" %( np.nanmax(F_theta_plot), np.nanmin(F_theta_plot) ) )
    print("Hall_par max        = %15.8e; Hall_par min        = %15.8e (A/m2)" %( np.nanmax(Hall_par_plot), np.nanmin(Hall_par_plot) ) )
    print("Hall_par_eff max    = %15.8e; Hall_par_eff min    = %15.8e (A/m2)" %( np.nanmax(Hall_par_eff_plot), np.nanmin(Hall_par_eff_plot) ) )
    print("nu_e_tot max        = %15.8e; nu_e_tot min        = %15.8e (A/m2)" %( np.nanmax(nu_e_tot_plot), np.nanmin(nu_e_tot_plot) ) )
    print("nu_e_tot_eff max    = %15.8e; nu_e_tot_eff min    = %15.8e (A/m2)" %( np.nanmax(nu_e_tot_eff_plot), np.nanmin(nu_e_tot_eff_plot) ) )
    print("nu_en max           = %15.8e; nu_en min           = %15.8e (A/m2)" %( np.nanmax(nu_en_plot), np.nanmin(nu_en_plot) ) )
    print("nu_ei1 max          = %15.8e; nu_ei1 min          = %15.8e (A/m2)" %( np.nanmax(nu_ei1_plot), np.nanmin(nu_ei1_plot) ) )
    print("nu_ei2 max          = %15.8e; nu_ei2 min          = %15.8e (A/m2)" %( np.nanmax(nu_ei2_plot), np.nanmin(nu_ei2_plot) ) )
    print("nu_i01 max          = %15.8e; nu_i01 min          = %15.8e (A/m2)" %( np.nanmax(nu_i01_plot), np.nanmin(nu_i01_plot) ) )
    print("nu_i02 max          = %15.8e; nu_i02 min          = %15.8e (A/m2)" %( np.nanmax(nu_i02_plot), np.nanmin(nu_i02_plot) ) )
    print("nu_i12 max          = %15.8e; nu_i12 min          = %15.8e (A/m2)" %( np.nanmax(nu_i12_plot), np.nanmin(nu_i12_plot) ) )
    print("lambdaD max         = %15.8e; lambdaD min         = %15.8e (mm)" %( np.nanmax(lambdaD_plot*1E3), np.nanmin(lambdaD_plot*1E3) ) )
    ###########################################################################
    print("Plotting...")
    ############################ GENERATING PLOTS #############################
    zs                = zs*1E2
    rs                = rs*1E2
    zscells           = zscells*1E2
    rscells           = rscells*1E2
    points            = points*1E2
    nodes[0,:]        = nodes[0,:]*1e2
    nodes[1,:]        = nodes[1,:]*1e2
    z_cath            = z_cath*1E2
    r_cath            = r_cath*1E2
    elem_geom[0,:]    = elem_geom[0,:]*1E2
    elem_geom[1,:]    = elem_geom[1,:]*1E2
    Ez_plot_anode     = np.copy(Ez_plot)*1E-3
    je_z_plot_anode   = np.copy(je_z_plot)*1E-3
    ji_z_plot_anode   = np.copy(ji_z_plot)*1E-2
    Efield            = Efield*1E-3
    Ez                = Ez*1E-3
    je_para           = je_para*1E-4
    je_perp           = je_perp*1E-4
    je_t              = je_t*1E-4
    ji_x              = ji_x*1E-3
    ji_z              = ji_z*1E-3
#    Efield_plot_cont  = np.copy(Efield_plot) # NOTE: Perform the norm of the time-averaged vector for plotting to avoid averaging errors
    Er_plot_cont      = np.copy(Er_plot)
    Ez_plot_cont      = np.copy(Ez_plot)
    Efield_plot_cont  = np.sqrt(Er_plot_cont**2 + Ez_plot_cont**2)
    Efield_plot       = Efield_plot*1E-3
    Er_plot           = Er_plot*1E-3
    Ez_plot           = Ez_plot*1E-3
    je_para_plot      = je_para_plot*1E-4 # This is A/cm2
    je_perp_plot      = je_perp_plot*1E-4 # This is A/cm2
    je_t_plot         = je_t_plot*1E-4    # This is A/cm2
    je_z_plot         = je_z_plot*1E-4    # This is A/cm2
    je_r_plot         = je_r_plot*1E-4    # This is A/cm2
    ji_x_plot         = ji_x_plot*1E-3
    ji_z_plot         = ji_z_plot*1E-3
#    je2D_plot         = je2D_plot*1E-4
#    nu_e_tot_plot     = nu_e_tot_plot*1E-6
#    nu_e_tot_eff_plot = nu_e_tot_eff_plot*1E-6
#    nu_en_plot        = nu_en_plot*1E-6
#    nu_ei1_plot       = nu_ei1_plot*1E-6
#    nu_ei2_plot       = nu_ei2_plot*1E-6
#    nu_i01_plot       = nu_i01_plot*1E-6
#    nu_i02_plot       = nu_i02_plot*1E-6
#    nu_i12_plot       = nu_i12_plot*1E-6
    lambdaD      = lambdaD*1E3
    lambdaD_plot = lambdaD_plot*1E3
    # B field in Gauss
    Bfield_plot = Bfield_plot*1e4
    Br_plot     = Br_plot*1e4
    Bz_plot     = Bz_plot*1e4 
    
    # Time in ms
    time       = time*1e3
    
    
    # Figures for axial position of max ne and Ez
    plt.figure(r'zmax_ne')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    plt.title(r"$z_{max}$ (cm)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    plt.plot(time, zmax_ne_1Dz, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[k], markeredgecolor = 'k', label=r"$n_e$")
    plt.plot(time, zmax_Ez_1Dz, linestyle=linestyles[k+1], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[k+1], markeredgecolor = 'k', label=r"$E_z$")
    plt.legend(fontsize = font_size_legend-2,loc=4,ncol=2) 
    
    # Prepare figure for paper
    [fig, axes] = plt.subplots(nrows=2, ncols=2, figsize=(15,12))
    ax1 = plt.subplot2grid( (2,2), (0,0) )
    ax2 = plt.subplot2grid( (2,2), (0,1) )
    ax3 = plt.subplot2grid( (2,2), (1,0) )
    ax4 = plt.subplot2grid( (2,2), (1,1) )
    
    ax1.set_title(r"(a) $n$ [m$^{-3}$]", fontsize = font_size,y=1.02)   
    ax1.set_xlabel(r"$z$ [cm]",fontsize = font_size)
    ax1.set_ylabel(r"$r$ [cm]",fontsize = font_size)
    ax1.tick_params(labelsize = ticks_size) 
    ax2.set_title(r"(b) $T_e$ [eV]", fontsize = font_size,y=1.02)   
    ax2.set_xlabel(r"$z$ [cm]",fontsize = font_size)
    ax2.set_ylabel(r"$r$ [cm]",fontsize = font_size)
    ax2.tick_params(labelsize = ticks_size) 
    ax3.set_title(r"(c) $\phi$ [V]", fontsize = font_size,y=1.02)   
    ax3.set_xlabel(r"$z$ [cm]",fontsize = font_size)
    ax3.set_ylabel(r"$r$ [cm]",fontsize = font_size)
    ax3.tick_params(labelsize = ticks_size) 
    ax4.set_title(r"(d) $|\boldsymbol{B}|$ [G]", fontsize = font_size,y=1.02)   
    ax4.set_xlabel(r"$z$ [cm]",fontsize = font_size)
    ax4.tick_params(labelsize = ticks_size) 
    
    
    
    # Plotting average profiles and contours
    # Obtain the vectors for the uniform mesh for streamlines plotting. It must be uniform and squared mesh
    delta_x = 0.11
    zvec = np.arange(zs[0,0],zs[0,-1]+delta_x,delta_x)
    rvec = np.copy(zvec)
    
#    rvec = np.arange(rs[0,0],rs[-1,0]+delta_x,delta_x)
#    zvec = np.copy(rvec)
       
    # Phi plot
    log_type         = 0
    auto             = 0
#        min_val0         = -2.0
#        max_val0         = 300.0
    min_val0         = -10.0
    max_val0         = 310.0
    cont             = 1
    lines            = 0
    cont_nlevels     = nlevels_2Dcontour
    auto_cbar_ticks  = 0 
    auto_lines_ticks = 0
    nticks_cbar      = 14
    nticks_lines     = 10
    cbar_ticks       = np.array([300, 250, 200, 150, 100, 50, 0])
#        cbar_ticks       = np.array([500, 450, 400, 450, 400, 350, 300, 250, 200, 150, 100, 50, 0])
#        cbar_ticks       = np.array([250, 200, 150, 100, 50, 0]) # TOPO1
#        cbar_ticks       = np.array([650,600,550,500,450,400,350,300,250, 200, 150, 100, 50, 0]) # TOPO2
#        lines_ticks      = np.array([305, 280, 250, 200, 150, 100, 75, 50, 40, 30, 25, 15, 10, 5, 1, 0, -1, -2, -3, -4, -5, -6])
#        lines_ticks      = np.array([305, 280, 250, 200, 150, 100, 75, 50, 40, 25, 15, 10, 5])
    lines_ticks      = np.array([5,10,15,20,25,30,40,50,75,100,150,200,250,280,290,302])
#        lines_ticks      = np.array([5,10,15,25,30,35,45,55,75,100,150,175,200,250,300,400,500]) # TOPO1
#        lines_ticks      = np.array([5,10,15,20,40,50,65,80,100,150,250,350,450,500,550,600,650]) # TOPO2 N3
#        lines_ticks      = np.array([10,40,50,65,80,100,150,250,350,450,500]) # TOPO2 N4
    lines_ticks_loc  = 'default'
    cbar_ticks_fmt    = '{%.0f}'
    lines_ticks_fmt   = '{%.0f}'
    lines_width       = line_width
    lines_ticks_color = 'k'
    lines_style       = '-'
    [CS,CS2] = contour_2D (ax3,'$z$ [cm]', '$r$ [cm]', font_size, ticks_size, zs, rs, phi_plot, nodes_flag, log_type, auto, 
                           min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                           nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                           lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)     
#    ax3.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, fontsize=ticks_size_isolines, zorder = 1)
    ax3.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
    ax3.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)    
    ax3.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
#    ax3.plot(2.10,0.5*(points[-1,1]+points[0,1]),'ko',linewidth = line_width_boundary,markersize = marker_size)  
#    ax3.plot(np.array([0.0,zs[0,-1]],dtype=float),np.array([0.5*(points[-1,1]+points[0,1]),0.5*(points[-1,1]+points[0,1])],dtype=float),'k-',linewidth = line_width_boundary,markersize = marker_size)   
#    if plot_cath_contours == 1:
#        ax3.plot(z_cath,r_cath,'wo',linewidth = line_width_boundary,markersize = marker_size)    
    ax3.set_xticks(np.arange(0,zs[0,-1]+1,2))
    ax3.set_yticks(np.arange(0,rs[-1,0]+1,2))
    

    
    # ne plot
    log_type         = 1
    auto             = 1
    min_val0         = 1E12
    max_val0         = 5E14
    cont             = 1
    lines            = 0
    cont_nlevels     = nlevels_2Dcontour
    auto_cbar_ticks  = 1 
    auto_lines_ticks = 0
    nticks_cbar      = 4
    nticks_lines     = 4
    cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
#        lines_ticks      = np.sort(np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16]))
#        lines_ticks      = np.sort(np.array([2E18,5E17,2E17,1E17,5E16,1E16,5E15,1E15]))
    lines_ticks      = np.sort(np.array([2E18,6E17,2E17,1E17,5E16,1E16,5E15,1E15])) # TOPO2 N4
#        lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
    lines_ticks_loc  = 'default'
    cbar_ticks_fmt    = '{%.2f}'
    lines_ticks_fmt   = '{%.1f}'
    lines_width       = line_width
    lines_ticks_color = 'k'
    lines_style       = '-'
    [CS,CS2] = contour_2D (ax1,'$z$ [cm]', '$r$ [cm]', font_size, ticks_size, zs, rs, ne_plot, nodes_flag, log_type, auto, 
                      min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                      nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                      lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
    # Isolines ticks (exponent)
#    ax1.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, fontsize=ticks_size_isolines, zorder = 1)
#        ax.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
    ax1.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
    ax1.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
    ax1.plot(2.10,0.5*(points[-1,1]+points[0,1]),'ko',linewidth = line_width_boundary,markersize = marker_size)  
    ax1.plot(np.array([0.0,zs[0,-1]],dtype=float),np.array([0.5*(points[-1,1]+points[0,1]),0.5*(points[-1,1]+points[0,1])],dtype=float),'k-',linewidth = line_width_boundary,markersize = marker_size)   
    
#        if Bline_all2Dplots == 1:        
#            plt.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
    if plot_cath_contours == 1:
        ax1.plot(z_cath,r_cath,'wo',linewidth = line_width_boundary,markersize = marker_size) 
    ax1.set_xticks(np.arange(0,zs[0,-1]+1,2))
    ax1.set_yticks(np.arange(0,rs[-1,0]+1,2))
        

    # Te plot
    log_type         = 0
    auto             = 0
    min_val0         = 0
    max_val0         = 45.0
    max_val0         = 35.0
    cont             = 1
    lines            = 0
    cont_nlevels     = nlevels_2Dcontour
    auto_cbar_ticks  = 0 
    auto_lines_ticks = 0
    nticks_cbar      = 14
    nticks_lines     = 10
    cbar_ticks       = np.array([45,40,35,30,25,20,15,10,5,0])
#        cbar_ticks       = np.array([25,20,15,10,5,0]) # TOPO 1
#        cbar_ticks       = np.array([65,60,55,50,45,40,35,30,25,20,15,10,5,0]) # TOPO 2
#        lines_ticks      = np.array([7,9,12,20,25,30,35,40,45])
#        lines_ticks      = np.array([3,4,5,6,7,8,9,12,15,18,20,22,25,30,35,40,45])
#        lines_ticks      = np.array([1,5,6,7,8,10,15,18,20,21]) # TOPO1
#        lines_ticks      = np.array([1,5,10,12,14,25,30,40,50,55,58]) # TOPO2 N3
    lines_ticks      = np.array([1,7,9,10,11,12,15,25,30,34,45,47]) # TOPO2 N4
#        lines_ticks_loc  = [(0.38,4.25),(0.88,4.25),(1.5,4.25),(2.7,4.6),(3.0,3.8),(3.6,4.8),(3.9,4.25),(4.5,4.25),(5.18,4.0),(5.3,3.2),(5.6,1.8),(3.7,6.8)]
    lines_ticks_loc  = 'default'
    cbar_ticks_fmt    = '{%.0f}'
    lines_ticks_fmt   = '{%.0f}'
    lines_width       = line_width
    lines_ticks_color = 'k'
    lines_style       = '-'
    [CS,CS2] = contour_2D (ax2,'$z$ [cm]', '$r$ [cm]', font_size, ticks_size, zs, rs, Te_plot, nodes_flag, log_type, auto, 
                           min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                           nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                           lines_ticks_fmt, lines_ticks_color, lines_style, lines_width) 
#    ax2.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, fontsize=ticks_size_isolines, zorder = 1)
#        ax.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
    ax2.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
    ax2.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
#    ax2.plot(2.10,0.5*(points[-1,1]+points[0,1]),'ko',linewidth = line_width_boundary,markersize = marker_size)  
#    ax2.plot(np.array([0.0,zs[0,-1]],dtype=float),np.array([0.5*(points[-1,1]+points[0,1]),0.5*(points[-1,1]+points[0,1])],dtype=float),'k-',linewidth = line_width_boundary,markersize = marker_size)   
    if Bline_all2Dplots == 1:        
        ax2.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)        
#    if plot_cath_contours == 1:
#        ax2.plot(z_cath,r_cath,'wo',linewidth = line_width_boundary,markersize = marker_size) 
    ax2.set_xticks(np.arange(0,zs[0,-1]+1,2))
    ax2.set_yticks(np.arange(0,rs[-1,0]+1,2))
        
    
    # B plot
    log_type         = 1
    auto             = 1
    min_val0         = 1E12
    max_val0         = 5E14
    cont             = 1
    lines            = 0
    cont_nlevels     = nlevels_2Dcontour
    auto_cbar_ticks  = 1 
    auto_lines_ticks = 0
    nticks_cbar      = 4
    nticks_lines     = 6
    cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
    lines_ticks      = np.array([2,5,10,20,50,100,200,300])
#        lines_ticks_loc  =  [(10.6,0.37),(10.02,2.21),(9.0,4.25),(7.68,4.25),(0.17,4.2),(0.7, 4.4),(1.7, 4.25),(2.5, 3.7),(5.8,4.25),(4.5,4.25),(3.2,4.25)]
    lines_ticks_loc  = "default"
    cbar_ticks_fmt    = '{%.2f}'
    lines_ticks_fmt   = '{%.1f}'
    lines_width       = line_width
    lines_ticks_color = 'k'
    lines_style       = '-'
    [CS,CS2] = contour_2D (ax4,'$z$ [cm]', '$r$ [cm]', font_size, ticks_size, zs, rs, Bfield_plot, nodes_flag, log_type, auto, 
                           min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                           nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                           lines_ticks_fmt, lines_ticks_color, lines_style, line_width_boundary)   
#    plot_MFAM_ax(ax4,faces,nodes,line_width_Blines)
    plot_MFAM_ax_nosig(ax4,faces,nodes,line_width_Blines)
    # Isolines ticks (exponent)
#    ax4.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, fontsize=ticks_size_isolines, zorder = 1)
#        ax4.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
    ax4.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
    ax4.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)
#    ax4.plot(2.10,0.5*(points[-1,1]+points[0,1]),'ko',linewidth = line_width_boundary,markersize = marker_size)  
#    ax4.plot(np.array([0.0,zs[0,-1]],dtype=float),np.array([0.5*(points[-1,1]+points[0,1]),0.5*(points[-1,1]+points[0,1])],dtype=float),'k-',linewidth = line_width_boundary,markersize = marker_size)   
#    if plot_cath_contours == 1:
#        ax4.plot(z_cath,r_cath,'wo',linewidth = line_width_boundary,markersize = marker_size) 
    ax4.set_xticks(np.arange(0,zs[0,-1]+1,2))
    ax4.set_yticks(np.arange(0,rs[-1,0]+1,2))
           

    plt.tight_layout()
    if save_flag == 1:
#        plt.savefig(path_out+"DMD_Vd300_fig"+figs_format,bbox_inches='tight') 
#        plt.savefig(path_out+"DMD_Vd300_fig_Blines"+figs_format,bbox_inches='tight') 
#        plt.savefig(path_out+"DMD_Vd300_fig_Blines_test"+figs_format,bbox_inches='tight') 
#        plt.savefig(path_out+"DMD_Vd300_fig_Blines_test2"+figs_format,bbox_inches='tight') 
        plt.savefig(path_out+"DMD_Vd300_fig_Blines_final"+figs_format,bbox_inches='tight') 
        plt.close()
                   
    
    ###########################################################################
