#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:10:12 2020

@author: adrian

###############################################################################
Description:    This Python function carries out the post processing of the 
                CORE resutls for the HET sims
###############################################################################
Inputs:         No inputs
###############################################################################
Outputs:        Plots and outputs for the simulations
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
from HET_sims_read_DMD import HET_sims_read_DMD
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
from pydmd import DMD
from pydmd import FbDMD
from pydmd import HODMD
import scipy
from matplotlib import animation
import types
from DMD import dmd_rom, dmd

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
video_type          = 0 # 0 deactivate video
                        # 1 video type 1 (reconstructed density and reconstruction error using all modes)
                        # 2 video type 2 (reconstructed density and reconstruction error using selected modes)


path_out      = "HET_DMD_videos/rel_last_steps6000_20modes_dpar50/"
path_out      = "HET_DMD_videos/rel_last_steps6000_20modes_dpar250/"
path_out_data = "DMD_data/"


if save_flag == 1 and os.path.isdir(path_out) != 1:  
    sys.exit("ERROR: path_out is not an existing directory")


# Set options for LaTeX font
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
font_size           = 18
font_size_legend    = font_size - 10
ticks_size          = 18
ticks_size_isolines = ticks_size - 15
text_size           = 25 - 10
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
marker_size            = 3
marker_size_cath       = 5
marker_size_moving     = 8
xticks_pad             = 6.0
nlevels_cont           = 50

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
###############################################################################




if prof_plots == 1:
    print("######## prof_plots ########")
    
    num_firstmax       = 20   # Number of first maximum values searched for in the FFTs of Id
    num_firstmax_print = 5    # Number of first maximum values above to be printed
    order              = 50   # Used for obtaining FFT (correctly identify integer number of cycles)
    order_fast         = 500  # Used for obtaining FFT (correctly identify integer number of cycles)
    
#    marker_size  = 4
    marker_size_cath = 14
    cathode_marker = '*'
    cathode_color  = orange
    ref_color      = 'c'
    marker_every = 3
#    font_size_legend    = font_size - 15
    font_size_legend    = 15
    
    # Radial index for axial profiles

    rind = 21   # New picM for SPT100
#    rind = 32   # New picM for SPT100 (picrm)
#    rind = 19   # picM for SPT100 thesis
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
#    zcat_pos       = 7.47248             # z coordinate of crossing point of cathode topo2 1200 Bline with rind = 17
#    zcat_pos       = 5.902074            # z coordinate of crossing point of cathode topo2 2853 Bline with rind = 17
    # TOPO 1 ------------------------------------------------------------------
    plot_zcath_012 = 2
#    zcat_pos       = 12.14428            # z coordinate of crossing point of cathode topo1 699 Bline with rind = 17
#    zcat_pos       = 7.3422075           # z coordinate of crossing point of cathode topo1 313 Bline with rind = 17   
#    zcat_pos       = 5.688635            # z coordinate of crossing point of cathode topo1 251 Bline with rind = 17 
    
    elems_cath_Bline    = range(407-1,483-1+2,2) # Elements along the cathode B line for cases C1, C2 and C3
    elems_cath_Bline_2  = range(875-1,951-1+2,2) # Elements along the cathode B line for case C5 (C4 thesis)
    elems_Bline         = range(330-1,406-1+2,2) # Elements along a B line
    ref_elem            = elems_Bline[len(elems_Bline)/2]
#    ref_elem            = elems_Bline[0]
    plot_Bline_cathBline = 1          # Only used for plots activated when plot_cath_Bline_prof = 1
    
    # Common reference potential PIC mesh node Python indeces
    phi_ref  = 0
    iphi_ref = 24
    jphi_ref = 28
    
    # Print out time steps
#    timestep = 'last'
    timestep = 1005
  
    allsteps_flag   = 1
    read_inst_data  = 1
    read_part_lists = 0
    read_flag       = 1
    
    mean_vars       = 1
    mean_type       = 0
    last_steps      = 6000
    last_steps_fast = last_steps*20
    step_i          = 421
    step_f          = 1108
    plot_mean_vars  = 1

    # DMD options
    var_name    = "ne" # DMD over Plasma density 
    
    use_PyDMD     = 1
    use_dmd_dom   = 0
    use_dmd       = 0
    
    read_vars_file = 1
    read_vars_file_name = "vars1"
    
    save_vars_file = 0
    save_vars_file_name = "rel_last_steps6000_20modes_dpar30"
    
#    nmodes      = 40
#    d_par       = 100 # Only for HODMD using PyDMD
    
#    nmodes      = 20
#    d_par       = 150 # Only for HODMD using PyDMD
    
    nmodes      = 20
    d_par       = 250 # Only for HODMD using PyDMD
    
    read_vars_file_name = "rel_last_steps"+str(last_steps)+"_"+str(nmodes)+"modes_dpar"+str(d_par)
    save_vars_file_name = "rel_last_steps"+str(last_steps)+"_"+str(nmodes)+"modes_dpar"+str(d_par)
    

    
    fr_tol = 5.0       # Tolerance for frequency kHz for selecting modes (select stable modes with frequency larger that this one)
    gr_tol = 1500.0    # Tolerance for growth rates for selecting modes (select stable modes with growth rates lower than this one in absolute value)

    if timestep == 'last':
        timestep = -1  
    if allsteps_flag == 0:
        mean_vars = 0

    
    # Simulation names
    nsims = 1

    # Flag for old sims (1: old sim files, 0: new sim files)
    oldpost_sim      = np.array([1,3,3,3,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([0,9,8,8,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)         
    
    oldpost_sim      = np.array([3,3,3,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
    oldsimparams_sim = np.array([10,8,8,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)    
    
    
    sim_names = [
#                 "../../../Rb_sims_files/SPT100_al0025_Ne5_C1",
#                 "../../../Sr_sims_files/SPT100_pm2em1_cat481_tmtetq25_RLC",
#                 "../../../Sr_sims_files/SPT100_DMD_pm2em2_cat3328_tmtetq2",
                 "../../../Sr_sims_files/SPT100_DMD_pm2em2_cat3328_tmtetq2_rel",
#                 "../../../Sr_sims_files/SPT100_DMD_pm2em2_cat3328_tmtetq2_Vd300",
                         
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
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                              "HT5k_PIC_mesh_rm3.hdf5",
                             ]
    elif topo_case == 0:    
        PIC_mesh_file_name = [
                              "SPT100_picM_Reference1500points_rm2.hdf5",
                
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

                        
    labels = [r"$\bar{n}_e$",
              r"$\bar{n}_n$"]

    
    # Line colors
    colors = ['k','r','g','b','m','c','m','y',orange,brown]
#    colors = ['k','m',orange,brown]
    # Markers
    markers = ['s','o','v','^','<', '>','D','p','*']
#    markers = ['s','<','D','p']
    # Line style
    linestyles = ['-','--','-.', ':','-','--','-.']
    linestyles = ['-','-','-','-','-','-','-']
    
    xmax = 18
              
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
#        [num_ion_spe,num_neu_spe,n_mp_cell_i,n_mp_cell_n,n_mp_cell_i_min,
#           n_mp_cell_i_max,n_mp_cell_n_min,n_mp_cell_n_max,min_ion_plasma_density,
#           m_A,spec_refl_prob,ene_bal,points,zs,rs,zscells,rscells,dims,
#           nodes_flag,cells_flag,cells_vol,volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,
#           eta_max,eta_min,xi_top,xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,
#           nsteps,nsteps_fast,nsteps_eFld,faces,nodes,elem_n,boundary_f,face_geom,elem_geom,
#           n_faces,n_elems,n_faces_boundary,bIDfaces_Dwall,bIDfaces_Awall,
#           bIDfaces_FLwall,IDfaces_Dwall,IDfaces_Awall,IDfaces_FLwall,zfaces_Dwall,
#           rfaces_Dwall,Afaces_Dwall,zfaces_Awall,rfaces_Awall,Afaces_Awall,
#           zfaces_FLwall,rfaces_FLwall,Afaces_FLwall,
#           cath_elem,z_cath,r_cath,V_cath,mass,ssIons1,ssIons2,ssNeutrals1,ssNeutrals2,
#           n_mp_i1_list,n_mp_i2_list,n_mp_n1_list,n_mp_n2_list,phi,phi_elems,Ez,Er,Efield,
#           Bz,Br,Bfield,Te,Te_elems,je_mag_elems,je_perp_elems,je_theta_elems,je_para_elems,
#           cs01,cs02,nn1,nn2,ni1,ni2,ne,ne_elems,fn1_x,fn1_y,
#           fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,
#           un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,
#           ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,
#           ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,
#           avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
#           ion_gen_weights1,ion_gen_weights2,surf_elems,n_imp_elems,imp_elems,
#           imp_elems_kbc,imp_elems_MkQ1,imp_elems_Te,imp_elems_dphi_kbc,
#           imp_elems_dphi_sh,imp_elems_nQ1,imp_elems_nQ2,imp_elems_ion_flux_in1,
#           imp_elems_ion_flux_out1,imp_elems_ion_ene_flux_in1,
#           imp_elems_ion_ene_flux_out1,imp_elems_ion_imp_ene1,
#           imp_elems_ion_flux_in2,imp_elems_ion_flux_out2,
#           imp_elems_ion_ene_flux_in2,imp_elems_ion_ene_flux_out2,
#           imp_elems_ion_imp_ene2,imp_elems_neu_flux_in1,imp_elems_neu_flux_out1,
#           imp_elems_neu_ene_flux_in1,imp_elems_neu_ene_flux_out1,
#           imp_elems_neu_imp_ene1,imp_elems_neu_flux_in2,imp_elems_neu_flux_out2,
#           imp_elems_neu_ene_flux_in2,imp_elems_neu_ene_flux_out2,
#           imp_elems_neu_imp_ene2,tot_mass_mp_neus,tot_mass_mp_ions,tot_num_mp_neus,
#           tot_num_mp_ions,tot_mass_exit_neus,tot_mass_exit_ions,mass_mp_neus,
#           mass_mp_ions,num_mp_neus,num_mp_ions,avg_dens_mp_neus,avg_dens_mp_ions,
#           eta_u,eta_prod,eta_thr,eta_div,eta_cur,thrust,thrust_ion,thrust_neu,
#           Id_inst,Id,Vd_inst,Vd,I_beam,I_tw_tot,Pd,Pd_inst,P_mat,P_inj,P_inf,P_ion,
#           P_ex,P_use_tot_i,P_use_tot_n,P_use_tot,P_use_z_i,P_use_z_n,P_use_z,
#           qe_wall,qe_wall_inst,Pe_faces_Dwall,Pe_faces_Awall,Pe_faces_FLwall,
#           Pe_faces_Dwall_inst,Pe_faces_Awall_inst,Pe_faces_FLwall_inst,
#           Pe_Dwall,Pe_Awall,Pe_FLwall,Pe_Dwall_inst,Pe_Awall_inst,Pe_FLwall_inst, 
#           Pi_Dwall,Pi_Awall,Pi_FLwall,Pi_FLwall_nonz,Pn_Dwall,Pn_Awall,Pn_FLwall,
#           Pn_FLwall_nonz,P_Dwall,P_Awall,P_FLwall,Pwalls,Pionex,Ploss,Pthrust,
#           Pnothrust,Pnothrust_walls,balP,err_balP,ctr_Pd,ctr_Ploss,ctr_Pwalls,
#           ctr_Pionex,ctr_P_DAwalls,ctr_P_FLwalls,ctr_P_FLwalls_in,ctr_P_FLwalls_i,
#           ctr_P_FLwalls_n,ctr_P_FLwalls_e,balP_Pthrust,err_balP_Pthrust,
#           ctr_balPthrust_Pd,ctr_balPthrust_Pnothrust,ctr_balPthrust_Pthrust,
#           ctr_balPthrust_Pnothrust_walls,ctr_balPthrust_Pnothrust_ionex,
#           err_def_balP,Isp_s,Isp_ms,dMdt_i1,dMdt_i2,dMdt_n1,dMdt_n2,dMdt_tot,
#           mflow_coll_i1,mflow_coll_i2,mflow_coll_n1,mflow_coll_n2,mflow_fw_i1,
#           mflow_fw_i2,mflow_fw_n1,mflow_fw_n2,mflow_tw_i1,mflow_tw_i2,mflow_tw_n1,
#           mflow_tw_n2,mflow_ircmb_picS_n1,mflow_ircmb_picS_n2,mflow_inj_i1,mflow_inj_i2,
#           mflow_fwmat_i1,mflow_fwmat_i2,mflow_inj_n1,mflow_fwmat_n1,mflow_inj_n2,
#           mflow_fwmat_n2,mflow_twmat_i1,mflow_twinf_i1,mflow_twa_i1,mflow_twmat_i2,
#           mflow_twinf_i2,mflow_twa_i2,mflow_twmat_n1,mflow_twinf_n1,mflow_twa_n1,
#           mflow_twmat_n2,mflow_twinf_n2,mflow_twa_n2,mbal_n1,mbal_i1,mbal_i2,mbal_tot,
#           err_mbal_n1,err_mbal_i1,err_mbal_i2,err_mbal_tot,ctr_mflow_coll_n1,
#           ctr_mflow_fw_n1,ctr_mflow_tw_n1,ctr_mflow_coll_i1,ctr_mflow_fw_i1,
#           ctr_mflow_tw_i1,ctr_mflow_coll_i2,ctr_mflow_fw_i2,ctr_mflow_tw_i2,
#           ctr_mflow_coll_tot,ctr_mflow_fw_tot,ctr_mflow_tw_tot,dEdt_i1,dEdt_i2,
#           dEdt_n1,dEdt_n2,eneflow_coll_i1,eneflow_coll_i2,eneflow_coll_n1,
#           eneflow_coll_n2,eneflow_fw_i1,eneflow_fw_i2,eneflow_fw_n1,eneflow_fw_n2,
#           eneflow_tw_i1,eneflow_tw_i2,eneflow_tw_n1,eneflow_tw_n2,Pfield_i1,
#           Pfield_i2,eneflow_inj_i1,eneflow_fwmat_i1,eneflow_inj_i2,
#           eneflow_fwmat_i2,eneflow_inj_n1,eneflow_fwmat_n1,eneflow_inj_n2,
#           eneflow_fwmat_n2,eneflow_twmat_i1,eneflow_twinf_i1,eneflow_twa_i1,
#           eneflow_twmat_i2,eneflow_twinf_i2,eneflow_twa_i2,eneflow_twmat_n1,
#           eneflow_twinf_n1,eneflow_twa_n1,eneflow_twmat_n2,eneflow_twinf_n2,
#           eneflow_twa_n2,ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,Te_cath,
#           nu_cath,ndot_cath,Q_cath,P_cath,V_cath_tot,ne_cath_avg,F_theta,Hall_par,
#           Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,
#           nu_ex,Boltz,Boltz_dim,Pfield_e,Ebal_e,ge_b,ge_b_acc,ge_sb_b,ge_sb_b_acc,
#           delta_see,delta_see_acc,err_interp_n] = HET_sims_read(path_simstate_inp,path_simstate_out,
#                                                    path_postdata_out,path_simparams_inp,
#                                                    path_picM,allsteps_flag,timestep,read_inst_data,
#                                                    read_part_lists,read_flag,oldpost_sim[k],oldsimparams_sim[k])
        
        [num_ion_spe,num_neu_spe,n_mp_cell_i,n_mp_cell_n,n_mp_cell_i_min,
           n_mp_cell_i_max,n_mp_cell_n_min,n_mp_cell_n_max,min_ion_plasma_density,
           m_A,spec_refl_prob,ene_bal,points,zs,rs,zscells,rscells,dims,
           nodes_flag,cells_flag,cells_vol,volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,
           eta_max,eta_min,xi_top,xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,
           nsteps,nsteps_fast,nsteps_eFld,faces,nodes,elem_n,boundary_f,face_geom,elem_geom,
           n_faces,n_elems,n_faces_boundary,bIDfaces_Dwall,bIDfaces_Awall,
           bIDfaces_FLwall,IDfaces_Dwall,IDfaces_Awall,IDfaces_FLwall,zfaces_Dwall,
           rfaces_Dwall,Afaces_Dwall,zfaces_Awall,rfaces_Awall,Afaces_Awall,
           zfaces_FLwall,rfaces_FLwall,Afaces_FLwall,
           cath_elem,z_cath,r_cath,V_cath,mass,ssIons1,ssIons2,ssNeutrals1,ssNeutrals2,
           n_mp_i1_list,n_mp_i2_list,n_mp_n1_list,n_mp_n2_list,phi,phi_elems,Ez,Er,Efield,
           Bz,Br,Bfield,Te,Te_elems,je_mag_elems,je_perp_elems,je_theta_elems,je_para_elems,
           cs01,cs02,nn1,nn2,ni1,ni2,ne,ne_elems,fn1_x,fn1_y,
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
           P_ex,P_use_tot_i,P_use_tot_n,P_use_tot,P_use_z_i,P_use_z_n,P_use_z,
           qe_wall,qe_wall_inst,Pe_faces_Dwall,Pe_faces_Awall,Pe_faces_FLwall,
           Pe_faces_Dwall_inst,Pe_faces_Awall_inst,Pe_faces_FLwall_inst,
           Pe_Dwall,Pe_Awall,Pe_FLwall,Pe_Dwall_inst,Pe_Awall_inst,Pe_FLwall_inst, 
           Pi_Dwall,Pi_Awall,Pi_FLwall,Pi_FLwall_nonz,Pn_Dwall,Pn_Awall,Pn_FLwall,
           Pn_FLwall_nonz,P_Dwall,P_Awall,P_FLwall,Pwalls,Pionex,Ploss,Pthrust,
           Pnothrust,Pnothrust_walls,balP,err_balP,ctr_Pd,ctr_Ploss,ctr_Pwalls,
           ctr_Pionex,ctr_P_DAwalls,ctr_P_FLwalls,ctr_P_FLwalls_in,ctr_P_FLwalls_i,
           ctr_P_FLwalls_n,ctr_P_FLwalls_e,balP_Pthrust,err_balP_Pthrust,
           ctr_balPthrust_Pd,ctr_balPthrust_Pnothrust,ctr_balPthrust_Pthrust,
           ctr_balPthrust_Pnothrust_walls,ctr_balPthrust_Pnothrust_ionex,
           err_def_balP,Isp_s,Isp_ms,dMdt_i1,dMdt_i2,dMdt_n1,dMdt_n2,dMdt_tot,
           mflow_coll_i1,mflow_coll_i2,mflow_coll_n1,mflow_coll_n2,mflow_fw_i1,
           mflow_fw_i2,mflow_fw_n1,mflow_fw_n2,mflow_tw_i1,mflow_tw_i2,mflow_tw_n1,
           mflow_tw_n2,mflow_ircmb_picS_n1,mflow_ircmb_picS_n2,mflow_inj_i1,mflow_inj_i2,
           mflow_fwmat_i1,mflow_fwmat_i2,mflow_inj_n1,mflow_fwmat_n1,mflow_inj_n2,
           mflow_fwmat_n2,mflow_twmat_i1,mflow_twinf_i1,mflow_twa_i1,mflow_twmat_i2,
           mflow_twinf_i2,mflow_twa_i2,mflow_twmat_n1,mflow_twinf_n1,mflow_twa_n1,
           mflow_twmat_n2,mflow_twinf_n2,mflow_twa_n2,mbal_n1,mbal_i1,mbal_i2,mbal_tot,
           err_mbal_n1,err_mbal_i1,err_mbal_i2,err_mbal_tot,ctr_mflow_coll_n1,
           ctr_mflow_fw_n1,ctr_mflow_tw_n1,ctr_mflow_coll_i1,ctr_mflow_fw_i1,
           ctr_mflow_tw_i1,ctr_mflow_coll_i2,ctr_mflow_fw_i2,ctr_mflow_tw_i2,
           ctr_mflow_coll_tot,ctr_mflow_fw_tot,ctr_mflow_tw_tot,dEdt_i1,dEdt_i2,
           dEdt_n1,dEdt_n2,eneflow_coll_i1,eneflow_coll_i2,eneflow_coll_n1,
           eneflow_coll_n2,eneflow_fw_i1,eneflow_fw_i2,eneflow_fw_n1,eneflow_fw_n2,
           eneflow_tw_i1,eneflow_tw_i2,eneflow_tw_n1,eneflow_tw_n2,Pfield_i1,
           Pfield_i2,eneflow_inj_i1,eneflow_fwmat_i1,eneflow_inj_i2,
           eneflow_fwmat_i2,eneflow_inj_n1,eneflow_fwmat_n1,eneflow_inj_n2,
           eneflow_fwmat_n2,eneflow_twmat_i1,eneflow_twinf_i1,eneflow_twa_i1,
           eneflow_twmat_i2,eneflow_twinf_i2,eneflow_twa_i2,eneflow_twmat_n1,
           eneflow_twinf_n1,eneflow_twa_n1,eneflow_twmat_n2,eneflow_twinf_n2,
           eneflow_twa_n2,ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,Te_cath,
           nu_cath,ndot_cath,Q_cath,P_cath,V_cath_tot,ne_cath_avg,F_theta,Hall_par,
           Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,
           nu_ex,Boltz,Boltz_dim,Pfield_e,Ebal_e,ge_b,ge_b_acc,ge_sb_b,ge_sb_b_acc,
           delta_see,delta_see_acc,err_interp_n] = HET_sims_read_DMD(path_simstate_inp,path_simstate_out,
                                                                     path_postdata_out,path_simparams_inp,
                                                                     path_picM,allsteps_flag,timestep,read_inst_data,
                                                                     read_part_lists,read_flag,oldpost_sim[k],oldsimparams_sim[k])
        
        # Obtain FFT of different signals -------------------------------------
        # Set initial time to zero
        time = time - time[0]
        time_fast = time_fast - time_fast[0]
        # Domain averaged (spatially averaged) variables using the nodal weighting volumes
        [Te_mean_dom,_] = domain_average(Te,time,vol)
        # Obtain FFT for Te_mean_dom considering an integer number of periods
        [fft_Te_mean_dom,freq_Te_mean_dom,max_fft_Te_mean_dom,max_freq_Te_mean_dom] = comp_FFT(time,Te_mean_dom,time[nsteps-last_steps::],Te_mean_dom[nsteps-last_steps::],order)
        # Obtain FFT for Id considering an integer number of periods
        [fft_Id,freq_Id,max_fft_Id,max_freq_Id] = comp_FFT(time,Id,time[nsteps-last_steps::],Id[nsteps-last_steps::],order)
        [maxs_fft_Id,maxs_freq_Id] = find_firstmax(freq_Id[1:],np.abs(fft_Id[1:]),num_firstmax)
        # Obtain FFT for Id_inst considering an integer number of periods
        [fft_Id_inst,freq_Id_inst,max_fft_Id_inst,max_freq_Id_inst] = comp_FFT(time,Id_inst,time[nsteps-last_steps::],Id_inst[nsteps-last_steps::],order)
        [maxs_fft_Id_inst,maxs_freq_Id_inst] = find_firstmax(freq_Id_inst[1:],np.abs(fft_Id_inst[1:]),num_firstmax)
        # Obtain FFT for I_beam considering an integer number of periods
        [fft_I_beam,freq_I_beam,max_fft_I_beam,max_freq_I_beam] = comp_FFT(time,I_beam,time[nsteps-last_steps::],I_beam[nsteps-last_steps::],order)
        # Obtain FFT for avg_dens_mp_ions considering an integer number of periods
        [fft_avg_dens_mp_ions,freq_avg_dens_mp_ions,max_fft_avg_dens_mp_ions,max_freq_avg_dens_mp_ions] = comp_FFT(time_fast,avg_dens_mp_ions,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_ions[nsteps_fast-last_steps_fast::],order_fast)
        # Obtain FFT for avg_dens_mp_neus considering an integer number of periods
        [fft_avg_dens_mp_neus,freq_avg_dens_mp_neus,max_fft_avg_dens_mp_neus,max_freq_avg_dens_mp_neus] = comp_FFT(time_fast,avg_dens_mp_neus,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_neus[nsteps_fast-last_steps_fast::],order_fast)
        
        
        plt.figure(r'FFT dens_e(t)')
        plt.semilogx(freq_avg_dens_mp_ions[1:], np.abs(fft_avg_dens_mp_ions[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            
        plt.figure(r'FFT dens_n(t)')     
        plt.semilogx(freq_avg_dens_mp_neus[1:], np.abs(fft_avg_dens_mp_neus[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])

        plt.figure(r'FFT Te(t)')
        plt.semilogx(freq_Te_mean_dom[1:], np.abs(fft_Te_mean_dom[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        
        plt.figure(r'FFT Id_inst(t)')
        plt.semilogx(freq_Id_inst[1:], np.abs(fft_Id_inst[1:]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        for i in range(0,len(maxs_freq_Id_inst)): 
            plt.semilogx(maxs_freq_Id_inst[i], np.abs(maxs_fft_Id_inst[i]), linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            plt.text(maxs_freq_Id_inst[i], np.abs(maxs_fft_Id_inst[i]),str(i+1),fontsize = text_size,color='r',ha='center',va='center')

        
        plt.figure(r'FFT Id(t)')
        plt.semilogx(freq_Id[1:], np.abs(fft_Id[1:]), linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_FFT, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
    
        for i in range(0,len(maxs_freq_Id_inst)):
            print("Id inst freq "+str(i+1)+" = "+str(maxs_freq_Id_inst[i]))
        # ---------------------------------------------------------------------
        
        # Perform the DMD analysis --------------------------------------------
        dt = time[1]-time[0]
        time_vec      = time[nsteps-last_steps::]
        time_vec_fast = time_fast[nsteps_fast-last_steps_fast::]
        if var_name == "ne":
            var      = np.copy(ne[:,:,nsteps-last_steps::])
            dims_var = np.shape(var)
        
        if use_PyDMD == 1:
            # ==== Use PyDMD ==================================================
            # Obtain a subset of the data considering only the last steps
    #        nindz = 19 # Only data at plume
            nindz = 0
            data  = [var[:,nindz::,t] for t in range(0,dims_var[2])]
            dim_data = np.shape(data)
#            ne_data = [ne[:,nindz::,t] for t in range(nsteps-last_steps,nsteps)]
#            dim_data = np.shape(ne_data)
            
            # Examinate singular values
            A = np.zeros((dim_data[1]*dim_data[2],dim_data[0]))
            for col in range(0,dim_data[0]):
                A[:,col] = (data[col].flatten()).T
            plt.figure("Singular values")
            sigmas = scipy.linalg.svdvals(A)
            plt.semilogy(sigmas, 'o')
            
            if read_vars_file == 0:
                print("read_vars_file = 0 >> HODMD is going to be performed")
                hodmd = HODMD(svd_rank=nmodes, exact=True, opt=True, d=d_par).fit(data)
                eigs     = hodmd.eigs
                modes    = hodmd.modes
                freqs    = hodmd.frequency
                amps     = hodmd.amplitudes
                dynamics = hodmd.dynamics
                reconst  = hodmd.reconstructed_data
                atilde    = hodmd.atilde
                data_orig = hodmd.snapshots
                if save_vars_file == 1:
                    print("save_vars_file = 1 >> saving HODMD results data file")
                    pickle.dump([eigs,modes,freqs,amps,dynamics,reconst,atilde,data_orig], open(path_out_data+save_vars_file_name+".p", "wb"))
                
#                # Store the time dynamics of all obtained modes
#                tvar_modes_complete = np.zeros((nmodes,dim_data[1]*dim_data[2],dim_data[0]),dtype=complex)
#                for ind_mode in range(0,nmodes):
#                    for ind_time in range(0,dim_data[0]):
#                        tvar_modes_complete[ind_mode,:,ind_time] = hodmd.modes[:,ind_mode]*hodmd.dynamics[ind_mode,ind_time]
#                # Store the eigenvalues of all the obtained modes
#                modes_eigs_complete = hodmd.eigs
#                # Store the frequencies of all the obtained modes (kHz)
#                modes_freq_complete = hodmd.frequency/dt*1E-3
#                # Store the growth rates of all the obtained modes
#                modes_gr_complete   = np.log(np.abs(modes_eigs_complete))/dt
#                # Store the amplitudes of all the obtained modes
#                modes_amp_complete  = hodmd.amplitudes

            elif read_vars_file == 1:
                print("read_vars_file = 1 >> HODMD data file is going to be read")
                eigs, modes, freqs, amps, dynamics, reconst, atilde, data_orig = pickle.load(open(path_out_data+read_vars_file_name+".p","rb"))
            
#            # From inspection, we select first nmodes
#            ntimes = dim_data[0]
#            plt.figure("Nodes evolution")
#            colors = ['k','r','g','b','m','c','y']
#            for ind_modes in range(0,nmodes):
#                plt.plot(hodmd.dmd_timesteps[0:ntimes], hodmd.dynamics[ind_modes,0:ntimes].T.real)
#            ind_modes = 10
#            plt.plot(hodmd.dmd_timesteps[0:ntimes], hodmd.dynamics[ind_modes,0:ntimes].T.real)
            
            # Store the time dynamics of all obtained modes
            tvar_modes_complete = np.zeros((nmodes,dim_data[1]*dim_data[2],dim_data[0]),dtype=complex)
            for ind_mode in range(0,nmodes):
                for ind_time in range(0,dim_data[0]):
                    tvar_modes_complete[ind_mode,:,ind_time] = modes[:,ind_mode]*dynamics[ind_mode,ind_time]
#                    tvar_modes_complete[ind_mode,:,ind_time] = np.abs(modes[:,ind_mode])*dynamics[ind_mode,ind_time]
            # Store the eigenvalues of all the obtained modes
            modes_eigs_complete = eigs
            # Store the frequencies of all the obtained modes (kHz)
            modes_freq_complete = freqs/dt*1E-3
            # Store the growth rates of all the obtained modes
            modes_gr_complete   = np.log(np.abs(modes_eigs_complete))/dt
            # Store the amplitudes of all the obtained modes
            modes_amp_complete  = amps
            # Store the dynamics of all the obtained modes
            modes_dynamics_complete = dynamics
            # Obtain the time evolution of the 2D(z,r) map of each mode
            tvar_modes_complete_zr = np.zeros((nmodes,dim_data[1],dim_data[2],dim_data[0]),dtype=complex)
            for i in range(0,nmodes):
                for ind_time in range(0,dim_data[0]):
                    tvar_modes_complete_zr[i,:,:,ind_time] = tvar_modes_complete[i,:,ind_time].reshape(zs.shape)
            
            # Check data reconstruction using all obtained modes
            data_rec_complete          = np.matmul(modes,modes_dynamics_complete)
            err_rec_approach_complete  = data_rec_complete - reconst
            print("Err reconst approach = "+str(np.max(np.max(np.abs(err_rec_approach_complete)))))
            # Obtain reconstructed data using all modes and error in 2D(z,r) plane and for each timestep
            data_rec_complete_zr       = np.zeros((dim_data[1],dim_data[2],dim_data[0]),dtype=complex)
            err_rec_complete_zr        = np.zeros((dim_data[1],dim_data[2],dim_data[0]),dtype=complex)
            for i in range(0,dim_data[0]):
                data_rec_complete_zr[:,:,i] = reconst[:,i].reshape(zs.shape)
                for i1 in range(0,dim_data[1]):
                    for i2 in range(0,dim_data[2]):
                        if var[i1,i2,i] != 0.0:
                            err_rec_complete_zr[i1,i2,i]  = np.abs(var[i1,i2,i] - np.abs(np.real(data_rec_complete_zr[i1,i2,i])))/var[i1,i2,i]
                        else:
                            err_rec_complete_zr[i1,i2,i]  = np.abs(var[i1,i2,i] - np.abs(np.real(data_rec_complete_zr[i1,i2,i])))
            
            print("Err reconst max = "+str(np.max(np.max(err_rec_complete_zr))))
            
            # Select only stable modes with positive and negative frequency (keep the complex conjugate)
            tvar_modes     = np.zeros((0,dim_data[1]*dim_data[2],dim_data[0]),dtype=complex)
            modes_dynamics = np.zeros((0,dim_data[0]),dtype=complex)
            modes_freq     = np.zeros(0,dtype=float)
            modes_gr       = np.zeros(0,dtype=float)
            modes_amp      = np.zeros(0,dtype=complex)
            modes_ind      = np.zeros(0,dtype=int)
            n_sel_modes = 0
            for i in range(0,nmodes):
                if ( (modes_freq_complete[i] >= fr_tol or modes_freq_complete[i] <= -fr_tol) and (np.abs(np.abs(modes_eigs_complete[i])-1.0) <= 1E-3) and (np.abs(modes_gr_complete[i]) < gr_tol) ):
                    modes_freq = np.append(modes_freq,np.array([modes_freq_complete[i]]),axis=0)
                    modes_gr   = np.append(modes_gr,np.array([modes_gr_complete[i]]),axis=0)
                    modes_amp  = np.append(modes_amp,np.array([modes_amp_complete[i]]),axis=0)
                    modes_ind  = np.append(modes_ind,np.array([i]),axis=0)
                    tvar_modes = np.append(tvar_modes,np.zeros([1,dim_data[1]*dim_data[2],dim_data[0]]),axis=0)
                    tvar_modes[n_sel_modes,:,:] = tvar_modes_complete[i,:,:]
                    modes_dynamics = np.append(modes_dynamics,np.zeros([1,dim_data[0]]),axis=0)
                    modes_dynamics[n_sel_modes,:] = modes_dynamics_complete[i,:]
#                    print(np.shape(tvar_modes))
                    n_sel_modes = n_sel_modes + 1
            modes_sel = modes[:,modes_ind]
            
            # Obtain the time evolution of the 2D(z,r) map of each selected mode
            tvar_modes_zr = np.zeros((n_sel_modes,dim_data[1],dim_data[2],dim_data[0]),dtype=complex)
            for i in range(0,n_sel_modes):
                for ind_time in range(0,dim_data[0]):
                    tvar_modes_zr[i,:,:,ind_time] = tvar_modes[i,:,ind_time].reshape(zs.shape)
                    
            # Reconstruct data using only selected modes modes
            data_rec = np.matmul(modes_sel,modes_dynamics)
            err_rec  = data_rec - reconst
            print("Err reconst sel = "+str(np.max(np.max(np.abs(err_rec)))))

             # Obtain reconstructed data only selected modes and error in 2D(z,r) plane and for each timestep
            data_rec_zr       = np.zeros((dim_data[1],dim_data[2],dim_data[0]),dtype=complex)
            err_rec_zr        = np.zeros((dim_data[1],dim_data[2],dim_data[0]),dtype=complex)
            for i in range(0,dim_data[0]):
                data_rec_zr[:,:,i] = data_rec[:,i].reshape(zs.shape)
                for i1 in range(0,dim_data[1]):
                    for i2 in range(0,dim_data[2]):
                        if var[i1,i2,i] != 0.0:
                            err_rec_zr[i1,i2,i]  = np.abs(var[i1,i2,i] - np.abs(np.real(data_rec_zr[i1,i2,i])))/var[i1,i2,i]
                        else:
                            err_rec_zr[i1,i2,i]  = np.abs(var[i1,i2,i] - np.abs(np.real(data_rec_zr[i1,i2,i])))
            
            print("Err reconst max sel = "+str(np.max(np.max(err_rec_zr))))
            
            # Get the number of modes without accounting for the complex conjugate ones
            n_sel_modes1 = n_sel_modes/2
            modes_ind1   = np.where(modes_freq > 0)[0]
            
            # =================================================================
        
        if use_dmd_dom == 1:
            last_steps = 3000
            time_vec = time[nsteps-last_steps::]
            nmodes = 20
            # ==== Use function dmd_dom =======================================
            Big_X = np.zeros((nsteps,dims[0],dims[1]))
            for i in range(0,nsteps):
                Big_X[i,:,:] = ne[:,:,i]
    #        Big_X = np.reshape(ne,(nsteps,dims[0],dims[1]))
            Big_X = Big_X[nsteps-last_steps::,:,:]
            dim_data = np.shape(Big_X)
            
            [Eigenvalues, Eigenvectors, ModeAmplitudes, ModeFrequencies, GrowthRates, POD_Mode_Energies] = dmd_rom(Big_X, nmodes, dt)
            Mode_freqs = ModeFrequencies/(2.0*np.pi)
            
            # Obtain the time dynamics of the modes
            time_dynamics = np.zeros((nmodes,dim_data[0]),dtype=complex)
            for i in range(0,dim_data[0]):
                time_dynamics[:,i] = ModeAmplitudes*np.exp((GrowthRates + ModeFrequencies*1j)*time_vec[i])
        
            # =================================================================
        
        if use_dmd == 1:
            last_steps = 3000
            time_vec = time[nsteps-last_steps::]
            nmodes = 20
            # ==== Use function dmd ===========================================
#            dims = np.shape(Big_X)
#            # Reshapes Big_X
#            if np.any(np.iscomplex(Big_X)):
#                Big_Xdata = np.zeros((dims[0],np.prod(dims[1::])),dtype=complex)
#            else:        
#                Big_Xdata = np.zeros((dims[0],np.prod(dims[1::])))
#            for i in range(0,dims[0]):
#                if len(dims) == 2:
#                    # 1D problems
#                    Big_Xdata[i,:] = np.reshape(Big_X[i,:],(1,np.prod(dims[1::])),order='F') 
#                elif len(dims) == 3:
#                    # 2D problems
#                    Big_Xdata[i,:] = np.reshape(Big_X[i,:,:],(1,np.prod(dims[1::])),order='F') 
#            Big_X = np.transpose(Big_Xdata)
            
#            var      = np.copy(ne)
            var      = np.copy(ne[:,:,nsteps-last_steps::])
            dims_var = np.shape(var)
            data     = np.zeros((dims_var[0]*dims_var[1],dims_var[2]),dtype=float)
            
            for i in range(0,dims_var[2]):
    #            data[:,i] = np.reshape(var[:,:,i],(np.prod(dims_var[0:2]),1),order='F') 
                data[:,i] = np.reshape(var[:,:,i],(np.prod(dims_var[0:2]),1),order='F')[:,0]
            # Split Big_X into two snapshot sets
            X = data[:,0:-1]
            Y = data[:,1::]
            
    #        [mu,Phi] = dmd(X, Y, truncate=None)
            [mu,Phi] = dmd(X, Y, truncate=20)
            
            dmd_om = np.angle(mu)/dt 
            dmd_fr = np.angle(mu)/(2.0*np.pi*dt)
            dmd_gr = np.log(np.abs(mu))/dt
            index = 0
            dmd_am = np.matmul(np.linalg.pinv(Phi),X[:,index])
            
            # Obtain the time dynamics of the modes
            dmd_time_dynamics = np.zeros((nmodes,dims_var[2]),dtype=complex)
            for i in range(0,dims_var[2]):
                dmd_time_dynamics[:,i] = dmd_am*np.exp((dmd_gr + dmd_om*1j)*time_vec[i])
            # =================================================================
        
        
        # Plot the time dynamics of the modes
        font_size_legend = 10
        [fig, axes] = plt.subplots(nrows=3, ncols=1, figsize=(15,12))
        ax1 = plt.subplot2grid( (3,1), (0,0) )
        ax2 = plt.subplot2grid( (3,1), (1,0) )
        ax3 = plt.subplot2grid( (3,1), (2,0) )
        ax1.set_title("MyDMD")
        ax2.set_title("PyDMD")
        ax3.set_title(r"$I_d$ (A)")
        ax3.set_xlabel(r"$t$ (ms)")
        for i in range(0,n_sel_modes):
#        for i in range(1,15):
            if modes_freq[i] > 0 and modes_freq[i] < 20:
                print(modes_freq[i])
    #            print(i,np.nanmax(np.real(time_dynamics[i,:])))
    #            ax1.plot(time_vec*1E3,np.real(dmd_time_dynamics[i,:]),label=str(i+1)+" "+str(np.abs(1E-3*dmd_fr[i])))
    #            ax2.plot(time_vec*1E3,np.real(time_dynamics[i,:]),label=str(i+1)+" "+str(np.abs(1E-3*Mode_freqs[i])))
                ax1.plot(time_vec*1E3,1.0+modes_dynamics[i,:]/np.abs(modes_amp[i]),label="M"+str(i+1)+" f="+str(np.abs(modes_freq[i])))
                ax1.plot(time_vec*1E3,1.0+np.real(np.exp(2.0*np.pi*modes_freq[i]*1j*time_vec*1E3)),linestyle='--',label="M"+str(i+1)+" f="+str(np.abs(modes_freq[i])))
    #            ax1.plot(time_vec*1E3,np.real(time_dynamics[i,:]),label=str(i+1)+" "+str(np.abs(1E-3*Mode_freqs[i])))
    #            ax2.plot(time_vec*1E3,fbdmd.dynamics[i,:],label=r"Mode "+str(i+1)+" "+str(np.abs(1E-3*modes_freq_complete[i])))
    #            ax1.plot(time_vec*1E3,np.real(time_dynamics[i,:])/np.nanmax(np.real(time_dynamics[i,:])))
    #            ax2.plot(time_vec*1E3,fbdmd.dynamics[i,:]/np.nanmax(fbdmd.dynamics[i,:]))
        ax1.plot(time_vec*1E3,2.0*Id_inst[nsteps-last_steps::]/np.max(Id_inst[nsteps-last_steps::]),linestyle='--',color='k',linewidth=2,label=r"$I_d$ f="+str(max_freq_Id_inst*1E-3))
        ax1.plot(time_vec_fast*1E3,2.0*avg_dens_mp_ions[nsteps_fast-last_steps_fast::]/np.max(avg_dens_mp_ions[nsteps_fast-last_steps_fast::]),linestyle='--',color='r',linewidth=2,label=r"$\bar{n}_e$ f="+str(max_freq_avg_dens_mp_ions*1E-3))
        ax1.legend(fontsize=font_size_legend,loc=1)
        ax3.legend(fontsize=font_size_legend,loc=1)
        ax2.legend(fontsize=font_size_legend,loc=1)

        
        
        # Prepare data for plotting
        zs = zs*1E2
        rs = rs*1E2
        z_cath = z_cath*1E2
        r_cath = r_cath*1E2
        points = points*1E2
        time = time*1E3
        time_fast = time_fast*1E3
        
        rind_point = rind
        zind_point = 15
        
        # Take index of mode with positive frequency (complex conjugate is index + 1) from lower to higher frequency
        mode_index = np.zeros(n_sel_modes1,dtype=int)
        copy_modes_freq = np.copy(modes_freq)
        
        for i in range(0,n_sel_modes1):
            min_freq = np.nanmin(np.abs(copy_modes_freq))
            mode_index[i] = np.where(copy_modes_freq == min_freq)[0]
            copy_modes_freq[mode_index[i]] = np.nan
            copy_modes_freq[mode_index[i]+1] = np.nan
            
        ne_plot                   = np.copy(ne)
        Te_plot                   = np.copy(Te)
        var_plot                  = np.copy(var)
        data_rec_complete_zr_plot = np.copy(data_rec_complete_zr)
        data_rec_zr_plot          = np.copy(data_rec_zr)
        tvar_modes_zr_plot        = np.copy(tvar_modes_zr)
        err_rec_complete_zr_plot  = np.copy(err_rec_complete_zr)
        err_rec_zr_plot           = np.copy(err_rec_zr)
        var_plot[np.where(nodes_flag == 0)]                  = np.nan
        ne_plot[np.where(nodes_flag == 0)]                   = np.nan
        Te_plot[np.where(nodes_flag == 0)]                   = np.nan
        data_rec_complete_zr_plot[np.where(nodes_flag == 0)] = np.nan + np.nan * 0j
        data_rec_zr_plot[np.where(nodes_flag == 0)]          = np.nan + np.nan * 0j
        err_rec_complete_zr_plot[np.where(nodes_flag == 0)]  = np.nan + np.nan * 0j
        err_rec_zr_plot[np.where(nodes_flag == 0)]           = np.nan + np.nan * 0j
        for i in range(0,n_sel_modes):
            tvar_modes_zr_plot_plot = tvar_modes_zr_plot[i,:,:,:]
            tvar_modes_zr_plot_plot[np.where(nodes_flag == 0)] = np.nan
            tvar_modes_zr_plot[i,:,:,:] = tvar_modes_zr_plot_plot

        
        first_step = nsteps - last_steps
#        final_step = len(Id)
        final_step = first_step + 2000
        first_step_fast = first_step*20
        final_step_fast = final_step*20
        index = 0
        i_rec = 1
        if video_type == 1:
            for ii in range(first_step,final_step):
#            for ii in range(first_step,first_step + 1):
                index = index + 1
                i_fast = 20*ii
#                i_rec = ii -last_steps
#                i_rec = ii

#                [fig, axes] = plt.subplots(nrows=2, ncols=3 ,figsize=(22.5,12))
#                ax1 = plt.subplot2grid( (2,3), (0,0) )
#                ax2 = plt.subplot2grid( (2,3), (1,0) )
#                ax3 = plt.subplot2grid( (2,3), (0,1) )
#                ax4 = plt.subplot2grid( (2,3), (1,1) )
#                ax5 = plt.subplot2grid( (2,3), (0,2) )
#                ax6 = plt.subplot2grid( (2,3), (1,2) )
                
                [fig, axes] = plt.subplots(nrows=4, ncols=4 ,figsize=(22.5,18))
                ax1 = plt.subplot2grid( (4,4), (0,0) )
                ax2 = plt.subplot2grid( (4,4), (1,0) )
                ax3 = plt.subplot2grid( (4,4), (2,0) )
                ax4 = plt.subplot2grid( (4,4), (3,0) )
                
                ax5 = plt.subplot2grid( (4,4), (0,1) )
                ax6 = plt.subplot2grid( (4,4), (1,1) )
                ax7 = plt.subplot2grid( (4,4), (2,1) )
                ax8 = plt.subplot2grid( (4,4), (3,1) )
                
                ax9  = plt.subplot2grid( (4,4), (0,2) )
                ax10 = plt.subplot2grid( (4,4), (1,2) )
                ax11 = plt.subplot2grid( (4,4), (2,2) )
                ax12 = plt.subplot2grid( (4,4), (3,2) )
                
                ax13 = plt.subplot2grid( (4,4), (0,3) )
                ax14 = plt.subplot2grid( (4,4), (1,3) )
                ax15 = plt.subplot2grid( (4,4), (2,3) )
                ax16 = plt.subplot2grid( (4,4), (3,3) )

            
            
                ax1.set_title(r"$I_d$ (A)",fontsize = font_size, y=1.02)
                ax1.set_xlabel(r'$t$ (ms)', fontsize = font_size)
                ax1.tick_params(labelsize = ticks_size) 
                ax1.tick_params(axis='x', which='major', pad=10)
                
                ax2.set_title(r"$\bar{n}_e$  (m$^{-3}$)",fontsize = font_size, y=1.02)
                ax2.set_xlabel(r'$t$ (ms)', fontsize = font_size)
                ax2.tick_params(labelsize = ticks_size) 
                ax2.tick_params(axis='x', which='major', pad=10)
                
                ax3.set_title(r"$\bar{n}_n$ (m$^{-3}$)",fontsize = font_size, y=1.02)
                ax3.set_xlabel(r'$t$ (ms)', fontsize = font_size)
                ax3.tick_params(labelsize = ticks_size) 
                ax3.tick_params(axis='x', which='major', pad=10)
                
                ax4.set_title(r"$\bar{T}_e$ (eV)",fontsize = font_size, y=1.02)
                ax4.set_xlabel(r'$t$ (ms)', fontsize = font_size)
                ax4.tick_params(labelsize = ticks_size) 
                ax4.tick_params(axis='x', which='major', pad=10)
                    
        
                    
                # Plot the time evolution of the discharge current
                ax1.semilogy(time[first_step:final_step], Id[first_step:final_step], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                ylims = ax1.get_ylim()  
#                marker_size_ID = 10
#                fact_x = np.array([0.97,1.03,1.0,0.98])
#                for i in range(0,len(prntstep_IDs)):
#                    ax1.semilogy(time[prntstep_IDs[i]]*np.ones(2), np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker="", color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#                    ax1.semilogy(time[prntstep_IDs[i]], Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#            #        ax1.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
#                    ax1.text(fact_x[i]*time[prntstep_IDs[i]-plot_tol], fact_y[i]*Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')                
#                    ax1.semilogy(time, Id_mean*np.ones(np.shape(time)), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")            
                ax1.semilogy(time[ii], Id[ii], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_moving, marker="o", color='r', markeredgecolor = 'k', label="")
                ax1.set_xlim([time[first_step],time[final_step]])
                
                # Plot the time evolution of the average plasma density in the domain
                ax2.semilogy(time_fast[first_step_fast:final_step_fast], avg_dens_mp_ions[first_step_fast:final_step_fast], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label=labels[0])            
                ax2.semilogy(time[first_step:final_step], ne[rind_point,zind_point,first_step:final_step], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="", color='b', markeredgecolor = 'k', label=labels[0])            
                ax2.set_xlim([time[first_step],time[final_step]])
                ax2.set_ylim(1E16,7E18)
                ylims = ax2.get_ylim()
#                fact_x = np.array([0.97,1.02,1.02,1.02])
#                marker_size_ID = 6
#                for i in range(0,len(fast_prntstep_IDs)):
#                    ax2.semilogy(time_fast[fast_prntstep_IDs[i]]*np.ones(2), np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker="", color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#                    ax2.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_ions[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#                    ax2.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_neus[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#                    ax2.text(fact_x[i]*time_fast[fast_prntstep_IDs[i]-plot_tol], 1.5*ylims[0],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')     
#            #        print(prntstep_IDs_text[i]+" time_fast = "+str(time_fast[fast_prntstep_IDs[i]])+", time = "+str(time[prntstep_IDs[i]]))
                ax2.semilogy(time_fast[i_fast], avg_dens_mp_ions[i_fast], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size_moving, marker="o", color='r', markeredgecolor = 'k', label="")            
                ax2.semilogy(time[ii], ne[rind_point,zind_point,ii], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size_moving, marker="o", color='r', markeredgecolor = 'k', label=labels[0])            

#                ax2.legend(fontsize = font_size,loc=2,ncol=2)
                
                # Plot the time evolution of the average neutral density in the domain
                ax3.semilogy(time_fast[first_step_fast:final_step_fast], avg_dens_mp_neus[first_step_fast:final_step_fast], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label=labels[1])            
                ax3.semilogy(time[first_step:final_step], nn1[rind_point,zind_point,first_step:final_step], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="", color='b', markeredgecolor = 'k', label=labels[0])            
                ax3.set_xlim([time[first_step],time[final_step]])
                ax3.set_ylim(4E17,1E19)
                ylims = ax3.get_ylim()
                ax3.semilogy(time_fast[i_fast], avg_dens_mp_neus[i_fast], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size_moving, marker="o", color='r', markeredgecolor = 'k', label="")            
                ax3.semilogy(time[ii], nn1[rind_point,zind_point,ii], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size_moving, marker="o", color='r', markeredgecolor = 'k', label=labels[0])            
#                ax3.legend(fontsize = font_size,loc=2,ncol=2)
                
                # Plot the time evolution of the average electron temperature in the domain
                ax4.plot(time[first_step:final_step], Te_mean_dom[first_step:final_step], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
                ax4.plot(time[first_step:final_step], Te[rind_point,zind_point,first_step:final_step], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="", color='b', markeredgecolor = 'k', label=labels[0])            
                ax4.set_xlim([time[first_step],time[final_step]])
                ax4.set_ylim(0.8*np.min(Te[rind_point,zind_point,first_step:final_step]),1.2*np.max(Te[rind_point,zind_point,first_step:final_step]))
                ylims = ax4.get_ylim()
                ax4.plot(time[ii], Te_mean_dom[ii], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size_moving, marker="o", color='r', markeredgecolor = 'k', label="")            
                ax4.plot(time[ii], Te[rind_point,zind_point,ii], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size_moving, marker="o", color='r', markeredgecolor = 'k', label=labels[0])            
#                ax4.legend(fontsize = font_size,loc=2,ncol=2)
                
                # Plot the plasma density contour
                ax5.set_title(r'$n_{e}$ (m$^{-3}$)', fontsize = font_size,y=1.02)
                log_type         = 1
                auto             = 0
                min_val0         = 1E14
                max_val0         = 1E19
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 1 
                auto_lines_ticks = -1
                nticks_cbar      = 4
                nticks_lines     = 4
                cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
                lines_ticks      = np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16])
                lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
            #    lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.0f}'
                lines_ticks_fmt   = '{%.1f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
#                var2plot = ne_plot[:,:,ii]
                var2plot = var_plot[:,:,i_rec]
                [CS,CS2] = contour_2D (ax5,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, var2plot, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
            #    # Isolines ticks (exponent)
            #    ax5.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax5.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax5.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            #    ax5.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
                ax5.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                ax5.plot(zs[rind_point,zind_point],rs[rind_point,zind_point],'ko',linewidth = line_width_boundary,markersize = marker_size)

                    
                # Plot the reconstructed plasma density contour
                ax9.set_title(r'Rec. $n_e$ (m$^{-3}$)', fontsize = font_size,y=1.02)
                log_type         = 1
                auto             = 0
#                log_type         = 0
#                auto             = 1
#                min_val0         = 1E-5
#                max_val0         = 1E2
                min_val0         = 1E14
                max_val0         = 1E19
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 1 
                auto_lines_ticks = -1
                nticks_cbar      = 4
                nticks_lines     = 4
                cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
                lines_ticks      = np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16])
                lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
            #    lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.2f}'
                lines_ticks_fmt   = '{%.1f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
                var2plot = np.abs(np.real(data_rec_complete_zr_plot[:,:,i_rec]))
#                var2plot = np.abs(np.real(data_rec_plot_1[:,:,i_rec]))
#                var2plot = np.abs(np.real(err_rec_complete_zr_plot[:,:,i_rec]))
#                var2plot = np.abs(np.real(err_rec_zr_plot[:,:,i_rec]))
                [CS,CS2] = contour_2D (ax9,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, var2plot, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
            #    # Isolines ticks (exponent)
            #    ax9.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax9.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax9.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            #    ax9.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
                ax9.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                
                
                # Plot the error in reconstructed plasma density contour
                ax13.set_title(r'Rec. error $\epsilon$ (-)', fontsize = font_size,y=1.02)
                log_type         = 0
                auto             = 0
                min_val0         = 0.0
                max_val0         = 1.0
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 0 
                auto_lines_ticks = -1
                nticks_cbar      = 8
                nticks_lines     = 4
                cbar_ticks       = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
                cbar_ticks       = np.array([0.0,0.2,0.4,0.6,0.8,1.0])
                lines_ticks      = np.sort(np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16]))
                lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
            #    lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.1f}'
                lines_ticks_fmt   = '{%.1f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
#                var2plot = np.abs(np.real(data_rec_plot[:,:,i_rec]))
#                var2plot = np.abs(np.real(data_rec_plot_1[:,:,i_rec]))
                var2plot = np.abs(np.real(err_rec_complete_zr_plot[:,:,i_rec]))
#                var2plot = np.abs(np.real(err_rec_zr_plot[:,:,i_rec]))
                [CS,CS2] = contour_2D (ax13,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, var2plot, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
            #    # Isolines ticks (exponent)
            #    ax13.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax13.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax13.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            #    ax13.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
                ax13.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                
                
                # Plot the mode index 1 contour (breathing mode) Add complex conjugate
                mind = 0
                text  = r'M'+str(mind+1)+', f = {:.2f}'.format(modes_freq[mode_index[mind]])+' (kHz)'
                ax6.set_title(text, fontsize = font_size,y=1.02)
                log_type         = 1
                auto             = 0
                min_val0         = 1E14
                max_val0         = 1E19
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 1 
                auto_lines_ticks = -1
                nticks_cbar      = 4
                nticks_lines     = 4
                cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
                lines_ticks      = np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16])
                lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
            #    lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.2f}'
                lines_ticks_fmt   = '{%.1f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
#                var2plot = np.abs(np.real(data_rec_plot_1[:,:,i_rec]))
#                var2plot = np.abs(np.real(data_rec_modes_plot[mode_index1,:,:,i_rec]))
                var2plot = np.abs(np.real(tvar_modes_zr_plot[mode_index[mind],:,:,i_rec] + tvar_modes_zr_plot[mode_index[mind]+1,:,:,i_rec]))
#                var2plot = np.abs(np.real(np.sum(tvar_modes_zr_plot[:,:,:,i_rec],axis=0)))
#                var2plot = np.abs(np.real(data_rec_modes_plot[mode_index2,:,:,i_rec]))
                [CS,CS2] = contour_2D (ax6,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, var2plot, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
            #    # Isolines ticks (exponent)
            #    ax7.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax6.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax6.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            #    ax7.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
                ax6.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                
                # Plot the mode index 2 contour. Add complex conjugate
                mind = 1
                text  = r'M'+str(mind+1)+', f = {:.2f}'.format(modes_freq[mode_index[mind]])+' (kHz)'
                ax10.set_title(text, fontsize = font_size,y=1.02)
                log_type         = 1
                auto             = 0
                min_val0         = 1E14
                max_val0         = 1E19
#                min_val0         = 1E10
#                max_val0         = 1E16
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 1 
                auto_lines_ticks = -1
                nticks_cbar      = 4
                nticks_lines     = 4
                cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
                lines_ticks      = np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16])
                lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
            #    lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.2f}'
                lines_ticks_fmt   = '{%.1f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
                var2plot = np.abs(np.real(tvar_modes_zr_plot[mode_index[mind],:,:,i_rec] + tvar_modes_zr_plot[mode_index[mind]+1,:,:,i_rec]))
                [CS,CS2] = contour_2D (ax10,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, var2plot, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
            #    # Isolines ticks (exponent)
            #    ax10.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax10.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax10.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            #    ax10.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
                ax10.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                
                
                # Plot the mode index 3 contour. Add complex conjugate
                mind = 2
                text  = r'M'+str(mind+1)+', f = {:.2f}'.format(modes_freq[mode_index[mind]])+' (kHz)'
                ax14.set_title(text, fontsize = font_size,y=1.02)
                log_type         = 1
                auto             = 0
                min_val0         = 1E14
                max_val0         = 1E19
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 1 
                auto_lines_ticks = -1
                nticks_cbar      = 4
                nticks_lines     = 4
                cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
                lines_ticks      = np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16])
                lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
            #    lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.2f}'
                lines_ticks_fmt   = '{%.1f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
                var2plot = np.abs(np.real(tvar_modes_zr_plot[mode_index[mind],:,:,i_rec] + tvar_modes_zr_plot[mode_index[mind]+1,:,:,i_rec]))
                [CS,CS2] = contour_2D (ax14,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, var2plot, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
            #    # Isolines ticks (exponent)
            #    ax14.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax14.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax14.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            #    ax14.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
                ax14.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                
                
                # Plot the mode index 4 contour. Add complex conjugate
                mind = 3
                text  = r'M'+str(mind+1)+', f = {:.2f}'.format(modes_freq[mode_index[mind]])+' (kHz)'
                ax7.set_title(text, fontsize = font_size,y=1.02)
                log_type         = 1
                auto             = 0
                min_val0         = 1E12
                max_val0         = 1E17
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 1 
                auto_lines_ticks = -1
                nticks_cbar      = 4
                nticks_lines     = 4
                cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
                lines_ticks      = np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16])
                lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
            #    lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.2f}'
                lines_ticks_fmt   = '{%.1f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
                var2plot = np.abs(np.real(tvar_modes_zr_plot[mode_index[mind],:,:,i_rec] + tvar_modes_zr_plot[mode_index[mind]+1,:,:,i_rec]))
                [CS,CS2] = contour_2D (ax7,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, var2plot, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
            #    # Isolines ticks (exponent)
            #    ax7.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax7.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax7.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            #    ax7.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
                ax7.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                
                
                # Plot the mode index 5 contour. Add complex conjugate
                mind = 4
                text  = r'M'+str(mind+1)+', f = {:.2f}'.format(modes_freq[mode_index[mind]])+' (kHz)'
                ax11.set_title(text, fontsize = font_size,y=1.02)
                log_type         = 1
                auto             = 0
                min_val0         = 1E12
                max_val0         = 1E17
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 1 
                auto_lines_ticks = -1
                nticks_cbar      = 4
                nticks_lines     = 4
                cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
                lines_ticks      = np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16])
                lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
            #    lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.2f}'
                lines_ticks_fmt   = '{%.1f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
                var2plot = np.abs(np.real(tvar_modes_zr_plot[mode_index[mind],:,:,i_rec] + tvar_modes_zr_plot[mode_index[mind]+1,:,:,i_rec]))
                [CS,CS2] = contour_2D (ax11,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, var2plot, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
            #    # Isolines ticks (exponent)
            #    ax11.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax11.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax11.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            #    ax11.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
                ax11.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                
                
                # Plot the mode index 6 contour. Add complex conjugate
                mind = 5
                text  = r'M'+str(mind+1)+', f = {:.2f}'.format(modes_freq[mode_index[mind]])+' (kHz)'
                ax15.set_title(text, fontsize = font_size,y=1.02)
                log_type         = 1
                auto             = 0
                min_val0         = 1E12
                max_val0         = 1E17
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 1 
                auto_lines_ticks = -1
                nticks_cbar      = 4
                nticks_lines     = 4
                cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
                lines_ticks      = np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16])
                lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
            #    lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.2f}'
                lines_ticks_fmt   = '{%.1f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
                var2plot = np.abs(np.real(tvar_modes_zr_plot[mode_index[mind],:,:,i_rec] + tvar_modes_zr_plot[mode_index[mind]+1,:,:,i_rec]))
                [CS,CS2] = contour_2D (ax15,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, var2plot, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
            #    # Isolines ticks (exponent)
            #    ax15.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax15.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax15.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            #    ax15.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
                ax15.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                
                
                # Plot the mode index 7 contour. Add complex conjugate
                mind = 6
                text  = r'M'+str(mind+1)+', f = {:.2f}'.format(modes_freq[mode_index[mind]])+' (kHz)'
                ax8.set_title(text, fontsize = font_size,y=1.02)
                log_type         = 1
                auto             = 0
                min_val0         = 1E12
                max_val0         = 1E17
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 1 
                auto_lines_ticks = -1
                nticks_cbar      = 4
                nticks_lines     = 4
                cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
                lines_ticks      = np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16])
                lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
            #    lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.2f}'
                lines_ticks_fmt   = '{%.1f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
                var2plot = np.abs(np.real(tvar_modes_zr_plot[mode_index[mind],:,:,i_rec] + tvar_modes_zr_plot[mode_index[mind]+1,:,:,i_rec]))
                [CS,CS2] = contour_2D (ax8,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, var2plot, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
            #    # Isolines ticks (exponent)
            #    ax8.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax8.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax8.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            #    ax8.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
                ax8.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                
                
                # Plot the mode index 8 contour. Add complex conjugate
                mind = 7
                text  = r'M'+str(mind+1)+', f = {:.2f}'.format(modes_freq[mode_index[mind]])+' (kHz)'
                ax12.set_title(text, fontsize = font_size,y=1.02)
                log_type         = 1
                auto             = 0
                min_val0         = 1E12
                max_val0         = 1E17
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 1 
                auto_lines_ticks = -1
                nticks_cbar      = 4
                nticks_lines     = 4
                cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
                lines_ticks      = np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16])
                lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
            #    lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.2f}'
                lines_ticks_fmt   = '{%.1f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
                var2plot = np.abs(np.real(tvar_modes_zr_plot[mode_index[mind],:,:,i_rec] + tvar_modes_zr_plot[mode_index[mind]+1,:,:,i_rec]))
                [CS,CS2] = contour_2D (ax12,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, var2plot, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
            #    # Isolines ticks (exponent)
            #    ax12.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax12.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax12.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            #    ax12.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
                ax12.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                
                
                # Plot the electron temperature contour
                ax16.set_title(r'$T_e$ (eV)', fontsize = font_size,y=1.02)
                log_type         = 0
                auto             = 0
                min_val0         = 0.0
                max_val0         = 45.0
                cont             = 1
                lines            = 0
                cont_nlevels     = nlevels_cont
                auto_cbar_ticks  = 0 
                auto_lines_ticks = -1
                nticks_cbar      = 5
                nticks_lines     = 10
                cbar_ticks       = np.array([45,40,35,30,25,20,15,10,5,0])
            #        lines_ticks      = np.array([7,9,12,20,25,30,35,40,45])
                lines_ticks      = np.array([7,9,12,20,30,35,40,45])
                lines_ticks_loc  = [(0.38,4.25),(0.88,4.25),(1.5,4.25),(2.7,4.6),(3.0,3.8),(3.6,4.8),(3.9,4.25),(4.5,4.25),(5.18,4.0),(5.3,3.2),(5.6,1.8),(3.7,6.8)]
            #        lines_ticks_loc  = 'default'
                cbar_ticks_fmt    = '{%.0f}'
                lines_ticks_fmt   = '{%.0f}'
                lines_width       = line_width
                lines_ticks_color = 'k'
                lines_style       = '-'
                [CS,CS2] = contour_2D (ax16,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, Te_plot[:,:,ii], nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width) 
            #    ax16.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                ax16.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
                ax16.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)          
            #    ax16.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)        
                ax16.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                ax16.plot(zs[rind_point,zind_point],rs[rind_point,zind_point],'ko',linewidth = line_width_boundary,markersize = marker_size)
                
                
                i_rec = i_rec + 1
                
                plt.tight_layout()
                if save_flag == 1:
                    fig.savefig(path_out+str(index)+figs_format,bbox_inches='tight') 
                    plt.close()  
                    
        ind = ind + 1
        if ind > 8:
            ind = 0
            ind2 = ind2 + 1
            if ind2 > 6:
                ind = 0
                ind2 = 0
                ind3 = ind3 + 1
        
###        tvar_mode = np.matmul(fbdmd.modes[:,0],fbdmd.dynamics[0,:].reshape((700,0)))
#        tvar_modes = np.zeros((nmodes,dim_data[1]*dim_data[2],dim_data[0]))
#        for ind_mode in range(0,nmodes):
#            for ind_time in range(0,dim_data[0]):
#                tvar_modes[ind_mode,:,ind_time] = fbdmd.modes[:,ind_mode]*fbdmd.dynamics[ind_mode,ind_time]
        
        
#        # Create video for the time evolution of each mode
#        FFMpegWriter = animation.writers['ffmpeg']
#        metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
#        # Change the video bitrate as you like and add some metadata.
#        writer = FFMpegWriter(fps=50, bitrate=1800, metadata=metadata)
#        # HOW TO IMPROVE VIDEO QUALITY:
#        # Increase dphi in figure and in save
#        # Increase bitrate in writer
#        # Keep low figsize (2,2) in this case
#        
#        plot_mode = 6
#        
#        # Approach 1: using pcolor --------------------------------------------
#        #fig = plt.figure(figsize=(8,6),dpi=900)
##        fig = plt.figure(figsize=(2,2),dpi=900)
#        fig = plt.figure(figsize=(2,2),dpi = 200)
#        
#        dmd_states = [tvar_modes[plot_mode,:,state].reshape(zs.shape) for state in range(0,dim_data[0])]
#            
#        frames = [[plt.pcolor(zs, rs, state.real)] for state in dmd_states]
##        frames = [[plt.contourf(zs, rs, state.real)] for state in dmd_states]
#
#        ani = animation.ArtistAnimation(fig, frames, interval=70, blit=False, repeat=False)
#        # ---------------------------------------------------------------------
#        
#        # Save the video
#        #ani.save("movie.mp4", writer=writer,dpi=900)
#        ani.save("DMD_mode"+str(plot_mode)+".mp4", writer=writer,dpi=200)
#        
#
#        # Approach 2: using contourf ------------------------------------------
##        fig = plt.figure(figsize=(2,2),dpi = 500)
##                
##        frames = []
##        for ind_time in range(0,dim_data[0]):
##            state = tvar_modes[plot_mode,:,ind_time].reshape(zs.shape)
##            im = plt.contourf(zs, rs, state.real)
##            
##            #################################################################
##            ## Bug fix for Quad Contour set not having attribute 'set_visible'
##            def setvisible(self,vis):
##                for c in self.collections: c.set_visible(vis)
##            im.set_visible = types.MethodType(setvisible,im)
##            im.axes = plt.gca()
##            im.figure=fig
##            ####################################################################
##            
##            frames.append([im])
##            
##        ani = animation.ArtistAnimation(fig, frames, interval=70, blit=False, repeat=False)
#        # ---------------------------------------------------------------------
#        
#
#        
#        
##        fig,ax = plt.subplots()
##        DATA = np.zeros((dim_data[1],dim_data[2],dim_data[0]))
##        for ind_time in range(0,dim_data[0]):
##            DATA[:,:,ind_time] = tvar_modes[plot_mode,:,ind_time].reshape(zs.shape)
##
##        def animate(i):
##              ax.clear()
##              ax.contourf(DATA[:,:,i])
##              ax.set_title('%03d'%(i)) 
##              return ax
##
##        interval = 2#in seconds     
##        ani = animation.FuncAnimation(fig,animate,5,interval=interval*1e+3,blit=False)
#        
#        
##        # Save the video
##        #ani.save("movie.mp4", writer=writer,dpi=900)
##        ani.save("DMD_mode"+str(plot_mode)+".mp4", writer=writer,dpi=200)
##        