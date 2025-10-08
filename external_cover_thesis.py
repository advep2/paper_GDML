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
from contour_2D_mod import contour_2D
from streamlines_function import streamplot, streamline_2D
from HET_sims_read import HET_sims_read
from HET_sims_mean import HET_sims_mean
from HET_sims_plotvars import HET_sims_plotvars
from HET_sims_plotvars import HET_sims_cp_vars
from HET_sims_post import max_min_mean_vals, comp_phase_shift, comp_FFT, comp_Boltz
from FFT import FFT
from scipy.signal import correlate
import pylab


# Close all existing figures
plt.close("all")

#cm.ScalarMappable(norm=None, cmap='YlGnBu')


################################ INPUTS #######################################
# Print out time step
timestep = 'last'
#timestep = 13
#timestep = 400
if timestep == 'last':
    timestep = -1

    
# Plots save flag
save_flag = 0
#figs_format = ".eps"
figs_format = ".png"
#figs_format = ".pdf"



#path_out = "../../HET_figs/zprof_figs/al0025_avge600steps/"
#path_out = "../../HET_figs/Id_figs/"
# THESIS ----------------------------------------------------------------------
path_out = "../../HET_figs/temp_figs/"
#path_out = "../../HET_figs/mesh_figs/"
# POST-THESIS -----------------------------------------------------------------
#path_out = "../../HET_figs/Vd_cases/"
#path_out = "../../HET_figs/mA_cases/"
#path_out = "../../../Thesis/Document/Figures/Ch_Discharges/temp_figs/"
#if generate_thesis_figures == 1 and save_flag == 1:
#    path_out = "../../../Thesis/Document/Figures/Ch_Discharges/"



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
    
    
    
    
######################## READ INPUT/OUTPUT FILES ##########################
    
ticks_size_isolines = 20
marker_every = 3

rind       = 19
rind_anode1 = rind
rind_anode2 = 17
zind_anode  = 8
elems_cath_Bline   = range(407-1,483-1+2,2) # Elements along the cathode B line for cases C1, C2 and C3
#    elems_cath_Bline   = range(875-1,951-1+2,2) # Elements along the cathode B line for case C5

# Cathode plotting flag and cathode position in cm (for plot_zcath_012 = 2) 
plot_zcath_012 = 2
zcat_pos       = 5.9394542444501024

# Print out time step
timestep = 0
timesteps = [996,1010,1050,1095]

allsteps_flag   = 1
read_inst_data  = 0
read_part_lists = 0
read_flag       = 1

mean_vars       = 1
mean_type       = 0
last_steps      = 600
step_i          = 100
step_f          = 190
plot_mean_vars  = 1


plot_fields_ref = 0
plot_dens_ref   = 0
plot_temp_ref   = 0
plot_vel_ref    = 0
plot_curr_ref   = 1
plot_nmp_ref    = 0
plot_freq_ref   = 0
plot_anode_ref  = 0    

Bline_all2Dplots = 1
cath2D_plots     = 0
cath2D_title     = r"(d)"


if allsteps_flag == 0:
    mean_vars = 0


# Simulation names
nsims = 1    
sim_names = ["../../Rb_hyphen/sim/sims/SPT100_al0025_Ne5_C1"]
#    sim_names = ["../../Rb_hyphen/sim/sims/SPT100_al0025_Ne100_C1"]
#    sim_names = ["../../Rb_hyphen/sim/sims/SPT100_also1510_V1"]


# Labels             
labels = [r"A",r"B",r"C",r"D",r"Average"]

# Line colors
colors = ['r','g','b','m','k','m','y']
# Markers
markers = ['^','>','v', '<', 's', 'o','*']
# Line style
#    linestyles = ['-','--','-.', ':','-','--','-.']
linestyles = ['-','-','-', '-','-','-','-']



plt.figure('ji zr ref')
#plt.title(r"(f) $|\tilde{\boldsymbol{j}}_{i}|$ (Am$^{-2}$)", fontsize = font_size,y=1.02)
    
    
k = 0
print("##### CASE "+str(k+1)+": "+sim_names[k][sim_names[k].find('S')::]+" #####")
# Obtain paths to simulation files
path_picM         = sim_names[k]+"/SET/inp/SPT100_picM.hdf5"
path_simstate_inp = sim_names[k]+"/CORE/inp/SimState.hdf5"
path_simstate_out = sim_names[k]+"/CORE/out/SimState.hdf5"
path_postdata_out = sim_names[k]+"/CORE/out/PostData.hdf5"
path_simparams_inp = sim_names[k]+"/CORE/inp/sim_params.inp"
print("Reading results...")
[num_ion_spe,num_neu_spe,n_mp_cell_i,n_mp_cell_n,n_mp_cell_i_min,
   n_mp_cell_i_max,n_mp_cell_n_min,n_mp_cell_n_max,min_ion_plasma_density,
   m_A,spec_refl_prob,ene_bal,points,zs,rs,zscells,rscells,dims,
   nodes_flag,cells_flag,cells_vol,volume,ind_maxr_c,ind_maxz_c,nr_c,nz_c,
   eta_max,eta_min,xi_top,xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,
   nsteps,nsteps_fast,nsteps_eFld,faces,nodes,boundary_f,face_geom,elem_geom,
   n_faces,n_elems,n_faces_boundary,bIDfaces_Dwall,bIDfaces_Awall,
   bIDfaces_FLwall,IDfaces_Dwall,IDfaces_Awall,IDfaces_FLwall,zfaces_Dwall,
   rfaces_Dwall,Afaces_Dwall,zfaces_Awall,rfaces_Awall,Afaces_Awall,
   zfaces_FLwall,rfaces_FLwall,Afaces_FLwall,
   cath_elem,z_cath,r_cath,V_cath,mass,ssIons1,ssIons2,ssNeutrals1,ssNeutrals2,
   n_mp_i1_list,n_mp_i2_list,n_mp_n1_list,n_mp_n2_list,phi,phi_elems,Ez,Er,Efield,
   Bz,Br,Bfield,Te,Te_elems,cs01,cs02,nn1,nn2,ni1,ni2,ne,ne_elems,fn1_x,fn1_y,
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
   nu_cath,ndot_cath,Q_cath,P_cath,F_theta,Hall_par,Hall_par_eff,
   nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,Boltz,Boltz_dim] = HET_sims_read(path_simstate_inp,path_simstate_out,
                                                                                                   path_postdata_out,path_simparams_inp,
                                                                                                   path_picM,allsteps_flag,timestep,read_inst_data,
                                                                                                   read_part_lists,read_flag)
print("Generating plotting variables (NaN in ghost nodes)...")                                                                                                      
[Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,fn1_x,fn1_y,fn1_z,
   fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,
   un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,ji1_z,
   ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,ue_perp,
   ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,
   avg_w_n2,avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
   ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,
   F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12] = HET_sims_plotvars(nodes_flag,cells_flag,Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,
                                                                                                                     fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,
                                                                                                                     un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,
                                                                                                                     ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,
                                                                                                                     ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
                                                                                                                     n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                                                                                                     neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                                                                                                     ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,F_theta,Hall_par,Hall_par_eff,
                                                                                                                     nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12)
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
       Boltz_mean,Boltz_dim_mean,phi_elems_mean,ne_elems_mean,Te_elems_mean] = HET_sims_mean(nsteps,mean_type,last_steps,step_i,step_f,phi,Er,Ez,Efield,Br,Bz,Bfield,
                                                                                    nn1,nn2,ni1,ni2,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,
                                                                                    fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,
                                                                                    ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,
                                                                                    je_perp,je_para,ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
                                                                                    n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                                                                    neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                                                                    ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,nu_cath,ndot_cath,F_theta,
                                                                                    Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,
                                                                                    nu_i02,nu_i12,Boltz,Boltz_dim,phi_elems,ne_elems,Te_elems)
                                                                                    
                                                                                    
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
       nu_en_plot,nu_ei1_plot,nu_ei2_plot,nu_i01_plot,nu_i02_plot,nu_i12_plot] = HET_sims_cp_vars(Br,Bz,Bfield,phi_mean,Er_mean,Ez_mean,Efield_mean,nn1_mean,nn2_mean,
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
                                                                                                  nu_i02_mean,nu_i12_mean)
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
       nu_en_plot,nu_ei1_plot,nu_ei2_plot,nu_i01_plot,nu_i02_plot,nu_i12_plot] = HET_sims_cp_vars(Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,
                                                                                                  fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,
                                                                                                  un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,
                                                                                                  ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,
                                                                                                  ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
                                                                                                  n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                                                                                  neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                                                                                  ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,nu_cath,ndot_cath,F_theta,
                                                                                                  Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,
                                                                                                  nu_i02,nu_i12)
                                                                                                                                                           
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
uimean_x_plot           = ji_x_plot/(e*ne_plot)
uimean_y_plot           = ji_y_plot/(e*ne_plot)
uimean_z_plot           = ji_z_plot/(e*ne_plot)
uimean_plot             = np.sqrt( uimean_x_plot**2 + uimean_y_plot**2 + uimean_z_plot**2 )
j_r_plot                = ji_x_plot + je_r_plot
j_t_plot                = ji_y_plot + je_t_plot
j_z_plot                = ji_z_plot + je_z_plot
j_plot                  = np.sqrt(j_r_plot**2 + j_t_plot**2 + j_z_plot**2)
erel_je_plot            = np.abs(je2_plot-je_plot)/np.abs(je_plot)
erel_ue_plot            = np.abs(ue2_plot-ue_plot)/np.abs(ue_plot)
erel_jeji_plot          = np.abs(je_plot-ji_plot)/np.abs(ji_plot)
erel_jz_plot            = np.abs(je_z_plot+ji_z_plot)/np.abs(ji_z_plot)
erel_jr_plot            = np.abs(je_r_plot+ji_x_plot)/np.abs(ji_x_plot)
ratio_ue_t_perp_plot    = ue_t_plot/ue_perp_plot
ratio_ue_t_para_plot    = ue_t_plot/ue_para_plot
ratio_ue_perp_para_plot = ue_perp_plot/ue_para_plot
je2D_plot               = np.sqrt(je_r_plot**2 + je_z_plot**2)
ji2D_plot               = np.sqrt(ji_x_plot**2 + ji_z_plot**2)
j2D_plot                = np.sqrt(j_r_plot**2 + j_z_plot**2)

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
##################################################
############################ GENERATING PLOTS #############################
zs                = zs*1E2
rs                = rs*1E2
zscells           = zscells*1E2
rscells           = rscells*1E2
points            = points*1E2
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
je_para_plot      = je_para_plot*1E-4
je_perp_plot      = je_perp_plot*1E-4
je_t_plot         = je_t_plot*1E-4
ji_x_plot         = ji_x_plot*1E-3
ji_z_plot         = ji_z_plot*1E-3
#    nu_e_tot_plot     = nu_e_tot_plot*1E-6
#    nu_e_tot_eff_plot = nu_e_tot_eff_plot*1E-6
#    nu_en_plot        = nu_en_plot*1E-6
#    nu_ei1_plot       = nu_ei1_plot*1E-6
#    nu_ei2_plot       = nu_ei2_plot*1E-6
#    nu_i01_plot       = nu_i01_plot*1E-6
#    nu_i02_plot       = nu_i02_plot*1E-6
#    nu_i12_plot       = nu_i12_plot*1E-6

# Obtain the vectors for the uniform mesh for streamlines plotting. It must be uniform and squared mesh
delta_x = 0.11
zvec = np.arange(zs[0,0],zs[0,-1]+delta_x,delta_x)
rvec = np.copy(zvec)

plt.figure('ji zr ref')
ax = plt.gca()
log_type         = 1
auto             = 1
min_val0         = 0.0
max_val0         = 0.0
cont             = 1
lines            = 0
cont_nlevels     = 500
auto_cbar_ticks  = -1 
auto_lines_ticks = 0
nticks_cbar      = 5
nticks_lines     = 10
cbar_ticks       = np.array([0.0,0.2,0.3,0.5,0.7,0.8,0.9,1])
#        lines_ticks      = np.array([1E-5,1E-4,1E-3,1E-2,1E-1,1E0,2E0,3E0,4E0,5E0,6E0,7E0,8E0,9E0,1E1,1E2,2E2,4E2,6E2,8E2])
lines_ticks      = np.array([5E1,1E2,2E2,4E2,6E2])   
lines_ticks_loc  = [(0.6,4.25),(1.15,4.25),(2.24,4.5),(4.3,4.25),(6.32,4.25),(7.65,6.75),(5.0,7.3),(3.6,1.9),(4.82,1.81),(7.48,1.24),(8.6,0.7),(9.95,0.25)]
#        lines_ticks_loc  = 'default'
cbar_ticks_fmt    = '{%.1f}'
lines_ticks_fmt   = '{%.2f}'
lines_width       = line_width
lines_ticks_color = 'k'
lines_style       = '-'
[CS,CS2] = contour_2D (ax,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, ji2D_plot, nodes_flag, log_type, auto, 
                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width) 
#ax.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
#plt.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
#plt.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
#plt.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)


# --- Plot the electron streamlines 
streamline_color = 'w'
# Define number of starting points for plotting the streamlines
plot_start_points = 0
#        nstart_r = 25
#        nstart_z = 15
nstart_r = 10
nstart_z = 3
start_points = np.zeros((0,2))
# Mid chamber axial line
zstart = np.linspace(1,len(zvec)-2,nstart_z)
rstart = 38*np.ones(nstart_z)
start_points = np.append(start_points,np.transpose(np.array([zstart,rstart])),axis=0)
## Top plume axial line
#zstart = np.linspace(28,len(zvec)-2,nstart_z)
#rstart = 68*np.ones(nstart_z)
#start_points = np.append(start_points,np.transpose(np.array([zstart,rstart])),axis=0)
# Radial line at plume 1
zstart = 40*np.ones(nstart_r)
rstart = np.linspace(1,len(zvec)-2,nstart_r)
start_points = np.append(start_points,np.transpose(np.array([zstart,rstart])),axis=0)
# Radial line at plume 2
zstart = 80*np.ones(nstart_r)
rstart = np.linspace(1,len(zvec)-2,nstart_r)
start_points = np.append(start_points,np.transpose(np.array([zstart,rstart])),axis=0)
## Radial chamber line
#zstart = 20*np.ones(nstart_r-5)
#rstart = np.linspace(30,50,nstart_r-5)
#start_points = np.append(start_points,np.transpose(np.array([zstart,rstart])),axis=0)
                    
stream = streamline_2D(ax,zvec,rvec,dims,zs,rs,-je_z_plot,-je_r_plot,flag_start,start_points,
                       plot_start_points,10,streamline_width,streamline_color,arrow_size,
                       arrow_style,min_length)
ax.set_xlim(zs[0,0],zs[0,-1])
ax.set_ylim(rs[0,0],rs[-1,0])


# --- Plot the ion streamlines 
streamline_color = 'b'
# Define number of starting points for plotting the streamlines
plot_start_points = 0
#        nstart_r = 25
#        nstart_z = 15
nstart_r = 5
nstart_z = 3
start_points = np.zeros((0,2))
# Radial line at plume 2
zstart = 90*np.ones(3)
rstart = np.array([22,32,43])
start_points = np.append(start_points,np.transpose(np.array([zstart,rstart])),axis=0)
# Top plume axial line
zstart = np.linspace(28,len(zvec)-2,nstart_z+2)
rstart = 68*np.ones(nstart_z+2)
start_points = np.append(start_points,np.transpose(np.array([zstart,rstart])),axis=0)
# Bottom plume axial line
zstart = np.linspace(28,len(zvec)-2,nstart_z)
rstart = 17*np.ones(nstart_z)
start_points = np.append(start_points,np.transpose(np.array([zstart,rstart])),axis=0)

zstart = np.linspace(28,len(zvec)-2,nstart_z)
rstart = 5*np.ones(nstart_z)
start_points = np.append(start_points,np.transpose(np.array([zstart,rstart])),axis=0)
                    
stream = streamline_2D(ax,zvec,rvec,dims,zs,rs,ji_z_plot,ji_x_plot,flag_start,start_points,
                       plot_start_points,10,streamline_width,streamline_color,arrow_size,
                       arrow_style,min_length)
ax.set_xlim(zs[0,0],zs[0,-1])
ax.set_ylim(rs[0,0],rs[-1,0])

ax.axis('off')

#plt.savefig(path_out+"external_cover_fig"+figs_format,bbox_inches='tight',dpi=800,transparent=True) 
#plt.savefig(path_out+"external_cover_fig"+figs_format,bbox_inches='tight',transparent=True) 
#plt.close()