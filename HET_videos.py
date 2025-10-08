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
# Do not show up figures
import matplotlib as mpl
mpl.use("Agg")
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
from HET_sims_post import max_min_mean_vals, comp_phase_shift, comp_FFT, comp_Boltz
from FFT import FFT
from scipy.signal import correlate
import pylab


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

    
# Plots save flag
save_flag = 1
#figs_format = ".eps"
figs_format = ".png"
#figs_format = ".pdf"

video_type   = 1
nlevels_cont = 100

#path_out = "../../HET_videos/video2/REF_case_100levels/"
#path_out = "../../HET_videos/video3/REF_case_100levels/"
#path_out = "ElAlmendro2021/HT5k_video/"
#path_out = "ElAlmendro2021/T2N4_video/"
path_out = "EPIC_2022/HT5k_video/"


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
marker_size_moving     = 8
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
###############################################################################



    
marker_size_ID = 10
props = dict(boxstyle='round', facecolor='white', edgecolor = 'k',alpha=1) 

timestep        = 0
allsteps_flag   = 1
read_inst_data  = 1
read_part_lists = 0
read_flag       = 0

mean_type       = 2 
order           = 50
order_fast      = 500
#last_steps      = 670
last_steps      = 1000
step_i          = 400
step_f          = 700
last_steps_fast = 33500
step_i_fast     = int(step_i*50)
step_f_fast     = int(step_f*50)
print_mean_vars = 1



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

#prntstep_IDs = []
#prntstep_IDs_text = []
#fast_prntstep_IDs = []
    
# Simulation names
nsims = 1  

# For case HT5k_rm6_tm08_8te1tq25s01_V300_m14_change_inj
oldpost_sim      = np.array([4,4,4,4,4,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
oldsimparams_sim = np.array([15,14,14,13,13,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)  

# For case HT5k_rm6_tm08_8te1tq25s01_V300_m14_no_RLC
#oldpost_sim      = np.array([5,4,4,4,4,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
#oldsimparams_sim = np.array([16,14,14,13,13,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)  

#sim_names = ["../../Rb_hyphen/sim/sims/SPT100_al0025_Ne5_C1"]
#sim_names = ["../../../Ca_hyphen/sim/sims/HT5k_rm6_tm08_8te1tq25s01_V300_m14_no_RLC"]
sim_names = ["../../../Ca_hyphen/sim/sims/HT5k_rm6_tm08_8te1tq25s01_V300_m14_change_inj"]
#sim_names = ["../../../Sr_hyphen/sim/sims/T2N4_pm1em1_cat1200_tm15_te1_tq125_0438e2a"]

PIC_mesh_file_name = ["HT5k_PIC_mesh_rm3.hdf5"]
#PIC_mesh_file_name = ["PIC_mesh_topo2_refined4.hdf5"]
             

# Labels  
labels = [r"$\bar{n}_e$",
          r"$\bar{n}_n$"]
                            



# Line colors
#    colors = ['r','g','b','k','c','m','y',orange,brown]
colors = ['k','r','g','b','m','c','y',orange,brown]
#    colors = ['k','m',orange,brown]
# Markers
markers = ['s','o','v', '^', '<', '>','*']
#    markers = ['s','','D','p']
markers = ['','','', '', '', '','']
# Line style
linestyles = ['-','--','-.', ':','-','--','-.']
#    linestyles = ['-','-','--','-.', ':','-','--','-.']
linestyles = ['-','-','-','-']

dashList = [(None,None),(None,None),(12,6,12,6,3,6),(12,6,3,6,3,6),(5,2,20,2)] 
          


k = 0
ind_ini_letter = sim_names[k].rfind('/') + 1
print("##### CASE "+str(k+1)+": "+sim_names[k][ind_ini_letter::]+" #####")
print("##### oldsimparams_sim = "+str(oldsimparams_sim[k])+" #####")
print("##### oldpost_sim      = "+str(oldpost_sim[k])+" #####")
print("##### last_steps       = "+str(last_steps)+" #####")
######################## READ INPUT/OUTPUT FILES ##########################
# Obtain paths to simulation files
#path_picM         = sim_names[k]+"/SET/inp/SPT100_picM.hdf5"
path_picM         = sim_names[k]+"/SET/inp/"+PIC_mesh_file_name[k]
path_simstate_inp = sim_names[k]+"/CORE/inp/SimState.hdf5"
path_simstate_out = sim_names[k]+"/CORE/out/SimState.hdf5"
path_postdata_out = sim_names[k]+"/CORE/out/PostData.hdf5"
path_simparams_inp = sim_names[k]+"/CORE/inp/sim_params.inp"
print("Reading results...")
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
                                                
# Obtain the FFT of the discharge current and the beam current 
if mean_type == 2:
    # Obtain FFT for Id considering an integer number of periods
    [fft_Id,freq_Id,max_fft_Id,max_freq_Id] = comp_FFT(time,Id,time[nsteps-last_steps::],Id[nsteps-last_steps::],order)
    # Obtain FFT for Id_inst considering an integer number of periods
    [fft_Id_inst,freq_Id_inst,max_fft_Id_inst,max_freq_Id_inst] = comp_FFT(time,Id_inst,time[nsteps-last_steps::],Id_inst[nsteps-last_steps::],order)
    # Obtain FFT for I_beam considering an integer number of periods
    [fft_I_beam,freq_I_beam,max_fft_I_beam,max_freq_I_beam] = comp_FFT(time,I_beam,time[nsteps-last_steps::],I_beam[nsteps-last_steps::],order)
    # Obtain FFT for avg_dens_mp_ions considering an integer number of periods
    [fft_avg_dens_mp_ions,freq_avg_dens_mp_ions,max_fft_avg_dens_mp_ions,max_freq_avg_dens_mp_ions] = comp_FFT(time_fast,avg_dens_mp_ions,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_ions[nsteps_fast-last_steps_fast::],order_fast)
    # Obtain FFT for avg_dens_mp_neus considering an integer number of periods
    [fft_avg_dens_mp_neus,freq_avg_dens_mp_neus,max_fft_avg_dens_mp_neus,max_freq_avg_dens_mp_neus] = comp_FFT(time_fast,avg_dens_mp_neus,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_neus[nsteps_fast-last_steps_fast::],order_fast)
#    # Obtain FFT for nu_cath considering an integer number of periods
#    [fft_nu_cath,freq_nu_cath,max_fft_nu_cath,max_freq_nu_cath] = comp_FFT(time,nu_cath,time[nsteps-last_steps::],nu_cath[nsteps-last_steps::],order)
    # Obtain FFT for P_cath considering an integer number of periods
    [fft_P_cath,freq_P_cath,max_fft_P_cath,max_freq_P_cath] = comp_FFT(time,P_cath,time[nsteps-last_steps::],P_cath[nsteps-last_steps::],order)
    
    # Obtain the phase shift of the signals Id and I_beam from the time between max peaks
    [_,_,time_shift_IdIbeam,phase_shift_IdIbeam_deg] = comp_phase_shift(time,Id,I_beam,time[nsteps-last_steps::],Id[nsteps-last_steps::],
                                                                        time[nsteps-last_steps::],I_beam[nsteps-last_steps::],order)
    # Obtain the phase shift of the signals avg_dens_mp_neus and avg_dens_mp_ions from the time between max peaks
    [_,_,time_shift_avg_dens_mp_neusions,phase_shift_avg_dens_mp_neusions_deg] = comp_phase_shift(time_fast,avg_dens_mp_neus,avg_dens_mp_ions,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_neus[nsteps_fast-last_steps_fast::],
                                                                                                  time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_ions[nsteps_fast-last_steps_fast::],order_fast)
    # Obtain the phase shift of the signals representing the contributions to the total heavy species mass balance
    [_,_,time_shift_ctr_mbal_tot,phase_shift_ctr_mbal_tot_deg] = comp_phase_shift(time,ctr_mflow_fw_tot,ctr_mflow_tw_tot,time[nsteps-last_steps::],ctr_mflow_fw_tot[nsteps-last_steps::],
                                                                                  time[nsteps-last_steps::],ctr_mflow_tw_tot[nsteps-last_steps::],order)     

# Obtain the utilization efficiency from the actual flows
eta_u_bis = (mflow_twinf_i1 + mflow_twinf_i2)/(mflow_inj_n1-(mflow_twa_i1+mflow_twa_i2+mflow_twa_n1))   

# Obtain the total net power of the heavy species deposited to the injection (anode) wall
P_inj_hs = eneflow_twa_i1 + eneflow_twa_i2 + eneflow_twa_n1 - (eneflow_inj_i1 + eneflow_inj_i2 + eneflow_inj_n1)
# Obtain the total net power of the heavy species deposited to the dielectric walls
P_mat_hs = eneflow_twmat_i1 + eneflow_twmat_i2 + eneflow_twmat_n1 - (eneflow_fwmat_i1 + eneflow_fwmat_i2 + eneflow_fwmat_n1)

# Obtain mean values
if mean_type == 2:
    [_,_,_,_,_,_,
     mean_min_mass_mp_ions1,mean_max_mass_mp_ions1,mass_mp_ions1_mean,
     max2mean_mass_mp_ions1,min2mean_mass_mp_ions1,amp_mass_mp_ions1,
     mins_ind_comp_mass_mp_ions1,maxs_ind_comp_mass_mp_ions1]                   = max_min_mean_vals(time_fast,time_fast[nsteps_fast-last_steps_fast::],mass_mp_ions[nsteps_fast-last_steps_fast::,0],order)
    [_,_,_,_,_,_,
     mean_min_mass_mp_ions2,mean_max_mass_mp_ions2,mass_mp_ions2_mean,
     max2mean_mass_mp_ions2,min2mean_mass_mp_ions2,amp_mass_mp_ions2,
     mins_ind_comp_mass_mp_ions2,maxs_ind_comp_mass_mp_ions2]                   = max_min_mean_vals(time_fast,time_fast[nsteps_fast-last_steps_fast::],mass_mp_ions[nsteps_fast-last_steps_fast::,1],order)
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
    [_,_,_,_,_,_,
     mean_min_thrust_i2,mean_max_thrust_i2,thrust_i2_mean,
     max2mean_thrust_i2,min2mean_thrust_i2,amp_thrust_i2,
     mins_ind_comp_thrust_i2,maxs_ind_comp_thrust_i2]                           = max_min_mean_vals(time,time[nsteps-last_steps::],thrust_ion[nsteps-last_steps::,1],order)
    [_,_,_,_,_,_,
     mean_min_thrust_n,mean_max_thrust_n,thrust_n_mean,
     max2mean_thrust_n,min2mean_thrust_n,amp_thrust_n,
     mins_ind_comp_thrust_n,maxs_ind_comp_thrust_n]                             = max_min_mean_vals(time,time[nsteps-last_steps::],thrust_neu[nsteps-last_steps::],order)
    [_,_,_,_,_,_,
     mean_min_Id,mean_max_Id,Id_mean,
     max2mean_Id,min2mean_Id,amp_Id,
     mins_ind_comp_Id,maxs_ind_comp_Id]                                         = max_min_mean_vals(time,time[nsteps-last_steps::],Id[nsteps-last_steps::],order)
    [_,_,_,_,_,_,
     mean_min_Id_inst,mean_max_Id_inst,Id_inst_mean,
     max2mean_Id_inst,min2mean_Id_inst,amp_Id_inst,
     mins_ind_comp_Id_inst,maxs_ind_comp_Id_inst]                               = max_min_mean_vals(time,time[nsteps-last_steps::],Id_inst[nsteps-last_steps::],order)
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
    [_,_,_,_,_,_,
     mean_min_P_cath,mean_max_P_cath,P_cath_mean,
     max2mean_P_cath,min2mean_P_cath,amp_P_cath,
     mins_ind_comp_P_cath,maxs_ind_comp_P_cath]                                 = max_min_mean_vals(time,time[nsteps-last_steps::],P_cath[nsteps-last_steps::],order)
#    [_,_,_,_,_,_,
#     mean_min_nu_cath,mean_max_nu_cath,nu_cath_mean,
#     max2mean_nu_cath,min2mean_nu_cath,amp_nu_cath,
#     mins_ind_comp_nu_cath,maxs_ind_comp_nu_cath]                               = max_min_mean_vals(time,time[nsteps-last_steps::],nu_cath[nsteps-last_steps::],order)
    [_,_,_,_,_,_,
     mean_min_I_tw_tot,mean_max_I_tw_tot,I_tw_tot_mean,
     max2mean_I_tw_tot,min2mean_I_tw_tot,amp_I_tw_tot,
     mins_ind_comp_I_tw_tot,maxs_ind_comp_I_tw_tot]                             = max_min_mean_vals(time,time[nsteps-last_steps::],I_tw_tot[nsteps-last_steps::],order)
    [_,_,_,_,_,_,
     mean_min_mflow_twinf_i1,mean_max_mflow_twinf_i1,mflow_twinf_i1_mean,
     max2mean_mflow_twinf_i1,min2mean_mflow_twinf_i1,amp_mflow_twinf_i1,
     mins_ind_comp_mflow_twinf_i1,maxs_ind_comp_mflow_twinf_i1]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twinf_i1[nsteps-last_steps::],order)
    [_,_,_,_,_,_,
     mean_min_mflow_twinf_i2,mean_max_mflow_twinf_i2,mflow_twinf_i2_mean,
     max2mean_mflow_twinf_i2,min2mean_mflow_twinf_i2,amp_mflow_twinf_i2,
     mins_ind_comp_mflow_twinf_i2,maxs_ind_comp_mflow_twinf_i2]                 = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twinf_i2[nsteps-last_steps::],order)
    [_,_,_,_,_,_,
     mean_min_mflow_inj_n1,mean_max_mflow_inj_n1,mflow_inj_n1_mean,
     max2mean_mflow_inj_n1,min2mean_mflow_inj_n1,amp_mflow_inj_n1,
     mins_ind_comp_mflow_inj_n1,maxs_ind_comp_mflow_inj_n1]                     = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_inj_n1[nsteps-last_steps::],order)
    [_,_,_,_,_,_,
     mean_min_mflow_twa_i1,mean_max_mflow_twa_i1,mflow_twa_i1_mean,
     max2mean_mflow_twa_i1,min2mean_mflow_twa_i1,amp_mflow_twa_i1,
     mins_ind_comp_mflow_twa_i1,maxs_ind_comp_mflow_twa_i1]                     = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twa_i1[nsteps-last_steps::],order)
    [_,_,_,_,_,_,
     mean_min_mflow_twa_i2,mean_max_mflow_twa_i2,mflow_twa_i2_mean,
     max2mean_mflow_twa_i2,min2mean_mflow_twa_i2,amp_mflow_twa_i2,
     mins_ind_comp_mflow_twa_i2,maxs_ind_comp_mflow_twa_i2]                     = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twa_i2[nsteps-last_steps::],order)
    [_,_,_,_,_,_,
     mean_min_mflow_twa_n1,mean_max_mflow_twa_n1,mflow_twa_n1_mean,
     max2mean_mflow_twa_n1,min2mean_mflow_twa_n1,amp_mflow_twa_n1,
     mins_ind_comp_mflow_twa_n1,maxs_ind_comp_mflow_twa_n1]                     = max_min_mean_vals(time,time[nsteps-last_steps::],mflow_twa_n1[nsteps-last_steps::],order)
    [_,_,_,_,_,_,
     mean_min_Pe_Dwall,mean_max_Pe_Dwall,Pe_Dwall_mean,
     max2mean_Pe_Dwall,min2mean_Pe_Dwall,amp_Pe_Dwall,
     mins_ind_comp_Pe_Dwall,maxs_ind_comp_Pe_Dwall]                             = max_min_mean_vals(time,time[nsteps-last_steps::],Pe_Dwall[nsteps-last_steps::],order)
    [_,_,_,_,_,_,
     mean_min_Pe_Awall,mean_max_Pe_Awall,Pe_Awall_mean,
     max2mean_Pe_Awall,min2mean_Pe_Awall,amp_Pe_Awall,
     mins_ind_comp_Pe_Awall,maxs_ind_comp_Pe_Awall]                             = max_min_mean_vals(time,time[nsteps-last_steps::],Pe_Awall[nsteps-last_steps::],order)
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
print("Obtaining final variables for plotting...") 
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
                                         nu_i02,nu_i12,err_interp_n,f_split_adv,f_split_qperp,f_split_qpara,
                                         f_split_qb,f_split_Pperp,f_split_Ppara,f_split_ecterm,f_split_inel)
                                                                                              
# Time in ms and dimensions in cm
time       = time*1e3
time_fast  = time_fast*1e3
steps      = steps*1e-3
steps_fast = steps_fast*1e-3
zs         = zs*1e2
rs         = rs*1e2
points     = points*1e2
z_cath     = z_cath*1e2
r_cath     = r_cath*1e2
Er_plot_cont      = np.copy(Er_plot)
Ez_plot_cont      = np.copy(Ez_plot)
Efield_plot_cont  = np.sqrt(Er_plot_cont**2 + Ez_plot_cont**2)
ratio_ni1_ni2_plot = np.divide(ni1_plot,ni2_plot)


# Do not plot units in axes
# SAFRAN CHEOPS 1: units in cm
#L_c = 3.725
#H_c = (0.074995-0.052475)*100
# HT5k: units in cm
L_c = 2.53
H_c = (0.0785-0.0565)*100
zs = zs/L_c
rs = rs/H_c
points[:,0] = points[:,0]/L_c
points[:,1] = points[:,1]/H_c
z_cath = z_cath/L_c
r_cath = r_cath/H_c
zscells = zscells/L_c
rscells = rscells/H_c

# cont_xlabel = '$z$ (cm)'
# cont_ylabel = '$r$ (cm)'

cont_xlabel = '$z/L_c$'
cont_ylabel = '$r/H_c$'

###########################################################################
print("Plotting...")
############################ GENERATING PLOTS #############################
first_step = np.where(Id == 0)[0][-1] + 1
#final_step = len(Id)
final_step = 1000

first_step = 100
first_step_fast = first_step*50
final_step_fast = final_step*50
index = 0

#plt.figure('test')
#vec = np.mean(Te_plot[0:5,int(xi_bottom),:],axis=0)
#pos = np.where(vec < 10.0)
#print(np.shape(pos))
#plt.plot(time, np.mean(Te_plot[0:3,int(xi_bottom),:],axis=0) )

for ii in range(first_step,final_step):
#for ii in range(first_step,first_step + 2):
    
    # Avoid considering snapshots with large Te in cathode for HT5k case
    if np.max(Te[0:6,int(xi_bottom)::,ii]) < 10.0:
        index = index + 1
        i_fast = 50*ii
    
    
        [fig, axes] = plt.subplots(nrows=2, ncols=3 ,figsize=(22.5,12))
        ax1 = plt.subplot2grid( (2,3), (0,0) )
        ax2 = plt.subplot2grid( (2,3), (1,0) )
        ax3 = plt.subplot2grid( (2,3), (0,1) )
        ax4 = plt.subplot2grid( (2,3), (1,1) )
        ax5 = plt.subplot2grid( (2,3), (0,2) )
        ax6 = plt.subplot2grid( (2,3), (1,2) )
    
    
        ax1.set_title(r"$I_d$ (A)",fontsize = font_size, y=1.02)
        ax1.set_xlabel(r'$t$ (ms)', fontsize = font_size)
        ax1.tick_params(labelsize = ticks_size) 
        ax1.tick_params(axis='x', which='major', pad=10)
        
        ax2.set_title(r"$\bar{n}_e$, $\bar{n}_n$ (m$^{-3}$)",fontsize = font_size, y=1.02)
        ax2.set_xlabel(r'$t$ (ms)', fontsize = font_size)
        ax2.tick_params(labelsize = ticks_size) 
        ax2.tick_params(axis='x', which='major', pad=10)
            
    
            
        # Plot the time evolution of the discharge current
    #    ax1.plot(time[first_step::], Id[first_step::], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
        ax1.plot(time[first_step:final_step], Id[first_step:final_step], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
        ylims = ax1.get_ylim()  
        marker_size_ID = 10
        fact_x = np.array([0.97,1.03,1.0,0.98])
    #    for i in range(0,len(prntstep_IDs)):
    #        ax1.plot(time[prntstep_IDs[i]]*np.ones(2), np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker="", color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
    #        ax1.plot(time[prntstep_IDs[i]], Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
    ##        ax1.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
    #        ax1.text(fact_x[i]*time[prntstep_IDs[i]-plot_tol], fact_y[i]*Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')                
    #        ax1.plot(time, Id_mean*np.ones(np.shape(time)), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")            
        ax1.plot(time[ii], Id[ii], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_moving, marker="o", color='r', markeredgecolor = 'k', label="")
    #    ax1.set_xlim([time[first_step],time[-1]])
        ax1.set_xlim([time[first_step],time[final_step]])
        ax1.set_ylim(10,20)
        
        # Plot the time evolution of both the average plasma and neutral density in the domain
    #    ax2.semilogy(time_fast[first_step_fast::], avg_dens_mp_ions[first_step_fast::], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label=labels[0])            
    #    ax2.semilogy(time_fast[first_step_fast::], avg_dens_mp_neus[first_step_fast::], linestyle='--', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label=labels[1])            
        ax2.semilogy(time_fast[first_step_fast:final_step_fast], avg_dens_mp_ions[first_step_fast:final_step_fast], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label=labels[0])            
        ax2.semilogy(time_fast[first_step_fast:final_step_fast], avg_dens_mp_neus[first_step_fast:final_step_fast], linestyle='--', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label=labels[1])            
        
        ax2.set_xlim([time[first_step],time[final_step]])
    #    ax2.set_ylim(10**15,10**19)
        ax2.set_ylim(10**17,2*10**18)    #HT5k
    #    ax2.set_ylim(10**17,2*10**17)
    #    ax2.set_ylim(9*10**15,7*10**17)  # T2N4
        ylims = ax2.get_ylim()
        fact_x = np.array([0.97,1.02,1.02,1.02])
        marker_size_ID = 6
    #    for i in range(0,len(fast_prntstep_IDs)):
    #        ax2.semilogy(time_fast[fast_prntstep_IDs[i]]*np.ones(2), np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker="", color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
    #        ax2.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_ions[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
    #        ax2.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_neus[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
    #        ax2.text(fact_x[i]*time_fast[fast_prntstep_IDs[i]-plot_tol], 1.5*ylims[0],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')     
    #        print(prntstep_IDs_text[i]+" time_fast = "+str(time_fast[fast_prntstep_IDs[i]])+", time = "+str(time[prntstep_IDs[i]]))
        ax2.semilogy(time_fast[i_fast], avg_dens_mp_ions[i_fast], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size_moving, marker="o", color='c', markeredgecolor = 'k', label="")            
        ax2.semilogy(time_fast[i_fast], avg_dens_mp_neus[i_fast], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size_moving, marker="o", color='y', markeredgecolor = 'k', label="")            
        ax2.legend(fontsize = font_size,loc=2,ncol=2)
        
        
        if video_type == 1:
        
            # Plot the plasma density contour
            ax3.set_title(r'$n_{e}$ (m$^{-3}$)', fontsize = font_size,y=1.02)
            log_type         = 1
            auto             = 0
            # T2N4
    #        min_val0         = 1E14
    #        max_val0         = 1E18
            # HT5k
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
            [CS,CS2] = contour_2D (ax3, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, ne_plot[:,:,ii], nodes_flag, log_type, auto, 
                              min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                              nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                              lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
        #    # Isolines ticks (exponent)
        #    ax3.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
            ax3.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax3.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
        #    ax3.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
            ax3.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
            
            # Plot the neutral density contour
            ax4.set_title(r'$n_{n}$ (m$^{-3}$)', fontsize = font_size,y=1.02)
            ax = plt.gca()
            log_type         = 1
            auto             = 0
            # T2N4
    #        min_val0         = 1E13
    #        max_val0         = 1E19
            # HT5k
            min_val0         = 1E13
            max_val0         = 1E19
            cont             = 1
            lines            = 0
            cont_nlevels     = nlevels_cont
            auto_cbar_ticks  = 1 
            auto_lines_ticks = -1
            nticks_cbar      = 4
            nticks_lines     = 10
            cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
            lines_ticks      = np.array([3E19,1.5E19,1E19,5E18,1E18,5E17,2E17,1E17,5E16,3E16])
            lines_ticks_loc  = 'default'
            cbar_ticks_fmt    = '{%.2f}'
            lines_ticks_fmt   = '{%.1f}'
            lines_width       = line_width
            lines_ticks_color = 'k'
            lines_style       = '-'
            [CS,CS2] = contour_2D (ax4, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, nn1_plot[:,:,ii], nodes_flag, log_type, auto, 
                              min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                              nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                              lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
        #    # Isolines ticks (exponent)
        #    ax4.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, fontsize=ticks_size_isolines, zorder = 1)
            ax4.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax4.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            ax4.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
            
            # Plot the electric potential contour
            ax5.set_title(r'$\phi$ (V)', fontsize = font_size,y=1.02)
            log_type         = 0
            auto             = 0
            # T2N4
    #        min_val0         = -110
    #        max_val0         = 550.0
            # HT5k
            min_val0         = -110
            max_val0         = 320
            cont             = 1
            lines            = 0
            cont_nlevels     = nlevels_cont
            auto_cbar_ticks  = 0 
            auto_lines_ticks = -1
            nticks_cbar      = 5
            nticks_lines     = 10
            cbar_ticks       = np.array([300, 250, 200, 150, 100, 50, 0, -50, -100])   # HT5k
    #        cbar_ticks       = np.array([550,500,450,400,350,300, 250, 200, 150, 100, 50, 0,-50,-100]) # T2N4
        #        lines_ticks      = np.array([305, 280, 250, 200, 150, 100, 75, 50, 40, 30, 25, 15, 10, 5, 1, 0, -1, -2, -3, -4, -5, -6])
            lines_ticks      = np.array([305, 280, 250, 200, 150, 100, 75, 50, 40, 25, 15, 10, 5])
            lines_ticks_loc  = 'default'
            cbar_ticks_fmt    = '{%.0f}'
            lines_ticks_fmt   = '{%.0f}'
            lines_width       = line_width
            lines_ticks_color = 'w'
            lines_style       = '-'
            [CS,CS2] = contour_2D (ax5, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, phi_plot[:,:,ii], nodes_flag, log_type, auto, 
                                   min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                   nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                   lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)     
        #        ax5.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, fontsize=ticks_size_isolines, zorder = 1)
            ax5.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax5.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)    
        #        ax5.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
            ax5.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)     
            
            # Plot the electron temperature contour
            ax6.set_title(r'$T_e$ (eV)', fontsize = font_size,y=1.02)
            log_type         = 0
            auto             = 0
            min_val0         = 0.0
            # T2N4
    #        max_val0         = 70.0
            # HT5k
            max_val0         = 40.0
            cont             = 1
            lines            = 0
            cont_nlevels     = nlevels_cont
            auto_cbar_ticks  = 0 
            auto_lines_ticks = -1
            nticks_cbar      = 5
            nticks_lines     = 10
    #        cbar_ticks       = np.array([65,60,55,50,45,40,35,30,25,20,15,10,5,0]) # T2N4
            cbar_ticks       = np.array([40,35,30,25,20,15,10,5,0])                 # HT5k
        #        lines_ticks      = np.array([7,9,12,20,25,30,35,40,45])
            lines_ticks      = np.array([7,9,12,20,30,35,40,45])
            lines_ticks_loc  = [(0.38,4.25),(0.88,4.25),(1.5,4.25),(2.7,4.6),(3.0,3.8),(3.6,4.8),(3.9,4.25),(4.5,4.25),(5.18,4.0),(5.3,3.2),(5.6,1.8),(3.7,6.8)]
        #        lines_ticks_loc  = 'default'
            cbar_ticks_fmt    = '{%.0f}'
            lines_ticks_fmt   = '{%.0f}'
            lines_width       = line_width
            lines_ticks_color = 'k'
            lines_style       = '-'
            [CS,CS2] = contour_2D (ax6, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, Te_plot[:,:,ii], nodes_flag, log_type, auto, 
                                   min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                   nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                   lines_ticks_fmt, lines_ticks_color, lines_style, lines_width) 
        #    ax6.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
            ax6.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax6.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)          
        #    ax6.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)        
            ax6.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
            
            
        elif video_type == 2:
            
            # Plot the plasma density contour
            ax3.set_title(r'$n_{e}$ (m$^{-3}$)', fontsize = font_size,y=1.02)
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
            [CS,CS2] = contour_2D (ax3, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, ne_plot[:,:,ii], nodes_flag, log_type, auto, 
                              min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                              nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                              lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
        #    # Isolines ticks (exponent)
        #    ax3.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
            ax3.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax3.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
        #    ax3.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
            ax3.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
            
            # Plot the neutral density contour
            ax4.set_title(r'$n_{n}$ (m$^{-3}$)', fontsize = font_size,y=1.02)
            ax = plt.gca()
            log_type         = 1
            auto             = 0
            min_val0         = 1E14
            max_val0         = 1E20
            cont             = 1
            lines            = 0
            cont_nlevels     = nlevels_cont
            auto_cbar_ticks  = 1 
            auto_lines_ticks = -1
            nticks_cbar      = 4
            nticks_lines     = 10
            cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
            lines_ticks      = np.array([3E19,1.5E19,1E19,5E18,1E18,5E17,2E17,1E17,5E16,3E16])
            lines_ticks_loc  = 'default'
            cbar_ticks_fmt    = '{%.2f}'
            lines_ticks_fmt   = '{%.1f}'
            lines_width       = line_width
            lines_ticks_color = 'k'
            lines_style       = '-'
            [CS,CS2] = contour_2D (ax4, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, nn1_plot[:,:,ii], nodes_flag, log_type, auto, 
                              min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                              nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                              lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
        #    # Isolines ticks (exponent)
        #    ax4.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, fontsize=ticks_size_isolines, zorder = 1)
            ax4.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax4.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            ax4.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
            
            # Plot the electric potential contour
            ax5.set_title(r'$\phi$ (V)', fontsize = font_size,y=1.02)
            log_type         = 0
            auto             = 0
            min_val0         = -110
            max_val0         = 320.0
            cont             = 1
            lines            = 0
            cont_nlevels     = nlevels_cont
            auto_cbar_ticks  = 0 
            auto_lines_ticks = -1
            nticks_cbar      = 5
            nticks_lines     = 10
            cbar_ticks       = np.array([300, 250, 200, 150, 100, 50, 0, -50, -100])
        #        lines_ticks      = np.array([305, 280, 250, 200, 150, 100, 75, 50, 40, 30, 25, 15, 10, 5, 1, 0, -1, -2, -3, -4, -5, -6])
            lines_ticks      = np.array([305, 280, 250, 200, 150, 100, 75, 50, 40, 25, 15, 10, 5])
            lines_ticks_loc  = 'default'
            cbar_ticks_fmt    = '{%.0f}'
            lines_ticks_fmt   = '{%.0f}'
            lines_width       = line_width
            lines_ticks_color = 'w'
            lines_style       = '-'
            [CS,CS2] = contour_2D (ax5, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, phi_plot[:,:,ii], nodes_flag, log_type, auto, 
                                   min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                   nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                   lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)     
        #        ax5.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, fontsize=ticks_size_isolines, zorder = 1)
            ax5.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax5.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)    
        #        ax5.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
            ax5.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)     
            
            
            # Plot the electric field contour
            ax6.set_title(r"$|\boldsymbol{E}|$ (Vm$^{-1}$)", fontsize = font_size,y=1.02)
            log_type         = 1
            auto             = 0
            min_val0         = 1E0
            max_val0         = 1E4
            cont             = 1
            lines            = 0
            cont_nlevels     = nlevels_cont
            auto_cbar_ticks  = 1 
            auto_lines_ticks = -1
            nticks_cbar      = 4
            nticks_lines     = 6
            cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
    #        lines_ticks      = np.array([0,200,500,1000,2000,3000,4000,20000,60000])*1E-3
    #        lines_ticks      = np.array([0,200,500,1000,2000,3000,4000,20000,60000])
            lines_ticks      = np.array([200,500,2000,4000,10000])
            lines_ticks_loc  = [(0.7,4.25),(1.6,4.25),(3.0,3.8),(4.0,4.25),(4.7,5.0),(5.3,3.5),(3.46,7.5),(3.6,0.7),(5.7,3.0),(7.9,1.8),(7.7,5.5),(9.5,2.9)]
    #        lines_ticks_loc  = 'default'
            cbar_ticks_fmt    = '{%.2f}'
            lines_ticks_fmt   = '{%.1f}'
            lines_width       = line_width
            lines_ticks_color = 'k'
            lines_style       = '-'
            [CS,CS2] = contour_2D (ax6, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, Efield_plot_cont[:,:,ii], nodes_flag, log_type, auto, 
                                   min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                   nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                   lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)   
             # Isolines ticks (exponent)
    #        ax6.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
            ax6.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax6.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            ax6.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)   
    #        ax.tick_params(axis='x',pad=xticks_pad)
            
            field_deltar = 1
            field_deltaz = 2
            Q1 = ax6.quiver(zs[::field_deltar,::field_deltaz],rs[::field_deltar,::field_deltaz],
                            Ez_plot_cont[::field_deltar,::field_deltaz,ii]/Efield_plot_cont[::field_deltar,::field_deltaz,ii],
                            Er_plot_cont[::field_deltar,::field_deltaz,ii]/Efield_plot_cont[::field_deltar,::field_deltaz,ii],
                            width = 0.004, pivot='mid', color='w',scale = 1/0.035)  
                            
                            
        elif video_type == 3:
            
            # Plot the i1 density contour
            ax3.set_title(r'$n_{i1}$ (m$^{-3}$)', fontsize = font_size,y=1.02)
            log_type         = 1
            auto             = 0
            min_val0         = 1E13
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
            [CS,CS2] = contour_2D (ax3, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, ni1_plot[:,:,ii], nodes_flag, log_type, auto, 
                              min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                              nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                              lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
        #    # Isolines ticks (exponent)
        #    ax3.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
            ax3.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax3.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
        #    ax3.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
            ax3.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
            
            
            # Plot the i2 density contour
            ax4.set_title(r'$n_{i2}$ (m$^{-3}$)', fontsize = font_size,y=1.02)
            log_type         = 1
            auto             = 0
            min_val0         = 1E13
            max_val0         = 1E18
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
            [CS,CS2] = contour_2D (ax4, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, ni2_plot[:,:,ii], nodes_flag, log_type, auto, 
                              min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                              nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                              lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
        #    # Isolines ticks (exponent)
        #    ax3.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
            ax4.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax4.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
        #    ax3.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
            ax4.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
            
            
            # Plot the i2/i1 density ratio contour
            ax5.set_title(r'$n_{i2}/n_{i1}$ (-)', fontsize = font_size,y=1.02)
            log_type         = 1
            auto             = 0
            min_val0         = 1E-3
            max_val0         = 1E0
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
            [CS,CS2] = contour_2D (ax5, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, 1.0/ratio_ni1_ni2_plot[:,:,ii], nodes_flag, log_type, auto, 
                              min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                              nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                              lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
        #    # Isolines ticks (exponent)
        #    ax5.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
            ax5.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax5.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
        #    ax5.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
            ax5.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
                        
                
            # Plot the electric field contour
            ax6.set_title(r"$|\boldsymbol{E}|$ (Vm$^{-1}$)", fontsize = font_size,y=1.02)
            log_type         = 1
            auto             = 0
            min_val0         = 1E0
            max_val0         = 1E4
            cont             = 1
            lines            = 0
            cont_nlevels     = nlevels_cont
            auto_cbar_ticks  = 1 
            auto_lines_ticks = -1
            nticks_cbar      = 4
            nticks_lines     = 6
            cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
    #        lines_ticks      = np.array([0,200,500,1000,2000,3000,4000,20000,60000])*1E-3
    #        lines_ticks      = np.array([0,200,500,1000,2000,3000,4000,20000,60000])
            lines_ticks      = np.array([200,500,2000,4000,10000])
            lines_ticks_loc  = [(0.7,4.25),(1.6,4.25),(3.0,3.8),(4.0,4.25),(4.7,5.0),(5.3,3.5),(3.46,7.5),(3.6,0.7),(5.7,3.0),(7.9,1.8),(7.7,5.5),(9.5,2.9)]
    #        lines_ticks_loc  = 'default'
            cbar_ticks_fmt    = '{%.2f}'
            lines_ticks_fmt   = '{%.1f}'
            lines_width       = line_width
            lines_ticks_color = 'k'
            lines_style       = '-'
            [CS,CS2] = contour_2D (ax6, cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, Efield_plot_cont[:,:,ii], nodes_flag, log_type, auto, 
                                   min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                   nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                   lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)   
             # Isolines ticks (exponent)
    #        ax6.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
            ax6.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            ax6.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            ax6.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)   
    #        ax.tick_params(axis='x',pad=xticks_pad)
            
            field_deltar = 1
            field_deltaz = 2
            Q1 = ax6.quiver(zs[::field_deltar,::field_deltaz],rs[::field_deltar,::field_deltaz],
                            Ez_plot_cont[::field_deltar,::field_deltaz,ii]/Efield_plot_cont[::field_deltar,::field_deltaz,ii],
                            Er_plot_cont[::field_deltar,::field_deltaz,ii]/Efield_plot_cont[::field_deltar,::field_deltaz,ii],
                            width = 0.004, pivot='mid', color='w',scale = 1/0.035)  
    
                    
        plt.tight_layout()
        if save_flag == 1:
            fig.savefig(path_out+str(index)+figs_format,bbox_inches='tight') 
            plt.close()  
    ###########################################################################    



        
