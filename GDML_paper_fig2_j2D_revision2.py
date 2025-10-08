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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
# from contour_2D import contour_2D
from GDML_paper_contour_2D_fig2 import contour_2D_fig2
from streamlines_function import streamplot, streamline_2D
from HET_sims_read import HET_sims_read
from HET_sims_mean import HET_sims_mean
from HET_sims_plotvars import HET_sims_plotvars
from HET_sims_plotvars import HET_sims_cp_vars
from HET_sims_post import max_min_mean_vals, comp_phase_shift, comp_FFT, comp_Boltz, domain_average
from FFT import FFT
from HET_sims_interp_MFAM_picM_plot import HET_sims_interp_MFAM_picM_plot
from HET_sims_interp_scan import HET_sims_interp_scan
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

save_flag = 1

    
# Plots save flag
# figs_format = ".eps"
figs_format = ".png"
figs_format2 = ".pdf"

# Plots to produce
ref_case_plots      = 1


# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/fig2_paper/figs_chamber_and_plume/new_sims/revision1/"
# path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/fig2_paper/figs_only_plume/new_sims/revision1/transposed/"
path_out = "VHT_MP_US_plume_sims/paper_GDML_figs/fig2_paper/figs_only_plume/new_sims/revision2/"



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
    








if ref_case_plots == 1:
    print("######## ref_case_plots ########")
#          
#    text_size           = text_size-8
#    ticks_size          = ticks_size-8
#    font_size           = font_size-8
    # text_size           = 20
    # ticks_size          = 20
    # font_size           = 20
    
    text_size           = 25
    ticks_size          = 25
    font_size           = 25
    marker_size         = 10 

#    ticks_size_isolines = 20
    ticks_size_isolines = 10
    marker_every = 3
    
    nlevels_2Dcontour = 100
    
    marker_size_fsplit = 5
    line_width_fsplit  = 0.5


    rind       = 15    # VHT_US MP and LP fine mesh Np and PPSX00
    rind_anode1 = rind
    rind_anode2 = 17
    zind_anode  = 8
    elems_cath_Bline = []
    
    # Cathode plotting flag and cathode position in cm (for plot_zcath_012 = 2) 
    plot_zcath_012     = 1
    zcat_pos           = 5.9394542444501024 # In prof figures if plot_zcath_012 = 2
    plot_cath_contours = 1
#    plot_cath_contours = 0
    cath_marker_color = 'w'
    cath_marker_edge_width = 0.5
    cath_marker_edge_color = 'k'
    
    # Plot cathode magnetic line 
    plot_cath_Bline = 0
    cath_Bline_color = 'g'
    
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
    last_steps      = 1200
#    last_steps      = 1000
    step_i          = 900
    step_f          = 1050
    plot_mean_vars  = 1
    
    # Flag to decide if interpolate from MFAM to a finer picM for plotting phi, Te and je components (Recommended = 1)
    interp_MFAM_picM_plot = 1
    
    # Flag to decide if plotting the angular profile points (Faraday scan)
    plot_scan_points    = 0
    # Options for plotting angular profiles in plume (Faraday probe scans)
    z_offset    = -2.9 # Distance (cm) from anode to axial position of the axis of Faraday probe scan
    r_offset    = 6.56  # Offset radius (cm) of the axis of the Faraday probe scan
    rscan       = 6.56  # Radius (cm) of the Faraday probe scan 
    ang_min     = -90   # Minimum angle for the profile (deg)
    ang_max     = 90  # Maximum angle for the profile (deg)
    Npoints_ang = 200 # Number of points for the profile 
    

    plot_curr_ref       = 1
    
    
    Bline_all2Dplots = 1
    cath2D_plots     = 0
    cath2D_title     = r"(b)"
    
    vline = 0         # Draw a vertical line on P3 and P4 corresponding to vertical P boundary of P2 and P3, respectively
    color_vline = 'w'
    
    only_plume = 1    # Only plot 2D maps in plume
    

    if allsteps_flag == 0:
        mean_vars = 0

    
    # Simulation names
    nsims = 6   


    # Flag for old sims (1: old sim files, 0: new sim files)    
    oldpost_sim      = np.array([6,6,6,6,6,6,6,6,6,6,6,6],dtype = int)
    oldsimparams_sim = np.array([21,21,21,21,21,21,21,21,20,20,20,20],dtype = int)   
    
    
    sim_names = [
        
                "../../../sim/sims/P1L_Tcath_new",
                "../../../sim/sims/P1G_Tcath_new",
                
                "../../../sim/sims/P2L_Tcath_new",
                "../../../sim/sims/P2G_Tcath_new",
                
                "../../../sim/sims/P3L_Tcath_new",
                "../../../sim/sims/P3G_Tcath_new",
                
                "../../../sim/sims/P4L_Fz_Tcath_new",
                "../../../sim/sims/P4G_Fz_Tcath_new",
                
                # "../../../sim/sims/P1L",
                # "../../../sim/sims/P1G",
                
                # "../../../sim/sims/P2L",
                # "../../../sim/sims/P2G",
                
                # "../../../sim/sims/P3L",
                # "../../../sim/sims/P3G",
                
                # "../../../sim/sims/P4L",
                # "../../../sim/sims/P4G",

                 ] 

    PIC_mesh_file_name = [
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
                               ]


    # Labels             
    labels = [r"A",r"B",r"C",r"D",r"Average"]
    
    # Line colors
    colors = ['r','g','b','m','k','m','y']
    # Markers
    markers = ['^','>','v', '<', 's', 'o','*']
    # Line style
#    linestyles = ['-','--','-.', ':','-','--','-.']
    linestyles = ['-','-','-', '-','-','-','-']
    
#    cont_xlabel = '$z$ (cm)'
#    cont_ylabel = '$r$ (cm)'
    
    cont_xlabel = '$z/L_\mathrm{c}$'
    cont_ylabel = '$r/H_\mathrm{c}$'
              
    # Profile and contour plots
    
        
    if plot_curr_ref == 1:
        
        
        # [fig1, axes1] = plt.subplots(nrows=4, ncols=2, figsize=(15,24))
        # [fig2, axes2] = plt.subplots(nrows=4, ncols=2, figsize=(15,24))
        # [fig3, axes3] = plt.subplots(nrows=4, ncols=2, figsize=(15,24))
        
        # [fig1, axes1] = plt.subplots(nrows=3, ncols=2, figsize=(15,18))
        # [fig2, axes2] = plt.subplots(nrows=3, ncols=2, figsize=(15,18))
        # [fig3, axes3] = plt.subplots(nrows=3, ncols=2, figsize=(15,18))
        
        [fig1, axes1] = plt.subplots(nrows=2, ncols=3, figsize=(22.5,12))
        [fig2, axes2] = plt.subplots(nrows=2, ncols=3, figsize=(22.5,12))
        [fig3, axes3] = plt.subplots(nrows=2, ncols=3, figsize=(22.5,12))
        # ax1 = plt.subplot2grid( (4,2), (0,0) )
        # ax2 = plt.subplot2grid( (4,2), (0,1) )
        # ax3 = plt.subplot2grid( (4,2), (1,0) )
        # ax4 = plt.subplot2grid( (4,2), (1,1) )
        # ax5 = plt.subplot2grid( (4,2), (2,0) )
        # ax6 = plt.subplot2grid( (4,2), (2,1) )
        # ax7 = plt.subplot2grid( (4,2), (3,0) )
        # ax8 = plt.subplot2grid( (4,2), (3,1) )
        
        mappable = None
        
    ######################## READ INPUT/OUTPUT FILES ##########################
    for k in range(0,nsims):
        ind_ini_letter = sim_names[k].rfind('/') + 1
        print("##### CASE "+str(k+1)+": "+sim_names[k][ind_ini_letter::]+" #####")
        # Obtain paths to simulation files
        path_picM         = sim_names[k]+"/SET/inp/"+PIC_mesh_file_name[k]
        path_picM_plot    = sim_names[k]+"/SET/inp/"+PIC_mesh_plot_file_name[k]
        path_simstate_inp = sim_names[k]+"/CORE/inp/SimState.hdf5"
        # --------------
        path_simstate_out = sim_names[k]+"/CORE/out/SimState.hdf5"
        path_postdata_out = sim_names[k]+"/CORE/out/PostData.hdf5"
        path_simparams_inp = sim_names[k]+"/CORE/inp/sim_params.inp"
    
        # --------------
        
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
            ue_perp_mp       = -je_perp_mp/(e*ne_mp)
            ue_theta_mp      = -je_theta_mp/(e*ne_mp)
            ue_para_mp       = -je_para_mp/(e*ne_mp)
            ue_z_mp          = -je_z_mp/(e*ne_mp)
            ue_r_mp          = -je_r_mp/(e*ne_mp)
            ue_mp            = np.sqrt(ue_r_mp**2 +ue_theta_mp**2 + ue_z_mp**2)
            Ekin_e_mp        = 0.5*me*ue_mp**2/e
            ratio_Ekin_Te_mp = Ekin_e_mp/Te_mp
            
        # Obtain angular profiles (Faraday probe scan) if required
        if plot_scan_points == 1:
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
        ji_x_plot               = ji1_x_plot + ji2_x_plot + ji3_x_plot + ji4_x_plot
        ji_y_plot               = ji1_y_plot + ji2_y_plot + ji3_y_plot + ji4_y_plot
        ji_z_plot               = ji1_z_plot + ji2_z_plot + ji3_z_plot + ji4_z_plot
        ji_plot                 = np.sqrt( ji_x_plot**2 + ji_y_plot**2 + ji_z_plot**2 )
        ji1_plot                = np.sqrt( ji1_x_plot**2 + ji1_y_plot**2 + ji1_z_plot**2 )
        ji2_plot                = np.sqrt( ji2_x_plot**2 + ji2_y_plot**2 + ji2_z_plot**2 )
        uimean_x_plot           = ji_x_plot/(e*ne_plot)
        uimean_y_plot           = ji_y_plot/(e*ne_plot)
        uimean_z_plot           = ji_z_plot/(e*ne_plot)
        uimean_plot             = np.sqrt( uimean_x_plot**2 + uimean_y_plot**2 + uimean_z_plot**2 )
        Z_avg                   = 1.0*ni1_plot/ne_plot + 2.0*ni2_plot/ne_plot + 1.0*ni3_plot/ne_plot + 2.0*ni4_plot/ne_plot
        cs_plot                 = np.sqrt(e*Z_avg*Te_plot/mass)
        Mi_plot                 = uimean_plot/cs_plot
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
        erel_jr_plot            = np.abs(je_r_plot+ji_x_plot)/np.abs(ji_x_plot)
        erel_jz_plot            = np.abs(je_z_plot+ji_z_plot)/np.abs(ji_z_plot)
        erel_r_plot            = np.abs(je_r_plot+ji_x_plot)/np.abs(ji_x_plot)
        ratio_ue_t_perp_plot    = ue_t_plot/ue_perp_plot
        ratio_ue_t_para_plot    = ue_t_plot/ue_para_plot
        ratio_ue_perp_para_plot = ue_perp_plot/ue_para_plot
        nu_ei_el_tot_plot       = nu_ei1_plot + nu_ei2_plot
        nu_ion_tot_plot         = nu_i01_plot + nu_i02_plot + nu_i12_plot
        lambdaD_plot            = np.sqrt(eps0*(e*Te_plot)/(ne_plot*e**2))
        nn_plot                 = nn1_plot + nn2_plot + nn3_plot
        pn1_plot                = nn1_plot*Tn1_plot*e*1E-2 # Neutral pressure in mbar (1Pa = 1E-2 mbar)
        pn2_plot                = nn2_plot*Tn2_plot*e*1E-2 # Neutral pressure in mbar (1Pa = 1E-2 mbar)
        pn3_plot                = nn3_plot*Tn3_plot*e*1E-2 # Neutral pressure in mbar (1Pa = 1E-2 mbar)
        pn_plot                 = pn1_plot + pn2_plot + pn3_plot
        ratio_ni1_ni2_plot      = np.divide(ni2_plot,ni1_plot)
        ratio_ni1_ni3_plot      = np.divide(ni3_plot,ni1_plot)
        ratio_ni1_ni4_plot      = np.divide(ni4_plot,ni1_plot)
        ratio_ne_neCEX_plot     = np.divide(ni3_plot + 2*ni4_plot,ne_plot)
        ratio_nn1_nn2_plot      = np.divide(nn2_plot,nn1_plot)
        ratio_nn1_nn3_plot      = np.divide(nn3_plot,nn1_plot)
        ratio_nn1_nnCEX_plot    = np.divide(nn2_plot+nn3_plot,nn1_plot)
        ratio_nn_nnCEX_plot     = np.divide(nn2_plot+nn3_plot,nn_plot)
        
        ue                      = np.sqrt(ue_r**2 +ue_t**2 + ue_z**2)
        ratio_Ekin_Te           = (0.5*me*ue**2/e)/Te
        ji_x                    = ji1_x + ji2_x + ji3_x + ji4_x
        ji_y                    = ji1_y + ji2_y + ji3_y + ji4_y
        ji_z                    = ji1_z + ji2_z + ji3_z + ji4_z
        j_r                     = ji_x + je_r
        j_t                     = ji_y + je_t
        j_z                     = ji_z + je_z
        je2D                    = np.sqrt(je_r**2 + je_z**2)
        ji2D                    = np.sqrt(ji_x**2 + ji_z**2)
        j2D                     = np.sqrt(j_r**2 + j_z**2)
        lambdaD                 = np.sqrt(eps0*(e*Te)/(ne*e**2))
        
        f_split_q_plot     = f_split_qperp_plot + f_split_qpara_plot + f_split_qb_plot
        f_split_eflux_plot = f_split_adv_plot + f_split_q_plot
        f_split_P_plot     = f_split_Pperp_plot + f_split_Ppara_plot
        f_split_LHS_plot   = f_split_eflux_plot + f_split_P_plot + f_split_ecterm_plot + f_split_inel_plot
        
        # Obtain effective Hall parameter in plume volume
        Hall_par_effect_plot = np.sqrt(Hall_par_plot*Hall_par_eff_plot)
        hall_effect_mean     = 0.0
        hall_eff_mean        = 0.0
        hall_mean            = 0.0
        pn_mean             = 0.0
        cells_vol_tot = 0.0
        for icell in range(0,dims[0]-1):
            for jcell in range(int(xi_bottom),dims[1]-1):
                hall_effect_cell = 0.25*(Hall_par_effect_plot[icell,jcell] + Hall_par_effect_plot[icell,jcell+1] + Hall_par_effect_plot[icell+1,jcell+1] +Hall_par_effect_plot[icell+1,jcell])
                hall_eff_cell    = 0.25*(Hall_par_eff_plot[icell,jcell] + Hall_par_eff_plot[icell,jcell+1] + Hall_par_eff_plot[icell+1,jcell+1] +Hall_par_eff_plot[icell+1,jcell])
                hall_cell        = 0.25*(Hall_par_plot[icell,jcell] + Hall_par_plot[icell,jcell+1] + Hall_par_plot[icell+1,jcell+1] +Hall_par_plot[icell+1,jcell])
                hall_effect_mean = hall_effect_mean + hall_effect_cell*cells_vol[icell,jcell]
                hall_eff_mean    = hall_eff_mean + hall_eff_cell*cells_vol[icell,jcell]
                hall_mean        = hall_mean + hall_cell*cells_vol[icell,jcell]
                pn_cell          = 0.25*(pn_plot[icell,jcell] + pn_plot[icell,jcell+1] + pn_plot[icell+1,jcell+1] +pn_plot[icell+1,jcell])
                pn_mean          = pn_mean + pn_cell*cells_vol[icell,jcell]
                cells_vol_tot    = cells_vol_tot + cells_vol[icell,jcell]
        hall_effect_mean = hall_effect_mean/cells_vol_tot
        hall_eff_mean    = hall_eff_mean/cells_vol_tot
        hall_mean        = hall_mean/cells_vol_tot
        pn_mean          = pn_mean/cells_vol_tot
        
        pos_tol = 5
        pos_null_j2D_point = np.where(j2D_plot == np.nanmin(np.nanmin(j2D_plot[pos_tol:dims[0]-pos_tol,int(xi_bottom)+pos_tol:-pos_tol:1])))
        z_null_j2D_point = zs[pos_null_j2D_point][0]
        r_null_j2D_point = rs[pos_null_j2D_point][0]
        j2D_null_point   = j2D_plot[pos_null_j2D_point][0]
    
    

        ###########################################################################
        print("Plotting...")
        ############################ GENERATING PLOTS #############################
        if interp_MFAM_picM_plot == 1:
            zs_mp                = zs_mp*1E2
            rs_mp                = rs_mp*1E2
        zs                = zs*1E2
        rs                = rs*1E2
        zscells           = zscells*1E2
        rscells           = rscells*1E2
        points            = points*1E2
        z_cath            = z_cath*1E2
        r_cath            = r_cath*1E2
        nodes[0,:]        = nodes[0,:]*1e2
        nodes[1,:]        = nodes[1,:]*1e2
        elem_geom[0,:]    = elem_geom[0,:]*1E2
        elem_geom[1,:]    = elem_geom[1,:]*1E2
        face_geom[0,:]    = face_geom[0,:]*1E2
        face_geom[1,:]    = face_geom[1,:]*1E2
        Ez_plot_anode     = np.copy(Ez_plot)*1E-3
        je_z_plot_anode   = np.copy(je_z_plot)*1E-3
        ji_z_plot_anode   = np.copy(ji_z_plot)*1E-2
        # B field in Gauss
        Bfield_plot       = Bfield_plot*1E4
        Br_plot           = Br_plot*1E4
        Bz_plot           = Bz_plot*1E4 
        Efield            = Efield*1E-3
        Ez                = Ez*1E-3
        je_para           = je_para*1E-4    # This is A/cm2
        je_perp           = je_perp*1E-4    # This is A/cm2
        je_t              = je_t*1E-4       # This is A/cm2
        ji_x              = ji_x*1E-4       # This is A/cm2
        ji_z              = ji_z*1E-4       # This is A/cm2
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
        ji_x_plot         = ji_x_plot*1E-4    # This is A/cm2  
        ji_z_plot         = ji_z_plot*1E-4    # This is A/cm2
        je2D_plot         = je2D_plot*1E-4    # This is A/cm2
        ji2D_plot         = ji2D_plot*1E-4    # This is A/cm2
        j2D_plot          = j2D_plot*1E-4     # This is A/cm2
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
        
        if plot_scan_points == 1:
            z_scan = z_scan*1E2
            r_scan = r_scan*1E2
            
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
        
        #Limit minium ionization frequencies
        min_nu = 1E-2
        nu_i01_plot[nu_i01_plot<min_nu] = min_nu
        nu_i02_plot[nu_i02_plot<min_nu] = min_nu
        nu_i12_plot[nu_i12_plot<min_nu] = min_nu
        nu_ion_tot_plot[nu_ion_tot_plot<min_nu] = min_nu
            
    
        # Obtain cathode Bline
        lambda_tol  = 1e-4
        cath_Bline_IDs = np.zeros(0,dtype=int)
        cath_Bline_z   = np.zeros(0,dtype=float)
        cath_Bline_r   = np.zeros(0,dtype=float)
        cath_Bline_d   = np.zeros(0,dtype=float)
        cath_Bline_IDs = np.append(cath_Bline_IDs,np.array([cath_elem[0]]),axis=0)
        if cath_type == 1:
            cath_lambda = face_geom[2,cath_elem[0]]
            cath_Bline_z = np.append(cath_Bline_z,np.array([face_geom[0,cath_elem[0]]]),axis=0)
            cath_Bline_r = np.append(cath_Bline_r,np.array([face_geom[1,cath_elem[0]]]),axis=0)
        elif cath_type == 2:
            cath_lambda = elem_geom[2,cath_elem[0]]
            cath_Bline_z = np.append(cath_Bline_z,np.array([elem_geom[0,cath_elem[0]]]),axis=0)
            cath_Bline_r = np.append(cath_Bline_r,np.array([elem_geom[1,cath_elem[0]]]),axis=0)
        cath_Bline_d = np.append(cath_Bline_d,np.array([0.0]),axis=0)
        
        for i in range(0,n_elems):
            if abs(elem_geom[2,i]-cath_lambda) < abs(cath_lambda)*lambda_tol:
                cath_Bline_IDs = np.append(cath_Bline_IDs,np.array([i]),axis=0)
                cath_Bline_z = np.append(cath_Bline_z,np.array([elem_geom[0,i]]),axis=0)
                cath_Bline_r = np.append(cath_Bline_r,np.array([elem_geom[1,i]]),axis=0)
                dcath = np.sqrt((cath_Bline_z[-1] - cath_Bline_z[0])**2 + (cath_Bline_r[-1] - cath_Bline_r[0])**2)
                cath_Bline_d = np.append(cath_Bline_d,np.array([dcath]),axis=0)
    
        cath_Bline_npoints = len(cath_Bline_IDs)
        pos_sort  = np.argsort(cath_Bline_d)
        cath_Bline_d = cath_Bline_d[pos_sort]
        cath_Bline_IDs = cath_Bline_IDs[pos_sort]
        cath_Bline_z   = cath_Bline_z[pos_sort]
        cath_Bline_r   = cath_Bline_r[pos_sort]
        
        
    #    # Do not plot units in axes
    #    # SAFRAN CHEOPS 1: units in cm
    ##    L_c = 3.725
    ##    H_c = (0.074995-0.052475)*100
    #    # HT5k: units in cm
    #    L_c = 2.53
    #    H_c = (0.0785-0.0565)*100
        # VHT_US (IEPC 2022)
        L_c = 2.9
        H_c = 2.22    
        # VHT_US PPSX00 testcase1 LP (TFM Alejandro)
    #    L_c = 2.5
    #    H_c = 1.1
        # PPSX00 testcase2 LP
    #    L_c = 2.5
    #    H_c = 1.5
        
        if interp_MFAM_picM_plot == 1:
            zs_mp = zs_mp/L_c
            rs_mp = rs_mp/H_c
        if plot_scan_points == 1:
            z_scan = z_scan/L_c
            r_scan = r_scan/H_c
        zs = zs/L_c
        rs = rs/H_c
        points[:,0] = points[:,0]/L_c
        points[:,1] = points[:,1]/H_c
        z_cath = z_cath/L_c
        r_cath = r_cath/H_c
        zscells = zscells/L_c
        rscells = rscells/H_c
        cath_Bline_z = cath_Bline_z/L_c
        cath_Bline_r = cath_Bline_r/H_c
        
        nodes[0,:]        = nodes[0,:]/L_c
        nodes[1,:]        = nodes[1,:]/H_c
        elem_geom[0,:]    = elem_geom[0,:]/L_c
        elem_geom[1,:]    = elem_geom[1,:]/H_c
        face_geom[0,:]    = face_geom[0,:]/L_c
        face_geom[1,:]    = face_geom[1,:]/H_c
        
        
        # Values for IEPC22
        print("hall_effect_mean in plume = %15.8e" %hall_effect_mean)
        print("hall_eff_mean in plume    = %15.8e" %hall_eff_mean)
        print("hall_mean in plume        = %15.8e" %hall_mean)
        print("z_null_j2D_point in plume = %15.8e" %(z_null_j2D_point*1E2/L_c))
        print("r_null_j2D_point in plume = %15.8e" %(r_null_j2D_point*1E2/H_c))
        print("j2D_null_point in plume   = %15.8e" %j2D_null_point)
        print("pn_mean in plume (mbar)   = %15.8e" %pn_mean)
    
    
        # Colors for plotting the MFAM
        lambda_color   = 'b'
        sigma_color    = 'r'
        boundary_color = 'k'
        
        # ylim_axes = 18.2
        # xlim_axes = 15.0
        ylim_axes = 16.0
        xlim_axes = 11.344827586206897
        yticks_lim = ylim_axes
    #    yticks_lim = rs[-1,-1]+1
    

        # Obtain the vectors for the uniform mesh for streamlines plotting. It must be uniform and squared mesh
        delta_x = 0.11     # Plot dimensional axes
        delta_x = 0.11/L_c # Plot non-dimensional axes
        delta_x = 0.11/H_c # Plot non-dimensional axes
        delta_x_inC = delta_x
        Npoints_stream = 1000
        Npoints_stream_inC = 1000
        # VHT_US MP sims (IEPC 2022)
        rvec = np.linspace(rs[0,-1],rs[-1,-1],Npoints_stream)
        zvec = np.copy(rvec)
        zvec_inC = np.linspace(zs[0,0],zs[0,int(xi_bottom)+5],Npoints_stream_inC)
        rvec_inC = np.copy(zvec_inC) + 2.4
        
        
        # Axes ticks flag (1: plot ticks 1 by 1, 2: plot ticks 1 by 1 with fixed limits)
        ax_ticks_flag = 2
        
        ind1 = k%2

        if k == 0 or k == 1:
            ind2 = 0
        elif k == 2 or k == 3:
            ind2 = 1
        elif k == 4 or k == 5:
            ind2 = 2
        elif k == 6 or k == 7:
            ind2 = 3
            
        # if ind2 == 0:
        #     if ind1 == 0:
        #         text_title = r"(a) "
        #     elif ind1 == 1:
        #         text_title = r"(c) "
        #     elif ind1 == 2:
        #         text_title = r"(e) "
        #     elif ind1 == 3:
        #         text_title = r"(g) "
        # elif ind2 == 1:
        #     if ind1 == 0:
        #         text_title = r"(b) "
        #     elif ind1 == 1:
        #         text_title = r"(d) "
        #     elif ind1 == 2:
        #         text_title = r"(f) "
        #     elif ind1 == 3:
        #         text_title = r"(h) "
        
        
        if ind1 == 0:
            if ind2 == 0:
                text_title = r"(a) LP1C1"
            elif ind2 == 1:
                text_title = r"(c) LP2C1"
            elif ind2 == 2:
                text_title = r"(e) LP3C1"
            elif ind2 == 3:
                text_title = r"(g) LP4C1"
        elif ind1 == 1:
            if ind2 == 0:
                text_title = r"(b) GP1C1"
            elif ind2 == 1:
                text_title = r"(d) GP2C1"
            elif ind2 == 2:
                text_title = r"(f) GP3C1"
            elif ind2 == 3:
                text_title = r"(h) GP4C1"
        
        
        # if ind2 == 0:
        #     if ind1 == 0:
        #         text_title = r"(a) LP1C1"
        #     elif ind1 == 1:
        #         text_title = r"(c) LP2C1"
        #     elif ind1 == 2:
        #         text_title = r"(e) LP3C1"
        #     elif ind1 == 3:
        #         text_title = r"(g) LP4C1"
        # elif ind2 == 1:
        #     if ind1 == 0:
        #         text_title = r"(b) GP1C1"
        #     elif ind1 == 1:
        #         text_title = r"(d) GP2C1"
        #     elif ind1 == 2:
        #         text_title = r"(f) GP3C1"
        #     elif ind1 == 3:
        #         text_title = r"(h) GP4C1"
            
        print(k,ind1,ind2)
        

           
           
                
        if plot_curr_ref == 1:
            
            zstext = 0.5
            rstext = 0.95
            # text = text_title+r"$-\tilde{\boldsymbol{\jmath}}_\mathrm{e}$ (Acm$^{-2}$)"
            text = text_title
            axes1[ind1,ind2].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes1[ind1,ind2].transAxes) 
            # text = text_title+r"$-\tilde{\boldsymbol{\jmath}}_\mathrm{i}$ (Acm$^{-2}$)"
            text = text_title
            axes2[ind1,ind2].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes2[ind1,ind2].transAxes) 
            # text = text_title+r"$\tilde{\boldsymbol{\jmath}}$ (Acm$^{-2}$)"
            text = text_title
            axes3[ind1,ind2].text(zstext,rstext,text,fontsize = text_size,color='k',ha='center',va='center',transform=axes3[ind1,ind2].transAxes) 
            
            # axes1[ind1,ind2].set_title(text_title+r"$-\tilde{\boldsymbol{\jmath}}_\mathrm{e}$ (Acm$^{-2}$)", fontsize = font_size,y=1.02)
            # axes2[ind1,ind2].set_title(text_title+r"$\tilde{\boldsymbol{\jmath}}_\mathrm{i}$ (Acm$^{-2}$)", fontsize = font_size,y=1.02)
            # axes3[ind1,ind2].set_title(text_title+r"$\tilde{\boldsymbol{\jmath}}$ (Acm$^{-2}$)", fontsize = font_size,y=1.02)
            
            # axes1[ind1,ind2].set_title(text_title+r"$-\tilde{\boldsymbol{\jmath}}_\mathrm{e}$ (Acm$^{-2}$)", fontsize = font_size,y=1.02)
            # axes2[ind1,ind2].set_title(text_title+r"$\tilde{\boldsymbol{\jmath}}_\mathrm{i}$ (Acm$^{-2}$)", fontsize = font_size,y=1.02)
            # axes3[ind1,ind2].set_title(text_title+r"$\tilde{\boldsymbol{\jmath}}$ (Acm$^{-2}$)", fontsize = font_size,y=1.02)
            
            
                
            # # Delete vars to free RAM memory
            # print("delete")
            # del je_para_mp, je_perp_mp
    
            log_type         = 1
            auto             = 0
    #        min_val0         = 1E-1
    #        max_val0         = 1E3
            min_val0         = 1E-5
            max_val0         = 1E0
            if only_plume == 1:
                max_val0         = 1E-1
            cont             = 1
            lines            = 0
            cont_nlevels     = nlevels_2Dcontour
            auto_cbar_ticks  = 1 
            auto_lines_ticks = -1
            nticks_cbar      = 5
            nticks_lines     = 10
            cbar_ticks       = np.array([0.0,0.2,0.3,0.5,0.7,0.8,0.9,1])
    #        lines_ticks      = np.array([8E1,1E2,5E2,1E3,2E3,5E3,1E4,2E4,3E4])
            lines_ticks      = np.array([1E2,5E2,1E3])
    #        lines_ticks_loc  = [(1.5,4.25),(3.4,4.25),(4.8,1.43),(7.56,7.4),(9.34,5.16),(8.45,0.6),(5.9,5.5),(4.0,6.4)]        
            lines_ticks_loc  = 'default'
            cbar_ticks_fmt    = '{%.1f}'
            lines_ticks_fmt   = '{%.2f}'
            lines_width       = line_width
            lines_ticks_color = 'k'
            lines_style       = '-'
            if interp_MFAM_picM_plot == 1:
                var = np.copy(je_2D_mp)
                if only_plume == 1:
                    var[np.where(zs_mp < zs[0,int(xi_bottom)])] = np.nan
                [CS,CS2] = contour_2D_fig2 (axes1[ind1,ind2],cont_xlabel, cont_ylabel, font_size, ticks_size, zs_mp, rs_mp, var, nodes_flag_mp, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width) 
            else:
                var = np.copy(je2D_plot)
                if only_plume == 1:
                    var[:,0:int(xi_bottom)] = np.nan
                [CS,CS2] = contour_2D_fig2 (axes1[ind1,ind2],cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, var, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width) 
                
    #         if k == 1 or k == 3 or k == 5:
    #         # if k == 1:
    #             # Generate colorbar if needed
    #             min_val_exp = np.floor(np.log10(min_val0))
    #             max_val_exp = np.floor(np.log10(max_val0))+1
    #             if auto_cbar_ticks == 1 or auto_cbar_ticks == 0:
    #                 # For the colorbar ticks, only if auto_cbar_ticks=1 we generate automatic
    #                 # ticks. Otherwise is given already in inputed cbar_ticks. Format for
    #                 # colorbar ticks is not selected in this case
    #                 if auto_cbar_ticks == 1:
    #                     cbar_nticks = np.int(max_val_exp - min_val_exp + 1)
    #                     ticks_exp   = np.linspace(min_val_exp,max_val_exp,cbar_nticks)
    #                     cbar_ticks  = np.power(10, ticks_exp)
    #                 CS.cmap.set_under('white')
    #                 CS.cmap.set_over('black')
    #                 cbar = plt.colorbar(CS,ticks=cbar_ticks,ax=axes1[ind1,ind2])
    # #                cbar = plt.colorbar(CS,ticks=cbar_ticks,format=ticker.FuncFormatter(fmt_func_exponent),ax=ax)
    #                 cbar.ax.tick_params(labelsize=ticks_size)
                
            # if mappable is None:
            #     CS.cmap.set_under('white')
            #     CS.cmap.set_over('black')
            #     mappable = CS
            
            
            
    #        ax.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, fontsize=ticks_size_isolines, zorder = 1)
    #        ax.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
                #        ax.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, fontsize=ticks_size_isolines)
    #        ax.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines)
            axes1[ind1,ind2].plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            axes1[ind1,ind2].plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            if plot_cath_contours == 1:
                axes1[ind1,ind2].plot(1.05*z_cath,r_cath,linewidth = line_width_boundary,markersize = marker_size,marker = 's',color = cath_marker_color,markeredgecolor = cath_marker_edge_color)
                axes1[ind1,ind2].plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
            # --- Plot the streamlines 
            # Define number of starting points for plotting the streamlines
            plot_start_points = 0
            start_points = np.zeros((0,2))
            streamline_density = 2
    #        flag_start = 1
            flag_start = 0
            if interp_MFAM_picM_plot == 1:
                stream = streamline_2D(axes1[ind1,ind2],zvec,rvec,dims_mp,zs_mp,rs_mp,-je_z_mp,-je_r_mp,flag_start,start_points,
                                       plot_start_points,streamline_density,streamline_width,streamline_color,arrow_size,
                                       arrow_style,min_length)
            else:
                stream = streamline_2D(axes1[ind1,ind2],zvec,rvec,dims,zs,rs,-je_z_plot,-je_r_plot,flag_start,start_points,
                                       plot_start_points,streamline_density,streamline_width,streamline_color,arrow_size,
                                       arrow_style,min_length)
            axes1[ind1,ind2].set_ylim(0,ylim_axes)
            if only_plume == 1:
                axes1[ind1,ind2].set_xlim(zs[0,int(xi_bottom)],xlim_axes)
            else:
                axes1[ind1,ind2].set_xlim(0,xlim_axes)

            if ax_ticks_flag == 2:
    #            ax.set_xticks(np.arange(0,zs[0,-1]+1,2))
    #            ax.set_yticks(np.arange(0,rs[-1,0]+1,2))
                if only_plume == 1:
                    # axes1[ind1,ind2].set_xticks(np.arange(zs[0,int(xi_bottom)],xlim_axes+1,1))
                    axes1[ind1,ind2].set_xticks(np.arange(zs[0,int(xi_bottom)],xlim_axes,1))
                else:
                    # axes1[ind1,ind2].set_xticks(np.arange(0,xlim_axes+1,1))
                    axes1[ind1,ind2].set_xticks(np.arange(0,xlim_axes,1))
                # axes1[ind1,ind2].set_yticks(np.arange(0,ylim_axes,1))
                    axes1[ind1,ind2].set_yticks(np.arange(0,ylim_axes+1,1))
            elif ax_ticks_flag == 1:
                axes1[ind1,ind2].set_xticks(np.arange(0,zs[0,-1]+1,1))
    #            ax.set_yticks(np.arange(0,rs[-1,-1],1))
                axes1[ind1,ind2].set_yticks(np.arange(0,rs[-1,-1]+1,1))
    #            ax.set_yticks(np.arange(0,rs[-1,-1],2))
            
            
            axes1[ind1,ind2].tick_params(labelsize = ticks_size)
            # if k == 4 or k == 5:
            #     axes1[ind1,ind2].set_xlabel(cont_xlabel, fontsize = font_size)
            #     # axes1[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            # else:
            #     axes1[ind1,ind2].tick_params(labelsize = ticks_size,labelbottom=False)
                
            # if k == 0 or k == 2 or k == 4:
            #     axes1[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            # else:
            #     axes1[ind1,ind2].tick_params(labelsize = ticks_size,labelleft=False)
            
            if k%2:
                axes1[ind1,ind2].set_xlabel(cont_xlabel, fontsize = font_size)
                # axes1[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            else:
                axes1[ind1,ind2].tick_params(labelsize = ticks_size,labelbottom=False)
                
            if ind2 == 0:
                axes1[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            else:
                axes1[ind1,ind2].tick_params(labelsize = ticks_size,labelleft=False)
            
            
            
            if vline == 1:
                if ind1 == 2 or ind1 == 3:
                    axes1[ind1,ind2].axvline(x=zP, linestyle='--',color=color_vline,linewidth = line_width_boundary)
                
    
                
                
            log_type         = 1
            auto             = 0
    #        min_val0         = 1E-1
    #        max_val0         = 1E3
            min_val0         = 1E-5
            max_val0         = 1E0
            if only_plume == 1:
                max_val0         = 1E-1
            cont             = 1
            lines            = 0
            cont_nlevels     = nlevels_2Dcontour
            auto_cbar_ticks  = 1 
            auto_lines_ticks = -1
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
    #        if interp_MFAM_picM_plot == 1:
    #            [CS,CS2] = contour_2D_fig2 (ax,cont_xlabel, cont_ylabel, font_size, ticks_size, zs_mp, rs_mp, ji_2D_mp, nodes_flag_mp, log_type, auto, 
    #                                   min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
    #                                   nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
    #                                   lines_ticks_fmt, lines_ticks_color, lines_style, lines_width) 
    #        else:
    #            
    #            [CS,CS2] = contour_2D_fig2 (ax,cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, ji2D_plot, nodes_flag, log_type, auto, 
    #                                   min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
    #                                   nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
    #                                   lines_ticks_fmt, lines_ticks_color, lines_style, lines_width) 
            var = np.copy(ji2D_plot)
            if only_plume == 1:
                var[:,0:int(xi_bottom)] = np.nan    
            [CS,CS2] = contour_2D_fig2 (axes2[ind1,ind2],cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, var, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width) 
    
    #         if k == 1 or k == 3 or k == 5:
    #         # if k == 1:
    #             # Generate colorbar if needed
    #             min_val_exp = np.floor(np.log10(min_val0))
    #             max_val_exp = np.floor(np.log10(max_val0))+1
    #             if auto_cbar_ticks == 1 or auto_cbar_ticks == 0:
    #                 # For the colorbar ticks, only if auto_cbar_ticks=1 we generate automatic
    #                 # ticks. Otherwise is given already in inputed cbar_ticks. Format for
    #                 # colorbar ticks is not selected in this case
    #                 if auto_cbar_ticks == 1:
    #                     cbar_nticks = np.int(max_val_exp - min_val_exp + 1)
    #                     ticks_exp   = np.linspace(min_val_exp,max_val_exp,cbar_nticks)
    #                     cbar_ticks  = np.power(10, ticks_exp)
    #                 CS.cmap.set_under('white')
    #                 CS.cmap.set_over('black')
    #                 cbar = plt.colorbar(CS,ticks=cbar_ticks,ax=axes2[ind1,ind2])
    # #                cbar = plt.colorbar(CS,ticks=cbar_ticks,format=ticker.FuncFormatter(fmt_func_exponent),ax=ax)
    #                 cbar.ax.tick_params(labelsize=ticks_size)
    
    
    
    
    #        ax.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
    #        ax.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines)
            axes2[ind1,ind2].plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            axes2[ind1,ind2].plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            if plot_cath_contours == 1:
                axes2[ind1,ind2].plot(1.05*z_cath,r_cath,linewidth = line_width_boundary,markersize = marker_size,marker = 's',color = cath_marker_color,markeredgecolor = cath_marker_edge_color)
                # axes2[ind1,ind2].plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
            # --- Plot the streamlines 
            # Define number of starting points for plotting the streamlines
            flag_start = 1
            plot_start_points = 0
            start_points = np.zeros((0,2))
            streamline_density = 2
    #        flag_start = 1
            flag_start = 0
    #        if interp_MFAM_picM_plot == 1:
    #            stream = streamline_2D(ax,zvec,rvec,dims_mp,zs_mp,rs_mp,ji_z_mp,ji_x_mp,flag_start,start_points,
    #                                   plot_start_points,streamline_density,streamline_width,streamline_color,arrow_size,
    #                                   arrow_style,min_length)
    #        else:
    #            stream = streamline_2D(ax,zvec,rvec,dims,zs,rs,ji_z_plot,ji_x_plot,flag_start,start_points,
    #                                   plot_start_points,streamline_density,streamline_width,streamline_color,arrow_size,
    #                                   arrow_style,min_length)
                
            stream = streamline_2D(axes2[ind1,ind2],zvec,rvec,dims,zs,rs,ji_z_plot,ji_x_plot,flag_start,start_points,
                                       plot_start_points,streamline_density,streamline_width,streamline_color,arrow_size,
                                       arrow_style,min_length)
            axes2[ind1,ind2].set_ylim(0,ylim_axes)
            if only_plume == 1:
                axes2[ind1,ind2].set_xlim(zs[0,int(xi_bottom)],xlim_axes)
            else:
                axes2[ind1,ind2].set_xlim(0,xlim_axes)
            if ax_ticks_flag == 2:
    #            ax.set_xticks(np.arange(0,zs[0,-1]+1,2))
    #            ax.set_yticks(np.arange(0,rs[-1,0]+1,2))
                if only_plume == 1:
                    # axes2[ind1,ind2].set_xticks(np.arange(zs[0,int(xi_bottom)],xlim_axes+1,1))
                    axes2[ind1,ind2].set_xticks(np.arange(zs[0,int(xi_bottom)],xlim_axes,1))
                else:
                    # axes2[ind1,ind2].set_xticks(np.arange(0,xlim_axes+1,1))
                    axes2[ind1,ind2].set_xticks(np.arange(0,xlim_axes,1))
                # axes2[ind1,ind2].set_yticks(np.arange(0,ylim_axes,1))
                    axes2[ind1,ind2].set_yticks(np.arange(0,ylim_axes+1,1))
            elif ax_ticks_flag == 1:
                axes2[ind1,ind2].set_xticks(np.arange(0,zs[0,-1]+1,1))
    #            ax.set_yticks(np.arange(0,rs[-1,-1],1))
                axes2[ind1,ind2].set_yticks(np.arange(0,rs[-1,-1]+1,1))
    #            ax.set_yticks(np.arange(0,rs[-1,-1],2))
                
                
            axes2[ind1,ind2].tick_params(labelsize = ticks_size)
            # if k == 4 or k == 5:
            #     axes2[ind1,ind2].set_xlabel(cont_xlabel, fontsize = font_size)
            #     # axes2[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            # else:
            #     axes2[ind1,ind2].tick_params(labelsize = ticks_size,labelbottom=False)
                
            # if k == 0 or k == 2 or k == 4:
            #     axes2[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            # else:
            #     axes2[ind1,ind2].tick_params(labelsize = ticks_size,labelleft=False)
            
            if k%2:
                axes2[ind1,ind2].set_xlabel(cont_xlabel, fontsize = font_size)
                # axes1[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            else:
                axes2[ind1,ind2].tick_params(labelsize = ticks_size,labelbottom=False)
                
            if ind2 == 0:
                axes2[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            else:
                axes2[ind1,ind2].tick_params(labelsize = ticks_size,labelleft=False)
                
            if vline == 1:
                if ind1 == 2 or ind1 == 3:
                    axes2[ind1,ind2].axvline(x=zP, linestyle='--',color=color_vline,linewidth = line_width_boundary)
            
    
    
            log_type         = 1
            auto             = 0
    #        min_val0         = 1E-1
    #        max_val0         = 1E3
            min_val0         = 1E-5
            max_val0         = 1E0
            if only_plume == 1:
                max_val0         = 1E-1
            cont             = 1
            lines            = 0
            cont_nlevels     = nlevels_2Dcontour
            auto_cbar_ticks  = 1 
            auto_lines_ticks = -1
            nticks_cbar      = 5
            nticks_lines     = 10
            cbar_ticks       = np.array([0.0,0.2,0.3,0.5,0.7,0.8,0.9,1])
            lines_ticks      = np.array([1E1,1E2,5E2,1E3,2E3,5E3,1E4,2E4])
            lines_ticks_loc  = 'default'
            cbar_ticks_fmt    = '{%.1f}'
            lines_ticks_fmt   = '{%.4f}'
            lines_width       = line_width
            lines_ticks_color = 'k'
            lines_style       = '-'
            if interp_MFAM_picM_plot == 1:
                var = np.copy(j_2D_mp)
                if only_plume == 1:
                    var[np.where(zs_mp < zs[0,int(xi_bottom)])] = np.nan
                [CS,CS2] = contour_2D_fig2 (axes3[ind1,ind2],cont_xlabel, cont_ylabel, font_size, ticks_size, zs_mp, rs_mp, var, nodes_flag_mp, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)     
            else:
                var = np.copy(j2D_plot)
                if only_plume == 1:
                    var[:,0:int(xi_bottom)] = np.nan
                [CS,CS2] = contour_2D_fig2 (axes3[ind1,ind2],cont_xlabel, cont_ylabel, font_size, ticks_size, zs, rs, var, nodes_flag, log_type, auto, 
                                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width) 
    
    
    #         if k == 1 or k == 3 or k == 5:
    #         # if k == 1:
    #             # Generate colorbar if needed
    #             min_val_exp = np.floor(np.log10(min_val0))
    #             max_val_exp = np.floor(np.log10(max_val0))+1
    #             if auto_cbar_ticks == 1 or auto_cbar_ticks == 0:
    #                 # For the colorbar ticks, only if auto_cbar_ticks=1 we generate automatic
    #                 # ticks. Otherwise is given already in inputed cbar_ticks. Format for
    #                 # colorbar ticks is not selected in this case
    #                 if auto_cbar_ticks == 1:
    #                     cbar_nticks = np.int(max_val_exp - min_val_exp + 1)
    #                     ticks_exp   = np.linspace(min_val_exp,max_val_exp,cbar_nticks)
    #                     cbar_ticks  = np.power(10, ticks_exp)
    #                 CS.cmap.set_under('white')
    #                 CS.cmap.set_over('black')
    #                 cbar = plt.colorbar(CS,ticks=cbar_ticks,ax=axes3[ind1,ind2])
    # #                cbar = plt.colorbar(CS,ticks=cbar_ticks,format=ticker.FuncFormatter(fmt_func_exponent),ax=ax)
    #                 cbar.ax.tick_params(labelsize=ticks_size)
    
    
    
    #        ax.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, fontsize=ticks_size_isolines, zorder = 1)
    #        ax.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, fontsize=ticks_size_isolines)
            axes3[ind1,ind2].plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
            axes3[ind1,ind2].plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
            if plot_cath_contours == 1:
                axes3[ind1,ind2].plot(1.05*z_cath,r_cath,linewidth = line_width_boundary,markersize = marker_size,marker = 's',color = cath_marker_color,markeredgecolor = cath_marker_edge_color)
                # axes3[ind1,ind2].plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
            # --- Plot the streamlines 
            # Define number of starting points for plotting the streamlines
            plot_start_points = 0
            start_points = np.zeros((0,2))
            streamline_density = 2
    #        flag_start = 1
            flag_start = 0
            if interp_MFAM_picM_plot == 1:
                stream = streamline_2D(axes3[ind1,ind2],zvec,rvec,dims_mp,zs_mp,rs_mp,j_z_mp,j_r_mp,flag_start,start_points,
                                       plot_start_points,streamline_density,streamline_width,streamline_color,arrow_size,
                                       arrow_style,min_length)
            else:
                
                stream = streamline_2D(axes3[ind1,ind2],zvec,rvec,dims,zs,rs,j_z_plot,j_r_plot,flag_start,start_points,
                                       plot_start_points,streamline_density,streamline_width,streamline_color,arrow_size,
                                       arrow_style,min_length)
            axes3[ind1,ind2].set_ylim(0,ylim_axes)
            if only_plume == 1:
                axes3[ind1,ind2].set_xlim(zs[0,int(xi_bottom)],xlim_axes)
            else:
                axes3[ind1,ind2].set_xlim(0,xlim_axes)
            if ax_ticks_flag == 2:
    #            ax.set_xticks(np.arange(0,zs[0,-1]+1,2))
    #            ax.set_yticks(np.arange(0,rs[-1,0]+1,2))
                if only_plume == 1:
                    # axes3[ind1,ind2].set_xticks(np.arange(zs[0,int(xi_bottom)],xlim_axes+1,1))
                    axes3[ind1,ind2].set_xticks(np.arange(zs[0,int(xi_bottom)],xlim_axes,1))
                else:
                    # axes3[ind1,ind2].set_xticks(np.arange(0,xlim_axes+1,1))
                    axes3[ind1,ind2].set_xticks(np.arange(0,xlim_axes,1))
                # axes3[ind1,ind2].set_yticks(np.arange(0,ylim_axes,1))
                axes3[ind1,ind2].set_yticks(np.arange(0,ylim_axes+1,1))
            elif ax_ticks_flag == 1:
                axes3[ind1,ind2].set_xticks(np.arange(0,zs[0,-1]+1,1))
    #            ax.set_yticks(np.arange(0,rs[-1,-1],1))
                axes3[ind1,ind2].set_yticks(np.arange(0,rs[-1,-1]+1,1))
    #            ax.set_yticks(np.arange(0,rs[-1,-1],2))
                
            axes3[ind1,ind2].tick_params(labelsize = ticks_size)
            # if k == 4 or k == 5:
            #     axes3[ind1,ind2].set_xlabel(cont_xlabel, fontsize = font_size)
            #     # axes3[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            # else:
            #     axes3[ind1,ind2].tick_params(labelsize = ticks_size,labelbottom=False)
                
            # if k == 0 or k == 2 or k == 4:
            #     axes3[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            # else:
            #     axes3[ind1,ind2].tick_params(labelsize = ticks_size,labelleft=False)
            
            if k%2:
                axes3[ind1,ind2].set_xlabel(cont_xlabel, fontsize = font_size)
                # axes1[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            else:
                axes3[ind1,ind2].tick_params(labelsize = ticks_size,labelbottom=False)
                
            if ind2 == 0:
                axes3[ind1,ind2].set_ylabel(cont_ylabel, fontsize = font_size)   
            else:
                axes3[ind1,ind2].tick_params(labelsize = ticks_size,labelleft=False)
                
            if vline == 1:
                if ind1 == 2 or ind1 == 3:
                    axes3[ind1,ind2].axvline(x=zP, linestyle='--',color=color_vline,linewidth = line_width_boundary)
                    
        if vline == 1:
            if k == 3 or k == 5:
                zP = zs_mp[0,-1]
            
   

#         if k == 1 or k == 3 or k == 5:
    #         # if k == 1:
    #             # Generate colorbar if needed
    #             min_val_exp = np.floor(np.log10(min_val0))
    #             max_val_exp = np.floor(np.log10(max_val0))+1
    #             if auto_cbar_ticks == 1 or auto_cbar_ticks == 0:
    #                 # For the colorbar ticks, only if auto_cbar_ticks=1 we generate automatic
    #                 # ticks. Otherwise is given already in inputed cbar_ticks. Format for
    #                 # colorbar ticks is not selected in this case
    #                 if auto_cbar_ticks == 1:
    #                     cbar_nticks = np.int(max_val_exp - min_val_exp + 1)
    #                     ticks_exp   = np.linspace(min_val_exp,max_val_exp,cbar_nticks)
    #                     cbar_ticks  = np.power(10, ticks_exp)
    #                 CS.cmap.set_under('white')
    #                 CS.cmap.set_over('black')
    #                 cbar = plt.colorbar(CS,ticks=cbar_ticks,ax=axes3[ind1,ind2])
    # #                cbar = plt.colorbar(CS,ticks=cbar_ticks,format=ticker.FuncFormatter(fmt_func_exponent),ax=ax)
    #                 cbar.ax.tick_params(labelsize=ticks_size)         
        
    # Add a common colorbar for all subplots in horizontal position          
    if mappable is None:
        CS.cmap.set_under('white')
        CS.cmap.set_over('black')
        mappable = CS
                
    min_val_exp = np.floor(np.log10(min_val0))
    max_val_exp = np.floor(np.log10(max_val0))+1
    
    if auto_cbar_ticks == 1:
        cbar_nticks = int(max_val_exp - min_val_exp + 1)
        ticks_exp = np.linspace(min_val_exp, max_val_exp, cbar_nticks)
        cbar_ticks = np.power(10, ticks_exp)


    # Position of the horizontal colorbar common to the complete figure
    # cbar_ax = fig1.add_axes([0.15, 0.08, 0.7, 0.025])  #  [left, bottom, width, height] in coordinates relative to figure
    # left: cuánto margen izquierdo deja (→ más pequeño = más a la izquierda).
    # bottom: altura vertical desde la base de la figura (→ más pequeño = más abajo).
    # width: longitud de la barra (entre 0 y 1, en fracción del ancho total).
    # height: grosor vertical de la barra.
        
    cbar_ax = fig1.add_axes([0.20, -0.03, 0.6, 0.020])  #  [left, bottom, width, height] in coordinates relative to figure
    # cbar = fig1.colorbar(mappable, ax=axes1.ravel().tolist(), orientation='horizontal',
    #                     ticks=cbar_ticks, fraction=0.05, pad=0.07)
    cbar = fig1.colorbar(mappable, cax=cbar_ax, orientation='horizontal', ticks=cbar_ticks)
    cbar.ax.tick_params(labelsize=ticks_size)
    cbar.set_label(r"$-\tilde{\boldsymbol{\jmath}}_\mathrm{e}$ (Acm$^{-2}$)", fontsize=font_size)
    
    cbar_ax = fig2.add_axes([0.20, -0.03, 0.6, 0.020])  # ajusta según convenga
    # cbar = fig2.colorbar(mappable, ax=axes2.ravel().tolist(), orientation='horizontal',
    #                     ticks=cbar_ticks, fraction=0.05, pad=0.07)
    cbar = fig2.colorbar(mappable, cax=cbar_ax, orientation='horizontal', ticks=cbar_ticks)
    cbar.ax.tick_params(labelsize=ticks_size)
    cbar.set_label(r"$\tilde{\boldsymbol{\jmath}}_\mathrm{i}$ (Acm$^{-2}$)", fontsize=font_size)
    
    cbar_ax = fig3.add_axes([0.20, -0.03, 0.6, 0.020])  # ajusta según convenga
    # cbar = fig3.colorbar(mappable, ax=axes3.ravel().tolist(), orientation='horizontal',
    #                     ticks=cbar_ticks, fraction=0.05, pad=0.07)
    cbar = fig3.colorbar(mappable, cax=cbar_ax, orientation='horizontal', ticks=cbar_ticks)
    cbar.ax.tick_params(labelsize=ticks_size)
    cbar.set_label(r"$\tilde{\boldsymbol{\jmath}}$ (Acm$^{-2}$)", fontsize=font_size)
    
    fig1.tight_layout()
    if save_flag == 1:
        fig1.savefig(path_out+"je2D_ref_complete"+figs_format,bbox_inches='tight')
        fig1.savefig(path_out+"je2D_ref_complete"+figs_format2,bbox_inches='tight')
        plt.close(fig1)
        
    fig2.tight_layout()
    if save_flag == 1:
        fig2.savefig(path_out+"ji2D_ref_complete"+figs_format,bbox_inches='tight')
        fig2.savefig(path_out+"ji2D_ref_complete"+figs_format2,bbox_inches='tight')
        plt.close(fig2)
        
    fig3.tight_layout()
    if save_flag == 1:
        fig3.savefig(path_out+"fig2_rev2"+figs_format,bbox_inches='tight')
        fig3.savefig(path_out+"fig2_rev2"+figs_format2,bbox_inches='tight')
        plt.close(fig3)
    