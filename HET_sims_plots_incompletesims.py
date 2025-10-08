#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:26:57 2019

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
from scipy.signal import correlate
import pylab
import scipy.io as sio

# ---- Deactivate/activate all types of python warnings ----
import warnings
warnings.filterwarnings("ignore") # Deactivate all types of warnings
#    warnings.simplefilter('always')   # Activate all types of warnings
# -------------------------------------------------



# Close all existing figures
plt.close("all")

save_flag = 0

    
# Plots save flag
#figs_format = ".eps"
figs_format = ".png"
#figs_format = ".pdf"

#path_out = "../../../HET_figs/SAFRAN_HET/topo2_n4/comp_cases/cat1200_tmtetq_tests/"
path_out = "HET_figs/"

# Set options for LaTeX font
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
font_size           = 25
#font_size_legend    = font_size - 10
font_size_legend    = font_size - 18
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


print("######## time_plots ########")

#    marker_size = 4
#    marker_size_ID = marker_size + 2
#    marker_size = 0
marker_size_ID = 10


font_size_legend = 8
props = dict(boxstyle='round', facecolor='white', edgecolor = 'k',alpha=1) 

timestep        = 0
allsteps_flag   = 1
read_inst_data  = 1
read_part_lists = 0
read_flag       = 0


step_i = 1
step_f = 3


plot_mass = 0
plot_dens = 0
plot_Nmp  = 0
plot_eff  = 0
plot_thr  = 0
plot_Te   = 0 # Usually set to 1 for complete sims
plot_Id   = 1 # Usually set to 1 for complete sims
plot_Vd   = 0 # Usually set to 1 for complete sims
plot_Pd   = 0
plot_cath = 0
plot_mbal = 0
plot_Pbal = 0
plot_Pbal_inistep = 300
plot_exp_data = 0

Nke_effects = 0

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

prntstep_IDs = []
prntstep_IDs_text = []
fast_prntstep_IDs = []

#### READ EXPERIMENTAL DATA IF REQUIRED
exp_data_time_plots = 0
#exp_datafile_name   = "TOPO1_n1_UC3M.CSV"
#exp_datafile_name   = "TOPO1_n2_UC3M.CSV"
#exp_datafile_name   = "TOPO2_n3_UC3M.CSV"
exp_datafile_name   = "TOPO2_n4_UC3M.CSV"
exp_data = np.genfromtxt(exp_datafile_name, delimiter='\t')
exp_time = exp_data[:,0]
exp_Vd   = exp_data[:,1]  # Vd experimental values
exp_Id   = exp_data[:,4]  # Id experimental values

exp_nsteps     = len(exp_time)
exp_last_steps = exp_nsteps
exp_order      = 20
########################################
    
# Simulation names
nsims = 1

# Flag for old sims (1: old sim files, 0: new sim files)
#oldpost_sim      = np.array([3,3,3,3,3,3,3,0,0,3,3,3,0,0,0,0],dtype = int)
#oldsimparams_sim = np.array([7,8,8,7,7,7,7,6,6,7,7,7,6,6,5,0],dtype = int)

#oldpost_sim      = np.array([3,3,3,3,3,3,3,0,0,3,3,3,0,0,0,0],dtype = int)
#oldsimparams_sim = np.array([7,7,7,7,7,7,7,6,6,7,7,7,6,6,5,0],dtype = int)

#oldpost_sim      = np.array([0,3,3,3,3,3,3,3,0,0,3,3,3,0,0,0,0],dtype = int)
#oldsimparams_sim = np.array([6,7,7,7,7,7,7,7,6,6,7,7,7,6,6,5,0],dtype = int)

#oldpost_sim      = np.array([0,3,0,3,3,0,0,3,3,3,0,0,0,0],dtype = int)
#oldsimparams_sim = np.array([5,7,6,7,7,6,6,7,7,7,6,6,5,0],dtype = int)

#oldpost_sim      = np.array([1,3,3,3,3,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
#oldsimparams_sim = np.array([0,9,7,7,7,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)   

oldpost_sim      = np.array([6,3,3,3,3,3,3,3,0,0,3,3,3,3,0,0,0,0],dtype = int)
oldsimparams_sim = np.array([20,9,7,7,7,7,7,7,6,6,7,7,7,7,6,6,5,0],dtype = int)   

sim_names = [
        
#             "../../../Rb_hyphen/sim/sims/SPT100_al0025_Ne5_C4",
#             "../../../Sr_hyphen/sim/sims/SPT100_pm1em1_cat878_tmtetq25",
#
#             "../../../Rb_sims_files/Topo2_n4_l200s200_cat1200_tm15_te1_tq125",
#             "../../../Ca_sims_files/Topo2_n4_l200s200_cat1200_tm15_te1_tq125_Check",
#             "../../../Ca_sims_files/Topo2_n4_l200s200_cat1200_tm15_te1_tq125_Check_acc",
             
#             "../../../H_sims/Mg/hyphen/sims/CHEOPS_LP/PPSX00_OP1_CEX",
#             "../../../Mg_hyphen_borja/sim/sims/PPSX00_OP1_tmte12_tq1",
             "../../../sim/sims/PPSX00_OP1_new_intel",

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
elif topo_case == 0:    
    PIC_mesh_file_name = [
                          "PIC_mesh.hdf5",
                          "PIC_mesh.hdf5",
                          "SPT100_picM.hdf5",
                          "SPT100_picM.hdf5",
                          "SPT100_picM.hdf5",
                          "SPT100_picM.hdf5"]


# Labels     
labels = [
            r"",
#            r"C4 thesis",
#            r"C4 new",
        


                #r"No RLC",                                                  # $\alpha_{tm} = 1-10$,$\alpha_{te,q} = 1-2.5$
                #r"$R = 100$ $\Omega$, $L = 340$ $\mu$H, $C = 6$ $\mu$F",    # $\alpha_{tm} = 1-10$,$\alpha_{te,q} = 1-2.5$
                #r"$R = 100$ $\Omega$, $L = 340$ $\mu$H, $C = 0.6$ $\mu$F",  # $\alpha_{tm} = 1-10$,$\alpha_{te,q} = 1-2.5$

#                r"No RLC",                                                  # $\alpha_{tm} = 1-5$,$\alpha_{te} = 1$,$\alpha_{tq} = 1-2.5$
#                r"$R = 100$ $\Omega$, $L = 340$ $\mu$H, $C = 6$ $\mu$F",    # $\alpha_{tm} = 1-5$,$\alpha_{te} = 1$,$\alpha_{tq} = 1-2.5$
#                r"$R = 100$ $\Omega$, $L = 340$ $\mu$H, $C = 0.6$ $\mu$F",  # $\alpha_{tm} = 1-5$,$\alpha_{te} = 1$,$\alpha_{tq} = 1-2.5$

#            r"orig",
#            r"Check",
#            r"Check acc"

          ]
          

# Titles for the reference case and S case
titles_Id_fig = [r"(e) $I_d$ (A)",
                 r"(e) $I_d$ instantaneous (A)",
                 r"(f) $I_d$ normalized amplitude (-)",
                 r"(e) Instantaneous $I_d$ normalized amplitude (-)",
                 r"(g) $I_{i \infty}$ (A)",
                 r"(h) $I_{i \infty}$ normalized amplitude (-)",
                 r"(c) $I_d$, $I_{i \infty}$ (A)",
                 r"(d) $I_d$, $I_{i \infty}$ normalized amplitude (-)"]
                 
titles_dens_fig = [r"(c) $\bar{n}_e$ (m$^{-3}$)",
                   r"(a) $\bar{n}_{n}$ (m$^{-3}$)",
                   r"(d) $\bar{n}_e$ normalized amplitude (-)",
                   r"(b) $\bar{n}_n$ normalized amplitude (-)",
                   r"(a) $\bar{n}_e$, $\bar{n}_n$ (m$^{-3}$)",
                   r"(b) $\bar{n}_e$, $\bar{n}_n$ normalized amplitude (-)"]
                   
titles_eff_fig  = [r"(e) $\eta_{u}$ (-)",
                   r"(f) $\eta_{thr}$ (-)"]
                   
                   
# Titles for the cathode cases
titles_Id_fig = [r"$I_d$ (A)",
                 r"(e) $I_d$ instantaneous (A)",
                 r"(b) $I_d$ normalized amplitude (-)",
                 r"(e) Instantaneous $I_d$ normalized amplitude (-)",
                 r"(g) $I_{i \infty}$ (A)",
                 r"(h) $I_{i \infty}$ normalized amplitude (-)",
                 r"(c) $I_d$, $I_{i \infty}$ (A)",
                 r"(d) $I_d$, $I_{i \infty}$ normalized amplitude (-)"]
                 
titles_dens_fig = [r"(c) $\bar{n}_e$ (m$^{-3}$)",
                   r"(d) $\bar{n}_{n}$ (m$^{-3}$)",
                   r"(d) $\bar{n}_e$ normalized amplitude (-)",
                   r"(b) $\bar{n}_n$ normalized amplitude (-)",
                   r"(a) $\bar{n}_e$, $\bar{n}_n$ (m$^{-3}$)",
                   r"(b) $\bar{n}_e$, $\bar{n}_n$ normalized amplitude (-)"]
                   
                   
#    # Titles for the alpha_t g1 cases
#    titles_Id_fig = [r"(a) $I_d$ (A)",
#                     r"(e) $I_d$ instantaneous (A)",
#                     r"(c) $I_d$ normalized amplitude (-)",
#                     r"(e) Instantaneous $I_d$ normalized amplitude (-)",
#                     r"(g) $I_{i \infty}$ (A)",
#                     r"(h) $I_{i \infty}$ normalized amplitude (-)",
#                     r"(c) $I_d$, $I_{i \infty}$ (A)",
#                     r"(d) $I_d$, $I_{i \infty}$ normalized amplitude (-)"]
#                     
#    titles_dens_fig = [r"(e) $\bar{n}_e$ (m$^{-3}$)",
#                       r"(g) $\bar{n}_{n}$ (m$^{-3}$)",
#                       r"(d) $\bar{n}_e$ normalized amplitude (-)",
#                       r"(b) $\bar{n}_n$ normalized amplitude (-)",
#                       r"(a) $\bar{n}_e$, $\bar{n}_n$ (m$^{-3}$)",
#                       r"(b) $\bar{n}_e$, $\bar{n}_n$ normalized amplitude (-)"]
                   
                   
#    # Titles for the alpha_t g2 cases
#    titles_Id_fig = [r"(b) $I_d$ (A)",
#                     r"(e) $I_d$ instantaneous (A)",
#                     r"(d) $I_d$ normalized amplitude (-)",
#                     r"(e) Instantaneous $I_d$ normalized amplitude (-)",
#                     r"(g) $I_{i \infty}$ (A)",
#                     r"(h) $I_{i \infty}$ normalized amplitude (-)",
#                     r"(c) $I_d$, $I_{i \infty}$ (A)",
#                     r"(d) $I_d$, $I_{i \infty}$ normalized amplitude (-)"]
#                     
#    titles_dens_fig = [r"(f) $\bar{n}_e$ (m$^{-3}$)",
#                       r"(h) $\bar{n}_{n}$ (m$^{-3}$)",
#                       r"(d) $\bar{n}_e$ normalized amplitude (-)",
#                       r"(b) $\bar{n}_n$ normalized amplitude (-)",
#                       r"(a) $\bar{n}_e$, $\bar{n}_n$ (m$^{-3}$)",
#                       r"(b) $\bar{n}_e$, $\bar{n}_n$ normalized amplitude (-)"]
                   
# Titles for the Vd_cases and mA_cases
#    titles_Id_fig = [r"(a) $I_d$ (A)",
#                     r"(e) $I_d$ instantaneous (A)",
#                     r"(b) $I_d$ normalized amplitude (-)",
#                     r"(e) Instantaneous $I_d$ normalized amplitude (-)",
#                     r"(g) $I_{i \infty}$ (A)",
#                     r"(h) $I_{i \infty}$ normalized amplitude (-)",
#                     r"(c) $I_d$, $I_{i \infty}$ (A)",
#                     r"(d) $I_d$, $I_{i \infty}$ normalized amplitude (-)"]
#                     
#    titles_dens_fig = [r"(c) $\bar{n}_e$ (m$^{-3}$)",
#                       r"(e) $\bar{n}_{n}$ (m$^{-3}$)",
#                       r"(d) $\bar{n}_e$ normalized amplitude (-)",
#                       r"(f) $\bar{n}_n$ normalized amplitude (-)",
#                       r"(a) $\bar{n}_e$, $\bar{n}_n$ (m$^{-3}$)",
#                       r"(b) $\bar{n}_e$, $\bar{n}_n$ normalized amplitude (-)"]
#                       
#    titles_eff_fig  = [r"(g) $\eta_{u}$ (-)",
#                       r"(g) $\eta_{thr}$ (-)"]


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
#    linestyles = ['-','-','-','-','-','-','-']

dashList = [(None,None),(None,None),(12,6,12,6,3,6),(12,6,3,6,3,6),(5,2,20,2)] 
          
          
if plot_mass == 1:
    # Plot the time evolution of the ions 1 mass
    plt.figure(r'mi1(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    plt.title(r"(a) $m_{i1}$ ($10^{-11}$ kg)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of the ions 2 mass
    plt.figure(r'mi2(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    plt.title(r"(b) $m_{i2}$ ($10^{-11}$ kg)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of the total ion mass
    plt.figure(r'mitot(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    plt.title(r"(b) $m_{i} = m_{i1} + m_{i2}$ ($10^{-11}$ kg)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of the neutral mass
    plt.figure(r'mn(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    plt.title(r"(b) $m_{n}$ ($10^{-11}$ kg)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
if plot_dens == 1:
    # Plot the time evolution of the average plasma density in the domain
    plt.figure(r'dens_e(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    plt.title(titles_dens_fig[0],fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of the average neutral density in the domain
    plt.figure(r'dens_n(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    plt.title(titles_dens_fig[1],fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of both the average plasma and neutral density in the domain
    plt.figure(r'dens_e_dens_n(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    plt.title(titles_dens_fig[4],fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of both the average plasma and neutral density in the domain (normalized)
    plt.figure(r'norm_dens_e_dens_n(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    plt.title(titles_dens_fig[4]+" (normalized)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
if plot_eff == 1:
    # Plot the time evolution of the utilization efficiency
    plt.figure(r'eta_u(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(titles_eff_fig[0],fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of the production efficiency
    plt.figure(r'eta_prod(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(r"(e) $\eta_{prod}$ (-)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of the current efficiency
    plt.figure(r'eta_cur(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(r"(e) $\eta_{cur}$ (-)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of the divergence efficiency
    plt.figure(r'eta_div(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(r"(e) $\eta_{div}$ (-)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of the thrust efficiency
    plt.figure(r'eta_thr(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(titles_eff_fig[1],fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
if plot_thr == 1:
    # Plot the time evolution of the total thrust
    plt.figure(r'T(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(r"(h) $F$ (mN)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
if plot_Te == 1:
    # Plot the time evolution of the average Te in the domain
    plt.figure(r'Te(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(r"(e) $T_e$ (eV)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of both the normalized Id and Te_mean_dom            
    plt.figure(r'Te_Id(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(r"(e) $T_e/\bar{T}_e$, $I_d/\bar{I}_d$ (-)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
if plot_Id == 1:
    # Plot the time evolution of the discharge current
    plt.figure(r'Id(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(titles_Id_fig[0],fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
if plot_Vd == 1:
    # Plot the time evolution of the discharge voltage
    plt.figure(r'Vd(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(r"$V_d$ (V)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
if plot_Pd == 1:
    # Plot the time evolution of the input power
    plt.figure(r'Pd(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(r"(e) $P_d$ (W)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
if plot_cath == 1:
    # Plot the time evolution of the cathode equivalent emission frequency
    plt.figure(r'nu_cat(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(r"(e) $\nu_{cat}$ (Hz)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    # Plot the time evolution of the cathode emission power
    plt.figure(r'P_cat(t)')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(r"(f) $P_{cat}$ (W)",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)




ind = 0
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
       ndot_CEX01_i3,ndot_CEX02_i4,
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
    
    # Set initial time to zero
    time = time - time[0]
    time_fast = time_fast - time_fast[0]
    # Get the last timestep for plotting for the experimental data
    if exp_data_time_plots == 1:
        last_ind_exp = np.where(exp_time <= time[-1])[0][-1]
            
    # Domain averaged (spatially averaged) variables using the nodal weighting volumes
    [Te_mean_dom,_] = domain_average(Te,time,vol)
                                                 
 
    ###########################################################################
    print("Plotting...")
    ############################ GENERATING PLOTS #############################

    # Time in ms
    time       = time*1e3
    time_fast  = time_fast*1e3
    steps      = steps*1e-3
    steps_fast = steps_fast*1e-3
    prntsteps_ID = np.linspace(0,nsteps-1,nsteps)
    tot_mass_mp_ions   = tot_mass_mp_ions*1e11
    tot_mass_mp_neus   = tot_mass_mp_neus*1e11
    tot_mass_exit_ions = tot_mass_exit_ions*1e11
    tot_mass_exit_neus = tot_mass_exit_neus*1e11
    mass_mp_ions       = mass_mp_ions*1e11
    mass_mp_neus       = mass_mp_neus*1e11
    tot_num_mp_ions    = tot_num_mp_ions*1e-6
    tot_num_mp_neus    = tot_num_mp_neus*1e-6
    num_mp_ions        = num_mp_ions*1e-6
    num_mp_neus        = num_mp_neus*1e-6
    thrust             = thrust*1e3
    thrust_ion         = thrust_ion*1e3
    thrust_neu         = thrust_neu*1e3
    mbal_n1            = mbal_n1*1E6
    dMdt_n1            = dMdt_n1*1E6
    mbal_i1            = mbal_i1*1E6
    dMdt_i1            = dMdt_i1*1E6
    mbal_i2            = mbal_i2*1E6
    dMdt_i2            = dMdt_i2*1E6
    
    #if Id[-1] != 0:
    #    ind_start = np.where(Id == 0)[0][-1]
    #    time      = time[ind_start+1::] - time[ind_start+1]
    #    Id        = Id[ind_start+1::]
    #    Vd        = Vd[ind_start+1::]
    #    print("Vd_mean     = "+str(np.mean(Vd)))
    #    print("Id_mean     = "+str(np.mean(Id)))
    #    print("Id_mean exp = "+str(np.mean(exp_Id[0:last_ind_exp+1])))

    #time = time -time[0]
    
    
    print("F_mean      = "+str(np.mean(thrust[step_i:step_f+1])))
    print("Id_mean     = "+str(np.mean(Id[step_i:step_f+1])))
    print("Vd_mean     = "+str(np.mean(Vd[step_i:step_f+1])))
    
    
    if plot_mass == 1:
        # Plot the time evolution of the ions 1 mass
        plt.figure(r'mi1(t)')
        if time2steps_axis == 1:
            plt.plot(steps_fast, mass_mp_ions[:,0], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        else:
            plt.plot(time_fast, mass_mp_ions[:,0], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        # Plot the time evolution of the ions 2 mass
        plt.figure(r'mi2(t)')
        if time2steps_axis == 1:
            plt.plot(steps_fast, mass_mp_ions[:,1], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        else:   
            plt.plot(time_fast, mass_mp_ions[:,1], linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        # Plot the time evolution of the ion total mass
        plt.figure(r'mitot(t)')
        if time2steps_axis == 1:
            plt.plot(steps_fast, tot_mass_mp_ions, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
        else:
            plt.plot(time_fast, tot_mass_mp_ions, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
        # Plot the time evolution of the neutral mass
        plt.figure(r'mn(t)')
        if time2steps_axis == 1:
            plt.plot(steps_fast, tot_mass_mp_neus, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        else:     
            plt.plot(time_fast, tot_mass_mp_neus, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
    if plot_dens == 1:
        # Plot the time evolution of the average plasma density in the domain
        plt.figure(r'dens_e(t)')
        if time2steps_axis == 1:
            plt.semilogy(steps_fast, avg_dens_mp_ions, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
        else:
#                plt.semilogy(time_fast, avg_dens_mp_ions, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])      
            plt.semilogy(time_fast, avg_dens_mp_ions, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        # Plot the time evolution of the average neutral density in the domain
        plt.figure(r'dens_n(t)')
        if time2steps_axis == 1:
            plt.semilogy(steps_fast, avg_dens_mp_neus, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
        else:
#                plt.semilogy(time_fast, avg_dens_mp_neus, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])  
            plt.semilogy(time_fast, avg_dens_mp_neus, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k]) 

        # Plot the time evolution of both the average plasma and neutral density in the domain
        plt.figure(r'dens_e_dens_n(t)')
        if time2steps_axis == 1:
            plt.semilogy(steps_fast, avg_dens_mp_ions, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
            plt.semilogy(steps_fast, avg_dens_mp_neus, linestyle='--', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            
        
        else:
            plt.semilogy(time_fast, avg_dens_mp_ions, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
            plt.semilogy(time_fast, avg_dens_mp_neus, linestyle='--', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            
#                for i in range(0,len(fast_prntstep_IDs)):
#                    plt.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_ions[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#                    plt.text(fact_x[i]*time_fast[fast_prntstep_IDs[i]-plot_tol], fact_y[i]*avg_dens_mp_ions[fast_prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')     
#                    plt.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_neus[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
#                    plt.text(fact_x[i]*time_fast[fast_prntstep_IDs[i]-plot_tol], fact_y[i]*avg_dens_mp_neus[fast_prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')                         
#                    print(prntstep_IDs_text[i]+" time_fast = "+str(time_fast[fast_prntstep_IDs[i]])+", time = "+str(time[prntstep_IDs[i]]))
        
#        # Plot the time evolution of both the average plasma and neutral density in the domain (normalized)
#        plt.figure(r'norm_dens_e_dens_n(t)')  
#        if time2steps_axis == 1:
#            plt.semilogy(steps_fast, avg_dens_mp_ions/avg_dens_mp_ions_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
#            plt.semilogy(steps_fast, avg_dens_mp_neus/avg_dens_mp_neus_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            
#        
#        else:
#            plt.semilogy(time_fast, avg_dens_mp_ions/avg_dens_mp_ions_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])            
#            plt.semilogy(time_fast, avg_dens_mp_neus/avg_dens_mp_neus_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            

    if plot_eff == 1:
        # Plot the time evolution of the utilization efficiency
        plt.figure(r'eta_u(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.semilogy(steps, eta_u, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif time2steps_axis == 1 and prntstepsID_axis == 1:    
            plt.semilogy(prntsteps_ID, eta_u, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        else:
            plt.semilogy(time, eta_u, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        # Plot the time evolution of the production efficiency
        plt.figure(r'eta_prod(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.semilogy(steps, eta_prod, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
            plt.semilogy(prntsteps_ID, eta_prod, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        else:
            plt.semilogy(time, eta_prod, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        # Plot the time evolution of the current efficiency
        plt.figure(r'eta_cur(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.semilogy(steps, eta_cur, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
            plt.semilogy(prntsteps_ID, eta_cur, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        else:
            plt.semilogy(time, eta_cur, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                plt.semilogy(time[nsteps-last_steps::], eta_cur_mean*np.ones(np.shape(time[nsteps-last_steps::])), linestyle=linestyles[ind3], linewidth = line_width-1, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")              
        # Plot the time evolution of the divergence efficiency
        plt.figure(r'eta_div(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.semilogy(steps, eta_div, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
            plt.semilogy(prntsteps_ID, eta_div, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        else:
            plt.semilogy(time, eta_div, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        # Plot the time evolution of the thrust efficiency
        plt.figure(r'eta_thr(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.semilogy(steps, eta_thr, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
            plt.semilogy(prntsteps_ID, eta_thr, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        else:
            plt.semilogy(time, eta_thr, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
    if plot_thr == 1:
        # Plot the time evolution of the total thrust
        plt.figure(r'T(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.plot(steps, thrust, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
            plt.plot(prntsteps_ID, thrust, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        else:
            plt.semilogy(time, thrust, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        
    if plot_Te == 1:
        # Plot the time evolution of the average Te in the domain
        plt.figure(r'Te(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.semilogy(steps, Te_mean_dom, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
            plt.semilogy(prntsteps_ID, Te_mean_dom, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            for i in range(0,len(prntstep_IDs)):
                plt.semilogy(prntstep_IDs[i], Te_mean_dom[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
        else:
            plt.semilogy(time, Te_mean_dom, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        # Plot the time evolution of both the normalized Id and Te_mean_dom            
#        plt.figure(r'Te_Id(t)')
#        if time2steps_axis == 1 and prntstepsID_axis == 0:
#            plt.semilogy(steps, Te_mean_dom/Te_mean_dom_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#            plt.semilogy(steps, Id/Id_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")    
#            for i in range(0,len(prntstep_IDs)):                
##                    plt.text(steps[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)          
#                plt.text(steps[prntstep_IDs[i]-plot_tol], Te_mean_dom[prntstep_IDs[i]-plot_tol]/Te_mean_dom_mean,prntstep_IDs_text[i],fontsize = text_size,color='r',ha='center',va='center')             
#        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
#            plt.semilogy(prntsteps_ID, Te_mean_dom/Te_mean_dom_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#            plt.semilogy(prntsteps_ID, Id/Id_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")
#            plt.text(time[prntstep_IDs[i]-plot_tol], Te_mean_dom[prntstep_IDs[i]-plot_tol]/Te_mean_dom_mean,prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
#            for i in range(0,len(prntstep_IDs)):
#                plt.semilogy(prntstep_IDs[i], Te_mean_dom[prntstep_IDs[i]]/Te_mean_dom_mean, linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
#                plt.semilogy(prntstep_IDs[i], Id[prntstep_IDs[i]]/Id_mean, linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
#        else:
#            plt.semilogy(time, Te_mean_dom/Te_mean_dom_mean, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#            plt.semilogy(time, Id/Id_mean, linestyle='--', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")  
#            for i in range(0,len(prntstep_IDs)):
#                plt.semilogy(time[prntstep_IDs[i]], Te_mean_dom[prntstep_IDs[i]]/Te_mean_dom_mean, linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
##                    plt.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
#                plt.text(fact_x[i]*time[prntstep_IDs[i]-plot_tol], fact_y[i]*Te_mean_dom[prntstep_IDs[i]-plot_tol]/Te_mean_dom_mean,prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')                
##                plt.semilogy(time, Id_mean*np.ones(np.shape(time)), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label="")            

    if plot_Id == 1:
        # Plot the time evolution of the discharge current
        plt.figure(r'Id(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.semilogy(steps, Id, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#            plt.semilogy(steps, Id_mean*np.ones(np.shape(steps)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
            for i in range(0,len(prntstep_IDs)): 
                plt.semilogy(steps[prntstep_IDs[i]], Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color='r', markeredgecolor = 'k', label="")
#                    plt.text(steps[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)            
                plt.text(steps[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='r',ha='center',va='center')            
        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
            plt.semilogy(prntsteps_ID, Id, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#            plt.semilogy(prntsteps_ID, Id_mean*np.ones(np.shape(prntsteps_ID)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")
            for i in range(0,len(prntstep_IDs)):
                plt.semilogy(prntstep_IDs[i], Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color='r', markeredgecolor = 'k', label="")
#                    plt.text(prntstep_IDs[i]+plot_tol, Id[prntstep_IDs[i]],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
                plt.text(prntstep_IDs[i]+plot_tol, Id[prntstep_IDs[i]],prntstep_IDs_text[i],fontsize = text_size,color='r',ha='center',va='center')
        else:
            plt.semilogy(time, Id, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                plt.plot(time, Id, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])

#                if k == 0 or k == 1:
#                    plt.semilogy(time, Id, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])
#                elif k == 2 or k == 3:
#                    plt.semilogy(time, Id, linestyle=linestyles[k], dashes=dashList[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[k], color=colors[ind], markeredgecolor = 'k', label=labels[k])

#                plt.semilogy(time, Id_mean*np.ones(np.shape(time)), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")              
#                plt.semilogy(time[nsteps-last_steps::], Id_mean*np.ones(np.shape(time[nsteps-last_steps::])), linestyle=linestyles[ind3], linewidth = line_width-0.5, markevery=marker_every_time, markersize=marker_size, marker="", color=colors[ind], markeredgecolor = 'k', label="")                              
            for i in range(0,len(prntstep_IDs)):
                plt.semilogy(time[prntstep_IDs[i]], Id[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color='r', markeredgecolor = 'k', label="")            
#                    plt.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='k',ha='center',va='center',bbox=props)
                plt.text(time[prntstep_IDs[i]-plot_tol], Id[prntstep_IDs[i]-plot_tol],prntstep_IDs_text[i],fontsize = text_size,color='r',ha='center',va='center')
            
    if plot_Vd == 1:
        # Plot the time evolution of the discharge voltage
        plt.figure(r'Vd(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.plot(steps, Vd, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
            plt.plot(prntsteps_ID, Vd, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            for i in range(0,len(prntstep_IDs)):
                plt.plot(prntstep_IDs[i], Vd[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
        else:
            plt.plot(time, Vd, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
    if plot_Pd == 1:
        # Plot the time evolution of the input power
        plt.figure(r'Pd(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.plot(steps, Pd, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
            plt.plot(prntsteps_ID, Pd, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
            for i in range(0,len(prntstep_IDs)):
                plt.plot(prntstep_IDs[i], Pd[prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker='s', color=colors[ind], markeredgecolor = 'k', label="")
        else:
            plt.plot(time, Pd, linestyle=linestyles[ind3], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])    
    
    if plot_cath == 1:
        # Plot the time evolution of the cathode equivalent emission frequency
        plt.figure(r'nu_cat(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.semilogy(steps, nu_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
            plt.semilogy(prntsteps_ID, nu_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        else:
            plt.semilogy(time, nu_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        # Plot the time evolution of the cathode emission power
        plt.figure(r'P_cat(t)')
        if time2steps_axis == 1 and prntstepsID_axis == 0:
            plt.semilogy(steps, P_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        elif time2steps_axis == 1 and prntstepsID_axis == 1:  
            plt.semilogy(prntsteps_ID, P_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
        else:
            plt.semilogy(time, P_cath, linestyle=linestyles[k], linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker=markers[ind2], color=colors[ind], markeredgecolor = 'k', label=labels[k])
    
    ind = ind + 1
    if ind > 8:
        ind = 0
        ind2 = ind2 + 1
        if ind2 > 6:
            ind = 0
            ind2 = 0
            ind3 = ind3 + 1
 
           
if exp_data_time_plots == 1:
    exp_time = exp_time*1e3
    plt.figure(r'Id_exp(t)')
    plt.semilogy(exp_time[0:last_ind_exp+1], exp_Id[0:last_ind_exp+1], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='D', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
    plt.xlabel(r"$t$ (ms)",fontsize = font_size)
    if time2steps_axis == 1 and prntstepsID_axis == 0:
        plt.xlabel(r"$N_{steps}$ ($10^{3}$ -)",fontsize = font_size)
    elif time2steps_axis == 1 and prntstepsID_axis == 1:
        plt.xlabel(r"$N_{prnt,steps}$ (-)",fontsize = font_size)
    plt.title(r"$I_d$ exp. data",fontsize = font_size,y=1.02)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    if save_flag == 1:
        plt.figure(r'Id_exp(t)')
        plt.savefig(path_out+"Id_exp_t"+figs_format,bbox_inches='tight') 
        plt.close()


if plot_mass == 1:
    plt.figure(r'mi1(t)')
    plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
    plt.figure(r'mi2(t)')
    plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
    plt.figure(r'mitot(t)')
    plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
    plt.figure(r'mn(t)')
    plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
if plot_dens == 1:
    plt.figure(r'dens_e(t)')
    ax = plt.gca()
    ax.set_ylim(10**15,10**19)
    plt.legend(fontsize = font_size_legend,loc=4,ncol=2)
    plt.figure(r'dens_n(t)')
    ax = plt.gca()
    ax.set_ylim(10**17,10**19)
    plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
    plt.figure(r'dens_e_dens_n(t)')
    ax = plt.gca()
    ax.set_ylim(10**15,10**19)
#        ax.set_xlim(0.0,0.9)
    ylims = ax.get_ylim()
    fact_x = np.array([0.97,1.02,1.02,1.02])
    marker_size_ID = 6
    for i in range(0,len(fast_prntstep_IDs)):
        plt.semilogy(time_fast[fast_prntstep_IDs[i]]*np.ones(2), np.array([ylims[0],ylims[1]]), linestyle=':', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker="", color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
        plt.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_ions[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
        plt.semilogy(time_fast[fast_prntstep_IDs[i]], avg_dens_mp_neus[fast_prntstep_IDs[i]], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size_ID, marker=prntstep_IDs_markers[i], color=prntstep_IDs_colors[i], markeredgecolor = 'k', label="")            
        plt.text(fact_x[i]*time_fast[fast_prntstep_IDs[i]-plot_tol], 1.5*ylims[0],prntstep_IDs_text[i],fontsize = text_size,color=prntstep_IDs_colors[i],ha='center',va='center')     
        print(prntstep_IDs_text[i]+" time_fast = "+str(time_fast[fast_prntstep_IDs[i]])+", time = "+str(time[prntstep_IDs[i]]))
    plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
    plt.figure(r'norm_dens_e_dens_n(t)')  
    plt.legend(fontsize = font_size_legend,loc=1,ncol=2)
if plot_eff == 1:
    plt.figure(r'eta_u(t)')
    plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
    ax = plt.gca()
    ax.set_ylim(1E-2,1E1)
    plt.figure(r'eta_prod(t)')
    plt.legend(fontsize = font_size_legend,loc=2,ncol=2)
    ax = plt.gca()
    ax.set_ylim(1E-2,1E0)
    plt.figure(r'eta_cur(t)')
    plt.legend(fontsize = font_size_legend,loc=2,ncol=2)
    ax = plt.gca()
    ax.set_ylim(1E-1,1E1)
    plt.figure(r'eta_div(t)')
    plt.legend(fontsize = font_size_legend,loc=2,ncol=2)
    plt.figure(r'eta_thr(t)')
    plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
if plot_thr == 1:
    plt.figure(r'T(t)')
    ax = plt.gca()
    ax.set_ylim(1E-4,1E4)
    plt.legend(fontsize = font_size_legend,loc=4,ncol=2)    
if plot_Te == 1:
    plt.figure(r'Te(t)')
    plt.legend(fontsize = font_size_legend,loc=2,ncol=1)   
    plt.figure(r'Te_Id(t)')
    plt.legend(fontsize = font_size_legend,loc=2,ncol=1)  
if plot_Id == 1:
    plt.figure(r'Id(t)')
    if exp_data_time_plots == 1:
        plt.semilogy(exp_time[0:last_ind_exp+1], exp_Id[0:last_ind_exp+1], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='D', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
#    plt.legend(fontsize = font_size_legend,loc=2,ncol=1) 
#        plt.legend(fontsize = font_size_legend,loc=3,ncol=2)
    plt.legend(fontsize = font_size_legend,loc=3,ncol=1)
    ax = plt.gca()
    ax.set_ylim(1E-1,1E2)
#    ax.set_ylim(1E-2,1E2)

if plot_Vd == 1:
    plt.figure(r'Vd(t)')
    if exp_data_time_plots == 1:
        plt.plot(exp_time[0:last_ind_exp+1], exp_Vd[0:last_ind_exp+1], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker='D', color=orange, markeredgecolor = 'k', label=r'Exp. Data')
    plt.legend(fontsize = font_size_legend,loc=3,ncol=1)
if plot_Pd == 1:
    plt.figure(r'Pd(t)')
    plt.legend(fontsize = font_size_legend,loc=1,ncol=2)  
if plot_cath == 1:
    plt.figure(r'nu_cat(t)')
    plt.legend(fontsize = font_size_legend,loc=2,ncol=1)    
    plt.figure(r'P_cat(t)')
    plt.legend(fontsize = font_size_legend,loc=2,ncol=1) 

if save_flag == 1:
    if plot_mass == 1:
        plt.figure(r'mi1(t)')
        plt.savefig(path_out+"mi1_t"+figs_format,bbox_inches='tight') 
        plt.close() 
        plt.figure(r'mi2(t)')
        plt.savefig(path_out+"mi2_t"+figs_format,bbox_inches='tight') 
        plt.close() 
        plt.figure(r'mitot(t)')
        plt.savefig(path_out+"mitot_t"+figs_format,bbox_inches='tight') 
        plt.close() 
        plt.figure(r'mn(t)')
        plt.savefig(path_out+"mn_t"+figs_format,bbox_inches='tight') 
        plt.close() 
    if plot_dens == 1:
        plt.figure(r'dens_e(t)')
        plt.savefig(path_out+"avg_dens_e_t"+figs_format,bbox_inches='tight') 
        plt.close()
        plt.figure(r'dens_n(t)')
        plt.savefig(path_out+"avg_dens_n_t"+figs_format,bbox_inches='tight') 
        plt.close()
        plt.figure(r'dens_e_dens_n(t)')
        plt.savefig(path_out+"avg_dens_e_dens_n_t"+figs_format,bbox_inches='tight') 
        plt.close()
        plt.figure(r'norm_dens_e_dens_n(t)')  
        plt.savefig(path_out+"norm_avg_dens_e_dens_n_t"+figs_format,bbox_inches='tight') 
        plt.close()
    if plot_eff == 1:
        plt.figure(r'eta_u(t)')
        plt.savefig(path_out+"eta_u_t"+figs_format,bbox_inches='tight') 
        plt.close()   
        plt.figure(r'eta_prod(t)')
        plt.savefig(path_out+"eta_prod_t"+figs_format,bbox_inches='tight') 
        plt.close()  
        plt.figure(r'eta_cur(t)')
        plt.savefig(path_out+"eta_cur_t"+figs_format,bbox_inches='tight') 
        plt.close()
        plt.figure(r'eta_div(t)')
        plt.savefig(path_out+"eta_div_t"+figs_format,bbox_inches='tight') 
        plt.close()
        plt.figure(r'eta_thr(t)')
        plt.savefig(path_out+"eta_thr_t"+figs_format,bbox_inches='tight') 
        plt.close()
    if plot_thr == 1:
        plt.figure(r'T(t)')
        plt.savefig(path_out+"T_t"+figs_format,bbox_inches='tight') 
        plt.close()    
    if plot_Te == 1:
        plt.figure(r'Te(t)')
        plt.savefig(path_out+"Te_mean_dom_t"+figs_format,bbox_inches='tight') 
        plt.close()
        plt.figure(r'Te_Id(t)')
        plt.savefig(path_out+"Te_dom_mean_Id_t"+figs_format,bbox_inches='tight') 
        plt.close()
    if plot_Id == 1:
        plt.figure(r'Id(t)')
        plt.savefig(path_out+"Id_t"+figs_format,bbox_inches='tight') 
        plt.close()
    if plot_Vd == 1:
        plt.figure(r'Vd(t)')
        plt.savefig(path_out+"Vd_t"+figs_format,bbox_inches='tight') 
        plt.close()
    if plot_Pd == 1:
        plt.figure(r'Pd(t)')
        plt.savefig(path_out+"Pd_t"+figs_format,bbox_inches='tight') 
        plt.close()
    if plot_cath == 1:
        plt.figure(r'nu_cat(t)')
        plt.savefig(path_out+"nu_cat_t"+figs_format,bbox_inches='tight') 
        plt.close()   
        plt.figure(r'P_cat(t)')
        plt.savefig(path_out+"P_cat_t"+figs_format,bbox_inches='tight') 
        plt.close()



###########################################################################   
