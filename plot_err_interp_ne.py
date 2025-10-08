#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 08:45:39 2020

@author: adrian
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from contour_2D import contour_2D

sim_name = "../../../sim/sims/Topo2_n4_l200s200_cat1200_tm15_te1_tq125"

step_i = 200
step_f = 400

rind = 17

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

nlevels_2Dcontour = 100

# Extra colors
orange ='#FD6A02'            
gold   ='#F9A602'
brown  ='#8B4000'
silver ='#C0C0C0'


# Open the PostData.hdf5 file
path_simstate_out = sim_name+"/CORE/out/SimState.hdf5"
path_postdata_out = sim_name+"/CORE/out/PostData.hdf5"
h5_out = h5py.File(path_simstate_out,"r+")
h5_post = h5py.File(path_postdata_out,"r+")

zs = h5_out['/picM/zs'][0:,0:]*1E2
rs = h5_out['/picM/rs'][0:,0:]*1E2
nodes_flag = h5_out['/picM/nodes_flag'][0:,0:]
err_interp_n = h5_post["/picM_data/err_interp_n"][0:,0:,:]

err_interp_n_mean = np.nanmean(err_interp_n[:,:,step_i:step_f+1],axis=2)

err_interp_n_mean[np.where(np.isinf(err_interp_n_mean))] = 0.0



plt.figure('err_interp_n prof')
plt.plot(zs[rind,:],err_interp_n_mean[rind,:], linestyle='-', linewidth = 2, markevery=1, markersize=3, marker='s', color='k', markeredgecolor = 'k', label='')


plt.figure('phi ref')
plt.title(r"(d) $n_e^{err}$ (-)", fontsize = font_size,y=1.02)
ax = plt.gca()
log_type         = 1
auto             = 0
#        min_val0         = -2.0
#        max_val0         = 300.0
min_val0         = 1E-5
max_val0         = 1E0
cont             = 1
lines            = 1
cont_nlevels     = nlevels_2Dcontour
auto_cbar_ticks  = 1 
auto_lines_ticks = 1
nticks_cbar      = 14
nticks_lines     = 10
cbar_ticks       = np.sort(np.array([1E-4,1E-3,1E-2,1E-1,1E0]))
lines_ticks      = np.sort(np.array([1E-4,1E-3,1E-2,1E-1,1E0])) 
lines_ticks_loc  = 'default'
cbar_ticks_fmt    = '{%.0f}'
lines_ticks_fmt   = '{%.0f}'
lines_width       = line_width
lines_ticks_color = 'k'
lines_style       = '-'
[CS,CS2] = contour_2D (ax,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, err_interp_n_mean, nodes_flag, log_type, auto, 
                       min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                       nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                       lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)     
#ax.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, fontsize=ticks_size_isolines, zorder = 1)
#plt.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
#plt.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)    
