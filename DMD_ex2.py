#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:16:37 2020

@author: adrian
"""

import numpy as np
import scipy
import scipy.integrate

from matplotlib import animation
from IPython.display import HTML
from matplotlib import pyplot as plt
from pydmd import DMD

# Close all existing figures
plt.close("all")

# Now we first create the 2D data set evolving in time: a hyperbolic sine with
# damped oscillations. We add noise to make it more realistic
Nx1 = 80
Nx2 = 80
x1 = np.linspace(-3, 3, 80)
x2 = np.linspace(-3, 3, 80)
x1grid, x2grid = np.meshgrid(x1, x2)

time = np.linspace(0, 6, 16)

data = [2/np.cosh(x1grid)/np.cosh(x2grid)*(1.2j**-t) for t in time]
noise = [np.random.normal(0.0, 0.4, size=x1grid.shape) for t in time]

snapshots = [d+n for d,n in zip(data, noise)]

fig = plt.figure("Original data",figsize=(18,12))
for id_subplot, snapshot in enumerate(snapshots, start=1):
    plt.subplot(4, 4, id_subplot)
    plt.pcolor(x1grid, x2grid, snapshot.real, vmin=-1, vmax=1)
    
# Alright, now it's time to apply the DMD to the collected data. First, we 
# create a new DMD instance; we note there are four optional parameters:
# svd_rank: since the dynamic mode decomposition relies on singular value 
#           decomposition, we can specify the number of the largest singular 
#           values used to approximate the input data.
# tlsq_rank: using the total least square, it is possible to perform a linear 
#            regression in order to remove the noise on the data; because this 
#            regression is based again on the singular value decomposition, 
#            this parameter indicates how many singular values are used.
# exact: boolean flag that allows to chose between the exact modes or the projected one.
# opt: boolean flag that allows to chose between the standard version and the optimized one.
    
dmd = DMD(svd_rank=1, tlsq_rank=2, exact=True, opt=True)
dmd.fit(snapshots)
dmd.plot_modes_2D(figsize=(12,5))

# The svd_rank can be set to zero for an automatic selection of the truncation rank;
# in some cases (as this tutorial) the singular values should be examinated 
# in order to select the proper truncation.
plt.figure("Singular values")
A = np.array([snapshot.flatten() for snapshot in snapshots]).T
plt.plot(scipy.linalg.svdvals(A), 'o')

# We can now plot the reconstructed states from DMD: the approximated system is
# similar to the original one and, moreover, the noise is greatly reduced.
fig = plt.figure("Reconstructed",figsize=(18,12))
for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T, start=1):
    plt.subplot(4, 4, id_subplot)
    plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)
    
## We can also manipulate the interval between the approximated states and 
## extend the temporal window where the data is reconstructed thanks to DMD. 
## Let's make the DMD delta time a quarter of the original and extend the temporal
## window to  [0,3torg] , where  torg  indicates the time when the last snapshot was caught.
#    
#print("Shape before manipulation: {}".format(dmd.reconstructed_data.shape))
#dmd.dmd_time['dt'] *= .25
#dmd.dmd_time['tend'] *= 3
#print("Shape after manipulation: {}".format(dmd.reconstructed_data.shape))

## Now the cool trick: we combine the reconstructed dataset to create an 
## animation that shows the evolution of our system.
##fig = plt.figure(figsize=(8,6),dpi=900)
#fig = plt.figure(figsize=(2,2),dpi=900)
#
#dmd_states = [state.reshape(x1grid.shape) for state in dmd.reconstructed_data.T]
#
#frames = [
#    [plt.pcolor(x1grid, x2grid, state.real, vmin=-1, vmax=1)]
#    for state in dmd_states
#]
#
#ani = animation.ArtistAnimation(fig, frames, interval=70, blit=False, repeat=False)
#
##HTML(ani.to_html5_video())
## Set up formatting for the movie files
##Writer = animation.writers['ffmpeg']
##writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
##writermp4 = animation.FFMpegWriter(fps=60) 
##ani.save('movie.mp4', writer=writermp4)
#
## Alternative for higher quality
#FFMpegWriter = animation.writers['ffmpeg']
#metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
#
## Change the video bitrate as you like and add some metadata.
#writer = FFMpegWriter(fps=15, bitrate=1800, metadata=metadata)
##ani.save("movie.mp4", writer=writer,dpi=900)
#ani.save("movie.mp4", writer=writer,dpi=900)
#
## HOW TO IMPROVE VIDEO QUALITY:
## Increase dphi in figure and in save
## Increase bitrate in writer
## Keep low figsize (2,2) in this case