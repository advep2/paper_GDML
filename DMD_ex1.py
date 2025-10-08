#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:37:50 2020

@author: adrian
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from pydmd import DMD
from DMD import dmd_rom

# Close all existing figures
plt.close("all")

# Create the input data adding two different functions
def f1(x,t): 
    return 1./np.cosh(x+3)*np.exp(2.3j*t)

def f2(x,t):
    return 2./np.cosh(x)*np.tanh(x)*np.exp(2.8j*t)



#Nx = 128
#Nt = 256
#x = np.linspace(-5, 5, Nx)
#t = np.linspace(0, 4*np.pi, Nt)

Nx = 400
Nt = 200
#Nx = 10
#Nt = 5
x = np.linspace(-10, 10, Nx)
t = np.linspace(0, 4*np.pi, Nt)

dt = t[1] - t[0]

xgrid, tgrid = np.meshgrid(x, t)

X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2 # Here we have one timestep for each row (we need one timestep per column)

titles = ['$f_1(x,t)$', '$f_2(x,t)$', '$f$']
data = [X1, X2, X]

fig = plt.figure(figsize=(17,6))
for n, title, d in zip(range(131,134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
plt.colorbar()


# Now we have the temporal snapshots in the input matrix rows: we can easily
# create a new DMD instance and exploit it in order to compute the 
# decomposition on the data. Since the snapshots must be arranged by columns, 
# in this case we need to transpose the matrix.
dmd = DMD(svd_rank=2)
dmd.fit(X.T)

# The dmd object contains the principal information about the decomposition:
# 1) The attribute modes is a 2D numpy array where the columns are the low-rank structures individuated; (these are the Phis)
# 2) The attribute dynamics is a 2D numpy array where the rows refer to the time evolution of each mode; (they include b terms or amplitudes)
# 3) The attribute eigs refers to the eigenvalues of the low dimensional operator;
# 4) The attribute reconstructed_data refers to the approximated system evolution.

# Thanks to the eigenvalues, we can check if the modes are stable or not: 
# if an eigenvalue is on the unit circle, the corresponding mode will be stable;
# while if an eigenvalue is inside or outside the unit circle, the mode will 
# converge or diverge, respectively. From the following plot, we can note that
# the two modes are stable.

for eig in dmd.eigs:
    print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))
dmd.plot_eigs(show_axes=True, show_unit_circle=True)

# We can plot the DMD modes and their time-dynamics
plt.figure("DMD Modes")
for mode in dmd.modes.T:
    plt.plot(x, mode.real)
    plt.title('Modes')

plt.figure("DMD modes dynamics")
for dynamic in dmd.dynamics:
    plt.plot(t, dynamic.real)
    plt.title('Dynamics')
    
# Finally, we can reconstruct the original dataset as the product of modes and
# dynamics. We plot the evolution of each mode to emphasize their similarity
# with the input functions and we plot the reconstructed data.
fig = plt.figure(figsize=(17,6))
for n, mode, dynamic in zip(range(131, 133), dmd.modes.T, dmd.dynamics):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T)
    
plt.subplot(133)
plt.pcolor(xgrid, tgrid, dmd.reconstructed_data.T.real)
plt.colorbar()


# Reconstruct the data and plot it to compare
X_dmd = np.matmul(dmd.modes,dmd.dynamics).transpose()

plt.figure("Reconstructed manually")
plt.pcolor(xgrid, tgrid, X_dmd.real)
plt.colorbar()

# Error between the data reconstructed automatically and manually
err = dmd.reconstructed_data.T.real - X_dmd.real

# Plot the time dynamics given by PyDMD
[fig, axes] = plt.subplots(nrows=2, ncols=1, figsize=(15,6))
ax1 = plt.subplot2grid( (2,1), (0,0) )
ax2 = plt.subplot2grid( (2,1), (1,0) )
ax1.plot(t,dmd.dynamics[0,:],label='PyDMD')
ax2.plot(t,dmd.dynamics[1,:],label='PyDMD')

# Obtain the frequencies in rad/s (i.e. omegas)
omegas = np.angle(dmd.eigs)/dt
omegas_dmd = dmd.frequency*(2.0*np.pi/dt) # Relation between omegas and frequencies given by PyDMD
# Obtain freqs in Hz
freqs = omegas/(2.0*np.pi)
freqs_dmd = dmd.frequency/dt

# Test the ad-hoc function
r = 2 # Select the number of modes
Big_X = X
[Eigenvalues, Eigenvectors, ModeAmplitudes, ModeFrequencies, GrowthRates, POD_Mode_Energies] = dmd_rom(Big_X, r, dt)
 
# Obtain the time dynamics of the modes
time_dynamics = np.zeros((r,Nt),dtype=complex)
for i in range(0,Nt):
    time_dynamics[:,i] = ModeAmplitudes*np.exp((GrowthRates + ModeFrequencies*1j)*t[i])
    
# Plot the time dynamocs obtained in this case
ax1.plot(t,np.real(time_dynamics[0,:]),linestyle='--',color='r',label='My DMD')
ax2.plot(t,np.real(time_dynamics[1,:]),linestyle='--',color='r',label='My DMD')
ax1.legend()
ax2.legend()
