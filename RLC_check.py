#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:38:57 2020

@author: adrian
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from RLC_filter_circuit import RLC_filter_circuit, RLC_filter_circuit2


plt.close("all")

sim_name = "../../../sim/sims/Topo2_n4_l200s200_cat1200_tm15_te1_tq125_orig"
path_postdata_out = sim_name+"/CORE/out/PostData.hdf5"

V_ps  = 4.98900000e+02
V_ref = 0.0
R     = 1.00000000e+02
L     = 3.40000000e-04
C     = 6.00000000e-06
dt    = 1.5E-8/5.0

# Open the PostData.hdf5 file
h5_post = h5py.File(path_postdata_out,"r+")
Id_inst = h5_post['/eFldM_data/boundary/Id'][:,0]
Id      = h5_post['/eFldM_data/boundary/Id_acc'][:,0]
Vd_inst = h5_post['/eFldM_data/boundary/Vd'][:,0]
Vd      = h5_post['/eFldM_data/boundary/Vd_acc'][:,0]
nsteps         = len(Id)
time           = np.zeros(nsteps,dtype=float)
for i in range(0,nsteps-1):
    time[i+1] = time[i] + 50*dt

# Use RLC_filter_circuit in core
Vd_out             = np.zeros(nsteps,dtype=float)
RLC_filter_int_out = np.zeros(nsteps,dtype=float)
Vd0                = V_ps
RLC_filter_int     = 0.0
for i in range(0,nsteps):
    [Vd1,RLC_filter_int]  = RLC_filter_circuit(V_ref,V_ps,Vd0,Id_inst[i],RLC_filter_int,50*dt,R,L,C)
    RLC_filter_int_out[i] = RLC_filter_int
    Vd_out[i] = Vd1
    Vd0       = Vd1


plt.figure("Vd_out")
plt.plot(time,Vd_out,'r')
#plt.plot(time,Vd_inst,'b')

plt.figure("RLC integral")
plt.plot(time,RLC_filter_int_out,'r')


# Use Barral approach
Vd_out         = np.zeros(nsteps,dtype=float)
deltaV_out     = np.zeros(nsteps,dtype=float)
Vd0 = V_ps
for i in range(0,nsteps-1):
    Id0 = Id_inst[i]
    Id1 = Id_inst[i+1]
    [Vd1,deltaV]  = RLC_filter_circuit2(Vd0,Id0,Id1,50*dt,R,L,C)
    deltaV_out[i] = deltaV
    Vd_out[i] = Vd1
    Vd0       = Vd1


plt.figure("Vd_out")
plt.plot(time,Vd_out,'k')

plt.figure("deltaV_out")
plt.plot(time,deltaV_out,'k')

#for i in range(0,100):
#    print(Id_inst[i])
















