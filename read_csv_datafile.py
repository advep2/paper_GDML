#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:59:12 2019

@author: adrian
"""


import numpy as np
import matplotlib.pyplot as plt

marker_size = 3
font_size           = 25
font_size_legend    = font_size - 10
ticks_size          = 25
ticks_size_isolines = ticks_size - 15
text_size           = 25
levels_step         = 100
cbar_nticks         = 10


plt.close("all")

path_inp = "TOPO2_n4_UC3M.CSV"

data = np.genfromtxt(path_inp, delimiter='\t')
time = data[:,0]
var1 = data[:,1] # Vd
var2 = data[:,2]
var3 = data[:,3]
var4 = data[:,4] # Id
var5 = data[:,5]
var6 = data[:,6]
var7 = data[:,7]
var8 = data[:,8]

print("#### COMPLETE SIGNALS ####")
print("Mean var1 = %15.8e" %np.mean(var1))
print("Mean var2 = %15.8e" %np.mean(var2))
print("Mean var3 = %15.8e" %np.mean(var3))
print("Mean var4 = %15.8e" %np.mean(var4))
print("Mean var5 = %15.8e" %np.mean(var5))

nsteps = len(time)

# time in ms
time = time*1e3
ind_ini = 0
ind_end = nsteps/10

time = time[ind_ini:ind_end+1]
var1 = var1[ind_ini:ind_end+1] # Vd
var2 = var2[ind_ini:ind_end+1]
var3 = var3[ind_ini:ind_end+1]
var4 = var4[ind_ini:ind_end+1] # Id
var5 = var5[ind_ini:ind_end+1]
var6 = var6[ind_ini:ind_end+1]
var7 = var7[ind_ini:ind_end+1]
var8 = var8[ind_ini:ind_end+1]

print("#### PARTIAL SIGNALS ####")
print("Mean var1 = %15.8e" %np.mean(var1))
print("Mean var2 = %15.8e" %np.mean(var2))
print("Mean var3 = %15.8e" %np.mean(var3))
print("Mean var4 = %15.8e" %np.mean(var4))
print("Mean var5 = %15.8e" %np.mean(var5))

#plt.figure("var1")
#plt.plot(time,var1,'ko-',markersize = marker_size)
#plt.plot(time,np.mean(var1)*np.ones(np.shape(var1)),'r')
#ax = plt.gca()
#plt.xlabel(r"t (ms)",fontsize = font_size)
#plt.title(r"$V_d$ (V)",fontsize = font_size,y=1.02)
#plt.xticks(fontsize = ticks_size) 
#plt.yticks(fontsize = ticks_size)

plt.figure("var2")
plt.plot(time,var2,'ko-',markersize = marker_size)
plt.plot(time,np.mean(var2)*np.ones(np.shape(var2)),'r')
plt.xlabel(r"t (ms)",fontsize = font_size)
plt.title(r"$\phi_{cat}$ (V)",fontsize = font_size,y=1.02)
plt.xticks(fontsize = ticks_size) 
plt.yticks(fontsize = ticks_size)

plt.figure("var3")
plt.plot(time,var3,'ko-',markersize = marker_size)
plt.plot(time,np.mean(var3)*np.ones(np.shape(var3)),'r')
plt.xlabel(r"t (ms)",fontsize = font_size)
plt.title(r"$I_{d,RMS}$ (A)",fontsize = font_size,y=1.02)
plt.xticks(fontsize = ticks_size) 
plt.yticks(fontsize = ticks_size)

#plt.figure("var4")
#plt.plot(time,var4,'ko-',markersize = marker_size)
#plt.plot(time,np.mean(var4)*np.ones(np.shape(var1)),'r')
#ax = plt.gca()
#plt.xlabel(r"t (ms)",fontsize = font_size)
#plt.title(r"$I_d$ (A)",fontsize = font_size,y=1.02)
#plt.xticks(fontsize = ticks_size) 
#plt.yticks(fontsize = ticks_size)

plt.figure("var5")
plt.plot(time,var5,'ko-',markersize = marker_size)

plt.figure("var6")
plt.plot(time,var6,'ko-',markersize = marker_size)

plt.figure("var7")
plt.plot(time,var7,'ko-',markersize = marker_size)

plt.figure("var8")
plt.plot(time,var8,'ko-',markersize = marker_size)

