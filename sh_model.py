#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:18:01 2020

@author: adrian
"""

import numpy as np
import matplotlib.pyplot as plt


e  = 1.6021766E-19
me = 9.1093829E-31

Ts       = 2.0
Es       = 50.0
delta_r0 = 0.4
Er       = 40
sigma_th = 0.3

Te       = 15
ne       = 2E17

delta_s = 2*Te/Es
delta_r = delta_r0*Er**2/(Te+Er)**2

delta_phi = np.linspace(0,20*Te,500)


gp_net_max = sigma_th*(1-delta_r)*ne*np.sqrt(e*Te/(2*np.pi*me))
gp_net = gp_net_max*np.exp(-delta_phi/Te)
gs = gp_net*delta_s
ge = gp_net - gs

E_W    = 2*e*Te*gp_net - 2*e*Ts*gs
E_W_ns = 2*e*Te*gp_net
E_Q = E_W + e*delta_phi*gp_net*(1-delta_s)
E_deltaphiQ = e*delta_phi*gp_net*(1-delta_s)

plt.close("Heats at W and Q")
plt.figure("Heats at W and Q")
plt.semilogy(delta_phi/Te,E_W,'r',label=r'EW')
plt.semilogy(delta_phi/Te,E_W_ns,'m',label=r'EW nosee')
plt.semilogy(delta_phi/Te,E_Q,'b',label=r'EQ')
plt.semilogy(delta_phi/Te,E_deltaphiQ,'c',label=r'deltaphi EQ')
plt.legend()

plt.close("Fluxes at W")
plt.figure("Fluxes at W")
plt.semilogy(delta_phi/Te,gp_net,'r',label=r'gpnet')
plt.semilogy(delta_phi/Te,gs,'b',label=r'gs')
plt.semilogy(delta_phi/Te,ge,'r',label=r'ge = gpnet-gs')
plt.legend()

print("delta_s = "+str(delta_s))

