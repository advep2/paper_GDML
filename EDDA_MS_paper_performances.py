#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:17:32 2021

@author: adrian
"""
import numpy as np

e = 1.6021766E-19
m = 2.1802878E-25

m_A      = np.array([8,8,8,8,10,10,10,10,10,14,14,14,14,14],dtype=float)*1E-6
eta_u    = np.array([94.8,97.7,99.4,99.5,91.0,95.5,98.0,98.4,98.6,92.3,95.4,96.3,96.6,97.3],dtype=float)*1E-2
eta_prod = np.array([51.8,49.5,50.0,54.2,57.2,51.6,49.5,50.2,55.4,57.6,51.8,49.9,51.2,57.4],dtype=float)*1E-2
eta_ch   = np.array([90.0,89.1,88.7,89.0,90.1,88.3,87.4,87.3,87.6,87.4,85.3,84.5,84.4,84.8],dtype=float)*1E-2
I_prod   = np.array([12.0,13.0,13.3,12.1,13.0,15.4,16.6,16.5,14.9,18.9,22.4,23.5,23.0,20.6],dtype=float)

f_eta_u_eta_prod   = eta_u/eta_prod 
f_eta_u_eta_prod_2 = eta_ch*I_prod/(e/m*m_A) 

for i in range(0,len(f_eta_u_eta_prod)):
    print(m_A[i],f_eta_u_eta_prod[i],f_eta_u_eta_prod_2[i])