#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:51:04 2020

@author: adrian

############################################################################
Description:    This function corresponds to the F(theta) erosion function 
############################################################################
Inputs:         1) Fmax parameter [-]
                2) theta_max parameter [deg] at which F(theta_max) = Fmax 
                3) a parameter [-]
                4) theta [deg]: vector or value of theta at which to obtain F(theta)
                5) Ntheta: length of vector theta
############################################################################
Output:         1) F(theta) values [-]
"""

def erosion_Ftheta(Fmax,theta_max,a,theta,Ntheta):
    
    import numpy as np
    
    Ftheta = np.zeros(Ntheta,dtype=float)
    
    # Convert theta values from deg to rad
    theta_max = np.copy(theta_max)*np.pi/180.0
    theta     = np.copy(theta)*np.pi/180.0
    
    for i in range(0,Ntheta):
        if theta[i] <= theta_max:
            Ftheta[i] = 1 + (Fmax-1)*np.exp(-a*(theta[i]-theta_max)**2/theta_max**2)
        else:
#            Ftheta[i] = Fmax*(1-np.pi/(2*theta[i]))*((1-theta_max/theta[i])/(1-2*theta_max/np.pi))**2.0
            Ftheta[i] = Fmax-Fmax*np.pi/(2*theta[i])*((1-theta_max/theta[i])/(1-2*theta_max/np.pi))**2.0
#            Ftheta[i] = Fmax*(1-theta_max/(2*theta[i]))*((1-theta_max/theta[i])/(1-2*theta_max/np.pi))**2.0
    
    return Ftheta
    
if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 

    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))
    
    save_flag = 1
    # Plots save flag
    #figs_format = ".eps"
    figs_format = ".png"
    #figs_format = ".pdf"
    
    path_out = "CHEOPS_Final_figs/"
    
    # Set options for LaTeX font
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    font_size           = 25
    font_size_legend    = font_size - 15
    ticks_size          = 18
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
    
    Fmax      = 3
    theta_max = 60
#    a = 1.0/8.0
    a         = 5.0
#    a = 3.63786
#    theta  = np.array([30],dtype=float) 
    theta  = np.linspace(0,90,91) 
    Ntheta = len(theta) 
    
    Ftheta = erosion_Ftheta(Fmax,theta_max,a,theta,Ntheta)
    
    plt.figure("F_theta")
    text1 = r"$a = $ "+"{:.2f}".format(a)+", $F_{max} = $ "+"{:.2f}".format(Fmax)+r", $ \theta = $ "+"{:.0f}".format(theta_max)+" (deg)"
    plt.plot(theta,Ftheta,'k',label=text1)
#    plt.plot(theta_max*np.ones(np.shape(theta)),np.linspace(Ftheta.min(),Ftheta.max(),Ntheta))
#    plt.plot(theta,np.zeros(Ntheta))
    plt.xlabel(r"$\theta$ (deg)",fontsize = font_size)
    plt.title(r"$F(\theta) (-)$",fontsize = font_size)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    plt.legend(fontsize = font_size_legend,loc=2) 
    if save_flag == 1:
        plt.savefig(path_out+"erosion_Ftheta"+figs_format,bbox_inches='tight')
        plt.close()
        
    # Plot for several values of a
    a = np.array([1.0/8.0,1,2,3,4,5],dtype=float)
    colors = ['k','r','g','b','m','c']
    for i in range(0,len(a)):
        
        Ftheta = erosion_Ftheta(Fmax,theta_max,a[i],theta,Ntheta)
        
        plt.figure("F_theta a values")
        text1 = r"$a = $ "+"{:.2f}".format(a[i])+", $F_{max} = $ "+"{:.2f}".format(Fmax)+r", $ \theta = $ "+"{:.0f}".format(theta_max)+" (deg)"
        plt.plot(theta,Ftheta,color=colors[i],label=text1)
    #    plt.plot(theta_max*np.ones(np.shape(theta)),np.linspace(Ftheta.min(),Ftheta.max(),Ntheta))
    #    plt.plot(theta,np.zeros(Ntheta))
    plt.xlabel(r"$\theta$ (deg)",fontsize = font_size)
    plt.title(r"$F(\theta) (-)$",fontsize = font_size)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    plt.legend(fontsize = font_size_legend,loc=3) 
    if save_flag == 1:
        plt.savefig(path_out+"erosion_Ftheta_a_values"+figs_format,bbox_inches='tight')
        plt.close()