#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:51:04 2020

@author: adrian

############################################################################
Description:    This function corresponds to the Y0(E) erosion function 
############################################################################
Inputs:         1) c parameter [mm3/C-eV]
                2) E_th parameter [eV]: threshold energy for erosion
                4) E [eV]: vector or value of energy at which to obtain Y0(E)
                5) NE: length of vector E
############################################################################
Output:         1) Y0(E) values [mm3/C]
"""

def erosion_Y0(c,E_th,E,NE):
    
    import numpy as np
    
    Y0_E = np.zeros(NE,dtype=float)
    
    for i in range(0,NE):
        if E[i] > E_th:
            Y0_E[i] = c*(E[i]-E_th)
    
    return Y0_E
    
if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 
    
    from erosion_Ftheta import erosion_Ftheta

    
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
    
    def fmt_func_exponent_cbar(x,pos):
        a, b = '{:.1e}'.format(x).split('e')
        b = int(b)
        return r'${}\cdot$10$^{{{}}}$'.format(a, b)
    
    def fmt_func_exponent_lines(x):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${}\cdot$10$^{{{}}}$'.format(a, b)
    
    # Data for Y0(E) ----------------------------------------------------------
    c    = 3.75e-5 # [mm3/C-eV]
    E_th = 60      # [eV]
    
    E  = np.linspace(30,200,100) 
    NE = len(E) 
    
    Y0_E = erosion_Y0(c,E_th,E,NE)
    
    plt.figure("Y0_E")
#    text1 = r"$c = $ "+"{:.2e}".format(c)+" (mm$^3$/C-eV), $E_{th} = $ "+"{:.0f}".format(E_th)+" (eV)"
    text1 = r"$c = $ "+fmt_func_exponent_lines(c)+" (mm$^3$/C-eV), $E_{th} = $ "+"{:.0f}".format(E_th)+" (eV)"
    plt.plot(E,Y0_E,'k',label=text1)
#    plt.plot(E_th*np.ones(np.shape(E)),np.linspace(Y0_E.min(),Y0_E.max(),NE))
    plt.xlabel(r"$E$ (eV)",fontsize = font_size)
    plt.title(r"$Y_0(E)$ (mm$^3$/C)",fontsize = font_size)
    plt.xticks(fontsize = ticks_size) 
    plt.yticks(fontsize = ticks_size)
    plt.legend(fontsize = font_size_legend,loc=4) 
    ax = plt.gca()
    ax.set_ylim(0,3E-3)
#    plt.grid()
    if save_flag == 1:
        plt.savefig(path_out+"erosion_Y0_E"+figs_format,bbox_inches='tight')
        plt.close()
    
    
#    # Data for F(theta) -------------------------------------------------------
#    Fmax = 3
#    theta_max = 60
#    a = 3
##    a = 3.63786
##    theta  = np.array([30],dtype=float) 
#    theta  = np.linspace(0,90,91) 
#    Ntheta = len(theta) 
#    
#    Ftheta = erosion_Ftheta(Fmax,theta_max,a,theta,Ntheta)
#    
#    print(Ftheta)
#    
#    plt.figure("F_theta")
#    plt.plot(theta,Ftheta)
#    plt.plot(theta_max*np.ones(np.shape(theta)),np.linspace(Ftheta.min(),Ftheta.max(),Ntheta))
#    plt.plot(theta,np.zeros(Ntheta))
#    plt.xlabel(r"$\theta$ (deg)")
#    plt.title(r"$F(\theta)$")
    