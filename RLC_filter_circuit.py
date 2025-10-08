#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:51:12 2020

@author: adrian
"""

def RLC_filter_circuit(V_ref,V_ps,Vd0,Id,RLC_filter_int,dt,R,L,C):
    
    import numpy as np
    

    Vd1            = Vd0 + dt*(RLC_filter_int + (V_ps - Vd0)/(R*C) - Id/C)
    RLC_filter_int = RLC_filter_int + (V_ps - Vd1)*dt/(L*C)
        
    
    
    return [Vd1,RLC_filter_int]


def RLC_filter_circuit2(Vd0,Id0,Id1,dt,R,L,C):
    
    import numpy as np
    
    deltaI = Id1-Id0
    deltaV = -deltaI/(1/R + dt/L + C/dt)
    Vd1    = Vd0 + deltaV
        
    
    
    return [Vd1,deltaV]


if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 
    from scipy import interpolate
    from scipy.interpolate import RegularGridInterpolator

    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))
    
    V_ps           = 4.98900000e+02
    R              = 1.00000000e+02
    L              = 3.40000000e-04
    C              = 6.00000000e-06
    Id             = 5.0 
    V_ref          = 0.0
    Vd0            = V_ps
    RLC_filter_int = 0.0
    dt             = 1.5E-8/5.0
    
    [Vd1,RLC_filter_int] = RLC_filter_circuit(V_ref,V_ps,Vd0,Id,RLC_filter_int,dt,R,L,C)
    
    
    print(Vd1)
    print(RLC_filter_int)
    
    
    

#        subroutine RLC_filter_circuit(sim_params,isD_eFld_f0,isD_eFld_f1)
#
#            ! This subroutine solves the RLC filter circuit updating the anode wall potential and corresponding discharge voltage
#            ! between the anode and the cathode
#
#            ! ----------------------------------------- DECLARATION PART OF THE SUBROUTINE ----------------------------------------
#            ! ---------------------------------------------------------------------------------------------------------------------
#
#            ! ---------------------------------------------------- INPUTS ---------------------------------------------------------
#            type(sim_params_t),intent(in)::           sim_params         ! Simulation parameters structure
#            type(isD_eFld_t),intent(in)::             isD_eFld_f0        ! Interstep data at the electron fluid mesh faces at
#                                                                         ! the previous electron fluid subiteration
#            ! ------------------------------------------------ INPUTS/OUTPUTS -----------------------------------------------------
#            type(isD_eFld_t),intent(inout)::          isD_eFld_f1        ! Interstep data at the electron fluid mesh faces at
#                                                                         ! the current electron fluid subiteration
#            ! ---------------------------------------------------- OUTPUTS --------------------------------------------------------
#
#            ! ----------------------------------------------- INTERNAL VARIABLES --------------------------------------------------
#            real*8                                    RC, LC, C, V_ps, Id
#
#            ! --------------------------------------- EXECUTABLE PART OF THE SUBROUTINE -------------------------------------------
#            ! ---------------------------------------------------------------------------------------------------------------------
#
#            ! Retrieve the C, LC and RC values
#            C  = sim_params%C_filter
#            RC = sim_params%R_filter*C
#            LC = sim_params%L_filter*C
#            ! Retrieve the power supply voltage with respect to the potential reference
#!            V_ps = sim_params%V_ps - sim_params%phi_ref
#            V_ps = sim_params%V_ps
#            ! Retrieve the discharge current obtained at current electron fluid subiteration
#            Id = isD_eFld_f1%it_match_conv_hist%Id
#
#            ! Update the discharge voltage solving the RLC circuit
#            isD_eFld_f1%Vd = isD_eFld_f0%Vd + sim_params%dt_e*(isD_eFld_f0%RLC_filter_int + (V_ps - isD_eFld_f0%Vd)/RC - Id/C)
#            ! Update the anode wall potential
#            isD_eFld_f1%anode_potential = isD_eFld_f1%Vd + sim_params%phi_ref
#
#            ! Update the integral part
#            isD_eFld_f1%RLC_filter_int = isD_eFld_f0%RLC_filter_int + (V_ps - isD_eFld_f1%Vd)*sim_params%dt_e/LC
#
#        end subroutine RLC_filter_circuit