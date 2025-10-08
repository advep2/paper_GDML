# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:16:42 2018

@author: adrian

############################################################################
Description:    This python script performs the time-average of the given data
                from HET sims
############################################################################
Inputs:         1) nsteps: number of last print-out steps over which the 
                               time-average is performed
                2) mean_type: 0 for mean using last steps; 1 for mean using
                              step_i and step_f
                3) last_steps: last number of print-out steps over which the 
                               time-average is performed in case mean_type = 0
                4) stepi,step_f: first and last print-out steps over which the 
                                 time-average is performed in case mean_type = 1
                5) Variables to be time-averaged
############################################################################
Output:        1) Time-averaged variables
"""

def HET_sims_mean(nsteps,mean_type,last_steps,step_i,step_f,Z_ion_spe,
                  num_ion_spe,num_neu_spe,phi,Er,Ez,
                  Efield,Br,Bz,Bfield,nn1,nn2,nn3,ni1,ni2,ni3,ni4,ne,fn1_x,
                  fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fn3_x,fn3_y,fn3_z,fi1_x,fi1_y,
                  fi1_z,fi2_x,fi2_y,fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,
                  un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,ui1_x,
                  ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,
                  ui4_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,ji3_z,
                  ji4_x,ji4_y,ji4_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,
                  ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Tn3,Ti1,Ti2,Ti3,Ti4,Te,
                  n_mp_n1,n_mp_n2,n_mp_n3,n_mp_i1,n_mp_i2,n_mp_i3,n_mp_i4,
                  avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                  neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                  ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
                  ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,ndot_CEX01_i3,ndot_CEX02_i4,
                  ne_cath,Te_cath,nu_cath,ndot_cath,
                  F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,
                  nu_ei2,nu_i01,nu_i02,nu_i12,nu_ex,
                  Boltz,Boltz_dim,phi_elems,phi_faces,ne_elems,ne_faces,Te_elems,
                  Te_faces,err_interp_n,
                  f_split_adv,f_split_qperp,f_split_qpara,f_split_qb,f_split_Pperp,
                  f_split_Ppara,f_split_ecterm,f_split_inel,
                  je_perp_elems,je_theta_elems,je_para_elems,je_z_elems,je_r_elems,
                  je_perp_faces,je_theta_faces,je_para_faces,je_z_faces,je_r_faces,
                  F_theta_elems,Hall_par_elems,Hall_par_eff_elems,nu_e_tot_elems,
                  nu_e_tot_eff_elems,F_theta_faces,Hall_par_faces,Hall_par_eff_faces,
                  nu_e_tot_faces,nu_e_tot_eff_faces,nu_en_elems,nu_ei1_elems,nu_ei2_elems,
                  nu_i01_elems,nu_i02_elems,nu_i12_elems,nu_ex_elems,nu_en_faces,
                  nu_ei1_faces,nu_ei2_faces,nu_i01_faces,nu_i02_faces,nu_i12_faces,
                  nu_ex_faces):
    
    import numpy as np
    
    # Parameters
    e = 1.6021766E-19
    
    if mean_type == 0:
        # Electric potential
        phi_mean    = np.nanmean(phi[:,:,nsteps-last_steps::],axis=2)
        # Electric field
        Er_mean     = np.nanmean(Er[:,:,nsteps-last_steps::],axis=2)
        Ez_mean     = np.nanmean(Ez[:,:,nsteps-last_steps::],axis=2)      
        Efield_mean = np.nanmean(Efield[:,:,nsteps-last_steps::],axis=2)            
        # Particle densities, fluxes and velocities
        nn1_mean   = np.nanmean(nn1[:,:,nsteps-last_steps::],axis=2)
        nn2_mean   = np.nanmean(nn2[:,:,nsteps-last_steps::],axis=2)
        nn3_mean   = np.nanmean(nn3[:,:,nsteps-last_steps::],axis=2)
        ni1_mean   = np.nanmean(ni1[:,:,nsteps-last_steps::],axis=2)
        ni2_mean   = np.nanmean(ni2[:,:,nsteps-last_steps::],axis=2)
        ni3_mean   = np.nanmean(ni3[:,:,nsteps-last_steps::],axis=2)
        ni4_mean   = np.nanmean(ni4[:,:,nsteps-last_steps::],axis=2)
        ne_mean    = np.nanmean(ne[:,:,nsteps-last_steps::],axis=2)
        # Particle fluxes, currents and fluid velocities
        fn1_x_mean = np.nanmean(fn1_x[:,:,nsteps-last_steps::],axis=2)
        fn1_y_mean = np.nanmean(fn1_y[:,:,nsteps-last_steps::],axis=2)
        fn1_z_mean = np.nanmean(fn1_z[:,:,nsteps-last_steps::],axis=2)
        fn2_x_mean = np.nanmean(fn2_x[:,:,nsteps-last_steps::],axis=2)
        fn2_y_mean = np.nanmean(fn2_y[:,:,nsteps-last_steps::],axis=2)
        fn2_z_mean = np.nanmean(fn2_z[:,:,nsteps-last_steps::],axis=2)
        fn3_x_mean = np.nanmean(fn3_x[:,:,nsteps-last_steps::],axis=2)
        fn3_y_mean = np.nanmean(fn3_y[:,:,nsteps-last_steps::],axis=2)
        fn3_z_mean = np.nanmean(fn3_z[:,:,nsteps-last_steps::],axis=2)
        fi1_x_mean = np.nanmean(fi1_x[:,:,nsteps-last_steps::],axis=2)
        fi1_y_mean = np.nanmean(fi1_y[:,:,nsteps-last_steps::],axis=2)
        fi1_z_mean = np.nanmean(fi1_z[:,:,nsteps-last_steps::],axis=2)
        fi2_x_mean = np.nanmean(fi2_x[:,:,nsteps-last_steps::],axis=2)
        fi2_y_mean = np.nanmean(fi2_y[:,:,nsteps-last_steps::],axis=2)
        fi2_z_mean = np.nanmean(fi2_z[:,:,nsteps-last_steps::],axis=2)
        fi3_x_mean = np.nanmean(fi3_x[:,:,nsteps-last_steps::],axis=2)
        fi3_y_mean = np.nanmean(fi3_y[:,:,nsteps-last_steps::],axis=2)
        fi3_z_mean = np.nanmean(fi3_z[:,:,nsteps-last_steps::],axis=2)
        fi4_x_mean = np.nanmean(fi4_x[:,:,nsteps-last_steps::],axis=2)
        fi4_y_mean = np.nanmean(fi4_y[:,:,nsteps-last_steps::],axis=2)
        fi4_z_mean = np.nanmean(fi4_z[:,:,nsteps-last_steps::],axis=2)
        un1_x_mean = np.divide(fn1_x_mean,nn1_mean) 
        un1_y_mean = np.divide(fn1_y_mean,nn1_mean) 
        un1_z_mean = np.divide(fn1_z_mean,nn1_mean)
        if np.all(nn2_mean == 0.0):
            un2_x_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
            un2_y_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
            un2_z_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
        else:
            un2_x_mean = np.divide(fn2_x_mean,nn2_mean) 
            un2_y_mean = np.divide(fn2_y_mean,nn2_mean) 
            un2_z_mean = np.divide(fn2_z_mean,nn2_mean)
        if np.all(nn3_mean == 0.0):
            un3_x_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
            un3_y_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
            un3_z_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
        else:
            un3_x_mean = np.divide(fn3_x_mean,nn3_mean) 
            un3_y_mean = np.divide(fn3_y_mean,nn3_mean) 
            un3_z_mean = np.divide(fn3_z_mean,nn3_mean)
        ui1_x_mean = np.divide(fi1_x_mean,ni1_mean) 
        ui1_y_mean = np.divide(fi1_y_mean,ni1_mean) 
        ui1_z_mean = np.divide(fi1_z_mean,ni1_mean)
        ui2_x_mean = np.divide(fi2_x_mean,ni2_mean) 
        ui2_y_mean = np.divide(fi2_y_mean,ni2_mean) 
        ui2_z_mean = np.divide(fi2_z_mean,ni2_mean)
        if np.all(ni3_mean == 0.0):
            ui3_x_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
            ui3_y_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
            ui3_z_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
        else:
            ui3_x_mean = np.divide(fi3_x_mean,ni3_mean) 
            ui3_y_mean = np.divide(fi3_y_mean,ni3_mean) 
            ui3_z_mean = np.divide(fi3_z_mean,ni3_mean)
        if np.all(ni4_mean == 0.0):
            ui4_x_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
            ui4_y_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
            ui4_z_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
        else:
            ui4_x_mean = np.divide(fi4_x_mean,ni4_mean) 
            ui4_y_mean = np.divide(fi4_y_mean,ni4_mean) 
            ui4_z_mean = np.divide(fi4_z_mean,ni4_mean)
        
        ji1_x_mean = Z_ion_spe[0]*e*fi1_x_mean
        ji1_y_mean = Z_ion_spe[0]*e*fi1_y_mean
        ji1_z_mean = Z_ion_spe[0]*e*fi1_z_mean
        ji2_x_mean = Z_ion_spe[1]*e*fi2_x_mean
        ji2_y_mean = Z_ion_spe[1]*e*fi2_y_mean
        ji2_z_mean = Z_ion_spe[1]*e*fi2_z_mean
        if num_ion_spe == 4:
            ji3_x_mean = Z_ion_spe[2]*e*fi3_x_mean
            ji3_y_mean = Z_ion_spe[2]*e*fi3_y_mean
            ji3_z_mean = Z_ion_spe[2]*e*fi3_z_mean
            ji4_x_mean = Z_ion_spe[3]*e*fi4_x_mean
            ji4_y_mean = Z_ion_spe[3]*e*fi4_y_mean
            ji4_z_mean = Z_ion_spe[3]*e*fi4_z_mean
        else:
            ji3_x_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
            ji3_y_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
            ji3_z_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
            ji4_x_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
            ji4_y_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
            ji4_z_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
                
        je_r_mean    = np.nanmean(je_r[:,:,nsteps-last_steps::],axis=2) 
        je_t_mean    = np.nanmean(je_t[:,:,nsteps-last_steps::],axis=2)
        je_z_mean    = np.nanmean(je_z[:,:,nsteps-last_steps::],axis=2)
        je_perp_mean = np.nanmean(je_perp[:,:,nsteps-last_steps::],axis=2)
        je_para_mean = np.nanmean(je_para[:,:,nsteps-last_steps::],axis=2)
        ue_r_mean    = np.divide(je_r_mean,-e*ne_mean) 
        ue_t_mean    = np.divide(je_t_mean,-e*ne_mean)
        ue_z_mean    = np.divide(je_z_mean,-e*ne_mean)
        ue_perp_mean = np.divide(je_perp_mean,-e*ne_mean)
        ue_para_mean = np.divide(je_para_mean,-e*ne_mean)
        # Compute azimuthal ExB drift velocity considering the system (z,theta,r) == (perp,theta,para)
        uthetaExB_mean = -1/Bfield**2*(Br*Ez_mean - Bz*Er_mean)
        # Temperatures
        Tn1_mean = np.nanmean(Tn1[:,:,nsteps-last_steps::],axis=2)
        Tn2_mean = np.nanmean(Tn2[:,:,nsteps-last_steps::],axis=2)
        Tn3_mean = np.nanmean(Tn3[:,:,nsteps-last_steps::],axis=2)
        Ti1_mean = np.nanmean(Ti1[:,:,nsteps-last_steps::],axis=2)
        Ti2_mean = np.nanmean(Ti2[:,:,nsteps-last_steps::],axis=2)
        Ti3_mean = np.nanmean(Ti3[:,:,nsteps-last_steps::],axis=2)
        Ti4_mean = np.nanmean(Ti4[:,:,nsteps-last_steps::],axis=2)
        Te_mean = np.nanmean(Te[:,:,nsteps-last_steps::],axis=2)
        # Number of particles per cell (PIC mesh cell variables have same size than PIC mesh nodes variables)
        n_mp_n1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_n2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_n3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_i1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_i2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_i3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_i4_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_n1_mean[0:-1,0:-1] = np.nanmean(n_mp_n1[0:-1,0:-1,nsteps-last_steps::],axis=2)
        n_mp_n2_mean[0:-1,0:-1] = np.nanmean(n_mp_n2[0:-1,0:-1,nsteps-last_steps::],axis=2)
        n_mp_n3_mean[0:-1,0:-1] = np.nanmean(n_mp_n3[0:-1,0:-1,nsteps-last_steps::],axis=2)
        n_mp_i1_mean[0:-1,0:-1] = np.nanmean(n_mp_i1[0:-1,0:-1,nsteps-last_steps::],axis=2)
        n_mp_i2_mean[0:-1,0:-1] = np.nanmean(n_mp_i2[0:-1,0:-1,nsteps-last_steps::],axis=2)
        n_mp_i3_mean[0:-1,0:-1] = np.nanmean(n_mp_i3[0:-1,0:-1,nsteps-last_steps::],axis=2)
        n_mp_i4_mean[0:-1,0:-1] = np.nanmean(n_mp_i4[0:-1,0:-1,nsteps-last_steps::],axis=2)
        # Average particle weight per cell (PIC mesh cell variables have same size than PIC mesh nodes variables)
        avg_w_n1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        avg_w_n2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        avg_w_i1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        avg_w_i2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        avg_w_n1_mean[0:-1,0:-1] = np.nanmean(avg_w_n1[0:-1,0:-1,nsteps-last_steps::],axis=2)
        avg_w_n2_mean[0:-1,0:-1] = np.nanmean(avg_w_n2[0:-1,0:-1,nsteps-last_steps::],axis=2)
        avg_w_i1_mean[0:-1,0:-1] = np.nanmean(avg_w_i1[0:-1,0:-1,nsteps-last_steps::],axis=2)
        avg_w_i2_mean[0:-1,0:-1] = np.nanmean(avg_w_i2[0:-1,0:-1,nsteps-last_steps::],axis=2)
        # Generation weights per cell (PIC mesh cell variables have same size than PIC mesh nodes variables)
        neu_gen_weights1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        neu_gen_weights2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ion_gen_weights1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ion_gen_weights2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        neu_gen_weights1_mean[0:-1,0:-1] = np.nanmean(neu_gen_weights1[0:-1,0:-1,nsteps-last_steps::],axis=2)
        neu_gen_weights2_mean[0:-1,0:-1] = np.nanmean(neu_gen_weights2[0:-1,0:-1,nsteps-last_steps::],axis=2)
        ion_gen_weights1_mean[0:-1,0:-1] = np.nanmean(ion_gen_weights1[0:-1,0:-1,nsteps-last_steps::],axis=2)
        ion_gen_weights2_mean[0:-1,0:-1] = np.nanmean(ion_gen_weights2[0:-1,0:-1,nsteps-last_steps::],axis=2)
        # Obtain the ionization source term (ni_dot) per cell for each ionization collision (PIC mesh cell variables have same size than PIC mesh nodes variables)
        ndot_ion01_n1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion02_n1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion12_i1_mean = np.zeros(np.shape(Te_mean),dtype=float)    
        ndot_ion01_n2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion02_n2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion01_n3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion02_n3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion12_i3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_CEX01_i3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_CEX02_i4_mean = np.zeros(np.shape(Te_mean),dtype=float)
        
        ndot_ion01_n1_mean[0:-1,0:-1] = np.nanmean(ndot_ion01_n1[0:-1,0:-1,nsteps-last_steps::],axis=2)
        ndot_ion02_n1_mean[0:-1,0:-1] = np.nanmean(ndot_ion02_n1[0:-1,0:-1,nsteps-last_steps::],axis=2)
        ndot_ion12_i1_mean[0:-1,0:-1] = np.nanmean(ndot_ion12_i1[0:-1,0:-1,nsteps-last_steps::],axis=2)
        ndot_ion01_n2_mean[0:-1,0:-1] = np.nanmean(ndot_ion01_n2[0:-1,0:-1,nsteps-last_steps::],axis=2)
        ndot_ion02_n2_mean[0:-1,0:-1] = np.nanmean(ndot_ion02_n2[0:-1,0:-1,nsteps-last_steps::],axis=2)
        ndot_ion01_n3_mean[0:-1,0:-1] = np.nanmean(ndot_ion01_n3[0:-1,0:-1,nsteps-last_steps::],axis=2)
        ndot_ion02_n3_mean[0:-1,0:-1] = np.nanmean(ndot_ion02_n3[0:-1,0:-1,nsteps-last_steps::],axis=2)
        ndot_ion12_i3_mean[0:-1,0:-1] = np.nanmean(ndot_ion12_i3[0:-1,0:-1,nsteps-last_steps::],axis=2)        
        ndot_CEX01_i3_mean[0:-1,0:-1] = np.nanmean(ndot_CEX01_i3[0:-1,0:-1,nsteps-last_steps::],axis=2)       
        ndot_CEX02_i4_mean[0:-1,0:-1] = np.nanmean(ndot_CEX02_i4[0:-1,0:-1,nsteps-last_steps::],axis=2)  
        
        # Obtain cathode plasma density, temperature, production frequency and source term
        ne_cath_mean   = np.mean(ne_cath[nsteps-last_steps::],axis=0)
        Te_cath_mean   = np.mean(Te_cath[nsteps-last_steps::],axis=0)
        nu_cath_mean   = np.mean(nu_cath[nsteps-last_steps::],axis=0)
        ndot_cath_mean = np.mean(ndot_cath[nsteps-last_steps::],axis=0)
        
        # Obtain anomalous term, Hall parameters and collision frequencies
        F_theta_mean      = np.nanmean(F_theta[:,:,nsteps-last_steps::],axis=2)
        Hall_par_mean     = np.nanmean(Hall_par[:,:,nsteps-last_steps::],axis=2)
        Hall_par_eff_mean = np.nanmean(Hall_par_eff[:,:,nsteps-last_steps::],axis=2)
        nu_e_tot_mean     = np.nanmean(nu_e_tot[:,:,nsteps-last_steps::],axis=2)
        nu_e_tot_eff_mean = np.nanmean(nu_e_tot_eff[:,:,nsteps-last_steps::],axis=2)
        nu_en_mean        = np.nanmean(nu_en[:,:,nsteps-last_steps::],axis=2)
        nu_ei1_mean       = np.nanmean(nu_ei1[:,:,nsteps-last_steps::],axis=2)
        nu_ei2_mean       = np.nanmean(nu_ei2[:,:,nsteps-last_steps::],axis=2)
        nu_i01_mean       = np.nanmean(nu_i01[:,:,nsteps-last_steps::],axis=2)
        nu_i02_mean       = np.nanmean(nu_i02[:,:,nsteps-last_steps::],axis=2)
        nu_i12_mean       = np.nanmean(nu_i12[:,:,nsteps-last_steps::],axis=2)
        nu_ex_mean        = np.nanmean(nu_ex[:,:,nsteps-last_steps::],axis=2)
        # Variables at the MFAM elements
        Boltz_mean              = np.nanmean(Boltz[nsteps-last_steps::,:],axis=0)
        Boltz_dim_mean          = np.nanmean(Boltz_dim[nsteps-last_steps::,:],axis=0)
        phi_elems_mean          = np.nanmean(phi_elems[nsteps-last_steps::,:],axis=0) 
        ne_elems_mean           = np.nanmean(ne_elems[nsteps-last_steps::,:],axis=0) 
        Te_elems_mean           = np.nanmean(Te_elems[nsteps-last_steps::,:],axis=0)
        je_perp_elems_mean      = np.nanmean(je_perp_elems[nsteps-last_steps::,:],axis=0)
        je_theta_elems_mean     = np.nanmean(je_theta_elems[nsteps-last_steps::,:],axis=0)
        je_para_elems_mean      = np.nanmean(je_para_elems[nsteps-last_steps::,:],axis=0)
        je_z_elems_mean         = np.nanmean(je_z_elems[nsteps-last_steps::,:],axis=0)
        je_r_elems_mean         = np.nanmean(je_r_elems[nsteps-last_steps::,:],axis=0)
        F_theta_elems_mean      = np.nanmean(F_theta_elems[nsteps-last_steps::,:],axis=0)
        Hall_par_elems_mean     = np.nanmean(Hall_par_elems[nsteps-last_steps::,:],axis=0)
        Hall_par_eff_elems_mean = np.nanmean(Hall_par_eff_elems[nsteps-last_steps::,:],axis=0)
        nu_e_tot_elems_mean     = np.nanmean(nu_e_tot_elems[nsteps-last_steps::,:],axis=0)
        nu_e_tot_eff_elems_mean = np.nanmean(nu_e_tot_eff_elems[nsteps-last_steps::,:],axis=0)
        nu_en_elems_mean        = np.nanmean(nu_en_elems[nsteps-last_steps::,:],axis=0)
        nu_ei1_elems_mean       = np.nanmean(nu_ei1_elems[nsteps-last_steps::,:],axis=0)
        nu_ei2_elems_mean       = np.nanmean(nu_ei2_elems[nsteps-last_steps::,:],axis=0)
        nu_i01_elems_mean       = np.nanmean(nu_i01_elems[nsteps-last_steps::,:],axis=0)
        nu_i02_elems_mean       = np.nanmean(nu_i02_elems[nsteps-last_steps::,:],axis=0)
        nu_i12_elems_mean       = np.nanmean(nu_i12_elems[nsteps-last_steps::,:],axis=0)
        nu_ex_elems_mean        = np.nanmean(nu_ex_elems[nsteps-last_steps::,:],axis=0)
        # Variables at the MFAM faces
        phi_faces_mean          = np.nanmean(phi_faces[nsteps-last_steps::,:],axis=0)
        ne_faces_mean           = np.nanmean(ne_faces[nsteps-last_steps::,:],axis=0) 
        Te_faces_mean           = np.nanmean(Te_faces[nsteps-last_steps::,:],axis=0)
        je_perp_faces_mean      = np.nanmean(je_perp_faces[nsteps-last_steps::,:],axis=0)
        je_theta_faces_mean     = np.nanmean(je_theta_faces[nsteps-last_steps::,:],axis=0)
        je_para_faces_mean      = np.nanmean(je_para_faces[nsteps-last_steps::,:],axis=0)
        je_z_faces_mean         = np.nanmean(je_z_faces[nsteps-last_steps::,:],axis=0)
        je_r_faces_mean         = np.nanmean(je_r_faces[nsteps-last_steps::,:],axis=0)
        F_theta_faces_mean      = np.nanmean(F_theta_faces[nsteps-last_steps::,:],axis=0)
        Hall_par_faces_mean     = np.nanmean(Hall_par_faces[nsteps-last_steps::,:],axis=0)
        Hall_par_eff_faces_mean = np.nanmean(Hall_par_eff_faces[nsteps-last_steps::,:],axis=0)
        nu_e_tot_faces_mean     = np.nanmean(nu_e_tot_faces[nsteps-last_steps::,:],axis=0)
        nu_e_tot_eff_faces_mean = np.nanmean(nu_e_tot_eff_faces[nsteps-last_steps::,:],axis=0)
        nu_en_faces_mean        = np.nanmean(nu_en_faces[nsteps-last_steps::,:],axis=0)
        nu_ei1_faces_mean       = np.nanmean(nu_ei1_faces[nsteps-last_steps::,:],axis=0)
        nu_ei2_faces_mean       = np.nanmean(nu_ei2_faces[nsteps-last_steps::,:],axis=0)
        nu_i01_faces_mean       = np.nanmean(nu_i01_faces[nsteps-last_steps::,:],axis=0)
        nu_i02_faces_mean       = np.nanmean(nu_i02_faces[nsteps-last_steps::,:],axis=0)
        nu_i12_faces_mean       = np.nanmean(nu_i12_faces[nsteps-last_steps::,:],axis=0)
        nu_ex_faces_mean        = np.nanmean(nu_ex_faces[nsteps-last_steps::,:],axis=0)
        
        # Intepolation error in plasma density
        err_interp_n_mean = np.nanmean(err_interp_n[:,:,nsteps-last_steps::],axis=2)
        
        # f_split variables
        f_split_adv_mean    = np.nanmean(f_split_adv[:,:,nsteps-last_steps::],axis=2)
        f_split_qperp_mean  = np.nanmean(f_split_qperp[:,:,nsteps-last_steps::],axis=2)
        f_split_qpara_mean  = np.nanmean(f_split_qpara[:,:,nsteps-last_steps::],axis=2)
        f_split_qb_mean     = np.nanmean(f_split_qb[:,:,nsteps-last_steps::],axis=2)
        f_split_Pperp_mean  = np.nanmean(f_split_Pperp[:,:,nsteps-last_steps::],axis=2)
        f_split_Ppara_mean  = np.nanmean(f_split_Ppara[:,:,nsteps-last_steps::],axis=2)
        f_split_ecterm_mean = np.nanmean(f_split_ecterm[:,:,nsteps-last_steps::],axis=2)
        f_split_inel_mean   = np.nanmean(f_split_inel[:,:,nsteps-last_steps::],axis=2)
    
    elif mean_type == 1:
        # Electric potential
        phi_mean    = np.nanmean(phi[:,:,step_i:step_f+1],axis=2)
        # Electric field
        Er_mean     = np.nanmean(Er[:,:,step_i:step_f+1],axis=2)
        Ez_mean     = np.nanmean(Ez[:,:,step_i:step_f+1],axis=2)      
        Efield_mean = np.nanmean(Efield[:,:,step_i:step_f+1],axis=2)            
        # Particle densities, fluxes and velocities
        nn1_mean   = np.nanmean(nn1[:,:,step_i:step_f+1],axis=2)
        nn2_mean   = np.nanmean(nn2[:,:,step_i:step_f+1],axis=2)
        nn3_mean   = np.nanmean(nn3[:,:,step_i:step_f+1],axis=2)
        ni1_mean   = np.nanmean(ni1[:,:,step_i:step_f+1],axis=2)
        ni2_mean   = np.nanmean(ni2[:,:,step_i:step_f+1],axis=2)
        ni3_mean   = np.nanmean(ni3[:,:,step_i:step_f+1],axis=2)
        ni4_mean   = np.nanmean(ni4[:,:,step_i:step_f+1],axis=2)
        ne_mean    = np.nanmean(ne[:,:,step_i:step_f+1],axis=2)
        # Particle fluxes, currents and fluid velocities
        fn1_x_mean = np.nanmean(fn1_x[:,:,step_i:step_f+1],axis=2)
        fn1_y_mean = np.nanmean(fn1_y[:,:,step_i:step_f+1],axis=2)
        fn1_z_mean = np.nanmean(fn1_z[:,:,step_i:step_f+1],axis=2)
        fn2_x_mean = np.nanmean(fn2_x[:,:,step_i:step_f+1],axis=2)
        fn2_y_mean = np.nanmean(fn2_y[:,:,step_i:step_f+1],axis=2)
        fn2_z_mean = np.nanmean(fn2_z[:,:,step_i:step_f+1],axis=2)
        fn3_x_mean = np.nanmean(fn3_x[:,:,step_i:step_f+1],axis=2)
        fn3_y_mean = np.nanmean(fn3_y[:,:,step_i:step_f+1],axis=2)
        fn3_z_mean = np.nanmean(fn3_z[:,:,step_i:step_f+1],axis=2)
        
        
        fi1_x_mean = np.nanmean(fi1_x[:,:,step_i:step_f+1],axis=2)
        fi1_y_mean = np.nanmean(fi1_y[:,:,step_i:step_f+1],axis=2)
        fi1_z_mean = np.nanmean(fi1_z[:,:,step_i:step_f+1],axis=2)
        fi2_x_mean = np.nanmean(fi2_x[:,:,step_i:step_f+1],axis=2)
        fi2_y_mean = np.nanmean(fi2_y[:,:,step_i:step_f+1],axis=2)
        fi2_z_mean = np.nanmean(fi2_z[:,:,step_i:step_f+1],axis=2)
        fi3_x_mean = np.nanmean(fi3_x[:,:,step_i:step_f+1],axis=2)
        fi3_y_mean = np.nanmean(fi3_y[:,:,step_i:step_f+1],axis=2)
        fi3_z_mean = np.nanmean(fi3_z[:,:,step_i:step_f+1],axis=2)
        fi4_x_mean = np.nanmean(fi4_x[:,:,step_i:step_f+1],axis=2)
        fi4_y_mean = np.nanmean(fi4_y[:,:,step_i:step_f+1],axis=2)
        fi4_z_mean = np.nanmean(fi4_z[:,:,step_i:step_f+1],axis=2)
        
        un1_x_mean = np.divide(fn1_x_mean,nn1_mean) 
        un1_y_mean = np.divide(fn1_y_mean,nn1_mean) 
        un1_z_mean = np.divide(fn1_z_mean,nn1_mean)
        if np.all(nn2_mean == 0.0):
            un2_x_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
            un2_y_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
            un2_z_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
        else:
            un2_x_mean = np.divide(fn2_x_mean,nn2_mean) 
            un2_y_mean = np.divide(fn2_y_mean,nn2_mean) 
            un2_z_mean = np.divide(fn2_z_mean,nn2_mean)
        if np.all(nn3_mean == 0.0):
            un3_x_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
            un3_y_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
            un3_z_mean = np.zeros(np.shape(un1_x_mean),dtype=float) 
        else:
            un3_x_mean = np.divide(fn3_x_mean,nn3_mean) 
            un3_y_mean = np.divide(fn3_y_mean,nn3_mean) 
            un3_z_mean = np.divide(fn3_z_mean,nn3_mean)
        ui1_x_mean = np.divide(fi1_x_mean,ni1_mean) 
        ui1_y_mean = np.divide(fi1_y_mean,ni1_mean) 
        ui1_z_mean = np.divide(fi1_z_mean,ni1_mean)
        ui2_x_mean = np.divide(fi2_x_mean,ni2_mean) 
        ui2_y_mean = np.divide(fi2_y_mean,ni2_mean) 
        ui2_z_mean = np.divide(fi2_z_mean,ni2_mean)
        if np.all(ni3_mean == 0.0):
            ui3_x_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
            ui3_y_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
            ui3_z_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
        else:
            ui3_x_mean = np.divide(fi3_x_mean,ni3_mean) 
            ui3_y_mean = np.divide(fi3_y_mean,ni3_mean) 
            ui3_z_mean = np.divide(fi3_z_mean,ni3_mean)
        if np.all(ni4_mean == 0.0):
            ui4_x_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
            ui4_y_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
            ui4_z_mean = np.zeros(np.shape(ui1_x_mean),dtype=float) 
        else:
            ui4_x_mean = np.divide(fi4_x_mean,ni4_mean) 
            ui4_y_mean = np.divide(fi4_y_mean,ni4_mean) 
            ui4_z_mean = np.divide(fi4_z_mean,ni4_mean)
        
        ji1_x_mean = Z_ion_spe[0]*e*fi1_x_mean
        ji1_y_mean = Z_ion_spe[0]*e*fi1_y_mean
        ji1_z_mean = Z_ion_spe[0]*e*fi1_z_mean
        ji2_x_mean = Z_ion_spe[1]*e*fi2_x_mean
        ji2_y_mean = Z_ion_spe[1]*e*fi2_y_mean
        ji2_z_mean = Z_ion_spe[1]*e*fi2_z_mean
        if num_ion_spe == 4:
            ji3_x_mean = Z_ion_spe[2]*e*fi3_x_mean
            ji3_y_mean = Z_ion_spe[2]*e*fi3_y_mean
            ji3_z_mean = Z_ion_spe[2]*e*fi3_z_mean
            ji4_x_mean = Z_ion_spe[3]*e*fi4_x_mean
            ji4_y_mean = Z_ion_spe[3]*e*fi4_y_mean
            ji4_z_mean = Z_ion_spe[3]*e*fi4_z_mean
        else:
            ji3_x_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
            ji3_y_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
            ji3_z_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
            ji4_x_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
            ji4_y_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
            ji4_z_mean = np.zeros(np.shape(ji1_x_mean),dtype=float) 
            
        je_r_mean    = np.nanmean(je_r[:,:,step_i:step_f+1],axis=2) 
        je_t_mean    = np.nanmean(je_t[:,:,step_i:step_f+1],axis=2)
        je_z_mean    = np.nanmean(je_z[:,:,step_i:step_f+1],axis=2)
        je_perp_mean = np.nanmean(je_perp[:,:,step_i:step_f+1],axis=2)
        je_para_mean = np.nanmean(je_para[:,:,step_i:step_f+1],axis=2)
        ue_r_mean    = np.divide(je_r_mean,-e*ne_mean) 
        ue_t_mean    = np.divide(je_t_mean,-e*ne_mean)
        ue_z_mean    = np.divide(je_z_mean,-e*ne_mean)
        ue_perp_mean = np.divide(je_perp_mean,-e*ne_mean)
        ue_para_mean = np.divide(je_para_mean,-e*ne_mean)
        # Compute azimuthal ExB drift velocity considering the system (z,theta,r) == (perp,theta,para)
        uthetaExB_mean = -1/Bfield**2*(Br*Ez_mean - Bz*Er_mean)
        # Temperatures
        Tn1_mean = np.nanmean(Tn1[:,:,step_i:step_f+1],axis=2)
        Tn2_mean = np.nanmean(Tn2[:,:,step_i:step_f+1],axis=2)
        Tn3_mean = np.nanmean(Tn3[:,:,step_i:step_f+1],axis=2)
        Ti1_mean = np.nanmean(Ti1[:,:,step_i:step_f+1],axis=2)
        Ti2_mean = np.nanmean(Ti2[:,:,step_i:step_f+1],axis=2)
        Ti3_mean = np.nanmean(Ti3[:,:,step_i:step_f+1],axis=2)
        Ti4_mean = np.nanmean(Ti4[:,:,step_i:step_f+1],axis=2)
        Te_mean = np.nanmean(Te[:,:,step_i:step_f+1],axis=2)
        # Number of particles per cell (PIC mesh cell variables have same size than PIC mesh nodes variables)
        n_mp_n1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_n2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_n3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_i1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_i2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_i3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_i4_mean = np.zeros(np.shape(Te_mean),dtype=float)
        n_mp_n1_mean[0:-1,0:-1] = np.nanmean(n_mp_n1[0:-1,0:-1,step_i:step_f+1],axis=2)
        n_mp_n2_mean[0:-1,0:-1] = np.nanmean(n_mp_n2[0:-1,0:-1,step_i:step_f+1],axis=2)
        n_mp_n3_mean[0:-1,0:-1] = np.nanmean(n_mp_n3[0:-1,0:-1,step_i:step_f+1],axis=2)
        n_mp_i1_mean[0:-1,0:-1] = np.nanmean(n_mp_i1[0:-1,0:-1,step_i:step_f+1],axis=2)
        n_mp_i2_mean[0:-1,0:-1] = np.nanmean(n_mp_i2[0:-1,0:-1,step_i:step_f+1],axis=2)
        n_mp_i3_mean[0:-1,0:-1] = np.nanmean(n_mp_i3[0:-1,0:-1,step_i:step_f+1],axis=2)
        n_mp_i4_mean[0:-1,0:-1] = np.nanmean(n_mp_i4[0:-1,0:-1,step_i:step_f+1],axis=2)
        # Average particle weight per cell (PIC mesh cell variables have same size than PIC mesh nodes variables)
        avg_w_n1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        avg_w_n2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        avg_w_i1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        avg_w_i2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        avg_w_n1_mean[0:-1,0:-1] = np.nanmean(avg_w_n1[0:-1,0:-1,step_i:step_f+1],axis=2)
        avg_w_n2_mean[0:-1,0:-1] = np.nanmean(avg_w_n2[0:-1,0:-1,step_i:step_f+1],axis=2)
        avg_w_i1_mean[0:-1,0:-1] = np.nanmean(avg_w_i1[0:-1,0:-1,step_i:step_f+1],axis=2)
        avg_w_i2_mean[0:-1,0:-1] = np.nanmean(avg_w_i2[0:-1,0:-1,step_i:step_f+1],axis=2)
        # Generation weights per cell (PIC mesh cell variables have same size than PIC mesh nodes variables)
        neu_gen_weights1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        neu_gen_weights2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ion_gen_weights1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ion_gen_weights2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        neu_gen_weights1_mean[0:-1,0:-1] = np.nanmean(neu_gen_weights1[0:-1,0:-1,step_i:step_f+1],axis=2)
        neu_gen_weights2_mean[0:-1,0:-1] = np.nanmean(neu_gen_weights2[0:-1,0:-1,step_i:step_f+1],axis=2)
        ion_gen_weights1_mean[0:-1,0:-1] = np.nanmean(ion_gen_weights1[0:-1,0:-1,step_i:step_f+1],axis=2)
        ion_gen_weights2_mean[0:-1,0:-1] = np.nanmean(ion_gen_weights2[0:-1,0:-1,step_i:step_f+1],axis=2)
        # Obtain the ionization source term (ni_dot) per cell for each ionization collision (PIC mesh cell variables have same size than PIC mesh nodes variables)
        ndot_ion01_n1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion02_n1_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion12_i1_mean = np.zeros(np.shape(Te_mean),dtype=float)  
        ndot_ion01_n2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion02_n2_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion01_n3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion02_n3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion12_i3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_CEX01_i3_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_CEX02_i4_mean = np.zeros(np.shape(Te_mean),dtype=float)
        ndot_ion01_n1_mean[0:-1,0:-1] = np.nanmean(ndot_ion01_n1[0:-1,0:-1,step_i:step_f+1],axis=2)
        ndot_ion02_n1_mean[0:-1,0:-1] = np.nanmean(ndot_ion02_n1[0:-1,0:-1,step_i:step_f+1],axis=2)
        ndot_ion12_i1_mean[0:-1,0:-1] = np.nanmean(ndot_ion12_i1[0:-1,0:-1,step_i:step_f+1],axis=2)        
        ndot_ion01_n2_mean[0:-1,0:-1] = np.nanmean(ndot_ion01_n2[0:-1,0:-1,step_i:step_f+1],axis=2)
        ndot_ion02_n2_mean[0:-1,0:-1] = np.nanmean(ndot_ion02_n2[0:-1,0:-1,step_i:step_f+1],axis=2)
        ndot_ion01_n3_mean[0:-1,0:-1] = np.nanmean(ndot_ion01_n3[0:-1,0:-1,step_i:step_f+1],axis=2)
        ndot_ion02_n3_mean[0:-1,0:-1] = np.nanmean(ndot_ion02_n3[0:-1,0:-1,step_i:step_f+1],axis=2)
        ndot_ion12_i3_mean[0:-1,0:-1] = np.nanmean(ndot_ion12_i3[0:-1,0:-1,step_i:step_f+1],axis=2)   
        ndot_CEX01_i3_mean[0:-1,0:-1] = np.nanmean(ndot_CEX01_i3[0:-1,0:-1,step_i:step_f+1],axis=2)   
        ndot_CEX02_i4_mean[0:-1,0:-1] = np.nanmean(ndot_CEX02_i4[0:-1,0:-1,step_i:step_f+1],axis=2)   
        
        # Obtain cathode plasma density, temperature, production frequency and source term
        ne_cath_mean   = np.mean(ne_cath[step_i:step_f+1],axis=0)
        Te_cath_mean   = np.mean(Te_cath[step_i:step_f+1],axis=0)
        nu_cath_mean   = np.mean(nu_cath[step_i:step_f+1],axis=0)
        ndot_cath_mean = np.mean(ndot_cath[step_i:step_f+1],axis=0)
        # Obtain anomalous term, Hall parameters and collision frequencies
        F_theta_mean      = np.nanmean(F_theta[:,:,step_i:step_f+1],axis=2)
        Hall_par_mean     = np.nanmean(Hall_par[:,:,step_i:step_f+1],axis=2)
        Hall_par_eff_mean = np.nanmean(Hall_par_eff[:,:,step_i:step_f+1],axis=2)
        nu_e_tot_mean     = np.nanmean(nu_e_tot[:,:,step_i:step_f+1],axis=2)
        nu_e_tot_eff_mean = np.nanmean(nu_e_tot_eff[:,:,step_i:step_f+1],axis=2)
        nu_en_mean        = np.nanmean(nu_en[:,:,step_i:step_f+1],axis=2)
        nu_ei1_mean       = np.nanmean(nu_ei1[:,:,step_i:step_f+1],axis=2)
        nu_ei2_mean       = np.nanmean(nu_ei2[:,:,step_i:step_f+1],axis=2)
        nu_i01_mean       = np.nanmean(nu_i01[:,:,step_i:step_f+1],axis=2)
        nu_i02_mean       = np.nanmean(nu_i02[:,:,step_i:step_f+1],axis=2)
        nu_i12_mean       = np.nanmean(nu_i12[:,:,step_i:step_f+1],axis=2)
        nu_ex_mean        = np.nanmean(nu_ex[:,:,step_i:step_f+1],axis=2)
        # Variables at the MFAM elements
        Boltz_mean              = np.nanmean(Boltz[step_i:step_f+1,:],axis=0)
        Boltz_dim_mean          = np.nanmean(Boltz_dim[step_i:step_f+1,:],axis=0)
        phi_elems_mean          = np.nanmean(phi_elems[step_i:step_f+1,:],axis=0) 
        ne_elems_mean           = np.nanmean(ne_elems[step_i:step_f+1,:],axis=0) 
        Te_elems_mean           = np.nanmean(Te_elems[step_i:step_f+1,:],axis=0)
        je_perp_elems_mean      = np.nanmean(je_perp_elems[step_i:step_f+1,:],axis=0)
        je_theta_elems_mean     = np.nanmean(je_theta_elems[step_i:step_f+1,:],axis=0)
        je_para_elems_mean      = np.nanmean(je_para_elems[step_i:step_f+1,:],axis=0)
        je_z_elems_mean         = np.nanmean(je_z_elems[step_i:step_f+1,:],axis=0)
        je_r_elems_mean         = np.nanmean(je_r_elems[step_i:step_f+1,:],axis=0)
        F_theta_elems_mean      = np.nanmean(F_theta_elems[step_i:step_f+1,:],axis=0)
        Hall_par_elems_mean     = np.nanmean(Hall_par_elems[step_i:step_f+1,:],axis=0)
        Hall_par_eff_elems_mean = np.nanmean(Hall_par_eff_elems[step_i:step_f+1,:],axis=0)
        nu_e_tot_elems_mean     = np.nanmean(nu_e_tot_elems[step_i:step_f+1,:],axis=0)
        nu_e_tot_eff_elems_mean = np.nanmean(nu_e_tot_eff_elems[step_i:step_f+1,:],axis=0)
        nu_en_elems_mean        = np.nanmean(nu_en_elems[step_i:step_f+1,:],axis=0)
        nu_ei1_elems_mean       = np.nanmean(nu_ei1_elems[step_i:step_f+1,:],axis=0)
        nu_ei2_elems_mean       = np.nanmean(nu_ei2_elems[step_i:step_f+1,:],axis=0)
        nu_i01_elems_mean       = np.nanmean(nu_i01_elems[step_i:step_f+1,:],axis=0)
        nu_i02_elems_mean       = np.nanmean(nu_i02_elems[step_i:step_f+1,:],axis=0)
        nu_i12_elems_mean       = np.nanmean(nu_i12_elems[step_i:step_f+1,:],axis=0)
        nu_ex_elems_mean        = np.nanmean(nu_ex_elems[step_i:step_f+1,:],axis=0)
        # Variables at the MFAM faces
        phi_faces_mean          = np.nanmean(phi_faces[step_i:step_f+1,:],axis=0)
        ne_faces_mean           = np.nanmean(ne_faces[step_i:step_f+1,:],axis=0) 
        Te_faces_mean           = np.nanmean(Te_faces[step_i:step_f+1,:],axis=0)
        je_perp_faces_mean      = np.nanmean(je_perp_faces[step_i:step_f+1,:],axis=0)
        je_theta_faces_mean     = np.nanmean(je_theta_faces[step_i:step_f+1,:],axis=0)
        je_para_faces_mean      = np.nanmean(je_para_faces[step_i:step_f+1,:],axis=0)
        je_z_faces_mean         = np.nanmean(je_z_faces[step_i:step_f+1,:],axis=0)
        je_r_faces_mean         = np.nanmean(je_r_faces[step_i:step_f+1,:],axis=0)
        F_theta_faces_mean      = np.nanmean(F_theta_faces[step_i:step_f+1,:],axis=0)
        Hall_par_faces_mean     = np.nanmean(Hall_par_faces[step_i:step_f+1,:],axis=0)
        Hall_par_eff_faces_mean = np.nanmean(Hall_par_eff_faces[step_i:step_f+1,:],axis=0)
        nu_e_tot_faces_mean     = np.nanmean(nu_e_tot_faces[step_i:step_f+1,:],axis=0)
        nu_e_tot_eff_faces_mean = np.nanmean(nu_e_tot_eff_faces[step_i:step_f+1,:],axis=0)
        nu_en_faces_mean        = np.nanmean(nu_en_faces[step_i:step_f+1,:],axis=0)
        nu_ei1_faces_mean       = np.nanmean(nu_ei1_faces[step_i:step_f+1,:],axis=0)
        nu_ei2_faces_mean       = np.nanmean(nu_ei2_faces[step_i:step_f+1,:],axis=0)
        nu_i01_faces_mean       = np.nanmean(nu_i01_faces[step_i:step_f+1,:],axis=0)
        nu_i02_faces_mean       = np.nanmean(nu_i02_faces[step_i:step_f+1,:],axis=0)
        nu_i12_faces_mean       = np.nanmean(nu_i12_faces[step_i:step_f+1,:],axis=0)
        nu_ex_faces_mean        = np.nanmean(nu_ex_faces[step_i:step_f+1,:],axis=0)
        
        # Intepolation error in plasma density
        err_interp_n_mean = np.nanmean(err_interp_n[:,:,step_i:step_f+1],axis=2)
        
        # f_split variables
        f_split_adv_mean    = np.nanmean(f_split_adv[:,:,step_i:step_f+1],axis=2)
        f_split_qperp_mean  = np.nanmean(f_split_qperp[:,:,step_i:step_f+1],axis=2)
        f_split_qpara_mean  = np.nanmean(f_split_qpara[:,:,step_i:step_f+1],axis=2)
        f_split_qb_mean     = np.nanmean(f_split_qb[:,:,step_i:step_f+1],axis=2)
        f_split_Pperp_mean  = np.nanmean(f_split_Pperp[:,:,step_i:step_f+1],axis=2)
        f_split_Ppara_mean  = np.nanmean(f_split_Ppara[:,:,step_i:step_f+1],axis=2)
        f_split_ecterm_mean = np.nanmean(f_split_ecterm[:,:,step_i:step_f+1],axis=2)
        f_split_inel_mean   = np.nanmean(f_split_inel[:,:,step_i:step_f+1],axis=2)
    
    
    
    return[phi_mean,Er_mean,Ez_mean,Efield_mean,nn1_mean,nn2_mean,nn3_mean,
           ni1_mean,ni2_mean,ni3_mean,ni4_mean,ne_mean,fn1_x_mean,fn1_y_mean,
           fn1_z_mean,fn2_x_mean,fn2_y_mean,fn2_z_mean,fn3_x_mean,fn3_y_mean,
           fn3_z_mean,fi1_x_mean,fi1_y_mean,fi1_z_mean,fi2_x_mean,fi2_y_mean,
           fi2_z_mean,fi3_x_mean,fi3_y_mean,fi3_z_mean,fi4_x_mean,fi4_y_mean,
           fi4_z_mean,un1_x_mean,un1_y_mean,un1_z_mean,un2_x_mean,un2_y_mean,
           un2_z_mean,un3_x_mean,un3_y_mean,un3_z_mean,ui1_x_mean,ui1_y_mean,
           ui1_z_mean,ui2_x_mean,ui2_y_mean,ui2_z_mean,ui3_x_mean,ui3_y_mean,
           ui3_z_mean,ui4_x_mean,ui4_y_mean,ui4_z_mean,ji1_x_mean,ji1_y_mean,
           ji1_z_mean,ji2_x_mean,ji2_y_mean,ji2_z_mean,ji3_x_mean,ji3_y_mean,
           ji3_z_mean,ji4_x_mean,ji4_y_mean,ji4_z_mean,je_r_mean,je_t_mean,
           je_z_mean,je_perp_mean,je_para_mean,ue_r_mean,ue_t_mean,ue_z_mean,
           ue_perp_mean,ue_para_mean,uthetaExB_mean,Tn1_mean,Tn2_mean,Tn3_mean,
           Ti1_mean,Ti2_mean,Ti3_mean,Ti4_mean,Te_mean,n_mp_n1_mean,n_mp_n2_mean,
           n_mp_n3_mean,n_mp_i1_mean,n_mp_i2_mean,n_mp_i3_mean,n_mp_i4_mean,
           avg_w_n1_mean,avg_w_n2_mean,avg_w_i1_mean,avg_w_i2_mean,
           neu_gen_weights1_mean,neu_gen_weights2_mean,ion_gen_weights1_mean,
           ion_gen_weights2_mean,ndot_ion01_n1_mean,ndot_ion02_n1_mean,
           ndot_ion12_i1_mean,ndot_ion01_n2_mean,ndot_ion02_n2_mean,
           ndot_ion01_n3_mean,ndot_ion02_n3_mean,ndot_ion12_i3_mean,
           ndot_CEX01_i3_mean,ndot_CEX02_i4_mean,
           ne_cath_mean,Te_cath_mean,nu_cath_mean,ndot_cath_mean,F_theta_mean,
           Hall_par_mean,Hall_par_eff_mean,nu_e_tot_mean,nu_e_tot_eff_mean,
           nu_en_mean,nu_ei1_mean,nu_ei2_mean,nu_i01_mean,nu_i02_mean,nu_i12_mean,nu_ex_mean,
           Boltz_mean,Boltz_dim_mean,phi_elems_mean,phi_faces_mean,ne_elems_mean,
           ne_faces_mean,Te_elems_mean,Te_faces_mean,err_interp_n_mean,f_split_adv_mean,
           f_split_qperp_mean,f_split_qpara_mean,f_split_qb_mean,f_split_Pperp_mean,
           f_split_Ppara_mean,f_split_ecterm_mean,f_split_inel_mean,
           je_perp_elems_mean,je_theta_elems_mean,je_para_elems_mean,
           je_z_elems_mean,je_r_elems_mean,je_perp_faces_mean,je_theta_faces_mean,
           je_para_faces_mean,je_z_faces_mean,je_r_faces_mean,
           F_theta_elems_mean,Hall_par_elems_mean,Hall_par_eff_elems_mean,
           nu_e_tot_elems_mean,nu_e_tot_eff_elems_mean,F_theta_faces_mean,
           Hall_par_faces_mean,Hall_par_eff_faces_mean,nu_e_tot_faces_mean,
           nu_e_tot_eff_faces_mean,nu_en_elems_mean,nu_ei1_elems_mean,
           nu_ei2_elems_mean,nu_i01_elems_mean,nu_i02_elems_mean,
           nu_i12_elems_mean,nu_ex_elems_mean,nu_en_faces_mean,
           nu_ei1_faces_mean,nu_ei2_faces_mean,nu_i01_faces_mean,
           nu_i02_faces_mean,nu_i12_faces_mean,nu_ex_faces_mean]
           
           
if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 
    from HET_sims_read import HET_sims_read
    from HET_sims_plotvars import HET_sims_plotvars

    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))
    
    # Parameters
    e  = 1.6021766E-19
    elems_cath_Bline   = range(407-1,483-1+2,2) # Elements along the cathode B line for cases C1, C2 and C3
#    elems_cath_Bline   = range(875-1,951-1+2,2) # Elements along the cathode B line for case C5
    
    
    sim_name = "../../Rb_hyphen/sim/sims/SPT100_al0025_Ne5_C1"

    timestep         = 65
    allsteps_flag    = 1
    read_inst_data   = 0
    read_part_tracks = 0
    read_part_lists  = 0
    read_flag        = 0
    
    mean_type  = 0
    last_steps = 670
    step_i = 200
    step_f = 300
    
    
    
    path_simstate_inp  = sim_name+"/CORE/inp/SimState.hdf5"
    path_simstate_out  = sim_name+"/CORE/out/SimState.hdf5"
    path_postdata_out  = sim_name+"/CORE/out/PostData.hdf5"
    path_simparams_inp = sim_name+"/CORE/inp/sim_params.inp"
    path_picM          = sim_name+"/SET/inp/SPT100_picM.hdf5"
    
    [num_ion_spe,num_neu_spe,n_mp_cell_i,n_mp_cell_n,n_mp_cell_i_min,
       n_mp_cell_i_max,n_mp_cell_n_min,n_mp_cell_n_max,min_ion_plasma_density,
       m_A,spec_refl_prob,ene_bal,points,zs,rs,zscells,rscells,dims,
       nodes_flag,cells_flag,cells_vol,volume,ind_maxr_c,ind_maxz_c,nr_c,nz_c,
       eta_max,eta_min,xi_top,xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,
       nsteps,nsteps_fast,nsteps_eFld,faces,nodes,boundary_f,face_geom,elem_geom,
       n_faces,n_elems,n_faces_boundary,bIDfaces_Dwall,bIDfaces_Awall,
       bIDfaces_FLwall,IDfaces_Dwall,IDfaces_Awall,IDfaces_FLwall,zfaces_Dwall,
       rfaces_Dwall,Afaces_Dwall,zfaces_Awall,rfaces_Awall,Afaces_Awall,
       zfaces_FLwall,rfaces_FLwall,Afaces_FLwall,
       cath_elem,z_cath,r_cath,V_cath,mass,ssIons1,ssIons2,ssNeutrals1,ssNeutrals2,
       n_mp_i1_list,n_mp_i2_list,n_mp_n1_list,n_mp_n2_list,phi,phi_elems,Ez,Er,Efield,
       Bz,Br,Bfield,Te,Te_elems,cs01,cs02,nn1,nn2,ni1,ni2,ne,ne_elems,fn1_x,fn1_y,
       fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,
       un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,
       ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,
       ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,
       avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
       ion_gen_weights1,ion_gen_weights2,surf_elems,n_imp_elems,imp_elems,
       imp_elems_kbc,imp_elems_MkQ1,imp_elems_Te,imp_elems_dphi_kbc,
       imp_elems_dphi_sh,imp_elems_nQ1,imp_elems_nQ2,imp_elems_ion_flux_in1,
       imp_elems_ion_flux_out1,imp_elems_ion_ene_flux_in1,
       imp_elems_ion_ene_flux_out1,imp_elems_ion_imp_ene1,
       imp_elems_ion_flux_in2,imp_elems_ion_flux_out2,
       imp_elems_ion_ene_flux_in2,imp_elems_ion_ene_flux_out2,
       imp_elems_ion_imp_ene2,imp_elems_neu_flux_in1,imp_elems_neu_flux_out1,
       imp_elems_neu_ene_flux_in1,imp_elems_neu_ene_flux_out1,
       imp_elems_neu_imp_ene1,imp_elems_neu_flux_in2,imp_elems_neu_flux_out2,
       imp_elems_neu_ene_flux_in2,imp_elems_neu_ene_flux_out2,
       imp_elems_neu_imp_ene2,tot_mass_mp_neus,tot_mass_mp_ions,tot_num_mp_neus,
       tot_num_mp_ions,tot_mass_exit_neus,tot_mass_exit_ions,mass_mp_neus,
       mass_mp_ions,num_mp_neus,num_mp_ions,avg_dens_mp_neus,avg_dens_mp_ions,
       eta_u,eta_prod,eta_thr,eta_div,eta_cur,thrust,thrust_ion,thrust_neu,
       Id_inst,Id,Vd_inst,Vd,I_beam,I_tw_tot,Pd,Pd_inst,P_mat,P_inj,P_inf,P_ion,
       P_ex,P_use_tot_i,P_use_tot_n,P_use_tot,P_use_z_i,P_use_z_n,P_use_z,
       qe_wall,qe_wall_inst,Pe_faces_Dwall,Pe_faces_Awall,Pe_faces_FLwall,
       Pe_faces_Dwall_inst,Pe_faces_Awall_inst,Pe_faces_FLwall_inst,
       Pe_Dwall,Pe_Awall,Pe_FLwall,Pe_Dwall_inst,Pe_Awall_inst,Pe_FLwall_inst, 
       Pi_Dwall,Pi_Awall,Pi_FLwall,Pi_FLwall_nonz,Pn_Dwall,Pn_Awall,Pn_FLwall,
       Pn_FLwall_nonz,P_Dwall,P_Awall,P_FLwall,Pwalls,Pionex,Ploss,Pthrust,
       Pnothrust,Pnothrust_walls,balP,err_balP,ctr_Pd,ctr_Ploss,ctr_Pwalls,
       ctr_Pionex,ctr_P_DAwalls,ctr_P_FLwalls,ctr_P_FLwalls_in,ctr_P_FLwalls_i,
       ctr_P_FLwalls_n,ctr_P_FLwalls_e,balP_Pthrust,err_balP_Pthrust,
       ctr_balPthrust_Pd,ctr_balPthrust_Pnothrust,ctr_balPthrust_Pthrust,
       ctr_balPthrust_Pnothrust_walls,ctr_balPthrust_Pnothrust_ionex,
       err_def_balP,Isp_s,Isp_ms,dMdt_i1,dMdt_i2,dMdt_n1,dMdt_n2,dMdt_tot,
       mflow_coll_i1,mflow_coll_i2,mflow_coll_n1,mflow_coll_n2,mflow_fw_i1,
       mflow_fw_i2,mflow_fw_n1,mflow_fw_n2,mflow_tw_i1,mflow_tw_i2,mflow_tw_n1,
       mflow_tw_n2,mflow_ircmb_picS_n1,mflow_ircmb_picS_n2,mflow_inj_i1,mflow_inj_i2,
       mflow_fwmat_i1,mflow_fwmat_i2,mflow_inj_n1,mflow_fwmat_n1,mflow_inj_n2,
       mflow_fwmat_n2,mflow_twmat_i1,mflow_twinf_i1,mflow_twa_i1,mflow_twmat_i2,
       mflow_twinf_i2,mflow_twa_i2,mflow_twmat_n1,mflow_twinf_n1,mflow_twa_n1,
       mflow_twmat_n2,mflow_twinf_n2,mflow_twa_n2,mbal_n1,mbal_i1,mbal_i2,mbal_tot,
       err_mbal_n1,err_mbal_i1,err_mbal_i2,err_mbal_tot,ctr_mflow_coll_n1,
       ctr_mflow_fw_n1,ctr_mflow_tw_n1,ctr_mflow_coll_i1,ctr_mflow_fw_i1,
       ctr_mflow_tw_i1,ctr_mflow_coll_i2,ctr_mflow_fw_i2,ctr_mflow_tw_i2,
       ctr_mflow_coll_tot,ctr_mflow_fw_tot,ctr_mflow_tw_tot,dEdt_i1,dEdt_i2,
       dEdt_n1,dEdt_n2,eneflow_coll_i1,eneflow_coll_i2,eneflow_coll_n1,
       eneflow_coll_n2,eneflow_fw_i1,eneflow_fw_i2,eneflow_fw_n1,eneflow_fw_n2,
       eneflow_tw_i1,eneflow_tw_i2,eneflow_tw_n1,eneflow_tw_n2,Pfield_i1,
       Pfield_i2,eneflow_inj_i1,eneflow_fwmat_i1,eneflow_inj_i2,
       eneflow_fwmat_i2,eneflow_inj_n1,eneflow_fwmat_n1,eneflow_inj_n2,
       eneflow_fwmat_n2,eneflow_twmat_i1,eneflow_twinf_i1,eneflow_twa_i1,
       eneflow_twmat_i2,eneflow_twinf_i2,eneflow_twa_i2,eneflow_twmat_n1,
       eneflow_twinf_n1,eneflow_twa_n1,eneflow_twmat_n2,eneflow_twinf_n2,
       eneflow_twa_n2,ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,Te_cath,
       nu_cath,ndot_cath,Q_cath,P_cath,F_theta,Hall_par,Hall_par_eff,
       nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,Boltz,Boltz_dim] = HET_sims_read(path_simstate_inp,path_simstate_out,
                                                                                                  path_postdata_out,path_simparams_inp,
                                                                                                  path_picM,allsteps_flag,timestep,read_inst_data,
                                                                                                  read_part_lists,read_flag)
                                                                                                                    
    [Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,fn1_x,fn1_y,fn1_z,
       fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,
       un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,ji1_z,
       ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,ue_perp,
       ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,
       avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
       ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,
       F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12] = HET_sims_plotvars(nodes_flag,cells_flag,Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,
                                                                                                                         fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,
                                                                                                                         un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,
                                                                                                                         ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,
                                                                                                                         ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
                                                                                                                         n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                                                                                                         neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                                                                                                         ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,F_theta,Hall_par,Hall_par_eff,
                                                                                                                         nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12)

    [phi_mean,Er_mean,Ez_mean,Efield_mean,nn1_mean,nn2_mean,
       ni1_mean,ni2_mean,ne_mean,fn1_x_mean,fn1_y_mean,fn1_z_mean,
       fn2_x_mean,fn2_y_mean,fn2_z_mean,fi1_x_mean,fi1_y_mean,fi1_z_mean,
       fi2_x_mean,fi2_y_mean,fi2_z_mean,un1_x_mean,un1_y_mean,un1_z_mean,
       un2_x_mean,un2_y_mean,un2_z_mean,ui1_x_mean,ui1_y_mean,ui1_z_mean,
       ui2_x_mean,ui2_y_mean,ui2_z_mean,ji1_x_mean,ji1_y_mean,ji1_z_mean,
       ji2_x_mean,ji2_y_mean,ji2_z_mean,je_r_mean,je_t_mean,je_z_mean,
       je_perp_mean,je_para_mean,ue_r_mean,ue_t_mean,ue_z_mean,ue_perp_mean,
       ue_para_mean,uthetaExB_mean,Tn1_mean,Tn2_mean,Ti1_mean,Ti2_mean,Te_mean,
       n_mp_n1_mean,n_mp_n2_mean,n_mp_i1_mean,n_mp_i2_mean,avg_w_n1_mean,
       avg_w_n2_mean,avg_w_i1_mean,avg_w_i2_mean,neu_gen_weights1_mean,
       neu_gen_weights2_mean,ion_gen_weights1_mean,ion_gen_weights2_mean,
       ndot_ion01_n1_mean,ndot_ion02_n1_mean,ndot_ion12_i1_mean,
       ne_cath_mean,nu_cath_mean,ndot_cath_mean,F_theta_mean,Hall_par_mean,
       Hall_par_eff_mean,nu_e_tot_mean,nu_e_tot_eff_mean,nu_en_mean,
       nu_ei1_mean,nu_ei2_mean,nu_i01_mean,nu_i02_mean,nu_i12_mean,
       Boltz_mean,Boltz_dim_mean,phi_elems_mean,ne_elems_mean,Te_elems_mean] = HET_sims_mean(nsteps,mean_type,last_steps,step_i,step_f,phi,Er,Ez,Efield,Br,Bz,Bfield,
                                                                                             nn1,nn2,ni1,ni2,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,
                                                                                             fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,
                                                                                             ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,
                                                                                             je_perp,je_para,ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
                                                                                             n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                                                                             neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                                                                             ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,nu_cath,ndot_cath,F_theta,
                                                                                             Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,
                                                                                             nu_i02,nu_i12,Boltz,Boltz_dim,phi_elems,ne_elems,Te_elems)
                                                                                
    
#    vlight = 3e8
#    print(np.nanmax(ue_t[:,:,step_i]),np.nanmin(ue_t[:,:,step_i]))
#    print(np.nanmax(ue_t[:,:,step_f]),np.nanmin(ue_t[:,:,step_f]))
#    print(np.nanmax(ue_t_mean[:,:]),np.nanmin(ue_t_mean[:,:]))
    
    phi_mean_ABS = abs(phi_mean)
    
    print(np.where(phi_mean_ABS == np.nanmin(phi_mean_ABS)))
    print(phi[17,20,step_i])
    print(phi[17,20,step_f])
    print(phi_mean[17,20])    
    print(0.5*(phi[17,20,step_i]+phi[17,20,step_f]))
    
    phi_cath_mean = 0.0
    Te_cath_mean = Te_elems_mean[cath_elem]
    ne_cath_mean = ne_elems_mean[cath_elem]
    Boltz_mean_bis = e*(phi_elems_mean-phi_cath_mean)/(e*Te_cath_mean) - np.log(ne_elems_mean/ne_cath_mean)
    
    rel_err = np.abs(Boltz_mean_bis[elems_cath_Bline] - Boltz_mean[elems_cath_Bline])/np.abs(Boltz_mean[elems_cath_Bline])
    
    
    plt.figure("Boltzmann at cathode B line")
    plt.plot(elem_geom[3,elems_cath_Bline],Boltz_mean[elems_cath_Bline],'r')
    plt.plot(elem_geom[3,elems_cath_Bline],Boltz_mean_bis[elems_cath_Bline],'g')
    plt.plot(elem_geom[3,cath_elem],Boltz_mean[cath_elem],'ks')
    plt.plot(elem_geom[3,cath_elem],Boltz_mean_bis[cath_elem],'ks')
    plt.plot(elem_geom[3,elems_cath_Bline],np.zeros(np.shape(elem_geom[3,elems_cath_Bline])),'--')
    plt.xlabel(r"$\sigma$ (m T)")
    plt.ylabel(r"Boltzmann relation (-)")
    
    plt.figure("phi at cathode B line")
    plt.plot(elem_geom[3,elems_cath_Bline],phi_elems_mean[elems_cath_Bline])
    plt.plot(elem_geom[3,cath_elem],phi_elems_mean[cath_elem],'ks')
    plt.plot(elem_geom[3,elems_cath_Bline],np.zeros(np.shape(elem_geom[3,elems_cath_Bline])),'--')
    plt.xlabel(r"$\sigma$ (m T)")
    plt.ylabel(r"$\phi$ (V)")
    
    plt.figure("Te at cathode B line")
    plt.plot(elem_geom[3,elems_cath_Bline],Te_elems_mean[elems_cath_Bline])
    plt.plot(elem_geom[3,cath_elem],Te_elems_mean[cath_elem],'ks')
    plt.plot(elem_geom[3,elems_cath_Bline],np.zeros(np.shape(elem_geom[3,elems_cath_Bline])),'--')
    plt.xlabel(r"$\sigma$ (m T)")
    plt.ylabel(r"$T_e$ (eV)")
    
    plt.figure("ne at cathode B line")
    plt.semilogy(elem_geom[3,elems_cath_Bline],ne_elems_mean[elems_cath_Bline])
    plt.semilogy(elem_geom[3,cath_elem],ne_elems_mean[cath_elem],'ks')
    plt.semilogy(elem_geom[3,elems_cath_Bline],np.zeros(np.shape(elem_geom[3,elems_cath_Bline])),'--')
    plt.xlabel(r"$\sigma$ (m T)")
    plt.ylabel(r"$n_e$ (m$^{-3}$)")