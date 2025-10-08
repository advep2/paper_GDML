# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:12:23 2019

@author: adrian
"""

"""
############################################################################
Description:    This python function finds succesive minimum and maximum 
                values of a given time signal and return those min and max
                values and its position, the average min and max values, and 
                the mean of the signal between the first and last min values
                (and thus it considers an integer number of periods for the
                average)
############################################################################
 Inputs:        1) time_comp: complete time vector of the signal (may be equal
                              to time)
                2) time:   time vector of the signal
                3) signal: signal evaluated at the time vector
                4) order:  number of neighbor points when looking for local 
                           maxima or minima. A value of 10-50 is recommended
                           For noisy signals or with a higher time resolution
                           a higher value is recommended (50 or higher)
############################################################################
Output:         1) mins: vector with the successive minimum values
                2) time_mins: vector with times at which we have the successive
                              minimum values
                3) mins_ind: vector with indeces of the successive minimum values 
                4) mean_min: mean minimum value of the signal between the first 
                             and last minimum found
                5) maxs: vector with the successive maximum values
                6) time_maxs: vector with times at which we have the successive
                              maximum values
                7) maxs_ind: vector with indeces of the successive maximum values 
                8) mean_min: mean of the minimum values found
                9) mean_max: mean of the maximum values found
               10) mean_val: mean value of the given signal between the first and the
                         last minimum values found
"""

def max_min_mean_vals(time_comp,time,signal,order):
    
    import numpy as np
    from scipy.signal import argrelextrema
    
    maxs_ind  = argrelextrema(signal, np.greater, order = order)[0]
    mins_ind  = argrelextrema(signal, np.less, order = order)[0]
    
    if len(maxs_ind) > 0 and len(mins_ind) > 0:
        mins      = signal[mins_ind]
        maxs      = signal[maxs_ind]
        time_mins = time[mins_ind]
        time_maxs = time[maxs_ind]
        mean_min  = np.mean(mins)
        mean_max  = np.mean(maxs)
        
        
        if len(maxs_ind) < len(mins_ind):
            mean_val  = np.mean(signal[mins_ind[0]:mins_ind[-1]+1])
        else:
            mean_val  = np.mean(signal[maxs_ind[0]:maxs_ind[-1]+1])
    
        max2mean  = mean_max/mean_val
        min2mean  = mean_min/mean_val
        amplitude = (mean_max-mean_min)/mean_val
        
        mins_ind_comp = np.zeros(np.shape(mins_ind),dtype=int)
        maxs_ind_comp = np.zeros(np.shape(maxs_ind),dtype=int)
        for i in range(0,len(mins_ind_comp)):
            mins_ind_comp[i] = np.where(time_comp == time_mins[i])[0][0]
        for i in range(0,len(maxs_ind_comp)):
            maxs_ind_comp[i] = np.where(time_comp == time_maxs[i])[0][0]
    
    else:
        mins          = 0 
        time_mins     = 0
        mins_ind      = 0
        maxs          = 0
        time_maxs     = 0
        maxs_ind      = 0
        mean_min      = 0
        mean_max      = 0 
        mean_val      = np.mean(signal)
        max2mean      = 0 
        min2mean      = 0    
        amplitude     = 0
        mins_ind_comp = 0
        maxs_ind_comp = 0
        
    
    return [mins,time_mins,mins_ind,maxs,
            time_maxs,maxs_ind,mean_min,mean_max,
            mean_val,max2mean,min2mean,amplitude,
            mins_ind_comp,maxs_ind_comp]
    
    
"""
############################################################################
Description:    This python function computes the average phase shift between
                two oscillating signals A and B featuring the same dominant
                frequency. The time and phase shift are computed considering
                the mean of the dominant frequency of each signal.
                Negative phase or time shift means that the signal A is ahead
                of the signal B (i.e. the peaks of A occur first).
############################################################################
 Inputs:        1) time_comp: complete time vector of the signals (may be equal
                              to timeA and or timeB)
                2) sigA_comp: complete signal A evaluated at time_comp
                3) sigB_comp: complete signal B evaluated at time_comp
                4) timeA: time vector of the signal A
                5) sigA: signal A evaluated at the time vector timeA
                6) timeB: time vector of the signal B
                7) sigB: signal B evaluated at the time vector timeB
                8) order: number of neighbor points when looking for local 
                          maxima or minima. A value of 10-50 is recommended
                          For noisy signals or with a higher time resolution
                          a higher value is recommended (50 or higher)
############################################################################
Output:         1) time_shift_vector: time shift vector containing the
                                      time shift between pairs of successive
                                      peaks of the signals A and B in degrees
                2) phase_shift_vector_deg: phase shift vector containing the
                                           phase shift between pairs of successive
                                           peaks of the signals A and B in degrees
                3) mean_time_shift: mean time shift between signals A and B 
                4) mean_phase_shift_deg: mean phase shift between signals A and
                                         B in degrees
"""

def comp_phase_shift(time_comp,sigA_comp,sigB_comp,timeA,sigA,timeB,sigB,order):
    
    import numpy as np
    from FFT import FFT
    
    # Obtain the maxima and minima of signals A and B and the proper range of
    # each signals containg an integer number of cycles for each of them
    [mins_sigA,time_mins_sigA,mins_ind_sigA,maxs_sigA,
     time_maxs_sigA,maxs_ind_sigA,mean_min_sigA,mean_max_sigA,
     mean_val_sigA,max2mean_sigA,min2mean_sigA,amplitude_sigA,
     mins_ind_comp_sigA,maxs_ind_comp_sigA] = max_min_mean_vals(time_comp,timeA,sigA,order)
     
    [mins_sigB,time_mins_sigB,mins_ind_sigB,maxs_sigB,
     time_maxs_sigB,maxs_ind_sigB,mean_min_sigB,mean_max_sigB,
     mean_val_sigB,max2mean_sigB,min2mean_sigB,amplitude_sigB,
     mins_ind_comp_sigB,maxs_ind_comp_sigB] = max_min_mean_vals(time_comp,timeB,sigB,order)
     
    # Obtain the FFT of each signal considering the proper time vector for each
    # of them with an integer number of cycles for each case
    time_vector_sigA = time_comp[mins_ind_comp_sigA[0]:mins_ind_comp_sigA[-1]+1]
    sigA_vector      = sigA_comp[mins_ind_comp_sigA[0]:mins_ind_comp_sigA[-1]+1]
    [fft_sigA,freq_sigA,max_fft_sigA,max_freq_sigA] = FFT(time_vector_sigA[1]-time_vector_sigA[0],time_vector_sigA,sigA_vector)
    
    time_vector_sigB = time_comp[mins_ind_comp_sigB[0]:mins_ind_comp_sigB[-1]+1]
    sigB_vector      = sigB_comp[mins_ind_comp_sigB[0]:mins_ind_comp_sigB[-1]+1]
    [fft_sigB,freq_sigB,max_fft_sigB,max_freq_sigB] = FFT(time_vector_sigB[1]-time_vector_sigB[0],time_vector_sigB,sigB_vector)
    
    # Consider the mean frequency of the two signals as the frequency of both 
    # of them in order to compute their phase shift
    freq_sig = 0.5*(max_freq_sigA + max_freq_sigB)
    
    # Check if both signals have the same dominant frequency. Send a warning in case
    # the relative error in their dominant frequency is larger than a given tolerance
    tol = 0.01
    rel_err_freqAB = np.abs(max_freq_sigA - max_freq_sigB)/max_freq_sigA
    if rel_err_freqAB > tol:
        print("comp_phase_shift: sigA and sigB do not have the same frequency. Error on the phase shift may be present.")
        print("                  Phase shift computed considering the mean frequecy of both signals")
        print("                  Freq. sigA = "+str(max_freq_sigA))
        print("                  Freq. sigB = "+str(max_freq_sigB))
        print("                  Rel. Err   = "+str(rel_err_freqAB)+" > "+str(tol))
        print("                  Mean. freq = "+str(freq_sig))

    # Obtain the time shift vector
    dimA = len(time_maxs_sigA)
    dimB = len(time_maxs_sigB)
    if dimA > dimB: 
        time_shift_vector = time_maxs_sigA[dimA-dimB::] - time_maxs_sigB
    elif dimA < dimB: 
        time_shift_vector = time_maxs_sigA - time_maxs_sigB[dimB-dimA::]
    else:
        time_shift_vector = time_maxs_sigA - time_maxs_sigB
    # Force the phase shift vector to be in [-pi:pi]
    period = 1.0/freq_sig
    phase_shift_vector = 2*np.pi*(((0.5 + time_shift_vector/period) % 1.0) - 0.5)
    # Convert to degrees
    phase_shift_vector_deg = phase_shift_vector*180.0/np.pi
    # Obtain mean values
    mean_time_shift      = np.mean(time_shift_vector)
    mean_phase_shift_deg = np.mean(phase_shift_vector_deg)
    
    
    return [time_shift_vector,phase_shift_vector_deg,mean_time_shift,mean_phase_shift_deg]
    
    
"""
############################################################################
Description:    This python function computes the FFT of a given oscillating 
                signal considering a time period including an integer number 
                of cycles.
############################################################################
Inputs:         1) time_comp: complete time vector of the signal (may be equal
                              to time)
                2) sig_comp:  signal values at time_comp
                3) time: time vector of the signal
                4) sig: signal evaluated at the time vector time
                5) order: number of neighbor points when looking for local 
                          maxima or minima. A value of 10-50 is recommended
                          For noisy signals or with a higher time resolution
                          a higher value is recommended (50 or higher)
############################################################################
Output:         1) time_shift_vector: time shift vector containing the
                                      time shift between pairs of successive
                                      peaks of the signals A and B in degrees
                2) phase_shift_vector_deg: phase shift vector containing the
                                           phase shift between pairs of successive
                                           peaks of the signals A and B in degrees
                3) mean_time_shift: mean time shift between signals A and B 
                4) mean_phase_shift_deg: mean phase shift between signals A and
                                         B in degrees
"""

def comp_FFT(time_comp,sig_comp,time,sig,order):
    
    from FFT import FFT
    
    # Obtain the maxima and minima of the signal and the proper range of
    # the signal containg an integer number of cycles
    [mins_sig,time_mins_sig,mins_ind_sig,maxs_sig,
     time_maxs_sig,maxs_ind_sig,mean_min_sig,mean_max_sig,
     mean_val_sig,max2mean_sig,min2mean_sig,amplitude_sig,
     mins_ind_comp_sig,maxs_ind_comp_sig] = max_min_mean_vals(time_comp,time,sig,order)
     
     
    # Obtain the FFT of the signal considering the proper time vector 
    # with an integer number of cycles
    time_vector_sig = time_comp[mins_ind_comp_sig[0]:mins_ind_comp_sig[-1]+1]
    sig_vector      = sig_comp[mins_ind_comp_sig[0]:mins_ind_comp_sig[-1]+1]
    [fft_sig,freq_sig,max_fft_sig,max_freq_sig] = FFT(time_vector_sig[1]-time_vector_sig[0],time_vector_sig,sig_vector)
    
    return [fft_sig,freq_sig,max_fft_sig,max_freq_sig]
    
    
"""
############################################################################
Description:    This python function computes the isothermal Boltzmann 
                relation for the electrons along the cathode magnetic field
                streamline. Besides gives the evolution along that line of 
                different plasma properties
############################################################################
Inputs:         1) time_comp: complete time vector of the signal (may be equal
                              to time)
                2) sig_comp:  signal values at time_comp
                3) time: time vector of the signal
                4) sig: signal evaluated at the time vector time
                5) order: number of neighbor points when looking for local 
                          maxima or minima. A value of 10-50 is recommended
                          For noisy signals or with a higher time resolution
                          a higher value is recommended (50 or higher)
############################################################################
Output:         1) time_shift_vector: time shift vector containing the
                                      time shift between pairs of successive
                                      peaks of the signals A and B in degrees
                2) phase_shift_vector_deg: phase shift vector containing the
                                           phase shift between pairs of successive
                                           peaks of the signals A and B in degrees
                3) mean_time_shift: mean time shift between signals A and B 
                4) mean_phase_shift_deg: mean phase shift between signals A and
                                         B in degrees
"""

def comp_Boltz(elems_cath_Bline,cath_elem,V_cath,V_cath_tot,zs,rs,elem_geom,face_geom,phi,Te,ne,cath_type):
    
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    
    # Parameters
    e  = 1.6021766E-19
    
    
    # Obtain the interpolation functions for phi, Te and ne at the PIC mesh
    rvec = rs[:,0]
    zvec = zs[0,:]
    interp_phi = RegularGridInterpolator((rvec,zvec), phi, method='nearest')
    interp_Te = RegularGridInterpolator((rvec,zvec), Te, method='nearest')
    interp_ne = RegularGridInterpolator((rvec,zvec), ne, method='nearest')
    
    # Obtain the vectors containing the coordinates (z,r) of the MFAM elements
    # representing the cathode magnetic field streamline
    r_cath_Bline = elem_geom[1,elems_cath_Bline]
    z_cath_Bline = elem_geom[0,elems_cath_Bline]
    if cath_type == 1:
        r_cath     = face_geom[1,cath_elem]
        z_cath     = face_geom[0,cath_elem]
    elif cath_type == 2:
        r_cath    = elem_geom[1,cath_elem]
        z_cath    = elem_geom[0,cath_elem]
        
    # Interpolate phi, Te and ne to the cathode magnetic field streamline
    cath_Bline_phi = interp_phi((r_cath_Bline,z_cath_Bline))
    cath_Bline_Te  = interp_Te((r_cath_Bline,z_cath_Bline))
    cath_Bline_ne  = interp_ne((r_cath_Bline,z_cath_Bline))
    cath_phi_vec   = interp_phi((r_cath,z_cath))
    cath_Te_vec    = interp_Te((r_cath,z_cath))
    cath_ne_vec    = interp_ne((r_cath,z_cath))

    if len(cath_elem) == 1:
        cath_phi = np.copy(cath_phi_vec)
        cath_Te  = np.copy(cath_Te_vec)
        cath_ne  = np.copy(cath_ne_vec)
    elif len(cath_elem) > 1:
        cath_phi = 0.0
        cath_Te  = 0.0
        cath_ne  = 0.0
        for ind_cath in range(0,len(cath_elem)):
            if cath_type == 2:
                cath_phi = cath_phi + cath_phi_vec[ind_cath]*V_cath[ind_cath]/V_cath_tot
                cath_Te = cath_Te + cath_Te_vec[ind_cath]*V_cath[ind_cath]/V_cath_tot
                cath_ne = cath_ne + cath_ne_vec[ind_cath]*V_cath[ind_cath]/V_cath_tot
            elif cath_type == 1:
                cath_phi = cath_phi + cath_phi_vec[ind_cath]*face_geom[-2,ind_cath]/np.sum(face_geom[-2,cath_elem],axis=0)
                cath_Te = cath_Te + cath_Te_vec[ind_cath]*face_geom[-2,ind_cath]/np.sum(face_geom[-2,cath_elem],axis=0)
                cath_ne = cath_ne + cath_ne_vec[ind_cath]*face_geom[-2,ind_cath]/np.sum(face_geom[-2,cath_elem],axis=0)
            
    # Obtain the Maxwell-Boltzmann equilibrium law (or isothermal Boltzmann relation)
    cath_Bline_nodim_Boltz = e*(cath_Bline_phi-cath_phi)/(e*cath_Te) - np.log(cath_Bline_ne/cath_ne)
    cath_Bline_dim_Boltz   = cath_Te*cath_Bline_nodim_Boltz
    
    cath_nodim_Boltz = e*(cath_phi-cath_phi)/(e*cath_Te) - np.log(cath_ne/cath_ne)
    cath_dim_Boltz   = cath_Te*cath_nodim_Boltz

    return [cath_Bline_phi, cath_phi, cath_Bline_Te, cath_Te, cath_Bline_ne,
            cath_ne, cath_Bline_nodim_Boltz, cath_nodim_Boltz,
            cath_Bline_dim_Boltz, cath_dim_Boltz]
    

"""
############################################################################
Description:    This python function computes the nodal weighting volume
                weighted average value in the simulation domain of a given
                magnitude (e.g. the elctron temperature)
############################################################################
Inputs:         1) var:  variable whose domain average is to be computed at
                         each given time
                2) time: time vector 
                3) vol: nodal weighting volumes
############################################################################
Output:         1) var_mean: vector containing the domain average value of the
                             given variable at each given time
                2) vol_tot: total domain volume, as the sum of the nodal 
                            weighting volumes
"""

def domain_average(var,time,vol):
    
    import numpy as np
    
    nsteps   = len(time)
    [nr,nz]  = np.shape(vol)
    var_mean = np.zeros(np.shape(time),dtype=float)
    vol_tot  = 0.0 
    
    for i in range(0,nr):
        for j in range(0,nz):
            vol_tot = vol_tot + vol[i,j] 
            
    
    for k in range(0,nsteps):
        for i in range(0,nr):
            for j in range(0,nz):                
                var_mean[k]   = var_mean[k] + var[i,j,k]*vol[i,j]

    var_mean   = var_mean/vol_tot

    return [var_mean,vol_tot]



if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 
    from scipy.signal import correlate
    from HET_sims_read import HET_sims_read
    from FFT import FFT
    from HET_sims_mean import HET_sims_mean
    from HET_sims_plotvars import HET_sims_plotvars
    
    
    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))
    
    
#    sim_name = "../../Rb_hyphen/sim/sims/SPT100_al0025_Ne5_C1_S"
    sim_name = "../../Rb_hyphen/sim/sims/SPT100_al0025_Ne5_C1"
#    sim_name = "../../Rb_hyphen/sim/sims/SPT100_also255_Ne5_C1"
#    sim_name = "../../Rb_hyphen/sim/sims/SPT100_also2575_Ne5_C1"
    
    sim_name = "../../../sim/sims/SPT100_al0025_Ne5_C1_qboundary"
    
    sim_name =  "../../../Sr_sims_files/SPT100_DMD_pm2em2_cat3328_tmtetq2_rel"
    
    elems_cath_Bline   = range(407-1,483-1+2,2) # Elements along the cathode B line for cases C1, C2 and C3
#    elems_cath_Bline   = range(875-1,951-1+2,2) # Elements along the cathode B line for case C5
    
    mean_type = 2 
    step_i = 0
    step_f = 0
    last_steps = 2500
#    last_steps = 600
    last_steps_fast = last_steps*20
    order = 50
    order_fast = 500

    timestep         = -1
    allsteps_flag    = 1
    read_inst_data   = 1
    read_part_tracks = 0
    read_part_lists  = 0
    read_flag        = 1
    oldpost_sim      = 0
    oldsimparams_sim = 1
    
    oldpost_sim      = 3
    oldsimparams_sim = 10
    
    
#    path_picM         = sim_name+"/SET/inp/SPT100_picM.hdf5"
    path_picM         = sim_name+"/SET/inp/SPT100_picM_Reference1500points_rm2.hdf5"
    
    path_simstate_inp = sim_name+"/CORE/inp/SimState.hdf5"
    path_simstate_out = sim_name+"/CORE/out/SimState.hdf5"
    path_postdata_out = sim_name+"/CORE/out/PostData.hdf5"
    path_simparams_inp = sim_name+"/CORE/inp/sim_params.inp"
    
    [num_ion_spe,num_neu_spe,n_mp_cell_i,n_mp_cell_n,n_mp_cell_i_min,
           n_mp_cell_i_max,n_mp_cell_n_min,n_mp_cell_n_max,min_ion_plasma_density,
           m_A,spec_refl_prob,ene_bal,points,zs,rs,zscells,rscells,dims,
           nodes_flag,cells_flag,cells_vol,volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,
           eta_max,eta_min,xi_top,xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,
           nsteps,nsteps_fast,nsteps_eFld,faces,nodes,elem_n,boundary_f,face_geom,elem_geom,
           n_faces,n_elems,n_faces_boundary,bIDfaces_Dwall,bIDfaces_Awall,
           bIDfaces_FLwall,IDfaces_Dwall,IDfaces_Awall,IDfaces_FLwall,zfaces_Dwall,
           rfaces_Dwall,Afaces_Dwall,zfaces_Awall,rfaces_Awall,Afaces_Awall,
           zfaces_FLwall,rfaces_FLwall,Afaces_FLwall,
           cath_elem,z_cath,r_cath,V_cath,mass,ssIons1,ssIons2,ssNeutrals1,ssNeutrals2,
           n_mp_i1_list,n_mp_i2_list,n_mp_n1_list,n_mp_n2_list,phi,phi_elems,Ez,Er,Efield,
           Bz,Br,Bfield,Te,Te_elems,je_mag_elems,je_perp_elems,je_theta_elems,je_para_elems,
           cs01,cs02,nn1,nn2,ni1,ni2,ne,ne_elems,fn1_x,fn1_y,
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
           nu_cath,ndot_cath,Q_cath,P_cath,V_cath_tot,ne_cath_avg,F_theta,Hall_par,
           Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,
           nu_ex,Boltz,Boltz_dim,Pfield_e,Ebal_e,ge_b,ge_b_acc,ge_sb_b,ge_sb_b_acc,
           delta_see,delta_see_acc,err_interp_n] = HET_sims_read(path_simstate_inp,path_simstate_out,
                                                                 path_postdata_out,path_simparams_inp,
                                                                 path_picM,allsteps_flag,timestep,read_inst_data,
                                                                 read_part_lists,read_flag,oldpost_sim,oldsimparams_sim)
    
    # For fast print-out signals #############################################
#    var = avg_dens_mp_neus                                                                          
#    [mins,time_mins,mins_ind,maxs,
#     time_maxs,maxs_ind,mean_min,mean_max,
#     mean_val,max2mean,min2mean,amplitude,
#     mins_ind_comp,maxs_ind_comp] = max_min_mean_vals(time_fast,time_fast[nsteps_fast-last_steps_fast::],var[nsteps_fast-last_steps_fast::],order)
#    
#    
#    time_vector = time_fast[mins_ind_comp[0]:mins_ind_comp[-1]+1]
#    var_vector      = var[mins_ind_comp[0]:mins_ind_comp[-1]+1]
#    [fft_var,freq_var,max_fft_var,max_freq_var]         = FFT(time_vector[1]-time_vector[0],time_vector,var_vector)                                               
#    
#    plt.figure("signal")
#    plt.plot(time_fast,var,'r')
#    plt.plot(time_mins,mins,'r^')
#    plt.plot(time_maxs,maxs,'rv')
#    plt.plot(time_vector,var_vector,'k')
#    plt.plot(time_fast,mean_val*np.ones(np.shape(time_fast)),'r--')
#    plt.plot(time_fast[nsteps_fast-last_steps_fast::],0.9*mean_val*np.ones(np.shape(time_fast[nsteps_fast-last_steps_fast::])),'k--')
#
#    plt.figure("FFT signal")
#    plt.semilogx(freq_var[1::],np.abs(fft_var[1::]),'r')
#    
#    
#    print("var mean_val (A)        = "+str(mean_val))
#    print("var mean_max (A)        = "+str(mean_max))
#    print("var mean_min (A)        = "+str(mean_min))   
#    
#    print("Freq var     (Hz)       = "+str(max_freq_var))
#    print("Freqs var     (Hz)      = "+str(1/np.diff(time_mins)))
    
    # For print-out signals #############################################
    var = Id                                                                         
    [mins,time_mins,mins_ind,maxs,
     time_maxs,maxs_ind,mean_min,mean_max,
     mean_val,max2mean,min2mean,amplitude,
     mins_ind_comp,maxs_ind_comp] = max_min_mean_vals(time,time[nsteps-last_steps::],var[nsteps-last_steps::],order)
    
    
    time_vector = time[mins_ind_comp[0]:mins_ind_comp[-1]+1]
    var_vector      = var[mins_ind_comp[0]:mins_ind_comp[-1]+1]
    [fft_var,freq_var,max_fft_var,max_freq_var]         = FFT(time_vector[1]-time_vector[0],time_vector,var_vector)                                               
    
    plt.figure("signal")
    plt.semilogy(time,var,'r')
    plt.semilogy(time_mins,mins,'r^')
    plt.semilogy(time_maxs,maxs,'rv')
    plt.semilogy(time_vector,var_vector,'k')
    plt.semilogy(time,mean_val*np.ones(np.shape(time)),'r--')
    plt.semilogy(time[nsteps-last_steps::],0.9*mean_val*np.ones(np.shape(time[nsteps-last_steps::])),'k--')

    plt.figure("FFT signal")
    plt.semilogx(freq_var[1::],np.abs(fft_var[1::]),'r')
    
    
    print("var mean_val (A)        = "+str(mean_val))
    print("var mean_max (A)        = "+str(mean_max))
    print("var mean_min (A)        = "+str(mean_min))   
    
    print("Freq var     (Hz)       = "+str(max_freq_var))
    print("Freqs var     (Hz)      = "+str(1/np.diff(time_mins)))
    
    
    # Analysis for Id and I_beam ##############################################
#    var = Id                                                                          
#    [mins_Id,time_mins_Id,mins_ind_Id,maxs_Id,
#     time_maxs_Id,maxs_ind_Id,mean_min_Id,mean_max_Id,
#     mean_val_Id,max2mean_Id,min2mean_Id,amplitude_Id,
#     mins_ind_comp_Id,maxs_ind_comp_Id] = max_min_mean_vals(time,time[nsteps-last_steps::],var[nsteps-last_steps::],order)
#     
#    var = I_beam                                                                          
#    [mins_I_beam,time_mins_I_beam,mins_ind_I_beam,maxs_I_beam,
#     time_maxs_I_beam,maxs_ind_I_beam,mean_min_I_beam,mean_max_I_beam,
#     mean_val_I_beam,max2mean_I_beam,min2mean_I_beam,amplitude_I_beam,
#     mins_ind_comp_I_beam,maxs_ind_comp_I_beam] = max_min_mean_vals(time,time[nsteps-last_steps::],var[nsteps-last_steps::],order)
#                                                                                      
#    
#    plt.figure("signal")
#    plt.plot(time,Id,'r')
#    plt.plot(time_mins_Id,mins_Id,'r^')
#    plt.plot(time_maxs_Id,maxs_Id,'rv')
#    plt.plot(time,mean_val_Id*np.ones(np.shape(time)),'r--')
#    plt.plot(time,I_beam,'g')
#    plt.plot(time_mins_I_beam,mins_I_beam,'g^')
#    plt.plot(time_maxs_I_beam,maxs_I_beam,'gv')
#    plt.plot(time,mean_val_I_beam*np.ones(np.shape(time)),'g--')
#    plt.plot(time[nsteps-last_steps::],0.9*mean_val_Id*np.ones(np.shape(time[nsteps-last_steps::])),'k--')
    

##    # Obtain the FFT considering a proper period (with an integer number of cycles on each case)
##    time_vector_Id = time[mins_ind_comp_Id[0]:mins_ind_comp_Id[-1]+1]
##    Id_vector      = Id[mins_ind_comp_Id[0]:mins_ind_comp_Id[-1]+1]
##    [fft_Id,freq_Id,max_fft_Id,max_freq_Id]                                 = FFT(time_vector_Id[1]-time_vector_Id[0],time_vector_Id,Id_vector)
##    
##    time_vector_I_beam = time[mins_ind_comp_I_beam[0]:mins_ind_comp_I_beam[-1]+1]
##    I_beam_vector      = I_beam[mins_ind_comp_I_beam[0]:mins_ind_comp_I_beam[-1]+1]
##    [fft_I_beam,freq_I_beam,max_fft_I_beam,max_freq_I_beam]                 = FFT(time_vector_I_beam[1]-time_vector_I_beam[0],time_vector_I_beam,I_beam_vector)
##    
##    plt.figure("FFT signal")
##    plt.semilogx(freq_Id[1::],np.abs(fft_Id[1::]),'r',label=r'$I_d$ example')    
##    plt.semilogx(freq_I_beam[1::],np.abs(fft_I_beam[1::]),'g-',label=r'$I_{i \infty}$ example')   
##    
##    
##    # Both signals have the same frequency    
##    # Obtain the time shift
##    time_shift = time_maxs_Id-time_maxs_I_beam
##    # Force the phase shift to be in [-pi:pi]
##    period = 1.0/max_freq_Id
##    phase_shift = 2*np.pi*(((0.5 + time_shift/period) % 1.0) - 0.5)
##    # Convert to degrees
##    phase_shift_deg = phase_shift*180.0/np.pi
##    
##    mean_time_shift      = np.mean(time_shift)
##    mean_phase_shift_deg = np.mean(phase_shift_deg)
##    
##    # Use the cross-correlation of the two signals
##    Id_vector     = Id[nsteps-last_steps::]
##    I_beam_vector = I_beam[nsteps-last_steps::]
##    time_vector   = time[nsteps-last_steps::]
##    nsamples = len(time_vector)
##    xcorr = correlate(Id_vector, I_beam_vector)
###    xcorr = correlate(I_beam_vector, Id_vector)
##    # The peak of the cross-correlation gives the shift between the two signals
##    # The xcorr array goes from -nsamples to nsamples
##    dt = np.linspace(-time_vector[-1], time_vector[-1], 2*nsamples-1)
##    time_shift_corr = dt[xcorr.argmax()]
##    # Force the phase shift to be in [-pi:pi]
##    phase_shift_corr = 2*np.pi*(((0.5 + time_shift_corr/period) % 1.0) - 0.5)
##    # Convert to degrees
##    phase_shift_corr_deg = phase_shift_corr*180.0/np.pi
##    
##    
##    print("Id mean_val (A)        = "+str(mean_val_Id))
##    print("Id mean_max (A)        = "+str(mean_max_Id))
##    print("Id mean_min (A)        = "+str(mean_min_Id))
##    
##    print("I_beam mean_val (A)    = "+str(mean_val_I_beam))
##    print("I_beam mean_max (A)    = "+str(mean_max_I_beam))
##    print("I_beam mean_min (A)    = "+str(mean_min_I_beam))    
##    
##    print("Freq Id     (Hz)       = "+str(max_freq_Id))
##    print("Freq I_beam (Hz)       = "+str(max_freq_I_beam))  
##    print("phase_shift (deg)      = "+str(phase_shift_deg))
##    print("mean phase_shift (deg) = "+str(mean_phase_shift_deg))
##    print("phase_shift_corr (deg) = "+str(phase_shift_corr_deg))
##
##    
##    plt.figure("phase_shift")
##    plt.plot(phase_shift_deg,'ks',label = r'example')
    
    
    ##### Using new functions 
    
#    # Obtain the FFT for Id and I_beam in a period with an integer number of cycles using the new function
#    [fft_Id2,freq_Id2,max_fft_Id2,max_freq_Id2] = comp_FFT(time,Id,time[nsteps-last_steps::],Id[nsteps-last_steps::],order)
#    [fft_I_beam2,freq_I_beam2,max_fft_I_beam2,max_freq_I_beam2] = comp_FFT(time,I_beam,time[nsteps-last_steps::],I_beam[nsteps-last_steps::],order)
#    
#    plt.figure("FFT signal")
#    plt.semilogx(freq_Id2[1::],np.abs(fft_Id2[1::]),'b--',label=r'$I_d$ New function')    
#    plt.semilogx(freq_I_beam2[1::],np.abs(fft_I_beam2[1::]),'k--',label=r'$I_{i \infty}$ New function')
#    plt.legend()
#    
#    # Using the new function for computing the phase shift
#    [time_shift_vector,phase_shift_vector_deg,
#     mean_time_shift,mean_phase_shift_deg] = comp_phase_shift(time,Id,I_beam,time[nsteps-last_steps::],Id[nsteps-last_steps::],
#                                                              time[nsteps-last_steps::],I_beam[nsteps-last_steps::],order)
#    print("phase_shift_func (deg) = "+str(mean_phase_shift_deg))
#    
#    plt.figure("phase_shift")
#    plt.plot(phase_shift_vector_deg,'r^',label = r'New function')
#    plt.legend()
    
    # Analysis for avg_dens_mp_ions and avg_dens_mp_neus ##############################################
#    var = avg_dens_mp_ions                                                                          
#    [mins_ne,time_mins_ne,mins_ind_ne,maxs_ne,
#     time_maxs_ne,maxs_ind_ne,mean_min_ne,mean_max_ne,
#     mean_val_ne,max2mean_ne,min2mean_ne,amplitude_ne,
#     mins_ind_comp_ne,maxs_ind_comp_ne] = max_min_mean_vals(time_fast,time_fast[nsteps_fast-last_steps_fast::],var[nsteps_fast-last_steps_fast::],order_fast)
#     
#    var = avg_dens_mp_neus                                                                         
#    [mins_nn,time_mins_nn,mins_ind_nn,maxs_nn,
#     time_maxs_nn,maxs_ind_nn,mean_min_nn,mean_max_nn,
#     mean_val_nn,max2mean_nn,min2mean_nn,amplitude_nn,
#     mins_ind_comp_nn,maxs_ind_comp_nn] = max_min_mean_vals(time_fast,time_fast[nsteps_fast-last_steps_fast::],var[nsteps_fast-last_steps_fast::],order_fast)
#                                                                                      
#    
#    plt.figure("signal")
#    plt.semilogy(time_fast,avg_dens_mp_ions,'r')
#    plt.semilogy(time_mins_ne,mins_ne,'r^')
#    plt.semilogy(time_maxs_ne,maxs_ne,'rv')
#    plt.semilogy(time_fast,mean_val_ne*np.ones(np.shape(time_fast)),'r--')
#    plt.semilogy(time_fast,avg_dens_mp_neus,'g')
#    plt.semilogy(time_mins_nn,mins_nn,'g^')
#    plt.semilogy(time_maxs_nn,maxs_nn,'gv')
#    plt.semilogy(time_fast,mean_val_nn*np.ones(np.shape(time_fast)),'g--')
#    plt.semilogy(time_fast[nsteps_fast-last_steps_fast::],0.9*mean_val_ne*np.ones(np.shape(time_fast[nsteps_fast-last_steps_fast::])),'k--')
#    ax = plt.gca()
#    ax.set_ylim(1E15,1E19)
#    
#    # Obtain the FFT for avg_dens_mp_ions and avg_dens_mp_neus in a period with an integer number of cycles using the new function
#    [fft_ne,freq_ne,max_fft_ne,max_freq_ne] = comp_FFT(time_fast,avg_dens_mp_ions,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_ions[nsteps_fast-last_steps_fast::],order_fast)
#    [fft_nn,freq_nn,max_fft_nn,max_freq_nn] = comp_FFT(time_fast,avg_dens_mp_neus,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_neus[nsteps_fast-last_steps_fast::],order_fast)
#    
#    plt.figure("FFT signal")
#    plt.semilogx(freq_ne[1::],np.abs(fft_ne[1::]),'b--',label=r'$\bar{n}_e$ New function')    
#    plt.semilogx(freq_nn[1::],np.abs(fft_nn[1::]),'k--',label=r'$\bar{n}_n$ New function')
#    plt.legend()
#    
#    print("Freq ne (Hz)       = "+str(max_freq_ne))
#    print("Freq nn (Hz)       = "+str(max_freq_nn))  
#    
#    # Using the new function for computing the phase shift nn-ne
#    [time_shift_vector,phase_shift_vector_deg,
#     mean_time_shift,mean_phase_shift_deg] = comp_phase_shift(time_fast,avg_dens_mp_neus,avg_dens_mp_ions,time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_neus[nsteps_fast-last_steps_fast::],
#                                                              time_fast[nsteps_fast-last_steps_fast::],avg_dens_mp_ions[nsteps_fast-last_steps_fast::],order_fast)
#    print("phase_shift_func (deg) = "+str(mean_phase_shift_deg))
#    
#    plt.figure("phase_shift")
#    plt.plot(phase_shift_vector_deg,'r^',label = r'New function')
#    plt.legend()


    # Analysis for two variables var1 and var2 ##############################################
#    f = 1.2e4
#    var1 = ctr_mflow_fw_tot    
##    var1 = np.sin(2*np.pi*f*time)                                                                          
#    [mins_var1,time_mins_var1,mins_ind_var1,maxs_var1,
#     time_maxs_var1,maxs_ind_var1,mean_min_var1,mean_max_var1,
#     mean_val_var1,max2mean_Id,min2mean_var1,amplitude_var1,
#     mins_ind_comp_var1,maxs_ind_comp_var1] = max_min_mean_vals(time,time[nsteps-last_steps::],var1[nsteps-last_steps::],order)
#     
#    var2 = ctr_mflow_tw_tot  
##    var2 = np.sin(2*np.pi*f*time + np.pi)                                                                         
#    [mins_var2,time_mins_var2,mins_ind_var2,maxs_var2,
#     time_maxs_var2,maxs_ind_var2,mean_min_var2,mean_max_var2,
#     mean_val_var2,max2mean_Id,min2mean_var2,amplitude_var2,
#     mins_ind_comp_var2,maxs_ind_comp_var2] = max_min_mean_vals(time,time[nsteps-last_steps::],var2[nsteps-last_steps::],order)
#                                                                                      
#    
#    plt.figure("signal")
#    plt.plot(time,var1,'r',label=r'var1')
#    plt.plot(time_mins_var1,mins_var1,'r^')
#    plt.plot(time_maxs_var1,maxs_var1,'rv')
#    plt.plot(time,mean_val_var1*np.ones(np.shape(time)),'r--')
#    plt.plot(time,var2,'g',label=r'var2')
#    plt.plot(time_mins_var2,mins_var2,'g^')
#    plt.plot(time_maxs_var2,maxs_var2,'gv')
#    plt.plot(time,mean_val_var2*np.ones(np.shape(time)),'g--')
#    plt.plot(time[nsteps-last_steps::],0.9*mean_val_var1*np.ones(np.shape(time[nsteps-last_steps::])),'k--')
#    plt.legend()
#    
#    
#    # Obtain the FFT for var1 and var2 in a period with an integer number of cycles using the new function
#    [fft_var1,freq_var1,max_fft_var1,max_freq_var1] = comp_FFT(time,var1,time[nsteps-last_steps::],var1[nsteps-last_steps::],order)
#    [fft_var2,freq_var2,max_fft_var2,max_freq_var2] = comp_FFT(time,var2,time[nsteps-last_steps::],var2[nsteps-last_steps::],order)
#    
#    plt.figure("FFT signal")
#    plt.semilogx(freq_var1[1::],np.abs(fft_var1[1::]),'b--',label=r'var1 New function')    
#    plt.semilogx(freq_var2[1::],np.abs(fft_var2[1::]),'k--',label=r'var2 New function')
#    plt.legend()
#    
#    print("Freq var1 (Hz)       = "+str(max_freq_var1))
#    print("Freq var2 (Hz)       = "+str(max_freq_var2))  
#    
#    # Using the new function for computing the phase shift
#    [time_shift_vector,phase_shift_vector_deg,
#     mean_time_shift,mean_phase_shift_deg] = comp_phase_shift(time,var1,var2,time[nsteps-last_steps::],var1[nsteps-last_steps::],
#                                                              time[nsteps-last_steps::],var2[nsteps-last_steps::],order)
#    print("phase_shift_func (deg) = "+str(mean_phase_shift_deg))
#    print("phase_shift vect (deg) = "+str(phase_shift_vector_deg))  
#    
#    plt.figure("phase_shift")
#    plt.plot(phase_shift_vector_deg,'r^',label = r'New function')
#    plt.legend()
    
    
    ##################### FUNCTION comp_Boltz #################################
    
#    [Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,fn1_x,fn1_y,fn1_z,
#       fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,
#       un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,ji1_z,
#       ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,ue_perp,
#       ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,
#       avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
#       ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,
#       F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12] = HET_sims_plotvars(nodes_flag,cells_flag,Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,
#                                                                                                                         fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,
#                                                                                                                         un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,
#                                                                                                                         ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,
#                                                                                                                         ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
#                                                                                                                         n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
#                                                                                                                         neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
#                                                                                                                         ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,F_theta,Hall_par,Hall_par_eff,
#                                                                                                                         nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12)
#
#    [phi_mean,Er_mean,Ez_mean,Efield_mean,nn1_mean,nn2_mean,
#       ni1_mean,ni2_mean,ne_mean,fn1_x_mean,fn1_y_mean,fn1_z_mean,
#       fn2_x_mean,fn2_y_mean,fn2_z_mean,fi1_x_mean,fi1_y_mean,fi1_z_mean,
#       fi2_x_mean,fi2_y_mean,fi2_z_mean,un1_x_mean,un1_y_mean,un1_z_mean,
#       un2_x_mean,un2_y_mean,un2_z_mean,ui1_x_mean,ui1_y_mean,ui1_z_mean,
#       ui2_x_mean,ui2_y_mean,ui2_z_mean,ji1_x_mean,ji1_y_mean,ji1_z_mean,
#       ji2_x_mean,ji2_y_mean,ji2_z_mean,je_r_mean,je_t_mean,je_z_mean,
#       je_perp_mean,je_para_mean,ue_r_mean,ue_t_mean,ue_z_mean,ue_perp_mean,
#       ue_para_mean,uthetaExB_mean,Tn1_mean,Tn2_mean,Ti1_mean,Ti2_mean,Te_mean,
#       n_mp_n1_mean,n_mp_n2_mean,n_mp_i1_mean,n_mp_i2_mean,avg_w_n1_mean,
#       avg_w_n2_mean,avg_w_i1_mean,avg_w_i2_mean,neu_gen_weights1_mean,
#       neu_gen_weights2_mean,ion_gen_weights1_mean,ion_gen_weights2_mean,
#       ndot_ion01_n1_mean,ndot_ion02_n1_mean,ndot_ion12_i1_mean,
#       ne_cath_mean,nu_cath_mean,ndot_cath_mean,F_theta_mean,Hall_par_mean,
#       Hall_par_eff_mean,nu_e_tot_mean,nu_e_tot_eff_mean,nu_en_mean,
#       nu_ei1_mean,nu_ei2_mean,nu_i01_mean,nu_i02_mean,nu_i12_mean,
#       Boltz_mean,Boltz_dim_mean,phi_elems_mean,ne_elems_mean,Te_elems_mean] = HET_sims_mean(nsteps,mean_type,last_steps,step_i,step_f,phi,Er,Ez,Efield,Br,Bz,Bfield,
#                                                                                             nn1,nn2,ni1,ni2,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,
#                                                                                             fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,
#                                                                                             ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,
#                                                                                             je_perp,je_para,ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
#                                                                                             n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
#                                                                                             neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
#                                                                                             ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,nu_cath,ndot_cath,F_theta,
#                                                                                             Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,
#                                                                                             nu_i02,nu_i12,Boltz,Boltz_dim,phi_elems,ne_elems,Te_elems)
#    
#    [cath_Bline_phi, cath_phi, cath_Bline_Te, cath_Te, cath_Bline_ne,
#     cath_ne, cath_Bline_nodim_Boltz, cath_nodim_Boltz,
#     cath_Bline_dim_Boltz, cath_dim_Boltz] = comp_Boltz(elems_cath_Bline,cath_elem,zs,rs,elem_geom,phi_mean,Te_mean,ne_mean)
#    
#    plt.figure("phi at cathode B line")
#    plt.plot(elem_geom[3,elems_cath_Bline],cath_Bline_phi-cath_phi)
#    plt.plot(elem_geom[3,cath_elem],cath_phi-cath_phi,'ks')
#    plt.plot(elem_geom[3,elems_cath_Bline],np.zeros(np.shape(elem_geom[3,elems_cath_Bline])),'--')
#    plt.xlabel(r"$\sigma$ (m T)")
#    plt.ylabel(r"$\phi$ (V)")
    
    
#    T = 1.0
#    freq = 1/T
#    nsteps = 1000
#    last_steps = 1000
#    t = np.linspace(0,2.5*T,nsteps)
#    var = np.sin(2*np.pi*freq*t)
#    order = 50
#    
#    [mins,time_mins,mins_ind,maxs,
#            time_maxs,maxs_ind,mean_min,mean_max,
#            mean_val,max2mean,min2mean,amplitude,
#            mins_ind_comp,maxs_ind_comp] = max_min_mean_vals(t,t[nsteps-last_steps::],var[nsteps-last_steps::],order)
#    
#    
#    plt.figure("test")
#    plt.plot(t,var,'k')
#    plt.plot(time_mins,mins,'r^')
#    plt.plot(time_maxs,maxs,'g^')
#    
#    mean_val_mins  = np.mean(var[mins_ind[0]:mins_ind[-1]+1])
#    mean_val_maxs  = np.mean(var[maxs_ind[0]:maxs_ind[-1]+1])
#    print(mean_val_mins)
#    print(mean_val_maxs)