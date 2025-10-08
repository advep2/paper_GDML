# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:02:02 2018

@author: adrian

###############################################################################
Description:    This Python function returns the FFT (Fast Fourier Transform)
                of a given data (temporal sample)
###############################################################################
Inputs:         1) dt: sampling interval
                2) time: time vector at which data has been sampled
                3) data: data vector with the data sample
###############################################################################
Outputs:        1) Fast Fourier Transform
"""


def FFT(dt,time,data,plot=False):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # If the data shape is uneven, it is fixed here to be even
    if data.shape[0] % 2 != 0:
        print("FFT: signal preferred to be even in size, auto fixing it...")
        time = time[0:-1]
        data = data[0:-1]
        
#    # Perform windowing of the signal to reduce effect of time finite signal
#    window = np.hamming(len(data))
    
    # Obtain the FFT
    dataFFT = np.fft.fft(data) / time.shape[0]  # divided by size t for coherent magnitude
#    dataFFT = np.fft.fft(data*window) / time.shape[0]  # divided by size t for coherent magnitude
    # Obtain the frequency vector
    freq = np.fft.fftfreq(time.shape[0], d=dt)    
    
    # We work with real signals, so that only right half of freq axis is needed.
    # Besides, we do not consider the first value (corresponding to freq = 0 term,
    # which is the mean funciton value). The FFT is divided by the length of the
    # signal and is multiplied by 2 in order to proper normalized it. However, the
    # value for freq = 0 must not be multiplied by 2.
    # NOTE: That is only true for sine analytic signals. Therefore, in order to
    # normalize we simply divide by the largest amplitude value
    firstNegInd  = np.argmax(freq < 0)
    freqAxisReal = freq[0:firstNegInd]
    dataFFTReal   = dataFFT[0:firstNegInd]  
    max_dataFFTReal = np.max(np.abs(dataFFTReal[1:]))
    max_freq        = freqAxisReal[np.where(np.abs(dataFFTReal[1:]) == np.max(np.abs(dataFFTReal[1:])))[0][0]+1]
    for i in range(1,firstNegInd):
#        dataFFTReal[i] = 2 * dataFFTReal[i]
        dataFFTReal[i] = dataFFTReal[i]/max_dataFFTReal
        
    if plot:
        plt.figure("Signal")
        plt.plot(time,data)
        plt.xlabel(r'$t$ (s)')
        plt.title(r'Temporal signal inputed')
        
        plt.figure("FFT")
        plt.semilogx(freqAxisReal[1:],np.abs(dataFFTReal[1:]))
        ax = plt.gca()
        ax.xaxis.grid(which='minor')
        plt.xlabel(r'$f$ (Hz)')
        plt.title(r'Normalized amplitude in frequency domain')
        
    
    
    return dataFFTReal, freqAxisReal, max_dataFFTReal, max_freq
    
    
if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 
    from HET_sims_read import HET_sims_read

    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))
    
    data_analytic_flag = 1
    
    if data_analytic_flag == 0:
        ############################## SIMULATION DATA ########################
        sim_name = "../../Rb_hyphen/sim/sims/SPT100_al0025_Ne5_C1"
#        sim_name = "../../Rb_hyphen/sim/sims/SPT100_al0025_Ne5_C1_Navghalf"
        timestep         = -1
        allsteps_flag    = 1
        read_inst_data   = 1
        read_part_tracks = 0
        read_part_lists  = 0
        read_flag        = 0
        
        last_step      = 600
        last_step_fast = 30000
        
        path_picM         = sim_name+"/SET/inp/SPT100_picM.hdf5"
        path_simstate_inp = sim_name+"/CORE/inp/SimState.hdf5"
        path_simstate_out = sim_name+"/CORE/out/SimState.hdf5"
        path_postdata_out = sim_name+"/CORE/out/PostData.hdf5"
        path_simparams_inp = sim_name+"/CORE/inp/sim_params.inp"
        
#        [num_ion_spe,num_neu_spe,n_mp_cell_i,n_mp_cell_n,n_mp_cell_i_min,
#           n_mp_cell_i_max,n_mp_cell_n_min,n_mp_cell_n_max,min_ion_plasma_density,
#           m_A,spec_refl_prob,ene_bal,points,zs,rs,zscells,rscells,dims,
#           nodes_flag,cells_flag,ind_maxr_c,ind_maxz_c,nr_c,nz_c,eta_max,
#           eta_min,xi_top,xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,
#           nsteps,nsteps_eFld,faces,nodes,boundary_f,face_geom,elem_geom,n_faces,
#           n_elems,cath_elem,z_cath,r_cath,V_cath,mass,ssIons1,ssIons2,ssNeutrals1,ssNeutrals2,
#           n_mp_i1_list,n_mp_i2_list,n_mp_n1_list,n_mp_n2_list,phi,Ez,Er,Efield,
#           Bz,Br,Bfield,Te,cs01,cs02,nn1,nn2,ni1,ni2,ne,fn1_x,fn1_y,fn1_z,fn2_x,
#           fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,
#           un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,
#           ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,
#           ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,
#           avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
#           ion_gen_weights1,ion_gen_weights2,surf_elems,n_imp_elems,imp_elems,
#           imp_elems_kbc,imp_elems_MkQ1,imp_elems_Te,imp_elems_dphi_kbc,
#           imp_elems_dphi_sh,imp_elems_nQ1,imp_elems_nQ2,imp_elems_ion_flux_in1,
#           imp_elems_ion_flux_out1,imp_elems_ion_ene_flux_in1,
#           imp_elems_ion_ene_flux_out1,imp_elems_ion_imp_ene1,
#           imp_elems_ion_flux_in2,imp_elems_ion_flux_out2,
#           imp_elems_ion_ene_flux_in2,imp_elems_ion_ene_flux_out2,
#           imp_elems_ion_imp_ene2,imp_elems_neu_flux_in1,imp_elems_neu_flux_out1,
#           imp_elems_neu_ene_flux_in1,imp_elems_neu_ene_flux_out1,
#           imp_elems_neu_imp_ene1,imp_elems_neu_flux_in2,imp_elems_neu_flux_out2,
#           imp_elems_neu_ene_flux_in2,imp_elems_neu_ene_flux_out2,
#           imp_elems_neu_imp_ene2,tot_mass_mp_neus,tot_mass_mp_ions,tot_num_mp_neus,
#           tot_num_mp_ions,tot_mass_exit_neus,tot_mass_exit_ions,mass_mp_neus,
#           mass_mp_ions,num_mp_neus,num_mp_ions,eta_u,eta_prod,eta_thr,eta_div,
#           eta_cur,thrust,thrust_ion,thrust_neu,Id_inst,Id,Vd_inst,Vd,dMdt_i1,
#           dMdt_i2,dMdt_n1,dMdt_n2,mflow_coll_i1,mflow_coll_i2,mflow_coll_n1,
#           mflow_coll_n2,mflow_fw_i1,mflow_fw_i2,mflow_fw_n1,mflow_fw_n2,
#           mflow_tw_i1,mflow_tw_i2,mflow_tw_n1,mflow_tw_n2,mflow_ircmb_picS_n1,
#           mflow_ircmb_picS_n2,mflow_inj_i1,mflow_inj_i2,mflow_fwmat_i1,
#           mflow_fwmat_i2,mflow_inj_n1,mflow_fwmat_n1,mflow_inj_n2,mflow_fwmat_n2,
#           mflow_twmat_i1,mflow_twinf_i1,mflow_twa_i1,mflow_twmat_i2,mflow_twinf_i2,
#           mflow_twa_i2,mflow_twmat_n1,mflow_twinf_n1,mflow_twa_n1,mflow_twmat_n2,
#           mflow_twinf_n2,mflow_twa_n2,dEdt_i1,dEdt_i2,dEdt_n1,dEdt_n2,
#           eneflow_coll_i1,eneflow_coll_i2,eneflow_coll_n1,eneflow_coll_n2,
#           eneflow_fw_i1,eneflow_fw_i2,eneflow_fw_n1,eneflow_fw_n2,eneflow_tw_i1,
#           eneflow_tw_i2,eneflow_tw_n1,eneflow_tw_n2,Pfield_i1,Pfield_i2,
#           eneflow_inj_i1,eneflow_fwmat_i1,eneflow_inj_i2,eneflow_fwmat_i2,
#           eneflow_inj_n1,eneflow_fwmat_n1,eneflow_inj_n2,eneflow_fwmat_n2,
#           eneflow_twmat_i1,eneflow_twinf_i1,eneflow_twa_i1,eneflow_twmat_i2,
#           eneflow_twinf_i2,eneflow_twa_i2,eneflow_twmat_n1,eneflow_twinf_n1,
#           eneflow_twa_n1,eneflow_twmat_n2,eneflow_twinf_n2,eneflow_twa_n2,
#           ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,nu_cath,ndot_cath,
#           F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,
#           nu_ei2,nu_i01,nu_i02,nu_i12] = HET_sims_read(path_simstate_inp,path_simstate_out,
#                                                        path_postdata_out,path_simparams_inp,
#                                                        path_picM,allsteps_flag,timestep,read_inst_data,
#                                                        read_part_lists,read_flag)
        
        
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
                                                        
        # Average Id
        [fft,freq,max_fft,max_freq] = FFT(time[1]-time[0],time[nsteps-last_step::],Id[nsteps-last_step::],plot=True)
#        [fft,freq,max_fft,max_freq] = FFT(time_fast[1]-time_fast[0],time_fast[steps_fast[-1]-last_step_fast::],tot_mass_mp_ions[steps_fast[-1]-last_step_fast::],plot=True)
        
    #    [frq,ft] = FFT(time_fast[1]-time_fast[0],time_fast[steps_fast[-1]-last_step_fast::],tot_mass_mp_ions[steps_fast[-1]-last_step_fast::])
        
    #    fig, ax = plt.subplots(2, 1)
    ##    ax[0].plot(time[nsteps-last_step::],Id[nsteps-last_step::])
    #    ax[0].plot(time_fast[steps_fast[-1]-last_step_fast::],tot_mass_mp_ions[steps_fast[-1]-last_step_fast::])
    #    ax[0].set_xlabel('t (s)')
    #    ax[0].set_ylabel('Id')
    #    ax[1].plot(frq[1::],abs(ft[1::])/np.max(abs(ft[1::])),'r') # plotting the spectrum
    #    ax[1].set_xlabel('f (Hz)')
    #    ax[1].set_ylabel('|Y(freq)|')
    #    
    #    print(np.shape(ft))
    #    print(frq[np.where(abs(ft[1::]) == np.max(abs(ft[1::])))])
    
    elif data_analytic_flag == 1:
        ############################## ANALYTIC FUNCTION ######################
    #    # Sampling time vector definition
    #    Fs = 150.0
    #    dt = 1/Fs
    #    t = np.arange(0,1+dt,dt)
    #    # Function (signal) definition
    #    ff = 5
    #    sig = 10 + np.sin(2*np.pi*ff*t)    
    
        n = 1000 # Number of data points
        dt = 5.0 # Sampling period 
        t = dt*np.arange(0,n) # t vector coordinates
        w1 = 100.0 # wavelength (seconds)
        w2 = 20.0 # wavelength (seconds)
        sig = np.sin(2*np.pi*t/w1) + 2*np.cos(2*np.pi*t/w2) # signal
        
        [fft,freq,max_fft, max_freq] = FFT(dt,t,sig,plot=True)
        
    
#    # Saving a .mat
#    import scipy.io as sio
#    # Create a dictionary
#    dic = {}
#    dic['Id'] = Id
#    dic['time'] = time
#    sio.savemat('Id_time.mat', dic)
    
    
