#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:54:50 2020

@author: adrian

############################################################################
Description:    This python script performs the time-average of the given data
                from HET sims at the boundary
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

def HET_sims_mean_bound(nsteps,mean_type,last_steps,step_i,step_f,delta_r,
                        delta_s,delta_s_csl,dphi_sh_b,je_b,ji_tot_b,gp_net_b,ge_sb_b,
                        relerr_je_b,qe_tot_wall,qe_tot_s_wall,qe_tot_b,qe_b,
                        qe_b_bc,qe_b_fl,imp_ene_e_wall,imp_ene_e_b,relerr_qe_b,
                        relerr_qe_b_cons,Te,phi,err_interp_phi,err_interp_Te,
                        err_interp_jeperp,err_interp_jetheta,err_interp_jepara,
                        err_interp_jez,err_interp_jer,n_inst,ni1_inst,ni2_inst,nn1_inst,
                        inst_dphi_sh_b_Te,inst_imp_ene_e_b,inst_imp_ene_e_b_Te,
                        inst_imp_ene_e_wall,inst_imp_ene_e_wall_Te,
                        
                        delta_r_nodes,delta_s_nodes,delta_s_csl_nodes,dphi_sh_b_nodes,je_b_nodes,
                        gp_net_b_nodes,ge_sb_b_nodes,relerr_je_b_nodes,qe_tot_wall_nodes,
                        qe_tot_s_wall_nodes,qe_tot_b_nodes,qe_b_nodes,qe_b_bc_nodes,
                        qe_b_fl_nodes,imp_ene_e_wall_nodes,imp_ene_e_b_nodes,
                        relerr_qe_b_nodes,relerr_qe_b_cons_nodes,Te_nodes,
                        phi_nodes,err_interp_n_nodes,n_inst_nodes,ni1_inst_nodes,ni2_inst_nodes,
                        nn1_inst_nodes,n_nodes,ni1_nodes,ni2_nodes,nn1_nodes,dphi_kbc_nodes,
                        MkQ1_nodes,ji1_nodes,ji2_nodes,ji3_nodes,ji4_nodes,ji_nodes,gn1_tw_nodes,
                        gn1_fw_nodes,gn2_tw_nodes,gn2_fw_nodes,gn3_tw_nodes,gn3_fw_nodes,gn_tw_nodes,                        
                        qi1_tot_wall_nodes,qi2_tot_wall_nodes,qi3_tot_wall_nodes,
                        qi4_tot_wall_nodes,qi_tot_wall_nodes,qn1_tw_nodes,qn1_fw_nodes,
                        qn2_tw_nodes,qn2_fw_nodes,qn3_tw_nodes,qn3_fw_nodes,qn_tot_wall_nodes,
                        imp_ene_i1_nodes,imp_ene_i2_nodes,imp_ene_i3_nodes,imp_ene_i4_nodes,
                        imp_ene_ion_nodes,imp_ene_ion_nodes_v2,imp_ene_n1_nodes,imp_ene_n2_nodes,
                        imp_ene_n3_nodes,imp_ene_n_nodes,imp_ene_n_nodes_v2,
                        inst_dphi_sh_b_Te_nodes,inst_imp_ene_e_b_nodes,
                        inst_imp_ene_e_b_Te_nodes,inst_imp_ene_e_wall_nodes,
                        inst_imp_ene_e_wall_Te_nodes,
                        
                        delta_r_surf,delta_s_surf,delta_s_csl_surf,dphi_sh_b_surf,je_b_surf,gp_net_b_surf,
                        ge_sb_b_surf,relerr_je_b_surf,qe_tot_wall_surf,qe_tot_s_wall_surf,
                        qe_tot_b_surf,qe_b_surf,qe_b_bc_surf,qe_b_fl_surf,
                        imp_ene_e_wall_surf,imp_ene_e_b_surf,relerr_qe_b_surf,
                        relerr_qe_b_cons_surf,Te_surf,phi_surf,nQ1_inst_surf,nQ1_surf,
                        nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,MkQ1_surf,ji1_surf,ji2_surf,
                        ji3_surf,ji4_surf,ji_surf,gn1_tw_surf,gn1_fw_surf,gn2_tw_surf,gn2_fw_surf,
                        gn3_tw_surf,gn3_fw_surf,gn_tw_surf,
                        qi1_tot_wall_surf,qi2_tot_wall_surf,qi3_tot_wall_surf,qi4_tot_wall_surf,
                        qi_tot_wall_surf,qn1_tw_surf,qn1_fw_surf,qn2_tw_surf,qn2_fw_surf,
                        qn3_tw_surf,qn3_fw_surf,qn_tot_wall_surf,imp_ene_i1_surf,imp_ene_i2_surf,
                        imp_ene_i3_surf,imp_ene_i4_surf,imp_ene_ion_surf,imp_ene_ion_surf_v2,
                        imp_ene_n1_surf,imp_ene_n2_surf,imp_ene_n3_surf,imp_ene_n_surf,imp_ene_n_surf_v2,
                        inst_dphi_sh_b_Te_surf,inst_imp_ene_e_b_surf,inst_imp_ene_e_b_Te_surf,
                        inst_imp_ene_e_wall_surf,inst_imp_ene_e_wall_Te_surf):
    
    import numpy as np
    
    # Parameters
    e = 1.6021766E-19
    
    
    # Since we have BM oscillations, time average of non-linear operations is not equal to the non-linear operation computed 
    # with time-averaged variables.
    # As for efficiencies, since they are defined after time-averaging current and/or power balances, it makes sense to compute them
    # with time average quantities. 
    # However, for wall-impact energies (ene = P/g), it is more correct to do the average of the instantaneous impact energies.
    # We shall compute here time-averaged impact energies using the average of instantaneous values.
    
    
    if mean_type == 0:
        delta_r_mean                = np.nanmean(delta_r[nsteps-last_steps::,:],axis=0)
        delta_s_mean                = np.nanmean(delta_s[nsteps-last_steps::,:],axis=0)
        delta_s_csl_mean            = np.nanmean(delta_s_csl[nsteps-last_steps::,:],axis=0)
        dphi_sh_b_mean              = np.nanmean(dphi_sh_b[nsteps-last_steps::,:],axis=0)
        je_b_mean                   = np.nanmean(je_b[nsteps-last_steps::,:],axis=0)
        ji_tot_b_mean               = np.nanmean(ji_tot_b[nsteps-last_steps::,:],axis=0)
        gp_net_b_mean               = np.nanmean(gp_net_b[nsteps-last_steps::,:],axis=0)
        ge_sb_b_mean                = np.nanmean(ge_sb_b[nsteps-last_steps::,:],axis=0)
        relerr_je_b_mean            = np.nanmean(relerr_je_b[nsteps-last_steps::,:],axis=0)
        qe_tot_wall_mean            = np.nanmean(qe_tot_wall[nsteps-last_steps::,:],axis=0)
        qe_tot_s_wall_mean          = np.nanmean(qe_tot_s_wall[nsteps-last_steps::,:],axis=0)
        qe_tot_b_mean               = np.nanmean(qe_tot_b[nsteps-last_steps::,:],axis=0)
        qe_b_mean                   = np.nanmean(qe_b[nsteps-last_steps::,:],axis=0)
        qe_b_bc_mean                = np.nanmean(qe_b_bc[nsteps-last_steps::,:],axis=0)
        qe_b_fl_mean                = np.nanmean(qe_b_fl[nsteps-last_steps::,:],axis=0)
        imp_ene_e_wall_mean         = np.nanmean(imp_ene_e_wall[nsteps-last_steps::,:],axis=0)
        imp_ene_e_b_mean            = np.nanmean(imp_ene_e_b[nsteps-last_steps::,:],axis=0)
        relerr_qe_b_mean            = np.nanmean(relerr_qe_b[nsteps-last_steps::,:],axis=0)
        relerr_qe_b_cons_mean       = np.nanmean(relerr_qe_b_cons[nsteps-last_steps::,:],axis=0)
        Te_mean                     = np.nanmean(Te[nsteps-last_steps::,:],axis=0)
        phi_mean                    = np.nanmean(phi[nsteps-last_steps::,:],axis=0)
        err_interp_phi_mean         = np.nanmean(err_interp_phi[nsteps-last_steps::,:],axis=0)
        err_interp_Te_mean          = np.nanmean(err_interp_Te[nsteps-last_steps::,:],axis=0)
        err_interp_jeperp_mean      = np.nanmean(err_interp_jeperp[nsteps-last_steps::,:],axis=0)
        err_interp_jetheta_mean     = np.nanmean(err_interp_jetheta[nsteps-last_steps::,:],axis=0)
        err_interp_jepara_mean      = np.nanmean(err_interp_jepara[nsteps-last_steps::,:],axis=0)
        err_interp_jez_mean         = np.nanmean(err_interp_jez[nsteps-last_steps::,:],axis=0)
        err_interp_jer_mean         = np.nanmean(err_interp_jer[nsteps-last_steps::,:],axis=0)
        n_inst_mean                 = np.nanmean(n_inst[nsteps-last_steps::,:],axis=0)
        ni1_inst_mean               = np.nanmean(ni1_inst[nsteps-last_steps::,:],axis=0)
        ni2_inst_mean               = np.nanmean(ni2_inst[nsteps-last_steps::,:],axis=0)
        nn1_inst_mean               = np.nanmean(nn1_inst[nsteps-last_steps::,:],axis=0)
        inst_dphi_sh_b_Te_mean      = np.nanmean(inst_dphi_sh_b_Te[nsteps-last_steps::,:],axis=0)
        inst_imp_ene_e_b_mean       = np.nanmean(inst_imp_ene_e_b[nsteps-last_steps::,:],axis=0)
        inst_imp_ene_e_b_Te_mean    = np.nanmean(inst_imp_ene_e_b_Te[nsteps-last_steps::,:],axis=0)
        inst_imp_ene_e_wall_mean    = np.nanmean(inst_imp_ene_e_wall[nsteps-last_steps::,:],axis=0)
        inst_imp_ene_e_wall_Te_mean = np.nanmean(inst_imp_ene_e_wall_Te[nsteps-last_steps::,:],axis=0)
        
        
        delta_r_nodes_mean                = np.nanmean(delta_r_nodes[:,:,nsteps-last_steps::],axis=2)
        delta_s_nodes_mean                = np.nanmean(delta_s_nodes[:,:,nsteps-last_steps::],axis=2)
        delta_s_csl_nodes_mean            = np.nanmean(delta_s_csl_nodes[:,:,nsteps-last_steps::],axis=2)
        dphi_sh_b_nodes_mean              = np.nanmean(dphi_sh_b_nodes[:,:,nsteps-last_steps::],axis=2)
        je_b_nodes_mean                   = np.nanmean(je_b_nodes[:,:,nsteps-last_steps::],axis=2)
        gp_net_b_nodes_mean               = np.nanmean(gp_net_b_nodes[:,:,nsteps-last_steps::],axis=2)
        ge_sb_b_nodes_mean                = np.nanmean(ge_sb_b_nodes[:,:,nsteps-last_steps::],axis=2)
        relerr_je_b_nodes_mean            = np.nanmean(relerr_je_b_nodes[:,:,nsteps-last_steps::],axis=2)
        qe_tot_wall_nodes_mean            = np.nanmean(qe_tot_wall_nodes[:,:,nsteps-last_steps::],axis=2)
        qe_tot_s_wall_nodes_mean          = np.nanmean(qe_tot_s_wall_nodes[:,:,nsteps-last_steps::],axis=2)
        qe_tot_b_nodes_mean               = np.nanmean(qe_tot_b_nodes[:,:,nsteps-last_steps::],axis=2)
        qe_b_nodes_mean                   = np.nanmean(qe_b_nodes[:,:,nsteps-last_steps::],axis=2)
        qe_b_bc_nodes_mean                = np.nanmean(qe_b_bc_nodes[:,:,nsteps-last_steps::],axis=2)
        qe_b_fl_nodes_mean                = np.nanmean(qe_b_fl_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_e_wall_nodes_mean         = np.nanmean(imp_ene_e_wall_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_e_b_nodes_mean            = np.nanmean(imp_ene_e_b_nodes[:,:,nsteps-last_steps::],axis=2)
        relerr_qe_b_nodes_mean            = np.nanmean(relerr_qe_b_nodes[:,:,nsteps-last_steps::],axis=2)
        relerr_qe_b_cons_nodes_mean       = np.nanmean(relerr_qe_b_cons_nodes[:,:,nsteps-last_steps::],axis=2)
        Te_nodes_mean                     = np.nanmean(Te_nodes[:,:,nsteps-last_steps::],axis=2)
        phi_nodes_mean                    = np.nanmean(phi_nodes[:,:,nsteps-last_steps::],axis=2)
        err_interp_n_nodes_mean           = np.nanmean(err_interp_n_nodes[:,:,nsteps-last_steps::],axis=2)
        n_inst_nodes_mean                 = np.nanmean(n_inst_nodes[:,:,nsteps-last_steps::],axis=2)
        ni1_inst_nodes_mean               = np.nanmean(ni1_inst_nodes[:,:,nsteps-last_steps::],axis=2)
        ni2_inst_nodes_mean               = np.nanmean(ni2_inst_nodes[:,:,nsteps-last_steps::],axis=2)
        nn1_inst_nodes_mean               = np.nanmean(nn1_inst_nodes[:,:,nsteps-last_steps::],axis=2)
        n_nodes_mean                      = np.nanmean(n_nodes[:,:,nsteps-last_steps::],axis=2)
        ni1_nodes_mean                    = np.nanmean(ni1_nodes[:,:,nsteps-last_steps::],axis=2)
        ni2_nodes_mean                    = np.nanmean(ni2_nodes[:,:,nsteps-last_steps::],axis=2)
        nn1_nodes_mean                    = np.nanmean(nn1_nodes[:,:,nsteps-last_steps::],axis=2)
        dphi_kbc_nodes_mean               = np.nanmean(dphi_kbc_nodes[:,:,nsteps-last_steps::],axis=2)
        MkQ1_nodes_mean                   = np.nanmean(MkQ1_nodes[:,:,nsteps-last_steps::],axis=2)
        ji1_nodes_mean                    = np.nanmean(ji1_nodes[:,:,nsteps-last_steps::],axis=2)
        ji2_nodes_mean                    = np.nanmean(ji2_nodes[:,:,nsteps-last_steps::],axis=2)
        ji3_nodes_mean                    = np.nanmean(ji3_nodes[:,:,nsteps-last_steps::],axis=2)
        ji4_nodes_mean                    = np.nanmean(ji4_nodes[:,:,nsteps-last_steps::],axis=2)
        ji_nodes_mean                     = np.nanmean(ji_nodes[:,:,nsteps-last_steps::],axis=2)
        gn1_tw_nodes_mean                 = np.nanmean(gn1_tw_nodes[:,:,nsteps-last_steps::],axis=2)
        gn1_fw_nodes_mean                 = np.nanmean(gn1_fw_nodes[:,:,nsteps-last_steps::],axis=2)
        gn2_tw_nodes_mean                 = np.nanmean(gn2_tw_nodes[:,:,nsteps-last_steps::],axis=2)
        gn2_fw_nodes_mean                 = np.nanmean(gn2_fw_nodes[:,:,nsteps-last_steps::],axis=2)
        gn3_tw_nodes_mean                 = np.nanmean(gn3_tw_nodes[:,:,nsteps-last_steps::],axis=2)
        gn3_fw_nodes_mean                 = np.nanmean(gn3_fw_nodes[:,:,nsteps-last_steps::],axis=2)
        gn_tw_nodes_mean                  = np.nanmean(gn_tw_nodes[:,:,nsteps-last_steps::],axis=2)
        qi1_tot_wall_nodes_mean           = np.nanmean(qi1_tot_wall_nodes[:,:,nsteps-last_steps::],axis=2)
        qi2_tot_wall_nodes_mean           = np.nanmean(qi2_tot_wall_nodes[:,:,nsteps-last_steps::],axis=2)
        qi3_tot_wall_nodes_mean           = np.nanmean(qi3_tot_wall_nodes[:,:,nsteps-last_steps::],axis=2)
        qi4_tot_wall_nodes_mean           = np.nanmean(qi4_tot_wall_nodes[:,:,nsteps-last_steps::],axis=2)
        qi_tot_wall_nodes_mean            = np.nanmean(qi_tot_wall_nodes[:,:,nsteps-last_steps::],axis=2)
        qn1_tw_nodes_mean                 = np.nanmean(qn1_tw_nodes[:,:,nsteps-last_steps::],axis=2)
        qn1_fw_nodes_mean                 = np.nanmean(qn1_fw_nodes[:,:,nsteps-last_steps::],axis=2)
        qn2_tw_nodes_mean                 = np.nanmean(qn2_tw_nodes[:,:,nsteps-last_steps::],axis=2)
        qn2_fw_nodes_mean                 = np.nanmean(qn2_fw_nodes[:,:,nsteps-last_steps::],axis=2)
        qn3_tw_nodes_mean                 = np.nanmean(qn3_tw_nodes[:,:,nsteps-last_steps::],axis=2)
        qn3_fw_nodes_mean                 = np.nanmean(qn3_fw_nodes[:,:,nsteps-last_steps::],axis=2)
        qn_tot_wall_nodes_mean            = np.nanmean(qn_tot_wall_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_i1_nodes_mean             = np.nanmean(imp_ene_i1_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_i2_nodes_mean             = np.nanmean(imp_ene_i2_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_i3_nodes_mean             = np.nanmean(imp_ene_i3_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_i4_nodes_mean             = np.nanmean(imp_ene_i4_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_ion_nodes_mean            = np.nanmean(imp_ene_ion_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_ion_nodes_v2_mean         = np.nanmean(imp_ene_ion_nodes_v2[:,:,nsteps-last_steps::],axis=2)
        imp_ene_n1_nodes_mean             = np.nanmean(imp_ene_n1_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_n2_nodes_mean             = np.nanmean(imp_ene_n2_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_n3_nodes_mean             = np.nanmean(imp_ene_n3_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_n_nodes_mean              = np.nanmean(imp_ene_n_nodes[:,:,nsteps-last_steps::],axis=2)
        imp_ene_n_nodes_v2_mean           = np.nanmean(imp_ene_n_nodes_v2[:,:,nsteps-last_steps::],axis=2)
        inst_dphi_sh_b_Te_nodes_mean      = np.nanmean(inst_dphi_sh_b_Te_nodes[:,:,nsteps-last_steps::],axis=2)
        inst_imp_ene_e_b_nodes_mean       = np.nanmean(inst_imp_ene_e_b_nodes[:,:,nsteps-last_steps::],axis=2)
        inst_imp_ene_e_b_Te_nodes_mean    = np.nanmean(inst_imp_ene_e_b_Te_nodes[:,:,nsteps-last_steps::],axis=2)
        inst_imp_ene_e_wall_nodes_mean    = np.nanmean(inst_imp_ene_e_wall_nodes[:,:,nsteps-last_steps::],axis=2)
        inst_imp_ene_e_wall_Te_nodes_mean = np.nanmean(inst_imp_ene_e_wall_Te_nodes[:,:,nsteps-last_steps::],axis=2)
        
        
        delta_r_surf_mean                = np.nanmean(delta_r_surf[:,nsteps-last_steps::],axis=1)
        delta_s_surf_mean                = np.nanmean(delta_s_surf[:,nsteps-last_steps::],axis=1)
        delta_s_csl_surf_mean            = np.nanmean(delta_s_csl_surf[:,nsteps-last_steps::],axis=1)
        dphi_sh_b_surf_mean              = np.nanmean(dphi_sh_b_surf[:,nsteps-last_steps::],axis=1)
        je_b_surf_mean                   = np.nanmean(je_b_surf[:,nsteps-last_steps::],axis=1)
        gp_net_b_surf_mean               = np.nanmean(gp_net_b_surf[:,nsteps-last_steps::],axis=1)
        ge_sb_b_surf_mean                = np.nanmean(ge_sb_b_surf[:,nsteps-last_steps::],axis=1)
        relerr_je_b_surf_mean            = np.nanmean(relerr_je_b_surf[:,nsteps-last_steps::],axis=1)
        qe_tot_wall_surf_mean            = np.nanmean(qe_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        qe_tot_s_wall_surf_mean          = np.nanmean(qe_tot_s_wall_surf[:,nsteps-last_steps::],axis=1)
        qe_tot_b_surf_mean               = np.nanmean(qe_tot_b_surf[:,nsteps-last_steps::],axis=1)
        qe_b_surf_mean                   = np.nanmean(qe_b_surf[:,nsteps-last_steps::],axis=1)
        qe_b_bc_surf_mean                = np.nanmean(qe_b_bc_surf[:,nsteps-last_steps::],axis=1)
        qe_b_fl_surf_mean                = np.nanmean(qe_b_fl_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_e_wall_surf_mean         = np.nanmean(imp_ene_e_wall_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_e_b_surf_mean            = np.nanmean(imp_ene_e_b_surf[:,nsteps-last_steps::],axis=1)
        relerr_qe_b_surf_mean            = np.nanmean(relerr_qe_b_surf[:,nsteps-last_steps::],axis=1)
        relerr_qe_b_cons_surf_mean       = np.nanmean(relerr_qe_b_cons_surf[:,nsteps-last_steps::],axis=1)
        Te_surf_mean                     = np.nanmean(Te_surf[:,nsteps-last_steps::],axis=1)
        phi_surf_mean                    = np.nanmean(phi_surf[:,nsteps-last_steps::],axis=1)
        nQ1_inst_surf_mean               = np.nanmean(nQ1_inst_surf[:,nsteps-last_steps::],axis=1)
        nQ1_surf_mean                    = np.nanmean(nQ1_surf[:,nsteps-last_steps::],axis=1)
        nQ2_inst_surf_mean               = np.nanmean(nQ2_inst_surf[:,nsteps-last_steps::],axis=1)
        nQ2_surf_mean                    = np.nanmean(nQ2_surf[:,nsteps-last_steps::],axis=1)
        dphi_kbc_surf_mean               = np.nanmean(dphi_kbc_surf[:,nsteps-last_steps::],axis=1)
        MkQ1_surf_mean                   = np.nanmean(MkQ1_surf[:,nsteps-last_steps::],axis=1)
        ji1_surf_mean                    = np.nanmean(ji1_surf[:,nsteps-last_steps::],axis=1)
        ji2_surf_mean                    = np.nanmean(ji2_surf[:,nsteps-last_steps::],axis=1)
        ji3_surf_mean                    = np.nanmean(ji3_surf[:,nsteps-last_steps::],axis=1)
        ji4_surf_mean                    = np.nanmean(ji4_surf[:,nsteps-last_steps::],axis=1)
        ji_surf_mean                     = np.nanmean(ji_surf[:,nsteps-last_steps::],axis=1)
        gn1_tw_surf_mean                 = np.nanmean(gn1_tw_surf[:,nsteps-last_steps::],axis=1)
        gn1_fw_surf_mean                 = np.nanmean(gn1_fw_surf[:,nsteps-last_steps::],axis=1)
        gn2_tw_surf_mean                 = np.nanmean(gn2_tw_surf[:,nsteps-last_steps::],axis=1)
        gn2_fw_surf_mean                 = np.nanmean(gn2_fw_surf[:,nsteps-last_steps::],axis=1)
        gn3_tw_surf_mean                 = np.nanmean(gn3_tw_surf[:,nsteps-last_steps::],axis=1)
        gn3_fw_surf_mean                 = np.nanmean(gn3_fw_surf[:,nsteps-last_steps::],axis=1)
        gn_tw_surf_mean                  = np.nanmean(gn_tw_surf[:,nsteps-last_steps::],axis=1)
        qi1_tot_wall_surf_mean           = np.nanmean(qi1_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        qi2_tot_wall_surf_mean           = np.nanmean(qi2_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        qi3_tot_wall_surf_mean           = np.nanmean(qi3_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        qi4_tot_wall_surf_mean           = np.nanmean(qi4_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        qi_tot_wall_surf_mean            = np.nanmean(qi_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        qn1_tw_surf_mean                 = np.nanmean(qn1_tw_surf[:,nsteps-last_steps::],axis=1)
        qn1_fw_surf_mean                 = np.nanmean(qn1_fw_surf[:,nsteps-last_steps::],axis=1)
        qn2_tw_surf_mean                 = np.nanmean(qn2_tw_surf[:,nsteps-last_steps::],axis=1)
        qn2_fw_surf_mean                 = np.nanmean(qn2_fw_surf[:,nsteps-last_steps::],axis=1)
        qn3_tw_surf_mean                 = np.nanmean(qn3_tw_surf[:,nsteps-last_steps::],axis=1)
        qn3_fw_surf_mean                 = np.nanmean(qn3_fw_surf[:,nsteps-last_steps::],axis=1)
        qn_tot_wall_surf_mean            = np.nanmean(qn_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_i1_surf_mean             = np.nanmean(imp_ene_i1_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_i2_surf_mean             = np.nanmean(imp_ene_i2_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_i3_surf_mean             = np.nanmean(imp_ene_i3_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_i4_surf_mean             = np.nanmean(imp_ene_i4_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_ion_surf_mean            = np.nanmean(imp_ene_ion_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_ion_surf_v2_mean         = np.nanmean(imp_ene_ion_surf_v2[:,nsteps-last_steps::],axis=1)
        imp_ene_n1_surf_mean             = np.nanmean(imp_ene_n1_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_n2_surf_mean             = np.nanmean(imp_ene_n2_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_n3_surf_mean             = np.nanmean(imp_ene_n3_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_n_surf_mean              = np.nanmean(imp_ene_n_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_n_surf_v2_mean           = np.nanmean(imp_ene_n_surf_v2[:,nsteps-last_steps::],axis=1)
        inst_dphi_sh_b_Te_surf_mean      = np.nanmean(inst_dphi_sh_b_Te_surf[:,nsteps-last_steps::],axis=1)
        inst_imp_ene_e_b_surf_mean       = np.nanmean(inst_imp_ene_e_b_surf[:,nsteps-last_steps::],axis=1)
        inst_imp_ene_e_b_Te_surf_mean    = np.nanmean(inst_imp_ene_e_b_Te_surf[:,nsteps-last_steps::],axis=1)
        inst_imp_ene_e_wall_surf_mean    = np.nanmean(inst_imp_ene_e_wall_surf[:,nsteps-last_steps::],axis=1)
        inst_imp_ene_e_wall_Te_surf_mean = np.nanmean(inst_imp_ene_e_wall_Te_surf[:,nsteps-last_steps::],axis=1)
        
        
    elif mean_type == 1:
        delta_r_mean                = np.nanmean(delta_r[step_i:step_f+1,:],axis=0)
        delta_s_mean                = np.nanmean(delta_s[step_i:step_f+1,:],axis=0)
        delta_s_csl_mean            = np.nanmean(delta_s_csl[step_i:step_f+1,:],axis=0)
        dphi_sh_b_mean              = np.nanmean(dphi_sh_b[step_i:step_f+1,:],axis=0)
        je_b_mean                   = np.nanmean(je_b[step_i:step_f+1,:],axis=0)
        ji_tot_b_mean               = np.nanmean(ji_tot_b[step_i:step_f+1,:],axis=0)
        gp_net_b_mean               = np.nanmean(gp_net_b[step_i:step_f+1,:],axis=0)
        ge_sb_b_mean                = np.nanmean(ge_sb_b[step_i:step_f+1,:],axis=0)
        relerr_je_b_mean            = np.nanmean(relerr_je_b[step_i:step_f+1,:],axis=0)
        qe_tot_wall_mean            = np.nanmean(qe_tot_wall[step_i:step_f+1,:],axis=0)
        qe_tot_s_wall_mean          = np.nanmean(qe_tot_s_wall[step_i:step_f+1,:],axis=0)
        qe_tot_b_mean               = np.nanmean(qe_tot_b[step_i:step_f+1,:],axis=0)
        qe_b_mean                   = np.nanmean(qe_b[step_i:step_f+1,:],axis=0)
        qe_b_bc_mean                = np.nanmean(qe_b_bc[step_i:step_f+1,:],axis=0)
        qe_b_fl_mean                = np.nanmean(qe_b_fl[step_i:step_f+1,:],axis=0)
        imp_ene_e_wall_mean         = np.nanmean(imp_ene_e_wall[step_i:step_f+1,:],axis=0)
        imp_ene_e_b_mean            = np.nanmean(imp_ene_e_b[step_i:step_f+1,:],axis=0)
        relerr_qe_b_mean            = np.nanmean(relerr_qe_b[step_i:step_f+1,:],axis=0)
        relerr_qe_b_cons_mean       = np.nanmean(relerr_qe_b_cons[step_i:step_f+1,:],axis=0)
        Te_mean                     = np.nanmean(Te[step_i:step_f+1,:],axis=0)
        phi_mean                    = np.nanmean(phi[step_i:step_f+1,:],axis=0)
        err_interp_phi_mean         = np.nanmean(err_interp_phi[step_i:step_f+1,:],axis=0)
        err_interp_Te_mean          = np.nanmean(err_interp_Te[step_i:step_f+1,:],axis=0)
        err_interp_jeperp_mean      = np.nanmean(err_interp_jeperp[step_i:step_f+1,:],axis=0)
        err_interp_jetheta_mean     = np.nanmean(err_interp_jetheta[step_i:step_f+1,:],axis=0)
        err_interp_jepara_mean      = np.nanmean(err_interp_jepara[step_i:step_f+1,:],axis=0)
        err_interp_jez_mean         = np.nanmean(err_interp_jez[step_i:step_f+1,:],axis=0)
        err_interp_jer_mean         = np.nanmean(err_interp_jer[step_i:step_f+1,:],axis=0)
        n_inst_mean                 = np.nanmean(n_inst[step_i:step_f+1,:],axis=0)
        ni1_inst_mean               = np.nanmean(ni1_inst[step_i:step_f+1,:],axis=0)
        ni2_inst_mean               = np.nanmean(ni2_inst[step_i:step_f+1,:],axis=0)
        nn1_inst_mean               = np.nanmean(nn1_inst[step_i:step_f+1,:],axis=0)
        inst_dphi_sh_b_Te_mean      = np.nanmean(inst_dphi_sh_b_Te[step_i:step_f+1,:],axis=0)
        inst_imp_ene_e_b_mean       = np.nanmean(inst_imp_ene_e_b[step_i:step_f+1,:],axis=0)
        inst_imp_ene_e_b_Te_mean    = np.nanmean(inst_imp_ene_e_b_Te[step_i:step_f+1,:],axis=0)
        inst_imp_ene_e_wall_mean    = np.nanmean(inst_imp_ene_e_wall[step_i:step_f+1,:],axis=0)
        inst_imp_ene_e_wall_Te_mean = np.nanmean(inst_imp_ene_e_wall_Te[step_i:step_f+1,:],axis=0)
        
        
        delta_r_nodes_mean                = np.nanmean(delta_r_nodes[:,:,step_i:step_f+1],axis=2)
        delta_s_nodes_mean                = np.nanmean(delta_s_nodes[:,:,step_i:step_f+1],axis=2)
        delta_s_csl_nodes_mean            = np.nanmean(delta_s_csl_nodes[:,:,step_i:step_f+1],axis=2)
        dphi_sh_b_nodes_mean              = np.nanmean(dphi_sh_b_nodes[:,:,step_i:step_f+1],axis=2)
        je_b_nodes_mean                   = np.nanmean(je_b_nodes[:,:,step_i:step_f+1],axis=2)
        gp_net_b_nodes_mean               = np.nanmean(gp_net_b_nodes[:,:,step_i:step_f+1],axis=2)
        ge_sb_b_nodes_mean                = np.nanmean(ge_sb_b_nodes[:,:,step_i:step_f+1],axis=2)
        relerr_je_b_nodes_mean            = np.nanmean(relerr_je_b_nodes[:,:,step_i:step_f+1],axis=2)
        qe_tot_wall_nodes_mean            = np.nanmean(qe_tot_wall_nodes[:,:,step_i:step_f+1],axis=2)
        qe_tot_s_wall_nodes_mean          = np.nanmean(qe_tot_s_wall_nodes[:,:,step_i:step_f+1],axis=2)
        qe_tot_b_nodes_mean               = np.nanmean(qe_tot_b_nodes[:,:,step_i:step_f+1],axis=2)
        qe_b_nodes_mean                   = np.nanmean(qe_b_nodes[:,:,step_i:step_f+1],axis=2)
        qe_b_bc_nodes_mean                = np.nanmean(qe_b_bc_nodes[:,:,step_i:step_f+1],axis=2)
        qe_b_fl_nodes_mean                = np.nanmean(qe_b_fl_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_e_wall_nodes_mean         = np.nanmean(imp_ene_e_wall_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_e_b_nodes_mean            = np.nanmean(imp_ene_e_b_nodes[:,:,step_i:step_f+1],axis=2)
        relerr_qe_b_nodes_mean            = np.nanmean(relerr_qe_b_nodes[:,:,step_i:step_f+1],axis=2)
        relerr_qe_b_cons_nodes_mean       = np.nanmean(relerr_qe_b_cons_nodes[:,:,step_i:step_f+1],axis=2)
        Te_nodes_mean                     = np.nanmean(Te_nodes[:,:,step_i:step_f+1],axis=2)
        phi_nodes_mean                    = np.nanmean(phi_nodes[:,:,step_i:step_f+1],axis=2)
        err_interp_n_nodes_mean           = np.nanmean(err_interp_n_nodes[:,:,step_i:step_f+1],axis=2)
        n_inst_nodes_mean                 = np.nanmean(n_inst_nodes[:,:,step_i:step_f+1],axis=2)
        ni1_inst_nodes_mean               = np.nanmean(ni1_inst_nodes[:,:,step_i:step_f+1],axis=2)
        ni2_inst_nodes_mean               = np.nanmean(ni2_inst_nodes[:,:,step_i:step_f+1],axis=2)
        nn1_inst_nodes_mean               = np.nanmean(nn1_inst_nodes[:,:,step_i:step_f+1],axis=2)
        n_nodes_mean                      = np.nanmean(n_nodes[:,:,step_i:step_f+1],axis=2)
        ni1_nodes_mean                    = np.nanmean(ni1_nodes[:,:,step_i:step_f+1],axis=2)
        ni2_nodes_mean                    = np.nanmean(ni2_nodes[:,:,step_i:step_f+1],axis=2)
        nn1_nodes_mean                    = np.nanmean(nn1_nodes[:,:,step_i:step_f+1],axis=2)
        dphi_kbc_nodes_mean               = np.nanmean(dphi_kbc_nodes[:,:,step_i:step_f+1],axis=2)
        MkQ1_nodes_mean                   = np.nanmean(MkQ1_nodes[:,:,step_i:step_f+1],axis=2)
        ji1_nodes_mean                    = np.nanmean(ji1_nodes[:,:,step_i:step_f+1],axis=2)
        ji2_nodes_mean                    = np.nanmean(ji2_nodes[:,:,step_i:step_f+1],axis=2)
        ji3_nodes_mean                    = np.nanmean(ji3_nodes[:,:,step_i:step_f+1],axis=2)
        ji4_nodes_mean                    = np.nanmean(ji4_nodes[:,:,step_i:step_f+1],axis=2)
        ji_nodes_mean                     = np.nanmean(ji_nodes[:,:,step_i:step_f+1],axis=2)
        gn1_tw_nodes_mean                 = np.nanmean(gn1_tw_nodes[:,:,step_i:step_f+1],axis=2)
        gn1_fw_nodes_mean                 = np.nanmean(gn1_fw_nodes[:,:,step_i:step_f+1],axis=2)
        gn2_tw_nodes_mean                 = np.nanmean(gn2_tw_nodes[:,:,step_i:step_f+1],axis=2)
        gn2_fw_nodes_mean                 = np.nanmean(gn2_fw_nodes[:,:,step_i:step_f+1],axis=2)
        gn3_tw_nodes_mean                 = np.nanmean(gn3_tw_nodes[:,:,step_i:step_f+1],axis=2)
        gn3_fw_nodes_mean                 = np.nanmean(gn3_fw_nodes[:,:,step_i:step_f+1],axis=2)
        gn_tw_nodes_mean                  = np.nanmean(gn_tw_nodes[:,:,step_i:step_f+1],axis=2)
        qi1_tot_wall_nodes_mean           = np.nanmean(qi1_tot_wall_nodes[:,:,step_i:step_f+1],axis=2)
        qi2_tot_wall_nodes_mean           = np.nanmean(qi2_tot_wall_nodes[:,:,step_i:step_f+1],axis=2)
        qi3_tot_wall_nodes_mean           = np.nanmean(qi3_tot_wall_nodes[:,:,step_i:step_f+1],axis=2)
        qi4_tot_wall_nodes_mean           = np.nanmean(qi4_tot_wall_nodes[:,:,step_i:step_f+1],axis=2)
        qi_tot_wall_nodes_mean            = np.nanmean(qi_tot_wall_nodes[:,:,step_i:step_f+1],axis=2)
        qn1_tw_nodes_mean                 = np.nanmean(qn1_tw_nodes[:,:,step_i:step_f+1],axis=2)
        qn1_fw_nodes_mean                 = np.nanmean(qn1_fw_nodes[:,:,step_i:step_f+1],axis=2)
        qn2_tw_nodes_mean                 = np.nanmean(qn2_tw_nodes[:,:,step_i:step_f+1],axis=2)
        qn2_fw_nodes_mean                 = np.nanmean(qn2_fw_nodes[:,:,step_i:step_f+1],axis=2)
        qn3_tw_nodes_mean                 = np.nanmean(qn3_tw_nodes[:,:,step_i:step_f+1],axis=2)
        qn3_fw_nodes_mean                 = np.nanmean(qn3_fw_nodes[:,:,step_i:step_f+1],axis=2)
        qn_tot_wall_nodes_mean            = np.nanmean(qn_tot_wall_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_i1_nodes_mean             = np.nanmean(imp_ene_i1_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_i2_nodes_mean             = np.nanmean(imp_ene_i2_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_i3_nodes_mean             = np.nanmean(imp_ene_i3_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_i4_nodes_mean             = np.nanmean(imp_ene_i4_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_ion_nodes_mean            = np.nanmean(imp_ene_ion_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_ion_nodes_v2_mean         = np.nanmean(imp_ene_ion_nodes_v2[:,:,step_i:step_f+1],axis=2)
        imp_ene_n1_nodes_mean             = np.nanmean(imp_ene_n1_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_n2_nodes_mean             = np.nanmean(imp_ene_n2_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_n3_nodes_mean             = np.nanmean(imp_ene_n3_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_n_nodes_mean              = np.nanmean(imp_ene_n_nodes[:,:,step_i:step_f+1],axis=2)
        imp_ene_n_nodes_v2_mean           = np.nanmean(imp_ene_n_nodes_v2[:,:,step_i:step_f+1],axis=2)
        inst_dphi_sh_b_Te_nodes_mean      = np.nanmean(inst_dphi_sh_b_Te_nodes[:,:,step_i:step_f+1],axis=2)
        inst_imp_ene_e_b_nodes_mean       = np.nanmean(inst_imp_ene_e_b_nodes[:,:,step_i:step_f+1],axis=2)
        inst_imp_ene_e_b_Te_nodes_mean    = np.nanmean(inst_imp_ene_e_b_Te_nodes[:,:,step_i:step_f+1],axis=2)
        inst_imp_ene_e_wall_nodes_mean    = np.nanmean(inst_imp_ene_e_wall_nodes[:,:,step_i:step_f+1],axis=2)
        inst_imp_ene_e_wall_Te_nodes_mean = np.nanmean(inst_imp_ene_e_wall_Te_nodes[:,:,step_i:step_f+1],axis=2)
        
        
        delta_r_surf_mean                = np.nanmean(delta_r_surf[:,step_i:step_f+1],axis=1)
        delta_s_surf_mean                = np.nanmean(delta_s_surf[:,step_i:step_f+1],axis=1)
        delta_s_csl_surf_mean            = np.nanmean(delta_s_csl_surf[:,step_i:step_f+1],axis=1)
        dphi_sh_b_surf_mean              = np.nanmean(dphi_sh_b_surf[:,step_i:step_f+1],axis=1)
        je_b_surf_mean                   = np.nanmean(je_b_surf[:,step_i:step_f+1],axis=1)
        gp_net_b_surf_mean               = np.nanmean(gp_net_b_surf[:,step_i:step_f+1],axis=1)
        ge_sb_b_surf_mean                = np.nanmean(ge_sb_b_surf[:,step_i:step_f+1],axis=1)
        relerr_je_b_surf_mean            = np.nanmean(relerr_je_b_surf[:,step_i:step_f+1],axis=1)
        qe_tot_wall_surf_mean            = np.nanmean(qe_tot_wall_surf[:,step_i:step_f+1],axis=1)
        qe_tot_s_wall_surf_mean          = np.nanmean(qe_tot_s_wall_surf[:,step_i:step_f+1],axis=1)
        qe_tot_b_surf_mean               = np.nanmean(qe_tot_b_surf[:,step_i:step_f+1],axis=1)
        qe_b_surf_mean                   = np.nanmean(qe_b_surf[:,step_i:step_f+1],axis=1)
        qe_b_bc_surf_mean                = np.nanmean(qe_b_bc_surf[:,step_i:step_f+1],axis=1)
        qe_b_fl_surf_mean                = np.nanmean(qe_b_fl_surf[:,step_i:step_f+1],axis=1)
        imp_ene_e_wall_surf_mean         = np.nanmean(imp_ene_e_wall_surf[:,step_i:step_f+1],axis=1)
        imp_ene_e_b_surf_mean            = np.nanmean(imp_ene_e_b_surf[:,step_i:step_f+1],axis=1)
        relerr_qe_b_surf_mean            = np.nanmean(relerr_qe_b_surf[:,step_i:step_f+1],axis=1)
        relerr_qe_b_cons_surf_mean       = np.nanmean(relerr_qe_b_cons_surf[:,step_i:step_f+1],axis=1)
        Te_surf_mean                     = np.nanmean(Te_surf[:,step_i:step_f+1],axis=1)
        phi_surf_mean                    = np.nanmean(phi_surf[:,step_i:step_f+1],axis=1)
        nQ1_inst_surf_mean               = np.nanmean(nQ1_inst_surf[:,step_i:step_f+1],axis=1)
        nQ1_surf_mean                    = np.nanmean(nQ1_surf[:,step_i:step_f+1],axis=1)
        nQ2_inst_surf_mean               = np.nanmean(nQ2_inst_surf[:,step_i:step_f+1],axis=1)
        nQ2_surf_mean                    = np.nanmean(nQ2_surf[:,step_i:step_f+1],axis=1)
        dphi_kbc_surf_mean               = np.nanmean(dphi_kbc_surf[:,step_i:step_f+1],axis=1)
        MkQ1_surf_mean                   = np.nanmean(MkQ1_surf[:,step_i:step_f+1],axis=1)
        ji1_surf_mean                    = np.nanmean(ji1_surf[:,step_i:step_f+1],axis=1)
        ji2_surf_mean                    = np.nanmean(ji2_surf[:,step_i:step_f+1],axis=1)
        ji3_surf_mean                    = np.nanmean(ji3_surf[:,step_i:step_f+1],axis=1)
        ji4_surf_mean                    = np.nanmean(ji4_surf[:,step_i:step_f+1],axis=1)
        ji_surf_mean                     = np.nanmean(ji_surf[:,step_i:step_f+1],axis=1)
        gn1_tw_surf_mean                 = np.nanmean(gn1_tw_surf[:,step_i:step_f+1],axis=1)
        gn1_fw_surf_mean                 = np.nanmean(gn1_fw_surf[:,step_i:step_f+1],axis=1)
        gn2_tw_surf_mean                 = np.nanmean(gn2_tw_surf[:,step_i:step_f+1],axis=1)
        gn2_fw_surf_mean                 = np.nanmean(gn2_fw_surf[:,step_i:step_f+1],axis=1)
        gn3_tw_surf_mean                 = np.nanmean(gn3_tw_surf[:,step_i:step_f+1],axis=1)
        gn3_fw_surf_mean                 = np.nanmean(gn3_fw_surf[:,step_i:step_f+1],axis=1)
        gn_tw_surf_mean                  = np.nanmean(gn_tw_surf[:,step_i:step_f+1],axis=1)
        qi1_tot_wall_surf_mean           = np.nanmean(qi1_tot_wall_surf[:,step_i:step_f+1],axis=1)
        qi2_tot_wall_surf_mean           = np.nanmean(qi2_tot_wall_surf[:,step_i:step_f+1],axis=1)
        qi3_tot_wall_surf_mean           = np.nanmean(qi3_tot_wall_surf[:,step_i:step_f+1],axis=1)
        qi4_tot_wall_surf_mean           = np.nanmean(qi4_tot_wall_surf[:,step_i:step_f+1],axis=1)
        qi_tot_wall_surf_mean            = np.nanmean(qi_tot_wall_surf[:,step_i:step_f+1],axis=1)
        qn1_tw_surf_mean                 = np.nanmean(qn1_tw_surf[:,step_i:step_f+1],axis=1)
        qn1_fw_surf_mean                 = np.nanmean(qn1_fw_surf[:,step_i:step_f+1],axis=1)
        qn2_tw_surf_mean                 = np.nanmean(qn2_tw_surf[:,step_i:step_f+1],axis=1)
        qn2_fw_surf_mean                 = np.nanmean(qn2_fw_surf[:,step_i:step_f+1],axis=1)
        qn3_tw_surf_mean                 = np.nanmean(qn3_tw_surf[:,step_i:step_f+1],axis=1)
        qn3_fw_surf_mean                 = np.nanmean(qn3_fw_surf[:,step_i:step_f+1],axis=1)
        qn_tot_wall_surf_mean            = np.nanmean(qn_tot_wall_surf[:,step_i:step_f+1],axis=1)
        imp_ene_i1_surf_mean             = np.nanmean(imp_ene_i1_surf[:,step_i:step_f+1],axis=1)
        imp_ene_i2_surf_mean             = np.nanmean(imp_ene_i2_surf[:,step_i:step_f+1],axis=1)
        imp_ene_i3_surf_mean             = np.nanmean(imp_ene_i3_surf[:,step_i:step_f+1],axis=1)
        imp_ene_i4_surf_mean             = np.nanmean(imp_ene_i4_surf[:,step_i:step_f+1],axis=1)
        imp_ene_ion_surf_mean            = np.nanmean(imp_ene_ion_surf[:,step_i:step_f+1],axis=1)
        imp_ene_ion_surf_v2_mean         = np.nanmean(imp_ene_ion_surf_v2[:,step_i:step_f+1],axis=1)
        imp_ene_n1_surf_mean             = np.nanmean(imp_ene_n1_surf[:,step_i:step_f+1],axis=1)
        imp_ene_n2_surf_mean             = np.nanmean(imp_ene_n2_surf[:,step_i:step_f+1],axis=1)
        imp_ene_n3_surf_mean             = np.nanmean(imp_ene_n3_surf[:,step_i:step_f+1],axis=1)
        imp_ene_n_surf_mean              = np.nanmean(imp_ene_n_surf[:,step_i:step_f+1],axis=1)
        imp_ene_n_surf_v2_mean           = np.nanmean(imp_ene_n_surf_v2[:,step_i:step_f+1],axis=1)
        inst_dphi_sh_b_Te_surf_mean      = np.nanmean(inst_dphi_sh_b_Te_surf[:,step_i:step_f+1],axis=1)
        inst_imp_ene_e_b_surf_mean       = np.nanmean(inst_imp_ene_e_b_surf[:,step_i:step_f+1],axis=1)
        inst_imp_ene_e_b_Te_surf_mean    = np.nanmean(inst_imp_ene_e_b_Te_surf[:,step_i:step_f+1],axis=1)
        inst_imp_ene_e_wall_surf_mean    = np.nanmean(inst_imp_ene_e_wall_surf[:,step_i:step_f+1],axis=1)
        inst_imp_ene_e_wall_Te_surf_mean = np.nanmean(inst_imp_ene_e_wall_Te_surf[:,step_i:step_f+1],axis=1)
    
    
    return[delta_r_mean,delta_s_mean,delta_s_csl_mean,dphi_sh_b_mean,je_b_mean,
           ji_tot_b_mean,gp_net_b_mean,ge_sb_b_mean,relerr_je_b_mean,qe_tot_wall_mean,
           qe_tot_s_wall_mean,qe_tot_b_mean,qe_b_mean,qe_b_bc_mean,qe_b_fl_mean,
           imp_ene_e_wall_mean,imp_ene_e_b_mean,
           relerr_qe_b_mean,relerr_qe_b_cons_mean,Te_mean,phi_mean,
           err_interp_phi_mean,err_interp_Te_mean,err_interp_jeperp_mean,
           err_interp_jetheta_mean,err_interp_jepara_mean,err_interp_jez_mean,
           err_interp_jer_mean,n_inst_mean,ni1_inst_mean,ni2_inst_mean,nn1_inst_mean,
           inst_dphi_sh_b_Te_mean,inst_imp_ene_e_b_mean,inst_imp_ene_e_b_Te_mean,
           inst_imp_ene_e_wall_mean,inst_imp_ene_e_wall_Te_mean,
           
           delta_r_nodes_mean,delta_s_nodes_mean,delta_s_csl_nodes_mean,
           dphi_sh_b_nodes_mean,je_b_nodes_mean,gp_net_b_nodes_mean,
           ge_sb_b_nodes_mean,relerr_je_b_nodes_mean,qe_tot_wall_nodes_mean,
           qe_tot_s_wall_nodes_mean,qe_tot_b_nodes_mean,qe_b_nodes_mean,
           qe_b_bc_nodes_mean,qe_b_fl_nodes_mean,relerr_qe_b_nodes_mean,
           imp_ene_e_wall_nodes_mean,imp_ene_e_b_nodes_mean,
           relerr_qe_b_cons_nodes_mean,Te_nodes_mean,phi_nodes_mean,
           err_interp_n_nodes_mean,n_inst_nodes_mean,ni1_inst_nodes_mean,
           ni2_inst_nodes_mean,nn1_inst_nodes_mean,n_nodes_mean,ni1_nodes_mean,
           ni2_nodes_mean,nn1_nodes_mean,dphi_kbc_nodes_mean,MkQ1_nodes_mean,
           ji1_nodes_mean,ji2_nodes_mean,ji3_nodes_mean,ji4_nodes_mean,
           ji_nodes_mean,gn1_tw_nodes_mean,gn1_fw_nodes_mean,gn2_tw_nodes_mean,
           gn2_fw_nodes_mean,gn3_tw_nodes_mean,gn3_fw_nodes_mean,gn_tw_nodes_mean,
           qi1_tot_wall_nodes_mean,qi2_tot_wall_nodes_mean,qi3_tot_wall_nodes_mean,
           qi4_tot_wall_nodes_mean,qi_tot_wall_nodes_mean,qn1_tw_nodes_mean,
           qn1_fw_nodes_mean,qn2_tw_nodes_mean,qn2_fw_nodes_mean,qn3_tw_nodes_mean,
           qn3_fw_nodes_mean,qn_tot_wall_nodes_mean,imp_ene_i1_nodes_mean,
           imp_ene_i2_nodes_mean,imp_ene_i3_nodes_mean,imp_ene_i4_nodes_mean,
           imp_ene_ion_nodes_mean,imp_ene_ion_nodes_v2_mean,
           imp_ene_n1_nodes_mean,imp_ene_n2_nodes_mean,imp_ene_n3_nodes_mean,
           imp_ene_n_nodes_mean,imp_ene_n_nodes_v2_mean,inst_dphi_sh_b_Te_nodes_mean, 
           inst_imp_ene_e_b_nodes_mean,inst_imp_ene_e_b_Te_nodes_mean,
           inst_imp_ene_e_wall_nodes_mean,inst_imp_ene_e_wall_Te_nodes_mean,
           
           delta_r_surf_mean,delta_s_surf_mean,delta_s_csl_surf_mean,
           dphi_sh_b_surf_mean,je_b_surf_mean,gp_net_b_surf_mean,ge_sb_b_surf_mean,
           relerr_je_b_surf_mean,qe_tot_wall_surf_mean,qe_tot_s_wall_surf_mean,
           qe_tot_b_surf_mean,qe_b_surf_mean,qe_b_bc_surf_mean,qe_b_fl_surf_mean,
           imp_ene_e_wall_surf_mean,imp_ene_e_b_surf_mean,
           relerr_qe_b_surf_mean,relerr_qe_b_cons_surf_mean,Te_surf_mean,
           phi_surf_mean,nQ1_inst_surf_mean,nQ1_surf_mean,nQ2_inst_surf_mean,
           nQ2_surf_mean,dphi_kbc_surf_mean,MkQ1_surf_mean,ji1_surf_mean,
           ji2_surf_mean,ji3_surf_mean,ji4_surf_mean,ji_surf_mean,gn1_tw_surf_mean,
           gn1_fw_surf_mean,gn2_tw_surf_mean,gn2_fw_surf_mean,gn3_tw_surf_mean,
           gn3_fw_surf_mean,gn_tw_surf_mean,qi1_tot_wall_surf_mean,
           qi2_tot_wall_surf_mean,qi3_tot_wall_surf_mean,qi4_tot_wall_surf_mean,
           qi_tot_wall_surf_mean,qn1_tw_surf_mean,qn1_fw_surf_mean,qn2_tw_surf_mean,
           qn2_fw_surf_mean,qn3_tw_surf_mean,qn3_fw_surf_mean,qn_tot_wall_surf_mean,
           imp_ene_i1_surf_mean,imp_ene_i2_surf_mean,imp_ene_i3_surf_mean,
           imp_ene_i4_surf_mean,imp_ene_ion_surf_mean,imp_ene_ion_surf_v2_mean,
           imp_ene_n1_surf_mean,imp_ene_n2_surf_mean,imp_ene_n3_surf_mean,
           imp_ene_n_surf_mean,imp_ene_n_surf_v2_mean,inst_dphi_sh_b_Te_surf_mean,
           inst_imp_ene_e_b_surf_mean,inst_imp_ene_e_b_Te_surf_mean,
           inst_imp_ene_e_wall_surf_mean,inst_imp_ene_e_wall_Te_surf_mean]
           
           
if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 
    from HET_sims_read_bound import HET_sims_read_bound

    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))
    
    # Parameters
    e  = 1.6021766E-19
    
    
    sim_name = "../../../Sr_sims_files/SPT100_orig_tmtetq2_Vd300_test_rel"

    timestep         = 65
    allsteps_flag    = 1
    
    oldpost_sim      = 3
    oldsimparams_sim = 8
    
    mean_type  = 0
    last_steps = 670
    step_i = 200
    step_f = 300
    
    
    
    path_simstate_inp  = sim_name+"/CORE/inp/SimState.hdf5"
    path_simstate_out  = sim_name+"/CORE/out/SimState.hdf5"
    path_postdata_out  = sim_name+"/CORE/out/PostData.hdf5"
    path_simparams_inp = sim_name+"/CORE/inp/sim_params.inp"
    path_picM          = sim_name+"/SET/inp/SPT100_picM.hdf5"
    
    [points,zs,rs,zscells,rscells,dims,nodes_flag,cells_flag,cells_vol,
       volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,eta_max,eta_min,xi_top,
       xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,nsteps,
       nsteps_fast,nsteps_eFld,faces,nodes,elem_n,boundary_f,face_geom,
       elem_geom,n_faces,n_elems,n_faces_boundary,
       nfaces_Dwall_bot,nfaces_Dwall_top,nfaces_Awall,nfaces_FLwall_ver,
       nfaces_FLwall_lat,nfaces_Axis,bIDfaces_Dwall_bot,bIDfaces_Dwall_top,
       bIDfaces_Awall,bIDfaces_FLwall_ver,bIDfaces_FLwall_lat,
       bIDfaces_Axis,IDfaces_Dwall_bot,IDfaces_Dwall_top,
       IDfaces_Awall,IDfaces_FLwall_ver,IDfaces_FLwall_lat,IDfaces_Axis,
       zfaces_Dwall_bot,rfaces_Dwall_bot,zfaces_Dwall_top,rfaces_Dwall_top,
       Afaces_Dwall_bot,Afaces_Dwall_top,zfaces_Awall,rfaces_Awall,
       Afaces_Awall,zfaces_FLwall_ver,rfaces_FLwall_ver,zfaces_FLwall_lat,
       rfaces_FLwall_lat,Afaces_FLwall_ver,Afaces_FLwall_lat,zfaces_Axis,
       rfaces_Axis,Afaces_Axis,sDwall_bot,sDwall_top,sAwall,sFLwall_ver,
       sFLwall_lat,sAxis,sc_bot,sc_top,
       
       nnodes_Dwall_bot,nnodes_Dwall_top,nnodes_Awall,nnodes_FLwall_ver,
       nnodes_FLwall_lat,nnodes_Axis,nnodes_bound,inodes_Dwall_bot,
       jnodes_Dwall_bot,inodes_Dwall_top,jnodes_Dwall_top,inodes_Awall,
       jnodes_Awall,inodes_FLwall_ver,jnodes_FLwall_ver,inodes_FLwall_lat,
       jnodes_FLwall_lat,inodes_Axis,jnodes_Axis,sDwall_bot_nodes,
       sDwall_top_nodes,sAwall_nodes,sFLwall_ver_nodes,sFLwall_lat_nodes,
       sAxis_nodes,
       
       nsurf_Dwall_bot,nsurf_Dwall_top,nsurf_Awall,nsurf_FLwall_ver,
       nsurf_FLwall_lat,nsurf_bound,indsurf_Dwall_bot,zsurf_Dwall_bot,
       rsurf_Dwall_bot,indsurf_Dwall_top,zsurf_Dwall_top,rsurf_Dwall_top,
       indsurf_Awall,zsurf_Awall,rsurf_Awall,indsurf_FLwall_ver,
       zsurf_FLwall_ver,rsurf_FLwall_ver,indsurf_FLwall_lat,
       zsurf_FLwall_lat,rsurf_FLwall_lat,
       
       delta_r,delta_s,dphi_sh_b,je_b,ji_tot_b,gp_net_b,ge_sb_b,relerr_je_b,
       qe_tot_wall,qe_tot_s_wall,qe_tot_b,qe_b,qe_b_bc,qe_b_fl,relerr_qe_b,
       relerr_qe_b_cons,Te,phi,err_interp_phi,err_interp_Te,
       err_interp_jeperp,err_interp_jetheta,err_interp_jepara,
       err_interp_jez,err_interp_jer,n_inst,ni1_inst,ni2_inst,nn1_inst,
       
       delta_r_nodes,delta_s_nodes,dphi_sh_b_nodes,je_b_nodes,
       gp_net_b_nodes,ge_sb_b_nodes,relerr_je_b_nodes,qe_tot_wall_nodes,
       qe_tot_s_wall_nodes,qe_tot_b_nodes,qe_b_nodes,qe_b_bc_nodes,
       qe_b_fl_nodes,relerr_qe_b_nodes,relerr_qe_b_cons_nodes,Te_nodes,
       phi_nodes,err_interp_n_nodes,n_inst_nodes,ni1_inst_nodes,ni2_inst_nodes,
       nn1_inst_nodes,n_nodes,ni1_nodes,ni2_nodes,nn1_nodes,dphi_kbc_nodes,
       MkQ1_nodes,ji1_nodes,ji2_nodes,ji_nodes,gn1_tw_nodes,gn1_fw_nodes,
       qi1_tot_wall_nodes,qi2_tot_wall_nodes,qi_tot_wall_nodes,qn1_tw_nodes,
       qn1_fw_nodes,imp_ene_i1_nodes,imp_ene_i2_nodes,imp_ene_n1_nodes,
       
       delta_r_surf,delta_s_surf,dphi_sh_b_surf,je_b_surf,gp_net_b_surf,
       ge_sb_b_surf,relerr_je_b_surf,qe_tot_wall_surf,qe_tot_s_wall_surf,
       qe_tot_b_surf,qe_b_surf,qe_b_bc_surf,qe_b_fl_surf,relerr_qe_b_surf,
       relerr_qe_b_cons_surf,Te_surf,phi_surf,nQ1_inst_surf,nQ1_surf,
       nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,MkQ1_surf,ji1_surf,ji2_surf,
       ji_surf,gn1_tw_surf,gn1_fw_surf,qi1_tot_wall_surf,qi2_tot_wall_surf,
       qi_tot_wall_surf,qn1_tw_surf,qn1_fw_surf,imp_ene_i1_surf,
       imp_ene_i2_surf,imp_ene_n1_surf] = HET_sims_read_bound(path_simstate_inp,path_simstate_out,path_postdata_out,path_simparams_inp,
                                                              path_picM,allsteps_flag,timestep,oldpost_sim,oldsimparams_sim)
    
    
    if (n_faces_boundary == nfaces_Dwall_bot+nfaces_Dwall_top+nfaces_Awall+nfaces_FLwall_ver+nfaces_FLwall_lat+nfaces_Axis):
        print("Boundary faces read correctly")
        print("n_boundary_faces  = "+str(n_faces_boundary))
        print("nfaces_Dwall_bot  = "+str(nfaces_Dwall_bot))
        print("nfaces_Dwall_top  = "+str(nfaces_Dwall_top))
        print("nfaces_Awall      = "+str(nfaces_Awall))
        print("nfaces_FLwall_ver = "+str(nfaces_FLwall_ver))
        print("nfaces_FLwall_lat = "+str(nfaces_FLwall_lat))
        print("nfaces_Axis       = "+str(nfaces_Axis))
        
        
    [delta_r_mean,delta_s_mean,dphi_sh_b_mean,je_b_mean,ji_tot_b_mean,
       gp_net_b_mean,ge_sb_b_mean,relerr_je_b_mean,qe_tot_wall_mean,
       qe_tot_s_wall_mean,qe_tot_b_mean,qe_b_mean,qe_b_bc_mean,qe_b_fl_mean,
       relerr_qe_b_mean,relerr_qe_b_cons_mean,Te_mean,phi_mean,
       err_interp_phi_mean,err_interp_Te_mean,err_interp_jeperp_mean,
       err_interp_jetheta_mean,err_interp_jepara_mean,err_interp_jez_mean,
       err_interp_jer_mean,n_inst_mean,ni1_inst_mean,ni2_inst_mean,nn1_inst_mean,
       
       delta_r_nodes_mean,delta_s_nodes_mean,dphi_sh_b_nodes_mean,
       je_b_nodes_mean,gp_net_b_nodes_mean,ge_sb_b_nodes_mean,
       relerr_je_b_nodes_mean,qe_tot_wall_nodes_mean,
       qe_tot_s_wall_nodes_mean,qe_tot_b_nodes_mean,qe_b_nodes_mean,
       qe_b_bc_nodes_mean,qe_b_fl_nodes_mean,relerr_qe_b_nodes_mean,
       relerr_qe_b_cons_nodes_mean,Te_nodes_mean,phi_nodes_mean,
       err_interp_n_nodes_mean,n_inst_nodes_mean,ni1_inst_nodes_mean,
       ni2_inst_nodes_mean,nn1_inst_nodes_mean,n_nodes_mean,ni1_nodes_mean,
       ni2_nodes_mean,nn1_nodes_mean,dphi_kbc_nodes_mean,MkQ1_nodes_mean,
       ji1_nodes_mean,ji2_nodes_mean,ji_nodes_mean,gn1_tw_nodes_mean,
       gn1_fw_nodes_mean,qi1_tot_wall_nodes_mean,qi2_tot_wall_nodes_mean,
       qi_tot_wall_nodes_mean,qn1_tw_nodes_mean,qn1_fw_nodes_mean,
       imp_ene_i1_nodes_mean,imp_ene_i2_nodes_mean,imp_ene_n1_nodes_mean,
       
       delta_r_surf_mean,delta_s_surf_mean,dphi_sh_b_surf_mean,
       je_b_surf_mean,gp_net_b_surf_mean,ge_sb_b_surf_mean,
       relerr_je_b_surf_mean,qe_tot_wall_surf_mean,qe_tot_s_wall_surf_mean,
       qe_tot_b_surf_mean,qe_b_surf_mean,qe_b_bc_surf_mean,qe_b_fl_surf_mean,
       relerr_qe_b_surf_mean,relerr_qe_b_cons_surf_mean,Te_surf_mean,
       phi_surf_mean,nQ1_inst_surf_mean,nQ1_surf_mean,nQ2_inst_surf_mean,
       nQ2_surf_mean,dphi_kbc_surf_mean,MkQ1_surf_mean,ji1_surf_mean,
       ji2_surf_mean,ji_surf_mean,gn1_tw_surf_mean,gn1_fw_surf_mean,
       qi1_tot_wall_surf_mean,qi2_tot_wall_surf_mean,qi_tot_wall_surf_mean,
       qn1_tw_surf_mean,qn1_fw_surf_mean,imp_ene_i1_surf_mean,
       imp_ene_i2_surf_mean,imp_ene_n1_surf_mean] = HET_sims_mean_bound(nsteps,mean_type,last_steps,step_i,step_f,delta_r,
                                                                        delta_s,dphi_sh_b,je_b,ji_tot_b,gp_net_b,ge_sb_b,
                                                                        relerr_je_b,qe_tot_wall,qe_tot_s_wall,qe_tot_b,qe_b,
                                                                        qe_b_bc,qe_b_fl,relerr_qe_b,relerr_qe_b_cons,Te,phi,
                                                                        err_interp_phi,err_interp_Te,err_interp_jeperp,
                                                                        err_interp_jetheta,err_interp_jepara,err_interp_jez,
                                                                        err_interp_jer,n_inst,ni1_inst,ni2_inst,nn1_inst,
                                                                        
                                                                        delta_r_nodes,delta_s_nodes,dphi_sh_b_nodes,je_b_nodes,
                                                                        gp_net_b_nodes,ge_sb_b_nodes,relerr_je_b_nodes,qe_tot_wall_nodes,
                                                                        qe_tot_s_wall_nodes,qe_tot_b_nodes,qe_b_nodes,qe_b_bc_nodes,
                                                                        qe_b_fl_nodes,relerr_qe_b_nodes,relerr_qe_b_cons_nodes,Te_nodes,
                                                                        phi_nodes,err_interp_n_nodes,n_inst_nodes,ni1_inst_nodes,ni2_inst_nodes,
                                                                        nn1_inst_nodes,n_nodes,ni1_nodes,ni2_nodes,nn1_nodes,dphi_kbc_nodes,
                                                                        MkQ1_nodes,ji1_nodes,ji2_nodes,ji_nodes,gn1_tw_nodes,gn1_fw_nodes,
                                                                        qi1_tot_wall_nodes,qi2_tot_wall_nodes,qi_tot_wall_nodes,qn1_tw_nodes,
                                                                        qn1_fw_nodes,imp_ene_i1_nodes,imp_ene_i2_nodes,imp_ene_n1_nodes,
                                                                        
                                                                        delta_r_surf,delta_s_surf,dphi_sh_b_surf,je_b_surf,gp_net_b_surf,
                                                                        ge_sb_b_surf,relerr_je_b_surf,qe_tot_wall_surf,qe_tot_s_wall_surf,
                                                                        qe_tot_b_surf,qe_b_surf,qe_b_bc_surf,qe_b_fl_surf,relerr_qe_b_surf,
                                                                        relerr_qe_b_cons_surf,Te_surf,phi_surf,nQ1_inst_surf,nQ1_surf,
                                                                        nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,MkQ1_surf,ji1_surf,ji2_surf,
                                                                        ji_surf,gn1_tw_surf,gn1_fw_surf,qi1_tot_wall_surf,qi2_tot_wall_surf,
                                                                        qi_tot_wall_surf,qn1_tw_surf,qn1_fw_surf,imp_ene_i1_surf,
                                                                        imp_ene_i2_surf,imp_ene_n1_surf)
                                                                                                                    
    
    print("Nodes:")
    print(delta_r_nodes_mean[inodes_Dwall_bot,jnodes_Dwall_bot])
    print("Surface elements:")
    print(delta_r_surf_mean[indsurf_Dwall_bot])
    
