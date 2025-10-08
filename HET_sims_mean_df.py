#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:54:50 2020

@author: adrian

############################################################################
Description:    This python script performs the time-average of the given data
                distribution functions data from HET sims
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

def HET_sims_mean_df(nsteps,mean_type,last_steps,step_i,step_f,
                     nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,
                     MkQ1_surf,ji1_surf,ji2_surf,ji3_surf,ji4_surf,ji_surf,
                     gn1_tw_surf,gn1_fw_surf,gn2_tw_surf,gn2_fw_surf,gn3_tw_surf,
                     gn3_fw_surf,gn_tw_surf,
                     qi1_tot_wall_surf,qi2_tot_wall_surf,qi3_tot_wall_surf,
                     qi4_tot_wall_surf,qi_tot_wall_surf,qn1_tw_surf,qn1_fw_surf,
                     qn2_tw_surf,qn2_fw_surf,qn3_tw_surf,qn3_fw_surf,
                     qn_tot_wall_surf,imp_ene_i1_surf,imp_ene_i2_surf,
                     imp_ene_i3_surf,imp_ene_i4_surf,imp_ene_n1_surf,
                     imp_ene_n2_surf,imp_ene_n3_surf,
                     
                     angle_df_i1,ene_df_i1,normv_df_i1,ene_angle_df_i1,
                     angle_df_i2,ene_df_i2,normv_df_i2,ene_angle_df_i2,
                     angle_df_i3,ene_df_i3,normv_df_i3,ene_angle_df_i3,
                     angle_df_i4,ene_df_i4,normv_df_i4,ene_angle_df_i4,
                     angle_df_n1,ene_df_n1,normv_df_n1,ene_angle_df_n1,
                     angle_df_n2,ene_df_n2,normv_df_n2,ene_angle_df_n2,
                     angle_df_n3,ene_df_n3,normv_df_n3,ene_angle_df_n3):
    
    import numpy as np
    
    # Parameters
    e = 1.6021766E-19
    
    if mean_type == 0:
        nQ1_inst_surf_mean         = np.nanmean(nQ1_inst_surf[:,nsteps-last_steps::],axis=1)
        nQ1_surf_mean              = np.nanmean(nQ1_surf[:,nsteps-last_steps::],axis=1)
        nQ2_inst_surf_mean         = np.nanmean(nQ2_inst_surf[:,nsteps-last_steps::],axis=1)
        nQ2_surf_mean              = np.nanmean(nQ2_surf[:,nsteps-last_steps::],axis=1)
        dphi_kbc_surf_mean         = np.nanmean(dphi_kbc_surf[:,nsteps-last_steps::],axis=1)
        MkQ1_surf_mean             = np.nanmean(MkQ1_surf[:,nsteps-last_steps::],axis=1)
        ji1_surf_mean              = np.nanmean(ji1_surf[:,nsteps-last_steps::],axis=1)
        ji2_surf_mean              = np.nanmean(ji2_surf[:,nsteps-last_steps::],axis=1)
        ji3_surf_mean              = np.nanmean(ji3_surf[:,nsteps-last_steps::],axis=1)
        ji4_surf_mean              = np.nanmean(ji4_surf[:,nsteps-last_steps::],axis=1)
        ji_surf_mean               = np.nanmean(ji_surf[:,nsteps-last_steps::],axis=1)
        gn1_tw_surf_mean           = np.nanmean(gn1_tw_surf[:,nsteps-last_steps::],axis=1)
        gn1_fw_surf_mean           = np.nanmean(gn1_fw_surf[:,nsteps-last_steps::],axis=1)
        gn2_tw_surf_mean           = np.nanmean(gn2_tw_surf[:,nsteps-last_steps::],axis=1)
        gn2_fw_surf_mean           = np.nanmean(gn2_fw_surf[:,nsteps-last_steps::],axis=1)
        gn3_tw_surf_mean           = np.nanmean(gn3_tw_surf[:,nsteps-last_steps::],axis=1)
        gn3_fw_surf_mean           = np.nanmean(gn3_fw_surf[:,nsteps-last_steps::],axis=1)
        gn_tw_surf_mean            = np.nanmean(gn_tw_surf[:,nsteps-last_steps::],axis=1)
        qi1_tot_wall_surf_mean     = np.nanmean(qi1_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        qi2_tot_wall_surf_mean     = np.nanmean(qi2_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        qi3_tot_wall_surf_mean     = np.nanmean(qi3_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        qi4_tot_wall_surf_mean     = np.nanmean(qi4_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        qi_tot_wall_surf_mean      = np.nanmean(qi_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        qn1_tw_surf_mean           = np.nanmean(qn1_tw_surf[:,nsteps-last_steps::],axis=1)
        qn1_fw_surf_mean           = np.nanmean(qn1_fw_surf[:,nsteps-last_steps::],axis=1)
        qn2_tw_surf_mean           = np.nanmean(qn2_tw_surf[:,nsteps-last_steps::],axis=1)
        qn2_fw_surf_mean           = np.nanmean(qn2_fw_surf[:,nsteps-last_steps::],axis=1)
        qn3_tw_surf_mean           = np.nanmean(qn3_tw_surf[:,nsteps-last_steps::],axis=1)
        qn3_fw_surf_mean           = np.nanmean(qn3_fw_surf[:,nsteps-last_steps::],axis=1)
        qn_tot_wall_surf_mean      = np.nanmean(qn_tot_wall_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_i1_surf_mean       = np.nanmean(imp_ene_i1_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_i2_surf_mean       = np.nanmean(imp_ene_i2_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_i3_surf_mean       = np.nanmean(imp_ene_i3_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_i4_surf_mean       = np.nanmean(imp_ene_i4_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_n1_surf_mean       = np.nanmean(imp_ene_n1_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_n2_surf_mean       = np.nanmean(imp_ene_n2_surf[:,nsteps-last_steps::],axis=1)
        imp_ene_n3_surf_mean       = np.nanmean(imp_ene_n3_surf[:,nsteps-last_steps::],axis=1)
        
        angle_df_i1_mean           = np.nanmean(angle_df_i1[:,:,nsteps-last_steps::],axis=2)
        ene_df_i1_mean             = np.nanmean(ene_df_i1[:,:,nsteps-last_steps::],axis=2)
        normv_df_i1_mean           = np.nanmean(normv_df_i1[:,:,nsteps-last_steps::],axis=2)
        ene_angle_df_i1_mean       = np.nanmean(ene_angle_df_i1[:,:,:,nsteps-last_steps::],axis=3)
        angle_df_i2_mean           = np.nanmean(angle_df_i2[:,:,nsteps-last_steps::],axis=2)
        ene_df_i2_mean             = np.nanmean(ene_df_i2[:,:,nsteps-last_steps::],axis=2)
        normv_df_i2_mean           = np.nanmean(normv_df_i2[:,:,nsteps-last_steps::],axis=2)
        ene_angle_df_i2_mean       = np.nanmean(ene_angle_df_i2[:,:,:,nsteps-last_steps::],axis=3)
        angle_df_i3_mean           = np.nanmean(angle_df_i3[:,:,nsteps-last_steps::],axis=2)
        ene_df_i3_mean             = np.nanmean(ene_df_i3[:,:,nsteps-last_steps::],axis=2)
        normv_df_i3_mean           = np.nanmean(normv_df_i3[:,:,nsteps-last_steps::],axis=2)
        ene_angle_df_i3_mean       = np.nanmean(ene_angle_df_i3[:,:,:,nsteps-last_steps::],axis=3)        
        angle_df_i4_mean           = np.nanmean(angle_df_i4[:,:,nsteps-last_steps::],axis=2)
        ene_df_i4_mean             = np.nanmean(ene_df_i4[:,:,nsteps-last_steps::],axis=2)
        normv_df_i4_mean           = np.nanmean(normv_df_i4[:,:,nsteps-last_steps::],axis=2)
        ene_angle_df_i4_mean       = np.nanmean(ene_angle_df_i4[:,:,:,nsteps-last_steps::],axis=3)
        angle_df_n1_mean           = np.nanmean(angle_df_n1[:,:,nsteps-last_steps::],axis=2)
        ene_df_n1_mean             = np.nanmean(ene_df_n1[:,:,nsteps-last_steps::],axis=2)
        normv_df_n1_mean           = np.nanmean(normv_df_n1[:,:,nsteps-last_steps::],axis=2)
        ene_angle_df_n1_mean       = np.nanmean(ene_angle_df_n1[:,:,:,nsteps-last_steps::],axis=3)
        angle_df_n2_mean           = np.nanmean(angle_df_n2[:,:,nsteps-last_steps::],axis=2)
        ene_df_n2_mean             = np.nanmean(ene_df_n2[:,:,nsteps-last_steps::],axis=2)
        normv_df_n2_mean           = np.nanmean(normv_df_n2[:,:,nsteps-last_steps::],axis=2)
        ene_angle_df_n2_mean       = np.nanmean(ene_angle_df_n2[:,:,:,nsteps-last_steps::],axis=3)
        angle_df_n3_mean           = np.nanmean(angle_df_n3[:,:,nsteps-last_steps::],axis=2)
        ene_df_n3_mean             = np.nanmean(ene_df_n3[:,:,nsteps-last_steps::],axis=2)
        normv_df_n3_mean           = np.nanmean(normv_df_n3[:,:,nsteps-last_steps::],axis=2)
        ene_angle_df_n3_mean       = np.nanmean(ene_angle_df_n3[:,:,:,nsteps-last_steps::],axis=3)
        
    elif mean_type == 1:   
        nQ1_inst_surf_mean         = np.nanmean(nQ1_inst_surf[:,step_i:step_f+1],axis=1)
        nQ1_surf_mean              = np.nanmean(nQ1_surf[:,step_i:step_f+1],axis=1)
        nQ2_inst_surf_mean         = np.nanmean(nQ2_inst_surf[:,step_i:step_f+1],axis=1)
        nQ2_surf_mean              = np.nanmean(nQ2_surf[:,step_i:step_f+1],axis=1)
        dphi_kbc_surf_mean         = np.nanmean(dphi_kbc_surf[:,step_i:step_f+1],axis=1)
        MkQ1_surf_mean             = np.nanmean(MkQ1_surf[:,step_i:step_f+1],axis=1)
        ji1_surf_mean              = np.nanmean(ji1_surf[:,step_i:step_f+1],axis=1)
        ji2_surf_mean              = np.nanmean(ji2_surf[:,step_i:step_f+1],axis=1)
        ji3_surf_mean              = np.nanmean(ji3_surf[:,step_i:step_f+1],axis=1)
        ji4_surf_mean              = np.nanmean(ji4_surf[:,step_i:step_f+1],axis=1)        
        ji_surf_mean               = np.nanmean(ji_surf[:,step_i:step_f+1],axis=1)
        gn1_tw_surf_mean           = np.nanmean(gn1_tw_surf[:,step_i:step_f+1],axis=1)
        gn1_fw_surf_mean           = np.nanmean(gn1_fw_surf[:,step_i:step_f+1],axis=1)
        gn2_tw_surf_mean           = np.nanmean(gn2_tw_surf[:,step_i:step_f+1],axis=1)
        gn2_fw_surf_mean           = np.nanmean(gn2_fw_surf[:,step_i:step_f+1],axis=1)        
        gn3_tw_surf_mean           = np.nanmean(gn3_tw_surf[:,step_i:step_f+1],axis=1)
        gn3_fw_surf_mean           = np.nanmean(gn3_fw_surf[:,step_i:step_f+1],axis=1)
        gn_tw_surf_mean            = np.nanmean(gn_tw_surf[:,step_i:step_f+1],axis=1)
        qi1_tot_wall_surf_mean     = np.nanmean(qi1_tot_wall_surf[:,step_i:step_f+1],axis=1)
        qi2_tot_wall_surf_mean     = np.nanmean(qi2_tot_wall_surf[:,step_i:step_f+1],axis=1)
        qi3_tot_wall_surf_mean     = np.nanmean(qi3_tot_wall_surf[:,step_i:step_f+1],axis=1)
        qi4_tot_wall_surf_mean     = np.nanmean(qi4_tot_wall_surf[:,step_i:step_f+1],axis=1)
        qi_tot_wall_surf_mean      = np.nanmean(qi_tot_wall_surf[:,step_i:step_f+1],axis=1)
        qn1_tw_surf_mean           = np.nanmean(qn1_tw_surf[:,step_i:step_f+1],axis=1)
        qn1_fw_surf_mean           = np.nanmean(qn1_fw_surf[:,step_i:step_f+1],axis=1)
        qn2_tw_surf_mean           = np.nanmean(qn2_tw_surf[:,step_i:step_f+1],axis=1)
        qn2_fw_surf_mean           = np.nanmean(qn2_fw_surf[:,step_i:step_f+1],axis=1)
        qn3_tw_surf_mean           = np.nanmean(qn3_tw_surf[:,step_i:step_f+1],axis=1)
        qn3_fw_surf_mean           = np.nanmean(qn3_fw_surf[:,step_i:step_f+1],axis=1)
        qn_tot_wall_surf_mean      = np.nanmean(qn_tot_wall_surf[:,step_i:step_f+1],axis=1)
        imp_ene_i1_surf_mean       = np.nanmean(imp_ene_i1_surf[:,step_i:step_f+1],axis=1)
        imp_ene_i2_surf_mean       = np.nanmean(imp_ene_i2_surf[:,step_i:step_f+1],axis=1)
        imp_ene_i3_surf_mean       = np.nanmean(imp_ene_i3_surf[:,step_i:step_f+1],axis=1)
        imp_ene_i4_surf_mean       = np.nanmean(imp_ene_i4_surf[:,step_i:step_f+1],axis=1)
        imp_ene_n1_surf_mean       = np.nanmean(imp_ene_n1_surf[:,step_i:step_f+1],axis=1)
        imp_ene_n2_surf_mean       = np.nanmean(imp_ene_n2_surf[:,step_i:step_f+1],axis=1)
        imp_ene_n3_surf_mean       = np.nanmean(imp_ene_n3_surf[:,step_i:step_f+1],axis=1)
        
        angle_df_i1_mean           = np.nanmean(angle_df_i1[:,:,step_i:step_f+1],axis=2)
        ene_df_i1_mean             = np.nanmean(ene_df_i1[:,:,step_i:step_f+1],axis=2)
        normv_df_i1_mean           = np.nanmean(normv_df_i1[:,:,step_i:step_f+1],axis=2)
        ene_angle_df_i1_mean       = np.nanmean(ene_angle_df_i1[:,:,:,step_i:step_f+1],axis=3)
        angle_df_i2_mean           = np.nanmean(angle_df_i2[:,:,step_i:step_f+1],axis=2)
        ene_df_i2_mean             = np.nanmean(ene_df_i2[:,:,step_i:step_f+1],axis=2)
        normv_df_i2_mean           = np.nanmean(normv_df_i2[:,:,step_i:step_f+1],axis=2)
        ene_angle_df_i2_mean       = np.nanmean(ene_angle_df_i2[:,:,:,step_i:step_f+1],axis=3)
        angle_df_i3_mean           = np.nanmean(angle_df_i3[:,:,step_i:step_f+1],axis=2)
        ene_df_i3_mean             = np.nanmean(ene_df_i3[:,:,step_i:step_f+1],axis=2)
        normv_df_i3_mean           = np.nanmean(normv_df_i3[:,:,step_i:step_f+1],axis=2)
        ene_angle_df_i3_mean       = np.nanmean(ene_angle_df_i3[:,:,:,step_i:step_f+1],axis=3)
        angle_df_i4_mean           = np.nanmean(angle_df_i4[:,:,step_i:step_f+1],axis=2)
        ene_df_i4_mean             = np.nanmean(ene_df_i4[:,:,step_i:step_f+1],axis=2)
        normv_df_i4_mean           = np.nanmean(normv_df_i4[:,:,step_i:step_f+1],axis=2)
        ene_angle_df_i4_mean       = np.nanmean(ene_angle_df_i4[:,:,:,step_i:step_f+1],axis=3)
        angle_df_n1_mean           = np.nanmean(angle_df_n1[:,:,step_i:step_f+1],axis=2)
        ene_df_n1_mean             = np.nanmean(ene_df_n1[:,:,step_i:step_f+1],axis=2)
        normv_df_n1_mean           = np.nanmean(normv_df_n1[:,:,step_i:step_f+1],axis=2)
        ene_angle_df_n1_mean       = np.nanmean(ene_angle_df_n1[:,:,:,step_i:step_f+1],axis=3)
        angle_df_n2_mean           = np.nanmean(angle_df_n2[:,:,step_i:step_f+1],axis=2)
        ene_df_n2_mean             = np.nanmean(ene_df_n2[:,:,step_i:step_f+1],axis=2)
        normv_df_n2_mean           = np.nanmean(normv_df_n2[:,:,step_i:step_f+1],axis=2)
        ene_angle_df_n2_mean       = np.nanmean(ene_angle_df_n2[:,:,:,step_i:step_f+1],axis=3)
        angle_df_n3_mean           = np.nanmean(angle_df_n3[:,:,step_i:step_f+1],axis=2)
        ene_df_n3_mean             = np.nanmean(ene_df_n3[:,:,step_i:step_f+1],axis=2)
        normv_df_n3_mean           = np.nanmean(normv_df_n3[:,:,step_i:step_f+1],axis=2)
        ene_angle_df_n3_mean       = np.nanmean(ene_angle_df_n3[:,:,:,step_i:step_f+1],axis=3)
    
    
    
    return[nQ1_inst_surf_mean,nQ1_surf_mean,nQ2_inst_surf_mean,
           nQ2_surf_mean,dphi_kbc_surf_mean,MkQ1_surf_mean,ji1_surf_mean,
           ji2_surf_mean,ji3_surf_mean,ji4_surf_mean,ji_surf_mean,
           gn1_tw_surf_mean,gn1_fw_surf_mean,gn2_tw_surf_mean,gn2_fw_surf_mean,
           gn3_tw_surf_mean,gn3_fw_surf_mean,gn_tw_surf_mean,
           qi1_tot_wall_surf_mean,qi2_tot_wall_surf_mean,
           qi3_tot_wall_surf_mean,qi4_tot_wall_surf_mean,qi_tot_wall_surf_mean,
           qn1_tw_surf_mean,qn1_fw_surf_mean,qn2_tw_surf_mean,qn2_fw_surf_mean,
           qn3_tw_surf_mean,qn3_fw_surf_mean,qn_tot_wall_surf_mean,
           imp_ene_i1_surf_mean,imp_ene_i2_surf_mean,imp_ene_i3_surf_mean,
           imp_ene_i4_surf_mean,imp_ene_n1_surf_mean,imp_ene_n2_surf_mean,
           imp_ene_n3_surf_mean,
           
           angle_df_i1_mean,ene_df_i1_mean,normv_df_i1_mean,ene_angle_df_i1_mean,
           angle_df_i2_mean,ene_df_i2_mean,normv_df_i2_mean,ene_angle_df_i2_mean,
           angle_df_i3_mean,ene_df_i3_mean,normv_df_i3_mean,ene_angle_df_i3_mean,
           angle_df_i4_mean,ene_df_i4_mean,normv_df_i4_mean,ene_angle_df_i4_mean,
           angle_df_n1_mean,ene_df_n1_mean,normv_df_n1_mean,ene_angle_df_n1_mean,
           angle_df_n2_mean,ene_df_n2_mean,normv_df_n2_mean,ene_angle_df_n2_mean,
           angle_df_n3_mean,ene_df_n3_mean,normv_df_n3_mean,ene_angle_df_n3_mean]
           
           
if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 
    from HET_sims_read_df import HET_sims_read_df

    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))
    
    # Parameters
    e  = 1.6021766E-19
    
    
#    sim_name = "../../../Sr_sims_files/SPT100_orig_tmtetq2_Vd300_test_rel"
    sim_name = "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_Kr"
    
    sim_name = "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd300"
    
    sim_name = "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_last"

    timestep         = -1
    allsteps_flag    = 1
    
    oldpost_sim      = 3
    oldsimparams_sim = 8
    
    mean_type  = 0
    last_steps = 670
    step_i = 200
    step_f = 300
    
    
    path_picM         = sim_name+"/SET/inp/SPT100_picM.hdf5"
    path_picM         = sim_name +"/SET/inp/PIC_mesh_topo2_refined4.hdf5"
#    path_picM         = sim_name +"/SET/inp/SPT100_picM_Reference1500points_rm.hdf5"
    path_simstate_inp = sim_name+"/CORE/inp/SimState.hdf5"
    path_simstate_out = sim_name+"/CORE/out/SimState.hdf5"
    path_postdata_out = sim_name+"/CORE/out/PostData.hdf5"
    path_simparams_inp = sim_name+"/CORE/inp/sim_params.inp"
    
    [points,zs,rs,zscells,rscells,dims,nodes_flag,cells_flag,cells_vol,
       volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,eta_max,eta_min,xi_top,
       xi_bottom,time,steps,dt,nsteps,sc_bot,sc_top,
       
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
       zsurf_FLwall_lat,rsurf_FLwall_lat,sDwall_bot_surf,sDwall_top_surf,
       sAwall_surf,sFLwall_ver_surf,sFLwall_lat_surf,
       
       nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,
       MkQ1_surf,ji1_surf,ji2_surf,ji_surf,gn1_tw_surf,gn1_fw_surf,
       qi1_tot_wall_surf,qi2_tot_wall_surf,qi_tot_wall_surf,qn1_tw_surf,
       qn1_fw_surf,imp_ene_i1_surf,imp_ene_i2_surf,imp_ene_n1_surf,
       
       angle_bins_i1,ene_bins_i1,normv_bins_i1,angle_bins_i2,ene_bins_i2,
       normv_bins_i2,angle_bins_n1,ene_bins_n1,normv_bins_n1,nbins_angle,
       nbins_ene,nbins_normv,
       
       angle_df_i1,ene_df_i1,normv_df_i1,angle_df_i2,ene_df_i2,normv_df_i2,
       angle_df_n1,ene_df_n1,normv_df_n1] = HET_sims_read_df(path_simstate_inp,path_simstate_out,path_postdata_out,path_simparams_inp,
                                                             path_picM,allsteps_flag,timestep,oldpost_sim,oldsimparams_sim)

        
        
    [nQ1_inst_surf_mean,nQ1_surf_mean,nQ2_inst_surf_mean,
       nQ2_surf_mean,dphi_kbc_surf_mean,MkQ1_surf_mean,ji1_surf_mean,
       ji2_surf_mean,ji_surf_mean,gn1_tw_surf_mean,gn1_fw_surf_mean,
       qi1_tot_wall_surf_mean,qi2_tot_wall_surf_mean,qi_tot_wall_surf_mean,
       qn1_tw_surf_mean,qn1_fw_surf_mean,imp_ene_i1_surf_mean,
       imp_ene_i2_surf_mean,imp_ene_n1_surf_mean,
       
       angle_df_i1_mean,ene_df_i1_mean,normv_df_i1_mean,angle_df_i2_mean,
       ene_df_i2_mean,normv_df_i2_mean,angle_df_n1_mean,ene_df_n1_mean,
       normv_df_n1_mean] = HET_sims_mean_df(nsteps,mean_type,last_steps,step_i,step_f,
                                            nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,
                                            MkQ1_surf,ji1_surf,ji2_surf,ji_surf,gn1_tw_surf,gn1_fw_surf,
                                            qi1_tot_wall_surf,qi2_tot_wall_surf,qi_tot_wall_surf,qn1_tw_surf,
                                            qn1_fw_surf,imp_ene_i1_surf,imp_ene_i2_surf,imp_ene_n1_surf,
                                             
                                            angle_df_i1,ene_df_i1,normv_df_i1,angle_df_i2,ene_df_i2,normv_df_i2,
                                            angle_df_n1,ene_df_n1,normv_df_n1)
                                                                                                                    
    

    
