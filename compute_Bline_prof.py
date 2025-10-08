#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:06:13 2024

@author: adrian

############################################################################
Description:    This python script provides the profiles of different plasma
                variables to be plotted in HET_sims_plots.py along a provided
                magnetic line (Bline)
############################################################################
Inputs:        1) Bline: Given Bline as a list of elements or faces IDs 
               2) Bline_type: Flag to decide if the given B line corresponds
                               to elements IDs or faces IDs

############################################################################
Output: 
"""




def compute_Bline_prof(Bline,Bline_type,cath_elem,n_elems,n_faces,elem_geom,
                       face_geom,faces,nodes,versors_e,versors_f,
                       mean_type,nsteps,last_steps,step_i,step_f,
                       zs,rs,ne_picM,nn_picM,
                       phi_elems,phi_faces,ne_elems,ne_faces,Te_elems,Te_faces,
                       je_perp_elems,je_theta_elems,je_para_elems,je_z_elems,
                       je_r_elems,je_perp_faces,je_theta_faces,je_para_faces,
                       je_z_faces,je_r_faces,
                       alpha_ano_elems,alpha_ano_e_elems,alpha_ano_q_elems,
                       alpha_ine_elems,alpha_ine_q_elems,alpha_ano_faces,
                       alpha_ano_e_faces,alpha_ano_q_faces,alpha_ine_faces,
                       alpha_ine_q_faces,
                       Hall_par_elems,Hall_par_eff_elems,nu_e_tot_elems,
                       nu_e_tot_eff_elems,Hall_par_faces,Hall_par_eff_faces,
                       nu_e_tot_faces,nu_e_tot_eff_faces,nu_en_elems,
                       nu_ei1_elems,nu_ei2_elems,nu_i01_elems,nu_i02_elems,
                       nu_i12_elems,nu_ex_elems,nu_en_faces,nu_ei1_faces,
                       nu_ei2_faces,nu_i01_faces,nu_i02_faces,nu_i12_faces,
                       nu_ex_faces,
                       felec_para_elems,felec_para_faces,felec_perp_elems,
                       felec_perp_faces,felec_z_elems,felec_z_faces,
                       felec_r_elems,felec_r_faces):
    
    
    
    import numpy as np
    from scipy import interpolate
    
    
    # When the magnetic line provided in Bline is the cathode magnetic line,
    # (CML), we consider that it does not contain the first element or face, 
    # which is instead given in cath_elem
    s_Bline = np.zeros(len(Bline) + 1,dtype=float)
    z_Bline = np.zeros(len(Bline) + 1,dtype=float)
    r_Bline = np.zeros(len(Bline) + 1,dtype=float)
    B_Bline               = np.zeros(np.shape(s_Bline),dtype=float)
    Bz_Bline              = np.zeros(np.shape(s_Bline),dtype=float)
    Br_Bline              = np.zeros(np.shape(s_Bline),dtype=float)
    alpha_ano_Bline       = np.zeros(np.shape(s_Bline),dtype=float)
    phi_Bline             = np.zeros((nsteps,len(s_Bline)),dtype=float)
    ne_Bline              = np.zeros((nsteps,len(s_Bline)),dtype=float)
    ne_Bline_mfam         = np.zeros((nsteps,len(s_Bline)),dtype=float)
    nn_Bline              = np.zeros((nsteps,len(s_Bline)),dtype=float)
    Te_Bline              = np.zeros((nsteps,len(s_Bline)),dtype=float)
    ratio_Ekin_Te_Bline   = np.zeros((nsteps,len(s_Bline)),dtype=float)
    je_perp_Bline         = np.zeros((nsteps,len(s_Bline)),dtype=float)
    je_theta_Bline        = np.zeros((nsteps,len(s_Bline)),dtype=float)
    je_para_Bline         = np.zeros((nsteps,len(s_Bline)),dtype=float)
    je_z_Bline            = np.zeros((nsteps,len(s_Bline)),dtype=float)
    je_r_Bline            = np.zeros((nsteps,len(s_Bline)),dtype=float)
    Hall_par_Bline        = np.zeros((nsteps,len(s_Bline)),dtype=float)
    Hall_par_eff_Bline    = np.zeros((nsteps,len(s_Bline)),dtype=float)
    Hall_par_effect_Bline = np.zeros((nsteps,len(s_Bline)),dtype=float)
    felec_para_Bline      = np.zeros((nsteps,len(s_Bline)),dtype=float)
    felec_perp_Bline      = np.zeros((nsteps,len(s_Bline)),dtype=float)
    felec_z_Bline         = np.zeros((nsteps,len(s_Bline)),dtype=float)
    felec_r_Bline         = np.zeros((nsteps,len(s_Bline)),dtype=float)


    if Bline_type == 0:
        # Bline is composed of elements
        # Obtain geometry of Bline
        z_Bline[0] = face_geom[0,cath_elem]
        r_Bline[0] = face_geom[1,cath_elem]
        for i in range(0,len(Bline)):
            z_Bline[i+1] = elem_geom[0,Bline[i]]
            r_Bline[i+1] = elem_geom[1,Bline[i]]
            s_Bline[i+1] = s_Bline[i] + np.sqrt((z_Bline[i+1]-z_Bline[i])**2 + (r_Bline[i+1]-r_Bline[i])**2)
        
        # Obtain variables along Bline
        # Variables computed directly at the MFAM
        B_Bline[0]                   = face_geom[-1,cath_elem]*1E4
        B_Bline[1::]                 = elem_geom[-1,Bline]*1E4
        Bz_Bline[0]                  = B_Bline[0]*versors_f[2,cath_elem[0]]
        Bz_Bline[1::]                = B_Bline[1::]*versors_e[2,Bline]
        Br_Bline[0]                  = B_Bline[0]*versors_f[3,cath_elem[0]]
        Br_Bline[1::]                = B_Bline[1::]*versors_e[3,Bline]
        alpha_ano_Bline[0]           = alpha_ano_faces[cath_elem[0]]
        alpha_ano_Bline[1::]         = alpha_ano_elems[Bline]
        phi_Bline[:,0]               = phi_faces[:,cath_elem[0]]
        phi_Bline[:,1::]             = phi_elems[:,Bline]
        Te_Bline[:,0]                = Te_faces[:,cath_elem[0]]
        Te_Bline[:,1::]              = Te_elems[:,Bline]
        ne_Bline_mfam[:,0]           = ne_faces[:,cath_elem[0]]
        ne_Bline_mfam[:,1::]         = ne_elems[:,Bline]
        je_para_Bline[:,0]           = je_para_faces[:,cath_elem[0]]
        je_para_Bline[:,1::]         = je_para_elems[:,Bline]
        Hall_par_Bline[:,0]          = Hall_par_faces[:,cath_elem[0]]
        Hall_par_Bline[:,1::]        = Hall_par_elems[:,Bline]
        Hall_par_eff_Bline[:,0]      = Hall_par_eff_faces[:,cath_elem[0]]
        Hall_par_eff_Bline[:,1::]    = Hall_par_eff_elems[:,Bline]
        Hall_par_effect_Bline        = np.sqrt(Hall_par_Bline*Hall_par_eff_Bline)
        felec_para_Bline[:,0]        = felec_para_faces[:,cath_elem[0]]
        felec_para_Bline[:,1::]      = felec_para_elems[:,Bline]
        felec_perp_Bline[:,0]        = felec_perp_faces[:,cath_elem[0]]
        felec_perp_Bline[:,1::]       = felec_perp_elems[:,Bline]
        felec_z_Bline[:,0]           = felec_z_faces[:,cath_elem[0]]
        felec_z_Bline[:,1::]         = felec_z_elems[:,Bline]
        felec_r_Bline[:,0]           = felec_r_faces[:,cath_elem[0]]
        felec_r_Bline[:,1::]         = felec_r_elems[:,Bline]
        
        # Prepare data at picM for interpolation
        dims = np.shape(zs)
        npoints_r = dims[0]
        npoints_z = dims[1]
        for i_step in range(0,nsteps):
            vec_points_picM     = np.zeros((int(npoints_r*npoints_z),2),dtype='float')
            # ji_x_points         = np.zeros((int(npoints_r*npoints_z),1),dtype='float')
            # ji_y_points         = np.zeros((int(npoints_r*npoints_z),1),dtype='float')
            # ji_z_points         = np.zeros((int(npoints_r*npoints_z),1),dtype='float')   
            # ji_2D_points        = np.zeros((int(npoints_r*npoints_z),1),dtype='float')      
            ne_points           = np.zeros((int(npoints_r*npoints_z),1),dtype='float')     
            nn_points           = np.zeros((int(npoints_r*npoints_z),1),dtype='float') 
            
            ind = 0
            for i in range(0,int(npoints_r)):
                for j in range(0,int(npoints_z)):
                   vec_points_picM[ind,0]     = zs[i,j]
                   vec_points_picM[ind,1]     = rs[i,j]
                   # ji_x_points[ind,0]         = ji_x[i,j]
                   # ji_y_points[ind,0]         = ji_y[i,j]
                   # ji_z_points[ind,0]         = ji_z[i,j]
                   # ji_2D_points[ind,0]        = ji_2D[i,j]
                   ne_points[ind,0]           = ne_picM[i,j,i_step]
                   nn_points[ind,0]           = nn_picM[i,j,i_step]
                   ind = ind + 1
            
            method1 = 'linear'
            method2 = 'nearest'
             
            # ne
            step_ne_Bline  = interpolate.griddata(vec_points_picM, ne_points, (z_Bline, r_Bline), method=method1)
            step_ne_Bline2 = interpolate.griddata(vec_points_picM, ne_points, (z_Bline, r_Bline), method=method2)
            arr_inds          = np.where(np.isnan(step_ne_Bline) == True)
            step_ne_Bline[arr_inds] = step_ne_Bline2[arr_inds]
            step_ne_Bline = step_ne_Bline[:,0]
            ne_Bline[i_step,:] = step_ne_Bline
            # nn
            step_nn_Bline  = interpolate.griddata(vec_points_picM, nn_points, (z_Bline, r_Bline), method=method1)
            step_nn_Bline2 = interpolate.griddata(vec_points_picM, nn_points, (z_Bline, r_Bline), method=method2)
            arr_inds          = np.where(np.isnan(step_nn_Bline) == True)
            step_nn_Bline[arr_inds] = step_nn_Bline2[arr_inds]
            step_nn_Bline = step_nn_Bline[:,0]
            nn_Bline[i_step,:] = step_nn_Bline
        
    elif Bline_type == 1:
        #Bline is composed of faces
        # Obtain geometry of Bline
        z_Bline[0] = np.nanmean(face_geom[0,cath_elem])
        r_Bline[0] = np.nanmean(face_geom[1,cath_elem])
        for i in range(0,len(Bline)):
            z_Bline[i+1] = face_geom[0,Bline[i]]
            r_Bline[i+1] = face_geom[1,Bline[i]]
            s_Bline[i+1] = s_Bline[i] + np.sqrt((z_Bline[i+1]-z_Bline[i])**2 + (r_Bline[i+1]-r_Bline[i])**2)
        
        # Obtain variables along Bline
        # Variables computed directly at the MFAM
        B_Bline[0]                   = np.nanmean(face_geom[-1,cath_elem])*1E4
        B_Bline[1::]                 = face_geom[-1,Bline]*1E4
        Bz_Bline[0]                  = np.nanmean(face_geom[-1,cath_elem]*versors_f[2,cath_elem])*1E4
        Bz_Bline[1::]                = B_Bline[1::]*versors_f[2,Bline]
        Br_Bline[0]                  = np.nanmean(face_geom[-1,cath_elem]*versors_f[3,cath_elem])*1E4   
        Br_Bline[1::]                = B_Bline[1::]*versors_f[3,Bline]
        alpha_ano_Bline[0]           = np.nanmean(alpha_ano_faces[cath_elem])
        alpha_ano_Bline[1::]         = alpha_ano_faces[Bline]
        phi_Bline[:,0]               = np.nanmean(phi_faces[:,cath_elem],axis=1)
        phi_Bline[:,1::]             = phi_faces[:,Bline]
        Te_Bline[:,0]                = np.nanmean(Te_faces[:,cath_elem],axis=1)
        Te_Bline[:,1::]              = Te_faces[:,Bline]
        ne_Bline_mfam[:,0]           = np.nanmean(ne_faces[:,cath_elem],axis=1)
        ne_Bline_mfam[:,1::]         = ne_faces[:,Bline]
        je_para_Bline[:,0]           = np.nanmean(je_para_faces[:,cath_elem],axis=1)
        je_para_Bline[:,1::]         = je_para_faces[:,Bline]
        Hall_par_Bline[:,0]          = np.nanmean(Hall_par_faces[:,cath_elem],axis=1)
        Hall_par_Bline[:,1::]        = Hall_par_faces[:,Bline]
        Hall_par_eff_Bline[:,0]      = np.nanmean(Hall_par_eff_faces[:,cath_elem],axis=1)
        Hall_par_eff_Bline[:,1::]    = Hall_par_eff_faces[:,Bline]
        Hall_par_effect_Bline        = np.sqrt(Hall_par_Bline*Hall_par_eff_Bline)
        felec_para_Bline[:,0]        = np.nanmean(felec_para_faces[:,cath_elem],axis=1)
        felec_para_Bline[:,1::]      = felec_para_faces[:,Bline]
        felec_perp_Bline[:,0]        = np.nanmean(felec_perp_faces[:,cath_elem],axis=1)
        felec_perp_Bline[:,1::]      = felec_perp_faces[:,Bline]
        felec_z_Bline[:,0]           = np.nanmean(felec_z_faces[:,cath_elem],axis=1)
        felec_z_Bline[:,1::]         = felec_z_faces[:,Bline]
        felec_r_Bline[:,0]           = np.nanmean(felec_r_faces[:,cath_elem],axis=1)
        felec_r_Bline[:,1::]         = felec_r_faces[:,Bline]
        
        # Prepare data at picM for interpolation
        dims = np.shape(zs)
        npoints_r = dims[0]
        npoints_z = dims[1]
        for i_step in range(0,nsteps):
            vec_points_picM     = np.zeros((int(npoints_r*npoints_z),2),dtype='float')
            # ji_x_points         = np.zeros((int(npoints_r*npoints_z),1),dtype='float')
            # ji_y_points         = np.zeros((int(npoints_r*npoints_z),1),dtype='float')
            # ji_z_points         = np.zeros((int(npoints_r*npoints_z),1),dtype='float')   
            # ji_2D_points        = np.zeros((int(npoints_r*npoints_z),1),dtype='float')      
            ne_points           = np.zeros((int(npoints_r*npoints_z),1),dtype='float')     
            nn_points           = np.zeros((int(npoints_r*npoints_z),1),dtype='float') 
            
            ind = 0
            for i in range(0,int(npoints_r)):
                for j in range(0,int(npoints_z)):
                   vec_points_picM[ind,0]     = zs[i,j]
                   vec_points_picM[ind,1]     = rs[i,j]
                   # ji_x_points[ind,0]         = ji_x[i,j]
                   # ji_y_points[ind,0]         = ji_y[i,j]
                   # ji_z_points[ind,0]         = ji_z[i,j]
                   # ji_2D_points[ind,0]        = ji_2D[i,j]
                   ne_points[ind,0]           = ne_picM[i,j,i_step]
                   nn_points[ind,0]           = nn_picM[i,j,i_step]
                   ind = ind + 1
            
            method1 = 'linear'
            method2 = 'nearest'
             
            # ne
            step_ne_Bline  = interpolate.griddata(vec_points_picM, ne_points, (z_Bline, r_Bline), method=method1)
            step_ne_Bline2 = interpolate.griddata(vec_points_picM, ne_points, (z_Bline, r_Bline), method=method2)
            arr_inds          = np.where(np.isnan(step_ne_Bline) == True)
            step_ne_Bline[arr_inds] = step_ne_Bline2[arr_inds]
            step_ne_Bline = step_ne_Bline[:,0]
            ne_Bline[i_step,:] = step_ne_Bline
            # nn
            step_nn_Bline  = interpolate.griddata(vec_points_picM, nn_points, (z_Bline, r_Bline), method=method1)
            step_nn_Bline2 = interpolate.griddata(vec_points_picM, nn_points, (z_Bline, r_Bline), method=method2)
            arr_inds          = np.where(np.isnan(step_nn_Bline) == True)
            step_nn_Bline[arr_inds] = step_nn_Bline2[arr_inds]
            step_nn_Bline = step_nn_Bline[:,0]
            nn_Bline[i_step,:] = step_nn_Bline
    

    
    # Compute time-averaged vars
    if mean_type == 0:
        phi_Bline_mean     = np.nanmean(phi_Bline[nsteps-last_steps::,:],axis=0)
        Te_Bline_mean      = np.nanmean(Te_Bline[nsteps-last_steps::,:],axis=0)
        ne_Bline_mean      = np.nanmean(ne_Bline[nsteps-last_steps::,:],axis=0)
        ne_Bline_mfam_mean         = np.nanmean(ne_Bline_mfam[nsteps-last_steps::,:],axis=0)
        nn_Bline_mean              = np.nanmean(nn_Bline[nsteps-last_steps::,:],axis=0)
        je_para_Bline_mean         = np.nanmean(je_para_Bline[nsteps-last_steps::,:],axis=0)
        Hall_par_Bline_mean        = np.nanmean(Hall_par_Bline[nsteps-last_steps::,:],axis=0)
        Hall_par_eff_Bline_mean    = np.nanmean(Hall_par_eff_Bline[nsteps-last_steps::,:],axis=0)
        Hall_par_effect_Bline_mean = np.nanmean(Hall_par_effect_Bline[nsteps-last_steps::,:],axis=0)
        felec_para_Bline_mean      = np.nanmean(felec_para_Bline[nsteps-last_steps::,:],axis=0)
        felec_perp_Bline_mean      = np.nanmean(felec_perp_Bline[nsteps-last_steps::,:],axis=0)
        felec_z_Bline_mean         = np.nanmean(felec_z_Bline[nsteps-last_steps::,:],axis=0)
        felec_r_Bline_mean         = np.nanmean(felec_r_Bline[nsteps-last_steps::,:],axis=0)
              
    elif mean_type == 1:
        phi_Bline_mean     = np.nanmean(phi_Bline[step_i:step_f+1,:],axis=0)
        Te_Bline_mean      = np.nanmean(Te_Bline[step_i:step_f+1,:],axis=0)
        ne_Bline_mean      = np.nanmean(ne_Bline[step_i:step_f+1,:],axis=0)
        ne_Bline_mfam_mean = np.nanmean(ne_Bline_mfam[step_i:step_f+1,:],axis=0)
        nn_Bline_mean      = np.nanmean(nn_Bline[step_i:step_f+1,:],axis=0)
        je_para_Bline_mean = np.nanmean(je_para_Bline[step_i:step_f+1,:],axis=0)
        Hall_par_Bline_mean        = np.nanmean(Hall_par_Bline[step_i:step_f+1,:],axis=0)
        Hall_par_eff_Bline_mean    = np.nanmean(Hall_par_eff_Bline[step_i:step_f+1,:],axis=0)
        Hall_par_effect_Bline_mean = np.nanmean(Hall_par_effect_Bline[step_i:step_f+1,:],axis=0)
        felec_para_Bline_mean      = np.nanmean(felec_para_Bline[step_i:step_f+1,:],axis=0)
        felec_perp_Bline_mean      = np.nanmean(felec_perp_Bline[step_i:step_f+1,:],axis=0)
        felec_z_Bline_mean         = np.nanmean(felec_z_Bline[step_i:step_f+1,:],axis=0)
        felec_r_Bline_mean         = np.nanmean(felec_r_Bline[step_i:step_f+1,:],axis=0)
    
    return [s_Bline,z_Bline,r_Bline,
            B_Bline,Bz_Bline,Br_Bline,alpha_ano_Bline,phi_Bline,ne_Bline,
            ne_Bline_mfam,nn_Bline,Te_Bline,ratio_Ekin_Te_Bline,je_perp_Bline,
            je_theta_Bline,je_para_Bline,je_z_Bline,je_r_Bline,Hall_par_Bline,
            Hall_par_eff_Bline,Hall_par_effect_Bline,felec_para_Bline,
            felec_perp_Bline,felec_z_Bline,felec_r_Bline,
            
            phi_Bline_mean,Te_Bline_mean,ne_Bline_mean,ne_Bline_mfam_mean,
            nn_Bline_mean,je_para_Bline_mean,Hall_par_Bline_mean,
            Hall_par_eff_Bline_mean,Hall_par_effect_Bline_mean,
            felec_para_Bline_mean,felec_perp_Bline_mean,felec_z_Bline_mean,
            felec_r_Bline_mean]