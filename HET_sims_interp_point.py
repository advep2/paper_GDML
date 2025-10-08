#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:23:15 2024

@author: adrian

############################################################################
Description:    This python script provides the value of different plasma
                variables at a point by interpolating to that point.
                This python script is adapted from HET_sims_interp_zprof.py
                FOR NOW (10/05/2025) it only returns the value of the electric
                potential and it is used to compute the coupling voltage
############################################################################
Inputs:         1) zpoint,rpoint: coordinates (cm) of the selected point at 
                                  which perform the interpolation 
############################################################################
Output:         
"""


def HET_sims_interp_point(zpoint,rpoint,
                          n_elems,n_faces,elem_geom,face_geom,versors_e,versors_f,
                          phi_elems,phi_faces,Te_elems,Te_faces,je_perp_elems,
                          je_theta_elems,je_para_elems,je_z_elems,je_r_elems,
                          je_perp_faces,je_theta_faces,je_para_faces,je_z_faces,
                          je_r_faces,
                          zs,rs,zs_mp,rs_mp,ji_x,ji_y,ji_z,ne,nn,Hall_par,
                          Hall_par_eff):
    
    import numpy as np
    from scipy import interpolate
    
    # Obtain the points of the axial profile.
    # If interp_MFAM_picM_plot is activated, the fine PIC mesh is used only for
    # those variables computed at the MFAM by HYPHEN.
    z_prof = zpoint
    r_prof = rpoint
    

    # Compute additional magnitudes to interpolate
    je_2D_elems = np.sqrt(je_r_elems**2 + je_z_elems**2)
    je_2D_faces = np.sqrt(je_r_faces**2 + je_z_faces**2)
    ji_2D       = np.sqrt(ji_x**2 + ji_z**2)
    # Obtain the magnetic field magnitude and components at elements and faces in Gauss
    B_elems     = elem_geom[-1,:]*1E4     
    B_faces     = face_geom[-1,:]*1E4     
    Br_elems    = B_elems*versors_e[3,:]
    Bz_elems    = B_elems*versors_e[2,:]
    Br_faces    = B_faces*versors_f[3,:]
    Bz_faces    = B_faces*versors_f[2,:]
    
    # Prepare data at MFAM for interpolation
    npoints = n_elems+n_faces
    vec_points  = np.zeros((int(npoints),2),dtype='float')
    vec_points[0:n_elems,0] = np.transpose(elem_geom[0,:])
    vec_points[n_elems::,0] = np.transpose(face_geom[0,:])
    vec_points[0:n_elems,1] = np.transpose(elem_geom[1,:])
    vec_points[n_elems::,1] = np.transpose(face_geom[1,:])
    phi_points         = np.zeros((int(npoints),1),dtype='float')
    Te_points          = np.zeros((int(npoints),1),dtype='float')
    je_perp_points     = np.zeros((int(npoints),1),dtype='float')
    je_theta_points    = np.zeros((int(npoints),1),dtype='float')
    je_para_points     = np.zeros((int(npoints),1),dtype='float')
    je_z_points        = np.zeros((int(npoints),1),dtype='float')
    je_r_points        = np.zeros((int(npoints),1),dtype='float')
    je_2D_points       = np.zeros((int(npoints),1),dtype='float')
    B_points           = np.zeros((int(npoints),1),dtype='float')
    Br_points          = np.zeros((int(npoints),1),dtype='float')
    Bz_points          = np.zeros((int(npoints),1),dtype='float')
    
    phi_points[0:n_elems,0]         = phi_elems[:]
    phi_points[n_elems::,0]         = phi_faces[:]
    Te_points[0:n_elems,0]          = Te_elems[:]
    Te_points[n_elems::,0]          = Te_faces[:]
    je_perp_points[0:n_elems,0]     = je_perp_elems[:]
    je_perp_points[n_elems::,0]     = je_perp_faces[:]
    je_theta_points[0:n_elems,0]    = je_theta_elems[:]
    je_theta_points[n_elems::,0]    = je_theta_faces[:]
    je_para_points[0:n_elems,0]     = je_para_elems[:]
    je_para_points[n_elems::,0]     = je_para_faces[:]
    je_z_points[0:n_elems,0]        = je_z_elems[:]
    je_z_points[n_elems::,0]        = je_z_faces[:]
    je_r_points[0:n_elems,0]        = je_r_elems[:]
    je_r_points[n_elems::,0]        = je_r_faces[:]
    je_2D_points[0:n_elems,0]       = je_2D_elems[:]
    je_2D_points[n_elems::,0]       = je_2D_faces[:]
    B_points[0:n_elems,0]           = B_elems[:]
    B_points[n_elems::,0]           = B_faces[:] 
    Br_points[0:n_elems,0]          = Br_elems[:]
    Br_points[n_elems::,0]          = Br_faces[:] 
    Bz_points[0:n_elems,0]          = Bz_elems[:]
    Bz_points[n_elems::,0]          = Bz_faces[:]
    
    
    # Prepare data at picM for interpolation
    dims = np.shape(zs)
    npoints_r = dims[0]
    npoints_z = dims[1]
    vec_points_picM     = np.zeros((int(npoints_r*npoints_z),2),dtype='float')
    ji_x_points         = np.zeros((int(npoints_r*npoints_z),1),dtype='float')
    ji_y_points         = np.zeros((int(npoints_r*npoints_z),1),dtype='float')
    ji_z_points         = np.zeros((int(npoints_r*npoints_z),1),dtype='float')   
    ji_2D_points        = np.zeros((int(npoints_r*npoints_z),1),dtype='float')      
    ne_points           = np.zeros((int(npoints_r*npoints_z),1),dtype='float')     
    nn_points           = np.zeros((int(npoints_r*npoints_z),1),dtype='float') 
    Hall_par_points     = np.zeros((int(npoints_r*npoints_z),1),dtype='float') 
    Hall_par_eff_points = np.zeros((int(npoints_r*npoints_z),1),dtype='float') 
    
    ind = 0
    for i in range(0,int(npoints_r)):
        for j in range(0,int(npoints_z)):
           vec_points_picM[ind,0]     = zs[i,j]
           vec_points_picM[ind,1]     = rs[i,j]
           ji_x_points[ind,0]         = ji_x[i,j]
           ji_y_points[ind,0]         = ji_y[i,j]
           ji_z_points[ind,0]         = ji_z[i,j]
           ji_2D_points[ind,0]        = ji_2D[i,j]
           ne_points[ind,0]           = ne[i,j]
           nn_points[ind,0]           = nn[i,j]
           Hall_par_points[ind,0]     = Hall_par[i,j]
           Hall_par_eff_points[ind,0] = Hall_par_eff[i,j]
           ind = ind + 1



    method1 = 'linear'
    method2 = 'nearest'
    
    # Interpolate variables from the MFAM
    # # B
    # B_prof  = interpolate.griddata(vec_points, B_points, (z_prof, r_prof), method=method1)
    # B_prof2 = interpolate.griddata(vec_points, B_points, (z_prof, r_prof), method=method2)
    # arr_inds         = np.where(np.isnan(B_prof) == True)
    # B_prof[arr_inds] = B_prof2[arr_inds]
    # B_prof = B_prof[:,0]
    # # Br
    # Br_prof  = interpolate.griddata(vec_points, Br_points, (z_prof, r_prof), method=method1)
    # Br_prof2 = interpolate.griddata(vec_points, Br_points, (z_prof, r_prof), method=method2)
    # arr_inds          = np.where(np.isnan(Br_prof) == True)
    # Br_prof[arr_inds] = Br_prof2[arr_inds]
    # Br_prof = Br_prof[:,0]
    # # Bz
    # Bz_prof = interpolate.griddata(vec_points, Bz_points, (z_prof, r_prof), method=method1)
    # Bz_prof2 = interpolate.griddata(vec_points, Bz_points, (z_prof, r_prof), method=method2)
    # arr_inds          = np.where(np.isnan(Bz_prof) == True)
    # Bz_prof[arr_inds] = Bz_prof2[arr_inds]
    # Bz_prof = Bz_prof[:,0]
    # phi
    phi_prof  = interpolate.griddata(vec_points, phi_points, (z_prof, r_prof), method=method1)
    phi_prof2 = interpolate.griddata(vec_points, phi_points, (z_prof, r_prof), method=method2)
    arr_inds           = np.where(np.isnan(phi_prof) == True)
    phi_prof[arr_inds] = phi_prof2[arr_inds]
    phi_prof = phi_prof[0]
    # # Te
    # Te_prof  = interpolate.griddata(vec_points, Te_points, (z_prof, r_prof), method=method1)
    # Te_prof2 = interpolate.griddata(vec_points, Te_points, (z_prof, r_prof), method=method2)
    # arr_inds          = np.where(np.isnan(Te_prof) == True)
    # Te_prof[arr_inds] = Te_prof2[arr_inds]
    # Te_prof = Te_prof[:,0]
    # # je_perp
    # je_perp_prof  = interpolate.griddata(vec_points, je_perp_points, (z_prof, r_prof), method=method1)
    # je_perp_prof2 = interpolate.griddata(vec_points, je_perp_points, (z_prof, r_prof), method=method2)
    # arr_inds               = np.where(np.isnan(je_perp_prof) == True)
    # je_perp_prof[arr_inds] = je_perp_prof2[arr_inds]
    # je_perp_prof = je_perp_prof[:,0]
    # # je_theta
    # je_theta_prof  = interpolate.griddata(vec_points, je_theta_points, (z_prof, r_prof), method=method1)
    # je_theta_prof2 = interpolate.griddata(vec_points, je_theta_points, (z_prof, r_prof), method=method2)
    # arr_inds                = np.where(np.isnan(je_theta_prof) == True)
    # je_theta_prof[arr_inds] = je_theta_prof2[arr_inds]
    # je_theta_prof = je_theta_prof[:,0]
    # # je_para
    # je_para_prof  = interpolate.griddata(vec_points, je_para_points, (z_prof, r_prof), method=method1)
    # je_para_prof2 = interpolate.griddata(vec_points, je_para_points, (z_prof, r_prof), method=method2)
    # arr_inds               = np.where(np.isnan(je_para_prof) == True)
    # je_para_prof[arr_inds] = je_para_prof2[arr_inds]
    # je_para_prof = je_para_prof[:,0]
    # # je_z
    # je_z_prof  = interpolate.griddata(vec_points, je_z_points, (z_prof, r_prof), method=method1)
    # je_z_prof2 = interpolate.griddata(vec_points, je_z_points, (z_prof, r_prof), method=method2)
    # arr_inds            = np.where(np.isnan(je_z_prof) == True)
    # je_z_prof[arr_inds] = je_z_prof2[arr_inds]
    # je_z_prof = je_z_prof[:,0]
    # # je_r
    # je_r_prof  = interpolate.griddata(vec_points, je_r_points, (z_prof, r_prof), method=method1)
    # je_r_prof2 = interpolate.griddata(vec_points, je_r_points, (z_prof, r_prof), method=method2)
    # arr_inds            = np.where(np.isnan(je_r_prof) == True)
    # je_r_prof[arr_inds] = je_r_prof2[arr_inds]
    # je_r_prof = je_r_prof[:,0]
    # # je_2D
    # je_2D_prof  = interpolate.griddata(vec_points, je_2D_points, (z_prof, r_prof), method=method1)
    # je_2D_prof2 = interpolate.griddata(vec_points, je_2D_points, (z_prof, r_prof), method=method2)
    # arr_inds             = np.where(np.isnan(je_2D_prof) == True)
    # je_2D_prof[arr_inds] = je_2D_prof2[arr_inds]
    # je_2D_prof = je_2D_prof[:,0]


    # # Interpolate ion current from the PIC mesh to compute electric current
    # # ji_x
    # ji_x_prof  = interpolate.griddata(vec_points_picM, ji_x_points, (z_prof, r_prof), method=method1)
    # ji_x_prof2 = interpolate.griddata(vec_points_picM, ji_x_points, (z_prof, r_prof), method=method2)
    # arr_inds            = np.where(np.isnan(ji_x_prof) == True)
    # ji_x_prof[arr_inds] = ji_x_prof2[arr_inds]
    # ji_x_prof = ji_x_prof[:,0]
    # # ji_y
    # ji_y_prof  = interpolate.griddata(vec_points_picM, ji_y_points, (z_prof, r_prof), method=method1)
    # ji_y_prof2 = interpolate.griddata(vec_points_picM, ji_y_points, (z_prof, r_prof), method=method2)
    # arr_inds            = np.where(np.isnan(ji_y_prof) == True)
    # ji_y_prof[arr_inds] = ji_y_prof2[arr_inds]
    # ji_y_prof = ji_y_prof[:,0]
    # # ji_z
    # ji_z_prof  = interpolate.griddata(vec_points_picM, ji_z_points, (z_prof, r_prof), method=method1)
    # ji_z_prof2 = interpolate.griddata(vec_points_picM, ji_z_points, (z_prof, r_prof), method=method2)
    # arr_inds            = np.where(np.isnan(ji_z_prof) == True)
    # ji_z_prof[arr_inds] = ji_z_prof2[arr_inds]
    # ji_z_prof = ji_z_prof[:,0]
    # # ji_2D
    # ji_2D_prof  = interpolate.griddata(vec_points_picM, ji_2D_points, (z_prof, r_prof), method=method1)
    # ji_2D_prof2 = interpolate.griddata(vec_points_picM, ji_2D_points, (z_prof, r_prof), method=method2)
    # arr_inds             = np.where(np.isnan(ji_2D_prof) == True)
    # ji_2D_prof[arr_inds] = ji_2D_prof2[arr_inds]
    # ji_2D_prof = ji_2D_prof[:,0]


    # # Obtain other variables
    # ji_prof   = np.sqrt(ji_x_prof**2+ji_y_prof**2+ji_z_prof**2)
    # je_prof   = np.sqrt(je_r_prof**2+je_theta_prof**2+je_z_prof**2)    
    # j_r_prof  = ji_x_prof + je_r_prof
    # j_t_prof  = ji_y_prof + je_theta_prof
    # j_z_prof  = ji_z_prof + je_z_prof
    # j_2D_prof = np.sqrt(j_r_prof**2+j_z_prof**2)
    # j_prof    = np.sqrt(j_r_prof**2+j_z_prof**2 + j_t_prof**2)


        



    # # Interpolate variables from the PIC mesh        
    # # ne
    # ne_prof  = interpolate.griddata(vec_points_picM, ne_points, (z_prof, r_prof), method=method1)
    # ne_prof2 = interpolate.griddata(vec_points_picM, ne_points, (z_prof, r_prof), method=method2)
    # arr_inds          = np.where(np.isnan(ne_prof) == True)
    # ne_prof[arr_inds] = ne_prof2[arr_inds]
    # ne_prof = ne_prof[:,0]
    # # nn
    # nn_prof  = interpolate.griddata(vec_points_picM, nn_points, (z_prof, r_prof), method=method1)
    # nn_prof2 = interpolate.griddata(vec_points_picM, nn_points, (z_prof, r_prof), method=method2)
    # arr_inds          = np.where(np.isnan(nn_prof) == True)
    # nn_prof[arr_inds] = nn_prof2[arr_inds]
    # nn_prof = nn_prof[:,0]
    # # Hall_par
    # Hall_par_prof  = interpolate.griddata(vec_points_picM, Hall_par_points, (z_prof, r_prof), method=method1)
    # Hall_par_prof2 = interpolate.griddata(vec_points_picM, Hall_par_points, (z_prof, r_prof), method=method2)
    # arr_inds          = np.where(np.isnan(Hall_par_prof) == True)
    # Hall_par_prof[arr_inds] = Hall_par_prof2[arr_inds]
    # Hall_par_prof = Hall_par_prof[:,0]
    # # Hall_par_eff
    # Hall_par_eff_prof = interpolate.griddata(vec_points_picM, Hall_par_eff_points, (z_prof, r_prof), method=method1)
    # Hall_par_eff_prof2 = interpolate.griddata(vec_points_picM, Hall_par_eff_points, (z_prof, r_prof), method=method2)
    # arr_inds          = np.where(np.isnan(Hall_par_eff_prof) == True)
    # Hall_par_eff_prof[arr_inds] = Hall_par_eff_prof2[arr_inds]
    # Hall_par_eff_prof = Hall_par_eff_prof[:,0]
    # # Hall_par_effect
    # Hall_par_effect_prof = np.sqrt(Hall_par_eff_prof*Hall_par_prof)

    
    return [phi_prof]
    
    # return [B_prof,Br_prof,Bz_prof,phi_prof,Te_prof,je_perp_prof,je_theta_prof,
    #         je_para_prof,je_z_prof,je_r_prof,je_2D_prof,je_prof,
    #         j_r_prof,j_t_prof,j_z_prof,j_2D_prof,j_prof,
        
    #         ji_x_prof,ji_y_prof,ji_z_prof,ji_2D_prof,ji_prof,ne_prof,nn_prof,
    #         Hall_par_prof,Hall_par_eff_prof,Hall_par_effect_prof]