#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 08:49:47 2023

@author: adrian

############################################################################
Description:    This python script performs the interpolation from the MFAM
                elements and faces to the nodes of a fine PIC mesh for several 
                variables computed at the MFAM by HYPHEN (phi, Te and je components)
                These variables are lated plotted at the fine PIC mesh, so that
                we reduce interpolation errors between the MFAM and the picM
                used for the HYPHEN simulation.
                The ion current is interpolated from the picM to the fine PIC
                mesh in order to plot the electric current in the fine PIC mesh
                The plasma density is also interpolated for obtaining other
                quantities
############################################################################
Inputs:         1) 
############################################################################
Output:         1)
"""


def HET_sims_interp_MFAM_picM_plot(path_picM_plot,n_elems,n_faces,elem_geom,
                                   face_geom,versors_e,versors_f,phi_elems,phi_faces,
                                   Te_elems,Te_faces,je_perp_elems,je_theta_elems,
                                   je_para_elems,je_z_elems,je_r_elems,
                                   je_perp_faces,je_theta_faces,je_para_faces,
                                   je_z_faces,je_r_faces,zs,rs,ji_x,ji_y,ji_z,ne,
                                   alpha_ano_elems,alpha_ano_e_elems,alpha_ano_q_elems,
                                   alpha_ine_elems,alpha_ine_q_elems,
                                   alpha_ano_faces,alpha_ano_e_faces,alpha_ano_q_faces,
                                   alpha_ine_faces,alpha_ine_q_faces):
    
    import h5py
    import numpy as np
    from scipy import interpolate
    
    
    
    # Open the HDF5 file containing the PIC mesh for plotting
    h5_picM = h5py.File(path_picM_plot,"r",swmr=True)
    # Retrieve 2D PIC mesh data coordinates
    zs_mp = h5_picM['/mesh_points/zs'][0:,0:]
    rs_mp = h5_picM['/mesh_points/rs'][0:,0:]
    dims_mp = np.shape(zs_mp)
    dataset = h5_picM['/mesh_flags/nodes_flag']
    nodes_flag_mp = dataset[...]
    dataset = h5_picM['/mesh_volumes/cells_vol']
    cells_vol_mp = dataset[...]
    xi_bottom_mp = h5_picM['/mesh_points/xi_bottom'][0][0] 
    xi_top_mp    = h5_picM['/mesh_points/xi_top'][0][0] 
    eta_min_mp   = h5_picM['/mesh_points/eta_min'][0][0] 
    eta_max_mp   = h5_picM['/mesh_points/eta_max'][0][0] 
    
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
    alpha_ano_points   = np.zeros((int(npoints),1),dtype='float')
    alpha_ano_e_points = np.zeros((int(npoints),1),dtype='float')
    alpha_ano_q_points = np.zeros((int(npoints),1),dtype='float')
    alpha_ine_points   = np.zeros((int(npoints),1),dtype='float')
    alpha_ine_q_points = np.zeros((int(npoints),1),dtype='float')
    
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
    alpha_ano_points[0:n_elems,0]   = alpha_ano_elems[:]
    alpha_ano_points[n_elems::,0]   = alpha_ano_faces[:]    
    alpha_ano_e_points[0:n_elems,0] = alpha_ano_e_elems[:]
    alpha_ano_e_points[n_elems::,0] = alpha_ano_e_faces[:]    
    alpha_ano_q_points[0:n_elems,0] = alpha_ano_q_elems[:]
    alpha_ano_q_points[n_elems::,0] = alpha_ano_q_faces[:]
    alpha_ine_points[0:n_elems,0]   = alpha_ine_elems[:]
    alpha_ine_points[n_elems::,0]   = alpha_ine_faces[:]
    alpha_ine_q_points[0:n_elems,0] = alpha_ine_q_elems[:]
    alpha_ine_q_points[n_elems::,0] = alpha_ine_q_faces[:]
    
    
    
    # Interpolate each variable from the MFAM to the fine PIC mesh
    var_mp         = np.zeros((dims_mp[0],dims_mp[1]),dtype='float')
    var_mp2        = np.zeros((dims_mp[0],dims_mp[1]),dtype='float')
    
    method1 = 'linear'
    method2 = 'nearest'
    
    
    # B
    var_points       = B_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    Bfield_mp = np.copy(var_mp)
    Bfield_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # Br
    var_points       = Br_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    Br_mp = np.copy(var_mp)
    Br_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # Bz
    var_points       = Bz_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    Bz_mp = np.copy(var_mp)
    Bz_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # alpha_ano
    var_points       = alpha_ano_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    alpha_ano_mp = np.copy(var_mp)
    alpha_ano_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # alpha_ano_e
    var_points       = alpha_ano_e_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    alpha_ano_e_mp = np.copy(var_mp)
    alpha_ano_e_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # alpha_ano_q
    var_points       = alpha_ano_q_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    alpha_ano_q_mp = np.copy(var_mp)
    alpha_ano_q_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # alpha_ine
    var_points       = alpha_ine_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    alpha_ine_mp = np.copy(var_mp)
    alpha_ine_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # alpha_ine
    var_points       = alpha_ine_q_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    alpha_ine_q_mp = np.copy(var_mp)
    alpha_ine_q_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # phi
    var_points       = phi_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    phi_mp = np.copy(var_mp)
    phi_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # Te
    var_points       = Te_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    Te_mp = np.copy(var_mp)
    Te_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # je_perp
    var_points       = je_perp_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    je_perp_mp = np.copy(var_mp)
    je_perp_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # je_theta
    var_points       = je_theta_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    je_theta_mp = np.copy(var_mp)
    je_theta_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # je_para
    var_points       = je_para_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    je_para_mp = np.copy(var_mp)
    je_para_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # je_z
    var_points       = je_z_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    je_z_mp = np.copy(var_mp)
    je_z_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # je_r
    var_points       = je_r_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    je_r_mp = np.copy(var_mp)
    je_r_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # je_2D
    var_points       = je_2D_points
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    je_2D_mp = np.copy(var_mp)
    je_2D_mp[np.where(nodes_flag_mp == 0)] = np.nan
    

    # Interpolate ji components and ne from HYPHEN picM to fine picM 
    dims = np.shape(zs)
    npoints_r = dims[0]
    npoints_z = dims[1]
    vec_points   = np.zeros((int(npoints_r*npoints_z),2),dtype='float')
    ji_x_points  = np.zeros((int(npoints_r*npoints_z),1),dtype='float')
    ji_y_points  = np.zeros((int(npoints_r*npoints_z),1),dtype='float')
    ji_z_points  = np.zeros((int(npoints_r*npoints_z),1),dtype='float')   
    ji_2D_points = np.zeros((int(npoints_r*npoints_z),1),dtype='float')      
    ne_points    = np.zeros((int(npoints_r*npoints_z),1),dtype='float')     
    ind = 0
    for i in range(0,int(npoints_r)):
        for j in range(0,int(npoints_z)):
           vec_points[ind,0]   = zs[i,j]
           vec_points[ind,1]   = rs[i,j]
           ji_x_points[ind,0]  = ji_x[i,j]
           ji_y_points[ind,0]  = ji_y[i,j]
           ji_z_points[ind,0]  = ji_z[i,j]
           ji_2D_points[ind,0] = ji_2D[i,j]
           ne_points[ind,0]    = ne[i,j]
           ind = ind + 1
    
    # ji_x
    var_points       = ji_x_points    
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    ji_x_mp = np.copy(var_mp)
    ji_x_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # ji_y
    var_points       = ji_y_points    
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    ji_y_mp = np.copy(var_mp)
    ji_y_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # ji_z
    var_points       = ji_z_points    
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    ji_z_mp = np.copy(var_mp)
    ji_z_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # ji_2D
    var_points       = ji_2D_points    
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    ji_2D_mp = np.copy(var_mp)
    ji_2D_mp[np.where(nodes_flag_mp == 0)] = np.nan
    
    # ne
    var_points       = ne_points    
    var_mp_interp    = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method1)
    var_mp_interp2   = interpolate.griddata(vec_points, var_points, (zs_mp, rs_mp), method=method2)
    var_mp[:,:]      = var_mp_interp[:,:,0]
    var_mp2[:,:]     = var_mp_interp2[:,:,0]
    arr_inds         = np.where(np.isnan(var_mp) == True)
    var_mp[arr_inds] = var_mp2[arr_inds]
    ne_mp = np.copy(var_mp)
    ne_mp[np.where(nodes_flag_mp == 0)] = np.nan
        
    
    # Obtain j components 
    j_r_mp  = je_r_mp + ji_x_mp
    j_t_mp  = je_theta_mp + ji_y_mp
    j_z_mp  = je_z_mp + ji_z_mp
    j_2D_mp = np.sqrt(j_r_mp**2 + j_z_mp**2)
    
    
    
    

    return[zs_mp,rs_mp,dims_mp,nodes_flag_mp,cells_vol_mp,xi_bottom_mp,
           xi_top_mp,eta_min_mp,eta_max_mp,phi_mp,Te_mp,je_perp_mp,je_theta_mp,
           je_para_mp,je_z_mp,je_r_mp,je_2D_mp,ji_x_mp,ji_y_mp,ji_z_mp,
           ji_2D_mp,j_r_mp,j_t_mp,j_z_mp,j_2D_mp,ne_mp,Bfield_mp,Br_mp,Bz_mp,
           alpha_ano_mp,alpha_ano_e_mp,alpha_ano_q_mp,alpha_ine_mp,
           alpha_ine_q_mp]