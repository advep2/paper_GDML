#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:23:15 2024

@author: adrian

############################################################################
Description:    This python script performs the interpolation from the MFAM
                elements and faces and from the PIC mesh nodes to the nodes of
                an angular profile (Faraday probe scan) for several 
                variables. For those variables computed at the MFAM by HYPHEN
                (phi, Te and je components), interpolation is performed from 
                the MFAM elements and faces. For those variables computed at
                the PIC mesh by HYPHEN, interpolation is performed from the
                nodes of the PIC mesh.
############################################################################
Inputs:         1) z_offset:    Distance (cm) from anode to axial position of 
                                the axis of Faraday probe scan. Positive if 
                                axial position of the axis of Faraday probe
                                scan is behind the thruster anode (behind 
                                the thruster). Negative otherwise. 
                2) r_offset:    Offset radius (cm) of the axis of the Faraday 
                                probe scan. It can only be positive
                3) rscan:       Radius (cm) of the Faraday probe scan 
                4) ang_min:     Minimum angle for the profile (deg)
                5) ang_max:     Maximum angle for the profile (deg)
                6) Npoints_ang: Number of points for the profile 
############################################################################
Output:         1) ang_scan:    Vector with angular points of the scan profiles (deg))
"""


def HET_sims_interp_scan(z_offset,r_offset,rscan,ang_min,ang_max,Npoints_ang,
                          n_elems,n_faces,elem_geom,face_geom,versors_e,versors_f,
                          phi_elems,phi_faces,Te_elems,Te_faces,je_perp_elems,
                          je_theta_elems,je_para_elems,je_z_elems,je_r_elems,
                          je_perp_faces,je_theta_faces,je_para_faces,je_z_faces,
                          je_r_faces,zs,rs,ji_x,ji_y,ji_z,ne,nn,Hall_par,
                          Hall_par_eff):
    
    import numpy as np
    from scipy import interpolate
    
    
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

    
    
    # Define points of the angular profile
    if rscan > zs[0,-1]*1E2 + z_offset:
        print("Warning (HET_sims_interp_scans): rscan is larger than axial dimension of plume. rscan taken as z_max + z_offset")
        rscan = zs[0,-1]*1E2 + z_offset
    ang_scan = np.linspace(ang_min,ang_max,Npoints_ang)
    r_scan = np.abs(rscan*1E-2*np.sin(ang_scan[:] * np.pi/180) + r_offset*1E-2)
    z_scan = rscan*1E-2*np.cos(ang_scan[:] * np.pi/180) - z_offset*1E-2
    

    

    method1 = 'linear'
    method2 = 'nearest'
    
    # Interpolate variables from the MFAM
    # B
    B_scan  = interpolate.griddata(vec_points, B_points, (z_scan, r_scan), method=method1)
    B_scan2 = interpolate.griddata(vec_points, B_points, (z_scan, r_scan), method=method2)
    arr_inds         = np.where(np.isnan(B_scan) == True)
    B_scan[arr_inds] = B_scan2[arr_inds]
    # Br
    Br_scan  = interpolate.griddata(vec_points, Br_points, (z_scan, r_scan), method=method1)
    Br_scan2 = interpolate.griddata(vec_points, Br_points, (z_scan, r_scan), method=method2)
    arr_inds          = np.where(np.isnan(Br_scan) == True)
    Br_scan[arr_inds] = Br_scan2[arr_inds]
    # Bz
    Bz_scan = interpolate.griddata(vec_points, Bz_points, (z_scan, r_scan), method=method1)
    Bz_scan2 = interpolate.griddata(vec_points, Bz_points, (z_scan, r_scan), method=method2)
    arr_inds          = np.where(np.isnan(Bz_scan) == True)
    Bz_scan[arr_inds] = Bz_scan2[arr_inds]
    # phi
    phi_scan  = interpolate.griddata(vec_points, phi_points, (z_scan, r_scan), method=method1)
    phi_scan2 = interpolate.griddata(vec_points, phi_points, (z_scan, r_scan), method=method2)
    arr_inds           = np.where(np.isnan(phi_scan) == True)
    phi_scan[arr_inds] = phi_scan2[arr_inds]
    # Te
    Te_scan  = interpolate.griddata(vec_points, Te_points, (z_scan, r_scan), method=method1)
    Te_scan2 = interpolate.griddata(vec_points, Te_points, (z_scan, r_scan), method=method2)
    arr_inds          = np.where(np.isnan(Te_scan) == True)
    Te_scan[arr_inds] = Te_scan2[arr_inds]
    # je_perp
    je_perp_scan  = interpolate.griddata(vec_points, je_perp_points, (z_scan, r_scan), method=method1)
    je_perp_scan2 = interpolate.griddata(vec_points, je_perp_points, (z_scan, r_scan), method=method2)
    arr_inds               = np.where(np.isnan(je_perp_scan) == True)
    je_perp_scan[arr_inds] = je_perp_scan2[arr_inds]
    # je_theta
    je_theta_scan  = interpolate.griddata(vec_points, je_theta_points, (z_scan, r_scan), method=method1)
    je_theta_scan2 = interpolate.griddata(vec_points, je_theta_points, (z_scan, r_scan), method=method2)
    arr_inds                = np.where(np.isnan(je_theta_scan) == True)
    je_theta_scan[arr_inds] = je_theta_scan2[arr_inds]
    # je_para
    je_para_scan  = interpolate.griddata(vec_points, je_para_points, (z_scan, r_scan), method=method1)
    je_para_scan2 = interpolate.griddata(vec_points, je_para_points, (z_scan, r_scan), method=method2)
    arr_inds               = np.where(np.isnan(je_para_scan) == True)
    je_para_scan[arr_inds] = je_para_scan2[arr_inds]
    # je_z
    je_z_scan  = interpolate.griddata(vec_points, je_z_points, (z_scan, r_scan), method=method1)
    je_z_scan2 = interpolate.griddata(vec_points, je_z_points, (z_scan, r_scan), method=method2)
    arr_inds            = np.where(np.isnan(je_z_scan) == True)
    je_z_scan[arr_inds] = je_z_scan2[arr_inds]
    # je_r
    je_r_scan  = interpolate.griddata(vec_points, je_r_points, (z_scan, r_scan), method=method1)
    je_r_scan2 = interpolate.griddata(vec_points, je_r_points, (z_scan, r_scan), method=method2)
    arr_inds            = np.where(np.isnan(je_r_scan) == True)
    je_r_scan[arr_inds] = je_r_scan2[arr_inds]
    # je_2D
    je_2D_scan  = interpolate.griddata(vec_points, je_2D_points, (z_scan, r_scan), method=method1)
    je_2D_scan2 = interpolate.griddata(vec_points, je_2D_points, (z_scan, r_scan), method=method2)
    arr_inds             = np.where(np.isnan(je_2D_scan) == True)
    je_2D_scan[arr_inds] = je_2D_scan2[arr_inds]
    
    
    # Interpolate variables from the PIC mesh
    # ji_x
    ji_x_scan  = interpolate.griddata(vec_points_picM, ji_x_points, (z_scan, r_scan), method=method1)
    ji_x_scan2 = interpolate.griddata(vec_points_picM, ji_x_points, (z_scan, r_scan), method=method2)
    arr_inds            = np.where(np.isnan(ji_x_scan) == True)
    ji_x_scan[arr_inds] = ji_x_scan2[arr_inds]
    # ji_y
    ji_y_scan  = interpolate.griddata(vec_points_picM, ji_y_points, (z_scan, r_scan), method=method1)
    ji_y_scan2 = interpolate.griddata(vec_points_picM, ji_y_points, (z_scan, r_scan), method=method2)
    arr_inds            = np.where(np.isnan(ji_y_scan) == True)
    ji_y_scan[arr_inds] = ji_y_scan2[arr_inds]
    # ji_z
    ji_z_scan  = interpolate.griddata(vec_points_picM, ji_z_points, (z_scan, r_scan), method=method1)
    ji_z_scan2 = interpolate.griddata(vec_points_picM, ji_z_points, (z_scan, r_scan), method=method2)
    arr_inds            = np.where(np.isnan(ji_z_scan) == True)
    ji_z_scan[arr_inds] = ji_z_scan2[arr_inds]
    # ji_2D
    ji_2D_scan  = interpolate.griddata(vec_points_picM, ji_2D_points, (z_scan, r_scan), method=method1)
    ji_2D_scan2 = interpolate.griddata(vec_points_picM, ji_2D_points, (z_scan, r_scan), method=method2)
    arr_inds             = np.where(np.isnan(ji_2D_scan) == True)
    ji_2D_scan[arr_inds] = ji_2D_scan2[arr_inds]
    # ne
    ne_scan  = interpolate.griddata(vec_points_picM, ne_points, (z_scan, r_scan), method=method1)
    ne_scan2 = interpolate.griddata(vec_points_picM, ne_points, (z_scan, r_scan), method=method2)
    arr_inds          = np.where(np.isnan(ne_scan) == True)
    ne_scan[arr_inds] = ne_scan2[arr_inds]
    # nn
    nn_scan  = interpolate.griddata(vec_points_picM, nn_points, (z_scan, r_scan), method=method1)
    nn_scan2 = interpolate.griddata(vec_points_picM, nn_points, (z_scan, r_scan), method=method2)
    arr_inds          = np.where(np.isnan(nn_scan) == True)
    nn_scan[arr_inds] = nn_scan2[arr_inds]
    # Hall_par
    Hall_par_scan  = interpolate.griddata(vec_points_picM, Hall_par_points, (z_scan, r_scan), method=method1)
    Hall_par_scan2 = interpolate.griddata(vec_points_picM, Hall_par_points, (z_scan, r_scan), method=method2)
    arr_inds          = np.where(np.isnan(Hall_par_scan) == True)
    Hall_par_scan[arr_inds] = Hall_par_scan2[arr_inds]
    # Hall_par_eff
    Hall_par_eff_scan = interpolate.griddata(vec_points_picM, Hall_par_eff_points, (z_scan, r_scan), method=method1)
    Hall_par_eff_scan2 = interpolate.griddata(vec_points_picM, Hall_par_eff_points, (z_scan, r_scan), method=method2)
    arr_inds          = np.where(np.isnan(Hall_par_eff_scan) == True)
    Hall_par_eff_scan[arr_inds] = Hall_par_eff_scan2[arr_inds]
    
    
    # Obtain other variables at the points of the angular profile
    ji_scan   = np.sqrt(ji_x_scan**2+ji_y_scan**2+ji_z_scan**2)
    je_scan   = np.sqrt(je_r_scan**2+je_theta_scan**2+je_z_scan**2)    
    j_r_scan  = ji_x_scan + je_r_scan
    j_t_scan  = ji_y_scan + je_theta_scan
    j_z_scan  = ji_z_scan + je_z_scan
    j_2D_scan = np.sqrt(j_r_scan**2+j_z_scan**2)
    j_scan    = np.sqrt(j_r_scan**2+j_z_scan**2 + j_t_scan**2)
    
    
    
    return [ang_scan,r_scan,z_scan,
            B_scan,Br_scan,Bz_scan,phi_scan,Te_scan,je_perp_scan,je_theta_scan,
            je_para_scan,je_z_scan,je_r_scan,je_2D_scan,je_scan,
            ji_x_scan,ji_y_scan,ji_z_scan,ji_2D_scan,ji_scan,ne_scan,nn_scan,
            Hall_par_scan,Hall_par_eff_scan,
            j_r_scan,j_t_scan,j_z_scan,j_2D_scan,j_scan]