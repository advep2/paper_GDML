#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:44:53 2020

@author: adrian

############################################################################
Description:    This python script reads the distribution functions data from HET sims
############################################################################
Inputs:         1) path_simstate_inp: path to input SimState.hdf5 file
                2) path_simstate_out: path to output SimState.hdf5 file
                3) path_postdata_out: path to output PostData.hdf5 file
                4) path_simparams_inp: path to input sim_params.inp file
                6) allsteps_flag: if 1 all print-out steps are read
                7) timestep: if allsteps_flag = 0, indicates the step ID to be read
                8) read_inst_data: if 1 all timed data printed at every step is read (other data)
               10) read_part_lists: if 1 the particle lists are read (time consuming)
               11) read_flag: if 1 extra electron-fluid variables at the PIC mesh
                              are read. This is for compatibility with old simulations
                              not containing those variables
               12) oldpost_sim,oldsimparams_sim: flags for old simulations inp/out files
                   oldpost_sim for reading q datasets: 1 Old sim, 0 new sim
                   oldsimparams_sim: 0 Old sim (thesis), 1 new sim 1, 2 new sim 2
############################################################################
Output:        1) Read variables
"""

def HET_sims_read_df(path_simstate_inp,path_simstate_out,path_postdata_out,path_simparams_inp,
                     path_picM,allsteps_flag,timestep,oldpost_sim,oldsimparams_sim):
    
    import h5py
    import numpy as np
    
    # Parameters
    e  = 1.6021766E-19
    g0 = 9.80665
    me = 9.1093829E-31
    
    # Open the SimState.hdf5 input/output files
    h5_inp = h5py.File(path_simstate_inp,"r+")
    h5_out = h5py.File(path_simstate_out,"r+")
    # Open the PostData.hdf5 file
    h5_post = h5py.File(path_postdata_out,"r+")
    # Open the PIC mesh HDF5 file
    h5_picM = h5py.File(path_picM,"r+")
    
    print("HET_sims_read_df: reading sim_params.inp...")
    # Open sim_params.inp file
    r = open(path_simparams_inp,'r')
    lines = r.readlines()
    r.close() 
    
    
    if oldsimparams_sim == 8:
        # SIMS from commit 7189590 (new sheath)
        line_num_ion_spe            = 56
        line_num_neu_spe            = 57
        line_Z_ion_spe              = 58
        line_n_mp_cell_i            = 59
        line_n_mp_cell_n            = 60
        line_n_mp_cell_i_min        = 63
        line_n_mp_cell_i_max        = 64
        line_n_mp_cell_n_min        = 65
        line_n_mp_cell_n_max        = 66
        line_min_ion_plasma_density = 76
        line_m_A                    = 79
        line_ene_bal                = 13
        line_nsteps_eFld            = 33
        line_override_cathode_e     = 134
        line_new_cathode_e          = 135
        line_T_cath                 = 185
        line_ncollisions            = 197
        line_ncollisions_e          = 187
        line_ids_collisions_e       = 188
        line_coll_spe_e             = 190
        line_out_coll_spe_e         = 191
    
    elif oldsimparams_sim == 20:
        # SIMS from commit 3590d0a (moving to intel ifort compiler): valid with oldpost_sim = 6 
        line_num_ion_spe            = 61
        line_num_neu_spe            = 62
        line_Z_ion_spe              = 63
        line_n_mp_cell_i            = 70
        line_n_mp_cell_n            = 71
        line_n_mp_cell_i_min        = 74
        line_n_mp_cell_i_max        = 75
        line_n_mp_cell_n_min        = 76
        line_n_mp_cell_n_max        = 77
        line_min_ion_plasma_density = 87
        line_m_A                    = 90
        line_ene_bal                = 13
        line_nsteps_eFld            = 34
        line_override_cathode_e     = 134
        line_new_cathode_e          = 135
        line_T_cath                 = 186
        line_cath_type              = 187
        line_ncollisions            = 200
        line_ncollisions_e          = 190
        line_ids_collisions_e       = 191
        line_coll_spe_e             = 193
        line_out_coll_spe_e         = 194
        line_interp_eFld2PIC_alldata = 16
        line_prnt_out_inst_vars     =  36
        line_print_out_picMformat   =  37
        line_n_cond_wall            = 152
        line_ff_c_bound_type_je     = 157
    
    # Obtain the number of ion and neutral species
    num_ion_spe = int(lines[line_num_ion_spe][lines[line_num_ion_spe].find('=')+1:lines[line_num_ion_spe].find('\n')])
    num_neu_spe = int(lines[line_num_neu_spe][lines[line_num_neu_spe].find('=')+1:lines[line_num_neu_spe].find('\n')])
    # Obtain the number of steps of the electron fluid module per PIC step
    nsteps_eFld = int(lines[line_nsteps_eFld][lines[line_nsteps_eFld].find('=')+1:lines[line_nsteps_eFld].find('\n')])
    
    
    print("HET_sims_read_df: reading mesh...")
    # Obtain PIC mesh boundary points
    points = h5_picM['mesh_points/points'][0:,0:]
    # Retrieve 2D PIC mesh data coordinates
    zs = h5_out['/picM/zs'][0:,0:]
    rs = h5_out['/picM/rs'][0:,0:]
    dims = np.shape(zs)
    dataset = h5_out['/picM/nodes_flag']
    nodes_flag = dataset[...]
    dataset = h5_out['/picM/cells_flag']
    cells_flag = dataset[...]
    dataset = h5_out['/picM/cells_vol']
    cells_vol = dataset[...]
    dataset = h5_out['/picM/vol']
    vol = dataset[...]
    volume    = np.sum(cells_vol[np.where(cells_flag == 1)]) 
    zscells = 0.25*(zs[0:-1,0:-1] + zs[0:-1,1::] + zs[1::,0:-1] + zs[1::,1::])
    rscells = 0.25*(rs[0:-1,0:-1] + rs[0:-1,1::] + rs[1::,0:-1] + rs[1::,1::])
    # Obtain indices nr and nz for chamber and plume zone
    ind_maxr_c = dims[0] - 1
    ind_maxz_c = dims[1] - 1 
    nr_c = ind_maxr_c + 1
    nz_c = ind_maxz_c + 1
    eta_max   = int(h5_out['/picM/eta_max'][0][0]) 
    eta_min   = int(h5_out['/picM/eta_min'][0][0]) 
    xi_top    = int(h5_out['/picM/xi_top'][0][0])
    xi_bottom = int(h5_out['/picM/xi_bottom'][0][0])
    # Retrieve picS data at important elements
    surf_elems  = h5_out['picS/surf_elems'][0:,0:]
    n_imp_elems = h5_out['picS/n_imp_elems'][0][0]
    imp_elems   = h5_out['picS/imp_elems'][0:,0:] - 1
    norm_vers   = h5_out['picS/norm_vers'][0:,0:,0:] # normal versor (r,z) components for each surface element (isurf,jsurf)
    
    # Obtain length of boundary parts, depending on the type of PIC mesh
    n_parts = len(points)
    L_parts = np.zeros(n_parts,dtype=float)
    points_append = np.append(points,[points[0,:]],axis=0)
    for i in range(0,n_parts):
        L_parts[i] = np.sqrt((points_append[i+1,0]-points_append[i,0])**2 + (points_append[i+1,1]-points_append[i,1])**2)
        
    # Obtain several important lengths along bottom and top dielectric walls
    if n_parts == 8:
        # No chamfer case
        Lplume_bot    = L_parts[1]
        Lplume_top    = L_parts[5]
        Lchamb_bot    = L_parts[0]
        Lchamb_top    = L_parts[6]
        Lfreeloss_ver = L_parts[3]
        Lfreeloss_lat = L_parts[4]
        Lfreeloss     = L_parts[3] + L_parts[4]
        Laxis         = L_parts[2]
    elif n_parts == 9:
        # One chamfer case
        if points[1,1] == points[2,1]:
            # Chamber is in top boundary
            Lplume_bot    = L_parts[1]
            Lplume_top    = L_parts[5]
            Lchamb_bot    = L_parts[0] 
            Lchamb_top    = L_parts[6] + L_parts[7]
            Lfreeloss_ver = L_parts[3]
            Lfreeloss_lat = L_parts[4]
            Lfreeloss     = L_parts[3] + L_parts[4]
            Laxis         = L_parts[2]
        else:
            # Chamfer is in bottom boundary
            Lplume_bot    = L_parts[2]
            Lplume_top    = L_parts[6]
            Lchamb_bot    = L_parts[0] + L_parts[1]
            Lchamb_top    = L_parts[7]
            Lfreeloss_ver = L_parts[4]
            Lfreeloss_lat = L_parts[5]
            Lfreeloss     = L_parts[4] + L_parts[5]
            Laxis         = L_parts[3]

    elif n_parts == 10:
        # Two chamfer case
        Lplume_bot    = L_parts[2]
        Lplume_top    = L_parts[6]
        Lchamb_bot    = L_parts[0] + L_parts[1]
        Lchamb_top    = L_parts[7] + L_parts[8]
        Lfreeloss_ver = L_parts[4]
        Lfreeloss_lat = L_parts[5]
        Lfreeloss     = L_parts[4] + L_parts[5]
        Laxis         = L_parts[3]
    Lanode     = L_parts[-1]
    
    # ---- PIC mesh nodes related vars for each type of boundary --------------
    # Obtain the number of PIC mesh nodes for each type of boundary
    nnodes_Dwall_bot  = int(xi_bottom + 1 + eta_min + 1 - 1)
    nnodes_Dwall_top  = int(xi_top + 1 + dims[0] - (eta_max + 1))
    nnodes_Awall      = int((eta_max + 1) - (eta_min + 1) + 1)
    nnodes_FLwall_ver = int(dims[0])
    nnodes_FLwall_lat = int(dims[1] - (xi_bottom + 1) + 1)
    nnodes_Axis       = int(dims[1] - (xi_bottom + 1) + 1)
    nnodes_bound = 0
    for i in range(0,dims[0]):
        for j in range(0,dims[1]):
            if nodes_flag[i,j] == -1:
                nnodes_bound = nnodes_bound + 1
    # Obtain the PIC mesh nodes indices vectors for each type of boundary
    inodes_Dwall_bot  = np.concatenate((eta_min*np.ones(xi_bottom+1,dtype=int),np.linspace(eta_min-1,0,eta_min,dtype=int)))
    jnodes_Dwall_bot  = np.concatenate((np.linspace(0,xi_bottom,xi_bottom+1,dtype=int),xi_bottom*np.ones(eta_min,dtype=int) ))
    inodes_Dwall_top  = np.concatenate((eta_max*np.ones(xi_top+1,dtype=int),np.linspace(eta_max+1,dims[0]-1,dims[0] - (eta_max + 1),dtype=int)))
    jnodes_Dwall_top  = np.concatenate((np.linspace(0,xi_top,xi_top+1,dtype=int),xi_top*np.ones(dims[0] - (eta_max + 1),dtype=int) ))
    inodes_Awall      = np.linspace(eta_min,eta_max,nnodes_Awall,dtype=int)
    jnodes_Awall      = np.zeros(nnodes_Awall,dtype=int)
    inodes_FLwall_ver = np.linspace(0,dims[0]-1,nnodes_FLwall_ver,dtype=int)
    jnodes_FLwall_ver = (dims[1]-1)*np.ones(nnodes_FLwall_ver,dtype=int)
    inodes_FLwall_lat = (dims[0]-1)*np.ones(nnodes_FLwall_lat,dtype=int)
    jnodes_FLwall_lat = np.linspace(xi_top,dims[1]-1,nnodes_FLwall_lat,dtype=int)
    inodes_Axis       = np.zeros(nnodes_Axis,dtype=int)
    jnodes_Axis       = np.linspace(xi_bottom,dims[1]-1,nnodes_Axis,dtype=int)
    
    # Obtain the arc length along each type of boundary
    sDwall_bot_nodes  = np.zeros(nnodes_Dwall_bot)
    sDwall_top_nodes  = np.zeros(nnodes_Dwall_top)
    sAwall_nodes      = np.zeros(nnodes_Awall)
    sFLwall_ver_nodes = np.zeros(nnodes_FLwall_ver)
    sFLwall_lat_nodes = np.zeros(nnodes_FLwall_lat)
    sAxis_nodes       = np.zeros(nnodes_Axis)
    for i in range(0,nnodes_Dwall_bot-1):
        z1 = zs[inodes_Dwall_bot[i],jnodes_Dwall_bot[i]]
        r1 = rs[inodes_Dwall_bot[i],jnodes_Dwall_bot[i]]
        z2 = zs[inodes_Dwall_bot[i+1],jnodes_Dwall_bot[i+1]]
        r2 = rs[inodes_Dwall_bot[i+1],jnodes_Dwall_bot[i+1]]
        sDwall_bot_nodes[i+1] = sDwall_bot_nodes[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    for i in range(0,nnodes_Dwall_top-1):
        z1 = zs[inodes_Dwall_top[i],jnodes_Dwall_top[i]]
        r1 = rs[inodes_Dwall_top[i],jnodes_Dwall_top[i]]
        z2 = zs[inodes_Dwall_top[i+1],jnodes_Dwall_top[i+1]]
        r2 = rs[inodes_Dwall_top[i+1],jnodes_Dwall_top[i+1]]
        sDwall_top_nodes[i+1] = sDwall_top_nodes[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    for i in range(0,nnodes_Awall-1):
        z1 = zs[inodes_Awall[i],jnodes_Awall[i]]
        r1 = rs[inodes_Awall[i],jnodes_Awall[i]]
        z2 = zs[inodes_Awall[i+1],jnodes_Awall[i+1]]
        r2 = rs[inodes_Awall[i+1],jnodes_Awall[i+1]]
        sAwall_nodes[i+1] = sAwall_nodes[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    for i in range(0,nnodes_FLwall_ver-1):
        z1 = zs[inodes_FLwall_ver[i],jnodes_FLwall_ver[i]]
        r1 = rs[inodes_FLwall_ver[i],jnodes_FLwall_ver[i]]
        z2 = zs[inodes_FLwall_ver[i+1],jnodes_FLwall_ver[i+1]]
        r2 = rs[inodes_FLwall_ver[i+1],jnodes_FLwall_ver[i+1]]
        sFLwall_ver_nodes[i+1] = sFLwall_ver_nodes[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    for i in range(0,nnodes_FLwall_lat-1):
        z1 = zs[inodes_FLwall_lat[i],jnodes_FLwall_lat[i]]
        r1 = rs[inodes_FLwall_lat[i],jnodes_FLwall_lat[i]]
        z2 = zs[inodes_FLwall_lat[i+1],jnodes_FLwall_lat[i+1]]
        r2 = rs[inodes_FLwall_lat[i+1],jnodes_FLwall_lat[i+1]]
        sFLwall_lat_nodes[i+1] = sFLwall_lat_nodes[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    for i in range(0,nnodes_Axis-1):
        z1 = zs[inodes_Axis[i],jnodes_Axis[i]]
        r1 = rs[inodes_Axis[i],jnodes_Axis[i]]
        z2 = zs[inodes_Axis[i+1],jnodes_Axis[i+1]]
        r2 = rs[inodes_Axis[i+1],jnodes_Axis[i+1]]
        sAxis_nodes[i+1] = sAxis_nodes[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
        
    # Obtain the arc length value at the end of the chamber for top and bottom boundaries (use the PIC mesh for simplicity)
#    sc_bot = np.sqrt((zs[int(eta_min),0] - zs[int(eta_min),int(xi_bottom)])**2 + (rs[int(eta_min),0] - rs[int(eta_min),int(xi_bottom)])**2)
#    sc_top = np.sqrt((zs[int(eta_max),0] - zs[int(eta_max),int(xi_top)])**2 + (rs[int(eta_max),0] - rs[int(eta_max),int(xi_top)])**2)
    
    # Improve the calculation commented above (old) by substracting the length of the plume vertical bottom and top parts
#    sc_bot = sDwall_bot[-1] - np.sqrt((zs[int(eta_min),int(xi_bottom)] - zs[0,int(xi_bottom)])**2 + (rs[int(eta_min),int(xi_bottom)] - rs[0,int(xi_bottom)])**2)
#    sc_top = sDwall_top[-1] - np.sqrt((zs[-1,int(xi_top)] - zs[int(eta_max),int(xi_top)])**2 + (rs[-1,int(xi_top)] - rs[int(eta_max),int(xi_top)])**2)
    
#    sc_bot = sDwall_bot[-1] - Lplume_bot
#    sc_top = sDwall_top[-1] - Lplume_top
    sc_bot = Lchamb_bot
    sc_top = Lchamb_top
    # -------------------------------------------------------------------------
    
    # ---- PIC mesh surface elements related vars for each type of boundary ---
    # Run over the nodes of each type of boundary. Obtain the number of surface
    # elements for each type of boundary. Obtain the index (position) of the  
    # surface elements in imp_elems for each type of boundary, and obtain its
    # coordinates
    nsurf_Dwall_bot  = 0
    nsurf_Dwall_top  = 0
    nsurf_Awall      = 0
    nsurf_FLwall_ver = 0
    nsurf_FLwall_lat = 0
    nsurf_bound      = 0
    indsurf_Dwall_bot  = np.zeros(nsurf_Dwall_bot,dtype=int)
    zsurf_Dwall_bot    = np.zeros(nsurf_Dwall_bot,dtype=float)
    rsurf_Dwall_bot    = np.zeros(nsurf_Dwall_bot,dtype=float)
    indsurf_Dwall_top  = np.zeros(nsurf_Dwall_top,dtype=int)
    zsurf_Dwall_top    = np.zeros(nsurf_Dwall_top,dtype=float)
    rsurf_Dwall_top    = np.zeros(nsurf_Dwall_top,dtype=float)
    indsurf_Awall      = np.zeros(nsurf_Awall,dtype=int)
    zsurf_Awall        = np.zeros(nsurf_Awall,dtype=float)
    rsurf_Awall        = np.zeros(nsurf_Awall,dtype=float)
    indsurf_FLwall_ver = np.zeros(nsurf_FLwall_ver,dtype=int)
    zsurf_FLwall_ver   = np.zeros(nsurf_FLwall_ver,dtype=float)
    rsurf_FLwall_ver   = np.zeros(nsurf_FLwall_ver,dtype=float)
    indsurf_FLwall_lat = np.zeros(nsurf_FLwall_lat,dtype=int)
    zsurf_FLwall_lat   = np.zeros(nsurf_FLwall_lat,dtype=float)
    rsurf_FLwall_lat   = np.zeros(nsurf_FLwall_lat,dtype=float)
    # Dwall_bot
    for i in range(0,nnodes_Dwall_bot-1):
        isurf = inodes_Dwall_bot[i] + inodes_Dwall_bot[i+1]
        jsurf = jnodes_Dwall_bot[i] + jnodes_Dwall_bot[i+1]
        for k in range(0,n_imp_elems):
            if imp_elems[k,0] == isurf and imp_elems[k,1] == jsurf:
                nsurf_Dwall_bot   = nsurf_Dwall_bot + 1
                nsurf_bound       = nsurf_bound + 1
                indsurf_Dwall_bot = np.append(indsurf_Dwall_bot,k)
                zsurf_Dwall_bot   = np.append(zsurf_Dwall_bot,0.5*(zs[inodes_Dwall_bot[i],jnodes_Dwall_bot[i]]+zs[inodes_Dwall_bot[i+1],jnodes_Dwall_bot[i+1]]))
                rsurf_Dwall_bot   = np.append(rsurf_Dwall_bot,0.5*(rs[inodes_Dwall_bot[i],jnodes_Dwall_bot[i]]+rs[inodes_Dwall_bot[i+1],jnodes_Dwall_bot[i+1]]))
    # Dwall_top
    for i in range(0,nnodes_Dwall_top-1):
        isurf = inodes_Dwall_top[i] + inodes_Dwall_top[i+1]
        jsurf = jnodes_Dwall_top[i] + jnodes_Dwall_top[i+1]
        for k in range(0,n_imp_elems):
            if imp_elems[k,0] == isurf and imp_elems[k,1] == jsurf:
                nsurf_Dwall_top   = nsurf_Dwall_top + 1
                nsurf_bound       = nsurf_bound + 1
                indsurf_Dwall_top = np.append(indsurf_Dwall_top,k)
                zsurf_Dwall_top   = np.append(zsurf_Dwall_top,0.5*(zs[inodes_Dwall_top[i],jnodes_Dwall_top[i]]+zs[inodes_Dwall_top[i+1],jnodes_Dwall_top[i+1]]))
                rsurf_Dwall_top   = np.append(rsurf_Dwall_top,0.5*(rs[inodes_Dwall_top[i],jnodes_Dwall_top[i]]+rs[inodes_Dwall_top[i+1],jnodes_Dwall_top[i+1]]))
    # Awall
    for i in range(0,nnodes_Awall-1):
        isurf = inodes_Awall[i] + inodes_Awall[i+1]
        jsurf = jnodes_Awall[i] + jnodes_Awall[i+1]
        for k in range(0,n_imp_elems):
            if imp_elems[k,0] == isurf and imp_elems[k,1] == jsurf:
                nsurf_Awall   = nsurf_Awall + 1
                nsurf_bound   = nsurf_bound + 1
                indsurf_Awall = np.append(indsurf_Awall,k)
                zsurf_Awall   = np.append(zsurf_Awall,0.5*(zs[inodes_Awall[i],jnodes_Awall[i]]+zs[inodes_Awall[i+1],jnodes_Awall[i+1]]))
                rsurf_Awall   = np.append(rsurf_Awall,0.5*(rs[inodes_Awall[i],jnodes_Awall[i]]+rs[inodes_Awall[i+1],jnodes_Awall[i+1]]))          
    # FLwall_ver
    for i in range(0,nnodes_FLwall_ver-1):
        isurf = inodes_FLwall_ver[i] + inodes_FLwall_ver[i+1]
        jsurf = jnodes_FLwall_ver[i] + jnodes_FLwall_ver[i+1]
        for k in range(0,n_imp_elems):
            if imp_elems[k,0] == isurf and imp_elems[k,1] == jsurf:
                nsurf_FLwall_ver   = nsurf_FLwall_ver + 1
                nsurf_bound        = nsurf_bound + 1
                indsurf_FLwall_ver = np.append(indsurf_FLwall_ver,k)
                zsurf_FLwall_ver   = np.append(zsurf_FLwall_ver,0.5*(zs[inodes_FLwall_ver[i],jnodes_FLwall_ver[i]]+zs[inodes_FLwall_ver[i+1],jnodes_FLwall_ver[i+1]]))
                rsurf_FLwall_ver   = np.append(rsurf_FLwall_ver,0.5*(rs[inodes_FLwall_ver[i],jnodes_FLwall_ver[i]]+rs[inodes_FLwall_ver[i+1],jnodes_FLwall_ver[i+1]]))
    # FLwall_lat
    for i in range(0,nnodes_FLwall_lat-1):
        isurf = inodes_FLwall_lat[i] + inodes_FLwall_lat[i+1]
        jsurf = jnodes_FLwall_lat[i] + jnodes_FLwall_lat[i+1]
        for k in range(0,n_imp_elems):
            if imp_elems[k,0] == isurf and imp_elems[k,1] == jsurf:
                nsurf_FLwall_lat   = nsurf_FLwall_lat + 1
                nsurf_bound        = nsurf_bound + 1
                indsurf_FLwall_lat = np.append(indsurf_FLwall_lat,k)
                zsurf_FLwall_lat   = np.append(zsurf_FLwall_lat,0.5*(zs[inodes_FLwall_lat[i],jnodes_FLwall_lat[i]]+zs[inodes_FLwall_lat[i+1],jnodes_FLwall_lat[i+1]]))
                rsurf_FLwall_lat   = np.append(rsurf_FLwall_lat,0.5*(rs[inodes_FLwall_lat[i],jnodes_FLwall_lat[i]]+rs[inodes_FLwall_lat[i+1],jnodes_FLwall_lat[i+1]]))
    
    # Obtain the arc length along each type of boundary
    sDwall_bot_surf  = np.zeros(nsurf_Dwall_bot,dtype=float)
    sDwall_top_surf  = np.zeros(nsurf_Dwall_top,dtype=float)
    sAwall_surf      = np.zeros(nsurf_Awall,dtype=float)
    sFLwall_ver_surf = np.zeros(nsurf_FLwall_ver,dtype=float)
    sFLwall_lat_surf = np.zeros(nsurf_FLwall_lat,dtype=float)
    for i in range(0,nsurf_Dwall_bot-1):
        z1 = zsurf_Dwall_bot[i]
        r1 = rsurf_Dwall_bot[i]
        z2 = zsurf_Dwall_bot[i+1]
        r2 = rsurf_Dwall_bot[i+1]
        sDwall_bot_surf[i+1] = sDwall_bot_surf[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    zp = zs[int(eta_min),0]
    rp = rs[int(eta_min),0]
    sDwall_bot_surf = sDwall_bot_surf + np.sqrt((zp-zsurf_Dwall_bot[0])**2 + (rp-rsurf_Dwall_bot[0])**2)
    for i in range(0,nsurf_Dwall_top-1):
        z1 = zsurf_Dwall_top[i]
        r1 = rsurf_Dwall_top[i]
        z2 = zsurf_Dwall_top[i+1]
        r2 = rsurf_Dwall_top[i+1]
        sDwall_top_surf[i+1] = sDwall_top_surf[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    zp = zs[int(eta_max),0]
    rp = rs[int(eta_max),0]
    sDwall_top_surf = sDwall_top_surf + np.sqrt((zp-zsurf_Dwall_top[0])**2 + (rp-rsurf_Dwall_top[0])**2)
    for i in range(0,nsurf_Awall-1):
        z1 = zsurf_Awall[i]
        r1 = rsurf_Awall[i]
        z2 = zsurf_Awall[i+1]
        r2 = rsurf_Awall[i+1]
        sAwall_surf[i+1] = sAwall_surf[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    zp = zs[int(eta_min),0]
    rp = rs[int(eta_min),0]
    sAwall_surf = sAwall_surf + np.sqrt((zp-zsurf_Awall[0])**2 + (rp-rsurf_Awall[0])**2)
    for i in range(0,nsurf_FLwall_ver-1):
        z1 = zsurf_FLwall_ver[i]
        r1 = rsurf_FLwall_ver[i]
        z2 = zsurf_FLwall_ver[i+1]
        r2 = rsurf_FLwall_ver[i+1]
        sFLwall_ver_surf[i+1] = sFLwall_ver_surf[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    zp = zs[0,-1]
    rp = rs[0,-1]
    sFLwall_ver_surf = sFLwall_ver_surf + np.sqrt((zp-zsurf_FLwall_ver[0])**2 + (rp-rsurf_FLwall_ver[0])**2)
    for i in range(0,nsurf_FLwall_lat-1):
        z1 = zsurf_FLwall_lat[i]
        r1 = rsurf_FLwall_lat[i]
        z2 = zsurf_FLwall_lat[i+1]
        r2 = rsurf_FLwall_lat[i+1]
        sFLwall_lat_surf[i+1] = sFLwall_lat_surf[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    zp = zs[-1,int(xi_bottom)]
    rp = rs[-1,int(xi_bottom)]
    sFLwall_lat_surf = sFLwall_lat_surf + np.sqrt((zp-zsurf_FLwall_lat[0])**2 + (rp-rsurf_FLwall_lat[0])**2)
    # -------------------------------------------------------------------------

    # Obtain the time vector
    time   = h5_post['times_sim'][:,0]
    dt     = time[1] - time[0]
    nsteps = len(time)
    steps  = h5_post['steps_sim'][:,0]
    
    # Obtain the distribution function bins vectors
    angle_bins_i1 = h5_out['ssD_picS_acc/imp_part_df_ions1/angle_bins'][0:,0]
    ene_bins_i1   = h5_out['ssD_picS_acc/imp_part_df_ions1/ene_bins'][0:,0]
    normv_bins_i1 = h5_out['ssD_picS_acc/imp_part_df_ions1/norm_vel_bins'][0:,0]
    angle_bins_i2 = np.zeros(np.shape(angle_bins_i1),dtype=float)
    ene_bins_i2   = np.zeros(np.shape(ene_bins_i1),dtype=float)
    normv_bins_i2 = np.zeros(np.shape(normv_bins_i1),dtype=float)
    angle_bins_i3 = np.zeros(np.shape(angle_bins_i1),dtype=float)
    ene_bins_i3   = np.zeros(np.shape(ene_bins_i1),dtype=float)
    normv_bins_i3 = np.zeros(np.shape(normv_bins_i1),dtype=float)
    angle_bins_i4 = np.zeros(np.shape(angle_bins_i1),dtype=float)
    ene_bins_i4   = np.zeros(np.shape(ene_bins_i1),dtype=float)
    normv_bins_i4 = np.zeros(np.shape(normv_bins_i1),dtype=float)
    if num_ion_spe == 2:
        angle_bins_i2 = h5_out['ssD_picS_acc/imp_part_df_ions2/angle_bins'][0:,0]
        ene_bins_i2   = h5_out['ssD_picS_acc/imp_part_df_ions2/ene_bins'][0:,0]
        normv_bins_i2 = h5_out['ssD_picS_acc/imp_part_df_ions2/norm_vel_bins'][0:,0]
    elif num_ion_spe == 4:
        angle_bins_i2 = h5_out['ssD_picS_acc/imp_part_df_ions2/angle_bins'][0:,0]
        ene_bins_i2   = h5_out['ssD_picS_acc/imp_part_df_ions2/ene_bins'][0:,0]
        normv_bins_i2 = h5_out['ssD_picS_acc/imp_part_df_ions2/norm_vel_bins'][0:,0]
        angle_bins_i3 = h5_out['ssD_picS_acc/imp_part_df_ions3/angle_bins'][0:,0]
        ene_bins_i3   = h5_out['ssD_picS_acc/imp_part_df_ions3/ene_bins'][0:,0]
        normv_bins_i3 = h5_out['ssD_picS_acc/imp_part_df_ions3/norm_vel_bins'][0:,0]
        angle_bins_i4 = h5_out['ssD_picS_acc/imp_part_df_ions4/angle_bins'][0:,0]
        ene_bins_i4   = h5_out['ssD_picS_acc/imp_part_df_ions4/ene_bins'][0:,0]
        normv_bins_i4 = h5_out['ssD_picS_acc/imp_part_df_ions4/norm_vel_bins'][0:,0]
        
    angle_bins_n1 = h5_out['ssD_picS_acc/imp_part_df_neus1/angle_bins'][0:,0]
    ene_bins_n1   = h5_out['ssD_picS_acc/imp_part_df_neus1/ene_bins'][0:,0]
    normv_bins_n1 = h5_out['ssD_picS_acc/imp_part_df_neus1/norm_vel_bins'][0:,0]
    angle_bins_n2 = np.zeros(np.shape(angle_bins_n1),dtype=float)
    ene_bins_n2   = np.zeros(np.shape(ene_bins_n1),dtype=float)
    normv_bins_n2 = np.zeros(np.shape(normv_bins_n1),dtype=float)
    if num_neu_spe == 3:
        angle_bins_n2 = h5_out['ssD_picS_acc/imp_part_df_neus2/angle_bins'][0:,0]
        ene_bins_n2   = h5_out['ssD_picS_acc/imp_part_df_neus2/ene_bins'][0:,0]
        normv_bins_n2 = h5_out['ssD_picS_acc/imp_part_df_neus2/norm_vel_bins'][0:,0]
        angle_bins_n3 = h5_out['ssD_picS_acc/imp_part_df_neus3/angle_bins'][0:,0]
        ene_bins_n3   = h5_out['ssD_picS_acc/imp_part_df_neus3/ene_bins'][0:,0]
        normv_bins_n3 = h5_out['ssD_picS_acc/imp_part_df_neus3/norm_vel_bins'][0:,0]

    nbins_angle = len(angle_bins_n1)
    nbins_ene   = len(ene_bins_n1)
    nbins_normv = len(normv_bins_n1)
        
    
    # Obtain the data at the at the PIC mesh surface elements
    # Decide if read all timesteps or a particular timestep
    if allsteps_flag == 1:
        print("HET_sims_read_df: reading all steps data...")        
        # Obtain data at the boundary PIC mesh surface elements
        # Data from PIC module
        nQ1_inst_surf         = h5_post['/picS_data/imp_elems_nQ1'][:,2,:]
        nQ1_surf              = h5_post['/picS_data/imp_elems_nQ1'][:,3,:]
        nQ2_inst_surf         = h5_post['/picS_data/imp_elems_nQ2'][:,2,:]
        nQ2_surf              = h5_post['/picS_data/imp_elems_nQ2'][:,3,:]
        dphi_kbc_surf         = h5_post['/picS_data/imp_elems_dphi_kbc'][:,3,:]
        MkQ1_surf             = h5_post['/picS_data/imp_elems_MkQ1'][:,3,:]
        ji1_surf              = h5_post['/picS_data/imp_elems_ion_flux_in1'][:,3,:]*e
        ji2_surf              = np.zeros(np.shape(ji1_surf),dtype=float)
        ji3_surf              = np.zeros(np.shape(ji1_surf),dtype=float)
        ji4_surf              = np.zeros(np.shape(ji1_surf),dtype=float)
        if num_ion_spe == 2:
            ji2_surf          = h5_post['/picS_data/imp_elems_ion_flux_in2'][:,3,:]*2*e
        elif num_ion_spe == 4:
            ji2_surf          = h5_post['/picS_data/imp_elems_ion_flux_in2'][:,3,:]*2*e
            ji3_surf          = h5_post['/picS_data/imp_elems_ion_flux_in3'][:,3,:]*e
            ji4_surf          = h5_post['/picS_data/imp_elems_ion_flux_in4'][:,3,:]*2*e
        ji_surf               = ji1_surf + ji2_surf + ji3_surf + ji4_surf
        gn1_tw_surf           = h5_post['/picS_data/imp_elems_neu_flux_in1'][:,3,:]
        gn1_fw_surf           = h5_post['/picS_data/imp_elems_neu_flux_out1'][:,3,:]
        gn2_tw_surf           = np.zeros(np.shape(gn1_tw_surf),dtype=float)
        gn2_fw_surf           = np.zeros(np.shape(gn1_tw_surf),dtype=float)
        gn3_tw_surf           = np.zeros(np.shape(gn1_tw_surf),dtype=float)
        gn3_fw_surf           = np.zeros(np.shape(gn1_tw_surf),dtype=float)
        if num_neu_spe == 3:
            gn2_tw_surf           = h5_post['/picS_data/imp_elems_neu_flux_in2'][:,3,:]
            gn2_fw_surf           = h5_post['/picS_data/imp_elems_neu_flux_out2'][:,3,:]
            gn3_tw_surf           = h5_post['/picS_data/imp_elems_neu_flux_in3'][:,3,:]
            gn3_fw_surf           = h5_post['/picS_data/imp_elems_neu_flux_out3'][:,3,:]
        gn_tw_surf            = gn1_tw_surf + gn2_tw_surf + gn3_tw_surf
        qi1_tot_wall_surf     = h5_post['/picS_data/imp_elems_ion_ene_flux_in1'][:,3,:]
        qi2_tot_wall_surf     = np.zeros(np.shape(qi1_tot_wall_surf),dtype=float)
        qi3_tot_wall_surf     = np.zeros(np.shape(qi1_tot_wall_surf),dtype=float)
        qi4_tot_wall_surf     = np.zeros(np.shape(qi1_tot_wall_surf),dtype=float)
        if num_ion_spe == 2:
            qi2_tot_wall_surf = h5_post['/picS_data/imp_elems_ion_ene_flux_in2'][:,3,:]
        elif num_ion_spe == 4:
            qi2_tot_wall_surf = h5_post['/picS_data/imp_elems_ion_ene_flux_in2'][:,3,:]
            qi3_tot_wall_surf = h5_post['/picS_data/imp_elems_ion_ene_flux_in3'][:,3,:]
            qi4_tot_wall_surf = h5_post['/picS_data/imp_elems_ion_ene_flux_in4'][:,3,:]
        qi_tot_wall_surf      = qi1_tot_wall_surf + qi2_tot_wall_surf + qi3_tot_wall_surf + qi4_tot_wall_surf
        qn1_tw_surf           = h5_post['/picS_data/imp_elems_neu_ene_flux_in1'][:,3,:]
        qn1_fw_surf           = h5_post['/picS_data/imp_elems_neu_ene_flux_out1'][:,3,:]
        qn2_tw_surf           = np.zeros(np.shape(qn1_tw_surf),dtype=float)
        qn2_fw_surf           = np.zeros(np.shape(qn1_tw_surf),dtype=float)
        qn3_tw_surf           = np.zeros(np.shape(qn1_tw_surf),dtype=float)
        qn3_fw_surf           = np.zeros(np.shape(qn1_tw_surf),dtype=float)
        if num_neu_spe == 3:
            qn2_tw_surf           = h5_post['/picS_data/imp_elems_neu_ene_flux_in2'][:,3,:]
            qn2_fw_surf           = h5_post['/picS_data/imp_elems_neu_ene_flux_out2'][:,3,:]
            qn3_tw_surf           = h5_post['/picS_data/imp_elems_neu_ene_flux_in3'][:,3,:]
            qn3_fw_surf           = h5_post['/picS_data/imp_elems_neu_ene_flux_out3'][:,3,:]
        qn_tot_wall_surf      = qn1_tw_surf + qn2_tw_surf + qn3_tw_surf
        imp_ene_i1_surf       = h5_post['/picS_data/imp_elems_ion_imp_ene1'][:,3,:]
        imp_ene_i2_surf       = np.zeros(np.shape(imp_ene_i1_surf),dtype=float)
        imp_ene_i3_surf       = np.zeros(np.shape(imp_ene_i1_surf),dtype=float)
        imp_ene_i4_surf       = np.zeros(np.shape(imp_ene_i1_surf),dtype=float)
        if num_ion_spe == 2:
            imp_ene_i2_surf   = h5_post['/picS_data/imp_elems_ion_imp_ene2'][:,3,:]
        elif num_ion_spe == 4:
            imp_ene_i2_surf   = h5_post['/picS_data/imp_elems_ion_imp_ene2'][:,3,:]
            imp_ene_i3_surf   = h5_post['/picS_data/imp_elems_ion_imp_ene3'][:,3,:]
            imp_ene_i4_surf   = h5_post['/picS_data/imp_elems_ion_imp_ene4'][:,3,:]
        imp_ene_n1_surf       = h5_post['/picS_data/imp_elems_neu_imp_ene1'][:,3,:]
        imp_ene_n2_surf       = np.zeros(np.shape(imp_ene_n1_surf),dtype=float)
        imp_ene_n3_surf       = np.zeros(np.shape(imp_ene_n1_surf),dtype=float)
        if num_neu_spe == 3:
            imp_ene_n2_surf       = h5_post['/picS_data/imp_elems_neu_imp_ene2'][:,3,:]
            imp_ene_n3_surf       = h5_post['/picS_data/imp_elems_neu_imp_ene3'][:,3,:]
        # Obtain the distribution functions data at important elements for all timesteps
        angle_df_i1     = np.zeros((n_imp_elems,nbins_angle,nsteps),dtype=float)
        ene_df_i1       = np.zeros((n_imp_elems,nbins_ene,nsteps),dtype=float)
        normv_df_i1     = np.zeros((n_imp_elems,nbins_normv,nsteps),dtype=float)
        ene_angle_df_i1 = np.zeros((n_imp_elems,nbins_ene,nbins_angle,nsteps),dtype=float)
        for k in range(0,nsteps):
            angle_df_i1[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_angle_ions1"][0:,0:]
            ene_df_i1[:,:,k]         = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_ions1"][0:,0:]
            normv_df_i1[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_norm_vel_ions1"][0:,0:]
            ene_angle_df_i1[:,:,:,k] = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_angle_ions1"][0:,0:,0:]
        angle_df_i2     = np.zeros((n_imp_elems,nbins_angle,nsteps),dtype=float)
        ene_df_i2       = np.zeros((n_imp_elems,nbins_ene,nsteps),dtype=float)
        normv_df_i2     = np.zeros((n_imp_elems,nbins_normv,nsteps),dtype=float)
        ene_angle_df_i2 = np.zeros((n_imp_elems,nbins_ene,nbins_angle,nsteps),dtype=float)
        angle_df_i3     = np.zeros((n_imp_elems,nbins_angle,nsteps),dtype=float)
        ene_df_i3       = np.zeros((n_imp_elems,nbins_ene,nsteps),dtype=float)
        normv_df_i3     = np.zeros((n_imp_elems,nbins_normv,nsteps),dtype=float)
        ene_angle_df_i3 = np.zeros((n_imp_elems,nbins_ene,nbins_angle,nsteps),dtype=float)
        angle_df_i4     = np.zeros((n_imp_elems,nbins_angle,nsteps),dtype=float)
        ene_df_i4       = np.zeros((n_imp_elems,nbins_ene,nsteps),dtype=float)
        normv_df_i4     = np.zeros((n_imp_elems,nbins_normv,nsteps),dtype=float)
        ene_angle_df_i4 = np.zeros((n_imp_elems,nbins_ene,nbins_angle,nsteps),dtype=float)
        if num_ion_spe == 2:
            for k in range(0,nsteps):
                angle_df_i2[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_angle_ions2"][0:,0:]
                ene_df_i2[:,:,k]         = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_ions2"][0:,0:]
                normv_df_i2[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_norm_vel_ions2"][0:,0:]
                ene_angle_df_i2[:,:,:,k] = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_angle_ions2"][0:,0:,0:]
        elif num_ion_spe == 4:
            for k in range(0,nsteps):
                angle_df_i2[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_angle_ions2"][0:,0:]
                ene_df_i2[:,:,k]         = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_ions2"][0:,0:]
                normv_df_i2[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_norm_vel_ions2"][0:,0:]
                ene_angle_df_i2[:,:,:,k] = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_angle_ions2"][0:,0:,0:]
                angle_df_i3[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_angle_ions3"][0:,0:]
                ene_df_i3[:,:,k]         = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_ions3"][0:,0:]
                normv_df_i3[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_norm_vel_ions3"][0:,0:]
                ene_angle_df_i3[:,:,:,k] = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_angle_ions3"][0:,0:,0:]
                angle_df_i4[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_angle_ions4"][0:,0:]
                ene_df_i4[:,:,k]         = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_ions4"][0:,0:]
                normv_df_i4[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_norm_vel_ions4"][0:,0:]
                ene_angle_df_i4[:,:,:,k] = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_angle_ions4"][0:,0:,0:]
                
        angle_df_n1     = np.zeros((n_imp_elems,nbins_angle,nsteps),dtype=float)
        ene_df_n1       = np.zeros((n_imp_elems,nbins_ene,nsteps),dtype=float)
        normv_df_n1     = np.zeros((n_imp_elems,nbins_normv,nsteps),dtype=float)
        ene_angle_df_n1 = np.zeros((n_imp_elems,nbins_ene,nbins_angle,nsteps),dtype=float)
        angle_df_n2     = np.zeros((n_imp_elems,nbins_angle,nsteps),dtype=float)
        ene_df_n2       = np.zeros((n_imp_elems,nbins_ene,nsteps),dtype=float)
        normv_df_n2     = np.zeros((n_imp_elems,nbins_normv,nsteps),dtype=float)
        ene_angle_df_n2 = np.zeros((n_imp_elems,nbins_ene,nbins_angle,nsteps),dtype=float)
        angle_df_n3     = np.zeros((n_imp_elems,nbins_angle,nsteps),dtype=float)
        ene_df_n3       = np.zeros((n_imp_elems,nbins_ene,nsteps),dtype=float)
        normv_df_n3     = np.zeros((n_imp_elems,nbins_normv,nsteps),dtype=float)
        ene_angle_df_n3 = np.zeros((n_imp_elems,nbins_ene,nbins_angle,nsteps),dtype=float)
        for k in range(0,nsteps):
            angle_df_n1[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_angle_neus1"][0:,0:]
            ene_df_n1[:,:,k]         = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_neus1"][0:,0:]
            normv_df_n1[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_norm_vel_neus1"][0:,0:]
            ene_angle_df_n1[:,:,:,k] = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_angle_neus1"][0:,0:,0:]
        if num_neu_spe == 3:
            for k in range(0,nsteps):
                angle_df_n2[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_angle_neus2"][0:,0:]
                ene_df_n2[:,:,k]         = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_neus2"][0:,0:]
                normv_df_n2[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_norm_vel_neus2"][0:,0:]
                ene_angle_df_n2[:,:,:,k] = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_angle_neus2"][0:,0:,0:]
                angle_df_n3[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_angle_neus3"][0:,0:]
                ene_df_n3[:,:,k]         = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_neus3"][0:,0:]
                normv_df_n3[:,:,k]       = h5_post['df_data/t'+str(k)+"/imp_part_df_norm_vel_neus3"][0:,0:]
                ene_angle_df_n3[:,:,:,k] = h5_post['df_data/t'+str(k)+"/imp_part_df_ene_angle_neus3"][0:,0:,0:]
        
        
        
    elif allsteps_flag == 0:
        print("HET_sims_read_bound: reading all steps data...")        
        # Obtain data at the boundary PIC mesh surface elements
        # Data from PIC module
        nQ1_inst_surf         = h5_post['/picS_data/imp_elems_nQ1'][:,2,timestep]
        nQ1_surf              = h5_post['/picS_data/imp_elems_nQ1'][:,3,timestep]
        nQ2_inst_surf         = h5_post['/picS_data/imp_elems_nQ2'][:,2,timestep]
        nQ2_surf              = h5_post['/picS_data/imp_elems_nQ2'][:,3,timestep]
        dphi_kbc_surf         = h5_post['/picS_data/imp_elems_dphi_kbc'][:,3,timestep]
        MkQ1_surf             = h5_post['/picS_data/imp_elems_MkQ1'][:,3,timestep]
        ji1_surf              = h5_post['/picS_data/imp_elems_ion_flux_in1'][:,3,timestep]*e
        ji2_surf              = np.zeros(np.shape(ji1_surf),dtype=float)
        if num_ion_spe == 2:
            ji2_surf          = h5_post['/picS_data/imp_elems_ion_flux_in2'][:,3,timestep]*2*e
        ji_surf               = ji1_surf + ji2_surf
        gn1_tw_surf           = h5_post['/picS_data/imp_elems_neu_flux_in1'][:,3,timestep]
        gn1_fw_surf           = h5_post['/picS_data/imp_elems_neu_flux_out1'][:,3,timestep]
        qi1_tot_wall_surf     = h5_post['/picS_data/imp_elems_ion_ene_flux_in1'][:,3,timestep]
        qi2_tot_wall_surf     = np.zeros(np.shape(qi1_tot_wall_surf),dtype=float)
        if num_ion_spe == 2:
            qi2_tot_wall_surf = h5_post['/picS_data/imp_elems_ion_ene_flux_in2'][:,3,timestep]
        qi_tot_wall_surf      = qi1_tot_wall_surf + qi2_tot_wall_surf
        qn1_tw_surf           = h5_post['/picS_data/imp_elems_neu_ene_flux_in1'][:,3,timestep]
        qn1_fw_surf           = h5_post['/picS_data/imp_elems_neu_ene_flux_out1'][:,3,timestep]
        imp_ene_i1_surf       = h5_post['/picS_data/imp_elems_ion_imp_ene1'][:,3,timestep]
        imp_ene_i2_surf       = np.zeros(np.shape(imp_ene_i1_surf),dtype=float)
        if num_ion_spe == 2:
            imp_ene_i2_surf   = h5_post['/picS_data/imp_elems_ion_imp_ene2'][:,3,timestep]
        imp_ene_n1_surf       = h5_post['/picS_data/imp_elems_neu_imp_ene1'][:,3,timestep]
        
        # Obtain the distribution functions data at important elements for the particular timestep
        if timestep == -1:
            timestep = nsteps - 1
        angle_df_i1 = h5_post['df_data/t'+str(timestep)+"/imp_part_df_angle_ions1"][0:,0:]
        ene_df_i1   = h5_post['df_data/t'+str(timestep)+"/imp_part_df_ene_ions1"][0:,0:]
        normv_df_i1 = h5_post['df_data/t'+str(timestep)+"/imp_part_df_norm_vel_ions1"][0:,0:]
        angle_df_i2 = np.zeros(np.shape(angle_df_i1),dtype=float)
        ene_df_i2   = np.zeros(np.shape(ene_df_i1),dtype=float)
        normv_df_i2 = np.zeros(np.shape(normv_df_i1),dtype=float)
        if num_ion_spe == 2:
            angle_df_i2 = h5_post['df_data/t'+str(timestep)+"/imp_part_df_angle_ions2"][0:,0:]
            ene_df_i2   = h5_post['df_data/t'+str(timestep)+"/imp_part_df_ene_ions2"][0:,0:]
            normv_df_i2 = h5_post['df_data/t'+str(timestep)+"/imp_part_df_norm_vel_ions2"][0:,0:]

        angle_df_n1 = h5_post['df_data/t'+str(timestep)+"/imp_part_df_angle_neus1"][0:,0:]
        ene_df_n1   = h5_post['df_data/t'+str(timestep)+"/imp_part_df_ene_neus1"][0:,0:]
        normv_df_n1 = h5_post['df_data/t'+str(timestep)+"/imp_part_df_norm_vel_neus1"][0:,0:]



    
    return[num_ion_spe,num_neu_spe,points,zs,rs,zscells,rscells,dims,nodes_flag,
           cells_flag,cells_vol,volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,
           eta_max,eta_min,xi_top,xi_bottom,time,steps,dt,nsteps,sc_bot,sc_top,
           Lplume_bot,Lplume_top,Lchamb_bot,Lchamb_top,Lanode,Lfreeloss_ver,
           Lfreeloss_lat,Lfreeloss,Laxis,
           
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
           sAwall_surf,sFLwall_ver_surf,sFLwall_lat_surf,imp_elems,norm_vers,
           
           nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,
           MkQ1_surf,ji1_surf,ji2_surf,ji3_surf,ji4_surf,ji_surf,
           gn1_tw_surf,gn1_fw_surf,gn2_tw_surf,gn2_fw_surf,gn3_tw_surf,gn3_fw_surf,
           gn_tw_surf,qi1_tot_wall_surf,qi2_tot_wall_surf,qi3_tot_wall_surf,
           qi4_tot_wall_surf,qi_tot_wall_surf,qn1_tw_surf,qn1_fw_surf,qn2_tw_surf,
           qn2_fw_surf,qn3_tw_surf,qn3_fw_surf,qn_tot_wall_surf,imp_ene_i1_surf,
           imp_ene_i2_surf,imp_ene_i3_surf,imp_ene_i4_surf,imp_ene_n1_surf,
           imp_ene_n2_surf,imp_ene_n3_surf,
           
           angle_bins_i1,ene_bins_i1,normv_bins_i1,angle_bins_i2,ene_bins_i2,
           normv_bins_i2,angle_bins_i3,ene_bins_i3,normv_bins_i3,angle_bins_i4,
           ene_bins_i4,normv_bins_i4,angle_bins_n1,ene_bins_n1,normv_bins_n1,
           angle_bins_n2,ene_bins_n2,normv_bins_n2,angle_bins_n3,ene_bins_n3,
           normv_bins_n3,nbins_angle,nbins_ene,nbins_normv,
           
           angle_df_i1,ene_df_i1,normv_df_i1,ene_angle_df_i1,
           angle_df_i2,ene_df_i2,normv_df_i2,ene_angle_df_i2,
           angle_df_i3,ene_df_i3,normv_df_i3,ene_angle_df_i3,
           angle_df_i4,ene_df_i4,normv_df_i4,ene_angle_df_i4,
           angle_df_n1,ene_df_n1,normv_df_n1,ene_angle_df_n1,
           angle_df_n2,ene_df_n2,normv_df_n2,ene_angle_df_n2,
           angle_df_n3,ene_df_n3,normv_df_n3,ene_angle_df_n3]

    
if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 

    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))
    
#    sim_name = "../../../Sr_sims_files/SPT100_orig_tmtetq2_Vd300_test_rel"
#    sim_name = "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_Kr"
#    sim_name = "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd300"
#    sim_name = "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_last" 
    sim_name = "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_71d0dcb"

    timestep         = -1
    allsteps_flag    = 1
    
    oldpost_sim      = 3
    oldsimparams_sim = 8
    
    
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
       sAwall_surf,sFLwall_ver_surf,sFLwall_lat_surf,imp_elems,norm_vers,
       
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
    
    
        
    if (nnodes_bound == nnodes_Dwall_bot + nnodes_Axis - 1 + nnodes_FLwall_ver -1 + nnodes_FLwall_lat - 1 + nnodes_Dwall_top - 1 + nnodes_Awall - 2):
        print("PIC mesh boundary nodes read correctly")
        print("nnodes_bound      = "+str(nnodes_bound))
        print("nnodes_Dwall_bot  = "+str(nnodes_Dwall_bot))
        print("nnodes_Dwall_top  = "+str(nnodes_Dwall_top))
        print("nnodes_Awall      = "+str(nnodes_Awall))
        print("nnodes_FLwall_ver = "+str(nnodes_FLwall_ver))
        print("nnodes_FLwall_lat = "+str(nnodes_FLwall_lat))
        print("nnodes_Axis       = "+str(nnodes_Axis))
    
    if (nsurf_bound == nsurf_Dwall_bot + nsurf_FLwall_ver + nsurf_FLwall_lat + nsurf_Dwall_top + nsurf_Awall):
        print("PIC mesh boundary surface elements read correctly")
        print("nsurf_bound      = "+str(nsurf_bound))
        print("nsurf_Dwall_bot  = "+str(nsurf_Dwall_bot))
        print("nsurf_Dwall_top  = "+str(nsurf_Dwall_top))
        print("nsurf_Awall      = "+str(nsurf_Awall))
        print("nsurf_FLwall_ver = "+str(nsurf_FLwall_ver))
        print("nsurf_FLwall_lat = "+str(nsurf_FLwall_lat))
        
    
    # Plotting part -----------------------------------------------------------
    line_width_boundary = 1
    line_width_bf       = 0.7
    line_width          = 1
    marker_size         = 3
    marker_size_bf      = 5
    marker_size_pic_bn  = 6
    
    # Extra colors
    orange ='#FD6A02'            
    gold   ='#F9A602'
    brown  ='#8B4000'
    silver ='#C0C0C0'
    
    
    zs_plot = np.copy(zs)
    rs_plot = np.copy(rs)
    for i in range(0,dims[0]):
        for j in range(0,dims[1]):
            if nodes_flag[i,j] == 0:
                rs_plot[i,j] = np.NaN
                zs_plot[i,j] = np.NaN
        
    plt.figure("PIC mesh and MFAM boundary faces")
    plt.plot(zs_plot,rs_plot,'ko-',linewidth=line_width,markersize = marker_size)
    plt.plot(zs_plot.transpose(),rs_plot.transpose(),'ko-',linewidth=line_width,markersize = marker_size)
    # Plot points defining mesh boundary
    # Inner wall
    plt.plot(points[0:3,0],points[0:3,1],'ro-',linewidth=line_width_boundary,markersize = marker_size)
    # Axis r = 0
    plt.plot(points[2:4,0],points[2:4,1],'mo-',linewidth=line_width_boundary,markersize = marker_size)
    # Free loss
    plt.plot(points[3:6,0],points[3:6,1],'bo-',linewidth=line_width_boundary,markersize = marker_size)
    # Outer wall
    plt.plot(points[5::,0],points[5::,1],'ro-',linewidth=line_width_boundary,markersize = marker_size)
    # Injection
    plt.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'go-',linewidth=line_width_boundary,markersize = marker_size)
    # Plot all mesh boundary points and domain points
    for i in range(0,dims[0]):
        for j in range(0,dims[1]):
            if nodes_flag[i,j] == -1:
                if i == 0:
                    # Axis r = 0
                    plt.plot(zs[i,j],rs[i,j],'mo',markeredgecolor='m',markersize = marker_size)
                elif j == dims[1] - 1 or i == dims[0] - 1:
                    # Free loss
                    plt.plot(zs[i,j],rs[i,j],'bo',markeredgecolor='b',markersize = marker_size)
                elif j == xi_top or i == eta_max or j == xi_bottom or i == eta_min:
                    # Material wall
                    plt.plot(zs[i,j],rs[i,j],'ro',markeredgecolor='r',markersize = marker_size)
                elif j == 0:
                    # Injection
                    plt.plot(zs[i,j],rs[i,j],'go',markeredgecolor='g',markersize = marker_size)
            if nodes_flag[i,j] == 1:
                plt.plot(zs[i,j],rs[i,j],'ko',markersize = marker_size)
#    # Plot the boundary faces centers for each type of boundary
#    # Dwall_bot
#    plt.plot(zfaces_Dwall_bot,rfaces_Dwall_bot,'cx-',linewidth = line_width_bf,markersize = marker_size_bf)
#    # Dwall_top
#    plt.plot(zfaces_Dwall_top,rfaces_Dwall_top,'cx-',linewidth = line_width_bf,markersize = marker_size_bf)
#    # Awall
#    plt.plot(zfaces_Awall,rfaces_Awall,'mx-',linewidth = line_width_bf,markersize = marker_size_bf)
#    # FLwall_ver
#    plt.plot(zfaces_FLwall_ver,rfaces_FLwall_ver,'yx-',linewidth = line_width_bf,markersize = marker_size_bf)
#    # FLwall_lat
#    plt.plot(zfaces_FLwall_lat,rfaces_FLwall_lat,'x-',color = orange,linewidth = line_width_bf,markersize = marker_size_bf)
#    # Axis
#    plt.plot(zfaces_Axis,rfaces_Axis,'x-',color = brown,linewidth = line_width_bf,markersize = marker_size_bf)
    # Plot the PIC mesh boundary nodes for each type of boundary
    # Dwall_bot
    plt.plot(zs[inodes_Dwall_bot,jnodes_Dwall_bot],rs[inodes_Dwall_bot,jnodes_Dwall_bot],'rD',markersize = marker_size_pic_bn)
    # Dwall_top
    plt.plot(zs[inodes_Dwall_top,jnodes_Dwall_top],rs[inodes_Dwall_top,jnodes_Dwall_top],'rD',markersize = marker_size_pic_bn)
    # Awall
    plt.plot(zs[inodes_Awall,jnodes_Awall],rs[inodes_Awall,jnodes_Awall],'gP',markersize = marker_size_pic_bn)
    # FLwall_ver
    plt.plot(zs[inodes_FLwall_ver,jnodes_FLwall_ver],rs[inodes_FLwall_ver,jnodes_FLwall_ver],'bp',markersize = marker_size_pic_bn)
    # FLwall_lat
    plt.plot(zs[inodes_FLwall_lat,jnodes_FLwall_lat],rs[inodes_FLwall_lat,jnodes_FLwall_lat],'cp',markersize = marker_size_pic_bn)
    # Axis
    plt.plot(zs[inodes_Axis,jnodes_Axis],rs[inodes_Axis,jnodes_Axis],'mp',markersize = marker_size_pic_bn)
    # Plot the PIC mesh boundary surface elements centers for each type of boundary
    # Dwall_bot
    plt.plot(zsurf_Dwall_bot,rsurf_Dwall_bot,'ko',markersize = marker_size)
    # Dwall_top
    plt.plot(zsurf_Dwall_top,rsurf_Dwall_top,'ko',markersize = marker_size)
    # Awall
    plt.plot(zsurf_Awall,rsurf_Awall,'ko',markersize = marker_size)
    # FLwall_ver
    plt.plot(zsurf_FLwall_ver,rsurf_FLwall_ver,'ko',markersize = marker_size)
    # FLwall_lat
    plt.plot(zsurf_FLwall_lat,rsurf_FLwall_lat,'ko',markersize = marker_size)
    
    tstep    = 1000
    ind_elem = 4
    e  = 1.6021766E-19
    # 1D distribution function checks
    # Computation of particle fluxes to the walls
    Iangle_i1 = 0
    Iene_i1   = 0
    Inormv_i1 = 0
    for k in range(0,nbins_angle-1):
        dangle = angle_bins_i1[k+1]-angle_bins_i1[k]
        Iangle_i1 = Iangle_i1 + 0.5*(angle_df_i1[ind_elem,k,tstep] + angle_df_i1[ind_elem,k+1,tstep])*dangle
    for k in range(0,nbins_ene-1):
        dene = ene_bins_i1[k+1]-ene_bins_i1[k]
        Iene_i1 = Iene_i1 + 0.5*(ene_df_i1[ind_elem,k,tstep] + ene_df_i1[ind_elem,k+1,tstep])*dene
    for k in range(0,nbins_normv-1):
        dnormv = normv_bins_i1[k+1]-normv_bins_i1[k]
        Inormv_i1 = Inormv_i1 + 0.5*(normv_df_i1[ind_elem,k,tstep] + normv_df_i1[ind_elem,k+1,tstep])*dnormv
    
    
    err_Iangle_i1 = np.abs(Iangle_i1 - ji1_surf[ind_elem,tstep]/e)/(ji1_surf[ind_elem,tstep]/e)
    err_Iene_i1   = np.abs(Iene_i1 - ji1_surf[ind_elem,tstep]/e)/(ji1_surf[ind_elem,tstep]/e)
    err_Inormv_i1 = np.abs(Inormv_i1 - ji1_surf[ind_elem,tstep]/e)/(ji1_surf[ind_elem,tstep]/e)
    
    print("Iangle_i1 = %15.8e" %Iangle_i1)
    print("Iene_i1   = %15.8e" %Iene_i1)
    print("Inormv_i1 = %15.8e" %Inormv_i1)
    
    print("err Iangle_i1 = %15.8e" %err_Iangle_i1)
    print("err Iene_i1   = %15.8e" %err_Iene_i1)
    print("err Inormv_i1 = %15.8e" %err_Inormv_i1)
    
    # Computation or mean values of impacting angle, energy and normal velocity
    angle_mean_i1   = 0
    imp_ene_mean_i1 = 0 
    for k in range(0,nbins_angle-1):
        dangle = angle_bins_i1[k+1]-angle_bins_i1[k]
        angle_mean_i1 = angle_mean_i1 + 0.5*(angle_bins_i1[k]*angle_df_i1[ind_elem,k,tstep] + angle_bins_i1[k+1]*angle_df_i1[ind_elem,k+1,tstep])*dangle
    for k in range(0,nbins_ene-1):
        dene = ene_bins_i1[k+1]-ene_bins_i1[k]
        imp_ene_mean_i1 = imp_ene_mean_i1 + 0.5*(ene_bins_i1[k]*ene_df_i1[ind_elem,k,tstep] + ene_bins_i1[k+1]*ene_df_i1[ind_elem,k+1,tstep])*dene
    imp_ene_mean_i1 = imp_ene_mean_i1/Iene_i1
    
    err_imp_ene_i1 = np.abs(imp_ene_mean_i1 - imp_ene_i1_surf[ind_elem,tstep]/e)/(imp_ene_i1_surf[ind_elem,tstep]/e)
    
    print("imp_ene_mean_i1 = %15.8e" %imp_ene_mean_i1)
    print("err imp_ene_i1  = %15.8e" %err_imp_ene_i1)