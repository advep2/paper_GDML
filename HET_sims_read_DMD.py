# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:16:42 2018

@author: adrian

############################################################################
Description:    This python script reads data from HET sims
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
               NOTE: Electron-fluid module collisions (new treatment)
                     ids_collisions_e = 1,4,2,2,5,2,5       
                     # Vector containing the IDs of the collisions to be
                     # simulated for electrons:
                     # 1 - Elastic collisions
                     # 2 - Ionization 
                     # 3 - Recombination
            	     # 4 - Excitation
                     # 5 - Coulomb (model implemented in the code)
                     # 6 - Rotational excitation
                     # 7 - Vibrational excitation
                     # 8 - Dissociation
                     # 9 - Dissociative ionization
############################################################################
Output:        1) Read variables
"""


def HET_sims_read_DMD(path_simstate_inp,path_simstate_out,path_postdata_out,path_simparams_inp,
                      path_picM,allsteps_flag,timestep,read_inst_data,read_part_lists,read_flag,
                      oldpost_sim,oldsimparams_sim):
    
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
    
    print("HET_sims_read: reading sim_params.inp...")
    # Open sim_params.inp file
    r = open(path_simparams_inp,'r')
    lines = r.readlines()
    r.close() 
    
    # Obtain number of species and number of particles per cell
    if oldsimparams_sim == 0:
        # OLD SIMS (THESIS SIMS (rev 675ef12) and following revs 387df40 and 67f2082)
        line_num_ion_spe            = 43
        line_num_neu_spe            = 44
        line_n_mp_cell_i            = 45
        line_n_mp_cell_n            = 46
        line_n_mp_cell_i_min        = 47
        line_n_mp_cell_i_max        = 48
        line_n_mp_cell_n_min        = 49
        line_n_mp_cell_n_max        = 50
        line_min_ion_plasma_density = 62
        line_m_A                    = 65 
        line_spec_refl_prob         = 66 
        line_ene_bal                = 13
        line_nsteps_eFld            = 29
        line_override_cathode_e     = 97
        line_new_cathode_e          = 98
        line_T_cath                 = 137
        line_ncollisions            = 144
    elif oldsimparams_sim == 1:
        # AFTER THESIS SIMS 1 
        line_num_ion_spe            = 51
        line_num_neu_spe            = 52
        line_n_mp_cell_i            = 53
        line_n_mp_cell_n            = 54
        line_n_mp_cell_i_min        = 57
        line_n_mp_cell_i_max        = 58
        line_n_mp_cell_n_min        = 59
        line_n_mp_cell_n_max        = 60
        line_min_ion_plasma_density = 70
        line_m_A                    = 73 
        line_spec_refl_prob         = 74 
        line_ene_bal                = 13
        line_nsteps_eFld            = 30
        line_override_cathode_e     = 105
        line_new_cathode_e          = 106
        line_T_cath                 = 151
        line_ncollisions            = 158
    elif oldsimparams_sim == 2:
        # AFTER THESIS SIMS 2 (rev 2a975ad)
        line_num_ion_spe            = 51
        line_num_neu_spe            = 52
        line_n_mp_cell_i            = 53
        line_n_mp_cell_n            = 54
        line_n_mp_cell_i_min        = 57
        line_n_mp_cell_i_max        = 58
        line_n_mp_cell_n_min        = 59
        line_n_mp_cell_n_max        = 60
        line_min_ion_plasma_density = 70
        line_m_A                    = 73 
        line_spec_refl_prob         = 74 
        line_ene_bal                = 13
        line_nsteps_eFld            = 30
        line_override_cathode_e     = 105
        line_new_cathode_e          = 106
        line_T_cath                 = 152
        line_ncollisions            = 159
    elif oldsimparams_sim == 3:
        # AFTER THESIS SIMS 2 (rev 1085401)
        line_num_ion_spe            = 51
        line_num_neu_spe            = 52
        line_n_mp_cell_i            = 53
        line_n_mp_cell_n            = 54
        line_n_mp_cell_i_min        = 57
        line_n_mp_cell_i_max        = 58
        line_n_mp_cell_n_min        = 59
        line_n_mp_cell_n_max        = 60
        line_min_ion_plasma_density = 70
        line_m_A                    = 73 
        line_ene_bal                = 13
        line_nsteps_eFld            = 30
        line_override_cathode_e     = 104
        line_new_cathode_e          = 105
        line_T_cath                 = 151
        line_ncollisions            = 158
    elif oldsimparams_sim == 4:
        # AFTER THESIS SIMS 2 (rev 6a2fb59)
        line_num_ion_spe            = 51
        line_num_neu_spe            = 52
        line_n_mp_cell_i            = 53
        line_n_mp_cell_n            = 54
        line_n_mp_cell_i_min        = 57
        line_n_mp_cell_i_max        = 58
        line_n_mp_cell_n_min        = 59
        line_n_mp_cell_n_max        = 60
        line_min_ion_plasma_density = 70
        line_m_A                    = 73 
        line_ene_bal                = 13
        line_nsteps_eFld            = 30
        line_override_cathode_e     = 104
        line_new_cathode_e          = 105
        line_T_cath                 = 152
        line_ncollisions            = 159
    elif oldsimparams_sim == 5:
        # AFTER THESIS SIMS 2 (rev e462529)
        line_num_ion_spe            = 53
        line_num_neu_spe            = 54
        line_n_mp_cell_i            = 55
        line_n_mp_cell_n            = 56
        line_n_mp_cell_i_min        = 59
        line_n_mp_cell_i_max        = 60
        line_n_mp_cell_n_min        = 61
        line_n_mp_cell_n_max        = 62
        line_min_ion_plasma_density = 72
        line_m_A                    = 75
        line_ene_bal                = 13
        line_nsteps_eFld            = 30
        line_override_cathode_e     = 106
        line_new_cathode_e          = 107
        line_T_cath                 = 154
        line_ncollisions            = 161
    elif oldsimparams_sim == 6:
        # AFTER THESIS SIMS 2 (rev 7aba29d)
        line_num_ion_spe            = 53
        line_num_neu_spe            = 54
        line_n_mp_cell_i            = 55
        line_n_mp_cell_n            = 56
        line_n_mp_cell_i_min        = 59
        line_n_mp_cell_i_max        = 60
        line_n_mp_cell_n_min        = 61
        line_n_mp_cell_n_max        = 62
        line_min_ion_plasma_density = 72
        line_m_A                    = 75
        line_ene_bal                = 13
        line_nsteps_eFld            = 30
        line_override_cathode_e     = 106
        line_new_cathode_e          = 107
        line_T_cath                 = 158
        line_ncollisions            = 165
    elif oldsimparams_sim == 7:
        # AFTER THESIS SIMS 2 (rev 57847d1,d184b49) (LAST CHEOPS REPORT: SAFRAN T1,T2)
        line_num_ion_spe            = 53
        line_num_neu_spe            = 54
        line_Z_ion_spe              = 55
        line_n_mp_cell_i            = 56
        line_n_mp_cell_n            = 57
        line_n_mp_cell_i_min        = 60
        line_n_mp_cell_i_max        = 61
        line_n_mp_cell_n_min        = 62
        line_n_mp_cell_n_max        = 63
        line_min_ion_plasma_density = 73
        line_m_A                    = 76
        line_ene_bal                = 13
        line_nsteps_eFld            = 30
        line_override_cathode_e     = 106
        line_new_cathode_e          = 107
        line_T_cath                 = 158
        line_ncollisions            = 170
        line_ncollisions_e          = 160
        line_ids_collisions_e       = 161
        line_coll_spe_e             = 163
        line_out_coll_spe_e         = 164
    elif oldsimparams_sim == 8:
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
    elif oldsimparams_sim == 9:
        # SIMS from commit 3f5a50e (after final report CHEOPS)
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
        line_T_cath                 = 184
        line_ncollisions            = 196
        line_ncollisions_e          = 186
        line_ids_collisions_e       = 187
        line_coll_spe_e             = 189
        line_out_coll_spe_e         = 190
    elif oldsimparams_sim == 10:
        # SIMS from commit 18345cd (after time-varying V_ps and mA)
        line_num_ion_spe            = 58
        line_num_neu_spe            = 59
        line_Z_ion_spe              = 60
        line_n_mp_cell_i            = 61
        line_n_mp_cell_n            = 62
        line_n_mp_cell_i_min        = 65
        line_n_mp_cell_i_max        = 66
        line_n_mp_cell_n_min        = 67
        line_n_mp_cell_n_max        = 68
        line_min_ion_plasma_density = 78
        line_m_A                    = 81
        line_ene_bal                = 13
        line_nsteps_eFld            = 33
        line_override_cathode_e     = 136
        line_new_cathode_e          = 137
        line_T_cath                 = 186
        line_ncollisions            = 198
        line_ncollisions_e          = 188
        line_ids_collisions_e       = 189
        line_coll_spe_e             = 191
        line_out_coll_spe_e         = 192
        line_interp_eFld2PIC_alldata = 16
        
        
    num_ion_spe = int(lines[line_num_ion_spe][lines[line_num_ion_spe].find('=')+1:lines[line_num_ion_spe].find('\n')])
    num_neu_spe = int(lines[line_num_neu_spe][lines[line_num_neu_spe].find('=')+1:lines[line_num_neu_spe].find('\n')])
    data_list = lines[line_n_mp_cell_i][lines[line_n_mp_cell_i].find('=')+1:].split(",")
    data_array = np.array(data_list)             
    n_mp_cell_i = data_array.astype(int)        
    data_list = lines[line_n_mp_cell_n][lines[line_n_mp_cell_n].find('=')+1:].split(",")
    data_array = np.array(data_list)             
    n_mp_cell_n = data_array.astype(int)      
    data_list = lines[line_n_mp_cell_i_min][lines[line_n_mp_cell_i_min].find('=')+1:].split(",")
    data_array = np.array(data_list)             
    n_mp_cell_i_min = data_array.astype(int)        
    data_list = lines[line_n_mp_cell_i_max][lines[line_n_mp_cell_i_max].find('=')+1:].split(",")
    data_array = np.array(data_list)             
    n_mp_cell_i_max = data_array.astype(int)     
    data_list = lines[line_n_mp_cell_n_min][lines[line_n_mp_cell_n_min].find('=')+1:].split(",")
    data_array = np.array(data_list)             
    n_mp_cell_n_min = data_array.astype(int)        
    data_list = lines[line_n_mp_cell_n_max][lines[line_n_mp_cell_n_max].find('=')+1:].split(",")
    data_array = np.array(data_list)             
    n_mp_cell_n_max = data_array.astype(int)    
    # Obtain the inoization minimum plasma density (background)
    min_ion_plasma_density = float(lines[line_min_ion_plasma_density][lines[line_min_ion_plasma_density].find('=')+1:lines[line_min_ion_plasma_density].find('\n')])
    # Obtain the injection mass flow 
    m_A = float(lines[line_m_A][lines[line_m_A].find('=')+1:lines[line_m_A].find('\n')])
    # Obtain the neutral specular reflection probability
    if oldsimparams_sim == 0 or oldsimparams_sim == 1 or oldsimparams_sim == 2:
        spec_refl_prob = float(lines[line_spec_refl_prob][lines[line_spec_refl_prob].find('=')+1:lines[line_spec_refl_prob].find('\n')])
    elif oldsimparams_sim > 2:
        # Obtain the neutral specular reflection probability
        spec_refl_prob = h5_out['/picS/spec_refl_prob'][0][0]
    # Obtain the energy balance flag
    ene_bal = int(lines[line_ene_bal][lines[line_ene_bal].find('=')+1:lines[line_ene_bal].find('\n')])
    # Obtain the number of steps of the electron fluid module per PIC step
    nsteps_eFld = int(lines[line_nsteps_eFld][lines[line_nsteps_eFld].find('=')+1:lines[line_nsteps_eFld].find('\n')])
    # Obtain the flag override_cathode_e
    override_cathode_e = int(lines[line_override_cathode_e][lines[line_override_cathode_e].find('=')+1:lines[line_override_cathode_e].find('\n')])
    # Obtain the new cathode element ID
    data_list = lines[line_new_cathode_e][lines[line_new_cathode_e].find('=')+1:].split(",")
    data_array = np.array(data_list)             
    new_cathode_e = data_array.astype(int)
#    new_cathode_e = int(lines[line_new_cathode_e][lines[line_new_cathode_e].find('=')+1:lines[line_new_cathode_e].find('\n')])
    # Obtain the cathode temperature
    T_cath = float(lines[line_T_cath][lines[line_T_cath].find('=')+1:lines[line_T_cath].find('\n')])
    # Obtain the number of collisions activated
    n_collisions = int(lines[line_ncollisions][lines[line_ncollisions].find('=')+1:lines[line_ncollisions].find('\n')])
    # Obtain the electron-fluid module collisions related data
    if oldsimparams_sim == 7 or oldsimparams_sim == 8 or oldsimparams_sim == 9 or oldsimparams_sim == 10:
        # Obtain the ion species charge number
        data_list = lines[line_Z_ion_spe][lines[line_Z_ion_spe].find('=')+1:].split(",")
        data_array = np.array(data_list)             
        Z_ion_spe = data_array.astype(int)
        # Obtain the number of collisions in the eFld module
        n_collisions_e = int(lines[line_ncollisions_e][lines[line_ncollisions_e].find('=')+1:lines[line_ncollisions_e].find('\n')])
        # Obtain the eFld module collisions ids
        data_list = lines[line_ids_collisions_e][lines[line_ids_collisions_e].find('=')+1:].split(",")
        data_array = np.array(data_list)             
        ids_collisions_e = data_array.astype(int)
        # Obtain the eFld module collisions input heavy species
        data_list = lines[line_coll_spe_e][lines[line_coll_spe_e].find('=')+1:].split(",")
        data_array = np.array(data_list)             
        coll_spe_e = data_array.astype(int)
        # Obtain the eFld module collisions output heavy species
        data_list = lines[line_out_coll_spe_e][lines[line_out_coll_spe_e].find('=')+1:].split(",")
        data_array = np.array(data_list)             
        out_coll_spe_e = data_array.astype(int)
        if oldsimparams_sim == 10:
            # Obtain the flag for the complete interpolation and print-out to PostData.hdf5 of the eFld module variables
            interp_eFld2PIC_alldata = int(lines[line_interp_eFld2PIC_alldata][lines[line_interp_eFld2PIC_alldata].find('=')+1:lines[line_interp_eFld2PIC_alldata].find('\n')])
    
    
    
    ############### READ ALWAYS VARIABLES FOR ENERGY BALANCE ##################
    ene_bal = 1
    ###########################################################################
    
    
    print("HET_sims_read: reading mesh and particle lists...")
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
    eta_max   = h5_out['/picM/eta_max'][0][0] 
    eta_min   = h5_out['/picM/eta_min'][0][0] 
    xi_top    = h5_out['/picM/xi_top'][0][0]
    xi_bottom = h5_out['/picM/xi_bottom'][0][0]
    # Read the MFAM variables needed to plot the MFAM mesh
    dataset = h5_out['eFldM/faces']
    faces = dataset[...]
    dataset = h5_out['eFldM/nodes']    
    nodes = dataset[...]
    dataset = h5_out['eFldM/elements_n']
    elem_n = dataset[...]
    dataset = h5_out['eFldM/element_geom']
    elem_geom = dataset[...]
    dataset = h5_out['eFldM/face_geom']
    face_geom = dataset[...]
    dataset = h5_out['eFldM/boundary_f']
    boundary_f = dataset[...][0,:] - 1
    n_faces = np.shape(faces)[1]
    n_faces_boundary = len(boundary_f)
    n_elems = np.shape(elem_geom)[1]
    # Obtain the number of MFAM boundary faces of each type
    nfaces_Dwall  = 0 # Dielectric walls
    nfaces_Awall  = 0 # Anode walls
    nfaces_FLwall = 0 # Free loss walls
    for i in range(0,n_faces_boundary):
        bface_id = boundary_f[i]
        if faces[2,bface_id] == 15 or faces[2,bface_id] == 16:
            nfaces_FLwall = nfaces_FLwall + 1
        elif faces[2,bface_id] == 11 or faces[2,bface_id] == 13 or faces[2,bface_id] == 17:
            nfaces_Dwall  = nfaces_Dwall + 1
        elif faces[2,bface_id] == 12 or faces[2,bface_id] == 18:
            nfaces_Awall  = nfaces_Awall + 1   
    # Obtain the ID in boundary_f (Python standard) of the faces of each type 
    # and the ID in faces and face_geom of the MFAM boundary faces of each type
    # IDs of faces of each type in the boundary, ordered as they appear in boundary_f
    bIDfaces_Dwall  = np.zeros(nfaces_Dwall,dtype=int)
    bIDfaces_Awall  = np.zeros(nfaces_Awall,dtype=int)
    bIDfaces_FLwall = np.zeros(nfaces_FLwall,dtype=int) 
    # IDs of faces of each type in the boundary 
    # (their original IDs in faces and face_geom, i.e. their position in faces
    # and face_geom)
    IDfaces_Dwall   = np.zeros(nfaces_Dwall,dtype=int)
    IDfaces_Awall   = np.zeros(nfaces_Awall,dtype=int)
    IDfaces_FLwall  = np.zeros(nfaces_FLwall,dtype=int)
    indD  = 0
    indA  = 0
    indFL = 0
    for i in range(0,n_faces_boundary):
        bface_id = boundary_f[i]
        if faces[2,bface_id] == 15 or faces[2,bface_id] == 16:
            bIDfaces_FLwall[indFL] = i
            IDfaces_FLwall[indFL]  = bface_id
            indFL = indFL + 1
        elif faces[2,bface_id] == 11 or faces[2,bface_id] == 13 or faces[2,bface_id] == 17:
            bIDfaces_Dwall[indD] = i            
            IDfaces_Dwall[indD] = bface_id
            indD = indD + 1
        elif faces[2,bface_id] == 12 or faces[2,bface_id] == 18:
            bIDfaces_Awall[indA] = i            
            IDfaces_Awall[indA] = bface_id
            indA = indA + 1  
    # Obtain the z,r, coordinates of the face center and the area for the MFAM boundary faces of each type
    zfaces_Dwall  = face_geom[0,IDfaces_Dwall]
    rfaces_Dwall  = face_geom[1,IDfaces_Dwall]
    Afaces_Dwall  = face_geom[4,IDfaces_Dwall]
    zfaces_Awall  = face_geom[0,IDfaces_Awall]
    rfaces_Awall  = face_geom[1,IDfaces_Awall]
    Afaces_Awall  = face_geom[4,IDfaces_Awall]
    zfaces_FLwall = face_geom[0,IDfaces_FLwall]
    rfaces_FLwall = face_geom[1,IDfaces_FLwall]
    Afaces_FLwall = face_geom[4,IDfaces_FLwall]
    # Obtain cathode position
    if override_cathode_e == 0:
        cath_elem = h5_out['eFldM/cathode_e'][0][0] - 1
    else:
        cath_elem = new_cathode_e - 1
        cath_elem = np.sort(cath_elem)
    z_cath    = elem_geom[0,cath_elem]
    r_cath    = elem_geom[1,cath_elem]
    V_cath    = elem_geom[4,cath_elem]
    # Obtain the time vector
    time      = h5_post['times_sim'][:,0]
    time_fast = h5_post['times_fast'][:,0]
    dt = time[1] - time[0]
    dt_e = 0.0
    if nsteps_eFld > 0:
        dt_e = dt/np.real(nsteps_eFld)
    nsteps      = len(time)
    nsteps_fast = len(time_fast)
    steps  = h5_post['steps_sim'][:,0]
    steps_fast = np.linspace(0,steps[-1],steps[-1]+1).astype(int)
    # Obtain the magnetic field
    Br = h5_out['/picM/Br'][0:,0:]
    Bz = h5_out['/picM/Bz'][0:,0:]
    Bfield = np.sqrt(Br**2 + Bz**2)
    # Obtain the particle lists at last print-out step
    mass         = h5_out['ssIons/ssIons1/mass'][0][0]
    ssIons1 = 0.0
    ssIons2 = 0.0
    ssNeutrals1 = 0.0
    ssNeutrals2 = 0.0
    n_mp_i1_list = int(0)
    n_mp_i2_list = int(0)
    n_mp_n1_list = int(0)
    n_mp_n2_list = int(0)
    if read_part_lists == 1:
        ssIons1      = h5_out['/ssIons/ssIons1/list'][:,:]
        ssIons2      = h5_out['/ssIons/ssIons2/list'][:,:]
        ssNeutrals1  = h5_out['/ssNeutrals/ssNeutrals1/list'][:,:]
        n_mp_i1_list = h5_out['/ssIons/ssIons1/n_mp'][0][0]    
        n_mp_i2_list = h5_out['/ssIons/ssIons2/n_mp'][0][0] 
        n_mp_n1_list = h5_out['/ssNeutrals/ssNeutrals1/n_mp'][0][0]  
        if num_neu_spe == 2:
            ssNeutrals2  = h5_out['/ssNeutrals/ssNeutrals2/list'][:,:]  
            n_mp_n2_list = h5_out['/ssNeutrals/ssNeutrals2/n_mp'][0][0]  

    # Obtain data at the PIC mesh (we only consider instantaneos instead of (time-averaged) values)
    # Decide if read all timesteps or a particular timestep
    if allsteps_flag == 1:
        print("HET_sims_read: reading all steps data...")
        # Read all timesteps
        # Electric potential
        phi = h5_post["/picM_data/phi"][0:,0:,:]
        # Electric potential at the MFAM elements (nsteps x n_elems)
        phi_elems = h5_post["/eFldM_data/elements/phi"][0:,0:]
        # Electric field
        Ez = h5_post["/picM_data/Ez"][0:,0:,:]
        Er = h5_post["/picM_data/Er"][0:,0:,:]
        Efield = np.sqrt(Ez**2 + Er**2)
        # Electron temperature
        Te = h5_post["/picM_data/Te_acc"][0:,0:,:]
        # Electron temperature at the MFAM elements (nsteps x n_elems)
        Te_elems = h5_post["/eFldM_data/elements/Te"][0:,0:]
        # Electron current vector at the MFAM elements
        je_mag_elems = h5_post["/eFldM_data/elements/je_mag"][0:,0:,0:]
        je_perp_elems = je_mag_elems[0:,0,0:]
        je_theta_elems = je_mag_elems[0:,1,0:]
        je_para_elems = je_mag_elems[0:,2,0:]
        
        # Ions sonic velocity
        cs01 = np.sqrt(e*Te/mass)
        cs02 = np.sqrt(2*e*Te/mass)
        # Particle densities
        nn1 = h5_post["/picM_data/nn1"][0:,0:,:]
        ni1 = h5_post["/picM_data/ni1"][0:,0:,:]
        ni2 = np.zeros(np.shape(ni1),dtype=float)
        if num_ion_spe > 1:
            ni2 = h5_post["/picM_data/ni2"][0:,0:,:]
        ne  = h5_post["/picM_data/n"][0:,0:,:]
        nn2 = np.zeros((dims[0],dims[1],nsteps),dtype=float)
        if num_neu_spe == 2:
            nn2 = h5_post["/picM_data/nn2"][0:,0:,:]
        # Obtain the plasma density at the MFAM elements
        ne_elems = h5_post["/eFldM_data/elements/n"][0:,0:]
        # Obtain plasma density, the electron temperature and the electric potential at the cathode element
        ne_cath = h5_post["/eFldM_data/elements/n"][:,cath_elem]
        Te_cath = h5_post["/eFldM_data/elements/Te"][:,cath_elem]
        phi_cath = np.zeros(np.shape(ne_cath),dtype=float)
        # Particle fluxes, currents and fluid velocities
        fn1_x = h5_post["/picM_data/fn_x1"][0:,0:,:]
        fn1_y = h5_post["/picM_data/fn_y1"][0:,0:,:]
        fn1_z = h5_post["/picM_data/fn_z1"][0:,0:,:]
        fi1_x = h5_post["/picM_data/fi_x1"][0:,0:,:]
        fi1_y = h5_post["/picM_data/fi_y1"][0:,0:,:]
        fi1_z = h5_post["/picM_data/fi_z1"][0:,0:,:]
        fi2_x = np.zeros(np.shape(fi1_x),dtype=float)
        fi2_y = np.zeros(np.shape(fi1_y),dtype=float)
        fi2_z = np.zeros(np.shape(fi1_z),dtype=float)
        if num_ion_spe > 1:
            fi2_x = h5_post["/picM_data/fi_x2"][0:,0:,:]
            fi2_y = h5_post["/picM_data/fi_y2"][0:,0:,:]
            fi2_z = h5_post["/picM_data/fi_z2"][0:,0:,:]
        un1_x = np.divide(fn1_x,nn1) 
        un1_y = np.divide(fn1_y,nn1) 
        un1_z = np.divide(fn1_z,nn1)
        ui1_x = np.divide(fi1_x,ni1) 
        ui1_y = np.divide(fi1_y,ni1) 
        ui1_z = np.divide(fi1_z,ni1)
        ui2_x = np.zeros(np.shape(ui1_x),dtype=float)
        ui2_y = np.zeros(np.shape(ui1_y),dtype=float)
        ui2_z = np.zeros(np.shape(ui1_z),dtype=float)
        if num_ion_spe > 1:
            ui2_x = np.divide(fi2_x,ni2) 
            ui2_y = np.divide(fi2_y,ni2) 
            ui2_z = np.divide(fi2_z,ni2)
        fn2_x = np.zeros((dims[0],dims[1],nsteps),dtype=float)
        fn2_y = np.zeros((dims[0],dims[1],nsteps),dtype=float)
        fn2_z = np.zeros((dims[0],dims[1],nsteps),dtype=float)
        un2_x = np.zeros((dims[0],dims[1],nsteps),dtype=float)
        un2_y = np.zeros((dims[0],dims[1],nsteps),dtype=float)
        un2_z = np.zeros((dims[0],dims[1],nsteps),dtype=float)
        if num_neu_spe == 2:
            fn2_x = h5_post["/picM_data/fn_x2"][0:,0:,:]
            fn2_y = h5_post["/picM_data/fn_y2"][0:,0:,:]
            fn2_z = h5_post["/picM_data/fn_z2"][0:,0:,:]
            un2_x = np.divide(fn2_x,nn2) 
            un2_y = np.divide(fn2_y,nn2) 
            un2_z = np.divide(fn2_z,nn2)
        ji1_x   = e*fi1_x
        ji1_y   = e*fi1_y
        ji1_z   = e*fi1_z
        ji2_x   = 2.0*e*fi2_x
        ji2_y   = 2.0*e*fi2_y
        ji2_z   = 2.0*e*fi2_z
        if (oldsimparams_sim != 10) or (oldsimparams_sim == 10 and interp_eFld2PIC_alldata == 1):
            je_r    = h5_post["/picM_data/je_r_acc"][0:,0:,:]
    	    je_t    = h5_post["/picM_data/je_theta_acc"][0:,0:,:]
    	    je_z    = h5_post["/picM_data/je_z_acc"][0:,0:,:]
    	    je_perp = h5_post["/picM_data/je_perp_acc"][0:,0:,:]
    	    je_para = h5_post["/picM_data/je_para_acc"][0:,0:,:]
    	    ue_r    = np.divide(je_r,-e*ne) 
    	    ue_t    = np.divide(je_t,-e*ne)
    	    ue_z    = np.divide(je_z,-e*ne)
    	    ue_perp = np.divide(je_perp,-e*ne)
    	    ue_para = np.divide(je_para,-e*ne)
        else:
    	    je_r    = np.zeros((dims[0],dims[1],nsteps),dtype=float)
    	    je_t    = np.zeros((dims[0],dims[1],nsteps),dtype=float)
    	    je_z    = np.zeros((dims[0],dims[1],nsteps),dtype=float)
    	    je_perp = np.zeros((dims[0],dims[1],nsteps),dtype=float)
    	    je_para = np.zeros((dims[0],dims[1],nsteps),dtype=float)
    	    ue_r    = np.zeros((dims[0],dims[1],nsteps),dtype=float)
    	    ue_t    = np.zeros((dims[0],dims[1],nsteps),dtype=float)
    	    ue_z    = np.zeros((dims[0],dims[1],nsteps),dtype=float)
    	    ue_perp = np.zeros((dims[0],dims[1],nsteps),dtype=float)
    	    ue_para = np.zeros((dims[0],dims[1],nsteps),dtype=float)
            
        # Compute azimuthal ExB drift velocity considering the system (z,theta,r) == (perp,theta,para)
        uthetaExB = np.zeros(np.shape(ue_para),dtype=float)
        for k in range(0,nsteps):
            uthetaExB[:,:,k] = -1/Bfield[:,:]**2*(Br[:,:]*Ez[:,:,k] - Bz[:,:]*Er[:,:,k])
        # Temperatures
        Tn1 = h5_post["/picM_data/temp_n1"][0:,0:,:]
        Ti1 = h5_post["/picM_data/temp_i1"][0:,0:,:]
        Ti2 = np.zeros(np.shape(Ti1),dtype=float)
        if num_ion_spe > 1:
            Ti2 = h5_post["/picM_data/temp_i2"][0:,0:,:]
        Tn2 = np.zeros((dims[0],dims[1],nsteps),dtype=float)
        if num_neu_spe == 2:
            Tn2 = h5_post["/picM_data/temp_n2"][0:,0:,:]
        # Number of particles per cell
        n_mp_n1 = h5_post['/picM_data/n_mp_n1'][0:,0:,:]
        n_mp_i1 = h5_post['/picM_data/n_mp_i1'][0:,0:,:]
        n_mp_i2 = np.zeros(np.shape(n_mp_i1),dtype=float)
        if num_ion_spe > 1:
            n_mp_i2 = h5_post['/picM_data/n_mp_i2'][0:,0:,:]
        n_mp_n2 = np.zeros((dims[0],dims[1],nsteps),dtype=float)
        if num_neu_spe == 2:
            n_mp_n2 = h5_post['/picM_data/n_mp_n2'][0:,0:,:]
        # Average particle weight per cell
        avg_w_n1 = h5_post['/picM_data/avg_w_n1'][0:,0:,:]
        avg_w_i1 = h5_post['/picM_data/avg_w_i1'][0:,0:,:]
        avg_w_i2 = np.zeros(np.shape(avg_w_i1),dtype=float)
        if num_ion_spe > 1:
            avg_w_i2 = h5_post['/picM_data/avg_w_i2'][0:,0:,:]
        avg_w_n2 = np.zeros((dims[0],dims[1],nsteps),dtype=float)
        if num_neu_spe == 2:
            avg_w_n2 = h5_post['/picM_data/avg_w_n2'][:,:,:]
        # Generation weights per cell
        neu_gen_weights1 = h5_post['/picM_data/neu_gen_weights1'][:,:,:]
        ion_gen_weights1 = h5_post['/picM_data/ion_gen_weights1'][:,:,:]
        ion_gen_weights2 = np.zeros(np.shape(ion_gen_weights1),dtype=float)
        if num_ion_spe > 1:
            ion_gen_weights2 = h5_post['/picM_data/ion_gen_weights2'][:,:,:]
        neu_gen_weights2 = np.zeros((dims[0],dims[1],nsteps),dtype=float)
        if num_neu_spe == 2:
            neu_gen_weights2 = h5_post['/picM_data/neu_gen_weights2'][:,:,:]
        # Obtain the relevant data from KBC at the important material elements
        surf_elems    = h5_out['picS/surf_elems']    
        n_imp_elems   = h5_out['picS/n_imp_elems'][0][0]
        imp_elems     = h5_out['picS/imp_elems'][:,:]
        imp_elems_kbc = h5_post['/picS_data/imp_elems_kbc'][:,:,:]   
        # Obtain important element variables
        imp_elems_MkQ1              = h5_post['/picS_data/imp_elems_MkQ1'][:,:,:]
        imp_elems_Te                = h5_post['/picS_data/imp_elems_Te'][:,:,:]
        imp_elems_dphi_kbc          = h5_post['/picS_data/imp_elems_dphi_kbc'][:,:,:]
        imp_elems_dphi_sh           = h5_post['/picS_data/imp_elems_dphi_sh'][:,:,:]
        imp_elems_nQ1               = h5_post['/picS_data/imp_elems_nQ1'][:,:,:]
        imp_elems_nQ2               = h5_post['/picS_data/imp_elems_nQ2'][:,:,:]
        imp_elems_ion_flux_in1      = h5_post['/picS_data/imp_elems_ion_flux_in1'][:,:,:]
        imp_elems_ion_flux_out1     = h5_post['/picS_data/imp_elems_ion_flux_out1'][:,:,:]
        imp_elems_ion_ene_flux_in1  = h5_post['/picS_data/imp_elems_ion_ene_flux_in1'][:,:,:]
        imp_elems_ion_ene_flux_out1 = h5_post['/picS_data/imp_elems_ion_ene_flux_out1'][:,:,:]
        imp_elems_ion_imp_ene1      = h5_post['/picS_data/imp_elems_ion_imp_ene1'][:,:,:]
        imp_elems_ion_flux_in2      = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
        imp_elems_ion_flux_out2     = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
        imp_elems_ion_ene_flux_in2  = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
        imp_elems_ion_ene_flux_out2 = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
        imp_elems_ion_imp_ene2      = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
        if num_ion_spe > 1:
            imp_elems_ion_flux_in2      = h5_post['/picS_data/imp_elems_ion_flux_in2'][:,:,:]
            imp_elems_ion_flux_out2     = h5_post['/picS_data/imp_elems_ion_flux_out2'][:,:,:]
            imp_elems_ion_ene_flux_in2  = h5_post['/picS_data/imp_elems_ion_ene_flux_in2'][:,:,:]
            imp_elems_ion_ene_flux_out2 = h5_post['/picS_data/imp_elems_ion_ene_flux_out2'][:,:,:]
            imp_elems_ion_imp_ene2      = h5_post['/picS_data/imp_elems_ion_imp_ene2'][:,:,:]
        imp_elems_neu_flux_in1      = h5_post['/picS_data/imp_elems_neu_flux_in1'][:,:,:]
        imp_elems_neu_flux_out1     = h5_post['/picS_data/imp_elems_neu_flux_out1'][:,:,:]
        imp_elems_neu_ene_flux_in1  = h5_post['/picS_data/imp_elems_neu_ene_flux_in1'][:,:,:]
        imp_elems_neu_ene_flux_out1 = h5_post['/picS_data/imp_elems_neu_ene_flux_out1'][:,:,:]
        imp_elems_neu_imp_ene1      = h5_post['/picS_data/imp_elems_neu_imp_ene1'][:,:,:] 
        imp_elems_neu_flux_in2      = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
        imp_elems_neu_flux_out2     = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
        imp_elems_neu_ene_flux_in2  = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
        imp_elems_neu_ene_flux_out2 = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
        imp_elems_neu_imp_ene2      = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
        if num_neu_spe == 2:
            imp_elems_neu_flux_in2      = h5_post['/picS_data/imp_elems_neu_flux_in2'][:,:,:]
            imp_elems_neu_flux_out2     = h5_post['/picS_data/imp_elems_neu_flux_out2'][:,:,:]
            imp_elems_neu_ene_flux_in2  = h5_post['/picS_data/imp_elems_neu_ene_flux_in2'][:,:,:]
            imp_elems_neu_ene_flux_out2 = h5_post['/picS_data/imp_elems_neu_ene_flux_out2'][:,:,:]
            imp_elems_neu_imp_ene2      = h5_post['/picS_data/imp_elems_neu_imp_ene2'][:,:,:] 
        # Obtain the efficiencies 
        eta_u = h5_post['/othr_data/ssD_othr_acc/eta_u'][:,0]
        eta_prod = h5_post['/othr_data/ssD_othr_acc/eta_prod'][:,0]
        eta_thr = h5_post['/othr_data/ssD_othr_acc/eta_thr'][:,0]
        eta_div = h5_post['/othr_data/ssD_othr_acc/eta_div'][:,0]
        eta_cur = h5_post['/othr_data/ssD_othr_acc/eta_cur'][:,0]
        # Obtain the thrust
        thrust = h5_post['/othr_data/ssD_othr_acc/thrust'][:,0]
        thrust_ion = h5_post['/othr_data/ssD_othr_acc/thrust_ion'][:,:]
        thrust_neu = h5_post['/othr_data/ssD_othr_acc/thrust_neu'][:,:]
        # Obtain Id and Vd
        Id_inst = h5_post['/eFldM_data/boundary/Id'][:,0]
        Id      = h5_post['/eFldM_data/boundary/Id_acc'][:,0]
        Vd_inst = h5_post['/eFldM_data/boundary/Vd'][:,0]
        Vd      = h5_post['/eFldM_data/boundary/Vd_acc'][:,0]
        # Obtain the ion beam current 
        I_beam = h5_post['/othr_data/ssD_othr_acc/I_twinf_tot'][:,0]
        # Obtain the total ion current to all walls
        I_tw_tot = h5_post['/othr_data/ssD_othr_acc/I_tw_tot'][:,0]
        # Obtain the input power (WE USE THE INSTANTANEOUS VOLTAGE BECAUSE IT IS CONSTANT TO 300)
        Pd      = Vd*Id
        Pd_inst = Vd_inst*Id_inst
        # Obtain the total power deposited to the material walls
        P_mat = h5_post['/othr_data/ssD_othr_acc/P_mat'][:,0]
        # Obtain the total power deposited to the injection (anode) walls
        P_inj = h5_post['/othr_data/ssD_othr_acc/P_inj'][:,0]
        # Obtain the total power deposited to the free loss walls
        P_inf = h5_post['/othr_data/ssD_othr_acc/P_inf'][:,0]
        if oldpost_sim != 3:
            # Obtain the total power spent in ionization
            P_ion = h5_post['/othr_data/ssD_othr_acc/Pion_e'][:,0]
            # Obtain the total power spent in excitation
            P_ex = h5_post['/othr_data/ssD_othr_acc/Pex_e'][:,0]
        elif oldpost_sim == 3:
            # Obtain the total power spent in ionization and excitation
            P_ion = np.zeros(np.shape(P_mat),dtype=float)
            P_ex  = np.zeros(np.shape(P_mat),dtype=float)
            for i in range(0,n_collisions_e):
                if ids_collisions_e[i] == 2:
                    # Ionization collision
                    P_ion = P_ion + h5_post['/othr_data/ssD_othr_acc/Pcoll_e'][:,i]
                elif ids_collisions_e[i] == 4:
                    # Excitation collisions
                    P_ex = P_ex + h5_post['/othr_data/ssD_othr_acc/Pcoll_e'][:,i]
            
        # Obtain the total ion and neutral useful power (total energy flow through the free loss surface)
        P_use_tot_i = h5_post['/othr_data/ssD_othr_acc/P_use_tot_i'][:,0]
        P_use_tot_n = h5_post['/othr_data/ssD_othr_acc/P_use_tot_n'][:,0]
        P_use_tot = P_use_tot_i + P_use_tot_n
        # Obtain the axial ion and neutral useful power (total axial energy flow through the free loss surface)
        P_use_z_i = h5_post['/othr_data/ssD_othr_acc/P_use_z_i'][:,0]
        P_use_z_n = h5_post['/othr_data/ssD_othr_acc/P_use_z_n'][:,0]
        P_use_z   = P_use_z_i + P_use_z_n
        # Obtain the electron energy flux deposited to the walls at all boundary MFAM faces
        if oldpost_sim == 1:
            qe_wall      = h5_post['/eFldM_data/boundary/qe_wall_acc'][:,:]
            qe_wall_inst = h5_post['/eFldM_data/boundary/qe_wall'][:,:]
        elif oldpost_sim == 0 or oldpost_sim == 3:
            qe_wall      = h5_post['/eFldM_data/boundary/qe_tot_wall_acc'][:,:]
            qe_wall_inst = h5_post['/eFldM_data/boundary/qe_tot_wall'][:,:]
        # Obtain the electron power deposited to the walls at all MFAM boundary faces (per type of boundary faces)
        Pe_faces_Dwall       = np.zeros((nsteps,nfaces_Dwall),dtype=float)
        Pe_faces_Awall       = np.zeros((nsteps,nfaces_Awall),dtype=float)
        Pe_faces_FLwall      = np.zeros((nsteps,nfaces_FLwall),dtype=float)
        Pe_faces_Dwall_inst  = np.zeros((nsteps,nfaces_Dwall),dtype=float)
        Pe_faces_Awall_inst  = np.zeros((nsteps,nfaces_Awall),dtype=float)
        Pe_faces_FLwall_inst = np.zeros((nsteps,nfaces_FLwall),dtype=float)
        for k in range(0,nsteps):
            Pe_faces_Dwall[k,:]       = qe_wall[k,bIDfaces_Dwall]*Afaces_Dwall
            Pe_faces_Awall[k,:]       = qe_wall[k,bIDfaces_Awall]*Afaces_Awall
            Pe_faces_FLwall[k,:]      = qe_wall[k,bIDfaces_FLwall]*Afaces_FLwall
            Pe_faces_Dwall_inst[k,:]  = qe_wall_inst[k,bIDfaces_Dwall]*Afaces_Dwall
            Pe_faces_Awall_inst[k,:]  = qe_wall_inst[k,bIDfaces_Awall]*Afaces_Awall
            Pe_faces_FLwall_inst[k,:] = qe_wall_inst[k,bIDfaces_FLwall]*Afaces_FLwall
        # Obtain the total electron power deposited to the different boundary walls
        Pe_Dwall       = np.sum(Pe_faces_Dwall,axis=1)
        Pe_Awall       = np.sum(Pe_faces_Awall,axis=1)
        # NOTE: At the free loss boundary the energy flux given by the sheath is zero, so that this value is computed below
        #       This is done whis way because in old sims (thesis) qe_wall is zero for FL wall. In new sims qe_wall is properly
        #       updated, and we have checked that Pe_FLwall      = np.sum(Pe_faces_FLwall,axis=1) is equivalent to the computation
        #       done below
#        Pe_FLwall      = np.sum(Pe_faces_FLwall,axis=1)
        Pe_Dwall_inst  = np.sum(Pe_faces_Dwall_inst,axis=1)
        Pe_Awall_inst  = np.sum(Pe_faces_Awall_inst,axis=1)
        Pe_FLwall_inst = np.sum(Pe_faces_FLwall_inst,axis=1)
        # Obtain the specific impulse (s) and (m/s)
        Isp_s = thrust/(g0*m_A)
        Isp_ms = thrust/m_A  
        # Obtain data for the mass and energy balance
        # Mass balance ------------------------------------------------------------
        dMdt_i1             = h5_post['/othr_data/ssD_othr_acc/dmass_mp_ions_acc'][:,0]
        dMdt_n1             = h5_post['/othr_data/ssD_othr_acc/dmass_mp_neus_acc'][:,0]
        mflow_coll_i1       = h5_post['/othr_data/ssD_othr_acc/mflow_coll_i_acc'][:,0]
        mflow_coll_n1       = h5_post['/othr_data/ssD_othr_acc/mflow_coll_n_acc'][:,0]
        mflow_fw_i1         = h5_post['/othr_data/ssD_othr_acc/mflow_fw_i'][:,0]
        mflow_fw_n1         = h5_post['/othr_data/ssD_othr_acc/mflow_fw_n'][:,0]
        mflow_tw_i1         = h5_post['/othr_data/ssD_othr_acc/mflow_tw_i'][:,0]
        mflow_tw_n1         = h5_post['/othr_data/ssD_othr_acc/mflow_tw_n'][:,0]
        mflow_ircmb_picS_n1 = h5_post['/othr_data/ssD_othr_acc/mflow_ircmb_picS_n'][:,0] 
        dMdt_i2             = np.zeros(np.shape(dMdt_i1),dtype=float)
        mflow_coll_i2       = np.zeros(np.shape(dMdt_i1),dtype=float)
        mflow_fw_i2         = np.zeros(np.shape(dMdt_i1),dtype=float)
        mflow_tw_i2         = np.zeros(np.shape(dMdt_i1),dtype=float)
        if num_ion_spe > 1:
            dMdt_i2             = h5_post['/othr_data/ssD_othr_acc/dmass_mp_ions_acc'][:,1]
            mflow_coll_i2       = h5_post['/othr_data/ssD_othr_acc/mflow_coll_i_acc'][:,1]
            mflow_fw_i2         = h5_post['/othr_data/ssD_othr_acc/mflow_fw_i'][:,1]
            mflow_tw_i2         = h5_post['/othr_data/ssD_othr_acc/mflow_tw_i'][:,1]
        dMdt_n2             = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_coll_n2       = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_fw_n2         = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_tw_n2         = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_ircmb_picS_n2 = np.zeros(np.shape(dMdt_n1),dtype=float)
        if num_neu_spe == 2:
            dMdt_n2             = h5_post['/othr_data/ssD_othr_acc/dmass_mp_neus_acc'][:,1]
            mflow_coll_n2       = h5_post['/othr_data/ssD_othr_acc/mflow_coll_n_acc'][:,1]
            mflow_fw_n2         = h5_post['/othr_data/ssD_othr_acc/mflow_fw_n'][:,1]
            mflow_tw_n2         = h5_post['/othr_data/ssD_othr_acc/mflow_tw_n'][:,1]
            mflow_ircmb_picS_n2 = h5_post['/othr_data/ssD_othr_acc/mflow_ircmb_picS_n'][:,1] 
        # Values composing flows from wall
        mflow_inj_i1        = h5_post['/othr_data/ssD_othr_acc/mflow_inj_i'][:,0]
        mflow_fwmat_i1      = h5_post['/othr_data/ssD_othr_acc/mflow_fwmat_i'][:,0]
        mflow_inj_i2        = np.zeros(np.shape(dMdt_i1),dtype=float)
        mflow_fwmat_i2      = np.zeros(np.shape(dMdt_i1),dtype=float)
        if num_ion_spe > 1:
            mflow_inj_i2        = h5_post['/othr_data/ssD_othr_acc/mflow_inj_i'][:,1]
            mflow_fwmat_i2      = h5_post['/othr_data/ssD_othr_acc/mflow_fwmat_i'][:,1]
        mflow_inj_n1        = h5_post['/othr_data/ssD_othr_acc/mflow_inj_n'][:,0]
        mflow_fwmat_n1      = h5_post['/othr_data/ssD_othr_acc/mflow_fwmat_n'][:,0]
        mflow_inj_n2        = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_fwmat_n2      = np.zeros(np.shape(dMdt_n1),dtype=float)
        if num_neu_spe == 2:
            mflow_inj_n2        = h5_post['/othr_data/ssD_othr_acc/mflow_inj_n'][:,1]
            mflow_fwmat_n2      = h5_post['/othr_data/ssD_othr_acc/mflow_fwmat_n'][:,1]
        # Values composing flows to wall
        mflow_twmat_i1      = h5_post['/othr_data/ssD_othr_acc/mflow_twmat_i'][:,0]
        mflow_twinf_i1      = h5_post['/othr_data/ssD_othr_acc/mflow_twinf_i'][:,0]
        mflow_twa_i1        = h5_post['/othr_data/ssD_othr_acc/mflow_twa_i'][:,0]
        mflow_twmat_i2      = np.zeros(np.shape(dMdt_i1),dtype=float)
        mflow_twinf_i2      = np.zeros(np.shape(dMdt_i1),dtype=float)
        mflow_twa_i2        = np.zeros(np.shape(dMdt_i1),dtype=float)
        if num_ion_spe > 1:
            mflow_twmat_i2      = h5_post['/othr_data/ssD_othr_acc/mflow_twmat_i'][:,1]
            mflow_twinf_i2      = h5_post['/othr_data/ssD_othr_acc/mflow_twinf_i'][:,1]
            mflow_twa_i2        = h5_post['/othr_data/ssD_othr_acc/mflow_twa_i'][:,1]
        mflow_twmat_n1      = h5_post['/othr_data/ssD_othr_acc/mflow_twmat_n'][:,0]
        mflow_twinf_n1      = h5_post['/othr_data/ssD_othr_acc/mflow_twinf_n'][:,0]
        mflow_twa_n1        = h5_post['/othr_data/ssD_othr_acc/mflow_twa_n'][:,0]
        mflow_twmat_n2      = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_twinf_n2      = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_twa_n2        = np.zeros(np.shape(dMdt_n1),dtype=float)
        if num_neu_spe == 2:
            mflow_twmat_n2      = h5_post['/othr_data/ssD_othr_acc/mflow_twmat_n'][:,1]
            mflow_twinf_n2      = h5_post['/othr_data/ssD_othr_acc/mflow_twinf_n'][:,1]
            mflow_twa_n2        = h5_post['/othr_data/ssD_othr_acc/mflow_twa_n'][:,1]
        # Obtain mass balances
        mbal_n1 = mflow_coll_n1 + mflow_fw_n1 - mflow_tw_n1
        mbal_i1 = mflow_coll_i1 + mflow_fw_i1 - mflow_tw_i1
        mbal_i2 = mflow_coll_i2 + mflow_fw_i2 - mflow_tw_i2
        mbal_tot = mbal_n1 + mbal_i1 + mbal_i2
        dMdt_tot = dMdt_n1 + dMdt_i1 + dMdt_i2
        # Obtain mass balance errors
#        err_mbal_n1     = np.abs(mbal_n1 - dMdt_n1)/np.abs(mflow_inj_n1)
#        err_mbal_i1     = np.abs(mbal_i1 - dMdt_i1)/np.abs(mflow_inj_n1)
#        err_mbal_i2     = np.abs(mbal_i2 - dMdt_i2)/np.abs(mflow_inj_n1)
        err_mbal_n1     = np.abs(mbal_n1 - dMdt_n1)/np.abs(m_A)
        err_mbal_i1     = np.abs(mbal_i1 - dMdt_i1)/np.abs(m_A)
        err_mbal_i2     = np.abs(mbal_i2 - dMdt_i2)/np.abs(m_A)
        err_mbal_tot    = np.abs(mbal_tot - dMdt_tot)/np.abs(m_A)
        # Obtain contributions to the mass balances
        abs_mbal_n1       = np.abs(mflow_coll_n1) + np.abs(mflow_fw_n1) + np.abs(mflow_tw_n1)
        ctr_mflow_coll_n1 = np.abs(mflow_coll_n1)/abs_mbal_n1
        ctr_mflow_fw_n1   = np.abs(mflow_fw_n1)/abs_mbal_n1
        ctr_mflow_tw_n1   = np.abs(mflow_tw_n1)/abs_mbal_n1
        abs_mbal_i1       = np.abs(mflow_coll_i1) + np.abs(mflow_fw_i1) + np.abs(mflow_tw_i1)
        ctr_mflow_coll_i1 = np.abs(mflow_coll_i1)/abs_mbal_i1
        ctr_mflow_fw_i1   = np.abs(mflow_fw_i1)/abs_mbal_i1
        ctr_mflow_tw_i1   = np.abs(mflow_tw_i1)/abs_mbal_i1
        abs_mbal_i2       = np.abs(mflow_coll_i2) + np.abs(mflow_fw_i2) + np.abs(mflow_tw_i2)
        ctr_mflow_coll_i2 = np.abs(mflow_coll_i2)/abs_mbal_i2
        ctr_mflow_fw_i2   = np.abs(mflow_fw_i2)/abs_mbal_i2
        ctr_mflow_tw_i2   = np.abs(mflow_tw_i2)/abs_mbal_i2
        abs_mbal_tot       = np.abs(mflow_coll_n1 + mflow_coll_i1 + mflow_coll_i2) + np.abs(mflow_fw_n1 + mflow_fw_i1 + mflow_fw_i2) + np.abs(mflow_tw_n1 + mflow_tw_i1 + mflow_tw_i2)
        ctr_mflow_coll_tot = np.abs(mflow_coll_n1 + mflow_coll_i1 + mflow_coll_i2)/abs_mbal_tot
        ctr_mflow_fw_tot   = np.abs(mflow_fw_n1 + mflow_fw_i1 + mflow_fw_i2)/abs_mbal_tot
        ctr_mflow_tw_tot   = np.abs(mflow_tw_n1 + mflow_tw_i1 + mflow_tw_i2)/abs_mbal_tot
        # Energy balance ----------------------------------------------------------  
        if ene_bal == 1:
            dEdt_i1         = h5_post['/othr_data/ssD_othr_acc/dene_mp_ions_acc'][:,0]
            eneflow_coll_i1 = h5_post['/othr_data/ssD_othr_acc/eneflow_coll_i_acc'][:,0]
            eneflow_fw_i1   = h5_post['/othr_data/ssD_othr_acc/eneflow_fw_i'][:,0]
            eneflow_tw_i1   = h5_post['/othr_data/ssD_othr_acc/eneflow_tw_i'][:,0]
            Pfield_i1       = h5_post['/othr_data/ssD_othr_acc/Pfield_i_acc'][:,0]
            dEdt_i2         = np.zeros(np.shape(dEdt_i1),dtype=float)
            eneflow_coll_i2 = np.zeros(np.shape(dEdt_i1),dtype=float)
            eneflow_fw_i2   = np.zeros(np.shape(dEdt_i1),dtype=float)
            eneflow_tw_i2   = np.zeros(np.shape(dEdt_i1),dtype=float)
            Pfield_i2       = np.zeros(np.shape(dEdt_i1),dtype=float)
            if num_ion_spe > 1:
                dEdt_i2         = h5_post['/othr_data/ssD_othr_acc/dene_mp_ions_acc'][:,1]
                eneflow_coll_i2 = h5_post['/othr_data/ssD_othr_acc/eneflow_coll_i_acc'][:,1]
                eneflow_fw_i2   = h5_post['/othr_data/ssD_othr_acc/eneflow_fw_i'][:,1]
                eneflow_tw_i2   = h5_post['/othr_data/ssD_othr_acc/eneflow_tw_i'][:,1]
                Pfield_i2       = h5_post['/othr_data/ssD_othr_acc/Pfield_i_acc'][:,1]
            dEdt_n1         = h5_post['/othr_data/ssD_othr_acc/dene_mp_neus_acc'][:,0]
            eneflow_coll_n1 = h5_post['/othr_data/ssD_othr_acc/eneflow_coll_n_acc'][:,0]
            eneflow_fw_n1   = h5_post['/othr_data/ssD_othr_acc/eneflow_fw_n'][:,0]
            eneflow_tw_n1   = h5_post['/othr_data/ssD_othr_acc/eneflow_tw_n'][:,0]
            dEdt_n2         = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_coll_n2 = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_fw_n2   = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_tw_n2   = np.zeros(np.shape(dEdt_n1),dtype=float)
            if num_neu_spe == 2:
                dEdt_n2         = h5_post['/othr_data/ssD_othr_acc/dene_mp_neus_acc'][:,1]
                eneflow_coll_n2 = h5_post['/othr_data/ssD_othr_acc/eneflow_coll_n_acc'][:,1]
                eneflow_fw_n2   = h5_post['/othr_data/ssD_othr_acc/eneflow_fw_n'][:,1]
                eneflow_tw_n2   = h5_post['/othr_data/ssD_othr_acc/eneflow_tw_n'][:,1]
                
            # Values composing flows from wall
            eneflow_inj_i1        = h5_post['/othr_data/ssD_othr_acc/eneflow_inj_i'][:,0]
            eneflow_fwmat_i1      = h5_post['/othr_data/ssD_othr_acc/eneflow_fwmat_i'][:,0]
            eneflow_inj_i2        = np.zeros(np.shape(dEdt_i1),dtype=float)
            eneflow_fwmat_i2      = np.zeros(np.shape(dEdt_i1),dtype=float)
            if num_ion_spe > 1:
                eneflow_inj_i2        = h5_post['/othr_data/ssD_othr_acc/eneflow_inj_i'][:,1]
                eneflow_fwmat_i2      = h5_post['/othr_data/ssD_othr_acc/eneflow_fwmat_i'][:,1]
            eneflow_inj_n1        = h5_post['/othr_data/ssD_othr_acc/eneflow_inj_n'][:,0]
            eneflow_fwmat_n1      = h5_post['/othr_data/ssD_othr_acc/eneflow_fwmat_n'][:,0]
            eneflow_inj_n2        = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_fwmat_n2      = np.zeros(np.shape(dEdt_n1),dtype=float)
            if num_neu_spe == 2:
                eneflow_inj_n2        = h5_post['/othr_data/ssD_othr_acc/eneflow_inj_n'][:,1]
                eneflow_fwmat_n2      = h5_post['/othr_data/ssD_othr_acc/eneflow_fwmat_n'][:,1]
            # Values composing flows to wall
            eneflow_twmat_i1      = h5_post['/othr_data/ssD_othr_acc/eneflow_twmat_i'][:,0]
            eneflow_twinf_i1      = h5_post['/othr_data/ssD_othr_acc/eneflow_twinf_i'][:,0]
            eneflow_twa_i1        = h5_post['/othr_data/ssD_othr_acc/eneflow_twa_i'][:,0]
            eneflow_twmat_i2      = np.zeros(np.shape(dEdt_i1),dtype=float)
            eneflow_twinf_i2      = np.zeros(np.shape(dEdt_i1),dtype=float)
            eneflow_twa_i2        = np.zeros(np.shape(dEdt_i1),dtype=float)
            if num_ion_spe > 1:
                eneflow_twmat_i2      = h5_post['/othr_data/ssD_othr_acc/eneflow_twmat_i'][:,1]
                eneflow_twinf_i2      = h5_post['/othr_data/ssD_othr_acc/eneflow_twinf_i'][:,1]
                eneflow_twa_i2        = h5_post['/othr_data/ssD_othr_acc/eneflow_twa_i'][:,1]
            eneflow_twmat_n1      = h5_post['/othr_data/ssD_othr_acc/eneflow_twmat_n'][:,0]
            eneflow_twinf_n1      = h5_post['/othr_data/ssD_othr_acc/eneflow_twinf_n'][:,0]
            eneflow_twa_n1        = h5_post['/othr_data/ssD_othr_acc/eneflow_twa_n'][:,0]
            eneflow_twmat_n2      = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_twinf_n2      = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_twa_n2        = np.zeros(np.shape(dEdt_n1),dtype=float)
            if num_neu_spe == 2:
                eneflow_twmat_n2      = h5_post['/othr_data/ssD_othr_acc/eneflow_twmat_n'][:,1]
                eneflow_twinf_n2      = h5_post['/othr_data/ssD_othr_acc/eneflow_twinf_n'][:,1]
                eneflow_twa_n2        = h5_post['/othr_data/ssD_othr_acc/eneflow_twa_n'][:,1] 
        else:
            dEdt_i1         = 0.0
            dEdt_i2         = 0.0
            dEdt_n1         = 0.0
            dEdt_n2         = 0.0
            eneflow_coll_i1 = 0.0
            eneflow_coll_i2 = 0.0
            eneflow_coll_n1 = 0.0
            eneflow_coll_n2 = 0.0
            eneflow_fw_i1   = 0.0
            eneflow_fw_i2   = 0.0
            eneflow_fw_n1   = 0.0
            eneflow_fw_n2   = 0.0
            eneflow_tw_i1   = 0.0
            eneflow_tw_i2   = 0.0
            eneflow_tw_n1   = 0.0
            eneflow_tw_n2   = 0.0
            Pfield_i1       = 0.0
            Pfield_i2       = 0.0
            # Values composing flows from wall
            eneflow_inj_i1        = 0.0
            eneflow_fwmat_i1      = 0.0
            eneflow_inj_i2        = 0.0
            eneflow_fwmat_i2      = 0.0
            eneflow_inj_n1        = 0.0
            eneflow_fwmat_n1      = 0.0
            eneflow_inj_n2        = 0.0
            eneflow_fwmat_n2      = 0.0
            # Values composing flows to wall
            eneflow_twmat_i1      = 0.0
            eneflow_twinf_i1      = 0.0
            eneflow_twa_i1        = 0.0
            eneflow_twmat_i2      = 0.0
            eneflow_twinf_i2      = 0.0
            eneflow_twa_i2        = 0.0
            eneflow_twmat_n1      = 0.0
            eneflow_twinf_n1      = 0.0
            eneflow_twa_n1        = 0.0
            eneflow_twmat_n2      = 0.0
            eneflow_twinf_n2      = 0.0
            eneflow_twa_n2        = 0.0
            
        # Obtain the total net ion and neutral power deposited to the walls at all boundaries (per type of boundary)
        Pi_Dwall       = eneflow_twmat_i1 + eneflow_twmat_i2 - (eneflow_fwmat_i1 + eneflow_fwmat_i2) 
        Pi_Awall       = eneflow_twa_i1 + eneflow_twa_i2 - (eneflow_inj_i1 + eneflow_inj_i2) 
        Pi_FLwall      = eneflow_twinf_i1 + eneflow_twinf_i2 
        Pn_Dwall       = eneflow_twmat_n1 - eneflow_fwmat_n1 
        Pn_Awall       = eneflow_twa_n1 - eneflow_inj_n1 
        Pn_FLwall      = eneflow_twinf_n1
        # NOTE: At the free loss boundary the energy flux given by the sheath is zero, so that the electron energy deposited is
        Pe_FLwall = P_inf - (Pi_FLwall + Pn_FLwall)
        # Obtain the ion and neutral power deposited to the free loss which is not axial (not generating thrust)
        Pi_FLwall_nonz = Pi_FLwall - P_use_z_i
        Pn_FLwall_nonz = Pn_FLwall - P_use_z_n
        # Obtain total net power deposited to the walls by both the electrons and the heavy species
        P_Dwall  = Pe_Dwall + Pi_Dwall + Pn_Dwall
        P_Awall  = Pe_Awall + Pi_Awall + Pn_Awall
        P_FLwall = Pe_FLwall + Pi_FLwall + Pn_FLwall
        # Obtain the total energy balance
        Pwalls  = P_Dwall + P_Awall + P_FLwall 
        Pionex  = P_ion + P_ex
        Ploss   = Pwalls + Pionex
        balP    = Pd - Ploss
        # Obtain the total power not used for generating thrust
        Pnothrust = Pionex + P_Dwall + P_Awall + Pe_FLwall + Pi_FLwall_nonz + Pn_FLwall_nonz
        Pnothrust_walls = Pnothrust - Pionex
        # Obtain the total power used for generating thrust
        Pthrust   = P_use_z
        # Alternative total energy balance
        balP_Pthrust = Pd - (Pnothrust + Pthrust)
        # Obtain error in total energy balance
        err_balP = np.abs(balP)/Pd  
        err_balP_Pthrust = np.abs(balP_Pthrust)/Pd 
        err_def_balP = np.abs(balP - balP_Pthrust)/np.abs(balP)            
        # Contributions to the energy balance
        abs_balP         = np.abs(Pd) + np.abs(Ploss)
        ctr_Pd           = np.abs(Pd)/abs_balP
        ctr_Ploss        = np.abs(Ploss)/abs_balP
        ctr_Pwalls       = np.abs(Pwalls)/abs_balP
        ctr_Pionex       = np.abs(Pionex)/abs_balP
        ctr_P_DAwalls    = np.abs(P_Dwall + P_Awall)/abs_balP
        ctr_P_FLwalls    = np.abs(P_FLwall)/abs_balP
        ctr_P_FLwalls_in = np.abs(Pi_FLwall+Pn_FLwall)/abs_balP
        ctr_P_FLwalls_i  = np.abs(Pi_FLwall)/abs_balP
        ctr_P_FLwalls_n  = np.abs(Pn_FLwall)/abs_balP
        ctr_P_FLwalls_e  = np.abs(Pe_FLwall)/abs_balP
        abs_balP_Pthrust         = np.abs(Pd) + np.abs(Pnothrust) + np.abs(Pthrust)
        ctr_balPthrust_Pd        = np.abs(Pd)/abs_balP_Pthrust
        ctr_balPthrust_Pnothrust = np.abs(Pnothrust)/abs_balP_Pthrust
        ctr_balPthrust_Pthrust   = np.abs(Pthrust)/abs_balP_Pthrust
        ctr_balPthrust_Pnothrust_walls = np.abs(Pnothrust_walls)/abs_balP_Pthrust
        ctr_balPthrust_Pnothrust_ionex = np.abs(Pionex)/abs_balP_Pthrust
            
        # Obtain the ionization source term (ni_dot) per cell for each ionization collision
        # This is obtained as the absolute value of the neutral mass loss per cell and collision
        # (i.e. neutral species)
        # NOTE: NOT ADAPTED FOR 2 NEUTRAL SPECIES
        if n_collisions > 0:
            ndot_ion01_n1 = abs(h5_post['/picM_data/dm_coll_acc/coll1_inp1_n1'][0:,0:,:])/(dt*mass) 
            ndot_ion02_n1 = np.zeros(np.shape(ndot_ion01_n1),dtype=float)
            ndot_ion12_i1 = np.zeros(np.shape(ndot_ion01_n1),dtype=float)
            if num_ion_spe > 1:
                ndot_ion02_n1 = abs(h5_post['/picM_data/dm_coll_acc/coll2_inp1_n1'][0:,0:,:])/(dt*mass) 
                ndot_ion12_i1 = abs(h5_post['/picM_data/dm_coll_acc/coll3_inp1_i1'][0:,0:,:])/(dt*mass)
            for i in range(0,dims[0]-1):
                for j in range(0,dims[1]-1):
                    ndot_ion01_n1[i,j,:] = ndot_ion01_n1[i,j,:]/cells_vol[i,j]
                    ndot_ion02_n1[i,j,:] = ndot_ion02_n1[i,j,:]/cells_vol[i,j]
                    ndot_ion12_i1[i,j,:] = ndot_ion12_i1[i,j,:]/cells_vol[i,j]
        else:
            ndot_ion01_n1 = 0.0
            ndot_ion02_n1 = 0.0
            ndot_ion12_i1 = 0.0
            
        # Obtain cathode production frequency, mass and energy source term and power
        if len(cath_elem) == 1:
            V_cath_tot  = V_cath
            ne_cath_avg = ne_cath
            nu_cath   = (Id/V_cath)/(e*ne_cath)
            ndot_cath = ne_cath*nu_cath
            Q_cath    = 3.0/2.0*ndot_cath*e*T_cath
            P_cath    = V_cath*Q_cath
        elif len(cath_elem) > 1:
            V_cath_tot  = np.sum(V_cath)
            ne_cath_avg = np.zeros(nsteps,dtype=float)
            for k in range(0,nsteps):
                for ind_cath in range(0,len(cath_elem)):
                    ne_cath_avg[k] = ne_cath_avg[k] + ne_cath[k,ind_cath]*V_cath[ind_cath]/V_cath_tot
            nu_cath   = (Id/V_cath_tot)/(e*ne_cath_avg)
            ndot_cath = ne_cath_avg*nu_cath
            Q_cath    = 3.0/2.0*ndot_cath*e*T_cath
            P_cath    = V_cath_tot*Q_cath
                
            
            
        # Obtain extra data at the PIC mesh if required, including anomalous term, Hall parameter and collision frequencies
        if read_flag == 1 and interp_eFld2PIC_alldata == 1:
            F_theta      = h5_post['/picM_data/F_theta_acc'][:,:,:]
            Hall_par     = h5_post['/picM_data/Hall_Beta_e_tot_acc'][:,:,:]
            Hall_par_eff = h5_post['/picM_data/Hall_Beta_e_tot_eff_acc'][:,:,:]
            nu_e_tot     = h5_post['/picM_data/freq_e_tot_acc'][:,:,:]
            nu_e_tot_eff = h5_post['/picM_data/freq_e_tot_eff_acc'][:,:,:]
            if oldpost_sim != 3:
                nu_en        = h5_post['/picM_data/freq_en_acc_prop1'][:,:,:]
                nu_ei1       = h5_post['/picM_data/freq_ei1_acc_prop1'][:,:,:]
                nu_ei2       = h5_post['/picM_data/freq_ei2_acc_prop1'][:,:,:]
                nu_i01       = h5_post['/picM_data/freq_i01_acc_prop1'][:,:,:]
                nu_i02       = h5_post['/picM_data/freq_i02_acc_prop1'][:,:,:]
                nu_i12       = h5_post['/picM_data/freq_i12_acc_prop1'][:,:,:]    
                nu_ex        = np.zeros(np.shape(nu_e_tot),dtype=float)
            elif oldpost_sim == 3:
                nu_en        = np.zeros(np.shape(nu_e_tot),dtype=float)
                nu_ei1       = np.zeros(np.shape(nu_e_tot),dtype=float)
                nu_ei2       = np.zeros(np.shape(nu_e_tot),dtype=float)
                nu_i01       = np.zeros(np.shape(nu_e_tot),dtype=float)
                nu_i02       = np.zeros(np.shape(nu_e_tot),dtype=float)
                nu_i12       = np.zeros(np.shape(nu_e_tot),dtype=float)  
                nu_ex        = np.zeros(np.shape(nu_e_tot),dtype=float) 
                freq_e = np.zeros((n_collisions_e,dims[0],dims[1],nsteps),dtype=float)
                for i in range(0,n_collisions_e):
                    freq_e[i,:,:,:] = h5_post['/picM_data/freq_e_acc'+str(i+1)][:,:,:]
                for i in range(0,n_collisions_e):
                    if ids_collisions_e[i] == 1:
                        # Electron-neutral elastic collision
                        nu_en = nu_en + freq_e[i,:,:,:]
                    if ids_collisions_e[i] == 2:
                        # Ionization collision
                        if coll_spe_e[i] < 0 and Z_ion_spe[out_coll_spe_e[i]-1] == 1:
                            # Ionization 0-1
                            nu_i01 = nu_i01 + freq_e[i,:,:,:]
                        elif coll_spe_e[i] < 0 and Z_ion_spe[out_coll_spe_e[i]-1] == 2:
                            # Ionization 0-2
                            nu_i02 = nu_i02 + freq_e[i,:,:,:]
                        elif coll_spe_e[i] > 0 and Z_ion_spe[out_coll_spe_e[i]-1] - Z_ion_spe[coll_spe_e[i]-1] == 1:
                            # Ionization 1-2
                            nu_i12 = nu_i12 + freq_e[i,:,:,:]
                    elif ids_collisions_e[i] == 4:
                        # Electron-neutral excitation collision
                        nu_ex = nu_ex + freq_e[i,:,:,:]
                    elif ids_collisions_e[i] == 5:
                        # Electron-ion Coulomb collision (elastic)
                        if Z_ion_spe[coll_spe_e[i]-1] == 1:
                            # Electron-singly charged ion collision
                            nu_ei1 = nu_ei1 + freq_e[i,:,:,:]
                        elif Z_ion_spe[coll_spe_e[i]-1] == 2:
                            # Electron-doubly charged ion collision
                            nu_ei2 = nu_ei2 + freq_e[i,:,:,:]
                        
        else:
            F_theta      = np.zeros(np.shape(Te),dtype=float)
            Hall_par     = np.zeros(np.shape(Te),dtype=float)
            Hall_par_eff = np.zeros(np.shape(Te),dtype=float)
            nu_e_tot     = np.zeros(np.shape(Te),dtype=float)
            nu_e_tot_eff = np.zeros(np.shape(Te),dtype=float)
            nu_en        = np.zeros(np.shape(Te),dtype=float)
            nu_ei1       = np.zeros(np.shape(Te),dtype=float)
            nu_ei2       = np.zeros(np.shape(Te),dtype=float)
            nu_i01       = np.zeros(np.shape(Te),dtype=float)
            nu_i02       = np.zeros(np.shape(Te),dtype=float)
            nu_i12       = np.zeros(np.shape(Te),dtype=float)
            nu_ex        = np.zeros(np.shape(Te),dtype=float)
            
        # NEW ENERGY BALANCE FOR ELECTRONS
        Pfield_e = h5_post['/othr_data/ssD_othr_acc/Pfield_e'][:,0]
        Ebal_e   = Pfield_e + P_cath - (Pe_Dwall + Pe_Awall + Pe_FLwall + Pionex)

        
        # Read extra boundary variables
        if oldpost_sim == 7:
            # Obtain the SEE yields and electron/ion total fluxes
            ge_b        = h5_post['/picM_data/ji_tot_b'][0:,0:,:]/e
            ge_b_acc    = h5_post['/picM_data/ji_tot_b_acc'][0:,0:,:]/e
            ge_sb_b     = h5_post['/picM_data/ge_sb_b'][0:,0:,:]
            ge_sb_b_acc = h5_post['/picM_data/ge_sb_b_acc'][0:,0:,:]
            delta_see     = np.divide(ge_sb_b,ge_sb_b + ge_b)
            delta_see_acc = np.divide(ge_sb_b_acc,ge_sb_b_acc + ge_b_acc)
            #delta_see      = ge_sb_b/(ge_sb_b + ge_b)
            #delta_see_acc  = ge_sb_b_acc/(ge_sb_b_acc + ge_b_acc)
        else:
            ge_b        = np.zeros(np.shape(Te),dtype=float)
            ge_b_acc    = np.zeros(np.shape(Te),dtype=float)
            ge_sb_b     = np.zeros(np.shape(Te),dtype=float)
            ge_sb_b_acc = np.zeros(np.shape(Te),dtype=float)
            delta_see     = np.zeros(np.shape(Te),dtype=float)
            delta_see_acc = np.zeros(np.shape(Te),dtype=float)
            
        # Read interpolation errors for density
        if oldpost_sim == 3:
#            err_interp_n = h5_post['/picM_data/err_interp_n_acc'][0:,0:,:]
            err_interp_n = np.zeros(np.shape(Te),dtype=float)
        else:
            err_interp_n = np.zeros(np.shape(Te),dtype=float)
                
    
            
            
    elif allsteps_flag == 0:
        print("HET_sims_read: reading given timestep data...")
        # Read only the given timestep
        # Electric potential
        phi = h5_post["/picM_data/phi_acc"][0:,0:,timestep]
        # Electric potential at the MFAM elements (nsteps x n_elems)
        phi_elems = h5_post["/eFldM_data/elements/phi_acc"][timestep,0:]
        # Electric field
        Ez = h5_post["/picM_data/Ez_acc"][0:,0:,timestep]
        Er = h5_post["/picM_data/Er_acc"][0:,0:,timestep]
        Efield = np.sqrt(Ez**2 + Er**2)
        # Electron temperature
        Te = h5_post["/picM_data/Te_acc"][0:,0:,timestep]
        # Electron temperature at the MFAM elements (nsteps x n_elems)
        Te_elems = h5_post["/eFldM_data/elements/Te_acc"][timestep,0:]
        # Electron current vector at the MFAM elements
        je_mag_elems = h5_post["/eFldM_data/elements/je_mag_acc"][timestep,0:,0:]
        je_perp_elems = je_mag_elems[0,0:]
        je_theta_elems = je_mag_elems[1,0:]
        je_para_elems = je_mag_elems[2,0:]
        
        # Ions sonic velocity
        cs01 = np.sqrt(e*Te/mass)
        cs02 = np.sqrt(2*e*Te/mass)
        # Particle densities
        nn1 = h5_post["/picM_data/nn_acc1"][0:,0:,timestep]
        ni1 = h5_post["/picM_data/ni_acc1"][0:,0:,timestep]
        ni2 = h5_post["/picM_data/ni_acc2"][0:,0:,timestep]
        ne  = h5_post["/picM_data/n_acc"][0:,0:,timestep]
        nn2 = np.zeros((dims[0],dims[1]),dtype=float)
        if num_neu_spe == 2:
            nn2 = h5_post["/picM_data/nn_acc2"][0:,0:,timestep]
        # Obtain the plasma density at the MFAM elements
        ne_elems = h5_post["/eFldM_data/elements/n"][timestep,0:]
        # Obtain plasma density, the electron temperature and the electric potential at the cathode element
        ne_cath = h5_post["/eFldM_data/elements/n"][timestep,cath_elem]
        Te_cath = h5_post["/eFldM_data/elements/Te_acc"][timestep,cath_elem]
        phi_cath = 0.0
        # Particle fluxes, currents and fluid velocities
        fn1_x = h5_post["/picM_data/fn_x_acc1"][0:,0:,timestep]
        fn1_y = h5_post["/picM_data/fn_y_acc1"][0:,0:,timestep]
        fn1_z = h5_post["/picM_data/fn_z_acc1"][0:,0:,timestep]
        fi1_x = h5_post["/picM_data/fi_x_acc1"][0:,0:,timestep]
        fi1_y = h5_post["/picM_data/fi_y_acc1"][0:,0:,timestep]
        fi1_z = h5_post["/picM_data/fi_z_acc1"][0:,0:,timestep]
        fi2_x = h5_post["/picM_data/fi_x_acc2"][0:,0:,timestep]
        fi2_y = h5_post["/picM_data/fi_y_acc2"][0:,0:,timestep]
        fi2_z = h5_post["/picM_data/fi_z_acc2"][0:,0:,timestep]
        
#        fn1_x = h5_post["/picM_data/fn_x1"][0:,0:,timestep]
#        fn1_y = h5_post["/picM_data/fn_y1"][0:,0:,timestep]
#        fn1_z = h5_post["/picM_data/fn_z1"][0:,0:,timestep]
#        fi1_x = h5_post["/picM_data/fi_x1"][0:,0:,timestep]
#        fi1_y = h5_post["/picM_data/fi_y1"][0:,0:,timestep]
#        fi1_z = h5_post["/picM_data/fi_z1"][0:,0:,timestep]
#        fi2_x = h5_post["/picM_data/fi_x2"][0:,0:,timestep]
#        fi2_y = h5_post["/picM_data/fi_y2"][0:,0:,timestep]
#        fi2_z = h5_post["/picM_data/fi_z2"][0:,0:,timestep]
        
        un1_x = np.divide(fn1_x,nn1) 
        un1_y = np.divide(fn1_y,nn1) 
        un1_z = np.divide(fn1_z,nn1)
        ui1_x = np.divide(fi1_x,ni1) 
        ui1_y = np.divide(fi1_y,ni1) 
        ui1_z = np.divide(fi1_z,ni1)
        ui2_x = np.divide(fi2_x,ni2) 
        ui2_y = np.divide(fi2_y,ni2) 
        ui2_z = np.divide(fi2_z,ni2)
        fn2_x = np.zeros((dims[0],dims[1]),dtype=float)
        fn2_y = np.zeros((dims[0],dims[1]),dtype=float)
        fn2_z = np.zeros((dims[0],dims[1]),dtype=float)
        un2_x = np.zeros((dims[0],dims[1]),dtype=float)
        un2_y = np.zeros((dims[0],dims[1]),dtype=float)
        un2_z = np.zeros((dims[0],dims[1]),dtype=float)
        if num_neu_spe == 2:
            fn2_x = h5_post["/picM_data/fn_x_acc2"][0:,0:,timestep]
            fn2_y = h5_post["/picM_data/fn_y_acc2"][0:,0:,timestep]
            fn2_z = h5_post["/picM_data/fn_z_acc2"][0:,0:,timestep]
            un2_x = np.divide(fn2_x,nn2) 
            un2_y = np.divide(fn2_y,nn2) 
            un2_z = np.divide(fn2_z,nn2)
        ji1_x   = e*fi1_x
        ji1_y   = e*fi1_y
        ji1_z   = e*fi1_z
        ji2_x   = 2.0*e*fi2_x
        ji2_y   = 2.0*e*fi2_y
        ji2_z   = 2.0*e*fi2_z
        if (oldsimparams_sim != 10) or (oldsimparams_sim == 10 and interp_eFld2PIC_alldata == 1):
            je_r    = h5_post["/picM_data/je_r_acc"][0:,0:,timestep]
            je_t    = h5_post["/picM_data/je_theta_acc"][0:,0:,timestep]
            je_z    = h5_post["/picM_data/je_z_acc"][0:,0:,timestep]
            je_perp = h5_post["/picM_data/je_perp_acc"][0:,0:,timestep]
            je_para = h5_post["/picM_data/je_para_acc"][0:,0:,timestep]
            ue_r    = np.divide(je_r,-e*ne) 
            ue_t    = np.divide(je_t,-e*ne)
            ue_z    = np.divide(je_z,-e*ne)
            ue_perp = np.divide(je_perp,-e*ne)
            ue_para = np.divide(je_para,-e*ne)
        else:
    	    je_r    = np.zeros((dims[0],dims[1]),dtype=float)
    	    je_t    = np.zeros((dims[0],dims[1]),dtype=float)
    	    je_z    = np.zeros((dims[0],dims[1]),dtype=float)
    	    je_perp = np.zeros((dims[0],dims[1]),dtype=float)
    	    je_para = np.zeros((dims[0],dims[1]),dtype=float)
    	    ue_r    = np.zeros((dims[0],dims[1]),dtype=float)
    	    ue_t    = np.zeros((dims[0],dims[1]),dtype=float)
    	    ue_z    = np.zeros((dims[0],dims[1]),dtype=float)
    	    ue_perp = np.zeros((dims[0],dims[1]),dtype=float)
    	    ue_para = np.zeros((dims[0],dims[1]),dtype=float)
            
        # Compute azimuthal ExB drift velocity considering the system (z,theta,r) == (perp,theta,para)
        uthetaExB = -1/Bfield**2*(Br*Ez - Bz*Er)
        # Temperatures
        Tn1 = h5_post["/picM_data/temp_n_acc1"][0:,0:,timestep]
        Ti1 = h5_post["/picM_data/temp_i_acc1"][0:,0:,timestep]
        Ti2 = h5_post["/picM_data/temp_i_acc2"][0:,0:,timestep]
        Tn2 = np.zeros((dims[0],dims[1]),dtype=float)
        if num_neu_spe == 2:
            Tn2 = h5_post["/picM_data/temp_n_acc2"][0:,0:,timestep]
        # Number of particles per cell
        n_mp_n1 = h5_post['/picM_data/n_mp_n_acc1'][0:,0:,timestep]
        n_mp_i1 = h5_post['/picM_data/n_mp_i_acc1'][0:,0:,timestep]
        n_mp_i2 = h5_post['/picM_data/n_mp_i_acc2'][0:,0:,timestep]
        n_mp_n2 = np.zeros((dims[0],dims[1]),dtype=float)
        if num_neu_spe == 2:
            n_mp_n2 = h5_post['/picM_data/n_mp_n_acc2'][0:,0:,timestep]
        # Average particle weight per cell
        avg_w_n1 = h5_post['/picM_data/avg_w_n_acc1'][0:,0:,timestep]
        avg_w_i1 = h5_post['/picM_data/avg_w_i_acc1'][0:,0:,timestep]
        avg_w_i2 = h5_post['/picM_data/avg_w_i_acc2'][0:,0:,timestep]
        avg_w_n2 = np.zeros((dims[0],dims[1]),dtype=float)
        if num_neu_spe == 2:
            avg_w_n2 = h5_post['/picM_data/avg_w_n_acc2'][:,:,timestep]
        # Generation weights per cell
        neu_gen_weights1 = h5_post['/picM_data/neu_gen_weights1'][:,:,timestep]
        ion_gen_weights1 = h5_post['/picM_data/ion_gen_weights1'][:,:,timestep]
        ion_gen_weights2 = h5_post['/picM_data/ion_gen_weights2'][:,:,timestep]
        neu_gen_weights2 = np.zeros((dims[0],dims[1]),dtype=float)
        if num_neu_spe == 2:
            neu_gen_weights2 = h5_post['/picM_data/neu_gen_weights2'][:,:,timestep]
        # Obtain the relevant data from KBC at the important material elements
        surf_elems    = h5_out['picS/surf_elems']    
        n_imp_elems   = h5_out['picS/n_imp_elems'][0][0]
        imp_elems     = h5_out['picS/imp_elems'][:,:]
        imp_elems_kbc = h5_post['/picS_data/imp_elems_kbc'][:,:,timestep]   
        # Obtain important element variables
#        imp_elems_MkQ1              = h5_post['/picS_data/imp_elems_MkQ1'][:,:,timestep]
#        imp_elems_Te                = h5_post['/picS_data/imp_elems_Te'][:,:,timestep]
#        imp_elems_dphi_kbc          = h5_post['/picS_data/imp_elems_dphi_kbc'][:,:,timestep]
#        imp_elems_dphi_sh           = h5_post['/picS_data/imp_elems_dphi_sh'][:,:,timestep]
#        imp_elems_nQ1               = h5_post['/picS_data/imp_elems_nQ1'][:,:,timestep]
#        imp_elems_nQ2               = h5_post['/picS_data/imp_elems_nQ2'][:,:,timestep]
#        imp_elems_ion_flux_in1      = h5_post['/picS_data/imp_elems_ion_flux_in1'][:,:,timestep]
#        imp_elems_ion_flux_out1     = h5_post['/picS_data/imp_elems_ion_flux_out1'][:,:,timestep]
#        imp_elems_ion_ene_flux_in1  = h5_post['/picS_data/imp_elems_ion_ene_flux_in1'][:,:,timestep]
#        imp_elems_ion_ene_flux_out1 = h5_post['/picS_data/imp_elems_ion_ene_flux_out1'][:,:,timestep]
#        imp_elems_ion_imp_ene1      = h5_post['/picS_data/imp_elems_ion_imp_ene1'][:,:,timestep]
#        imp_elems_ion_flux_in2      = h5_post['/picS_data/imp_elems_ion_flux_in2'][:,:,timestep]
#        imp_elems_ion_flux_out2     = h5_post['/picS_data/imp_elems_ion_flux_out2'][:,:,timestep]
#        imp_elems_ion_ene_flux_in2  = h5_post['/picS_data/imp_elems_ion_ene_flux_in2'][:,:,timestep]
#        imp_elems_ion_ene_flux_out2 = h5_post['/picS_data/imp_elems_ion_ene_flux_out2'][:,:,timestep]
#        imp_elems_ion_imp_ene2      = h5_post['/picS_data/imp_elems_ion_imp_ene2'][:,:,timestep]
#        imp_elems_neu_flux_in1      = h5_post['/picS_data/imp_elems_neu_flux_in1'][:,:,timestep]
#        imp_elems_neu_flux_out1     = h5_post['/picS_data/imp_elems_neu_flux_out1'][:,:,timestep]
#        imp_elems_neu_ene_flux_in1  = h5_post['/picS_data/imp_elems_neu_ene_flux_in1'][:,:,timestep]
#        imp_elems_neu_ene_flux_out1 = h5_post['/picS_data/imp_elems_neu_ene_flux_out1'][:,:,timestep]
#        imp_elems_neu_imp_ene1      = h5_post['/picS_data/imp_elems_neu_imp_ene1'][:,:,timestep] 
#        imp_elems_neu_flux_in2      = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
#        imp_elems_neu_flux_out2     = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
#        imp_elems_neu_ene_flux_in2  = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
#        imp_elems_neu_ene_flux_out2 = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
#        imp_elems_neu_imp_ene2      = np.zeros(np.shape(imp_elems_MkQ1),dtype=float)
#        if num_neu_spe == 2:
#            imp_elems_neu_flux_in2      = h5_post['/picS_data/imp_elems_neu_flux_in2'][:,:,timestep]
#            imp_elems_neu_flux_out2     = h5_post['/picS_data/imp_elems_neu_flux_out2'][:,:,timestep]
#            imp_elems_neu_ene_flux_in2  = h5_post['/picS_data/imp_elems_neu_ene_flux_in2'][:,:,timestep]
#            imp_elems_neu_ene_flux_out2 = h5_post['/picS_data/imp_elems_neu_ene_flux_out2'][:,:,timestep]
#            imp_elems_neu_imp_ene2      = h5_post['/picS_data/imp_elems_neu_imp_ene2'][:,:,timestep]   
        
        
        imp_elems_MkQ1              = 0
        imp_elems_Te                = 0
        imp_elems_dphi_kbc          = 0
        imp_elems_dphi_sh           = 0
        imp_elems_nQ1               = 0
        imp_elems_nQ2               = 0
        imp_elems_ion_flux_in1      = 0
        imp_elems_ion_flux_out1     = 0
        imp_elems_ion_ene_flux_in1  = 0
        imp_elems_ion_ene_flux_out1 = 0
        imp_elems_ion_imp_ene1      = 0
        imp_elems_ion_flux_in2      = 0
        imp_elems_ion_flux_out2     = 0
        imp_elems_ion_ene_flux_in2  = 0
        imp_elems_ion_ene_flux_out2 = 0
        imp_elems_ion_imp_ene2      = 0
        imp_elems_neu_flux_in1      = 0
        imp_elems_neu_flux_out1     = 0
        imp_elems_neu_ene_flux_in1  = 0
        imp_elems_neu_ene_flux_out1 = 0
        imp_elems_neu_imp_ene1      = 0 
        imp_elems_neu_flux_in2      = 0
        imp_elems_neu_flux_out2     = 0
        imp_elems_neu_ene_flux_in2  = 0
        imp_elems_neu_ene_flux_out2 = 0
        imp_elems_neu_imp_ene2      = 0
     
        # Obtain the efficiencies 
        eta_u = h5_post['/othr_data/ssD_othr_acc/eta_u'][timestep,0]
        eta_prod = h5_post['/othr_data/ssD_othr_acc/eta_prod'][timestep,0]
        eta_thr = h5_post['/othr_data/ssD_othr_acc/eta_thr'][timestep,0]
        eta_div = h5_post['/othr_data/ssD_othr_acc/eta_div'][timestep,0]
        eta_cur = h5_post['/othr_data/ssD_othr_acc/eta_cur'][timestep,0]
        # Obtain the thrust
        thrust = h5_post['/othr_data/ssD_othr_acc/thrust'][timestep,:]
        thrust_ion = h5_post['/othr_data/ssD_othr_acc/thrust_ion'][timestep,:]
        thrust_neu = h5_post['/othr_data/ssD_othr_acc/thrust_neu'][timestep,:]
        # Obtain Id and Vd
        Id_inst = h5_post['/eFldM_data/boundary/Id'][timestep,0]
        Id      = h5_post['/eFldM_data/boundary/Id_acc'][timestep,0]
        Vd_inst = h5_post['/eFldM_data/boundary/Vd'][timestep,0]
        Vd      = h5_post['/eFldM_data/boundary/Vd_acc'][timestep,0]
        # Obtain the ion beam current 
        I_beam = h5_post['/othr_data/ssD_othr_acc/I_twinf_tot'][timestep,0]
        # Obtain the total ion current to all walls
        I_tw_tot = h5_post['/othr_data/ssD_othr_acc/I_tw_tot'][timestep,0]
        # Obtain the input power (WE USE THE INSTANTANEOUS VOLTAGE BECAUSE IT IS CONSTANT TO 300)
        Pd      = Vd*Id
        Pd_inst = Vd_inst*Id_inst
        # Obtain the total power deposited to the material walls
        P_mat = h5_post['/othr_data/ssD_othr_acc/P_mat'][timestep,0]
        # Obtain the total power deposited to the injection (anode) walls
        P_inj = h5_post['/othr_data/ssD_othr_acc/P_inj'][timestep,0]
        # Obtain the total power deposited to the free loss walls
        P_inf = h5_post['/othr_data/ssD_othr_acc/P_inf'][timestep,0]
        if oldpost_sim != 3:
            # Obtain the total power spent in ionization
            P_ion = h5_post['/othr_data/ssD_othr_acc/Pion_e'][timestep,0]
            # Obtain the total power spent in excitation
            P_ex = h5_post['/othr_data/ssD_othr_acc/Pex_e'][timestep,0]
        elif oldpost_sim == 3:
            # Obtain the total power spent in ionization and excitation
            P_ion = np.zeros(np.shape(P_mat),dtype=float)
            P_ex  = np.zeros(np.shape(P_mat),dtype=float)
            for i in range(0,n_collisions_e):
                if ids_collisions_e[i] == 2:
                    # Ionization collision
                    P_ion = P_ion + h5_post['/othr_data/ssD_othr_acc/Pcoll_e'][timestep,i]
                elif ids_collisions_e[i] == 4:
                    # Excitation collisions
                    P_ex = P_ex + h5_post['/othr_data/ssD_othr_acc/Pcoll_e'][timestep,i]
        # Obtain the total ion and neutral useful power (total energy flow through the free loss surface)
        P_use_tot_i = h5_post['/othr_data/ssD_othr_acc/P_use_tot_i'][timestep,0]
        P_use_tot_n = h5_post['/othr_data/ssD_othr_acc/P_use_tot_n'][timestep,0]
        P_use_tot   = P_use_tot_i + P_use_tot_n
        # Obtain the axial ion and neutral useful power (total axial energy flow through the free loss surface)
        P_use_z_i = h5_post['/othr_data/ssD_othr_acc/P_use_z_i'][timestep,0]
        P_use_z_n = h5_post['/othr_data/ssD_othr_acc/P_use_z_n'][timestep,0]
        P_use_z   = P_use_z_i + P_use_z_n
        # Obtain the electron energy flux deposited to the walls at all boundary MFAM faces
        if oldpost_sim == 1:
            qe_wall      = h5_post['/eFldM_data/boundary/qe_wall_acc'][timestep,:]
            qe_wall_inst = h5_post['/eFldM_data/boundary/qe_wall'][timestep,:]
        elif oldpost_sim == 0 or oldpost_sim == 3:
            qe_wall      = h5_post['/eFldM_data/boundary/qe_tot_wall_acc'][timestep,:]
            qe_wall_inst = h5_post['/eFldM_data/boundary/qe_tot_wall'][timestep,:]
        # Obtain the electron power deposited to the walls at all MFAM boundary faces (per type of boundary faces)
        Pe_faces_Dwall       = qe_wall[bIDfaces_Dwall]*Afaces_Dwall
        Pe_faces_Awall       = qe_wall[bIDfaces_Awall]*Afaces_Awall
        Pe_faces_FLwall      = qe_wall[bIDfaces_FLwall]*Afaces_FLwall
        Pe_faces_Dwall_inst  = qe_wall_inst[bIDfaces_Dwall]*Afaces_Dwall
        Pe_faces_Awall_inst  = qe_wall_inst[bIDfaces_Awall]*Afaces_Awall
        Pe_faces_FLwall_inst = qe_wall_inst[bIDfaces_FLwall]*Afaces_FLwall
        # Obtain the total electron power deposited to the different boundary walls
        Pe_Dwall       = np.sum(Pe_faces_Dwall)
        Pe_Awall       = np.sum(Pe_faces_Awall)
        # NOTE: At the free loss boundary the energy flux given by the sheath is zero, so that this value is computed below
#        Pe_FLwall      = np.sum(Pe_faces_FLwall)
        Pe_Dwall_inst  = np.sum(Pe_faces_Dwall_inst)
        Pe_Awall_inst  = np.sum(Pe_faces_Awall_inst)
        Pe_FLwall_inst = np.sum(Pe_faces_FLwall_inst)
        # Obtain the specific impulse (s) and (m/s)
        Isp_s = thrust/(g0*m_A)
        Isp_ms = thrust/m_A  
        # Obtain data for the mass and energy balance
        # Mass balance ------------------------------------------------------------
        dMdt_i1             = h5_post['/othr_data/ssD_othr_acc/dmass_mp_ions_acc'][timestep,0]
        dMdt_i2             = h5_post['/othr_data/ssD_othr_acc/dmass_mp_ions_acc'][timestep,1]
        dMdt_n1             = h5_post['/othr_data/ssD_othr_acc/dmass_mp_neus_acc'][timestep,0]
        mflow_coll_i1       = h5_post['/othr_data/ssD_othr_acc/mflow_coll_i_acc'][timestep,0]
        mflow_coll_i2       = h5_post['/othr_data/ssD_othr_acc/mflow_coll_i_acc'][timestep,1]
        mflow_coll_n1       = h5_post['/othr_data/ssD_othr_acc/mflow_coll_n_acc'][timestep,0]
        mflow_fw_i1         = h5_post['/othr_data/ssD_othr_acc/mflow_fw_i'][timestep,0]
        mflow_fw_i2         = h5_post['/othr_data/ssD_othr_acc/mflow_fw_i'][timestep,1]
        mflow_fw_n1         = h5_post['/othr_data/ssD_othr_acc/mflow_fw_n'][timestep,0]
        mflow_tw_i1         = h5_post['/othr_data/ssD_othr_acc/mflow_tw_i'][timestep,0]
        mflow_tw_i2         = h5_post['/othr_data/ssD_othr_acc/mflow_tw_i'][timestep,1]
        mflow_tw_n1         = h5_post['/othr_data/ssD_othr_acc/mflow_tw_n'][timestep,0]
        mflow_ircmb_picS_n1 = h5_post['/othr_data/ssD_othr_acc/mflow_ircmb_picS_n'][timestep,0] 
        dMdt_n2             = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_coll_n2       = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_fw_n2         = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_tw_n2         = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_ircmb_picS_n2 = np.zeros(np.shape(dMdt_n1),dtype=float)
        if num_neu_spe == 2:
            dMdt_n2             = h5_post['/othr_data/ssD_othr_acc/dmass_mp_neus_acc'][timestep,1]
            mflow_coll_n2       = h5_post['/othr_data/ssD_othr_acc/mflow_coll_n_acc'][timestep,1]
            mflow_fw_n2         = h5_post['/othr_data/ssD_othr_acc/mflow_fw_n'][timestep,1]
            mflow_tw_n2         = h5_post['/othr_data/ssD_othr_acc/mflow_tw_n'][timestep,1]
            mflow_ircmb_picS_n2 = h5_post['/othr_data/ssD_othr_acc/mflow_ircmb_picS_n'][timestep,1] 
        # Values composing flows from wall
        mflow_inj_i1        = h5_post['/othr_data/ssD_othr_acc/mflow_inj_i'][timestep,0]
        mflow_fwmat_i1      = h5_post['/othr_data/ssD_othr_acc/mflow_fwmat_i'][timestep,0]
        mflow_inj_i2        = h5_post['/othr_data/ssD_othr_acc/mflow_inj_i'][timestep,1]
        mflow_fwmat_i2      = h5_post['/othr_data/ssD_othr_acc/mflow_fwmat_i'][timestep,1]
        mflow_inj_n1        = h5_post['/othr_data/ssD_othr_acc/mflow_inj_n'][timestep,0]
        mflow_fwmat_n1      = h5_post['/othr_data/ssD_othr_acc/mflow_fwmat_n'][timestep,0]
        mflow_inj_n2        = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_fwmat_n2      = np.zeros(np.shape(dMdt_n1),dtype=float)
        if num_neu_spe == 2:
            mflow_inj_n2        = h5_post['/othr_data/ssD_othr_acc/mflow_inj_n'][timestep,1]
            mflow_fwmat_n2      = h5_post['/othr_data/ssD_othr_acc/mflow_fwmat_n'][timestep,1]
        # Values composing flows to wall
        mflow_twmat_i1      = h5_post['/othr_data/ssD_othr_acc/mflow_twmat_i'][timestep,0]
        mflow_twinf_i1      = h5_post['/othr_data/ssD_othr_acc/mflow_twinf_i'][timestep,0]
        mflow_twa_i1        = h5_post['/othr_data/ssD_othr_acc/mflow_twa_i'][timestep,0]
        mflow_twmat_i2      = h5_post['/othr_data/ssD_othr_acc/mflow_twmat_i'][timestep,1]
        mflow_twinf_i2      = h5_post['/othr_data/ssD_othr_acc/mflow_twinf_i'][timestep,1]
        mflow_twa_i2        = h5_post['/othr_data/ssD_othr_acc/mflow_twa_i'][timestep,1]
        mflow_twmat_n1      = h5_post['/othr_data/ssD_othr_acc/mflow_twmat_n'][timestep,0]
        mflow_twinf_n1      = h5_post['/othr_data/ssD_othr_acc/mflow_twinf_n'][timestep,0]
        mflow_twa_n1        = h5_post['/othr_data/ssD_othr_acc/mflow_twa_n'][timestep,0]
        mflow_twmat_n2      = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_twinf_n2      = np.zeros(np.shape(dMdt_n1),dtype=float)
        mflow_twa_n2        = np.zeros(np.shape(dMdt_n1),dtype=float)
        if num_neu_spe == 2:
            mflow_twmat_n2      = h5_post['/othr_data/ssD_othr_acc/mflow_twmat_n'][timestep,1]
            mflow_twinf_n2      = h5_post['/othr_data/ssD_othr_acc/mflow_twinf_n'][timestep,1]
            mflow_twa_n2        = h5_post['/othr_data/ssD_othr_acc/mflow_twa_n'][timestep,1]
        # Obtain mass balances
        mbal_n1 = mflow_coll_n1 + mflow_fw_n1 - mflow_tw_n1
        mbal_i1 = mflow_coll_i1 + mflow_fw_i1 - mflow_tw_i1
        mbal_i2 = mflow_coll_i2 + mflow_fw_i2 - mflow_tw_i2
        mbal_tot = mbal_n1 + mbal_i1 + mbal_i2
        dMdt_tot = dMdt_n1 + dMdt_i1 + dMdt_i2
        # Obtain mass balance errors
#        err_mbal_n1     = np.abs(mbal_n1 - dMdt_n1)/np.abs(mflow_inj_n1)
#        err_mbal_i1     = np.abs(mbal_i1 - dMdt_i1)/np.abs(mflow_inj_n1)
#        err_mbal_i2     = np.abs(mbal_i2 - dMdt_i2)/np.abs(mflow_inj_n1)
        err_mbal_n1     = np.abs(mbal_n1 - dMdt_n1)/np.abs(m_A)
        err_mbal_i1     = np.abs(mbal_i1 - dMdt_i1)/np.abs(m_A)
        err_mbal_i2     = np.abs(mbal_i2 - dMdt_i2)/np.abs(m_A)
        err_mbal_tot    = np.abs(mbal_tot - dMdt_tot)/np.abs(m_A)
        # Obtain contributions to the mass balances
        abs_mbal_n1        = np.abs(mflow_coll_n1) + np.abs(mflow_fw_n1) + np.abs(mflow_tw_n1)
        ctr_mflow_coll_n1  = np.abs(mflow_coll_n1)/abs_mbal_n1
        ctr_mflow_fw_n1    = np.abs(mflow_fw_n1)/abs_mbal_n1
        ctr_mflow_tw_n1    = np.abs(mflow_tw_n1)/abs_mbal_n1
        abs_mbal_i1        = np.abs(mflow_coll_i1) + np.abs(mflow_fw_i1) + np.abs(mflow_tw_i1)
        ctr_mflow_coll_i1  = np.abs(mflow_coll_i1)/abs_mbal_i1
        ctr_mflow_fw_i1    = np.abs(mflow_fw_i1)/abs_mbal_i1
        ctr_mflow_tw_i1    = np.abs(mflow_tw_i1)/abs_mbal_i1
        abs_mbal_i2        = np.abs(mflow_coll_i2) + np.abs(mflow_fw_i2) + np.abs(mflow_tw_i2)
        ctr_mflow_coll_i2  = np.abs(mflow_coll_i2)/abs_mbal_i2
        ctr_mflow_fw_i2    = np.abs(mflow_fw_i2)/abs_mbal_i2
        ctr_mflow_tw_i2    = np.abs(mflow_tw_i2)/abs_mbal_i2
        abs_mbal_tot       = np.abs(mflow_coll_n1 + mflow_coll_i1 + mflow_coll_i2) + np.abs(mflow_fw_n1 + mflow_fw_i1 + mflow_fw_i2) + np.abs(mflow_tw_n1 + mflow_tw_i1 + mflow_tw_i2)
        ctr_mflow_coll_tot = np.abs(mflow_coll_n1 + mflow_coll_i1 + mflow_coll_i2)/abs_mbal_tot
        ctr_mflow_fw_tot   = np.abs(mflow_fw_n1 + mflow_fw_i1 + mflow_fw_i2)/abs_mbal_tot
        ctr_mflow_tw_tot   = np.abs(mflow_tw_n1 + mflow_tw_i1 + mflow_tw_i2)/abs_mbal_tot
        # Energy balance ----------------------------------------------------------  
        if ene_bal == 1:
            dEdt_i1         = h5_post['/othr_data/ssD_othr_acc/dene_mp_ions_acc'][timestep,0]
            dEdt_i2         = h5_post['/othr_data/ssD_othr_acc/dene_mp_ions_acc'][timestep,1]
            dEdt_n1         = h5_post['/othr_data/ssD_othr_acc/dene_mp_neus_acc'][timestep,0]
            eneflow_coll_i1 = h5_post['/othr_data/ssD_othr_acc/eneflow_coll_i_acc'][timestep,0]
            eneflow_coll_i2 = h5_post['/othr_data/ssD_othr_acc/eneflow_coll_i_acc'][timestep,1]
            eneflow_coll_n1 = h5_post['/othr_data/ssD_othr_acc/eneflow_coll_n_acc'][timestep,0]
            eneflow_fw_i1   = h5_post['/othr_data/ssD_othr_acc/eneflow_fw_i'][timestep,0]
            eneflow_fw_i2   = h5_post['/othr_data/ssD_othr_acc/eneflow_fw_i'][timestep,1]
            eneflow_fw_n1   = h5_post['/othr_data/ssD_othr_acc/eneflow_fw_n'][timestep,0]
            eneflow_tw_i1   = h5_post['/othr_data/ssD_othr_acc/eneflow_tw_i'][timestep,0]
            eneflow_tw_i2   = h5_post['/othr_data/ssD_othr_acc/eneflow_tw_i'][timestep,1]
            eneflow_tw_n1   = h5_post['/othr_data/ssD_othr_acc/eneflow_tw_n'][timestep,0]
            Pfield_i1       = h5_post['/othr_data/ssD_othr_acc/Pfield_i_acc'][timestep,0]
            Pfield_i2       = h5_post['/othr_data/ssD_othr_acc/Pfield_i_acc'][timestep,1]
            dEdt_n2         = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_coll_n2 = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_fw_n2   = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_tw_n2   = np.zeros(np.shape(dEdt_n1),dtype=float)
            if num_neu_spe == 2:
                dEdt_n2         = h5_post['/othr_data/ssD_othr_acc/dene_mp_neus_acc'][timestep,1]
                eneflow_coll_n2 = h5_post['/othr_data/ssD_othr_acc/eneflow_coll_n_acc'][timestep,1]
                eneflow_fw_n2   = h5_post['/othr_data/ssD_othr_acc/eneflow_fw_n'][timestep,1]
                eneflow_tw_n2   = h5_post['/othr_data/ssD_othr_acc/eneflow_tw_n'][timestep,1]
                
            # Values composing flows from wall
            eneflow_inj_i1        = h5_post['/othr_data/ssD_othr_acc/eneflow_inj_i'][timestep,0]
            eneflow_fwmat_i1      = h5_post['/othr_data/ssD_othr_acc/eneflow_fwmat_i'][timestep,0]
            eneflow_inj_i2        = h5_post['/othr_data/ssD_othr_acc/eneflow_inj_i'][timestep,1]
            eneflow_fwmat_i2      = h5_post['/othr_data/ssD_othr_acc/eneflow_fwmat_i'][timestep,1]
            eneflow_inj_n1        = h5_post['/othr_data/ssD_othr_acc/eneflow_inj_n'][timestep,0]
            eneflow_fwmat_n1      = h5_post['/othr_data/ssD_othr_acc/eneflow_fwmat_n'][timestep,0]
            eneflow_inj_n2        = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_fwmat_n2      = np.zeros(np.shape(dEdt_n1),dtype=float)
            if num_neu_spe == 2:
                eneflow_inj_n2        = h5_post['/othr_data/ssD_othr_acc/eneflow_inj_n'][timestep,1]
                eneflow_fwmat_n2      = h5_post['/othr_data/ssD_othr_acc/eneflow_fwmat_n'][timestep,1]
            # Values composing flows to wall
            eneflow_twmat_i1      = h5_post['/othr_data/ssD_othr_acc/eneflow_twmat_i'][timestep,0]
            eneflow_twinf_i1      = h5_post['/othr_data/ssD_othr_acc/eneflow_twinf_i'][timestep,0]
            eneflow_twa_i1        = h5_post['/othr_data/ssD_othr_acc/eneflow_twa_i'][timestep,0]
            eneflow_twmat_i2      = h5_post['/othr_data/ssD_othr_acc/eneflow_twmat_i'][timestep,1]
            eneflow_twinf_i2      = h5_post['/othr_data/ssD_othr_acc/eneflow_twinf_i'][timestep,1]
            eneflow_twa_i2        = h5_post['/othr_data/ssD_othr_acc/eneflow_twa_i'][timestep,1]
            eneflow_twmat_n1      = h5_post['/othr_data/ssD_othr_acc/eneflow_twmat_n'][timestep,0]
            eneflow_twinf_n1      = h5_post['/othr_data/ssD_othr_acc/eneflow_twinf_n'][timestep,0]
            eneflow_twa_n1        = h5_post['/othr_data/ssD_othr_acc/eneflow_twa_n'][timestep,0]
            eneflow_twmat_n2      = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_twinf_n2      = np.zeros(np.shape(dEdt_n1),dtype=float)
            eneflow_twa_n2        = np.zeros(np.shape(dEdt_n1),dtype=float)
            if num_neu_spe == 2:
                eneflow_twmat_n2      = h5_post['/othr_data/ssD_othr_acc/eneflow_twmat_n'][timestep,1]
                eneflow_twinf_n2      = h5_post['/othr_data/ssD_othr_acc/eneflow_twinf_n'][timestep,1]
                eneflow_twa_n2        = h5_post['/othr_data/ssD_othr_acc/eneflow_twa_n'][timestep,1] 
        else:
            dEdt_i1         = 0.0
            dEdt_i2         = 0.0
            dEdt_n1         = 0.0
            dEdt_n2         = 0.0
            eneflow_coll_i1 = 0.0
            eneflow_coll_i2 = 0.0
            eneflow_coll_n1 = 0.0
            eneflow_coll_n2 = 0.0
            eneflow_fw_i1   = 0.0
            eneflow_fw_i2   = 0.0
            eneflow_fw_n1   = 0.0
            eneflow_fw_n2   = 0.0
            eneflow_tw_i1   = 0.0
            eneflow_tw_i2   = 0.0
            eneflow_tw_n1   = 0.0
            eneflow_tw_n2   = 0.0
            Pfield_i1       = 0.0
            Pfield_i2       = 0.0
            # Values composing flows from wall
            eneflow_inj_i1        = 0.0
            eneflow_fwmat_i1      = 0.0
            eneflow_inj_i2        = 0.0
            eneflow_fwmat_i2      = 0.0
            eneflow_inj_n1        = 0.0
            eneflow_fwmat_n1      = 0.0
            eneflow_inj_n2        = 0.0
            eneflow_fwmat_n2      = 0.0
            # Values composing flows to wall
            eneflow_twmat_i1      = 0.0
            eneflow_twinf_i1      = 0.0
            eneflow_twa_i1        = 0.0
            eneflow_twmat_i2      = 0.0
            eneflow_twinf_i2      = 0.0
            eneflow_twa_i2        = 0.0
            eneflow_twmat_n1      = 0.0
            eneflow_twinf_n1      = 0.0
            eneflow_twa_n1        = 0.0
            eneflow_twmat_n2      = 0.0
            eneflow_twinf_n2      = 0.0
            eneflow_twa_n2        = 0.0
            
        # Obtain the total net ion and neutral power deposited to the walls at all boundaries (per type of boundary)
        Pi_Dwall       = eneflow_twmat_i1 + eneflow_twmat_i2 - (eneflow_fwmat_i1 + eneflow_fwmat_i2) 
        Pi_Awall       = eneflow_twa_i1 + eneflow_twa_i2 - (eneflow_inj_i1 + eneflow_inj_i2) 
        Pi_FLwall      = eneflow_twinf_i1 + eneflow_twinf_i2 
        Pn_Dwall       = eneflow_twmat_n1 - eneflow_fwmat_n1 
        Pn_Awall       = eneflow_twa_n1 - eneflow_inj_n1 
        Pn_FLwall      = eneflow_twinf_n1
        # NOTE: At the free loss boundary the energy flux given by the sheath is zero, so that the electron energy deposited is
        Pe_FLwall = P_inf - (Pi_FLwall + Pn_FLwall)
        # Obtain the ion and neutral power deposited to the free loss which is not axial (not generating thrust)
        Pi_FLwall_nonz = Pi_FLwall - P_use_z_i
        Pn_FLwall_nonz = Pn_FLwall - P_use_z_n
        # Obtain total net power deposited to the walls by both the electrons and the heavy species
        P_Dwall  = Pe_Dwall + Pi_Dwall + Pn_Dwall
        P_Awall  = Pe_Awall + Pi_Awall + Pn_Awall
        P_FLwall = Pe_FLwall + Pi_FLwall + Pn_FLwall
        # Obtain the total energy balance
        Pwalls  = P_Dwall + P_Awall + P_FLwall 
        Pionex  = P_ion + P_ex
        Ploss   = Pwalls + Pionex
        balP    = Pd - Ploss
        # Obtain the total power not used for generating thrust
        Pnothrust       = Pionex + P_Dwall + P_Awall + Pe_FLwall + Pi_FLwall_nonz + Pn_FLwall_nonz
        Pnothrust_walls = Pnothrust - Pionex
        # Obtain the total power used for generating thrust
        Pthrust   = P_use_z
        # Alternative total energy balance
        balP_Pthrust = Pd - (Pnothrust + Pthrust)
        # Obtain error in total energy balance
        err_balP         = np.abs(balP)/Pd  
        err_balP_Pthrust = np.abs(balP_Pthrust)/Pd 
        err_def_balP = np.abs(balP - balP_Pthrust)/np.abs(balP)
        # Contributions to the energy balance
        abs_balP         = np.abs(Pd) + np.abs(Ploss)
        ctr_Pd           = np.abs(Pd)/abs_balP
        ctr_Ploss        = np.abs(Ploss)/abs_balP
        ctr_Pwalls       = np.abs(Pwalls)/abs_balP
        ctr_Pionex       = np.abs(Pionex)/abs_balP
        ctr_P_DAwalls    = np.abs(P_Dwall + P_Awall)/abs_balP
        ctr_P_FLwalls    = np.abs(P_FLwall)/abs_balP
        ctr_P_FLwalls_in = np.abs(Pi_FLwall+Pn_FLwall)/abs_balP
        ctr_P_FLwalls_i  = np.abs(Pi_FLwall)/abs_balP
        ctr_P_FLwalls_n  = np.abs(Pn_FLwall)/abs_balP
        ctr_P_FLwalls_e  = np.abs(Pe_FLwall)/abs_balP
        abs_balP_Pthrust         = np.abs(Pd) + np.abs(Pnothrust) + np.abs(Pthrust)
        ctr_balPthrust_Pd        = np.abs(Pd)/abs_balP_Pthrust
        ctr_balPthrust_Pnothrust = np.abs(Pnothrust)/abs_balP_Pthrust
        ctr_balPthrust_Pthrust   = np.abs(Pthrust)/abs_balP_Pthrust
        ctr_balPthrust_Pnothrust_walls = np.abs(Pnothrust_walls)/abs_balP_Pthrust
        ctr_balPthrust_Pnothrust_ionex = np.abs(Pionex)/abs_balP_Pthrust
        
        # Obtain the ionization source term (ni_dot) per cell for each ionization collision
        # This is obtained as the absolute value of the neutral mass loss per cell and collision
        # (i.e. neutral species)
        # NOTE: NOT ADAPTED FOR 2 NEUTRAL SPECIES
        if n_collisions > 0:
            ndot_ion01_n1 = abs(h5_post['/picM_data/dm_coll_acc/coll1_inp1_n1'][0:,0:,timestep])/(dt*mass) 
            ndot_ion02_n1 = abs(h5_post['/picM_data/dm_coll_acc/coll2_inp1_n1'][0:,0:,timestep])/(dt*mass) 
            ndot_ion12_i1 = abs(h5_post['/picM_data/dm_coll_acc/coll3_inp1_i1'][0:,0:,timestep])/(dt*mass)
            for i in range(0,dims[0]-1):
                for j in range(0,dims[1]-1):
                    ndot_ion01_n1[i,j] = ndot_ion01_n1[i,j]/cells_vol[i,j]
                    ndot_ion02_n1[i,j] = ndot_ion02_n1[i,j]/cells_vol[i,j]
                    ndot_ion12_i1[i,j] = ndot_ion12_i1[i,j]/cells_vol[i,j]
        else:
            ndot_ion01_n1 = 0.0
            ndot_ion02_n1 = 0.0
            ndot_ion12_i1 = 0.0
            
        # Obtain cathode production frequency, mass and energy source term and power
        nu_cath   = (Id/V_cath)/(e*ne_cath)
        ndot_cath = ne_cath*nu_cath
        Q_cath    = 3.0/2.0*ndot_cath*e*T_cath
        P_cath    = V_cath*Q_cath
        
        # Obtain cathode production frequency, mass and energy source term and power
        if len(cath_elem) == 1:
            V_cath_tot  = V_cath
            ne_cath_avg = ne_cath
            nu_cath   = (Id/V_cath)/(e*ne_cath)
            ndot_cath = ne_cath*nu_cath
            Q_cath    = 3.0/2.0*ndot_cath*e*T_cath
            P_cath    = V_cath*Q_cath
        elif len(cath_elem) > 1:
            V_cath_tot  = np.sum(V_cath)
            ne_cath_avg = 0.0
            for ind_cath in range(0,len(cath_elem)):
                ne_cath_avg = ne_cath_avg + ne_cath[ind_cath]*V_cath[ind_cath]/V_cath_tot
            nu_cath   = (Id/V_cath_tot)/(e*ne_cath_avg)
            ndot_cath = ne_cath_avg*nu_cath
            Q_cath    = 3.0/2.0*ndot_cath*e*T_cath
            P_cath    = V_cath_tot*Q_cath
        
            
        # Obtain extra data at the PIC mesh if required, including anomalous term, Hall parameter and collision frequencies
        if read_flag == 1:
            F_theta      = h5_post['/picM_data/F_theta_acc'][:,:,timestep]
            Hall_par     = h5_post['/picM_data/Hall_Beta_e_tot_acc'][:,:,timestep]
            Hall_par_eff = h5_post['/picM_data/Hall_Beta_e_tot_eff_acc'][:,:,timestep]
            nu_e_tot     = h5_post['/picM_data/freq_e_tot_acc'][:,:,timestep]
            nu_e_tot_eff = h5_post['/picM_data/freq_e_tot_eff_acc'][:,:,timestep]
            if oldpost_sim != 3:
                nu_en        = h5_post['/picM_data/freq_en_acc_prop1'][:,:,timestep]
                nu_ei1       = h5_post['/picM_data/freq_ei1_acc_prop1'][:,:,timestep]
                nu_ei2       = h5_post['/picM_data/freq_ei2_acc_prop1'][:,:,timestep]
                nu_i01       = h5_post['/picM_data/freq_i01_acc_prop1'][:,:,timestep]
                nu_i02       = h5_post['/picM_data/freq_i02_acc_prop1'][:,:,timestep]
                nu_i12       = h5_post['/picM_data/freq_i12_acc_prop1'][:,:,timestep] 
                nu_ex        = np.zeros(np.shape(nu_e_tot),dtype=float)
            elif oldpost_sim == 3:
                nu_en        = np.zeros(np.shape(nu_e_tot),dtype=float)
                nu_ei1       = np.zeros(np.shape(nu_e_tot),dtype=float)
                nu_ei2       = np.zeros(np.shape(nu_e_tot),dtype=float)
                nu_i01       = np.zeros(np.shape(nu_e_tot),dtype=float)
                nu_i02       = np.zeros(np.shape(nu_e_tot),dtype=float)
                nu_i12       = np.zeros(np.shape(nu_e_tot),dtype=float)  
                nu_ex        = np.zeros(np.shape(nu_e_tot),dtype=float) 
                freq_e = np.zeros((n_collisions_e,dims[0],dims[1]),dtype=float)
                for i in range(0,n_collisions_e):
                    freq_e[i,:,:] = h5_post['/picM_data/freq_e_acc'+str(i+1)][:,:,timestep]
                for i in range(0,n_collisions_e):
                    if ids_collisions_e[i] == 1:
                        # Electron-neutral elastic collision
                        nu_en = nu_en + freq_e[i,:,:]
                    if ids_collisions_e[i] == 2:
                        # Ionization collision
                        if coll_spe_e[i] < 0 and Z_ion_spe[out_coll_spe_e[i]-1] == 1:
                            # Ionization 0-1
                            nu_i01 = nu_i01 + freq_e[i,:,:]
                        elif coll_spe_e[i] < 0 and Z_ion_spe[out_coll_spe_e[i]-1] == 2:
                            # Ionization 0-2
                            nu_i02 = nu_i02 + freq_e[i,:,:]
                        elif coll_spe_e[i] > 0 and Z_ion_spe[out_coll_spe_e[i]-1] - Z_ion_spe[coll_spe_e[i]-1] == 1:
                            # Ionization 1-2
                            nu_i12 = nu_i12 + freq_e[i,:,:]
                    elif ids_collisions_e[i] == 4:
                        # Electron-neutral excitation collision
                        nu_ex = nu_ex + freq_e[i,:,:]
                    elif ids_collisions_e[i] == 5:
                        # Electron-ion Coulomb collision (elastic)
                        if Z_ion_spe[coll_spe_e[i]-1] == 1:
                            # Electron-singly charged ion collision
                            nu_ei1 = nu_ei1 + freq_e[i,:,:]
                        elif Z_ion_spe[coll_spe_e[i]-1] == 2:
                            # Electron-doubly charged ion collision
                            nu_ei2 = nu_ei2 + freq_e[i,:,:]
        else:
            F_theta      = np.zeros(np.shape(Te),dtype=float)
            Hall_par     = np.zeros(np.shape(Te),dtype=float)
            Hall_par_eff = np.zeros(np.shape(Te),dtype=float)
            nu_e_tot     = np.zeros(np.shape(Te),dtype=float)
            nu_e_tot_eff = np.zeros(np.shape(Te),dtype=float)
            nu_en        = np.zeros(np.shape(Te),dtype=float)
            nu_ei1       = np.zeros(np.shape(Te),dtype=float)
            nu_ei2       = np.zeros(np.shape(Te),dtype=float)
            nu_i01       = np.zeros(np.shape(Te),dtype=float)
            nu_i02       = np.zeros(np.shape(Te),dtype=float)
            nu_i12       = np.zeros(np.shape(Te),dtype=float) 
            nu_ex        = np.zeros(np.shape(Te),dtype=float) 

        
        # NEW ENERGY BALANCE FOR ELECTRONS
        Pfield_e = h5_post['/othr_data/ssD_othr_acc/Pfield_e'][timestep,0]
        Ebal_e   = Pfield_e + P_cath - (Pe_Dwall + Pe_Awall + Pe_FLwall + Pionex)

        # Read boundary variables
        if oldpost_sim == 7:
            # Obtain the SEE yields and electron/ion total fluxes
            ge_b        = h5_post['/picM_data/ji_tot_b'][0:,0:,timestep]/e
            ge_b_acc    = h5_post['/picM_data/ji_tot_b_acc'][0:,0:,timestep]/e
            ge_sb_b     = h5_post['/picM_data/ge_sb_b'][0:,0:,timestep]
            ge_sb_b_acc = h5_post['/picM_data/ge_sb_b_acc'][0:,0:,timestep]
            delta_see     = np.divide(ge_sb_b,ge_sb_b + ge_b)
            delta_see_acc = np.divide(ge_sb_b_acc,ge_sb_b_acc + ge_b_acc)
            #delta_see      = ge_sb_b/(ge_sb_b + ge_b)
            #delta_see_acc  = ge_sb_b_acc/(ge_sb_b_acc + ge_b_acc)
        else:
            ge_b        = np.zeros(np.shape(Te),dtype=float)
            ge_b_acc    = np.zeros(np.shape(Te),dtype=float)
            ge_sb_b     = np.zeros(np.shape(Te),dtype=float)
            ge_sb_b_acc = np.zeros(np.shape(Te),dtype=float)
            delta_see     = np.zeros(np.shape(Te),dtype=float)
            delta_see_acc = np.zeros(np.shape(Te),dtype=float)
        
        # Read interpolation errors for density
        if oldpost_sim == 3:
#            err_interp_n = h5_post['/picM_data/err_interp_n_acc'][0:,0:,timestep]
            err_interp_n = np.zeros(np.shape(Te),dtype=float)
        else:
            err_interp_n = np.zeros(np.shape(Te),dtype=float)
        
        
    # Obtain other instantaneous data (mass and number of particles)
    tot_mass_mp_neus   = 0.0
    tot_mass_mp_ions   = 0.0
    tot_num_mp_neus    = 0
    tot_num_mp_ions    = 0    
    tot_mass_exit_neus = 0.0
    tot_mass_exit_ions = 0.0
    mass_mp_neus   = 0.0
    mass_mp_ions   = 0.0
    num_mp_neus    = 0
    num_mp_ions    = 0   
    avg_dens_mp_neus = 0.0
    avg_dens_mp_ions = 0.0
    if read_inst_data == 1:    
        tot_mass_mp_neus = h5_post['/othr_data/isD_othr/tot_mass_mp_neus'][:,0]   
        tot_mass_mp_ions = h5_post['/othr_data/isD_othr/tot_mass_mp_ions'][:,0] 
        tot_num_mp_neus = h5_post['/othr_data/isD_othr/tot_num_mp_neus'][:,0]   
        tot_num_mp_ions = h5_post['/othr_data/isD_othr/tot_num_mp_ions'][:,0] 
        tot_mass_exit_neus = h5_post['/othr_data/isD_othr/tot_mass_exit_neus'][:,0] 
        tot_mass_exit_ions = h5_post['/othr_data/isD_othr/tot_mass_exit_ions'][:,0] 
        mass_mp_neus = h5_post['/othr_data/isD_othr/mass_mp_neus'][:,:]   
        mass_mp_ions = h5_post['/othr_data/isD_othr/mass_mp_ions'][:,:] 
        num_mp_neus = h5_post['/othr_data/isD_othr/num_mp_neus'][:,:]   
        num_mp_ions = h5_post['/othr_data/isD_othr/num_mp_ions'][:,:] 
        avg_dens_mp_neus   = tot_mass_mp_neus/(mass*volume)
        avg_dens_mp_ions   = tot_mass_mp_ions/(mass*volume)                            # This is total density of ions
        if num_ion_spe > 1:
            avg_dens_mp_ions   = (mass_mp_ions[:,0] + 2.0*mass_mp_ions[:,1])/(mass*volume)  # This is total density of electrons (plasma)

    if allsteps_flag == 1:
        # Set to zero possible nan and inf values in the total energy balance relative error
        err_balP[np.where(np.isinf(err_balP))] = 0.0
        err_balP[np.where(np.isnan(err_balP))] = 0.0
        err_balP_Pthrust[np.where(np.isinf(err_balP_Pthrust))] = 0.0
        err_balP_Pthrust[np.where(np.isnan(err_balP_Pthrust))] = 0.0
        err_def_balP[np.where(np.isinf(err_def_balP))] = 0.0
        err_def_balP[np.where(np.isnan(err_def_balP))] = 0.0
        
    # Obtain the Maxwell-Boltzmann equilibrium law (or isothermal Boltzmann relation)
    Boltz     = np.zeros(np.shape(ne_elems),dtype=float)
    Boltz_dim = np.zeros(np.shape(ne_elems),dtype=float)
    warning_flag = 0
    warning_cont = 0
    if len(cath_elem) == 1:
        if allsteps_flag == 1:
            for i in range(0,nsteps):
                ### TEST: fix a value below 0.2 for the cathode temperature in case C3. Take value from near PIC node
                if Te_cath[i] < 0.2:
                    Te_cath[i] = Te[6,20,i]
                    Te_elems[i,cath_elem] = Te[6,20,i]
                    warning_flag = 1
                    warning_cont = warning_cont + 1
                ######################################################################
                Boltz[i,:] = e*(phi_elems[i,:]-phi_cath[i])/(e*Te_cath[i]) - np.log(ne_elems[i,:]/ne_cath[i])
                Boltz_dim[i,:] = Te_cath[i]*Boltz[i,:]
        elif allsteps_flag == 0:
            ### TEST: fix a value below 0.2 for the cathode temperature in case C3. Take value from near PIC node
            if Te_cath < 0.2:
                Te_cath             = Te[6,20]
                Te_elems[cath_elem] = Te[6,20]
                warning_flag = 1
                warning_cont = warning_cont + 1
            ######################################################################
            Boltz = e*(phi_elems-phi_cath)/(e*Te_cath) - np.log(ne_elems/ne_cath)
            Boltz_dim = Te_cath*Boltz
            
        if warning_flag == 1:
            print("HET_sims_read WARNING: Te at the cathode MFAM element has been modified (too low values) "+str(warning_cont)+" times")
        
        
    
    
#    # TEST ====================================================================
#    e    = 1.6021766E-19
#    me   = 9.1093829E-31
#    Te_inst      = h5_post["/picM_data/Te"][0:,0:,:]
#    ne_inst      = h5_post["/picM_data/n"][0:,0:,:]
#    je_r_inst    = h5_post["/picM_data/je_r"][0:,0:,:]
#    je_t_inst    = h5_post["/picM_data/je_theta"][0:,0:,:]
#    je_z_inst    = h5_post["/picM_data/je_z"][0:,0:,:]
#    je_perp_inst = h5_post["/picM_data/je_perp"][0:,0:,:]
#    je_para_inst = h5_post["/picM_data/je_para"][0:,0:,:]
#    ue_r_inst    = np.divide(je_r_inst,-e*ne_inst) 
#    ue_t_inst    = np.divide(je_t_inst,-e*ne_inst)
#    ue_z_inst    = np.divide(je_z_inst,-e*ne_inst)
#    ue_perp_inst = np.divide(je_perp_inst,-e*ne_inst)
#    ue_para_inst = np.divide(je_para_inst,-e*ne_inst)
#    ue_inst     = np.sqrt(ue_r_inst**2 + ue_t_inst**2 + ue_z_inst**2)
#    
#    ratio_Ekin_Te_inst = 0.5*me*ue_inst**2/e/Te_inst
#    print(np.nanmax(ratio_Ekin_Te_inst[:,:,-1]),np.nanmin(ratio_Ekin_Te_inst[:,:,-1]))
#    # =========================================================================
        
        

    
    return[num_ion_spe,num_neu_spe,n_mp_cell_i,n_mp_cell_n,n_mp_cell_i_min,
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
           delta_see,delta_see_acc,err_interp_n]
           
       
       
"""
############################################################################
Description:    This python function interpolate data from PIC mesh cells 
                centers to PIC mesh nodes
############################################################################
Inputs:         1) dims: PIC mesh dimensions
                2) zs,rs: PIC mesh nodes matrices
                3) zscells,rscells: PIC mesh cells centers matrices
                4) var_cells: variable at PIC mesh cells to be interpolated
############################################################################
Output:         1) var_nodes: variable interpolated to the PIC mesh nodes
"""
           
def interp_cells2nodes(dims,zs,rs,zscells,rscells,var_cells):
    
    import numpy as np
    from scipy import interpolate
    
    var_nodes = np.zeros(dims,dtype=float)    
    npoints_r = dims[0]-1
    npoints_z = dims[1]-1
    points     = np.zeros((int(npoints_r*npoints_z),2),dtype='float')
    var_points = np.zeros((int(npoints_r*npoints_z),1),dtype='float')    
    
    ind = 0
    for i in range(0,int(npoints_r)):
        for j in range(0,int(npoints_z)):
           points[ind,0] = zscells[i,j]
           points[ind,1] = rscells[i,j]
           var_points[ind,0] = var_cells[i,j]
           ind = ind + 1

    var_interp = interpolate.griddata(points, var_points, (zs, rs), method='linear')
    var_nodes[:,:] = var_interp[:,:,0]

    return var_nodes
    
    
           
if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 
    from scipy import interpolate
    from scipy.interpolate import RegularGridInterpolator

    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))
    
    
#    sim_name = "../../Rb_hyphen/sim/sims/SPT100_al0025_Ne5_C1"
#    sim_name = "../../../Rb_hyphen/sim/sims/SPT100_al0025_Ne5_C1"
#    sim_name = "../../../sim/sims/SPT100_al0025_Ne5_C1_qboundary"
#    sim_name = "../../../Rb_hyphen/sim/sims/SAFRAN_HET_topo2_l200s200_cat3298_relaunch"
    
#    sim_name = "../../../Rb_hyphen/sim/sims/SAFRAN_HET_topo2_l200s200_cat1200_also125_RLC"
#    sim_name = "../../../Rb_hyphen/sim/sims/SAFRAN_HET_topo2_l200s200_cat1200_tm110_tetq125"
    
#    sim_name = "../../../sim/sims/SAFRAN_HET_topo2_l200s200_cat1200_tm110_tetq125_ECath"

#    sim_name = "../../../Rb_hyphen/sim/sims/SAFRAN_HET_topo2_l200s200_cat1200_tm110_tetq125_RLC"
#    sim_name = "../../../Rb_hyphen/sim/sims/SAFRAN_HET_topo2_l200s200_cat1200_tm110_tetq125_RLC_Coll"
    
#    sim_name = "../../../Ca_sims_files/SPT100_thesis_REF_MFAMjesus_rm2_picrm"
    sim_name = "../../../sim/sims/SPT100_thesis_REF_MFAMjesus_rm2_picrm"
#    sim_name = "../../../Ca_sims_files/SPT100_thesis_REF_MFAMjesus_rm2_picrm_aljpara"

    timestep         = -1
    allsteps_flag    = 1
    read_inst_data   = 1
    read_part_tracks = 0
    read_part_lists  = 0
    read_flag        = 1
    
    oldpost_sim      = 3
    oldsimparams_sim = 8
    
#    elems_cath_Bline   = range(407-1,483-1+2,2) # Elements along the cathode B line for cases C1, C2 and C3
#    elems_cath_Bline   = range(875-1,951-1+2,2) # Elements along the cathode B line for case C5
    elems_cath_Bline   = range(3238-1,3330-1+1,1)  # Elements along cathode B line for SPT100_MFAM_Ref1500pts_rm2
    
    elems_cath_Bline_2   = range(1739-1,1555-1+2,-2)
    
    
    last_steps = 700
    
    
#    path_picM         = sim_name+"/SET/inp/SPT100_picM.hdf5"
#    path_picM         = sim_name +"/SET/inp/PIC_mesh_topo2_refined4.hdf5"
    path_picM         = sim_name +"/SET/inp/SPT100_picM_Reference1500points_rm.hdf5"
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
       delta_see,delta_see_acc] = HET_sims_read(path_simstate_inp,path_simstate_out,
                                                              path_postdata_out,path_simparams_inp,
                                                              path_picM,allsteps_flag,timestep,read_inst_data,
                                                              read_part_lists,read_flag,oldpost_sim,oldsimparams_sim)
    
#    # Obtain the total net power of the heavy species deposited to the injection (anode) wall
#    P_inj_hs = eneflow_twa_i1 + eneflow_twa_i2 + eneflow_twa_n1 - (eneflow_inj_i1 + eneflow_inj_i2 + eneflow_inj_n1)
#    
#    P_inj_electrons = P_inj - P_inj_hs
#    P_inj_electrons2 = np.sum(Pe_faces_Awall,axis=1)
#    P_mat_electrons  = np.sum(Pe_faces_Awall,axis=1) 
#    
#    plt.figure("P_inj")
##    plt.plot(time,P_inj_electrons,'r')
#    plt.plot(time,P_inj_electrons2,'g')
##    plt.plot(time,P_inj,'b')
#    plt.plot(time,P_inj_hs,'k')
#    
#    print("Mean value P_inj_electrons2 = %15.8e" %np.mean(P_inj_electrons2[600::])  ) 
#    print("Mean value P_inj_electrons2 = %15.8e" %np.mean(Pe_Awall[600::])  ) 
    

#    print("Mean value err_balP = %15.8e" %np.mean(err_balP[600::])  ) 
#    print("Mean value Pd       = %15.8e" %np.mean(Pd[600::])  ) 
#    print("Mean value Ploss    = %15.8e" %np.mean(Ploss[600::])  ) 
#    print("Mean value Pwalls   = %15.8e" %np.mean(Pwalls[600::])  ) 
#    print("Mean value Pion_ex  = %15.8e" %np.mean(Pionex[600::])  ) 
    

    
    
    plt.figure("balP")
#    plt.plot(time,P_inj_electrons,'r')
    plt.plot(time,balP,'k')
    plt.plot(time,Pd,'r',label='Pd')
    plt.plot(time,Ploss,'g',label='Ploss = Pionex + Pwall')
    plt.plot(time,Pionex,'b',label='Pionex')
    plt.plot(time,Pwalls,'m',label='Pwall')
    plt.legend()
    
    plt.figure("balP_contr")
    plt.plot(time,ctr_Pd,'r',label='Pd')
    plt.plot(time,ctr_Ploss,'g',label='Ploss = Pwalls + Pionex')
    plt.plot(time,ctr_Pwalls,'b',label='Pwalls')
    plt.plot(time,ctr_Pionex,'k',label='Pionex')
    plt.plot(time,ctr_Pwalls+ctr_Pionex,'k--',label='Pwalls + Pionex')
    plt.legend()
    
    plt.figure("balP_Pthrust_contr")
    plt.plot(time,ctr_balPthrust_Pd,'r',label='Pd')
    plt.plot(time,ctr_balPthrust_Pthrust,'g',label='Pthrust')
    plt.plot(time,ctr_balPthrust_Pnothrust,'b',label='Pnothrust')
    plt.plot(time,ctr_balPthrust_Pnothrust_walls,'c',label='Pnothrustwalls')
    plt.plot(time,ctr_balPthrust_Pnothrust_ionex,'m',label='Pnothrustionex')
    plt.plot(time,ctr_balPthrust_Pthrust+ctr_balPthrust_Pnothrust,'k--',label='Pthrust + Pnothrust')
    plt.plot(time,ctr_balPthrust_Pnothrust_walls+ctr_balPthrust_Pnothrust_ionex,'y--',label='Pnothrustwalls + Pnothrustionex')
    plt.legend()
    
    plt.figure("err_balP")
    plt.plot(time,err_balP_Pthrust,'r',label='Pd')
    

    print("Mean value Pthrust   = %15.8e" %np.mean(Pthrust[600::])  ) 
    print("Mean value Pnothrust  = %15.8e" %np.mean(Pnothrust[600::])  ) 
    print("ratio mean Pthrust/mean Pd  = %15.8e" %(np.mean(Pthrust[600::])/np.mean(Pd[600::]) )  ) 
    
    Boltz_mean = np.mean(Boltz[600::,:],axis=0)
    phi_elems_mean = np.mean(phi_elems[600::,:],axis=0)
    Te_cath_mean   = np.mean(Te_elems[600::,cath_elem],axis=0)
    
    plt.figure("Boltzmann at cathode B line")
    plt.plot(elem_geom[3,elems_cath_Bline],Boltz_mean[elems_cath_Bline])
    plt.plot(elem_geom[3,cath_elem],Boltz_mean[cath_elem],'ks')
    plt.plot(elem_geom[3,elems_cath_Bline],np.zeros(np.shape(elem_geom[3,elems_cath_Bline])),'--')
    plt.xlabel(r"$\sigma$ (m T)")
    plt.ylabel(r"Boltzmann relation (-)")
    
    plt.figure("phi at cathode B line")
    plt.plot(elem_geom[3,elems_cath_Bline],phi_elems_mean[elems_cath_Bline])
    plt.plot(elem_geom[3,cath_elem],phi_elems_mean[cath_elem],'ks')
    plt.plot(elem_geom[3,elems_cath_Bline],np.zeros(np.shape(elem_geom[3,elems_cath_Bline])),'--')
    plt.xlabel(r"$\sigma$ (m T)")
    plt.ylabel(r"$\phi$ (V)")
    
    
    plt.figure("D walls check electron energy computation")
    plt.plot(time,P_mat,'k-')
    plt.plot(time,P_Dwall,'r-')
    
    plt.figure("A walls check electron energy computation")
    plt.plot(time,P_inj,'k-')
    plt.plot(time,P_Awall,'r-')
    
    plt.figure("FL walls check electron energy computation")
    plt.plot(time,P_inf,'k-')
    plt.plot(time,P_FLwall,'r-')
    
    
    # CHECK FOR NEW COLLISIONS IN EFLD MODULE
    max_relerr_coll = np.zeros(nsteps,dtype=float)
    if oldpost_sim == 3:
        # In new simulatiojns (after the change in the treatment of the collisions in the electron fluid module) the momentum
        # transfer in excitation and ionization collisions is considered, so that they are added to the elastic collision term.
        # Besides, the generation term with the corresponding chrage number jump is also considered. Therefore, the total electron
        # collision frequency contains the excitation and the ionization collision frequencies considered as elastic collisions
        # producing momentum transfer (the latter here without the charge number jump, since this contribution is not from the
        # generation of new electrons from ionization)
        for k in range(0,nsteps):
            sum_nu_coll = nu_en[:,:,k] + nu_ex[:,:,k] + nu_i01[:,:,k] + nu_i02[:,:,k] + nu_i12[:,:,k] + 1.0*nu_i01[:,:,k] + 2.0*nu_i02[:,:,k] + 1.0*nu_i12[:,:,k] + nu_ei1[:,:,k] + nu_ei2[:,:,k]
            tot_nu_coll = nu_e_tot[:,:,k]
            if np.any(tot_nu_coll) != 0.0:
                max_relerr_coll[k] = np.nanmax(np.abs(sum_nu_coll - tot_nu_coll)/np.abs(tot_nu_coll))
    elif oldpost_sim != 3:
        # In old simulations (before the change in the treatment of the collisions in the electron fluid module) the excitation 
        # and ionization collisions are not considered as elastic collisions for momentum transfer (i.e. the momentum transfer due
        # to these collisions is neglected). Therefore, the ionization collisions only appear in the electron momentum equation 
        # due to the generation term. The total electron collision frequency does not consider the momentum transfer in excitation 
        # nor in ionization collisions.
        # On the other hand, the computed frequencies are already given from the electrons perspective, so that the ionization
        # frequencies are already multiplied by the corresponding charge number jump
        for k in range(0,nsteps):
            sum_nu_coll = nu_en[:,:,k] + nu_i01[:,:,k] + nu_i02[:,:,k] + nu_i12[:,:,k] + nu_ei1[:,:,k] + nu_ei2[:,:,k]
            tot_nu_coll = nu_e_tot[:,:,k]
            if np.any(tot_nu_coll) != 0.0:
                max_relerr_coll[k] = np.nanmax(np.abs(sum_nu_coll - tot_nu_coll)/np.abs(tot_nu_coll))
                        
    plt.figure("max error collisions")
    plt.plot(time,max_relerr_coll,'k-')
    
    
    

    # PLOT je along the magnetic line
    # Obtain je_para values directly at the MFAM elements
    je_para_mean      = np.mean(je_para_elems[nsteps-last_steps::,:],axis=0)
    je_para_mean_picM = np.nanmean(je_para[:,:,nsteps-last_steps::],axis=2)
    
    je_para_mean      = np.mean(phi_elems[nsteps-last_steps::,:],axis=0)
    je_para_mean_picM = np.nanmean(phi[:,:,nsteps-last_steps::],axis=2)
    
#    je_para_mean      = np.mean(Te_elems[nsteps-last_steps::,:],axis=0)
#    je_para_mean_picM = np.nanmean(Te[:,:,nsteps-last_steps::],axis=2)
    
    je_para_mean      = je_para_elems[-1,:]
    je_para_mean_picM = je_para[:,:,-1]
    
    je_para_mean      = phi_elems[-1,:]
    je_para_mean_picM = phi[:,:,-1]
    
    
     # Obtain the interpolation function for je_para at the PIC mesh
    rvec = rs[:,0]
    zvec = zs[0,:]
    interp_jepara = RegularGridInterpolator((rvec,zvec), je_para_mean_picM,method='nearest')
    
    # Obtain the vectors containing the coordinates (z,r) of the MFAM elements
    # representing the cathode magnetic field streamline
    r_cath_Bline = elem_geom[1,elems_cath_Bline]
    z_cath_Bline = elem_geom[0,elems_cath_Bline]
    r_cath_Bline_2 = elem_geom[1,elems_cath_Bline_2]
    z_cath_Bline_2 = elem_geom[0,elems_cath_Bline_2]
    
    # Interpolate jepara to the cathode magnetic field streamline
    cath_Bline_jepara = interp_jepara((r_cath_Bline,z_cath_Bline))
    cath_elem_jepara  = interp_jepara((r_cath,z_cath))
    cath_Bline_2_jepara = interp_jepara((r_cath_Bline_2,z_cath_Bline_2))

    
    plt.figure("je_para at cathode Bline")
#    val = -1e-4
    val = 1.0
    plt.plot(elem_geom[3,elems_cath_Bline],val*je_para_mean[elems_cath_Bline],'ks-')
    plt.plot(elem_geom[3,cath_elem],val*je_para_mean[cath_elem],'mo')
    plt.plot(elem_geom[3,elems_cath_Bline],val*cath_Bline_jepara,'b^-')
    plt.plot(elem_geom[3,cath_elem],val*cath_elem_jepara,'ro')
    
#    plt.plot(elem_geom[3,elems_cath_Bline_2],val*je_para_mean[elems_cath_Bline_2],'gv-')
#    plt.plot(elem_geom[3,elems_cath_Bline_2],val*cath_Bline_2_jepara,'r<-')
    
    error = np.abs(cath_Bline_jepara - je_para_mean[elems_cath_Bline])/np.abs(je_para_mean[elems_cath_Bline])
    error[np.where( elems_cath_Bline == cath_elem)] = 0.0
    err_max = np.nanmax(error)
    err_max_2 = np.nanmax(np.abs(cath_Bline_2_jepara - je_para_mean[elems_cath_Bline_2])/np.abs(je_para_mean[elems_cath_Bline_2]))
    print("err_max = "+str(err_max))
    print("err_max_2 = "+str(err_max_2))
    
    
    

