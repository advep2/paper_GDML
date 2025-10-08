#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:44:53 2020

@author: adrian

############################################################################
Description:    This python script reads boundary data from HET sims
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

def HET_sims_read_bound(path_simstate_inp,path_simstate_out,path_postdata_out,path_simparams_inp,
                        path_picM,allsteps_flag,timestep,oldpost_sim,oldsimparams_sim):
    
    import h5py
    import numpy as np
    
    # Parameters
    e  = 1.6021766E-19
    g0 = 9.80665
    me = 9.1093829E-31
    
#    # Open the SimState.hdf5 input/output files
#    h5_inp = h5py.File(path_simstate_inp,"r+")
#    h5_out = h5py.File(path_simstate_out,"r+")
#    # Open the PostData.hdf5 file
#    h5_post = h5py.File(path_postdata_out,"r+")
#    # Open the PIC mesh HDF5 file
#    h5_picM = h5py.File(path_picM,"r+")
    
    # Open the SimState.hdf5 input/output files
    h5_inp = h5py.File(path_simstate_inp,"r")
    h5_out = h5py.File(path_simstate_out,"r")
    # Open the PostData.hdf5 file
    h5_post = h5py.File(path_postdata_out,"r")
    # Open the PIC mesh HDF5 file
    h5_picM = h5py.File(path_picM,"r")
    
    print("HET_sims_read_bound: reading sim_params.inp...")
    # Open sim_params.inp file
    r = open(path_simparams_inp,'r')
    lines = r.readlines()
    r.close() 
    
    if oldsimparams_sim == 7:
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
        line_interp_eFld2PIC_alldata = 16
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
    elif oldsimparams_sim == 11:
        # SIMS from commit 1140de6 (after time-varying V_ps and mA, alphas and new alpha Q parameters)
        line_num_ion_spe            = 59
        line_num_neu_spe            = 60
        line_Z_ion_spe              = 61
        line_n_mp_cell_i            = 62
        line_n_mp_cell_n            = 63
        line_n_mp_cell_i_min        = 66
        line_n_mp_cell_i_max        = 67
        line_n_mp_cell_n_min        = 68
        line_n_mp_cell_n_max        = 69
        line_min_ion_plasma_density = 79
        line_m_A                    = 82
        line_ene_bal                = 13
        line_nsteps_eFld            = 33
        line_override_cathode_e     = 137
        line_new_cathode_e          = 138
        line_T_cath                 = 188
        line_ncollisions            = 200
        line_ncollisions_e          = 190
        line_ids_collisions_e       = 191
        line_coll_spe_e             = 193
        line_out_coll_spe_e         = 194
        line_interp_eFld2PIC_alldata = 16
    elif oldsimparams_sim == 12:
        # SIMS from commit 2e77087 (after PHI changes)        
        line_num_ion_spe            = 59
        line_num_neu_spe            = 60
        line_Z_ion_spe              = 61
        line_n_mp_cell_i            = 62
        line_n_mp_cell_n            = 63
        line_n_mp_cell_i_min        = 66
        line_n_mp_cell_i_max        = 67
        line_n_mp_cell_n_min        = 68
        line_n_mp_cell_n_max        = 69
        line_min_ion_plasma_density = 79
        line_m_A                    = 82
        line_ene_bal                = 13
        line_nsteps_eFld            = 33
        line_override_cathode_e     = 137
        line_new_cathode_e          = 138
        line_T_cath                 = 189
        line_cath_type              = 190
        line_ncollisions            = 202
        line_ncollisions_e          = 192
        line_ids_collisions_e       = 193
        line_coll_spe_e             = 195
        line_out_coll_spe_e         = 196
        line_interp_eFld2PIC_alldata = 16
    elif oldsimparams_sim == 13:
        # SIMS from commit 0f90f1f to commit e1de2ae (after removing it_matching and changing post)        
        line_num_ion_spe            = 61
        line_num_neu_spe            = 62
        line_Z_ion_spe              = 63
        line_n_mp_cell_i            = 64
        line_n_mp_cell_n            = 65
        line_n_mp_cell_i_min        = 68
        line_n_mp_cell_i_max        = 69
        line_n_mp_cell_n_min        = 70
        line_n_mp_cell_n_max        = 71
        line_min_ion_plasma_density = 81
        line_m_A                    = 84
        line_ene_bal                = 13
        line_nsteps_eFld            = 33
#        line_override_cathode_e     = 130
#        line_new_cathode_e          = 131
#        line_T_cath                 = 182
#        line_cath_type              = 183
#        line_ncollisions            = 195
#        line_ncollisions_e          = 185
#        line_ids_collisions_e       = 186
#        line_coll_spe_e             = 188
#        line_out_coll_spe_e         = 189
#        line_interp_eFld2PIC_alldata = 16
#        line_prnt_out_inst_vars     =  36
#        line_print_out_picMformat   =  37
        
#        line_override_cathode_e     = 134
#        line_new_cathode_e          = 135
#        line_T_cath                 = 187
#        line_cath_type              = 188
#        line_ncollisions            = 200
#        line_ncollisions_e          = 190
#        line_ids_collisions_e       = 191
#        line_coll_spe_e             = 193
#        line_out_coll_spe_e         = 194
#        line_interp_eFld2PIC_alldata = 16
#        line_prnt_out_inst_vars     =  36
#        line_print_out_picMformat   =  37
        
        line_override_cathode_e     = 134
        line_new_cathode_e          = 135
        line_T_cath                 = 189
        line_cath_type              = 190
        line_ncollisions            = 202
        line_ncollisions_e          = 192
        line_ids_collisions_e       = 193
        line_coll_spe_e             = 195
        line_out_coll_spe_e         = 196
        line_interp_eFld2PIC_alldata = 16
        line_prnt_out_inst_vars     =  36
        line_print_out_picMformat   =  37
        line_n_cond_wall            = 152
    
    elif oldsimparams_sim == 14:
        # SIMS from commit 1f4d17a (after Jiewei changes related to structure and obsolet inputs)        
        line_num_ion_spe            = 60
        line_num_neu_spe            = 61
        line_Z_ion_spe              = 62
        line_n_mp_cell_i            = 63
        line_n_mp_cell_n            = 64
        line_n_mp_cell_i_min        = 67
        line_n_mp_cell_i_max        = 68
        line_n_mp_cell_n_min        = 69
        line_n_mp_cell_n_max        = 70
        line_min_ion_plasma_density = 80
        line_m_A                    = 83
        line_ene_bal                = 13
        line_nsteps_eFld            = 33
        line_override_cathode_e     = 132
        line_new_cathode_e          = 133
        line_T_cath                 = 186
        line_cath_type              = 187
        line_ncollisions            = 199
        line_ncollisions_e          = 189
        line_ids_collisions_e       = 190
        line_coll_spe_e             = 192
        line_out_coll_spe_e         = 193
        line_interp_eFld2PIC_alldata = 16
        line_prnt_out_inst_vars     =  35
        line_print_out_picMformat   =  36
        line_n_cond_wall            = 150
    elif oldsimparams_sim == 15:
        # SIMS from commit 0438e2a (after imposing Tcat as boundary condition at the wall cathode faces)  
        # 09/12/2021: Do the changes for reading new efficiencies and P_use_z_e here, for oldsimparams_sim >= 15 and oldpost_sim = 5 
        # (value of oldpost_sim = 4 is valid from after changes in cont+mom up to the change in efficiencies and P_use_z_e is read) 
        line_num_ion_spe            = 60
        line_num_neu_spe            = 61
        line_Z_ion_spe              = 62
        line_n_mp_cell_i            = 63
        line_n_mp_cell_n            = 64
        line_n_mp_cell_i_min        = 67
        line_n_mp_cell_i_max        = 68
        line_n_mp_cell_n_min        = 69
        line_n_mp_cell_n_max        = 70
        line_min_ion_plasma_density = 80
        line_m_A                    = 83
        line_ene_bal                = 13
        line_nsteps_eFld            = 33
        line_override_cathode_e     = 132
        line_new_cathode_e          = 133
        line_T_cath                 = 186
        line_cath_type              = 187
        line_ncollisions            = 200
        line_ncollisions_e          = 190
        line_ids_collisions_e       = 191
        line_coll_spe_e             = 193
        line_out_coll_spe_e         = 194
        line_interp_eFld2PIC_alldata = 16
        line_prnt_out_inst_vars     =  35
        line_print_out_picMformat   =  36
        line_n_cond_wall            = 150
        line_ff_c_bound_type_je     = 160
    elif oldsimparams_sim == 16:
        # SIMS from commit bc4d1b4: valid with oldpost_sim = 5 or oldpost_sim = 6 (this last for commit 52174e9 including mflow/eneflow changes) 
        line_num_ion_spe            = 60
        line_num_neu_spe            = 61
        line_Z_ion_spe              = 62
        line_n_mp_cell_i            = 65
        line_n_mp_cell_n            = 66
        line_n_mp_cell_i_min        = 69
        line_n_mp_cell_i_max        = 70
        line_n_mp_cell_n_min        = 71
        line_n_mp_cell_n_max        = 72
        line_min_ion_plasma_density = 82
        line_m_A                    = 85
        line_ene_bal                = 13
        line_nsteps_eFld            = 33
        line_override_cathode_e     = 136
        line_new_cathode_e          = 137
        line_T_cath                 = 190
        line_cath_type              = 191
        line_ncollisions            = 204
        line_ncollisions_e          = 194
        line_ids_collisions_e       = 195
        line_coll_spe_e             = 197
        line_out_coll_spe_e         = 198
        line_interp_eFld2PIC_alldata = 16
        line_prnt_out_inst_vars     =  35
        line_print_out_picMformat   =  36
        line_n_cond_wall            = 154
        line_ff_c_bound_type_je     = 164
    elif oldsimparams_sim == 17:
        # SIMS from commit 695a2ac (CSL condition): valid with oldpost_sim = 6 
        line_num_ion_spe            = 60
        line_num_neu_spe            = 61
        line_Z_ion_spe              = 62
        line_n_mp_cell_i            = 65
        line_n_mp_cell_n            = 66
        line_n_mp_cell_i_min        = 69
        line_n_mp_cell_i_max        = 70
        line_n_mp_cell_n_min        = 71
        line_n_mp_cell_n_max        = 72
        line_min_ion_plasma_density = 82
        line_m_A                    = 85
        line_ene_bal                = 13
        line_nsteps_eFld            = 33
        line_override_cathode_e     = 137
        line_new_cathode_e          = 138
        line_T_cath                 = 191
        line_cath_type              = 192
        line_ncollisions            = 205
        line_ncollisions_e          = 195
        line_ids_collisions_e       = 196
        line_coll_spe_e             = 198
        line_out_coll_spe_e         = 199
        line_interp_eFld2PIC_alldata = 16
        line_prnt_out_inst_vars     =  35
        line_print_out_picMformat   =  36
        line_n_cond_wall            = 155
        line_ff_c_bound_type_je     = 165
    elif oldsimparams_sim == 18:
        # SIMS from commit baf74f6 (05/07/2022): valid with oldpost_sim = 6  (AFTER IEPC22)
        line_num_ion_spe            = 60
        line_num_neu_spe            = 61
        line_Z_ion_spe              = 62
        line_n_mp_cell_i            = 65
        line_n_mp_cell_n            = 66
        line_n_mp_cell_i_min        = 69
        line_n_mp_cell_i_max        = 70
        line_n_mp_cell_n_min        = 71
        line_n_mp_cell_n_max        = 72
        line_min_ion_plasma_density = 82
        line_m_A                    = 85
        line_ene_bal                = 13
        line_nsteps_eFld            = 33
        line_override_cathode_e     = 129
        line_new_cathode_e          = 130
        line_T_cath                 = 179
        line_cath_type              = 180
        line_ncollisions            = 193
        line_ncollisions_e          = 183
        line_ids_collisions_e       = 184
        line_coll_spe_e             = 186
        line_out_coll_spe_e         = 187
        line_interp_eFld2PIC_alldata = 16
        line_prnt_out_inst_vars     =  35
        line_print_out_picMformat   =  36
        line_n_cond_wall            = 147
        line_ff_c_bound_type_je     = 152
    elif oldsimparams_sim == 19:
        # SIMS from commit 8da48f4 (2D VDF and 3D VDF in bulk): valid with oldpost_sim = 6 
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
        line_T_cath                 = 185
        line_cath_type              = 186
        line_ncollisions            = 199
        line_ncollisions_e          = 189
        line_ids_collisions_e       = 190
        line_coll_spe_e             = 192
        line_out_coll_spe_e         = 193
        line_interp_eFld2PIC_alldata = 16
        line_prnt_out_inst_vars     =  36
        line_print_out_picMformat   =  37
        line_n_cond_wall            = 152
        line_ff_c_bound_type_je     = 157
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
    elif oldsimparams_sim == 21:
        # SIMS from commit 06e883b (latest commit after GDML paper on 27/09/2024): valid with oldpost_sim = 6 
        line_num_ion_spe            = 61
        line_num_neu_spe            = 62
        line_Z_ion_spe              = 63
        line_n_mp_cell_i            = 72
        line_n_mp_cell_n            = 73
        line_n_mp_cell_i_min        = 76
        line_n_mp_cell_i_max        = 77
        line_n_mp_cell_n_min        = 78
        line_n_mp_cell_n_max        = 79
        line_min_ion_plasma_density = 90
        line_m_A                    = 93
        line_ene_bal                = 13
        line_nsteps_eFld            = 34
        line_override_cathode_e     = 137
        line_new_cathode_e          = 138
        line_T_cath                 = 192
        line_cath_type              = 193
        line_ncollisions            = 215
        line_ncollisions_e          = 196
        line_ids_collisions_e       = 197
        line_coll_spe_e             = 199
        line_out_coll_spe_e         = 200
        line_interp_eFld2PIC_alldata = 16
        line_prnt_out_inst_vars     =  36
        line_print_out_picMformat   =  37
        line_n_cond_wall            = 155
        line_ff_c_bound_type_je     = 160
        line_B_fact                 = 170
    
    if oldsimparams_sim >= 8:
        line_interp_check           = 17
    
    # Obtain the number of ion and neutral species
    num_ion_spe = int(lines[line_num_ion_spe][lines[line_num_ion_spe].find('=')+1:lines[line_num_ion_spe].find('\n')])
    num_neu_spe = int(lines[line_num_neu_spe][lines[line_num_neu_spe].find('=')+1:lines[line_num_neu_spe].find('\n')])
    # Obtain the number of steps of the electron fluid module per PIC step
    nsteps_eFld = int(lines[line_nsteps_eFld][lines[line_nsteps_eFld].find('=')+1:lines[line_nsteps_eFld].find('\n')])
    # Obtain the interpolation check flag
    if oldsimparams_sim >=8:
        interp_check = int(lines[line_interp_check][lines[line_interp_check].find('=')+1:lines[line_interp_check].find('\n')])
    else:
        interp_check = 0
        
    if oldsimparams_sim >= 13:
        prnt_out_inst_vars   = int(lines[line_prnt_out_inst_vars][lines[line_prnt_out_inst_vars].find('=')+1:lines[line_prnt_out_inst_vars].find('\n')])
        print_out_picMformat = int(lines[line_print_out_picMformat][lines[line_print_out_picMformat].find('=')+1:lines[line_print_out_picMformat].find('\n')])    
    else:
        prnt_out_inst_vars   = 1
#        prnt_out_inst_vars   = 0  # Ucomment for DMD paper figure 
        print_out_picMformat = 0
        
    if oldsimparams_sim >=15:
        ff_c_bound_type_je = int(lines[line_ff_c_bound_type_je][lines[line_ff_c_bound_type_je].find('=')+1:lines[line_ff_c_bound_type_je].find('\n')])
    else:
        ff_c_bound_type_je = 1

    
    
    
    # Define the reshape variable function ####################################
    def reshape_var(f,name,vartype,dims1,dims2,nsteps,timestep):
        
        import numpy as np
        
        if timestep == 'all':
            var = np.zeros((dims1,dims2,nsteps),dtype=vartype)
            for i in range(0,nsteps):
                var[:,:,i] = np.reshape(f[name][i,0:],(dims1,dims2),order='F')
        else:
            var = np.zeros((dims1,dims2),dtype=vartype)
            var = np.reshape(f[name][timestep,0:],(dims1,dims2),order='F')
            
        return var
    
    print("HET_sims_read_bound: reading mesh...")
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
    surf_areas  = h5_out['picS/surf_areas'][0:,0:]
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
        
    # ---- MFAM boundary faces related vars for each type of boundary ---------
    # Obtain the number of MFAM boundary faces of each type
    nfaces_Dwall_bot  = 0 # Dielectric walls along the bottom boundary
    nfaces_Dwall_top  = 0 # Dielectric walls along the top boundary
    nfaces_Awall      = 0 # Anode walls
    nfaces_FLwall_ver = 0 # Free loss walls vertical
    nfaces_FLwall_lat = 0 # Free loss walls lateral
    nfaces_Axis       = 0 # Axis boundary 
    for i in range(0,n_faces_boundary):
        bface_id = boundary_f[i]
        if faces[2,bface_id] == 15:
            if (np.abs(zs[0,-1] - face_geom[0,bface_id])/face_geom[0,bface_id] <= 1E-6):
                nfaces_FLwall_ver = nfaces_FLwall_ver + 1
            else:
                nfaces_FLwall_lat = nfaces_FLwall_lat + 1
        elif faces[2,bface_id] == 11 or faces[2,bface_id] == 13 or faces[2,bface_id] == 17 or faces[2,bface_id] == 16:
#            if (rs[int(dims[0]/2.0),0] > face_geom[1,bface_id]):
            if (0.5*(rs[int(eta_min),0]+rs[int(eta_max),0]) > face_geom[1,bface_id]):
                nfaces_Dwall_bot  = nfaces_Dwall_bot + 1
#            elif (rs[int(dims[0]/2.0),0] < face_geom[1,bface_id]):
            elif (0.5*(rs[int(eta_min),0]+rs[int(eta_max),0]) < face_geom[1,bface_id]):
                nfaces_Dwall_top  = nfaces_Dwall_top + 1
        elif faces[2,bface_id] == 12 or faces[2,bface_id] == 18:
            nfaces_Awall  = nfaces_Awall + 1   
        elif faces[2,bface_id] == 14:
            nfaces_Axis  = nfaces_Axis + 1   
    # Obtain the ID in boundary_f (Python standard) of the faces of each type 
    # and the ID in faces and face_geom of the MFAM boundary faces of each type
    # IDs of faces of each type in the boundary, ordered as they appear in boundary_f
    bIDfaces_Dwall_bot  = np.zeros(nfaces_Dwall_bot,dtype=int)
    bIDfaces_Dwall_top  = np.zeros(nfaces_Dwall_top,dtype=int)
    bIDfaces_Awall      = np.zeros(nfaces_Awall,dtype=int)
    bIDfaces_FLwall_ver = np.zeros(nfaces_FLwall_ver,dtype=int) 
    bIDfaces_FLwall_lat = np.zeros(nfaces_FLwall_lat,dtype=int) 
    bIDfaces_Axis       = np.zeros(nfaces_Axis,dtype=int)
    # IDs of faces of each type in the boundary 
    # (their original IDs in faces and face_geom, i.e. their position in faces
    # and face_geom)
    IDfaces_Dwall_bot  = np.zeros(nfaces_Dwall_bot,dtype=int)
    IDfaces_Dwall_top  = np.zeros(nfaces_Dwall_top,dtype=int)
    IDfaces_Awall      = np.zeros(nfaces_Awall,dtype=int)
    IDfaces_FLwall_ver = np.zeros(nfaces_FLwall_ver,dtype=int)
    IDfaces_FLwall_lat = np.zeros(nfaces_FLwall_lat,dtype=int)
    IDfaces_Axis       = np.zeros(nfaces_Axis,dtype=int)
    indD_bot  = 0
    indD_top  = 0
    indA      = 0
    indFL_ver = 0
    indFL_lat = 0
    indAx     = 0
    for i in range(0,n_faces_boundary):
        bface_id = boundary_f[i]
        if faces[2,bface_id] == 15:
            if (np.abs(zs[0,-1] - face_geom[0,bface_id])/face_geom[0,bface_id] <= 1E-6):
                bIDfaces_FLwall_ver[indFL_ver] = i
                IDfaces_FLwall_ver[indFL_ver]  = bface_id
                indFL_ver = indFL_ver + 1
            else:
                bIDfaces_FLwall_lat[indFL_lat] = i
                IDfaces_FLwall_lat[indFL_lat]  = bface_id
                indFL_lat = indFL_lat + 1
        elif faces[2,bface_id] == 11 or faces[2,bface_id] == 13 or faces[2,bface_id] == 17 or faces[2,bface_id] == 16: 
            # NOTE (04/05/2023): FOR NOW CATHODE FACES ARE INCLUDED HERE: THIS SHOULD BE CHANGED IN FUTURE
#            if (rs[int(dims[0]/2.0),0] > face_geom[1,bface_id]):
            if (0.5*(rs[int(eta_min),0]+rs[int(eta_max),0]) > face_geom[1,bface_id]):
                bIDfaces_Dwall_bot[indD_bot] = i            
                IDfaces_Dwall_bot[indD_bot] = bface_id
                indD_bot = indD_bot + 1
#            elif (rs[int(dims[0]/2.0),0] < face_geom[1,bface_id]):
            elif (0.5*(rs[int(eta_min),0]+rs[int(eta_max),0]) < face_geom[1,bface_id]):
                bIDfaces_Dwall_top[indD_top] = i            
                IDfaces_Dwall_top[indD_top] = bface_id
                indD_top = indD_top + 1
        elif faces[2,bface_id] == 12 or faces[2,bface_id] == 18:
            bIDfaces_Awall[indA] = i            
            IDfaces_Awall[indA] = bface_id
            indA = indA + 1  
        elif faces[2,bface_id] == 14:
            bIDfaces_Axis[indAx] = i            
            IDfaces_Axis[indAx] = bface_id
            indAx = indAx + 1    
    # Obtain the z,r, coordinates of the face center and the area for the MFAM boundary faces of each type
    zfaces_Dwall_bot  = face_geom[0,IDfaces_Dwall_bot]
    rfaces_Dwall_bot  = face_geom[1,IDfaces_Dwall_bot]
    Afaces_Dwall_bot  = face_geom[4,IDfaces_Dwall_bot]
    zfaces_Dwall_top  = face_geom[0,IDfaces_Dwall_top]
    rfaces_Dwall_top  = face_geom[1,IDfaces_Dwall_top]
    Afaces_Dwall_top  = face_geom[4,IDfaces_Dwall_top]
    zfaces_Awall      = face_geom[0,IDfaces_Awall]
    rfaces_Awall      = face_geom[1,IDfaces_Awall]
    Afaces_Awall      = face_geom[4,IDfaces_Awall]
    zfaces_FLwall_ver = face_geom[0,IDfaces_FLwall_ver]
    rfaces_FLwall_ver = face_geom[1,IDfaces_FLwall_ver]
    Afaces_FLwall_ver = face_geom[4,IDfaces_FLwall_ver]
    zfaces_FLwall_lat = face_geom[0,IDfaces_FLwall_lat]
    rfaces_FLwall_lat = face_geom[1,IDfaces_FLwall_lat]
    Afaces_FLwall_lat = face_geom[4,IDfaces_FLwall_lat]
    zfaces_Axis       = face_geom[0,IDfaces_Axis]
    rfaces_Axis       = face_geom[1,IDfaces_Axis]
    Afaces_Axis       = face_geom[4,IDfaces_Axis]
    
    # Sort the boundary faces of each type so that they form a continous line
    d_Dwall_bot  = np.zeros(nfaces_Dwall_bot,dtype=float)
    d_Dwall_top  = np.zeros(nfaces_Dwall_top,dtype=float)
    d_Awall      = np.zeros(nfaces_Awall,dtype=float)
    d_FLwall_ver = np.zeros(nfaces_FLwall_ver,dtype=float)
    d_FLwall_lat = np.zeros(nfaces_FLwall_lat,dtype=float)
    d_Axis       = np.zeros(nfaces_Axis,dtype=float)
    zp = zs[int(eta_min),0]
    rp = rs[int(eta_min),0]
    for i in range(0,nfaces_Dwall_bot):
        d_Dwall_bot[i] = np.sqrt((zfaces_Dwall_bot[i]-zp)**2 + (rfaces_Dwall_bot[i]-rp)**2)
    zp = zs[int(eta_max),0]
    rp = rs[int(eta_max),0]
    for i in range(0,nfaces_Dwall_top):
        d_Dwall_top[i] = np.sqrt((zfaces_Dwall_top[i]-zp)**2 + (rfaces_Dwall_top[i]-rp)**2)
    zp = zs[int(eta_min),0]
    rp = rs[int(eta_min),0]
    for i in range(0,nfaces_Awall):
        d_Awall[i] = np.sqrt((zfaces_Awall[i]-zp)**2 + (rfaces_Awall[i]-rp)**2)
    zp = zs[0,-1]
    rp = rs[0,-1]
    for i in range(0,nfaces_FLwall_ver):
        d_FLwall_ver[i] = np.sqrt((zfaces_FLwall_ver[i]-zp)**2 + (rfaces_FLwall_ver[i]-rp)**2)    
    zp = zs[-1,int(xi_bottom)]
    rp = rs[-1,int(xi_bottom)]
    for i in range(0,nfaces_FLwall_lat):
        d_FLwall_lat[i] = np.sqrt((zfaces_FLwall_lat[i]-zp)**2 + (rfaces_FLwall_lat[i]-rp)**2) 
    zp = zs[0,int(xi_bottom)]
    rp = rs[0,int(xi_bottom)]
    for i in range(0,nfaces_Axis):
        d_Axis[i] = np.sqrt((zfaces_Axis[i]-zp)**2 + (rfaces_Axis[i]-rp)**2)
    
    # Obtain the indeces for sorting (permutation vectors)
    indsort_Dwall_bot  = np.argsort(d_Dwall_bot)
    indsort_Dwall_top  = np.argsort(d_Dwall_top)
    indsort_Awall      = np.argsort(d_Awall)
    indsort_FLwall_ver = np.argsort(d_FLwall_ver)
    indsort_FLwall_lat = np.argsort(d_FLwall_lat)
    indsort_Axis       = np.argsort(d_Axis)

    # Sort the variables related to each type of boundary faces
    bIDfaces_Dwall_bot = bIDfaces_Dwall_bot[indsort_Dwall_bot]
    IDfaces_Dwall_bot  = IDfaces_Dwall_bot[indsort_Dwall_bot]
    zfaces_Dwall_bot   = zfaces_Dwall_bot[indsort_Dwall_bot]
    rfaces_Dwall_bot   = rfaces_Dwall_bot[indsort_Dwall_bot]
    Afaces_Dwall_bot   = Afaces_Dwall_bot[indsort_Dwall_bot]
    d_Dwall_bot        = d_Dwall_bot[indsort_Dwall_bot]
    
    bIDfaces_Dwall_top = bIDfaces_Dwall_top[indsort_Dwall_top]
    IDfaces_Dwall_top  = IDfaces_Dwall_top[indsort_Dwall_top]
    zfaces_Dwall_top   = zfaces_Dwall_top[indsort_Dwall_top]
    rfaces_Dwall_top   = rfaces_Dwall_top[indsort_Dwall_top]
    Afaces_Dwall_top   = Afaces_Dwall_top[indsort_Dwall_top]
    d_Dwall_top        = d_Dwall_top[indsort_Dwall_top]
    
    bIDfaces_Awall     = bIDfaces_Awall[indsort_Awall]
    IDfaces_Awall      = IDfaces_Awall[indsort_Awall]
    zfaces_Awall       = zfaces_Awall[indsort_Awall]
    rfaces_Awall       = rfaces_Awall[indsort_Awall]
    Afaces_Awall       = Afaces_Awall[indsort_Awall]
    d_Awall            = d_Awall[indsort_Awall]
    
    bIDfaces_FLwall_ver = bIDfaces_FLwall_ver[indsort_FLwall_ver]
    IDfaces_FLwall_ver  = IDfaces_FLwall_ver[indsort_FLwall_ver]
    zfaces_FLwall_ver   = zfaces_FLwall_ver[indsort_FLwall_ver]
    rfaces_FLwall_ver   = rfaces_FLwall_ver[indsort_FLwall_ver]
    Afaces_FLwall_ver   = Afaces_FLwall_ver[indsort_FLwall_ver]
    d_FLwall_ver        = d_FLwall_ver[indsort_FLwall_ver]
    
    bIDfaces_FLwall_lat = bIDfaces_FLwall_lat[indsort_FLwall_lat]
    IDfaces_FLwall_lat  = IDfaces_FLwall_lat[indsort_FLwall_lat]
    zfaces_FLwall_lat   = zfaces_FLwall_lat[indsort_FLwall_lat]
    rfaces_FLwall_lat   = rfaces_FLwall_lat[indsort_FLwall_lat]
    Afaces_FLwall_lat   = Afaces_FLwall_lat[indsort_FLwall_lat]
    d_FLwall_lat        = d_FLwall_lat[indsort_FLwall_lat]
    
    bIDfaces_Axis     = bIDfaces_Axis[indsort_Axis]
    IDfaces_Axis      = IDfaces_Axis[indsort_Axis]
    zfaces_Axis       = zfaces_Axis[indsort_Axis]
    rfaces_Axis       = rfaces_Axis[indsort_Axis]
    Afaces_Axis       = Afaces_Axis[indsort_Axis]
    d_Axis            = d_Axis[indsort_Axis]
    
    # Obtain the arc length along each type of boundary surface
    sDwall_bot  = np.zeros(nfaces_Dwall_bot)
    sDwall_top  = np.zeros(nfaces_Dwall_top)
    sAwall      = np.zeros(nfaces_Awall)
    sFLwall_ver = np.zeros(nfaces_FLwall_ver)
    sFLwall_lat = np.zeros(nfaces_FLwall_lat)
    sAxis       = np.zeros(nfaces_Axis)
    for i in range(0,nfaces_Dwall_bot-1):
        z1 = zfaces_Dwall_bot[i]
        r1 = rfaces_Dwall_bot[i]
        z2 = zfaces_Dwall_bot[i+1]
        r2 = rfaces_Dwall_bot[i+1]
        sDwall_bot[i+1] = sDwall_bot[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    sDwall_bot = sDwall_bot + d_Dwall_bot[0]
    for i in range(0,nfaces_Dwall_top-1):
        z1 = zfaces_Dwall_top[i]
        r1 = rfaces_Dwall_top[i]
        z2 = zfaces_Dwall_top[i+1]
        r2 = rfaces_Dwall_top[i+1]
        sDwall_top[i+1] = sDwall_top[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    sDwall_top = sDwall_top + d_Dwall_top[0]
    for i in range(0,nfaces_Awall-1):
        z1 = zfaces_Awall[i]
        r1 = rfaces_Awall[i]
        z2 = zfaces_Awall[i+1]
        r2 = rfaces_Awall[i+1]
        sAwall[i+1] = sAwall[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    sAwall = sAwall + d_Awall[0]
    for i in range(0,nfaces_FLwall_ver-1):
        z1 = zfaces_FLwall_ver[i]
        r1 = rfaces_FLwall_ver[i]
        z2 = zfaces_FLwall_ver[i+1]
        r2 = rfaces_FLwall_ver[i+1]
        sFLwall_ver[i+1] = sFLwall_ver[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    sFLwall_ver = sFLwall_ver + d_FLwall_ver[0]
    for i in range(0,nfaces_FLwall_lat-1):
        z1 = zfaces_FLwall_lat[i]
        r1 = rfaces_FLwall_lat[i]
        z2 = zfaces_FLwall_lat[i+1]
        r2 = rfaces_FLwall_lat[i+1]
        sFLwall_lat[i+1] = sFLwall_lat[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    sFLwall_lat = sFLwall_lat + d_FLwall_lat[0]
    for i in range(0,nfaces_Axis-1):
        z1 = zfaces_Axis[i]
        r1 = rfaces_Axis[i]
        z2 = zfaces_Axis[i+1]
        r2 = rfaces_Axis[i+1]
        sAxis[i+1] = sAxis[i] + np.sqrt((z2-z1)**2+(r2-r1)**2)
    sAxis = sAxis + d_Axis[0]
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
    # Substract the length of the plume vertical bottom and top parts
#    sc_bot_nodes = sDwall_bot_nodes[-1] - np.sqrt((zs[int(eta_min),int(xi_bottom)] - zs[0,int(xi_bottom)])**2 + (rs[int(eta_min),int(xi_bottom)] - rs[0,int(xi_bottom)])**2)
#    sc_top_nodes = sDwall_top_nodes[-1] - np.sqrt((zs[-1,int(xi_top)] - zs[int(eta_max),int(xi_top)])**2 + (rs[-1,int(xi_top)] - rs[int(eta_max),int(xi_top)])**2)
    
    sc_bot_nodes = sDwall_bot_nodes[-1] - Lplume_bot
    sc_top_nodes = sDwall_top_nodes[-1] - Lplume_top
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
        
    # Obtain the arc length value at the end of the chamber for top and bottom boundaries (use the PIC mesh for simplicity)
    # Substract the length of the plume vertical bottom and top parts
#    sc_bot_surf = sDwall_bot_surf[-1] - np.sqrt((zs[int(eta_min),int(xi_bottom)] - zs[0,int(xi_bottom)])**2 + (rs[int(eta_min),int(xi_bottom)] - rs[0,int(xi_bottom)])**2)
#    sc_top_surf = sDwall_top_surf[-1] - np.sqrt((zs[-1,int(xi_top)] - zs[int(eta_max),int(xi_top)])**2 + (rs[-1,int(xi_top)] - rs[int(eta_max),int(xi_top)])**2)
    
#    sc_bot_surf = sDwall_bot_surf[-1] - Lplume_bot
#    sc_top_surf = sDwall_top_surf[-1] - Lplume_top
    
    sc_bot_surf = Lchamb_bot
    sc_top_surf = Lchamb_top
    # -------------------------------------------------------------------------

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
    
    # Obtain the data at the boundary faces, the PIC mesh nodes and the PIC mesh important surface elements
    # Decide if read all timesteps or a particular timestep
    if allsteps_flag == 1:
        print("HET_sims_read_bound: reading all steps data...")
        # Obtain data at the boundary faces
        # Bfield data
        Bfield = face_geom[-1,:]
        # Accumulated data ----------------------------------------------------
        dphi_sh_b          = h5_post['/eFldM_data/boundary/dphi_sh_b_acc'][:,:]
        je_b               = h5_post['/eFldM_data/boundary/je_b_acc'][:,:]
        ji_tot_b           = h5_post['/eFldM_data/boundary/ji_tot_b_acc'][:,:]
        ge_sb_b            = h5_post['/eFldM_data/boundary/ge_sb_b_acc'][:,:]
        relerr_je_b        = h5_post['/eFldM_data/boundary/normal_flux_rel_err_b_acc'][:,:]
        qe_tot_wall        = h5_post['/eFldM_data/boundary/qe_tot_wall_acc'][:,:]
        qe_tot_b           = h5_post['/eFldM_data/boundary/qe_tot_b_acc'][:,:]
        qe_b               = h5_post['/eFldM_data/boundary/qe_b_acc'][:,:]
        if oldpost_sim <= 3:
            qe_b_bc            = h5_post['/eFldM_data/boundary/qe_b_bc_acc'][:,:]
            qe_b_fl            = h5_post['/eFldM_data/boundary/qe_b_fl_acc'][:,:]
            relerr_qe_b        = h5_post['/eFldM_data/boundary/normal_heat_flux_rel_err_b_acc'][:,:]
            relerr_qe_b_cons   = h5_post['/eFldM_data/boundary/normal_cons_heat_flux_rel_err_b_acc'][:,:]
        elif oldpost_sim > 3:
            qe_b_bc            = h5_post['/eFldM_data/boundary/qe_tot_b_bc_acc'][:,:]
            qe_b_fl            = h5_post['/eFldM_data/boundary/qe_tot_b_fl_acc'][:,:]
            relerr_qe_b        = h5_post['/eFldM_data/boundary/normal_energy_flux_rel_err_b_acc'][:,:]
            relerr_qe_b_cons   = h5_post['/eFldM_data/boundary/normal_cons_energy_flux_rel_err_b_acc'][:,:]
        Te                 = h5_post['/eFldM_data/faces/Te_acc'][:,:]
        phi                = h5_post['/eFldM_data/faces/phi_acc'][:,:]
        n_inst             = h5_post['/eFldM_data/faces/n'][:,:]
        if oldpost_sim <= 3:
            ni1_inst           = h5_post['/eFldM_data/faces/ni'][:,0,:]
            ni2_inst           = np.zeros(np.shape(ni1_inst),dtype=float)
            if num_ion_spe == 2:
                ni2_inst       = h5_post['/eFldM_data/faces/ni'][:,1,:]
            nn1_inst           = h5_post['/eFldM_data/faces/nn'][:,0,:]
        elif oldpost_sim > 3:
            ni1_inst           = np.zeros(np.shape(n_inst),dtype=float)
            ni2_inst           = np.zeros(np.shape(n_inst),dtype=float)
            nn1_inst           = np.zeros(np.shape(n_inst),dtype=float)
            
        if oldpost_sim > 3:
            delta_r            = h5_post['/eFldM_data/boundary/delta_r_acc'][:,:]
            delta_s            = h5_post['/eFldM_data/boundary/delta_s_acc'][:,:]
            gp_net_b           = h5_post['/eFldM_data/boundary/gp_net_b_acc'][:,:]
            qe_tot_s_wall      = h5_post['/eFldM_data/boundary/qe_tot_s_wall_acc'][:,:]
            delta_s_csl        = np.zeros(np.shape(je_b),dtype=float)
            if oldpost_sim >=6 and oldsimparams_sim >= 17:
                delta_s_csl    = h5_post['/eFldM_data/boundary/delta_s_csl_acc'][:,:]
            if interp_check == 1:
                err_interp_phi     = h5_post['/eFldM_data/faces/err_interp_acc'][:,0,:]
                err_interp_Te      = h5_post['/eFldM_data/faces/err_interp_acc'][:,1,:]
                err_interp_jeperp  = h5_post['/eFldM_data/faces/err_interp_acc'][:,2,:]
                err_interp_jetheta = h5_post['/eFldM_data/faces/err_interp_acc'][:,3,:]
                err_interp_jepara  = h5_post['/eFldM_data/faces/err_interp_acc'][:,4,:]
                err_interp_jez     = h5_post['/eFldM_data/faces/err_interp_acc'][:,5,:]
                err_interp_jer     = h5_post['/eFldM_data/faces/err_interp_acc'][:,6,:]
            else:
                err_interp_phi     = np.zeros(np.shape(je_b),dtype=float)
                err_interp_Te      = np.zeros(np.shape(je_b),dtype=float)
                err_interp_jeperp  = np.zeros(np.shape(je_b),dtype=float)
                err_interp_jetheta = np.zeros(np.shape(je_b),dtype=float)
                err_interp_jepara  = np.zeros(np.shape(je_b),dtype=float)
                err_interp_jez     = np.zeros(np.shape(je_b),dtype=float)
                err_interp_jer     = np.zeros(np.shape(je_b),dtype=float)
        else:
            # Provisional for reading sims from version 71d0dcb
            delta_r            = np.zeros(np.shape(je_b),dtype=float)
            delta_s            = np.zeros(np.shape(je_b),dtype=float)
            delta_s_csl        = np.zeros(np.shape(je_b),dtype=float)
            gp_net_b           = np.zeros(np.shape(je_b),dtype=float)
            err_interp_phi     = np.zeros(np.shape(je_b),dtype=float)
            err_interp_Te      = np.zeros(np.shape(je_b),dtype=float)
            err_interp_jeperp  = np.zeros(np.shape(je_b),dtype=float)
            err_interp_jetheta = np.zeros(np.shape(je_b),dtype=float)
            err_interp_jepara  = np.zeros(np.shape(je_b),dtype=float)
            err_interp_jez     = np.zeros(np.shape(je_b),dtype=float)
            err_interp_jer     = np.zeros(np.shape(je_b),dtype=float)
            qe_tot_s_wall      = np.zeros(np.shape(je_b),dtype=float)
        
        if oldsimparams_sim >= 21: 
            inst_dphi_sh_b_Te       = h5_post['/eFldM_data/boundary/dphi_sh_b_Te_acc'][:,:]   
            inst_imp_ene_e_b        = h5_post['/eFldM_data/boundary/imp_ene_e_b_acc'][:,:]   
            inst_imp_ene_e_b_Te     = h5_post['/eFldM_data/boundary/imp_ene_e_b_Te_acc'][:,:]   
            inst_imp_ene_e_wall     = h5_post['/eFldM_data/boundary/imp_ene_e_wall_acc'][:,:]   
            inst_imp_ene_e_wall_Te  = h5_post['/eFldM_data/boundary/imp_ene_e_wall_Te_acc'][:,:]   
            
        else:   
            inst_dphi_sh_b_Te       = np.zeros(np.shape(je_b),dtype=float) 
            inst_imp_ene_e_b        = np.zeros(np.shape(je_b),dtype=float) 
            inst_imp_ene_e_b_Te     = np.zeros(np.shape(je_b),dtype=float) 
            inst_imp_ene_e_wall     = np.zeros(np.shape(je_b),dtype=float) 
            inst_imp_ene_e_wall_Te  = np.zeros(np.shape(je_b),dtype=float)
            

        # Obtain data at the boundary PIC mesh nodes
        # Bfield data
        Br_nodes = h5_out['/picM/Br'][0:,0:]
        Bz_nodes = h5_out['/picM/Bz'][0:,0:]
        Bfield_nodes = np.sqrt(Br_nodes**2.0+Bz_nodes**2.0)
        # Data from the eFld module
        if print_out_picMformat == 1:
            dphi_sh_b_nodes     = reshape_var(h5_post,"/picM_data/dphi_sh_acc","float",dims[0],dims[1],nsteps,"all")
            je_b_nodes          = reshape_var(h5_post,"/picM_data/je_b_acc","float",dims[0],dims[1],nsteps,"all")
            ge_sb_b_nodes       = reshape_var(h5_post,"/picM_data/ge_sb_b_acc","float",dims[0],dims[1],nsteps,"all")
            relerr_je_b_nodes   = reshape_var(h5_post,"/picM_data/normal_flux_rel_err_b_acc","float",dims[0],dims[1],nsteps,"all")
            qe_tot_wall_nodes   = reshape_var(h5_post,"/picM_data/qe_tot_wall_acc","float",dims[0],dims[1],nsteps,"all")
            qe_tot_b_nodes      = reshape_var(h5_post,"/picM_data/qe_tot_b_acc","float",dims[0],dims[1],nsteps,"all")
            qe_b_nodes          = reshape_var(h5_post,"/picM_data/qe_b_acc","float",dims[0],dims[1],nsteps,"all")
            
            if oldpost_sim <= 3:
                qe_b_bc_nodes          = reshape_var(h5_post,"/picM_data/qe_b_bc_acc","float",dims[0],dims[1],nsteps,"all")
                qe_b_fl_nodes          = reshape_var(h5_post,"/picM_data/qe_b_fl_acc","float",dims[0],dims[1],nsteps,"all")
                relerr_qe_b_nodes      = reshape_var(h5_post,"/picM_data/normal_heat_flux_rel_err_b_acc","float",dims[0],dims[1],nsteps,"all")
                relerr_qe_b_cons_nodes = reshape_var(h5_post,"/picM_data/normal_cons_heat_flux_rel_err_b_acc","float",dims[0],dims[1],nsteps,"all")
            elif oldpost_sim > 3:
                qe_b_bc_nodes          = reshape_var(h5_post,"/picM_data/qe_tot_b_bc_acc","float",dims[0],dims[1],nsteps,"all")
                qe_b_fl_nodes          = reshape_var(h5_post,"/picM_data/qe_tot_b_fl_acc","float",dims[0],dims[1],nsteps,"all")
                relerr_qe_b_nodes      = reshape_var(h5_post,"/picM_data/normal_energy_flux_rel_err_b_acc","float",dims[0],dims[1],nsteps,"all")
                relerr_qe_b_cons_nodes = reshape_var(h5_post,"/picM_data/normal_cons_energy_flux_rel_err_b_acc","float",dims[0],dims[1],nsteps,"all")

            Te_nodes               = reshape_var(h5_post,"/picM_data/Te_acc","float",dims[0],dims[1],nsteps,"all")
            phi_nodes              = reshape_var(h5_post,"/picM_data/phi_acc","float",dims[0],dims[1],nsteps,"all")
            if oldpost_sim > 3:
                delta_r_nodes       = reshape_var(h5_post,"/picM_data/delta_r_acc","float",dims[0],dims[1],nsteps,"all")
                delta_s_nodes       = reshape_var(h5_post,"/picM_data/delta_s_acc","float",dims[0],dims[1],nsteps,"all")
                delta_s_csl_nodes   = np.zeros(np.shape(je_b_nodes),dtype=float)
                if oldpost_sim >=6 and oldsimparams_sim >= 17:
                    delta_s_csl_nodes = reshape_var(h5_post,"/picM_data/delta_s_csl_acc","float",dims[0],dims[1],nsteps,"all")
                gp_net_b_nodes      = reshape_var(h5_post,"/picM_data/gp_net_b_acc","float",dims[0],dims[1],nsteps,"all")
                qe_tot_s_wall_nodes = reshape_var(h5_post,"/picM_data/qe_tot_s_wall_acc","float",dims[0],dims[1],nsteps,"all")
                if interp_check == 1:
                    err_interp_n_nodes     = reshape_var(h5_post,"/picM_data/err_interp_n_acc","float",dims[0],dims[1],nsteps,"all")
                else:
                    err_interp_n_nodes  = np.zeros(np.shape(je_b_nodes),dtype=float)
            else:
                delta_r_nodes       = np.zeros(np.shape(je_b_nodes),dtype=float)
                delta_s_nodes       = np.zeros(np.shape(je_b_nodes),dtype=float)
                delta_s_csl_nodes   = np.zeros(np.shape(je_b_nodes),dtype=float)
                gp_net_b_nodes      = np.zeros(np.shape(je_b_nodes),dtype=float)
                err_interp_n_nodes  = np.zeros(np.shape(je_b_nodes),dtype=float)
                qe_tot_s_wall_nodes = np.zeros(np.shape(je_b_nodes),dtype=float) 
                
            if oldsimparams_sim >= 21:
                inst_dphi_sh_b_Te_nodes       = reshape_var(h5_post,"/picM_data/dphi_sh_b_Te_acc","float",dims[0],dims[1],nsteps,"all")  
                inst_imp_ene_e_b_nodes        = reshape_var(h5_post,"/picM_data/imp_ene_e_b_acc","float",dims[0],dims[1],nsteps,"all")     
                inst_imp_ene_e_b_Te_nodes     = reshape_var(h5_post,"/picM_data/imp_ene_e_b_Te_acc","float",dims[0],dims[1],nsteps,"all")    
                inst_imp_ene_e_wall_nodes     = reshape_var(h5_post,"/picM_data/imp_ene_e_wall_acc","float",dims[0],dims[1],nsteps,"all")     
                inst_imp_ene_e_wall_Te_nodes  = reshape_var(h5_post,"/picM_data/imp_ene_e_wall_Te_acc","float",dims[0],dims[1],nsteps,"all")    
                
            else:  
                inst_dphi_sh_b_Te_nodes       = np.zeros(np.shape(je_b_nodes),dtype=float) 
                inst_imp_ene_e_b_nodes        = np.zeros(np.shape(je_b_nodes),dtype=float) 
                inst_imp_ene_e_b_Te_nodes     = np.zeros(np.shape(je_b_nodes),dtype=float) 
                inst_imp_ene_e_wall_nodes     = np.zeros(np.shape(je_b_nodes),dtype=float) 
                inst_imp_ene_e_wall_Te_nodes  = np.zeros(np.shape(je_b_nodes),dtype=float) 
    
            # Data from PIC module            
            if prnt_out_inst_vars == 1:
                n_inst_nodes           = reshape_var(h5_post,"/picM_data/n","float",dims[0],dims[1],nsteps,"all")
                ni1_inst_nodes         = reshape_var(h5_post,"/picM_data/ni1","float",dims[0],dims[1],nsteps,"all")
                ni2_inst_nodes         = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
                if num_ion_spe == 2:
                    ni2_inst_nodes     = reshape_var(h5_post,"/picM_data/ni2","float",dims[0],dims[1],nsteps,"all")
                nn1_inst_nodes         = reshape_var(h5_post,"/picM_data/nn1","float",dims[0],dims[1],nsteps,"all")
            else:
                n_inst_nodes    = np.zeros(np.shape(je_b_nodes),dtype=float)
                ni1_inst_nodes  = np.zeros(np.shape(je_b_nodes),dtype=float)
                ni2_inst_nodes  = np.zeros(np.shape(je_b_nodes),dtype=float)
                nn1_inst_nodes  = np.zeros(np.shape(je_b_nodes),dtype=float)

                
            n_nodes                = reshape_var(h5_post,"/picM_data/n_acc","float",dims[0],dims[1],nsteps,"all")
            ni1_nodes              = reshape_var(h5_post,"/picM_data/ni_acc1","float",dims[0],dims[1],nsteps,"all")
            ni2_nodes              = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            if num_ion_spe == 2:
                ni2_nodes          = reshape_var(h5_post,"/picM_data/ni_acc2","float",dims[0],dims[1],nsteps,"all")
            nn1_nodes              = reshape_var(h5_post,"/picM_data/nn_acc1","float",dims[0],dims[1],nsteps,"all")
            
            dphi_kbc_nodes         = reshape_var(h5_post,"/picM_data/dphi_kbc_acc","float",dims[0],dims[1],nsteps,"all")
            MkQ1_nodes             = reshape_var(h5_post,"/picM_data/MkQ1_acc","float",dims[0],dims[1],nsteps,"all")
            ji1_nodes              = reshape_var(h5_post,"/picM_data/ion_flux_in_acc1","float",dims[0],dims[1],nsteps,"all")*e
            ji2_nodes              = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            ji3_nodes              = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            ji4_nodes              = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            if num_ion_spe == 2:
                ji2_nodes          = reshape_var(h5_post,"/picM_data/ion_flux_in_acc2","float",dims[0],dims[1],nsteps,"all")*2*e
            elif num_ion_spe == 4:
                ji2_nodes          = reshape_var(h5_post,"/picM_data/ion_flux_in_acc2","float",dims[0],dims[1],nsteps,"all")*2*e
                ji3_nodes          = reshape_var(h5_post,"/picM_data/ion_flux_in_acc3","float",dims[0],dims[1],nsteps,"all")*e
                ji4_nodes          = reshape_var(h5_post,"/picM_data/ion_flux_in_acc4","float",dims[0],dims[1],nsteps,"all")*2*e
            ji_nodes               = ji1_nodes + ji2_nodes + ji3_nodes + ji4_nodes
            
            gn1_tw_nodes           = reshape_var(h5_post,"/picM_data/neu_flux_in_acc1","float",dims[0],dims[1],nsteps,"all")
            gn1_fw_nodes           = reshape_var(h5_post,"/picM_data/neu_flux_out_acc1","float",dims[0],dims[1],nsteps,"all")
            gn2_tw_nodes           = np.zeros(np.shape(gn1_tw_nodes),dtype=float)
            gn2_fw_nodes           = np.zeros(np.shape(gn1_tw_nodes),dtype=float)
            gn3_tw_nodes           = np.zeros(np.shape(gn1_tw_nodes),dtype=float)
            gn3_fw_nodes           = np.zeros(np.shape(gn1_tw_nodes),dtype=float)
            if num_neu_spe == 3:
                gn2_tw_nodes           = reshape_var(h5_post,"/picM_data/neu_flux_in_acc2","float",dims[0],dims[1],nsteps,"all")
                gn2_fw_nodes           = reshape_var(h5_post,"/picM_data/neu_flux_out_acc2","float",dims[0],dims[1],nsteps,"all")
                gn3_tw_nodes           = reshape_var(h5_post,"/picM_data/neu_flux_in_acc3","float",dims[0],dims[1],nsteps,"all")
                gn3_fw_nodes           = reshape_var(h5_post,"/picM_data/neu_flux_out_acc3","float",dims[0],dims[1],nsteps,"all")
            gn_tw_nodes            = gn1_tw_nodes + gn2_tw_nodes + gn3_tw_nodes
            
            qi1_tot_wall_nodes     = reshape_var(h5_post,"/picM_data/ion_ene_flux_in_acc1","float",dims[0],dims[1],nsteps,"all")
            qi2_tot_wall_nodes     = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            qi3_tot_wall_nodes     = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            qi4_tot_wall_nodes     = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            if num_ion_spe == 2:
                qi2_tot_wall_nodes = reshape_var(h5_post,"/picM_data/ion_ene_flux_in_acc2","float",dims[0],dims[1],nsteps,"all")
            elif num_ion_spe == 4:
                qi2_tot_wall_nodes = reshape_var(h5_post,"/picM_data/ion_ene_flux_in_acc2","float",dims[0],dims[1],nsteps,"all")
                qi3_tot_wall_nodes = reshape_var(h5_post,"/picM_data/ion_ene_flux_in_acc3","float",dims[0],dims[1],nsteps,"all")
                qi4_tot_wall_nodes = reshape_var(h5_post,"/picM_data/ion_ene_flux_in_acc4","float",dims[0],dims[1],nsteps,"all")
            qi_tot_wall_nodes      = qi1_tot_wall_nodes + qi2_tot_wall_nodes + qi3_tot_wall_nodes + qi4_tot_wall_nodes
            
            qn1_tw_nodes           = reshape_var(h5_post,"/picM_data/neu_ene_flux_in_acc1","float",dims[0],dims[1],nsteps,"all")
            qn1_fw_nodes           = reshape_var(h5_post,"/picM_data/neu_ene_flux_out_acc1","float",dims[0],dims[1],nsteps,"all")
            qn2_tw_nodes           = np.zeros(np.shape(qn1_tw_nodes),dtype=float)
            qn2_fw_nodes           = np.zeros(np.shape(qn1_tw_nodes),dtype=float)
            qn3_tw_nodes           = np.zeros(np.shape(qn1_tw_nodes),dtype=float)
            qn3_fw_nodes           = np.zeros(np.shape(qn1_tw_nodes),dtype=float)
            if num_neu_spe == 3:
                qn2_tw_nodes           = reshape_var(h5_post,"/picM_data/neu_ene_flux_in_acc2","float",dims[0],dims[1],nsteps,"all")
                qn2_fw_nodes           = reshape_var(h5_post,"/picM_data/neu_ene_flux_out_acc2","float",dims[0],dims[1],nsteps,"all")
                qn3_tw_nodes           = reshape_var(h5_post,"/picM_data/neu_ene_flux_in_acc3","float",dims[0],dims[1],nsteps,"all")
                qn3_fw_nodes           = reshape_var(h5_post,"/picM_data/neu_ene_flux_out_acc3","float",dims[0],dims[1],nsteps,"all")
            qn_tot_wall_nodes = qn1_tw_nodes + qn2_tw_nodes + qn3_tw_nodes

            imp_ene_i1_nodes       = reshape_var(h5_post,"/picM_data/ion_imp_ene_acc1","float",dims[0],dims[1],nsteps,"all")
            imp_ene_i2_nodes       = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            imp_ene_i3_nodes       = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            imp_ene_i4_nodes       = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            if num_ion_spe == 2:
                imp_ene_i2_nodes   = reshape_var(h5_post,"/picM_data/ion_imp_ene_acc2","float",dims[0],dims[1],nsteps,"all")
            elif num_ion_spe == 4:
                imp_ene_i2_nodes   = reshape_var(h5_post,"/picM_data/ion_imp_ene_acc2","float",dims[0],dims[1],nsteps,"all")
                imp_ene_i3_nodes   = reshape_var(h5_post,"/picM_data/ion_imp_ene_acc3","float",dims[0],dims[1],nsteps,"all")
                imp_ene_i4_nodes   = reshape_var(h5_post,"/picM_data/ion_imp_ene_acc4","float",dims[0],dims[1],nsteps,"all")
            
            imp_ene_n1_nodes       = reshape_var(h5_post,"/picM_data/neu_imp_ene_acc1","float",dims[0],dims[1],nsteps,"all")
            imp_ene_n2_nodes       = np.zeros(np.shape(imp_ene_n1_nodes),dtype=float)
            imp_ene_n3_nodes       = np.zeros(np.shape(imp_ene_n1_nodes),dtype=float)
            if num_neu_spe == 3:
                imp_ene_n2_nodes       = reshape_var(h5_post,"/picM_data/neu_imp_ene_acc2","float",dims[0],dims[1],nsteps,"all")
                imp_ene_n3_nodes       = reshape_var(h5_post,"/picM_data/neu_imp_ene_acc3","float",dims[0],dims[1],nsteps,"all")
        
        else:
            dphi_sh_b_nodes        = h5_post['/picM_data/dphi_sh_acc'][:,:,:]
            je_b_nodes             = h5_post['/picM_data/je_b_acc'][:,:,:]
            ge_sb_b_nodes          = h5_post['/picM_data/ge_sb_b_acc'][:,:,:]
            relerr_je_b_nodes      = h5_post['/picM_data/normal_flux_rel_err_b_acc'][:,:,:]
            qe_tot_wall_nodes      = h5_post['/picM_data/qe_tot_wall_acc'][:,:,:]
            qe_tot_b_nodes         = h5_post['/picM_data/qe_tot_b_acc'][:,:,:]
            qe_b_nodes             = h5_post['/picM_data/qe_b_acc'][:,:,:]
            if oldpost_sim <= 3:
                qe_b_bc_nodes          = h5_post['/picM_data/qe_b_bc_acc'][:,:,:]
                qe_b_fl_nodes          = h5_post['/picM_data/qe_b_fl_acc'][:,:,:]
                relerr_qe_b_nodes      = h5_post['/picM_data/normal_heat_flux_rel_err_b_acc'][:,:,:]
                relerr_qe_b_cons_nodes = h5_post['/picM_data/normal_cons_heat_flux_rel_err_b_acc'][:,:,:]
            elif oldpost_sim > 3:
                qe_b_bc_nodes          = h5_post['/picM_data/qe_tot_b_bc_acc'][:,:,:]
                qe_b_fl_nodes          = h5_post['/picM_data/qe_tot_b_fl_acc'][:,:,:]
                relerr_qe_b_nodes      = h5_post['/picM_data/normal_energy_flux_rel_err_b_acc'][:,:,:]
                relerr_qe_b_cons_nodes = h5_post['/picM_data/normal_cons_energy_flux_rel_err_b_acc'][:,:,:]
            Te_nodes               = h5_post['/picM_data/Te_acc'][:,:,:]
            phi_nodes              = h5_post['/picM_data/phi_acc'][:,:,:]
            if oldpost_sim > 3:
                delta_r_nodes          = h5_post['/picM_data/delta_r_acc'][:,:,:]
                delta_s_nodes          = h5_post['/picM_data/delta_s_acc'][:,:,:]
                delta_s_csl_nodes      = np.zeros(np.shape(je_b_nodes),dtype=float)
                if oldpost_sim >=6 and oldsimparams_sim >= 17:
                    delta_s_csl_nodes  = h5_post['/picM_data/delta_s_csl_acc'][:,:,:]
                gp_net_b_nodes         = h5_post['/picM_data/gp_net_b_acc'][:,:,:]
                qe_tot_s_wall_nodes    = h5_post['/picM_data/qe_tot_s_wall_acc'][:,:,:]
                if interp_check == 1:
                    err_interp_n_nodes     = h5_post['/picM_data/err_interp_n_acc'][:,:,:]
                else:
                    err_interp_n_nodes  = np.zeros(np.shape(je_b_nodes),dtype=float)
            else:
                delta_r_nodes       = np.zeros(np.shape(je_b_nodes),dtype=float)
                delta_s_nodes       = np.zeros(np.shape(je_b_nodes),dtype=float)
                delta_s_csl_nodes   = np.zeros(np.shape(je_b_nodes),dtype=float)
                gp_net_b_nodes      = np.zeros(np.shape(je_b_nodes),dtype=float)
                err_interp_n_nodes  = np.zeros(np.shape(je_b_nodes),dtype=float)
                qe_tot_s_wall_nodes = np.zeros(np.shape(je_b_nodes),dtype=float) 
                
            if oldsimparams_sim >= 21:
                inst_dphi_sh_b_Te_nodes       = h5_post['/picM_data/dphi_sh_b_Te_acc'][:,:,:] 
                inst_imp_ene_e_b_nodes        = h5_post['/picM_data/imp_ene_e_b_acc'][:,:,:]  
                inst_imp_ene_e_b_Te_nodes     = h5_post['/picM_data/imp_ene_e_b_Te_acc'][:,:,:]   
                inst_imp_ene_e_wall_nodes     = h5_post['/picM_data/imp_ene_e_wall_acc'][:,:,:]    
                inst_imp_ene_e_wall_Te_nodes  = h5_post['/picM_data/imp_ene_e_wall_Te_acc'][:,:,:]   
                
            else:  
                inst_dphi_sh_b_Te_nodes       = np.zeros(np.shape(je_b_nodes),dtype=float) 
                inst_imp_ene_e_b_nodes        = np.zeros(np.shape(je_b_nodes),dtype=float) 
                inst_imp_ene_e_b_Te_nodes     = np.zeros(np.shape(je_b_nodes),dtype=float) 
                inst_imp_ene_e_wall_nodes     = np.zeros(np.shape(je_b_nodes),dtype=float) 
                inst_imp_ene_e_wall_Te_nodes  = np.zeros(np.shape(je_b_nodes),dtype=float) 
    
            # Data from PIC module
            n_inst_nodes           = h5_post['/picM_data/n'][:,:,:]
            ni1_inst_nodes         = h5_post['/picM_data/ni1'][:,:,:]
            ni2_inst_nodes         = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            if num_ion_spe == 2:
                ni2_inst_nodes     = h5_post['/picM_data/ni2'][:,:,:]
            nn1_inst_nodes         = h5_post['/picM_data/nn1'][:,:,:]
            n_nodes                = h5_post['/picM_data/n_acc'][:,:,:]
            ni1_nodes              = h5_post['/picM_data/ni_acc1'][:,:,:]
            ni2_nodes              = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            if num_ion_spe == 2:
                ni2_nodes          = h5_post['/picM_data/ni_acc2'][:,:,:]
            nn1_nodes              = h5_post['/picM_data/nn_acc1'][:,:,:]
            dphi_kbc_nodes         = h5_post['/picM_data/dphi_kbc_acc'][:,:,:]
            MkQ1_nodes             = h5_post['/picM_data/MkQ1_acc'][:,:,:]
            ji1_nodes              = h5_post['/picM_data/ion_flux_in_acc1'][:,:,:]*e
            ji2_nodes              = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            ji3_nodes              = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            ji4_nodes              = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            if num_ion_spe == 2:
                ji2_nodes          = h5_post['/picM_data/ion_flux_in_acc2'][:,:,:]*2*e
            elif num_ion_spe == 4:
                ji2_nodes          = h5_post['/picM_data/ion_flux_in_acc2'][:,:,:]*2*e
                ji3_nodes          = h5_post['/picM_data/ion_flux_in_acc3'][:,:,:]*e
                ji4_nodes          = h5_post['/picM_data/ion_flux_in_acc4'][:,:,:]*2*e
            ji_nodes               = ji1_nodes + ji2_nodes + ji3_nodes + ji4_nodes
            gn1_tw_nodes           = h5_post['/picM_data/neu_flux_in_acc1'][:,:,:]
            gn1_fw_nodes           = h5_post['/picM_data/neu_flux_out_acc1'][:,:,:]
            gn2_tw_nodes           = np.zeros(np.shape(gn1_tw_nodes),dtype=float)
            gn2_fw_nodes           = np.zeros(np.shape(gn1_tw_nodes),dtype=float)
            gn3_tw_nodes           = np.zeros(np.shape(gn1_tw_nodes),dtype=float)
            gn3_fw_nodes           = np.zeros(np.shape(gn1_tw_nodes),dtype=float)
            if num_neu_spe == 3:
                gn2_tw_nodes           = h5_post['/picM_data/neu_flux_in_acc2'][:,:,:]
                gn2_fw_nodes           = h5_post['/picM_data/neu_flux_out_acc2'][:,:,:]
                gn3_tw_nodes           = h5_post['/picM_data/neu_flux_in_acc3'][:,:,:]
                gn3_fw_nodes           = h5_post['/picM_data/neu_flux_out_acc3'][:,:,:]
            gn_tw_nodes            = gn1_tw_nodes + gn2_tw_nodes + gn3_tw_nodes
                
            qi1_tot_wall_nodes     = h5_post['/picM_data/ion_ene_flux_in_acc1'][:,:,:]
            qi2_tot_wall_nodes     = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            qi3_tot_wall_nodes     = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            qi4_tot_wall_nodes     = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            if num_ion_spe == 2:
                qi2_tot_wall_nodes = h5_post['/picM_data/ion_ene_flux_in_acc2'][:,:,:]
            elif num_ion_spe == 4:
                qi2_tot_wall_nodes = h5_post['/picM_data/ion_ene_flux_in_acc2'][:,:,:]
                qi3_tot_wall_nodes = h5_post['/picM_data/ion_ene_flux_in_acc3'][:,:,:]
                qi4_tot_wall_nodes = h5_post['/picM_data/ion_ene_flux_in_acc4'][:,:,:]
            qi_tot_wall_nodes      = qi1_tot_wall_nodes + qi2_tot_wall_nodes + qi3_tot_wall_nodes + qi4_tot_wall_nodes
            qn1_tw_nodes           = h5_post['/picM_data/neu_ene_flux_in_acc1'][:,:,:]
            qn1_fw_nodes           = h5_post['/picM_data/neu_ene_flux_out_acc1'][:,:,:]
            qn2_tw_nodes           = np.zeros(np.shape(qn1_tw_nodes),dtype=float)
            qn2_fw_nodes           = np.zeros(np.shape(qn1_tw_nodes),dtype=float)
            qn3_tw_nodes           = np.zeros(np.shape(qn1_tw_nodes),dtype=float)
            qn3_fw_nodes           = np.zeros(np.shape(qn1_tw_nodes),dtype=float)
            if num_neu_spe == 3:
                qn2_tw_nodes           = h5_post['/picM_data/neu_ene_flux_in_acc2'][:,:,:]
                qn2_fw_nodes           = h5_post['/picM_data/neu_ene_flux_out_acc2'][:,:,:]
                qn3_tw_nodes           = h5_post['/picM_data/neu_ene_flux_in_acc3'][:,:,:]
                qn3_fw_nodes           = h5_post['/picM_data/neu_ene_flux_out_acc3'][:,:,:]
            qn_tot_wall_nodes = qn1_tw_nodes + qn2_tw_nodes + qn3_tw_nodes
            
            imp_ene_i1_nodes       = h5_post['/picM_data/ion_imp_ene_acc1'][:,:,:]
            imp_ene_i2_nodes       = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            imp_ene_i3_nodes       = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            imp_ene_i4_nodes       = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
            if num_ion_spe == 2:
                imp_ene_i2_nodes   = h5_post['/picM_data/ion_imp_ene_acc2'][:,:,:]
            elif num_ion_spe == 4:
                imp_ene_i2_nodes   = h5_post['/picM_data/ion_imp_ene_acc2'][:,:,:]
                imp_ene_i3_nodes   = h5_post['/picM_data/ion_imp_ene_acc3'][:,:,:]
                imp_ene_i4_nodes   = h5_post['/picM_data/ion_imp_ene_acc4'][:,:,:]
                
            imp_ene_n1_nodes       = h5_post['/picM_data/neu_imp_ene_acc1'][:,:,:]
            imp_ene_n2_nodes       = np.zeros(np.shape(imp_ene_n1_nodes),dtype=float)
            imp_ene_n3_nodes       = np.zeros(np.shape(imp_ene_n1_nodes),dtype=float)
            if num_neu_spe == 3:
                imp_ene_n2_nodes       = h5_post['/picM_data/neu_imp_ene_acc2'][:,:,:]
                imp_ene_n3_nodes       = h5_post['/picM_data/neu_imp_ene_acc3'][:,:,:]
                
        
        
        # Obtain data at the boundary PIC mesh surface elements
        # Data from the eFld module
        dphi_sh_b_surf        = h5_post['/picS_data/imp_elems_dphi_sh'][:,3,:]
        je_b_surf             = h5_post['/picS_data/imp_elems_je_b'][:,3,:]
        ge_sb_b_surf          = h5_post['/picS_data/imp_elems_ge_sb_b'][:,3,:]
        relerr_je_b_surf      = h5_post['/picS_data/imp_elems_normal_flux_rel_err_b'][:,3,:]
        qe_tot_wall_surf      = h5_post['/picS_data/imp_elems_qe_tot_wall'][:,3,:]      
        qe_tot_b_surf         = h5_post['/picS_data/imp_elems_qe_tot_b'][:,3,:]
        qe_b_surf             = h5_post['/picS_data/imp_elems_qe_b'][:,3,:]
        if oldpost_sim <= 3:
            qe_b_bc_surf          = h5_post['/picS_data/imp_elems_qe_b_bc'][:,3,:]
            qe_b_fl_surf          = h5_post['/picS_data/imp_elems_qe_b_fl'][:,3,:]
            relerr_qe_b_surf      = h5_post['/picS_data/imp_elems_normal_heat_flux_rel_err_b'][:,3,:]
            relerr_qe_b_cons_surf = h5_post['/picS_data/imp_elems_normal_cons_heat_flux_rel_err_b'][:,3,:]
        elif oldpost_sim > 3:
            qe_b_bc_surf          = h5_post['/picS_data/imp_elems_qe_tot_b_bc'][:,3,:]
            qe_b_fl_surf          = h5_post['/picS_data/imp_elems_qe_tot_b_fl'][:,3,:]
            relerr_qe_b_surf      = h5_post['/picS_data/imp_elems_normal_energy_flux_rel_err_b'][:,3,:]
            relerr_qe_b_cons_surf = h5_post['/picS_data/imp_elems_normal_cons_energy_flux_rel_err_b'][:,3,:]
        Te_surf               = h5_post['/picS_data/imp_elems_Te'][:,3,:]
        phi_surf              = h5_post['/picS_data/imp_elems_phi'][:,3,:]
        if oldpost_sim > 3:
            delta_r_surf       = h5_post['/picS_data/imp_elems_delta_r'][:,3,:]
            delta_s_surf       = h5_post['/picS_data/imp_elems_delta_s'][:,3,:]
            delta_s_csl_surf   = np.zeros(np.shape(je_b_surf),dtype=float)
            if oldpost_sim >=6 and oldsimparams_sim >= 17:
                delta_s_csl_surf = h5_post['/picS_data/imp_elems_delta_s_csl'][:,3,:]
            gp_net_b_surf      = h5_post['/picS_data/imp_elems_gp_net_b'][:,3,:]
            qe_tot_s_wall_surf = h5_post['/picS_data/imp_elems_qe_tot_s_wall'][:,3,:]
        else:
            delta_r_surf       = np.zeros(np.shape(je_b_surf),dtype=float)
            delta_s_surf       = np.zeros(np.shape(je_b_surf),dtype=float)
            delta_s_csl_surf   = np.zeros(np.shape(je_b_surf),dtype=float)
            gp_net_b_surf      = np.zeros(np.shape(je_b_surf),dtype=float)
            qe_tot_s_wall_surf = np.zeros(np.shape(je_b_surf),dtype=float)
            
        if oldsimparams_sim >= 21:
            inst_dphi_sh_b_Te_surf      = h5_post['/picS_data/imp_elems_dphi_sh_b_Te'][:,3,:]
            inst_imp_ene_e_b_surf       = h5_post['/picS_data/imp_elems_imp_ene_e_b'][:,3,:]
            inst_imp_ene_e_b_Te_surf    = h5_post['/picS_data/imp_elems_imp_ene_e_b_Te'][:,3,:]
            inst_imp_ene_e_wall_surf    = h5_post['/picS_data/imp_elems_imp_ene_e_wall'][:,3,:]
            inst_imp_ene_e_wall_Te_surf = h5_post['/picS_data/imp_elems_imp_ene_e_wall_Te'][:,3,:]
        else:
            inst_dphi_sh_b_Te_surf      = np.zeros(np.shape(je_b_surf),dtype=float)
            inst_imp_ene_e_b_surf       = np.zeros(np.shape(je_b_surf),dtype=float)
            inst_imp_ene_e_b_Te_surf    = np.zeros(np.shape(je_b_surf),dtype=float)
            inst_imp_ene_e_wall_surf    = np.zeros(np.shape(je_b_surf),dtype=float)
            inst_imp_ene_e_wall_Te_surf = np.zeros(np.shape(je_b_surf),dtype=float)
            
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
        
        # Obtain the infinite potential
        phi_inf = np.zeros(nsteps,dtype=float)
        if ff_c_bound_type_je == 2 or ff_c_bound_type_je == 3:
            # phi_inf = h5_post['/eFldM_data/boundary/fl_cms/V_acc'][:,0] 
            phi_inf = h5_post['/eFldM_data/boundary/fl_cms/V'][:,0] 
        
    
        # Computation of wall impact energies in eV
        # Electrons
        imp_ene_e_wall       = qe_tot_wall/(-je_b/e)/e
        imp_ene_e_wall_nodes = qe_tot_wall_nodes/(-je_b_nodes/e)/e
        imp_ene_e_wall_surf  = qe_tot_wall_surf/(-je_b_surf/e)/e
        imp_ene_e_b          = qe_tot_b/(-je_b/e)/e
        imp_ene_e_b_nodes    = qe_tot_b_nodes/(-je_b_nodes/e)/e
        imp_ene_e_b_surf     = qe_tot_b_surf/(-je_b_surf/e)/e
        # Ions
        den                  = ji1_nodes/e + ji2_nodes/(2*e) + ji3_nodes/e + ji4_nodes/(2*e)
        imp_ene_ion_nodes    = (imp_ene_i1_nodes*ji1_nodes/e     + \
                                imp_ene_i2_nodes*ji2_nodes/(2*e) + \
                                imp_ene_i3_nodes*ji3_nodes/e     + \
                                imp_ene_i4_nodes*ji4_nodes/(2*e))/den/e
        imp_ene_ion_nodes_v2 = qi_tot_wall_nodes/(ji_nodes/e)/e
        den                  = ji1_surf/e + ji2_surf/(2*e) + ji3_surf/e + ji4_surf/(2*e)
        imp_ene_ion_surf     = (imp_ene_i1_surf*ji1_surf/e     + \
                                imp_ene_i2_surf*ji2_surf/(2*e) + \
                                imp_ene_i3_surf*ji3_surf/e     + \
                                imp_ene_i4_surf*ji4_surf/(2*e))/den/e
        imp_ene_ion_surf_v2  = qi_tot_wall_surf/(ji_surf/e)/e
        imp_ene_i1_nodes     = imp_ene_i1_nodes/e
        imp_ene_i2_nodes     = imp_ene_i2_nodes/e
        imp_ene_i3_nodes     = imp_ene_i3_nodes/e
        imp_ene_i4_nodes     = imp_ene_i4_nodes/e
        imp_ene_i1_surf      = imp_ene_i1_surf/e
        imp_ene_i2_surf      = imp_ene_i2_surf/e
        imp_ene_i3_surf      = imp_ene_i3_surf/e
        imp_ene_i4_surf      = imp_ene_i4_surf/e
        
        # Neutrals
        imp_ene_n_nodes      = (imp_ene_n1_nodes*gn1_tw_nodes + imp_ene_n2_nodes*gn2_tw_nodes + imp_ene_n3_nodes*gn3_tw_nodes)/gn_tw_nodes/e
        imp_ene_n_nodes_v2   = qn_tot_wall_nodes/gn_tw_nodes/e
        imp_ene_n_surf       = (imp_ene_n1_surf*gn1_tw_surf + imp_ene_n2_surf*gn2_tw_surf + imp_ene_n3_surf*gn3_tw_surf)/gn_tw_surf/e
        imp_ene_n_surf_v2    = qn_tot_wall_surf/gn_tw_surf/e
        imp_ene_n1_nodes     = imp_ene_n1_nodes/e
        imp_ene_n2_nodes     = imp_ene_n2_nodes/e
        imp_ene_n3_nodes     = imp_ene_n3_nodes/e
        imp_ene_n1_surf      = imp_ene_n1_surf/e
        imp_ene_n2_surf      = imp_ene_n2_surf/e
        imp_ene_n3_surf      = imp_ene_n3_surf/e
    
    
    
    elif allsteps_flag == 0:
        print("HET_sims_read_bound: reading all steps data...")
        # Obtain data at the boundary faces
        # Bfield data
        Bfield = face_geom[-1,:]
        # Accumulated data ----------------------------------------------------
        delta_r            = h5_post['/eFldM_data/boundary/delta_r_acc'][timestep,:]
        delta_s            = h5_post['/eFldM_data/boundary/delta_s_acc'][timestep,:]
        dphi_sh_b          = h5_post['/eFldM_data/boundary/dphi_sh_b_acc'][timestep,:]
        je_b               = h5_post['/eFldM_data/boundary/je_b_acc'][timestep,:]
        ji_tot_b           = h5_post['/eFldM_data/boundary/ji_tot_b_acc'][timestep,:]
        gp_net_b           = h5_post['/eFldM_data/boundary/gp_net_b_acc'][timestep,:]
        ge_sb_b            = h5_post['/eFldM_data/boundary/ge_sb_b_acc'][timestep,:]
        relerr_je_b        = h5_post['/eFldM_data/boundary/normal_flux_rel_err_b_acc'][timestep,:]
        qe_tot_wall        = h5_post['/eFldM_data/boundary/qe_tot_wall_acc'][timestep,:]
        qe_tot_s_wall      = h5_post['/eFldM_data/boundary/qe_tot_s_wall_acc'][timestep,:]
        qe_tot_b           = h5_post['/eFldM_data/boundary/qe_tot_b_acc'][timestep,:]
        qe_b               = h5_post['/eFldM_data/boundary/qe_b_acc'][timestep,:]
        qe_b_bc            = h5_post['/eFldM_data/boundary/qe_b_bc_acc'][timestep,:]
        qe_b_fl            = h5_post['/eFldM_data/boundary/qe_b_fl_acc'][timestep,:]
        relerr_qe_b        = h5_post['/eFldM_data/boundary/normal_heat_flux_rel_err_b_acc'][timestep,:]
        relerr_qe_b_cons   = h5_post['/eFldM_data/boundary/normal_cons_heat_flux_rel_err_b_acc'][timestep,:]
        Te                 = h5_post['/eFldM_data/faces/Te_acc'][timestep,:]
        phi                = h5_post['/eFldM_data/faces/phi_acc'][timestep,:]
        err_interp_phi     = h5_post['/eFldM_data/faces/err_interp_acc'][timestep,0,:]
        err_interp_Te      = h5_post['/eFldM_data/faces/err_interp_acc'][timestep,1,:]
        err_interp_jeperp  = h5_post['/eFldM_data/faces/err_interp_acc'][timestep,2,:]
        err_interp_jetheta = h5_post['/eFldM_data/faces/err_interp_acc'][timestep,3,:]
        err_interp_jepara  = h5_post['/eFldM_data/faces/err_interp_acc'][timestep,4,:]
        err_interp_jez     = h5_post['/eFldM_data/faces/err_interp_acc'][timestep,5,:]
        err_interp_jer     = h5_post['/eFldM_data/faces/err_interp_acc'][timestep,6,:]
        n_inst             = h5_post['/eFldM_data/faces/n'][timestep,:]
        ni1_inst           = h5_post['/eFldM_data/faces/ni'][timestep,0,:]
        ni2_inst           = np.zeros(np.shape(ni1_inst),dtype=float)
        if num_ion_spe == 2:
            ni2_inst       = h5_post['/eFldM_data/faces/ni'][timestep,1,:]
        nn1_inst           = h5_post['/eFldM_data/faces/nn'][timestep,0,:]
        
        # Obtain data at the boundary PIC mesh nodes
        # Bfield data
        Br_nodes               = h5_out['/picM/Br'][0:,0:]
        Bz_nodes               = h5_out['/picM/Bz'][0:,0:]
        Bfield_nodes           = np.sqrt(Br_nodes**2.0+Bz_nodes**2.0)
        # Data from the eFld module
        delta_r_nodes          = h5_post['/picM_data/delta_r_acc'][:,:,timestep]
        delta_s_nodes          = h5_post['/picM_data/delta_s_acc'][:,:,timestep]
        dphi_sh_b_nodes        = h5_post['/picM_data/dphi_sh_acc'][:,:,timestep]
        je_b_nodes             = h5_post['/picM_data/je_b_acc'][:,:,timestep]
        gp_net_b_nodes         = h5_post['/picM_data/gp_net_b_acc'][:,:,timestep]
        ge_sb_b_nodes          = h5_post['/picM_data/ge_sb_b_acc'][:,:,timestep]
        relerr_je_b_nodes      = h5_post['/picM_data/normal_flux_rel_err_b_acc'][:,:,timestep]
        qe_tot_wall_nodes      = h5_post['/picM_data/qe_tot_wall_acc'][:,:,timestep]
        qe_tot_s_wall_nodes    = h5_post['/picM_data/qe_tot_s_wall_acc'][:,:,timestep]
        qe_tot_b_nodes         = h5_post['/picM_data/qe_tot_b_acc'][:,:,timestep]
        qe_b_nodes             = h5_post['/picM_data/qe_b_acc'][:,:,timestep]
        qe_b_bc_nodes          = h5_post['/picM_data/qe_b_bc_acc'][:,:,timestep]
        qe_b_fl_nodes          = h5_post['/picM_data/qe_b_fl_acc'][:,:,timestep]
        relerr_qe_b_nodes      = h5_post['/picM_data/normal_heat_flux_rel_err_b_acc'][:,:,timestep]
        relerr_qe_b_cons_nodes = h5_post['/picM_data/normal_cons_heat_flux_rel_err_b_acc'][:,:,timestep]
        Te_nodes               = h5_post['/picM_data/Te_acc'][:,:,timestep]
        phi_nodes              = h5_post['/picM_data/phi_acc'][:,:,timestep]
        err_interp_n_nodes     = h5_post['/picM_data/err_interp_n_acc'][:,:,timestep]
        # Data from PIC module
        n_inst_nodes           = h5_post['/picM_data/n'][:,:,timestep]
        ni1_inst_nodes         = h5_post['/picM_data/ni1'][:,:,timestep]
        ni2_inst_nodes         = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
        if num_ion_spe == 2:
            ni2_inst_nodes     = h5_post['/picM_data/ni2'][:,:,timestep]
        nn1_inst_nodes         = h5_post['/picM_data/nn1'][:,:,timestep]
        n_nodes                = h5_post['/picM_data/n_acc'][:,:,timestep]
        ni1_nodes              = h5_post['/picM_data/ni_acc1'][:,:,timestep]
        ni2_nodes              = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
        if num_ion_spe == 2:
            ni2_nodes          = h5_post['/picM_data/ni_acc2'][:,:,timestep]
        nn1_nodes              = h5_post['/picM_data/nn_acc1'][:,:,timestep]
        dphi_kbc_nodes         = h5_post['/picM_data/dphi_kbc_acc'][:,:,timestep]
        MkQ1_nodes             = h5_post['/picM_data/MkQ1_acc'][:,:,timestep]
        ji1_nodes              = h5_post['/picM_data/ion_flux_in_acc1'][:,:,timestep]*e
        ji2_nodes              = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
        if num_ion_spe == 2:
            ji2_nodes          = h5_post['/picM_data/ion_flux_in_acc2'][:,:,timestep]*2*e
        ji_nodes               = ji1_nodes + ji2_nodes
        gn1_tw_nodes           = h5_post['/picM_data/neu_flux_in_acc1'][:,:,timestep]
        gn1_fw_nodes           = h5_post['/picM_data/neu_flux_out_acc1'][:,:,timestep]
        qi1_tot_wall_nodes     = h5_post['/picM_data/ion_ene_flux_in_acc1'][:,:,timestep]
        qi2_tot_wall_nodes     = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
        if num_ion_spe == 2:
            qi2_tot_wall_nodes = h5_post['/picM_data/ion_ene_flux_in_acc2'][:,:,timestep]
        qi_tot_wall_nodes      = qi1_tot_wall_nodes + qi2_tot_wall_nodes
        qn1_tw_nodes           = h5_post['/picM_data/neu_ene_flux_in_acc1'][:,:,timestep]
        qn1_fw_nodes           = h5_post['/picM_data/neu_ene_flux_out_acc1'][:,:,timestep]
        imp_ene_i1_nodes       = h5_post['/picM_data/ion_imp_ene_acc1'][:,:,timestep]
        imp_ene_i2_nodes       = np.zeros(np.shape(ni1_inst_nodes),dtype=float)
        if num_ion_spe == 2:
            imp_ene_i2_nodes   = h5_post['/picM_data/ion_imp_ene_acc2'][:,:,timestep]
        imp_ene_n1_nodes       = h5_post['/picM_data/neu_imp_ene_acc1'][:,:,timestep]
        
        # Obtain data at the boundary PIC mesh surface elements
        # Data from the eFld module
        delta_r_surf          = h5_post['/picS_data/imp_elems_delta_r'][:,3,timestep]
        delta_s_surf          = h5_post['/picS_data/imp_elems_delta_s'][:,3,timestep]
        dphi_sh_b_surf        = h5_post['/picS_data/imp_elems_dphi_sh'][:,3,timestep]
        je_b_surf             = h5_post['/picS_data/imp_elems_je_b'][:,3,timestep]
        gp_net_b_surf         = h5_post['/picS_data/imp_elems_gp_net_b'][:,3,timestep]
        ge_sb_b_surf          = h5_post['/picS_data/imp_elems_ge_sb_b'][:,3,timestep]
        relerr_je_b_surf      = h5_post['/picS_data/imp_elems_normal_flux_rel_err_b'][:,3,timestep]
        qe_tot_wall_surf      = h5_post['/picS_data/imp_elems_qe_tot_wall'][:,3,timestep]
        qe_tot_s_wall_surf    = h5_post['/picS_data/imp_elems_qe_tot_s_wall'][:,3,timestep]
        qe_tot_b_surf         = h5_post['/picS_data/imp_elems_qe_tot_b'][:,3,timestep]
        qe_b_surf             = h5_post['/picS_data/imp_elems_qe_b'][:,3,timestep]
        qe_b_bc_surf          = h5_post['/picS_data/imp_elems_qe_b_bc'][:,3,timestep]
        qe_b_fl_surf          = h5_post['/picS_data/imp_elems_qe_b_fl'][:,3,timestep]
        relerr_qe_b_surf      = h5_post['/picS_data/imp_elems_normal_heat_flux_rel_err_b'][:,3,timestep]
        relerr_qe_b_cons_surf = h5_post['/picS_data/imp_elems_normal_cons_heat_flux_rel_err_b'][:,3,timestep]
        Te_surf               = h5_post['/picS_data/imp_elems_Te'][:,3,timestep]
        phi_surf              = h5_post['/picS_data/imp_elems_phi'][:,3,timestep]
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
        
        



    
    return[num_ion_spe,num_neu_spe,points,zs,rs,zscells,rscells,dims,nodes_flag,
           cells_flag,cells_vol,volume,vol,ind_maxr_c,ind_maxz_c,nr_c,nz_c,eta_max,
           eta_min,xi_top,xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,nsteps,
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
           sFLwall_lat,sAxis,sc_bot,sc_top,sc_bot_nodes,sc_top_nodes,sc_bot_surf,
           sc_top_surf,Lplume_bot,Lplume_top,Lchamb_bot,Lchamb_top,Lanode,
           Lfreeloss_ver,Lfreeloss_lat,Lfreeloss,Laxis,
           
           nnodes_Dwall_bot,nnodes_Dwall_top,nnodes_Awall,nnodes_FLwall_ver,
           nnodes_FLwall_lat,nnodes_Axis,nnodes_bound,inodes_Dwall_bot,
           jnodes_Dwall_bot,inodes_Dwall_top,jnodes_Dwall_top,inodes_Awall,
           jnodes_Awall,inodes_FLwall_ver,jnodes_FLwall_ver,inodes_FLwall_lat,
           jnodes_FLwall_lat,inodes_Axis,jnodes_Axis,sDwall_bot_nodes,
           sDwall_top_nodes,sAwall_nodes,sFLwall_ver_nodes,sFLwall_lat_nodes,
           sAxis_nodes,
           
           imp_elems,surf_areas,nsurf_Dwall_bot,nsurf_Dwall_top,nsurf_Awall,
           nsurf_FLwall_ver,nsurf_FLwall_lat,nsurf_bound,indsurf_Dwall_bot,
           zsurf_Dwall_bot,rsurf_Dwall_bot,indsurf_Dwall_top,zsurf_Dwall_top,
           rsurf_Dwall_top,indsurf_Awall,zsurf_Awall,rsurf_Awall,
           indsurf_FLwall_ver,zsurf_FLwall_ver,rsurf_FLwall_ver,
           indsurf_FLwall_lat,zsurf_FLwall_lat,rsurf_FLwall_lat,sDwall_bot_surf,
           sDwall_top_surf,sAwall_surf,sFLwall_ver_surf,sFLwall_lat_surf,
           
           delta_r,delta_s,delta_s_csl,dphi_sh_b,je_b,ji_tot_b,gp_net_b,ge_sb_b,
           relerr_je_b,qe_tot_wall,qe_tot_s_wall,qe_tot_b,qe_b,qe_b_bc,qe_b_fl,
           imp_ene_e_wall,imp_ene_e_b,relerr_qe_b,relerr_qe_b_cons,Te,phi,
           err_interp_phi,err_interp_Te,err_interp_jeperp,err_interp_jetheta,
           err_interp_jepara,err_interp_jez,err_interp_jer,n_inst,ni1_inst,
           ni2_inst,nn1_inst,Bfield,inst_dphi_sh_b_Te,inst_imp_ene_e_b,
           inst_imp_ene_e_b_Te,inst_imp_ene_e_wall,inst_imp_ene_e_wall_Te,
           
           delta_r_nodes,delta_s_nodes,delta_s_csl_nodes,dphi_sh_b_nodes,
           je_b_nodes,gp_net_b_nodes,ge_sb_b_nodes,relerr_je_b_nodes,
           qe_tot_wall_nodes,qe_tot_s_wall_nodes,qe_tot_b_nodes,qe_b_nodes,
           qe_b_bc_nodes,qe_b_fl_nodes,imp_ene_e_wall_nodes,imp_ene_e_b_nodes,
           relerr_qe_b_nodes,relerr_qe_b_cons_nodes,Te_nodes,phi_nodes,
           err_interp_n_nodes,n_inst_nodes,ni1_inst_nodes,ni2_inst_nodes,
           nn1_inst_nodes,n_nodes,ni1_nodes,ni2_nodes,nn1_nodes,dphi_kbc_nodes,
           MkQ1_nodes,ji1_nodes,ji2_nodes,ji3_nodes,ji4_nodes,ji_nodes,
           gn1_tw_nodes,gn1_fw_nodes,gn2_tw_nodes,gn2_fw_nodes,gn3_tw_nodes,
           gn3_fw_nodes,gn_tw_nodes,qi1_tot_wall_nodes,qi2_tot_wall_nodes,
           qi3_tot_wall_nodes,qi4_tot_wall_nodes,qi_tot_wall_nodes,qn1_tw_nodes,
           qn1_fw_nodes,qn2_tw_nodes,qn2_fw_nodes,qn3_tw_nodes,qn3_fw_nodes,
           qn_tot_wall_nodes,imp_ene_i1_nodes,imp_ene_i2_nodes,imp_ene_i3_nodes,
           imp_ene_i4_nodes,imp_ene_ion_nodes,imp_ene_ion_nodes_v2,
           imp_ene_n1_nodes,imp_ene_n2_nodes,imp_ene_n3_nodes,imp_ene_n_nodes,
           imp_ene_n_nodes_v2,Bfield_nodes,inst_dphi_sh_b_Te_nodes,
           inst_imp_ene_e_b_nodes,inst_imp_ene_e_b_Te_nodes,
           inst_imp_ene_e_wall_nodes,inst_imp_ene_e_wall_Te_nodes,
           
           delta_r_surf,delta_s_surf,delta_s_csl_surf,dphi_sh_b_surf,je_b_surf,
           gp_net_b_surf,ge_sb_b_surf,relerr_je_b_surf,qe_tot_wall_surf,
           qe_tot_s_wall_surf,qe_tot_b_surf,qe_b_surf,qe_b_bc_surf,qe_b_fl_surf,
           imp_ene_e_wall_surf,imp_ene_e_b_surf,relerr_qe_b_surf,
           relerr_qe_b_cons_surf,Te_surf,phi_surf,nQ1_inst_surf,nQ1_surf,
           nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,MkQ1_surf,ji1_surf,ji2_surf,
           ji3_surf,ji4_surf,ji_surf,gn1_tw_surf,gn1_fw_surf,gn2_tw_surf,
           gn2_fw_surf,gn3_tw_surf,gn3_fw_surf,gn_tw_surf,qi1_tot_wall_surf,
           qi2_tot_wall_surf,qi3_tot_wall_surf,qi4_tot_wall_surf,qi_tot_wall_surf,
           qn1_tw_surf,qn1_fw_surf,qn2_tw_surf,qn2_fw_surf,qn3_tw_surf,qn3_fw_surf,
           qn_tot_wall_surf,imp_ene_i1_surf,imp_ene_i2_surf,imp_ene_i3_surf,
           imp_ene_i4_surf,imp_ene_ion_surf,imp_ene_ion_surf_v2,
           imp_ene_n1_surf,imp_ene_n2_surf,imp_ene_n3_surf,imp_ene_n_surf,
           imp_ene_n_surf_v2,phi_inf,inst_dphi_sh_b_Te_surf,
           inst_imp_ene_e_b_surf,inst_imp_ene_e_b_Te_surf,
           inst_imp_ene_e_wall_surf,inst_imp_ene_e_wall_Te_surf]

    
if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 

    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))
    
#    sim_name = "../../../Sr_sims_files/SPT100_orig_tmtetq2_Vd300_test_rel"
    sim_name = "../../../Ca_sims_files/T2N3_pm1em1_cat1200_tm15_te1_tq125_Kr"
    
    sim_name = "../../../Ca_sims_files/SPT100_orig_tmtetq2_Vd300"

    timestep         = -1
    allsteps_flag    = 1
    
    oldpost_sim      = 3
    oldsimparams_sim = 8
    
    
    path_picM         = sim_name+"/SET/inp/SPT100_picM.hdf5"
#    path_picM         = sim_name +"/SET/inp/PIC_mesh_topo2_refined4.hdf5"
#    path_picM         = sim_name +"/SET/inp/SPT100_picM_Reference1500points_rm.hdf5"
    path_simstate_inp = sim_name+"/CORE/inp/SimState.hdf5"
    path_simstate_out = sim_name+"/CORE/out/SimState.hdf5"
    path_postdata_out = sim_name+"/CORE/out/PostData.hdf5"
    path_simparams_inp = sim_name+"/CORE/inp/sim_params.inp"
    
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
       
       imp_elems,surf_areas,nsurf_Dwall_bot,nsurf_Dwall_top,nsurf_Awall,nsurf_FLwall_ver,
       nsurf_FLwall_lat,nsurf_bound,indsurf_Dwall_bot,zsurf_Dwall_bot,
       rsurf_Dwall_bot,indsurf_Dwall_top,zsurf_Dwall_top,rsurf_Dwall_top,
       indsurf_Awall,zsurf_Awall,rsurf_Awall,indsurf_FLwall_ver,
       zsurf_FLwall_ver,rsurf_FLwall_ver,indsurf_FLwall_lat,
       zsurf_FLwall_lat,rsurf_FLwall_lat,sDwall_bot_surf,sDwall_top_surf,
       sAwall_surf,sFLwall_ver_surf,sFLwall_lat_surf,
       
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
    # Plot the boundary faces centers for each type of boundary
    # Dwall_bot
    plt.plot(zfaces_Dwall_bot,rfaces_Dwall_bot,'cx-',linewidth = line_width_bf,markersize = marker_size_bf)
    # Dwall_top
    plt.plot(zfaces_Dwall_top,rfaces_Dwall_top,'cx-',linewidth = line_width_bf,markersize = marker_size_bf)
    # Awall
    plt.plot(zfaces_Awall,rfaces_Awall,'mx-',linewidth = line_width_bf,markersize = marker_size_bf)
    # FLwall_ver
    plt.plot(zfaces_FLwall_ver,rfaces_FLwall_ver,'yx-',linewidth = line_width_bf,markersize = marker_size_bf)
    # FLwall_lat
    plt.plot(zfaces_FLwall_lat,rfaces_FLwall_lat,'x-',color = orange,linewidth = line_width_bf,markersize = marker_size_bf)
    # Axis
    plt.plot(zfaces_Axis,rfaces_Axis,'x-',color = brown,linewidth = line_width_bf,markersize = marker_size_bf)
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