# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:16:42 2018

@author: adrian

############################################################################
Description:    This python script obtains the plotting variables including
                NaN's at those ghost nodes and cells so that contour, isolines
                and streamlines plots can be properly performed 
############################################################################
Inputs:        1) Variables to be modified for proper plotting
############################################################################
Output:        1) Plot variables
"""

def HET_sims_plotvars(nodes_flag,cells_flag,Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,
                      nn2,nn3,ni1,ni2,ni3,ni4,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,
                      fn2_z,fn3_x,fn3_y,fn3_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,
                      fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,un1_x,un1_y,
                      un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,ui1_x,ui1_y,
                      ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,
                      ui4_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,
                      ji3_z,ji4_x,ji4_y,ji4_z,je_r,je_t,je_z,je_perp,je_para,
                      ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Tn3,
                      Ti1,Ti2,Ti3,Ti4,Te,n_mp_n1,n_mp_n2,n_mp_n3,
                      n_mp_i1,n_mp_i2,n_mp_i3,n_mp_i4,avg_w_n1,avg_w_n2,
                      avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
                      ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,
                      ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
                      ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,ndot_CEX01_i3,
                      ndot_CEX02_i4,F_theta,Hall_par,Hall_par_eff,nu_e_tot,
                      nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,nu_ex,
                      err_interp_n,f_split_adv,f_split_qperp,f_split_qpara,
                      f_split_qb,f_split_Pperp,f_split_Ppara,f_split_ecterm,
                      f_split_inel,alpha_ano,alpha_ano_e,alpha_ano_q,alpha_ine,
                      alpha_ine_q):
    
    import numpy as np
    
    # Magnetic field
    Br[np.where(nodes_flag == 0)]     = np.nan
    Bz[np.where(nodes_flag == 0)]     = np.nan
    Bfield[np.where(nodes_flag == 0)] = np.nan  
    # Electric potential
    phi[np.where(nodes_flag == 0)]    = np.nan
    # Electric field
    Er[np.where(nodes_flag == 0)]     = np.nan
    Ez[np.where(nodes_flag == 0)]     = np.nan
    Efield[np.where(nodes_flag == 0)] = np.nan           
    # Particle densities, fluxes and velocities
    nn1[np.where(nodes_flag == 0)]    = np.nan
    nn2[np.where(nodes_flag == 0)]    = np.nan
    nn3[np.where(nodes_flag == 0)]    = np.nan
    ni1[np.where(nodes_flag == 0)]    = np.nan
    ni2[np.where(nodes_flag == 0)]    = np.nan
    ni3[np.where(nodes_flag == 0)]    = np.nan
    ni4[np.where(nodes_flag == 0)]    = np.nan
    ne[np.where(nodes_flag == 0)]     = np.nan
    # Particle fluxes, currents and fluid velocities
    fn1_x[np.where(nodes_flag == 0)]   = np.nan
    fn1_y[np.where(nodes_flag == 0)]   = np.nan
    fn1_z[np.where(nodes_flag == 0)]   = np.nan
    fn2_x[np.where(nodes_flag == 0)]   = np.nan
    fn2_y[np.where(nodes_flag == 0)]   = np.nan
    fn2_z[np.where(nodes_flag == 0)]   = np.nan
    fn3_x[np.where(nodes_flag == 0)]   = np.nan
    fn3_y[np.where(nodes_flag == 0)]   = np.nan
    fn3_z[np.where(nodes_flag == 0)]   = np.nan
    fi1_x[np.where(nodes_flag == 0)]   = np.nan
    fi1_y[np.where(nodes_flag == 0)]   = np.nan
    fi1_z[np.where(nodes_flag == 0)]   = np.nan
    fi2_x[np.where(nodes_flag == 0)]   = np.nan
    fi2_y[np.where(nodes_flag == 0)]   = np.nan
    fi2_z[np.where(nodes_flag == 0)]   = np.nan
    fi3_x[np.where(nodes_flag == 0)]   = np.nan
    fi3_y[np.where(nodes_flag == 0)]   = np.nan
    fi3_z[np.where(nodes_flag == 0)]   = np.nan
    fi4_x[np.where(nodes_flag == 0)]   = np.nan
    fi4_y[np.where(nodes_flag == 0)]   = np.nan
    fi4_z[np.where(nodes_flag == 0)]   = np.nan
    un1_x[np.where(nodes_flag == 0)]   = np.nan
    un1_y[np.where(nodes_flag == 0)]   = np.nan
    un1_z[np.where(nodes_flag == 0)]   = np.nan
    un2_x[np.where(nodes_flag == 0)]   = np.nan
    un2_y[np.where(nodes_flag == 0)]   = np.nan
    un2_z[np.where(nodes_flag == 0)]   = np.nan
    un3_x[np.where(nodes_flag == 0)]   = np.nan
    un3_y[np.where(nodes_flag == 0)]   = np.nan
    un3_z[np.where(nodes_flag == 0)]   = np.nan
    ui1_x[np.where(nodes_flag == 0)]   = np.nan 
    ui1_y[np.where(nodes_flag == 0)]   = np.nan 
    ui1_z[np.where(nodes_flag == 0)]   = np.nan
    ui2_x[np.where(nodes_flag == 0)]   = np.nan
    ui2_y[np.where(nodes_flag == 0)]   = np.nan
    ui2_z[np.where(nodes_flag == 0)]   = np.nan
    ui3_x[np.where(nodes_flag == 0)]   = np.nan
    ui3_y[np.where(nodes_flag == 0)]   = np.nan
    ui3_z[np.where(nodes_flag == 0)]   = np.nan
    ui4_x[np.where(nodes_flag == 0)]   = np.nan
    ui4_y[np.where(nodes_flag == 0)]   = np.nan
    ui4_z[np.where(nodes_flag == 0)]   = np.nan
    ji1_x[np.where(nodes_flag == 0)]   = np.nan
    ji1_y[np.where(nodes_flag == 0)]   = np.nan
    ji1_z[np.where(nodes_flag == 0)]   = np.nan
    ji2_x[np.where(nodes_flag == 0)]   = np.nan
    ji2_y[np.where(nodes_flag == 0)]   = np.nan
    ji2_z[np.where(nodes_flag == 0)]   = np.nan
    ji3_x[np.where(nodes_flag == 0)]   = np.nan
    ji3_y[np.where(nodes_flag == 0)]   = np.nan
    ji3_z[np.where(nodes_flag == 0)]   = np.nan
    ji4_x[np.where(nodes_flag == 0)]   = np.nan
    ji4_y[np.where(nodes_flag == 0)]   = np.nan
    ji4_z[np.where(nodes_flag == 0)]   = np.nan
    je_r[np.where(nodes_flag == 0)]    = np.nan
    je_t[np.where(nodes_flag == 0)]    = np.nan
    je_z[np.where(nodes_flag == 0)]    = np.nan
    je_perp[np.where(nodes_flag == 0)] = np.nan
    je_para[np.where(nodes_flag == 0)] = np.nan
    ue_r[np.where(nodes_flag == 0)]    = np.nan
    ue_t[np.where(nodes_flag == 0)]    = np.nan
    ue_z[np.where(nodes_flag == 0)]    = np.nan
    ue_perp[np.where(nodes_flag == 0)] = np.nan
    ue_para[np.where(nodes_flag == 0)] = np.nan
    uthetaExB[np.where(nodes_flag == 0)] = np.nan
    # Temperatures
    Tn1[np.where(nodes_flag == 0)]     = np.nan
    Tn2[np.where(nodes_flag == 0)]     = np.nan
    Tn3[np.where(nodes_flag == 0)]     = np.nan
    Ti1[np.where(nodes_flag == 0)]     = np.nan
    Ti2[np.where(nodes_flag == 0)]     = np.nan
    Ti3[np.where(nodes_flag == 0)]     = np.nan
    Ti4[np.where(nodes_flag == 0)]     = np.nan
    Te[np.where(nodes_flag == 0)]      = np.nan
    # Number of particles per cell
    n_mp_n1[np.where(nodes_flag == 0)] = np.nan
    n_mp_n2[np.where(nodes_flag == 0)] = np.nan
    n_mp_n3[np.where(nodes_flag == 0)] = np.nan
    n_mp_i1[np.where(nodes_flag == 0)] = np.nan
    n_mp_i2[np.where(nodes_flag == 0)] = np.nan
    n_mp_i3[np.where(nodes_flag == 0)] = np.nan
    n_mp_i4[np.where(nodes_flag == 0)] = np.nan
    n_mp_n1[np.where(cells_flag == 0)] = np.nan
    n_mp_n2[np.where(cells_flag == 0)] = np.nan
    n_mp_n3[np.where(cells_flag == 0)] = np.nan
    n_mp_i1[np.where(cells_flag == 0)] = np.nan
    n_mp_i2[np.where(cells_flag == 0)] = np.nan
    n_mp_i3[np.where(cells_flag == 0)] = np.nan
    n_mp_i4[np.where(cells_flag == 0)] = np.nan
    n_mp_n1[-1,:] = np.nan
    n_mp_n2[-1,:] = np.nan
    n_mp_n3[-1,:] = np.nan
    n_mp_i1[-1,:] = np.nan
    n_mp_i2[-1,:] = np.nan
    n_mp_i3[-1,:] = np.nan
    n_mp_i4[-1,:] = np.nan
    n_mp_n1[:,-1] = np.nan
    n_mp_n2[:,-1] = np.nan
    n_mp_n3[:,-1] = np.nan
    n_mp_i1[:,-1] = np.nan
    n_mp_i2[:,-1] = np.nan
    n_mp_i3[:,-1] = np.nan
    n_mp_i4[:,-1] = np.nan
    
    # Average particle weight per cell
    avg_w_n1[np.where(nodes_flag == 0)] = np.nan
    avg_w_n2[np.where(nodes_flag == 0)] = np.nan
    avg_w_i1[np.where(nodes_flag == 0)] = np.nan
    avg_w_i2[np.where(nodes_flag == 0)] = np.nan
    avg_w_n1[np.where(cells_flag == 0)] = np.nan
    avg_w_n2[np.where(cells_flag == 0)] = np.nan
    avg_w_i1[np.where(cells_flag == 0)] = np.nan
    avg_w_i2[np.where(cells_flag == 0)] = np.nan
    avg_w_n1[-1,:] = np.nan
    avg_w_n2[-1,:] = np.nan
    avg_w_i1[-1,:] = np.nan
    avg_w_i2[-1,:] = np.nan
    avg_w_n1[:,-1] = np.nan
    avg_w_n2[:,-1] = np.nan
    avg_w_i1[:,-1] = np.nan
    avg_w_i2[:,-1] = np.nan
    # Generation weights per cell
    neu_gen_weights1[np.where(nodes_flag == 0)] = np.nan
    neu_gen_weights2[np.where(nodes_flag == 0)] = np.nan
    ion_gen_weights1[np.where(nodes_flag == 0)] = np.nan
    ion_gen_weights2[np.where(nodes_flag == 0)] = np.nan
    neu_gen_weights1[np.where(cells_flag == 0)] = np.nan
    neu_gen_weights2[np.where(cells_flag == 0)] = np.nan
    ion_gen_weights1[np.where(cells_flag == 0)] = np.nan
    ion_gen_weights2[np.where(cells_flag == 0)] = np.nan
    neu_gen_weights1[-1,:] = np.nan
    neu_gen_weights2[-1,:] = np.nan
    ion_gen_weights1[-1,:] = np.nan
    ion_gen_weights2[-1,:] = np.nan
    neu_gen_weights1[:,-1] = np.nan
    neu_gen_weights2[:,-1] = np.nan
    ion_gen_weights1[:,-1] = np.nan
    ion_gen_weights2[:,-1] = np.nan
    # Ionization source term (ni_dot) per cell for each ionization collision
    ndot_ion01_n1[np.where(nodes_flag == 0)] = np.nan
    ndot_ion02_n1[np.where(nodes_flag == 0)] = np.nan
    ndot_ion12_i1[np.where(nodes_flag == 0)] = np.nan
    ndot_ion01_n2[np.where(nodes_flag == 0)] = np.nan
    ndot_ion02_n2[np.where(nodes_flag == 0)] = np.nan
    ndot_ion01_n3[np.where(nodes_flag == 0)] = np.nan
    ndot_ion02_n3[np.where(nodes_flag == 0)] = np.nan
    ndot_ion12_i3[np.where(nodes_flag == 0)] = np.nan
    ndot_ion01_n1[np.where(cells_flag == 0)] = np.nan
    ndot_ion02_n1[np.where(cells_flag == 0)] = np.nan
    ndot_ion12_i1[np.where(cells_flag == 0)] = np.nan
    ndot_ion01_n2[np.where(cells_flag == 0)] = np.nan
    ndot_ion02_n2[np.where(cells_flag == 0)] = np.nan
    ndot_ion01_n3[np.where(cells_flag == 0)] = np.nan
    ndot_ion02_n3[np.where(cells_flag == 0)] = np.nan
    ndot_ion12_i3[np.where(cells_flag == 0)] = np.nan
    ndot_CEX01_i3[np.where(cells_flag == 0)] = np.nan
    ndot_CEX02_i4[np.where(cells_flag == 0)] = np.nan
    ndot_ion01_n1[-1,:] = np.nan
    ndot_ion02_n1[-1,:] = np.nan
    ndot_ion12_i1[-1,:] = np.nan
    ndot_ion01_n2[-1,:] = np.nan
    ndot_ion02_n2[-1,:] = np.nan
    ndot_ion01_n3[-1,:] = np.nan
    ndot_ion02_n3[-1,:] = np.nan
    ndot_ion12_i3[-1,:] = np.nan
    ndot_ion01_n1[:,-1] = np.nan
    ndot_ion02_n1[:,-1] = np.nan
    ndot_ion12_i1[:,-1] = np.nan
    ndot_ion01_n2[:,-1] = np.nan
    ndot_ion02_n2[:,-1] = np.nan
    ndot_ion01_n3[:,-1] = np.nan
    ndot_ion02_n3[:,-1] = np.nan
    ndot_ion12_i3[:,-1] = np.nan
    ndot_CEX01_i3[:,-1] = np.nan
    ndot_CEX02_i4[:,-1] = np.nan
    

    F_theta[np.where(nodes_flag == 0)]      = np.nan
    Hall_par[np.where(nodes_flag == 0)]     = np.nan
    Hall_par_eff[np.where(nodes_flag == 0)] = np.nan
    nu_e_tot[np.where(nodes_flag == 0)]     = np.nan
    nu_e_tot_eff[np.where(nodes_flag == 0)] = np.nan
    nu_en[np.where(nodes_flag == 0)]        = np.nan
    nu_ei1[np.where(nodes_flag == 0)]       = np.nan
    nu_ei2[np.where(nodes_flag == 0)]       = np.nan
    nu_i01[np.where(nodes_flag == 0)]       = np.nan
    nu_i02[np.where(nodes_flag == 0)]       = np.nan
    nu_i12[np.where(nodes_flag == 0)]       = np.nan
    nu_ex[np.where(nodes_flag == 0)]        = np.nan
    
    # Interpolation errors in density
    err_interp_n[np.where(nodes_flag == 0)] = np.nan
    
    # f_split variables
    f_split_adv[np.where(nodes_flag == 0)]    = np.nan
    f_split_qperp[np.where(nodes_flag == 0)]  = np.nan
    f_split_qpara[np.where(nodes_flag == 0)]  = np.nan
    f_split_qb[np.where(nodes_flag == 0)]     = np.nan
    f_split_Pperp[np.where(nodes_flag == 0)]  = np.nan
    f_split_Ppara[np.where(nodes_flag == 0)]  = np.nan
    f_split_ecterm[np.where(nodes_flag == 0)] = np.nan
    f_split_inel[np.where(nodes_flag == 0)]   = np.nan
    
    # Anomalous transport parameters
    alpha_ano[np.where(nodes_flag == 0)]       = np.nan
    alpha_ano_e[np.where(nodes_flag == 0)]     = np.nan
    alpha_ano_q[np.where(nodes_flag == 0)]     = np.nan
    alpha_ine[np.where(nodes_flag == 0)]       = np.nan
    alpha_ine_q[np.where(nodes_flag == 0)]     = np.nan
    
    
    
    return[Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,
           nn2,nn3,ni1,ni2,ni3,ni4,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,
           fn2_z,fn3_x,fn3_y,fn3_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,
           fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,un1_x,un1_y,
           un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,ui1_x,ui1_y,
           ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,
           ui4_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,
           ji3_z,ji4_x,ji4_y,ji4_z,je_r,je_t,je_z,je_perp,je_para,
           ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Tn3,
           Ti1,Ti2,Ti3,Ti4,Te,n_mp_n1,n_mp_n2,n_mp_n3,
           n_mp_i1,n_mp_i2,n_mp_i3,n_mp_i4,avg_w_n1,avg_w_n2,
           avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
           ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,
           ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
           ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,ndot_CEX01_i3,ndot_CEX02_i4,
           F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,
           nu_ei2,nu_i01,nu_i02,nu_i12,nu_ex,err_interp_n,f_split_adv,
           f_split_qperp,f_split_qpara,f_split_qb,f_split_Pperp,
           f_split_Ppara,f_split_ecterm,f_split_inel,alpha_ano,alpha_ano_e,
           alpha_ano_q,alpha_ine,alpha_ine_q]
               
           
"""
############################################################################
Description:    This python fucntion return the given variables as a copy with
                the plotting name
############################################################################
Inputs:        1) Input variables to be copied
############################################################################
Output:        1) Plot variables
"""
           
#def HET_sims_cp_vars(Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,
#                     fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,
#                     un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,
#                     ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,
#                     ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
#                     n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
#                     neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
#                     ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,nu_cath,ndot_cath,F_theta,
#                     Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,
#                     nu_i02,nu_i12,err_interp_n,f_split_adv,f_split_qperp,f_split_qpara,f_split_qb,
#                     f_split_Pperp,f_split_Ppara,f_split_ecterm,f_split_inel):


def HET_sims_cp_vars(Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,
                     nn2,nn3,ni1,ni2,ni3,ni4,ne,fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,
                     fn2_z,fn3_x,fn3_y,fn3_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,
                     fi2_z,fi3_x,fi3_y,fi3_z,fi4_x,fi4_y,fi4_z,un1_x,un1_y,
                     un1_z,un2_x,un2_y,un2_z,un3_x,un3_y,un3_z,ui1_x,ui1_y,
                     ui1_z,ui2_x,ui2_y,ui2_z,ui3_x,ui3_y,ui3_z,ui4_x,ui4_y,
                     ui4_z,ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,ji3_x,ji3_y,
                     ji3_z,ji4_x,ji4_y,ji4_z,je_r,je_t,je_z,je_perp,je_para,
                     ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Tn3,
                     Ti1,Ti2,Ti3,Ti4,Te,n_mp_n1,n_mp_n2,n_mp_n3,n_mp_i1,n_mp_i2,
                     n_mp_i3,n_mp_i4,avg_w_n1,avg_w_n2,
                     avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
                     ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,
                     ndot_ion02_n1,ndot_ion12_i1,ndot_ion01_n2,ndot_ion02_n2,
                     ndot_ion01_n3,ndot_ion02_n3,ndot_ion12_i3,ndot_CEX01_i3,
                     ndot_CEX02_i4,ne_cath,nu_cath,
                     ndot_cath,F_theta,Hall_par,Hall_par_eff,nu_e_tot,
                     nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12,nu_ex,
                     err_interp_n,f_split_adv,f_split_qperp,f_split_qpara,
                     f_split_qb,f_split_Pperp,f_split_Ppara,f_split_ecterm,
                     f_split_inel):
    
    
    
    
                         
    import numpy as np
    
    Br_plot               = np.copy(Br)
    Bz_plot               = np.copy(Bz)
    Bfield_plot           = np.copy(Bfield)
    phi_plot              = np.copy(phi)
    Er_plot               = np.copy(Er)
    Ez_plot               = np.copy(Ez)
    Efield_plot           = np.copy(Efield)
    nn1_plot              = np.copy(nn1)
    nn2_plot              = np.copy(nn2)
    nn3_plot              = np.copy(nn3)
    ni1_plot              = np.copy(ni1)
    ni2_plot              = np.copy(ni2)
    ni3_plot              = np.copy(ni3)
    ni4_plot              = np.copy(ni4)
    ne_plot               = np.copy(ne)
    fn1_x_plot            = np.copy(fn1_x)
    fn1_y_plot            = np.copy(fn1_y)
    fn1_z_plot            = np.copy(fn1_z)
    fn2_x_plot            = np.copy(fn2_x)
    fn2_y_plot            = np.copy(fn2_y)
    fn2_z_plot            = np.copy(fn2_z)
    fn3_x_plot            = np.copy(fn3_x)
    fn3_y_plot            = np.copy(fn3_y)
    fn3_z_plot            = np.copy(fn3_z)
    fi1_x_plot            = np.copy(fi1_x)
    fi1_y_plot            = np.copy(fi1_y)
    fi1_z_plot            = np.copy(fi1_z)
    fi2_x_plot            = np.copy(fi2_x)
    fi2_y_plot            = np.copy(fi2_y)
    fi2_z_plot            = np.copy(fi2_z)
    fi3_x_plot            = np.copy(fi3_x)
    fi3_y_plot            = np.copy(fi3_y)
    fi3_z_plot            = np.copy(fi3_z)
    fi4_x_plot            = np.copy(fi4_x)
    fi4_y_plot            = np.copy(fi4_y)
    fi4_z_plot            = np.copy(fi4_z)
    un1_x_plot            = np.copy(un1_x)
    un1_y_plot            = np.copy(un1_y)
    un1_z_plot            = np.copy(un1_z)
    un2_x_plot            = np.copy(un2_x)
    un2_y_plot            = np.copy(un2_y)
    un2_z_plot            = np.copy(un2_z)
    un3_x_plot            = np.copy(un3_x)
    un3_y_plot            = np.copy(un3_y)
    un3_z_plot            = np.copy(un3_z)
    ui1_x_plot            = np.copy(ui1_x)
    ui1_y_plot            = np.copy(ui1_y)
    ui1_z_plot            = np.copy(ui1_z)
    ui2_x_plot            = np.copy(ui2_x)
    ui2_y_plot            = np.copy(ui2_y)
    ui2_z_plot            = np.copy(ui2_z)
    ui3_x_plot            = np.copy(ui3_x)
    ui3_y_plot            = np.copy(ui3_y)
    ui3_z_plot            = np.copy(ui3_z)    
    ui4_x_plot            = np.copy(ui4_x)
    ui4_y_plot            = np.copy(ui4_y)
    ui4_z_plot            = np.copy(ui4_z)  
    ji1_x_plot            = np.copy(ji1_x)
    ji1_y_plot            = np.copy(ji1_y)
    ji1_z_plot            = np.copy(ji1_z)
    ji2_x_plot            = np.copy(ji2_x)
    ji2_y_plot            = np.copy(ji2_y)
    ji2_z_plot            = np.copy(ji2_z)
    ji3_x_plot            = np.copy(ji3_x)
    ji3_y_plot            = np.copy(ji3_y)
    ji3_z_plot            = np.copy(ji3_z)                                    
    ji4_x_plot            = np.copy(ji4_x)
    ji4_y_plot            = np.copy(ji4_y)
    ji4_z_plot            = np.copy(ji4_z)    
    je_r_plot             = np.copy(je_r)
    je_t_plot             = np.copy(je_t)
    je_z_plot             = np.copy(je_z)
    je_perp_plot          = np.copy(je_perp)
    je_para_plot          = np.copy(je_para)
    ue_r_plot             = np.copy(ue_r)
    ue_t_plot             = np.copy(ue_t)
    ue_z_plot             = np.copy(ue_z)
    ue_perp_plot          = np.copy(ue_perp)
    ue_para_plot          = np.copy(ue_para)
    uthetaExB_plot        = np.copy(uthetaExB)
    Tn1_plot              = np.copy(Tn1)
    Tn2_plot              = np.copy(Tn2)
    Tn3_plot              = np.copy(Tn3)
    Ti1_plot              = np.copy(Ti1)
    Ti2_plot              = np.copy(Ti2)
    Ti3_plot              = np.copy(Ti3)
    Ti4_plot              = np.copy(Ti4)
    Te_plot               = np.copy(Te)
    n_mp_n1_plot          = np.copy(n_mp_n1)
    n_mp_n2_plot          = np.copy(n_mp_n2)
    n_mp_n3_plot          = np.copy(n_mp_n3)
    n_mp_i1_plot          = np.copy(n_mp_i1)
    n_mp_i2_plot          = np.copy(n_mp_i2)
    n_mp_i3_plot          = np.copy(n_mp_i3)
    n_mp_i4_plot          = np.copy(n_mp_i4)
    avg_w_n1_plot         = np.copy(avg_w_n1)
    avg_w_n2_plot         = np.copy(avg_w_n2)
    avg_w_i1_plot         = np.copy(avg_w_i1)
    avg_w_i2_plot         = np.copy(avg_w_i2)
    neu_gen_weights1_plot = np.copy(neu_gen_weights1)           
    neu_gen_weights2_plot = np.copy(neu_gen_weights2)
    ion_gen_weights1_plot = np.copy(ion_gen_weights1)
    ion_gen_weights2_plot = np.copy(ion_gen_weights2)
    ndot_ion01_n1_plot    = np.copy(ndot_ion01_n1)
    ndot_ion02_n1_plot    = np.copy(ndot_ion02_n1)
    ndot_ion12_i1_plot    = np.copy(ndot_ion12_i1)
    ndot_ion01_n2_plot    = np.copy(ndot_ion01_n2)
    ndot_ion02_n2_plot    = np.copy(ndot_ion02_n2)
    ndot_ion01_n3_plot    = np.copy(ndot_ion01_n3)
    ndot_ion02_n3_plot    = np.copy(ndot_ion02_n3)
    ndot_ion12_i3_plot    = np.copy(ndot_ion12_i3)
    ndot_CEX01_i3_plot    = np.copy(ndot_CEX01_i3)
    ndot_CEX02_i4_plot    = np.copy(ndot_CEX02_i4)
    ne_cath_plot          = np.copy(ne_cath)
    nu_cath_plot          = np.copy(nu_cath)
    ndot_cath_plot        = np.copy(ndot_cath)
    F_theta_plot          = np.copy(F_theta)
    Hall_par_plot         = np.copy(Hall_par)
    Hall_par_eff_plot     = np.copy(Hall_par_eff)
    nu_e_tot_plot         = np.copy(nu_e_tot)
    nu_e_tot_eff_plot     = np.copy(nu_e_tot_eff)
    nu_en_plot            = np.copy(nu_en)
    nu_ei1_plot           = np.copy(nu_ei1)
    nu_ei2_plot           = np.copy(nu_ei2)
    nu_i01_plot           = np.copy(nu_i01)
    nu_i02_plot           = np.copy(nu_i02)
    nu_i12_plot           = np.copy(nu_i12)
    nu_ex_plot            = np.copy(nu_ex)
    err_interp_n_plot     = np.copy(err_interp_n) 
    
    f_split_adv_plot      = np.copy(f_split_adv) 
    f_split_qperp_plot    = np.copy(f_split_qperp) 
    f_split_qpara_plot    = np.copy(f_split_qpara) 
    f_split_qb_plot       =  np.copy(f_split_qb)  
    f_split_Pperp_plot    = np.copy(f_split_Pperp) 
    f_split_Ppara_plot    = np.copy(f_split_Ppara) 
    f_split_ecterm_plot   = np.copy(f_split_ecterm) 
    f_split_inel_plot     = np.copy(f_split_inel) 
    
    
    return[Br_plot,Bz_plot,Bfield_plot,phi_plot,Er_plot,Ez_plot,Efield_plot,
           nn1_plot,nn2_plot,nn3_plot,ni1_plot,ni2_plot,ni3_plot,ni4_plot,
           ne_plot,fn1_x_plot,fn1_y_plot,fn1_z_plot,fn2_x_plot,fn2_y_plot,
           fn2_z_plot,fn3_x_plot,fn3_y_plot,fn3_z_plot,fi1_x_plot,fi1_y_plot,
           fi1_z_plot,fi2_x_plot,fi2_y_plot,fi2_z_plot,fi3_x_plot,fi3_y_plot,
           fi3_z_plot,fi4_x_plot,fi4_y_plot,fi4_z_plot,un1_x_plot,un1_y_plot,
           un1_z_plot,un2_x_plot,un2_y_plot,un2_z_plot,un3_x_plot,un3_y_plot,
           un3_z_plot,ui1_x_plot,ui1_y_plot,ui1_z_plot,ui2_x_plot,ui2_y_plot,
           ui2_z_plot,ui3_x_plot,ui3_y_plot,ui3_z_plot,ui4_x_plot,ui4_y_plot,
           ui4_z_plot,ji1_x_plot,ji1_y_plot,ji1_z_plot,ji2_x_plot,ji2_y_plot,
           ji2_z_plot,ji3_x_plot,ji3_y_plot,ji3_z_plot,ji4_x_plot,ji4_y_plot,
           ji4_z_plot,je_r_plot,je_t_plot,je_z_plot,je_perp_plot,je_para_plot,
           ue_r_plot,ue_t_plot,ue_z_plot,ue_perp_plot,ue_para_plot,uthetaExB_plot,
           Tn1_plot,Tn2_plot,Tn3_plot,Ti1_plot,Ti2_plot,Ti3_plot,Ti4_plot,
           Te_plot,n_mp_n1_plot,n_mp_n2_plot,n_mp_n3_plot,
           n_mp_i1_plot,n_mp_i2_plot,n_mp_i3_plot,n_mp_i4_plot,
           avg_w_n1_plot,avg_w_n2_plot,avg_w_i1_plot,
           avg_w_i2_plot,neu_gen_weights1_plot,neu_gen_weights2_plot,
           ion_gen_weights1_plot,ion_gen_weights2_plot,ndot_ion01_n1_plot,
           ndot_ion02_n1_plot,ndot_ion12_i1_plot,ndot_ion01_n2_plot,
           ndot_ion02_n2_plot,ndot_ion01_n3_plot,ndot_ion02_n3_plot,
           ndot_ion12_i3_plot,ndot_CEX01_i3_plot,ndot_CEX02_i4_plot,ne_cath_plot,
           nu_cath_plot,ndot_cath_plot,
           F_theta_plot,Hall_par_plot,Hall_par_eff_plot,nu_e_tot_plot,
           nu_e_tot_eff_plot,nu_en_plot,nu_ei1_plot,nu_ei2_plot,nu_i01_plot,
           nu_i02_plot,nu_i12_plot,nu_ex_plot,err_interp_n_plot,f_split_adv_plot,
           f_split_qperp_plot,f_split_qpara_plot,f_split_qb_plot,
           f_split_Pperp_plot,f_split_Ppara_plot,f_split_ecterm_plot,
           f_split_inel_plot]
    
    
"""
############################################################################
Description:    This python fucntion return the given boundary variables as 
                a copy with the plotting name
############################################################################
Inputs:        1) Input variables to be copied
############################################################################
Output:        1) Plot variables
    
"""

def HET_sims_cp_vars_bound(delta_r,delta_s,delta_s_csl,dphi_sh_b,je_b,ji_tot_b,
                           gp_net_b,ge_sb_b,relerr_je_b,qe_tot_wall,qe_tot_s_wall,
                           qe_tot_b,qe_b,qe_b_bc,qe_b_fl,imp_ene_e_wall,imp_ene_e_b,
                           relerr_qe_b,relerr_qe_b_cons,Te,phi,err_interp_phi,
                           err_interp_Te,err_interp_jeperp,err_interp_jetheta,
                           err_interp_jepara,err_interp_jez,err_interp_jer,
                           n_inst,ni1_inst,ni2_inst,nn1_inst,inst_dphi_sh_b_Te,
                           inst_imp_ene_e_b,inst_imp_ene_e_b_Te,
                           inst_imp_ene_e_wall,inst_imp_ene_e_wall_Te,
                           
                           delta_r_nodes,delta_s_nodes,delta_s_csl_nodes,
                           dphi_sh_b_nodes,je_b_nodes,gp_net_b_nodes,ge_sb_b_nodes,
                           relerr_je_b_nodes,qe_tot_wall_nodes,qe_tot_s_wall_nodes,
                           qe_tot_b_nodes,qe_b_nodes,qe_b_bc_nodes,qe_b_fl_nodes,
                           imp_ene_e_wall_nodes,imp_ene_e_b_nodes,
                           relerr_qe_b_nodes,relerr_qe_b_cons_nodes,Te_nodes,
                           phi_nodes,err_interp_n_nodes,n_inst_nodes,ni1_inst_nodes,
                           ni2_inst_nodes,nn1_inst_nodes,n_nodes,ni1_nodes,ni2_nodes,
                           nn1_nodes,dphi_kbc_nodes,MkQ1_nodes,ji1_nodes,ji2_nodes,
                           ji3_nodes,ji4_nodes,ji_nodes,gn1_tw_nodes,gn1_fw_nodes,
                           gn2_tw_nodes,gn2_fw_nodes,gn3_tw_nodes,gn3_fw_nodes,
                           gn_tw_nodes,qi1_tot_wall_nodes,qi2_tot_wall_nodes,
                           qi3_tot_wall_nodes,qi4_tot_wall_nodes,qi_tot_wall_nodes,
                           qn1_tw_nodes,qn1_fw_nodes,qn2_tw_nodes,qn2_fw_nodes,
                           qn3_tw_nodes,qn3_fw_nodes,qn_tot_wall_nodes,
                           imp_ene_i1_nodes,imp_ene_i2_nodes,imp_ene_i3_nodes,
                           imp_ene_i4_nodes,imp_ene_ion_nodes,imp_ene_ion_nodes_v2,
                           imp_ene_n1_nodes,imp_ene_n2_nodes,imp_ene_n3_nodes,
                           imp_ene_n_nodes,imp_ene_n_nodes_v2,
                           inst_dphi_sh_b_Te_nodes,inst_imp_ene_e_b_nodes,
                           inst_imp_ene_e_b_Te_nodes,inst_imp_ene_e_wall_nodes,
                           inst_imp_ene_e_wall_Te_nodes,
                           
                           delta_r_surf,delta_s_surf,delta_s_csl_surf,dphi_sh_b_surf,
                           je_b_surf,gp_net_b_surf,ge_sb_b_surf,relerr_je_b_surf,
                           qe_tot_wall_surf,qe_tot_s_wall_surf,qe_tot_b_surf,
                           qe_b_surf,qe_b_bc_surf,qe_b_fl_surf,imp_ene_e_wall_surf,
                           imp_ene_e_b_surf,relerr_qe_b_surf,relerr_qe_b_cons_surf,
                           Te_surf,phi_surf,nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,
                           nQ2_surf,dphi_kbc_surf,MkQ1_surf,ji1_surf,ji2_surf,
                           ji3_surf,ji4_surf,ji_surf,gn1_tw_surf,gn1_fw_surf,
                           gn2_tw_surf,gn2_fw_surf,gn3_tw_surf,gn3_fw_surf,
                           gn_tw_surf,qi1_tot_wall_surf,qi2_tot_wall_surf,
                           qi3_tot_wall_surf,qi4_tot_wall_surf,qi_tot_wall_surf,
                           qn1_tw_surf,qn1_fw_surf,qn2_tw_surf,qn2_fw_surf,
                           qn3_tw_surf,qn3_fw_surf,qn_tot_wall_surf,
                           imp_ene_i1_surf,imp_ene_i2_surf,imp_ene_i3_surf,
                           imp_ene_i4_surf,imp_ene_ion_surf,imp_ene_ion_surf_v2,
                           imp_ene_n1_surf,imp_ene_n2_surf,imp_ene_n3_surf,
                           imp_ene_n_surf,imp_ene_n_surf_v2,
                           inst_dphi_sh_b_Te_surf,inst_imp_ene_e_b_surf,
                           inst_imp_ene_e_b_Te_surf,inst_imp_ene_e_wall_surf,
                           inst_imp_ene_e_wall_Te_surf):

    
    
    import numpy as np
        
    delta_r_plot                = np.copy(delta_r)
    delta_s_plot                = np.copy(delta_s)
    delta_s_csl_plot            = np.copy(delta_s_csl)
    dphi_sh_b_plot              = np.copy(dphi_sh_b)
    je_b_plot                   = np.copy(je_b)
    ji_tot_b_plot               = np.copy(ji_tot_b)
    gp_net_b_plot               = np.copy(gp_net_b)
    ge_sb_b_plot                = np.copy(ge_sb_b)
    relerr_je_b_plot            = np.copy(relerr_je_b)
    qe_tot_wall_plot            = np.copy(qe_tot_wall)
    qe_tot_s_wall_plot          = np.copy(qe_tot_s_wall)
    qe_tot_b_plot               = np.copy(qe_tot_b)
    qe_b_plot                   = np.copy(qe_b)
    qe_b_bc_plot                = np.copy(qe_b_bc)
    qe_b_fl_plot                = np.copy(qe_b_fl)
    imp_ene_e_wall_plot         = np.copy(imp_ene_e_wall)
    imp_ene_e_b_plot            = np.copy(imp_ene_e_b)
    relerr_qe_b_plot            = np.copy(relerr_qe_b)
    relerr_qe_b_cons_plot       = np.copy(relerr_qe_b_cons)
    Te_plot                     = np.copy(Te)
    phi_plot                    = np.copy(phi)
    err_interp_phi_plot         = np.copy(err_interp_phi)
    err_interp_Te_plot          = np.copy(err_interp_Te)
    err_interp_jeperp_plot      = np.copy(err_interp_jeperp)
    err_interp_jetheta_plot     = np.copy(err_interp_jetheta)
    err_interp_jepara_plot      = np.copy(err_interp_jepara)
    err_interp_jez_plot         = np.copy(err_interp_jez)
    err_interp_jer_plot         = np.copy(err_interp_jer)
    n_inst_plot                 = np.copy(n_inst)
    ni1_inst_plot               = np.copy(ni1_inst)
    ni2_inst_plot               = np.copy(ni2_inst)
    nn1_inst_plot               = np.copy(nn1_inst)
    inst_dphi_sh_b_Te_plot      = np.copy(inst_dphi_sh_b_Te)
    inst_imp_ene_e_b_plot       = np.copy(inst_imp_ene_e_b)
    inst_imp_ene_e_b_Te_plot    = np.copy(inst_imp_ene_e_b_Te)
    inst_imp_ene_e_wall_plot    = np.copy(inst_imp_ene_e_wall)
    inst_imp_ene_e_wall_Te_plot = np.copy(inst_imp_ene_e_wall_Te)
    
    delta_r_nodes_plot                = np.copy(delta_r_nodes)
    delta_s_nodes_plot                = np.copy(delta_s_nodes)
    delta_s_csl_nodes_plot            = np.copy(delta_s_csl_nodes)
    dphi_sh_b_nodes_plot              = np.copy(dphi_sh_b_nodes)
    je_b_nodes_plot                   = np.copy(je_b_nodes)
    gp_net_b_nodes_plot               = np.copy(gp_net_b_nodes)
    ge_sb_b_nodes_plot                = np.copy(ge_sb_b_nodes)
    relerr_je_b_nodes_plot            = np.copy(relerr_je_b_nodes)
    qe_tot_wall_nodes_plot            = np.copy(qe_tot_wall_nodes)
    qe_tot_s_wall_nodes_plot          = np.copy(qe_tot_s_wall_nodes)
    qe_tot_b_nodes_plot               = np.copy(qe_tot_b_nodes)
    qe_b_nodes_plot                   = np.copy(qe_b_nodes)
    qe_b_bc_nodes_plot                = np.copy(qe_b_bc_nodes)
    qe_b_fl_nodes_plot                = np.copy(qe_b_fl_nodes)
    imp_ene_e_wall_nodes_plot         = np.copy(imp_ene_e_wall_nodes)
    imp_ene_e_b_nodes_plot            = np.copy(imp_ene_e_b_nodes)
    relerr_qe_b_nodes_plot            = np.copy(relerr_qe_b_nodes)
    relerr_qe_b_cons_nodes_plot       = np.copy(relerr_qe_b_cons_nodes)
    Te_nodes_plot                     = np.copy(Te_nodes)
    phi_nodes_plot                    = np.copy(phi_nodes)
    err_interp_n_nodes_plot           = np.copy(err_interp_n_nodes)
    n_inst_nodes_plot                 = np.copy(n_inst_nodes)
    ni1_inst_nodes_plot               = np.copy(ni1_inst_nodes)
    ni2_inst_nodes_plot               = np.copy(ni2_inst_nodes)
    nn1_inst_nodes_plot               = np.copy(nn1_inst_nodes)
    n_nodes_plot                      = np.copy(n_nodes)
    ni1_nodes_plot                    = np.copy(ni1_nodes)
    ni2_nodes_plot                    = np.copy(ni2_nodes)
    nn1_nodes_plot                    = np.copy(nn1_nodes)
    dphi_kbc_nodes_plot               = np.copy(dphi_kbc_nodes)
    MkQ1_nodes_plot                   = np.copy(MkQ1_nodes)
    ji1_nodes_plot                    = np.copy(ji1_nodes)
    ji2_nodes_plot                    = np.copy(ji2_nodes)
    ji3_nodes_plot                    = np.copy(ji3_nodes)
    ji4_nodes_plot                    = np.copy(ji4_nodes)
    ji_nodes_plot                     = np.copy(ji_nodes)
    gn1_tw_nodes_plot                 = np.copy(gn1_tw_nodes)
    gn1_fw_nodes_plot                 = np.copy(gn1_fw_nodes)
    gn2_tw_nodes_plot                 = np.copy(gn2_tw_nodes)
    gn2_fw_nodes_plot                 = np.copy(gn2_fw_nodes)
    gn3_tw_nodes_plot                 = np.copy(gn3_tw_nodes)
    gn3_fw_nodes_plot                 = np.copy(gn3_fw_nodes)
    gn_tw_nodes_plot                  = np.copy(gn_tw_nodes)
    qi1_tot_wall_nodes_plot           = np.copy(qi1_tot_wall_nodes)
    qi2_tot_wall_nodes_plot           = np.copy(qi2_tot_wall_nodes)
    qi3_tot_wall_nodes_plot           = np.copy(qi3_tot_wall_nodes)
    qi4_tot_wall_nodes_plot           = np.copy(qi4_tot_wall_nodes)
    qi_tot_wall_nodes_plot            = np.copy(qi_tot_wall_nodes)
    qn1_tw_nodes_plot                 = np.copy(qn1_tw_nodes)
    qn1_fw_nodes_plot                 = np.copy(qn1_fw_nodes)
    qn2_tw_nodes_plot                 = np.copy(qn2_tw_nodes)
    qn2_fw_nodes_plot                 = np.copy(qn2_fw_nodes)
    qn3_tw_nodes_plot                 = np.copy(qn3_tw_nodes)
    qn3_fw_nodes_plot                 = np.copy(qn3_fw_nodes)
    qn_tot_wall_nodes_plot            = np.copy(qn_tot_wall_nodes)
    imp_ene_i1_nodes_plot             = np.copy(imp_ene_i1_nodes)
    imp_ene_i2_nodes_plot             = np.copy(imp_ene_i2_nodes)
    imp_ene_i3_nodes_plot             = np.copy(imp_ene_i3_nodes)
    imp_ene_i4_nodes_plot             = np.copy(imp_ene_i4_nodes)
    imp_ene_ion_nodes_plot            = np.copy(imp_ene_ion_nodes)
    imp_ene_ion_nodes_v2_plot         = np.copy(imp_ene_ion_nodes_v2)
    imp_ene_n1_nodes_plot             = np.copy(imp_ene_n1_nodes)
    imp_ene_n2_nodes_plot             = np.copy(imp_ene_n2_nodes)
    imp_ene_n3_nodes_plot             = np.copy(imp_ene_n3_nodes)
    imp_ene_n_nodes_plot              = np.copy(imp_ene_n_nodes)
    imp_ene_n_nodes_v2_plot           = np.copy(imp_ene_n_nodes_v2)
    inst_dphi_sh_b_Te_nodes_plot      = np.copy(inst_dphi_sh_b_Te_nodes)
    inst_imp_ene_e_b_nodes_plot       = np.copy(inst_imp_ene_e_b_nodes)
    inst_imp_ene_e_b_Te_nodes_plot    = np.copy(inst_imp_ene_e_b_Te_nodes)
    inst_imp_ene_e_wall_nodes_plot    = np.copy(inst_imp_ene_e_wall_nodes)
    inst_imp_ene_e_wall_Te_nodes_plot = np.copy(inst_imp_ene_e_wall_Te_nodes)
    
    delta_r_surf_plot                = np.copy(delta_r_surf)
    delta_s_surf_plot                = np.copy(delta_s_surf)
    delta_s_csl_surf_plot            = np.copy(delta_s_csl_surf)
    dphi_sh_b_surf_plot              = np.copy(dphi_sh_b_surf)
    je_b_surf_plot                   = np.copy(je_b_surf)
    gp_net_b_surf_plot               = np.copy(gp_net_b_surf)
    ge_sb_b_surf_plot                = np.copy(ge_sb_b_surf)
    relerr_je_b_surf_plot            = np.copy(relerr_je_b_surf)
    qe_tot_wall_surf_plot            = np.copy(qe_tot_wall_surf)
    qe_tot_s_wall_surf_plot          = np.copy(qe_tot_s_wall_surf)
    qe_tot_b_surf_plot               = np.copy(qe_tot_b_surf)
    qe_b_surf_plot                   = np.copy(qe_b_surf)
    qe_b_bc_surf_plot                = np.copy(qe_b_bc_surf)
    qe_b_fl_surf_plot                = np.copy(qe_b_fl_surf)
    imp_ene_e_wall_surf_plot         = np.copy(imp_ene_e_wall_surf)
    imp_ene_e_b_surf_plot            = np.copy(imp_ene_e_b_surf)
    relerr_qe_b_surf_plot            = np.copy(relerr_qe_b_surf)
    relerr_qe_b_cons_surf_plot       = np.copy(relerr_qe_b_cons_surf)
    Te_surf_plot                     = np.copy(Te_surf)
    phi_surf_plot                    = np.copy(phi_surf)
    nQ1_inst_surf_plot               = np.copy(nQ1_inst_surf)
    nQ1_surf_plot                    = np.copy(nQ1_surf)
    nQ2_inst_surf_plot               = np.copy(nQ2_inst_surf)
    nQ2_surf_plot                    = np.copy(nQ2_surf)
    dphi_kbc_surf_plot               = np.copy(dphi_kbc_surf)
    MkQ1_surf_plot                   = np.copy(MkQ1_surf)
    ji1_surf_plot                    = np.copy(ji1_surf)
    ji2_surf_plot                    = np.copy(ji2_surf)
    ji3_surf_plot                    = np.copy(ji3_surf)
    ji4_surf_plot                    = np.copy(ji4_surf)
    ji_surf_plot                     = np.copy(ji_surf)
    gn1_tw_surf_plot                 = np.copy(gn1_tw_surf)
    gn1_fw_surf_plot                 = np.copy(gn1_fw_surf)
    gn2_tw_surf_plot                 = np.copy(gn2_tw_surf)
    gn2_fw_surf_plot                 = np.copy(gn2_fw_surf)    
    gn3_tw_surf_plot                 = np.copy(gn3_tw_surf)
    gn3_fw_surf_plot                 = np.copy(gn3_fw_surf)
    gn_tw_surf_plot                  = np.copy(gn_tw_surf)
    qi1_tot_wall_surf_plot           = np.copy(qi1_tot_wall_surf)
    qi2_tot_wall_surf_plot           = np.copy(qi2_tot_wall_surf)
    qi3_tot_wall_surf_plot           = np.copy(qi3_tot_wall_surf)
    qi4_tot_wall_surf_plot           = np.copy(qi4_tot_wall_surf)
    qi_tot_wall_surf_plot            = np.copy(qi_tot_wall_surf)
    qn1_tw_surf_plot                 = np.copy(qn1_tw_surf)
    qn1_fw_surf_plot                 = np.copy(qn1_fw_surf)
    qn2_tw_surf_plot                 = np.copy(qn2_tw_surf)
    qn2_fw_surf_plot                 = np.copy(qn2_fw_surf)
    qn3_tw_surf_plot                 = np.copy(qn3_tw_surf)
    qn3_fw_surf_plot                 = np.copy(qn3_fw_surf)
    qn_tot_wall_surf_plot            = np.copy(qn_tot_wall_surf)
    imp_ene_i1_surf_plot             = np.copy(imp_ene_i1_surf)
    imp_ene_i2_surf_plot             = np.copy(imp_ene_i2_surf)
    imp_ene_i3_surf_plot             = np.copy(imp_ene_i3_surf)
    imp_ene_i4_surf_plot             = np.copy(imp_ene_i4_surf)
    imp_ene_ion_surf_plot            = np.copy(imp_ene_ion_surf)
    imp_ene_ion_surf_v2_plot         = np.copy(imp_ene_ion_surf_v2)
    imp_ene_n1_surf_plot             = np.copy(imp_ene_n1_surf)
    imp_ene_n2_surf_plot             = np.copy(imp_ene_n2_surf)
    imp_ene_n3_surf_plot             = np.copy(imp_ene_n3_surf)
    imp_ene_n_surf_plot              = np.copy(imp_ene_n_surf)
    imp_ene_n_surf_v2_plot           = np.copy(imp_ene_n_surf_v2)
    inst_dphi_sh_b_Te_surf_plot      = np.copy(inst_dphi_sh_b_Te_surf)
    inst_imp_ene_e_b_surf_plot       = np.copy(inst_imp_ene_e_b_surf)
    inst_imp_ene_e_b_Te_surf_plot    = np.copy(inst_imp_ene_e_b_Te_surf)
    inst_imp_ene_e_wall_surf_plot    = np.copy(inst_imp_ene_e_wall_surf)
    inst_imp_ene_e_wall_Te_surf_plot = np.copy(inst_imp_ene_e_wall_Te_surf)
    
    

    return[delta_r_plot,delta_s_plot,delta_s_csl_plot,dphi_sh_b_plot,je_b_plot,
           ji_tot_b_plot,gp_net_b_plot,ge_sb_b_plot,relerr_je_b_plot,qe_tot_wall_plot,
           qe_tot_s_wall_plot,qe_tot_b_plot,qe_b_plot,qe_b_bc_plot,
           qe_b_fl_plot,imp_ene_e_wall_plot,imp_ene_e_b_plot,
           relerr_qe_b_plot,relerr_qe_b_cons_plot,Te_plot,
           phi_plot,err_interp_phi_plot,err_interp_Te_plot,
           err_interp_jeperp_plot,err_interp_jetheta_plot,
           err_interp_jepara_plot,err_interp_jez_plot,err_interp_jer_plot,
           n_inst_plot,ni1_inst_plot,ni2_inst_plot,nn1_inst_plot,
           inst_dphi_sh_b_Te_plot,inst_imp_ene_e_b_plot,inst_imp_ene_e_b_Te_plot,
           inst_imp_ene_e_wall_plot,inst_imp_ene_e_wall_Te_plot,
           
           delta_r_nodes_plot,delta_s_nodes_plot,delta_s_csl_nodes_plot,
           dphi_sh_b_nodes_plot,je_b_nodes_plot,gp_net_b_nodes_plot,
           ge_sb_b_nodes_plot,relerr_je_b_nodes_plot,qe_tot_wall_nodes_plot,
           qe_tot_s_wall_nodes_plot,qe_tot_b_nodes_plot,qe_b_nodes_plot,
           qe_b_bc_nodes_plot,qe_b_fl_nodes_plot,imp_ene_e_wall_nodes_plot,
           imp_ene_e_b_nodes_plot,relerr_qe_b_nodes_plot,
           relerr_qe_b_cons_nodes_plot,Te_nodes_plot,phi_nodes_plot,
           err_interp_n_nodes_plot,n_inst_nodes_plot,ni1_inst_nodes_plot,
           ni2_inst_nodes_plot,nn1_inst_nodes_plot,n_nodes_plot,
           ni1_nodes_plot,ni2_nodes_plot,nn1_nodes_plot,dphi_kbc_nodes_plot,
           MkQ1_nodes_plot,ji1_nodes_plot,ji2_nodes_plot,ji3_nodes_plot,
           ji4_nodes_plot,ji_nodes_plot,gn1_tw_nodes_plot,gn1_fw_nodes_plot,
           gn2_tw_nodes_plot,gn2_fw_nodes_plot,gn3_tw_nodes_plot,
           gn3_fw_nodes_plot,gn_tw_nodes_plot,qi1_tot_wall_nodes_plot,
           qi2_tot_wall_nodes_plot,qi3_tot_wall_nodes_plot,qi4_tot_wall_nodes_plot,
           qi_tot_wall_nodes_plot,qn1_tw_nodes_plot,qn1_fw_nodes_plot,
           qn2_tw_nodes_plot,qn2_fw_nodes_plot,qn3_tw_nodes_plot,
           qn3_fw_nodes_plot,qn_tot_wall_nodes_plot,imp_ene_i1_nodes_plot,
           imp_ene_i2_nodes_plot,imp_ene_i3_nodes_plot,imp_ene_i4_nodes_plot,
           imp_ene_ion_nodes_plot,imp_ene_ion_nodes_v2_plot,
           imp_ene_n1_nodes_plot,imp_ene_n2_nodes_plot,imp_ene_n3_nodes_plot,
           imp_ene_n_nodes_plot,imp_ene_n_nodes_v2_plot,
           inst_dphi_sh_b_Te_nodes_plot,inst_imp_ene_e_b_nodes_plot,
           inst_imp_ene_e_b_Te_nodes_plot,inst_imp_ene_e_wall_nodes_plot,
           inst_imp_ene_e_wall_Te_nodes_plot,
           
           delta_r_surf_plot,delta_s_surf_plot,delta_s_csl_surf_plot,
           dphi_sh_b_surf_plot,je_b_surf_plot,gp_net_b_surf_plot,ge_sb_b_surf_plot,
           relerr_je_b_surf_plot,qe_tot_wall_surf_plot,qe_tot_s_wall_surf_plot,
           qe_tot_b_surf_plot,qe_b_surf_plot,qe_b_bc_surf_plot,qe_b_fl_surf_plot,
           imp_ene_e_wall_surf_plot,imp_ene_e_b_surf_plot,
           relerr_qe_b_surf_plot,relerr_qe_b_cons_surf_plot,Te_surf_plot,
           phi_surf_plot,nQ1_inst_surf_plot,nQ1_surf_plot,nQ2_inst_surf_plot,
           nQ2_surf_plot,dphi_kbc_surf_plot,MkQ1_surf_plot,ji1_surf_plot,
           ji2_surf_plot,ji3_surf_plot,ji4_surf_plot,ji_surf_plot,gn1_tw_surf_plot,
           gn1_fw_surf_plot,gn2_tw_surf_plot,gn2_fw_surf_plot,gn3_tw_surf_plot,
           gn3_fw_surf_plot,gn_tw_surf_plot,qi1_tot_wall_surf_plot,
           qi2_tot_wall_surf_plot,qi3_tot_wall_surf_plot,qi4_tot_wall_surf_plot,
           qi_tot_wall_surf_plot,qn1_tw_surf_plot,qn1_fw_surf_plot,
           qn2_tw_surf_plot,qn2_fw_surf_plot,qn3_tw_surf_plot,qn3_fw_surf_plot,
           qn_tot_wall_surf_plot,imp_ene_i1_surf_plot,imp_ene_i2_surf_plot,
           imp_ene_i3_surf_plot,imp_ene_i4_surf_plot,imp_ene_ion_surf_plot,
           imp_ene_ion_surf_v2_plot,imp_ene_n1_surf_plot,imp_ene_n2_surf_plot,
           imp_ene_n3_surf_plot,imp_ene_n_surf_plot,imp_ene_n_surf_v2_plot,
           inst_dphi_sh_b_Te_surf_plot,inst_imp_ene_e_b_surf_plot,
           inst_imp_ene_e_b_Te_surf_plot,inst_imp_ene_e_wall_surf_plot,
           inst_imp_ene_e_wall_Te_surf_plot]

    

"""
############################################################################
Description:    This python fucntion return the given boundary and distribution
                function variables as a copy with the plotting name
############################################################################
Inputs:        1) Input variables to be copied
############################################################################
Output:        1) Plot variables
    
"""

def HET_sims_cp_vars_df(nQ1_inst_surf,nQ1_surf,nQ2_inst_surf,nQ2_surf,dphi_kbc_surf,
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
        

    nQ1_inst_surf_plot     = np.copy(nQ1_inst_surf)
    nQ1_surf_plot          = np.copy(nQ1_surf)
    nQ2_inst_surf_plot     = np.copy(nQ2_inst_surf)
    nQ2_surf_plot          = np.copy(nQ2_surf)
    dphi_kbc_surf_plot     = np.copy(dphi_kbc_surf)
    MkQ1_surf_plot         = np.copy(MkQ1_surf)
    ji1_surf_plot          = np.copy(ji1_surf)
    ji2_surf_plot          = np.copy(ji2_surf)
    ji3_surf_plot          = np.copy(ji3_surf)
    ji4_surf_plot          = np.copy(ji4_surf)
    ji_surf_plot           = np.copy(ji_surf)
    gn1_tw_surf_plot       = np.copy(gn1_tw_surf)
    gn1_fw_surf_plot       = np.copy(gn1_fw_surf)
    gn2_tw_surf_plot       = np.copy(gn2_tw_surf)
    gn2_fw_surf_plot       = np.copy(gn2_fw_surf)    
    gn3_tw_surf_plot       = np.copy(gn3_tw_surf)
    gn3_fw_surf_plot       = np.copy(gn3_fw_surf)
    gn_tw_surf_plot        = np.copy(gn_tw_surf)
    qi1_tot_wall_surf_plot = np.copy(qi1_tot_wall_surf)
    qi2_tot_wall_surf_plot = np.copy(qi2_tot_wall_surf)
    qi3_tot_wall_surf_plot = np.copy(qi3_tot_wall_surf)
    qi4_tot_wall_surf_plot = np.copy(qi4_tot_wall_surf)
    qi_tot_wall_surf_plot  = np.copy(qi_tot_wall_surf)
    qn1_tw_surf_plot       = np.copy(qn1_tw_surf)
    qn1_fw_surf_plot       = np.copy(qn1_fw_surf)
    qn2_tw_surf_plot       = np.copy(qn2_tw_surf)
    qn2_fw_surf_plot       = np.copy(qn2_fw_surf)    
    qn3_tw_surf_plot       = np.copy(qn3_tw_surf)
    qn3_fw_surf_plot       = np.copy(qn3_fw_surf)
    qn_tot_wall_surf_plot  = np.copy(qn_tot_wall_surf)
    imp_ene_i1_surf_plot   = np.copy(imp_ene_i1_surf)
    imp_ene_i2_surf_plot   = np.copy(imp_ene_i2_surf)
    imp_ene_i3_surf_plot   = np.copy(imp_ene_i3_surf)
    imp_ene_i4_surf_plot   = np.copy(imp_ene_i4_surf)
    imp_ene_n1_surf_plot   = np.copy(imp_ene_n1_surf)
    imp_ene_n2_surf_plot   = np.copy(imp_ene_n2_surf)
    imp_ene_n3_surf_plot   = np.copy(imp_ene_n3_surf)
    
    angle_df_i1_plot       = np.copy(angle_df_i1)
    ene_df_i1_plot         = np.copy(ene_df_i1)
    normv_df_i1_plot       = np.copy(normv_df_i1)
    ene_angle_df_i1_plot   = np.copy(ene_angle_df_i1)
    angle_df_i2_plot       = np.copy(angle_df_i2)
    ene_df_i2_plot         = np.copy(ene_df_i2)
    normv_df_i2_plot       = np.copy(normv_df_i2)
    ene_angle_df_i2_plot   = np.copy(ene_angle_df_i2)
    angle_df_i3_plot       = np.copy(angle_df_i3)
    ene_df_i3_plot         = np.copy(ene_df_i3)
    normv_df_i3_plot       = np.copy(normv_df_i3)
    ene_angle_df_i3_plot   = np.copy(ene_angle_df_i3)
    angle_df_i4_plot       = np.copy(angle_df_i4)
    ene_df_i4_plot         = np.copy(ene_df_i4)
    normv_df_i4_plot       = np.copy(normv_df_i4)
    ene_angle_df_i4_plot   = np.copy(ene_angle_df_i4)
    
    angle_df_n1_plot       = np.copy(angle_df_n1)
    ene_df_n1_plot         = np.copy(ene_df_n1)
    normv_df_n1_plot       = np.copy(normv_df_n1)
    ene_angle_df_n1_plot   = np.copy(ene_angle_df_n1)
    angle_df_n2_plot       = np.copy(angle_df_n2)
    ene_df_n2_plot         = np.copy(ene_df_n2)
    normv_df_n2_plot       = np.copy(normv_df_n2)
    ene_angle_df_n2_plot   = np.copy(ene_angle_df_n2)
    angle_df_n3_plot       = np.copy(angle_df_n3)
    ene_df_n3_plot         = np.copy(ene_df_n3)
    normv_df_n3_plot       = np.copy(normv_df_n3)
    ene_angle_df_n3_plot   = np.copy(ene_angle_df_n3)
    
    

    return[nQ1_inst_surf_plot,nQ1_surf_plot,nQ2_inst_surf_plot,
           nQ2_surf_plot,dphi_kbc_surf_plot,MkQ1_surf_plot,ji1_surf_plot,
           ji2_surf_plot,ji3_surf_plot,ji4_surf_plot,ji_surf_plot,
           gn1_tw_surf_plot,gn1_fw_surf_plot,gn2_tw_surf_plot,gn2_fw_surf_plot,
           gn3_tw_surf_plot,gn3_fw_surf_plot,gn_tw_surf_plot,
           
           qi1_tot_wall_surf_plot,qi2_tot_wall_surf_plot,qi3_tot_wall_surf_plot,
           qi4_tot_wall_surf_plot,qi_tot_wall_surf_plot,
           qn1_tw_surf_plot,qn1_fw_surf_plot,qn2_tw_surf_plot,qn2_fw_surf_plot,
           qn3_tw_surf_plot,qn3_fw_surf_plot,qn_tot_wall_surf_plot,
           
           imp_ene_i1_surf_plot,imp_ene_i2_surf_plot,imp_ene_i3_surf_plot,
           imp_ene_i4_surf_plot,imp_ene_n1_surf_plot,imp_ene_n2_surf_plot,
           imp_ene_n3_surf_plot,
           
           angle_df_i1_plot,ene_df_i1_plot,normv_df_i1_plot,ene_angle_df_i1_plot,
           angle_df_i2_plot,ene_df_i2_plot,normv_df_i2_plot,ene_angle_df_i2_plot,
           angle_df_i3_plot,ene_df_i3_plot,normv_df_i3_plot,ene_angle_df_i3_plot,
           angle_df_i4_plot,ene_df_i4_plot,normv_df_i4_plot,ene_angle_df_i4_plot,
           angle_df_n1_plot,ene_df_n1_plot,normv_df_n1_plot,ene_angle_df_n1_plot,
           angle_df_n2_plot,ene_df_n2_plot,normv_df_n2_plot,ene_angle_df_n2_plot,
           angle_df_n3_plot,ene_df_n3_plot,normv_df_n3_plot,ene_angle_df_n3_plot]
           
           
if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 
    from HET_sims_read import HET_sims_read

    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))
    
    
    sim_name = "../../Rb_hyphen/sim/sims/SPT100_al005_Ne5_CID878"
    
    timestep         = 45
    allsteps_flag    = 0
    read_inst_data   = 0
    read_part_tracks = 0
    read_part_lists  = 0
    read_flag        = 1
    
    last_steps = 2
    
    
    
    path_simstate_inp  = sim_name+"/CORE/inp/SimState.hdf5"
    path_simstate_out  = sim_name+"/CORE/out/SimState.hdf5"
    path_postdata_out  = sim_name+"/CORE/out/PostData.hdf5"
    path_simparams_inp = sim_name+"/CORE/inp/sim_params.inp"
    path_picM          = sim_name+"/SET/inp/SPT100_picM.hdf5"
    
    [num_ion_spe,num_neu_spe,n_mp_cell_i,n_mp_cell_n,n_mp_cell_i_min,
       n_mp_cell_i_max,n_mp_cell_n_min,n_mp_cell_n_max,min_ion_plasma_density,
       m_A,spec_refl_prob,ene_bal,points,zs,rs,zscells,rscells,dims,
       nodes_flag,cells_flag,ind_maxr_c,ind_maxz_c,nr_c,nz_c,eta_max,
       eta_min,xi_top,xi_bottom,time,time_fast,steps,steps_fast,dt,dt_e,
       nsteps,nsteps_eFld,faces,nodes,boundary_f,face_geom,elem_geom,n_faces,
       n_elems,cath_elem,z_cath,r_cath,V_cath,mass,ssIons1,ssIons2,ssNeutrals1,ssNeutrals2,
       n_mp_i1_list,n_mp_i2_list,n_mp_n1_list,n_mp_n2_list,phi,Ez,Er,Efield,
       Bz,Br,Bfield,Te,cs01,cs02,nn1,nn2,ni1,ni2,ne,fn1_x,fn1_y,fn1_z,fn2_x,
       fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,
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
       mass_mp_ions,num_mp_neus,num_mp_ions,eta_u,eta_prod,eta_thr,eta_div,
       eta_cur,thrust,thrust_ion,thrust_neu,Id_inst,Id,Vd_inst,Vd,dMdt_i1,
       dMdt_i2,dMdt_n1,dMdt_n2,mflow_coll_i1,mflow_coll_i2,mflow_coll_n1,
       mflow_coll_n2,mflow_fw_i1,mflow_fw_i2,mflow_fw_n1,mflow_fw_n2,
       mflow_tw_i1,mflow_tw_i2,mflow_tw_n1,mflow_tw_n2,mflow_ircmb_picS_n1,
       mflow_ircmb_picS_n2,mflow_inj_i1,mflow_inj_i2,mflow_fwmat_i1,
       mflow_fwmat_i2,mflow_inj_n1,mflow_fwmat_n1,mflow_inj_n2,mflow_fwmat_n2,
       mflow_twmat_i1,mflow_twinf_i1,mflow_twa_i1,mflow_twmat_i2,mflow_twinf_i2,
       mflow_twa_i2,mflow_twmat_n1,mflow_twinf_n1,mflow_twa_n1,mflow_twmat_n2,
       mflow_twinf_n2,mflow_twa_n2,dEdt_i1,dEdt_i2,dEdt_n1,dEdt_n2,
       eneflow_coll_i1,eneflow_coll_i2,eneflow_coll_n1,eneflow_coll_n2,
       eneflow_fw_i1,eneflow_fw_i2,eneflow_fw_n1,eneflow_fw_n2,eneflow_tw_i1,
       eneflow_tw_i2,eneflow_tw_n1,eneflow_tw_n2,Pfield_i1,Pfield_i2,
       eneflow_inj_i1,eneflow_fwmat_i1,eneflow_inj_i2,eneflow_fwmat_i2,
       eneflow_inj_n1,eneflow_fwmat_n1,eneflow_inj_n2,eneflow_fwmat_n2,
       eneflow_twmat_i1,eneflow_twinf_i1,eneflow_twa_i1,eneflow_twmat_i2,
       eneflow_twinf_i2,eneflow_twa_i2,eneflow_twmat_n1,eneflow_twinf_n1,
       eneflow_twa_n1,eneflow_twmat_n2,eneflow_twinf_n2,eneflow_twa_n2,
       ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,nu_cath,ndot_cath,
       F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,
       nu_ei2,nu_i01,nu_i02,nu_i12] = HET_sims_read(path_simstate_inp,path_simstate_out,
                                                    path_postdata_out,path_simparams_inp,
                                                    path_picM,allsteps_flag,timestep,read_inst_data,
                                                    read_part_lists,read_flag)
                                                    
    print("E_max = %15.8e; E_min = %15.8e (V/m)" %( np.nanmax(Efield), np.nanmin(Efield) )) 
    
    [Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,fn1_x,fn1_y,fn1_z,
       fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,un1_x,un1_y,un1_z,
       un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,ji1_x,ji1_y,ji1_z,
       ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,ue_r,ue_t,ue_z,ue_perp,
       ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,
       avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,neu_gen_weights1,neu_gen_weights2,
       ion_gen_weights1,ion_gen_weights2,ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,
       F_theta,Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12] = HET_sims_plotvars(nodes_flag,cells_flag,Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,
                                                                                                                         fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,
                                                                                                                         un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,
                                                                                                                         ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,
                                                                                                                         ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
                                                                                                                         n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                                                                                                         neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                                                                                                         ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,F_theta,Hall_par,Hall_par_eff,
                                                                                                                         nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,nu_i02,nu_i12)
    print("E_max = %15.8e; E_min = %15.8e (V/m)" %( np.nanmax(Efield), np.nanmin(Efield) ))                                                                                                    
                                                                                                        
                                                                                                        
    [Br_plot,Bz_plot,Bfield_plot,phi_plot,Er_plot,Ez_plot,Efield_plot,
       nn1_plot,nn2_plot,ni1_plot,ni2_plot,ne_plot,
       fn1_x_plot,fn1_y_plot,fn1_z_plot,fn2_x_plot,fn2_y_plot,fn2_z_plot,
       fi1_x_plot,fi1_y_plot,fi1_z_plot,fi2_x_plot,fi2_y_plot,fi2_z_plot,
       un1_x_plot,un1_y_plot,un1_z_plot,un2_x_plot,un2_y_plot,un2_z_plot,
       ui1_x_plot,ui1_y_plot,ui1_z_plot,ui2_x_plot,ui2_y_plot,ui2_z_plot,
       ji1_x_plot,ji1_y_plot,ji1_z_plot,ji2_x_plot,ji2_y_plot,ji2_z_plot,
       je_r_plot,je_t_plot,je_z_plot,je_perp_plot,je_para_plot,
       ue_r_plot,ue_t_plot,ue_z_plot,ue_perp_plot,ue_para_plot,uthetaExB_plot,
       Tn1_plot,Tn2_plot,Ti1_plot,Ti2_plot,Te_plot,n_mp_n1_plot,n_mp_n2_plot,
       n_mp_i1_plot,n_mp_i2_plot,avg_w_n1_plot,avg_w_n2_plot,avg_w_i1_plot,
       avg_w_i2_plot,neu_gen_weights1_plot,neu_gen_weights2_plot,
       ion_gen_weights1_plot,ion_gen_weights2_plot,ndot_ion01_n1_plot,
       ndot_ion02_n1_plot,ndot_ion12_i1_plot,ne_cath_plot,nu_cath_plot,ndot_cath_plot,
       F_theta_plot,Hall_par_plot,Hall_par_eff_plot,nu_e_tot_plot,nu_e_tot_eff_plot,
       nu_en_plot,nu_ei1_plot,nu_ei2_plot,nu_i01_plot,nu_i02_plot,nu_i12_plot] = HET_sims_cp_vars(Br,Bz,Bfield,phi,Er,Ez,Efield,nn1,nn2,ni1,ni2,ne,
                                                                                                  fn1_x,fn1_y,fn1_z,fn2_x,fn2_y,fn2_z,fi1_x,fi1_y,fi1_z,fi2_x,fi2_y,fi2_z,
                                                                                                  un1_x,un1_y,un1_z,un2_x,un2_y,un2_z,ui1_x,ui1_y,ui1_z,ui2_x,ui2_y,ui2_z,
                                                                                                  ji1_x,ji1_y,ji1_z,ji2_x,ji2_y,ji2_z,je_r,je_t,je_z,je_perp,je_para,
                                                                                                  ue_r,ue_t,ue_z,ue_perp,ue_para,uthetaExB,Tn1,Tn2,Ti1,Ti2,Te,
                                                                                                  n_mp_n1,n_mp_n2,n_mp_i1,n_mp_i2,avg_w_n1,avg_w_n2,avg_w_i1,avg_w_i2,
                                                                                                  neu_gen_weights1,neu_gen_weights2,ion_gen_weights1,ion_gen_weights2,
                                                                                                  ndot_ion01_n1,ndot_ion02_n1,ndot_ion12_i1,ne_cath,nu_cath,ndot_cath,F_theta,
                                                                                                  Hall_par,Hall_par_eff,nu_e_tot,nu_e_tot_eff,nu_en,nu_ei1,nu_ei2,nu_i01,
                                                                                                  nu_i02,nu_i12)
    print("E_max = %15.8e; E_min = %15.8e (V/m)" %( np.nanmax(Efield_plot), np.nanmin(Efield_plot) )) 
    
#    vlight = 3e8
#    print(np.nanmax(ue_t[:,:,timestep]),np.nanmin(ue_t[:,:,timestep]))
#    print(np.nanmax(ue_t_plot[:,:,timestep]),np.nanmin(ue_t_plot[:,:,timestep]))