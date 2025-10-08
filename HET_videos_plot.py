############################ GENERATING PLOTS #############################
first_step = np.where(Id == 0)[0][-1] + 1
final_step = len(Id)-1

first_step = 500
first_step_fast = first_step*50
final_step_fast = final_step*50
ind = 0
#for i in range(first_step,final_step+1):
for i in range(first_step,first_step + 200):
    ind = ind + 1
    i_fast = 50*i


    [fig, axes] = plt.subplots(nrows=2, ncols=3 ,figsize=(22.5,12))
    ax1 = plt.subplot2grid( (2,3), (0,0) )
    ax2 = plt.subplot2grid( (2,3), (1,0) )
    ax3 = plt.subplot2grid( (2,3), (0,1) )
    ax4 = plt.subplot2grid( (2,3), (1,1) )
    ax5 = plt.subplot2grid( (2,3), (0,2) )
    ax6 = plt.subplot2grid( (2,3), (1,2) )


    ax1.set_title(r"$I_d$ (A)",fontsize = font_size, y=1.02)
    ax1.set_xlabel(r'$t$ (ms)', fontsize = font_size)
    ax1.tick_params(labelsize = ticks_size) 
    ax1.tick_params(axis='x', which='major', pad=10)
    
    ax2.set_title(r"$\bar{n}_e$, $\bar{n}_n$ (m$^{-3}$)",fontsize = font_size, y=1.02)
    ax2.set_xlabel(r'$t$ (ms)', fontsize = font_size)
    ax2.tick_params(labelsize = ticks_size) 
    ax2.tick_params(axis='x', which='major', pad=10)
        

        
    # Plot the time evolution of the discharge current
    ax1.semilogy(time[first_step::], Id[first_step::], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="", color='k', markeredgecolor = 'k', label="")
    ax1.semilogy(time[i], Id[i], linestyle='-', linewidth = line_width, markevery=marker_every_time, markersize=marker_size, marker="o", color='r', markeredgecolor = 'k', label="")
    ax1.set_xlim([time[first_step],time[-1]])
    
    # Plot the time evolution of both the average plasma and neutral density in the domain
    ax2.semilogy(time_fast[first_step_fast::], avg_dens_mp_ions[first_step_fast::], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="", color='b', markeredgecolor = 'k', label=labels[0])            
    ax2.semilogy(time_fast[first_step_fast::], avg_dens_mp_neus[first_step_fast::], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="", color='g', markeredgecolor = 'k', label=labels[1])            
    ax2.semilogy(time_fast[i_fast], avg_dens_mp_ions[i_fast], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="o", color='r', markeredgecolor = 'k', label="")            
    ax2.semilogy(time_fast[i_fast], avg_dens_mp_neus[i_fast], linestyle='-', linewidth = line_width, markevery=marker_every_time_fast, markersize=marker_size, marker="o", color='k', markeredgecolor = 'k', label="")            
    ax2.set_xlim([time[first_step],time[-1]])
    
    # Plot the plasma density contour
    ax3.set_title(r'$n_{e}$ (m$^{-3}$)', fontsize = font_size,y=1.02)
    log_type         = 1
    auto             = 0
    min_val0         = 1E14
    max_val0         = 1E19
    cont             = 1
    lines            = 0
    cont_nlevels     = 100
    auto_cbar_ticks  = 1 
    auto_lines_ticks = -1
    nticks_cbar      = 4
    nticks_lines     = 4
    cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
    lines_ticks      = np.array([5E18,3E18,2E18,1E18,5E17,3E17,2E17,1E17,5E16,2E16])
    lines_ticks_loc  = [(0.7,4.25),(3.13,4.0),(4.3,4.25),(6.14,4.25),(9.0,4.25),(9.2,6.24),(4.11,2.52),(4.15,1.02),(7.17,1.16),(7.65,0.7),(4.24,7.34)]
#    lines_ticks_loc  = 'default'
    cbar_ticks_fmt    = '{%.2f}'
    lines_ticks_fmt   = '{%.1f}'
    lines_width       = line_width
    lines_ticks_color = 'k'
    lines_style       = '-'
    [CS,CS2] = contour_2D (ax3,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, ne_plot[:,:,i], nodes_flag, log_type, auto, 
                      min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                      nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                      lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
#    # Isolines ticks (exponent)
#    ax3.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
    ax3.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
    ax3.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
#    ax3.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
    ax3.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
    
    # Plot the neutral density contour
    ax4.set_title(r'$n_{n}$ (m$^{-3}$)', fontsize = font_size,y=1.02)
    ax = plt.gca()
    log_type         = 1
    auto             = 0
    min_val0         = 1E14
    max_val0         = 1E20
    cont             = 1
    lines            = 0
    cont_nlevels     = 100
    auto_cbar_ticks  = 1 
    auto_lines_ticks = -1
    nticks_cbar      = 4
    nticks_lines     = 10
    cbar_ticks       = np.array([1E11,1E12,1E13,1E14])
    lines_ticks      = np.array([3E19,1.5E19,1E19,5E18,1E18,5E17,2E17,1E17,5E16,3E16])
    lines_ticks_loc  = 'default'
    cbar_ticks_fmt    = '{%.2f}'
    lines_ticks_fmt   = '{%.1f}'
    lines_width       = line_width
    lines_ticks_color = 'k'
    lines_style       = '-'
    [CS,CS2] = contour_2D (ax4,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, nn1_plot[:,:,i], nodes_flag, log_type, auto, 
                      min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                      nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                      lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)  
#    # Isolines ticks (exponent)
#    ax4.clabel(CS2, CS2.levels, fmt=fmt_func_exponent_lines, inline=1, fontsize=ticks_size_isolines, zorder = 1)
    ax4.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
    ax4.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)     
    ax4.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)
    
    # Plot the electric potential contour
    ax5.set_title(r'$\phi$ (V)', fontsize = font_size,y=1.02)
    log_type         = 0
    auto             = 0
    min_val0         = -110.0
    max_val0         = 320.0
    cont             = 1
    lines            = 0
    cont_nlevels     = 100
    auto_cbar_ticks  = 0 
    auto_lines_ticks = -1
    nticks_cbar      = 5
    nticks_lines     = 10
    cbar_ticks       = np.array([300, 250, 200, 150, 100, 50, 0, -50, -100])
#        lines_ticks      = np.array([305, 280, 250, 200, 150, 100, 75, 50, 40, 30, 25, 15, 10, 5, 1, 0, -1, -2, -3, -4, -5, -6])
    lines_ticks      = np.array([305, 280, 250, 200, 150, 100, 75, 50, 40, 25, 15, 10, 5])
    lines_ticks_loc  = 'default'
    cbar_ticks_fmt    = '{%.0f}'
    lines_ticks_fmt   = '{%.0f}'
    lines_width       = line_width
    lines_ticks_color = 'w'
    lines_style       = '-'
    [CS,CS2] = contour_2D (ax5,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, phi_plot[:,:,i], nodes_flag, log_type, auto, 
                           min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                           nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                           lines_ticks_fmt, lines_ticks_color, lines_style, lines_width)     
#        ax5.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, fontsize=ticks_size_isolines, zorder = 1)
    ax5.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
    ax5.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)    
#        ax5.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)
    ax5.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)     
    
    # Plot the electron temperature contour
    ax6.set_title(r'$T_e$ (eV)', fontsize = font_size,y=1.02)
    log_type         = 0
    auto             = 0
    min_val0         = 0.0
    max_val0         = 55.0
    cont             = 1
    lines            = 0
    cont_nlevels     = 100
    auto_cbar_ticks  = 0 
    auto_lines_ticks = -1
    nticks_cbar      = 5
    nticks_lines     = 10
    cbar_ticks       = np.array([55,50,45,40,35,30,25,20,15,10,5,0])
#        lines_ticks      = np.array([7,9,12,20,25,30,35,40,45])
    lines_ticks      = np.array([7,9,12,20,30,35,40,45])
    lines_ticks_loc  = [(0.38,4.25),(0.88,4.25),(1.5,4.25),(2.7,4.6),(3.0,3.8),(3.6,4.8),(3.9,4.25),(4.5,4.25),(5.18,4.0),(5.3,3.2),(5.6,1.8),(3.7,6.8)]
#        lines_ticks_loc  = 'default'
    cbar_ticks_fmt    = '{%.0f}'
    lines_ticks_fmt   = '{%.0f}'
    lines_width       = line_width
    lines_ticks_color = 'k'
    lines_style       = '-'
    [CS,CS2] = contour_2D (ax6,'$z$ (cm)', '$r$ (cm)', font_size, ticks_size, zs, rs, Te_plot[:,:,i], nodes_flag, log_type, auto, 
                           min_val0, max_val0, cont, lines, cont_nlevels, auto_cbar_ticks, auto_lines_ticks,
                           nticks_cbar, nticks_lines, cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt,
                           lines_ticks_fmt, lines_ticks_color, lines_style, lines_width) 
#    ax6.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, manual = lines_ticks_loc, fontsize=ticks_size_isolines, zorder = 1)
    ax6.plot(points[:,0],points[:,1],'k-',linewidth = line_width_boundary,markersize = marker_size)
    ax6.plot(np.array([points[-1,0],points[0,0]]),np.array([points[-1,1],points[0,1]]),'k-',linewidth = line_width_boundary,markersize = marker_size)          
#    ax6.plot(elem_geom[0,elems_cath_Bline],elem_geom[1,elems_cath_Bline],'r-',linewidth = line_width_boundary,markersize = marker_size)        
    ax6.plot(z_cath,r_cath,'ks',linewidth = line_width_boundary,markersize = marker_size)

                
    plt.tight_layout()
    if save_flag == 1:
        fig.savefig(path_out+str(ind)+figs_format,bbox_inches='tight') 
    plt.close("all") 
    ###########################################################################    



        
