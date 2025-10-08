# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:51:31 2017

@author: adry

############################################################################
Description:    This python script sets a 2D contour plot
############################################################################
Inputs:         1) ax: figure axis handle
                2) labx: x-axis label
                3) laby: y-axis label
                4) font_size: font size for labels
                5) ticks_size: size for ticks (in axes, colorbar and isolines) 
                6) xs,ys: x,y coordinates meshgrid matrices
                7) vals: matrix with values to be plotted
                8) log_type: flag indicating if log contour
                             0 - linear contour
                             1 - log contour 
                9) auto: flag for setting automatically the min and max values 
                         for the contour levels
                         0 - min_val0 and max_val0 are used as extremes
                         1 - min and max values are those of vals
               10) min_val0,max_val0: min and max values for the contour
                                      levels in case auto = 0
               11) cont: flag to plot the coloured contour 
                         0 - Do not plot the coloured contour
                         1 - Plot the coloured contour
               12) lines: flag to plot the isolines 
                          0 - Do not plot the isolines
                          1 - Plot the isolines
               13) cont_nlevels: number of levels used for the contour
               14) auto_cbar_ticks: flag to set the ticks in colorbar
                               -1 - Do not generate the colorbar. We can generate 
                                    the colorbar out of this function in case 
                                    we need more complex format on colorbar ticks.
                                0 - Not automatic ticks in colorbar Consider ticks
                                    given in cbar_ticks.
                                1 - Automatic ticks in colorbar. A number of 
                                    ticks equal to nticks_cbar is generated 
                                    between min_val and max_val           
               15) auto_lines_ticks: flag to set the ticks in isolines
                               -1 - Do not set ticks in isolines (generate a number
                                    os isolines equal to the length of isolines_ticks)
                                    We can generate the isolines ticks out of this
                                    function in case we need more complex format.
                                0 - Not automatic ticks in isolines. Consider 
                                    ticks given in lines_ticks.
                                1 - Automatic ticks in isolines. A number of 
                                    ticks (lines) equal to nticks_lines is 
                                    generated between min_val and max_val
               16) nticks_cbar: Number of ticks in colorbar when generated
                                automatically (auto_cbar_ticks = 1). Not used
                                in logarithm plots.
               17) nticks_lines: Number of ticks in isolines when generated
                                 automatically (auto_lines_ticks = 1)
               18) cbar_ticks: vector containing the ticks for the colorbar in 
                               case they are not automatically set.
               19) lines_ticks: vector containing the ticks for the isolines in
                                case they are not automatically set.
               19) lines_ticks_loc: vector including the position of the isolines
                                    ticks. Only available when ticks are not set
                                    automatically (auto_ticks = 0). If set to 
                                    'default', default locations will be taken
               20) cbar_ticks_fmt: format for colorbar ticks. If set to 'default'
                                   the default format is selected. For logarithm 
                                   plots is not used. If we want more complex 
                                   format it is better to introduce 
                                   auto_cbar_ticks = -1 and generate the colorbar
                                   out of this function.
               21) lines_ticks_fmt: format for isolines ticks. If set to 'default'
                                   the default format is selected. For logarithm
                                   plots is not used. If we want more complex 
                                   format it is better to introduce 
                                   auto_lines_ticks = -1 and generate the isolines
                                   ticks out of this function.
               22) lines_ticks_color: color for isolines and their ticks
               23) lines_style: isolines style
               24) lines_width: isolines width
############################################################################
Output:        1) CS: contour handle
               2) CS2: isolines handle
               
"""


def contour_2D_fig2 (ax, labx, laby, font_size, ticks_size, xs, ys, vals, nodes_flag,
                log_type, auto, min_val0, max_val0, cont, lines, cont_nlevels,
                auto_cbar_ticks, auto_lines_ticks, nticks_cbar, nticks_lines,
                cbar_ticks, lines_ticks, lines_ticks_loc, cbar_ticks_fmt, 
                lines_ticks_fmt, lines_ticks_color, lines_style,lines_width):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker   
    
    
    def fmt_func_exponent(x,pos):
        a, b = '{:.0e}'.format(x).split('e')
        b = int(b)
        return r'${}\cdot$10$^{{{}}}$'.format(a, b)
    
#    def fmt_func_exponent(x,pos):
#        a, b = '{:.0e}'.format(x).split('e')
#        b = int(b)
#        return r'$10$^{{{}}}$'.format(a,b)
    
    
    # Colormap for 2D contours
    cmap_name = 'jet'
    
    # Consider automatic max and min or inputed max and min values
    if auto == 1:
        min_val = np.nanmin(vals[np.where(nodes_flag != 0)])
        max_val = np.nanmax(vals[np.where(nodes_flag != 0)])
        
#        if min_val < min_val0:
#            min_val = min_val0
#        if max_val > max_val0:
#            max_val = max_val0
      
    else:
        min_val = min_val0
        max_val = max_val0
        
    
    if log_type == 0: 
        # Non-logarithmic plot
        # Plot coloured contour and colorbar with ticks if needed
        if cont == 1: 
            levels = np.linspace(min_val, max_val, cont_nlevels)
            # Contour map without extend for out of colorbar values
            CS = ax.contourf(xs, ys, vals, levels, cmap=cmap_name) 
            # Contour map with extend for out of colorbar values
#            CS = ax.contourf(xs, ys, vals, levels, cmap=cmap_name, extend='both') 
            # Generate colorbar if needed
            if auto_cbar_ticks == 1 or auto_cbar_ticks == 0:
                # For the colorbar ticks, only if auto_cbar_ticks=1 we generate automatic
                # ticks. Otherwise is given already in inputed cbar_ticks
                if auto_cbar_ticks == 1:
                    cbar_ticks = np.linspace(min_val, max_val, nticks_cbar)
#                CS.cmap.set_under('white')
#                CS.cmap.set_over('black')
                if cbar_ticks_fmt == 'default':
                    cbar = plt.colorbar(CS,ticks=cbar_ticks,ax=ax)
                else:
                    cbar = plt.colorbar(CS,ticks=cbar_ticks,format=cbar_ticks_fmt,ax=ax)
#                    cbar = plt.colorbar(CS,ticks=cbar_ticks,format=ticker.FuncFormatter(fmt_func_exponent),ax=ax)
                cbar.ax.tick_params(labelsize=ticks_size)
#                cbar.set_label(cbar_lab, size = font_size, rotation=360, labelpad=cbar_labelpad, y=cbar_y)

        # Plot isolines if needed
        if lines == 1:
            # For the isolines ticks, only if auto_lines_ticks=1 we generate automatic
            # ticks. Otherwise is given already in inputed lines_ticks
            if auto_lines_ticks == 1:
                lines_ticks = np.linspace(min_val, max_val, nticks_lines)
            CS2 = ax.contour(xs, ys, vals, lines_ticks, colors=lines_ticks_color,
                             linestyles=lines_style,linewidths=lines_width)
            # Isolines ticks: select format and manual location if needed
            if auto_lines_ticks == 1:
                if lines_ticks_fmt == 'default':
                    # ax.clabel(CS2, CS2.levels, inline=1, 
                    #           fontsize=ticks_size, zorder = 1)
                    ax.clabel(CS2, CS2.levels, inline=1, 
                              fontsize=ticks_size)
                else:
                    # ax.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, 
                    #           fontsize=ticks_size, zorder = 1)
                    ax.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, 
                              fontsize=ticks_size)
            elif auto_lines_ticks == 0:
                if lines_ticks_fmt == 'default':
                    if lines_ticks_loc == 'default':
                        # ax.clabel(CS2, CS2.levels, inline=1, fontsize=ticks_size,
                        #           zorder = 1)
                        ax.clabel(CS2, CS2.levels, inline=1, fontsize=ticks_size)
                    else:
                        # ax.clabel(CS2, CS2.levels, inline=1, manual = lines_ticks_loc,
                        #           fontsize=ticks_size, zorder = 1)
                        ax.clabel(CS2, CS2.levels, inline=1, manual = lines_ticks_loc,
                                  fontsize=ticks_size)
                else:
                    if lines_ticks_loc == 'default':
                        # ax.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, 
                        #           fontsize=ticks_size,zorder = 1)
                        ax.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, 
                                  fontsize=ticks_size)
                    else:
                        # ax.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, 
                        #           manual = lines_ticks_loc, fontsize=ticks_size,
                        #           zorder = 1)
                        ax.clabel(CS2, CS2.levels, fmt=lines_ticks_fmt, inline=1, 
                                  manual = lines_ticks_loc, fontsize=ticks_size)
        
    elif log_type == 1:
        # Log plot
        # Plot coloured contour and colorbar with ticks if needed
        min_val_exp = np.floor(np.log10(min_val))
#        max_val_exp = np.floor(np.log10(max_val))+0.5
        max_val_exp = np.floor(np.log10(max_val))+1
#        if 10**max_val_exp/max_val >= 5.0:
#            max_val_exp = max_val_exp -1
#        max_val_exp = np.floor(np.log10(max_val))
        if cont == 1:    
            levels_exp = np.linspace(min_val_exp,max_val_exp,cont_nlevels)
            levels     = np.power(10, levels_exp)  
            # Contour map without extend for out of colorbar values
            CS = ax.contourf(xs, ys, vals, levels, locator=ticker.LogLocator(),cmap=cmap_name) 
            # Contour map with extend for out of colorbar values in max
            # CS = ax.contourf(xs, ys, vals, levels, locator=ticker.LogLocator(),cmap=cmap_name,extend='max') 
            # Contour map with extend for out of colorbar values in both max and min
            # CS = ax.contourf(xs, ys, vals, levels, locator=ticker.LogLocator(),cmap=cmap_name,extend='both') 
#             # Generate colorbar if needed
#             if auto_cbar_ticks == 1 or auto_cbar_ticks == 0:
#                 # For the colorbar ticks, only if auto_cbar_ticks=1 we generate automatic
#                 # ticks. Otherwise is given already in inputed cbar_ticks. Format for
#                 # colorbar ticks is not selected in this case
#                 if auto_cbar_ticks == 1:
#                     cbar_nticks = np.int(max_val_exp - min_val_exp + 1)
#                     ticks_exp   = np.linspace(min_val_exp,max_val_exp,cbar_nticks)
#                     cbar_ticks  = np.power(10, ticks_exp)
#                 CS.cmap.set_under('white')
#                 CS.cmap.set_over('black')
#                 cbar = plt.colorbar(CS,ticks=cbar_ticks,ax=ax)
# #                cbar = plt.colorbar(CS,ticks=cbar_ticks,format=ticker.FuncFormatter(fmt_func_exponent),ax=ax)
#                 cbar.ax.tick_params(labelsize=ticks_size)
# #                cbar.set_label(r"$\frac{n_{i+}^{Slow}}{n_{i+}^{Fast} + n_{i+}^{Slow}}$", size = font_size, rotation=360, labelpad=-20, y=1.05)

        # Plot isolines if needed
        if lines == 1:
            # For the isolines ticks, only if auto_lines_ticks=1 we generate automatic
            # ticks. Otherwise is given already in inputed lines_ticks
            if auto_lines_ticks == 1:
                lines_nticks = np.int(max_val_exp - min_val_exp + 1)
                ticks_exp   = np.linspace(min_val_exp,max_val_exp,lines_nticks)
                lines_ticks  = np.power(10, ticks_exp)
            CS2 = ax.contour(xs, ys, vals, lines_ticks, colors=lines_ticks_color,
                             linestyles=lines_style,linewidths=lines_width)
            # Isolines ticks: format is not selected in this case. We only set 
            # manual location if needed
            if auto_lines_ticks == 1:
                # ax.clabel(CS2, CS2.levels, fmt=ticker.LogFormatterMathtext(), inline=1, 
                #           fontsize=ticks_size, zorder = 1)
                ax.clabel(CS2, CS2.levels, fmt=ticker.LogFormatterMathtext(), inline=1, 
                          fontsize=ticks_size)
            elif auto_lines_ticks == 0:
                if lines_ticks_loc == 'default':
                    # ax.clabel(CS2, CS2.levels, fmt=ticker.LogFormatterMathtext(), inline=1, 
                    #           fontsize=ticks_size, zorder = 1)
                    ax.clabel(CS2, CS2.levels, fmt=ticker.LogFormatterMathtext(), inline=1, 
                              fontsize=ticks_size)
                else:
                    ax.clabel(CS2, CS2.levels, fmt=ticker.LogFormatterMathtext(), inline=1,
                              manual = lines_ticks_loc, fontsize=ticks_size, zorder = 1)
            

    # Generate colorbar if needed
#    if cont == 1:
#        if auto_ticks == 1 or auto_ticks == 0:
#            CS.cmap.set_under('white')
#            CS.cmap.set_over('black')
#            if cbar_ticks_fmt == 'default':
#                cbar = plt.colorbar(CS,ticks=cbar_ticks,ax=ax)
#            else:
#                cbar = plt.colorbar(CS,ticks=cbar_ticks,format=cbar_ticks_fmt,ax=ax)
#            cbar.ax.tick_params(labelsize=ticks_size)
#    cbar.set_label(cbar_lab, size = font_size, rotation=360, labelpad=cbar_labelpad, y=cbar_y)
    # ax.set_xlabel(labx, fontsize = font_size)
    # ax.set_ylabel(laby, fontsize = font_size)   
    # ax.tick_params(labelsize = ticks_size)
#    plt.xticks(fontsize = ticks_size)
#    plt.yticks(fontsize = ticks_size)

    
    if cont == 0:
        CS = 'None'
    if lines == 0:
        CS2 = 'None'
    
    return CS, CS2
    
    
    

