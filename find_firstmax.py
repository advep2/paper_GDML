# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:16:42 2018

@author: adrian

############################################################################
Description:    This python script gives the a number num of first maximum
                values of an array y and its corresponding x values
############################################################################
Inputs:         1) x,y vectors
                2) num: Number of first maximum values of y
############################################################################
Output:        1) Read variables
"""


def find_firstmax(x,y,num):
    

    import numpy as np

    ymaxs = np.zeros(num,dtype=float)
    xmaxs = np.zeros(num,dtype=float)

    for i in range(0,num):
        ind = np.where(y == np.nanmax(y))[0][0]
        ymaxs[i] = y[ind]
        xmaxs[i] = x[ind]
        y[ind] = np.nan



    return[ymaxs,xmaxs]

if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt 

    
    # Close all existing figures
    plt.close("all")
    # Change current working directory to the current script folder:
    os.chdir(os.path.dirname(os.path.abspath('__file__')))

    y = np.array([5,0.5,100,500,0,2,5000],dtype=float)
    x = np.linspace(0,1,len(y))

    num = 3

    [ymaxs,xmaxs] = find_firstmax(x,y,num)

    print(ymaxs)
    print(xmaxs)
