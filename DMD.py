#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:08:49 2020

@author: adrian
"""

def dmd(X, Y, truncate=None):
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from numpy import dot, multiply, diag, power
    from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
    from numpy.linalg import inv, eig, pinv
    from scipy.linalg import svd, svdvals
    from scipy.integrate import odeint, ode, complex_ode
    from warnings import warn

    U2,Sig2,Vh2 = svd(X, False) # SVD of input matrix
    r = len(Sig2) if truncate is None else truncate # rank truncation
    U = U2[:,:r]
    Sig = diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]
    Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig)) # build A tilde
    mu,W = eig(Atil)
    Phi = dot(dot(dot(Y, V), inv(Sig)), W) # build DMD modes
    return mu, Phi

def dmd_rom(Big_X, rval, dt):
    """
    %This function is a wrapper for performing DMD on reduced order models
    %of fluid dynamical systems. The wrapper is built such that the common
    %DMD outputs are easily available, so only the birds-eye theoretical
    %grasp is needed for operation. 
    
    %====INPUTS===
    %Big_X: The ND matrix to analyze the DMD modes. First dimension is
    %time. All other dimensions are arbitrarily chosen and will be reshaped
    %back when the eigenvectors are calculated.
    
    %r: Number of reduced order POD modes to keep. Usually much less than
    %the number of snapshots. If in doubt, process the data set with this function and
    %plot stem(POD_Mode_Energies). The first few modes should have more
    %energy than the remaining modes.
    
    %dt: Time delay between snapshots. Gives the Nyquist limit of the data
    %set and allows the output ModeFrequencies to be meaningful physical or
    %dimensionless frequencies. If unused, make dt=1.
    
    %====OUTPUTS===
    %Eigenvalues: DMD mode complex eigenvalues (r by 1). Think that for each time
    %step the corresponding eigenvector is multiplied by
    %exp(Eigenvalue*dt). To understand more, google "z-transform".
    
    %Eigenvectors: The complex mode shapes themselves (r by [N-1]D).
    %Enables one to visualize the important dynamics of the system. Output
    %is already in the same dimensions as the input, where the first (time) dimension now is
    %replaced by mode number.
    
    %ModeAmplitudes: The complex amplitudes of the mode shapes. Determines
    %the dominant modes. i.e., ModeAmplitudes=max(abs(ModeAmplitudes)) is
    %the dominant mode.
    
    %ModeFrequencies: The real, two-sided frequencies related to each mode.
    %DMD being a linear decomposition will have each mode be related to a
    %single frequency. Outputs are omegas in rad/s
    
    %GrowthRates: The growth/decay rates related to each mode. Units [1/s].
    %If GrowthRates<0, mode decays. If GrowthRates>0, mode grows. In
    %realistic systems, modes GrowthRates<<0 will decay fast whereas modes
    %with GrowthRates~0 probably correspond to a limit-cycle of some sort.
    
    %POD_Mode_Energies: The mode energies of all POD modes. If the user
    %doesn't know the appropriate value of the input "r", plotting
    %stem(POD_Mode_Energies) or
    %stem(cumsum(POD_Mode_Energies)/sum(POD_Mode_Energies)) can give
    %insight on how many modes to keep. Good guesses would be to retain at
    %least 70% of the total energy of the system.
    
    """
    
    import numpy as np
    from numpy.matlib import repmat
    
    dims = np.shape(Big_X)
    newDims = np.zeros(np.shape(dims))
    newDims = dims
    newDims_l = list(newDims)
    newDims_l[0] = rval
    newDims = tuple(newDims_l)
    # Removes mean. Note: Not removing the mean biases the modes as the
    # data points centroid is shifted. If one wants to capture only the
    # oscillations around the mean, the mean MUST be removed.
#    Big_X = Big_X- repmat(np.mean(Big_X,axis=0).reshape(1,dims[1]), dims[0], np.ones((1,len(dims)-1),dtype=int)[0][0])
    
    # Reshapes Big_X
    if np.any(np.iscomplex(Big_X)):
        Big_Xdata = np.zeros((dims[0],np.prod(dims[1::])),dtype=complex)
    else:        
        Big_Xdata = np.zeros((dims[0],np.prod(dims[1::])))
    for i in range(0,dims[0]):
        if len(dims) == 2:
            # 1D problems
            Big_Xdata[i,:] = np.reshape(Big_X[i,:],(1,np.prod(dims[1::])),order='F') 
        elif len(dims) == 3:
            # 2D problems
            Big_Xdata[i,:] = np.reshape(Big_X[i,:,:],(1,np.prod(dims[1::])),order='F') 
    Big_X = np.transpose(Big_Xdata)
    
    
    # Split Big_X into two snapshot sets
    X = Big_X[:,0:-1]
    Y = Big_X[:,1::]
        
    # SVD on X
#    [U, Svec, VH] = np.linalg.svd(X,full_matrices=True, compute_uv=True)
    [U, Svec, VH] = np.linalg.svd(X)
    # Obtain the same result as in Matlab
    S = np.zeros(np.shape(X))
    np.fill_diagonal(S, Svec)
    V = VH.T.conj()  
    
    # Before reducing rank returns the mode energies for further analysis of
    # the ROM validity
    POD_Mode_Energies = Svec**2
    
    # Reduce rank
    U = U[:,0:rval]
    V = V[:,0:rval]
    S = np.zeros((rval,rval), dtype=complex)
    S[:rval, :rval] = np.diag(Svec[:rval])
    
    # Get A_tilde
    U_transp_conj = np.transpose(np.conjugate(U)) 
    Sinv = np.linalg.inv(S)
    M1 = np.matmul(U_transp_conj,Y)
    M2 = np.matmul(M1,V)
    A_tilde = np.matmul(M2,Sinv)
    #A_tilde = U.H'*Y*V/S;
    
    # (For debugging), we can compare if A_tilde=A, for r=max(r):
    # A=Y*pinv(X);
    
    # Compute A_tilde eigenvalues and eigenvectors
#    [eVecs, Eigenvalues] = np.eig(A_tilde);
    [Eigenvalues, eVecs] = np.linalg.eig(A_tilde)
    
    # Gets the DMD eigenvectors back
    M3 = np.matmul(Y,V)
    M4 = np.matmul(M3,Sinv)
    Eigenvectors = np.matmul(M4,eVecs)
#    Eigenvectors = Y*V*inv(S)*eVecs;
#    Eigenvalues=diag(Eigenvalues);
    
    # Gets the mode amplitudes
    index = 0
    ModeAmplitudes = np.matmul(np.linalg.pinv(Eigenvectors),X[:,index])
#    ModeAmplitudes=Eigenvectors\X(:,1);
    
    # Gets the frequencies associated with the modes
    fNY = 1.0/(2*dt);
#    ModeFrequencies=(angle(Eigenvalues)/pi)*fNY;
    ModeFrequencies = np.angle(Eigenvalues)/dt
    
    # Gets the growth rates    
    GrowthRates = np.log(np.abs(Eigenvalues))/dt
    
    # Reshapes the Eigenvectors back to original Big_X dims
    Eigenvectors = np.transpose(Eigenvectors)
    Eigenvectors = np.reshape(Eigenvectors,newDims)
    
    return [Eigenvalues, Eigenvectors, ModeAmplitudes, ModeFrequencies, GrowthRates, POD_Mode_Energies]