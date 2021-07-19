import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from wrf import getvar, destagger,interplevel, vinterp, latlon_coords, get_cartopy
import metpy.constants as mpconst
import metpy.calc as mpcalc
import numpy.ma as ma

def streamfunction_boundary_condition(psi0, u, v, msfm, dlat, dlon, level="full", mapfactor=True):
    '''
    Calculate the Non-divergent Boundary Condition streamfunction. Prepare the Psi field for 
    further Streamfunction inversion.
    -----------
    Parameters:
    -----------
    psi0: array like
        Streamfunction initial guess, a.k.a. geopotential field/f0
        
    u, v: array like
        wind field
        
    msfm: 2d array
        mapfactor from model output
        
    dlat, dlon: float
        model zonal and meridional interval in [meters]
        
    level: 'full' (default) or a list of integers
        level(s) to be calculated, 'full' means calculated all vertical levels
        
    mapfactor: boolean
        weather to use mapfactor for streamfunction boudary condition calculation.
    -----------
    Returns:
    -----------
    psi: array like
        streamfunction field with non-divergent boundary condition.
    '''
    psi = psi0
    nz, ny, nx = psi.shape
    dsum = 0
    circum = 0
    if level == 'full':
        nzz = np.arange(nz)
    else:
        nzz = level
        
    if mapfactor == True:

        
        for k in nzz:

            for j in range(ny-1,0,-1):                 # western edge N->S
                avgmsfm =  (msfm[j,0]+msfm[j-1,0])/2 
                circum += avgmsfm * dlat
                dsum += (u[k,j,0] + u[k,j-1,0])/2 * dlat / avgmsfm
            for i in range(nx-1):                      # southern edge
                avgmsfm = (msfm[0,i]+msfm[0,i+1])/2
                dsum += (v[k,0,i] + v[k,0,i+1])/2 * dlon / avgmsfm
                circum += avgmsfm * dlon
            for j in np.arange(ny-1):                  # eastern edge
                avgmsfm = (msfm[j,-1]+msfm[j+1,-1])/2 
                dsum -= (u[k,j,-1] + u[k,j+1,-1])/2 * dlat / avgmsfm
                circum += avgmsfm * dlat
            for i in np.arange(nx-1,0,-1):             # northern edge
                avgmsfm = (msfm[-1,i]+msfm[-1,i-1])/2
                dsum -= (v[k,-1,i] + v[k,-1,i-1])/2 * dlon / avgmsfm
                circum += avgmsfm * dlon


            dsum /= circum

            for j in range(ny-1,0,-1):    #left edge N->S
                avgmsfm =  (msfm[j,0]+msfm[j-1,0])/2 
                psi[k,j-1,0] = psi[k,j,0] +  (-dsum * avgmsfm
                                              + (u[k,j,0]+u[k,j-1,0])/2 / avgmsfm )* dlat 
            for i in range(nx-1):
                avgmsfm = (msfm[0,i]+msfm[0,i+1])/2
                psi[k,0,i+1] = psi[k,0,i] + (-dsum * avgmsfm
                                             + (v[k,0,i] + v[k,0,i+1])/2 / avgmsfm )* dlon 
            for j in np.arange(ny-1):
                avgmsfm = (msfm[j,-1]+msfm[j+1,-1])/2 
                psi[k,j+1,-1] = psi[k,j,-1] + (-dsum * avgmsfm
                                               - (u[k,j,-1] + u[k,j+1,-1])/2 / avgmsfm )* dlat 
            for i in np.arange(nx-1,0,-1):
                avgmsfm = (msfm[-1,i]+msfm[-1,i-1])/2
                psi[k,-1,i-1] = psi[k,-1,i] + (-dsum * avgmsfm
                                               - (v[k,-1,i] + v[k,-1,i-1])/2 / avgmsfm )* dlon  

        return psi
    
    if mapfactor == False:
        
        for k in nzz:

            for i in range(nx-1):
                dsum += (v[k,0,i] + v[k,0,i+1])/2 * dlon    #.....v on southern edge (+)
                dsum -= (v[k,-1,i] + v[k,-1,i+1])/2 * dlon  #...v on northwen edge (-)
            for j in range(ny-1):
                dsum += (u[k,j,0] + u[k,j+1,0])/2 * dlat        #....u on western edge (+)
                dsum -= (u[k,j,-1] + u[k,j+1,-1])/2 * dlat        #....u on eastern edge (-)
            print(dsum)

            dsum /= 2 * dlat * (ny-1) + dlon * (nx-1) * 2
            print(dsum)
            '''
            integrate  by Davis (2.40) to get the whole psi
            '''   
            print('psi st', psi[k,-1,0])
            psi_st = psi[k,-1,0]
            for j in range(ny-1,0,-1):    #left edge N->S
                psi[k,j-1,0] = psi[k,j,0] +  (-dsum + (u[k,j,0]+u[k,j-1,0])/2 )* dlat
            for i in range(nx-1):
                psi[k,0,i+1] = psi[k,0,i] + (-dsum + (v[k,0,i] + v[k,0,i+1])/2 )* dlon
            for j in np.arange(ny-1):
                psi[k,j+1,-1] = psi[k,j,-1] + (-dsum - (u[k,j,-1] + u[k,j+1,-1])/2 )* dlat 
            for i in np.arange(nx-1,0,-1):
                psi[k,-1,i-1] = psi[k,-1,i] + (-dsum - (v[k,-1,i] + v[k,-1,i-1])/2 )* dlon 

        return psi
    
def qgpv_inversion_2d(psi, vor, msfm, err_thrs, itrs_thrs, omegas=1.8):
    '''
    Use SOR to inver the potential vorticity field abided by the quasi-geostrophic constraint.
    ----------
    returns:
    psi: array like
        inverted streamfunction
    error_list: list
        a list of error of each iteration, sanity check for convergence.
        
    '''
    ny, nx = psi.shape
    AI_1 = 1.
    AI_2 = 1.
    AI_4 = 1.
    AI_5 = 1.
    AI_3v = -4.
    
    error_list = []
    for itr in range(itrs_thrs):
        error = 0
        for j in range(1, ny-1):
            for i in range(1,nx-1):
                lap = (AI_1 * psi[j-1,i]
                       + AI_2 * psi[j,i-1]
                       + AI_3v * psi[j,i]
                       + AI_4 * psi[j,i+1]
                       + AI_5 * psi[j+1,i])
                res = lap - vor[j,i] * msfm[j,i]
                psi[j,i] = psi[j,i] - omegas*res/AI_3v
                error = error+abs(res)

        unit_err = error/(nx-2)/(ny-2)
        error_list.append(unit_err)
        if unit_err < err_thrs:
            print('n iter', itr)
            print('error', unit_err)
            return psi, error_list
        
    return psi, error_list
