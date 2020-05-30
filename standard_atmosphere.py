import numpy as np


#https://www.numeric-gmbh.ch/posts/standard-atmosphere-calculations.html

# Set constants
hPa = 0.01
g = 9.8066         # m/s2
kappa = 0.2856219  # poisson exponent = Rd/Cp_d
R = 287.05         # Gas constant for dry air at the surface; J/(K.kg)
p0 = 1.e5          # Reference pressure for potential temperature; Pa
alpha = -1./5.255877  # Density of dry air at 0C and 1000mb?
cp = 1004.7        # Specific heat at constant pressure for dry air; J/(K.kg)
r_earth = 6341.624  # I assume this is earth radius, but is that 6341 or 6371?
gamma = .0065      #The dry adiabatic lapse rate

alpha = -1./5.255877
beta = -6341.624

p00= 1e5 * hPa

def gravity_adjust(Z):
    """
    Adjust gravity with changing altitude.

    Parameters
    ---------
    Z : array_like
        Array of altitudes.

    Returns
    ------
    g2 : array_like
        Altitude adjusted gravity.
    """
    r_earth = 6371.624  # I assume this is earth radius, but is that 6341 or 6371?
    r2 = Z/1000 + r_earth
    g2 = g / (r2/r_earth)**2

    return g2

def cal_zttheta(p):
    """
    Calculate standard temperature, altitude, and potential temperature from standard pressure.

    Parameters
    ----------
    p : array_like
        Standard pressure. 
        !Caution: input pressure should increase in sequence and with unit of 'Pa'

    Returns
    -------
    T: array_like
        Standard temperature.
    Z: array_like
        Standard altitude.
    theta: array_like
        Standard potential temperature.
    """
    nlevels = p.size

    Z = np.zeros(nlevels)
    T = np.zeros(nlevels)
    theta = np.zeros(nlevels)
    #if unit == 'Pa':
    P_tropopause = 22632 #Pa
    tropopause_idx = np.where(p>=P_tropopause)[0][0]
    
    # calculate variables below the tropopause
    Z[tropopause_idx:] = (288.15/gamma) * ( 1. - (101325/p[tropopause_idx:])**alpha)
    T[tropopause_idx:] = 288.15 - 0.0065*Z[tropopause_idx:]
    theta[tropopause_idx:] = T[tropopause_idx:] * (p00/p[tropopause_idx:])**(R/cp)

    # calculate variables above the tropopause
    T[:tropopause_idx] = 216.65
    Z[:tropopause_idx] = 11.e3 + beta * np.log(p[:tropopause_idx] / 22632)
    theta[:tropopause_idx] = T[:tropopause_idx] * (p00/p[:tropopause_idx])**(R/cp)
    
    return T, Z, theta

def diff(a, i=1):
    """
    Calculate the n-th discrete difference given by ``out[k] = a[k+i] - a[k]``
    Basically np.diff() does not work in higher orders

    Parameters
    ----------
    a : array_like
        Input array
    i : int, optional
        The number of times values are differenced

    Returns
    -------
    diff : ndarray
        The i-th differences.
    """
    n = len(a)
    return a[i:]-a[:n-i]

def atmos_structure(p, unit='Pa'):
    """
    Calculate geopotential height, temperature, and stratification parameter in standard atmosphere.

    Parameters
    ----------
    p : array_like
        Standard pressure. Unit default to 'Pa', but 'hPa' is also acceptable.
        The return values should be in the same sequence as p (i.e. increase or decrease).
    unit : str
        Default to 'Pa', but 'hPa' and 'mb' are also acceptable.
        If you are using hPa, but did not provide the unit argument, this function will automaticly convert it into Pa  when the largest pressure values < 2000.

    Returns
    -------
    Z : array_like
        Altitude for the given standard atmosphe
    g*Z : array_like
        Geopotential height for the given standard atmosphere.
    T : array_like
        Temperature structure for the given standard atmosphere.
    S_half : array_like
        Stratification parameter for the given standard atmosphere.
        len(S) = nlevs - 1
    """
    ### reverse the pressure if it is an increasing sequence
    if p[-1]<p[0]:
        p = p[::-1]
        reverse_flag = True
    else:
        reverse_flag = False
    
    ### Convert into Pa if in hPa
    if p.max() < 2000 or unit=='hPa' or unit=='mb':
        p = p*100
    
    # Begin calculation
    R = 287.05
    
    nlevs = len(p)
    nlevels_half = nlevs - 1
    pb = max(p[0], p[-1])
    pt = min(p[0], p[-1])
    p_half = (p[1:] + p[:-1])/2

    # Calculate temperature, altitude, potential temperature structure.
    T, Z, theta = cal_zttheta(p)
    T_half, Z_half, theta_half = cal_zttheta(p_half)

    # Calculate stratification parameter.
    dthetadp = np.zeros(nlevels_half)
    dthetadp[0] = (theta[1] - theta[0])/(p[1]-p[0])
    dthetadp[1:] = diff(theta,i=2) / (diff(p, i=2))
    S_half = - R * (T_half / theta_half) * dthetadp / p_half  # p[:-1]? or p_half
    
    # Reverse back the return values
    if reverse_flag == True:
        Z = Z[::-1]
        T = T[::-1]
        S_half = S_half[::-1]
        
    g2 = gravity_adjust(Z)

    return Z, g2*Z, T, S_half


