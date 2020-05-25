import numpy as np

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

def cal_zttheta(p, unit='Pa'):
    """
    Calculate standard temperature, altitude, and potential temperature from standard pressure.

    Parameters
    ----------
    p : array_like
        Standard pressure. Unit default to 'Pa', but 'hPa' is also acceptable

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
    if unit == 'hPa' or unit == 'mb':
        P_tropopause = 226.32 #hPa
        p00 = 1013.25
    elif unit == 'Pa':
        P_tropopause = 22632 #hPa
        p00 = 101325
    tropopause_idx = np.where(p>=P_tropopause)[0][-1] + 1

    # calculate variables below the tropopause
    Z[:tropopause_idx] = (288.15/gamma) * ( 1. - (1013.25/p[:tropopause_idx])**alpha)
    T[:tropopause_idx] = 288.15 - 0.0065*Z[:tropopause_idx]
    theta[:tropopause_idx] = T[:tropopause_idx] * (p00/p[:tropopause_idx])**(R/cp)

    # calculate variables above the tropopause
    T[tropopause_idx:] = 216.65
    Z[tropopause_idx:] = (11.e3 + beta * np.log(p[tropopause_idx:] / 226.32))
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

def atmos_structure(p):
    """
    Calculate geopotential height, temperature, and stratification parameter in standard atmosphere.

    Parameters
    ----------
    nlevs : int
        Number of levels for the standard atmosphere.
        Default to 19 layers.
    pb : int
        Pressure at bottom atmosphere level.
    pt : int
        Pressure at top atmosphere level.

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
    R = 287.05
    
    nlevs = len(p)
    nlevels_half = nlevs - 1
    pb = max(p[0], p[-1])
    pt = min(p[0], p[-1])
    p_half = (p[1:] + p[:-1])/2

    # Calculate temperature, altitude, potential temperature structure.
    T, Z, theta = cal_zttheta(p, unit='Pa')
    T_half, Z_half, theta_half = cal_zttheta(p_half, unit='Pa')
    g2 = gravity_adjust(Z)

    # Calculate stratification parameter.
    dthetadp = np.zeros(nlevels_half)
    dthetadp[0] = (theta[1] - theta[0])/(1.e2*(p[1]-p[0]))
    dthetadp[1:] = diff(theta,i=2) / (1.e2 * diff(p, i=2))
    S_half = - R * (T_half / theta_half) * dthetadp / p_half  # p[:-1]? or p_half

    return Z, g2*Z, T, S_half
