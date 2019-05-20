# -*- coding: utf-8 -*-
# Copied half from wobble by Megan Bedell

import numpy as np
import sys

speed_of_light = 2.99792458e8   # m/s

__all__ = ["fit_continuum", "define_stuff","bin_data", "doppler","lorentzian_profile"]

def lorentzian_profile(kappa, S, gamma, kappa0):
    '''
    Calculate a Lorentzian absorption profile.
    Parameters
    ----------
    kappa : ndarray
        The array of wavenumbers at which to sample the profile.
    S : float
        The absorption line "strength" (the integral of the entire line is equal to S).
    gamma : float
        The linewidth parameter of the profile.
    kappa0 : float
        The center position of the profile (in wavenumbers).
    Returns
    -------
    L : ndarray
        The sampled absorption profile.
    '''
    L = (S / np.pi) * gamma / ((kappa - kappa0)**2 + gamma**2)
    return(L)


  
def define_stuff():
    """
    Define variables for fitting
    like array of nights
    and folders
    """
    nights = np.array(["2015-03-23_SunI2" , "2015-04-23_SunI2" , "20150505" , "20150605" , "20150617" ,  "20151002",\
                       "2015-03-25_SunI2",  "2015-04-24_SunI2",  "20150511",  "20150607",  "20150702",  "20151011", \
                       "2015-04-20_SunI2" , "2015-04-28_SunI2" , "20150513"  , "20160227", \
                       "2015-04-21_SunI2",  "20150429" ,         "20150604", "20150612",  "20150929",  "20160401"])

    # Note:
    # Removed "20150611", bc of bad tau values
    # Removed 20150717 bc never worked
    # removed 20150612 bc tau values bad
    # Define fit ranges (chose 10)
    fit_ranges = np.arange(9010,20000,10)

    v_lims     = np.zeros((len(fit_ranges)-1,2))
    for i in range(len(fit_ranges)-1):
        v_lims[i] = np.array([fit_ranges[i],fit_ranges[i+1]])

    return nights, v_lims


def doppler(v, tensors=True):
    frac = (1. - v/speed_of_light) / (1. + v/speed_of_light)
    return np.sqrt(frac)


def fit_continuum(x, y, ivars, order=6, nsigma=[0.3,3.0], maxniter=50):
    """Fit the continuum using sigma clipping
    Args:
        x: The wavelengths
        y: The log-fluxes
        order: The polynomial order to use
        nsigma: The sigma clipping threshold: tuple (low, high)
        maxniter: The maximum number of iterations to do
    Returns:
        The value of the continuum at the wavelengths in x
    """
    A = np.vander(x - np.nanmean(x), order+1)
    m = np.ones(len(x), dtype=bool)
    for i in range(maxniter):
        m[ivars == 0] = 0  # mask out the bad pixels
        w = np.linalg.solve(np.dot(A[m].T, A[m]), np.dot(A[m].T, y[m]))
        mu = np.dot(A, w)
        resid = y - mu
        sigma = np.sqrt(np.nanmedian(resid**2))
        #m_new = np.abs(resid) < nsigma*sigma
        m_new = (resid > -nsigma[0]*sigma) & (resid < nsigma[1]*sigma)
        if m.sum() == m_new.sum():
            m = m_new
            break
        m = m_new
    return mu

def bin_data(xs, ys, ivars, xps):
    """
    Bin data onto a uniform grid using medians.
    
    Args:
        `xs`: `[N, M]` array of xs
        `ys`: `[N, M]` array of ys
        `ivars`: `[N, M]` array of y-ivars
        `xps`: `M'` grid of x-primes for output template
    
    Returns:
        `yps`: `M'` grid of y-primes
    
    """
    all_ys, all_xs, all_ivars = np.ravel(ys), np.ravel(xs), np.ravel(ivars)
    dx = xps[1] - xps[0] # ASSUMES UNIFORM GRID
    yps = np.zeros_like(xps)
    for i,t in enumerate(xps):
        ind = (all_xs >= t-dx/2.) & (all_xs < t+dx/2.)
        if np.sum(ind) > 0:
            #yps[i] = np.nanmedian(all_ys[ind])
            yps[i] = np.nansum(all_ys[ind] * all_ivars[ind]) / np.nansum(all_ivars[ind] + 1.) # MAGIC
    ind_nan = np.isnan(yps)
    yps.flat[ind_nan] = np.interp(xps[ind_nan], xps[~ind_nan], yps[~ind_nan])
    return yps