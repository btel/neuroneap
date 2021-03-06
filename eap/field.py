#!/usr/bin/env python
#coding=utf-8

import numpy as np
import scipy.signal
import re

def select_sections(coords, type):
    """Filter segments according to their name (taken from name field
    in coords)
    
    - type - regular expression that the name should match
    """ 
    sec_type = np.zeros(len(coords), dtype=np.bool)
    for i, name in enumerate(coords['name']):
        if re.match(type, name) is not None:
            sec_type[i] = True
    return sec_type

def hp_fir(N, cutoff, dt):
    Fs = 1./(dt*1.E-3)
    h = scipy.signal.firwin(N, cutoff/Fs/2., pass_zero=False)
    def _filter_func(s):
        filt_sig = scipy.signal.filtfilt(h, [1], s)
        return filt_sig
    return _filter_func
        
def calc_v_ext(pos, coord,  I, eta=3.5):
    """
    conductivity [eta] = Ohm.m
    segments coordinates [coord] = um
    measurement position [pos] = um
    membrane current density [I] = mA/cm2 
    """
    x0, y0, z0 = pos
    r = np.sqrt((coord['y']-y0)**2 + (coord['x']-x0)**2 +
                (coord['z']-z0)**2)*1.E-6 # m
    S = np.pi*coord['diam']*coord['L']*1.E-12 #m2
    I = I*1.E4 # mA/m2
    v_ext = np.sum(1./(4*np.pi)*eta*I*S/r,1) * 1E6 # nV
    return v_ext

def _cylindric_coords(pt1, pt2, pos):

    assert not (pt1==pt2).all(axis=0).any(), ('first and last point of segment '
                                              'must be different to define a line')
        
    #calculate distance from line (from wikipedia)
    n = pt2-pt1
    n = n/_vlen(n) #normal vector of cylinder axis
    a = pt1
    pos = pos[:, np.newaxis]

    rad_dist = _vlen((a - pos)-((a-pos)*n).sum(0)[None,:]*n)

    longl_dist = (((pos-a)*n).sum(0))

    return rad_dist, longl_dist

def _vlen(x):
    return np.sqrt(np.sum(x**2,0))

def estimate_lsa(pos, coord, I, eta=3.5):

    pos = np.array(pos)
    pt1 = np.vstack((coord['x0'], coord['y0'], coord['z0']))
    pt2 = np.vstack((coord['x1'], coord['y1'], coord['z1']))
    diam = coord['diam']
    
    r, d  = _cylindric_coords(pt1, pt2, pos)
    #l = _vlen(pt1-pt2)
    l = coord['L']
    assert (r>=0).all()
    I = I*1.E4*np.pi*diam*1E-6
    C = 1./(4*np.pi)*eta
    #v_ext = C*I*np.log(np.abs(r**2+(d+l)**2)/np.abs(r**2+d**2))/2.
    #v_ext = C*I*np.log(np.abs(np.sqrt(d**2+r)-d)/np.abs(np.sqrt((l+d)**2+r**2)-l-d))
    numerator = l-d+np.sqrt(r**2+(l-d)**2)
    denominator = -d+np.sqrt(r**2+d**2)
    v_ext = C*I*np.log(numerator/denominator)
    v_ext[np.isnan(v_ext)] = 0
    v_ext = v_ext*1E6 # nV
    return v_ext.sum(1)

def estimate_on_grid(coords, I, XX, YY, z=0):
    """Estimate field on a grid.
    
    Arguments:

    * coord (structured array) -- coordinates of neuron segments (see
    estimate_lsa for details)
    * I (2d array, float) -- current densities in time and (neuronal) space
    * X (2d array, float) -- X coordinates of grid data 
    * Y (2d array, float) -- Y coordinates of grid data
    * z (scalar, float) - z coordinate 

    """
    if not XX.shape==YY.shape:
        raise TypeError, "XX and YY must have the same dimensions"
    
    ts, _ = I.shape
    xs, ys  = XX.shape
    v = np.zeros((ts, xs, ys))
    for i in range(xs):
        for j in range(ys):
            v_ext = estimate_lsa((XX[i,j], YY[i,j], z), coords, I)
            v[:, i,j] = v_ext

    return v

def calc_grid(xrange, yrange, n_samp):
    xmin, xmax = xrange
    ymin, ymax = yrange
    
    try:
        n_x, n_y = n_samp
    except TypeError:
        n_x = n_y = n_samp

    x = np.linspace(xmin, xmax, n_x)
    y = np.linspace(ymin, ymax, n_y)

    XX, YY = np.meshgrid(x, y)
    
    return XX, YY

def calc_dipole_moment(coords, i_axial):
    """calculate current densities given coordinates array coord
    and axial current densities in each segment i_axial.
    Current dipole has dimensions of mA*um"""

    n = [coords['x1']-coords['x0'],
         coords['y1']-coords['y0'],
         coords['z1']-coords['z0']]

    n = np.array(n)
    #convert densities in mA/cm2 to intensities in mA
    i_current = i_axial*coords['diam'][None,:]**2*1e-8/4.*np.pi 
    dipole_moment = (n[None, :, :]*i_current[:, None, :]).sum(2)

    return dipole_moment.T

