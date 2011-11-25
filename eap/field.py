#!/usr/bin/env python
#coding=utf-8

import numpy as np
import scipy.signal
import re

def filter_sections(coords, type):
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

def calc_lsa(pos, coord, I, eta=3.5):

    def _vlen(x):
        return np.sqrt(np.sum(x**2,0))

    def _cylindric_coords(pt1, pt2, pos):
        
        dx1 = pt1-pos[:, np.newaxis]
        dx2 = pt2-pos[:, np.newaxis]

        dx12 = pt1-pt2

        rad_dist = _vlen(np.cross(dx1, dx2, axis=0))/_vlen(dx12)


        dxs = np.vstack((np.sum(dx1**2,0), np.sum(dx2**2,0)))
        r = np.min(dxs,0)
        longl_dist = np.sqrt(r-rad_dist**2)
        
        return rad_dist, longl_dist

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
    v_ext = C*I*np.log(np.abs(r**2+(d+l)**2)/np.abs(r**2+d**2))/2.
    #v_ext = C*I*np.log(np.abs(np.sqrt(d**2+r)-d)/np.abs(np.sqrt((l+d)**2+r**2)-l-d))
    v_ext[np.isnan(v_ext)] = 0
    v_ext = v_ext*1E6 # nV
    return v_ext.sum(1) 

