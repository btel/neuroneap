#!/usr/bin/env python
#coding=utf-8

import numpy as np
import scipy.signal
import re

def filter_sections(type):
    sec_type = []
    for sec in h.allsec():
        istype = re.match(type, sec.name()) is not None
        sec_type += [istype]*sec.nseg
    return np.array(sec_type)

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

if __name__ == '__main__':

    tstop=50
    
    fir = hp_fir(401, 800., h.dt)
    
    initialize()
    t, I = integrate(tstop)
    c = get_coords()
    seg_coords = get_seg_coords()
    
    selection = filter_sections("(dend)")

    pos = (0, 2500,0)
    x0, y0, z0= pos
       
    v_ext = calc_lsa(pos, seg_coords[selection], I[:, selection])
    
    fig = plt.figure()
    ax = plt.subplot(111)
    S = np.sqrt(np.pi)*c['diam']*c['L']
    plot_neuron(seg_coords, np.log(np.abs(I*S).max(0)), cmap=cm.jet)
    plt.plot([x0], [y0], 'ro')
    
    plt.figure()
    plt.plot(t, fir(calc_lsa(pos, seg_coords, I)), 'b--')
    plt.plot(t, fir(calc_lsa(pos, seg_coords[selection], I[:, selection])), 'b-')
    plt.plot(t, fir(calc_lsa(pos, seg_coords[~selection], I[:, ~selection])), 'r-')
    
    plt.figure()
    isnode = filter_sections("node")
    plt.plot(I[:, isnode][np.array([1080, 1160]),:].T)
    
    plt.figure()
    plt.plot(t, fir(calc_v_ext(pos, c[selection], I[:, selection])), 'r-')
    plt.plot(t, fir(v_ext), 'b-')
   
    #plt.figure()
    #contour_p2p(seg_coords[selection], I[:, selection])
   
    plt.show()
