#!/usr/bin/env python
#coding=utf-8

import sitecustomize
import neuron
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, collections, transforms
from scipy.interpolate import griddata
from functools import partial
import scipy.signal
import re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

h = neuron.h

def hp_fir(N, cutoff, dt):
    Fs = 1./(dt*1.E-3)
    h = scipy.signal.firwin(N, cutoff/Fs/2., pass_zero=False)
    def _filter_func(s):
        filt_sig = scipy.signal.filtfilt(h, [1], s)
        return filt_sig
    return _filter_func
        

def insert_extracellular():
    for sec in h.allsec():
        sec.insert("extracellular")

def get_v():
    v = []
    for sec in h.allsec():
        for seg in sec:
            v.append(seg.v)
    return v

def get_i():
    v = []
    for sec in h.allsec():
        for seg in sec:
            v.append(seg.i_membrane)
    return v

def get_nsegs():
    nsegs = 0
    for sec in h.allsec():
        nsegs += sec.nseg
    return nsegs

def filter_sections(type):
    sec_type = []
    for sec in h.allsec():
        istype = re.match(type, sec.name()) is not None
        sec_type += [istype]*sec.nseg
    return np.array(sec_type)

def get_coords():
    total_segs = get_nsegs()
    coords = np.zeros(total_segs,
                      dtype=[("x", np.float32),
                             ("y", np.float32),
                             ("z", np.float32),
                             ("L", np.float32),
                             ("diam", np.float32)
                            ])
    j = 0
    for sec in h.allsec():
        n3d = int(h.n3d(sec))
        x = np.array([h.x3d(i,sec) for i in range(n3d)])
        y = np.array([h.y3d(i,sec) for i in range(n3d)])
        z = np.array([h.z3d(i,sec) for i in range(n3d)])
        nseg = sec.nseg
        pt3d_x = np.arange(n3d)
        seg_x = np.arange(nseg)+0.5

        if len(pt3d_x)<1:
            x_coord = y_coord = z_coord =np.ones(nseg)*np.nan
        else:
            x_coord = np.interp(seg_x, pt3d_x, x)
            y_coord = np.interp(seg_x, pt3d_x, y)
            z_coord = np.interp(seg_x, pt3d_x, z)
      
        lengths = np.zeros(nseg)
        diams = np.zeros(nseg)
        
        lengths = np.ones(nseg)*sec.L*1./nseg
        diams = np.ones(nseg)*sec.diam

        coords['x'][j:j+nseg]=x_coord
        coords['y'][j:j+nseg]=y_coord
        coords['z'][j:j+nseg]=z_coord
        coords['L'][j:j+nseg]=lengths
        coords['diam'][j:j+nseg]=diams

        j+=nseg


    return coords

def get_seg_coords():
    total_segs = get_nsegs()
    coords = np.zeros(total_segs,
                      dtype=[("x0", np.float32),
                             ("y0", np.float32),
                             ("z0", np.float32),
                             ("x1", np.float32),
                             ("y1", np.float32),
                             ("z1", np.float32),
                             ("L", np.float32),
                             ("diam", np.float32)
                            ])
    j = 0
    for sec in h.allsec():
        n3d = int(h.n3d(sec))
        x = np.array([h.x3d(i,sec) for i in range(n3d)])
        y = np.array([h.y3d(i,sec) for i in range(n3d)])
        z = np.array([h.z3d(i,sec) for i in range(n3d)])

        arcl = np.sqrt(np.diff(x)**2+np.diff(y)**2+np.diff(z)**2)
        arcl = np.cumsum(np.concatenate(([0], arcl)))
        nseg = sec.nseg
        pt3d_x = arcl/arcl[-1]
        diams = np.ones(nseg)*sec.diam
        lengths = np.ones(nseg)*sec.L*1./nseg
      
        seg_x = np.arange(nseg)*1./nseg
        
        x_coord = np.interp(seg_x, pt3d_x, x)
        y_coord = np.interp(seg_x, pt3d_x, y)
        z_coord = np.interp(seg_x, pt3d_x, z)
      

        coords['x0'][j:j+nseg]=x_coord
        coords['y0'][j:j+nseg]=y_coord
        coords['z0'][j:j+nseg]=z_coord
        
        seg_x = (np.arange(nseg)+1.)/nseg

        x_coord = np.interp(seg_x, pt3d_x, x)
        y_coord = np.interp(seg_x, pt3d_x, y)
        z_coord = np.interp(seg_x, pt3d_x, z)
        
        coords['x1'][j:j+nseg]=x_coord
        coords['y1'][j:j+nseg]=y_coord
        coords['z1'][j:j+nseg]=z_coord

        coords['diam'][j:j+nseg] = diams
        coords['L'][j:j+nseg] = lengths

        j+=nseg


    return coords


def initialize():
    #insert_extracellular()
    h.fcurrent()

def integrate(tstop):
    V_all = []
    while h.t < tstop:
        h.fadvance()
        v= get_i()
        V_all.append(v)
    t = np.arange(0, len(V_all))*h.dt
    return t, np.array(V_all)

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
    l = _vlen(pt1-pt2)
    #l = coord['L']
    assert (r>=0).all()
    I = I*1.E4*np.pi*diam*1E-6
    C = 1./(4*np.pi)*eta
    v_ext = C*I*np.log(np.abs(r**2+(d+l)**2)/np.abs(r**2+d**2))/2.
    #v_ext = C*I*np.log(np.abs(np.sqrt(d**2+r)-d)/np.abs(np.sqrt((l+d)**2+r**2)-l-d))
    v_ext[np.isnan(v_ext)] = 0
    v_ext = v_ext*1E6 # nV
    return v_ext.sum(1) 

def plot_neuron(coords, scalar, cmap=cm.hot):
   
    a = plt.gca()
    line_segs = [[(c['x0'], c['y0']), (c['x1'], c['y1'])] for c in
                 coords]
    colors = cmap(plt.normalize()(scalar))
    col = collections.LineCollection(line_segs)
    a.add_collection(col, autolim=True)
    col.set_color(colors)
    a.autoscale_view()
    plt.axis('equal')

def contour_p2p(coords, I, xrange=(-4000, 4000), yrange=(-4000, 4000),
               z=0):
    xmin, xmax = xrange
    ymin, ymax = yrange
    

    n_x = n_y = 20
    x = np.linspace(xmin, xmax, n_x)
    y = np.linspace(ymin, ymax, n_y)

    XX, YY = np.meshgrid(x, y)

    p2p = np.zeros(XX.shape)

    for i in range(n_x):
        for j in range(n_y):
            v_ext = calc_lsa((XX[i,j], YY[i,j], z), coords, I)
            p2p[i,j] = np.log(v_ext.max() - v_ext.min())

    plt.contour(XX, YY, p2p)

if __name__ == '__main__':

    h.load_file("demo_ext.hoc")
    tstop=50
    h.dt = 0.025
    

    fir = hp_fir(401, 800., h.dt)
    
    initialize()
    t, I = integrate(tstop)
    c = get_coords()
    seg_coords = get_seg_coords()
    
    selection = filter_sections("(node)|(myelin)")

    #l1 = c['L']
    #pt1 = np.vstack((seg_coords['x0'], seg_coords['y0'], seg_coords['z0']))
    #pt2 = np.vstack((seg_coords['x1'], seg_coords['y1'], seg_coords['z1']))
    #l2 = np.sqrt(np.sum((pt1-pt2)**2,0))
    #
    #plt.plot((l2-l1)/l1)
    #plt.show()
 
    pos = (0, 2500,0)
    x0, y0, z0= pos
       
    v_ext = calc_lsa(pos, seg_coords[selection], I[:, selection])
    
    fig = plt.figure()
    ax = plt.subplot(111)
    S = np.sqrt(np.pi)*c['diam']*c['L']
    plot_neuron(seg_coords, np.log(np.abs(I*S).max(0)), cmap=cm.jet)
    #print isdend
    #plt.plot([seg_coords['y0'], seg_coords['y1']],
    #         [seg_coords['x0'], seg_coords['x1']],
    #         color=np.log(np.abs(I*S).max(0)))
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
