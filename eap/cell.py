#!/usr/bin/env python
#coding=utf-8

import neuron
import numpy as np

h = neuron.h

def integrate(tstop, i_axial=False):
    """Run Neuron simulation and return time and 2d array of 
    transmembrane currents (n_pts x n_segs). If i_axial is true
    return in addition axial currents"""
    i_membrane_all = []
    i_axial_all = [] 
    while h.t < tstop:
        h.fadvance()
        v = get_i_membrane()
        i_membrane_all.append(v)
        if i_axial:
            iax = get_i_axial()
            i_axial_all.append(iax)
    t = np.arange(0, len(i_membrane_all))*h.dt
    if i_axial:
        return t, np.array(i_membrane_all), np.array(i_axial_all)
    else:
        return t, np.array(i_membrane_all)

def insert_extracellular():
    for sec in h.allsec():
        sec.insert("extracellular")

def get_v():
    v = []
    for sec in h.allsec():
        for seg in sec:
            v.append(seg.v)
    return v

def get_i_membrane():
    v = []
    for sec in h.allsec():
        i_sec = [seg.i_membrane for seg in sec]
        x = [seg.x for seg in sec]
        #add currents from point processes at the beginning and end of section
        c_factor = 100 #from  [i/area]=nA/um2 to [i_membrane]=mA/cm2
        area0 = h.area(x[0], sec=sec)
        area1 = h.area(x[-1], sec=sec)
        i_sec[0] += sum(pp.i for pp in sec(0).point_processes())/area0*c_factor
        i_sec[-1] += sum(pp.i for pp in sec(1).point_processes())/area1*c_factor
        v += i_sec
    return v

def get_i_axial():
    """return axial current density in mA/cm2"""
    currents = []
    for sec in h.allsec():
        v0 = sec(0).v
        for seg in sec:
            v1 = seg.v
            l = sec.L/sec.nseg #length in um
            r = sec.Ra #resistance in ohm*cm
            iax = (v1-v0)/(r*l*1e-4)
            currents.append(iax)
            v0 = v1

    return currents

def get_nsegs():
    nsegs = 0
    for sec in h.allsec():
        nsegs += sec.nseg
    return nsegs


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
    nchars = 12
    total_segs = get_nsegs()
    coords = np.zeros(total_segs,
                      dtype=[("x0", np.float32),
                             ("y0", np.float32),
                             ("z0", np.float32),
                             ("x1", np.float32),
                             ("y1", np.float32),
                             ("z1", np.float32),
                             ("L", np.float32),
                             ("diam", np.float32),
                             ("name", "|S%d" % nchars)
                            ])
    j = 0
    for sec in h.allsec():
        nseg = sec.nseg
        
        diams = np.ones(nseg)*sec.diam
        lengths = np.ones(nseg)*sec.L*1./nseg
        names = np.repeat(sec.name()[:nchars], nseg).astype("|S%d"%nchars)

        seg_x = np.arange(nseg+1)*1./nseg

        x_coord, y_coord, z_coord = get_locs_coord(sec, seg_x)
      

        coords['x0'][j:j+nseg] = x_coord[:-1]
        coords['y0'][j:j+nseg] = y_coord[:-1]
        coords['z0'][j:j+nseg] = z_coord[:-1]
        
        coords['x1'][j:j+nseg] = x_coord[1:]
        coords['y1'][j:j+nseg] = y_coord[1:]
        coords['z1'][j:j+nseg] = z_coord[1:]

        coords['diam'][j:j+nseg] = diams
        coords['L'][j:j+nseg] = lengths
        coords['name'][j:j+nseg] = names 
        j+=nseg

    return coords

def get_locs_coord(sec, loc):
    """get 3d coordinates of section locations"""
    n3d = int(h.n3d(sec))
    x = np.array([h.x3d(i,sec) for i in range(n3d)])
    y = np.array([h.y3d(i,sec) for i in range(n3d)])
    z = np.array([h.z3d(i,sec) for i in range(n3d)])

    arcl = np.sqrt(np.diff(x)**2+np.diff(y)**2+np.diff(z)**2)
    arcl = np.cumsum(np.concatenate(([0], arcl)))
    nseg = sec.nseg
    pt3d_x = arcl/arcl[-1]
        
    x_coord = np.interp(loc, pt3d_x, x)
    y_coord = np.interp(loc, pt3d_x, y)
    z_coord = np.interp(loc, pt3d_x, z)

    return x_coord, y_coord, z_coord

def get_pp_coord(point_process):
    """get 3d coordinates of a point process such as synapse:
        point_process -- neuron object"""

    loc = point_process.get_loc()
    sec = h.cas()
    coord = get_locs_coord(sec, loc)
    h.pop_section()
    return coord

def get_point_processes():
    """Returns record array with all point processes and their
    coordinates as tuple.
    """
    point_processes = []
    for sec in h.allsec():
        pp_in_sec = []
        for seg in sec.allseg():
            pp_in_sec += seg.point_processes()
        locs = [pp.get_loc() for pp in pp_in_sec]
        #remove section from stack to avoid overflow
        [h.pop_section() for pp in pp_in_sec]
        x, y, z = get_locs_coord(sec, locs)
        point_processes += zip(pp_in_sec, x, y, z)
    return point_processes

def initialize(dt=0.025):
    #insert_extracellular()
    h.finitialize()
    h.dt = dt
    h.fcurrent()
    h.frecord_init()

def load_model(hoc_name, dll_name=None):
    if dll_name:
        h.nrn_load_dll(dll_name)
    h.load_file(hoc_name)

