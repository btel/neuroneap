#!/usr/bin/env python
#coding=utf-8

import numpy as np

from eap import cell
from neuron import h
from nose import with_setup
def configure_cell():
    h('vinit=-65')
    h('create cable')
    h('create soma')
    h('connect soma(1), cable(0)')
    h('access cable')
    h('nseg = 10')
    h('L = 10')
    h('forall insert extracellular')
    h('define_shape()')

configure_cell()

def add_synapse():
    h('objref synapse')
    h('cable synapse = new AlphaSynapse(0.1)')
    h('synapse.onset=0')
    h('synapse.tau=1')
    h('synapse.e=-100')
    h('synapse.gmax=0.00001')

def reset_cell():
    h('objref synapse')
    h('access cable')
    h.t = 0

def total_current(coord, I):
    return (I[:,:]*(coord['diam']*coord['L']*h.PI)[None,:]).sum(1)
@with_setup(add_synapse, reset_cell)
def test_current_balance_synapse_in_segment():
    h('synapse.loc(0.1)')
    h('soma {insert pas}')
    cell.initialize(dt=0.025)
    t, I = cell.integrate(1)
    coord = cell.get_seg_coords()
    assert (np.abs(total_current(coord, I))<1e-6).all()

@with_setup(add_synapse, reset_cell)
def test_current_balance_synapse_at_section_end():

    h('synapse.loc(0)')
    cell.initialize(dt=0.025)
    t, I = cell.integrate(1)
    coord = cell.get_seg_coords()
    assert (np.abs(total_current(coord, I))<1e-6).all()

def test_location_coord():

    x, y, z = cell.get_locs_coord(h.cable, 0.15)
    assert x == 1.5
    assert y == 0
    assert z == 0

@with_setup(add_synapse, reset_cell)
def test_point_process_coord():
    h('synapse.loc(0.15)')
    h('access soma')
    x, y, z = cell.get_pp_coord(h.synapse)
    assert x == 1.5
    assert y == 0
    assert z == 0

@with_setup(add_synapse, reset_cell)
def test_get_point_processes():
    h('synapse.loc(0.15)')
    h('access soma')
    pps = cell.get_point_processes()
    assert len(pps)==1
    assert pps[0][0].same(h.synapse)
    assert pps[0][1] == 1.5

def test_axial_currents():
    h('soma {insert pas}')
    isim = h.IClamp(1, sec=h.cable)
    isim.delay = 0
    isim.dur = 1
    isim.amp = 2 #nA
    h.cable.Ra = 0.1
    cell.initialize(0.1)
    t, imem_density, iax_density = cell.integrate(0.2, i_axial=True)
    coord = cell.get_seg_coords()
    iax = iax_density*coord['diam'][None,:]**2*1e-8/4.*h.PI #mA
    
    assert np.abs(iax[-1,1]*1e6 - isim.amp)<0.1*isim.amp
