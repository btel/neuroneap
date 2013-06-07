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
    h('access cable')
    h('nseg = 10')
    h('L = 10')
    h('insert pas')
    h('e_pas=-65')
    h('forall insert extracellular')
    h('define_shape()')

configure_cell()

def add_synapse():
    h('objref synapse')
    h('synapse = new AlphaSynapse(0.1)')
    h('synapse.onset=0')
    h('synapse.tau=1')
    h('synapse.e=-100')
    h('synapse.gmax=0.00001')

def reset_cell():
    h('objref synapse')
    h('access cable')
    h.t = 0

@with_setup(add_synapse, reset_cell)
def test_current_balance_synapse_in_segment():

    h('synapse.loc(0.1)')
    cell.initialize(dt=0.025)
    t, I = cell.integrate(1)
    assert (np.abs(I.sum(1))<1e-12).all()

@with_setup(add_synapse, reset_cell)
def test_current_balance_synapse_at_section_end():

    h('synapse.loc(0)')
    cell.initialize(dt=0.025)
    t, I = cell.integrate(1)
    assert (np.abs(I.sum(1))<1e-12).all()

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
