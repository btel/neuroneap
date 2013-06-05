#!/usr/bin/env python
#coding=utf-8

import numpy as np

from eap import cell
from neuron import h
from nose import with_setup
def configure_cell():
    h('vinit=-65')
    h('create cable')
    h('access cable')
    h('nseg = 10')
    h('insert extracellular')
    h('insert pas')
    h('e_pas=-65')

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

