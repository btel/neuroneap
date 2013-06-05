#!/usr/bin/env python
#coding=utf-8

import numpy as np

from eap import cell
from neuron import h

def test_current_balance_cable():

    h('vinit=-65')
    h('create cable')
    h('access cable')
    h('objref synapse')
    h('nseg = 10')
    h('insert extracellular')
    h('insert pas')
    h('e_pas=-65')
    h('synapse = new AlphaSynapse(0.1)')
    h('synapse.onset=0')
    h('synapse.tau=1')
    h('synapse.e=-100')
    h('synapse.gmax=0.00001')
    cell.initialize(dt=0.025)
    t, I = cell.integrate(1)
    assert (np.abs(I.sum(1))<1e-12).all()
