#!/usr/bin/env python
#coding=utf-8
r"""Simulation of extracelullar field due to a single action
potential in layer 5 pyramidal neuron. Morphology of the cell and
channel dynamics adapted from Mainen and Sejnowski (1995).
(\textbf{{A}}) Position of a simulated recording electrode (red
circle) in relation to the cell (black).  Distance of the recording
site from cell $d$={dist}~mm;  angle from cell main axis
$alpha$={alpha}) (\textbf{{B}}) Extracellular field potential in a
conductive medium (conductivity $\rho$ = {rho} $\Ohm\cdot m$) due to a
single action potenatial initiated by current injection to soma.
(\textbf{{C}}) High-pass filtered extracellular potential (zero-phase
FIR filter, order {order:d}, cutoff frequency $f$={cutoff}~Hz)."""

import numpy as np
import matplotlib.pyplot as plt

from eap import field, cell, graph

import platform
ARCH = platform.uname()[4]

dt = 0.025
tstop=50

# Parameters
dist     = 2.5  #mm
rho      = 3.5  #conductivity, Ohm.m
alpha    = 0.   #angle, rad
cutoff   = 800. #high-pass cutoff, Hz
order    = 401  #filter order

# Electrode position and filter
pos = (dist*np.sin(alpha)*1000, dist*np.cos(alpha)*1000, 0)
fir = field.hp_fir(order, cutoff, dt)

# Simulation
cell.load_model('models/Mainen/demo_ext.hoc',
                'models/Mainen/%s/.libs/libnrnmech.so' % ARCH)
cell.initialize(dt=dt)
t, I = cell.integrate(tstop)

# Calculation of field
coords = cell.get_seg_coords()
v_ext = field.estimate_lsa(pos, coords, I) 

# PLOTS
fig = plt.figure()
fig.subplots_adjust(left=0.05, wspace=0)
ax1= fig.add_subplot(1,2,1, frameon=False)
graph.plot_neuron(coords)
plt.plot([pos[0]], [pos[1]], 'ro')
ax1.text(0.05, 0.9, 'A', weight='bold',
         transform=ax1.transAxes)

## scalebar
xp, yp =  -1500, -2000
w, h = 1000, 1000
ax1.plot([xp, xp], [yp, yp+h], 'k-')
ax1.plot([xp, xp+h], [yp, yp], 'k-')
ax1.text(xp-100, yp+h/2., "1 mm", ha='right', va='center',
         transform=ax1.transData)
ax1.text(xp+h/2., yp-100, "1 mm", ha='center', va='top',
         transform=ax1.transData)

plt.xlim([-3000, 3000])
plt.ylim([-3000, 3000])
plt.xticks([])
plt.yticks([])

ax2 = fig.add_subplot(2,2,2)
plt.plot(t, v_ext, 'r-')
plt.xlim([10, 40])
plt.ylabel("EAP (nV)")
ax2.text(0.05, 0.9, 'B', weight='bold',
         transform=ax2.transAxes)

ax3 = fig.add_subplot(2,2,4)
plt.plot(t, fir(v_ext), 'r-')
plt.xlim([10, 40])
plt.xlabel('time (ms)')
plt.ylabel("high-frequency EAP (nV)")
ax3.text(0.05, 0.9, 'C', weight='bold',
         transform=ax3.transAxes)

print __doc__.format(**vars())
plt.savefig('fig1.pdf')
