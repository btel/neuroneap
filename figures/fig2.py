#!/usr/bin/env python
#coding=utf-8
r"""Differential contributions of distinct neuronal structures. 
(\textbf{{A}}) Morphology of the neuron with color-coded structures:
dendrites, soma, axon hillock (hill), axon initial segment (AIS)
and axon. Axon was truncated for clarity. (\textbf{{B}}) Contribution of
different neuronal structures  to extracellular potential at distance
of $d$={dist}~mm  above neuron (angle $\alpha$={alpha}) in homogeneous
medium (resistivity $\rho$={rho} $\Omega\cdot m$). For color legend
see (A). (\textbf{{C}}) Comparision of contributions of these
structures to high-frequency potential (cutoff {cutoff} Hz).
"""

import numpy as np
import matplotlib.pyplot as plt

from eap import field, cell, graph

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
                    'models/Mainen/i686/.libs/libnrnmech.so')
cell.initialize(dt=dt)
t, I = cell.integrate(tstop)
coords = cell.get_seg_coords()

# Select segments
dend = field.select_sections(coords, "dend")
soma = field.select_sections(coords, "soma")
axon = field.select_sections(coords, "(node)|(myelin)")
iseg = field.select_sections(coords, "iseg")
hill = field.select_sections(coords, "hill")
all  = field.select_sections(coords, ".*")

colors = {"dend": "r",
          "soma": "c",
          "axon": "b",
          "all" : "k",
          "iseg" : "g",
          "hill" : "m"}

# Calculation of field
v_dend = field.estimate_lsa(pos, coords[dend], I[:, dend])
v_soma = field.estimate_lsa(pos, coords[soma], I[:, soma])
v_axon = field.estimate_lsa(pos, coords[axon], I[:, axon])
v_iseg = field.estimate_lsa(pos, coords[iseg], I[:, iseg])
v_hill = field.estimate_lsa(pos, coords[hill], I[:, hill])

# PLOTS
fig = plt.figure()
fig.subplots_adjust(hspace=0.15, wspace=0.2, left=0.05, right=0.95)
ax1= fig.add_subplot(1,2,1, frameon=False)
l_dend = graph.plot_neuron(coords[dend], colors=colors['dend'])
l_axon = graph.plot_neuron(coords[axon], colors=colors['axon'])
l_iseg = graph.plot_neuron(coords[iseg], colors=colors['iseg'])
l_hill = graph.plot_neuron(coords[hill], colors=colors['hill'])
l_soma = graph.plot_neuron(coords[soma], colors=colors['soma'])
plt.legend((l_dend, l_axon, l_iseg, l_hill, l_soma), 
           ('dendrites', 'axon', 'AIS','hill', 'soma'),
           frameon=False,
           bbox_transform=ax1.transAxes,
           bbox_to_anchor=(0.15, 0.1, 0.3, 0.3))

xp, yp = -300, 350
w, h = 100, 100
plt.plot([xp, xp], [yp, yp+h], 'k-')
plt.plot([xp, xp+h], [yp, yp], 'k-')
plt.text(xp-10, yp+h/2., u"100 µm", ha='right', va='center',
         transform=ax1.transData)
plt.text(xp+h/2., yp-10, u"100 µm", ha='center', va='top',
         transform=ax1.transData)

plt.xlim([-450, 10])
plt.ylim([-170, 600])
plt.xticks([])
plt.yticks([])
ax1.text(0.05, 0.95, 'A', weight='bold',
         transform=ax1.transAxes)

ax2 = fig.add_subplot(2,2,2)
plt.plot(t, v_dend, colors['dend'])
plt.plot(t, v_axon, colors['axon'])
plt.plot(t, v_iseg, colors['iseg'])
plt.plot(t, v_hill, colors['hill'])
plt.plot(t, v_soma, colors['soma'])
plt.xlim([20, 40])
plt.ylabel("EAP (nV)")
ax2.text(0.05, 0.9, 'B', weight='bold',
         transform=ax2.transAxes)

ax3 = fig.add_subplot(2,2,4)
plt.plot(t, fir(v_dend), colors['dend'])
plt.plot(t, fir(v_axon), colors['axon'])
plt.plot(t, fir(v_iseg), colors['iseg'])
plt.plot(t, fir(v_hill), colors['hill'])
plt.plot(t, fir(v_soma), colors['soma'])
plt.xlim([20, 40])
plt.ylabel("high-frequency EAP (nV)")
ax3.text(0.05, 0.9, 'C', weight='bold',
         transform=ax3.transAxes)
plt.xlabel('time (ms)')

print __doc__.format(**vars())
plt.show()

