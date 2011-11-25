#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, collections, transforms, colors, ticker
import field

def plot_neuron(coords, scalar=None, colors=None, cmap=cm.jet):
   
    a = plt.gca()
    line_segs = [[(c['x0'], c['y0']), (c['x1'], c['y1'])] for c in
                 coords]

    if colors is None:
        if scalar is None:
            colors = [(0,0,0)] # all black
        else:
            colors = cmap(plt.normalize()(scalar))

    col = collections.LineCollection(line_segs)
    a.add_collection(col, autolim=True)
    col.set_color(colors)
    a.autoscale_view()
    plt.axis('equal')
    return col

def logcontour(xx, yy, zz, n_contours=10):
    
    v = np.logspace(np.log10(np.min(zz[:])),
                    np.log10(np.max(zz[:])), n_contours)
    lev_exp = np.arange(np.floor(np.log10(v.min())-1),
                           np.floor(np.log10(v.max())+1))
    
    levs = np.power(10, lev_exp)*np.array([1, 2, 5])[:, np.newaxis]
    levs = np.hstack(levs).astype(int)
   
    def pow_fmt(q, m):
        if (m < 2) and (m > 0):
            return "%d" % (10**m * q)
        if q == 1:
            return  r"$10^{%d}$" % (m,)
        else:
            return r"$%d\cdot10^{%d}$" % (q,m)


    fmt = [ pow_fmt(q,m) for q in [1,2,5] for m in lev_exp]

    fmt = dict(zip(levs, fmt))

    cs = plt.contour(xx, yy, zz, levs, norm=colors.LogNorm() )
    plt.clabel(cs, cs.levels, fmt=fmt, inline=1)

def spike_multiplies():
    pass
