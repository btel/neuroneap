#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, collections, transforms, colors, ticker
import field

def plot_neuron(coords, scalar=None, colors=None,
                norm=colors.Normalize(), cmap=cm.jet):
   
    a = plt.gca()
    line_segs = [[(c['x0'], c['y0']), (c['x1'], c['y1'])] for c in
                 coords]

    col = collections.LineCollection(line_segs, cmap=cmap, norm=norm)
    a.add_collection(col, autolim=True)
    if scalar is not None:
        col.set_array(scalar)
    else:
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

def plot_multiplies(xx, yy, vv, t=None, w=0.1, h=0.1):
    """Plot small mutiplies of potential  on a grid
    
    * w, h -- multiplies width/height in axes coords.
    
    Notes:
    You have to make sure that the main axis limits are set correctly
    prior to plotting."""

    if t is None:
        t = np.arange(vv.shape[0])
    fig = plt.gcf() 
    ax = plt.gca()
    
    # calc transform for inset placement
    transDataToFigure = (ax.transData+fig.transFigure.inverted())
    ax_inset = []
    last_inset = None
    nx, ny = xx.shape
    for i in range(nx):
        for j in range(ny):
            x, y = transDataToFigure.transform((xx[i,j], yy[i,j]))
            last_inset = fig.add_axes([x-w/2., y-h/2., w, h], frameon=False,
                              sharey=last_inset, sharex=last_inset)
            plt.plot(t, vv[:,i,j], 'k-')
            plt.xticks([])
            plt.yticks([])
            ax_inset.append(last_inset)
    return ax_inset
