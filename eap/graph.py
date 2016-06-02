#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, collections, transforms, colors, ticker
import field

def plot_neuron(coords, scalar=None, colors=None,
                norm=colors.Normalize(), cmap=cm.jet, show_diams=False, width_min=0.5, width_max=5):
   
    a = plt.gca()
    line_segs = [[(c['x0'], c['y0']), (c['x1'], c['y1'])] for c in
                 coords]

    if show_diams:
        diams = coords['diam']
        widths = (diams - diams.min()) / (diams.max() - diams.min())
        widths *= (width_max - width_min)
        widths += width_min
    else:
        widths = None
    
    col = collections.LineCollection(line_segs, cmap=cmap, norm=norm, linewidths=widths)
    a.add_collection(col, autolim=True)
    if scalar is not None:
        col.set_array(scalar)
    else:
        col.set_color(colors)

    a.autoscale_view()
    plt.axis('equal')
    return col

def logcontour(xx, yy, zz, n_contours=10, linecolors=None, linewidths=None, unit=''):
    
    v = np.logspace(np.log10(np.min(zz[:])),
                    np.log10(np.max(zz[:])), n_contours)
    lev_exp = np.arange(np.floor(np.log10(v.min())-1),
                           np.floor(np.log10(v.max())+1))
    
    levs = np.power(10, lev_exp)*np.array([1, 2, 5])[:, np.newaxis]
    levs = np.hstack(levs).astype(int)
    levs.sort()
   
    def pow_fmt(q, m, unit=unit):
        if (m < 2) and (m > 0):
            return r"$%d$ %s" % (10**m * q, unit)
        if q == 1:
            return  r"$10^{%d}$ %s" % (m, unit)
        else:
            return r"$%d\cdot10^{%d}$ %s" % (q, m, unit)


    fmt = [ pow_fmt(q,m) for q in [1,2,5] for m in lev_exp]

    fmt = dict(zip(levs, fmt))

    cs = plt.contour(xx, yy, zz, levs, norm=colors.LogNorm(), colors=linecolors, linewidths=linewidths)
    plt.clabel(cs, cs.levels, fmt=fmt, inline=1)

def plot_multiplies(xx, yy, vv, t=None, w=0.1, h=0.1, sharey=True):
    """Plot small mutiplies of potential  on a grid
    
    * w, h -- multiplies width/height in axes coords.
    
    Notes:
    You have to make sure that the main axis limits are set correctly
    prior to plotting."""


    def axes_locator(xx, yy):
        def __ax_loc(ax_inset, renderer):
            transDataToFigure = (ax.transData+fig.transFigure.inverted())
            x, y = transDataToFigure.transform((xx, yy))
            bbox = transforms.Bbox.from_bounds(x-w/2., y-h/2., w, h)
            return bbox
        return __ax_loc
        

    if t is None:
        t = np.arange(vv.shape[0])
    fig = plt.gcf() 
    ax = plt.gca()
    
    # calc transform for inset placement
    transDataToFigure = (ax.transData+fig.transFigure.inverted())
    ax_inset = []
    lines = []
    last_inset = None
    nx, ny = xx.shape
    for i in range(nx):
        for j in range(ny):
            x, y = transDataToFigure.transform((xx[i,j], yy[i,j]))
            last_inset = fig.add_axes([x-w/2., y-h/2., w, h], frameon=False,
                              sharey=last_inset if sharey else None, 
                              sharex=last_inset)
            last_inset.set_axes_locator(axes_locator(xx[i,j],yy[i,j]))
            l, = plt.plot(t, vv[:,i,j], 'k-')
            plt.xticks([])
            plt.yticks([])
            ax_inset.append(last_inset)
            lines.append(l)
    return lines, ax_inset
