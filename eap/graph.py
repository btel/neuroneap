#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, collections, transforms

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

def contour_p2p(coords, I, xrange=(-4000, 4000), yrange=(-4000, 4000),
               z=0):
    xmin, xmax = xrange
    ymin, ymax = yrange
    

    n_x = n_y = 20
    x = np.linspace(xmin, xmax, n_x)
    y = np.linspace(ymin, ymax, n_y)

    XX, YY = np.meshgrid(x, y)

    p2p = np.zeros(XX.shape)

    for i in range(n_x):
        for j in range(n_y):
            v_ext = calc_lsa((XX[i,j], YY[i,j], z), coords, I)
            p2p[i,j] = np.log(v_ext.max() - v_ext.min())

    plt.contour(XX, YY, p2p)


