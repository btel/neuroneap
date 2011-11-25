#!/usr/bin/env python
#coding=utf-8

import neuron
h = neuron.h
h.load_file("demo_ext.hoc")

from calc_csd import get_seg_coords, initialize, integrate

coords = get_seg_coords()
import numpy as np

def get_max_current(c):
    tstop=50
    h.dt = 0.025
    
    initialize()
    t, I = integrate(tstop)
    L = np.sqrt((c['x0']-c['x1'])**2+
                (c['y0']-c['y1'])**2+
                (c['z0']-c['z1'])**2
               )
    S = np.sqrt(np.pi)*c['diam']*L
    i_max = np.abs(I*S).max(0)
    i_norm = np.log(i_max)
    i_norm -= i_norm[~np.isinf(i_norm)].min()
    i_norm[np.isinf(i_norm)] = 0
    i_norm = i_norm/i_norm.max()
    print i_norm.min(), i_norm.max()
    return i_norm

# The number of points per line
N = 300

# The scalar parameter for each line
t = np.linspace(-2*np.pi, 2*np.pi, N)

from enthought.mayavi import mlab
mlab.figure(1, size=(400, 400), bgcolor=(0, 0, 0))
mlab.clf()

# We create a list of positions and connections, each describing a line.
# We will collapse them in one array before plotting.

pts = np.zeros(2*len(coords),
                  dtype=[("x", np.float32),
                         ("y", np.float32),
                         ("z", np.float32)
                        ])
pts['x'][:] = np.concatenate((coords['x0'], coords['x1']))
pts['y'][:] = np.concatenate((coords['y0'], coords['y1']))
pts['z'][:] = np.concatenate((coords['z0'], coords['z1']))

idx = np.arange(len(pts))

pts = np.unique(pts)

coords1 = np.zeros(len(coords), 
                  dtype=[("x", np.float32),
                         ("y", np.float32),
                         ("z", np.float32)
                        ])
coords2 = np.zeros(len(coords), 
                  dtype=[("x", np.float32),
                         ("y", np.float32),
                         ("z", np.float32)
                        ])
coords1['x'] = coords['x0']
coords1['y'] = coords['y0']
coords1['z'] = coords['z0']

coords2['x'] = coords['x1']
coords2['y'] = coords['y1']
coords2['z'] = coords['z1']

i_coord = get_max_current(coords)

connections = []
scalars = np.zeros(len(pts))
currents = np.zeros(len(pts))

for i in range(len(coords1)):
    node1, node2 = (coords1[i], coords2[i])
    i1, = np.where(pts==node1)
    i2, = np.where(pts==node2)
    connections.append([i1[0], i2[0]])
    scalars[i1] = coords['diam'][i] 
    currents[i1] = i_coord[i]

connections = np.array(connections)

scalars = scalars-scalars.min()
scalars = scalars/scalars.max()
# Create the points
#src = mlab.pipeline.scalar_scatter(x, y, z)
src = mlab.points3d(pts['x'], pts['y'], pts['z'],
                    scalars,
                                    scale_factor=0.0, resolution=10)

#current = np.random.rand(len(scalars))
#add new attribute
src.mlab_source.dataset.point_data.add_array(currents)
# We need to give a name to our new dataset.
src.mlab_source.dataset.point_data.get_array(1).name = 'current'
# Make sure that the dataset is up to date with the different arrays:
src.mlab_source.dataset.point_data.update()


# Connect them
src.mlab_source.dataset.lines = connections

# The stripper filter cleans up connected lines
tubes = mlab.pipeline.tube(src, tube_radius=0.5)
tubes.filter.radius_factor = 10.
tubes.filter.vary_radius = 'vary_radius_by_scalar'

lines = mlab.pipeline.stripper(tubes)

# Finally, display the set of lines
lines2 = mlab.pipeline.set_active_attribute(lines,
                                    point_scalars='current')
mlab.pipeline.surface(lines2)

# And choose a nice view
mlab.view(33.6, 106, 5.5, [0, 0, .05])
mlab.roll(125)
mlab.show()

