#!/usr/bin/env python
#coding=utf-8

from eap import field
import numpy as np
from numpy.testing import assert_almost_equal

L=1.
diam=1.
I_0=1.
eta    = 1.

def conf_cylinder(pos=(0,0,0), theta=0, L=L):
    x0, y0, z0 = pos
    coord = np.zeros((1,),  dtype=[("x0", np.float32),
                            ("y0", np.float32),
                            ("z0", np.float32),
                            ("x1", np.float32),
                            ("y1", np.float32),
                            ("z1", np.float32),
                            ("L", np.float32),
                            ("diam", np.float32)
                            ])
    coord[0] = (x0,y0,z0,x0+np.cos(theta)*L,y0+np.sin(theta)*L,z0,L,diam)

    I = np.zeros((1,1))
    I[0] = I_0

    return coord, I

def test_lsa_cylinder_longl():
    x_pos  = -1.
    pos = (x_pos, 0, 0)
    coord, I = conf_cylinder()
    v = field.estimate_lsa(pos, coord, I, eta=eta)
    v_analytical = 1e6/(4*np.pi) * eta*I_0*1E4*(np.pi*diam*1E-6) * np.log((L-x_pos)/L)
    np.testing.assert_almost_equal(v[0], v_analytical)

def test_lsa_cylinder_radial():
    y_pos = 1.
    pos = (0, y_pos, 0)
    coord, I = conf_cylinder()
    v = field.estimate_lsa(pos, coord, I, eta=eta)
    v_analytical = (1e6/(4*np.pi) * eta*I_0*1E4*(np.pi*diam*1E-6) *
                    np.log((y_pos**2+L**2)/y_pos**2))/2.

    np.testing.assert_almost_equal(v[0], v_analytical)

def test_lsa_cylinder_radial_inv():
    y_pos = 1.
    pos = (0, 1., 0)
    coord, I = conf_cylinder((-1.,0.,0.))
    v = field.estimate_lsa(pos, coord, I, eta=eta)
    v_analytical = (1e6/(4*np.pi) * eta*I_0*1E4*(np.pi*diam*1E-6) *
                    np.log((y_pos**2+L**2)/y_pos**2))/2.

    np.testing.assert_almost_equal(v[0], v_analytical)
    
def test_lsa_cylinder_far():
    y_pos = 1E8
    pos = (0., y_pos, 0)
    coord, I = conf_cylinder()
    v = field.estimate_lsa(pos, coord, I, eta=eta)

    assert_almost_equal(v[0], 0)

def test_lsa_symmetric_cylinders():
    coord1, I1 = conf_cylinder()
    coord2, I2 = conf_cylinder((-L, 0, 0))
    
    coord = np.hstack((coord1, coord2))
    I = np.hstack((I1, -I2))
    print coord

    pos = (0, 10, 0)
    v = field.estimate_lsa(pos, coord, I, eta=eta)

    assert_almost_equal(v[0], 0) 

def test_lsa_symmetry():
    coord1, I1 = conf_cylinder((-L/2.,0,0))
    coord2, I2 = conf_cylinder((0, -L/2, 0), theta=np.pi/2.)
    
    coord = np.hstack((coord1, coord2))
    I = np.hstack((I1, I2))

    pos = (1., 1., 0)
    v1 = field.estimate_lsa(pos, coord, I, eta=eta)
    pos = (-1., 1., 0)
    v2 = field.estimate_lsa(pos, coord, I, eta=eta)
    pos = (1., -1., 0)
    v3 = field.estimate_lsa(pos, coord, I, eta=eta)
    pos = (-1., -1., 0)
    v4 = field.estimate_lsa(pos, coord, I, eta=eta)

    assert_almost_equal(v1, v2) 
    assert_almost_equal(v2, v3) 
    assert_almost_equal(v3, v4) 

def test_lsa_cylinder_divide():

    pos = np.random.randn(3)*10
    theta = np.random.rand()*np.pi 
    coord_total, I = conf_cylinder(theta=theta)
    
    N = 10
    dl = np.array([L*1.], dtype=np.float64)/N
    first, _= conf_cylinder((0, 0, 0), theta=theta, L=dl)
    short_cylinders = [first]
    new = first
    for i in range(N-1):
        new,_ = conf_cylinder((new['x1'][0], new['y1'][0],
                               new['z1'][0]), theta=theta, L=dl)
        short_cylinders.append(new)

    coord_division = np.hstack(short_cylinders)

    I_div = np.ones((1,len(coord_division)))*I_0
    v_division = field.estimate_lsa(pos, coord_division, I_div)
    v_total = field.estimate_lsa(pos, coord_total, I)

    assert_almost_equal(v_total, v_division, decimal=4)

def test_lsa_tripole_cylinder():
    L = 1.
    cable = [conf_cylinder((i*L, 0, 0), L=L)[0] for i in range(3)]
    cable = np.hstack(cable)
    I = np.zeros((1, 3))
    I[0,0] = I_0
    I[0,1:] = -I_0/2.

    v1 = field.estimate_lsa((-1.5, 7, 0), cable, I)
    v2 = field.estimate_lsa((5, 7, 0), cable, I)

    show_potential_on_grid(cable, I)
    assert v1>0, "Potential is negative"
    assert v2<0, "Potential is positive"

def show_potential_on_grid(cable, I):
    import matplotlib.pyplot as plt
    xx, yy = field.calc_grid([-10,10], [-10,10], 10)
    v_ext = field.estimate_on_grid(cable, I, xx, yy)
    plt.contour(xx, yy, v_ext[0,:,:])
    plt.show()



