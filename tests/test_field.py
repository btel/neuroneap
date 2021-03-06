#!/usr/bin/env python
#coding=utf-8

from eap import field
import numpy as np
from numpy.testing import assert_almost_equal

L=1.
diam=1.
I_0=1.
eta    = 1.

def conf_cylinder(pos=(0,0,0), theta=0, L=L, I_0=I_0):
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

def concat_cylinders(coord_current):

    coords, currents = zip(*coord_current)

    coord = np.hstack(coords)
    current = np.hstack(currents)

    return coord, current

def test_lsa_cylinder_longl():
    x_pos  = -1.
    pos = (x_pos, 0, 0)
    coord, I = conf_cylinder()
    v = field.estimate_lsa(pos, coord, I, eta=eta)
    v_analytical = 1e6/(4*np.pi) * eta*I_0*1E4*(np.pi*diam*1E-6) * np.log((L-x_pos)/(-x_pos))
    np.testing.assert_almost_equal(v[0], v_analytical)

def test_lsa_cylinder_radial():
    y_pos = 1.
    pos = (0, y_pos, 0)
    coord, I = conf_cylinder()
    v = field.estimate_lsa(pos, coord, I, eta=eta)
    v_analytical = (1e6/(4*np.pi) * 
                     eta*I_0*1E4*(np.pi*diam*1E-6) *
                     np.log((L+np.sqrt(y_pos**2+L**2))/y_pos))

    np.testing.assert_almost_equal(v[0], v_analytical)

def test_lsa_cylinder_radial_inv():
    y_pos = 1.
    pos = (0, 1., 0)
    coord, I = conf_cylinder((-1.,0.,0.))
    v = field.estimate_lsa(pos, coord, I, eta=eta)
    v_analytical = (1e6/(4*np.pi) * 
                     eta*I_0*1E4*(np.pi*diam*1E-6) *
                     np.log((L+np.sqrt(y_pos**2+L**2))/y_pos))

    np.testing.assert_almost_equal(v[0], v_analytical)
    
def test_lsa_cylinder_dipole_ratio():
    """test if v of dipole falls with 1/r2"""
    y_pos = 1e3
    pos = np.array([-L, y_pos, 0])
    coord = np.hstack([conf_cylinder((-L,0,0))[0],
                       conf_cylinder((0,0,0))[0]])
    I = np.array([[-1,1]])
    v1 = field.estimate_lsa(pos, coord, I, eta=eta)
    v2 = field.estimate_lsa(2*pos, coord, I, eta=eta)

    assert_almost_equal(v1/v2, 4, decimal=5)

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

    #show_potential_on_grid(cable, I)
    assert v1>0, "Potential is negative"
    assert v2<0, "Potential is positive"


def test_cylindric_coordinates():

    #position, expected result, comment
    positions = [([0,0], [0,0], "at begin"),
                 ([1,0], [0, 1], "at end"),
                 ([-1, 0], [0, -1], "to the left"),
                 ([2, 0], [0, 2], "to the right"),
                 ([0, 1], [1, 0], "above left end"),
                 ([0.5, 1], [1, 0.5], "above in the middle"),
                 ([-1, -1], [1, -1], "below left")
                ]

    pt1 = np.array([[0, 0]]).T
    pt2 = np.array([[1, 0]]).T
    for pos, expected, comment in positions:
        obtained = field._cylindric_coords(pt1, pt2, np.array(pos))
        assert (np.concatenate(obtained)==np.array(expected)).all(), comment

def show_potential_on_grid(cable, I):
    import matplotlib.pyplot as plt
    xx, yy = field.calc_grid([-10,10], [-10,10], 10)
    v_ext = field.estimate_on_grid(cable, I, xx, yy)
    cs=plt.contour(xx, yy, v_ext[0,:,:])
    plt.clabel(cs)
    plt.show()


def test_current_dipole_of_single_cylinder():
    cylinder_pos = (0, 0, 0)
    L = 1

    def Q_analytical(L, theta, I0):
        return (L*I0*np.cos(theta)*diam**2/4*1e-8, 
                L*I0*np.sin(theta)*diam**2/4*1e-8, 0)
   
    pi = np.pi
    #length, theta, current, expected dipole moment
    tests = [(L,  0,     1.),
             (L,  pi/2., 1.),
             (L,  pi/4., 1.),
             (L,  pi,    1.),
             (2*L,0,     1.),
             (L,  0.,    2.)
            ]

    for l, theta, I0 in tests:
        coord, I_axial = conf_cylinder(cylinder_pos, 
                                       theta=theta,
                                       L=l,
                                       I_0=I0)
        Q = field.calc_dipole_moment(coord, I_axial)
        Q_expected = Q_analytical(l, theta, I0)

        assert_almost_equal(Q, np.array(Q_expected)[:, None],
            err_msg='failed for theta=%f, l=%f' % (theta,l))

def test_current_dipole_of_two_cylinders():

    I0, l, theta = 1., 1., 0.

    pi = np.pi

    def assert_dipole_equal(cylinders, Q_expected):
        coord, I_axial = concat_cylinders(cylinders)
        Q = field.calc_dipole_moment(coord, I_axial)
        assert_almost_equal(
                Q,  np.array(Q_expected)[:, None]*1e-8/4*diam**2)

    cylinders = [
        conf_cylinder((0,0,0), theta=theta, L=l, I_0=I0),
        conf_cylinder((L,0,0), theta=theta, L=l, I_0=I0)
    ]
    assert_dipole_equal(cylinders, (2*L*I0, 0, 0))
    
    cylinders = [
        conf_cylinder((0,0,0), theta=0, L=l, I_0=I0),
        conf_cylinder((2*L,0,0), theta=np.pi, L=l, I_0=I0)
    ]
    assert_dipole_equal(cylinders, (0, 0, 0))

    cylinders = [
        conf_cylinder((0,0,0), theta=0, L=l/2., I_0=I0),
        conf_cylinder((L,0,0), theta=np.pi/2., L=l/2., I_0=I0)
    ]
    assert_dipole_equal(cylinders, (0.5, 0.5, 0))
