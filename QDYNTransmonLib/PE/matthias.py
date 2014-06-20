# -*- coding: utf-8 -*-
"""
In the PE project, Matthias Müller did an optimization of the same system using
CRAB.

This module contains routines for his data, and calculating specific
functionals or other quantities in his way.


"""
import numpy as np


def read(datafile):
    """
    Read datafile of optimized gates from Matthias, return numpy array of
    times, array of numpy matrices U. The data file looks like this:

            #T/ns U11       U12     U13     U14     U21     U22     U23     U24     U31     U32     U33     U34     U41     U42     U43     U44
            100 (0.547393713542607,-0.454458520389049) (9.838246186775240E-002,0.161091326939856) (0.130351388903328,-7.759137606017047E-003) (-0.159971446587199,-0.641349012601028) (6.533274251377814E-002,-0.206931989808090) (-0.968553307016588,-3.255592622490506E-002) (-2.909705580269366E-003,-1.026713274349717E-002) (-8.526115366523079E-002,6.775565681155411E-002) (4.756229690411446E-002,-6.462733099329637E-002) (4.543331550429077E-003,3.845453667971317E-003) (-0.849064243771215,-0.442138087145397) (9.075252689661747E-002,-0.102987785394157) (3.117779807557363E-002,-0.638478425017202) (0.113977890968937,-0.104542240939655) (3.266193387827638E-002,9.126670844015491E-002) (0.645455415236018,0.323628868290356)
            ...
    """
    time = np.genfromtxt(datafile, usecols=(0,), unpack=True, dtype=np.int)
    Us = []
    with open(datafile) as in_fh:
        for line in in_fh:
            if not line.startswith('#'):
                U = np.zeros((4,4), dtype=np.complex128)
                xy = line.split()[1:]
                U[0,0] = complex(*[float(v) for v in xy[ 0][1:-1].split(",")])
                U[0,1] = complex(*[float(v) for v in xy[ 1][1:-1].split(",")])
                U[0,2] = complex(*[float(v) for v in xy[ 2][1:-1].split(",")])
                U[0,3] = complex(*[float(v) for v in xy[ 3][1:-1].split(",")])
                U[1,0] = complex(*[float(v) for v in xy[ 4][1:-1].split(",")])
                U[1,1] = complex(*[float(v) for v in xy[ 5][1:-1].split(",")])
                U[1,2] = complex(*[float(v) for v in xy[ 6][1:-1].split(",")])
                U[1,3] = complex(*[float(v) for v in xy[ 7][1:-1].split(",")])
                U[2,0] = complex(*[float(v) for v in xy[ 8][1:-1].split(",")])
                U[2,1] = complex(*[float(v) for v in xy[ 9][1:-1].split(",")])
                U[2,2] = complex(*[float(v) for v in xy[10][1:-1].split(",")])
                U[2,3] = complex(*[float(v) for v in xy[11][1:-1].split(",")])
                U[3,0] = complex(*[float(v) for v in xy[12][1:-1].split(",")])
                U[3,1] = complex(*[float(v) for v in xy[13][1:-1].split(",")])
                U[3,2] = complex(*[float(v) for v in xy[14][1:-1].split(",")])
                U[3,3] = complex(*[float(v) for v in xy[15][1:-1].split(",")])
                Us.append(np.matrix(U))
    return time, Us


def gs_closest_unitary(U):
    """
    Calculate a "closest unitary" by Gram-Schmidt Orthonormalization

    See email of May 21:
    >> we are currently summarizing the results of Giulia's calculations into
    >> figures. In order to compare to your calculations, we have two
    >> questions:
    >> - How to you determine \ilde U (the projection of the actual U onto
    >>         SU(4))?
    > I calculate tilde U by first projecting U on the logical subspace. Then
    > I take the row vectors, and perform Gram-Schimdt orthonormalization
    > starting from the first row.
    """
    def norm(v):
        return np.sqrt(np.vdot(v, v).real)
    v1 = np.array(U, np.complex128)[0,:]
    v2 = np.array(U, np.complex128)[1,:]
    v3 = np.array(U, np.complex128)[2,:]
    v4 = np.array(U, np.complex128)[3,:]

    u1 = v1 / norm(v1)

    u2 = v2 - np.vdot(u1, v2) * u1
    u2 = u2 / norm(u2)

    u3 = v3 - np.vdot(u2, v3) * u2 \
            - np.vdot(u1, v3) * u1
    u3 = u3 / norm(u3)

    u4 = v4 - np.vdot(u3, v4) * u3 \
            - np.vdot(u2, v4) * u2 \
            - np.vdot(u1, v4) * u1
    u4 = u4 / norm(u4)
    return np.matrix(np.row_stack((u1, u2, u3, u4)))
