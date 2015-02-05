import numpy as np
"""
Routines for constructing various Hamiltonians in the "full" qubit-qubit-cavity
Hilbert space

To write out a Hamiltonian, use e.g.

    params = QDYNTransmonLib.io.read_params('config', 'GHz')
    n_qubit = params['n_qubit']
    n_cavity = params['n_cavity']
    w_c = params['w_c'] - params['w_d'] # in RWA
    w_1 = params['w_1'] - params['w_d'] # in RWA
    w_2 = params['w_2'] - params['w_d'] # in RWA
    g_1 = params['g_1']
    g_2 = params['g_2']
    zeta = params['zeta']
    alpha_1 = params['alpha_1']
    alpha_2 = params['alpha_2']

    Hdrift, Hctrl2, Hctrlderiv \
    = construct_H3(n_qubit, n_cavity, w_c, w_1, w_2, alpha_1, alpha_2,
                   g_1, g_2, zeta)

    write_full_ham_matrix(Hdrift, "ham_drift.dat", comment, n_qubit,
                          n_cavity, rwa_factor=1.0, conversion_factor=1.0,
                          unit='GHz')
    write_full_ham_matrix(Hctrl2, "ham_ctrl2.dat", comment, n_qubit,
                          n_cavity, rwa_factor=0.25,
                          conversion_factor=6.579683918175572e6, unit='au')
    write_full_ham_matrix(Hctrlderiv, "ham_ctrlderiv.dat", comment,
                          n_qubit, n_cavity, rwa_factor=0.5,
                          conversion_factor=6.579683918175572e6, unit='au',
                          complex=True)
"""

def standard_ops(n_qubit, n_cavity):
    """
    Return lowering operators, number operators, and identities for qubit and
    cavity Hilbert space

    >>> a, b, nq, nc, Iq, Ic = standard_ops(n_qubit, n_cavity)
    """
    a = np.matrix(np.zeros(shape=(n_cavity,n_cavity), dtype=np.complex128))
    for i in xrange(1,n_cavity):
        a[i-1, i] = np.sqrt(i)
    nc = a.H * a

    b = np.matrix(np.zeros(shape=(n_qubit,n_qubit), dtype=np.complex128))
    for i in xrange(1,n_qubit):
        b[i-1, i] = np.sqrt(i)
    nq = b.H * b

    Iq = np.matrix(np.identity(n_qubit))
    Ic = np.matrix(np.identity(n_cavity))

    return a, b, nq, nc, Iq, Ic


def tensor(a, b, c):
    """ Qubit-Qubit-Cavity tensor product """
    return np.kron(a, np.kron(b, c))


def inv(h):
    """ Calculate the inverse of a diagonal Operator h """
    n, m = h.shape
    h_inv = h.copy()
    for i in xrange(n):
        for j in xrange(m):
            if (i == j):
                h_inv[i,j] = 1.0 / h_inv[i,j]
            else:
                assert(h_inv[i,j] == 0.0), "h must be diadonal to be inverted"
    return h_inv


def write_logical_eigenstates(n_qubit, n_cavity):
    with open('logical_states.dat', 'w') as out:
        print >> out, "# Eigenstates representing the logical subspace in "   \
                      "the canonical (|ijn>) representation"
        print >> out, "#                      00                       01"    \
                      "                       10                       11"    \
                      "    i    j    n"
        for i in xrange(n_qubit):
            for j in xrange(n_qubit):
                for n in xrange(n_cavity):
                    p00 = p01 = p10 = p11 = 0.0
                    if (i == 0) and (j == 0) and (n == 0): p00 = 1.0
                    if (i == 0) and (j == 1) and (n == 0): p01 = 1.0
                    if (i == 1) and (j == 0) and (n == 0): p10 = 1.0
                    if (i == 1) and (j == 1) and (n == 0): p11 = 1.0
                    out.write("%25.16E%25.16E%25.16E%25.16E%5d%5d%5d\n"      \
                              % (p00, p01, p10, p11, i, j, n))


def construct_Hfull(n_qubit, n_cavity, w_c, w_1, w_2, alpha_1, alpha_2, g_1,
    g_2):
    """
    Return Matrices Hdrift, Hctrl describing the Hamiltonian after the
    dispersive approximation
    """

    a, b, nq, nc, Iq, Ic = standard_ops(n_qubit, n_cavity)

    Hdrift =  w_c * tensor(Iq, Iq, nc)                                        \
            + w_1 * tensor(nq, Iq, Ic)                                        \
            + w_2 * tensor(Iq, nq, Ic)                                        \
            + 0.5 * alpha_1 * tensor(nq*nq - nq, Iq, Ic)                      \
            + 0.5 * alpha_2 * tensor(Iq, nq*nq - nq, Ic)                      \
            + g_1 * (tensor(b.H, Iq, a) + tensor(b, Iq, a.H))                 \
            + g_2 * (tensor(Iq, b.H, a) + tensor(Iq, b, a.H))

    Hctrl =   ( tensor(Iq, Iq, a) + tensor(Iq, Iq, a.H) )

    return Hdrift, Hctrl

