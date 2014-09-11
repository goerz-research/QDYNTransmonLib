import numpy as np
from numpy import kron
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


def chi1(n, Delta, g, alpha):
    chi = np.matrix(np.zeros(shape=(n,n), dtype=np.complex128))
    for i in xrange(0, n):
        chi[i,i] = i * g**2 / ( (i-1)*alpha + Delta )
    return chi


def chi2(n, Delta, g, alpha):
    chi = np.matrix(np.zeros(shape=(n,n), dtype=np.complex128))
    for i in xrange(0, n):
        chi[i,i] =    (g**2 * (alpha - Delta))                                \
                    / ( ((i-1)*alpha + Delta)  * (i*alpha + Delta)  )
    return chi


def chi3(n, Delta, g, alpha, offset=0):
    chi = np.matrix(np.zeros(shape=(n,n), dtype=np.complex128))
    for i in xrange(0+offset, n+offset):
        numer = (alpha**2 * g**4 * i * (i-1)) / ((2*i-3)*alpha + 2*Delta)
        denom = ((i-1)*alpha + Delta)**2 * ((i-2)*alpha + Delta)**2
        chi[i-offset,i-offset] =  numer / denom
    return chi


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


def alpha_polaron(n_qubit, delta, Delta1, Delta2, g1, g2, alpha1, alpha2):
    """
    Polaron operator (in two-qubit space, i.e. dimension n_qubit * n_qubit)

    n_qubit: number of qubit levels
    delta  : detuning of cavity from the drive
    Delta1 : detuning of qubit 1 from cavity (w_1 - w_c)
    Delta2 : detuning of qubit 2 from cavity (w_2 - w_c)
    g1     : coupling qubit 1 to cavity
    g2     : coupling qubit 2 to cavity
    alpha1 : anharmonicity qubit 1
    alpha2 : anharmonicity qubit 2
    """
    Iq = np.matrix(np.identity(n_qubit))
    chi_q1 = chi2(n_qubit, Delta1, g1, alpha1)
    chi_q2 = chi2(n_qubit, Delta2, g2, alpha2)
    alpha = kron(chi_q1, Iq) + kron(Iq, chi_q2) + delta * kron(Iq, Iq)
    for i in xrange(n_qubit * n_qubit):
        alpha[i,i] = -1.0 / alpha[i,i]
    return alpha


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


def construct_H1(n_qubit, n_cavity, w_c, w_1, w_2, alpha_1, alpha_2, g_1, g_2,
    zeta):
    """
    Return Matrices Hdrift, Hctrl describing the Hamiltonian after the
    dispersive approximation
    """

    a, b, nq, nc, Iq, Ic = standard_ops(n_qubit, n_cavity)

    Hdrift =   w_c * tensor(Iq, Iq, nc)       \
            + w_1 * tensor(nq, Iq, Ic)        \
            + w_2 * tensor(Iq, nq, Ic)        \
            + 0.5 * alpha_1 * tensor(nq*nq - nq, Iq, Ic) \
            + 0.5 * alpha_2 * tensor(Iq, nq*nq - nq, Ic) \
            + tensor(chi1(n_qubit, w_1-w_c, g_1, alpha_1), Iq, Ic) \
            + tensor(Iq, chi1(n_qubit, w_2-w_c, g_2, alpha_2), Ic) \
            + tensor(chi2(n_qubit, w_1-w_c, g_1, alpha_1), Iq, nc) \
            + tensor(Iq, chi2(n_qubit, w_2-w_c, g_2, alpha_2), nc) \
            + 4 * tensor(chi3(n_qubit, w_1-w_c, g_1, alpha_1), Iq, nc) \
            + 4 * tensor(Iq, chi3(n_qubit, w_2-w_c, g_2, alpha_2), nc) \
            + 2 * tensor(chi3(n_qubit, w_1-w_c, g_1, alpha_1), Iq, Ic) \
            + 2 * tensor(Iq, chi3(n_qubit, w_2-w_c, g_2, alpha_2), Ic) \
            + tensor(chi3(n_qubit, w_1-w_c, g_1, alpha_1), Iq, a.H*a.H*a*a) \
            + tensor(Iq, chi3(n_qubit, w_2-w_c, g_2, alpha_2), a.H*a.H*a*a) \
            + tensor(chi3(n_qubit, w_1-w_c, g_1, alpha_1, 2), Iq, a.H*a.H*a*a)\
            + tensor(Iq, chi3(n_qubit, w_2-w_c, g_2, alpha_2, 2), a.H*a.H*a*a)\
            + zeta * tensor(nq, nq, Ic)

    lambda_1 = - 2 * g_1 / (w_1 - w_c)
    lambda_2 = - 2 * g_2 / (w_2 - w_c)

    Hctrl =   lambda_1 * ( tensor(b, Iq, Ic) + tensor(b.H, Iq, Ic) ) \
            + lambda_2 * ( tensor(Iq, b, Ic) + tensor(Iq, b.H, Ic) ) \
            + ( tensor(Iq, Iq, a) + tensor(Iq, Iq, a.H) )

    return Hdrift, Hctrl


def construct_H2(n_qubit, n_cavity, w_c, w_1, w_2, alpha_1, alpha_2, g_1, g_2,
    zeta):
    """
    Return matrices Hdrift, Hctrl, Hctrl2 (Stark shift), Hctrlderiv (coupling
    to derivative of pulse) of the H1 in the approximation of off-resonant pulses
    """

    a, b, nq, nc, Iq, Ic = standard_ops(n_qubit, n_cavity)

    Hdrift =   w_c * tensor(Iq, Iq, nc)       \
            + w_1 * tensor(nq, Iq, Ic)        \
            + w_2 * tensor(Iq, nq, Ic)        \
            + 0.5 * alpha_1 * tensor(nq*nq - nq, Iq, Ic) \
            + 0.5 * alpha_2 * tensor(Iq, nq*nq - nq, Ic) \
            + tensor(chi1(n_qubit, w_1-w_c, g_1, alpha_1), Iq, Ic) \
            + tensor(Iq, chi1(n_qubit, w_2-w_c, g_2, alpha_2), Ic) \
            + tensor(chi2(n_qubit, w_1-w_c, g_1, alpha_1), Iq, nc) \
            + tensor(Iq, chi2(n_qubit, w_2-w_c, g_2, alpha_2), nc) \
            + zeta * tensor(nq, nq, Ic)

    Hctrl = tensor(Iq, Iq, a) + tensor(Iq, Iq, a.H)

    Hctrl2 =   (4.0 / (w_1-w_c)**2 ) \
                   * tensor(chi2(n_qubit, w_1-w_c, g_1, alpha_1), Iq, Ic)  \
             + (4.0 / (w_2-w_c)**2 ) \
                   * tensor(Iq, chi2(n_qubit, w_2-w_c, g_2, alpha_2), Ic)

    lambda_1 = - 2 * g_1 / (w_1 - w_c)
    lambda_2 = - 2 * g_2 / (w_2 - w_c)

    Hctrlderiv =    tensor(lambda_1 * 1j                                      \
                           * inv(alpha_1 * nq + (w_1-w_c)*Iq)                 \
                           * b, Iq, Ic)                                       \
                  - tensor(lambda_1 * 1j                                      \
                           * inv(alpha_1 * nq - alpha_1*Iq + (w_1-w_c)*Iq)    \
                           * b.H, Iq, Ic)                                     \
                  + tensor(Iq,                                                \
                           lambda_2 * 1j                                      \
                           * inv(alpha_2 * nq + (w_2-w_c)*Iq)                 \
                           * b, Ic)                                           \
                  - tensor(Iq,                                                \
                           lambda_2 * 1j                                      \
                           * inv(alpha_2 * nq - alpha_2*Iq + (w_2-w_c)*Iq)    \
                           * b.H, Ic)


    return Hdrift, Hctrl, Hctrl2, Hctrlderiv


def construct_H3(n_qubit, n_cavity, w_c, w_1, w_2, alpha_1, alpha_2, g_1, g_2,
    zeta):
    """
    Return matrices Hdrift, Hctrl2, Hctrlderiv of Hamiltonian after polaron
    transform
    """

    a, b, nq, nc, Iq, Ic = standard_ops(n_qubit, n_cavity)

    alpha_pol = alpha_polaron(n_qubit, w_c, w_1, w_2, g_1, g_2,
                              alpha_1, alpha_2)

    Hdrift =  w_1 * tensor(nq, Iq, Ic)        \
            + w_2 * tensor(Iq, nq, Ic)        \
            + 0.5 * alpha_1 * tensor(nq*nq - nq, Iq, Ic) \
            + 0.5 * alpha_2 * tensor(Iq, nq*nq - nq, Ic) \
            + tensor(chi1(n_qubit, w_1-w_c, g_1, alpha_1), Iq, Ic) \
            + tensor(Iq, chi1(n_qubit, w_2-w_c, g_2, alpha_2), Ic) \
            + zeta * tensor(nq, nq, Ic)

    chi2_q1 = kron(chi2(n_qubit, w_1-w_c, g_1, alpha_1), Iq)
    chi2_q2 = kron(Iq, chi2(n_qubit, w_2-w_c, g_2, alpha_2))

    Hctrl2 =    w_c * kron(alpha_pol.H * alpha_pol, Ic) \
             - kron(chi2_q1 * alpha_pol.H * alpha_pol, Ic) \
             - kron(chi2_q2 * alpha_pol.H * alpha_pol, Ic) \
             + (4.0 / (w_1-w_c)**2 ) \
                   * tensor(chi2(n_qubit, w_1-w_c, g_1, alpha_1), Iq, Ic)  \
             + (4.0 / (w_2-w_c)**2 ) \
                   * tensor(Iq, chi2(n_qubit, w_2-w_c, g_2, alpha_2), Ic)

    lambda_1 = - 2 * g_1 / (w_1 - w_c)
    lambda_2 = - 2 * g_2 / (w_2 - w_c)

    Hctrlderiv =    tensor(lambda_1 * 1j                                      \
                           * inv(alpha_1 * nq + (w_1-w_c)*Iq)                 \
                           * b, Iq, Ic)                                       \
                  - tensor(lambda_1 * 1j                                      \
                           * inv(alpha_1 * nq - alpha_1*Iq + (w_1-w_c)*Iq)    \
                           * b.H, Iq, Ic)                                     \
                  + tensor(Iq,                                                \
                           lambda_2 * 1j                                      \
                           * inv(alpha_2 * nq + (w_2-w_c)*Iq)                 \
                           * b, Ic)                                           \
                  - tensor(Iq,                                                \
                           lambda_2 * 1j                                      \
                           * inv(alpha_2 * nq - alpha_2*Iq + (w_2-w_c)*Iq)    \
                           * b.H, Ic)                                         \
                  - kron(                                                     \
                    inv(                                                      \
                      w_c * kron(Iq, Iq)                                      \
                      + kron(chi2(n_qubit, w_1-w_c, g_1, alpha_1), Iq)        \
                      + kron(Iq, chi2(n_qubit, w_2-w_c, g_2, alpha_2))        \
                    ), (a + a.H))

    return Hdrift, Hctrl2, Hctrlderiv

