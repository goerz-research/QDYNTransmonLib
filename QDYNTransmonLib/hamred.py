import numpy as np
from QDYNTransmonLib.hamfull import chi1, chi2, alpha_polaron

"""
Routines for constructing various Hamiltonians in the "reduced" qubit-qubit
Hilbert space
"""

def tensor(a, b):
    """ Qubit-Qubit tensor product """
    return np.kron(a, b)

def write_logical_eigenstates(n_qubit, n_start=0):
    with open('logical_states.dat', 'w') as out:
        print >> out, "# Eigenstates representing the logical subspace in "   \
                      "the canonical (|ij>) representation"
        print >> out, "#                      00                       01"    \
                      "                       10                       11"    \
                      "    i    j"
        for i in xrange(n_start, n_qubit+n_start):
            for j in xrange(n_start, n_qubit+n_start):
                p00 = p01 = p10 = p11 = 0.0
                if (i == 0) and (j == 0): p00 = 1.0
                if (i == 0) and (j == 1): p01 = 1.0
                if (i == 1) and (j == 0): p10 = 1.0
                if (i == 1) and (j == 1): p11 = 1.0
                out.write("%25.16E%25.16E%25.16E%25.16E%5d%5d\n"      \
                            % (p00, p01, p10, p11, i, j))


def construct_H3(n_qubit, w_c, w_1, w_2, alpha_1, alpha_2, g_1, g_2, zeta):

    b = np.matrix(np.zeros(shape=(n_qubit,n_qubit), dtype=np.complex128))
    for i in xrange(1,n_qubit):
        b[i-1, i] = np.sqrt(i)
    nq = b.H * b

    Iq = np.matrix(np.identity(n_qubit))
    alpha_pol = alpha_polaron(n_qubit, w_c, w_1, w_2, g_1, g_2,
                              alpha_1, alpha_2)

    Hdrift =  w_1 * tensor(nq, Iq)        \
            + w_2 * tensor(Iq, nq)        \
            + 0.5 * alpha_1 * tensor(nq*nq - nq, Iq) \
            + 0.5 * alpha_2 * tensor(Iq, nq*nq - nq) \
            + tensor(chi1(n_qubit, w_1-w_c, g_1, alpha_1), Iq) \
            + tensor(Iq, chi1(n_qubit, w_2-w_c, g_2, alpha_2)) \
            + zeta * tensor(nq, nq)

    Hctrl2 =    w_c * alpha_pol.H * alpha_pol \
             + tensor(chi2(n_qubit, w_1-w_c, g_1, alpha_1), Iq) \
               * alpha_pol.H * alpha_pol \
             + tensor(Iq, chi2(n_qubit, w_2-w_c, g_2, alpha_2)) \
               * alpha_pol.H * alpha_pol \
             + (4.0 / (w_1-w_c)**2 ) \
                   * tensor(chi2(n_qubit, w_1-w_c, g_1, alpha_1), Iq)  \
             + (4.0 / (w_2-w_c)**2 ) \
                   * tensor(chi2(n_qubit, w_1-w_c, g_1, alpha_1), Iq)  \

    lambda_1 = - 2 * g_1 / (w_1 - w_c)
    lambda_2 = - 2 * g_2 / (w_2 - w_c)

    d1 = np.matrix(np.zeros(shape=(n_qubit,n_qubit), dtype=np.complex128))
    for i in xrange(n_qubit):
        d1[i, i] = 1.0j / (i * alpha_1 + w_1 - w_c)
    d2 = np.matrix(np.zeros(shape=(n_qubit,n_qubit), dtype=np.complex128))
    for i in xrange(n_qubit):
        d2[i, i] = 1.0j / (i * alpha_2 + w_2 - w_c)

    Hctrlderiv =    tensor(lambda_1 * d1   * b, Iq)   \
                  + tensor(lambda_1 * d1.H * b.H, Iq) \
                  + tensor(Iq, lambda_2 * d2   * b)   \
                  + tensor(Iq, lambda_2 * d2.H * b.H)

    return Hdrift, Hctrl2, Hctrlderiv

