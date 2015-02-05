import numpy as np
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

