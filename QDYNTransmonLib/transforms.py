from numpy import kron
from QDYNTransmonLib.hamfull import tensor, inv, standard_ops, chi2


def S1(n_qubit, n_cavity, w_c, w_1, w_2, alpha_1, alpha_2, g_1, g_2):
    """
    Transformation between the full Hamiltonian and the dispersive Hamiltonian
    (H_1)
    """
    a, b, nq, nc, Iq, Ic = standard_ops(n_qubit, n_cavity)

    S =                                                                       \
      tensor(g_1 * inv(alpha_1*nq + (w_1-w_c)*Iq) * b, Iq, a.H)               \
    + tensor(Iq, g_2 * inv(alpha_2*nq + (w_2-w_c)*Iq) * b, a.H)

    return S - S.H


def S2(n_qubit, n_cavity, w_c, w_1, w_2, alpha_1, alpha_2, g_1, g_2,
    omega=1.0):
    """
    Transformation leading to H_2
    """
    a, b, nq, nc, Iq, Ic = standard_ops(n_qubit, n_cavity)

    lambda_1 = - 2 * g_1 / (w_1 - w_c)
    lambda_2 = - 2 * g_2 / (w_2 - w_c)

    S =                                                                       \
      tensor(lambda_1 * omega * inv(alpha_1*nq + (w_1-w_c)*Iq) * b, Iq, Ic)   \
    + tensor(Iq, lambda_2 * omega * inv(alpha_2*nq + (w_2-w_c)*Iq) * b, Ic)

    return S - S.H


def S_Polaron(n_qubit, n_cavity, w_c, w_1, w_2, alpha_1, alpha_2, g_1, g_2,
    omega=1.0):
    """
    Polaron transformation
    """
    a, b, nq, nc, Iq, Ic = standard_ops(n_qubit, n_cavity)

    S = -omega                                                                \
        * kron(inv(    w_c*kron(Iq,Iq)                                        \
                 + kron(chi2(n_qubit, w_1-w_c, g_1, alpha_1), Iq)             \
                 + kron(Iq, chi2(n_qubit, w_2-w_c, g_2, alpha_2))             \
        ), a.H)

    return S - S.H

