from QDYNTransmonLib.hamfull import tensor, standard_ops
import numpy as np
import numpy.linalg
from QDYNTransmonLib.io import full_qnums

"""
Routines for numerically determining H_1

example:

    HD = FullHamLevels(infile)
    params = read_params(config, 'GHz')
    chi_coeffs = get_chi_coeffs(HD, params, N_Taylor=3)
    print "chi_1^(1) = %f + %f n_1 + %f n_1**2" % tuple(chi_coeffs[0:3])
    print "chi_2^(1) = %f + %f n_1 + %f n_1**2" % tuple(chi_coeffs[3:6])
    print "chi_1^(2) = %f + %f n_1 + %f n_1**2" % tuple(chi_coeffs[6:9])
    print "chi_2^(2) = %f + %f n_1 + %f n_1**2" % tuple(chi_coeffs[9:12])
    delta = test(HD, params, chi_coeffs)

"""

def get_zeta(HD):
    """
    Get zeta from the diagonalized Hamiltonian
    """
    E000 = HD.get(0, 0, 0)
    E010 = HD.get(0, 1, 0)
    E100 = HD.get(1, 0, 0)
    E110 = HD.get(1, 1, 0)
    zeta = E000 - E010 - E100 + E110
    return zeta


def construct_H1_num(n_qubit, n_cavity, w_c, w_1, w_2, alpha_1, alpha_2, g_1,
    g_2, zeta, chi_coeffs):
    """
    Return Matrices Hdrift, Hctrl describing the Hamiltonian after the
    dispersive approximation
    """

    a, b, nq, nc, Iq, Ic = standard_ops(n_qubit, n_cavity)

    N_taylor = len(chi_coeffs) / (2*2) # two qubits, two chi's per qubit

    Hdrift =                                                               \
        w_c * tensor(Iq, Iq, nc)                                           \
      + w_1 * tensor(nq, Iq, Ic)                                           \
      + w_2 * tensor(Iq, nq, Ic)                                           \
      + 0.5 * alpha_1 * tensor(nq*nq - nq, Iq, Ic)                         \
      + 0.5 * alpha_2 * tensor(Iq, nq*nq - nq, Ic)                         \
      + tensor(chi_num(nq, chi_coeffs[0:N_taylor]), Iq, Ic)                \
      + tensor(Iq, chi_num(nq, chi_coeffs[N_taylor:2*N_taylor]), Ic)       \
      + tensor(chi_num(nq, chi_coeffs[2*N_taylor:3*N_taylor]), Iq, nc)     \
      + tensor(Iq, chi_num(nq, chi_coeffs[3*N_taylor:4*N_taylor]), nc)     \
      + zeta * tensor(nq, nq, Ic)

    lambda_1 = - 2 * g_1 / (w_1 - w_c)
    lambda_2 = - 2 * g_2 / (w_2 - w_c)

    Hctrl =   lambda_1 * ( tensor(b, Iq, Ic) + tensor(b.H, Iq, Ic) ) \
            + lambda_2 * ( tensor(Iq, b, Ic) + tensor(Iq, b.H, Ic) ) \
            + ( tensor(Iq, Iq, a) + tensor(Iq, Iq, a.H) )

    return Hdrift, Hctrl


def chi_num(nq, taylor_coeffs):
    chi = taylor_coeffs[0] * nq**0
    for k, a in enumerate(taylor_coeffs):
        chi += a * nq**k
    return chi


def get_H_tilde(HD, n_qubit, n_cavity, w_1, w_2, w_c, alpha_1, alpha_2):
    """
    Return the entries of the diagonalized Hamltonian, without energy levels
    and the zeta terms. Right hand side of the equation system.

    Also return array of indicies of lines from the equation system that should
    be dropped, since they require undefined quantum numbers

    HD      : Diagonalized Hamiltonian
    n_qubit : number of qubit levels
    n_cavity: number of cavity levels
    """
    zeta = get_zeta(HD)
    N = n_qubit * n_qubit * n_cavity
    b = np.zeros(N, dtype=np.complex128)
    b = []
    rows_to_delete = []
    for row in xrange(N):
        i, j, n = full_qnums(row+1, n_qubit, n_cavity)
        try:
            hval = HD.get(i, j, n)
            b.append(hval \
                    - (  (w_1 + 0.5*alpha_1*(i-1)) * i
                        + (w_2 + 0.5*alpha_2*(j-1)) * j
                        + w_c * n
                        + zeta * i * j
                    ))
        except KeyError:
            # For higher levels, the eigenstates can no longer directly be
            # associated with a set of quantum numbers. That means that not all
            # quantum numbers might be present in n HD. If the lookup fails,
            # the best thing to do is to disregard that condition from the
            # equation system entirely
            rows_to_delete.append(row)
    return (np.array(b, dtype=np.complex128), rows_to_delete)


def get_coeff_matrix(n_qubit, n_cavity, N_Taylor):
    """
    Construct the coefficient matrix

    n_qubit : number of qubit levels
    n_cavity: number of cavity levels
    """
    # matrix of coefficients
    N_chi = 2 # number of chis
    N_unknowns   = 2 * N_Taylor * N_chi # for two qubits
    N_conditions = n_qubit * n_qubit * n_cavity
    # C: over-determined coefficient matrix
    C = np.zeros(shape=(N_conditions, N_unknowns))
    # each condition is from one diagonal element the matrix
    for row in xrange(N_conditions):
        i, j, n = full_qnums(row+1, n_qubit, n_cavity)
        col = -1
        for i_chi in xrange(N_chi):
            for i_qubit in xrange(2):
                for i_taylor in xrange(N_Taylor):
                    col += 1
                    if i_qubit == 0:
                        C[row,col] = float(i**i_taylor)
                    else:
                        C[row,col] = float(j**i_taylor)
                    if i_chi == 1: # second chi
                        C[row,col] *= float(n)
    return C


def get_chi_coeffs(HD, params, N_taylor):
    """
    Calculate coefficients that most closely approximate HD
    """
    w_d     = params['w_d']
    w_1     = params['w_1'] - w_d
    w_2     = params['w_2'] - w_d
    w_c     = params['w_c'] - w_d
    alpha_1 = params['alpha_1']
    alpha_2 = params['alpha_2']
    n_qubit  = params['n_qubit']
    n_cavity = params['n_cavity']

    # coefficient matrix
    C = get_coeff_matrix(n_qubit, n_cavity, N_taylor)

    # right hand side (Hamiltonian with known terms subtracted)
    b, rows_to_delete = get_H_tilde(HD, n_qubit, n_cavity, w_1, w_2, w_c,
                                    alpha_1, alpha_2)

    C = np.delete(C, rows_to_delete, axis=0)

    # Find coefficients that best match the Hamiltonian, using least-squares
    chi_coeffs, residuals, rank, singular_values = numpy.linalg.lstsq(C, b)
    return chi_coeffs


def test(HD, params, chi_coeffs):
    """
    Check how close the matrix contructed from the coefficients is
    """
    w_d      = params['w_d']
    w_1      = params['w_1'] - w_d
    w_2      = params['w_2'] - w_d
    w_c      = params['w_c'] - w_d
    alpha_1  = params['alpha_1']
    alpha_2  = params['alpha_2']
    g_1      = params['g_1']
    g_2      = params['g_2']
    n_qubit  = params['n_qubit']
    n_cavity = params['n_cavity']
    zeta = get_zeta(HD)
    Hnum, __ = construct_H1_num(n_qubit, n_cavity, w_c, w_1, w_2, alpha_1,
                                alpha_2, g_1, g_2, zeta, chi_coeffs)
    N = n_qubit*n_qubit*n_cavity
    delta = np.zeros(N, dtype=np.complex128)
    for row in xrange(N):
        i, j, n = full_qnums(row+1, n_qubit, n_cavity)
        E = HD.get(i,j,n)
        delta[row] = abs(E - Hnum[row,row])
        print "delta \\Ket{%d %d %d}: %f MHz (%f%%)" % (
               i, j, n, delta[row]*1000.0, 100.0*delta[row]/abs(E))
    return delta

