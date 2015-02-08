"""
Routines for constructing various Hamiltonians in the "full" qubit-qubit-cavity
Hilbert space
"""
from warnings import warn
import numpy as np

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


def prime_factors(n):
    """Return the prime decomposition of n"""
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def qubit_cavity_decomposition(n, nc=None):
    """
    Return (nq, nc) of a decomposition n = nq*nq*nc, where nc >= nq
    """
    if nc is not None:
        n = n // nc
    factors = prime_factors(n)
    qubit_factors = []
    cavity_factors = []
    for factor in set(factors):
        count = factors.count(factor)
        while count >= 2:
            qubit_factors.append(factor)
            count += -2
        if count > 0:
            cavity_factors.append(factor)
    # Ensure nc >= nq
    if nc is None:
        while np.array(qubit_factors).prod() > np.array(cavity_factors).prod():
            factor = qubit_factors.pop(0)
            cavity_factors = [factor, factor] + cavity_factors
        nc = np.array(cavity_factors).prod()
    nq = np.array(qubit_factors).prod()
    return nq, nc


from scipy.sparse.coo import coo_matrix
class FullHamMatrix(coo_matrix):
    """
    Subclass of scipy.sparse.coo_matrix that contains a Hamiltonian matrix for
    the full system

    Attributes
    ----------

    nq      : number of qubit levels
    nc      : number of cavity levels
    filename: Default name of file to which FullHamMatrix is to be written

    Constructor Arguments
    ---------------------

    In addition to the keyword arguments of `scipy.sparse.coo_matrix`, the
    following (optional) keyword arguments are understood:

    nc : str
        The number of cavity levels. If not given qubit and cavity levels will
        be guessed.
    """

    def __init__(self, *args, **kwargs):
        if 'dtype' in kwargs:
            if kwargs['dtype'] is not np.complex128:
                warn('dtype is ignored (must be numpy.complex128)')
        kwargs['dtype'] = np.complex128
        nc = None
        if 'nc' in kwargs:
            nc = kwargs['nc'] # delay until after construction
            self.__nc = nc
            del kwargs['nc']
        coo_matrix.__init__(self, *args, **kwargs)
        n = self.shape[0]
        self.__nq, self.__nc = qubit_cavity_decomposition(n, nc=nc)

    @property
    def nq(self):
        return self.__nq

    @property
    def nc(self):
        return self.__nc

    def quantum_numbers(self, level_index):
        """
        Given a 1-based level_index, return tuple of 0-based quantum numbers
        (i, j, n)
        """
        l = level_index - 1
        nn = self.nq * self.nc
        i = l / nn
        l = l - i * nn
        j = l / self.nc
        n = l - j * self.nc
        return (i, j, n)

    def write(self, outfile, comment, rwa_factor, unit, complex=False):
        """
        Write out numpy matrix in sparse format to outfile

        The matrix file is assumed to be in the given unit; that unit will
        be written to the header of the output file. In the output, each
        value will be multiplied by rwa_factor. This is to
        account for the reduction of the pulse amplitude by a factor 1/2,
        or, more generally, a Hamiltonian connected to Omega^n will have to
        receive an rwa_factor for 1 / 2^n

        If complex is True, an extra column is written for the imaginary
        part
        """
        if complex:
            header_fmt = "#    row  column %24s %24s %20s %24s"
            header = header_fmt % ('Re(value) [%s]' % unit,
                                'Im(value) [%s]' % unit,
                                'i,j,n (row)', 'i,j,n (col)')
            fmt = "%8d%8d%25.16E%25.16E%7d%7d%7d    %7d%7d%7d"
        else:
            header_fmt = "#    row  column %24s %20s %24s"
            header = header_fmt % ('value [%s]' % unit,
                                'i,j,n (row)', 'i,j,n (col)')
            fmt = "%8d%8d%25.16E%7d%7d%7d    %7d%7d%7d"
        with open(outfile, 'w') as out_fh:
            print >> out_fh, comment
            print >> out_fh, header
            for i_val in xrange(self.nnz):
                i = self.col[i_val] + 1 # 1-based indexing
                j = self.row[i_val] + 1
                v = self.data[i_val]
                v *= rwa_factor
                if (not complex):
                    assert(v.imag == 0.0), \
                    "matrix has unexpected complex values"
                if (j >= i):
                    ii, ji, ni = self.quantum_numbers(i)
                    ij, jj, nj = self.quantum_numbers(j)
                    if complex:
                        print >> out_fh, fmt % (i, j, v.real, v.imag,
                                                ii, ji, ni, ij, jj, nj)
                    else:
                        print >> out_fh, fmt % (i, j, v.real,
                                                ii, ji, ni, ij, jj, nj)

    def write_logical_states(self, outfile):
        """
        Write the canonical logical states to the given outfile
        """
        with open(outfile, 'w') as out:
            out.write("# States representing the logical subspace in "
                        "the canonical (|ijn>) representation\n")
            out.write("#                      00                       01"
                      "                       10                       11"
                      "    i    j    n\n")
            for i in xrange(self.nq):
                for j in xrange(self.nq):
                    for n in xrange(self.nc):
                        p00 = p01 = p10 = p11 = 0.0
                        if (i == 0) and (j == 0) and (n == 0): p00 = 1.0
                        if (i == 0) and (j == 1) and (n == 0): p01 = 1.0
                        if (i == 1) and (j == 0) and (n == 0): p10 = 1.0
                        if (i == 1) and (j == 1) and (n == 0): p11 = 1.0
                        out.write("%25.16E%25.16E%25.16E%25.16E%5d%5d%5d\n"
                                  % (p00, p01, p10, p11, i, j, n))

