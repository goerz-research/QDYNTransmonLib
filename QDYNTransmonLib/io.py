"""
Reading and writing data
"""
import re
import numpy as np
import scipy
from QDYN.units import NumericConverter


def read_params(config, unit):
    """
    Extract the qubit parameters from the given config file and return them in
    a dictionary
    """
    result = {}

    float_params = ['w_c', 'w_1', 'w_2', 'w_d', 'alpha_1', 'alpha_2', 'g_1',
                    'g_2', 'zeta', 'kappa', 'E_C', "E_J", "E_JJ"]
    int_params = ['n_qubit', 'n_cavity']

    float_rxs = {}
    int_rxs = {}
    rx_float = r'(?P<val>[\d.edED+-]+)(_(?P<unit>\w+))?'
    rx_int = r'(?P<val>\d+)'
    for param in float_params:
        rx = re.compile(param + r'\s*=\s*' + rx_float)
        float_rxs[param] = rx
    for param in int_params:
        rx = re.compile(param + r'\s*=\s*' + rx_int)
        int_rxs[param] = rx

    convert = NumericConverter()

    with open(config) as in_fh:
        for line in in_fh:
            for param, rx in float_rxs.items():
                m = rx.search(line)
                if m:
                    val = float(m.group('val'))
                    val_unit = m.group('unit')
                    if val_unit is None:
                        val_unit = 'au'
                    if val_unit != unit:
                        val = convert.to_au(val, val_unit)
                        val = convert.from_au(val, unit)
                    result[param] = val
            for param, rx in int_rxs.items():
                m = rx.search(line)
                if m:
                    val = int(m.group('val'))
                    result[param] = val

    return result


class FullHamLevels():
    """
    Class representing data in 'eigenvalues.dat' of full model, as generate
    by tm_en_logical_eigenstates
    """

    def __init__(self, filename):
        """
        Read in diagonalized energy levels in full Hamiltonian
        filename should be eigenvalues.dat
        """
        self.E = np.genfromtxt(filename, usecols=(2, ), unpack=True)
        self.level, self.level_i, self.level_j, self.level_n \
        = np.genfromtxt(filename, usecols=(0,4,5,6), unpack=True,
                        dtype=np.int)

    def get(self, i, j, n):
        """
        Return the Energy in GHz for the level with the given quantum
        numbers
        """
        for ii, jj, nn, E  \
        in zip(self.level_i, self.level_j, self.level_n, self.E):
            if (ii == i) and (jj == j) and (nn == n):
                return E
        raise KeyError("(%d, %d, %d) not found" % (i, j, n))

    def print_dressed_params(self):
        """
        Print a summary of the dressed frame parameters
        """
        w_1 = self.get(1,0,0)
        w_2 = self.get(0,1,0)
        w_c = self.get(0,0,1)
        alpha_2 = self.get(0,2,0) - 2*w_2
        alpha_1 = self.get(2,0,0) - 2*w_1
        print "w_c (MHz)     = ", w_c * 1000.0
        print "w_1 (MHz)     = ", w_1 * 1000.0
        print "w_2 (MHz)     = ", w_2 * 1000.0
        print "w_d (MHz)     = ", 0.0
        print "alpha_1 (MHz) = ", alpha_1 * 1000.0
        print "alpha_2 (MHz) = ", alpha_2 * 1000.0


class RedHamLevels():
    """
    Class representing data in 'ham_eff_drift.dat' of reduced model
    Assumes that the Hamiltonian is fully diagonal
    """
    def __init__(self, filename):
        """
        Read in energy levels in reduced Hamiltonian.

        filename should be ham_eff_drift.dat
        """
        self.E = np.genfromtxt(filename, usecols=(2, ), unpack=True)
        self.level, self.level_i, self.level_j, \
        = np.genfromtxt(filename, usecols=(0,3,4), unpack=True,
                        dtype=np.int)

    def get(self, i, j):
        """
        Return the Energy in Ghz for the level with the given quantum
        numbers
        """
        for ii, jj, E in zip(self.level_i, self.level_j, self.E):
            if (ii == i) and (jj == j):
                return E

def weyl_path(datfile):
    """
    Extract the Weyl chamber coordinates c1, c2, c3 as numpy arrays over
    iteration number the local_invariants_oct_iter.dat file written during
    optimization.
    """
    c1, c2, c3 = np.genfromtxt(datfile, usecols=(8,9,10), unpack=True)
    return zip(c1, c2, c3)


def red_qnums(level_index, nq, n_start=0):
    """
    Given 1-based level_index, return tuple of 0-based quantum numbers (i, j)
    (or n_start based if given)
    """
    l = level_index - 1
    i = l / nq
    j = l - i*nq
    return (i+n_start,j+n_start)


def write_chi(chi, outfile):
    """
    Write the given diagonal chi matrix to outfile

    The entries are in the default energy unit (probably GHz)
    """
    N = chi.shape[0]
    with open(outfile, 'w') as out:
        print >> out, "# %3s%25s" % ("i", "val [energy]")
        for i in xrange(N):
            E = chi[i,i]
            assert E.imag == 0.0, "All chi entries must be real"
            print >> out, "%5d%25.16E" % (i+1, E.real)


def write_red_ham_matrix(h, outfile, comment, nq, rwa_factor,
    conversion_factor, unit, complex=False, n_start=0):
    """ Write out numpy matrix in sparse format to outfile

        The given 'unit' will be written to the header of outfile, and the
        Hamiltonian will be converted using conversion_factor, as well as
        multiplied with rwa_factor. Multiplying with rwa_factor is to
        account for the reduction of the pulse amplitude by a factor 1/2,
        which was not taken into account in the analytic derivations.
        Consequently, an Hamiltonian connected to Omega^n will have to
        receive an rwa_factor for 1 / 2^n

        If complex is True, an extra column is written for the imaginary part

        If n_start is given, start counting levels at n_start instead of zero
    """
    if complex:
        header_fmt = "#    row  column %24s %24s %13s %17s"
        header = header_fmt % ('Re(value) [%s]' % unit,
                               'Im(value) [%s]' % unit,
                               'i,j (row)', 'i,j col')
        fmt = "%8d%8d%25.16E%25.16E%7d%7d    %7d%7d"
    else:
        header_fmt = "#    row  column %24s %13s %17s"
        header = header_fmt % ('value [%s]' % unit, 'i,j (row)', 'i,j col')
        fmt = "%8d%8d%25.16E%7d%7d    %7d%7d"
    with open(outfile, 'w') as out_fh:
        print >> out_fh, comment
        print >> out_fh, header
        sparse_h = scipy.sparse.coo_matrix(h)
        for i_val in xrange(sparse_h.nnz):
            i = sparse_h.col[i_val] + 1 # 1-based indexing
            j = sparse_h.row[i_val] + 1
            v = sparse_h.data[i_val]
            v *= conversion_factor * rwa_factor
            if (not complex):
                assert(v.imag == 0.0), "matrix has unexpected complex values"
            if (j >= i):
                ii, ji = red_qnums(i, nq, n_start)
                ij, jj = red_qnums(j, nq, n_start)
                if complex:
                    print >> out_fh, fmt % (i, j, v.real, v.imag,
                                            ii, ji, ij, jj)
                else:
                    print >> out_fh, fmt % (i, j, v.real, ii, ji, ij, jj)

