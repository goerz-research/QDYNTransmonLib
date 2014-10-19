"""
Reading and writing data
"""
import re
import os
import numpy as np
import scipy
from QDYN.units import NumericConverter
import sys


def read_params(config, unit):
    """
    Extract the qubit parameters from the given config file and return them in
    a dictionary
    """
    result = {}

    float_params = ['w_c', 'w_1', 'w_2', 'w_d', 'alpha_1', 'alpha_2', 'g_1',
                    'g_2', 'zeta', 'kappa', 'E_C']
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


class ExcitationDataSet():
    """ Class holding information about integrated qubit or cavity population
        over time

        Attributes
        ----------

        tgrid   Time grid array
        mean    Array of mean excitation over time
        sd      Standard Deviation from the mean, over time
    """
    def __init__(self, filename):
        """ Load data from the given filename

            The filename must contain the time grid in the first column and
            then one column for each qubit or cavity quantum number, giving the
            integrated populations for that quantum number
            The tm_*_prop programs produces files such as 'psi00_cavity.dat',
            'psi00_q1.dat', 'psi00_q2.dat', 'rho00_cavity.dat', 'rho00_q1.dat',
            'rho00_q2.dat' in the right format.
        """
        table       = np.genfromtxt(filename)
        self.tgrid  = table[:,0] # first column
        nt          = len(self.tgrid)
        n           = len(table[1,:]) - 1 # number of columns, -1 for time grid
        self.mean   = np.zeros(nt)
        self.sd     = np.zeros(nt)
        for i_t, t in enumerate(self.tgrid):
            for i_c in xrange(n):
                if not np.isnan(table[i_t,i_c+1]):
                    self.mean[i_t] += float(i_c) * table[i_t,i_c+1]
            for i_c in xrange(n):
                if not np.isnan(table[i_t,i_c+1]):
                    self.sd[i_t] += (float(i_c) - self.mean[i_t])**2 \
                                    * table[i_t,i_c+1]
            self.sd[i_t] = np.sqrt(self.sd[i_t])


class PopulationDataSet():
    """
    Class holding information about population dynamics in the logical subspace
    over time

    Attributes
    ----------

    tgrid   Time grid array
    pop00   Population in the 00 state, over time
    pop01   Population in the 01 state, over time
    pop10   Population in the 10 state, over time
    pop11   Population in the 11 state, over time
    """
    def __init__(self, filename, hilbert_space):
        """
        Load data from the given filename

        If the given file is for the propagation of a state in Hilbert
        space, it must contain 9 columns: The time grid, and the real and
        imaginary part of the projection onto the states 00, 01, 10, and 11
        over time, respectively. The tm_en_prop and tm_eff_prop programs
        generate e.g. the file psi00_phases.dat in the correct format

        If the propagation is in Liouville space, the file must contain 5
        columns: The time grid, and the population in the four logical
        basis states, over time. The tm_en_rho_prop and tm_eff_rho_prop
        programs generage e.g. rho00_popdyn.dat in the correct format
        """
        if hilbert_space:
            table       = np.genfromtxt(filename)
            self.tgrid  = table[:,0] # first column
            nt          = len(self.tgrid)
            self.pop00  = np.zeros(nt)
            self.pop01  = np.zeros(nt)
            self.pop10  = np.zeros(nt)
            self.pop11  = np.zeros(nt)
            for i_t, t in enumerate(self.tgrid):
                x00 = table[i_t,1]
                y00 = table[i_t,2]
                x01 = table[i_t,3]
                y01 = table[i_t,4]
                x10 = table[i_t,5]
                y10 = table[i_t,6]
                x11 = table[i_t,7]
                y11 = table[i_t,8]
                self.pop00[i_t] = x00**2 + y00**2
                self.pop01[i_t] = x01**2 + y01**2
                self.pop10[i_t] = x10**2 + y10**2
                self.pop11[i_t] = x11**2 + y11**2
        else:
            self.tgrid, self.pop00, self.pop01, self.pop10, self.pop11 \
            = np.genfromtxt(filename, unpack=True)

    def write(self, outfile):
        """ Write stored data to outfile, one column per attribute
            (i.e. data as it will be plotted)
        """
        data = np.column_stack((self.tgrid.flatten(),
                                self.pop00.flatten(),
                                self.pop01.flatten(),
                                self.pop10.flatten(),
                                self.pop11.flatten()))
        header = ("%23s"+"%25s"*4) % ("time [ns]",
                                      "pop00", "pop01", "pop10", "pop11")
        np.savetxt(outfile, data, fmt='%25.16E'*5, header=header)



def collect_pop_plot_data(data_files, runfolder, write_plot_data=False):
    r'''
    Take an array of data_files that must math the regular expression

        (psi|rho)(00|01|10|11)_(cavity|q1|q2|phases|popdyn)\.dat

    Filenames not following this pattern will raise a ValueError.

    For each file, create a instance of ExcitationDataSet (if filename ends in
    cavity, q1, or q2) or PopulationDataSet (if filename ends in phases or
    popdyn)

    Return an array of the data set instances. Each file in data_files must
    exist inside the given runfolder (otherwise an IOError is rased).

    If write_plot_data is given as True, call the `write` method of the data
    set instance. The outfile is in the given runfolder, with its name
    depending on the corresponding input file. For example,

        psi00_cavity.dat    => psi00_cavity_excitation_plot.dat
        psi01_q1.dat        => psi01_q1_exciation_plot.dat
        psi10_q2.dat        => psi10_q1_exciation_plot.dat
        psi11_phases.dat    => psi11_logical_pop_plot.dat
        rho00_popdyn.dat    => rho00_logical_pop_plot.dat
        ...

    '''
    file_pt = r'(psi|rho)(00|01|10|11)_(cavity|q1|q2|phases|popdyn)\.dat'
    file_rx = re.compile(file_pt)
    data = [] # array for collecting *all* datasets
    for file in data_files:
        m = file_rx.match(file)
        if not m:
            raise ValueError("File must match pattern %s" % file_rx.pattern)
        filename = os.path.join(runfolder, file)
        if os.path.isfile(filename):
            if m.group(3) in ['phases', 'popdyn']:
                data.append(PopulationDataSet(filename,
                            hilbert_space=(m.group(1)=='psi')))
            else:
                data.append(ExcitationDataSet(filename))
            if write_plot_data:
                prefix = m.group(1) + m.group(2)
                outfile = {
                    'phases': "%s_logical_pop_plot.dat" % prefix,
                    'popdyn': "%s_logical_pop_plot.dat" % prefix,
                    'cavity': "%s_cavity_excitation_plot.dat" % prefix,
                    'q1'    : "%s_q1_excitation_plot.dat" % prefix,
                    'q2'    : "%s_q2_excitation_plot.dat" % prefix
                }
                data[-1].write(os.path.join(runfolder,
                               outfile[m.group(3)]))
        else:
            raise IOError("File %s must exist in runfolder" % filename)
    return data


def full_qnums(level_index, n_qubit, n_cavity):
    """
    Given a 1-based level_index, return tuple of 0-based quantum numbers
    (i, j, n)
    """
    l = level_index - 1
    nn = n_qubit * n_cavity
    i = l / nn
    l = l - i * nn
    j = l / n_cavity
    n = l - j * n_cavity
    return (i, j, n)


def red_qnums(level_index, nq):
    """
    Given 1-based level_index, return tuple of 0-based quantum numbers (i, j)
    """
    l = level_index - 1
    i = l / nq
    j = l - i*nq
    return (i,j)


def write_full_ham_matrix(h, outfile, comment, nq, nc, rwa_factor,
    conversion_factor, unit, complex=False, check_hermitian=True):
    """ Write out numpy matrix in sparse format to outfile

        The given 'unit' will be written to the header of outfile, and the
        Hamiltonian will be converted using conversion_factor, as well as
        multiplied with rwa_factor. Multiplying with rwa_factor is to
        account for the reduction of the pulse amplitude by a factor 1/2,
        which was not taken into account in the analytic derivations.
        Consequently, an Hamiltonian connected to Omega^n will have to
        receive an rwa_factor for 1 / 2^n

        If complex is True, an extra column is written for the imaginary part
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
    if check_hermitian:
        for i in xrange(nq*nq*nc):
            if (h[i,i].imag != 0.0):
                print >> sys.stderr, "Matrix hast complex entries on diagonal"
                sys.exit(1)
            for j in xrange(i+1, nq*nq*nc):
                if (abs(h[i,j] - h[j,i].conjugate()) > 1.0e-15):
                    print >> sys.stderr, "Matrix is not Hermitian"
                    sys.exit(1)
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
                ii, ji, ni = full_qnums(i, nq, nc)
                ij, jj, nj = full_qnums(j, nq, nc)
                if complex:
                    print >> out_fh, fmt % (i, j, v.real, v.imag,
                                            ii, ji, ni, ij, jj, nj)
                else:
                    print >> out_fh, fmt % (i, j, v.real,
                                            ii, ji, ni, ij, jj, nj)


def write_red_ham_matrix(h, outfile, comment, nq, rwa_factor,
    conversion_factor, unit, complex=False):
    """ Write out numpy matrix in sparse format to outfile

        The given 'unit' will be written to the header of outfile, and the
        Hamiltonian will be converted using conversion_factor, as well as
        multiplied with rwa_factor. Multiplying with rwa_factor is to
        account for the reduction of the pulse amplitude by a factor 1/2,
        which was not taken into account in the analytic derivations.
        Consequently, an Hamiltonian connected to Omega^n will have to
        receive an rwa_factor for 1 / 2^n

        If complex is True, an extra column is written for the imaginary part
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
                ii, ji = red_qnums(i, nq)
                ij, jj = red_qnums(j, nq)
                if complex:
                    print >> out_fh, fmt % (i, j, v.real, v.imag,
                                            ii, ji, ij, jj)
                else:
                    print >> out_fh, fmt % (i, j, v.real, ii, ji, ij, jj)

