"""
Reading and writing data
"""
import re
import numpy as np
from QDYN.units import NumericConverter


def read_params(config, unit):
    """
    Extract the qubit parameters from the given config file and return them in
    a dictionary
    """
    result = {}

    float_params = ['w_c', 'w_1', 'w_2', 'w_d', 'alpha_1', 'alpha_2', 'g_1',
                    'g_2', 'zeta']
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

