#!/usr/bin/evn python
# coding: utf-8

### Preamble

# In[2]:

import os
from functools import partial
import numpy as np
import sympy
from sympy.physics.quantum import TensorProduct
sympy.init_printing()
import QDYNTransmonLib.io
from QDYNTransmonLib.io import write_red_ham_matrix


def mat_sympy_to_numpy(m):
    """ Convert a symbolic SymPy Matrix to a NumPy 2D array

        Obviously, this only works after all symbols have been substituted with
        numerical values
    """
    return np.array(np.array(m), np.float64)

### Definition of Operators

# In[6]:

def b(nq):
    """ Return SymPy Matrix for the (single qubit) annihilation operator in the
        Hilbert space with nq levels.
    """
    result = sympy.Matrix(np.zeros((nq,nq), dtype=np.int))
    for i in xrange(1,nq):
        result[i-1,i] = sympy.sqrt(i)
    return result


# In[7]:

def Pi1(n, m, nq):
    """ Return the Sympy Matrix for the \identity \otimes |n><m| operator in the
        Hilbert space with nq levels
    """
    result = sympy.Matrix(np.zeros((nq,nq), dtype=np.int))
    result[n,m] = sympy.Integer(1)
    return TensorProduct(result, sympy.eye(nq))


# In[8]:

def Pi2(n, m, nq):
    """ Return the Sympy Matrix for the |n><m| \otimes \identity operator in the
        Hilbert space with nq levels
    """
    result = sympy.Matrix(np.zeros((nq,nq), dtype=np.int))
    result[n,m] = sympy.Integer(1)
    return TensorProduct(sympy.eye(nq), result)


# In[9]:

def b1dn_b1n_b2dm_b2m(n, m, nq):
    """ Return the operator
        (\Op{b}^\dagger)^n \Op{b}^n \otimes (\Op{b}^\dagger)^m \Op{b}^m

        Note that as a special case this also includes the number operator of
        the left (n=1, m=0) and the right qubit (n=0, m=1), as well as the
        identity (n=0, m=0)
    """
    left = sympy.eye(nq)
    for i in xrange(n):
        left *= b(nq).H
    for i in xrange(n):
        left *= b(nq)
    right = sympy.eye(nq)
    for i in xrange(m):
        right *= b(nq).H
    for i in xrange(m):
        right *= b(nq)
    return TensorProduct(left, right)


# In[10]:

def ham_ops(nq):
    """ Return dictionary of operators occurring in the Hamiltonian with nq
        levels
    """
    identity = sympy.eye(nq)
    Op = {
    'b_1'                 : TensorProduct(b(nq), identity),
    'b_2'                 : TensorProduct(identity, b(nq)),
    'n_1'                 : TensorProduct(b(nq).H*b(nq), identity),
    'n_2'                 : TensorProduct(identity, b(nq).H*b(nq)),
    'n_1 n_2'             : TensorProduct(b(nq).H*b(nq), b(nq).H*b(nq)),
    'b_1^+ b_1^+ b_1 b_1' : TensorProduct(b(nq).H*b(nq).H*b(nq)*b(nq),
                                          identity),
    'b_2^+ b_2^+ b_2 b_2' : TensorProduct(identity,
                                          b(nq).H*b(nq).H*b(nq)*b(nq)),
    'Pi_1'                : partial(Pi1, nq=nq),      # stays a function
    'Pi_2'                : partial(Pi2, nq=nq),      # stays a function
    '(b_1^+)^n b_1^n (b_2^+)^m b_2^m': partial(b1dn_b1n_b2dm_b2m, nq=nq)
    }
    return Op


### Symbols

# In[11]:

def ham_symbols(nq):
    """ Return dictionary of symbols occurring in the Hamiltonian with nq
        levels
    """
    symbols = {}
    for symbol in [
    "delta_1", "delta_2", "alpha_1", "alpha_2", "Delta", "Delta_1", "Delta_2",
    "lambda_1", "lambda_2", "zeta"]:
        symbols[symbol] = sympy.Symbol(symbol, real=True)
    symbols['chi_1'] = sympy.Symbol('chi_{1,1}', real=True)
    symbols['chi_2'] = sympy.Symbol('chi_{2,1}', real=True)
    chis1 = [sympy.Symbol('chi_{1,0}', real=True), symbols['chi_1']]
    chis2 = [sympy.Symbol('chi_{2,0}', real=True), symbols['chi_2']]
    for i in xrange(2,nq):
        chis1.append(sympy.Symbol(r'chi_{1,%d}' % i, real=True))
        chis2.append(sympy.Symbol(r'chi_{2,%d}' % i, real=True))
    symbols['chis1'] = chis1
    symbols['chis2'] = chis2
    return symbols


### Effective Hamiltonian

# We define the Hamiltonian as
# $$\Op{H} = \Op{H}_0 + \Omega \Op{H}_1 + \Omega^2 \Op{H}_2 + \Omega^3 \Op{H}_3$$
# with one routine for each $\Op{H}_i$.
#
# Each of these routines gets a dictionary of symbols to use (as returned by
# `ham_symbols`), and a dictionary of operators (as returned by `ham_ops`).
# Also the parameter `get_parts` can be set to `True` in order to get multiple
# matrices for each term, for easier interpretation.

# In[12]:

def ham0(ham_symbols, Op, get_parts=False):
    """ Return formula for the drift Hamiltonian """
    alpha_1  = ham_symbols['alpha_1']
    alpha_2  = ham_symbols['alpha_2']
    Delta_1  = ham_symbols['Delta_1']
    Delta_2  = ham_symbols['Delta_2']
    zeta     = ham_symbols['zeta']
    chis1    = ham_symbols['chis1']
    chis2    = ham_symbols['chis2']
    nq = len(chis1)

    parts = []
    parts.append((Delta_1) * Op['n_1'])
    parts.append((Delta_2) * Op['n_2'])
    parts.append((alpha_1/2) * Op['b_1^+ b_1^+ b_1 b_1'])
    parts.append((alpha_2/2) * Op['b_2^+ b_2^+ b_2 b_2'])
    parts.append(zeta * Op['n_1 n_2'])

    chi_terms1 = (chis1[0])* Op['(b_1^+)^n b_1^n (b_2^+)^m b_2^m'](1,0)
    for n in xrange(1,nq):
        chi_terms1 += (chis1[n] / (n+1))* Op['(b_1^+)^n b_1^n (b_2^+)^m b_2^m'](n+1,0)
    parts.append(chi_terms1)

    chi_terms2 = (chis2[0])* Op['(b_1^+)^n b_1^n (b_2^+)^m b_2^m'](0,1)
    for n in xrange(1,nq):
        chi_terms2 += (chis2[n] / (n+1))* Op['(b_1^+)^n b_1^n (b_2^+)^m b_2^m'](0,n+1)
    parts.append(chi_terms2)

    if get_parts:
        return parts
    else:
        ham = parts[0]
        for i in xrange(1,len(parts)):
            ham += parts[i]
        return ham


# In[13]:

def ham1(ham_symbols, Op, get_parts=False):
    """ Return formula for drive Hamiltonian linear in field """
    lambda_1 = ham_symbols['lambda_1']
    lambda_2 = ham_symbols['lambda_2']
    parts = []
    parts.append(lambda_1 * (Op['b_1'].H + Op['b_1']))
    parts.append(lambda_2 * (Op['b_2'].H + Op['b_2']))
    if get_parts:
        return parts
    else:
        ham = parts[0]
        for i in xrange(1,len(parts)):
            ham += parts[i]
        return ham


# In[14]:

def ham2(ham_symbols, Op, get_parts=False):
    """ Return formula for drive Hamiltonian quadratic in field """

    chi_1 = ham_symbols['chi_1']
    chi_2 = ham_symbols['chi_2']
    Delta = ham_symbols['Delta']
    chis1 = ham_symbols['chis1']
    chis2 = ham_symbols['chis2']
    nq    = len(chis1)

    term1 = -(chis1[0] * (1/Delta**2)) * Op['n_1']
    for n in xrange(1, nq): # 1..nq-1
        term1 -= (chis1[n] * (1/Delta**2)) * Op['(b_1^+)^n b_1^n (b_2^+)^m b_2^m'](n+1,0)

    term2 = -(chis2[0] * (1/Delta**2)) * Op['n_2']
    for n in xrange(1, nq): # 1..nq-1
        term2 -= (chis2[n] * (1/Delta**2)) * Op['(b_1^+)^n b_1^n (b_2^+)^m b_2^m'](0,n+1)

    parts = []
    parts.append(term1)
    parts.append(term2)
    parts.append((-chi_1 * (chi_1**2 + 3 * chi_2**2) / Delta**4) * Op['n_1'])
    parts.append((-chi_2 * (3 * chi_1**2 + chi_2**2) / Delta**4) * Op['n_2'])

    def ho_term(n, m, Delta, chi1n, chi2m, Op):
        """ Return term proportional to
            (b_1^\dagger)^n (b_1)^n \otimes (b_2^\dagger)^m (b_2)^m
        """
        s1 = (2 * chi1n * chi2m) / Delta**3
        s2 = (4 * chi1n * chi2m * (chi1n**2 + chi2m**2)) / Delta**5
        return (s1 + s2) * Op['(b_1^+)^n b_1^n (b_2^+)^m b_2^m'](n, m)

    ho_terms =  ho_term(1, 1, Delta, chis1[1], chis2[1], Op)
    for n in xrange(1,nq): # 1..nq-1
        for m in xrange(1,nq): # 1..nq-1
            if not (n == m == 1):
                ho_terms += ho_term(n, m, Delta, chis1[n], chis2[m], Op)

    parts.append(ho_terms)

    if get_parts:
        return parts
    else:
        ham = parts[0]
        for i in xrange(1,len(parts)):
            ham += parts[i]
        return ham


# In[15]:

def ham3(ham_symbols, Op, get_parts=False):
    """ Return formula for drive Hamiltonian cubic in field """
    lambda_1 = ham_symbols['lambda_1']
    lambda_2 = ham_symbols['lambda_2']
    Delta    = ham_symbols['Delta']
    chi11    = ham_symbols['chis1'][1]
    chi21    = ham_symbols['chis2'][1]

    parts = []
    parts.append((- (2 * lambda_1 * chi11**2) / Delta**4)
                 * (Op['b_1'].H + Op['b_1']))
    parts.append((- (2 * lambda_2 * chi21**2) / Delta**4)
                 * (Op['b_2'].H + Op['b_2']))

    if get_parts:
        return parts
    else:
        ham = parts[0]
        for i in xrange(1,len(parts)):
            ham += parts[i]
        return ham


# The above routines combine into `get_ham_parts`, which constructs the
# matrices for a common set of operators and symbos, prints them on screen, and
# returns them as a list

# In[16]:

def get_ham_parts(ops, symbols, summed=False):
    """ Return and the individual terms of the
        Hamiltonian

        The returned list will contain 4 Hamiltonians associated with Omega^i
    """

    def process_parts(parts, ham_terms, summed):
        if summed:
            ham_terms.append(parts)
        else:
            for i, part in enumerate(parts):
                if i == 0:
                    ham_terms.append(part)
                else:
                    ham_terms[-1] += part

    ham_terms = []

    parts = ham0(symbols, ops, get_parts=(not summed))
    process_parts(parts, ham_terms, summed)

    parts = ham1(symbols, ops, get_parts=(not summed))
    process_parts(parts, ham_terms, summed)

    parts = ham2(symbols, ops, get_parts=(not summed))
    process_parts(parts, ham_terms, summed)

    parts = ham3(symbols, ops, get_parts=(not summed))
    process_parts(parts, ham_terms, summed)

    return ham_terms



#### Entangling Energy

# In[17]:

def entangling_energy(ham_parts, Omega, formulas=None):
    """ Return the entangling energy chi = E00 - E01 - E10 + E11

        ham_parts is an array of Hamiltonian matrices (symbolic or numeric),
        [H_0, H_1, H_2, ...] where H_i connect to Omega^i
    """
    nq = int(sympy.sqrt(ham_parts[0].shape[0]))
    E00 = ham_parts[0][0,0]
    E01 = ham_parts[0][1,1]
    E10 = ham_parts[0][nq,nq]
    E11 = ham_parts[0][nq+1,nq+1]
    for i, H in enumerate(ham_parts[1:]):
        E00 += H[0,0]       * Omega**(i+1)
        E01 += H[1,1]       * Omega**(i+1)
        E10 += H[nq,nq]     * Omega**(i+1)
        E11 += H[nq+1,nq+1] * Omega**(i+1)
    result = E00 - E01 - E10 + E11
    if formulas is None:
        return result
    else:
        return result.subs(formulas)


# In[18]:

### Plugging in the parameters

# In[21]:

def get_formulas(params, symbols):
    """ Given a dictionary of parameters and a dictionary of symbols, generate
        a list of all the replacement substitutions that plug in the parameters
        for the symbols
    """
    formulas = []
    delta_1  = params['w_1'] - params['w_c']
    delta_2  = params['w_2'] - params['w_c']
    Delta    = params['w_c'] - params['w_d'] - params['g_1'] / delta_1 - params['g_2'] / delta_2
    Delta_1 = params['w_1'] - params['w_d']
    Delta_2 = params['w_2'] - params['w_d']
    for n, chi_1n in enumerate(symbols['chis1']):
        numerator = (n + 1) * params['g_1']**2 * (-params['alpha_1'])**n
        denominator = 1.0
        for m in xrange(n+1):
            denominator *= ((m * params['alpha_1']) + delta_1)
        formulas.append((chi_1n, numerator/denominator))
    for n, chi_2n in enumerate(symbols['chis2']):
        numerator = (n + 1) * params['g_2']**2 * (-params['alpha_2'])**n
        denominator = 1.0
        for m in xrange(n+1):
            denominator *= ((m * params['alpha_2']) + delta_2)
        formulas.append((chi_2n, numerator/denominator))
    formulas += [
        (symbols['lambda_1'], -2.0 * params['g_1']/delta_1),
        (symbols['lambda_2'], -2.0 * params['g_2']/delta_2),
        (symbols['delta_1'],  delta_1),
        (symbols['delta_2'],  delta_2),
        (symbols['Delta'],    Delta),
        (symbols['Delta_1'],  Delta_1),
        (symbols['Delta_2'],  Delta_2),
        (symbols['zeta'],     params['zeta']),
        (symbols['alpha_1'],  params['alpha_1']),
        (symbols['alpha_2'],  params['alpha_2']),
    ]
    return formulas


def get_ham_matrices(ops, symbols, params, formulas, print_vals=False):
    """ Return list of 4 numpy matrices for ham0, ham1, ham2, ham3
        with all parameters plugged in
    """
    if print_vals:
        printed = []
        for sym_name, sym in symbols.items():
            if isinstance(sym, list):
                for i, sub_sym in enumerate(sym):
                    print "%s%d = %f # MHz" % (sym_name, i,
                                               sympy.N(sub_sym.subs(formulas)))
            else:
                if not sym_name in printed:
                    print "%s = %f # MHz" % (sym_name,
                                             sympy.N(sym.subs(formulas)))
                    printed.append(sym_name)
        for sym_name, val in params.items():
            if not sym_name in printed:
                print "%s = %f # MHZ" % (sym_name, val)
                printed.append(sym_name)

    ham_matrices = []

    h0 = ham0(symbols, ops, get_parts=False).subs(formulas)
    ham_matrices.append(mat_sympy_to_numpy(h0))

    h1 = ham1(symbols, ops, get_parts=False).subs(formulas)
    ham_matrices.append(mat_sympy_to_numpy(h1))

    h2 = ham2(symbols, ops, get_parts=False).subs(formulas)
    ham_matrices.append(mat_sympy_to_numpy(h2))

    h3 = ham3(symbols, ops, get_parts=False).subs(formulas)
    ham_matrices.append(mat_sympy_to_numpy(h3))

    return ham_matrices


### Output

def write_ham_matrices(ham_matrices, params, outfolder):
    """ Write the given array of 4 Hamiltonian matrices
        to the files ham_eff_drift.dat, ham_eff_ctrl.dat,
        ham_eff_ctrl2.dat, ham_eff_ctrl3.dat inside the given runfolder

        The value for each Hamiltonian will be scaled by 1 / 2^n, to account
        for the RWA
    """
    try:
        os.makedirs(outfolder)
    except OSError:
        pass # folder exists

    nq = params['n_qubit']

    comment = "# "
    comment += "generated by %s [RWA];" % os.path.basename(__file__)
    for sym_name, val in params.items():
        comment += " %s = %f  MHZ," % (sym_name, val)



    h0 = ham_matrices[0]
    outfile = os.path.join(outfolder, 'ham_eff_drift.dat')
    write_red_ham_matrix(h0, outfile, comment, nq, 1.0, 1.0e-3, 'GHz')

    h1 = ham_matrices[1]
    outfile = os.path.join(outfolder, 'ham_eff_ctrl.dat')
    write_red_ham_matrix(h1, outfile, comment, nq, 0.5, 1.0, 'au')

    h2 = ham_matrices[2]
    outfile = os.path.join(outfolder, 'ham_eff_ctrl2.dat')
    write_red_ham_matrix(h2, outfile, comment, nq, 0.25, 6.5796839e+09, 'au')

    h3 = ham_matrices[3]
    outfile = os.path.join(outfolder, 'ham_eff_ctrl3.dat')
    write_red_ham_matrix(h3, outfile, comment, nq, 0.125, 4.3292241e+19, 'au')


def generate_ham_files(config, outfolder):
    params = QDYNTransmonLib.io.read_params(config, 'MHz')
    nq = params['n_qubit']
    ops = ham_ops(nq)
    symbols = ham_symbols(nq)
    formulas = get_formulas(params, symbols)
    ham_matrices = get_ham_matrices(ops, symbols, params, formulas,
                                    print_vals=True)
    write_ham_matrices(ham_matrices, params, outfolder=outfolder)
