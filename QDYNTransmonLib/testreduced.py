#import tempfile
"""
Code for testing the reduced model vs the full model
"""
import os
import shutil
from subprocess import call
from multiprocessing.dummy import Pool
import numpy as np
from QDYN.pulse import Pulse, tgrid_from_config, blackman
from QDYN.shutil import tail
from QDYN.gate2q import Gate2Q
from QDYNTransmonLib.holonomic import generate_ham_files
from QDYNTransmonLib.io import FullHamLevels, RedHamLevels, read_params


def get_config(config, replacements):
    """
    Return a modified config file string

    Parameters
    ----------

    config : str
        Multiline string, contents of config file
    replacements : array of tuples
        Each element of `replacements` is a tuple `(string, replacement)`

    Returns
    -------

    config : str
        Copy of input `config` with the given replacements
    """

    for s, r in replacements:
        config = config.replace(s, r)
    return config


def field_free_energy_diff(runfolder_full, runfolder_red):
    """
    Extract the energy levels in the logical subspace from the full and reduced
    runfolders and return their difference in GHz
    """
    full_levels = FullHamLevels(
                  os.path.join(runfolder_full, 'eigenvalues.dat'))
    full_E00 = full_levels.get(0, 0, 0)
    full_E01 = full_levels.get(0, 1, 0)
    full_E10 = full_levels.get(1, 0, 0)
    full_E11 = full_levels.get(1, 1, 0)

    red_levels = RedHamLevels(
                 os.path.join(runfolder_red, 'ham_eff_drift.dat'))
    red_E00 = 0.0
    red_E01 = red_levels.get(0, 1)
    red_E10 = red_levels.get(1, 0)
    red_E11 = red_levels.get(1, 1)

    delta00 = full_E00 - red_E00
    delta01 = full_E01 - red_E01
    delta10 = full_E10 - red_E10
    delta11 = full_E11 - red_E11

    return delta00, delta01, delta10, delta11



def run(config, nq_reduced, pulse_E0, notebook=True, folder="./"):
    """
    Run a propagation of the full and the reduced model for comparison

    Parameters
    ----------

    config : str
        Multiline string defining config file for the full propagation
    nq_reduced : int
        Number of qubit levels to use in the reduced model. Config file for
        reduced propagation will differ from config file for full propagation
        only by n_qubit
    pulse_E0 : float
        Ampltiude of the pulse to be used, in MHz. The pulse shape will be
        Blackman
    notebook : bool
        If True, print output, assuming that we're running inside an iPython
        Notebook
        Otherwise, no output
    folder: str
        Folder in which to do the propagation. The full and reduced propagation
        will each be done in a subfolder 'full' and 'reduced' respectively

    Returns
    -------

    Fidelity : float
        Hilbert-Schmidt-Overlap between the gates induced in the full and in
        the reduced model

    DeltaG : float
        Distance between full and reduced gate in Weyl chamber

    """
    runfolder_full = os.path.join(folder, 'full')
    runfolder_red = os.path.join(folder, 'reduced')
    for folder in (runfolder_full, runfolder_red):
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        call("mkdir -p %s" % folder, shell=True)

    # (1) full Hamiltonian ####################################################

    # Create config file
    config_full = config
    config_full_filename = os.path.join(runfolder_full, 'config')
    with open(config_full_filename, 'w') as config_fh:
        print >> config_fh, config_full

    # Create pulse
    tgrid, time_unit = tgrid_from_config(config_full_filename)
    dt = tgrid[1] - tgrid[0]
    t_stop = round(tgrid[-1] + 0.5*dt)
    ampl = pulse_E0 * blackman(tgrid, t_start=0.0, t_stop=t_stop)
    pulse = Pulse(tgrid=tgrid, amplitude=ampl, time_unit=time_unit,
                  ampl_unit='MHz', freq_unit='MHz', mode='real')
    pulse.write(os.path.join(runfolder_full, 'pulse.guess'))

    # Create run script (full)
    prop_script = os.path.join(runfolder_full, 'prop.sh')
    with open(prop_script, 'w') as run_fh:
        lines = [
        "#!/usr/bin/bash",
        "cd %s" % runfolder_full,
        "export OMP_NUM_THREADS=4",
        "tm_en_gh --rwa .",
        "tm_en_logical_eigenstates.py .",
        "tm_en_prop . > prop.log",
        "tail prop.log"
        ]
        for line in lines:
            print >> run_fh, line

    # Run
    if notebook:
        print "*** Full"
    call('bash %s' % prop_script, shell=True)
    if notebook:
        tail(prop_script, 12)
    U_full = Gate2Q(file=os.path.join(runfolder_full, 'U.dat'))
    if notebook:
        print "Concurrence (full) : ", U_full.concurrence()
        #show_U_arrow(U_full, name="U_{full}")

    # (2) reduced Hamiltonian #################################################

    # Create config file
    full_E = FullHamLevels(os.path.join(runfolder_full, 'eigenvalues.dat'))
    E00 = full_E.get(0, 0, 0) # GHz
    E01 = full_E.get(0, 1, 0) # GHz
    E10 = full_E.get(1, 0, 0) # GHz
    E11 = full_E.get(1, 1, 0) # GHz
    zeta = (E00 - E01 - E10 + E11) * 1000.0 # zeta in MHz
    config_red = get_config(config_full, replacements=[
                 ("n_qubit = 10", "n_qubit = %d" % nq_reduced),
                 ("zeta     = 0_MHz", "zeta = %f_MHz" % zeta)
                 ])
    config_red_filename = os.path.join(runfolder_red, 'config')
    with open(config_red_filename, 'w') as config_fh:
        print >> config_fh, config_red

    # Create pulse
    pulse.write(os.path.join(runfolder_red, 'pulse.guess'))

    # Write reduced Hamiltonian
    # tm_eff_gh_orig will give us the correct logical_eigenstates.dat
    call('tm_eff_gh_orig %s' % runfolder_red, shell=True)
    # we then overwrite the hamiltonian files with the proper ones
    generate_ham_files(config_red_filename, runfolder_red)

    # Create run script (full)
    prop_script = os.path.join(runfolder_red, 'prop.sh')
    with open(prop_script, 'w') as run_fh:
        lines = [
        "#!/usr/bin/bash",
        "cd %s" % runfolder_red,
        "export OMP_NUM_THREADS=4",
        "tm_eff_prop . > prop.log",
        ]
        for line in lines:
            print >> run_fh, line

    if notebook:
        print "*** Reduced"
    call('bash %s' % prop_script, shell=True)
    if notebook:
        tail(prop_script, 12)
    U_red = Gate2Q(file=os.path.join(runfolder_red, 'U.dat'))
    if notebook:
        print "Concurrence (red) : ", U_red.concurrence()
        #show_U_arrow(U_red, name="U_{red}")

    # (3) Compare #############################################################
    overlap = (U_full.H * U_red).trace()[0, 0]
    fidelity = abs(overlap**2) \
               / ( (U_full.H * U_full).trace()[0,0].real
                 * (U_red.H * U_red).trace()[0,0].real )

    g1_full, g2_full, g3_full = U_full.local_invariants()
    g1_red, g2_red, g3_red = U_red.local_invariants()
    DeltaG = np.sqrt((g1_red - g1_full)**2 + (g2_red - g2_full)**2
                     + (g3_red - g3_full)**2)
    return fidelity, DeltaG


def scan_dispersive(config_template, delta_1s):
    """
    Scan over different values of w_c - w_1, to see how the dispersive limit
    affects the effective model
    """
    with open('dispersive_scan.dat', 'w') as dat:
        print >> dat, "# scan over different qubit-cavity detunings, " \
                      "comparing full and effective model energy levels"
        print >> dat, "# e.g. D01 = E01 (full) - E01 (red); delta_1 = w_1-w_c"
        print >> dat, "# %13s%15s%15s%15s%15s%15s%15s%15s" % (
            'delta_1 [MHz]', 'g/delta_1', 'D00 [GHz]', 'D01 [GHz]',
            'D10 [GHz]', 'D11 [GHz]', '1-fid', 'DeltaG'
        )

        def worker(paramset):
            """pool thread worker"""
            fid, DeltaG = run(**paramset)
            folder = paramset['folder']
            runfolder_full = os.path.join(folder, 'full')
            runfolder_red = os.path.join(folder, 'reduced')
            D00, D01, D10, D11 \
            = field_free_energy_diff(runfolder_full, runfolder_red)
            qubit_params = read_params(os.path.join(runfolder_full, 'config'),
                                       'MHz')
            delta_1 = qubit_params['w_1'] - qubit_params['w_c']
            g = qubit_params['g_1']
            return "%15.1f%15.7E%15.7E%15.7E%15.7E%15.7E%15.7E%15.7E" % (
                    delta_1, g/delta_1, D00, D01, D10, D11, 1.0-fid, DeltaG)

        paramsets = []
        for delta_1 in delta_1s:
            w_c = 6850.0 - delta_1
            w_d = w_c - 40.0
            config = get_config(config_template, replacements={
                ('w_c      = 6000.0_MHz', 'w_c      = %f_MHz' % w_c),
                ('w_d      = 5960.0_MHz', 'w_d      = %f_MHz' % w_d),
            })
            paramsets.append(
                {'config' : config, 'nq_reduced' : 2, 'pulse_E0' :0.0,
                 'notebook' : False,
                 'folder': "./scan_dispersive/%d" % delta_1})
        pool = Pool(processes=4)
        results = pool.map(worker, paramsets)
        #results = []
        #for paramset in paramsets:
            #results.append(worker(paramset))
        print >> dat, "\n".join(results)

