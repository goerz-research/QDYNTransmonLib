import os
import subprocess
from QDYN.gate2q import Gate2Q
from QDYN.linalg import norm
from QDYN.shutil import mkdir
from QDYNTransmonLib.io import read_params

# If more than a feasible number of levels are required to converge, we'll
# simply give up on that set of parameters
NQ_MAX = 17 # maximum feasible number of qubit levels
NC_MAX = 250 # maximum feasible number of cavity levels

NQ_MAX_USED = 0 # maximum qubit levels required, over all propagations
NC_MAX_USED = 0 # maximum cavity levels required, over all propagations


# If propagate is False, not propagation is done, but an existing U.dat in the
# runfolder is read in and returned.
PROPAGATE = True


class NotConvergedError(Exception):
    pass


def propagate(write_run, config, params, pulses, commands, runfolder):
    """
    Propagate the Transmon system

    Arguments
    ---------

    write_run: callable
        `write_run(config, params, pulses, runfolder)` writes all the run data
        to the given runfolder
    config: str
        Template string for the config files.
    param: dict
        Dictionary of replacements for the config template
        Keys 'T' and 'nt' will be set automatically based on the first pulse
    pulses: list of QDYN.pulse.Pulse objects
        List of pulses to be propagated
    commands: str of list of str
        Shell commands to do the propagation

    Returns
    -------

    U: QDYN.gate2q.Gate2Q object
        Gate resulting from the propagation

    As a side effect, `params[T]` and `params[nt]` will be set according to the
    properties of `pulses[0]`
    """
    U_dat_file = os.path.join(runfolder, 'U.dat')
    if not PROPAGATE:
        if os.path.isfile(U_dat_file):
            U = Gate2Q(file=U_dat_file)
            return U
    mkdir(runfolder)
    if len(pulses) > 0:
        params['T'] = pulses[0].T()
        params['nt'] = len(pulses[0].tgrid) + 1
    if len(pulses) > 1:
        for pulse in pulses[1:]:
            assert abs(pulse.T() - params['T']) < 1.0e-15, \
            "All pulses must be defined on the same time grid"
            assert len(pulse.tgrid) + 1 == params['nt'], \
            "All pulses must be defined on the same time grid"
    write_run(config, params, pulses, runfolder)
    with open(os.path.join(runfolder, 'prop.sh'), 'w') as prop_sh:
        if type(commands) in [list, tuple]:
            commands = "\n".join(commands)
        else:
            commands = str(commands)
        prop_sh.write(commands)
    if os.path.isfile(U_dat_file):
        os.unlink(U_dat_file)
    subprocess.call(['bash', 'prop.sh'], cwd=runfolder)
    U = Gate2Q(file=U_dat_file)
    return U


def converged_propagate(write_run, config, params, pulses, commands, runfolder,
    nq_step=1, nc_step=10, limit=0.01):
    """
    Run `propagate` repeatedly, varying the number of qubit levels nq, and
    cavity levels nc until convergence is reached (the norm of the obtained
    gate is stable within the given limit)

    The initial values for nq and nc are taken from params['nq'] and
    params['ns'].

    Raises a NotConvergedError if convergence cannot be obtained.

    Paramters
    ---------

    write_run, config, params, pulses, commands, runfolder: see `propagate`

    nq_step: int
        step by which to increase nq in each iteration
    nc_step: int
        step by which to increase nc in each iteration
    limit: float
        convergence is reached as soon as norm(U1 - U2) < limit.

    Returns
    -------

    U: QDYN.gate2q.Gate2Q object
        Gate resulting from the propagation

    As a side effect, params['nq'] and params['nc'] will contain the
    "converged" values for the number of qubit and cavity levels.

    Note that params['nq'] will always be increased by at least nq_step,
    compared to the input value, and equivalently for params['nc'].
    """
    global NQ_MAX_USED
    global NC_MAX_USED

    def compare_results(U1, U2):
        return norm(U1 - U2) < limit

    if not PROPAGATE:
        U_dat_file = os.path.join(runfolder, 'U.dat')
        if os.path.isfile(U_dat_file):
            config_params = read_params(os.path.join(runfolder, 'config'),
                                        unit='MHz')
            params['nq'] = config_params['n_qubit']
            params['nc'] = config_params['n_cavity']
            U = Gate2Q(file=U_dat_file)
            return U

    with open(os.path.join(runfolder, 'converged_prop.log'), 'w') as log_fh:
        log_header_fmt = "# "+"%6s"+"%8s"+"%15s"*2+"\n"
        log_fh.write(log_header_fmt%("nq", "nc", "Norm(U)", "C(U)"))
        log_fmt = "%8d"*2+"%15.6e"*2+"\n"
        U = propagate(write_run, config, params, pulses, commands, runfolder)
        log_fh.write(log_fmt%(params['nq'], params['nc'], norm(U),
                     U.closest_unitary().concurrence()))
        converged = False
        while not converged:
            # first, we increase the cavity until we reach convergence
            c_converged = False
            while not c_converged:
                params['nc'] += nc_step
                if (params['nc'] > NC_MAX):
                    raise NotConvergedError
                next_U = propagate(write_run, config, params, pulses, commands,
                                runfolder)
                log_fh.write(log_fmt%(params['nq'], params['nc'], norm(next_U),
                             next_U.closest_unitary().concurrence()))
                c_converged = compare_results(U, next_U)
                U = next_U
            params['nc'] -= nc_step
            # then, we increase the qubit until we reach convergence
            q_converged = False
            while not q_converged:
                params['nq'] += nq_step
                if (params['nq'] > NQ_MAX):
                    raise NotConvergedError
                next_U = propagate(write_run, config, params, pulses, commands,
                                runfolder)
                log_fh.write(log_fmt%(params['nq'], params['nc'], norm(next_U),
                             next_U.closest_unitary().concurrence()))
                q_converged = compare_results(U, next_U)
                U = next_U
            params['nq'] -= nq_step
            # check that we are really converged with respect to both the qubit
            # and the cavity. If necessary, we start over.
            converged = False
            params['nc'] += nc_step
            params['nq'] += nq_step
            next_U = propagate(write_run, config, params, pulses, commands,
                            runfolder)
            log_fh.write(log_fmt%(params['nq'], params['nc'], norm(next_U),
                            next_U.closest_unitary().concurrence()))
            converged = compare_results(U, next_U)
            U = next_U
            if params['nq'] > NQ_MAX_USED:
                NQ_MAX_USED = params['nq']
            if params['nc'] > NC_MAX_USED:
                NC_MAX_USED = params['nc']
    return U
