import os
from QDYN.pulse import Pulse
import numpy as np
import re
from mgplottools.mpl import get_color
import matplotlib.patches as mpatches
import matplotlib.pylab as plt

class FullPopPlot(object):

    def __init__(self, runfolder, hilbert_space=True, figsize=(12,4), dpi=300,
        left_margin=0.1, right_margin=0.2, bottom_margin=0.1, top_margin=0.1,
        gap=0.05, legend_gap=0.05, h_pop=0.25):
        self.pop           = {'00': None, '01': None, '10': None, '11': None}
        self.exc = {}
        self.exc['cavity'] = {'00': None, '01': None, '10': None, '11': None}
        self.exc['q1']     = {'00': None, '01': None, '10': None, '11': None}
        self.exc['q2']     = {'00': None, '01': None, '10': None, '11': None}
        self.pulse = None
        self.tgrid = None
        self.figsize = figsize
        self.left_margin   = left_margin
        self.right_margin  = right_margin
        self.bottom_margin = bottom_margin
        self.top_margin    = top_margin
        self.gap = gap
        self.legend_gap = legend_gap
        self.h_pop = h_pop
        self.dpi = dpi
        self._ax = {}
        self.load(runfolder, hilbert_space)

    def load(self, runfolder, hilbert_space=True):
        if hilbert_space:
            files = [
            'psi00_cavity.dat', 'psi00_q1.dat', 'psi00_q2.dat', 'psi00_phases.dat',
            'psi01_cavity.dat', 'psi01_q1.dat', 'psi01_q2.dat', 'psi01_phases.dat',
            'psi10_cavity.dat', 'psi10_q1.dat', 'psi10_q2.dat', 'psi10_phases.dat',
            'psi11_cavity.dat', 'psi11_q1.dat', 'psi11_q2.dat', 'psi11_phases.dat'
            ]
        else:
            files = [
            'rho00_cavity.dat', 'rho00_q1.dat', 'rho00_q2.dat', 'rho00_popdyn.dat',
            'rho01_cavity.dat', 'rho01_q1.dat', 'rho01_q2.dat', 'rho01_popdyn.dat',
            'rho10_cavity.dat', 'rho10_q1.dat', 'rho10_q2.dat', 'rho10_popdyn.dat',
            'rho11_cavity.dat', 'rho11_q1.dat', 'rho11_q2.dat', 'rho11_popdyn.dat'
            ]
        self.exc['cavity']['00'], self.exc['q1']['00'], self.exc['q2']['00'], self.pop['00'], \
        self.exc['cavity']['01'], self.exc['q1']['01'], self.exc['q2']['01'], self.pop['01'], \
        self.exc['cavity']['10'], self.exc['q1']['10'], self.exc['q2']['10'], self.pop['10'], \
        self.exc['cavity']['11'], self.exc['q1']['11'], self.exc['q2']['11'], self.pop['11'] \
        = collect_pop_plot_data(files, runfolder)
        self.tgrid = self.exc['cavity']['00'].tgrid

        # Load pulse
        filename = os.path.join(runfolder, 'pulse.dat')
        if os.path.isfile(filename):
            self.pulse = Pulse(filename)

    def render(self, basis_state, ax_pop, ax_q1, ax_q2, ax_cavity,
        legend=False):

        # population dynamics
        p00, = ax_pop.plot(self.tgrid, self.pop[basis_state].pop00,
                           color=get_color('red'))
        p01, = ax_pop.plot(self.tgrid, self.pop[basis_state].pop01,
                           color=get_color('blue'))
        p10, = ax_pop.plot(self.tgrid, self.pop[basis_state].pop10,
                           color=get_color('orange'))
        p11, = ax_pop.plot(self.tgrid, self.pop[basis_state].pop11,
                           color=get_color('purple'))
        pop_sum =   self.pop[basis_state].pop00 + self.pop[basis_state].pop10 \
                  + self.pop[basis_state].pop01 + self.pop[basis_state].pop11
        tot, = ax_pop.plot(self.tgrid, pop_sum, color='Grey')
        ax_pop.axhline(y=1, ls='--', color='Gray')
        ax_pop.fill_between(self.pulse.tgrid,
             np.abs(self.pulse.amplitude)/np.max(np.abs(self.pulse.amplitude)),
             color='Blue', facecolor='Blue', alpha=0.1, rasterized=True)
        blue_patch = mpatches.Patch(color='Blue', alpha=0.1, label='pulse')
        if legend:
            # TODO: position to the right
            ax_pop.legend([tot, p00, p01, p10, p11, blue_patch],
                        ["log. subsp.", "00", "01", "10", "11"],
                        ncol=1, loc='center left',
                        handlelength=3, labelspacing=0.3, borderpad=0.0,
                        borderaxespad=0.0)

        # left qubit excitation
        q1_data = self.exc['q1'][basis_state]
        ax_q1.fill_between(q1_data.tgrid, q1_data.sd, color='LightGray',
                    facecolor='LightGray', rasterized=True)
        pmq1, = ax_q1.plot(q1_data.tgrid, q1_data.mean, color='black',
                           rasterized=True)
        if legend:
            gray_patch = mpatches.Patch(color='LightGray')
            ax_q1.legend([pmq1, gray_patch],
                         [r'$\langle i \rangle$', r'$\sigma_i$'],
                         title="left qubit",
                         ncol=1, loc='center left',
                         handlelength=3, labelspacing=0.3, borderpad=0.0,
                         borderaxespad=0.0)

        # right qubit excitation
        q2_data = self.exc['q2'][basis_state]
        ax_q2.fill_between(q2_data.tgrid, q2_data.sd, color='LightGray',
                    facecolor='LightGray', rasterized=True)
        pmq2, = ax_q2.plot(q2_data.tgrid, q2_data.mean, color='black',
                           rasterized=True)
        if legend:
            gray_patch = mpatches.Patch(color='LightGray')
            ax_q2.legend([pmq2, gray_patch],
                         [r'$\langle j \rangle$', r'$\sigma_j$'],
                         title="right qubit",
                         ncol=1, loc='center left',
                         handlelength=3, labelspacing=0.3, borderpad=0.0,
                         borderaxespad=0.0)

        # cavity excitation
        cavity_data = self.exc['cavity'][basis_state]
        ax_cavity.fill_between(cavity_data.tgrid, cavity_data.sd,
                               color='LightGray', facecolor='LightGray',
                               rasterized=True)
        pmcavity, = ax_cavity.plot(cavity_data.tgrid, cavity_data.mean,
                                   color='black', rasterized=True)
        if legend:
            gray_patch = mpatches.Patch(color='LightGray')
            ax_cavity.legend([pmcavity, gray_patch],
                         [r'$\langle n \rangle$', r'$\sigma_n$'],
                         title="cavity",
                         ncol=1, loc='center left',
                         handlelength=3, labelspacing=0.3, borderpad=0.0,
                         borderaxespad=0.0)

    def plot(self, basis_states=('00', '01', '10', '11'), fig=None,
        legend=True):
        """
        Generate a plot of the Bloch sphere on the given figure
        """
        if fig is None:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        h = 1.0 - self.top_margin - self.bottom_margin # total height for state
        h_pop = self.h_pop
        h_exc = float(h - h_pop) / 3.0
        n_states = len(basis_states)
        w = (1.0 - self.left_margin - self.right_margin
             - (n_states-1)*self.gap) / n_states
        self._ax = {}
        ax = self._ax
        for i_state, basis_state in enumerate(basis_states):
            left_offset = self.left_margin + i_state*(w+self.gap)
            ax[basis_state] = {}
            ax[basis_state]['pop'] = fig.add_axes(
                [left_offset, self.bottom_margin, w, h_pop])
            ax[basis_state]['q1'] = fig.add_axes(
                [left_offset, self.bottom_margin+h_pop, w, h_exc])
            ax[basis_state]['q2'] = fig.add_axes(
                [left_offset, self.bottom_margin+h_pop+h_exc, w, h_exc])
            ax[basis_state]['cavity'] = fig.add_axes(
                [left_offset, self.bottom_margin+h_pop+2*h_exc, w, h_exc])

            show_legend = False
            if legend:
                if i_state == len(basis_states)-1:
                    show_legend = True
            self.render(basis_state, ax[basis_state]['pop'],
                        ax[basis_state]['q1'], ax[basis_state]['q2'],
                        ax[basis_state]['cavity'], legend=show_legend)

    def show(self):
        """
        Show a plot of all the populatio dynamics
        """
        self.plot()
        plt.show()


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
    Take an array of data_files that must match the regular expression

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


