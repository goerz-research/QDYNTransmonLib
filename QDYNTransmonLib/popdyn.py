"""
Tools for showing population dynamics of the Transmon system
"""
import os
from QDYN.pulse import Pulse
import numpy as np
import re
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pylab as plt
from matplotlib.offsetbox import AnchoredText

class FullPopPlot(object):
    """
    Plot generator for population dynamics of the transmon system

    Attributes
    ---------

    runfolder: str (read only)
        Folder from which data was read
    hilbert_space: boolean (read only)
        Indication for whether data was loaded from Hilbert space propagation
        files (psi*.dat) or Liouville space propagation files (rho*.dat)
    dpi: int
        Resolution when writing figure to file
    left_margin: float
        distance from the left figure edge to the leftmost panel (in unit).
        Makes room for y-axis labels and ticks
    right_margin: float
        distance from the rightmost panel to the right figure edge (in unit).
        Makes room for legend
    bottom_margin: float
        distance from bottom figure edge to the bottom panel (in unit). Makes
        room for the x-axis label and ticks
    top_margin: float
        distance from the top figure edge to the top panel (in unit). Makes
        room for the titles
    panel_width: float
        Width (in unit) of each panel
    gap: float
        Vertical gap (in unit) between the columns of panels. Makes room for
        y-axis ticks
    legend_gap: float
        Vertical gap (in unit) between the rightmost panels and the left edge
        of the legend.
    h_pop: float
        Height (in unit) of the population dynamics panels
    h_exc: float
        Height (in unit) of the qubit and cavity excitation dynamics panels
    xaxis_minor: int or None
        Number of minor ticks between majore ticks, for the x-axis.
    pop_top_buffer: float
        Space in the population dynamics panels (in unit) reserved for the
        in-panel legend (e.g. the pulse patch)
    exc_top_buffer: float
        Space in the excitation dynamics panel (in unit) reserved for in-panel
        labels
    title: str or None
        Pattern for title of each panel column. The placeholder '%s' will be
        filled by '00', '01', '10', '11', as appropriate.
    styles: dict of dicts
        Dictionary of line styles to be used when plotting the various lines in
        the figure. The keys are
        '00': styles to be used for the projection onto the 00 states
        '01': styles to be used for the projection onto the 01 states
        '10': styles to be used for the projection onto the 10 states
        '11': styles to be used for the projection onto the 11 states
        'tot': styles to be used for the projection onto the logical subspace
        'pop_hline': styles for the horizontal line indicating 1.0 in the
                     population plot
        'pulse': styles to be used for the plot of the pulse shape
        'sd': styles to be used for the plot of the standard deviations of the
              mean excitation numbers
        'mean': styles to be used for the plot of the mean excitation numbers
    unit: 'cm' or 'inch'
        Unit in which the layout parameters are given
    pop: dict
        pop['00'] contains a PopulationDataSet for the propagation of the 00
        state, and equivalently for 01, 10, and 11
    exc: dict
        pop['cavity']['00'] contains an ExcitationDataSet for cavity in the
        propagation of the 00 state, and equivalently for 01, 10, and 11.
        pop['q1']['00'] and pop['q2']['00'] contain the data for the qubit
        excitations
    ax: dict
        Dictionary of all the panels (instances of matplotlib.axes.Axes)
        that have been rendered onto a figure.  E.g. for the 00 state, the four
        panels are ax['00']['pop'], ax['00']['q1'], ax['00']['q2'],
        ax['00']['cavity']
    pulse: QDYN.pulse.Pulse instance
        Pulse loaded from the runfolder
    tgrid: numpy array
        time grid for the propagation
    peak_qubit_excitation: float (read only)
        Peak qubit excitation, from the propagation of any of the logical
        states
    peak_cavity_excitation float (read only)
        Peak cavity excitation, from the propagation of any of the logical
        states
    """

    def __init__(self, runfolder, hilbert_space=True, dpi=300,
        left_margin=1.2, right_margin=2.0, bottom_margin=1.25, top_margin=0.5,
        panel_width=5.0, gap=1.0, legend_gap=0.25, h_pop=2.5, h_exc=1.8,
        xaxis_minor=None, pop_top_buffer=0.8, exc_top_buffer=0.5,
        title=r'$\vert \Psi(t=0) \rangle = \vert %s \rangle$',
        unit='cm'):
        """
        Load data from the given runfolder. All keyword arguments set the
        corresponding attributes.
        """
        self.pop           = {'00': None, '01': None, '10': None, '11': None}
        self.exc = {}
        self.exc['cavity'] = {'00': None, '01': None, '10': None, '11': None}
        self.exc['q1']     = {'00': None, '01': None, '10': None, '11': None}
        self.exc['q2']     = {'00': None, '01': None, '10': None, '11': None}
        self.pulse = None
        self.tgrid = None
        self._runfolder = runfolder
        self._hilbert_space = hilbert_space
        self.left_margin   = float(left_margin)
        self.right_margin  = float(right_margin)
        self.bottom_margin = float(bottom_margin)
        self.top_margin    = float(top_margin)
        self.panel_width   = float(panel_width)
        self.gap           = float(gap)
        self.legend_gap    = float(legend_gap)
        self.h_pop         = float(h_pop)
        self.h_exc         = float(h_exc)
        self.title = title
        self.unit = unit
        self.dpi = dpi
        self.pop_top_buffer = pop_top_buffer
        self.exc_top_buffer = exc_top_buffer
        self.xaxis_minor = xaxis_minor
        self._peak_qubit_excitation = 0.0
        self._peak_cavity_excitation = 0.0
        self.ax = {}
        self.styles = {
          '00' : {'label': '00'},
          '01' : {'label': '01'},
          '10' : {'label': '10'},
          '11' : {'label': '11'},
          'tot': {'color': 'black', 'label': 'total'},
          'pop_hline': {'ls': '--', 'color': 'Gray'},
          'pulse': {'color': 'Blue', 'alpha': 0.1,
                    'rasterized': True, 'label': 'pulse'},
          'sd': {'color': 'LightGray', 'rasterized': True},
          'mean': {'color': 'black', 'rasterized': True},
        }
        self.load(runfolder, hilbert_space)

    @property
    def runfolder(self):
        """Return the runfolder that init() was called with"""
        return self._runfolder

    @property
    def hilbert_space(self):
        """Return the hilbert_space value that init() was called with"""
        return self._hilbert_space

    @property
    def peak_qubit_excitation(self):
        """Return the peak qubit exitation, over all propagations"""
        return self._peak_qubit_excitation

    @property
    def peak_cavity_excitation(self):
        """Return the peak cavity exitation, over all propagations"""
        return self._peak_cavity_excitation

    def load(self, runfolder, hilbert_space=True):
        """
        Load data from the given runfolder

        If hilbert_space is True, load from files generated by Hilbert space
        proapgation (psi*.dat), otherwise load from files generated by
        Liouville space propagation (rho*.dat)

        Sets the exc, pop, pulse, tgrid, peak_qubit_excitation, and
        peak_cavity_excitation attributes.
        """
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
        self._peak_qubit_excitation = 0.0
        self._peak_cavity_excitation = 0.0
        self.exc['cavity']['00'], self.exc['q1']['00'], self.exc['q2']['00'], self.pop['00'], \
        self.exc['cavity']['01'], self.exc['q1']['01'], self.exc['q2']['01'], self.pop['01'], \
        self.exc['cavity']['10'], self.exc['q1']['10'], self.exc['q2']['10'], self.pop['10'], \
        self.exc['cavity']['11'], self.exc['q1']['11'], self.exc['q2']['11'], self.pop['11'] \
        = collect_pop_plot_data(files, runfolder)
        self.tgrid = self.exc['cavity']['00'].tgrid

        for qubit in ['q1', 'q2']:
            for basis_state in ['00', '01', '10', '11']:
                peak_qubit_excitation = np.max(
                                        self.exc[qubit][basis_state].mean
                                        + self.exc[qubit][basis_state].sd)
                if peak_qubit_excitation > self._peak_qubit_excitation:
                    self._peak_qubit_excitation = peak_qubit_excitation
        for basis_state in ['00', '01', '10', '11']:
            peak_cavity_excitation = np.max(
                                        self.exc['cavity'][basis_state].mean
                                        + self.exc['cavity'][basis_state].sd)
            if peak_cavity_excitation > self._peak_cavity_excitation:
                self._peak_cavity_excitation = peak_cavity_excitation

        # Load pulse
        filename = os.path.join(runfolder, 'pulse.dat')
        if os.path.isfile(filename):
            self.pulse = Pulse(filename)

    def render(self, basis_state, ax_pop, ax_q1, ax_q2, ax_cavity,
        pops=('00', '01', '10', '11', 'tot'), legend=False, scale=1.0):
        """
        Render data for the propagation of one of the logical basis states onto
        the given axes, based on the data and settings stored in attributes.

        Arguments
        ---------

        basis_state: '00', '01', '10', '11'
        ax_pop: matplotlib.axes.Axes instance
            axes on which to render population dynamics
        ax_q1:  matplotlib.axes.Axes instance
            axes on which to render left qubit excitation dynamics
        ax_q2:  matplotlib.axes.Axes instance
            axes on which to render right qubit excitation dynamics
        ax_cavity: matplotlib.axes.Axes instance
            axes on which to render qubit cavity dynamics
        pops: list
            subset of ('00', '01', '10', '11', 'tot'), indicating which
            populations should be shown in the population dynamics panel
        legend: boolean
            If true, render a legend to the right of each panel
        scale: float
            Scale all layout parameters by the given factor
        """
        legend_gap  = float(scale * self.legend_gap)
        panel_width = float(scale * self.panel_width)
        if basis_state not in ['00', '01', '10', '11']:
            raise ValueError("basis_state is %s, " % basis_state
                             +"must be one of '00', '01', '10', '11'")
        # population dynamics
        legend_lines = []
        if '00' in pops:
            p00, = ax_pop.plot(self.tgrid, self.pop[basis_state].pop00,
                            **self.styles['00'])
            legend_lines.append(p00)
        if '01' in pops:
            p01, = ax_pop.plot(self.tgrid, self.pop[basis_state].pop01,
                            **self.styles['01'])
            legend_lines.append(p01)
        if '10' in pops:
            p10, = ax_pop.plot(self.tgrid, self.pop[basis_state].pop10,
                            **self.styles['10'])
            legend_lines.append(p10)
        if '11' in pops:
            p11, = ax_pop.plot(self.tgrid, self.pop[basis_state].pop11,
                            **self.styles['11'])
            legend_lines.append(p11)
        if 'tot' in pops:
            pop_sum =   self.pop[basis_state].pop00 \
                      + self.pop[basis_state].pop10 \
                      + self.pop[basis_state].pop01 \
                      + self.pop[basis_state].pop11
            tot, = ax_pop.plot(self.tgrid, pop_sum, **self.styles['tot'])

        ax_pop.axhline(y=1, **self.styles['pop_hline'])
        ax_pop.fill_between(self.pulse.tgrid,
             np.abs(self.pulse.amplitude)/np.max(np.abs(self.pulse.amplitude)),
             **self.styles['pulse'])
        pulse_patch = mpatches.Patch(**self.styles['pulse'])

        if 'tot' in pops:
            in_panel_legend = ax_pop.legend([pulse_patch, tot],
                                            [self.styles["pulse"]["label"],
                                            self.styles["tot"]["label"]],
                                            ncol=2, loc='upper center',
                                            frameon=False, borderaxespad=0.0)
        else:
            in_panel_legend = ax_pop.legend((pulse_patch, ),
                                            (self.styles["pulse"]["label"],),
                                            ncol=1, loc='upper right',
                                            frameon=False, borderaxespad=0.0)
        ax_pop.add_artist(in_panel_legend)
        if legend:
            legend_offset = 1.0 + scale*legend_gap/panel_width
            legend_labels = [l.get_label() for l in legend_lines]
            ax_pop.legend(legend_lines, legend_labels,
                          ncol=1, loc='center left', frameon=False,
                          bbox_to_anchor=(legend_offset, 0.5),
                          handlelength=3, labelspacing=0.3, borderpad=0.0,
                          borderaxespad=0.0)

        # left qubit excitation
        q1_data = self.exc['q1'][basis_state]
        ax_q1.fill_between(q1_data.tgrid, q1_data.mean-q1_data.sd,
                    q1_data.mean+q1_data.sd, **self.styles['sd'])
        pmq1, = ax_q1.plot(q1_data.tgrid, q1_data.mean, **self.styles['mean'])
        ax_q1.add_artist(AnchoredText(
            "left qubit", loc=1, frameon=False, borderpad=0.0))
        if legend:
            legend_offset = 1.0 + scale*legend_gap/panel_width
            sd_patch = mpatches.Patch(**self.styles['sd'])
            ax_q1.legend([pmq1, sd_patch],
                         [r'$\langle i \rangle$', r'$\sigma_i$'],
                         ncol=1, loc='center left', frameon=False,
                         bbox_to_anchor=(legend_offset, 0.5),
                         handlelength=3, labelspacing=0.3, borderpad=0.0,
                         borderaxespad=0.0)

        # right qubit excitation
        q2_data = self.exc['q2'][basis_state]
        ax_q2.fill_between(q2_data.tgrid, q2_data.mean-q2_data.sd,
                    q2_data.mean+q2_data.sd, **self.styles['sd'])
        pmq2, = ax_q2.plot(q2_data.tgrid, q2_data.mean, **self.styles['mean'])
        ax_q2.add_artist(AnchoredText(
            "right qubit", loc=1, frameon=False, borderpad=0.0))
        if legend:
            legend_offset = 1.0 + scale*legend_gap/panel_width
            sd_patch = mpatches.Patch(**self.styles['sd'])
            ax_q2.legend([pmq2, sd_patch],
                         [r'$\langle j \rangle$', r'$\sigma_j$'],
                         ncol=1, loc='center left', frameon=False,
                         bbox_to_anchor=(legend_offset, 0.5),
                         handlelength=3, labelspacing=0.3, borderpad=0.0,
                         borderaxespad=0.0)

        # cavity excitation
        cavity_data = self.exc['cavity'][basis_state]
        ax_cavity.fill_between(cavity_data.tgrid,
                               cavity_data.mean-cavity_data.sd,
                               cavity_data.mean+cavity_data.sd,
                               **self.styles['sd'])
        pmcavity, = ax_cavity.plot(cavity_data.tgrid, cavity_data.mean,
                                   **self.styles['mean'])
        ax_cavity.add_artist(AnchoredText(
            "cavity", loc=1, frameon=False, borderpad=0.0))
        if legend:
            legend_offset = 1.0 + scale*legend_gap/panel_width
            sd_patch = mpatches.Patch(**self.styles['sd'])
            ax_cavity.legend([pmcavity, sd_patch],
                         [r'$\langle n \rangle$', r'$\sigma_n$'],
                         ncol=1, loc='center left', frameon=False,
                         bbox_to_anchor=(legend_offset, 0.5),
                         handlelength=3, labelspacing=0.3, borderpad=0.0,
                         borderaxespad=0.0)

    def plot(self, basis_states=('00', '01', '10', '11'),
        pops=('00', '01', '10', '11', 'tot'), fig=None,
        legend=True, quiet=True, scale=1.0):
        """
        Generate a plot of the dynamics starting from the given states,
        rendering it onto the  given figure.

        Arguments
        ---------

        basis_states: list
            subset of ('00', '01', '10', '11'), indicating the propagated
            states that should be included in the figure. Each item will
            produce a column of panels in the resulting plot
        pops: list
            passed to render(), for each of the basis states
        fig: matplotlib.figure.Figure instance
            Figure onto which to render the plots. If not given, create a new
            pylab figure
        legend: boolean
            If True, render a legend at the very right of the figure (one
            legend for each row of panels)
        quiet: boolean
            If True, print information about the figure layout, as well as the
            matplotlib backend and rc file
        scale: float
            Scaling factor for the layout attributes (margins etc). Useful for
            quickly scaling the figure in an interactive context (IPython
            notebook), whithout setting new attributes.
        """
        n_states = len(basis_states)
        h_pop         = float(scale * self.h_pop)
        h_exc         = float(scale * self.h_exc)
        w             = float(scale * self.panel_width)
        left_margin   = float(scale * self.left_margin)
        right_margin  = float(scale * self.right_margin)
        bottom_margin = float(scale * self.bottom_margin)
        top_margin    = float(scale * self.top_margin)
        gap           = float(scale * self.gap)
        fig_width = float(left_margin +right_margin
                          + n_states*w + (n_states-1)*gap)
        fig_height = float(bottom_margin + top_margin
                           + h_pop + 3*h_exc)
        if not quiet:
            backend = matplotlib.get_backend().lower()
            print "Using backend: ", backend
            print "Using maplotlibrc: ", matplotlib.matplotlib_fname()
            print "Figure height: ", fig_height, " %s" % self.unit
            print "Figure width : ", fig_width, " %s" % self.unit
        if fig is None:
            if self.unit == 'cm':
                fig = plt.figure(
                        figsize=(fig_width*0.39370079, fig_height*0.39370079),
                        dpi=self.dpi)
            else:
                fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.dpi)
        self.ax = {}
        ax = self.ax
        for i_state, basis_state in enumerate(basis_states):
            left_offset = float(left_margin + i_state*(w+gap))
            if not quiet:
                print "Panels for %s at offset %f %s from left border" % (
                      basis_state, left_offset, self.unit)
            ax[basis_state] = {}
            ax[basis_state]['pop'] = fig.add_axes(
                [left_offset/fig_width, bottom_margin/fig_height,
                 w/fig_width, h_pop/fig_height])
            ax[basis_state]['q1'] = fig.add_axes(
                [left_offset/fig_width, (bottom_margin+h_pop)/fig_height,
                 w/fig_width, h_exc/fig_height])
            ax[basis_state]['q2'] = fig.add_axes(
                [left_offset/fig_width,
                 (bottom_margin+h_pop+h_exc)/fig_height,
                 w/fig_width, h_exc/fig_height])
            title = self.title
            if self.title is not None:
                if '%s' in self.title:
                    title = self.title % str(basis_state)
            ax[basis_state]['cavity'] = fig.add_axes(
                [left_offset/fig_width,
                 (bottom_margin+h_pop+2*h_exc)/fig_height,
                 w/fig_width, h_exc/fig_height], title=title)

            show_legend = False
            if legend:
                if i_state == len(basis_states)-1:
                    show_legend = True
            self.render(basis_state, ax[basis_state]['pop'],
                        ax[basis_state]['q1'], ax[basis_state]['q2'],
                        ax[basis_state]['cavity'], pops=pops,
                        legend=show_legend, scale=scale)

        # synchronize all axes
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator
        for i_state, basis_state in enumerate(basis_states):
            for panel in ['pop', 'q1', 'q2', 'cavity']:
                # synchronize x-axis
                current_ax = ax[basis_state][panel]
                current_ax.set_xlim(self.tgrid[0], self.tgrid[-1])
                current_ax.xaxis.set_minor_locator(
                    AutoMinorLocator(self.xaxis_minor))
                if panel == 'pop':
                    current_ax.set_xlabel('time (%s)' % self.pulse.time_unit)
                else:
                    current_ax.set_xticklabels([])
                # synchronize y-axis
                if panel in ['q1', 'q2']:
                    y_lim_scale = 1.0 + float(self.exc_top_buffer)/self.h_exc
                    current_ax.set_ylim(
                        0.0, y_lim_scale*self.peak_qubit_excitation)
                    current_ax.yaxis.set_major_locator(
                        MaxNLocator(nbins=5, steps=(1,5,10), symmetric=False))
                    current_ax.yaxis.set_minor_locator(AutoMinorLocator())
                    if panel == 'q2' and i_state == 0:
                        current_ax.set_ylabel('qubit and cavity excitation')
                    # prune the tick labels in the buffer region
                    tick_labels = current_ax.get_yticklabels()
                    tick_locs = current_ax.yaxis.get_ticklocs()
                    for i_tick, tick_loc in enumerate(tick_locs):
                        if tick_loc > self.peak_qubit_excitation:
                            tick_labels[i_tick].set_visible(False)
                elif panel == 'cavity':
                    y_lim_scale = 1.0 + float(self.exc_top_buffer)/self.h_exc
                    current_ax.set_ylim(
                        0.0, y_lim_scale*self.peak_cavity_excitation)
                    current_ax.yaxis.set_major_locator(
                        MaxNLocator(nbins=5, steps=(1,5,10), symmetric=False))
                    current_ax.yaxis.set_minor_locator(AutoMinorLocator())
                elif panel == 'pop':
                    if i_state == 0:
                        current_ax.set_ylabel('population')
                        y_max = 1.0 + float(self.pop_top_buffer)/self.h_pop
                    current_ax.set_ylim(0.0, y_max)
                    current_ax.set_yticks(
                        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    current_ax.yaxis.set_minor_locator(AutoMinorLocator(2))


    def _repr_png_(self):
        """Return png representation"""
        return self._figure_data(fmt='png')

    def _repr_svg_(self):
        """Return svg representation"""
        return self._figure_data(fmt='svg')

    def _figure_data(self, fmt='png', display=False):
        """Return image representation"""
        from IPython.core.pylabtools import print_figure
        from IPython.display import Image, SVG
        from IPython.display import display as ipy_display
        fig_width = float(self.left_margin + self.right_margin
                          + 4*self.panel_width + 3*self.gap)
        fig_height = float(self.bottom_margin + self.top_margin
                           + self.h_pop + 3*self.h_exc)
        if self.unit == 'cm':
            fig = plt.figure(
                    figsize=(fig_width*0.39370079, fig_height*0.39370079),
                    dpi=self.dpi)
        else:
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.dpi)
        self.plot(fig=fig)
        fig_data = print_figure(fig, fmt)
        if fmt=='svg':
            fig_data = fig_data.decode('utf-8')
        # We MUST close the figure, otherwise IPython's display machinery
        # will pick it up and send it as output, resulting in a
        # double display
        plt.close(fig)
        if display:
            if fmt=='svg':
                ipy_display(SVG(fig_data))
            else:
                ipy_display(Image(fig_data))
        else:
            return fig_data

    def show(self, **kwargs):
        """
        Show a plot of all the population dynamics. All arguments are passed to
        the plot() method
        """
        self.plot(**kwargs)
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
        for i_t, __ in enumerate(self.tgrid):
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
