"""
Plotting functions:
    slider_plot - A plot with a slider to vary time
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cbook
from matplotlib.widgets import Slider
import warnings

plt.style.use('./paper.mplstyle')

def slider_plot(sols):
    """
    Create slider plot showing time-dependent variables
    INPUT:
        sols - a list of solution classes
    """
    # Turn off deprecation warning
    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

    # Define the base solution (for x and O(1)) and number of solutions
    sol = sols[0]
    n = len(sols)
    if n==1:
        sol.linestyle = 'k-'

    # All inputs have same value of x, t, c0, I and q
    fig, ax = plt.subplots(figsize=(16,9))

    plt.subplots_adjust(top=0.92, bottom=0.2, left=0.10, right=0.9, hspace=0.5,
                        wspace=0.5)

    plt.subplot(331)
    plt.title('Electrolyte')
    plt.xlabel('x [m]')
    plt.ylabel('c [Molar]')
    c = []
    for k in range(n):
        _, = plt.plot(sol.x, sols[k].c[0]/1e3, sols[k].linestyle, lw=2)
        c.append(_)
    plt.axis([0, sol.x[-1], 0, 1.1*np.nanmax(sol.c)/1e3])
    plt.subplot(331).xaxis.set_major_locator(plt.MaxNLocator(5))

    plt.subplot(332)
    plt.title('Interface')
    plt.xlabel('x [m]')
    plt.ylabel('$\epsilon$ [-]')
    eps = []
    for k in range(n):
        _, = plt.plot(sol.x, sols[k].eps[0], sols[k].linestyle, lw=2)
        eps.append(_)
    plt.axis([0, sol.x[-1], 0, 1])
    plt.subplot(332).xaxis.set_major_locator(plt.MaxNLocator(5))

    plt.subplot(333)
    plt.title('Time series')
    plt.xlabel('t [h]')
    plt.ylabel('$I_{circuit}$ [A]')
    I, = plt.plot(sol.t, sol.Icircuit, 'k-', lw=2)
    plt.axis([0, sol.t[-1], 0.9*np.nanmin(sol.Icircuit), 1.1*np.nanmax(sol.Icircuit)])
    plt.subplot(333).xaxis.set_major_locator(plt.MaxNLocator(5))
    I_line, = plt.plot([sol.t[0],sol.t[0]],
                       [0.9*np.nanmin(sol.Icircuit), 1.1*np.nanmax(sol.Icircuit)],
                       'k-')

    plt.subplot(334)
    plt.xlabel('x [m]')
    plt.ylabel('$\Phi$ [V]')
    phi = []
    for k in range(n):
        _, = plt.plot(sol.x, sols[k].phi[0], sols[k].linestyle, lw=2)
        phi.append(_)
    plt.axis([0, sol.x[-1], 0.9*np.nanmin(sol.phi), 1.1*np.nanmax(sol.phi)])
    plt.subplot(334).xaxis.set_major_locator(plt.MaxNLocator(5))

    plt.subplot(335)
    plt.xlabel('x [m]')
    plt.ylabel('$\eta$ [V]')
    eta = []
    for k in range(n):
        _, = plt.plot(sol.x, sols[k].eta[0], sols[k].linestyle, lw=2)
        eta.append(_)
    plt.axis([0, sol.x[-1], 1.1*np.nanmin(sol.eta), 1.1*np.nanmax(sol.eta)])
    plt.subplot(335).xaxis.set_major_locator(plt.MaxNLocator(5))

    plt.subplot(336)
    plt.xlabel('t [h]')
    plt.ylabel('$V_{circuit}$ [V]')
    for k in range(n):
        plt.plot(sol.t, sols[k].Vcircuit, sols[k].linestyle, lw=2)
    ymin, ymax = plt.ylim()
    plt.axis([0, sol.t[-1], ymin, ymax])
    plt.subplot(336).xaxis.set_major_locator(plt.MaxNLocator(5))
    V_line, = plt.plot([sol.t[0],sol.t[0]], [ymin, ymax], 'k-')

    plt.subplot(337)
    plt.xlabel('x [m]')
    plt.ylabel('i [A.m-2]')
    i = []
    for k in range(n):
        _, = plt.plot(sol.x, sols[k].i[0], sols[k].linestyle, lw=2, label=str(sols[k]))
        i.append(_)
    legend = plt.legend(bbox_to_anchor=(0.05, -0.4),
                        loc=2, borderaxespad=0., fontsize=12)
    plt.axis([0, sol.x[-1], 0, 1.1*np.nanmax(sol.i)])
    plt.subplot(337).xaxis.set_major_locator(plt.MaxNLocator(5))

    plt.subplot(338)
    plt.xlabel('x [m]')
    plt.ylabel('j [A.m-2]')
    j = []
    for k in range(n):
        _, = plt.plot(sol.x, sols[k].j[0], sols[k].linestyle, lw=2)
        j.append(_)
    plt.axis([0, sol.x[-1], 1.1*np.nanmin(sol.j[1:]), 1.1*np.nanmax(sol.j[1:])])
    plt.subplot(338).xaxis.set_major_locator(plt.MaxNLocator(5))

    plt.subplot(339)
    plt.xlabel('t [h]')
    plt.ylabel('q [-]')
    q, = plt.plot(sol.t, sol.q, 'k-', lw=2)
    plt.axis([0, sol.t[-1], 0, 1])
    plt.subplot(339).xaxis.set_major_locator(plt.MaxNLocator(5))
    q_line, = plt.plot([sol.t[0],sol.t[0]], [0, 1], 'k-')

    axfreq = plt.axes([0.4, 0.05, 0.37, 0.03])
    sfreq = Slider(axfreq, 'Time', 0, sol.t.size, valinit=0)

    def update(val):
        it = int(round(sfreq.val))
        for k in range(n):
            c[k].set_ydata(sols[k].c[it]/1e3)
            eps[k].set_ydata(sols[k].eps[it])
            phi[k].set_ydata(sols[k].phi[it])
            eta[k].set_ydata(sols[k].eta[it])
            i[k].set_ydata(sols[k].i[it])
            j[k].set_ydata(sols[k].j[it])

        # Time series
        I_line.set_xdata([sol.t[it], sol.t[it]])
        V_line.set_xdata([sol.t[it], sol.t[it]])
        q_line.set_xdata([sol.t[it], sol.t[it]])

        fig.canvas.draw_idle()

    sfreq.on_changed(update)

    plt.show()


def voltage(filename, solutions):
    """Plot voltages"""
    # Format filename
    filename = "out/Figures/" + filename + ".eps"
    # Set max t
    maxt = max([sol.t[np.where(~np.isnan(sol.V))][-1] for sol in solutions])
    # Plot
    plt.figure()
    for sol in solutions:
        plt.plot(sol.t, sol.Vcircuit, sol.linestyle, label=str(sol))
    # Axes and legend
    plt.xlabel(r'Time, $t$ [h]')
    plt.ylabel(r'Voltage, $V_{circuit}$ [V]')
    plt.xlim([0, maxt])
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
#    legend = plt.legend(fontsize=size)

    # Show and save
    plt.show()
    plt.savefig(filename, format='eps', dpi=1000)


def error(filename, errors, colors):
    """Make a log-log plot of errors"""
    size = 20
    # Add directories to filename
    filename = "out/Figures/" + filename + ".eps"
    plt.figure()
    for method in errors.keys():
        if method != 'Numerical':
            plt.loglog(list(errors[method].keys()), list(errors[method].values()),
                       color=colors[method], label=method)
    # Axes and legend
    plt.xlabel(r'Current, $I_{circuit}$ [h]')
    plt.ylabel(r'Error [\%]')
    legend = plt.legend(loc='best')

    # Show and save
    plt.show()
    plt.savefig(filename, format='eps', dpi=1000)
