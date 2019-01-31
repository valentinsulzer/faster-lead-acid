"""
Calculate the analytical (reduced order) solutions:
    LeadingOrderQuasiStatic
    FirstOrderQuasiStatic
    Composite

Also includes a separate class for the voltage breakdown (VoltageBreakdown)
"""
import warnings

import numpy as np
import scipy.integrate as it
import scipy.interpolate as interp

from src.functions import *
import pdb

class LeadingOrderQuasiStatic:
    """A class for the O(1) analytical (reduced order)  solution"""

    def __init__(self, time, p, g, Vonly=False):
        # Linestyle
        self.linestyle = 'g-.'
        self.color = 'green'

        # Time and space: we add a new axis to simplify taking the outer product
        self.t = np.copy(time)[:, np.newaxis]
        self.x = np.copy(g.x)

        # Current in the external circuit
        self.Icircuit = p.Icircuit(self.t * p.scales.time)

        # Current
        self.icell = p.icell(self.t)
        self.intI = it.cumtrapz(self.icell, self.t, initial=0.0, axis=0)

        # Porosities
        self.eps0n = p.epsn0 - p.beta_surf_n / p.ln * self.intI
        self.eps0p = p.epsp0 + p.beta_surf_p / p.lp * self.intI
        self.eps0 = np.hstack([self.eps0n * np.ones(g.nn),
                               p.epss0 * np.ones((len(self.t), g.ns)),
                               self.eps0p * np.ones(g.np)])

        # Concentration
        self.c0_v = (((p.epsn0 * p.ln + p.epss0 * p.ls + p.epsp0 * p.lp) * p.cinit
                      + (p.sn - p.sp) * self.intI)
                     / (self.eps0n * p.ln + p.epss0 * p.ls + self.eps0p * p.lp))
        self.c0 = self.c0_v * np.ones(g.n)

        # Implement cut-off if concentration goes below zero
        self.cutoff(p, c=self.c0)

        # Potentials and voltage
        phi0 = - p.U_Pb(self.c0_v) - np.arcsinh(self.icell / (2 * p.iota_ref_n * self.c0_v * p.ln))
        self.phi0 = phi0 * np.ones(g.n)
        self.V0 = (p.U_PbO2(self.c0_v)
                   - np.arcsinh(self.icell / (2 * p.iota_ref_p * self.c0_v ** 2 *
                                              p.cw(self.c0_v) * p.lp))
                   + phi0)

        # Implement cut-off if voltage goes below cut-off
        self.cutoff(p, V=self.V0)

        # External circuit
        self.V0circuit = self.V0 * 6
        self.Vcircuit = np.copy(self.V0circuit)

        self.phis0 = np.hstack([np.zeros((len(self.t), g.nn)),
                                np.nan * np.ones((len(self.t), g.ns)),
                                self.V0 * np.ones(g.np)])

        # Overpotentials
        self.eta0n = (self.phis0[:, 0] - self.phi0[:, 0] - p.U_Pb(self.c0[:, 0]))[:, np.newaxis]
        self.eta0p = (self.phis0[:, -1] - self.phi0[:, -1] - p.U_PbO2(self.c0[:, 0]))[:, np.newaxis]

        # Only continue if we care about other variables than V
        if not Vonly:
            self.eta0 = np.hstack([self.eta0n * np.ones(g.nn),
                                   np.nan * np.ones((len(self.t), g.ns)),
                                   self.eta0p * np.ones(g.np)])

            # Current densities
            self.j0 = np.hstack([self.icell / p.ln * np.ones(g.nn),
                                 np.nan * np.ones((len(self.t), g.ns)),
                                 -self.icell / p.lp * np.ones(g.np)])
            self.i0 = np.hstack([self.icell * np.linspace(0, 1, g.nn),
                                 self.icell * np.ones((len(self.t), g.ns)),
                                 self.icell * np.linspace(1, 0, g.np)])
            self.isolid0 = self.icell - self.i0

            # SOC
            self.q = 1 - self.intI / p.qmax

            # Totals
            for attr in ['c', 'eps', 'phi', 'phis', 'eta', 'j', 'i', 'isolid', 'V']:
                self.__dict__[attr] = np.copy(self.__dict__[attr + '0'])

    def __str__(self):
        return 'LOQS'

    def latexlinestyle(self, opacity=1):
        """Define linestyle for plotting in Latex"""
        return '[dashdotdotted, color=green!70!black, opacity={}]'.format(opacity)

    def cutoff(self, p, c=None, V=None):
        """
        Cut off solution after a certain index by replacing with NaNs
        (to avoid plotting issues) rather than deleting
            c: concentration cut-off if concentration goes below 0
            V: voltage cut-off if voltage goes below specified cut-off voltage
        """
        for attr in dir(self):
            # Exclude some attributes which are not time-dependent, and t itself
            exclude = (attr.startswith('__')  # inbuilt functions
                       or attr in ['t', 'x', 'dimensionalise', 'cutoff',
                                   'linestyle', 'color', 'latexlinestyle'])
            if not exclude:
                if self.__dict__[attr].shape[0] != len(self.t):
                    # Need to tranpose for this to work if attr has wrong shape
                    self.__dict__[attr] = self.__dict__[attr].T
                # Replace all values corresponding to where concentration is
                # below zero or voltage is below cut-off with NaNs
                if c is not None:
                    self.__dict__[attr] = np.where(c[:, -1][:, np.newaxis] > 0,
                                                   self.__dict__[attr], np.nan)
                if V is not None:
                    # Ignore warning if V has NaN values
                    if np.isnan(V).any():
                        warnings.simplefilter("ignore", RuntimeWarning)
                    else:
                        warnings.simplefilter("always")
                    self.__dict__[attr] = np.where(V > p.voltage_cutoff,
                                                   self.__dict__[attr], np.nan)

    def dimensionalise(self, p, g):
        """Dimensionalise results"""
        for attr in dir(self):
            # Exclude attributes that are functions
            exclude = (attr.startswith('__')
                       or attr in ['dimensionalise', 'cutoff', 'linestyle',
                                   'color', 'Icircuit', 'latexlinestyle'])
            if not exclude:
                scale = p.scales.match[attr]
                # Some variables have different non-dimensionalisation in negative
                # and positive electrodes (and are not defined in the separator)
                if attr in ['phis0', 'phis']:
                    # Multiplicative scale
                    self.__dict__[attr] *= p.scales.__dict__[scale[0]]
                    # Additive scale (only in pos)
                    self.__dict__[attr][:, g.nn + g.ns:] += scale[1]
                elif attr in ['j0', 'j']:
                    # Multiplicative scales (different in neg and pos)
                    self.__dict__[attr][:, :g.nn + g.ns] *= p.scales.__dict__[scale[0]]
                    self.__dict__[attr][:, g.nn + g.ns:] *= p.scales.__dict__[scale[1]]
                    # No additive scale
                elif isinstance(scale, tuple):
                    # Multiplicative scales
                    self.__dict__[attr] *= p.scales.__dict__[scale[0]]
                    # Additive scale (for potentials)
                    self.__dict__[attr] += scale[1]
                else:
                    # Multiplicative scales only
                    self.__dict__[attr] *= p.scales.__dict__[scale]


class FirstOrderQuasiStatic(LeadingOrderQuasiStatic):
    """A class for the O(Da) analytical (reduced order) quasi-steady solution"""

    def __init__(self, time, p, g, Vonly=False):
        # Inherit properties from O1
        LeadingOrderQuasiStatic.__init__(self, time, p, g, Vonly)

        # Set linestyle
        self.linestyle = 'r:'
        self.color = 'red'

        # Simplify notation
        c0 = self.c0_v
        I = self.icell
        snI = p.sn * I / p.ln
        spI = p.sp * I / p.lp

        # Derivatives of leading order
        deps0ndt = -p.beta_surf_n / p.ln * I
        deps0pdt = p.beta_surf_p / p.lp * I
        dc0dt = ((p.sn - p.sp) * I
                 / (self.eps0n * p.ln + p.epss0 * p.ls + self.eps0p * p.lp)
                 - ((p.epsn0 * p.ln + p.epss0 * p.ls + p.epsp0 * p.lp)
                    + (p.sn - p.sp) * self.intI)
                 / (self.eps0n * p.ln + p.epss0 * p.ls + self.eps0p * p.lp) ** 2
                 * (p.ln * deps0ndt + p.lp * deps0pdt))
        deps0nc0dt = self.eps0n * dc0dt + deps0ndt * c0
        deps0sc0dt = p.epss0 * dc0dt
        deps0pc0dt = self.eps0p * dc0dt + deps0pdt * c0

        # Concentration
        D0n = p.D_eff(c0, self.eps0n)
        D0s = p.D_eff(c0, p.epss0)
        D0p = p.D_eff(c0, self.eps0p)

        c1n_ = 1 / (2 * D0n) * (deps0nc0dt - snI) * (g.xn ** 2 - p.ln ** 2)
        c1s_ = (1 / (2 * D0s) * deps0sc0dt * (g.xs - p.ln) ** 2
                + (1 / (D0n) * (deps0nc0dt - snI)) * p.ln * (g.xs - p.ln))
        c1p_ = (1 / (2 * D0p) * (deps0pc0dt + spI) * ((g.xp - 1) ** 2 - p.lp ** 2)
                + ((p.ls / (2 * D0s) * deps0sc0dt
                    + p.ln / D0n * (deps0nc0dt - snI)) * p.ls))

        k = (1 / (self.eps0n * p.ln + p.epss0 * p.ls + self.eps0p * p.lp)
             * ((self.eps0n * p.ln ** 3 / (3 * D0n) - p.epss0 * p.ls ** 2 * p.ln / (2 * D0n))
                * (deps0nc0dt - snI)
                - p.epss0 * p.ls ** 3 / (6 * D0s) * deps0sc0dt
                + self.eps0p * p.lp ** 3 / (3 * D0p) * (deps0pc0dt + spI)
                - (p.ls / (2 * D0s) * deps0sc0dt
                   + p.ln / (D0n) * (deps0nc0dt - snI)) * self.eps0p * p.ls * p.lp))
        # No need to make c1 an attribute of this class (we never plot it)
        self.c1 = np.hstack([c1n_, c1s_, c1p_]) + k

        # Total concentration and concentration cut-off
        self.c = self.c0 + p.Cd * self.c1
        self.cutoff(p, c=self.c)

        # Get the other variables from c
        self = V_from_c(self, p, g, Vonly)

    def __str__(self):
        return 'FOQS'


    def latexlinestyle(self, opacity=1):
        """Define linestyle for plotting in Latex"""
        return '[dotted, color=red, opacity={}]'.format(opacity)

class Composite(LeadingOrderQuasiStatic):
    """A class for the O(Da) analytical (reduced order) composite solution"""

    def __init__(self, time, p, g, Vonly=False):
         # Inherit properties from O1
        LeadingOrderQuasiStatic.__init__(self, time, p, g, Vonly)

        # Set linestyle
        self.linestyle = 'b--'
        self.color = 'blue'

        # Simplify notation
        c0 = self.c0_v
        I = self.icell
        snI = p.sn * I / p.ln
        spI = p.sp * I / p.lp

        # Set initial conditions
        yinit = initial_conditions(p, g, 'composite')

        # Termination event: need a wrapper
        def negative_concentration_wrapper(t, y):
            return negative_concentration(t, y, p, g)

        negative_concentration_wrapper.terminal = True

        # Solve ODE for ctilde
        ODEsol = it.solve_ivp(lambda t, y: derivs(t, y, p, g, 'composite'),
                           (time[0], time[-1]), yinit, t_eval=time,
                           method='BDF',
                           events=negative_concentration_wrapper)

        # Extract c from u (we already have the porosity)
        (c_centres, *rest) = get_vars(ODEsol.y, p, g, 'composite', t=ODEsol.t)

        # Interpolate to cell edges
        c = interp.interp2d(ODEsol.t, g.xc, c_centres, kind='cubic')(ODEsol.t, g.x)
        # Transpose
        c = c.T

        # Pad with NaNs (things are already cut off by ivp solve)
        self.c = np.pad(c, ((0, len(self.t) - len(ODEsol.t)), (0, 0)),
                        'constant', constant_values=np.nan)

        # Calculate c1
        self.c1 = (self.c - self.c0) / p.Cd

        # Get the other variables from c
        self = V_from_c(self, p, g, Vonly)

    def __str__(self):
        return 'Composite'

    def latexlinestyle(self, opacity=1):
        """Define linestyle for plotting in Latex"""
        return '[dashed, color=blue, opacity={}]'.format(opacity)

def V_from_c(sol_, p, g, Vonly):
    """Solve the rest of the model from c and leading-order solution"""
    # Simplify notation
    c0 = sol_.c0_v
    I = sol_.icell

    # Electrolyte potential
    k0n = p.kappa_eff(c0, sol_.eps0n)
    k0s = p.kappa_eff(c0, p.epss0)
    k0p = p.kappa_eff(c0, sol_.eps0p)
    chi0 = p.chi(c0)
    cw0 = p.cw(c0)
    dcw0dc = p.dcwdc(c0)

    intc1n = np.trapz(sol_.c1[:, :g.nn], x=g.xn)[:, np.newaxis]
    An = (1 / p.ln * (np.tanh(sol_.eta0n)/c0 - p.dUPbdc(c0) - chi0/c0) * intc1n
          + I * p.ln / (6 * k0n))

    phi1n_ = - I / (2 * p.ln * k0n) * g.xn ** 2
    phi1s_ = - I * ((g.xs-p.ln)/k0s + p.ln/(2 * k0n))
    phi1p_ = - I * (p.ln/(2*k0n) + p.ls/(k0s) + (p.lp**2 - (1-g.xp)**2) / (2*p.lp*k0p))
    phi1 = chi0 / c0 * sol_.c1 + np.hstack([phi1n_, phi1s_, phi1p_]) + An

    # Voltage
    intc1p = np.trapz(sol_.c1[:, g.nn + g.ns:], x=g.xp)[:, np.newaxis]
    intphi1p = np.trapz(phi1[:, g.nn + g.ns:], x=g.xp)[:, np.newaxis]
    V1 = 1 / p.lp * ((p.dUPbO2dc(c0) - (2 * cw0 + c0 * dcw0dc) / (c0 * cw0) * np.tanh(sol_.eta0p)) * intc1p
                     + intphi1p)

    sol_.V = sol_.V0 + p.Cd * V1

    # Voltage cut-off
    sol_.cutoff(p, V=sol_.V)

    # Voltage in the external circuit
    sol_.Vcircuit = sol_.V * 6

    # If we only care about V, we are done (return None to exit)
    if not Vonly:
        # Electrode potential
        phis1 = np.hstack([np.zeros((len(sol_.t), g.nn)),
                           np.nan * np.ones((len(sol_.t), g.ns)),
                           V1 * np.ones(g.np)])

        # Overpotential
        eta1n = phis1[:, :g.nn] - phi1[:, :g.nn] - sol_.c1[:, :g.nn] * p.dUPbdc(c0)
        eta1p = phis1[:, g.nn + g.ns:] - phi1[:, g.nn + g.ns:] - sol_.c1[:, g.nn + g.ns:] * p.dUPbO2dc(c0)
        eta1 = np.hstack([eta1n, np.nan * np.ones((len(sol_.t), g.ns)), eta1p])

        # Current densities
        j1n = 2 * p.iota_ref_n * (sol_.c1[:, :g.nn] * np.sinh(sol_.eta0n) + c0 * eta1n * np.cosh(sol_.eta0n))
        j1p = 2 * p.iota_ref_p * ((2 * c0 * cw0 + c0 ** 2 * dcw0dc) * sol_.c1[:, g.nn + g.ns:] * np.sinh(sol_.eta0p)
                                  + c0 ** 2 * cw0 * eta1p * np.cosh(sol_.eta0p))
        j1 = np.hstack([j1n, np.nan * np.ones((len(sol_.t), g.ns)), j1p])

        i1n = it.cumtrapz(j1n, g.xn, initial=0.0, axis=1)
        i1p = it.cumtrapz(j1p, g.xp, initial=0.0, axis=1)
        i1 = np.hstack([i1n, np.zeros((len(sol_.t), g.ns)), i1p])
        isolid1 = -i1

        # Porosity
        eps1n = -p.beta_surf_n * it.cumtrapz(j1n, sol_.t, initial=0.0, axis=0)
        eps1p = -p.beta_surf_p * it.cumtrapz(j1p, sol_.t, initial=0.0, axis=0)
        eps1 = np.hstack([eps1n, np.zeros((len(sol_.t), g.ns)), eps1p])

        # Combine O(1) and O(Da) to make solution
        for attr in ['eps', 'phi', 'phis', 'eta', 'j', 'i', 'isolid']:
            sol_.__dict__[attr] = sol_.__dict__[attr + '0'] + p.Cd * eval(attr + '1')

        return sol_

class VoltageBreakdown:
    """
    A class containing the breakdown of the voltage from the first-order
    analytical solutions (quasi-static or composite)
    """
    def __init__(self, sol_, p, g):
        c0 = sol_.c0_v
        I = sol_.icell
        cw0 = p.cw(c0)
        dcw0dc = p.dcwdc(c0)
        k0n = p.kappa_eff(c0, sol_.eps0n)
        k0s = p.kappa_eff(c0, p.epss0)
        k0p = p.kappa_eff(c0, sol_.eps0p)

        # Integrate c1
        c1bar_n = np.trapz(sol_.c1[:, :g.nn], x=g.xn)[:,np.newaxis]/p.ln
        c1bar_p = np.trapz(sol_.c1[:, g.nn + g.ns:], x=g.xp)[:,np.newaxis]/p.lp

        # Initial value (for stack plot)
        self.init = np.where(~np.isnan(c0),(p.U_PbO2_ref/p.scales.pot + p.U_PbO2(c0[0])
                                            - p.U_Pb_ref/p.scales.pot - p.U_Pb(c0[0]))*np.ones(c0.shape),
                                            np.nan)

        # OCVs
        self.Un = p.U_Pb(c0[0]) - p.U_Pb(c0) - p.Cd*c1bar_n*p.dUPbdc(c0)
        self.Up = -p.U_PbO2(c0[0]) + p.U_PbO2(c0) + p.Cd*c1bar_p*p.dUPbO2dc(c0)

        # Kinetic overpotential
        self.kn = (- np.arcsinh(I / (2 * p.iota_ref_n * c0 * p.ln))
                     + p.Cd*c1bar_n/c0 * np.tanh(sol_.eta0n))
        self.kp = (- np.arcsinh(I / (2 * p.iota_ref_p * c0**2 * cw0 * p.lp))
                     - p.Cd*c1bar_p * (2*cw0+c0*dcw0dc)/(c0*cw0) * np.tanh(sol_.eta0p))

        # Concentration overpotential
        self.c = p.Cd*p.chi(c0)/c0 * (c1bar_p - c1bar_n)

        # Ohmic overpotential in the electrolyte
        self.o = - p.Cd*I * (p.ln/(3*k0n) + p.ls/k0s + p.lp/(3*k0p))

        # Totals
        self.tot = self.Un + self.Up + self.kn + self.kp + self.c + self.o

    def dimensionalise(self, p):
        """Dimensionalise using voltage scales"""
        self.Un = 6*(- p.U_Pb_ref + p.scales.pot*self.Un)
        self.Up = 6*(p.U_PbO2_ref + p.scales.pot*self.Up)
        for attr in ['kn', 'kp', 'c', 'o']:
            self.__dict__[attr] *= 6*p.scales.pot
        self.tot = self.Un + self.Up + self.kn + self.kp + self.c + self.o
