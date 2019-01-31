"""
Calculate the full solution numerically
"""
import numpy as np
import scipy.integrate as it
import scipy.interpolate as interp
from src.functions import *
from src.analytical import LeadingOrderQuasiStatic
from src.util import my_linear_interp


class Numerical(LeadingOrderQuasiStatic):
    """A class for the full numerical solution"""

    def __init__(self, time, p, g, Vonly=False):
        LeadingOrderQuasiStatic.__init__(self, time, p, g)
        # Set linestyle
        self.linestyle = "k-"
        self.color = "black"

        # Solve the system of PDEs (via method of lines & finite volumes)
        # Initial conditions
        yinit = initial_conditions(p, g, "full")

        # Termination event: need a wrapper
        def negative_concentration_wrapper(t, y):
            return negative_concentration(t, y, p, g)

        negative_concentration_wrapper.terminal = True
        negative_concentration_wrapper.direction = -1

        # Solve
        sol = it.solve_ivp(
            lambda t, y: derivs(t, y, p, g, "full"),
            (time[0], time[-1]),
            yinit,
            t_eval=time,
            method="BDF",
            events=negative_concentration_wrapper,
        )

        # Extract solution
        (self.c, self.eps, self.xin, self.xip) = get_vars(sol.y, p, g, "full", sol.t)

        # Post-process to find other attributes
        # Calculate current using the fluxes function
        (_, self.i_n, self.i_p, self.jn, self.jp, self.etan, self.etap) = fluxes(
            self.c,
            self.eps,
            p,
            g,
            "full",
            self.xin,
            self.xip,
            np.transpose(self.icell[: len(sol.t)]),
            sol.t,
        )

        # Potential in the electrolyte
        self.i = np.vstack(
            [self.i_n, np.transpose(self.icell[: len(sol.t)] * np.ones(g.ns)), self.i_p]
        )
        self.phi = calculate_phi(sol.t, self, p, g)

        # Pad with NaNs
        for attr in [
            "c",
            "eps",
            "phi",
            "xin",
            "xip",
            "i_n",
            "i_p",
            "i",
            "jn",
            "jp",
            "etan",
            "etap",
        ]:
            self.__dict__[attr] = np.pad(
                self.__dict__[attr],
                ((0, 0), (0, len(self.t) - len(sol.t))),
                "constant",
                constant_values=np.nan,
            )

        # Potential in the electrodes
        self.phisn = self.xin + self.phi[: g.nn - 1]
        self.phisp = self.xip + self.phi[g.nn + g.ns :]

        # Voltage
        self.V = self.phisp[-1][:, np.newaxis]
        self.Vcircuit = self.V * 6

        # Voltage cut-off (also transposes)
        self.cutoff(p, V=self.V)

        # If we only care about V, we are done (return None to exit)
        if Vonly:
            return None

        # Transpose i and calculate current density in the solid
        self.isolid = self.icell - self.i

        # Interpolate to cell edges (manually)

        for attr in ["c", "eps", "phi"]:
            self.__dict__[attr] = my_linear_interp(self.__dict__[attr])

        # Interpolate to cell edges and combine
        for attr in ["phis", "eta", "j"]:
            attr_n = my_linear_interp(self.__dict__[attr + "n"])
            attr_p = my_linear_interp(self.__dict__[attr + "p"])
            self.__dict__[attr] = np.hstack(
                [attr_n, np.nan * np.ones((len(self.t), g.ns)), attr_p]
            )

            # After shifting, phis won' be exactly zero at x=0, but very close to it

    def __str__(self):
        return "Numerical"

    def latexlinestyle(self, opacity=1):
        """Define linestyle for plotting in Latex"""
        return "[color=black, opacity={}]".format(opacity)
