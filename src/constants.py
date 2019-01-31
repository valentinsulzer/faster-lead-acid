"""
Define the classes
    Parameters - dimensional and dimensionless parameters in the model
    Scales - scales for nondimensionalisation
    Grid - information regarding the grid
"""
import numpy as np
import pdb


class Parameters:
    """A class to store the parameters in the model"""

    def __init__(self, fit=None, Icircuit=None, Ibar=None, gridpars_only=False):
        """Dimensional parameters"""
        # Constants
        self.R = 8.314  # Gas constant [J.K-1.mol-1]
        self.F = 96487  # Faraday constant [C.mol-1]
        self.T = 21.7 + 273.15  # Reference temperature [K]

        # Lengths
        self.Ln = (1.8e-3) / 2  # Half-width of negative electrode [m]
        self.Ls = 1.5e-3  # Width of separator [m]
        self.Lp = (2.5e-3) / 2  # Half-width of positive electrode [m]
        self.L = self.Ln + self.Ls + self.Lp  # Total width [m]
        self.H = 0.114  # Cell height [m]
        self.W = 0.065  # Cell depth [m]
        # Dimensionless
        self.ln = self.Ln / self.L  # Dimensionless half-width of negative electrode
        self.ls = self.Ls / self.L  # Dimensionless width of separator
        self.lp = self.Lp / self.L  # Dimensionless half-width of positive electrode

        # If we only needed the grid parameters, return
        if gridpars_only:
            return None

        # Current collectors
        self.A_cs = self.H * self.W  # Area of the current collectors [m2]
        self.ibar = Ibar / (8 * self.A_cs)  # Specified scale for the current [A.m-2]
        self.Icircuit = Icircuit
        self.voltage_cutoff_circuit = 10  # Voltage cut-off for the circuit[V]
        self.Q = 17  # Capacity [Ah]
        self.Crate = Ibar / self.Q  # C-rate [-]

        # Microstructure
        self.An = (
            2.5e6
        )  # Negative electrode surface area density [m-1] (or 1e4 or 1e6?)
        self.Ap = 20.5e6  # Positive electrode surface area density [m-1]
        self.epsnmax = fit["epsnmax"]  # Max porosity of negative electrode [-]
        self.epssmax = fit["epssmax"]  # Max porosity of separator [-]
        self.epspmax = fit["epspmax"]  # Max porosity of positive electrode [-]

        # Stoichiometric coefficients
        self.spn = -1  # s_+ in the negative electrode [-]
        self.spp = -3  # s_+ in the positive electrode [-]

        # Electrolyte physical properties
        self.cmax = fit["cmax"] * 1e3  # Maximum electrolye concentration [mol.m-3]
        self.tpw = 0.7  # Cation transference number [-]
        self.Vw = 17.5e-6  # Partial molar volume of water [m3.mol-1]
        self.Vp = 13.5e-6  # Partial molar volume of cations [m3.mol-1]
        self.Vn = 31.5e-6  # Partial molar volume of anions [m3.mol-1]
        self.Ve = self.Vn + self.Vp  # Partial molar volume of electrolyte [m3.mol-1]
        self.Mw = 18.01e-3  # Molar mass of water [kg.mol-1]
        self.Mp = 1e-3  # Molar mass of cations [kg.mol-1]
        self.Mn = 9.7e-2  # Molar mass of anions [kg.mol-1]
        self.Me = self.Mn + self.Mp  # Molar mass of electrolyte [kg.mol-1]
        self.DeltaVliqN = (
            self.Vn - self.Vp
        )  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]
        self.DeltaVliqP = (
            2 * self.Vw - self.Vn - 3 * self.Vp
        )  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]

        # Electrode physical properties
        self.VPb = 207e-3 / 11.34e3  # Molar volume of lead [m3.mol-1]
        self.VPbO2 = 239e-3 / 9.38e3  # Molar volume of lead dioxide [m3.mol-1]
        self.VPbSO4 = 303e-3 / 6.29e3  # Molar volume of lead sulfate [m3.mol-1]
        self.DeltaVsurfN = (
            self.VPb - self.VPbSO4
        )  # Net Molar Volume consumed in neg electrode [m3.mol-1]
        self.DeltaVsurfP = (
            self.VPbSO4 - self.VPbO2
        )  # Net Molar Volume consumed in pos electrode [m3.mol-1]
        self.sigma_eff_n = (
            4.8e6 * (1 - self.epsnmax) ** 1.5
        )  # Effective lead conductivity [S/m-1]
        self.sigma_eff_p = (
            8e3 * (1 - self.epspmax) ** 1.5
        )  # Effective lead dioxide conductivity [S/m-1]
        self.d = 1e-7  # Pore size [m]

        # Butler-Volmer
        self.jref_n = fit["jref_n"]  # Reference exchange-current density (neg) [A.m-2]
        self.jref_p = fit["jref_p"]  # Reference exchange-current density (pos) [A.m-2]
        self.Cdl = 0.2  # Double-layer capacity [F.m-2]
        self.U_Pb_ref = -0.294  # Reference OCP in the lead [V]
        self.U_PbO2_ref = 1.628  # Reference OCP in the lead dioxide [V]

        self.scales = Scales(self)

        """Dimensionless variables"""
        self.Cd = (
            (self.L ** 2)
            / self.D_hat(self.cmax)
            / (self.cmax * self.F * self.L / self.ibar)
        )  # Diffusional C-rate: diffusion timescale/discharge timescale
        self.alpha = (2 * self.Vw - self.Ve) * self.cmax  # Excluded volume fraction
        self.sn = -(self.spn + 2 * self.tpw) / 2  # Dimensionless rection rate (neg)
        self.sp = -(self.spp + 2 * self.tpw) / 2  # Dimensionless rection rate (pos)
        self.iota_s_n = (
            self.sigma_eff_n * self.R * self.T / (self.F * self.L) / self.ibar
        )  # Dimensionless lead conductivity
        self.iota_s_p = (
            self.sigma_eff_p * self.R * self.T / (self.F * self.L) / self.ibar
        )  # Dimensionless lead dioxide conductivity
        self.iota_ref_n = self.jref_n / (
            self.ibar / (self.An * self.L)
        )  # Dimensionless exchange-current density (neg)
        self.iota_ref_p = self.jref_p / (
            self.ibar / (self.Ap * self.L)
        )  # Dimensionless exchange-current density (pos)
        self.beta_surf_n = (
            -self.cmax * self.DeltaVsurfN / 2
        )  # Molar volume change (lead)
        self.beta_surf_p = (
            -self.cmax * self.DeltaVsurfP / 2
        )  # Molar volume change (lead dioxide)
        self.beta_liq_n = (
            -self.cmax * self.DeltaVliqN / 2
        )  # Molar volume change (electrolyte, neg)
        self.beta_liq_p = (
            -self.cmax * self.DeltaVliqP / 2
        )  # Molar volume change (electrolyte, pos)
        self.beta_n = (
            self.beta_surf_n + self.beta_liq_n
        )  # Total molar volume change (neg)
        self.beta_p = (
            self.beta_surf_p + self.beta_liq_p
        )  # Total molar volume change (pos)
        self.omega_i = (
            self.cmax
            * self.Me
            / self.rho_hat(self.cmax)
            * (1 - self.Mw * self.Ve / self.Vw * self.Me)
        )  # Diffusive kinematic relationship coefficient
        self.omega_c = (
            self.cmax
            * self.Me
            / self.rho_hat(self.cmax)
            * (self.tpw + self.Mn / self.Me)
        )  # Migrative kinematic relationship coefficient
        self.gamma_dl_n = (
            self.Cdl
            * self.R
            * self.T
            * self.An
            * self.L
            / (self.F * self.ibar)
            / (self.cmax * self.F * self.L / self.ibar)
        )  # Dimensionless double-layer capacity (neg)
        self.gamma_dl_p = (
            self.Cdl
            * self.R
            * self.T
            * self.Ap
            * self.L
            / (self.F * self.ibar)
            / (self.cmax * self.F * self.L / self.ibar)
        )  # Dimensionless double-layer capacity (pos)
        self.voltage_cutoff = (
            self.F
            / (self.R * self.T)
            * (self.voltage_cutoff_circuit / 6 - (self.U_PbO2_ref - self.U_Pb_ref))
        )  # Dimensionless voltage cut-off
        self.U_rxn = self.ibar / (self.cmax * self.F)  # Reaction velocity scale
        self.pi_os = (
            self.mu_hat(self.cmax)
            * self.U_rxn
            * self.L
            / (self.d ** 2 * self.R * self.T * self.cmax)
        )  # Ratio of viscous pressure scale to osmotic pressure scale

        # Initial conditions
        self.qinit = fit["qinit"]  # Initial SOC [-]
        self.qmax = (
            (self.Ln * self.epsnmax + self.Ls * self.epssmax + self.Lp * self.epspmax)
            / self.L
            / (self.sp - self.sn)
        )  # Dimensionless max capacity
        self.epsDeltan = self.beta_surf_n / self.ln * self.qmax
        self.epsDeltap = self.beta_surf_p / self.lp * self.qmax
        self.cinit = self.qinit
        self.epsn0 = self.epsnmax - self.epsDeltan * (
            1 - self.qinit
        )  # Initial pororsity (neg) [-]
        self.epss0 = self.epssmax  # Initial pororsity (sep) [-]
        self.epsp0 = self.epspmax - self.epsDeltap * (
            1 - self.qinit
        )  # Initial pororsity (pos) [-]

    def icell(self, t):
        """The dimensionless current function (could be some data)"""
        # This is a function of dimensionless time; Icircuit is a function of
        # time in *hours*
        return self.Icircuit(t * self.scales.time) / (8 * self.A_cs) / self.ibar

    def D_hat(self, c):
        """Dimensional effective Fickian diffusivity in the electrolyte [m2.s-1]"""
        return (1.75 + 260e-6 * c) * 1e-9

    def D_eff(self, c, eps):
        """Dimensionless effective Fickian diffusivity in the electrolyte"""
        return self.D_hat(c * self.cmax) / self.D_hat(self.cmax) * (eps ** 1.5)

    def kappa_hat(self, c):
        """Dimensional effective conductivity in the electrolyte [S.m-1]"""
        return c * np.exp(6.23 - 1.34e-4 * c - 1.61e-8 * c ** 2) * 1e-4

    def kappa_eff(self, c, eps):
        """Dimensionless molar conductivity in the electrolyte"""
        kappa_scale = (
            self.F ** 2 * self.cmax * self.D_hat(self.cmax) / (self.R * self.T)
        )
        return self.kappa_hat(c * self.cmax) / kappa_scale * (eps ** 1.5)

    def chi_hat(self, c):
        """Dimensional Darken thermodynamic factor in the electrolyte [-]"""
        return 0.49 + 4.1e-4 * c

    def chi(self, c):
        """Dimensionless Darken thermodynamic factor in the electrolyte"""
        chi_scale = 1 / (2 * (1 - self.tpw))
        return self.chi_hat(c * self.cmax) / chi_scale / (1 + self.alpha * c)

    def curlyK_hat(self, eps):
        """Dimensional permeability [m2]"""
        return eps ** 3 * self.d ** 2 / (180 * (1 - eps) ** 2)

    def curlyK(self, eps):
        """Dimensionless permeability"""
        return self.curlyK_hat(eps) / self.d ** 2

    def mu_hat(self, c):
        """Dimensional viscosity of electrolyte [kg.m-1.s-1]"""
        return 0.89e-3 + 1.11e-7 * c + 3.29e-11 * c ** 2

    def mu(self, c):
        """Dimensionless viscosity of electrolyte"""
        return self.mu_hat(c * self.cmax) / self.mu_hat(self.cmax)

    def rho_hat(self, c):
        """Dimensional density of electrolyte [kg.m-3]"""
        return self.Mw / self.Vw * (1 + (self.Me * self.Vw / self.Mw - self.Ve) * c)

    def rho(self, c):
        """Dimensionless density of electrolyte"""
        return self.rho_hat(c * self.cmax) / self.rho_hat(self.cmax)

    def cw_hat(self, c):
        """Dimensional solvent concentration [mol.m-3]"""
        return (1 - c * self.Ve) / self.Vw

    def cw(self, c):
        """Dimensionless solvent concentration"""
        return self.cw_hat(c * self.cmax) / self.cw_hat(self.cmax)

    def dcwdc(self, c):
        """Dimensionless derivative of cw with respect to c"""
        # Must have the same shape as c
        return 0 * c - self.Ve / self.Vw

    def m(self, c):
        """Dimensional electrolyte molar mass [mol.kg-1]"""
        return c * self.Vw / ((1 - c * self.Ve) * self.Mw)

    def dmdc(self, c):
        """Dimensional derivative of m with respect to c [kg-1]"""
        return self.Vw / ((1 - c * self.Ve) ** 2 * self.Mw)

    def U_Pb(self, c):
        """Dimensionless OCP in the negative electrode"""
        m = self.m(c * self.cmax)  # dimensionless
        U = (
            self.F
            / (self.R * self.T)
            * (
                -0.074 * np.log10(m)
                - 0.030 * np.log10(m) ** 2
                - 0.031 * np.log10(m) ** 3
                - 0.012 * np.log10(m) ** 4
            )
        )
        return U

    def U_Pb_hat(self, c):
        """Dimensional OCP in the negative electrode [V]"""
        return self.U_Pb_ref + self.R * self.T / self.F * self.U_Pb(c / self.cmax)

    def dUPbdc(self, c):
        """Dimensionless derivative of U_Pb with respect to c"""
        m = self.m(c * self.cmax)  # dimensionless
        dUdm = (
            self.F
            / (self.R * self.T)
            * (
                -0.074 / m / np.log(10)
                - 0.030 * 2 * np.log(m) / (m * np.log(10) ** 2)
                - 0.031 * 3 * np.log(m) ** 2 / m / np.log(10) ** 3
                - 0.012 * 4 * np.log(m) ** 3 / m / np.log(10) ** 4
            )
        )
        dmdc = self.dmdc(c * self.cmax) * self.cmax  # dimensionless
        return dmdc * dUdm

    def U_PbO2(self, c):
        """Dimensionless OCP in the positive electrode"""
        m = self.m(c * self.cmax)
        U = (
            self.F
            / (self.R * self.T)
            * (
                0.074 * np.log10(m)
                + 0.033 * np.log10(m) ** 2
                + 0.043 * np.log10(m) ** 3
                + 0.022 * np.log10(m) ** 4
            )
        )
        return U

    def U_PbO2_hat(self, c):
        """Dimensional OCP in the positive electrode [V]"""
        return self.U_PbO2_ref + self.R * self.T / self.F * self.U_PbO2(c / self.cmax)

    def dUPbO2dc(self, c):
        """Dimensionless derivative of U_Pb with respect to c"""
        m = self.m(c * self.cmax)  # dimensionless
        dUdm = (
            self.F
            / (self.R * self.T)
            * (
                0.074 / m / np.log(10)
                + 0.033 * 2 * np.log(m) / (m * np.log(10) ** 2)
                + 0.043 * 3 * np.log(m) ** 2 / m / np.log(10) ** 3
                + 0.022 * 4 * np.log(m) ** 3 / m / np.log(10) ** 4
            )
        )
        dmdc = self.dmdc(c * self.cmax) * self.cmax  # dimensionless
        return dmdc * dUdm


class Scales:
    """A class to store the scales we used for nondimensionalisation"""

    def __init__(self, pars):
        self.length = pars.L  # Length scale [m]
        self.time = (
            pars.cmax * pars.F * pars.L / pars.ibar / 3600
        )  # Discharge time scale [h]
        self.conc = pars.cmax  # Concentration scale [mol.m-3]
        self.current = pars.ibar  # Current scale (user defined) [A.m-2]
        self.jn = pars.ibar / (
            pars.An * pars.L
        )  # Interfacial current scale (neg) [A.m-2]
        self.jp = pars.ibar / (
            pars.Ap * pars.L
        )  # Interfacial current scale (pos) [A.m-2]
        self.pot = pars.R * pars.T / pars.F  # Thermal voltage [V]
        self.one = 1  # Porosity, SOC [-]

        # Combined scales
        self.It = self.current * self.time

        """Dictionary matching solution attributes to re-dimensionalisation scales"""
        self.match = {
            "t": "time",
            "x": "length",
            "icell": "current",
            "Icircuit": "current",
            "intI": "It",
            "c0": "conc",
            "c0_v": "conc",
            "c1": "conc",
            "c": "conc",
            "eps0n": "one",
            "eps0s": "one",
            "eps0p": "one",
            "epsn": "one",
            "epss": "one",
            "epsp": "one",
            "eps0": "one",
            "eps": "one",
            "phi0": ("pot", -pars.U_Pb_ref),
            "phi": ("pot", -pars.U_Pb_ref),
            "phis0": ("pot", pars.U_PbO2_ref - pars.U_Pb_ref),
            "phis": ("pot", pars.U_PbO2_ref - pars.U_Pb_ref),
            "phisn": "pot",
            "phisp": ("pot", pars.U_PbO2_ref - pars.U_Pb_ref),
            "eta0n": "pot",
            "eta0p": "pot",
            "eta0": "pot",
            "eta": "pot",
            "etan": "pot",
            "etap": "pot",
            "xin": ("pot", -pars.U_Pb_ref),
            "xip": ("pot", pars.U_PbO2_ref),
            "V0": ("pot", pars.U_PbO2_ref - pars.U_Pb_ref),
            "V1": ("pot", pars.U_PbO2_ref - pars.U_Pb_ref),
            "V": ("pot", pars.U_PbO2_ref - pars.U_Pb_ref),
            "V0circuit": ("pot", 6 * (pars.U_PbO2_ref - pars.U_Pb_ref)),
            "Vcircuit": ("pot", 6 * (pars.U_PbO2_ref - pars.U_Pb_ref)),
            "j0": ("jn", "jp"),
            "j": ("jn", "jp"),
            "jn": "jn",
            "jp": "jp",
            "i0": "current",
            "i": "current",
            "i_n": "current",
            "i_p": "current",
            "isolid0": "current",
            "isolid": "current",
            "q": "one",
        }


class Grid:
    """A class for information regarding the dimensionless grid"""

    def __init__(self, target_npts):
        # Generate parameters
        pars = Parameters(gridpars_only=True)

        # We aim to create the grid as uniformly as possible
        targetmeshsize = min(pars.ln, pars.ls, pars.lp) / target_npts
        # Negative electrode
        self.nn = round(pars.ln / targetmeshsize) + 1
        self.dxn = pars.ln / (self.nn - 1)
        # Separator
        self.ns = round(pars.ls / targetmeshsize) - 1
        self.dxs = pars.ls / (self.ns + 1)
        # Positive electrode
        self.np = round(pars.lp / targetmeshsize) + 1
        self.dxp = pars.lp / (self.np - 1)
        # Totals
        self.n = self.nn + self.ns + self.np

        # Grid: edges
        self.xn = np.arange(0.0, pars.ln + self.dxn / 2, self.dxn)
        self.xs = np.arange(
            pars.ln + self.dxs, pars.ln + pars.ls - self.dxs / 2, self.dxs
        )
        self.xp = np.arange(pars.ln + pars.ls, 1 + self.dxp / 2, self.dxp)
        self.x = np.concatenate([self.xn, self.xs, self.xp])
        self.dx = np.diff(self.x)

        # Grid: centres
        self.xcn = np.arange(self.dxn / 2, pars.ln, self.dxn)
        self.xcs = np.arange(pars.ln + self.dxp / 2, pars.ln + pars.ls, self.dxs)
        self.xcp = np.arange(pars.ln + pars.ls + self.dxp / 2, 1, self.dxp)
        self.xc = np.concatenate([self.xcn, self.xcs, self.xcp])
        self.dxc = np.diff(self.xc)
