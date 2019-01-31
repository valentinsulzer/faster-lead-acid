"""
Functions for solving ODEs and post-processing
"""
import numpy as np

def initial_conditions(p, g, method):
    """Calculates the initial conditions, returning them as a vector"""
    # Variables is defined at cell centers (finite volumes)
    c = p.cinit*np.ones(g.n-1)
    epsn = p.epsn0*np.ones(g.nn-1)
    epsp = p.epsp0*np.ones(g.np-1)

    # For the composite we are done
    if method == 'composite':
        return np.concatenate((c,epsn,epsp))

    # For the full numerical we also need the xis
    elif method == 'full':
        xin = p.U_Pb(p.cinit)*np.ones(g.nn-1)
        xip = p.U_PbO2(p.cinit)*np.ones(g.np-1)
        return np.concatenate((c,epsn,epsp,xin,xip))


def get_vars(y, p, g, method, t=None):
    """Extracts the variables from the vector u in a more readable form"""
    c = y[:g.n-1]
    epsn = y[g.n-1:g.n-1+g.nn-1]
    epsp = y[g.n-1+g.nn-1:g.n-1+g.nn-1+g.np-1]
    if y.size == len(y):
        eps = np.concatenate([epsn, p.epss0*np.ones(g.ns+1), epsp])
    else:
        eps = np.concatenate([epsn, p.epss0*np.ones((g.ns+1,len(t))), epsp])

    if method == 'composite':
        return (c, eps)

    elif method == 'full':
        xin = y[g.n-1+g.nn-1+g.np-1:g.n-1+2*(g.nn-1)+g.np-1]
        xip = y[g.n-1+2*(g.nn-1)+g.np-1:g.n-1+2*(g.nn-1)+2*(g.np-1)]
        return (c, eps, xin, xip)


def fluxes(c, eps, p, g, method, xin=None, xip=None, I=None, t=None):
    """Compute the flux coefficients at cell edges"""
    # hack to make this work for the full solution (c is a (n-1)x(len(t)) matrix)
    # as well as ODE solver (c is a vector of length (n-1))
    dx_ = g.dx
    dxc_ = g.dxc
    if I.size > 1:
        dx_ = dx_[:,np.newaxis]
        dxc_ = dxc_[:,np.newaxis]

    # Mesh ratio for harmonic means
    mesh_ratio = dx_[:-1] / (dx_[:-1]+dx_[1:])
    # Flux at cell centres
    D_centres = p.D_eff(c, eps)
    # Flux coefficients at cell edges: harmonic mean
    D = ((D_centres[:-1]*D_centres[1:])
         /((1-mesh_ratio)*D_centres[:-1] + mesh_ratio*D_centres[1:]))
    # Fluxes at cell edges
    Ddcdx = D*(c[1:] - c[:-1])/dxc_
    # Boundary conditions
    Ddcdx = np.concatenate([np.zeros(I.shape),Ddcdx,np.zeros(I.shape)])

    # For composite we are done
    if method == 'composite':
        return Ddcdx

    # For the full numerical we also need the currents
    elif method == 'full':
        # We don't care about the separator here but include it with NaNs
        # to simplify the code
        # There must be a more pythonic way of doing this:
        if I.size == 1:
            iota_s = np.concatenate([p.iota_s_n*np.ones(g.nn-1),
                                     np.nan*np.ones(g.ns+1),
                                     p.iota_s_p*np.ones(g.np-1)])
            xi = np.concatenate([xin, np.nan*np.ones(g.ns+1), xip])
        elif I.size > 1:
            iota_s = np.concatenate([p.iota_s_n*np.ones((g.nn-1,len(t))),
                                     np.nan*np.ones((g.ns+1,len(t))),
                                     p.iota_s_p*np.ones((g.np-1,len(t)))])
            xi = np.concatenate([xin, np.nan*np.ones((g.ns+1,len(t))), xip])

        # Fluxes at cell centres
        kappa = p.kappa_eff(c, eps)

        kchi_centres = kappa*p.chi(c)/c/(p.Cd + kappa/iota_s)
        k_centres = kappa/(p.Cd + kappa/iota_s)
        # Flux coefficients at cell edges: harmonic mean
        kchi = ((kchi_centres[:-1]*kchi_centres[1:])
                /((1-mesh_ratio)*kchi_centres[:-1] + mesh_ratio*kchi_centres[1:]))
        k = ((k_centres[:-1]*k_centres[1:])
              /((1-mesh_ratio)*k_centres[:-1] + mesh_ratio*k_centres[1:]))

        # Source terms
        kI_centres = (kappa*I/iota_s)/(p.Cd + kappa/iota_s)
        # Source terms at cell edges: arithmetic mean
        kI = (kI_centres[1:] + kI_centres[:-1])/2

        # Current density
        i = kchi*(c[1:] - c[:-1])/dxc_ + k*(xi[1:] - xi[:-1])/dxc_ + kI

        # Split and apply boundary conditions
        i_n = np.concatenate([np.zeros(I.shape), i[:g.nn-2], I])
        i_p = np.concatenate([I, i[g.nn+g.ns:], np.zeros(I.shape)])

        # Interfacial current density and overpotentials
        jn = (i_n[1:] - i_n[:-1])/g.dxn
        jp = (i_p[1:] - i_p[:-1])/g.dxp
        etan = xin - p.U_Pb(c[:g.nn-1])
        etap = xip - p.U_PbO2(c[g.nn+g.ns:])

        return Ddcdx, i_n, i_p, jn, jp, etan, etap


def negative_concentration(_, y, p, g):
    """Termination event for the concentration or porosity going negative"""
    (c, eps) = get_vars(y, p, g, 'composite')
    # Could also implement a voltage cut-off, but this requires calculating potentials
    # unless we think carefully about a lower bound based on the xis
    TOL = 1e-3
    return np.min(c) - TOL


def derivs(t, y, p, g, method):
    """Calculate the time derivatives for the system of ODEs"""
    # Current
    I = np.array([p.icell(t)])

    if method == 'composite':
        # Unpack y
        (c, eps0) = get_vars(y, p, g, 'composite')
        # Calculate fluxes: for code simplicity we use D(c) instead of D(c0),
        # but this makes almost no difference
        Ddcdx = fluxes(c, eps0, p, g, 'composite', I=I)

        """Time derivatives"""
        # Porosity: for code simplicity we calculate eps0n and eps0p across each
        # electrode even though they are uniform; this is only slighlty slower
        # than calculating them just once
        deps0ndt = - p.beta_surf_n*I/p.ln*np.ones(g.nn-1)
        deps0pdt = p.beta_surf_p*I/p.lp*np.ones(g.np-1)
        deps0dt = np.concatenate([deps0ndt, np.zeros(g.ns+1),deps0pdt])

        # Concentration
        sj0 = np.concatenate([p.sn*I/p.ln*np.ones(g.nn-1),
                              np.zeros(g.ns+1),
                              - p.sp*I/p.lp*np.ones(g.np-1)])
        dcdt = 1/eps0*(1/p.Cd*(Ddcdx[1:]-Ddcdx[:-1])/g.dx + sj0 - c*deps0dt)

        # Put everything together
        return np.concatenate((dcdt, deps0ndt, deps0pdt))

    elif method == 'full':
        # Unpack y
        (c, eps, xin, xip) = get_vars(y, p, g, 'full')
        c = np.maximum(0.001, np.minimum(c, 1))
        # Calculate fluxes using finite volumes
        (Ddcdx, i_n, i_p, jn, jp, etan, etap) = fluxes(c, eps, p, g, 'full', xin, xip, I, t)

        """Time derivatives"""
        # Porosities
        depsndt = - p.beta_surf_n*jn
        depspdt = - p.beta_surf_p*jp
        depsdt = np.concatenate([depsndt, np.zeros(g.ns+1), depspdt])

        # Concentration
        sj = np.concatenate([p.sn*jn,
                             np.zeros(g.ns+1),
                             p.sp*jp])
        dcdt = 1/eps*(1/p.Cd*(Ddcdx[1:]-Ddcdx[:-1])/g.dx + sj - c*depsdt)

        # Potentials
        cn = c[:g.nn-1]
        cp = c[g.nn+g.ns:]
        dxindt = 1/p.gamma_dl_n*(jn - 2 * p.iota_ref_n * cn * np.sinh(etan))
        dxipdt = 1/p.gamma_dl_p*(jp - 2 * p.iota_ref_p * cp**2 * p.cw(cp) *
                                 np.sinh(etap))

#        print(t)
#        if t>0.5:
#            pause
        return np.concatenate((dcdt, depsndt, depspdt, dxindt, dxipdt))

def calculate_phi(t, sol, p, g):
    """Calculate phi in the post-processing, based on the concentration and current"""
    # Set up grid
    dx_ = g.dx[:,np.newaxis]
    dxc_ = g.dxc[:,np.newaxis]
    mesh_ratio = dx_[:-1] / (dx_[:-1]+dx_[1:])

    # Slightly modified coefficients (equivalent to sending iota_s to infinity)
    kappa_centres = p.kappa_eff(sol.c, sol.eps)
    kchi_centres = kappa_centres*p.chi(sol.c)/sol.c

    # Flux coefficients at cell edges: harmonic mean
    kchi = ((kchi_centres[:-1]*kchi_centres[1:])
            /((1-mesh_ratio)*kchi_centres[:-1] + mesh_ratio*kchi_centres[1:]))
    k = ((kappa_centres[:-1]*kappa_centres[1:])
          /((1-mesh_ratio)*kappa_centres[:-1] + mesh_ratio*kappa_centres[1:]))
    # Flux at cell edges
    kchidcdx = kchi*(sol.c[1:] - sol.c[:-1])/dxc_

    # Integrate equation for phi; for now let phi=0 at x=0
    phi = np.concatenate([np.zeros((1,len(t))),
                          np.cumsum(dxc_/k*(kchidcdx - p.Cd*sol.i[1:-1]), axis=0)])
    # Now shift so that phis = 0 at x = 0 (phi = -xi at x=0)
    phi -= sol.xin[0]

    return phi
