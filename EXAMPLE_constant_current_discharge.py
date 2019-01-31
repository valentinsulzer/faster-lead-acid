"""
Main script to run a discharge
"""
import matplotlib.pyplot as plt
from src.analytical import *
from src.constants import *
from src.numerical import Numerical
from src.plotting import slider_plot
import time
import pdb

# Close all figures
plt.close("all")


def current(t_, size, step_time=None):
    """
    Define the current. Output must be the same type as input t_
    """
    constant = np.where(t_ > 0, size, 0)
    step = np.where(t_ < step_time, size, 0)

    return constant


# Specify current and time parameters
Crate = float(input("Enter C-rate: "))
Ibar = Crate * 17
tfinal = 24 / Ibar  # hours
tsteps = 100
t_dim = np.linspace(0, tfinal, tsteps)


def Icircuit(t):
    return current(t, Ibar, step_time=tfinal / 2)


# Load parameters, scales and grid
fitting_parameters = {
    "qinit": 1,
    "epsnmax": 0.53,
    "epssmax": 0.92,
    "epspmax": 0.57,
    "cmax": 5.6,
    "jref_n": 0.08,
    "jref_p": 0.006,
}
grid = Grid(10)
pars = Parameters(fitting_parameters, Icircuit, Ibar)
t = t_dim / pars.scales.time
# Set up solver_times dict
solver_times = {}

# Numerical
start = time.time()
num = Numerical(t, pars, grid)
solver_times["Numerical"] = time.time() - start
num.dimensionalise(pars, grid)

# Leading-order quasi-static
start = time.time()
loqs = LeadingOrderQuasiStatic(t, pars, grid)
solver_times["Quasi-static O(1)"] = time.time() - start
loqs.dimensionalise(pars, grid)

# First-order quasi-static
start = time.time()
foqs = FirstOrderQuasiStatic(t, pars, grid)
solver_times["Quasi-static O(Da)"] = time.time() - start
foqs.dimensionalise(pars, grid)

# Composite
start = time.time()
comp = Composite(t, pars, grid)
solver_times["Composite"] = time.time() - start
comp.dimensionalise(pars, grid)

# Compare solver_times
print("-" * 28)
print("Time taken (seconds)")
print("-" * 28)
for v, k in sorted([(v, k) for k, v in solver_times.items()]):
    print("{:18}: {:f}".format(k, v))

# Slider plot
slider_plot([num, comp, foqs, loqs])
