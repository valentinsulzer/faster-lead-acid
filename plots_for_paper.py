"""
Script to make plots and tables for paper:
    1. Comparing results from a single run for four solutions:
       numerical, leading-order quasi-static, first-order quasi-static and composite
                - Voltages
                - Space-dependent variables:
                    - concentration
                    - overpotential
                    - interfacial current density
    2. Table averaging over several runs:
        - runtimes
        - speed gain compared to numerical
    3. Errors
    4. Voltage breakdown
    5. Parameter fitting
    6. Calculating capacities for Peukert's law
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict, OrderedDict
from src.analytical import *
from src.constants import *
from src.numerical import Numerical
import src.plotting as myplt
from src.util import calculate_error, make_results_table, save_data, load_data
from parameter_fitting import do_parameter_fit
import pdb


def current(t_, size, step_time=None):
    """
    Define the current. Output must be the same type as input t_
    Also needs to work with data (interpolate)
    """
    constant = np.where(t_ > 0, size, 0)
    step = np.where(t_ < step_time, size, 0)

    return constant


def compare_solutions(base_parameters, grid, Crates, solution_methods, timeaxis="time"):
    """
    1. Compare solutions for a range of constant-current discharges
    """
    # filenames
    if timeaxis == "time":
        filename_end = ""
    elif timeaxis == "capacity":
        filename_end = "_capacity"
    if set(solution_methods) == set(
        [Numerical, LeadingOrderQuasiStatic, FirstOrderQuasiStatic, Composite]
    ):
        solution_file_str = "all"
    elif set(solution_methods) == set(
        [Numerical, LeadingOrderQuasiStatic, FirstOrderQuasiStatic]
    ):
        solution_file_str = "numerical_qs_only"
    elif set(solution_methods) == set([Numerical, LeadingOrderQuasiStatic]):
        solution_file_str = "numerical_qsO1_only"
    elif solution_methods == [Numerical]:
        solution_file_str = "numerical_only"
    filename = "out/plot_calls/voltages_{}{}.txt".format(
        solution_file_str, filename_end
    )
    # compute and save
    with open(filename, "w") as volt_plot_calls:
        for k, Crate in enumerate(Crates):
            # Set time and current
            Ibar = Crate * 17
            tfinal = 26 / Ibar  # hours
            tsteps = 1000
            t_dim = np.linspace(0, tfinal, tsteps)

            def Icircuit(t):
                return current(t, Ibar, step_time=tfinal / 2)

            pars = Parameters(fit=base_parameters, Icircuit=Icircuit, Ibar=Ibar)
            t = t_dim / pars.scales.time
            # Solution storage
            solutions = []
            # Prepare files for latex plotting
            if len(solution_methods) > 1:
                # group plot if there are several solution methods, for clarity
                if k in [0, 3]:
                    volt_plot_calls.write(
                        (
                            "\\nextgroupplot[title={{\\textbf{{({})}} {}C }}, "
                            + "ylabel={{Voltage, $\\di{{V}}_\\text{{circuit}}$ [V]}}]\n"
                        ).format(chr(k + 97), Crate)
                    )
                else:
                    volt_plot_calls.write(
                        ("\\nextgroupplot[title={{\\textbf{{({})}} {}C }}]\n").format(
                            chr(k + 97), Crate
                        )
                    )
            file = "{}/{}C.txt".format(solution_file_str, Crate)
            with open(
                "out/plot_calls/concs/{}".format(file), "w"
            ) as conc_plot_calls, open(
                "out/plot_calls/water_concs/{}".format(file), "w"
            ) as water_conc_plot_calls, open(
                "out/plot_calls/porosities/{}".format(file), "w"
            ) as por_plot_calls, open(
                "out/plot_calls/overpotentials/{}".format(file), "w"
            ) as over_plot_calls, open(
                "out/plot_calls/electrolyte_potentials/{}".format(file), "w"
            ) as elecpot_plot_calls, open(
                "out/plot_calls/interfacial_currents/{}".format(file), "w"
            ) as intcur_plot_calls:
                for method in solution_methods:
                    soln = method(t, pars, grid)
                    soln.dimensionalise(pars, grid)
                    # Store solution
                    solutions.append(soln)
                    # Store voltage
                    if len(solution_methods) > 1:
                        linestyle = soln.latexlinestyle()
                    else:
                        linestyle = ""
                    if timeaxis == "time":
                        save_data(
                            "voltages/{!s}/{}C".format(soln, Crate),
                            soln.t,
                            soln.Vcircuit,
                            volt_plot_calls,
                            linestyle,
                        )
                    elif timeaxis == "capacity":
                        dt = soln.t[1] - soln.t[0]
                        capacity_used = np.cumsum(soln.Icircuit) * dt
                        save_data(
                            "voltages/{!s}_capacity/{}C".format(soln, Crate),
                            capacity_used,
                            soln.Vcircuit,
                            volt_plot_calls,
                            linestyle,
                        )

                    # Add labels to first entry
                    if len(solution_methods) > 1 and k == 0:
                        volt_plot_calls.write("\\label{{pgfplots:{!s}}}\n".format(soln))
                    # Store internal variables
                    # Find indices where SoC is 100%, 75%, 50%, 25%
                    idcs = [0] + [
                        np.nanargmin(abs(soln.q - x)) for x in [0.75, 0.5, 0.25]
                    ]
                    for i, idx in enumerate(idcs):
                        opacity = 0.1 + 0.3 * i
                        # Display SOC?
                        if Crate == 0.1 and (
                            str(soln) == "LOQS" or solution_file_str == "numerical_only"
                        ):
                            if i == 0:
                                SOCstr = "{:.0f}\\% SOC".format(soln.q[idx][0] * 100)
                            else:
                                SOCstr = "{:.0f}\\%".format(soln.q[idx][0] * 100)
                            extra_feature = (
                                "\nnode[above left, pos=1, opacity=1, text=black, "
                                + "font=\\footnotesize]{{{}}}"
                            ).format(SOCstr)
                        else:
                            extra_feature = ""
                        if (
                            Crate < 2 or i < 3 or solution_file_str == "numerical_only"
                        ):  # Don't include i=4 for Crate =2
                            save_data(
                                "concs/{!s}/{}C_idx={}".format(soln, Crate, idx),
                                soln.x,
                                soln.c[idx] / 1e3,
                                conc_plot_calls,
                                soln.latexlinestyle(opacity),
                                extra_feature=extra_feature,
                            )
                            save_data(
                                "water_concs/{!s}/{}C_idx={}".format(soln, Crate, idx),
                                soln.x,
                                pars.cw_hat(soln.c)[idx] / 1e3,
                                water_conc_plot_calls,
                                "[color=blue, opacity={}]".format(opacity),
                            )
                            save_data(
                                "porosities/{!s}/{}C_idx={}".format(soln, Crate, idx),
                                soln.x,
                                soln.eps[idx],
                                por_plot_calls,
                                soln.latexlinestyle(opacity),
                            )
                            save_data(
                                "overpotentials/{!s}/{}C_idx={}".format(
                                    soln, Crate, idx
                                ),
                                soln.x,
                                soln.eta[idx],
                                over_plot_calls,
                                soln.latexlinestyle(opacity),
                            )
                            save_data(
                                "electrolyte_potentials/{!s}/{}C_idx={}".format(
                                    soln, Crate, idx
                                ),
                                soln.x,
                                soln.phi[idx],
                                elecpot_plot_calls,
                                soln.latexlinestyle(opacity),
                            )
                            save_data(
                                "interfacial_currents/{!s}/{}C_idx={}".format(
                                    soln, Crate, idx
                                ),
                                soln.x,
                                soln.j[idx],
                                intcur_plot_calls,
                                soln.latexlinestyle(opacity),
                            )
                # Add legend to fifth entry in voltages
                if len(solution_methods) > 1:
                    if k == 4:
                        volt_plot_calls.write(
                            (
                                "\\legend{{{!s}"
                                + ",{!s}" * (len(solutions) - 1)
                                + "}}\n"
                            ).format(*tuple(solutions))
                        )
        if len(solution_methods) == 1:
            volt_plot_calls.write(
                ("\\legend{{{!s}C" + ", {!s}C" * (len(Crates) - 1) + "}}\n").format(
                    *tuple(Crates)
                )
            )


def time_average_behaviour(base_parameters, grid, Crates, solution_methods):
    """
    2. Make table showing average behaviour
    """
    # Set up solver times as dict
    solver_times = {}
    for k, Crate in enumerate(Crates):
        # Set time and current
        Ibar = Crate * 17
        tfinal = 21.7 / Ibar  # hours
        tsteps = 100
        t_dim = np.linspace(0, tfinal, tsteps)

        def Icircuit(t):
            return current(t, Ibar, step_time=tfinal / 2)

        pars = Parameters(fit=base_parameters, Icircuit=Icircuit, Ibar=Ibar)
        t = t_dim / pars.scales.time
        # Set up solver_times as a defaultdict
        solver_times[Crate] = defaultdict(int)
        n = 1
        for _ in range(n):
            for method in solution_methods:
                start = time.time()
                soln = method(t, pars, grid)
                solver_times[Crate][str(soln)] += time.time() - start

        for k in solver_times[Crate].keys():
            solver_times[Crate][k] /= n
    make_results_table("speed", solver_times)


def calculate_errors(base_parameters, grid, Crates, solution_methods):
    """
    3. Errors for logarithmically spaced out C-rates
    """
    numVs = {}
    with open("out/plot_calls/absolute_errors.txt", "w") as abserror_plot_calls, open(
        "out/plot_calls/relative_errors.txt", "w"
    ) as relerror_plot_calls:
        for method in solution_methods:
            # Initialise errors
            abs_errors = np.array([])
            rel_errors = np.array([])
            for Crate in Crates:
                # Set time and current
                Ibar = Crate * 17
                tfinal = 21.7 / Ibar  # hours
                tsteps = 1000
                t_dim = np.linspace(0, tfinal, tsteps)

                def Icircuit(t):
                    return current(t, Ibar, step_time=tfinal / 2)

                pars = Parameters(fit=base_parameters, Icircuit=Icircuit, Ibar=Ibar)
                t = t_dim / pars.scales.time
                # Calculate and store numerical solution
                if method == Numerical:
                    num = Numerical(t, pars, grid)
                    num.dimensionalise(pars, grid)
                    numVs[Crate] = num.V
                else:
                    soln = method(t, pars, grid)
                    soln.dimensionalise(pars, grid)
                    # Store error
                    abs_errors = np.append(
                        abs_errors, calculate_error(soln.V, numVs[Crate], "absolute")
                    )
                    rel_errors = np.append(
                        rel_errors, calculate_error(soln.V, numVs[Crate], "relative")
                    )
            if method != Numerical:
                save_data(
                    "voltages/absolute_errors_{!s}".format(soln),
                    Crates,
                    abs_errors,
                    abserror_plot_calls,
                    soln.latexlinestyle(),
                )
                save_data(
                    "voltages/relative_errors_{!s}".format(soln),
                    Crates,
                    rel_errors,
                    relerror_plot_calls,
                    soln.latexlinestyle(),
                )


def calculate_voltage_breakdown(base_parameters, grid, Crates, solution_methods):
    """
    4. Voltage breakdown using composite solution
    """
    for k, Crate in enumerate(Crates):
        # Set time and current
        Ibar = Crate * 17
        tfinal = 21.7 / Ibar  # hours
        tsteps = 100
        t_dim = np.linspace(0, tfinal, tsteps)

        def Icircuit(t):
            return current(t, Ibar, step_time=tfinal / 2)

        pars = Parameters(fit=base_parameters, Icircuit=Icircuit, Ibar=Ibar)
        t = t_dim / pars.scales.time

        # Make composite solution
        comp = Composite(t, pars, grid)

        # Use dimensionless composite solution to calculate voltage breakdown
        V_breakdown = VoltageBreakdown(comp, pars, grid)
        running_tot = np.zeros(V_breakdown.tot.shape)

        # Dimensionalise for plotting
        comp.dimensionalise(pars, grid)

        with open(
            "out/plot_calls/voltage_breakdowns/{}C.txt".format(Crate), "w"
        ) as breakdown_plot_calls:
            for attr in ["init", "Un", "Up", "kn", "kp", "c", "o"]:
                save_data(
                    "voltages/breakdowns/{}/{}C".format(attr, Crate),
                    comp.t,
                    6 * pars.scales.pot * V_breakdown.__dict__[attr],
                    breakdown_plot_calls,
                    extra_feature="\closedcycle",
                )
                # labels
                if k == 0:
                    breakdown_plot_calls.write("\\label{{pgfplots:V{}}}\n".format(attr))


def parameter_fit():
    imei = "861508033133471"
    dates = OrderedDict()
    dates["3A"] = ("2017-03-25", "2017-03-25")
    dates["2.5A"] = ("2017-03-26", "2017-03-26")
    dates["2A"] = ("2017-03-27", "2017-03-27")
    dates["1.5A"] = ("2017-03-28", "2017-03-29")
    # dates['1A (0)'] = ('2017-03-30','2017-03-31')
    dates["1A"] = ("2017-03-31", "2017-04-01")
    dates["0.5A"] = ("2017-04-02", "2017-04-04")

    datas = load_data("March17", imei=imei, dates=dates, wt=10, ignore_start=True)

    # Load existing fits or initiate
    try:
        fits = np.load("out/data/fits/dict_all.npy")[np.newaxis][0]
    except FileNotFoundError:
        fits = defaultdict(dict)
    # Choose least squares solver and model solver
    for lsq_solver in ["scipy", "dfogn"]:
        for model_solver in [
            LeadingOrderQuasiStatic,
            FirstOrderQuasiStatic,
            Composite,
            Numerical,
        ]:
            print("-" * 60)
            # Define the starting point
            x0 = np.concatenate(
                [np.array([0.6, 0.9, 0.08, 1 / 6]), np.ones(len(datas) - 1)]
            )

            fits[lsq_solver][model_solver] = do_parameter_fit(
                x0, datas, lsq_solver, model_solver
            )  # (x, f, soln_time)
            np.save(
                "out/data/fits/dict_all.npy".format(lsq_solver, model_solver.__name__),
                fits,
            )

            # fig is saved inside parameter_fitting.py

    # Fill tables
    for lsq_solver in ["scipy", "dfogn"]:
        with open(
            "out/tables/fits/{}_performance.txt".format(lsq_solver), "w"
        ) as perf_table_row:
            for model_solver in [
                LeadingOrderQuasiStatic,
                FirstOrderQuasiStatic,
                Composite,
                Numerical,
            ]:
                # Make entries for performance table
                perf_table_row.write(
                    "& {1:.2f} & {2:.0f} ".format(*fits[lsq_solver][model_solver])
                )  # cost and time taken
                # Make entries for parameters table
                with open(
                    "out/tables/fits/{}_{}_params.txt".format(
                        lsq_solver, model_solver.__name__
                    ),
                    "w",
                ) as par_table_row:
                    for par in fits[lsq_solver][model_solver][0]:
                        par_table_row.write("& {:.2f}".format(par))


def calculate_capacities(base_parameters, grid, Crates, solution_methods):
    """
    6. Calculating capacities for Peukert's law
    """
    with open("out/plot_calls/capacities.txt", "w") as capacities_plot_calls:
        for method in solution_methods:
            # Initialise errors
            caps = np.array([])
            for Crate in Crates:
                # Set time and current
                Ibar = Crate * 17
                tfinal = 30 / Ibar  # hours
                tsteps = 2000
                t_dim = np.linspace(0, tfinal, tsteps)

                def Icircuit(t):
                    return current(t, Ibar, step_time=tfinal / 2)

                pars = Parameters(fit=base_parameters, Icircuit=Icircuit, Ibar=Ibar)
                t = t_dim / pars.scales.time
                soln = method(t, pars, grid)
                soln.dimensionalise(pars, grid)
                # Calculate and store capacity
                dt = soln.t[1] - soln.t[0]
                capacity = np.nansum(soln.Icircuit) * dt
                caps = np.append(caps, capacity)
            save_data(
                "capacities/{!s}".format(soln),
                Crates,
                caps,
                capacities_plot_calls,
                soln.latexlinestyle(),
            )


if __name__ == "__main__":
    # Close all figures
    plt.close("all")

    # Make base grid (doesn't change as we change other parameters)
    grid = Grid(10)
    base_parameters = {
        "qinit": 1,
        "epsnmax": 0.53,
        "epssmax": 0.92,
        "epspmax": 0.57,
        "cmax": 6.4,
        "jref_n": 0.08,
        "jref_p": 0.006,
    }
    solution_methods = [
        Numerical,
        LeadingOrderQuasiStatic,
        FirstOrderQuasiStatic,
        Composite,
    ]

    print("Compare")
    Crates = [0.1, 0.2, 0.5, 1, 2, 5]
    compare_solutions(base_parameters, grid, Crates, solution_methods, timeaxis="time")
    compare_solutions(base_parameters, grid, Crates, [Numerical], timeaxis="time")
    compare_solutions(base_parameters, grid, Crates, [Numerical], timeaxis="capacity")

    print("Time")
    Crates = [0.1, 0.5, 2, 5]
    time_average_behaviour(base_parameters, grid, Crates, solution_methods)

    print("Errors")
    Crates = np.logspace(-2, 1, 30)  # C-rates
    calculate_errors(base_parameters, grid, Crates, solution_methods)

    print("Breakdown")
    Crates = [0.1, 0.5, 5]
    calculate_voltage_breakdown(base_parameters, grid, Crates, solution_methods)

    print("fit")
    parameter_fit()

    print("Capacities")
    Crates = np.logspace(-2, 1, 30)  # C-rates
    calculate_capacities(base_parameters, grid, Crates, [Numerical])
