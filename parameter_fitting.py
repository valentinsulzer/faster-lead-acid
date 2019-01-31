"""
Fit parameters using data (synthetic or real)
"""
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt
import dfogn
import dfols
import matplotlib.pyplot as plt
import matplotlib.cbook
from collections import OrderedDict
import logging
import warnings
from src.analytical import *
from src.constants import *
from src.numerical import Numerical
from src.util import load_data, save_data
import pdb
import time


def do_parameter_fit(x0, datas, lsq_solver, model_solver, LARGE_V=50):
    """
    Function to fit parameters given a starting point and datas.
    Choice of which optimisation algorithm and which model to use.
    """

    def prediction_error(x, show_plots=True, show_stars=True, save_plot_data=False):
        """
        Compare model to data and return array of differences
        """
        # Rescale x
        x = lower + (upper - lower) * x
        # Initial SOCs
        q0s = np.append([1], x[4:])
        # Prepare dict
        modelVs = {}
        for i, name in enumerate(datas.keys()):
            data = datas[name]

            # Make dict of fitting parameters from x
            pars = {
                "epsnmax": x[0],
                "epssmax": x[1],
                "epspmax": x[0],
                "cmax": 5.65,
                "jref_n": x[2],
                "jref_p": x[2] / 10,
                "qinit": q0s[i],
            }

            # Shape preserving interpolation of current: do the hard work offline
            interpclass = interp.PchipInterpolator(data["time"], data["current"])

            def Icircuit(t):
                return interpclass(t)

            # Load parameters
            pars = Parameters(
                fit=pars, Icircuit=Icircuit, Ibar=np.max(Icircuit(data["time"]))
            )

            # Nondimensionalise time
            t = data["time"] / pars.scales.time

            # Run model
            model = model_solver(t, pars, grid, Vonly=True)
            model.dimensionalise(pars, grid)

            # Take away resistance of the wires
            model.Vcircuit -= x[3] * data["current"][:, np.newaxis]

            # Remove NaNs
            model.Vcircuit[np.isnan(model.Vcircuit)] = LARGE_V

            # Store voltages
            modelVs[name] = model.Vcircuit

        # Plot data vs model
        if show_plots:
            plt.ion()
            fig = plt.figure(1)
            plt.clf()
            for i, name in enumerate(datas.keys()):
                plt.plot(
                    datas[name]["time"],
                    datas[name]["voltage"],
                    "o",
                    markersize=1,
                    color="C" + str(i),
                )
                plt.plot(
                    datas[name]["time"], modelVs[name], color="C" + str(i), label=name
                )
            # plt.title('params={}'.format(x))
            plt.xlabel("Time [h]")
            plt.ylabel("Voltage [V]")
            legend = plt.legend()
            fig.canvas.flush_events()
            time.sleep(0.01)

        # Make vector of differences (including weights)
        diffs = np.concatenate(
            [
                (data["voltage"] - modelVs[name][:, 0]) * data["weights"]
                for name, data in datas.items()
            ]
        )

        # Show progress
        # if show_stars:
        #     plt.figure(2)
        #     plt.plot(x, np.dot(diffs, diffs), 'rx')
        #     plt.pause(.01)

        # Save plot data
        if save_plot_data:
            # Set linestyles
            linestyles = {
                "3A": "",
                "2.5A": "",
                "2A": "",
                "1.5A": "",
                "1A": "",
                "0.5A": "",
            }
            with open(
                "out/plot_calls/fits/{}_{}.txt".format(
                    lsq_solver, model_solver.__name__
                ),
                "w",
            ) as fit_plot_calls:
                # Save data and model from each current
                for i, name in enumerate(datas.keys()):
                    save_data(
                        "fits/{}/{}/{}_model".format(
                            lsq_solver, model_solver.__name__, name
                        ),
                        datas[name]["time"],
                        modelVs[name],
                        fit_plot_calls,
                        linestyles[name],
                        n_coarse_points=100,
                    )
                    save_data(
                        "fits/{}/{}/{}_data".format(
                            lsq_solver, model_solver.__name__, name
                        ),
                        datas[name]["time"],
                        datas[name]["voltage"],
                        fit_plot_calls,
                        linestyles[name],
                        n_coarse_points=100,
                    )
                # Add legend (two commas to ignore the data entries)
                fit_plot_calls.write(
                    (
                        "\\legend{{{!s}" + ", ,{!s}" * (len(datas.keys()) - 1) + "}}\n"
                    ).format(*tuple(datas.keys()))
                )
        return diffs

    # Compute grid (doesn't depend on base parameters)
    grid = Grid(10)

    fit_currents = datas.keys()

    # Set the bounds
    lower = np.concatenate([np.array([0.0, 0.0, 0.01, 0.0]), np.zeros(len(datas) - 1)])
    upper = np.concatenate([np.array([1.0, 1.0, 1.0, 1.0]), np.ones(len(datas) - 1)])

    # Rescale x0 so that all fitting parameters go from 0 to 1
    x0 = (x0 - lower) / (upper - lower)
    # errs = np.array([])
    # js = np.linspace(0.01,1,10)
    # for j in js:
    #     diffs = prediction_error(j, show_plots=True, show_stars=False)
    #     errs = np.append(errs, np.dot(diffs, diffs))
    # plt.figure(2)
    # plt.plot(js, errs)
    # plt.pause(0.01)
    # Do curve fitting
    print("Fit using {} on the {} model".format(lsq_solver, model_solver.__name__))
    print("-" * 60)
    if lsq_solver in ["dfogn", "dfols"]:
        # Set logging to INFO to view progress
        logging.basicConfig(level=logging.INFO, format="%(message)s")

        # Call and time DFO-GN or DFO-LS
        start = time.time()
        if lsq_solver == "dfogn":
            soln = dfogn.solve(
                prediction_error,
                x0,
                lower=np.zeros(len(x0)),
                upper=np.ones(len(x0)),
                rhoend=1e-5,
            )
        elif lsq_solver == "dfols":
            soln = dfols.solve(
                prediction_error,
                x0,
                bounds=(np.zeros(len(x0)), np.ones(len(x0))),
                rhoend=1e-5,
            )
        soln_time = time.time() - start

        # Scale x back to original scale
        x = lower + (upper - lower) * soln.x

        # Display output
        print(" *** DFO-GN results *** ")
        print("Solution xmin = %s" % str(x))
        print("Objective value f(xmin) = %.10g" % soln.f)
        print("Needed %g objective evaluations" % soln.nf)
        print("Exit flag = %g" % soln.flag)
        print(soln.msg)

        # Save solution parameters
        # save_output = {'y': True, 'n': False}[input('Save fitted params? (y/n): ')]
        # if save_output:
        #     filename = "out/fits/dfogn_{}.txt".format(model_solver.__name__)
        #     np.savetxt(filename, lower+(upper-lower)*soln.x)

        return (x, soln.f, soln_time)

    elif lsq_solver == "scipy":
        # Call and time scipy least squares
        start = time.time()
        soln = opt.least_squares(
            prediction_error,
            x0,
            bounds=(np.zeros(len(x0)), np.ones(len(x0))),
            method="trf",
            jac="2-point",
            diff_step=1e-5,
            ftol=1e-4,
            verbose=2,
        )
        soln_time = time.time() - start

        # Scale x back to original scale
        x = lower + (upper - lower) * soln.x

        # Display output
        print(" *** SciPy results *** ")
        print("Solution xmin = %s" % str(x))
        print("Objective value f(xmin) = %.10g" % (soln.cost * 2))
        print("Needed %g objective evaluations" % soln.nfev)
        print("Exit flag = %g" % soln.status)
        print(soln.message)

        # Save solution parameters
        # save_output = {'y': True, 'n': False}[input('Save fitted params? (y/n): ')]
        # if save_output:
        #     filename = "out/fits/scipy_{}.txt".format(model_solver.__name__)
        #     np.savetxt(filename, x)

        return (x, soln.cost * 2, soln_time)

    elif lsq_solver is None:
        # Do nothing
        diffs = prediction_error(x0)
        return (x0, np.dot(diffs, diffs), 0)

    # save plots (hacky!)
    prediction_error(x, save_plot_data=True)


if __name__ == "__main__":
    plt.close("all")
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    # Load real data or make synthetic data (long way so that order is preserved)
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
    # imei = 'batt1_run3_C01'
    # imei = 'batt2_run2_C01'
    # datas = load_data('Nov17', imei=imei)

    # Choose least squares solver and model solver
    lsq_solver = ["dfogn", "dfols", "scipy", None][
        int(input("Choose lsq solver (0: dfogn, 1: dfols, 2: scipy.opt, 3: None): "))
    ]
    model_solver = [
        Numerical,
        LeadingOrderQuasiStatic,
        FirstOrderQuasiStatic,
        Composite,
    ][
        int(
            input(
                "Choose model solver (0: Numerical, 1: LOQS, 2: FOQS, 3: Composite): "
            )
        )
    ]
    print("-" * 60)

    # Define the starting point
    fits = np.load("out/data/fits/dict_all.npy")[np.newaxis][0]
    x0 = fits["dfogn"][Numerical][0]
    # x0 = np.concatenate([np.array([0.6, 0.9, 0.08,0,6]), np.ones(len(datas)-1)])

    print(do_parameter_fit(x0, datas, lsq_solver, model_solver))

    # plt.savefig("out/figures/fits/{}_{}.eps".format(lsq_solver, model_solver.__name__),
    #             format='eps', dpi=1000)
    # plt.savefig("out/figures/fits/experiments_{}_{}.eps".format(lsq_solver, model_solver.__name__),
    #             format='eps', dpi=1000)
    plt.ioff()
    plt.show()
