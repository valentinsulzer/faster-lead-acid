# -*- coding: utf-8 -*-
"""
Useful extra functions
"""
import numpy as np
import pandas as pd
import numpy.linalg as LA
from collections import OrderedDict
import pdb


def calculate_error(x, y, error_type):
    """Calculate the normalised error of x relative to y as a percentage"""
    # remove NaNs
    notnan = ~np.isnan(x + y)
    abserror = LA.norm(x[notnan] - y[notnan])
    if abserror < 1e-10:
        return np.nan
    if error_type == "absolute":
        return abserror
    if error_type == "relative":
        return abserror / LA.norm(y[notnan])


def my_linear_interp(var):
    """Linearly interpolate/extrapolate data from cell centres to cell edges"""
    shifted = np.zeros((var.shape[0], var.shape[1] + 1))
    shifted[:, 0] = (3 * var[:, 0] - var[:, 1]) / 2
    shifted[:, 1:-1] = (var[:, 1:] + var[:, :-1]) / 2
    shifted[:, -1] = (3 * var[:, -1] - var[:, -2]) / 2
    return shifted


def datetime_to_hours(datetime, ref_datetime):
    """Convert a Timestamp to a number of hours elapsed since reference Timestamp"""
    return (datetime - ref_datetime).total_seconds() / 3600


def load_data(
    data_name,
    imei=None,
    dates=None,
    ignore_start=False,
    ignore_end=False,
    wt=1,
    n_coarse_points=0,
):
    """Load data as a list of dicts of lists"""
    datas = OrderedDict()
    if data_name == "March17":
        for name, value in dates.items():
            # Decompose dates
            (start, end) = value
            # Read the file (parsing dates)
            fname = "".join(
                ("inputs/telemetry_", imei, "_", start, "_", end, ".csv")
            )
            df = pd.read_csv(
                fname,
                header=1,
                parse_dates=[0],
                names=["time", "voltage", "current", "temperature"],
            )

            # Get rid of NaN values in current or voltage (equivalently voltage)
            df.dropna(subset=["voltage", "current"], inplace=True)
            # Drop initial values and/or end values (or neither)
            if ignore_start:
                df.drop(
                    df[(df.voltage > 12) & (df.current < 0.25)].index,
                    inplace=True,
                )
            if ignore_end:
                df.drop(
                    df[(df.voltage < 12) & (df.current < 0.25)].index,
                    inplace=True,
                )
            # Convert datetime to seconds since start
            df.time = df.time.apply(
                datetime_to_hours, args=(df.time[df.index[0]],)
            )
            # Sort by time
            df = df.sort_values(by=["time"])
            # Convert to dictionary of lists
            df_dict = df.to_dict("list")
            # Convert to dictionary of numpy arrays
            datas[name] = {k: np.asarray(v) for (k, v) in df_dict.items()}

            # Coarsen data if required
            if n_coarse_points:  # If n_coarse_points=0 (deault), do nothing
                # Define fine time and coarse time
                fine_time = datas[name]["time"]
                coarse_time = np.linspace(
                    fine_time[0], fine_time[-1], n_coarse_points
                )
                # Interpolate to coarse time
                datas[name]["current"] = np.interp(
                    coarse_time, fine_time, datas[name]["current"]
                )
                datas[name]["voltage"] = np.interp(
                    coarse_time, fine_time, datas[name]["voltage"]
                )
                datas[name]["time"] = coarse_time

            # Add weights - no weight for the initial relaxation then more weight at the end
            weights = np.ones(datas[name]["voltage"].shape)
            # weights[(datas[name]['voltage'] < 12) &
            #         (datas[name]['current'] < 0.25)] = 0
            weights[-1] = wt
            datas[name]["weights"] = weights

    elif data_name == "Nov17":
        fname = (
            "inputs/lead_acid_GITT_C-20_rest_2h_GEIS_100mA_10mHz_10kHz_"
            + imei
            + ".csv"
        )

        # Get headers (maybe not the most efficient way ...)
        with open(fname, "rb") as f:
            headers = f.read().splitlines()[99]
            headers = str(headers)[2:-4].split("\\t")

        # Remove units from headers
        headers = [s.split("/", 1)[0] for s in headers]

        # Read file
        df = pd.read_csv(fname, header=97, sep="\t", names=headers)
        # Convert current from mA to A and switch sign
        df.I = -df.I / 1e3
        # Convert time from seconds to hours
        df.time /= 3600
        # Change column names
        df = df.rename(columns={"Ewe": "voltage", "I": "current"})

        # Discharge only, no GEIS
        df_discharge_clean = df[(df["Ns"].isin([4, 5]))]
        mintime = df_discharge_clean.time[df_discharge_clean.index[0]]
        df_discharge_clean.time -= mintime
        # Convert to dictionary of lists
        df_dict = df_discharge_clean.to_dict("list")
        # Convert to dictionary of numpy arrays
        datas["all"] = {k: np.asarray(v) for (k, v) in df_dict.items()}
        datas["all"]["weights"] = np.ones(datas["all"]["time"].shape)
    return datas


def make_results_table(filename, times):
    """Print CPU times and speed-up to a text file, to be used as a LaTeX Table"""
    # Add directories to filename
    filename = "out/tables/" + filename + ".txt"

    # Read keys
    Crates = list(times.keys())
    methods = [
        k
        for k in sorted(
            times[Crates[0]], key=times[Crates[0]].get, reverse=True
        )
    ]

    with open(filename, "w") as text_file:
        # Table set-up
        text_file.write(
            ("\\begin{{tabular}}{{|c|{}}}\n").format("cc|" * len(Crates))
        )
        text_file.write("\hline\n")
        # Current row
        for Crate in Crates:
            text_file.write("& \multicolumn{{2}}{{c|}}{{{}C}}".format(Crate))
        text_file.write(r"\\" + "\n")
        text_file.write("\cline{{2-{}}}\n".format(2 * len(Crates) + 1))
        # Defining the columns
        text_file.write(
            ("Solution Method {}" + r"\\" + "\n").format(
                "& Time & Speed-up " * len(Crates)
            )
        )
        text_file.write("\hline\n")
        # Fill in values method-by-method
        for i, method in enumerate(methods):
            text_file.write(method)
            # Current-by-current: v is a dict with key: method and value: time
            for v in times.values():
                if method == "Numerical":
                    text_file.write(" & {:.3g} & {} ".format(v[method], "-"))
                else:
                    text_file.write(
                        " & {:.3f} & {:d} ".format(
                            v[method], round(v["Numerical"] / v[method])
                        )
                    )
            text_file.write(r"\\" + "\n")
        # End
        text_file.write("\hline\n\end{tabular}")
    #
    # with open(filename, "r") as text_file:
    #     print(text_file.read())


def save_data(
    datafile, x, y, callfile, linestyle="", extra_feature="", n_coarse_points=0
):
    """Save data in table form for plotting using pgfplots"""
    datafile = "out/data/" + datafile + ".dat"
    # Coarsen data if required
    if n_coarse_points:  # If n_coarse_points=0 (deault), do nothing
        # Define target x values
        xvals = np.linspace(x[0], x[-1], n_coarse_points)
        # Make sure y is a 1D object
        if len(y.shape) > 1:
            y = y[:, 0]
        # Interpolate to coarse time
        y = np.interp(xvals, x, y)
        x = xvals

    # Save data to datafile
    np.savetxt(datafile, np.column_stack((x, y)))

    # Add line to pgfplots callfile
    callfile.write(
        ("\\addplot{} table{{code/{}}}{};\n").format(
            linestyle, datafile, extra_feature
        )
    )
