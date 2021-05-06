"""Contains functions for plotting."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from matplotlib import axes, figure
from matplotlib import pyplot as plt
from matplotlib import ticker

# pylint: disable=useless-type-doc
from htma_py.utils.paths import get_plot_path

plt.rc("figure", dpi=300)


def plot_histogram(
    samples_from_distribution: np.array, x_label: str
) -> Tuple[figure.Figure, axes.Axes, Dict[str, Any]]:
    """
    Plot histogram from samples.

    Parameters
    ----------
    samples_from_distribution : np.array
        Size: (N,)
        The samples drawn from the distribution
    x_label : str
        Name to put on the x-axis

    Returns
    -------
    fig : figure.Figure
        The figure object
    axis : axes.Axes
        The axis object
    histogram_output: dict of str, Any
        The output from .hist
        - counts are the counts of hits in each bin
        - bins are the edges of each bin
        - patches are the corresponding patches object
    """
    fig = plt.figure(figsize=(5, 3))
    axis = fig.add_subplot(111)
    counts, bins, patches = axis.hist(
        samples_from_distribution, bins=100, density=False, alpha=0.75
    )
    histogram_output = {"counts": counts, "bins": bins, "patches": patches}
    axis.set_xlabel(x_label)
    axis.set_ylabel("Number of hits in a bin")
    axis = make_axis_pretty(axis)
    return fig, axis, histogram_output


def plot_loss(
    x_min: float, x_max: float, lin_loss_array: np.array, x_label: str
) -> Tuple[figure.Figure, axes.Axes]:
    """
    Plot the loss function.

    Parameters
    ----------
    x_min : float
        The minimum value for the dependant variable
    x_max : float
        The maximum value for the dependant variable
    lin_loss_array : np.array
        The loss array
    x_label : str
        Name to put on the x-axis

    Returns
    -------
    fig : figure.Figure
        The figure object
    axis : axes.Axes
        The axis object
    """
    x_var = np.linspace(x_min, x_max, lin_loss_array.size)

    fig = plt.figure(figsize=(5, 3))
    axis = fig.add_subplot(111)

    axis.plot(x_var, lin_loss_array, color="red", label="Loss function")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Monetary loss [$]")
    axis = make_axis_pretty(axis)

    # Make legend
    make_legend(axis)

    return fig, axis


def make_legend(axis: axes.Axes) -> None:
    """
    Make the legend.

    Parameters
    ----------
    axis : axes.Axes
        The axis object
    """
    legend = axis.legend(loc="best", fancybox=True, numpoints=1)
    legend.get_frame().set_alpha(0.5)


def plot_bar(
    x_var: np.array, y_var: np.array, plot_properties: Optional[Dict[str, Any]] = None
) -> Tuple[figure.Figure, axes.Axes]:
    """
    Plot a bar plot.

    Parameters
    ----------
    x_var : np.array
        The minimum value for the dependant variable
    y_var : np.array
        The maximum value for the dependant variable
    plot_properties : dict of str
        The plot properties consisting of
        - x_label : str
            - Name to put on the x-axis
        - y_label : str
            - Name to put on the x-axis
        - label : str
            - Label to put on the legend
        - step : int
            - Number of the steps used when plotting a slice of the data
              step = 1 means that the original data will be plotted
              step = 2 means that every second data point will be plotted etc
        - color : str
            - Color to use on the bar
        - width : float
            - Width of the bars

    Returns
    -------
    fig : figure.Figure
        The figure object
    axis : axes.Axes
        The axis object
    """
    if plot_properties is None:
        plot_properties = dict()
    if "label" not in plot_properties.keys():
        plot_properties["label"] = None
    if "step" not in plot_properties.keys():
        plot_properties["step"] = 1
    if "color" not in plot_properties.keys():
        plot_properties["color"] = "#1f77b4"
    if "width" not in plot_properties.keys():
        plot_properties["width"] = 0.1

    fig = plt.figure(figsize=(5, 3))
    axis = fig.add_subplot(111)

    bars = axis.bar(
        x_var[:: plot_properties["step"]],
        y_var[:: plot_properties["step"]],
        color=plot_properties["color"],
        label=plot_properties["label"],
        width=plot_properties["width"],
        alpha=0.75,
    )
    for patch in bars:
        patch.set_color(plot_properties["color"])
    axis.set_xlabel(plot_properties["x_label"])
    axis.set_ylabel(plot_properties["y_label"])
    axis = make_axis_pretty(axis)

    if plot_properties["label"] is not None:
        # Make legend
        legend = axis.legend(loc="best", fancybox=True, numpoints=1)
        legend.get_frame().set_alpha(0.5)

    return fig, axis


def save_plot(fig: figure.Figure, name: str) -> None:
    """
    Save the figure.

    Parameters
    ----------
    fig : figure.Figure
        The figure to save
    name : str
        The filename to use
    """
    save_path = get_plot_path().joinpath(name).absolute()
    # Save
    fig.savefig(
        save_path,
        transparent=True,
        bbox_inches="tight",
    )
    print(f"Saved plot to {save_path}")


def plot_number_formatter(val: float, _: Optional[float], precision: int = 3) -> str:
    """
    Format numbers in the plot.

    Parameters
    ----------
    val : float
        The value.
    _ : [None | float]
        The position (needed as input from FuncFormatter).
    precision : int
        Precision of the tick

    Returns
    -------
    tick_string : str
        The string to use for the tick
    """
    tick_string = "${{:.{}g}}".format(precision).format(val)
    check_for_period = False
    # Special case if 0.000x or 0.00x
    if "0.000" in tick_string:
        check_for_period = True
        tick_string = tick_string.replace("0.000", "")
        if tick_string[1] == "-":
            tick_string = "{}.{}e-03".format(tick_string[0:3], tick_string[3:])
        else:
            tick_string = "{}.{}e-03".format(tick_string[0:2], tick_string[2:])
    elif "0.00" in tick_string:
        check_for_period = True
        tick_string = tick_string.replace("0.00", "")
        if tick_string[1] == "-":
            tick_string = "{}.{}e-02".format(tick_string[0:3], tick_string[3:])
        else:
            tick_string = "{}.{}e-02".format(tick_string[0:2], tick_string[2:])
    if check_for_period:
        # The last character before e-0x should not be a period
        if len(tick_string) > 5 and tick_string[-5] == ".":
            tick_string = tick_string.replace(".", "")
    if "e+" in tick_string:
        tick_string = tick_string.replace("e+0", r"\cdot 10^{")
        tick_string = tick_string.replace("e+", r"\cdot 10^{")
        tick_string += "}$"
    elif "e-" in tick_string:
        tick_string = tick_string.replace("e-0", r"\cdot 10^{-")
        tick_string = tick_string.replace("e-", r"\cdot 10^{-")
        tick_string += "}$"
    else:
        tick_string += "$"

    return tick_string


def make_axis_pretty(axis: axes.Axes) -> axes.Axes:
    """
    Make the axis pretty.

    Parameters
    ----------
    axis : axes.Axes
        The axis object to beatify

    Returns
    -------
    axis : axes.Axes
        The beatified axis object to beatify
    """
    # https://stackoverflow.com/a/63755285/2786884
    label_format = "{:,.0f}"
    x_ticks = axis.get_xticks().tolist()
    axis.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    axis.set_xticklabels([label_format.format(x) for x in x_ticks], rotation=45)
    axis.get_yaxis().set_major_formatter(ticker.FuncFormatter(plot_number_formatter))
    axis.grid(True)
    return axis
