"""Contains functions for plotting."""

from typing import Optional, Tuple

import numpy as np
from matplotlib import axes, figure
from matplotlib import pyplot as plt
from matplotlib import ticker


# pylint: disable=useless-type-doc
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


def plot_histogram(
    samples_from_distribution: np.array, x_label: str
) -> Tuple[figure.Figure, axes.Axes]:
    """
    Plot histogram from samples.

    Parameters
    ----------
    samples_from_distribution : np.ndarray
        Size: (N,)
        The samples drawn from the distribution
    x_label : str
        Name to put on the x-axis

    Returns
    -------
    fig : figure.Figure
        The figure object
    fig : axes.Axes
        The axis object
    """
    plt.rc("figure", dpi=300)
    fig = plt.figure(figsize=(5, 3))
    axis = fig.add_subplot(111)
    axis.hist(samples_from_distribution, bins=100, density=True, alpha=0.75)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Probability [fraction]")
    # https://stackoverflow.com/a/63755285/2786884
    label_format = "{:,.0f}"
    x_ticks = axis.get_xticks().tolist()
    axis.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    axis.set_xticklabels([label_format.format(x) for x in x_ticks], rotation=45)
    axis.get_yaxis().set_major_formatter(ticker.FuncFormatter(plot_number_formatter))
    axis.grid(True)
    return fig, axis
