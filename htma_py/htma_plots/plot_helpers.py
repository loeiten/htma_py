"""Helper functions for plotting routines."""

from typing import Any, Dict, Optional

from matplotlib import axes, figure, ticker

from htma_py.utils.paths import get_plot_path


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
    tick_string = f"{val:.{precision}g}"
    check_for_period = False
    # Special case if 0.000x or 0.00x
    if "0.000" in tick_string:
        check_for_period = True
        tick_string = tick_string.replace("0.000", "")
        if tick_string[1] == "-":
            tick_string = f"{tick_string[0:3]}.{tick_string[3:]}e-03"
        else:
            tick_string = f"{tick_string[0:2]}.{tick_string[2:]}e-03"
    elif "0.00" in tick_string:
        check_for_period = True
        tick_string = tick_string.replace("0.00", "")
        if tick_string[1] == "-":
            tick_string = f"{tick_string[0:3]}.{tick_string[3:]}e-02"
        else:
            tick_string = f"{tick_string[0:2]}.{tick_string[2:]}e-02"
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


def set_labels_and_legends(
    axis: axes.Axes, plot_properties: Dict[str, Any]
) -> axes.Axes:
    """
    Set the labels and legends.

    Parameters
    ----------
    axis : axes.Axes
        The axis object to modify
    plot_properties : dict of str
        The plot properties consisting of
        - x_label : str
            - Name to put on the x-axis
        - y_label : str
            - Name to put on the x-axis
        - label : str
            - Label to put on the legend

    Returns
    -------
    axis : axes.Axes
        The modified axis object
    """
    axis.set_xlabel(plot_properties["x_label"])
    axis.set_ylabel(plot_properties["y_label"])
    axis = make_axis_pretty(axis)
    if plot_properties["label"] is not None:
        make_legend(axis)
    return axis


def make_axis_pretty(axis: axes.Axes, rotation: int = 45) -> axes.Axes:
    """
    Make the axis pretty.

    Parameters
    ----------
    axis : axes.Axes
        The axis object to beatify
    rotation : int
        Degrees of rotation

    Returns
    -------
    axis : axes.Axes
        The beatified axis object to beatify
    """
    # https://stackoverflow.com/a/63755285/2786884
    label_format = "{:,.0f}"
    x_ticks = axis.get_xticks().tolist()
    axis.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    axis.set_xticklabels([label_format.format(x) for x in x_ticks], rotation=rotation)
    axis.get_yaxis().set_major_formatter(ticker.FuncFormatter(plot_number_formatter))
    axis.grid(True)
    return axis
