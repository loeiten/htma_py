"""Contains methods which return common paths."""

from pathlib import Path


def get_root_path() -> Path:
    """
    Return the absolute path to the root directory.

    Returns
    -------
    Path
        The path to the root directory
    """
    return Path(__file__).absolute().parents[2]


def get_plot_path() -> Path:
    """
    Return the absolute path to the plot directory.

    Returns
    -------
    Path
        The path to the plot directory
    """
    root_dir = get_root_path()
    plot_dir = root_dir.joinpath("plots")
    if not plot_dir.is_dir():
        plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir
