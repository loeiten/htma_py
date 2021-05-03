"""Contains methods which return common paths."""


import configparser
import time
from pathlib import Path
from typing import Optional


def get_FIXME_package_path() -> Path:
    """
    Return the absolute path to the FIXME package.

    Returns
    -------
    Path
        The path to the root directory
    """
    return Path(__file__).absolute().parents[1]


def get_config_path() -> Path:
    """
    Return the absolute path to the configurations.

    Returns
    -------
    Path
        The path to the configuration directory
    """
    return get_FIXME_package_path().joinpath("config")


def get_logger_config_path() -> Path:
    """
    Return the absolute path to the logger configuration.

    Returns
    -------
    Path
        The path to the logger configuration file
    """
    return get_config_path().joinpath("logging_config.yaml")


def get_FIXME_config_path() -> Path:
    """
    Return the absolute path to the FIXME configuration.

    Returns
    -------
    Path
        The path to the FIXME configuration file
    """
    return get_config_path().joinpath("FIXME.ini")


def get_FIXME_configuration() -> configparser.ConfigParser:
    """
    Return the FIXME configuration.

    Returns
    -------
    config : configparser.ConfigParser
        The configuration of FIXME
    """
    config = configparser.ConfigParser()
    config.read(get_FIXME_config_path())
    return config


def get_log_file_directory() -> Path:
    """
    Return the log_file directory.

    Returns
    -------
    log_file_directory : Path
        Path to the log_file directory
    """
    config = get_FIXME_configuration()
    path_str = config["log"]["directory"]
    if path_str.lower() == "none":
        log_file_dir = get_FIXME_package_path().parent.joinpath("logs")
    else:
        log_file_dir = Path(path_str)

    log_file_dir.mkdir(exist_ok=True, parents=True)
    return log_file_dir


def get_log_file_path(
    log_file_dir: Optional[Path] = None, name: Optional[str] = None
) -> Path:
    """
    Return the absolute path to the log file path.

    Parameters
    ----------
    log_file_dir : Path or None
        Path to the log file directory
        If None, default log file directory will be used
    name : str or None
        Name of the log file
        If None, current date will be used

    Returns
    -------
    log_file_path : Path
        The path to the log file
    """
    if log_file_dir is None:
        log_file_dir = get_log_file_directory()
    if name is None:
        name = time.strftime("%Y%m%d.log")
    log_file_path = log_file_dir.joinpath(name)

    return log_file_path
