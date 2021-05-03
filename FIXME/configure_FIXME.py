"""Module for configuring FIXME."""


import logging
from pathlib import Path
from typing import Optional

import yaml

from FIXME.utils.logs import get_log_config, set_up_logger
from FIXME.utils.paths import (
    get_FIXME_config_path,
    get_FIXME_configuration,
    get_log_file_directory,
    get_logger_config_path,
)


def set_log_level(level: Optional[str] = None) -> None:
    """
    Set the log level.

    Parameters
    ----------
    level : None or str
        The logging level to use
        If None the caller will be prompted

    Raises
    ------
    ValueError
        If the level is not one of the possibilities
    """
    config = get_log_config()

    possibilities = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    if level is None:
        current_level = config["root"]["level"]

        possibilities_map = dict(enumerate(possibilities))
        question = (
            f"Please set the log level by entering a number:\n"
            f"   (empty input will reuse the current level "
            f"[{current_level}])\n"
        )
        for key, val in possibilities_map.items():
            question += f'{" "*3}({key}) - {val}\n'
        # Set an answer to start the wile loop
        answer_int = -1
        possibilities_keys = possibilities_map.keys()
        while answer_int not in possibilities_keys:
            answer_input = input(question)
            print(f"Your answered: '{answer_input}'")
            if answer_input != "":
                answer_int = int(answer_input)
            if answer_input == "":
                # Reverse the dict
                answer_int = list(possibilities_map.keys())[
                    list(possibilities_map.values()).index(current_level)
                ]
                break
        level = possibilities_map[answer_int]

    if level not in possibilities:
        msg = f"`level` in `set_log_level` must be one of " f"{possibilities}"
        raise ValueError(msg)

    print(f"Setting logging level to {level}")

    config["handlers"]["file_handler"]["level"] = level
    config["handlers"]["console_handler"]["level"] = level
    config["root"]["level"] = level

    with get_logger_config_path().open("w") as log_file:
        log_file.write(yaml.dump(config))

    set_up_logger(config)
    logging.debug("Logging level set to %s", level)
    print("")


def set_log_file_directory(log_dir: Optional[Path] = None) -> None:
    """
    Set the directory of the log files.

    Parameters
    ----------
    log_dir : None or Path
        The directory to keep the log files
        If None the caller will be prompted
    """
    config = get_FIXME_configuration()
    if log_dir is None:
        current_dir = get_log_file_directory()
        question = (
            f"Please enter the directory for log files:\n"
            f"Empty input will reuse the current directory "
            f"{current_dir}\n"
        )
        answer = input(question)
        print(f"Your answered: '{answer}'")
        if answer == "":
            config["log"]["directory"] = str(current_dir)
        else:
            config["log"]["directory"] = answer
    else:
        config["log"]["directory"] = str(log_dir)

    with get_FIXME_config_path().open("w") as configfile:
        config.write(configfile)

    print(f"Setting logging dir to {config['log']['directory']}")

    set_up_logger()
    logging.debug("Logging directory set to %s", config["log"]["directory"])
    print("")


if __name__ == "__main__":
    set_log_level()
    set_log_file_directory()
