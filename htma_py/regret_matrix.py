"""Contains functions to work with regret (aka. opportunity loss) matrices."""

import numpy as np


def get_regret_matrix(revenue_matrix: np.array) -> np.array:
    """
    Return the regret matrix.

    Parameters
    ----------
    revenue_matrix : np.array
        Shape: (choices, outcomes)
        The revenue_matrix matrix

    Returns
    -------
    regret_matrix : np.array
        Shape: (choices, outcomes)
        The corresponding regret matrix
    """
    regret_matrix = np.full(revenue_matrix.shape, -1)
    for col_nr, col in enumerate(revenue_matrix.T):
        regret_matrix[:, col_nr] = col.max() - col
    return regret_matrix


def evolve_one_probability_from_max_uncertainty(
    n_choices: int, choice_index_to_evolve: int = 0, n_evolves: int = 100
) -> np.array:
    """
    Evolve n discrete choices from max uncertainty to max certainty.

    Parameters
    ----------
    n_choices : int
        Number of choices
    choice_index_to_evolve : int
        Which index (i.e. choice) of choice to evolve to full certainty
    n_evolves : int
        Number of evolves until full certainty

    Returns
    -------
    result : np.array
        Shape: (n_choices, n_evolves)
        The matrix containing the evolutions
    """
    result = np.zeros((n_choices, n_evolves))
    for prob_dist_nr in range(n_choices):
        result[prob_dist_nr, :] = np.linspace(
            1 / n_choices, 1 if prob_dist_nr == choice_index_to_evolve else 0, n_evolves
        )
    return result
