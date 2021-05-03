import numpy as np


def get_regret_matrix(payoff):
    regret = np.full(payoff.shape, -1)
    for col_nr, col in enumerate(payoff.T):
        regret[:, col_nr] = col.max() - col
    return regret


def evolve_one_probability_from_max_uncertainty(choices, choice_to_evolve=0, n_prob_dist=100):
    result = np.zeros((choices, n_prob_dist))
    for prob_dist_nr in range(choices):
        result[prob_dist_nr, :] = np.linspace(1/choices, 1 if prob_dist_nr == choice_to_evolve else 0, n_prob_dist)
    return result

