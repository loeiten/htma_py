"""
Calculate discrete expected opportunity loss.

References
----------
https://www.youtube.com/watch?v=JehD3NcUKIY
"""

import numpy as np

from htma_py.regret_matrix import (
    evolve_one_probability_from_max_uncertainty,
    get_regret_matrix,
)


def main() -> None:
    """Calculate the expected opportunity loss."""
    payoff = np.array([[40, 45, 5], [70, 30, -13], [53, 45, -5]])
    regret = get_regret_matrix(payoff)
    prob_distributions = evolve_one_probability_from_max_uncertainty(
        payoff.shape[0], 0, 5
    )
    print("\nPayoff matrix")
    print(payoff)
    print("\nRegret matrix")
    print(regret)
    print("\nProbability distribution")
    print(prob_distributions)
    print("\nExpected opportunity loss")
    print(regret @ prob_distributions)


if __name__ == "__main__":
    main()
