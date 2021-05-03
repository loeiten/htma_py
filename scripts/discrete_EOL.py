import numpy as np
from src.regret_matrix import (
    evolve_one_probability_from_max_uncertainty,
    get_regret_matrix,
)


def main():
    payoff_ = np.array([[40, 45, 5], [70, 30, -13], [53, 45, -5]])
    regret_ = get_regret_matrix(payoff_)
    prob_distributions = evolve_one_probability_from_max_uncertainty(
        payoff_.shape[0], 0, 5
    )
    print("\nPayoff matrix")
    print(payoff_)
    print("\nRegret matrix")
    print(regret_)
    print("\nProbability distribution")
    print(prob_distributions)
    print("\nExpected opportunity loss")
    print(regret_ @ prob_distributions)


if __name__ == "__main__":
    main()
