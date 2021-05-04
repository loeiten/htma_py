"""Replication of HTMA_3rd_Ch6-2.xlsx."""


import numpy as np

from htma_py.distribution import Gaussian


def main() -> None:
    """Calculate the risk of breakeven for the machine lease example."""
    maintenance_savings_dist = Gaussian(10, 20)
    labor_savings_dist = Gaussian(-2, 8)
    raw_materials_savings_dist = Gaussian(3, 9)
    production_level_dist = Gaussian(1.5e4, 3.5e4)
    n_samples = 15170
    # NOTE: We are here doing Monte Carlo
    savings = (
        maintenance_savings_dist.sample(n_samples)
        + labor_savings_dist.sample(n_samples)
        + raw_materials_savings_dist.sample(n_samples)
    ) * production_level_dist.sample(n_samples)
    risk_pct = 100 * np.where(savings < 4e5)[0].size / n_samples
    print(
        f"The risk that the breakeven will not be met "
        f"(the new machine lease is a loss) = {risk_pct:.0f} %"
    )


if __name__ == "__main__":
    main()
