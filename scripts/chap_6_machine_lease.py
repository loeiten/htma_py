"""Replication of HTMA_3rd_Ch6-2.xlsx"""


import numpy as np
from src.distribution import Gaussian


def main():
    ms = Gaussian(10, 20)
    ls = Gaussian(-2, 8)
    rms = Gaussian(3, 9)
    pl = Gaussian(1.5e4, 3.5e4)
    n = 15170
    # NOTE: We are here doing Monte Carlo
    savings = (ms.sample(n) + ls.sample(n) + rms.sample(n)) * pl.sample(n)
    risk_pct = 100 * np.where(savings < 4e5)[0].size / n
    print(
        f"The risk that the breakeven will not be met (the new machine lease is a loss) = {risk_pct:.0f} %"
    )


if __name__ == "__main__":
    main()
