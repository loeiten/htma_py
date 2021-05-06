"""Script plotting the chance of being 90 % calibrated (similar to exhibit 5.6)."""

from matplotlib import pyplot as plt
from scipy.stats import binom

from htma_py.plots import make_axis_pretty, save_plot


def main() -> None:
    """Save a plot of the chance of being 90 % calibrated given x/10 hits."""
    target = 10
    probability_of_single_hit = 0.9  # This is the calibration we are aiming for
    hits = list(range(11))
    # Chance of hits given target and probability of single hit
    # I.e.: If x was the case, what is the chance of this observation
    chance = [100 * binom.pmf(hit, target, probability_of_single_hit) for hit in hits]

    fig = plt.figure(figsize=(5, 3))
    axis = fig.add_subplot(111)
    axis.plot(hits, chance, "o")
    axis.set_xlabel("Hits out of 10 possible")
    axis.set_ylabel("Probability [%]")
    axis.set_title("Chance of being 90 % calibrated")
    make_axis_pretty(axis, rotation=0)
    save_plot(fig, "calibration_probability.png")


if __name__ == "__main__":
    main()
