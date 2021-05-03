from matplotlib import pyplot as plt
from scipy.stats import binom


def main():
    target = 10
    probability_of_single_hit = 0.9  # This is the calibration we are aiming for
    hits = list(range(11))
    # Chance of hits given target and probability of single hit
    # I.e.: If x was the case, what is the chance of this observation
    chance = [100*binom.pmf(hit, target, probability_of_single_hit) for hit in hits]

    plt.rc("figure", dpi=300)
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.plot(hits, chance, 'o')
    ax.set_xlabel("Hits out of 10 possible")
    ax.set_ylabel("Probability [%]")
    ax.set_title("Chance of being 90 % calibrated")
    ax.grid(True)
    fig.savefig("prob_calibrated.png", transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    main()
