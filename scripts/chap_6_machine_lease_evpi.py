from src.continuous_evpi import get_evpi
from src.distribution import Gaussian


def main():
    ms = Gaussian(10, 20)
    ls = Gaussian(-2, 8)
    rms = Gaussian(3, 9)
    pl = Gaussian(1.5e4, 3.5e4)
    n = 2000
    threshold_payoff = 4e5

    ms_sample = ms.sample(n)
    ls_sample = ls.sample(n)
    rms_sample = rms.sample(n)
    pl_sample = pl.sample(n)

    ms_mean = ms.mean
    ls_mean = ls.mean
    rms_mean = rms.mean
    pl_mean = pl.mean

    payoff = (ms_sample + ls_sample + rms_sample) * pl_sample
    overall_evpi = get_evpi(payoff, threshold_payoff, n)
    print(
        f"The overall EVPI is {overall_evpi:.0f}\tOverall threshold: {threshold_payoff:.0f}"
    )

    print(
        "\n" + "=" * 80 + "\nUsing method 1 where all variables are kept at the mean:"
    )
    payoff_ms = (ms_sample + ls_mean + rms_mean) * pl_mean
    threshold_ms = (threshold_payoff / (pl_mean)) - ls_mean - rms_mean
    evpi_ms = get_evpi(payoff_ms, threshold_payoff, n)
    print(f"MS EVPI is {evpi_ms:.0f}\tThreshold for mean MS: {threshold_ms:.0f}")

    payoff_ls = (ms_mean + ls_sample + rms_mean) * pl_mean
    threshold_ls = (threshold_payoff / (pl_mean)) - ms_mean - rms_mean
    evpi_ls = get_evpi(payoff_ls, threshold_payoff, n)
    print(f"LS EVPI is {evpi_ls:.0f}\tThreshold for mean LS: {threshold_ls:.0f}")

    payoff_rms = (ms_mean + ls_mean + rms_sample) * pl_mean
    threshold_rms = (threshold_payoff / (pl_mean)) - ms_mean - ls_mean
    evpi_rms = get_evpi(payoff_rms, threshold_payoff, n)
    print(f"RMS EVPI is {evpi_rms:.0f}\tThreshold for mean RMS: {threshold_rms:.0f}")

    payoff_pl = (ms_mean + ls_mean + rms_mean) * pl_sample
    threshold_pl = threshold_payoff / ((ms_mean + ls_mean + rms_mean))
    evpi_pl = get_evpi(payoff_pl, threshold_payoff, n)
    print(f"PL EVPI is {evpi_pl:.0f}\tThreshold for mean PL: {threshold_pl:.0f}")

    print(
        "\n"
        + "=" * 80
        + "\nUsing method 2 where only one variable is assumed known (kept at mean):"
    )
    payoff_ms = (ms_mean + ls_sample + rms_sample) * pl_sample
    evpi_ms = overall_evpi - get_evpi(payoff_ms, threshold_payoff, n)
    print(f"Individual MS EVPI: {evpi_ms:.0f}")

    payoff_ls = (ms_sample + ls_mean + rms_sample) * pl_sample
    evpi_ls = overall_evpi - get_evpi(payoff_ls, threshold_payoff, n)
    print(f"Individual LS EVPI: {evpi_ls:.0f}")

    payoff_rms = (ms_sample + ls_sample + rms_mean) * pl_sample
    evpi_rms = overall_evpi - get_evpi(payoff_rms, threshold_payoff, n)
    print(f"Individual RMS EVPI: {evpi_rms:.0f}")

    payoff_pl = (ms_sample + ls_sample + rms_sample) * pl_mean
    evpi_pl = overall_evpi - get_evpi(payoff_pl, threshold_payoff, n)
    print(f"Individual PL EVPI: {evpi_pl:.0f}")


if __name__ == "__main__":
    main()
