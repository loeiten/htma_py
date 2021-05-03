from matplotlib import pylab as plt


def plot_distribution(x_array, distribution):
    fig, ax = plt.subplots()
    ax.plot(x_array, distribution.pdf(x_array))
