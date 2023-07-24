import numpy as np
from matplotlib import pyplot as plt

from cr_utils.plotting import apply_paper_formatting


def plot_single_isotherm():
    T = np.logspace(-4, 4, 1000)
    fraction_bound = T / (1 + T)
    plt.plot(T, fraction_bound, color='k')


def plot_single_isotherm_readout(readout_center):
    plot_single_isotherm()

    # readout_center = 0.25
    T_center = readout_center / (1 - readout_center)

    y = np.array([readout_center, readout_center, -1])
    x = np.array([1e-4, T_center, T_center])
    plt.plot(x, y, color='r')


def plot_single_isotherm_range(readout_center, readout_min, readout_max):
    plot_single_isotherm_readout(readout_center)
    T_max = 1e5 if readout_max >= 1.0 else readout_max / (1 - readout_max)
    T_min = 1e-5 if readout_min <= 0 else readout_min / (1 - readout_min)

    x = np.array([1e-4, T_max, T_max, T_min, T_min, 1e-4])
    y = np.array([readout_max, readout_max, -1, -1, readout_min, readout_min])
    plt.fill(x, y, alpha=0.5)


def fix_labeling():
    plt.xlabel('$T\cdot K_A$')
    plt.xscale('log')
    plt.xticks([1e-4, 1e-2, 1e-0, 1e2, 1e4])
    plt.ylabel('Relative Signal')
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.xlim((1e-4, 1e4))
    plt.ylim((-0.05, 1.05))
    plt.grid()


if __name__ == '__main__':
    apply_paper_formatting(18)
    readout_value = 0.5
    delta = 0.05

    plot_single_isotherm_range(0.5, readout_value - delta, readout_value + delta)
    fix_labeling()
    plt.show()
