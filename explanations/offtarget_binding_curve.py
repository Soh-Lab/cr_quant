"""
Code to generate Fig 1b) in the paper.
"""

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from cr_utils.plotting import apply_paper_formatting, save_figure
from cr_utils.utils import get_readouts

if __name__ == '__main__':
    apply_paper_formatting(18)

    # define KD values
    K_D = np.array([[1.0, 10.0]]) * 1e-3
    K_A = 1.0 / K_D

    # Number of colors in the gradient
    num_colors = 5
    # Generate equally spaced steps between 0 and 1
    gradient_steps = np.linspace(0, 0.8, num_colors)
    # Choose the colormap
    color_map = mpl.cm.get_cmap('plasma')

    colors = [color_map(step) for step in gradient_steps]
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

    # plotting background
    m_reagents = K_D.shape[0]
    # create a 2D-log-meshgrid of target concentrations
    A, B = np.logspace(-7, 0, 1000), np.logspace(-4, 0, 5)
    AA, BB = np.meshgrid(A, B)

    target_concs = np.stack([AA, BB], axis=0)
    readouts = get_readouts(K_A, target_concs)
    for i in reversed(range(len(B))):
        plt.semilogx(A * K_A[0, 0], readouts[0, i],
                     label='$[T_2]\cdot K_A^{2} = 10^{' + f'{int(np.log10(B[i] * K_A[0, 1]))}' + '}$')
    # figure formatting
    plt.grid()
    plt.xlabel('$[T_1]\cdot K_A^{1}$')
    plt.ylabel('Normalized Signal')
    plt.legend(fontsize=13)
    plt.xlim(1e-3, 1e3)

    plt.gcf().set_size_inches(5.9, 4.5)
    plt.gcf().set_layout_engine('compressed')
    save_figure(plt.gcf(), '../output/amplitude.svg')
    plt.show()
