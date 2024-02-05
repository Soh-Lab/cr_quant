"""Collection of plotting functions."""
from __future__ import annotations

import os

import numpy as np
from matplotlib import pyplot as plt, colors


def apply_paper_formatting(font_size):
    """
    Configures matplotlib to use prettier formatting settings.

    :param font_size: font size to be used in all plots.
    """
    # sans-serif fonts
    #plt.rcParams.update({'font.size': font_size, 'font.family': 'Avenir',
                         # 'mathtext.fontset': 'stixsans',
    # serif fonts
    plt.rcParams.update({'font.size': font_size, 'font.family': 'serif',
                         'mathtext.fontset': 'dejavuserif',
                         'xtick.labelsize': 15, 'ytick.labelsize': 15})
    SMALL_SIZE = 20
    MEDIUM_SIZE = 25
    BIGGER_SIZE = 30
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def save_figure(fig, path):
    """
    Saves the figure at the specified path.

    :param fig: Figure to be saved.
    :param path: Path to save the figure at.
    """
    directory = os.path.dirname(path)
    # Check if the directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    filetype = os.path.splitext(path)[1].replace('.', '')
    fig.savefig(path, format=filetype, dpi=300)


def plot_estimate_performance(estimate: np.ndarray,
                              true_conc: np.ndarray,
                              ax: None | plt.Axes = None):
    """
    Plot the performance of an estimate vs the true target concentrations.

    2022-12-09 Linus A. Hein, adapted from code by Sharon S. Newman.

    :param estimate: (...) numpy array containing the estimates.
    :param true_conc: (...) numpy array containing the ground truth.
    :param ax: (optional) Axes object to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    assert estimate.shape == true_conc.shape
    estimate = estimate.reshape((estimate.shape[0], -1))
    true_conc = true_conc.reshape((true_conc.shape[0], -1))
    with np.errstate(divide='ignore', invalid='ignore'):
        error = np.log10(estimate) - np.log10(true_conc)
    error[np.isnan(error)] = 1000  # set ridiculously high value to enable plotting

    im = ax.imshow(error.T, cmap='Spectral', vmin=-2, vmax=2)
    ax.set_title('Log-Estimation error')
    ax.set_yticks(np.arange(true_conc.shape[1]),
                  labels=[f'Target {i + 1}' for i in range(true_conc.shape[1])])
    plt.colorbar(im, ax=ax, location='bottom', orientation='horizontal')


def plot_lower_upper_performance(lower_upper_bounds: np.ndarray,
                                 true_conc: np.ndarray):
    """
    Plot the performance of the lower/upper bound estimation.

    2022-12-09 Linus A. Hein.

    :param lower_upper_bounds: (n, 2, ...) lower and upper bounds on all the target molecules.
    :param true_conc: (n, ...) true concentrations of all target concentrations.
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey='all')
    bounds_shape = lower_upper_bounds.shape
    true_conc_shape = true_conc.shape

    assert bounds_shape[0] == true_conc_shape[0]  # ensure that n is same between the two arrays
    assert bounds_shape[2:] == true_conc_shape[1:]  # ensure the dimensionality is matched correctly

    lower_upper_bounds = lower_upper_bounds.reshape((bounds_shape[0], bounds_shape[1], -1))
    true_conc = true_conc.reshape((true_conc_shape[0], -1))
    lower_bound = lower_upper_bounds[:, 0, :]
    upper_bound = lower_upper_bounds[:, 1, :]

    # plot error
    with np.errstate(divide='ignore', invalid='ignore'):
        # geometric mean may not lie inside the constraint set
        # geom_mean = np.exp(0.5 * (np.log(lower_bound) + np.log(upper_bound)))
        mean = 0.5 * (lower_bound + upper_bound)
    plot_estimate_performance(mean, true_conc, axs[0])

    # plot confidence
    with np.errstate(divide='ignore', invalid='ignore'):
        confidence = np.log10(upper_bound) - np.log10(lower_bound)
    confidence[np.isinf(confidence)] = 1000  # set ridiculously high value to enable plotting
    confidence[np.isnan(confidence)] = 1000  # set ridiculously high value to enable plotting
    im = axs[1].imshow(confidence.T, cmap='plasma', vmin=0, vmax=3)
    axs[1].set_title('Log-Confidence in Estimate')
    plt.colorbar(im, ax=axs[1], location='bottom', orientation='horizontal')

    fig.tight_layout()
    plt.show()


def plot_2d_boundedness_results(target_concs: np.ndarray,
                                boundedness: np.ndarray,
                                scaling: str = 'log',
                                plotting_method: str = 'contourf') -> list[plt.Axes]:
    """
    Plots boundedness of two target molecules. Colormap:
        - 0: (green)    lower bound on target exists.
        - 1: (blue)     no lower bound on target exists
        - 2: (red)      neither lower bound nor upper bound on the target exists

    2022-12-10 Linus A. Hein.

    :param target_concs: (2, t2, t1) 2d grid of target concentrations (result of np.meshgrid).
    :param boundedness: (2, 1, t2, t1) 2d grid of boundedness.
    :param scaling: Axis scaling. Possible values: "linear", "log", "symlog", "logit".
    :param plotting_method: ('contourf', or 'pcolormesh') matplotlib plotting method to use during
        plotting.
    :return: Axes objects on which the binding curves were plotted.
    """
    fig, axs = plt.subplots(1, 2, sharex='all', sharey='all')
    assert target_concs.shape[0] == 2

    # change the colormap to some custom values
    colormap = colors.ListedColormap(
        [colors.hex2color('#51FF4A'),
         colors.hex2color('#CDC9FF'),
         colors.hex2color('#FFDDBF')])

    for i in range(2):
        if plotting_method == 'contourf':
            axs[i].contourf(target_concs[0], target_concs[1], boundedness[i, 0],
                            cmap=colormap, vmin=0, vmax=2.0, shading='nearest')
        elif plotting_method == 'pcolormesh':
            axs[i].pcolormesh(target_concs[0], target_concs[1], boundedness[i, 0],
                              cmap=colormap, vmin=0, vmax=2.0, shading='nearest')
        else:
            raise Exception(f'plotting_method {plotting_method} is not in list ["contourf", '
                            f'"pcolormesh"]')
        axs[i].set_aspect('equal')
        axs[i].grid()
        axs[i].set_xscale(scaling)
        axs[i].set_yscale(scaling)
        axs[i].set_xlim([np.min(target_concs[0]), np.max(target_concs[0])])
        axs[i].set_ylim([np.min(target_concs[1]), np.max(target_concs[1])])
        axs[i].set_xlabel("$T_1$")
        axs[i].set_ylabel("$T_2$")
        axs[i].set_title(f'Target {i + 1}')
    fig.set_size_inches(13, 6)
    return axs


def plot_2d_fields(target_concs: np.ndarray,
                   fields: np.ndarray,
                   field_name: str = '',
                   axs: list[plt.Axes] = None,
                   scaling: str = 'log',
                   limits: tuple[float, float] = (0.0, 1.0),
                   colorbar_scale: float = 0.6) -> list[plt.Axes]:
    """
    Plot readout values on 2D map.

    2022-12-10 Linus A. Hein.

    :param target_concs: (2, t2, t1) 2d grid of target concentrations (result of meshgrid).
    :param field_name: String to be displayed on top of each axis.
    :param fields: (k, t2, t1) readouts produced by the m affinity reagents.
    :param field_name: name of the k axis in fields (Example: "Affinity Reagent").
    :param axs: (optional) List of k axes objects to plot on. Should have length of at least m.
    :param scaling: Axis scaling. Possible values: "linear", "log" (default), "symlog", "logit".
    :param limits: tuple of limits on the readout values (minimum, maximum).
        Default value (0.0, 1.0).
    :param colorbar_scale: Scale of the color bar.
    :return: List of k Axes objects on which the binding curves were plotted.
    """
    m_reagents = fields.shape[0]
    if axs is None:
        fig, axs = plt.subplots(1, m_reagents, sharey='all', sharex='all', layout='constrained')
    v = np.linspace(limits[0], limits[1], 7, endpoint=True)
    for i in range(m_reagents):
        im = axs[i].contourf(target_concs[0], target_concs[1], fields[i], v, extend='both',
                             cmap='viridis', vmin=limits[0], vmax=limits[1])
        if i == m_reagents - 1:
            plt.colorbar(im, orientation='vertical', ticks=v, shrink=colorbar_scale)
        # plt.colorbar(im, ax=axs[i], location='bottom', orientation='horizontal', ticks=v)
        axs[i].set_xlabel('$T_1$')
        axs[i].set_ylabel('$T_2$')
        axs[i].set_title(f'{field_name} {i + 1}')
        axs[i].grid()
        axs[i].set_xlim(np.min(target_concs[0]), np.max(target_concs[0]))
        axs[i].set_ylim(np.min(target_concs[1]), np.max(target_concs[1]))
        axs[i].set_xscale(scaling)
        axs[i].set_yscale(scaling)
    plt.gcf().set_size_inches(4 * m_reagents, 4.25)
    return axs


def plot_feasible_region(K_A_i: np.ndarray,
                         affine_bounds_i: np.ndarray,
                         ax: plt.Axes,
                         t1_limits: tuple[float, float],
                         scaling: str = 'log',
                         color: str = 'r'):
    """
    Plot feasible set onto existing axis object.

    2022-12-10 Linus A. Hein.

    :param K_A_i: (2) i-th row of K_A matrix with two target molecules.
    :param affine_bounds_i: (2) i-th row of affine lower/upper bounds.
    :param ax: Axes object to plot on.
    :param t1_limits: left and right limits for plotting. If scaling is "log", actual limits are
        [10^t1_limits[0], 10^t1_limits[1]]
    :param scaling: Axis scaling. Possible values: "linear", "log" (default).
    :param color: named matplotlib color of the shape that will be filled in.
    """
    if scaling == 'log':
        x = np.logspace(t1_limits[0], t1_limits[1], 1000)
    elif scaling == 'linear':
        x = np.linspace(t1_limits[0], t1_limits[1], 1000)
    else:
        raise Exception(f'scaling is {scaling}, which is not one of ["linear", "log"]')

    if affine_bounds_i[1] == np.infty:
        y_top = np.ones_like(x) * 1e10
    else:
        y_top = (affine_bounds_i[1] - K_A_i[0] * x) / K_A_i[1]

    if affine_bounds_i[0] <= 0.0:
        y_btm = np.ones_like(x) * 1e-10
    else:
        y_btm = (affine_bounds_i[0] - K_A_i[0] * x[::-1]) / K_A_i[1]

    ax.fill(np.concatenate([x, x[::-1]]), np.concatenate([y_top, y_btm]),
            alpha=0.8, color=color)


def plot_feasible_line(K_A_i: np.ndarray,
                       readout_value: float,
                       ax: plt.Axes,
                       t1_limits: tuple[float, float],
                       scaling: str = 'log',
                       color: str = 'r'):
    """
    Plot feasible set onto existing axis object.

    2022-12-10 Linus A. Hein.

    :param K_A_i: (2) i-th row of K_A matrix with two target molecules.
    :param readout_value: readout value for which to plot the line.
    :param ax: Axes object to plot on.
    :param t1_limits: left and right limits for plotting. If scaling is "log", actual limits are
        [10^t1_limits[0], 10^t1_limits[1]]
    :param scaling: Axis scaling. Possible values: "linear", "log" (default).
    :param color: named matplotlib color of the shape that will be filled in.
    """
    if scaling == 'log':
        x = np.logspace(t1_limits[0], t1_limits[1], 1000)
    elif scaling == 'linear':
        x = np.linspace(t1_limits[0], t1_limits[1], 1000)
    else:
        raise Exception(f'scaling is {scaling}, which is not one of ["linear", "log"]')

    y = (readout_value - K_A_i[0] * x) / K_A_i[1]

    ax.plot(x, y, color=color, linewidth=3)


def plot_ellipse(center, B, ax):
    """
    Plot an ellipse onto an axis object.

    :param center: (2) Coordinates of the center of the ellipse.
    :param B: (2, 2) Matrix describing the ellipsoid to be plotted. Edge of the ellipse is given by
        B @ u where u is an element of the set of all unit vectors.
    :param ax: Axes object to plot on.
    """
    n_points = 100
    rad_values = np.linspace(0, 2 * np.pi, n_points)
    u = np.zeros((2, n_points))
    u[0] = np.cos(rad_values)
    u[1] = np.sin(rad_values)
    coors = B @ u + center[:, np.newaxis]
    ax.fill(coors[0], coors[1], alpha=0.5, color='m')
