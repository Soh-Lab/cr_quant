"""
Generates an animation to give intuition for how signal readout resolution works for a single-target
binder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# define relative and absolute error in readout
delta_abs = 0.05
delta_rel = 0.05

# Select whether to linearly sweep target concentrations or signal readout values
LINEAR_TARGET_SWEEP = False

fig, ax = plt.subplots()

T = np.logspace(-4, 4, 1000)
fraction_bound = T / (1 + T)

# plot the binding curve (sigmoid)
ax.semilogx(T, fraction_bound)

# label axes
plt.xlabel('$T_∎/K_d^∎$')
plt.xticks([1e-4, 1e-2, 1e-0, 1e2, 1e4])
plt.ylabel('Relative Signal')
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.xlim((1e-4, 1e4))
plt.ylim((-0.05, 1.05))
plt.grid()

# plot the readout line
readout_center = 0.25
T_center = readout_center / (1 - readout_center)
y = np.array([readout_center, readout_center, -1])
x = np.array([1e-4, T_center, T_center])
line, = ax.semilogx(x, y)

# plot readout uncertainty
readout_max = readout_center + delta_abs
readout_min = readout_center - delta_abs
T_center = readout_center / (1 - readout_center)

# define the upper and lower bound for the target concentrations
T_max = 1e5 if readout_max >= 1.0 else readout_max / (1 - readout_max)
T_min = 1e-5 if readout_min <= 0 else readout_min / (1 - readout_min)

# plot the surrounding area of the readout area
x = np.array([1e-4, T_max, T_max, T_min, T_min, 1e-4])
y = np.array([readout_max, readout_max, -1, -1, readout_min, readout_min])
area = ax.fill(x, y, alpha=0.5)

# create the lookup table of which target concentrations to plot
repeats = 1000

if LINEAR_TARGET_SWEEP:
    Ts = np.logspace(-3, 3, repeats // 2 + 1)
    Ts = np.concatenate([Ts, Ts[1:-1][::-1]])
else:
    readouts = np.linspace(0.01, 0.99, repeats // 2 + 1)
    readouts = np.concatenate([readouts, readouts[1:-1][::-1]])


def animate(i):  # function that gets called for every frame
    if LINEAR_TARGET_SWEEP:
        T_center = Ts[(i + repeats // 4) % repeats]
        readout_center = T_center / (1 + T_center)
    else:
        readout_center = readouts[(i + repeats // 4) % repeats]
        T_center = readout_center / (1 - readout_center)

    # update the center line
    y = np.array([readout_center, readout_center, -1])
    x = np.array([1e-4, T_center, T_center])
    line.set_xdata(x)
    line.set_ydata(y)

    # update the area
    readout_max = readout_center * (1 + delta_rel) + delta_abs  # readout_center + delta
    readout_min = readout_center * (1 - delta_rel) - delta_abs  # readout_center - delta
    T_max = 1e5 if readout_max >= 1.0 else readout_max / (1 - readout_max)
    T_min = 1e-5 if readout_min <= 0 else readout_min / (1 - readout_min)

    x = np.array([1e-4, T_max, T_max, T_min, T_min, 1e-4])
    y = np.array([readout_max, readout_max, -1, -1, readout_min, readout_min])
    xy = np.stack([x, y])
    area[0].set(xy=xy.T)

    return None


# create animation
ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=False, save_count=49, frames=repeats)

writer = animation.FFMpegWriter(fps=50, metadata=dict(artist='Me'), bitrate=1800)
ani.save("../output/1d_readout_band_animation.mp4", writer=writer)
