"""
Generates an animation to give intuition for how signal readout works for a single-target binder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# create the lookup table of which target concentrations to plot
repeats = 1000

if LINEAR_TARGET_SWEEP:
    Ts = np.logspace(-3, 3, repeats // 2 + 1)
    Ts = np.concatenate([Ts, Ts[1:-1][::-1]])  # repeat lookup table in reverse
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

    # update the values of the readout line being plotted
    y = np.array([readout_center, readout_center, -1])
    x = np.array([1e-4, T_center, T_center])
    line.set_xdata(x)
    line.set_ydata(y)
    return line,


# create animation
ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=49, frames=repeats)

writer = animation.FFMpegWriter(fps=50, metadata=dict(artist='Me'), bitrate=1800)
ani.save("../output/1D_readout_animation.mp4", writer=writer)
