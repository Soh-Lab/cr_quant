"""
Generates an animation to give intuition for how signal readout resolution works for a two-target
binder.
"""

import numpy as np
from matplotlib import pyplot as plt, colors, animation

# define animation parameters
repeats = 1000
log_from, log_to = -4, 4
log_steps = 1024

fig, ax = plt.subplots()

# plot the background
A, B = np.logspace(log_from, log_to, log_steps), np.logspace(log_from, log_to, log_steps)
AA, BB = np.meshgrid(A, B)
readout = (AA + BB) / (1 + AA + BB)
plt.contourf(AA, BB, readout, cmap='viridis')

# plot true target concentration as red dot
A_true, B_true = 1, 1
r_value = (A_true + B_true) / (1 + A_true + B_true)
dots = ax.scatter(A_true, B_true, c='r')

# plot the line of other target concentrations that will produce an identical signal readout
x = np.logspace(log_from, log_to, log_steps)
y = (r_value) / (1 - r_value) - x
line = ax.plot(x, y, color='r')

# plot the area of concentrations that could produce the same signal readout
r_top = r_value + 0.05
r_bottom = r_value - 0.05

x_top = x
if r_top < 1.0:
    y_top = (r_top) / (1 - r_top) - x_top
else:
    y_top = np.ones_like(x_top) * np.power(10., log_to + 1)

x_bottom = x[::-1]
if r_bottom > 0.0:
    y_bottom = (r_bottom) / (1 - r_bottom) - x_bottom
else:
    y_bottom = np.ones_like(x_bottom) * np.power(10., log_from - 1)

y = np.concatenate([y_top, y_bottom])
area = ax.fill(np.concatenate([x_top, x_bottom]), y, alpha=0.5, color='r')

# create a lookup table for target concentrations to animate
param = np.linspace(0, 2 * np.pi, repeats)
As, Bs = np.power(10, 2 * np.sin(param)), np.power(10, 2 * np.sin(2 * param))

# label axes
plt.xlabel("$T_∎/K_d^∎$")
plt.ylabel("$T_▲/K_d^▲$")
plt.grid()
plt.ylim([10 ** log_from, 10 ** log_to])
plt.xlim([10 ** log_from, 10 ** log_to])
plt.xscale("log")
plt.yscale("log")
plt.gcf().set_size_inches(6, 6)


def animate(i):  # function that gets called for every frame
    A, B = As[i], Bs[i]
    r_value = (A + B) / (1 + A + B)

    # update dot location
    dots.set_offsets(np.array([A, B]))

    # update mid-line
    y = (r_value) / (1 - r_value) - x
    line[0].set(ydata=y)

    # update area
    r_top = r_value + 0.05
    r_bottom = r_value - 0.05

    if r_top < 1.0:
        y_top = (r_top) / (1 - r_top) - x_top
    else:  # handle no upper bound
        y_top = np.ones_like(x_top) * np.power(10., log_to + 1)

    if r_bottom > 0.0:
        y_bottom = (r_bottom) / (1 - r_bottom) - x_bottom
    else:  # handle no lower bound
        y_bottom = np.ones_like(x_bottom) * np.power(10., log_from - 1)

    xy = np.stack([np.concatenate([x_top, x_bottom]), np.concatenate([y_top, y_bottom])])
    area[0].set(xy=xy.T)

    return None


# create animation
ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=False, save_count=49, frames=repeats)

writer = animation.FFMpegWriter(fps=50, metadata=dict(artist='Me'), bitrate=1800)
ani.save("../output/2d_readout_band_animation.mp4", writer=writer)
