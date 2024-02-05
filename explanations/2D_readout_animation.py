"""
Generates an animation to give intuition for how signal readout works for a two-target binder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# create a lookup table for target concentrations to animate
param = np.linspace(0, 2 * np.pi, repeats)
As, Bs = np.power(10, 2 * np.sin(param)), np.power(10, 2 * np.sin(2 * param))

# label axes
plt.xlabel("$T_∎/K_d^∎$")
plt.ylabel("$T_▲/K_d^▲$")
plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.gcf().set_size_inches(6, 6)


def animate(i):  # function that gets called for every frame
    # lookup the new true concentration
    A, B = As[i], Bs[i]
    # calculate readout value for that concentration
    r_value = (A + B) / (1 + A + B)

    # update the dot marking the true target concentration
    dots.set_offsets(np.array([A, B]))

    # update the line of other target concentrations that would produce the same signal readout
    y = (r_value) / (1 - r_value) - x
    line[0].set(ydata=y)

    return line[0], dots


# create animation
ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=49, frames=repeats)

writer = animation.FFMpegWriter(fps=50, metadata=dict(artist='Me'), bitrate=1800)
ani.save("../output/2d_readout_animation.mp4", writer=writer)
