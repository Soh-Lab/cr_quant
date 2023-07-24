import numpy as np
from matplotlib import pyplot as plt, colors, animation

log_from, log_to = -4, 4
log_steps = 1024
A, B = np.logspace(log_from, log_to, log_steps), np.logspace(log_from, log_to, log_steps)
AA, BB = np.meshgrid(A, B)

readout = (AA + BB) / (1 + AA + BB)

fig, ax = plt.subplots()
plt.contourf(AA, BB, readout, cmap='viridis')

A_true, B_true = 1, 1
r_value = (A_true + B_true) / (1 + A_true + B_true)
dots = ax.scatter(A_true, B_true, c='r')

x = np.logspace(log_from, log_to, log_steps)
y = (r_value) / (1 - r_value) - x
line = ax.plot(x, y, color='r')

r_top = r_value + 0.05
r_bottom = r_value - 0.05

x_top = x
y_top = (r_top) / (1 - r_top) - x_top if r_top < 1.0 else np.ones_like(x_top) * np.power(10.,
                                                                                         log_to + 1)

x_bottom = x[::-1]
y_bottom = (r_bottom) / (1 - r_bottom) - x_bottom if r_bottom > 0.0 else np.ones_like(
    x_bottom) * np.power(10., log_from - 1)

y = np.concatenate([y_top, y_bottom])
area = ax.fill(np.concatenate([x_top, x_bottom]), y, alpha=0.5, color='r')

plt.xlabel("$T_∎/K_d^∎$")
plt.ylabel("$T_▲/K_d^▲$")
plt.grid()
plt.ylim([10 ** log_from, 10 ** log_to])
plt.xlim([10 ** log_from, 10 ** log_to])
plt.xscale("log")
plt.yscale("log")
plt.gcf().set_size_inches(6,6)
# plt.savefig('/Users/linus/workspace/xplex/LGM/figs/2D_readout.png')

repeats = 1000

param = np.linspace(0, 2 * np.pi, repeats)
# param = np.linspace(0.01, 0.99, repeats // 2 + 1)
# param = np.concatenate([param, param[1:-1][::-1]])
As, Bs = np.power(10, 2 * np.sin(param)), np.power(10, 2 * np.sin(2*param))#np.logspace(-2,2,repeats), np.logspace(-2,2,repeats)

def animate(i):
    A, B = As[i], Bs[i]
    r_value = (A+B)/(1+A+B)

    # update dot location
    dots.set_offsets(np.array([A,B]))

    # update mid-line
    y = (r_value) / (1 - r_value) - x

    line[0].set(ydata=y)  # update the data.

    # update area
    r_top = r_value + 0.05
    r_bottom = r_value - 0.05
    y_top = (r_top) / (1 - r_top) - x_top if r_top < 1.0 else np.ones_like(x_top) * np.power(10.,
                                                                                             log_to + 1)
    y_bottom = (r_bottom) / (1 - r_bottom) - x_bottom if r_bottom > 0.0 else np.ones_like(
        x_bottom) * np.power(10., log_from - 1)

    xy = np.stack([np.concatenate([x_top, x_bottom]), np.concatenate([y_top, y_bottom])])
    area[0].set(xy=xy.T)

    return None #line[0], dots

ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=False, save_count=49, frames=repeats)

writer = animation.FFMpegWriter(fps=50, metadata=dict(artist='Me'), bitrate=1800)
ani.save("/Users/linus/workspace/cr_quant/output/2dreadout_intervals.mp4", writer=writer)
# plt.savefig('/Users/linus/workspace/xplex/LGM/figs/2D_readout_area.png')
# plt.show()
