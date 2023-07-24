import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

repeats = 1000

fig, ax = plt.subplots()

log_from, log_to = -4, 4
log_steps = 1024
A, B = np.logspace(log_from, log_to, log_steps), np.logspace(log_from, log_to, log_steps)
AA, BB = np.meshgrid(A, B)

readout = (AA + BB) / (1 + AA + BB)

plt.contourf(AA, BB, readout, cmap='viridis')

A_true, B_true = 1, 1
r_value = (A_true+B_true)/(1+A_true+B_true)
dots = ax.scatter(A_true, B_true, c='r')

x = np.logspace(log_from, log_to, log_steps)
y = (r_value) / (1-r_value) - x
line = ax.plot(x, y, color='r')

param = np.linspace(0, 2 * np.pi, repeats)
# param = np.linspace(0.01, 0.99, repeats // 2 + 1)
# param = np.concatenate([param, param[1:-1][::-1]])
As, Bs = np.power(10, 2 * np.sin(param)), np.power(10, 2 * np.sin(2*param))#np.logspace(-2,2,repeats), np.logspace(-2,2,repeats)

def animate(i):
    A, B = As[i], Bs[i]
    r_value = (A+B)/(1+A+B)

    dots.set_offsets(np.array([A,B]))

    y = (r_value) / (1 - r_value) - x

    line[0].set(ydata=y)  # update the data.
    return line[0], dots

plt.xlabel("$T_∎/K_d^∎$")
plt.ylabel("$T_▲/K_d^▲$")
plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.gcf().set_size_inches(6,6)

ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=49, frames=repeats)

writer = animation.FFMpegWriter(fps=50, metadata=dict(artist='Me'), bitrate=1800)
ani.save("/Users/linus/workspace/cr_quant/output/2d_readout.mp4", writer=writer)
# plt.show()