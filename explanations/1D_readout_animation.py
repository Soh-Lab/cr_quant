import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

T = np.logspace(-4, 4, 1000)
fraction_bound = T / (1 + T)

ax.semilogx(T, fraction_bound)

plt.xlabel('$T_∎/K_d^∎$')
plt.xticks([1e-4, 1e-2, 1e-0, 1e2, 1e4])
plt.ylabel('Relative Signal')
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.xlim((1e-4, 1e4))
plt.ylim((-0.05, 1.05))
plt.grid()

readout_center = 0.25
T_center = readout_center / (1 - readout_center)

y = np.array([readout_center, readout_center, -1])
x = np.array([1e-4, T_center, T_center])

line, = ax.semilogx(x, y)

repeats = 1000
readouts = np.linspace(0.01, 0.99, repeats // 2 + 1)
readouts = np.concatenate([readouts, readouts[1:-1][::-1]])
print(readouts.shape)

Ts = np.logspace(-3, 3, repeats // 2 + 1)
Ts = np.concatenate([Ts, Ts[1:-1][::-1]])
print(Ts.shape)


def animate(i):
    T_center = Ts[(i + repeats // 4) % repeats]
    readout_center = T_center/(1+T_center)
    # readout_center = readouts[(i + repeats // 4) % repeats]
    # T_center = readout_center / (1 - readout_center)

    y = np.array([readout_center, readout_center, -1])
    x = np.array([1e-4, T_center, T_center])

    line.set_xdata(x)
    line.set_ydata(y)  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=49, frames=repeats)

writer = animation.FFMpegWriter(fps=50, metadata=dict(artist='Me'), bitrate=1800)
ani.save("/Users/linus/workspace/cr_quant/output/movie01.mp4", writer=writer)

plt.show()
