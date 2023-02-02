import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def gaussian_wave_packet(x, t, k, sigma, N=100):
    omega = 2 * np.pi
    a = np.zeros(2 * N)
    g = np.zeros(x.shape, dtype=complex)
    for n in range(-N, N):
        a[n + N] = np.exp(-0.5 * sigma**2 * (n * omega - k)**2)
        g += a[n + N] * np.exp(-1j * n * omega * t) * np.exp(-1j * n * omega * x)
    g *= (2 * np.pi)**(-0.25)
    return g

x = np.linspace(-100, 1000, 1000)
t = np.linspace(0, 10, 1000)
k = 100
sigma = 0.1

fig, ax = plt.subplots()
line, = ax.plot(x, np.real(gaussian_wave_packet(x, t[0], k, sigma)))

def update(i):
    line.set_ydata(np.real(gaussian_wave_packet(x, t[i], k, sigma)))
    return line,

ani = FuncAnimation(fig, update, frames=range(len(t)), interval=100, blit=True)
plt.show()
