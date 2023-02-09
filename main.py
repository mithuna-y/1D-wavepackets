import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad
from functools import partial

def gaussian(x, t, k, sigma):
    return np.real(np.exp(-sigma**2 * (x - k * t)**2 + 1j * k * x))

# The fourier series approx will be from -L to L. L will be set externally because it's the width of the graph.
L = 1000
c = 1
def gaussian_fourier(x, t, k, sigma):
    n = 50

    fc = lambda x: gaussian(x, 0, k, sigma) * np.cos(index * np.pi * x/ L)
    fs = lambda x: gaussian(x, 0, k, sigma) * np.sin(index * np.pi * x / L)

    sum = quad(partial(gaussian, t=0, k=k, sigma=sigma), -L, L)[0] * (1.0 / L)

    for index in range(1, n+1):
        an = quad(fc, -L, L)[0] * (1.0 / L)
        bn = quad(fs, -L, L)[0] * (1.0 / L)
        sum += an * np.cos(index * np.pi * x/ L - (c * index * np.pi / L) * t) + bn * np.sin(index * np.pi * x/ L- (c * index * np.pi / L) * t)

    return sum




x = np.linspace(-100, 1000, 1000)
t = np.linspace(0, 10, 1000)
k = 200
sigma = 0.01

fig, ax = plt.subplots()
line, = ax.plot(x, np.real(gaussian(x, t[0], k, sigma)))

def update(i):

    #line.set_ydata(gaussian(x, t[i], k, sigma))
    line.set_ydata(gaussian(x, t[i], k, sigma))
    return line,

ani = FuncAnimation(fig, update, frames=range(len(t)), interval=100, blit=True)
plt.show()
