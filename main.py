import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import quad
from functools import partial


def gaussian(x, t, k, sigma):
    return np.real(np.exp(-sigma ** 2 * (x - k * t) ** 2 + 1j * k * x))

def gaussian_fourier(x, t, k, sigma):
    n = 50

    fc = lambda x: gaussian(x, 0, k, sigma) * np.cos(index * np.pi * x / L)
    fs = lambda x: gaussian(x, 0, k, sigma) * np.sin(index * np.pi * x / L)

    sum = quad((partial(gaussian, t=0, k=k, sigma=sigma)), -L, L)[0] * (1.0 / L)

    for index in range(1, n + 1):
        an = quad(fc, -L, L)[0] * (1.0 / L)
        bn = quad(fs, -L, L)[0] * (1.0 / L)
        sum += an * np.cos(index * np.pi * x / L - (c * index * np.pi / L) * t) + bn * np.sin(
            index * np.pi * x / L - (c * index * np.pi / L) * t)

    return sum


# Define the parameters of the wave packet
sigma = 1
k_0 = 1
frequency = 1
omega = 2 * np.pi * frequency
c = 50
x_0 = -c * 5


# Define the time and spatial coordinates
t_min = 0
t_max = 10
dt = 0.01
times = np.arange(t_min, t_max, dt)
x_min = 0
x_max = 100
L = 100
dx = 0.01
x = np.arange(x_min, x_max, dx)

# Define the wave packet
def wave_packet(x, t):
    return (2 * np.pi * sigma ** 2) ** (-1 / 4) * np.exp(-((x - x_0 - c * t) ** 2) / (4 * sigma ** 2 * t ** 2 +0.01) + 1j * (k_0 * (x - c * t) - omega * t))

def wave_packet_fourier(x, t):
    n = 50
    t0 = 5
    fc = lambda x: wave_packet(x, t0) * np.cos(index * np.pi * x / L)
    fs = lambda x: wave_packet(x, t0) * np.sin(index * np.pi * x / L)

    sum = quad(partial(wave_packet, t=t0), -L, L)[0] * (1.0 / L)

    for index in range(1, n + 1):
        an = quad(fc, -L, L)[0] * (1.0 / L)
        bn = quad(fs, -L, L)[0] * (1.0 / L)
        sum += an * np.cos(index * np.pi * x / L - (c * index * np.pi / L) * t) + bn * np.sin(
            index * np.pi * x / L - (c * index * np.pi / L) * t)

    return sum


# Set up the animation
fig, ax = plt.subplots()
line, = ax.plot(x, np.real(wave_packet_fourier(x, 0)))
ax.set_xlabel('x')
ax.set_ylabel('Re[$\psi(x,t)$]')

def animate(i):
    line.set_ydata(np.real(wave_packet_fourier(x, times[i])))
    ax.set_title('Gaussian wave packet at time t = {:.2f}'.format(times[i]))
    return line,

ani = animation.FuncAnimation(fig, animate, len(times), interval=20)
plt.show()
