import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad
from functools import partial


def gaussian(x, t, k, sigma):
    return np.real(np.exp(-sigma ** 2 * (x - k * t) ** 2 + 1j * k * x))





def gaussian_chatGPT(x, t, k0, x0):
    w0 = 2 * np.pi * frequency
    return np.real((2 * np.pi * sigma ** 2) ** (-1 / 4) * np.exp(
        -((x - x0) ** 2) / (4 * (sigma * t + 0.001) ** 2) + 1j * (k0 * x - w0 * t)))




# The fourier series approx will be from -L to L. L will be set externally because it's the width of the graph.
L = 1000


def gaussian_fourier(x, t, k, sigma):
    n = 50

    fc = lambda x: gaussian(x, 0, k, sigma) * np.cos(index * np.pi * x / L)
    fs = lambda x: gaussian(x, 0, k, sigma) * np.sin(index * np.pi * x / L)

    sum = quad(partial(gaussian, t=0, k=k, sigma=sigma), -L, L)[0] * (1.0 / L)

    for index in range(1, n + 1):
        an = quad(fc, -L, L)[0] * (1.0 / L)
        bn = quad(fs, -L, L)[0] * (1.0 / L)
        sum += an * np.cos(index * np.pi * x / L - (c * index * np.pi / L) * t) + bn * np.sin(
            index * np.pi * x / L - (c * index * np.pi / L) * t)

    return sum

#
# x = np.linspace(-100, 1000, 1000)
# t = np.linspace(0, 10, 1000)
# k = 200
# sigma = 0.01
#
# fig, ax = plt.subplots()
# line, = ax.plot(x, np.real(gaussian(x, t[0], k, sigma)))
#
#
# def update(i):
#     # line.set_ydata(gaussian(x, t[i], k, sigma))
#     line.set_ydata(psi(x, t[i], k, sigma))
#     return line,
#
#
# ani = FuncAnimation(fig, update, frames=range(len(t)), interval=100, blit=True)
# plt.show()



# Define the parameters of the wave packet
sigma = 1
x_0 = 0
k_0 = 1
omega = 2 * np.pi
c = 1

# Define the time and spatial coordinates
t_min = 0
t_max = 10
dt = 0.01
times = np.arange(t_min, t_max, dt)
x_min = -10
x_max = 10
dx = 0.01
x = np.arange(x_min, x_max, dx)

# Define the wave packet
def wave_packet(x, t):
    return (2 * np.pi * sigma ** 2) ** (-1 / 4) * np.exp(-((x - x_0 - c * t) ** 2) / (4 * sigma ** 2 * t ** 2 +0.01) + 1j * (k_0 * (x - c * t) - omega * t))

def wave_packet_fourier(x, t):
    n = 50

    fc = lambda x: wave_packet(x, 1) * np.cos(index * np.pi * x / L)
    fs = lambda x: wave_packet(x, 1) * np.sin(index * np.pi * x / L)

    sum = quad(partial(wave_packet, t=1), -L, L)[0] * (1.0 / L)

    for index in range(1, n + 1):
        an = quad(fc, -L, L)[0] * (1.0 / L)
        bn = quad(fs, -L, L)[0] * (1.0 / L)
        sum += an * np.cos(index * np.pi * x / L - (c * index * np.pi / L) * t) + bn * np.sin(
            index * np.pi * x / L - (c * index * np.pi / L) * t)

    return np.real(sum)

# Calculate the wave packet for each time step
psi = []
for t in times:
    psi.append(wave_packet(x, t))

# Plot the wave packet over time
for i, psi_t in enumerate(psi):
    plt.clf()
    plt.plot(x, np.real(psi_t))
    plt.xlabel('x')
    plt.ylabel('Re[$\psi(x,t)$]')
    plt.title('Gaussian wave packet at time t = {}'.format(times[i]))
    plt.pause(0.01)

plt.show()
