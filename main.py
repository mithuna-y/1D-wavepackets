import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create time array
t = np.linspace(0, 10, 1000)

# Define original wave packet
def wave_packet(x, t, k, sigma):
    return np.exp(-sigma**2 * (x - k * t)**2 + 1j * k * x)

# Define shifted wave packet
def shifted_wave_packet(x, t, k, sigma):
    return np.exp(-sigma**2 * (x - k * t)**2 + 1j * k * x) * 1j

def added_waves(x, t, k, sigma):
    return 0.5 * wave_packet(x, t, k, sigma) + 0.5 * shifted_wave_packet(x, t, k, sigma)

# Set up figure and axis
fig, ax = plt.subplots()
n_steps = 100
x = np.linspace(-10, 100, n_steps)
k = 10
sigma = 0.1
line1, = ax.plot(x, np.real(wave_packet(x, t[0], k, sigma)))
line2, = ax.plot(x, np.real(shifted_wave_packet(x, t[0], k, sigma)))
line3, = ax.plot(x, np.real(added_waves(x, t[0], k, sigma)))


# Update function for animation
def update(frame):
    line1.set_ydata(np.real(wave_packet(x, t[frame], k, sigma)))
    line2.set_ydata(np.real(shifted_wave_packet(x, t[frame], k, sigma)))
    line3.set_ydata(np.real(added_waves(x, t[frame], k, sigma)))

    return line1, line2

# Animate the wave packets
ani = FuncAnimation(fig, update, frames=range(n_steps), interval=100)
plt.show()
