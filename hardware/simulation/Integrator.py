import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

# Define the time axis with reduced sampling points
t = np.linspace(0, 2 * np.pi, 1000)

# Input sine wave
input_signal = np.sin(t)

# Perform integration
output_signal = cumtrapz(input_signal, t, initial=0)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, input_signal, 'o-', label='Input: Sine Wave')
plt.plot(t, output_signal, 'o--', label='Output: Integrated Signal')
plt.title('Integration of a Sine Wave (Discrete Iteration)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()