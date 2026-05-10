# Communication-Engineering
Implementation of the Discrete Fourier Transform (DFT) in Python based on the mathematical definition of the transform. The project compares the execution time of a manually implemented DFT algorithm with NumPy’s optimized FFT implementation using np.fft.fft(). Includes visualization of the frequency-domain results and runtime analysis.

####################################################
# Code by Communications Engineering Lab (CEL), 2022
# Communication Lab - Chapter: DFT
#
# Task: DFT implementation
####################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

t_s = 0.001
f_s = 1 / t_s
N = 1024

t = np.arange(0, N * t_s, t_s)
f = np.arange(0, N) * f_s / N

x = np.exp(-(t - 512 * t_s) ** 2 * 1e5)

def dft_impl(x):
    M = len(x)
    X_dft = np.zeros(M, dtype=complex)

    for mu in range(M):
        for n in range(M):
            X_dft[mu] += x[n] * np.exp(-1j * 2 * np.pi / M * mu * n)

    return X_dft

start = time.time()
X_dft = dft_impl(x)
timer_dft = time.time() - start

start = time.time()
X_fft = np.fft.fft(x)
timer_fft = time.time() - start

print("Runtime of own implementation: " + str(timer_dft))
print("Runtime of FFT: " + str(timer_fft))

# Plot settings
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': False,
    'font.size': 10,
    'figure.max_open_warning': 100
})

plt.figure(figsize=(7, 5))

plt.subplot(211)
plt.plot(f, np.abs(X_dft))
plt.ylabel("DFT")
plt.grid()

plt.subplot(212)
plt.plot(f, np.abs(X_fft))
plt.ylabel("FFT")
plt.xlabel("Frequency [Hz]")
plt.grid()

plt.tight_layout()
plt.show()
