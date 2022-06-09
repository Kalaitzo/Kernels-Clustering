import numpy as np
import matplotlib.pyplot as plt


# Kernel function
def K(x, h):
    return np.exp(-0.5 * np.square(x) / h) / np.sqrt(2 * np.pi * h)


# Create implementations
N = 1000
X = np.random.rand(N)

# Axis samples
samples = 4 * N
X_ax = np.linspace(-3, 3, samples)
hi = 0.0001

f_hat = K(X_ax[:, None] - X, hi).mean(axis=-1)
plt.plot(X_ax, f_hat)
plt.show()

print('DONE')
