import numpy as np

def f(x):
    return 3 * x + 2 + np.random.rand(len(x))

x = np.linspace(-2, 2, 100)
y = f(x)

A = np.c_[np.ones(len(x)), x]        # columns: [1, x]
w = np.linalg.lstsq(A, y, rcond=None)[0]   # w = [intercept, slope]
fit = A @ w


import matplotlib.pyplot as plt

plt.plot(x, y, "b.")
plt.plot(x, fit, "r-")
plt.show()
