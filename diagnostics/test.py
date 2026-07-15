import numpy as np

n_inputs = 6
a = np.ones((n_inputs,3,4))
print(a)
x_in = a.reshape(n_inputs, -1)
print(x_in)

