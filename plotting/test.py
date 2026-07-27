import numpy as np


data_path = "data/lasing_testing/03_n2f_testing/simulation_gpumeep/n2f_map_.npz"

import matplotlib.pyplot as plt

data = np.load(data_path)

print(data)
plt.imshow(data["E2"].T)
plt.show()
