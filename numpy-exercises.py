__author__ = 'johnnychang'

import numpy as np

print(np.__version__)
np.__config__.show()

n = np.zeros(10)
n[4] = 1
print(n)


