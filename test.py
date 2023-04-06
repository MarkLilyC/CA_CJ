import numpy as np
import logging

a = np.ones((10), int)
b = np.zeros((40), int)
c = np.hstack((a, b))
print(c)
np.random.shuffle(c)
print(c)