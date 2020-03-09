# https://www.sharpsightlabs.com/blog/numpy-axes-explained/

import numpy as np

np_array_2d = np.arange(0, 6).reshape([2,3])
print(np_array_2d)

a = np.sum(np_array_2d, axis = 0) #Collapse 0-axis
print(a)

a = np.sum(np_array_2d, axis = 1)
print(a)

np_array_1s = np.array([[1,1,1],[1,1,1]])
np_array_9s = np.array([[9,9,9],[9,9,9]])

print(np.concatenate([np_array_1s, np_array_9s], axis = 0))
print(np.concatenate([np_array_1s, np_array_9s], axis = 1))

np_array_1s_1dim = np.array([1,1,1])
np_array_9s_1dim = np.array([9,9,9])
print(np.concatenate([np_array_1s_1dim, np_array_9s_1dim], axis = 0))

# np.concatenate([np_array_1s_1dim, np_array_9s_1dim], axis = 1)
# error