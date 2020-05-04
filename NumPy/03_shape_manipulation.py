import numpy as np

from numpy.random import Generator, PCG64
a = np.ones((3,4), dtype=int)
print(a)
print(a.shape)
rg = Generator(PCG64())
a = rg.standard_normal((3,4))
print(a)
print(a.shape)

# The shape of an array can be changed with various commands.
# Note that the following three commands all return a modified array,
# but do not change the original array:
b = a.ravel()  # returns the array, flattened
print(b)
b = a.reshape(6,2)
print(b)
b = a.T # T is the transpose
print(b)
print(a.T.shape)

a.resize((2,6))
print(a)
# If a dimension is given as -1 in a reshaping operation,
# the other dimensions are automatically calculated:
a = a.reshape(3,-1)
print(a)

print('Stacking together different arrays')
a = np.floor(10*rg.standard_normal((2,2)))
b = np.floor(10*rg.standard_normal((2,2)))
c = np.vstack((a,b))
print(c)
c = np.hstack((a,b))
print(c)

#.... missing

print(np.row_stack is np.vstack) #True