import numpy as np
# When operating and manipulating arrays,
# their data is sometimes copied into a new array and sometimes not.
# This is often a source of confusion for beginners. There are three cases:

a = np.array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
b = a
print(b is a)

# Python passes mutable objects as references, so function calls make no copy.
def f(x):
    print(id(x))     # id is a unique identifier of an object

print(id(a))
print(f(a)) # same value, because the object is the same (copied reference, not object)

#Different array objects can share the same data.
# The view method creates a new array object that looks at the same data.
c = a.view()
print(c is a)
print(c.base is a) # c is a view of the data owned by a
print(c.flags.owndata)
c = c.reshape((2, 6)) # a's shape doesn't change
print(a.shape) 
c[0, 4] = 1234 # a's data changes
print(a) 

# Slicing an array returns a view of it:
s = a[ : , 1:3]
s[:] = 10
print(a)

#deep copy
d = a.copy() 
print(d is a)
print(d.base is a)
d[0,0] = 9999
print(a)

# Sometimes copy should be called after slicing if the original array is not required anymore. For example, suppose a is a huge intermediate result and the final result b only contains a small fraction of a, a deep copy should be made when constructing b with slicing:
a = np.arange(int(1e8))
b = a[:100].copy()
del a # the memory of ``a`` can be released.
# If b = a[:100] is used instead, a is referenced by b and will persist in memory even if del a is executed.
print(b)