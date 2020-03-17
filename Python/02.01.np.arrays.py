import numpy as np
x = np.array([3, 6, 9, 12])
x = x/3.0
print(x)

y = [3, 6, 9, 12]
#y/3.0 #Error
print(y)

# https://www.pythoncentral.io/the-difference-between-a-list-and-an-array/
# Arrays have to be declared while lists don't because they are part of Python's syntax,
# so lists are generally used more often between the two, which works fine most of the time.
# However, if you're going to perform arithmetic functions to your lists,
# you should really be using arrays instead.
# Additionally, arrays will store your data more compactly and efficiently,
# so if you're storing a large amount of data, you may consider using arrays as well.