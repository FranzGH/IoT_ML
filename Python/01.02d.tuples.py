d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)        # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"

my_tuple= tuple(i for i in range(1, 10))
print(my_tuple)
print(type(my_tuple))    # Prints "<class 'tuple'>"

my_tuple= my_tuple[3:]
print(my_tuple)

el= my_tuple[2]
print(el)
