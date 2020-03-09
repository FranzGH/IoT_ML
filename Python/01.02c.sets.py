animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
animals.add('fish')       # Add an element to a set
print('fish' in animals)  # Prints "True"
print(len(animals))       # Number of elements in a set; prints "3"
animals.add('cat')        # Adding an element that is already in the set does nothing
print(len(animals))       # Prints "3"
animals.remove('cat')     # Remove an element from a set
print(len(animals))       # Prints "2"

animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print(f'{idx}: {animal}')
# Prints "#1: fish", "#2: dog", "#3: cat"

print('\nset comprehension')
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # Prints "{0, 1, 2, 3, 4, 5}"

my_set = {x for x in range(10)}
print(type(my_set))    # Prints "<class 'set'>"

'''
You cannot access items in a set by referring to an index,
since sets are unordered the items has no index.

my_set= my_set[3:]
print(my_set)

el= my_set[2]
print(el)

But you can loop through the set items using a for loop,
or ask if a specified value is present in a set, by using the in keyword.
'''

thisset = {"apple", "banana", "cherry"}

for x in thisset:
  print(x)
print("banana" in thisset)
