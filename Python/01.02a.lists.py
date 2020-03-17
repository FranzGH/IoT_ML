print('\n#lists')
xs = [3, 1, 2]    # Create a list
print(xs, xs[2])  # Prints "[3, 1, 2] 2"
print(xs[-1])     # Negative indices count from the end of the list; prints "2"
xs[2] = 'foo'     # Lists can contain elements of different types
print(xs)         # Prints "[3, 1, 'foo']"
xs.append('bar')  # Add a new element to the end of the list
print(xs)         # Prints "[3, 1, 'foo', 'bar']"
x = xs.pop()      # Remove and return the last element of the list
print(x, xs)      # Prints "bar [3, 1, 'foo']"
xs.append(['restaurant', 'hotel'])  # Add a new element to the end of the list
print(xs)         # Prints "[3, 1, 'foo', 'bar', ['restaurant', 'hotel']]"
xs.extend(['restaurant', 'hotel'])  # Add a new element to the end of the list
print(xs)         # Prints "[3, 1, 'foo', 'bar', ['restaurant', 'hotel'], 'restaurant', 'hotel']"
ys = xs + ['pub', 'osteria']
print(ys)         # Prints "[3, 1, 'foo', 'bar', ['restaurant', 'hotel'], 'restaurant', 'hotel', 'pub', 'osteria']"

print('\n#slicing')
nums = list(range(5))     # range is a built-in function that creates a list of integers
print(nums)               # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(nums[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # Assign a new sublist to a slice
print(nums)               # Prints "[0, 1, 8, 9, 4]"

print('\n#loops')
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# Prints "cat", "dog", "monkey", each on its own line.
for idx, animal in enumerate(animals):
    print(f'{idx}- {animal}')

print('\n#list comprehension')
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # Prints [0, 1, 4, 9, 16]

nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"