## Pyhton  list Iterables vs Iterables

#list Iterables
lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in lst:
    print(i)

print(iter(lst))

'''
Iterators
Definition: An iterator is an object that enables traversing through all the elements of a collection, one at a time.

Characteristics:

Not Indexable: You cannot access elements directly by index.

Stateful:       Maintains its current position during iteration.

Memory Efficient: Generates items on the fly, which is beneficial for large or infinite sequences.

Single Use: Once exhausted, cannot be reused without reinitialization.

my_list = ['a', 'b', 'c', 'd']
my_iter = iter(my_list)
print(next(my_iter))  # Output: 'a'
print(next(my_iter))  # Output: 'b'

'''

lst1 = iter(lst)

print(next(lst1))
print(next(lst1))

# the main functionality of Iterators is that what ever value is there inside it is will not get stored or
# Initialized in the memory at once oky one by one it will get in  call each slice  then it will stored

for i in lst1:
    print(i)