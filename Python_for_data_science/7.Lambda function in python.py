# lambda function in Python

# Is also called Anonymous function
# A function with no more
# only for single operation

def addition(a, b):
    return a + b

print(addition(4,5))

addition = lambda a, b : a + b

print(addition(34,34))


def even_odd(num):
    if num % 2 == 0:
        return True

print(even_odd(24))

even_odd = lambda num : num % 2 == 0

print(even_odd(12))

def addition(x,y,z):
    return x + y + z

print(addition(2,4,7))

addition = lambda x, y, z : x + y + z

print(addition(34,45,56))