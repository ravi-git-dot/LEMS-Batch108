# filter function in Python

def even(num):
    if num  % 2 == 0:
        return  True

lst = [1,2,3,4,5,6,7,8,9,0]

print(list(filter(even, lst)))

print(list(filter(lambda num : num % 2 == 0, lst)
))

print(list(map(lambda num : num % 2 == 0, lst)
))