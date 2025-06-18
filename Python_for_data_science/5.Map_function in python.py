# Map function in Python

def even_or_odd(num):
    if num % 2 == 0:
        return  "The numer {} is Even".format(num)
    else:
        return "The numer {} is Odd".format(num)

print(even_or_odd(34))

lst = [1,2,3,4,5,6,7,82,34,56,78,34,21,59]

# Use map with the defined function
print(list(map(even_or_odd, lst))
)