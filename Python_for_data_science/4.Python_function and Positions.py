## function in python
num = 24

def even_odd(num):
    if num % 2 == 0:
        print(f"{num} is even number")
    else:
        print(f"{num} is not even number")

print(even_odd(12))

# print vs return
def hello_world():
  print("Hello world")

print(hello_world())


def hello_welcome():
    return  "Hello world"

val = hello_welcome()
print(val)

def add(num1,num2):
    return  num1+num2

out = add(24,34)
print(out)

# position and variables

def hello(name, age= 25):
    print(f" My name is {name} and my age is {age}")

print(hello("Ravi", 25))

def hello(*args, **kwargs):
    print("Positional arguments:", args)
    print("Keyword arguments:", kwargs)

hello("Ravi", "Raj", age=25, dob=2000)

lst = ['Ravi', 'Raj']
dict_args = {'age': 25, 'dob': 2000}

hello(*lst, **dict_args)

def even_odd_sum(lst):
    even_sum = 0
    odd_sum = 0
    for i in lst:
        if i % 2 == 0:
            even_sum += i
        else:
            odd_sum += i
    return even_sum, odd_sum

# Example usage
lst = [1, 2, 3, 4, 5, 6]
even_sum, odd_sum = even_odd_sum(lst)
print("Even sum:", even_sum)
print("Odd sum:", odd_sum)



