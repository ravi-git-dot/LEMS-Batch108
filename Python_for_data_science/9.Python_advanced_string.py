import pandas as pd

## String formating in python
print('hello world')

strs = 'Hello Everyone'
print(strs)

def greating(name):
    return 'Hello {}.Welcome to the Community'.format(name)

print(greating('Ravi'))

def welcome_email(fristname, age):
    return 'welcome{} .Your age is {}'.format(fristname, age)

print(welcome_email('Ravi', 24))


def welcome_email(fristname, age):
    return 'welcome {fristname} .Your age is {age}'.format(age = age , fristname = fristname)

print(welcome_email('Ravi', 24))
