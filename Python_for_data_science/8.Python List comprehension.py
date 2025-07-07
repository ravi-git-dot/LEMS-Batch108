# Python List Comprehension

"""
list comprehension provide a concise way to create list, It consists of brackets containing an expression followed
by a for clause, then zero or more for it case, The expression can be anything, meaning you can put in all kinds of objects
in lists
"""
lst1 = []

def lst_square(lst):
    for i in lst:
        lst1.append(i* i)
    return lst1

print(lst_square([1,2,3,4,5,6]))


# what is should return
lst = [1,2,3,4,5,6,7]
print([i*i for i in lst])


print([i*i for i in lst if i % 2 == 0])

print(["even" if i % 2 == 0 else "odd" for i in lst])

import lightgbm as lgb
print("LightGBM imported successfully.")
