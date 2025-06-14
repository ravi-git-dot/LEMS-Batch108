# Numpy Tutorials

"""
Numpy is a general processing array-processing package. It provides a high-Performance multidimensional
array output, and tools for working with these arrays. it is the fundamental package for scientific computing with
python
"""
import pandas as pd

# what is array

"""
An array is a data structure that store value of same data type. In python this is the main difference between 
arrays and lists. while python list contain values corresponding to different data type, array in python can only
contain values corresponding to same data type
"""

## initially lets import numpy

import numpy as np

my_list = [1, 2, 3, 4, 5]

arr = np.array(my_list)

print(arr)

print("Type:", {type(arr)})

# how many no of rows and columns
print(arr.shape)

# mutiple array
my_list1 = [1,2,3,4,5,6]
my_list2 = [2,4,6,7,8,9]
my_list3 = [3,2,4,5,7,6]

arr = np.array([my_list1,my_list2,my_list3])

print(arr)

print("before:",arr.shape)

print("Rearrange:", arr.reshape(6,3))

# check the array shape
print("after:", arr.shape)

print("change the array rows and columns:", arr.reshape(1,18))

## Indexing
arr1 = np.array([1,2,3,4,5,6,7,8,9])
print(arr1)

print(arr1[0])

arr2 = np.array([my_list1,my_list2,my_list3])

print("Two Dimension array:", arr2)

print( "first two rows:\n", arr2[0:2,:])
print("First two rows with two columns:\n", arr2[0:2,0:2])

print("first row and third column:\n", arr2[1:, 3:])

## arrange

arr2 = np.arange(0,10,step = 2) # start, stop, step

print("after arrange:", arr2)

print(np.linspace(1,10,50))

# copy function

arr = np.array([1,2,3,4,5,6,7,8,9])

arr[3:] = [100] * (len(arr) - 3)
print(arr)

arr[3:] = [500] * (len(arr) - 3)
print(arr)

arr1 =arr.copy()

# print(arr)
# arr1[3:] = [1000] - (len(arr)-3)
# print(arr1)

# Some Condition are very use in exploratory data analysis

var = 2

print("which are less than 2 in array:",arr < 2)

"""
arr * 2
arr / 2
arr % 2
"""

#  create and array and reshape
print(np.arange(0,10).reshape(5,2))

# arr1 = np.arange(0,10).reshape(2,5)
# print("Array with 2:",arr1)
# arr2 = np.arange(0,10).reshape(5,2)
# print(arr2)
#
# print(arr1 * arr2)

arr = np.array([[0, 1,4,9,16],
                 [25,36,49,64,81]])

print( "new array:\n", arr)

print(np.ones(4, dtype = int))

print(np.ones((2,5), dtype = float))


# random Distribution

print(np.random.rand(4,4))

arr_ex = np.random.rand(4,4)

print("update array:\n",arr_ex)


import seaborn as sns
import pandas as pd
sns.histplot(pd.DataFrame(arr_ex), kde=True, stat="density")

print(np.random.randint(0,100,8))
print(np.random.randint(0,100,8).reshape(4,2))

print(np.random.random_sample((1,5)))

