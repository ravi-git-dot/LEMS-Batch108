## Pandas Tutorials

"""
Pandas is an open source.BSD-licensed library providing high-performance, easy-to-use data structure
and data analysis tools for the python programming language
"""
"""
What is data Frames?
What is data series?
Difference operation in Pandas 
"""

# first step import library as pandas
import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(0,20).reshape(5,4), index = ["Row1", "Row2","Row3", "Row4","Row5"],
                  columns=["Column1", "Column2","Column3","Column4"])

print(df.head())

df.to_csv("Test1.csv")

# Accessing the element
#with two method 1. loc, 2.iloc

print(df.loc["Row1"])

# data series either be one row are one columns
type(df.loc["Row1"])

print(df.iloc[:,:])

##  Important in R program index start with 1 but in python Program index start with 0
print(df.iloc[0:2,0:3])

## Data frame is more than one row and one columns
print(type(df.iloc[0:2,0:2]))

print(df.iloc[0:2, 0])

# Take the element from the column2
print(df.iloc[:,1:])

# data frame convert into array
print(df.iloc[:, 1:].values)

print(df.isnull().sum())

print(df["Column2"].value_counts())

print(df["Column2"].unique())

# df = pd.read_csv("mercedesbenz.csv")
# print(df.head())

print(df[["Column2", "Column3"]])


