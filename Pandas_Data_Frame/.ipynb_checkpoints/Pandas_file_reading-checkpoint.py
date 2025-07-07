import pandas as pd

# read the csv file
df = pd.read_csv("mercedesbenz.csv")
print(df.head())

# data information
print(df.info())

# data describe in details this only consider numerical columns only not categorical column
print(df.describe())

# data separated with semi-column
test_df = pd.read_csv("Test1.csv",sep = ';')
print(test_df.head())

# get the unique category counts
print(df["X0"].value_counts())

print(df[df["y"]>100])

# CSV

from io import StringIO, BytesIO

data = ("col1,col2,col3\n"
        "x,y,1\n"
        "a,b,2\n"
        "c,d,3")

print(type(data))

# StringIo is helping to memory power
df1 = pd.read_csv(StringIO(data))

print(df1)

## read from specific column
df = pd.read_csv(StringIO(data),usecols =["col1", "col3"])

print(df)

print(df.to_csv("Test.csv"))

# specifying columns data types
data = ("a, b, c, d\n"
        "1,2,3,4\n"
        "5,6,7,8\n"
        "9,10,11,12")

df =pd.read_csv(StringIO(data),dtype = object)

print(df['a'])

df = pd.read_csv(StringIO(data), dtype={"b":int, "c":float, "a":"Int64"})

print(df)

# Check the datatypes

print(df.dtypes)

data =("index, a, b, c\n"
       "4, apple, bat, 5.7\n"
       "8, orange, cow, 10")

#set as index default if 1 mean apple, orange are seen as index
df = pd.read_csv(StringIO(data), index_col=0) #

print(df)

data =("a, b, c\n"
       "4, apple, bat \n"
       "8, orange, cow")

df = pd.read_csv(StringIO(data))

print(df)

print(pd.read_csv(StringIO(data), index_col=False))

import pandas as pd
from io import StringIO

data = """a,b,c
        4,apple,bat
        8,orange,cow"""

df = pd.read_csv(StringIO(data), usecols=["b", "c"], index_col=False)
print(df)

## Quoting and Escape Character. very use full in NLP

import pandas as pd
from io import StringIO

data = '''a,b,c
hello,"Bob, nice to see you",5'''

df = pd.read_csv(StringIO(data))
print(df)

# print(pd.read_csv(StringIO(data), escapechar=\\))

# URL to CSV

import pandas as pd

url = "https://download.bls.gov/pub/time.series/cu/cu.item"
df = pd.read_csv(url, sep='\t')

print(df.head())
