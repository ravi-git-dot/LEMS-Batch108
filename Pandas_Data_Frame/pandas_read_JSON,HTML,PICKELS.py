import pandas as pd
import numpy as np

data = '''
{
  "employee_name": "James",
  "email": "james@gmail.com",
  "job_profile": [
    {
      "Title1": "Team lead",
      "Title2": "Supp.Developer"
    }
  ]
}
'''

df1 = pd.read_json(data)
print(df1)

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

print(df.head())

# convert Json to csv
print(df.to_csv("wine.csv"))

# converting json file into another json formate
print(df.to_json(orient="index"))

print(df1.to_json())

print(df1.to_json(orient="records"))


## Reading HTML content

url = "https://www.fdic.gov/bank/individual/failed/banklist.html"

dfs = pd.read_html(url)
print(dfs)

# find the data type
print(type(dfs))

url_mcc = "https://en.wikipedia.org/wiki/Mobile_country_code"

dfc = pd.read_html(url_mcc, match="country", header=0)
print(dfc)

print(dfc[0])


## Reading Excel file

df_excel = pd.read_excel("Excel_Sample.xlsx")
print(df_excel.head())

## Pickling

'''
All pandas object are equipped with to_pickle method which use python's cpickle module to save data 
structure to disk using the pickle formate 
'''

df_excel.to_pickle("df_excel.pkl")     # Save with extension
df = pd.read_pickle("df_excel.pkl")    # Load back

print(df.head())