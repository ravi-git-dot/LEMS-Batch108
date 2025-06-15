import pandas as pd
import numpy as np
import pandas as pd

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

df = pd.read_json(data)
print(df)

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

print(df.head())

# convert Json to csv
print(df.to_csv("wine.csv"))

# converting json file into another json formate
print(df.to_json(orient="index"))
