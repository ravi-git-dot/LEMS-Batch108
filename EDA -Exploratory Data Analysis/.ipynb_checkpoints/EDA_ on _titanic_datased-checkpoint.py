
## Logistic Regression with Python
"""
For this lecture we will be working with the Titanic Data Set from Kaggle. This is a very famous data set and very often is a student's first step in machine learning!

We'll be trying to predict a classification- survival or deceased. Let's begin our understanding of implementing Logistic Regression in Python for classification.

We'll use a "semi-cleaned" version of the titanic data set, if you use the data set hosted directly on Kaggle, you may need to do some additional cleaning not shown in this lecture notebook.
"""
# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic_train.csv')

# read first five data set
print(df.head())

# read last five data set
print(df.tail())

# read the column

dataset_column =df.columns
print(f"Dataset_ column names,\n {dataset_column}")

# find out the number of rows and columns
df.shape

# find  the columns information
print(f"data information,\n {df.info()}")

# find each column data types
df.dtypes

# find the data details
df.describe()

# find the duplicated
df.duplicated()

# find the total duplicated
total_duplicated = df.duplicated().sum()
print(total_duplicated)

# find the missing data
miss_data = df.isnull()
print(miss_data)

# find the total null data in dataset
total_missdata= df.isnull().sum()
print(total_missdata)

# find the missing data using visualizzation

sns.heatmap(df.isnull(), yticklabels = False, cbar= False, cmap = "viridis")
plt.show()

sns.set_style("whitegrid")
sns.countplot(x = "Survived", data = df, palette = "Set2")
plt.show()

sns.set_style("whitegrid")
sns.countplot(x ="Survived", hue = "Sex", data =df, palette ="BrBG" )
plt.show()

sns.set_style("whitegrid")
sns.countplot(x="Survived", hue="Pclass", data=df, palette="Set1")
plt.show()

sns.displot(df["Age"].dropna(), kde = False, color = "darkred", bins = 20)
plt.show()

sns.set_style("whitegrid")
sns.pairplot(data=df, hue="Survived", palette="rainbow")
plt.show()

import matplotlib.pyplot as plt

df["Age"].hist(bins = 30, color = "darkred", alpha = 0.3)
plt.show()

sns.countplot(x = "SibSp", data =df, palette = "pastel")

df["Fare"].hist(color = "red", bins =40, figsize = (8,4))
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Age', data=df, palette='winter')
plt.title('Age Distribution Across Passenger Classes')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()

from sklearn.impute import SimpleImputer

imputer_object = SimpleImputer(strategy = "mean")
df["Age"] = imputer_object.fit_transform(df[["Age"]])
df["Age"].head(30)

sns.heatmap(df.isnull(), yticklabels = False, cbar= False, cmap = "viridis")
plt.show()

df.drop("Cabin", axis = 1, inplace = True)
df.head()

df.info()

# find the categorical data
data_columns = df.columns
categorical_data = []
for col in data_columns:
    if df[col].dtype == "O":
        categorical_data.append(col)
print(f"Categorical columns are \n, {categorical_data}")

pd.get_dummies(df['Embarked'],drop_first=True).head()

df.head()

sex = pd.get_dummies(df['Sex'],drop_first=True)
embark = pd.get_dummies(df['Embarked'],drop_first=True)

df.drop(["Name", "Ticket","Embarked"], axis =1, inplace =True)
print(df.head())

from sklearn.preprocessing import LabelEncoder

label_object = LabelEncoder()
df["Sex"] = label_object.fit_transform(df["Sex"])
df["Sex"].head()

## Building a Logistic Regression model
"""
Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).
"""

x = df.drop("Survived", axis =1)
y = df["Survived"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

prediction  = model.predict(x_test)
print(prediction)

from sklearn.metrics import confusion_matrix
accuarcy = confusion_matrix(y_test, prediction)
print(accuarcy)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, prediction)
print(score)

"""
Evaluation
We can check precision,recall,f1-score using classification report!
"""

from sklearn.metrics import classification_report

print(classification_report(y_test,prediction))

"""
.Accuracy = true positive +true negative / total
.Precision = True Positive/ (tp+fp)
.recall = tp /(tp +fn)
.f1_score = 2 x (precision x recall)/ (precision + recall)
"""

